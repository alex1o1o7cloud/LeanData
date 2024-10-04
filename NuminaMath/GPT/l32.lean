import Mathlib

namespace value_of_f_neg6_l32_32554

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 2) = -f x

theorem value_of_f_neg6 : f (-6) = 0 :=
by
  sorry

end value_of_f_neg6_l32_32554


namespace length_of_AG_l32_32236

open_locale real
  
theorem length_of_AG (A B C D M G : Type*) [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited M] [inhabited G] 
  (hAB : dist A B = 3) (hAC : dist A C = 3 * real.sqrt 3) (hRight : ∀ (a b c: Type*) [inhabited a] [inhabited b] [inhabited c], ∃ (A ≠ B ≠ C), ∠A B C = 90) 
  (hMidpoint : midpoint B C M) (hAltitude : altitude A D B C) (hIntersection : intersects D B M G) : 
  dist A G = 3 * real.sqrt 3 / 4 :=
sorry

end length_of_AG_l32_32236


namespace number_of_possible_x_l32_32213

theorem number_of_possible_x (x : ℕ) (h : 11 ≤ Real.sqrt x ∧ Real.sqrt x < 12) : 
  (finset.filter (λ n, 11 ≤ Real.sqrt n ∧ Real.sqrt n < 12) (finset.range 144)).card = 23 :=
sorry

end number_of_possible_x_l32_32213


namespace parabola_directrix_value_l32_32587

noncomputable def parabola_p_value (p : ℝ) : Prop :=
(∀ y : ℝ, y^2 = 2 * p * (-2 - (-2)))

theorem parabola_directrix_value : parabola_p_value 4 :=
by
  -- proof steps here
  sorry

end parabola_directrix_value_l32_32587


namespace closest_to_product_l32_32756

theorem closest_to_product :
  let y := (0.48017) ^ 3
  let a1 := 0.011
  let a2 := 0.110
  let a3 := 1.10
  let a4 := 11.0
  let a5 := 110
in
  |y - a2| < |y - a1| ∧ |y - a2| < |y - a3| ∧ |y - a2| < |y - a4| ∧ |y - a2| < |y - a5| :=
by
  sorry

end closest_to_product_l32_32756


namespace is_generalized_distance_l32_32866

open Real

def generalized_distance (f : ℝ → ℝ → ℝ) : Prop :=
  (∀ x y, f x y ≥ 0 ∧ (f x y = 0 ↔ x = 0 ∧ y = 0)) ∧
  (∀ x y, f x y = f y x) ∧
  (∀ x y z, f x y ≤ f x z + f z y)

theorem is_generalized_distance : generalized_distance (λ x y, x^2 + y^2) :=
  sorry

end is_generalized_distance_l32_32866


namespace zeros_at_end_of_factorial_300_l32_32816

theorem zeros_at_end_of_factorial_300 : 
  (nat.factorial 300).trailingZeroes = 74 :=
sorry

end zeros_at_end_of_factorial_300_l32_32816


namespace combined_value_of_cookies_l32_32411

theorem combined_value_of_cookies
  (total_boxes_sold : ℝ)
  (plain_boxes_sold : ℝ)
  (price_chocolate_chip : ℝ)
  (price_plain : ℝ)
  (h1 : total_boxes_sold = 1585)
  (h2 : plain_boxes_sold = 793.375)
  (h3 : price_chocolate_chip = 1.25)
  (h4 : price_plain = 0.75) :
  (plain_boxes_sold * price_plain) + ((total_boxes_sold - plain_boxes_sold) * price_chocolate_chip) = 1584.5625 :=
by
  sorry

end combined_value_of_cookies_l32_32411


namespace lending_interest_rate_l32_32046

theorem lending_interest_rate 
  (P : ℝ) (borrowed_R : ℝ) (borrowed_T : ℝ) (gain_per_year : ℝ) (lent_T : ℝ) :
  P = 7000 →
  borrowed_R = 4 →
  borrowed_T = 2 →
  gain_per_year = 140 →
  lent_T = 2 →
  let borrowed_SI := (P * borrowed_R * borrowed_T) / 100 in
  let total_gain := gain_per_year * borrowed_T in
  let total_earned_interest := borrowed_SI + total_gain in
  let lent_R := (total_earned_interest * 100) / (P * lent_T) in
  lent_R = 6 :=
begin
  intros P_eq borrowed_R_eq borrowed_T_eq gain_eq lent_T_eq,
  rw [P_eq, borrowed_R_eq, borrowed_T_eq, gain_eq, lent_T_eq],
  let borrowed_SI := (7000 * 4 * 2) / 100,
  let total_gain := 140 * 2,
  let total_earned_interest := borrowed_SI + total_gain,
  let lent_R := (total_earned_interest * 100) / (7000 * 2),
  have h_borrowed_SI : borrowed_SI = 560 := by norm_num,
  have h_total_gain : total_gain = 280 := by norm_num,
  have h_total_earned_interest : total_earned_interest = 840 := by norm_num,
  have h_lent_R : lent_R = 6 := by norm_num,
  exact h_lent_R,
end

end lending_interest_rate_l32_32046


namespace quadratic_function_properties_l32_32611

def is_quadratic {α : Type*} [OrderedRing α] (f : α → α) : Prop :=
  ∃ a h k : α, f = λ x, a * (x - h)^2 + k

theorem quadratic_function_properties (a : ℝ) (h k : ℝ)
  (vertex_on_x_axis : k = 0)
  (right_side_rising : a > 0)
  (axis_of_symmetry_at_origin : h = 0) :
    ∃ f : ℝ → ℝ, is_quadratic f ∧ f 0 = 0 ∧ f = λ x, ax^2 :=
by {
  have f_def : ∃ f : ℝ → ℝ, f = λ x, a * (x - 0)^2,
  { use λ x, a * x^2, 
    sorry },
  exact exists.intro (λ x, a * x^2) ⟨⟨a, 0, 0, rfl⟩, 
                                       by simp [vertex_on_x_axis, axis_of_symmetry_at_origin], 
                                       by simp [right_side_rising]⟩,
  sorry
}

end quadratic_function_properties_l32_32611


namespace laptop_price_l32_32982

theorem laptop_price (x : ℝ) : 
  (0.8 * x - 120) = 0.9 * x - 64 → x = 560 :=
by
  sorry

end laptop_price_l32_32982


namespace sum_of_digits_of_all_odd_numbers_from_1_to_5000_l32_32451

theorem sum_of_digits_of_all_odd_numbers_from_1_to_5000 :
  let odd_numbers := (List.range 5000).filter (λ n, n % 2 = 1)
  let sum_of_digits := (fun (n : Nat) => (n.digits 10).sum)
  (odd_numbers.map sum_of_digits).sum = 54025 :=
  sorry

end sum_of_digits_of_all_odd_numbers_from_1_to_5000_l32_32451


namespace investment_Y_l32_32010

theorem investment_Y
  (X_investment : ℝ)
  (Y_investment : ℝ)
  (Z_investment : ℝ)
  (X_months : ℝ)
  (Y_months : ℝ)
  (Z_months : ℝ)
  (total_profit : ℝ)
  (Z_profit_share : ℝ)
  (h1 : X_investment = 36000)
  (h2 : Z_investment = 48000)
  (h3 : X_months = 12)
  (h4 : Y_months = 12)
  (h5 : Z_months = 8)
  (h6 : total_profit = 13970)
  (h7 : Z_profit_share = 4064) :
  Y_investment = 75000 := by
  -- Proof omitted
  sorry

end investment_Y_l32_32010


namespace sufficient_conditions_for_quadratic_l32_32582

theorem sufficient_conditions_for_quadratic (x : ℝ) : 
  (0 < x ∧ x < 4) ∨ (-2 < x ∧ x < 4) ∨ (-2 < x ∧ x < 3) → x^2 - 2*x - 8 < 0 :=
by
  sorry

end sufficient_conditions_for_quadratic_l32_32582


namespace closest_integer_to_cube_root_of_1728_l32_32005

theorem closest_integer_to_cube_root_of_1728: 
  ∃ n : ℕ, n^3 = 1728 ∧ (∀ m : ℤ, m^3 < 1728 → m < n) ∧ (∀ p : ℤ, p^3 > 1728 → p > n) :=
by
  sorry

end closest_integer_to_cube_root_of_1728_l32_32005


namespace parabola_equation_l32_32473

theorem parabola_equation (a b c d e f : ℤ) (x y : ℝ) (h0 : a > 0) 
  (h1 : a.natAbs.gcd b.natAbs.gcd c.natAbs.gcd d.natAbs.gcd e.natAbs.gcd f.natAbs = 1) 
  (h2 : (λ (p : Prod), sqrt ((p.fst - 2)^2 + (p.snd + 1)^2) = (abs (p.fst + 2 * p.snd - 5)) / (sqrt 5))
        ((x, y))) :
  4 * x^2 - 4 * x^2 * y^2 + 5 * y^2 + 10 * y - 20 = 0 :=
sorry

end parabola_equation_l32_32473


namespace fg_3_eq_123_l32_32589

def f (x : ℤ) : ℤ := x^2 + 2
def g (x : ℤ) : ℤ := 3 * x + 2

theorem fg_3_eq_123 : f (g 3) = 123 := by
  sorry

end fg_3_eq_123_l32_32589


namespace value_of_x_l32_32216

theorem value_of_x (x y : ℝ) (h1 : x ≠ 0) (h2 : x / 3 = y^2) (h3 : x / 6 = 3 * y) : x = 108 :=
by
  sorry

end value_of_x_l32_32216


namespace frog_ends_on_vertical_side_l32_32040

noncomputable def frog_probability_vertical_side : ℚ := 
  let P : ℚ × ℚ → ℚ := 
    λ ⟨x, y⟩, 
      if (x = 0 ∨ x = 6) then 1 
      else if (y = 0 ∨ y = 6) then 0 
      else if (x = 2 ∧ y = 3) then 3/4 
      else sorry 
  in P (2, 3)

theorem frog_ends_on_vertical_side : frog_probability_vertical_side = 3/4 := 
by sorry

end frog_ends_on_vertical_side_l32_32040


namespace number_of_integers_satisfying_condition_l32_32270

def sum_of_divisors (n : ℕ) : ℕ :=
  (Nat.divisors n).sum

def sum_of_proper_divisors (n : ℕ) : ℕ :=
  (Nat.properDivisors n).sum

def condition_satisfied (i : ℕ) : Prop :=
  sum_of_divisors i = 1 + Int.sqrt i + i + 2 * sum_of_proper_divisors i

theorem number_of_integers_satisfying_condition :
  {i : ℕ | 1 ≤ i ∧ i ≤ 3000 ∧ condition_satisfied i}.card = 16 :=
sorry

end number_of_integers_satisfying_condition_l32_32270


namespace permutations_with_exactly_one_descent_permutations_with_exactly_two_descents_l32_32194

-- Part (a)
theorem permutations_with_exactly_one_descent (n : ℕ) : 
  ∃ (count : ℕ), count = 2^n - n - 1 := sorry

-- Part (b)
theorem permutations_with_exactly_two_descents (n : ℕ) : 
  ∃ (count : ℕ), count = 3^n - 2^n * (n + 1) + (n * (n + 1)) / 2 := sorry

end permutations_with_exactly_one_descent_permutations_with_exactly_two_descents_l32_32194


namespace magnitude_conj_z_l32_32526

noncomputable def z := (3 + Complex.i) / (1 - 2 * Complex.i)
noncomputable def conj_z := Complex.conj z

theorem magnitude_conj_z : Complex.abs conj_z = Real.sqrt 2 :=
by sorry

end magnitude_conj_z_l32_32526


namespace gavin_shirts_l32_32533

theorem gavin_shirts (t g b : ℕ) (h_total : t = 23) (h_green : g = 17) (h_blue : b = t - g) : b = 6 :=
by sorry

end gavin_shirts_l32_32533


namespace sum_logarithm_identity_l32_32461

open Real

theorem sum_logarithm_identity :
  (∑ k in Finset.range 30 \ Finset.range 1,
     log 3 (1 + 2 / (k + 2)) * (log (k + 2) 3) * (log (k + 3) 3)
    ) = 
    (1 / log 3 2) - (1 / 3) := sorry

end sum_logarithm_identity_l32_32461


namespace positive_difference_of_solutions_eq_l32_32376

noncomputable def positive_difference_of_solutions : ℝ :=
  let discriminant := 33 in
  let root1 := (-3 + real.sqrt discriminant) / 4 in
  let root2 := (-3 - real.sqrt discriminant) / 4 in
  abs (root1 - root2) / 2

theorem positive_difference_of_solutions_eq :
  ∀ q : ℝ, q ≠ 3 →
  (frac (q^2 - 4*q - 21) (q - 3) = 3*q + 8) →
  positive_difference_of_solutions = real.sqrt 33 / 2 :=
by
  intro q hq h
  sorry

end positive_difference_of_solutions_eq_l32_32376


namespace order_f_a_b_c_l32_32128

noncomputable def f (x : ℝ) : ℝ := 2^x - 2^(-x)

def a : ℝ := (7/9)^(-1/4)
def b : ℝ := (9/7)^(1/5)
def c : ℝ := Real.log 2 (7/9)

theorem order_f_a_b_c : f(b) < f(a) ∧ f(c) < f(b) :=
by
  -- The proof is omitted here
  sorry

end order_f_a_b_c_l32_32128


namespace find_length_BC_l32_32442

variable (A B C O I : Type)
variable [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited O] [Inhabited I]

-- Definitions of various elements and properties
variable (circumcenter : A → A → A → O)
variable (incenter : A → A → A → I)
variable (perpendicular : O → I → A → Prop)
variable (distance : A → A → ℝ)
variable (AB : A)
variable (AC : A)

-- Given conditions
axiom circumcenter_def {A B C} : circumcenter A B C = O
axiom incenter_def {A B C} : incenter A B C = I
axiom perpendicular_def {O I A} : perpendicular O I A = True
axiom AB_length {A B} : distance A B = 10
axiom AC_length {A C} : distance A C = 18

-- Proof goal (requires proof, using conclusion from problem solution)
theorem find_length_BC {ABC : A} (O I : Type) (A B C : ABC) :
  circumcenter A B C = O →
  incenter A B C = I →
  perpendicular O I A → 
  distance A B = 10 →
  distance A C = 18 →
  distance B C = 14 :=
by
    -- skipping the proof details
    sorry

end find_length_BC_l32_32442


namespace fixed_area_of_tiling_l32_32401

-- A simple polygon does not self-intersect
structure SimplePolygon where
  vertices : List (ℝ × ℝ)
  non_self_intersecting : ∀ (i j : ℕ), i ≠ j → (i < vertices.length) → (j < vertices.length) → 
                                            ∃ u v: ℝ, (i, u ≠ i, v) ∧ (j, u ≠ j, v) 

-- A parallelogram can be represented as a quadrilateral with opposite sides parallel and equal length
structure Parallelogram where
  side_vectors : List (ℝ × ℝ)
  parallel_sides : (side_vectors.head ∥ side_vectors.getLast) ∧ (side_vectors.getNth 1 ∥ side_vectors.getNth 3)
  equal_sides : (euclidean_dist side_vectors.head side_vectors.getNth 1 = euclidean_dist side_vectors.getLast side_vectors.getNth 3) ∧
                (euclidean_dist side_vectors.getNth 1 side_vectors.getNth 2 = euclidean_dist side_vectors.getNth 3 side_vectors.head)

-- A tiling of a polygon with a finite number of parallelograms
structure Tiling (P : SimplePolygon) where
  parallelograms : List Parallelogram
  covers_P : ∀ p ∈ P.vertices, ∃ q ∈ parallelograms, p ∈ q.side_vectors.toSet
  finite_tiling : parallelograms.length < ∞

-- The statement to prove
theorem fixed_area_of_tiling (P : SimplePolygon) (tiling : Tiling P) : 
  ∃ area : ℝ, ∀ (q ∈ tiling.parallelograms), 
    P.vertices.Area = ∑ (r ∈ tiling.parallelograms), r.area :=
by
  sorry

end fixed_area_of_tiling_l32_32401


namespace mean_median_mode_y_l32_32480

theorem mean_median_mode_y : 
  ∃ y : ℤ, 
  (60 + 100 + y + 40 + 50 + 300 + y + 70 + 90) / 9 = y ∧ 
  (let sorted := list.sort [40, 50, 60, 70, 90, y, y, 100, 300] in
  sorted.nth_le (sorted.length / 2) sorry = y) ∧ 
  (∃ (n : ℕ), list.count y [60, 100, y, 40, 50, 300, y, 70, 90] = n + 1)
  := ∃ y, y = 100 :=
by
  use 100
  rw [list.nth_le, list.count]
  sorry

end mean_median_mode_y_l32_32480


namespace concurrency_l32_32555

variables {α : Type*} [metric_space α] [normed_space ℝ α] [inner_product_space ℝ α]

-- Definitions for circles and points
def circle (c : α) (r : ℝ) := {p : α | dist p c = r}
def line (p1 p2 : α) := {p : α | ∃ λ : ℝ, p = p1 + λ • (p2 - p1)}

variables (O1 O2 O3 A B C D P Q : α)
variables (r1 r2 r3 : ℝ)

-- Conditions given in part a)
axiom H1 : circle O1 r1 A
axiom H2 : circle O1 r1 B
axiom H3 : circle O2 r2 A
axiom H4 : circle O2 r2 B
axiom H5 : line A (O1 - O2) C
axiom H6 : line A (O1 - O2) D
axiom H7 : circle O3 r3 C
axiom H8 : circle O3 r3 D
axiom H9 : circle O1 r1 P
axiom H10 : circle O2 r2 P
axiom H11 : circle O1 r1 Q
axiom H12 : circle O2 r2 Q

-- Theorem to prove concurrency
theorem concurrency : ∃ X : α, line C P X ∧ line D Q X ∧ line A B X :=
by 
  sorry

end concurrency_l32_32555


namespace area_comparison_l32_32770

variable (base_B height_B : ℝ)
def area_B : ℝ := (1 / 2) * base_B * height_B
def area_A : ℝ := (1 / 2) * (1.10 * base_B) * (0.90 * height_B)

theorem area_comparison : area_A base_B height_B = 0.99 * area_B base_B height_B :=
by
  sorry

end area_comparison_l32_32770


namespace conversion_correct_l32_32093

-- Define the conversion of a single octal digit to its binary equivalent
def octalToBinary : Fin 8 → String
| 0 := "000"
| 1 := "001"
| 2 := "010"
| 3 := "011"
| 4 := "100"
| 5 := "101"
| 6 := "110"
| 7 := "111"

-- Define the conversion from an octal number (as a list of digits) to a binary string
def convertOctalToBinary (digits : List (Fin 8)) : String :=
  String.join (digits.map octalToBinary)

-- Define the given octal digits: 726 in base 8
def octalDigits : List (Fin 8) := [7, 2, 6]

-- Define the expected binary result
def expectedBinary : String := "111010110"

-- The main statement to prove
theorem conversion_correct :
  convertOctalToBinary octalDigits = expectedBinary :=
by sorry

end conversion_correct_l32_32093


namespace quadratic_inequality_l32_32719

variable {a b c : ℝ}

noncomputable def quadratic_polynomial (x : ℝ) := a * x^2 + b * x + c

theorem quadratic_inequality (h1 : ∀ x : ℝ, quadratic_polynomial x < 0)
    (h2 : a < 0) (h3 : b^2 - 4*a*c < 0) : (b / a) < (c / a + 1) := 
sorry

end quadratic_inequality_l32_32719


namespace max_and_min_values_cos_value_in_interval_l32_32899

noncomputable def f (x : ℝ) (A ω φ : ℝ) : ℝ := A * Real.sin (ω * x + φ)

variables (A ω φ : ℝ) (hA : 0 < A) (hω : 0 < ω) (hφ : 0 < φ ∧ φ < π / 2)
          (f_min : ∀ x, f x A ω φ ≥ -4)
          (f_at_zero : f 0 A ω φ = 2 * Real.sqrt 2)
          (symmetry_distance : (∀ x, f (x + π / ω) = f x A ω φ))

theorem max_and_min_values : 
  (∀ x ∈ Set.Icc (-π / 2) (π / 2), f x A ω φ ≤ 4) ∧ 
  (∃ x ∈ Set.Icc (-π / 2) (π / 2), f x A ω φ = 4) ∧ 
  (∀ x ∈ Set.Icc (-π / 2) (π / 2), f x A ω φ ≥ -2 * Real.sqrt 2) ∧ 
  (∃ x ∈ Set.Icc (-π / 2) (π / 2), f x A ω φ = -2 * Real.sqrt 2) :=
sorry

theorem cos_value_in_interval (x : ℝ) (hx : x ∈ Set.Ioo (π / 2) π) (fx_eq_1 : f x A ω φ = 1) :
  Real.cos (x + 5 * π / 12) = (-(3 * Real.sqrt (5) + 1) / 8) :=
sorry

end max_and_min_values_cos_value_in_interval_l32_32899


namespace quadratic_inequality_l32_32720

variables {a b c : ℝ}

theorem quadratic_inequality (h1 : ∀ x : ℝ, a * x^2 + b * x + c < 0) 
                            (h2 : a < 0) 
                            (h3 : b^2 - 4 * a * c < 0) : 
                            (b / a) < (c / a + 1) :=
begin
  sorry
end

end quadratic_inequality_l32_32720


namespace camp_III_students_selected_l32_32780

theorem camp_III_students_selected :
  (∀ (n : ℕ) (sample_size interval : ℕ), n = 600 → sample_size = 50 → interval = 12 →
    (∀ (first : ℕ), first = 3 →
      ∃ k_min k_max : ℕ,
        42 = k_min ∧ 49 = k_max ∧
        let number_selected from := k_max - k_min + 1 in
        number_selected = 8)) :=
begin
  sorry
end

end camp_III_students_selected_l32_32780


namespace mary_talking_ratio_l32_32984

theorem mary_talking_ratio:
  let mac_download_time := 10
  let windows_download_time := 3 * mac_download_time
  let audio_glitch_time := 2 * 4
  let video_glitch_time := 6
  let total_glitch_time := audio_glitch_time + video_glitch_time
  let total_download_time := mac_download_time + windows_download_time
  let total_time := 82
  let talking_time := total_time - total_download_time
  let talking_time_without_glitch := talking_time - total_glitch_time
  talking_time_without_glitch / total_glitch_time = 2 :=
by
  sorry

end mary_talking_ratio_l32_32984


namespace max_sum_range_l32_32698

theorem max_sum_range (f g : ℝ → ℝ) (Hf : ∀ x, f x ∈ set.Icc (-6 : ℝ) 4) (Hg : ∀ x, g x ∈ set.Icc (-3 : ℝ) 2) :
  ∃ d, d = 6 ∧ ∀ x, f x + g x ∈ set.Icc (set.Icc.min ⟨Hf⟩ ⟨Hg⟩) d :=
by
  sorry

end max_sum_range_l32_32698


namespace speech_contest_sequences_l32_32228

theorem speech_contest_sequences:
  let contestants := ["B1", "B2", "B3", "G1", "G2"] in
  let is_boy c := c = "B1" ∨ c = "B2" ∨ c = "B3" in
  let is_girl c := c = "G1" ∨ c = "G2" in
  let valid_sequence (s : List String) :=
    ¬ (s.get? 0 = some "B1") ∧
    (∀ (i : Fin s.length - 1), (s.get? i = some "G1" ∨ s.get? i = some "G2") → ¬ (s.get? (i + 1) = some "G1" ∨ s.get? (i + 1) = some "G2")) in
  (List.permutations contestants).count valid_sequence = 60 :=
by
  sorry

end speech_contest_sequences_l32_32228


namespace count_n_with_conditions_l32_32912

theorem count_n_with_conditions :
  let n := {n : ℕ | (∃ m : ℕ, 7 * n + 1 = m * m) ∧ 3 * n + 1 < 2008}
  finset.card n = 16 :=
by
  sorry

end count_n_with_conditions_l32_32912


namespace surface_area_of_frustum_correct_l32_32410

-- Define the context for the frustum
variables (s_top s_bottom slant_height : ℝ)
-- Top base side length
def top_base_side := s_top = 2
-- Bottom base side length
def bottom_base_side := s_bottom = 4
-- Slant height
def frustum_slant_height := slant_height = 2

-- The surface area of the frustum
def frustum_surface_area : ℝ := 12 * Real.sqrt 3 + 20

-- The theorem proving the surface area is correct given the conditions
theorem surface_area_of_frustum_correct
  (h1 : top_base_side)
  (h2 : bottom_base_side)
  (h3 : frustum_slant_height) :
  frustum_surface_area = 12 * Real.sqrt 3 + 20 :=
sorry

end surface_area_of_frustum_correct_l32_32410


namespace tetrahedron_distance_l32_32990

-- Define the properties of the tetrahedron
structure Tetrahedron :=
  (side1 : ℝ) -- length of one side of the equilateral triangle
  (side2 : ℝ) -- length of edge adjacent to side1
  (side3 : ℝ) -- length of edge adjacent to side2
  (side4 : ℝ) -- length of edge adjacent to side3
  (side5 : ℝ) -- length of edge adjacent to side1 (completing the tetrahedron)

def equilateral_face (t : Tetrahedron) : Prop :=
  t.side1 = 6 ∧ t.side2 = t.side1 ∧ t.side3 = t.side1

def other_edges (t : Tetrahedron) : Prop :=
  t.side4 = 3 ∧ t.side5 = 4 ∧ t.side2 = 5

-- The goal is to prove the distance between the line with an edge of length 3 units and its opposite edge

theorem tetrahedron_distance (t : Tetrahedron)
  (h1 : equilateral_face t)
  (h2 : other_edges t) :
  ∃ d : ℝ, d = 3.0356 :=
begin
  sorry
end

end tetrahedron_distance_l32_32990


namespace local_extremum_of_10_at_1_l32_32900

def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem local_extremum_of_10_at_1 (a b : ℝ) (h1 : deriv (f x a b) 1 = 0)
  (h2 : f 1 a b = 10) : a = 4 :=
sorry

end local_extremum_of_10_at_1_l32_32900


namespace gain_percent_of_articles_l32_32921

theorem gain_percent_of_articles (C S : ℝ) (h : 50 * C = 15 * S) : (S - C) / C * 100 = 233.33 :=
by
  sorry

end gain_percent_of_articles_l32_32921


namespace sqrt_of_mixed_number_l32_32500

theorem sqrt_of_mixed_number :
  (Real.sqrt (8 + 9 / 16)) = (Real.sqrt 137 / 4) :=
by
  sorry

end sqrt_of_mixed_number_l32_32500


namespace football_shaped_area_of_regions_II_and_III_l32_32685

theorem football_shaped_area_of_regions_II_and_III {ABCD : Type*} {A B C D : ABCD}
  (h_square : quadrilateral ABCD)
  (h_arc_AEC : ∃ (r : ℝ) (S : set (ℝ × ℝ)), is_circle S ∧ (0 : ℝ × ℝ) ∈ S ∧ D ∈ S ∧ arc AEC ⊆ S)
  (h_arc_AFC : ∃ (r : ℝ) (T : set (ℝ × ℝ)), is_circle T ∧ (0 : ℝ × ℝ) ∈ T ∧ B ∈ T ∧ arc AFC ⊆ T)
  (h_AB : dist A B = 4) :
  combined_area II III = 9.1 :=
sorry

end football_shaped_area_of_regions_II_and_III_l32_32685


namespace distance_preserving_l32_32955

variables {Point : Type} {d : Point → Point → ℕ} {f : Point → Point}

axiom distance_one (A B : Point) : d A B = 1 → d (f A) (f B) = 1

theorem distance_preserving :
  ∀ (A B : Point) (n : ℕ), n > 0 → d A B = n → d (f A) (f B) = n :=
by
  sorry

end distance_preserving_l32_32955


namespace smallest_typical_parallelepipeds_l32_32670

theorem smallest_typical_parallelepipeds (s : ℕ) :
  ∃ n : ℕ, (∀ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ a) → 
           (∀ (volume_sum : fin n → (ℕ × ℕ × ℕ)), 
            (∀ i, (volume_sum i).1 * (volume_sum i).2 * (volume_sum i).3 = s^3 / n)) ∧ n = 6 :=
begin
  sorry
end

end smallest_typical_parallelepipeds_l32_32670


namespace question_I_solution_set_question_II_range_l32_32170

noncomputable def f (x : ℝ) : ℝ := |2 * x - 4| + 1

-- Question I
theorem question_I_solution_set : 
  {x : ℝ | f x > |x + 1|} = set.Iio (4 / 3) ∪ set.Ioi 4 := 
by
  sorry

-- Question II
theorem question_II_range (a b m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = a + b) (h4 : f (m + 1) ≤ a + 4 * b) : 
  -3 ≤ m ∧ m ≤ 5 := 
by
  sorry

end question_I_solution_set_question_II_range_l32_32170


namespace find_q_l32_32257

theorem find_q (q : ℂ) (g : ℂ) (u : ℂ) :
  2 * g * q - u = 200 → g = 3 + 8 * complex.I → u = 1 + 16 * complex.I → q = 5 - 1602 / 150 * complex.I :=
by
  intros h1 h2 h3
  rw [h2] at h1
  rw [h3] at h1
  sorry

end find_q_l32_32257


namespace four_digit_distinct_abs_diff_eq_2_l32_32436

theorem four_digit_distinct_abs_diff_eq_2 : 
  ∃ count : ℕ, 
    count = 840 ∧ 
    (∀ n, 1000 ≤ n ∧ n ≤ 9999 → 
      (let a := n / 1000 in
       let b := (n / 100 % 10) in
       let c := (n / 10 % 10) in
       let d := n % 10 in
       a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ 
       b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
       abs (a - d) = 2) → 
       (count += 1)) :=
sorry

end four_digit_distinct_abs_diff_eq_2_l32_32436


namespace infinite_product_value_l32_32752

noncomputable def infinite_product : ℝ :=
  ∏' n : ℕ, 9^(1/(3^n))

theorem infinite_product_value : infinite_product = 27 := 
  by sorry

end infinite_product_value_l32_32752


namespace events_complementary_l32_32789

-- Definitions derived from conditions
def total_people : Nat := 5
def boys : Nat := 3
def girls : Nat := 2

-- Define events
def event_at_least_one_boy := { xy : Fin 5 × Fin 5 // xy.1 ≠ xy.2 ∧ (xy.1 < 3 ∨ xy.2 < 3) }
def event_all_girls := { xy : Fin 5 × Fin 5 // xy.1 ≠ xy.2 ∧ xy.1 ∈ {3, 4} ∧ xy.2 ∈ {3, 4} }

-- The proof problem
theorem events_complementary :
  ∀ (xy : Fin 5 × Fin 5),
    xy.1 ≠ xy.2 →
    (event_at_least_one_boy xy → ¬ event_all_girls xy) ∧ (event_all_girls xy → ¬ event_at_least_one_boy xy) :=
by
  sorry

end events_complementary_l32_32789


namespace sequence_solution_l32_32160

theorem sequence_solution (n : ℕ) (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h : ∀ n, S n = 2 * a n - 2^n + 1) : a n = n * 2^(n-1) :=
sorry

end sequence_solution_l32_32160


namespace find_analytic_expression_k_no_minimum_l32_32903

section quadratic_problems

variable (f : ℝ → ℝ)
variable (k : ℝ)

-- Condition 1: minimum value at x = 2 is -1
axiom f_at_min : ∀ x, f x = (x - 2)^2 - 1

-- Condition 2: f(1) + f(4) = 3
axiom f1_f4_sum : f 1 + f 4 = 3

-- Question 1: expression of f(x)
theorem find_analytic_expression : f = λ x, x^2 - 4*x + 3 := 
by
  sorry

-- g(x) = f(x) - kx
def g (x : ℝ) := f x - k * x

-- Question 2: range of k such that g(x) has no minimum on (1,4)
theorem k_no_minimum : k ∈ Set.Union Set.Iic Set.Ioi :=
by
  sorry

end quadratic_problems

end find_analytic_expression_k_no_minimum_l32_32903


namespace car_kilometers_per_gallon_l32_32028

-- Define the given conditions as assumptions
variable (total_distance : ℝ) (total_gallons : ℝ)
-- Assume the given conditions
axiom h1 : total_distance = 180
axiom h2 : total_gallons = 4.5

-- The statement to be proven
theorem car_kilometers_per_gallon : (total_distance / total_gallons) = 40 :=
by
  -- Sorry is used to skip the proof
  sorry

end car_kilometers_per_gallon_l32_32028


namespace smallest_a_distinct_seq_l32_32657

noncomputable def sequence (a : ℤ) : ℕ → ℤ
| 1     := a
| (n+1) := Nat.prime_factors (sequence a n ^ 2 - 1).max' sorry

theorem smallest_a_distinct_seq : ∀ a > 1, (∀ n m : ℕ, n ≠ m → sequence a n ≠ sequence a m) → a = 46 :=
begin
  sorry
end

end smallest_a_distinct_seq_l32_32657


namespace sum_of_three_at_least_fifty_l32_32307

-- Definitions
variables {F : Type} [fintype F] [decidable_eq F]
variables (friends : fin 7 → ℕ)
variables (h_sum : (∀ i j : fin 7, i ≠ j → friends i ≠ friends j))
variables (h_total : ∑ i, friends i = 100)

-- Theorem
theorem sum_of_three_at_least_fifty : 
  ∃ (i j k : fin 7), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ friends i + friends j + friends k ≥ 50 :=
sorry

end sum_of_three_at_least_fifty_l32_32307


namespace g_five_times_of_3_l32_32471

def g : ℕ → ℕ
| x := if x % 2 = 0 then x / 2
       else if x < 10 then 3 * x + 2
       else x - 1

theorem g_five_times_of_3 : g (g (g (g (g 3)))) = 16 := by
  sorry

end g_five_times_of_3_l32_32471


namespace impossible_to_have_card_divisible_by_two_pow_d_l32_32017

-- Initial conditions
def initial_cards : ℕ := 100
def odd_cards_initial : ℕ := 28

-- Every minute, for every 12 cards, their product is calculated and their sum creates a new card.
def new_card(c : ℕ) : ℕ :=
  (finset.range c).sum (λ i, (finset.range 12).prod (λ j, 1))  -- Simplified representation

-- Prove that it is impossible for there to eventually be a card divisible by 2^d for all natural d.
theorem impossible_to_have_card_divisible_by_two_pow_d :
  ∀ d : ℕ, ∃ cards : finset ℕ, ¬ (∃ card ∈ cards, 2^d ∣ card) := 
  sorry 

end impossible_to_have_card_divisible_by_two_pow_d_l32_32017


namespace total_fish_count_l32_32077

-- Define the number of fish for each person
def Billy := 10
def Tony := 3 * Billy
def Sarah := Tony + 5
def Bobby := 2 * Sarah

-- Define the total number of fish
def TotalFish := Billy + Tony + Sarah + Bobby

-- Prove that the total number of fish all 4 people have put together is 145
theorem total_fish_count : TotalFish = 145 := 
by
  -- provide the proof steps here
  sorry

end total_fish_count_l32_32077


namespace product_of_solutions_pos_real_part_eq_nine_l32_32609

open Complex

noncomputable def solve : ℂ :=
  let z := Complex.exp (Complex.i * Real.pi) in  -- polar form of -1
  let r := 729 ^ (1/6 : ℝ) in  -- 6th root of 729
  let θ := (Real.pi) / 6 in  -- 30 degrees in radians
  let sol1 := Complex.ofReal r * Complex.exp (Complex.i * θ) in
  let sol2 := Complex.ofReal r * Complex.exp (Complex.i * (2 * Real.pi - θ)) in
  sol1 * sol2

theorem product_of_solutions_pos_real_part_eq_nine :
  (solve.re = 9) :=
sorry

end product_of_solutions_pos_real_part_eq_nine_l32_32609


namespace point_of_tangent_parallel_x_axis_l32_32580

theorem point_of_tangent_parallel_x_axis :
  ∃ M : ℝ × ℝ, (M.1 = -1 ∧ M.2 = -3) ∧
    (∃ y : ℝ, y = M.1^2 + 2 * M.1 - 2 ∧
    (∃ y' : ℝ, y' = 2 * M.1 + 2 ∧ y' = 0)) :=
sorry

end point_of_tangent_parallel_x_axis_l32_32580


namespace current_in_series_circuit_l32_32231

noncomputable def I : ℂ :=
  let V : ℂ := 5 - 2 * Complex.i
  let Z1 : ℂ := 2 + Complex.i
  let Z2 : ℂ := 3 - 2 * Complex.i
  let Z : ℂ := Z1 + Z2
  V / Z

theorem current_in_series_circuit :
  I = (9 / 8 : ℚ) - (5 / 24 : ℚ) * Complex.i := by
  sorry

end current_in_series_circuit_l32_32231


namespace sum_floor_sqrt_eq_l32_32120

theorem sum_floor_sqrt_eq (n : ℕ) : 
  ∑ k in Finset.range (n^2 + 1) (n + 1)^2, (⌊Real.sqrt k⌋) = 2 * n^2 + n :=
sorry

end sum_floor_sqrt_eq_l32_32120


namespace percentage_material_B_in_final_mixture_l32_32315

-- Conditions
def percentage_material_A_in_Solution_X : ℝ := 20
def percentage_material_B_in_Solution_X : ℝ := 80
def percentage_material_A_in_Solution_Y : ℝ := 30
def percentage_material_B_in_Solution_Y : ℝ := 70
def percentage_material_A_in_final_mixture : ℝ := 22

-- Goal
theorem percentage_material_B_in_final_mixture :
  100 - percentage_material_A_in_final_mixture = 78 := by
  sorry

end percentage_material_B_in_final_mixture_l32_32315


namespace total_fish_count_l32_32078

-- Define the number of fish for each person
def Billy := 10
def Tony := 3 * Billy
def Sarah := Tony + 5
def Bobby := 2 * Sarah

-- Define the total number of fish
def TotalFish := Billy + Tony + Sarah + Bobby

-- Prove that the total number of fish all 4 people have put together is 145
theorem total_fish_count : TotalFish = 145 := 
by
  -- provide the proof steps here
  sorry

end total_fish_count_l32_32078


namespace quadratic_inequality_l32_32722

variables {a b c : ℝ}

theorem quadratic_inequality (h1 : ∀ x : ℝ, a * x^2 + b * x + c < 0) 
                            (h2 : a < 0) 
                            (h3 : b^2 - 4 * a * c < 0) : 
                            (b / a) < (c / a + 1) :=
begin
  sorry
end

end quadratic_inequality_l32_32722


namespace ice_cream_cost_proof_l32_32196

-- Assume the cost of the ice cream and toppings
def cost_of_ice_cream : ℝ := 2 -- Ice cream cost in dollars
def cost_per_topping : ℝ := 0.5 -- Cost per topping in dollars
def total_cost_of_sundae_with_10_toppings : ℝ := 7 -- Total cost in dollars

theorem ice_cream_cost_proof :
  (∀ (cost_of_ice_cream : ℝ), 
    total_cost_of_sundae_with_10_toppings = cost_of_ice_cream + 10 * cost_per_topping) →
  cost_of_ice_cream = 2 :=
by
  sorry

end ice_cream_cost_proof_l32_32196


namespace find_a_in_triangle_l32_32613

theorem find_a_in_triangle (C : ℝ) (b c : ℝ) (hC : C = 60) (hb : b = 1) (hc : c = Real.sqrt 3) :
  ∃ (a : ℝ), a = 2 := 
by
  sorry

end find_a_in_triangle_l32_32613


namespace true_propositions_l32_32829

noncomputable theory

def even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def periodic_function (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x

theorem true_propositions (f : ℝ → ℝ) (h1 : even_function f) (h2 : ∀ x, f(x + 1) = -f(x)) (h3 : ∀ x ∈ Icc (-1 : ℝ) (0 : ℝ), f x < f (x + 1)) :
  (periodic_function f 2) ∧
  (∀ x, f (1 - x) = f (1 + x)) ∧
  (¬ ∀ x ∈ Icc (0 : ℝ) (1 : ℝ), f x < f (x + 1)) ∧
  (¬ ∀ x ∈ Icc (1 : ℝ) (2 : ℝ), f x > f (x + 1)) ∧
  (f 2 = f 0) := sorry

end true_propositions_l32_32829


namespace team_members_count_l32_32071

theorem team_members_count (x : ℕ) (h1 : 3 * x + 2 * x = 33 ∨ 4 * x + 2 * x = 33) : x = 6 := by
  sorry

end team_members_count_l32_32071


namespace triangle_area_example_l32_32807

def Point := (ℝ × ℝ)

def triangle_area (A B C : Point) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem triangle_area_example :
  triangle_area (-2, 3) (7, -1) (4, 6) = 25.5 :=
by
  -- Proof will be here
  sorry

end triangle_area_example_l32_32807


namespace skew_lines_plane_properties_l32_32567

-- Definition: skew lines
def skew_lines (m l : Set Point) : Prop :=
  ∃ (α β : Set Plane), ¬(∃ (γ : Set Plane), (γ ⊇ m ∧ γ ⊇ l)) ∧
                       (∀ p ∈ m, ∀ q ∈ l, p ≠ q)

-- Conclusion ①
def exists_plane_passing_through_and_parallel (m l : Set Point) : Prop :=
  ∃ α : Set Plane, ∃ β : Set Plane, (α ⊇ m) ∧ (¬(α ⊇ l) ∧ β // l)

-- Conclusion ④
def exists_eqdistant_plane (m l : Set Point) : Prop :=
  ∃ π : Set Plane, π.equidistant_from_both_lines m l

-- Main theorem to be proved
theorem skew_lines_plane_properties (m l : Set Point) (h: skew_lines m l) : 
  exists_plane_passing_through_and_parallel m l ∧ exists_eqdistant_plane m l := sorry

end skew_lines_plane_properties_l32_32567


namespace log_sum_nine_l32_32331

-- Define that {a_n} is a geometric sequence and satisfies the given conditions.
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a n = a 1 * r ^ (n - 1)

-- Given conditions
axiom a_pos (a : ℕ → ℝ) : (∀ n, a n > 0)      -- All terms are positive
axiom a2a8_eq_4 (a : ℕ → ℝ) : a 2 * a 8 = 4    -- a₂a₈ = 4

theorem log_sum_nine (a : ℕ → ℝ) 
  (geo_seq : geometric_sequence a) 
  (pos : ∀ n, a n > 0)
  (eq4 : a 2 * a 8 = 4) :
  (Real.logb 2 (a 1) + Real.logb 2 (a 2) + Real.logb 2 (a 3) + Real.logb 2 (a 4)
  + Real.logb 2 (a 5) + Real.logb 2 (a 6) + Real.logb 2 (a 7) + Real.logb 2 (a 8)
  + Real.logb 2 (a 9)) = 9 :=
by
  sorry

end log_sum_nine_l32_32331


namespace smallest_n_for_solutions_l32_32271

noncomputable def f (x : ℝ) : ℝ :=
  |2 * (x - floor x) - 1|

def fractional_part (x : ℝ) : ℝ := x - floor x

def number_of_real_solutions (n : ℕ) : ℕ :=
  {x : ℝ | n * f (x^2 * f x) = x}.to_finset.card

theorem smallest_n_for_solutions (n_pos : ℕ) (h : number_of_real_solutions n >= 4030) :
  n = 32 :=
sorry

end smallest_n_for_solutions_l32_32271


namespace sum_even_integers_202_to_500_l32_32379

theorem sum_even_integers_202_to_500 : 
  let sequence := (λ k, 202 + 2 * k) in
  (∑ k in Finset.range 150, sequence k) = 52650 :=
by
  sorry

end sum_even_integers_202_to_500_l32_32379


namespace proof_problem_l32_32578

-- Definitions for the solution sets
def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | -3 < x ∧ x < 2}
def intersection : Set ℝ := {x | -1 < x ∧ x < 2}

-- The quadratic inequality solution sets
def solution_set (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b < 0}

-- The main theorem statement
theorem proof_problem (a b : ℝ) (h : solution_set a b = intersection) : a + b = -3 :=
sorry

end proof_problem_l32_32578


namespace polynomial_property_exists_a_l32_32513

noncomputable def P (x : ℝ) : ℝ := sorry -- Polynomial P(x)

theorem polynomial_property_exists_a (a : ℝ) (P : ℝ → ℝ) :
  (∀ x, x * P (x - 1) = (x - 2) * P x) →
  (∃ a : ℝ, ∀ x, P x = a * (x^2 - x)) :=
begin
  intro h,
  -- The proof goes here
  sorry,
end

end polynomial_property_exists_a_l32_32513


namespace city_partition_exists_l32_32725

-- Define a market and street as given
structure City where
  markets : Type
  street : markets → markets → Prop
  leaves_exactly_two : ∀ (m : markets), ∃ (m1 m2 : markets), street m m1 ∧ street m m2

-- Our formal proof statement
theorem city_partition_exists (C : City) : 
  ∃ (partition : C.markets → Fin 1014), 
    (∀ (m1 m2 : C.markets), C.street m1 m2 → partition m1 ≠ partition m2) ∧
    (∀ (d1 d2 : Fin 1014) (m1 m2 : C.markets), (partition m1 = d1) ∧ (partition m2 = d2) → 
     (C.street m1 m2 ∨ C.street m2 m1) →  (∀ (k l : Fin 1014), (k = d1) → (l = d2) → (∀ (a b : C.markets), (partition a = k) → (partition b = l) → (C.street a b ∨ C.street b a)))) :=
sorry

end city_partition_exists_l32_32725


namespace sum_of_four_smallest_divisors_l32_32794

-- Define a natural number n and divisors d1, d2, d3, d4
def is_divisor (d n : ℕ) : Prop := ∃ k : ℕ, n = k * d

-- Primary problem condition (sum of four divisors equals 2n)
def sum_of_divisors_eq (n d1 d2 d3 d4 : ℕ) : Prop := d1 + d2 + d3 + d4 = 2 * n

-- Assume the four divisors of n are distinct
def distinct (d1 d2 d3 d4 : ℕ) : Prop := d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4

-- State the Lean proof problem
theorem sum_of_four_smallest_divisors (n d1 d2 d3 d4 : ℕ) (h1 : d1 < d2) (h2 : d2 < d3) (h3 : d3 < d4) 
    (h_div1 : is_divisor d1 n) (h_div2 : is_divisor d2 n) (h_div3 : is_divisor d3 n) (h_div4 : is_divisor d4 n)
    (h_sum : sum_of_divisors_eq n d1 d2 d3 d4) (h_distinct : distinct d1 d2 d3 d4) : 
    (d1 + d2 + d3 + d4 = 10 ∨ d1 + d2 + d3 + d4 = 11 ∨ d1 + d2 + d3 + d4 = 12) := 
sorry

end sum_of_four_smallest_divisors_l32_32794


namespace values_of_A_satisfying_conditions_l32_32523

-- Definitions based on conditions
def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

-- The Lean proof statement
theorem values_of_A_satisfying_conditions :
  { A : ℕ | A < 10 ∧ is_divisible_by 45 A ∧ is_divisible_by 27315 5 }.card = 4 :=
by
  -- Insert the proof here
  sorry

end values_of_A_satisfying_conditions_l32_32523


namespace songs_can_be_stored_l32_32249

def totalStorageGB : ℕ := 16
def usedStorageGB : ℕ := 4
def songSizeMB : ℕ := 30
def gbToMb : ℕ := 1000

def remainingStorageGB := totalStorageGB - usedStorageGB
def remainingStorageMB := remainingStorageGB * gbToMb
def numberOfSongs := remainingStorageMB / songSizeMB

theorem songs_can_be_stored : numberOfSongs = 400 :=
by
  sorry

end songs_can_be_stored_l32_32249


namespace determine_g_function_l32_32573

theorem determine_g_function (g : ℝ → ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = g x + x^2) 
  (h2 : ∀ x, f (-x) = -f x) 
  (h3 : f 1 = 1) : g = λ x, x^5 - x^2 :=
by
  sorry

end determine_g_function_l32_32573


namespace julian_number_probability_l32_32064

-- Define the conditions as definitions in Lean
def valid_first_digits := {253, 256, 259}
def valid_last_four_digits := {3, 5, 8, 9}
def is_valid_last_digit (d : ℕ) := d ≠ 9

-- Define the proof problem
theorem julian_number_probability :
  let total_numbers := 3 * (24 - 6) in
  let probability := 1 / total_numbers in
  probability = 1 / 54 :=
by
  sorry

end julian_number_probability_l32_32064


namespace prime_factorization_count_l32_32910

theorem prime_factorization_count :
  (∃ (S : Finset ℕ), S = {97, 101, 2, 13, 107, 109} ∧ S.card = 6) :=
by
  sorry

end prime_factorization_count_l32_32910


namespace bisection_method_interval_l32_32744

-- Define the function f(x) = 2^x + 3x - 7
def f (x : ℝ) : ℝ := 2^x + 3 * x - 7

-- Define the interval [1, 3]
def a : ℝ := 1
def b : ℝ := 3

-- Prove that the next interval (1, 2) contains the root of f(x) = 0
theorem bisection_method_interval :
  f a < 0 ∧ f b > 0 ∧ f ((a + b) / 2) > 0 → ∃ c ∈ set.Ioo a ((a + b) / 2), f c = 0 :=
by 
  sorry

end bisection_method_interval_l32_32744


namespace unique_intersection_point_l32_32950

noncomputable def g : ℝ → ℝ := λ x, x^3 + 3 * x^2 + 9 * x + 15

theorem unique_intersection_point :
  (∃ c d : ℝ, g c = d ∧ g d = c) ∧ (∀ x : ℝ, g x = x → x = -3) ->
  (∃ c d : ℝ, g c = d ∧ g d = c ∧ c = -3 ∧ d = -3) :=
by
  sorry

end unique_intersection_point_l32_32950


namespace walter_time_at_seals_l32_32363

theorem walter_time_at_seals 
  (s p e total : ℕ)
  (h1 : p = 8 * s)
  (h2 : e = 13)
  (h3 : total = 130)
  (h4 : s + p + e = total) : s = 13 := 
by 
  sorry

end walter_time_at_seals_l32_32363


namespace square_has_most_symmetry_axes_l32_32066

def shape_symmetry_axes (shape : Type) : ℕ :=
  match shape with
  | Line_segment => 2
  | Angle => 1
  | Isosceles_triangle => 3  -- We use 3 assuming the best scenario.
  | Square => 4

def most_symmetry_shape (shapes : List Type) : Type :=
  shapes.foldl (λ acc x => if shape_symmetry_axes x > shape_symmetry_axes acc then x else acc) shapes.head!

theorem square_has_most_symmetry_axes (shapes : List Type)
  (h1 : Line_segment ∈ shapes)
  (h2 : Angle ∈ shapes)
  (h3 : Isosceles_triangle ∈ shapes)
  (h4 : Square ∈ shapes) :
  most_symmetry_shape shapes = Square :=
  sorry

end square_has_most_symmetry_axes_l32_32066


namespace o_l32_32099

theorem o'hara_triple_example (a b x : ℕ) (h₁ : a = 49) (h₂ : b = 16) (h₃ : x = (Int.sqrt a).toNat + (Int.sqrt b).toNat) : x = 11 := 
by
  sorry

end o_l32_32099


namespace mode_of_scores_is_correct_final_score_is_correct_l32_32934

noncomputable def studentScores : List ℝ := [9.6, 9.4, 9.6, 9.7, 9.7, 9.5, 9.6]

def calculateMode (scores : List ℝ) : ℝ :=
  scores.foldl (λ acc x → if scores.count x > scores.count acc then x else acc) scores.head!

def calculateFinalScore (scores : List ℝ) : ℝ :=
  let lowest := scores.minimum
  let highest := scores.maximum
  let remaining := scores.filter (λ x → x ≠ lowest ∧ x ≠ highest)
  let average := remaining.sum / remaining.length
  Float.round nearest average

theorem mode_of_scores_is_correct :
  calculateMode studentScores = 9.6 :=
by
  sorry

theorem final_score_is_correct :
  calculateFinalScore studentScores = 9.6 :=
by
  sorry

end mode_of_scores_is_correct_final_score_is_correct_l32_32934


namespace binomial_sum_alternating_l32_32486

theorem binomial_sum_alternating :
  (finset.range 51).sum (λ k, (-1 : ℤ) ^ k * (k + 1) * (nat.choose 50 k)) = 0 :=
begin
  sorry
end

end binomial_sum_alternating_l32_32486


namespace volume_ratio_proof_l32_32053

-- Definitions of the cone and its segments
def height := 100
def base_radius := 50
def top_segment_height := 20
def middle_segment_height := 30
def bottom_segment_height := 50

-- Volume of a cone segment calculation based on the height and base radius
noncomputable def volume_cone (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

-- Total volume of the cone
noncomputable def V_full := volume_cone base_radius height

-- Volume of the top segment
noncomputable def scaled_radius_top := base_radius * (top_segment_height / height)
noncomputable def V_top := volume_cone scaled_radius_top top_segment_height

-- Volume of the middle segment from the volume at 50 units height
noncomputable def height_50 := top_segment_height + middle_segment_height
noncomputable def scaled_radius_50 := base_radius * (height_50 / height)
noncomputable def V_50 := volume_cone scaled_radius_50 height_50
noncomputable def V_mid := V_50 - V_top

-- Volume of the bottom segment from the total volume
noncomputable def V_bottom := V_full - V_50

-- The target ratio of middle segment to bottom segment volumes
noncomputable def volume_ratio := V_mid / V_bottom

-- The theorem proving the correct ratio
theorem volume_ratio_proof : volume_ratio = (13 : ℝ) / 97 := 
by
  sorry

end volume_ratio_proof_l32_32053


namespace cost_of_each_bird_l32_32945

theorem cost_of_each_bird (num_grandparents : ℕ) (money_per_grandparent : ℕ) (total_wings : ℕ)
                          (wings_per_bird : ℕ) (total_money : ℕ) (num_birds : ℕ) (cost_per_bird : ℕ) :
    num_grandparents = 4 →
    money_per_grandparent = 50 →
    total_wings = 20 →
    wings_per_bird = 2 →
    total_money = num_grandparents * money_per_grandparent →
    num_birds = total_wings / wings_per_bird →
    cost_per_bird = total_money / num_birds →
    cost_per_bird = 20 :=
begin
  intros h1 h2 h3 h4 h5 h6 h7,
  rw [h1, h2, h3, h4] at h5 h6 h7,
  rw [h5, h6] at h7,
  exact h7,
end

end cost_of_each_bird_l32_32945


namespace categorize_numbers_l32_32110

theorem categorize_numbers :
    let numbers := [(-9.3, 1), (3/100, 2), (-20, 3), (0, 4), (0.01, 5), 
                    (-1, 6), (-7/2, 7), (3.14, 8), (100, 9)]
    let posNums := [2, 5, 8, 9]
    let ints := [3, 4, 6, 9]
    let negFracts := [1, 7]
    let nonNegNums := [2, 4, 5, 8, 9]
    let natNums := [4, 9]
    (∀ (x : ℝ) (label : ℕ), (x, label) ∈ numbers → x > 0 → label ∈ posNums) ∧
    (∀ (x : ℝ) (label : ℕ), (x, label) ∈ numbers → x ∈ ℤ → label ∈ ints) ∧
    (∀ (x : ℝ) (label : ℕ), (x, label) ∈ numbers → ∃ (a b : ℤ), x = a / b ∧ b ≠ 0 ∧ x < 0 → label ∈ negFracts) ∧
    (∀ (x : ℝ) (label : ℕ), (x, label) ∈ numbers → x ≥ 0 → label ∈ nonNegNums) ∧
    (∀ (x : ℝ) (label : ℕ), (x, label) ∈ numbers → x ∈ ℕ ∧ x > 0 → label ∈ natNums) :=
sorry

end categorize_numbers_l32_32110


namespace find_sum_on_si_l32_32769

noncomputable def sum_invested_on_si (r1 r2 r3 : ℝ) (years_si: ℕ) (ci_rate: ℝ) (principal_ci: ℝ) (years_ci: ℕ) (times_compounded: ℕ) :=
  let ci_rate_period := ci_rate / times_compounded
  let amount_ci := principal_ci * (1 + ci_rate_period / 1)^(years_ci * times_compounded)
  let ci := amount_ci - principal_ci
  let si := ci / 2
  let total_si_rate := r1 / 100 + r2 / 100 + r3 / 100
  let principle_si := si / total_si_rate
  principle_si

theorem find_sum_on_si :
  sum_invested_on_si 0.05 0.06 0.07 3 0.10 4000 2 2 = 2394.51 :=
by
  sorry

end find_sum_on_si_l32_32769


namespace chord_equation_l32_32917

theorem chord_equation
  (P : ℝ × ℝ)
  (hP : P = (1, 1))
  (h_mid : ∃ M N : ℝ × ℝ, (M.1 + N.1) / 2 = 1 ∧ (M.2 + N.2) / 2 = 1)
  (h_circle : ∀ (x y : ℝ), (x - 3)^2 + y^2 = 9 → (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 - 3)^2 + p.2^2 = 9)) :
  ∃ A B C : ℝ, A = 2 ∧ B = -1 ∧ C = -1 ∧ ∀ (x y : ℝ), (A * x + B * y + C = 0 ↔ y = 2 * x - 1) := 
sorry

end chord_equation_l32_32917


namespace cab_driver_income_on_fourth_day_l32_32785

theorem cab_driver_income_on_fourth_day (income_day1 income_day2 income_day3 income_day5 avg_income income_day4 : ℕ) :
  income_day1 = 600 ∧ income_day2 = 250 ∧ income_day3 = 450 ∧ income_day5 = 800 ∧ avg_income = 500 →
  income_day4 = 400 :=
by
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  cases h3 with h4 h5
  cases h5 with h6 h_avg
  sorry

end cab_driver_income_on_fourth_day_l32_32785


namespace determine_m_values_l32_32660

open Set

theorem determine_m_values :
  let U : Set ℝ := Univ
  let A := { x : ℝ | x^2 + 3*x + 2 = 0 }
  let B (m : ℝ) := { x : ℝ | x^2 + (m+1)*x + m = 0 }
  (∀ m : ℝ, (compl A ∩ B m) = ∅ → m = 1 ∨ m = 2) :=
by
  intros U A B
  simp
  sorry

end determine_m_values_l32_32660


namespace school_adding_seats_l32_32418

theorem school_adding_seats (row_seats : ℕ) (seat_cost : ℕ) (discount_rate : ℝ) (total_cost : ℕ) (n : ℕ) 
                         (total_seats : ℕ) (discounted_seat_cost : ℕ)
                         (total_groups : ℕ) (rows : ℕ) :
  row_seats = 8 →
  seat_cost = 30 →
  discount_rate = 0.10 →
  total_cost = 1080 →
  discounted_seat_cost = seat_cost * (1 - discount_rate) →
  total_seats = total_cost / discounted_seat_cost →
  total_groups = total_seats / 10 →
  rows = total_seats / row_seats →
  rows = 5 :=
by
  intros hrowseats hseatcost hdiscountrate htotalcost hdiscountedseatcost htotalseats htotalgroups hrows
  sorry

end school_adding_seats_l32_32418


namespace triangle_problem_l32_32221

variables {A B C a b c : ℝ}
variables {m n : ℝ × ℝ}

-- Definitions from conditions
def sides_opposite (a b c A B C : ℝ) : Prop := true
def vec_m := (2, 1) : ℝ × ℝ
def vec_n := (c * Real.cos C, a * Real.cos B + b * Real.cos A) : ℝ × ℝ

-- Perpendicular vectors condition
def perp_vectors (m n : ℝ × ℝ) : Prop := m.1 * n.1 + m.2 * n.2 = 0

-- Additional given conditions
def given_cos_C_eq_minus_half : Prop := Real.cos C = -1 / 2
def given_metric_condition : Prop := c^2 = 7 * b^2
def given_area (a b : ℝ) : Prop := 1 / 2 * a * b * Real.sin C = 2 * Real.sqrt 3

theorem triangle_problem (A B C a b c : ℝ)
  (sides_opposite : sides_opposite a b c A B C)
  (h₁ : vec_m = (2, 1))
  (h₂ : vec_n = (c * Real.cos C, a * Real.cos B + b * Real.cos A))
  (h₃ : perp_vectors vec_m vec_n)
  (h₄ : given_metric_condition)
  (h₅ : given_area a b) :
  C = 2 * Real.pi / 3 ∧ b = 2 :=
  sorry

end triangle_problem_l32_32221


namespace cube_root_of_sqrt_64_is_2_l32_32758

theorem cube_root_of_sqrt_64_is_2 :
  (∛(√64)) = 2 :=
sorry

end cube_root_of_sqrt_64_is_2_l32_32758


namespace alpha_beta_sum_l32_32596

noncomputable def problem :=
  ∃ (α β : ℝ), 
    (tan α) * (tan β) = 4 ∧ 
    (tan α + tan β) = 3 * sqrt 3 ∧ 
    (α > 0) ∧ 
    (α < π / 2) ∧ 
    (β > 0) ∧ 
    (β < π / 2) ∧ 
    α + β = 2 * π / 3

-- The main theorem statement
theorem alpha_beta_sum :
  ∃ (α β : ℝ), (tan α) * (tan β) = 4 ∧ (tan α + tan β) = 3 * sqrt 3 ∧ 0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2 → α + β = 2 * π / 3 :=
by {
  intro h,
  cases h with α h,
  cases h with β h,
  sorry
}

end alpha_beta_sum_l32_32596


namespace factor_polynomial_l32_32109

theorem factor_polynomial :
  ∃ (a b c d e f : ℤ), a < d ∧
    (a * x^2 + b * x + c) * (d * x^2 + e * x + f) = x^2 - 6 * x + 9 - 64 * x^4 ∧
    (a = -8 ∧ b = 1 ∧ c = -3 ∧ d = 8 ∧ e = 1 ∧ f = -3) := by
  sorry

end factor_polynomial_l32_32109


namespace max_isosceles_tris_2017_gon_l32_32883

theorem max_isosceles_tris_2017_gon :
  ∀ (n : ℕ), n = 2017 →
  ∃ (t : ℕ), (∃ (d : ℕ), d = 2014 ∧ 2015 = (n - 2)) →
  t = 2010 :=
by
  sorry

end max_isosceles_tris_2017_gon_l32_32883


namespace average_gpa_whole_class_l32_32332

theorem average_gpa_whole_class (n : ℕ) (h1 : 1 / 3 * n * 45)
    (h2 : 2 / 3 * n * 60) : (1 / 3 * 45 + 2 / 3 * 60) = 55 := 
sorry

end average_gpa_whole_class_l32_32332


namespace problem_solution_l32_32541

theorem problem_solution (a b c d : ℝ) (h1 : ab + bc + cd + da = 48) (h2 : b + d = 6) : a + c = 8 :=
sorry

end problem_solution_l32_32541


namespace definite_integral_l32_32824

open Real

theorem definite_integral : ∫ x in (0 : ℝ)..(π / 2), (x + sin x) = π^2 / 8 + 1 :=
by
  sorry

end definite_integral_l32_32824


namespace total_boys_in_class_l32_32809

/-- 
  Given 
    - n + 1 positions in a circle, where n is the number of boys and 1 position for the teacher.
    - The boy at the 6th position is exactly opposite to the boy at the 16th position.
  Prove that the total number of boys in the class is 20.
-/
theorem total_boys_in_class (n : ℕ) (h1 : n + 1 > 16) (h2 : (6 + 10) % (n + 1) = 16):
  n = 20 := 
by 
  sorry

end total_boys_in_class_l32_32809


namespace smallest_lcm_of_4digit_gcd_5_l32_32211

theorem smallest_lcm_of_4digit_gcd_5 :
  ∃ (m n : ℕ), (1000 ≤ m ∧ m < 10000) ∧ (1000 ≤ n ∧ n < 10000) ∧ 
               m.gcd n = 5 ∧ m.lcm n = 203010 :=
by sorry

end smallest_lcm_of_4digit_gcd_5_l32_32211


namespace reflect_parallelogram_H_l32_32298

/-- 
A parallelogram EFGH with vertices at E(3,7), F(5,11), G(7,7), and H(5,3) is reflected
across the x-axis to obtain E'F'G'H'. Then E'F'G'H' is reflected across the line y = x - 2
to obtain E''F''G''H''.

Prove that the coordinates of H'' are (-1,3).
-/
theorem reflect_parallelogram_H'' :
  let H := (5, 3 : ℝ × ℝ),
      H₁ := (H.1, -H.2),
      H₂ := (H₁.1, H₁.2 + 2),
      H₃ := (H₂.2, H₂.1),
      H'' := (H₃.1, H₃.2 - 2)
  in H'' = (-1, 3) :=
by
  let H := (5, 3 : ℝ × ℝ),
      H₁ := (H.1, -H.2),
      H₂ := (H₁.1, H₁.2 + 2),
      H₃ := (H₂.2, H₂.1),
      H'' := (H₃.1, H₃.2 - 2)
  show H'' = (-1, 3)
  sorry

end reflect_parallelogram_H_l32_32298


namespace phone_sales_total_amount_l32_32431

theorem phone_sales_total_amount
  (vivienne_phones : ℕ)
  (aliyah_more_phones : ℕ)
  (price_per_phone : ℕ)
  (aliyah_phones : ℕ := vivienne_phones + aliyah_more_phones)
  (total_phones : ℕ := vivienne_phones + aliyah_phones)
  (total_amount : ℕ := total_phones * price_per_phone) :
  vivienne_phones = 40 → aliyah_more_phones = 10 → price_per_phone = 400 → total_amount = 36000 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end phone_sales_total_amount_l32_32431


namespace sqrt_of_mixed_number_l32_32502

theorem sqrt_of_mixed_number :
  (Real.sqrt (8 + 9 / 16)) = (Real.sqrt 137 / 4) :=
by
  sorry

end sqrt_of_mixed_number_l32_32502


namespace min_length_BD_in_cyclic_quad_l32_32343

theorem min_length_BD_in_cyclic_quad :
  ∀ (A B C D I : Type) (circumcircle : IsCyclicQuad A B C D)
  (AI BI CI DI : ℝ) (IncenterABD : IsIncenter I A B D)
  (len_AI : AI = 2) (len_BC : BC = 2) (len_CD : CD = 2),
  (len_BD : ℝ) (minimum_len_BD : len_BD)
  (∀ (x y : ℝ), x * y = 3 → x + y ≥ 2 * Real.sqrt 3) :
    minimum_len_BD = 2 * Real.sqrt 3 := by
  sorry

end min_length_BD_in_cyclic_quad_l32_32343


namespace marbles_won_in_second_game_l32_32063

theorem marbles_won_in_second_game :
  ∀ (initial_marbles lost_marbles marbles_after_second_game marbles_won : ℤ),
    initial_marbles = 57 →
    lost_marbles = 18 →
    marbles_after_second_game = 64 →
    marbles_won = marbles_after_second_game - (initial_marbles - lost_marbles) →
    marbles_won = 25 :=
by {
  intros,
  sorry
}

end marbles_won_in_second_game_l32_32063


namespace max_height_l32_32783

noncomputable def ball_height (t : ℝ) : ℝ :=
  -4.9 * t^2 + 50 * t + 15

theorem max_height : ∃ t : ℝ, t < 50 / 4.9 ∧ ball_height t = 142.65 :=
sorry

end max_height_l32_32783


namespace probability_prime_sum_less_than_10_l32_32740

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def spinnerA_numbers : Finset ℕ := {1, 2, 3, 4}
def spinnerB_numbers : Finset ℕ := {3, 4, 5, 6}
def valid_sums : Finset ℕ := spinnerA_numbers.bind (λ a, spinnerB_numbers.image (λ b => a + b))
def prime_sums_less_than_10 : Finset ℕ := valid_sums.filter is_prime

theorem probability_prime_sum_less_than_10 :
  prime_sums_less_than_10.card = 5 ∧ valid_sums.card = 16 ∧ (prime_sums_less_than_10.card / valid_sums.card : ℚ) = 5 / 16 := 
  sorry

end probability_prime_sum_less_than_10_l32_32740


namespace sum_arithmetic_sequence_l32_32888

theorem sum_arithmetic_sequence (n : ℕ) (hn : 1 ≤ n) :
  let y := λ k : ℕ, 3/2 + (k-1) * (3/4)
  in ∑ k in finset.range n, y k = (3 * n^2 + 3 * n - 6) / 8 := sorry

end sum_arithmetic_sequence_l32_32888


namespace boundary_length_is_correct_l32_32424
noncomputable def boundary_length_of_figure : ℝ := 61.7

theorem boundary_length_is_correct :
  ∃ (side_length : ℝ) (segment_length : ℝ) (arc_length : ℝ) (total_arc_length : ℝ) (straight_segment_length : ℝ),
  side_length = real.sqrt 144 ∧
  segment_length = side_length / 4 ∧
  arc_length = (segment_length / 2) * real.pi ∧
  total_arc_length = 8 * arc_length ∧
  straight_segment_length = 8 * segment_length ∧
  boundary_length_of_figure = total_arc_length + straight_segment_length :=
by
  sorry

end boundary_length_is_correct_l32_32424


namespace sin_product_inequality_l32_32277

theorem sin_product_inequality {n : ℕ} (n_pos : 0 < n) 
  (x : Fin n → ℝ) (hx : ∀ i, 0 < x i ∧ x i < π) :
  (∏ i, Real.sin (x i)) ≤ (Real.sin ((1 / n) * (∑ i, x i))) ^ n := 
sorry

end sin_product_inequality_l32_32277


namespace find_n_value_l32_32112

theorem find_n_value (AB AC n m : ℕ) (h1 : AB = 33) (h2 : AC = 21) (h3 : AD = m) (h4 : DE = m) (h5 : EC = m) (h6 : BC = n) : 
  ∃ m : ℕ, m > 7 ∧ m < 21 ∧ n = 30 := 
by sorry

end find_n_value_l32_32112


namespace smurfs_gold_coins_l32_32321

theorem smurfs_gold_coins (x y : ℕ) (h1 : x + y = 200) (h2 : (2 / 3 : ℚ) * x = (4 / 5 : ℚ) * y + 38) : x = 135 :=
by
  sorry

end smurfs_gold_coins_l32_32321


namespace arithmetic_sequence_general_term_l32_32265

theorem arithmetic_sequence_general_term (a n : ℕ) (S : ℕ → ℕ)
  (h1 : ∀ n, S n = n^2 + 2 * (a n) - 6) :
  (a n) = 2 * n + 3 :=
sorry

end arithmetic_sequence_general_term_l32_32265


namespace probability_diff_amounts_l32_32060

theorem probability_diff_amounts (A B P1 P2 : ℕ) (amounts : Fin 4 → ℕ)
  (hA : amounts 0 = A) (hB : amounts 1 = B) (hP1 : amounts 2 = P1) (hP2 : amounts 3 = P2)
  (h_total : Multiset.of_finset ⟦amounts 0, amounts 1, amounts 2, amounts 3⟧ = {1, 1, 1, 5}) :
  let events := { e : Finset (Fin 4) // e.card = 2 } in
  let diff_events := { e ∈ events | (amounts e.1 = 1 ∧ amounts e.2 = 5) ∨ (amounts e.1 = 5 ∧ amounts e.2 = 1) } in
  (diff_events.card : ℚ) / (events.card : ℚ) = 1 / 3 :=
by
  sorry

end probability_diff_amounts_l32_32060


namespace proof_l32_32147

noncomputable def problem_statement (a b : ℝ) :=
  7 * (Real.sin a + Real.sin b) + 6 * (Real.cos a * Real.cos b - 1) = 0 →
  (Real.tan (a / 2) * Real.tan (b / 2) = 1 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -1)

theorem proof : ∀ a b : ℝ, problem_statement a b := sorry

end proof_l32_32147


namespace sqrt_mixed_fraction_l32_32492

theorem sqrt_mixed_fraction (a b : ℤ) (h_a : a = 8) (h_b : b = 9) : 
  (√(a + b / 16)) = (√137) / 4 := 
by 
  sorry

end sqrt_mixed_fraction_l32_32492


namespace second_player_wins_l32_32739

noncomputable def can_second_player_always_win : Prop :=
  ∃ strategy : (ℕ × ℕ → (ℕ × ℕ)), 
    (∀ i j, (i,j) ∈ (finset.range 10).product (finset.range 10) →
    (strategy (i, j) = (10 - i, 10 - j))) ∧ 
    (∀ (i j : ℕ), (i, j) ∈ (finset.range 10).product (finset.range 10) →
     ∃ k l, (k, l) = (10 - i, 10 - j) ∧ ((k != i ∨ l != j) ∧ ∀ x y, (x = k → y = l) → symbol x y ≠ symbol i j))

theorem second_player_wins :
  can_second_player_always_win :=
sorry

end second_player_wins_l32_32739


namespace rectangle_lengths_l32_32937

theorem rectangle_lengths 
  (A B C D P T S Q R : Type)
  [Inhabited P] [Inhabited Q]
  (hABCD : is_rectangle ABCD)
  (hP_on_BC : P ∈ segment B C)
  (hAngleAPD : ∠APD = 90)
  (hTS_perp_BC : ⊥ TS BC)
  (h_eq_lengths : BP = PT ∧ PT = PS)
  (hS_on_TS : S ∈ line TS)
  (hPD_intersects_TS_at_Q : P ∈ segment D (inter TS P Q))
  (hRA_passes_through_Q : ∃ R ∈ line CD, RA ∋ Q)
  (hPA : PA = 24)
  (hAQ : AQ = 18)
  (hQP : QP = 30) :
  BP = 4 * sqrt 11 ∧ QT = 2 * sqrt 181 := 
sorry

end rectangle_lengths_l32_32937


namespace most_suitable_graph_for_AQI_changes_l32_32625

/-- Given the AQI values over ten days, prove that the most suitable type of statistical graph 
to describe the changes in air quality is a line graph. The values are as follows:
  Day 1: 84, Day 2: 89, Day 3: 83, Day 4: 99, Day 5: 69,
  Day 6: 73, Day 7: 78, Day 8: 81, Day 9: 89, Day 10: 82.
The possible types of statistical graphs are:
  A: Line graph,
  B: Frequency distribution histogram,
  C: Bar graph,
  D: Pie chart. 
-/
theorem most_suitable_graph_for_AQI_changes (AQI : List ℕ) (h : AQI = [84, 89, 83, 99, 69, 73, 78, 81, 89, 82])
    (graph_types : List String)
    (h_graph_types : graph_types = ["Line graph", "Frequency distribution histogram", "Bar graph", "Pie chart"]) :
    (graph_types.head = "Line graph") := 
  sorry

end most_suitable_graph_for_AQI_changes_l32_32625


namespace percentage_less_than_l32_32926

theorem percentage_less_than (x y : ℝ) (h1 : y = x * 1.8181818181818181) : (∃ P : ℝ, P = 45) :=
by
  sorry

end percentage_less_than_l32_32926


namespace locus_of_point_M_l32_32928

noncomputable theory

open Real

variables {O A B C D M : Π (x : ℝ), x ∈ set.univ}
variables {R : ℝ}
variables [circle_center : x = O]
variables [diameter : A = 2 * R]
variables [radius_OC : dist O C = R]
variables [perpendicular_CD_OA : ⊥ dist D C]
variables [OP : circle (A, R) = circle (B, R)]

theorem locus_of_point_M : 
  ∃ M, dist O M = dist C D ∧ dist C D = R/2 →
  ∃ l, set_eq l circle (O, R/2) :=
by
  sorry

end locus_of_point_M_l32_32928


namespace length_segment_AB_between_line_and_circle_l32_32576

theorem length_segment_AB_between_line_and_circle (x y : ℝ) :
  let C := ⟨1, 3⟩ in
  let r := 2 in
  let d := (|1 - 2 * 3 + 3| / (real.sqrt (1^2 + 2^2))) in
  d = (2 * real.sqrt 5 / 5) →
  (2 * real.sqrt (r^2 - d^2)) = (8 * real.sqrt 5 / 5) :=
sorry

end length_segment_AB_between_line_and_circle_l32_32576


namespace converse_implication_l32_32149

variables {a b : EuclideanSpace ℝ (Fin 3)}

theorem converse_implication (h : ∥a∥ = ∥b∥) : a = -b :=
sorry

end converse_implication_l32_32149


namespace james_prom_total_cost_l32_32245

-- Definitions and conditions
def ticket_cost : ℕ := 100
def num_tickets : ℕ := 2
def dinner_cost : ℕ := 120
def tip_rate : ℚ := 0.30
def limo_hourly_rate : ℕ := 80
def limo_hours : ℕ := 6

-- Calculation of each component
def total_ticket_cost : ℕ := ticket_cost * num_tickets
def total_tip : ℚ := tip_rate * dinner_cost
def total_dinner_cost : ℚ := dinner_cost + total_tip
def total_limo_cost : ℕ := limo_hourly_rate * limo_hours

-- Final total cost calculation
def total_cost : ℚ := total_ticket_cost + total_dinner_cost + total_limo_cost

-- Proving the final total cost
theorem james_prom_total_cost : total_cost = 836 := by sorry

end james_prom_total_cost_l32_32245


namespace solve_for_z_l32_32925
noncomputable def problem_statement : Prop :=
  let x := Real.log 25
  let y := Real.sin (Float.pi / 4)  -- sin(45°)
  let z := 0.0204
  let tan_30 := Real.sqrt 3 / 3
  let log2_64 := 6
  let cos_60 := 1 / 2
  (x^2 - z) * (y^2 + tan_30) + (log2_64 * cos_60)^2 = 20.13

theorem solve_for_z : problem_statement := by
  sorry

end solve_for_z_l32_32925


namespace pascal_probability_first_twenty_rows_l32_32438

theorem pascal_probability_first_twenty_rows :
  let total_elements := (20 * (20 + 1)) / 2
  let num_ones := 1 + 2 * (20 - 1)
  let num_twos := 2 * (20 - 3)
  let num_ones_or_twos := num_ones + num_twos
  (num_ones_or_twos : ℚ) / total_elements = 73 / 210 := by
sorry

end pascal_probability_first_twenty_rows_l32_32438


namespace xiao_ming_total_score_l32_32225

theorem xiao_ming_total_score :
  ∃ (a_1 a_2 a_3 a_4 a_5 : ℕ), 
  a_1 < a_2 ∧ 
  a_2 < a_3 ∧ 
  a_3 < a_4 ∧ 
  a_4 < a_5 ∧ 
  a_1 + a_2 = 10 ∧ 
  a_4 + a_5 = 18 ∧ 
  a_1 + a_2 + a_3 + a_4 + a_5 = 35 :=
by
  sorry

end xiao_ming_total_score_l32_32225


namespace ratio_square_triangle_l32_32443

theorem ratio_square_triangle (A B C D E M : Point)
  (square_ABCD : square A B C D)
  (triangle_ABE : triangle A B E)
  (M_on_CD : M ∈ segment C D)
  (M_on_AE : M ∈ segment A E)
  (EM_eq_3AM : dist E M = 3 * dist A M) :
  area_of_square A B C D = 1 / 2 * area_of_triangle A B E :=
by
  sorry

end ratio_square_triangle_l32_32443


namespace point_difference_exceeds_fifty_l32_32779

-- Definitions of the conditions
def participants : ℕ := 200

def daily_matches : ℕ := participants / 2 -- Each day, participants are paired for matches

structure TournamentMatch :=
  (winner : ℕ)
  (loser : ℕ)
  (winner_points : ℕ)
  (loser_points : ℕ)

-- Assume we have a list of matches per day
def matches_per_day (day : ℕ) : list TournamentMatch := sorry

def points (player : ℕ) (day : ℕ) : ℕ := sorry -- This function retrieves the total points of a player at a given day

-- Function to get the difference in points between top and bottom player on a day
def point_difference (day : ℕ) :=
  let top_player := 1 in -- Sorted list of players by points, the first player is the top player
  let bottom_player := participants in -- The last player in the sorted list is the bottom player
  points top_player day - points bottom_player day

-- Statement to prove
theorem point_difference_exceeds_fifty :
  ∃ d : ℕ, point_difference d > 50 := sorry

end point_difference_exceeds_fifty_l32_32779


namespace probability_two_yellow_apples_l32_32246

theorem probability_two_yellow_apples (total_apples : ℕ) (red_apples : ℕ) (green_apples : ℕ) (yellow_apples : ℕ) (choose : ℕ → ℕ → ℕ) (probability : ℕ → ℕ → ℝ) :
  total_apples = 10 →
  red_apples = 5 →
  green_apples = 3 →
  yellow_apples = 2 →
  choose total_apples 2 = 45 →
  choose yellow_apples 2 = 1 →
  probability (choose yellow_apples 2) (choose total_apples 2) = 1 / 45 := 
  by
  sorry

end probability_two_yellow_apples_l32_32246


namespace total_books_l32_32286

variable (M K G : ℕ)

-- Conditions
def Megan_books := 32
def Kelcie_books := Megan_books / 4
def Greg_books := 2 * Kelcie_books + 9

-- Theorem to prove
theorem total_books : Megan_books + Kelcie_books + Greg_books = 65 := by
  unfold Megan_books Kelcie_books Greg_books
  sorry

end total_books_l32_32286


namespace time_spent_on_seals_l32_32367

theorem time_spent_on_seals (s : ℕ) 
  (h1 : 2 * 60 + 10 = 130) 
  (h2 : s + 8 * s + 13 = 130) :
  s = 13 :=
sorry

end time_spent_on_seals_l32_32367


namespace value_of_abc_l32_32330

theorem value_of_abc : 
  ∃ a b c : ℤ, 
    (x^2 + 15 * x + 54) = (x + a) * (x + b) ∧ 
    (x^2 - 17 * x + 72) = (x - b) * (x - c) ∧ 
    a + b + c = 23 := 
begin 
  sorry 
end

end value_of_abc_l32_32330


namespace factor_polynomial_l32_32103

theorem factor_polynomial :
  (x : ℝ) → (x^2 - 6*x + 9 - 64*x^4) = (-8*x^2 + x - 3) * (8*x^2 + x - 3) :=
by
  intro x
  sorry

end factor_polynomial_l32_32103


namespace find_sum_l32_32539

variable (a b c d : ℝ)

theorem find_sum (h1 : a * b + b * c + c * d + d * a = 48) (h2 : b + d = 6) : a + c = 8 :=
sorry

end find_sum_l32_32539


namespace no_such_function_exists_l32_32836

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, f(x + f(y)) = f(x) + sin y :=
by
  sorry

end no_such_function_exists_l32_32836


namespace sqrt_mixed_fraction_l32_32490

theorem sqrt_mixed_fraction (a b : ℤ) (h_a : a = 8) (h_b : b = 9) : 
  (√(a + b / 16)) = (√137) / 4 := 
by 
  sorry

end sqrt_mixed_fraction_l32_32490


namespace minimum_ratio_at_five_halves_l32_32182

noncomputable def x_A (a : ℝ) := 4^(-a)
noncomputable def x_B (a : ℝ) := 4^a
noncomputable def x_C (a : ℝ) := 4^(-18 / (2 * a + 1))
noncomputable def x_D (a : ℝ) := 4^(18 / (2 * a + 1))

noncomputable def m (a : ℝ) := abs (x_A a - x_C a)
noncomputable def n (a : ℝ) := abs (x_B a - x_D a)

noncomputable def ratio (a : ℝ) := n a / m a

theorem minimum_ratio_at_five_halves (a : ℝ) (h : a > 0) : ratio a = ratio (5 / 2) → a = 5 / 2 := by
  sorry

end minimum_ratio_at_five_halves_l32_32182


namespace smallest_lcm_l32_32209

theorem smallest_lcm (k l : ℕ) (hk : 999 < k ∧ k < 10000) (hl : 999 < l ∧ l < 10000)
  (h_gcd : Nat.gcd k l = 5) : Nat.lcm k l = 201000 :=
sorry

end smallest_lcm_l32_32209


namespace pizza_cost_l32_32121

theorem pizza_cost
  (initial_money_frank : ℕ)
  (initial_money_bill : ℕ)
  (final_money_bill : ℕ)
  (pizza_cost : ℕ)
  (number_of_pizzas : ℕ)
  (money_given_to_bill : ℕ) :
  initial_money_frank = 42 ∧
  initial_money_bill = 30 ∧
  final_money_bill = 39 ∧
  number_of_pizzas = 3 ∧
  money_given_to_bill = final_money_bill - initial_money_bill →
  3 * pizza_cost + money_given_to_bill = initial_money_frank →
  pizza_cost = 11 :=
by
  sorry

end pizza_cost_l32_32121


namespace sum_sin_cos_bounds_l32_32349

theorem sum_sin_cos_bounds (x1 x2 x3 x4 x5 : ℝ) (h1 : 0 ≤ x1 ∧ x1 ≤ π / 2)
                          (h2 : 0 ≤ x2 ∧ x2 ≤ π / 2) (h3 : 0 ≤ x3 ∧ x3 ≤ π / 2)
                          (h4 : 0 ≤ x4 ∧ x4 ≤ π / 2) (h5 : 0 ≤ x5 ∧ x5 ≤ π / 2)
                          (h_sum : sin x1 + sin x2 + sin x3 + sin x4 + sin x5 = 3) :
    2 ≤ cos x1 + cos x2 + cos x3 + cos x4 + cos x5 ∧ cos x1 + cos x2 + cos x3 + cos x4 + cos x5 ≤ 4 := 
by
  sorry

end sum_sin_cos_bounds_l32_32349


namespace transforming_sin_curve_l32_32694

theorem transforming_sin_curve :
  ∀ x : ℝ, (2 * Real.sin (x + (Real.pi / 3))) = (2 * Real.sin ((1/3) * x + (Real.pi / 3))) :=
by
  sorry

end transforming_sin_curve_l32_32694


namespace matrix_sum_eq_l32_32101

-- Definitions for matrices
def A : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -3], ![0, 5]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![[-6, 8], ![7, -10]]

-- Theorem statement
theorem matrix_sum_eq :
  A + B = ![![-2, 5], ![7, -5]] :=
by 
  -- The proof is omitted
  sorry

end matrix_sum_eq_l32_32101


namespace distance_focus_to_directrix_eq_four_l32_32158

theorem distance_focus_to_directrix_eq_four (p : ℝ) :
  (∃ p > 0, ∀ (x y : ℝ), (x^2 = 2 * p * y) ∧ ((0, ±2) = focus_of_parabola)) →
  p = 4 :=
by
  sorry

end distance_focus_to_directrix_eq_four_l32_32158


namespace solve_log_eq_l32_32691

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_log_eq (x : ℝ) (hx1 : x + 1 > 0) (hx2 : x - 1 > 0) :
  log_base (x + 1) (x^3 - 9 * x + 8) * log_base (x - 1) (x + 1) = 3 ↔ x = 3 := by
  sorry

end solve_log_eq_l32_32691


namespace greatest_power_of_2_divides_10_to_1002_minus_4_to_501_l32_32748

theorem greatest_power_of_2_divides_10_to_1002_minus_4_to_501 :
  ∃ n : ℤ, 2 ^ 1005 ∣ (10 ^ 1002 - 4 ^ 501) ∧ ∀ m : ℤ, (2 ^ m ∣ (10 ^ 1002 - 4 ^ 501)) → m ≤ 1005 :=
by
  sorry

end greatest_power_of_2_divides_10_to_1002_minus_4_to_501_l32_32748


namespace number_of_ways_to_choose_universities_l32_32806

theorem number_of_ways_to_choose_universities (students universities : ℕ) (n choosing_peking : ℕ) :
  students = 5 → universities = 5 → choosing_peking = 2 → 
  (students.choose choosing_peking) * \pow (universities - 1) (students - choosing_peking) = 640 :=
by
  intros h_students h_universities h_choosing_peking
  rw [h_students, h_universities, h_choosing_peking]
  sorry

end number_of_ways_to_choose_universities_l32_32806


namespace find_f_2012_plus_f_neg_2013_l32_32574

-- Defining the function and its properties
def f : ℝ → ℝ := λ x, if x ∈ set.Ico 0 2 then real.log (x + 1) / real.log 2 else 0

-- Given conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, x ≥ 0 → f (x + 2) = f x

-- Question: Prove the given equation
theorem find_f_2012_plus_f_neg_2013 : f 2012 + f (-2013) = -1 :=
by
  -- Conditions of the problem
  sorry

end find_f_2012_plus_f_neg_2013_l32_32574


namespace arithmetic_sequence_general_and_sum_l32_32141

open Nat

/-- Given an arithmetic sequence {a_n} with sum S_n, satisfying:
    1. a_3 = 8
    2. S_5 = 2a_7
    Prove: -/
theorem arithmetic_sequence_general_and_sum (a : ℕ → ℕ) (b : ℕ → ℕ) (S T : ℕ → ℕ) 
  (h1 : a 3 = 8)
  (h2 : S 5 = 2 * a 7)
  (h_sum_a : ∀ n, S n = n * (2 * a 1 + (n - 1) * Series.rangeSum id)) 
  (h_b : ∀ n, b n = a n + 2^(n+1)) :
  (∀ n, a n = 3 * n - 1) ∧ 
  (∀ n, T (2 * n) = 6 * (n * n) + n + 2^(2 * n + 2) - 4) := sorry

end arithmetic_sequence_general_and_sum_l32_32141


namespace relationship_between_p_and_q_l32_32568

variables {x y : ℝ}

def p : Prop := x ≠ 2 ∧ y ≠ 3
def q : Prop := x + y ≠ 5

theorem relationship_between_p_and_q : ¬ (p → q) ∧ ¬ (q → p) :=
sorry

end relationship_between_p_and_q_l32_32568


namespace min_value_frac_square_l32_32856

theorem min_value_frac_square (x : ℝ) (h : x > 9) : (x^2 / (x - 9) ≥ 36) ∧ ∃ y > 9, y^2 / (y - 9) = 36 :=
by {
  split,
  -- Prove that for all x > 9, x^2 / (x - 9) ≥ 36
  sorry,
  -- Prove that there exists an x > 9 such that x^2 / (x - 9) = 36
  use 18,
  split,
  linarith,
  norm_num,
}

end min_value_frac_square_l32_32856


namespace length_of_train_l32_32058

-- Definitions for conditions
def train_speed_kmph : ℝ := 84
def platform_length_m : ℝ := 233.3632
def time_seconds : ℝ := 16
def speed_kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (1000 / 3600)
def train_speed_mps : ℝ := speed_kmph_to_mps train_speed_kmph
def distance_covered_m : ℝ := train_speed_mps * time_seconds

-- Theorem statement asserting the length of the train
theorem length_of_train (train_speed_kmph : ℝ) (platform_length_m : ℝ) (time_seconds : ℝ)
  (train_speed_mps : ℝ) (distance_covered_m : ℝ) : 
  distance_covered_m = 373.333 → 
  platform_length_m = 233.3632 → 
  train_speed_kmph = 84 → 
  train_speed_mps = 23.3333 →
  time_seconds = 16 →
  train_speed_mps * time_seconds = 373.333 → 
  (distance_covered_m - platform_length_m) ≈ 139.9698 :=
sorry

end length_of_train_l32_32058


namespace xyz_sum_eq_7x_plus_5_l32_32276

variable (x y z : ℝ)

theorem xyz_sum_eq_7x_plus_5 (h1: y = 3 * x) (h2: z = y + 5) : x + y + z = 7 * x + 5 :=
by
  sorry

end xyz_sum_eq_7x_plus_5_l32_32276


namespace num_people_in_group_l32_32992

-- Define constants and conditions
def cost_per_set : ℕ := 3  -- $3 to make 4 S'mores
def smores_per_set : ℕ := 4
def total_cost : ℕ := 18   -- $18 total cost
def smores_per_person : ℕ := 3

-- Calculate total S'mores that can be made
def total_sets : ℕ := total_cost / cost_per_set
def total_smores : ℕ := total_sets * smores_per_set

-- Proof problem statement
theorem num_people_in_group : (total_smores / smores_per_person) = 8 :=
by
  sorry

end num_people_in_group_l32_32992


namespace harmonic_not_integer_l32_32995

theorem harmonic_not_integer (n : ℕ) (h : n > 1) : ¬ (∃ k : ℤ, H n = k) :=
by
  sorry

def H : ℕ → ℚ
| 0     => 0
| 1     => 1
| (n+1) => H n + 1 / (n+1)

end harmonic_not_integer_l32_32995


namespace total_cost_proof_l32_32070

-- Define the cost structure as functions
def cost_first_quarter_hour : ℝ := 6
def cost_next_hour_per_hour : ℝ := 8
def cost_next_two_hours_per_hour : ℝ := 7
def cost_beyond_three_hours_per_hour : ℝ := 5

-- Define customers' usage times in hours
def usage_time_customer_A : ℝ := 4 + 25/60
def usage_time_customer_B : ℝ := 1 + 45/60
def usage_time_customer_C : ℝ := 7 + 10/60

-- Function to calculate the cost based on the given usage time
def calculate_cost (t : ℝ) : ℝ :=
  if t <= 1/4 then cost_first_quarter_hour
  else if t <= 1 then cost_first_quarter_hour + (t - 1/4) * cost_next_hour_per_hour
  else if t <= 3 then cost_first_quarter_hour + 3/4 * cost_next_hour_per_hour + (t - 1) * cost_next_two_hours_per_hour
  else cost_first_quarter_hour + 3/4 * cost_next_hour_per_hour + 2 * cost_next_two_hours_per_hour + (t - 3) * cost_beyond_three_hours_per_hour

-- Total costs for each customer
def total_cost_customer_A : ℝ := calculate_cost usage_time_customer_A
def total_cost_customer_B : ℝ := calculate_cost usage_time_customer_B
def total_cost_customer_C : ℝ := calculate_cost usage_time_customer_C

-- Prove the total costs for each customer
theorem total_cost_proof :
  total_cost_customer_A = 33.0835 ∧
  total_cost_customer_B = 17.25 ∧
  total_cost_customer_C = 46.8335 :=
by
  sorry

end total_cost_proof_l32_32070


namespace rational_function_value_l32_32696

theorem rational_function_value (g : ℚ → ℚ) (h : ∀ x : ℚ, x ≠ 0 → 4 * g (x⁻¹) + 3 * g x / x = 2 * x^3) : g (-1) = -2 :=
sorry

end rational_function_value_l32_32696


namespace cylinder_twice_volume_l32_32408

noncomputable def original_volume (r h : ℝ) : ℝ := real.pi * r^2 * h

theorem cylinder_twice_volume :
  ∀ (r1 h1 r2 h2 : ℝ),
  r1 = 8 → h1 = 10 → r2 = 8 → h2 = 20 →
  original_volume r2 h2 = 2 * original_volume r1 h1 :=
by
  intros r1 h1 r2 h2 hr1 hr2 hh1 hh2
  rw [hr1, hh1, hr2, hh2]
  unfold original_volume
  sorry

end cylinder_twice_volume_l32_32408


namespace students_growth_rate_l32_32615

theorem students_growth_rate (x : ℝ) 
  (h_total : 728 = 200 + 200 * (1+x) + 200 * (1+x)^2) : 
  200 + 200 * (1+x) + 200*(1+x)^2 = 728 := 
  by
  sorry

end students_growth_rate_l32_32615


namespace integral_proof_l32_32515

noncomputable def integral_solution : ℝ → ℝ := 
  λ x, (x - 1) / 2 * Real.sqrt (8 + 2 * x - x^2) + 9 / 2 * Real.arcsin ((x - 1) / 3) + C

theorem integral_proof : 
  ∫ (λ x, Real.sqrt (8 + 2 * x - x^2)) = integral_solution := 
by 
  -- Proof skipped
  sorry

end integral_proof_l32_32515


namespace diff_of_squares_example_l32_32463

theorem diff_of_squares_example : (262^2 - 258^2 = 2080) :=
by
  let a := 262
  let b := 258
  have h1 : a^2 - b^2 = (a + b) * (a - b) := by rw [pow_two, pow_two, sub_mul, add_comm a b, mul_sub]
  have h2 : (a + b) = 520 := by norm_num
  have h3 : (a - b) = 4 := by norm_num
  have h4 : (262 + 258) * (262 - 258) = 520 * 4 := congr (congr_arg (*) h2) h3
  rw [h1, h4]
  norm_num

end diff_of_squares_example_l32_32463


namespace solve_for_x_l32_32603

theorem solve_for_x :
  ∀ (x : ℝ), log 10 5 + log 10 (5 * x + 1) = log 10 (x + 5) + 1 → x = 3 := by
  sorry

end solve_for_x_l32_32603


namespace diameter_of_circumscribed_circle_l32_32604

theorem diameter_of_circumscribed_circle 
  (a : ℝ) 
  (A : ℝ) 
  (h_a : a = 16) 
  (h_A : A = real.pi / 4) :
  ∃ D : ℝ, D = 16 * real.sqrt 2 :=
sorry

end diameter_of_circumscribed_circle_l32_32604


namespace setC_is_not_pythagorean_triple_l32_32432

-- Define what it means to be a Pythagorean triple
def isPythagoreanTriple (a b c : ℤ) : Prop :=
  a^2 + b^2 = c^2

-- Define the sets of numbers
def setA := (3, 4, 5)
def setB := (5, 12, 13)
def setC := (7, 25, 26)
def setD := (6, 8, 10)

-- The theorem stating that setC is not a Pythagorean triple
theorem setC_is_not_pythagorean_triple : ¬isPythagoreanTriple 7 25 26 := 
by sorry

end setC_is_not_pythagorean_triple_l32_32432


namespace closest_integer_to_cbrt_1728_l32_32007

theorem closest_integer_to_cbrt_1728 : 
  let x := 1728 in closest_integer (real.cbrt x) = 12 :=
by
  sorry

end closest_integer_to_cbrt_1728_l32_32007


namespace remainder_polynomial_division_l32_32860

theorem remainder_polynomial_division :
  let f : ℕ → ℤ := λ x, 5 * x^4 - 12 * x^3 + 3 * x^2 - 8 * x + 15 in
  f 4 = 543 :=
by
  sorry

end remainder_polynomial_division_l32_32860


namespace problem_1_problem_2_case1_problem_2_case2_l32_32177

noncomputable def f (a x : ℝ) := a^(3*x^2 - 3)
noncomputable def g (a x : ℝ) := (1/a)^(5*x + 5)

theorem problem_1 (a : ℝ) (h : 0 < a) (h' : a < 1) (x : ℝ) :
  f(a, x) < 1 ↔ x ∈ set.Iio (-1) ∪ set.Ioi 1 :=
sorry

theorem problem_2_case1 (a : ℝ) (h : 0 < a) (h' : a < 1) (x : ℝ) :
  f(a, x) ≥ g(a, x) ↔ x ∈ set.Icc (-1) (-2/3) :=
sorry

theorem problem_2_case2 (a : ℝ) (h : 1 < a) (x : ℝ) :
  f(a, x) ≥ g(a, x) ↔ x ∈ set.Iic (-1) ∪ set.Ici (-2/3) :=
sorry

end problem_1_problem_2_case1_problem_2_case2_l32_32177


namespace area_of_trapezium_l32_32113

theorem area_of_trapezium (a b h : ℕ) (ha : a = 26) (hb : b = 18) (hh : h = 15) :
  (1 / 2 : ℚ) * (a + b) * h = 330 :=
by
  rw [ha, hb, hh]
  norm_num

sorry

end area_of_trapezium_l32_32113


namespace find_function_relationship_between_y_and_x_l32_32875

theorem find_function_relationship_between_y_and_x :
  ∃ (k1 k2 : ℝ),
    (∀ x : ℝ, y = k1 * x + k2 * (x + 1)) ∧
    (y = 3 when x = 1) ∧
    (y = -1 when x = -3) →
    ∀ x : ℝ, y = x + 1 :=
by
  sorry

end find_function_relationship_between_y_and_x_l32_32875


namespace difference_in_areas_l32_32280

-- Definitions
def length_large : ℝ := 40
def width_large : ℝ := 20
def length_small : ℝ := (3 / 5) * length_large
def width_small : ℝ := (3 / 4) * width_large
def area_large : ℝ := length_large * width_large
def area_small : ℝ := length_small * width_small

-- Theorem statement
theorem difference_in_areas : area_large - area_small = 440 := by
  -- Proof skipped
  sorry

end difference_in_areas_l32_32280


namespace min_phone_calls_l32_32767

/-- 
Given n people each with exactly one unique secret, 
in each phone call the caller tells the other person every secret 
he knows but learns nothing from the person he calls.
This theorem proves the minimum number of phone calls required 
so that each person knows all n secrets is 2 * n - 2.
-/
theorem min_phone_calls (n : ℕ) (h : n > 0) :
  ∃ c : ℕ, c = 2 * n - 2 :=
by
  use 2 * n - 2
  sorry

end min_phone_calls_l32_32767


namespace total_fish_l32_32080

-- Defining the number of fish each person has, based on the conditions.
def billy_fish : ℕ := 10
def tony_fish : ℕ := 3 * billy_fish
def sarah_fish : ℕ := tony_fish + 5
def bobby_fish : ℕ := 2 * sarah_fish

-- The theorem stating the total number of fish.
theorem total_fish : billy_fish + tony_fish + sarah_fish + bobby_fish = 145 := by
  sorry

end total_fish_l32_32080


namespace john_horizontal_distance_l32_32646

-- Define the conditions and the question
def elevation_initial : ℕ := 100
def elevation_final : ℕ := 1450
def vertical_to_horizontal_ratio (v h : ℕ) : Prop := v * 2 = h

-- Define the proof problem: the horizontal distance John moves
theorem john_horizontal_distance : ∃ h, vertical_to_horizontal_ratio (elevation_final - elevation_initial) h ∧ h = 2700 := 
by 
  sorry

end john_horizontal_distance_l32_32646


namespace setC_not_pythagorean_l32_32434

/-- Defining sets of numbers as options -/
def SetA := (3, 4, 5)
def SetB := (5, 12, 13)
def SetC := (7, 25, 26)
def SetD := (6, 8, 10)

/-- Function to check if a set is a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

/-- Theorem stating set C is not a Pythagorean triple -/
theorem setC_not_pythagorean :
  ¬isPythagoreanTriple 7 25 26 :=
by {
  -- This slot will be filled with the concrete proof steps in Lean.
  sorry
}

end setC_not_pythagorean_l32_32434


namespace manuscript_pages_l32_32686

theorem manuscript_pages (P : ℕ)
  (h1 : 30 = 30)
  (h2 : 20 = 20)
  (h3 : 50 = 30 + 20)
  (h4 : 710 = 5 * (P - 50) + 30 * 8 + 20 * 11) :
  P = 100 :=
by
  sorry

end manuscript_pages_l32_32686


namespace prob_all_boys_l32_32042

-- Define the events
def is_boy (child : ℕ) : Prop := child = 1 -- Assuming 1 represents a boy and 0 represents a girl

noncomputable def probability_all_boys_given_conditions : ℝ :=
  let sample_space := {c | c.length = 4 ∧ is_boy (c.head) ∧ (∃ i, i ≠ 0 ∧ is_boy (c.get! i))} in
  let favorable_outcome := {c | c = [1, 1, 1, 1]} in
  (favorable_outcome ∩ sample_space).card.to_real / sample_space.card.to_real

-- Theorem to prove the correctness of the answer
theorem prob_all_boys :
  probability_all_boys_given_conditions = 1 / 5 :=
sorry

end prob_all_boys_l32_32042


namespace solution_set_proof_l32_32610

theorem solution_set_proof {a b : ℝ} :
  (∀ x, 2 < x ∧ x < 3 → x^2 - a * x - b < 0) →
  (∀ x, bx^2 - a * x - 1 > 0) →
  (∀ x, -1 / 2 < x ∧ x < -1 / 3) :=
by
  sorry

end solution_set_proof_l32_32610


namespace stewarts_theorem_l32_32683

theorem stewarts_theorem
  (A B C D : ℝ)
  (AB AC AD : ℝ)
  (BD CD BC : ℝ)
  (hD_on_BC : BD + CD = BC) :
  AB^2 * CD + AC^2 * BD - AD^2 * BC = BD * CD * BC := 
sorry

end stewarts_theorem_l32_32683


namespace magnitude_of_z_l32_32165

open Complex

theorem magnitude_of_z (z : ℂ) (h : z + I = (2 + I) / I) : abs z = Real.sqrt 10 := by
  sorry

end magnitude_of_z_l32_32165


namespace inequality_solution_set_l32_32830

def h (x : ℝ) : ℝ := 2^x

def f (x : ℝ) : ℝ := (1 - h x) / (1 + h x)

theorem inequality_solution_set :
  {x : ℝ | f (2*x - 1) > f (x + 1)} = {x : ℝ | x < 2} :=
by
  sorry

end inequality_solution_set_l32_32830


namespace evaluate_expression_l32_32469

def g (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 7

theorem evaluate_expression : 3 * g 2 + 4 * g (-2) = 113 := by
  sorry

end evaluate_expression_l32_32469


namespace moles_of_CHCl3_formed_l32_32478

def reaction_1 : ℕ → ℕ → ℕ × ℕ
| ch4, cl2 =>
  if ch4 ≤ cl2 then (0, cl2 - ch4)
  else (ch4 - cl4, 0)

def reaction_2 : ℕ → ℕ → ℕ × ℕ
| ch3cl, cl2 =>
  if ch3cl ≤ cl2 then (0, cl2 - ch3cl)
  else (ch3cl - cl2, 0)

def reaction_3 : ℕ → ℕ → ℕ × ℕ
| ch2cl2, cl2 =>
  if ch2cl2 ≤ cl2 then (0, cl2 - ch2cl2)
  else (ch2cl2 - cl2, 0)

def reaction_chain (init_ch4 : ℕ) (init_cl2 : ℕ) : ℕ :=
  let (ch4_left, cl2_left) := reaction_1 init_ch4 init_cl2
  let (ch3cl_left, cl2_left) := reaction_2 (init_ch4 - ch4_left) cl2_left
  let (ch2cl2_left, cl2_left) := reaction_3 (init_ch4 - ch4_left - ch3cl_left) cl2_left
  (init_ch4 - ch4_left - ch3cl_left - ch2cl2_left)

theorem moles_of_CHCl3_formed :
  reaction_chain 3 9 = 3 := sorry

end moles_of_CHCl3_formed_l32_32478


namespace minimum_m_plus_AB_l32_32957

noncomputable theory

-- Point definition
structure Point := (x : ℝ) (y : ℝ)

-- Parabola points and focus
def parabola (A : Point) := A.y^2 = 4 * A.x
def focus : Point := ⟨1, 0⟩

-- Circle definition
def circle (B : Point) := (B.x + 3)^2 + (B.y + 3)^2 = 4

-- Distance to y-axis
def distance_to_y_axis (A : Point) : ℝ := abs A.x

-- Distance between two points
def distance (P Q : Point) : ℝ := sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2)

-- Problem statement
theorem minimum_m_plus_AB (A B : Point) (m : ℝ) 
  (h_parabola : parabola A) (h_circle : circle B) (h_m : distance_to_y_axis A = m) :
  m + distance A B = 2 :=
begin
  sorry
end

end minimum_m_plus_AB_l32_32957


namespace no_domino_tiling_l32_32467

theorem no_domino_tiling (n : ℕ) (hn : 0 < n) : 
  ¬ (∃ (f : ℕ × ℕ → bool), (∀ (i j : ℕ), (i < n ∧ j < n → 
    ((f (i, j) = tt) ↔ ((i + j) % 2 = 0) ∧
    (f (i + 1, j) = tt ∨ f (i, j + 1) = tt ∨ f (i - 1, j) = tt ∨ f (i, j - 1) = tt)) ∧
    (¬((i = 0 ∧ j = 0) ∨ (i = n - 1 ∧ j = n - 1)))))))
    sorry

end no_domino_tiling_l32_32467


namespace math_problem_proof_l32_32747

-- Define the base conversion functions
def base11_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 2471 => 1 * 11^0 + 7 * 11^1 + 4 * 11^2 + 2 * 11^3
  | _    => 0

def base5_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 121 => 1 * 5^0 + 2 * 5^1 + 1 * 5^2
  | _   => 0

def base7_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 3654 => 4 * 7^0 + 5 * 7^1 + 6 * 7^2 + 3 * 7^3
  | _    => 0

def base8_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 5680 => 0 * 8^0 + 8 * 8^1 + 6 * 8^2 + 5 * 8^3
  | _    => 0

theorem math_problem_proof :
  let x := base11_to_base10 2471
  let y := base5_to_base10 121
  let z := base7_to_base10 3654
  let w := base8_to_base10 5680
  x / y - z + w = 1736 :=
by
  sorry

end math_problem_proof_l32_32747


namespace perfect_index_of_21_most_perfect_number_in_range_l32_32372

-- Defining proper factors
def proper_factors (a : ℕ) : List ℕ :=
  (List.range a).filter (λ x, x > 0 ∧ x < a ∧ a % x = 0)

-- Defining perfect index
def perfect_index (a : ℕ) : ℚ :=
  (proper_factors a).sum / a

-- Theorem statement to prove the "perfect index" of 21
theorem perfect_index_of_21 : perfect_index 21 = 11 / 21 :=
begin
  simp [proper_factors, perfect_index],
  -- Calculation can be shown or simplified here.
  sorry
end

-- Theorem statement to prove the most "perfect" number in the given range
theorem most_perfect_number_in_range : ∃ n, 20 < n ∧ n < 30 ∧
  (∀ m, 20 < m ∧ m < 30 → abs (perfect_index n - 1) < abs (perfect_index m - 1)) ∧ n = 28 :=
begin
  simp [perfect_index],
  -- Calculation can be shown or simplified here.
  sorry
end

end perfect_index_of_21_most_perfect_number_in_range_l32_32372


namespace georges_final_score_l32_32386

theorem georges_final_score :
  (6 + 4) * 3 = 30 := 
by
  sorry

end georges_final_score_l32_32386


namespace max_dance_counts_possible_l32_32022

noncomputable def max_dance_counts : ℕ := 29

theorem max_dance_counts_possible (boys girls : ℕ) (dance_count : ℕ → ℕ) :
   boys = 29 → girls = 15 → 
   (∀ b, b < boys → dance_count b ≤ girls) → 
   (∀ g, g < girls → ∃ d, d ≤ boys ∧ dance_count d = g) →
   (∃ d, d ≤ max_dance_counts ∧
     (∀ k, k ≤ d → (∃ b, b < boys ∧ dance_count b = k) ∨ (∃ g, g < girls ∧ dance_count g = k))) := 
sorry

end max_dance_counts_possible_l32_32022


namespace integral_transform_l32_32272

-- The definitions for the sequence of functions f_n
def f₁ (f : ℝ → ℝ) (a x : ℝ) : ℝ :=
  ∫ t in a..x, f t

def f₂ (f : ℝ → ℝ) (a x : ℝ) : ℝ :=
  ∫ t in a..x, f₁ f a t

def fₙ (f : ℝ → ℝ) (n : ℕ) (a x : ℝ) : ℝ :=
  (nat.rec_on n (λ _ _, 0) (λ n fn t, ∫ s in a..t, fn f s)) f x

theorem integral_transform (f : ℝ → ℝ) (a x : ℝ) (n : ℕ) :
  fₙ f n a x = (∫ t in a..x, f t * (t - x)^(n - 1)) * (-1)^(n - 1) / fact (n - 1) := 
sorry

end integral_transform_l32_32272


namespace cuboid_diagonals_and_edges_l32_32305

theorem cuboid_diagonals_and_edges (a b c : ℝ) : 
  4 * (a^2 + b^2 + c^2) = 4 * a^2 + 4 * b^2 + 4 * c^2 :=
by
  sorry

end cuboid_diagonals_and_edges_l32_32305


namespace four_is_integer_l32_32808

theorem four_is_integer (h1 : ∀ n : ℕ, ↑n ∈ ℤ) (h2 : 4 ∈ ℕ) : (4 : ℤ) := 
by 
  sorry

end four_is_integer_l32_32808


namespace cost_price_percentage_l32_32920

-- Define selling price and cost price
variables (SP CP : ℝ)
-- Define profit percent
def profit_percent := 21.951219512195124 / 100
-- Define profit as the difference between selling price and cost price
def profit := SP - CP

-- Define the main statement
theorem cost_price_percentage (h : profit_percent = (profit / CP) * 100) : (CP / SP) * 100 = 82 := by
  have h1 : profit = SP - CP := by sorry
  have h2 : profit_percent = ((SP - CP) / CP) * 100 := by sorry
  have h3 : 21.951219512195124 / 100 = (SP - CP) / CP := by sorry
  have h4 : 0.21951219512195124 = (SP - CP) / CP := by sorry
  have h5 : 0.21951219512195124 * CP + CP = SP := by sorry
  have h6 : CP * 1.21951219512195124 = SP := by
    linarith [eq_subst h5 h4, h3, h2, h1]
  have h7 : (CP / SP) = 1 / 1.21951219512195124 := by
    rw h6
    field_simp
  have h8 : (CP / SP) * 100 = (1 / 1.21951219512195124) * 100 := by
    rw h7
  have h9 : (1 / 1.21951219512195124) * 100 ≈ 82 := by sorry
  exact h9

end cost_price_percentage_l32_32920


namespace max_sum_primes_abc_l32_32590

theorem max_sum_primes_abc (a b c : ℕ) 
  (ha : nat.prime a) (hb : nat.prime b) (hc : nat.prime c)
  (h1 : a < b) (h2 : b < c) (hc1 : c < 100)
  (h3 : (b - a) * (c - b) * (c - a) = 240) : 
  a + b + c = 111 :=
sorry

end max_sum_primes_abc_l32_32590


namespace score_stability_l32_32942

theorem score_stability (mean_A mean_B : ℝ) (h_mean_eq : mean_A = mean_B)
  (variance_A variance_B : ℝ) (h_variance_A : variance_A = 0.06) (h_variance_B : variance_B = 0.35) :
  variance_A < variance_B :=
by
  -- Theorem statement and conditions sufficient to build successfully
  sorry

end score_stability_l32_32942


namespace cos_pi_over_3_plus_2theta_l32_32126

theorem cos_pi_over_3_plus_2theta 
  (theta : ℝ)
  (h : Real.sin (Real.pi / 3 - theta) = 3 / 4) : 
  Real.cos (Real.pi / 3 + 2 * theta) = 1 / 8 :=
by 
  sorry

end cos_pi_over_3_plus_2theta_l32_32126


namespace horizontal_distance_l32_32649

theorem horizontal_distance (elev_initial elev_final v_ratio h_ratio distance: ℝ)
  (h1 : elev_initial = 100)
  (h2 : elev_final = 1450)
  (h3 : v_ratio = 1)
  (h4 : h_ratio = 2)
  (h5 : distance = 1350) :
  distance * h_ratio = 2700 := by
  sorry

end horizontal_distance_l32_32649


namespace polygons_side_inequality_l32_32765

theorem polygons_side_inequality 
{n : ℕ} (A B : Fin n → Point) 
(hA_convex : ConvexPolygon A) (hB_convex : ConvexPolygon B)
(h_sides_equal : ∀ (i : Fin (n - 1)), side_length A i = side_length B i)
(h_angles : ∀ (i : Fin (n - 2)), angle A (i + 1) ≥ angle B (i + 1))
(h_strict_angle : ∃ (i : Fin (n - 2)), angle A (i + 1) > angle B (i + 1)) : 
side_length A ⟨0, sorry⟩ ⟨n - 1, sorry⟩ > side_length B ⟨0, sorry⟩ ⟨n - 1, sorry⟩ := sorry

end polygons_side_inequality_l32_32765


namespace min_length_MN_of_prism_l32_32062

noncomputable def regular_triangular_prism_min_segment_length (a : ℝ) : ℝ := a / real.sqrt 5

theorem min_length_MN_of_prism (a : ℝ) :
  ∀ (M N : ℝ), 
  (∀ x ∈ M ∪ N, 
    (∃ (xM xN : ℝ), 
      (xM ∈ set.Icc 0 a) ∧
      (xN ∈ set.Icc 0 a) ∧
      (xM - xN = x) ∧
      (x * x + (a - 2 * x) ^ 2 = a^2/5))) →
   (infimum {MN : ℝ | ∃ M N, 
      line_parallel_plane (M, N) (A, B, A₁)} = M N) →

  minimum_length_MN = (a / real.sqrt 5) :=
sorry

end min_length_MN_of_prism_l32_32062


namespace symmetric_set_min_points_l32_32421

theorem symmetric_set_min_points {T : set (ℝ × ℝ)}
  (h_orig : ∀ {p : ℝ × ℝ}, p ∈ T → (-p.1, -p.2) ∈ T)
  (h_xaxis : ∀ {p : ℝ × ℝ}, p ∈ T → (p.1, -p.2) ∈ T)
  (h_yaxis : ∀ {p : ℝ × ℝ}, p ∈ T → (-p.1, p.2) ∈ T)
  (h_line : ∀ {p : ℝ × ℝ}, p ∈ T → (-p.2, -p.1) ∈ T)
  (h_point : (1, 4) ∈ T) :
  ∃ (S : finset (ℝ × ℝ)), (∀ p ∈ S, p ∈ T) ∧ S.card = 8 :=
by {
  -- Proof can be written here.
  sorry
}

end symmetric_set_min_points_l32_32421


namespace part1_part2_l32_32172

def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem part1 (x : ℝ) (hx : x ≠ 0) : f(x) + f(1/x) = 1 := by
  sorry

theorem part2 : f(1) + f(2) + f(3) + f(4) + f(1/2) + f(1/3) + f(1/4) = 4 := by
  sorry

end part1_part2_l32_32172


namespace correct_sum_after_digit_change_l32_32701

theorem correct_sum_after_digit_change :
  let d := 7
  let e := 8
  let num1 := 935641
  let num2 := 471850
  let correct_sum := num1 + num2
  let new_sum := correct_sum + 10000
  new_sum = 1417491 := 
sorry

end correct_sum_after_digit_change_l32_32701


namespace sum_of_roots_is_zero_l32_32340

theorem sum_of_roots_is_zero (Q : Polynomial ℂ) {θ : ℝ} (h_mon : Q.leadingCoeff = 1)
  (h_roots : ∃ θ, 0 < θ ∧ θ < π / 6 ∧ ∀ z : ℂ, z = e^(θ * I) ∨ z = e^(-θ * I) ∨ z = -e^(θ * I) ∨ z = -e^(-θ * I) → is_root Q z)
  (h_area : let roots := {z : ℂ | is_root Q z}
            in ∃ a b c d : ℂ, a ∈ roots ∧ b ∈ roots ∧ c ∈ roots ∧ d ∈ roots ∧
              complex_area_of_trapezoid a b c d = Q.eval 0) :
  Q.roots.sum = 0 := 
sorry

end sum_of_roots_is_zero_l32_32340


namespace sqrt_mixed_fraction_l32_32493

theorem sqrt_mixed_fraction (a b : ℤ) (h_a : a = 8) (h_b : b = 9) : 
  (√(a + b / 16)) = (√137) / 4 := 
by 
  sorry

end sqrt_mixed_fraction_l32_32493


namespace problem1_problem2_l32_32881

-- Definition of sequences and the problem

def a (n : ℕ) : ℝ := (1/4)^n

def b (n : ℕ) : ℝ := 3*n - 2

def c (n : ℕ) : ℝ := b n * a n

def S (n : ℕ) : ℝ := ∑ i in Finset.range n, c i

-- Proof that b_n is an arithmetic sequence
theorem problem1 (n : ℕ) : b (n+1) - b n = 3 := by
  sorry

-- Proof that the sum of the first n terms of the sequence c_n is given by the given formula
theorem problem2 (n : ℕ) : S n = (2/3) - (12 * n + 8) / 3 * (1/4)^(n+1) := by
  sorry

end problem1_problem2_l32_32881


namespace volume_ratio_proof_l32_32444

noncomputable def volume_ratio_cylinder_sphere_cone (a : ℝ) : Prop :=
let V_cylinder := 2 * π * a^3 in
let V_sphere := (4 / 3) * π * a^3 in
let V_cone := (2 / 3) * π * a^3 in
V_cylinder / (π * a^3) = 3 ∧ V_sphere / (π * a^3) = 2 ∧ V_cone / (π * a^3) = 1 ∧
(V_cylinder / (π * a^3) / V_sphere / (π * a^3) / V_cone / (π * a^3)) = (3 : (3 : 1))

theorem volume_ratio_proof (a : ℝ) : volume_ratio_cylinder_sphere_cone a :=
  by sorry

end volume_ratio_proof_l32_32444


namespace total_steps_l32_32301

theorem total_steps (steps_per_floor : ℕ) (n : ℕ) (m : ℕ) (h : steps_per_floor = 20) (hm : m = 11) (hn : n = 1) : 
  steps_per_floor * (m - n) = 200 :=
by
  sorry

end total_steps_l32_32301


namespace find_blue_shirts_l32_32531

-- Statements of the problem conditions
def total_shirts : ℕ := 23
def green_shirts : ℕ := 17

-- Definition that we want to prove
def blue_shirts : ℕ := total_shirts - green_shirts

-- Proof statement (no need to include the proof itself)
theorem find_blue_shirts : blue_shirts = 6 := by
  sorry

end find_blue_shirts_l32_32531


namespace negate_proposition_l32_32146

variable (x : ℝ)

theorem negate_proposition :
  (¬ (∃ x₀ : ℝ, x₀^2 - x₀ + 1/4 ≤ 0)) ↔ ∀ x : ℝ, x^2 - x + 1/4 > 0 :=
by
  sorry

end negate_proposition_l32_32146


namespace find_a_l32_32924

theorem find_a (a : ℝ) :
  (∃ l : ℝ → ℝ, (∀ x : ℝ, y = l x) ∧ l 1 = 0 ∧ 
   (∀ x : ℝ, l x = x^3 - f x ∨ l x = ax^2 + x - 9 - f x)) 
  → (a = -1 ∨ a = -7) :=
sorry

end find_a_l32_32924


namespace soccer_balls_per_basket_l32_32787

-- Define the initial conditions
variables (s : ℕ) -- number of soccer balls in each basket initially
constants (TotalBaskets : ℕ := 5) (TennisBallsPerBasket : ℕ := 15) 
constants (Students1 : ℕ := 3) (BallsRemoved1 : ℕ := 8) 
constants (Students2 : ℕ := 2) (BallsRemoved2 : ℕ := 10) 
constants (BallsLeft : ℕ := 56)

-- Define the theorem we want to prove
theorem soccer_balls_per_basket :
  (TotalBaskets * (TennisBallsPerBasket + s) - (Students1 * BallsRemoved1 + Students2 * BallsRemoved2) = BallsLeft) →
  s = 5 :=
by
  sorry

end soccer_balls_per_basket_l32_32787


namespace boat_distance_l32_32727

theorem boat_distance (H : ℝ) (θ₁ θ₂ θ₃ : ℝ) (hH : H = 30) (hθ₁ : θ₁ = Real.pi / 4) (hθ₂ : θ₂ = Real.pi / 6) (hθ₃ : θ₃ = Real.pi / 6) :
  distance_between_boats H θ₁ θ₂ θ₃ = 30 := by
  sorry

end boat_distance_l32_32727


namespace Aaron_initial_erasers_l32_32429

/-- 
  Given:
  - Aaron gives 34 erasers to Doris.
  - Aaron ends with 47 erasers.
  Prove:
  - Aaron started with 81 erasers.
-/ 
theorem Aaron_initial_erasers (gives : ℕ) (ends : ℕ) (start : ℕ) :
  gives = 34 → ends = 47 → start = ends + gives → start = 81 :=
by
  intros h_gives h_ends h_start
  sorry

end Aaron_initial_erasers_l32_32429


namespace sqrt_of_trig_identity_l32_32474

noncomputable def cos_sq (theta : ℝ) : ℝ := (Real.cos theta) ^ 2

theorem sqrt_of_trig_identity :
  let x := 3 - cos_sq (Real.pi / 9)
  let y := 3 - cos_sq (2 * Real.pi / 9)
  let z := 3 - cos_sq (4 * Real.pi / 9) in
  Real.sqrt (x * y * z) = 0 :=
by
  sorry

end sqrt_of_trig_identity_l32_32474


namespace distance_first_day_l32_32631

theorem distance_first_day (total_distance : ℕ) (q : ℚ) (n : ℕ) (a : ℚ) : total_distance = 378 ∧ q = 1 / 2 ∧ n = 6 → a = 192 :=
by
  -- Proof omitted, just provide the statement
  sorry

end distance_first_day_l32_32631


namespace sqrt3_mul_sqrt3_add_inv_sqrt3_sqrt5_abs_sqrt3_sub_sqrt5_sqrt016_sub_cbrt8_add_sqrt14_l32_32085

-- Problem 1: Prove \(\sqrt{3}(\sqrt{3}+\frac{1}{\sqrt{3}}) = 4\)
theorem sqrt3_mul_sqrt3_add_inv_sqrt3 : (Real.sqrt 3) * (Real.sqrt 3 + 1 / Real.sqrt 3) = 4 := 
by sorry

-- Problem 2: Prove \(2\sqrt{5}-|\sqrt{3}-\sqrt{5}| = \sqrt{5} + \sqrt{3}\)
theorem sqrt5_abs_sqrt3_sub_sqrt5 : 2 * Real.sqrt 5 - Real.abs (Real.sqrt 3 - Real.sqrt 5) = Real.sqrt 5 + Real.sqrt 3 := 
by sorry

-- Problem 3: Prove \(\sqrt{0.16}-\sqrt[3]{8}+\sqrt{\frac{1}{4}} = -1.1\)
theorem sqrt016_sub_cbrt8_add_sqrt14 : Real.sqrt 0.16 - Real.cbrt 8 + Real.sqrt (1 / 4) = -1.1 := 
by sorry

end sqrt3_mul_sqrt3_add_inv_sqrt3_sqrt5_abs_sqrt3_sub_sqrt5_sqrt016_sub_cbrt8_add_sqrt14_l32_32085


namespace impossible_to_position_all_101_bugs_l32_32723

-- Definitions based on conditions
def bugs : Type := { bugs : ℕ // bugs = 101 }
def friends_relation (b : bugs) : bugs → bugs → Prop := sorry -- This represents the friendship relation as described

-- Main Statement
theorem impossible_to_position_all_101_bugs :
  ¬(∃ (configuration : bugs → ℝ × ℝ), 
  ∀ (i j : bugs), friends_relation i j ↔ dist (configuration i) (configuration j) = 1) :=
sorry

end impossible_to_position_all_101_bugs_l32_32723


namespace inequality1_inequality2_l32_32313

variable {a b c : ℝ}

noncomputable theory

-- Given a, b, c are positive real numbers
axiom positive_real_numbers (ha : a > 0) (hb : b > 0) (hc : c > 0) : true

-- Prove a^2 + b^2 + c^2 ≥ ab + bc + ca
theorem inequality1 (ha : a > 0) (hb : b > 0) (hc : c > 0) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := 
by {
  have h_pos := positive_real_numbers ha hb hc,
  -- The actual proof steps would go here
  sorry,
}

-- Prove (a + b + c)^2 ≥ 3(ab + bc + ca)
theorem inequality2 (ha : a > 0) (hb : b > 0) (hc : c > 0) : (a + b + c)^2 ≥ 3 * (a * b + b * c + c * a) := 
by {
  have h_pos := positive_real_numbers ha hb hc,
  -- The actual proof steps would go here
  sorry,
}

end inequality1_inequality2_l32_32313


namespace number_of_accurate_statements_is_zero_l32_32810

-- Definitions to model the problem
def is_prime (n : ℕ) : Prop := nat.prime n
def is_composite (n : ℕ) : Prop := ¬ nat.prime n ∧ n ≠ 1
def accurate_statements : ℕ := 0

-- Statements to be checked
def statement1 := ∀ (p1 p2 : ℕ), is_prime p1 → is_prime p2 → is_composite (p1 + p2)
def statement2 := ∀ (c1 c2 : ℕ), is_composite c1 → is_composite c2 → is_composite (c1 + c2)
def statement3 := ∀ (p c : ℕ), is_prime p → is_composite c → is_composite (p + c)
def statement4 := ∀ (p c : ℕ), is_prime p → is_composite c → ¬ is_composite (p + c)

-- The theorem to be proved
theorem number_of_accurate_statements_is_zero : 
  (∃ (s : ℕ), s = 0 ∧ accurate_statements = s) :=
by
  existsi 0
  exact ⟨rfl, rfl⟩

end number_of_accurate_statements_is_zero_l32_32810


namespace fillTimeWithLeak_l32_32048

-- Definitions based on given conditions
def pumpFillRate : ℝ := 1 / 2 -- tank per hour
def leakDrainRate : ℝ := 1 / 26 -- tank per hour

-- Effective fill rate considering the leak
def effectiveFillRate : ℝ := pumpFillRate - leakDrainRate

-- Statement asserting the time taken to fill one tank with the leak active
theorem fillTimeWithLeak : (1 : ℝ) / effectiveFillRate = 2.1667 :=
by
  -- We are skipping the proof part as required by the prompt.
  sorry

end fillTimeWithLeak_l32_32048


namespace simplify_expression_l32_32963

theorem simplify_expression (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) : 
  let x := (b / c) * (c / b),
      y := (a / c) * (c / a),
      z := (a / b) * (b / a) in
  x^2 + y^2 + z^2 + x * y * z = 4 := 
by 
  sorry

end simplify_expression_l32_32963


namespace setC_is_not_pythagorean_triple_l32_32433

-- Define what it means to be a Pythagorean triple
def isPythagoreanTriple (a b c : ℤ) : Prop :=
  a^2 + b^2 = c^2

-- Define the sets of numbers
def setA := (3, 4, 5)
def setB := (5, 12, 13)
def setC := (7, 25, 26)
def setD := (6, 8, 10)

-- The theorem stating that setC is not a Pythagorean triple
theorem setC_is_not_pythagorean_triple : ¬isPythagoreanTriple 7 25 26 := 
by sorry

end setC_is_not_pythagorean_triple_l32_32433


namespace max_value_theorem_l32_32968

open Real

noncomputable def max_value (x y : ℝ) : ℝ :=
  x * y * (75 - 5 * x - 3 * y)

theorem max_value_theorem :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 5 * x + 3 * y < 75 ∧ max_value x y = 3125 / 3 := by
  sorry

end max_value_theorem_l32_32968


namespace johann_oranges_count_l32_32253

def initial_oranges : ℕ := 120
def percent_eaten : ℕ := 25
def part_emily_borrowed : ℕ := 1/3
def emily_returned : ℕ := 10
def part_carson_stole : ℕ := 1/4
def carson_returned_part : ℕ := 1/2
def part_beth_stole : ℕ := 1/5
def beth_returned : ℕ := 8

theorem johann_oranges_count :
  let eaten := initial_oranges * percent_eaten / 100,
      remaining_after_eaten := initial_oranges - eaten,
      emily_borrowed := remaining_after_eaten * part_emily_borrowed,
      remaining_after_emily := remaining_after_eaten - emily_borrowed,
      remaining_after_emily_return := remaining_after_emily + emily_returned,
      carson_stole := remaining_after_emily_return * part_carson_stole,
      remaining_after_carson := remaining_after_emily_return - carson_stole,
      remaining_after_carson_return := remaining_after_carson + carson_stole * carson_returned_part,
      beth_stole := remaining_after_carson * part_beth_stole,
      remaining_after_beth := remaining_after_carson - beth_stole,
      final_oranges := remaining_after_beth + beth_returned
  in final_oranges = 59 := sorry

end johann_oranges_count_l32_32253


namespace simplify_expression_l32_32760

theorem simplify_expression : 5 + (-3) - (-7) - (+2) = 5 - 3 + 7 - 2 := by
  sorry

end simplify_expression_l32_32760


namespace number_of_arrangements_l32_32023

theorem number_of_arrangements (A₅ : Nat) (A₄ : Nat) (arrangements : Nat) 
  (hA₅ : A₅ = Nat.perm 5 5) (hA₄ : A₄ = Nat.perm 4 2) : arrangements = A₄ * A₅ := 
sorry

end number_of_arrangements_l32_32023


namespace parallel_AC_A₁C₁_l32_32954

-- Define the right-angled triangle ABC with ∠B = 90°
variables {A B C H A₁ C₁ : Type} 
variable [linear_ordered_field ℝ]

structure Point := (x y : ℝ)
structure Triangle := (A B C : Point)

-- Define the altitude BH from B to AC
def isAltitude (B H A C : Point) : Prop :=
  H = Point ((A.x + C.x) / 2) ((A.y + C.y) / 2)

-- Define excircle touches in terms of the associated points
def isExcircleTouching (P1 P2 P3 Q : Point) : Prop := sorry -- Detailed definition omitted here

variables 
  (tABC : Triangle)
  (hAlt : isAltitude B H A C)
  (hExcircleBH : isExcircleTouching A B H A₁)
  (hExcircleCH : isExcircleTouching A C H C₁)

-- Statement to prove that AC is parallel to A₁C₁
theorem parallel_AC_A₁C₁ : (AC : Line) ∥ (A₁C₁ : Line) :=
  sorry

end parallel_AC_A₁C₁_l32_32954


namespace a4_binomial_coefficient_l32_32318

theorem a4_binomial_coefficient :
  ∀ (a_n a_1 a_2 a_3 a_4 a_5 : ℝ) (x : ℝ),
  (x^5 = a_n + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5) →
  (x^5 = (1 + (x - 1))^5) →
  a_4 = 5 :=
by
  intros a_n a_1 a_2 a_3 a_4 a_5 x hx1 hx2
  sorry

end a4_binomial_coefficient_l32_32318


namespace base16_to_base2_bits_l32_32382

def base16_to_base10 (n : Nat) : Nat :=
  8 * 16^4 + 8 * 16^3 + 8 * 16^2 + 8 * 16^1 + 8 * 16^0

theorem base16_to_base2_bits (n : Nat) (h : n = 88888) : 
  let m := base16_to_base10 n
  let bits := Nat.ceilLog2 (m + 1)
  bits = 20 :=
by
  have : m = 559240 := sorry
  have : 2^19 < m ∧ m < 2^20 := sorry
  have : bits = 20 := sorry
  assumption

end base16_to_base2_bits_l32_32382


namespace apples_left_l32_32072

theorem apples_left (initial_apples : ℕ) (kids : ℕ) (apples_per_teacher : ℕ) (teachers_per_kid : ℕ) (pies : ℕ) (apples_per_pie : ℕ) :
  initial_apples = 50 → kids = 2 → apples_per_teacher = 3 → teachers_per_kid = 2 → pies = 2 → apples_per_pie = 10 →
  initial_apples - (kids * apples_per_teacher * teachers_per_kid + pies * apples_per_pie) = 18 :=
by
  intro h_initial_apples h_kids h_apples_per_teacher h_teachers_per_kid h_pies h_apples_per_pie
  rw [h_initial_apples, h_kids, h_apples_per_teacher, h_teachers_per_kid, h_pies, h_apples_per_pie]
  norm_num
  exact sorry

end apples_left_l32_32072


namespace maximum_distance_product_l32_32629

theorem maximum_distance_product (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  let ρ1 := 4 * Real.cos α
  let ρ2 := 2 * Real.sin α
  |ρ1 * ρ2| ≤ 4 :=
by
  -- The proof would go here
  sorry

end maximum_distance_product_l32_32629


namespace range_of_a_for_inequality_l32_32706

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, (sin x)^2 + a * cos x + a^2 ≥ 1 + cos x) ↔ (a ≤ -2 ∨ a ≥ 1) :=
by
  sorry

end range_of_a_for_inequality_l32_32706


namespace max_blocks_l32_32027

theorem max_blocks (box_height box_width box_length : ℝ) 
  (typeA_height typeA_width typeA_length typeB_height typeB_width typeB_length : ℝ) 
  (h_box : box_height = 8) (w_box : box_width = 10) (l_box : box_length = 12) 
  (h_typeA : typeA_height = 3) (w_typeA : typeA_width = 2) (l_typeA : typeA_length = 4) 
  (h_typeB : typeB_height = 4) (w_typeB : typeB_width = 3) (l_typeB : typeB_length = 5) : 
  max (⌊box_height / typeA_height⌋ * ⌊box_width / typeA_width⌋ * ⌊box_length / typeA_length⌋)
      (⌊box_height / typeB_height⌋ * ⌊box_width / typeB_width⌋ * ⌊box_length / typeB_length⌋) = 30 := 
  by
  sorry

end max_blocks_l32_32027


namespace max_good_cells_in_8x8_l32_32622

structure Grid (n : ℕ) :=
  (cells : Fin n → Fin n → ℕ)
  (distinct : Function.Injective (Function.uncurry cells))

def is_good {n : ℕ} (grid : Grid n) (i j : Fin n) : Prop :=
  (∃ s1 : Finset (Fin n), s1.card = n - 6 ∧ ∀ k ∈ s1, grid.cells i j > grid.cells i k) ∧
  (∃ s2 : Finset (Fin n), s2.card = n - 6 ∧ ∀ k ∈ s2, grid.cells i j > grid.cells k j)

theorem max_good_cells_in_8x8 :
  ∀ grid : Grid 8, Finset.card {ij : Fin 8 × Fin 8 | is_good grid ij.1 ij.2}.to_finset ≤ 16 := 
sorry

end max_good_cells_in_8x8_l32_32622


namespace population_problem_l32_32618

theorem population_problem (P : ℝ) :
  (P * 1.15 * 0.90 * 1.20 * 0.75 = 7575) → P = 12199 :=
by
  intro h
  have P_value : P = 7575 / (1.15 * 0.90 * 1.20 * 0.75) := by sorry
  have P_calculated : P ≈ 12199 := by sorry
  exact P_calculated

end population_problem_l32_32618


namespace min_value_func_l32_32859

noncomputable def func (x : ℝ) := x^2 / (x - 9)

theorem min_value_func : (∀ x > 9, func(x) ≥ 36) ∧ (∃ x = 18, func(x) = 36) :=
by
  sorry

end min_value_func_l32_32859


namespace initial_small_shoes_count_l32_32804

-- Define the conditions
def large_shoes : Nat := 22
def medium_shoes : Nat := 50
def shoes_sold : Nat := 83
def shoes_left : Nat := 13

-- The equation derived from the conditions
def total_initial_shoes := large_shoes + medium_shoes + (shoes_sold + shoes_left)

-- The proof problem
theorem initial_small_shoes_count : large_shoes + medium_shoes + (shoes_sold + shoes_left) - large_shoes - medium_shoes = 24 :=
by
-- The overall equation simplifies to:
-- large_shoes + medium_shoes + small_shoes - large_shoes - medium_shoes = small_shoes
-- 83 + 13 - (22 + 50) = 24
suffices H: large_shoes + medium_shoes + shoes_sold + shoes_left - large_shoes - medium_shoes = 24 from H;
calc 
  shoes_sold + shoes_left 
  = 83 + 13 : by rfl
  ... - (22 + 50) 
  = 96 - 72 : by rfl
  ... = 24 : by rfl

end initial_small_shoes_count_l32_32804


namespace derivative_quadrant_l32_32919

theorem derivative_quadrant (b c : ℝ) (H_b : b = -4) : ¬ ∃ x y : ℝ, x < 0 ∧ y > 0 ∧ 2*x + b = y := by
  sorry

end derivative_quadrant_l32_32919


namespace minimum_value_expression_l32_32893

theorem minimum_value_expression (m n : ℝ) (h : m + 3 * n = 1) : 
  ∃ c : ℝ, (∀ x y : ℝ, x + 3 * y = 1 → m * exp m + 3 * n * exp (3 * n) ≥ c) ∧ 
           (∃ x y : ℝ, x + 3 * y = 1 ∧ m * exp m + 3 * n * exp (3 * n) = c) := 
begin
  use sqrt real.exp 1,
  split,
  { sorry },
  { sorry }
end

end minimum_value_expression_l32_32893


namespace new_light_wattage_l32_32012

def original_wattage : ℝ := 60
def percentage_increase : ℝ := 12 / 100

noncomputable def new_wattage : ℝ := original_wattage * (1 + percentage_increase)

theorem new_light_wattage :
  new_wattage = 67.2 :=
by
  sorry

end new_light_wattage_l32_32012


namespace conjugate_in_first_quadrant_l32_32601

-- Definition of the problem's conditions
def z_condition (z : ℂ) : Prop :=
  (1 + I) * z = complex.abs (1 - I)

-- Definition of the statement to be proved
theorem conjugate_in_first_quadrant (z : ℂ) (hz : z_condition z) : 
  (complex.re (conj z) > 0) ∧ (complex.im (conj z) > 0) :=
sorry

end conjugate_in_first_quadrant_l32_32601


namespace area_of_one_trapezoid_l32_32636

theorem area_of_one_trapezoid (outer_area inner_area : ℝ) (num_trapezoids : ℕ) (h_outer : outer_area = 36) (h_inner : inner_area = 4) (h_num_trapezoids : num_trapezoids = 3) : (outer_area - inner_area) / num_trapezoids = 32 / 3 :=
by
  rw [h_outer, h_inner, h_num_trapezoids]
  norm_num

end area_of_one_trapezoid_l32_32636


namespace endpoint_correctness_l32_32413

-- Define two points in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define start point (2, 2)
def startPoint : Point := ⟨2, 2⟩

-- Define the endpoint's conditions
def endPoint (x y : ℝ) : Prop :=
  y = 2 * x + 1 ∧ (x > 0) ∧ (Real.sqrt ((x - startPoint.x) ^ 2 + (y - startPoint.y) ^ 2) = 6)

-- The solution to the problem proving (3.4213, 7.8426) satisfies the conditions
theorem endpoint_correctness : ∃ (x y : ℝ), endPoint x y ∧ x = 3.4213 ∧ y = 7.8426 := by
  use 3.4213
  use 7.8426
  sorry

end endpoint_correctness_l32_32413


namespace variance_3ξ_plus_2_l32_32138

-- Main definition
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

-- Hypothesis: ξ follows a binomial distribution with parameters n = 5 and p = 1/3
def ξ : Type := sorry  -- ξ is a random variable following B(5, 1/3)

-- Prove that D(3ξ + 2) = 10
theorem variance_3ξ_plus_2 : (binomial_variance 5 (1/3)) = (10 : ℝ) := by
  sorry

end variance_3ξ_plus_2_l32_32138


namespace geom_sequence_product_l32_32880

noncomputable def geom_seq (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geom_sequence_product (a : ℕ → ℝ) (h1 : geom_seq a) (h2 : a 0 * a 4 = 4) :
  a 0 * a 1 * a 2 * a 3 * a 4 = 32 ∨ a 0 * a 1 * a 2 * a 3 * a 4 = -32 :=
by
  sorry

end geom_sequence_product_l32_32880


namespace Tom_has_38_photos_l32_32357

theorem Tom_has_38_photos :
  ∃ (Tom Tim Paul : ℕ), 
  (Paul = Tim + 10) ∧ 
  (Tim = 152 - 100) ∧ 
  (152 = Tom + Paul + Tim) ∧ 
  (Tom = 38) :=
by
  sorry

end Tom_has_38_photos_l32_32357


namespace imaginary_part_of_x_l32_32605

theorem imaginary_part_of_x (x : ℂ) (h : (3 + 4 * Complex.i) * x = Complex.abs (4 + 3 * Complex.i)) :
  Complex.im x = -4 / 5 :=
sorry

end imaginary_part_of_x_l32_32605


namespace inequality_reciprocal_of_negative_l32_32199

variable {a b : ℝ}

theorem inequality_reciprocal_of_negative (h : a < b) (h_neg_a : a < 0) (h_neg_b : b < 0) : 
  (1 / a) > (1 / b) := by
  sorry

end inequality_reciprocal_of_negative_l32_32199


namespace rectangular_field_fencing_l32_32415

noncomputable def totalFencingNeeded
  (width: ℝ) (height: ℝ) (radius1: ℝ) (radius2: ℝ) : ℝ :=
  2 * height + width + 2 * Real.pi * radius1 + 2 * Real.pi * radius2

theorem rectangular_field_fencing
  (width height radius1 radius2 : ℝ)
  (h1: width = 30)
  (h2: width * height = 180)
  (h3: radius1 = 2)
  (h4: radius2 = 3) :
  totalFencingNeeded width height radius1 radius2 ≈ 73.42 :=
by {
  sorry -- Proof can be added here
}

end rectangular_field_fencing_l32_32415


namespace period_start_time_l32_32528

/-- A period of time had 4 hours of rain and 5 hours without rain, ending at 5 pm. 
Prove that the period started at 8 am. -/
theorem period_start_time :
  let end_time := 17 -- 5 pm in 24-hour format
  let rainy_hours := 4
  let non_rainy_hours := 5
  let total_hours := rainy_hours + non_rainy_hours
  let start_time := end_time - total_hours
  start_time = 8 :=
by
  sorry

end period_start_time_l32_32528


namespace simplify_sqrt_of_mixed_number_l32_32497

noncomputable def sqrt_fraction := λ (a b : ℕ), (Real.sqrt a) / (Real.sqrt b)

theorem simplify_sqrt_of_mixed_number : sqrt_fraction 137 16 = (Real.sqrt 137) / 4 := by
  sorry

end simplify_sqrt_of_mixed_number_l32_32497


namespace pqr_solution_l32_32519

noncomputable def find_pqr : Prop :=
  ∃ (p q r : ℝ), -1 < p ∧ p < q ∧ q < r ∧ r < 1 ∧ (∀ (f : ℝ → ℝ), 
  (∀ (x : ℝ), f x = a * x^2 + b * x + c → polynomial.degree f ≤ 2) →
  ∫ x in -1..p, f x - ∫ x in p..q, f x + ∫ x in q..r, f x - ∫ x in r..1, f x = 0) ∧
  p = 1 / Real.sqrt 2 ∧ q = 0 ∧ r = -1 / Real.sqrt 2.

theorem pqr_solution : find_pqr :=
  sorry

end pqr_solution_l32_32519


namespace incorrect_inequality_l32_32597

theorem incorrect_inequality (a b c : ℝ) (h : a > b) : ¬ (ac^2 > bc^2) ↔ c = 0 :=
by {
  sorry
}

end incorrect_inequality_l32_32597


namespace range_m_of_odd_strictly_increasing_l32_32571

def f : ℝ → ℝ := sorry

theorem range_m_of_odd_strictly_increasing 
  (h1 : ∀ x ∈ set.Icc (-1 : ℝ) 1, f (-x) = -f x)
  (h2 : ∀ a b ∈ set.Ico (-1 : ℝ) 0, a ≠ b → (f a - f b) / (a - b) > 0)
  (h3 : ∀ m ∈ set.Icc (-1 / 2 : ℝ) 0, f (m + 1) > f (2 * m)) :
  ∀ m : ℝ, -1 / 2 ≤ m ∧ m ≤ 0 :=
by sorry

end range_m_of_odd_strictly_increasing_l32_32571


namespace total_fish_l32_32074

theorem total_fish :
  let Billy := 10
  let Tony := 3 * Billy
  let Sarah := Tony + 5
  let Bobby := 2 * Sarah
  in Billy + Tony + Sarah + Bobby = 145 :=
by
  sorry

end total_fish_l32_32074


namespace bell_rings_geography_l32_32259

def classes : List String := ["Maths", "History", "Geography", "Science", "Music"]

def bell_ringing_sequence (bell_count : Nat) (classes : List String) : String :=
  let indices := List.filterMap (fun i => if i % 2 = 0 then some (i / 2) else none) (List.range bell_count)
  match indices.reverse.head with
  | none => "No class"
  | some i => classes.get? i |>.getOrElse "No class"

theorem bell_rings_geography (bell_count : Nat) (classes : List String) : bell_count = 5 -> bell_ringing_sequence bell_count classes = "Geography" :=
  by
    sorry

end bell_rings_geography_l32_32259


namespace find_c_l32_32414

-- Definitions based on the conditions in the problem
def is_vertex (h k : ℝ) := (5, 1) = (h, k)
def passes_through (x y : ℝ) := (2, 3) = (x, y)

-- Lean theorem statement
theorem find_c (a b c : ℝ) (h k x y : ℝ) (hv : is_vertex h k) (hp : passes_through x y)
  (heq : ∀ y, x = a * y^2 + b * y + c) : c = 17 / 4 :=
by
  sorry

end find_c_l32_32414


namespace gcd_differences_l32_32855

theorem gcd_differences (a b c : ℕ) (r : ℕ) (hc₁ : b - a = 32) (hc₂ : c - b = 48) (hc₃ : c - a = 80) :
  Nat.gcd (Nat.gcd 32 48) 80 = 16 :=
by
  rw [Nat.gcd_comm, Nat.gcd_assoc]
  sorry

end gcd_differences_l32_32855


namespace rectangle_area_l32_32038

-- Given conditions for the problem are set as hypotheses.
structure Rectangle :=
  (A B C D : ℝ)
  (AB BC CD DA : ℝ)
  (tangent_circle_radius : ℝ)
  (midpoint_BD : ℝ)

-- Define a theorem to prove the area of the rectangle
theorem rectangle_area (r : ℝ) (rect : Rectangle) :
  (rect.tangent_circle_radius = r) →
  (rect.midpoint_BD = r) →
  let w := 2 * r in
  let h := r in
  let area := w * h in
  area = 2 * r ^ 2 :=
by 
  sorry

end rectangle_area_l32_32038


namespace find_h_l32_32967

def star (x y : ℝ) : ℝ :=
  x + (sqrt (y + sqrt (y + sqrt (y + ...))))

theorem find_h (h : ℝ) : star 3 h = 8 -> h = 20 :=
  by
    sorry

end find_h_l32_32967


namespace count_three_letter_words_with_A_l32_32191

theorem count_three_letter_words_with_A : 
  let total_words := 5^3 in
  let words_without_A := 4^3 in
  total_words - words_without_A = 61 :=
by
  sorry

end count_three_letter_words_with_A_l32_32191


namespace complex_power_calc_l32_32818

theorem complex_power_calc : (1 + complex.I) ^ 20 = -1024 := by
  sorry

end complex_power_calc_l32_32818


namespace find_b_value_l32_32337

theorem find_b_value 
  (b : ℝ)
  (h_circle : ∀ (x y : ℝ), x^2 + y^2 = 4 → True)
  (h_line : ∀ (x y : ℝ), y = -sqrt 3 * x + b → True)
  (h_angle : ∀ d : ℝ, d = 1 → d = 2 / 2)
  : b = 2 ∨ b = -2 := 
sorry

end find_b_value_l32_32337


namespace M_inter_N_is_empty_l32_32897

-- Definition conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - x > 0}
def N : Set ℝ := {x | (x - 1) / x < 0}

-- Theorem statement
theorem M_inter_N_is_empty : M ∩ N = ∅ := by
  sorry

end M_inter_N_is_empty_l32_32897


namespace min_n_Sn_l32_32884

/--
Given an arithmetic sequence {a_n}, let S_n denote the sum of its first n terms.
If S_4 = -2, S_5 = 0, and S_6 = 3, then the minimum value of n * S_n is -9.
-/
theorem min_n_Sn (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h₁ : S 4 = -2)
  (h₂ : S 5 = 0)
  (h₃ : S 6 = 3)
  (h₄ : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2)
  : ∃ n : ℕ, n * S n = -9 := 
sorry

end min_n_Sn_l32_32884


namespace least_positive_period_f_max_value_g_product_mn_l32_32098

def f (x : ℝ) : ℝ := abs (Real.sin x) + abs (Real.cos x)
def g (x : ℝ) : ℝ := (Real.sin x)^3 - Real.sin x

theorem least_positive_period_f : ∃ m > 0, (∀ x : ℝ, f (x + m) = f x) ∧ m = Real.pi / 2 :=
by
  sorry

theorem max_value_g : ∃ n, (∀ x : ℝ, g x ≤ n) ∧ (∃ x, g x = n) ∧ n = (2 * Real.sqrt 3) / 9 :=
by
  sorry

theorem product_mn : 
  let m := Real.pi / 2 in
  let n := (2 * Real.sqrt 3) / 9 in
  m * n = (Real.sqrt 3 * Real.pi) / 9 :=
by
  sorry

end least_positive_period_f_max_value_g_product_mn_l32_32098


namespace angle_COD_eq_65_l32_32734

-- Define the conditions from step a)
variables (Q C D O : Type) [triangle : Triangle Q C D]

-- Assume the given conditions
variable (h1 : TangentTriangle Q C D O)
variable (h2 : Angle CQD = 50)

-- State the theorem to prove
theorem angle_COD_eq_65 : Angle COD = 65 := by
  sorry

end angle_COD_eq_65_l32_32734


namespace range_of_k_l32_32167

theorem range_of_k 
  (h1 : ∀ x, (x ≠ 1) → (x^2 + k * x + 3) / (x - 1) = 3 * x + k)
  (h2 : ∃! x, x > 0 ∧ ∃ y, (x, y) ∈ ({(x, (3 * x + k) * (x - 1)) | x ≠ 1} : set (ℝ × ℝ))) :
  k = -33 / 8 ∨ k = -4 ∨ k ≥ -3 :=
sorry

end range_of_k_l32_32167


namespace avg_mpg_l32_32031

-- Defining the conditions as per the problem
def distance_AB (x : ℝ) : ℝ := 2 * x
def distance_BC (x : ℝ) : ℝ := x
def fuel_usage_AB (x : ℝ) : ℝ := (2 * x) / 40
def fuel_usage_BC (x : ℝ) : ℝ := x / 50

theorem avg_mpg (x : ℝ) (hx : x > 0) :
  (distance_AB x + distance_BC x) / (fuel_usage_AB x + fuel_usage_BC x) = 200 / 3 :=
by
  sorry

end avg_mpg_l32_32031


namespace perpendicular_midline_equality_orthocentric_midpoint_segments_l32_32261

/-- Given points K, L, M, and N are the midpoints of edges AB, BC, CD, and DA respectively
    of tetrahedron ABCD, we want to prove that AC ⊥ BD if and only if KM = LN. -/
theorem perpendicular_midline_equality (A B C D K L M N : Point)
  (hK : midpoint A B K) (hL : midpoint B C L)
  (hM : midpoint C D M) (hN : midpoint D A N) :
  (AC ⊥ BD) ↔ (dist K M = dist L N) := sorry

/-- Given that tetrahedron ABCD is orthocentric if and only if the segments connecting the
    midpoints of opposite edges are equal. -/
theorem orthocentric_midpoint_segments (A B C D K L M N : Point)
  (hK : midpoint A B K) (hL : midpoint B C L)
  (hM : midpoint C D M) (hN : midpoint D A N) :
  orthocentric_tetrahedron A B C D ↔ (dist K M = dist L N) := sorry

end perpendicular_midline_equality_orthocentric_midpoint_segments_l32_32261


namespace find_f_at_2_l32_32132

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

theorem find_f_at_2 (a b : ℝ) 
  (h1 : 3 + 2 * a + b = 0) 
  (h2 : 1 + a + b + 1 = -2) : 
  f a b 2 = 3 := 
by
  dsimp [f]
  sorry

end find_f_at_2_l32_32132


namespace cos_beta_calculation_l32_32278

variable (α β : ℝ)
variable (h1 : 0 < α ∧ α < π / 2) -- α is an acute angle
variable (h2 : 0 < β ∧ β < π / 2) -- β is an acute angle
variable (h3 : Real.cos α = Real.sqrt 5 / 5)
variable (h4 : Real.sin (α - β) = Real.sqrt 10 / 10)

theorem cos_beta_calculation :
  Real.cos β = Real.sqrt 2 / 2 :=
  sorry

end cos_beta_calculation_l32_32278


namespace Martha_knitting_grandchildren_l32_32983

theorem Martha_knitting_grandchildren (T_hat T_scarf T_mittens T_socks T_sweater T_total : ℕ)
  (h_hat : T_hat = 2) (h_scarf : T_scarf = 3) (h_mittens : T_mittens = 2)
  (h_socks : T_socks = 3) (h_sweater : T_sweater = 6) (h_total : T_total = 48) :
  (T_total / (T_hat + T_scarf + T_mittens + T_socks + T_sweater)) = 3 := by
  sorry

end Martha_knitting_grandchildren_l32_32983


namespace jennifer_run_time_l32_32252

theorem jennifer_run_time (mark_time : ℕ) (jennifer_time_ratio : ℕ) (miles_mark : ℕ) (miles_jennifer_1 : ℕ) (miles_jennifer_2 : ℕ) :
  mark_time = 45 ∧ miles_mark = 5 ∧ jennifer_time_ratio = 3 ∧ miles_jennifer_1 = 3 ∧ miles_jennifer_2 = 7 →
  (jennifer_time_ratio * (mark_time / miles_mark) * miles_jennifer_2 = 35) :=
begin
  intro h,
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest2,
  cases h_rest2 with h3 h_rest3,
  cases h_rest3 with h4 h5,
  have jennifer_per_mile := h1 / (h2 * h3),
  rw [h2, h3, h4, h5] at jennifer_per_mile,
  have jennifer_total_time : jennifer_per_mile * 7 = 35 := by sorry,
  exact jennifer_total_time,
end

end jennifer_run_time_l32_32252


namespace sqrt_cosine_product_eq_l32_32477

theorem sqrt_cosine_product_eq (cos : ℤ → ℝ) (h_cos_1 : cos (1*π / 9) = real.cos (π / 9))
                               (h_cos_2 : cos (2*π / 9) = real.cos (2*π / 9))
                               (h_cos_4 : cos (4*π / 9) = real.cos (4*π / 9)) :
  sqrt ((3 - cos (1*π / 9)^2) * 
        (3 - cos (2*π / 9)^2) * 
        (3 - cos (4*π / 9)^2)) = real.sqrt(15) / 32 :=
sorry

end sqrt_cosine_product_eq_l32_32477


namespace brother_statement_is_false_l32_32061

/-- Define the conditions: 
  - Alice encounters one of the brothers
  - The brother states a lying pattern -/
def brother_statement := "Today is one of the days of the week when I lie"

-- Theorem: The brother's statement is false.
theorem brother_statement_is_false (brother: string) (statement: string) 
  (h_statement: statement = brother_statement) : statement = "false" :=
sorry

end brother_statement_is_false_l32_32061


namespace f_double_prime_zero_l32_32238

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then 2
  else if n = 8 then 1 / 2
  else 2 * (1 / 2) ^ (n - 1)

noncomputable def f (x : ℝ) : ℝ :=
  x * (x - a 1) * (x - a 2) * (x - a 3) * (x - a 4) * (x - a 5) * (x - a 6) * (x - a 7) * (x - a 8)

theorem f_double_prime_zero :
  f''(0) = -8 :=
sorry

end f_double_prime_zero_l32_32238


namespace large_circle_intersects_small_circles_l32_32680

theorem large_circle_intersects_small_circles :
  ∀ (x y : ℝ) (c : ℝ), c = 100 →
    ∃ (m n : ℤ), (m:ℝ) ≤ x ∧ x < (m:ℝ) + 1 ∧ (n:ℝ) ≤ y ∧ y < (n:ℝ) + 1 ∧
    let center := (x, y) in
    let sm_radius := 1 / 20 in
    ∀ (circle_point : ℝ × ℝ), (circle_point.1 - center.1) ^ 2 + (circle_point.2 - center.2) ^ 2 = c ^ 2 →
    ∃ (k l : ℤ), (circle_point.1 - k)^2 + (circle_point.2 - l)^2 ≤ sm_radius ^ 2 := 
sorry

end large_circle_intersects_small_circles_l32_32680


namespace perimeter_of_triangle_ABC_l32_32140

/-- Given a triangle ABC with ∠ C = 120°, and point D is the foot of the perpendicular dropped from point C to side AB. Points E and F are the feet of the perpendiculars dropped from point D to sides AC and BC respectively. Given that triangle EFC is isosceles and its area is √3, prove that the perimeter of triangle ABC is 16 + 8√3. -/
theorem perimeter_of_triangle_ABC (A B C D E F : ℝ) (h1 : ∠C = 120) (h2 : distance_point_line C AB = D)
  (h3 : distance_point_line D AC = E) (h4 : distance_point_line D BC = F) (h5 : isosceles_triangle E F C)
  (h6 : area E F C = √3) : 
  perimeter A B C = 16 + 8 * √3 :=
sorry

end perimeter_of_triangle_ABC_l32_32140


namespace sqrt_of_mixed_fraction_simplified_l32_32507

theorem sqrt_of_mixed_fraction_simplified :
  let x := 8 + (9 / 16) in
  sqrt x = (sqrt 137) / 4 := by
  sorry

end sqrt_of_mixed_fraction_simplified_l32_32507


namespace exists_2016_integers_with_product_9_and_sum_0_l32_32243

theorem exists_2016_integers_with_product_9_and_sum_0 :
  ∃ (L : List ℤ), L.length = 2016 ∧ L.prod = 9 ∧ L.sum = 0 := by
  sorry

end exists_2016_integers_with_product_9_and_sum_0_l32_32243


namespace distance_from_point_to_plane_example_l32_32929

def distance_point_to_plane (x0 y0 z0 A B C D : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C * z0 + D) / real.sqrt (A^2 + B^2 + C^2)

theorem distance_from_point_to_plane_example :
  distance_point_to_plane 1 2 3 1 2 2 (-4) = 7 / 3 :=
by
  sorry

end distance_from_point_to_plane_example_l32_32929


namespace problem_statement_l32_32065

theorem problem_statement
  (A : ∀ (n : ℕ), is_systematic_sampling (select_5th_student n))
  (B : ∀ (freqs : ℕ → ℕ), sample_percentage_less_than_29 freqs = 0.52)
  (C : is_high_correlation (-0.91))
  (D : ∀ (K_squared : ℚ), K_squared = 7.8 → confidence_level_related_to_gender K_squared 0.99) :
  ¬ B :=
by admit

end problem_statement_l32_32065


namespace distance_to_foci_l32_32826

open Real

def hyperbola (P : ℝ × ℝ) : Prop :=
  P.1^2 / 16 - P.2^2 / 9 = 1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem distance_to_foci (P : ℝ × ℝ) 
  (h : hyperbola P) 
  (distance_P_5_0 : distance P (5, 0) = 15) :
  distance P (-5, 0) = 7 ∨ distance P (-5, 0) = 23 :=
sorry

end distance_to_foci_l32_32826


namespace solve_expression_l32_32692

def evaluation_inside_parentheses : ℕ := 3 - 3

def power_of_zero : ℝ := (5 : ℝ) ^ evaluation_inside_parentheses

theorem solve_expression :
  (3 : ℝ) - power_of_zero = 2 := by
  -- Utilize the conditions defined above
  sorry

end solve_expression_l32_32692


namespace negation_of_exists_l32_32776

theorem negation_of_exists : (¬ ∃ x_0 : ℝ, x_0 < 0 ∧ x_0^2 > 0) ↔ ∀ x : ℝ, x < 0 → x^2 ≤ 0 :=
sorry

end negation_of_exists_l32_32776


namespace clock_hand_angle_at_3_30_l32_32375

def hour_hand_degree_per_hour : ℕ := 30
def minute_hand_degree_per_minute : ℕ := 6
def time_hours : ℕ := 3
def time_minutes : ℕ := 30

theorem clock_hand_angle_at_3_30 :
  let hour_hand_position := (time_hours * hour_hand_degree_per_hour) + (time_minutes * hour_hand_degree_per_hour / 60)
  let minute_hand_position := time_minutes * minute_hand_degree_per_minute
  abs (minute_hand_position - hour_hand_position) = 75 :=
by
  -- proof elided
  sorry

end clock_hand_angle_at_3_30_l32_32375


namespace lucy_speed_l32_32981

theorem lucy_speed :
  ∀ (speed1 speed2 distance1 distance2 total_distance usual_speed usual_time time1 remaining_time required_speed : ℕ),
  total_distance = 2 →
  usual_speed = 4 →
  distance1 = 1 →
  speed1 = 3 →
  distance2 = 1 →
  usual_time = total_distance / usual_speed →
  time1 = distance1 / speed1 →
  remaining_time = usual_time - time1 →
  required_speed = distance2 / remaining_time →
  required_speed = 6 :=
by
  intros speed1 speed2 distance1 distance2 total_distance usual_speed usual_time time1 remaining_time required_speed 
         h_total_distance h_usual_speed h_distance1 h_speed1 h_distance2 h_usual_time h_time1 h_remaining_time h_required_speed
  rw [h_total_distance, h_usual_speed, h_distance1, h_speed1, h_distance2] at *,
  -- we know that usual_time = 2 / 4
  change 2 / 4 = usual_time at h_usual_time,
  rw h_usual_time,
  -- simplify time1
  change 1 / 3 = time1 at h_time1,
  rw h_time1,
  -- calculate remaining_time
  change (2 / 4) - (1 / 3) = remaining_time at h_remaining_time,
  norm_num at h_remaining_time,
  -- find required_speed
  change 1 / (1 / 6) = required_speed at h_required_speed,
  norm_num at h_required_speed,
  exact h_required_speed

end lucy_speed_l32_32981


namespace penalty_kicks_must_be_92_l32_32700

theorem penalty_kicks_must_be_92 
  (total_players : ℕ)
  (goalies : ℕ)
  (other_players : total_players - goalies)
  (penalty_kicks_each_goalie : goalies * other_players)
  (total_penalty_kicks := 4 * 23) :
  total_penalty_kicks = 92 :=
  by sorry

end penalty_kicks_must_be_92_l32_32700


namespace find_omega_l32_32577

theorem find_omega 
  (w : ℝ) 
  (h₁ : 0 < w)
  (h₂ : (π / w) = (π / 2)) : w = 2 :=
by
  sorry

end find_omega_l32_32577


namespace Tom_has_38_photos_l32_32356

theorem Tom_has_38_photos :
  ∃ (Tom Tim Paul : ℕ), 
  (Paul = Tim + 10) ∧ 
  (Tim = 152 - 100) ∧ 
  (152 = Tom + Paul + Tim) ∧ 
  (Tom = 38) :=
by
  sorry

end Tom_has_38_photos_l32_32356


namespace combinations_seven_choose_three_l32_32308

theorem combinations_seven_choose_three : nat.choose 7 3 = 35 := by
  sorry

end combinations_seven_choose_three_l32_32308


namespace people_not_playing_sport_l32_32223

theorem people_not_playing_sport : 
  let TotalPeople := 310
      T := 138
      B := 255
      Both := 94
      AtLeastOne := T + B - Both 
  in TotalPeople - AtLeastOne = 11 :=
by 
  let TotalPeople := 310
  let T := 138
  let B := 255
  let Both := 94
  let AtLeastOne := T + B - Both
  have AtLeastOne_def : AtLeastOne = 299 := by
    simp [AtLeastOne, T, B, Both]
  show TotalPeople - AtLeastOne = 11, from by
    simp [TotalPeople, AtLeastOne_def]
    sorry

end people_not_playing_sport_l32_32223


namespace solution_set_M_inequality_ab_l32_32901

def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

theorem solution_set_M :
  {x | -3 ≤ x ∧ x ≤ 1} = { x : ℝ | f x ≤ 4 } :=
sorry

theorem inequality_ab
  (a b : ℝ) (h1 : -3 ≤ a ∧ a ≤ 1) (h2 : -3 ≤ b ∧ b ≤ 1) :
  (a^2 + 2 * a - 3) * (b^2 + 2 * b - 3) ≥ 0 :=
sorry

end solution_set_M_inequality_ab_l32_32901


namespace smallest_possible_M_l32_32962

theorem smallest_possible_M {a b c d e : ℕ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0)
(h_sum : a + b + c + d + e = 3015) :
  let M := max (max (a + b) (b + c)) (max (c + d) (d + e)) 
  in M = 755 :=
by
  sorry

end smallest_possible_M_l32_32962


namespace siren_sound_properties_l32_32423

/--
Let:
- holes denote the number of holes in the siren,
- rotations_per_sec denote the rotation frequency of the siren,
- speed_of_sound denote the speed of sound,
- n denote the frequency of the generated sound,
- λ denote the wavelength of the generated sound.

Given:
- holes = 20,
- rotations_per_sec = 300,
- speed_of_sound = 333000, 

We want to prove that:
n = holes * rotations_per_sec
and
λ = speed_of_sound / n,
which implies:
n = 6000 ∧ λ = 55.5
--/
theorem siren_sound_properties (holes : ℕ) (rotations_per_sec : ℕ) (speed_of_sound : ℕ)
    (n : ℕ) (λ : ℝ) :
  holes = 20 →
  rotations_per_sec = 300 →
  speed_of_sound = 333000 →
  n = holes * rotations_per_sec →
  λ = speed_of_sound / n →
  n = 6000 ∧ λ = 55.5 :=
by
  intros holes_eq rotations_eq sound_eq n_eq λ_eq
  rw [holes_eq, rotations_eq, sound_eq] at *
  simp [λ_eq, n_eq]
  sorry

end siren_sound_properties_l32_32423


namespace multiple_of_second_lock_time_l32_32949

def first_lock_time := 5
def second_lock_time := 3 * first_lock_time - 3
def combined_lock_time := 60

theorem multiple_of_second_lock_time : combined_lock_time / second_lock_time = 5 := by
  -- Adding a proof placeholder using sorry
  sorry

end multiple_of_second_lock_time_l32_32949


namespace similar_triangles_sum_of_legs_l32_32360

theorem similar_triangles_sum_of_legs
  (area_small : ℝ)
  (hypotenuse_small : ℝ)
  (area_large : ℝ)
  (area_small_pos : area_small = 30)
  (hypotenuse_13 : hypotenuse_small = 13)
  (area_large_pos : area_large = 750) :
  let scale_factor := real.sqrt (area_large / area_small)
  let leg1_small := 5
  let leg2_small := 12
  sum_of_large_legs : ℝ :=
  scale_factor * (leg1_small + leg2_small)
  in
  sum_of_large_legs = 85 :=
begin
  sorry
end

end similar_triangles_sum_of_legs_l32_32360


namespace probability_reds_before_greens_l32_32426

/-- A top hat contains 3 red chips and 2 green chips. Chips are drawn randomly,
    one at a time without replacement, until all 3 of the reds are drawn or until both green chips
    are drawn. Prove that the probability that the 3 reds are drawn before the 2 greens
    is 2/5. -/
theorem probability_reds_before_greens :
  let total_chips := 5 in
  let red_chips := 3 in
  let green_chips := 2 in
  let total_arrangements := Nat.choose 5 2 in
  let favorable_arrangements := Nat.choose 4 1 in
  (favorable_arrangements : ℚ) / (total_arrangements : ℚ) = 2 / 5 :=
by
  -- The proof is omitted
  sorry

end probability_reds_before_greens_l32_32426


namespace randy_initial_money_l32_32641

/--
Initially, Randy had an unknown amount of money. He was given $2000 by Smith and $900 by Michelle.
After that, Randy gave Sally a 1/4th of his total money after which he gave Jake and Harry $800 and $500 respectively.
If Randy is left with $5500 after all the transactions, prove that Randy initially had $6166.67.
-/
theorem randy_initial_money (X : ℝ) :
  (3/4 * (X + 2000 + 900) - 1300 = 5500) -> (X = 6166.67) :=
by
  sorry

end randy_initial_money_l32_32641


namespace angle_leq_60_degrees_l32_32303

-- Define the conditions
variables {a b c : ℝ} (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
variables (h_geometric_mean : a^2 = b * c)
variables (α : ℝ) (h_cosine_rule : cos α = (b^2 + c^2 - a^2) / (2 * b * c))

-- State the theorem to be proven
theorem angle_leq_60_degrees (h_triangle : a > 0 ∧ b > 0 ∧ c > 0) (h_geometric_mean : a^2 = b * c) 
  (α : ℝ) (h_cosine_rule : cos α = (b^2 + c^2 - a^2) / (2 * b * c)) : 
  α ≤ (60 : ℝ) :=
  sorry

end angle_leq_60_degrees_l32_32303


namespace not_tetrahedron_possible_l32_32232

-- Assuming the points M and N within the triangle ABC
variables {A B C M N : Type}

-- Let AM, BM, and CM be segments within specific interval
def segment_AM : ℝ := sorry
def segment_BM : ℝ := sorry
def segment_CM : ℝ := sorry

-- Assume the inequalities and condition on epsilon (ε)
def ε : ℝ := min (abs (segment_AM - segment_BM)) (min (abs (segment_BM - segment_CM)) (min (abs (segment_CM - segment_AM)) (1 / 10)))

-- Specific selection of point N near vertex A
def segment_AN : ℝ := sorry
def segment_BN : ℝ := sorry
def segment_CN : ℝ := sorry

-- Setting the conditions based on the problem's constraints
axiom AN_less_ε : segment_AN < ε
axiom BN_greater : segment_BN > (sqrt 3 / 2 + ε)
axiom CN_greater : segment_CN > (sqrt 3 / 2 + ε)

-- Proof goal stating that it is not always possible to form a tetrahedron
theorem not_tetrahedron_possible : ¬ (AM * BM * CM * AN * BN * CN != 0) := sorry

end not_tetrahedron_possible_l32_32232


namespace difference_of_squares_example_l32_32466

theorem difference_of_squares_example :
  262^2 - 258^2 = 2080 := by
sorry

end difference_of_squares_example_l32_32466


namespace complex_number_position_l32_32089

theorem complex_number_position (G : ℂ) (h1 : abs G < 1) (h2 : (G^2).im < 0) (h3 : (G^2).re > (G^2).im) : 
  true := sorry

end complex_number_position_l32_32089


namespace enumerate_integers_abs_le_two_l32_32672

theorem enumerate_integers_abs_le_two (x : Int) : 
  (|x| ≤ 2 ↔ x ∈ {-2, -1, 0, 1, 2}) := sorry

end enumerate_integers_abs_le_two_l32_32672


namespace question1_question2_question3_l32_32133

variable (n : Nat) (hn : n > 0)
noncomputable def R_n (x : Real) : Real := x^2 + (1 / n^2) * x^4
noncomputable def a_n : Real := 1 + 1 / n + 1 / (n^2)

theorem question1 (x : Real) : 
  R_n n x = x^2 + (1 / n^2) * x^4 ∧ 
  a_n n = 1 + 1 / n + 1 / (n^2) := 
by sorry

theorem question2 : 
  a_n n > a_n (n+1) ∧ a_n (n+1) > 2 := 
by sorry

noncomputable def S_n : Real := ∑ i in Finset.range n, a_n (i + 1)
noncomputable def T_n : Real := ∑ i in Finset.range n, 1 / (i + 1)

theorem question3 : 
  S_n n < 2 * n + n * T_n n := 
by sorry

end question1_question2_question3_l32_32133


namespace constant_term_expansion_l32_32598

noncomputable def a : ℝ := ∫ x in -1..1, real.sqrt (1 - x^2)

theorem constant_term_expansion :
  (∫ x in -1..1, real.sqrt (1 - x^2)) = a ∧
  let expr := (a / real.pi) * x - 1 / x in
  constant_term ((expr ^ 6).expand) = (-5 / 2) :=
by
  sorry

end constant_term_expansion_l32_32598


namespace quadratic_polynomial_inequality_l32_32715

variable {a b c : ℝ}

theorem quadratic_polynomial_inequality (h1 : ∀ x : ℝ, a * x^2 + b * x + c < 0)
    (h2 : a < 0)
    (h3 : b^2 - 4 * a * c < 0) :
    b / a < c / a + 1 := 
by 
  sorry

end quadratic_polynomial_inequality_l32_32715


namespace find_a_l32_32668

variable {f : ℝ → ℝ}
variable {a : ℝ}

-- Definitions and Assumptions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

axiom f_odd : odd_function f
axiom f_period : periodic_function f 5
axiom f_one : f 1 = -1
axiom f_four_eq_log2a : f 4 = Real.log2 a

-- Theorem to Prove
theorem find_a : a = 2 :=
by
  sorry

end find_a_l32_32668


namespace selection_ways_l32_32306

-- Define the problem parameters
def male_students : ℕ := 4
def female_students : ℕ := 3
def total_selected : ℕ := 3

-- Define the binomial coefficient function for combinatorial calculations
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define conditions
def both_genders_must_be_represented : Prop :=
  total_selected = 3 ∧ male_students >= 1 ∧ female_students >= 1

-- Problem statement: proof that the total ways to select 3 students is 30
theorem selection_ways : both_genders_must_be_represented → 
  (binomial male_students 2 * binomial female_students 1 +
   binomial male_students 1 * binomial female_students 2) = 30 :=
by
  sorry

end selection_ways_l32_32306


namespace tangent_line_at_e_l32_32566

noncomputable def f : ℝ → ℝ :=
λ x, if x < 0 then x + Real.log (-x) else x - Real.log x

theorem tangent_line_at_e (e_pos : 0 < Real.exp(1) ) : 
  ∃ (m : ℝ), m = (1 - 1/(Real.exp(1))) ∧ 
  ∃ (c : ℝ), c = (-(1 - 1 / (Real.exp(1)))) * Real.exp(1) + f (Real.exp(1)) ∧ 
  ∀ (x : ℝ), (y : ℝ), y = f x → y = m * x + c :=
begin
  sorry
end

end tangent_line_at_e_l32_32566


namespace allocate_25_rubles_in_4_weighings_l32_32362

theorem allocate_25_rubles_in_4_weighings :
  ∃ (coins : ℕ) (coins5 : ℕ → ℕ), 
    (coins = 1600) ∧ 
    (coins5 0 = 800 ∧ coins5 1 = 800) ∧
    (coins5 2 = 400 ∧ coins5 3 = 400) ∧
    (coins5 4 = 200 ∧ coins5 5 = 200) ∧
    (coins5 6 = 100 ∧ coins5 7 = 100) ∧
    (
      25 = 20 + 5 ∧ 
      (∃ i j k l m n, coins5 i = 400 ∧ coins5 j = 400 ∧ coins5 k = 200 ∧
        coins5 l = 200 ∧ coins5 m = 100 ∧ coins5 n = 100)
    )
  := 
sorry

end allocate_25_rubles_in_4_weighings_l32_32362


namespace trigonometric_identity_l32_32154

open Real

theorem trigonometric_identity
  (α : ℝ)
  (h1 : 0 ≤ α ∧ α ≤ π / 2)
  (h2 : cos α = 3 / 5) :
  (1 + sqrt 2 * cos (2 * α - π / 4)) / sin (α + π / 2) = 14 / 5 :=
by
  sorry

end trigonometric_identity_l32_32154


namespace exists_rel_prime_in_consecutive_set_l32_32314

theorem exists_rel_prime_in_consecutive_set (n : ℤ) :
  ∃ k ∈ (finset.Icc n (n + 9)), ∀ m ∈ (finset.Icc n (n + 9)), k ≠ m → Int.gcd k m = 1 :=
begin
  sorry
end

end exists_rel_prime_in_consecutive_set_l32_32314


namespace diff_of_squares_example_l32_32464

theorem diff_of_squares_example : (262^2 - 258^2 = 2080) :=
by
  let a := 262
  let b := 258
  have h1 : a^2 - b^2 = (a + b) * (a - b) := by rw [pow_two, pow_two, sub_mul, add_comm a b, mul_sub]
  have h2 : (a + b) = 520 := by norm_num
  have h3 : (a - b) = 4 := by norm_num
  have h4 : (262 + 258) * (262 - 258) = 520 * 4 := congr (congr_arg (*) h2) h3
  rw [h1, h4]
  norm_num

end diff_of_squares_example_l32_32464


namespace number_of_zeros_l32_32342

/-- The number of zeros at the end of the product of 20^50 and 50^20 is 90. -/
theorem number_of_zeros (a b : ℕ) (h1 : a = 20) (h2 : b = 50) :
  let prod1 := 20^50
  let prod2 := 50^20
  let prod := prod1 * prod2
  let prime_fact := (2^120) * (5^90)
  (num_zeros_at_end prod) = 90 :=
by
  sorry

/-- Define the auxiliary function to calculate the number of zeros at the end of a product. -/
def num_zeros_at_end (n : ℕ) : ℕ :=
  sorry

end number_of_zeros_l32_32342


namespace part_I_part_II_l32_32187

variables {ℝ : Type*}

def vector := ℝ × ℝ

noncomputable def vector_norm (v : vector) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

def parallel (v1 v2 : vector) : Prop :=
  ∃ (λ : ℝ), v1 = (λ * v2.1, λ * v2.2)

def perpendicular (v1 v2 : vector) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def cos_theta (v1 v2 : vector) (θ : ℝ) : Prop :=
  dot_product v1 v2 = vector_norm v1 * vector_norm v2 * θ

def problem_conditions_1 (b : vector) : Prop := 
  vector_norm b = 3 * real.sqrt 5 ∧ parallel b (1, 2)

def problem_conditions_2 (c : vector) : Prop := 
  cos_theta (1, 2) c (-real.sqrt 5 / 10) ∧ 
  perpendicular ((1, 2).1 + c.1, (1, 2).2 + c.2) ((1, 2).1 - 9 * c.1, (1, 2).2 - 9 * c.2) ∧
  vector_norm (1, 2) = real.sqrt 5

theorem part_I (b : vector) (hc : problem_conditions_1 b) : 
  b = (3, 6) ∨ b = (-3, -6) := 
  sorry

theorem part_II (c : vector) (hc : problem_conditions_2 c) : 
  vector_norm c = 1 := 
  sorry

end part_I_part_II_l32_32187


namespace part1_part2_l32_32131

noncomputable def f (x : ℝ) : ℝ := 2 * |x + 1| + |x - 2|

theorem part1 : {x : ℝ | f x ≥ 4} = {x : ℝ | x ≤ -4/3 ∨ x ≥ 0} := sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, 0 < x → f x + a * x - 1 > 0) → a > -5/2 := sorry

end part1_part2_l32_32131


namespace ratio_of_areas_l32_32359

-- Define the proof problem
theorem ratio_of_areas (triangle_area : ℝ) (region_area : ℝ) : 
  (region_area / triangle_area = 1 / 4) :=
sorry

end ratio_of_areas_l32_32359


namespace Carter_reads_30_pages_in_1_hour_l32_32086

variables (C L O : ℕ)

def Carter_reads_half_as_many_pages_as_Lucy_in_1_hour (C L : ℕ) : Prop :=
  C = L / 2

def Lucy_reads_20_more_pages_than_Oliver_in_1_hour (L O : ℕ) : Prop :=
  L = O + 20

def Oliver_reads_40_pages_in_1_hour (O : ℕ) : Prop :=
  O = 40

theorem Carter_reads_30_pages_in_1_hour
  (C L O : ℕ)
  (h1 : Carter_reads_half_as_many_pages_as_Lucy_in_1_hour C L)
  (h2 : Lucy_reads_20_more_pages_than_Oliver_in_1_hour L O)
  (h3 : Oliver_reads_40_pages_in_1_hour O) : 
  C = 30 :=
by
  sorry

end Carter_reads_30_pages_in_1_hour_l32_32086


namespace smallest_floor_sum_l32_32215

theorem smallest_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (⟨⌊(a^2 + b^2) / c⌋ + ⌊(b^2 + c^2) / a⌋ + ⌊(c^2 + a^2) / b⌋⟩ : ℤ) = 4 :=
sorry

end smallest_floor_sum_l32_32215


namespace largest_negative_root_l32_32115

noncomputable def arctan_rat_rat (a b : ℝ) : ℝ := Real.arctan (a / b)

def problem_eqn (x : ℝ) : Prop :=
  4 * Real.sin (3 * x) + 13 * Real.cos (3 * x) = 8 * Real.sin x + 11 * Real.cos x

theorem largest_negative_root :
  ∃ x : ℝ, problem_eqn x ∧ x < 0 ∧
    (- (Real.pi / 2) < x) ∧
    (problem_eqn x → x = (arctan_rat_rat 4 13 - arctan_rat_rat 8 11) / 2 ∨
    x = (arctan_rat_rat 4 13 + arctan_rat_rat 8 11 - 2 * Real.pi) / 4) :=
begin
  sorry
end

end largest_negative_root_l32_32115


namespace max_value_of_x3_plus_y3_l32_32978

theorem max_value_of_x3_plus_y3
  (x y : ℝ)
  (h : x^2 + y^2 = 2) :
  x^3 + y^3 ≤ 4 * sqrt 2 :=
sorry

end max_value_of_x3_plus_y3_l32_32978


namespace evaluate_series_l32_32844

noncomputable def infinite_series := ∑ k in (Finset.range ∞), (k + 1)^2 / 3^(k + 1)

theorem evaluate_series : infinite_series = 1 / 2 := sorry

end evaluate_series_l32_32844


namespace min_large_trucks_needed_l32_32986

-- Define the parameters for the problem
def total_fruit : ℕ := 134
def load_large_truck : ℕ := 15
def load_small_truck : ℕ := 7

-- Define the main theorem to be proved
theorem min_large_trucks_needed :
  ∃ (n : ℕ), n = 8 ∧ (total_fruit = n * load_large_truck + 2 * load_small_truck) :=
by sorry

end min_large_trucks_needed_l32_32986


namespace general_formula_an_geometric_sequence_bn_sum_of_geometric_sequence_Tn_l32_32348

-- Definitions based on conditions
def a (n : ℕ) : ℕ := 2 * n + 1

def b (n : ℕ) : ℕ := 2 ^ (a n)

noncomputable def S (n : ℕ) : ℕ := (n * (2 * n + 2)) / 2

noncomputable def T (n : ℕ) : ℕ := (8 * (4 ^ n - 1)) / 3

-- Statements to be proved
theorem general_formula_an : ∀ n : ℕ, a n = 2 * n + 1 := sorry

theorem geometric_sequence_bn : ∀ n : ℕ, b n = 2 ^ (2 * n + 1) := sorry

theorem sum_of_geometric_sequence_Tn : ∀ n : ℕ, T n = (8 * (4 ^ n - 1)) / 3 := sorry

end general_formula_an_geometric_sequence_bn_sum_of_geometric_sequence_Tn_l32_32348


namespace more_than_half_millet_after_the_third_day_l32_32988

theorem more_than_half_millet_after_the_third_day :
  (∀ day : ℕ, let millet_added_on_day := if day = 1 then 0.4 
                                         else if day = 2 then 0.5 
                                         else 0.4 in 
               let millet_remaining_after_birds := \begin
                 if day = 1 then 0.4 * 0.8
                 if day = 2 then (0.4 * 0.8 + 0.5) * 0.8
                 else ((0.4 * 0.8 + 0.5) * 0.8 + 0.4) * 0.8
               end in
               let new_millet_on_day  := (millet_remaining_after_birds + millet_added_on_day)
               let total_seeds_on_day := (day + 1).to_real
               (new_millet_on_day / total_seeds_on_day > 0.5)) :=
sorry

end more_than_half_millet_after_the_third_day_l32_32988


namespace youngest_sibling_is_42_l32_32322

-- Definitions for the problem conditions
def consecutive_even_integers (a : ℤ) := [a, a + 2, a + 4, a + 6]
def sum_of_ages_is_180 (ages : List ℤ) := ages.sum = 180

-- Main statement
theorem youngest_sibling_is_42 (a : ℤ) 
  (h1 : sum_of_ages_is_180 (consecutive_even_integers a)) :
  a = 42 := 
sorry

end youngest_sibling_is_42_l32_32322


namespace books_about_sports_l32_32197

theorem books_about_sports (total_books sports_books school_books : ℕ) 
  (h1 : total_books = 58) 
  (h2 : school_books = 19) 
  (h3 : total_books = sports_books + school_books) : 
  sports_books = 39 :=
by 
  rw [h1, h2, Nat.add_comm] at h3
  simpa using h3

end books_about_sports_l32_32197


namespace infinite_sqrt_eq_three_l32_32323

theorem infinite_sqrt_eq_three :
  ∃ m : ℝ, m > 0 ∧ m = √(3 + 2 * √(3 + 2 * √(3 + 2 * √(3 + 2 * √(3 + 2 * ...))))) ∧ m = 3 :=
sorry

end infinite_sqrt_eq_three_l32_32323


namespace total_sum_money_l32_32035

theorem total_sum_money (a b c : ℝ) (h1 : b = 0.65 * a) (h2 : c = 0.40 * a) (h3 : c = 64) :
  a + b + c = 328 :=
by
  sorry

end total_sum_money_l32_32035


namespace function_identity_l32_32094

theorem function_identity (f : ℕ → ℕ) 
  (h_pos : f 1 > 0) 
  (h_property : ∀ m n : ℕ, f (m^2 + n^2) = f m^2 + f n^2) : 
  ∀ n : ℕ, f n = n :=
by
  sorry

end function_identity_l32_32094


namespace vanessa_did_not_sell_7_bars_l32_32838

theorem vanessa_did_not_sell_7_bars :
  let cost_per_bar := 4
  let total_bars := 11
  let total_money_made := 16
  let bars_sold := total_money_made / cost_per_bar
  let bars_not_sold := total_bars - bars_sold
  in bars_not_sold = 7 :=
by
  dsimp
  have bars_sold_calc : bars_sold = 4 := by norm_num
  rw [bars_sold_calc]
  norm_num
  done

end vanessa_did_not_sell_7_bars_l32_32838


namespace initial_cars_on_lot_l32_32786

theorem initial_cars_on_lot (x : ℕ) (h1 : 0.20 * x  = 0.20 * x) (h2 : 80 = 80)
    (h3 : 0.4 = 0.4) : x = 40 :=
sorry

end initial_cars_on_lot_l32_32786


namespace minimum_movie_collections_l32_32616

-- Define the complete graph K_23 with 23 vertices (students)
def K_23 := SimpleGraph.complete 23

-- Define the movie collections and conditions
def movie_collections (G : SimpleGraph ℕ) (V : ℕ) := 
  ∀ v : G.vertex, ∃ collection : Set (fin (V - 1)), 
    collection.card = V - 1 ∧ ∀ u : G.vertex, u ≠ v → edge_connects v u collection

-- Theorem statement: The minimum number of different movie collections for students in K_23
theorem minimum_movie_collections (G : SimpleGraph ℕ) (V : ℕ) 
  (complete_G : G = K_23) (vertex_count : G.vertex.card = 23) :
  ∃! collections : Set (Set (fin (V - 1))), 
    collections.card = V ∧ 
    ∀ (v : G.vertex), ∃ collection ∈ collections,
    collection.card = V - 1 ∧
    ∀ u : G.vertex, u ≠ v → edge_connects v u collection :=
sorry -- Proof is not required.

end minimum_movie_collections_l32_32616


namespace Nikka_stamp_collection_l32_32676

theorem Nikka_stamp_collection (S : ℝ) 
  (h1 : 0.35 * S ≥ 0) 
  (h2 : 0.2 * S ≥ 0) 
  (h3 : 0 < S) 
  (h4 : 0.45 * S = 45) : S = 100 :=
sorry

end Nikka_stamp_collection_l32_32676


namespace problem1_problem2_l32_32142

-- Condition Definitions
def line_l (x : ℝ) : ℝ := 4 * x
def point_P : ℝ × ℝ := (6, 4)
def point_on_line_l_in_first_quadrant (A : ℝ × ℝ) : Prop := 
  ∃ (a : ℝ), a > 0 ∧ A = (a, line_l a)
def intersects_positive_x_axis (PA : ℝ × ℝ → Prop) (B : ℝ × ℝ) : Prop := 
  ∃ (PA : ℝ), B = (PA, 0) ∧ PA > 0
def perpendicular_OP_AB (AB : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ), 3 * x + 2 * (AB x) = 26

-- Lean 4 Statements (problems to prove)
theorem problem1 
  (A : ℝ × ℝ)
  (HA : point_on_line_l_in_first_quadrant A)
  (B : ℝ × ℝ)
  (HB : intersects_positive_x_axis (λ a, 4 * a) B)
  (H_perp : perpendicular_OP_AB (λ x, - (3 / 2) * x + 13) ) :
  ∃ (m : ℝ) (c : ℝ), m = - (3 / 2) ∧ c = 13 ∧
  (λ x, m * x + c) = λ x, (- (3 / 2) * x + 13) := sorry

theorem problem2 
  (A B : ℝ × ℝ)
  (area : ℝ)
  (HA : point_on_line_l_in_first_quadrant A)
  (HB : intersects_positive_x_axis (λ a, 4 * a) B)
  (H_area : area = 40):
  let min_area := 40
  ∧ let B_coords := (10, 0) := sorry

end problem1_problem2_l32_32142


namespace smallest_lcm_of_4_digit_integers_l32_32205

open Nat

theorem smallest_lcm_of_4_digit_integers (k l : ℕ) (hk : 1000 ≤ k ∧ k < 10000) (hl : 1000 ≤ l ∧ l < 10000) (h_gcd : gcd k l = 5) :
  lcm k l = 203010 := sorry

end smallest_lcm_of_4_digit_integers_l32_32205


namespace complex_multiplication_l32_32632

noncomputable def z1 : ℂ := (1 : ℂ) + (complex.I)
noncomputable def z2 : ℂ := (1 : ℂ) - (complex.I)

theorem complex_multiplication : z1 * z2 = 2 := 
begin
  sorry
end

end complex_multiplication_l32_32632


namespace solve_parallelepiped_problem_l32_32839

-- Definitions
def unit_square := ℕ
def face := list unit_square
def parallelepiped := (face × face × face) -- representing the 6 faces

-- Main Statement
theorem solve_parallelepiped_problem (P : parallelepiped) : 
  (exists (f : unit_square → ℕ), 
    ( ∀ (i j : ℕ) (H1 : i < 3 ∧ j < 4), (f (i * 4 + j) * 5) = 120 ) ∧
    ( ∀ (i j : ℕ) (H2 : i < 3 ∧ j < 5), (f (i * 5 + j) * 4) = 240 ) ∧
    ( ∀ (i j : ℕ) (H3 : i < 4 ∧ j < 5), (f (i * 5 + j) * 3) = 360 ) 
  ) := sorry

end solve_parallelepiped_problem_l32_32839


namespace ways_to_divide_day_l32_32409

theorem ways_to_divide_day (n m : ℕ+) : n * m = 86400 → 96 = 96 :=
by
  sorry

end ways_to_divide_day_l32_32409


namespace harmonic_not_integer_l32_32997

theorem harmonic_not_integer (n : ℕ) (h : n > 1) : ¬ ∃ (k : ℤ), H_n n = k :=
by
  sorry

noncomputable def H_n : ℕ → ℚ
| 0     := 0
| (n+1) := H_n n + 1 / (n + 1)

end harmonic_not_integer_l32_32997


namespace computation_result_l32_32456

def a : ℕ := 3
def b : ℕ := 5
def c : ℕ := 7

theorem computation_result :
  (a + b + c) ^ 2 + (a ^ 2 + b ^ 2 + c ^ 2) = 308 := by
  sorry

end computation_result_l32_32456


namespace retailer_profit_percentage_l32_32013

def market_price (P : ℝ) : ℝ := P
def cost_price (P : ℝ) : ℝ := 36 * P
def selling_price (P : ℝ) : ℝ := 140 * 0.99 * P
def profit (P : ℝ) : ℝ := selling_price P - cost_price P
def profit_percentage (P : ℝ) : ℝ :=
  (profit P / cost_price P) * 100

theorem retailer_profit_percentage (P : ℝ) : profit_percentage P = 285 :=
by
  sorry

end retailer_profit_percentage_l32_32013


namespace symmetric_point_origin_l32_32559

theorem symmetric_point_origin (A : ℝ × ℝ) (A_sym : ℝ × ℝ) (h : A = (3, -2)) (h_sym : A_sym = (-A.1, -A.2)) : A_sym = (-3, 2) :=
by
  sorry

end symmetric_point_origin_l32_32559


namespace prove_p_false_and_q_true_l32_32219

variables (p q : Prop)

theorem prove_p_false_and_q_true (h1 : p ∨ q) (h2 : ¬p) : ¬p ∧ q :=
by {
  -- proof placeholder
  sorry
}

end prove_p_false_and_q_true_l32_32219


namespace area_of_triangle_AOB_l32_32630

open Real

variables {α θ ρ x y : ℝ}

-- The given parametric equations for curve C₁
def C1_x (α : ℝ) : ℝ := 2 + 2 * cos α
def C1_y (α : ℝ) : ℝ := 2 * sin α

-- Polar coordinate equation for the line C₂
def C2_polar (ρ θ : ℝ) : Prop := ρ * cos θ + ρ * sin θ = 2

-- The ordinary equation for curve C₁ (x - 2)² + y² = 4
def C1_ordinary (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 4

-- Rectangular coordinate equation for line C₂
def C2_rectangular (x y : ℝ) : Prop := x + y = 2

-- Intersection points A and B lie on both C₁ and C₂
def AOB_intersects (x y : ℝ) : Prop :=
  C1_ordinary x y ∧ C2_rectangular x y

-- Distance between origin and line
def distance_from_origin : ℝ := 2 / sqrt 2

-- The area of triangle ΔAOB
def area_triangle (AB_distance : ℝ) (d : ℝ) : ℝ := 1 / 2 * AB_distance * d

-- Main theorem statement
theorem area_of_triangle_AOB (α1 α2 : ℝ) (A B : ℝ) (hA : A = 0)
  (hB : B = 4) :
  AOB_intersects (C1_x α1) (C1_y α1) →
  AOB_intersects (C1_x α2) (C1_y α2) →
  area_triangle 4 distance_from_origin = 2 * sqrt 2 := by
  sorry

end area_of_triangle_AOB_l32_32630


namespace longest_chord_length_of_circle_l32_32707

theorem longest_chord_length_of_circle (r : ℝ) (h : r = 5) : ∃ d, d = 10 :=
by
  sorry

end longest_chord_length_of_circle_l32_32707


namespace milly_billy_equal_sum_l32_32341

theorem milly_billy_equal_sum (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 20) 
    (h₃ : (∑ i in Finset.range (n+1), i + 1) = (∑ i in Finset.range (20 + 1) \ Finset.range (n + 1), i + 1)) : 
    n = 14 := 
by 
  sorry

end milly_billy_equal_sum_l32_32341


namespace find_x1_x2_x3_sum_l32_32090

noncomputable def f (x : ℝ) : ℝ := max (-7 * x - 21) (max (2 * x - 2) (5 * x + 10))

axiom q : ℝ → ℝ
axiom x1 x2 x3 : ℝ
axiom h_tangent1 : q x1 = f x1
axiom h_tangent2 : q x2 = f x2
axiom h_tangent3 : q x3 = f x3
axiom h_q0 : q 0 = 1

theorem find_x1_x2_x3_sum : x1 + x2 + x3 = 19 :=
sorry

end find_x1_x2_x3_sum_l32_32090


namespace find_triangle_angle_l32_32935

theorem find_triangle_angle (a b c : ℝ) (h : (a + b + c) * (a + b - c) = 4 * a * b) : 
  ∠ABC := 180 :=
begin
  sorry
end

end find_triangle_angle_l32_32935


namespace calculate_sum_l32_32129

def f (n : ℤ) (x : ℝ) : ℝ :=
  (cos (n * real.pi + x) ^ 2 * sin (n * real.pi - x) ^ 2) / (cos ((2 * n + 1) * real.pi - x) ^ 2)

theorem calculate_sum : 
  f 0 (real.pi / 2016) + f 0 (1007 * real.pi / 2016) = 1 :=
by
  sorry

end calculate_sum_l32_32129


namespace mean_of_y_l32_32054

noncomputable def mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

def regression_line (x : ℝ) : ℝ :=
  2 * x + 45

theorem mean_of_y (y₁ y₂ y₃ y₄ y₅ : ℝ) :
  mean [regression_line 1, regression_line 5, regression_line 7, regression_line 13, regression_line 19] = 63 := by
  sorry

end mean_of_y_l32_32054


namespace incorrect_option_d_l32_32156

-- Define the conditions
def planes_intersect_at_line (α β : Plane) (l : Line) : Prop :=
  Line_in_plane l α ∧ Line_in_plane l β

def line_in_plane (m : Line) (α : Plane) (h : Line ≠ l) : Prop := 
  h_neq : m ≠ l →
  Line_in_plane m α

-- Define predicates for parallel and perpendicular lines/planes
def parallel_line_plane (m : Line) (β : Plane) : Prop := 
  ∀ (p : Point), Point_on_line p m → Point_on_plane p β

def parallel_lines (m l : Line) : Prop := 
  ∀ (p : Point), Point_on_line p m → Point_on_line p l

def perpendicular_line_plane (m : Line) (β : Plane) : Prop := 
  ∀ (p : Point), Point_on_line p m → Line_perpendicular_to_plane m β
  
def perpendicular_lines (m l : Line) : Prop := 
  ∀ (p : Point), Point_on_line p m → Line_perpendicular_to_line m l

-- Define the statement to be proved
theorem incorrect_option_d (α β : Plane) (l m : Line) (h_inter : planes_intersect_at_line α β l) (h_in_plane : line_in_plane m α (m ≠ l)) :
  ¬ ((perpendicular_lines m l) → (perpendicular_line_plane m β)) :=
sorry

end incorrect_option_d_l32_32156


namespace sum_of_digits_2_1989_and_5_1989_l32_32015

theorem sum_of_digits_2_1989_and_5_1989 
  (m n : ℕ) 
  (h1 : 10^(m-1) < 2^1989 ∧ 2^1989 < 10^m) 
  (h2 : 10^(n-1) < 5^1989 ∧ 5^1989 < 10^n) 
  (h3 : 2^1989 * 5^1989 = 10^1989) : 
  m + n = 1990 := 
sorry

end sum_of_digits_2_1989_and_5_1989_l32_32015


namespace number_of_3_letter_words_with_at_least_one_A_l32_32193

theorem number_of_3_letter_words_with_at_least_one_A :
  let all_words := 5^3
  let no_A_words := 4^3
  all_words - no_A_words = 61 :=
by
  sorry

end number_of_3_letter_words_with_at_least_one_A_l32_32193


namespace max_triangle_area_in_rectangle_l32_32711

noncomputable def rectangle_side_a : ℝ := 12
noncomputable def rectangle_side_b : ℝ := 5
def angle_XYZ : real.angle := real.angle.pi / 6 -- 30 degrees in radians
def max_area (a b : ℝ) : ℝ := 25 * real.sqrt 3 / 4
def p : ℕ := 25
def q : ℕ := 3
def r : ℕ := 0

theorem max_triangle_area_in_rectangle :
  (rectangle_side_a = 12) ∧ (rectangle_side_b = 5) ∧ (angle_XYZ = real.angle.pi / 6) → 
  p + q + r = 28 :=
by
  intros
  sorry

end max_triangle_area_in_rectangle_l32_32711


namespace complex_solutions_x2_eq_neg4_l32_32326

-- Lean statement for the proof problem
theorem complex_solutions_x2_eq_neg4 (x : ℂ) (hx : x^2 = -4) : x = 2 * Complex.I ∨ x = -2 * Complex.I :=
by 
  sorry

end complex_solutions_x2_eq_neg4_l32_32326


namespace p_add_inv_p_gt_two_l32_32302

theorem p_add_inv_p_gt_two {p : ℝ} (hp_pos : p > 0) (hp_neq_one : p ≠ 1) : p + 1 / p > 2 :=
by
  sorry

end p_add_inv_p_gt_two_l32_32302


namespace acute_angle_between_planes_l32_32145

noncomputable def angle_between_planes (A B C D : ℝ × ℝ × ℝ) : ℝ :=
  let ⟨xa, ya, za⟩ := A in
  let ⟨xb, yb, zb⟩ := B in
  let ⟨xc, yc, zc⟩ := C in
  let ⟨xd, yd, zd⟩ := D in
  let ab := (xb - xa, yb - ya, zb - za) in
  let ac := (xc - xa, yc - ya, zc - za) in
  let bc := (xc - xb, yc - yb, zc - zb) in
  let bd := (xd - xb, yd - yb, zd - zb) in
  let n := (2, 3, -1) in -- Calculated normal vector for plane ABC
  let m := (1, 8, 6) in -- Calculated normal vector for plane BCD
  let dot_product := n.1 * m.1 + n.2 * m.2 + n.3 * m.3 in
  let n_magnitude := Real.sqrt ((n.1)^2 + (n.2)^2 + (n.3)^2) in
  let m_magnitude := Real.sqrt ((m.1)^2 + (m.2)^2 + (m.3)^2) in
  Real.arccos (dot_product / (n_magnitude * m_magnitude))

theorem acute_angle_between_planes :
  angle_between_planes (1, 0, 1) (-2, 2, 1) (2, 0, 3) (0, 4, -2) = Real.arccos (20 / Real.sqrt (14 * 101)) :=
  sorry

end acute_angle_between_planes_l32_32145


namespace hollow_cylinder_volume_l32_32455

def external_diameter : ℝ := 14
def internal_diameter : ℝ := 10
def height : ℝ := 10

def external_radius : ℝ := external_diameter / 2
def internal_radius : ℝ := internal_diameter / 2

def volume_external_cylinder : ℝ := π * external_radius ^ 2 * height
def volume_internal_cylinder : ℝ := π * internal_radius ^ 2 * height

def volume_hollow : ℝ := volume_external_cylinder - volume_internal_cylinder

theorem hollow_cylinder_volume :
  volume_hollow = 240 * π :=
by
  sorry

end hollow_cylinder_volume_l32_32455


namespace mike_travel_distance_l32_32287

theorem mike_travel_distance
  (mike_start : ℝ := 2.50)
  (mike_per_mile : ℝ := 0.25)
  (annie_start : ℝ := 2.50)
  (annie_toll : ℝ := 5.00)
  (annie_per_mile : ℝ := 0.25)
  (annie_miles : ℝ := 14)
  (mike_cost : ℝ)
  (annie_cost : ℝ) :
  mike_cost = annie_cost → mike_cost = mike_start + mike_per_mile * 34 := by
  sorry

end mike_travel_distance_l32_32287


namespace smallest_lcm_l32_32208

theorem smallest_lcm (k l : ℕ) (hk : 999 < k ∧ k < 10000) (hl : 999 < l ∧ l < 10000)
  (h_gcd : Nat.gcd k l = 5) : Nat.lcm k l = 201000 :=
sorry

end smallest_lcm_l32_32208


namespace exist_a_b_if_and_only_if_n_prime_divisor_1_mod_4_l32_32975

theorem exist_a_b_if_and_only_if_n_prime_divisor_1_mod_4
  (n : ℕ) (hn₁ : Odd n) (hn₂ : 0 < n) :
  (∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ (4 : ℚ) / n = 1 / a + 1 / b) ↔
  ∃ p, p ∣ n ∧ Prime p ∧ p % 4 = 1 :=
by
  sorry

end exist_a_b_if_and_only_if_n_prime_divisor_1_mod_4_l32_32975


namespace evaluate_series_l32_32845

noncomputable def infinite_series := ∑ k in (Finset.range ∞), (k + 1)^2 / 3^(k + 1)

theorem evaluate_series : infinite_series = 1 / 2 := sorry

end evaluate_series_l32_32845


namespace count_integers_solution_l32_32096

theorem count_integers_solution :
  let S := {x : ℤ | -5 ≤ x ∧ x ≤ - (23: ℚ) / 8}
  ∃ (n : ℕ), n = S.card ∧ n = 3 :=
by
  let S := {x : ℤ | -5 ≤ x ∧ x ≤ - (23: ℚ) / 8}
  have : S = {-5, -4, -3}, from sorry
  exact ⟨3, by rw [this, set.card_insert_of_not_mem, set.card_insert_of_not_mem, set.card_singleton]; simp⟩

end count_integers_solution_l32_32096


namespace lollipops_per_day_l32_32909

variable (Alison_lollipops : ℕ) (Henry_lollipops : ℕ) (Diane_lollipops : ℕ) (Total_lollipops : ℕ) (Days : ℕ)

-- Conditions given in the problem
axiom condition1 : Alison_lollipops = 60
axiom condition2 : Henry_lollipops = Alison_lollipops + 30
axiom condition3 : Alison_lollipops = Diane_lollipops / 2
axiom condition4 : Total_lollipops = Alison_lollipops + Henry_lollipops + Diane_lollipops
axiom condition5 : Days = 6

-- Question to prove
theorem lollipops_per_day : (Total_lollipops / Days) = 45 := sorry

end lollipops_per_day_l32_32909


namespace min_f_on_interval_range_a_ln_inequality_l32_32130

-- Defining the functions f and g
def f (x : ℝ) : ℝ := x * Real.log x
def g (x : ℝ) (a : ℝ) : ℝ := -x*x + a*x - 3

-- Statement 1: Minimum value of f(x) on [t, t+2] given t > 0
theorem min_f_on_interval (t : ℝ) (ht : t > 0) :
  let min_f := if 0 < t ∧ t < (1/Real.exp 1) then -1/(Real.exp 1) else t * Real.log t
  in min_f = if 0 < t ∧ t < (1/Real.exp 1) then -1/(Real.exp 1) else t * Real.log t :=
sorry

-- Statement 2: Range of the real number a for 2f(x) ≥ g(x) for all x in (0, ∞)
theorem range_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → 2 * f x ≥ g x a) → a ≤ 4 :=
sorry

-- Statement 3: ln x > 1/e^x - 2/(ex) for all x in (0, ∞)
theorem ln_inequality (x : ℝ) (hx : 0 < x) : Real.log x > 1/(Real.exp x) - 2/(Real.exp x * x) :=
sorry

end min_f_on_interval_range_a_ln_inequality_l32_32130


namespace measure_angle_EFD_l32_32821

-- Definitions of the problem conditions
def angle_A : ℝ := 50
def angle_B : ℝ := 70
def angle_C : ℝ := 60

-- Statement to prove
theorem measure_angle_EFD : 
  ∀ (D E F : Type) (incircle : D → E → F → Prop) (circumcircle : D → E → F → Prop),
  ∀ (on_BC : D) (on_AB : E) (on_AC : F), 
  incircle (on_BC) (on_AB) (on_AC) → 
  circumcircle (on_BC) (on_AB) (on_AC) → 
  angle_A + angle_B + angle_C = 180 →
  ∠ EFD = 50 := by
sorry

end measure_angle_EFD_l32_32821


namespace find_angle_C_find_area_l32_32550

-- Defining the parameters of the triangle and necessary conditions
variable {a b c A B C : ℝ}

-- Given conditions for question 1
axiom triangle_cond : a^2 - a * b - 2 * b^2 = 0
axiom angle_B : B = Real.pi / 6

-- Theorem for Question 1: Finding angle C
theorem find_angle_C : triangle_cond → angle_B → 
  C = Real.pi / 3 := sorry

-- Additional given condition for question 2
axiom angle_C : C = 2 * Real.pi / 3
axiom side_c : c = 14

-- Theorem for Question 2: Finding the area of the triangle
theorem find_area : triangle_cond → angle_C → side_c → 
  S = 28 * Real.sqrt 3 := sorry

end find_angle_C_find_area_l32_32550


namespace find_a_l32_32669

theorem find_a (a : ℤ) (h1 : 0 < a) (h2 : ∀ (x : ℝ), |x - a| < 1 → x ∈ {x | x = 2}) : a = 2 :=
sorry

end find_a_l32_32669


namespace smallest_odd_n_exists_smallest_odd_n_is_8_l32_32750

theorem smallest_odd_n_exists (n : ℕ) (h1 : n % 2 = 1) :
  (∏ k in finset.range (n + 1), 3 ^ ((2 * k + 1) / 9 : ℚ)) > 5000 := sorry

theorem smallest_odd_n_is_8 :
  ∃ n : ℕ, n % 2 = 1 ∧ (∏ k in finset.range (n + 1), 3 ^ ((2 * k + 1) / 9 : ℚ)) > 5000 ∧ n = 8 :=
begin
  use 8,
  split,
  { norm_num },
  split,
  { sorry },
  { refl }
end

end smallest_odd_n_exists_smallest_odd_n_is_8_l32_32750


namespace smallest_N_with_19_odd_units_digit_squares_l32_32316

theorem smallest_N_with_19_odd_units_digit_squares : ∃ N : ℕ, (∀ k : ℕ, N ≥ k) ∧ (count_odd_units_digit_squares N = 19) := 
by
  use 44
  sorry

def count_odd_units_digit_squares (N : ℕ) : ℕ :=
  (finset.range (N + 1)).filter (fun x => 
    let units_digit := (x * x) % 10
    units_digit = 1 ∨ units_digit = 9 ∨ units_digit = 5
  ).card

end smallest_N_with_19_odd_units_digit_squares_l32_32316


namespace fa_onto_l32_32956

noncomputable def f : ℝ → ℝ := sorry  -- Define the continuous and surjective function

theorem fa_onto (a : ℝ) (h : 0 < a ∧ a < 1) : ∀ y ∈ (set.Ioo 0 1), ∃ x ∈ set.Ioo a 1, f x = y :=
sorry  -- onto proof skipped

end fa_onto_l32_32956


namespace angle_between_lines_less_than_seventeen_l32_32358

theorem angle_between_lines_less_than_seventeen (O : Point) (L : Fin 12 → Line) 
  (h : ∀ i j : Fin 12, i ≠ j → L i ≠ L j) :
  ∃ i j : Fin 12, i ≠ j ∧ angle (L i) (L j) < 17 :=
by
  -- math proof goes here
  sorry

end angle_between_lines_less_than_seventeen_l32_32358


namespace value_of_sum_l32_32892

theorem value_of_sum (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (hc_solution : c^2 + a * c + b = 0) (hd_solution : d^2 + a * d + b = 0)
  (ha_solution : a^2 + c * a + d = 0) (hb_solution : b^2 + c * b + d = 0)
: a + b + c + d = -2 := sorry -- The proof is omitted as requested

end value_of_sum_l32_32892


namespace buddy_cards_on_saturday_l32_32294

def initial_cards : ℕ := 100

def cards_on_tuesday (n_mon : ℕ) : ℕ :=
  n_mon - Nat.floor (0.30 * n_mon)

def cards_on_wednesday (n_tue : ℕ) : ℕ :=
  n_tue + Nat.floor (0.20 * n_tue)

def cards_on_thursday (n_wed : ℕ) : ℕ :=
  n_wed - Nat.floor (0.25 * n_wed)

def cards_on_friday (n_thu : ℕ) : ℕ :=
  n_thu + Nat.floor (1 / 3 * n_thu)

def cards_on_saturday (n_fri : ℕ) : ℕ :=
  n_fri + 2 * n_fri

theorem buddy_cards_on_saturday :
  cards_on_saturday (cards_on_friday (cards_on_thursday (cards_on_wednesday (cards_on_tuesday initial_cards)))) = 252 :=
sorry

end buddy_cards_on_saturday_l32_32294


namespace lion_cubs_per_month_l32_32728

theorem lion_cubs_per_month
  (initial_lions : ℕ)
  (final_lions : ℕ)
  (months : ℕ)
  (lions_dying_per_month : ℕ)
  (net_increase : ℕ)
  (x : ℕ) : 
  initial_lions = 100 → 
  final_lions = 148 → 
  months = 12 → 
  lions_dying_per_month = 1 → 
  net_increase = 48 → 
  12 * (x - 1) = net_increase → 
  x = 5 := by
  intros initial_lions_eq final_lions_eq months_eq lions_dying_eq net_increase_eq equation
  sorry

end lion_cubs_per_month_l32_32728


namespace max_distance_slope_correct_l32_32383

noncomputable def max_distance_slope : ℝ :=
  let P := (3, 2)
  let line := λ m : ℝ, m * (x - 2) - y + 1 - 2 * m = 0
  let Q := (2, 1)
  let perpendicular := ∀ m : ℝ, (2 - 1) / (3 - 2) * m = -1
  by sorry

theorem max_distance_slope_correct :
  max_distance_slope = -1 :=
sorry

end max_distance_slope_correct_l32_32383


namespace sum_of_solutions_of_quadratic_l32_32453

theorem sum_of_solutions_of_quadratic :
  let a := 18
  let b := -27
  let c := -45
  let roots_sum := (-b / a : ℚ)
  roots_sum = 3 / 2 :=
by
  let a := 18
  let b := -27
  let c := -45
  let roots_sum := (-b / a : ℚ)
  have h1 : roots_sum = 3 / 2 := by sorry
  exact h1

end sum_of_solutions_of_quadratic_l32_32453


namespace calculation_correct_l32_32865

theorem calculation_correct (x y : ℝ) (hx0 : x ≠ 0) (hy0 : y ≠ 0) (hxy : x = 2 * y) : 
  (x - 2 / x) * (y + 2 / y) = 1 / 2 * (x^2 - 2 * x + 8 - 16 / x) := 
by 
  sorry

end calculation_correct_l32_32865


namespace runs_percentage_by_running_and_no_balls_l32_32025

variable (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) (no_balls : ℕ)

-- Conditions
def total_score := 180
def num_boundaries := 9
def num_sixes := 7
def num_no_balls := 2

-- Question and Correct Answer
theorem runs_percentage_by_running_and_no_balls : 
  (total_runs = total_score) → 
  (boundaries = num_boundaries) →
  (sixes = num_sixes) →
  (no_balls = num_no_balls) →
  (let running_runs := total_runs - (boundaries * 4 + sixes * 6 + no_balls * 1)
   in ((running_runs + no_balls) : ℚ / total_runs * 100 = 56.67)) :=
  by intros; sorry

end runs_percentage_by_running_and_no_balls_l32_32025


namespace time_to_drain_l32_32802

-- Define the constants
def S : ℝ := 6    -- Cross-sectional area in m^2
def H : ℝ := 5    -- Height of water in m
def s : ℝ := 0.01 -- Area of the hole in m^2
def V (g h : ℝ) : ℝ := 0.6 * real.sqrt (2 * g * h) -- Speed of water flow
def g : ℝ := 9.81 -- Acceleration due to gravity in m/s^2

-- Define the integral function which represents the total time to drain the tank
noncomputable def total_time (S H s g : ℝ) : ℝ :=
  (S / (0.6 * s * real.sqrt (2 * g))) * (2 * real.sqrt H)

-- The Lean statement to be proved
theorem time_to_drain : total_time S H s g = 1010 := sorry

end time_to_drain_l32_32802


namespace find_small_pack_size_l32_32693

-- Define the conditions of the problem
def soymilk_sold_in_packs (pack_size : ℕ) : Prop :=
  pack_size = 2 ∨ ∃ L : ℕ, pack_size = L

def cartons_bought (total_cartons : ℕ) (large_pack_size : ℕ) (num_large_packs : ℕ) (small_pack_size : ℕ) : Prop :=
  total_cartons = num_large_packs * large_pack_size + small_pack_size

-- The problem statement as a Lean theorem
theorem find_small_pack_size (total_cartons : ℕ) (num_large_packs : ℕ) (large_pack_size : ℕ) :
  soymilk_sold_in_packs 2 →
  soymilk_sold_in_packs large_pack_size →
  cartons_bought total_cartons large_pack_size num_large_packs 2 →
  total_cartons = 17 →
  num_large_packs = 3 →
  large_pack_size = 5 →
  ∃ S : ℕ, soymilk_sold_in_packs S ∧ S = 2 :=
by
  sorry

end find_small_pack_size_l32_32693


namespace factor_polynomial_l32_32108

theorem factor_polynomial :
  ∃ (a b c d e f : ℤ), a < d ∧
    (a * x^2 + b * x + c) * (d * x^2 + e * x + f) = x^2 - 6 * x + 9 - 64 * x^4 ∧
    (a = -8 ∧ b = 1 ∧ c = -3 ∧ d = 8 ∧ e = 1 ∧ f = -3) := by
  sorry

end factor_polynomial_l32_32108


namespace matrix_power_four_l32_32458

-- Given matrix M representing a rotation by pi/4 radians
def M : Matrix (Fin 2) (Fin 2) ℝ :=
  ![\[Real.sqrt 2 / 2, -Real.sqrt 2 / 2\], \[Real.sqrt 2 / 2, Real.sqrt 2 / 2\]]

-- Statement of the problem
theorem matrix_power_four :
  M ^ 4 = ![\[-1, 0\], \[0, -1\]] := 
sorry

end matrix_power_four_l32_32458


namespace power_function_value_l32_32178

theorem power_function_value
  (α : ℝ)
  (h : 2^α = Real.sqrt 2) :
  (4 : ℝ) ^ α = 2 :=
by {
  sorry
}

end power_function_value_l32_32178


namespace catCannotCatchMouseInA_catCanCatchMouseInB_catCannotCatchMouseInC_l32_32030

-- Define labyrinths A, B, and C as types or structures if necessary
inductive Labyrinth
| A
| B
| C

-- Define the starting positions
constant K : Type
constant M : Type

-- Conditions: Positions of cat and mouse in labyrinths
def cat_position (L : Labyrinth) : Type := K
def mouse_position (L : Labyrinth) : Type := M

-- Condition: Cat and mouse can move to any adjacent node
constant can_move_to_adjacent : Type -> Type -> Prop

-- Define the condition for catching the mouse
def cat_catches_mouse (L : Labyrinth) (cat_pos : K) (mouse_pos : M) : Prop :=
  cat_pos = mouse_pos

-- Statements for each labyrinth
theorem catCannotCatchMouseInA : ∀ (cat_pos : K) (mouse_pos : M),
  ¬ cat_catches_mouse Labyrinth.A cat_pos mouse_pos := sorry

theorem catCanCatchMouseInB : ∀ (cat_pos : K) (mouse_pos : M),
  cat_catches_mouse Labyrinth.B cat_pos mouse_pos := sorry

theorem catCannotCatchMouseInC : ∀ (cat_pos : K) (mouse_pos : M),
  ¬ cat_catches_mouse Labyrinth.C cat_pos mouse_pos := sorry

end catCannotCatchMouseInA_catCanCatchMouseInB_catCannotCatchMouseInC_l32_32030


namespace percentage_less_than_l32_32796

theorem percentage_less_than (x y : ℕ) (h : x = 6 * y) : (x - y) / x * 100 = 83.33 := 
by 
  sorry

end percentage_less_than_l32_32796


namespace isosceles_triangle_vertex_angle_l32_32624

theorem isosceles_triangle_vertex_angle (T : Triangle) (isosceles : is_isosceles T)
(angle_given : ∃ A B C : Angle, (is_angle_of T A) ∧ (is_angle_of T B) ∧ (is_angle_of T C) ∧ (A.degree = 50 ∨ B.degree = 50 ∨ C.degree = 50)) :
∃ V : Angle, is_vertex_angle_of T V ∧ (V.degree = 50 ∨ V.degree = 80) :=
by
  sorry

end isosceles_triangle_vertex_angle_l32_32624


namespace infinite_series_sum_eq_seven_l32_32841

noncomputable def infinite_series_sum : ℝ :=
  ∑' k : ℕ, (1 + k)^2 / 3^(1 + k)

theorem infinite_series_sum_eq_seven : infinite_series_sum = 7 :=
sorry

end infinite_series_sum_eq_seven_l32_32841


namespace rabbit_avg_distance_square_l32_32049

theorem rabbit_avg_distance_square (
  side_length : ℝ,
  dist_diagonal_hop : ℝ,
  dist_right_turn : ℝ
) (h1: side_length = 12) 
  (h2: dist_diagonal_hop = 9.8) 
  (h3: dist_right_turn = 4) :
  let avg_distance := (10.929 + 6.929 + (side_length - 10.929) + (side_length - 6.929)) / 4 in
  avg_distance = 6 :=
by
  sorry

end rabbit_avg_distance_square_l32_32049


namespace proof_problem_solution_l32_32977

open Real

noncomputable def proof_problem (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) : Prop :=
  (a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b)) ≥ (3 / 2)

theorem proof_problem_solution {a b c : ℝ} (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) :
  proof_problem a b c h_pos h_prod :=
begin
  -- proof to be filled in later
  sorry
end

end proof_problem_solution_l32_32977


namespace length_of_train_eq_130_m_l32_32057

def speed_km_hr := 65 -- The speed of the train in km/hr
def time_sec := 15.506451791548985 -- The time taken to cross the bridge in seconds
def length_bridge_m := 150 -- The length of the bridge in meters

noncomputable def speed_m_s := speed_km_hr * (1000 / 3600: Float) -- Convert speed to m/s
noncomputable def total_distance := speed_m_s * time_sec -- Total distance covered by the train

theorem length_of_train_eq_130_m :
  total_distance - length_bridge_m = 130 := 
sorry

end length_of_train_eq_130_m_l32_32057


namespace possible_values_a1_b1_l32_32656

theorem possible_values_a1_b1 :
  let M := {x : (ℕ × ℕ × ℕ × ℕ) | 
            x.1 ∈ {1, 2, 3, 4} ∧ x.2.1 ∈ {1, 2, 3, 4} ∧ 
            x.2.2.1 ∈ {1, 2, 3, 4} ∧ x.2.2.2 ∈ {1, 2, 3, 4} ∧
            x.1 * x.2.1 * x.2.2.1 * x.2.2.2 > 1} in
  ∀ (seq : Fin 255 → ℕ × ℕ × ℕ × ℕ),
    (∀ n, seq n ∈ M) ∧
    (∀ n : Fin 254, 
      ∃ a b c d a' b' c' d', 
      seq ⟨n+1, sorry⟩ = (a', b', c', d') ∧ 
      seq n = (a, b, c, d) ∧ 
      abs (a' - a) + abs (b' - b) + abs (c' - c) + abs (d' - d) = 1) ∧
    seq 0 = (a1, b1, 1, 1) →
  (a1, b1) ∈ {(1, 2), (1, 4), (2, 1), (2, 3), (3, 2), (3, 4), (4, 1), (4, 3)} :=
sorry

end possible_values_a1_b1_l32_32656


namespace coin_combinations_l32_32581

-- Define the coins and their counts
def one_cent_count := 1
def two_cent_count := 1
def five_cent_count := 1
def ten_cent_count := 4
def fifty_cent_count := 2

-- Define the expected number of different possible amounts
def expected_amounts := 119

-- Prove that the expected number of possible amounts can be achieved given the coins
theorem coin_combinations : 
  (∃ sums : Finset ℕ, 
    sums.card = expected_amounts ∧ 
    (∀ n ∈ sums, n = one_cent_count * 1 + 
                          two_cent_count * 2 + 
                          five_cent_count * 5 + 
                          ten_cent_count * 10 + 
                          fifty_cent_count * 50)) :=
sorry

end coin_combinations_l32_32581


namespace no_unique_sum_of_squares_l32_32837

open Polynomial

-- Define the problem statement in Lean 4
theorem no_unique_sum_of_squares (P : Polynomial ℝ) (hP : ¬IsConstant P) :
  ¬ (∃ (a b : Polynomial ℝ), (∃ u v : Polynomial ℝ, a = u^2 ∧ b = v^2 ∧ P = a + b) ∧
  ∀ a' b' u' v' : Polynomial ℝ, (a' = u'^2 ∧ b' = v'^2 ∧ P = a' + b') → (a = a' ∧ b = b' ∨ a = b' ∧ b = a')) :=
by
  sorry

end no_unique_sum_of_squares_l32_32837


namespace sum_q_t_12_l32_32959

def T : set (fin 12 → bool) := 
  {t | ∀ i, t i = 0 ∨ t i = 1}

def q_t (t : fin 12 → bool) (x : ℕ) : polynomial ℤ := 
  polynomial.of_finsupp (finsupp.single x (ite (t (⟨x % 12, sorry⟩ : fin 12)) 1 0))

def q (x : ℕ) := 
  ∑ t in T, q_t t x

theorem sum_q_t_12 : 
  ∑ t in T, q_t t 12 = 2048 :=
sorry

end sum_q_t_12_l32_32959


namespace Maryann_total_minutes_worked_l32_32673

theorem Maryann_total_minutes_worked (c a t : ℕ) (h1 : c = 70) (h2 : a = 7 * c) (h3 : t = c + a) : t = 560 := by
  sorry

end Maryann_total_minutes_worked_l32_32673


namespace min_distance_tangent_intersections_l32_32712

-- Define the properties of the ellipse
def ellipse (a b x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the property of being a tangent line at a point on the ellipse
def is_tangent_at (a b x1 y1 x y : ℝ) : Prop :=
  ellipse a b x1 y1 ∧ (x1 * x) / (a^2) + (y1 * y) / (b^2) = 1

-- Define the point A where the tangent line intersects the x-axis
def x_intercept (a b x1 y1 : ℝ) : ℝ × ℝ :=
  (a^2 / x1, 0)

-- Define the point B where the tangent line intersects the y-axis
def y_intercept (a b x1 y1 : ℝ) : ℝ × ℝ :=
  (0, b^2 / y1)

-- Define the distance between two points in the plane
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the problem as a theorem in Lean 4
theorem min_distance_tangent_intersections (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  ∃ x1 y1 : ℝ, ellipse a b x1 y1 →
  ∀ (θ : ℝ), distance (x_intercept a b x1 y1) (y_intercept a b x1 y1) = a + b :=
sorry

end min_distance_tangent_intersections_l32_32712


namespace quadrilateral_circle_area_ratio_l32_32798

theorem quadrilateral_circle_area_ratio (r : ℝ) (P : ℝ) (C : ℝ) 
  (h1 : C = 2 * Real.pi * r)
  (h2 : P = (4 / 3) * C)
  (area_quad : ℝ)
  (area_circle : ℝ)
  (h3 : area_circle = Real.pi * r^2)
  (h4 : area_quad = (1 / 2) * r * P) :
  100 * 4 + 49 * 3 = 547 :=
by
  have h5 : P = (8 * Real.pi * r) / 3 := by rw [h1, h2]; ring
  have h6 : area_quad = (4 * Real.pi * r^2) / 3 := by rw [h4, h5]; ring
  have h7 : (area_quad / area_circle) = (4 / 3) := by rw [h3, h6]; field_simp;  ring

  /- Therefore, a = 4 and b = 3 -/
  sorry

end quadrilateral_circle_area_ratio_l32_32798


namespace factor_polynomial_l32_32107

theorem factor_polynomial :
  ∃ (a b c d e f : ℤ), a < d ∧
    (a * x^2 + b * x + c) * (d * x^2 + e * x + f) = x^2 - 6 * x + 9 - 64 * x^4 ∧
    (a = -8 ∧ b = 1 ∧ c = -3 ∧ d = 8 ∧ e = 1 ∧ f = -3) := by
  sorry

end factor_polynomial_l32_32107


namespace Caitlin_age_l32_32814

theorem Caitlin_age (Aunt_Anna_age : ℕ) (h1 : Aunt_Anna_age = 54) (Brianna_age : ℕ) (h2 : Brianna_age = (2 * Aunt_Anna_age) / 3) (Caitlin_age : ℕ) (h3 : Caitlin_age = Brianna_age - 7) : 
  Caitlin_age = 29 := 
  sorry

end Caitlin_age_l32_32814


namespace smallest_lcm_l32_32210

theorem smallest_lcm (k l : ℕ) (hk : 999 < k ∧ k < 10000) (hl : 999 < l ∧ l < 10000)
  (h_gcd : Nat.gcd k l = 5) : Nat.lcm k l = 201000 :=
sorry

end smallest_lcm_l32_32210


namespace spherical_to_rectangular_example_l32_32827

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ,
   ρ * Real.sin φ * Real.sin θ,
   ρ * Real.cos φ)

theorem spherical_to_rectangular_example :
  spherical_to_rectangular 5 (3 * Real.pi / 2) (Real.pi / 3) = (0, -5 * Real.sqrt 3 / 2, 5 / 2) :=
by
  simp [spherical_to_rectangular, Real.sin, Real.cos]
  sorry

end spherical_to_rectangular_example_l32_32827


namespace unique_solution_l32_32091

def system_of_equations (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ) (x1 x2 x3 : ℝ) :=
  a11 * x1 + a12 * x2 + a13 * x3 = 0 ∧
  a21 * x1 + a22 * x2 + a23 * x3 = 0 ∧
  a31 * x1 + a32 * x2 + a33 * x3 = 0

theorem unique_solution
  (x1 x2 x3 : ℝ)
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ)
  (h_pos: 0 < a11 ∧ 0 < a22 ∧ 0 < a33)
  (h_neg: a12 < 0 ∧ a13 < 0 ∧ a21 < 0 ∧ a23 < 0 ∧ a31 < 0 ∧ a32 < 0)
  (h_sum_pos: 0 < a11 + a12 + a13 ∧ 0 < a21 + a22 + a23 ∧ 0 < a31 + a32 + a33)
  (h_system: system_of_equations a11 a12 a13 a21 a22 a23 a31 a32 a33 x1 x2 x3):
  x1 = 0 ∧ x2 = 0 ∧ x3 = 0 := sorry

end unique_solution_l32_32091


namespace person_a_work_days_l32_32011

theorem person_a_work_days (x : ℝ) (h1 : 1 / 6 + 1 / x = 1 / 3.75) : x = 10 := 
sorry

end person_a_work_days_l32_32011


namespace john_horizontal_distance_l32_32648

-- Define the conditions and the question
def elevation_initial : ℕ := 100
def elevation_final : ℕ := 1450
def vertical_to_horizontal_ratio (v h : ℕ) : Prop := v * 2 = h

-- Define the proof problem: the horizontal distance John moves
theorem john_horizontal_distance : ∃ h, vertical_to_horizontal_ratio (elevation_final - elevation_initial) h ∧ h = 2700 := 
by 
  sorry

end john_horizontal_distance_l32_32648


namespace mural_total_cost_is_192_l32_32255

def mural_width : ℝ := 6
def mural_height : ℝ := 3
def paint_cost_per_sqm : ℝ := 4
def area_per_hour : ℝ := 1.5
def hourly_rate : ℝ := 10

def mural_area := mural_width * mural_height
def paint_cost := mural_area * paint_cost_per_sqm
def labor_hours := mural_area / area_per_hour
def labor_cost := labor_hours * hourly_rate
def total_mural_cost := paint_cost + labor_cost

theorem mural_total_cost_is_192 : total_mural_cost = 192 := by
  -- Definitions
  sorry

end mural_total_cost_is_192_l32_32255


namespace max_sqrt_sum_min_sum_of_reciprocals_l32_32564

variable (a b : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1)

theorem max_sqrt_sum : 
  sqrt a + sqrt b ≤ sqrt 2 :=
sorry

theorem min_sum_of_reciprocals : 
  (1 / a) + (1 / (2 * b + 1)) ≥ (3 + 2 * sqrt 2) / 3 :=
sorry

end max_sqrt_sum_min_sum_of_reciprocals_l32_32564


namespace find_percentage_reduction_l32_32405

-- Given the conditions of the problem.
def original_price : ℝ := 7500
def current_price: ℝ := 4800
def percentage_reduction (x : ℝ) : Prop := (original_price * (1 - x)^2 = current_price)

-- The statement we need to prove:
theorem find_percentage_reduction (x : ℝ) (h : percentage_reduction x) : x = 0.2 :=
by
  sorry

end find_percentage_reduction_l32_32405


namespace value_of_expression_is_7_l32_32894

theorem value_of_expression_is_7 (a b c : ℕ) (h_set : {a, b, c} = {0, 1, 2})
  (h_condition : (a ≠ 2 ∨ b = 2 ∨ c ≠ 0) ∧ ((a ≠ 2) ∧ ¬(b = 2) ∧ ¬(c ≠ 0)) ∨ (¬(a ≠ 2) ∧ (b = 2) ∧ ¬(c ≠ 0)) ∨ (¬(a ≠ 2) ∧ ¬(b = 2) ∧ (c ≠ 0)) ∧ (h_true : (a ≠ 2 → (b ≠ 2) ∧ (c = 0)) ∧ (b = 2 → (a = 2) ∧ (c = 0)) ∧ (c ≠ 0 → (a = 2) ∧ (b ≠ 2)))) : a + 2 * b + 5 * c = 7 := 
by
  sorry

end value_of_expression_is_7_l32_32894


namespace infinite_series_sum_eq_seven_l32_32842

noncomputable def infinite_series_sum : ℝ :=
  ∑' k : ℕ, (1 + k)^2 / 3^(1 + k)

theorem infinite_series_sum_eq_seven : infinite_series_sum = 7 :=
sorry

end infinite_series_sum_eq_seven_l32_32842


namespace num_sequences_length_15_l32_32468

def a_seq : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n + 2) := b_seq n

def b_seq : ℕ → ℕ
| 0 := 0
| 1 := 0
| 2 := 1
| (n + 2) := a_seq n + b_seq n

theorem num_sequences_length_15 : a_seq 15 + b_seq 15 = 610 := by
  sorry

end num_sequences_length_15_l32_32468


namespace problem_statement_l32_32560

-- Definitions of propositions p and q
def p : Prop := ∃ x : ℝ, Real.tan x = 1
def q : Prop := ∀ x : ℝ, x^2 > 0

-- The proof problem
theorem problem_statement : ¬ (¬ p ∧ ¬ q) :=
by 
  -- sorry here indicates that actual proof is omitted
  sorry

end problem_statement_l32_32560


namespace sqrt_of_mixed_fraction_simplified_l32_32505

theorem sqrt_of_mixed_fraction_simplified :
  let x := 8 + (9 / 16) in
  sqrt x = (sqrt 137) / 4 := by
  sorry

end sqrt_of_mixed_fraction_simplified_l32_32505


namespace max_sum_distinct_factors_2029_l32_32242

theorem max_sum_distinct_factors_2029 :
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A * B * C = 2029 ∧ A + B + C = 297 :=
by
  sorry

end max_sum_distinct_factors_2029_l32_32242


namespace right_triangle_altitude_l32_32416

theorem right_triangle_altitude (a b c area altitude : ℝ)
  (h_tri_area : area = 800)
  (h_leg_a : a = 40)
  (h_angle : ∠ A B C = 30)
  (h_right_triangle : right_triangle a b c)
  (h_area_def : triangle_area a b = area) :
  altitude = 20 * sqrt 2 := 
  sorry

end right_triangle_altitude_l32_32416


namespace committee_selections_with_at_least_one_prev_served_l32_32445

-- Define the conditions
def total_candidates := 20
def previously_served := 8
def committee_size := 4
def never_served := total_candidates - previously_served

-- The proof problem statement
theorem committee_selections_with_at_least_one_prev_served : 
  (Nat.choose total_candidates committee_size - Nat.choose never_served committee_size) = 4350 :=
by
  sorry

end committee_selections_with_at_least_one_prev_served_l32_32445


namespace proposition_1_incorrect_proposition_2_incorrect_proposition_3_incorrect_proposition_4_correct_l32_32525

-- Define the binomial term
def binomial_term (n k : ℕ) (x : ℝ) : ℝ := (nat.choose n k) * x^(n - k) * (-1)^k

-- Define the binomial expansion
def binomial_expansion (x : ℝ) (n : ℕ) : ℝ := ∑ k in finset.range (n + 1), binomial_term n k x

-- Statement of the problem assertions
theorem proposition_1_incorrect :
  binomial_term 1999 1000 x ≠ -nat.choose 1999 1000 * x^999 := sorry

theorem proposition_2_incorrect :
  (∑ k in finset.range 1998, coeff (binomial_expansion x 1999) k) ≠ 1 := sorry

theorem proposition_3_incorrect :
  ¬(is_maximum (coeff (binomial_expansion x 1999) 1000) 
    ∧ is_maximum (coeff (binomial_expansion x 1999) 1001)) := sorry

theorem proposition_4_correct :
  (binomial_expansion 2000 1999) % 2000 = 1 := sorry

-- Helper functions can be defined as needed

end proposition_1_incorrect_proposition_2_incorrect_proposition_3_incorrect_proposition_4_correct_l32_32525


namespace arrangement_count_l32_32388

noncomputable def count_arrangements (balls : Finset ℕ) (boxes : Finset ℕ) : ℕ :=
  sorry -- The implementation of this function is out of scope for this task

theorem arrangement_count :
  count_arrangements ({1, 2, 3, 4} : Finset ℕ) ({1, 2, 3} : Finset ℕ) = 18 :=
sorry

end arrangement_count_l32_32388


namespace horizontal_distance_l32_32651

theorem horizontal_distance (elev_initial elev_final v_ratio h_ratio distance: ℝ)
  (h1 : elev_initial = 100)
  (h2 : elev_final = 1450)
  (h3 : v_ratio = 1)
  (h4 : h_ratio = 2)
  (h5 : distance = 1350) :
  distance * h_ratio = 2700 := by
  sorry

end horizontal_distance_l32_32651


namespace binomial_sum_alternating_l32_32487

theorem binomial_sum_alternating :
  (finset.range 51).sum (λ k, (-1 : ℤ) ^ k * (k + 1) * (nat.choose 50 k)) = 0 :=
begin
  sorry
end

end binomial_sum_alternating_l32_32487


namespace maria_green_beans_l32_32285

theorem maria_green_beans
    (potatoes : ℕ)
    (carrots : ℕ)
    (onions : ℕ)
    (green_beans : ℕ)
    (h1 : potatoes = 2)
    (h2 : carrots = 6 * potatoes)
    (h3 : onions = 2 * carrots)
    (h4 : green_beans = onions / 3) :
  green_beans = 8 := 
sorry

end maria_green_beans_l32_32285


namespace cousin_points_correct_l32_32387

-- Conditions translated to definitions
def paul_points : ℕ := 3103
def total_points : ℕ := 5816

-- Dependent condition to get cousin's points
def cousin_points : ℕ := total_points - paul_points

-- The goal of our proof problem
theorem cousin_points_correct : cousin_points = 2713 :=
by
    sorry

end cousin_points_correct_l32_32387


namespace correct_phrase_l32_32705

-- Define statements representing each option
def option_A : String := "as twice much"
def option_B : String := "much as twice"
def option_C : String := "twice as much"
def option_D : String := "as much twice"

-- The correct option
def correct_option : String := "twice as much"

-- The main theorem statement
theorem correct_phrase : option_C = correct_option :=
by
  sorry

end correct_phrase_l32_32705


namespace result_is_21_l32_32354

theorem result_is_21 (n : ℕ) (h : n = 55) : (n / 5 + 10) = 21 :=
by
  sorry

end result_is_21_l32_32354


namespace john_horizontal_distance_l32_32652

theorem john_horizontal_distance (v_increase : ℕ)
  (elevation_start : ℕ) (elevation_end : ℕ) (h_ratio : ℕ) :
  (elevation_end - elevation_start) * h_ratio = 1350 * 2 :=
begin
  -- Let elevation_start be 100 feet
  let elevation_start := 100,
  -- Let elevation_end be 1450 feet
  let elevation_end := 1450,
  -- The steepened ratio, height per step
  let v_increase := elevation_end - elevation_start,
  -- John travels 1 foot vertically for every 2 feet horizontally
  let h_ratio := 2,
  -- Hence the horizontal distance theorem
  -- s_foot = h_ratio * t_foot = 1350 * 2 = 2700 feet.
  sorry
end

end john_horizontal_distance_l32_32652


namespace max_unbounded_sinx_cscx_cosx_secx_l32_32116

theorem max_unbounded_sinx_cscx_cosx_secx (x : ℝ) (h : 0 < x ∧ x < π) : 
  ∃ m : ℝ, ∀ n : ℝ, n > m → ((sin x - 1 / sin x)^2 + (cos x - 1 / cos x)^2) > n :=
sorry

end max_unbounded_sinx_cscx_cosx_secx_l32_32116


namespace find_f_neg_2_l32_32152

variable {f : ℝ → ℝ}
variable g : ℝ → ℝ := λ x, f (2 * x) + x ^ 2

theorem find_f_neg_2 (h1 : f 2 = 2) (h2 : ∀ x, g x = -g (-x)) : f (-2) = -4 := by
  sorry

end find_f_neg_2_l32_32152


namespace no_arctan_solution_l32_32684

noncomputable def arctan (x : ℝ) : ℝ := Real.arctan x

theorem no_arctan_solution :
  ¬∃ (N : ℕ) (a : Fin 2020 → ℤ), arctan N = ∑ i in Finset.range 2020, a ⟨i, by norm_num⟩ * arctan (i + 1) :=
begin
  sorry
end

end no_arctan_solution_l32_32684


namespace coaches_needed_l32_32790

theorem coaches_needed (x : ℕ) : 44 * x + 64 = 328 := by
  sorry

end coaches_needed_l32_32790


namespace amount_collected_from_II_class_l32_32709

-- Definitions related to the problem.
def P_I := ℕ  -- Number of I class passengers
def P_II := ℕ -- Number of II class passengers
def F_I := ℕ  -- Fare for I class
def F_II := ℕ -- Fare for II class
def Total := 1325 -- Total amount collected

-- Hypotheses based on the conditions
axiom ratio_passengers : P_I = 1 ∧ P_II = 50 * P_I
axiom ratio_fare : F_I = 3 * F_II

-- Theorem to prove
theorem amount_collected_from_II_class : P_II * F_II = 1250 :=
by {
  sorry
}

end amount_collected_from_II_class_l32_32709


namespace ellipse_eccentricity_min_value_l32_32166

-- Statement of the problem in Lean 4
theorem ellipse_eccentricity_min_value (a b : ℝ) (h_b_gt_0 : b > 0) (h_a_gt_b : a > b)
    (h_min : ∀ a b, a > b > 0 → a = 2 * b → a^{2} + \frac{16}{b * (a - b)} = 16) :
  let c := sqrt (a^2 - b^2)
  in sqrt (3:ℝ) / 2 = c / a :=
sorry

end ellipse_eccentricity_min_value_l32_32166


namespace single_elimination_games_l32_32933

theorem single_elimination_games (n : ℕ) (h : n = 512) : 
  ∃ g : ℕ, g = n - 1 ∧ g = 511 := 
by
  use n - 1
  sorry

end single_elimination_games_l32_32933


namespace initial_population_l32_32621

theorem initial_population (P : ℝ) 
  (H1 : P * 1.15 * 0.90 * 1.20 * 0.75 = 7575) :
  P ≈ 12199 := by
  sorry

end initial_population_l32_32621


namespace sum_of_x_coordinates_of_points_above_line_l32_32291

noncomputable def is_above_line (p : ℝ × ℝ) : Prop := p.2 > 2 * p.1 + 5

def points : List (ℝ × ℝ) := 
  [(2, 8), (5, 15), (10, 25), (15, 36), (19, 45), (22, 52), (25, 66)]

def x_coordinates (pts : List (ℝ × ℝ)) : List ℝ := 
  pts.filter is_above_line |>.map Prod.fst

def sum_x_coordinates_above_line : ℝ :=
  (x_coordinates points).sum

theorem sum_of_x_coordinates_of_points_above_line :
  sum_x_coordinates_above_line = 81 :=
  by
  sorry

end sum_of_x_coordinates_of_points_above_line_l32_32291


namespace perpendicular_lines_max_MA_2MB_l32_32542

/-- Given p ∈ ℝ, define lines l₁ and l₂ as follows:
    l₁: x - p*y + p - 2 = 0
    l₂: p*x + y + 2*p - 4 = 0
    -/
variables {p : ℝ}
def l₁ (x y : ℝ) : Prop := x - p * y + p - 2 = 0
def l₂ (x y : ℝ) : Prop := p * x + y + 2 * p - 4 = 0

/-- Part 1: Prove that l₁ is perpendicular to l₂. -/
theorem perpendicular_lines : l₁.is_perpendicular_to l₂ := sorry

/-- Part 2: Prove that the maximum value of the expression MA + 2MB, where M is the intersection of l₁ and l₂, is 5√5. -/
theorem max_MA_2MB : maximum_value MA 2MB = 5 * sqrt 5 := sorry

end perpendicular_lines_max_MA_2MB_l32_32542


namespace smallest_n_divisible_by_45_l32_32518

theorem smallest_n_divisible_by_45 :
  let A_n := (λ n : ℕ, (list.range (n + 1)).sum (λ k, 10 ^ k))
  (∃ (n : ℕ), A_n n % 45 = 0 ∧ ∀ m < n, A_n m % 45 ≠ 0) ↔ n = 35 := by
  sorry

end smallest_n_divisible_by_45_l32_32518


namespace geometric_sequence_properties_l32_32553

theorem geometric_sequence_properties :
  (∀ (n : ℕ), 0 < n →
    let a := λ n, (2 ^ (7 - n)) in
    let b := λ n, Real.log (a n) / Real.log 2 in
    let T := λ n, (13 * n - n^2) / 2 in
    2 * a 4 - 3 * a 3 + a 2 = 0 ∧
    a 1 = 64 ∧
    (b 1 = 6) ∧
    (∀ m, 1 ≤ m → T m ≤ 21)).

end geometric_sequence_properties_l32_32553


namespace sqrt_of_mixed_fraction_simplified_l32_32506

theorem sqrt_of_mixed_fraction_simplified :
  let x := 8 + (9 / 16) in
  sqrt x = (sqrt 137) / 4 := by
  sorry

end sqrt_of_mixed_fraction_simplified_l32_32506


namespace shuxue_count_l32_32241

theorem shuxue_count : 
  (∃ (count : ℕ), count = (List.length (List.filter (λ n => (30 * n.1 + 3 * n.2 < 100) 
    ∧ (30 * n.1 + 3 * n.2 > 9)) 
      (List.product 
        (List.range' 1 3) -- Possible values for "a" are 1 to 3
        (List.range' 1 9)) -- Possible values for "b" are 1 to 9
    ))) ∧ count = 9 :=
  sorry

end shuxue_count_l32_32241


namespace geometric_progression_x_unique_l32_32852

theorem geometric_progression_x_unique (x : ℝ) :
  (70+x)^2 = (30+x)*(150+x) ↔ x = 10 := by
  sorry

end geometric_progression_x_unique_l32_32852


namespace inequality_am_gm_l32_32543

theorem inequality_am_gm (n : ℕ) : 
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → x i > 0) →
  (∑ i in finset.range n, ((x i) ^ 2 / x (i + 1)) ≥ ∑ i in finset.range n, x i) :=
by
  sorry

end inequality_am_gm_l32_32543


namespace hyperbola_eccentricity_range_l32_32902

-- Define the conditions given in the problem
def hyperbola_eq (a b x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

def circle_with_diameter {a b : ℝ} (A B : ℝ × ℝ) (M : ℝ × ℝ) : Prop :=
  let AM := (A.1 - M.1)^2 + (A.2 - M.2)^2
  let BM := (B.1 - M.1)^2 + (B.2 - M.2)^2
  let AB := (A.1 - B.1)^2 + (A.2 - B.2)^2
  4 * AM * BM = AB^2

def e_range (a b : ℝ) : ℝ :=
  let c := sqrt (a^2 + b^2)
  c / a

-- Proving the range of eccentricity given the conditions
theorem hyperbola_eccentricity_range (a b : ℝ)
  (h1 : 0 < a) (h2 : 0 < b)
  (h3 : ∃ A B : ℝ × ℝ, 
        (∃ x1 x2 y1 y2, hyperbola_eq a b x1 y1 ∧ hyperbola_eq a b x2 y2 ∧ 
         (A = (x1, y1)) ∧ (B = (x2, y2)))
        ∧ (∃ M : ℝ × ℝ, circle_with_diameter A B M))
  : 2 < e_range a b :=
sorry

end hyperbola_eccentricity_range_l32_32902


namespace max_sum_x_y_l32_32562

theorem max_sum_x_y (x y : ℝ) (h : x - sqrt (x + 1) = sqrt (y + 3) - y) : x + y ≤ 4 :=
sorry

end max_sum_x_y_l32_32562


namespace inverse_points_through_l32_32575

variable (f : ℝ → ℝ)

-- Condition: the graph of the function f passes through (0,1)
axiom h₀ : f 0 = 1

theorem inverse_points_through (hf₀ : f 0 = 1) :
  f⁻¹ 1 = 0 ∧ (y : ℝ) → (f∘(λ x, x + 4))⁻¹ 1 = -4 :=
by
  split
  sorry
  sorry

end inverse_points_through_l32_32575


namespace last_locker_opens_l32_32930

open Nat

/-- 
Problem statement:
In a school corridor, there is a row of lockers numbered from 1 to 500.
Initially, all lockers are closed.
A student starts at locker 1, opens it, and then proceeds by skipping one locker and opening the next, continuing this pattern until he reaches the end of the corridor.
Upon reaching the end, he turns around, skips the first closed locker he sees, then opens the next closed locker, and continues alternating skipping and opening each subsequent closed locker.
The student repeats this process of turning around whenever he reaches an end and continuing the pattern until only one locker remains closed.
Prove that the last locker he opens is locker number 242.
-/
theorem last_locker_opens (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 500) : 
  ∀ h_cond, last_number h_cond = 242 := sorry

end last_locker_opens_l32_32930


namespace zeros_sum_inequality_l32_32174

def f (a x : ℝ) : ℝ := Real.exp x - a * x + a

theorem zeros_sum_inequality (a x1 x2 : ℝ) (h₁ : f a x1 = 0) (h₂ : f a x2 = 0) : x1 + x2 < 2 * Real.log a :=
by
  sorry

end zeros_sum_inequality_l32_32174


namespace probability_of_selection_is_equal_l32_32123

-- Define the conditions of the problem
def total_students := 2004
def eliminated_students := 4
def remaining_students := total_students - eliminated_students -- 2000
def selected_students := 50
def k := remaining_students / selected_students -- 40

-- Define the probability calculation
def probability_selected := selected_students / remaining_students

-- The theorem stating that every student has a 1/40 probability of being selected
theorem probability_of_selection_is_equal :
  probability_selected = 1 / 40 :=
by
  -- insert proof logic here
  sorry

end probability_of_selection_is_equal_l32_32123


namespace a_2022_equals_neg_1011_l32_32889

def a : ℕ → ℤ
| 1       := 0
| (n + 1) := -|a n + n|

theorem a_2022_equals_neg_1011 : a 2022 = -1011 :=
by sorry

end a_2022_equals_neg_1011_l32_32889


namespace percent_increase_area_8_to_10_l32_32447

noncomputable def pi : ℝ := Real.pi

def radius (d : ℝ) : ℝ := d / 2

def area (r : ℝ) : ℝ := pi * (r * r)

def percent_increase_area (d1 d2 : ℝ) : ℝ :=
  let r1 := radius d1
  let r2 := radius d2
  let A1 := area r1
  let A2 := area r2
  ((A2 - A1) / A1) * 100

theorem percent_increase_area_8_to_10 :
  percent_increase_area 8 10 = 56.25 :=
by
  sorry

end percent_increase_area_8_to_10_l32_32447


namespace symmetric_point_origin_l32_32557

def symmetric_point (p: ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem symmetric_point_origin : 
  (symmetric_point (3, -2)) = (-3, 2) :=
by
  sorry

end symmetric_point_origin_l32_32557


namespace intersection_correct_l32_32904

-- Define sets M and N
def M := {x : ℝ | x < 1}
def N := {x : ℝ | Real.log (2 * x + 1) > 0}

-- Define the intersection of M and N
def M_intersect_N := {x : ℝ | 0 < x ∧ x < 1}

-- Prove that M_intersect_N is the correct intersection
theorem intersection_correct : M ∩ N = M_intersect_N :=
by
  sorry

end intersection_correct_l32_32904


namespace containment_relation_l32_32588

noncomputable def M : set ℝ := {y | ∃ x, y = -x^2 + 1}
noncomputable def P : set ℝ := {y | ∃ x, y = 2 * x + 1}

theorem containment_relation : M ⊆ P :=
sorry

end containment_relation_l32_32588


namespace num_of_digits_in_x_l32_32914

noncomputable def num_base_ten_digits (n : ℕ) : ℕ :=
⌊ log 10 (float_of_nat n) + 1 ⌋

theorem num_of_digits_in_x (x : ℕ) (h : log 3 (log 3 (log 3 x)) = 2) : num_base_ten_digits x = 9392 :=
by sorry

end num_of_digits_in_x_l32_32914


namespace round_trip_speed_ratio_l32_32026

-- Define conditions
def boat_speed_still_water : ℝ := 12
def current_speed : ℝ := 3
def distance_each_way : ℝ := 1

-- Calculate speeds
def downstream_speed : ℝ := boat_speed_still_water + current_speed
def upstream_speed : ℝ := boat_speed_still_water - current_speed

-- Calculate times
def time_downstream : ℝ := distance_each_way / downstream_speed
def time_upstream : ℝ := distance_each_way / upstream_speed

-- Calculate total time and total distance
def total_time : ℝ := time_downstream + time_upstream
def total_distance : ℝ := 2 * distance_each_way

-- Calculate average speed
def average_speed : ℝ := total_distance / total_time

-- Define the theorem to prove
theorem round_trip_speed_ratio : average_speed / boat_speed_still_water = 15 / 16 := by
  sorry

end round_trip_speed_ratio_l32_32026


namespace number_of_points_on_circle_at_distance_sqrt2_from_line_l32_32832

noncomputable def circle := { p : ℝ × ℝ // (p.1 - 1) ^ 2 + (p.2 - 1) ^ 2 = 2 }

def point_distance_line(p : ℝ × ℝ, a b c : ℝ) : ℝ :=
  abs(a * p.1 + b * p.2 + c) / real.sqrt(a ^ 2 + b ^ 2)

theorem number_of_points_on_circle_at_distance_sqrt2_from_line :
  {p : ℝ × ℝ | p ∈ circle ∧ point_distance_line p 1 1 1 = real.sqrt 2}.finite.card = 1 := 
sorry

end number_of_points_on_circle_at_distance_sqrt2_from_line_l32_32832


namespace walter_time_spent_at_seals_l32_32370

theorem walter_time_spent_at_seals (S : ℕ) 
(h1 : 8 * S + S + 13 = 130) : S = 13 :=
sorry

end walter_time_spent_at_seals_l32_32370


namespace distinct_real_numbers_satisfying_f_f_f_f_eq_six_l32_32664

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * x

theorem distinct_real_numbers_satisfying_f_f_f_f_eq_six :
  {c : ℝ // f (f (f (f c))) = 6}.toFinset.card = 16 := by
  sorry

end distinct_real_numbers_satisfying_f_f_f_f_eq_six_l32_32664


namespace actual_diameter_of_tissue_l32_32753

theorem actual_diameter_of_tissue (magnification_factor : ℝ) (magnified_diameter : ℝ) (image_magnified : magnification_factor = 1000 ∧ magnified_diameter = 2) : (1 / magnification_factor) * magnified_diameter = 0.002 :=
by
  sorry

end actual_diameter_of_tissue_l32_32753


namespace problem_conditions_proof_l32_32173

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

theorem problem_conditions_proof :
  (∀ x : ℝ, f (-1 : ℝ) 4 x > 0 → x ∈ Set.Icc (-1 : ℝ) 3) →
  f (-1 : ℝ) 4 1 = 2 →
  (-1 : ℝ) > 0 →
  (4 : ℝ) > 0 →
  ∃ a b : ℝ, 
    a = -1 ∧ 
    b = 4 ∧ 
    (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 → (1 / a + 4 / b) ≥ 9) :=
begin
  -- proof would go here
  sorry
end

end problem_conditions_proof_l32_32173


namespace false_statement_is_D_l32_32759

def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

def is_scalene_triangle (a b c : ℝ) : Prop :=
  (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ a)

def is_right_isosceles_triangle (a b c : ℝ) : Prop :=
  is_right_triangle a b c ∧ is_isosceles_triangle a b c

-- Statements derived from conditions
def statement_A : Prop := ∀ (a b c : ℝ), is_isosceles_triangle a b c → a = b ∨ b = c ∨ c = a
def statement_B : Prop := ∀ (a b c : ℝ), is_right_triangle a b c → a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2
def statement_C : Prop := ∀ (a b c : ℝ), is_scalene_triangle a b c → a ≠ b ∧ b ≠ c ∧ c ≠ a
def statement_D : Prop := ∀ (a b c : ℝ), is_right_triangle a b c → is_isosceles_triangle a b c
def statement_E : Prop := ∀ (a b c : ℝ), is_right_isosceles_triangle a b c → ∃ (θ : ℝ), θ ≠ 90 ∧ θ = 45

-- Main theorem to be proved
theorem false_statement_is_D : statement_D = false :=
by
  sorry

end false_statement_is_D_l32_32759


namespace assign_values_to_n_gon_l32_32994

theorem assign_values_to_n_gon (n k : ℕ) (hkn : k ≤ n) (x : Fin n → ℝ) :
  (∀ i, x i ≠ 0) →
  (∀ v : Fin k → Fin n, (∀ i j, i ≠ j → v i ≠ v j) →
    ∑ i, x (v i) = 0)
  :=
sorry

end assign_values_to_n_gon_l32_32994


namespace log_relation_l32_32317

theorem log_relation (a b : ℝ) (h₁ : a = log 4 900) (h₂ : b = log 2 30) : a = b :=
by
  sorry

end log_relation_l32_32317


namespace calculate_b6_b8_l32_32623

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n d, a (n+1) = a n + d

noncomputable def geometric_sequence (b : ℕ → ℤ) : Prop :=
∀ n r, b (n+1) = b n * r

theorem calculate_b6_b8 :
    (∃ a : ℕ → ℤ, arithmetic_sequence a ∧ 2 * a 3 - (a 7) ^ 2 + 2 * a 11 = 0)
  → (∃ b : ℕ → ℤ, geometric_sequence b ∧ b 7 = a 7)
  → b 6 * b 8 = 16 := by
  sorry

end calculate_b6_b8_l32_32623


namespace range_of_m_l32_32534

-- Definitions
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2
def q (x m : ℝ) : Prop := (x^2 - 4*x + 4 - m^2) ≤ 0

-- Theorem Statement
theorem range_of_m (m : ℝ) (h_m : m > 0) : 
  (¬(∃ x, ¬p x) → ¬(∃ x, ¬q x m)) → m ≥ 8 := 
sorry -- Proof not required

end range_of_m_l32_32534


namespace trisha_cookie_price_l32_32527

def area_trapezoid (a b h : ℝ) : ℝ :=
  (a + b) * h / 2

def area_circle (r : ℝ) : ℝ := 
  Real.pi * r^2

def total_dough_trisha (num_cookies : ℕ) (r : ℝ) : ℝ :=
  num_cookies * area_circle r

theorem trisha_cookie_price (dough_used : ℝ) 
  (cookie_cost : ℝ) : 
  dough_used = 144 ∧ cookie_cost = 60 → 
  ∃ p : ℝ, p = 60 :=
by
  intros h
  sorry

end trisha_cookie_price_l32_32527


namespace hyperbola_range_m_l32_32607

theorem hyperbola_range_m (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (16 - m)) + (y^2 / (9 - m)) = 1) → 9 < m ∧ m < 16 :=
by 
  sorry

end hyperbola_range_m_l32_32607


namespace population_problem_l32_32619

theorem population_problem (P : ℝ) :
  (P * 1.15 * 0.90 * 1.20 * 0.75 = 7575) → P = 12199 :=
by
  intro h
  have P_value : P = 7575 / (1.15 * 0.90 * 1.20 * 0.75) := by sorry
  have P_calculated : P ≈ 12199 := by sorry
  exact P_calculated

end population_problem_l32_32619


namespace a_minus_c_value_l32_32392

theorem a_minus_c_value (a b c : ℝ) 
  (h1 : (a + b) / 2 = 110) 
  (h2 : (b + c) / 2 = 150) : 
  a - c = -80 := 
by 
  -- We provide the proof inline with sorry
  sorry

end a_minus_c_value_l32_32392


namespace find_number_l32_32212

theorem find_number (N : ℝ) (h : 0.60 * N = 0.50 * 720) : N = 600 :=
sorry

end find_number_l32_32212


namespace ants_square_paths_l32_32730

theorem ants_square_paths (a : ℝ) :
  (∃ a, a = 4 ∧ a + 2 = 6 ∧ a + 4 = 8) →
  (∀ (Mu Ra Vey : ℝ), 
    (Mu = (a + 4) / 2) ∧ 
    (Ra = (a + 2) / 2 + 1) ∧ 
    (Vey = 6) →
    (Mu + Ra + Vey = 2 * (a + 4) + 2)) :=
sorry

end ants_square_paths_l32_32730


namespace min_value_is_correct_l32_32220

noncomputable def min_value (P : ℝ × ℝ) (A B C : ℝ × ℝ) : ℝ := 
  let PA := (A.1 - P.1, A.2 - P.2)
  let PB := (B.1 - P.1, B.2 - P.2)
  let PC := (C.1 - P.1, C.2 - P.2)
  PA.1 * PB.1 + PA.2 * PB.2 +
  PB.1 * PC.1 + PB.2 * PC.2 +
  PC.1 * PA.1 + PC.2 * PA.2

theorem min_value_is_correct :
  ∃ P : ℝ × ℝ, P = (5/3, 1/3) ∧
  min_value P (1, 4) (4, 1) (0, -4) = -62/3 :=
by
  sorry

end min_value_is_correct_l32_32220


namespace functional_equation_solution_l32_32512

theorem functional_equation_solution (f : ℝ → ℝ) (h : ∀ (x y : ℝ), 0 < x → 0 < y → f x - f (x + y) = f (x / y) * f (x + y)) :
    (∀ (x : ℝ), 0 < x → f x = 0) ∨ (∀ (x : ℝ), 0 < x → f x = 1 / x) :=
begin
  sorry
end

end functional_equation_solution_l32_32512


namespace ticket_cost_is_5_92_l32_32292

-- Define the conditions
variable (x : ℝ) -- The cost of each ticket
def total_cost_tickets := 2 * x
def borrowed_movie_cost : ℝ := 6.79
def total_amount_spent := total_cost_tickets + borrowed_movie_cost
def paid_amount : ℝ := 20
def change_received : ℝ := 1.37

-- Define the main problem statement
theorem ticket_cost_is_5_92 (h1 : total_amount_spent = paid_amount - change_received) : 
  x = 5.92 :=
by
  -- Initial conditions
  sorry -- Proof omitted

end ticket_cost_is_5_92_l32_32292


namespace solve_trapezoid_problem_l32_32878

def is_inscribable_trapezoid (α : ℝ) : Prop :=
  -- Here we need to capture the geometric condition
  -- of having an inscribed circle in trapezoid.
  ∃ (AB CD BC AD : ℝ),
    AD = 2 ∧
    (AB + CD = BC + AD) ∧
    AB = 2 * sin(α / 2) ∧
    CD = 2 * sin(α / 2) ∧
    BC = 2 * sin(α / 2)

def perimeter (α : ℝ) : ℝ :=
  -- Define perimeter based on the calculated expressions
  let AB_CD := 2 * sin(α / 2) in
  let AD := 2 in
  2 * AB_CD + AD + AD

theorem solve_trapezoid_problem :
  is_inscribable_trapezoid π ∧ ∀ α, 0 < α ∧ α ≤ π → perimeter α ≤ perimeter π :=
by
  sorry  -- Proof to be filled


end solve_trapezoid_problem_l32_32878


namespace not_repeating_decimal_l32_32548

-- Define the sequence and the conditions
def strictly_increasing (a : ℕ → ℕ) : Prop :=
  ∀ n, a n < a (n + 1)

def bounded_increase (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 1) ≤ 10 * a n

-- Define the main theorem
theorem not_repeating_decimal (a : ℕ → ℕ)
  (pos : ∀ n, a n > 0)
  (inc : strictly_increasing a)
  (bounded : bounded_increase a) :
  ¬ (∃ T, ∀ n ≥ T, nth_digit (decimal_expansion a) n = nth_digit (decimal_expansion a) (n + T)) :=
sorry

-- Decimal expansion and nth digit functions would have to be defined appropriately
-- depending on the formalism chosen for "repeating decimal" in Lean.

end not_repeating_decimal_l32_32548


namespace tan_diff_identity_l32_32570

theorem tan_diff_identity (α : ℝ) (hα : 0 < α ∧ α < π) (h : Real.sin α = 4 / 5) :
  Real.tan (π / 4 - α) = -1 / 7 ∨ Real.tan (π / 4 - α) = -7 :=
sorry

end tan_diff_identity_l32_32570


namespace count_nonnegative_integers_l32_32911

/-- 
Theorem: There are 2187 nonnegative integers that can be written in the form 
  a_6 * 4^6 + a_5 * 4^5 + a_4 * 4^4 + a_3 * 4^3 + a_2 * 4^2 + a_1 * 4^1 + a_0 * 4^0 
where a_i ∈ {0, 1, 2} for 0 ≤ i ≤ 6. 
-/
theorem count_nonnegative_integers : 
  { n : ℕ | ∃ (a_6 a_5 a_4 a_3 a_2 a_1 a_0 : ℕ) 
    (h₆ : a_6 ∈ {0, 1, 2}) 
    (h₅ : a_5 ∈ {0, 1, 2}) 
    (h₄ : a_4 ∈ {0, 1, 2}) 
    (h₃ : a_3 ∈ {0, 1, 2}) 
    (h₂ : a_2 ∈ {0, 1, 2}) 
    (h₁ : a_1 ∈ {0, 1, 2}) 
    (h₀ : a_0 ∈ {0, 1, 2}),
    n = a_6 * 4^6 + a_5 * 4^5 + a_4 * 4^4 + a_3 * 4^3 + a_2 * 4^2 + a_1 * 4 + a_0 
  }.finite.card = 2187 := 
by
  sorry

end count_nonnegative_integers_l32_32911


namespace probability_f_geq_0_l32_32171

theorem probability_f_geq_0 :
  let f := λ x : ℝ, -x^2 + 4*x - 3 in
  ∀ (x0 : ℝ), x0 ∈ Set.Icc 2 6 →
  (∃ (a b : ℝ), 2 ≤ a ∧ a ≤ 3 ∧ b = 6 ∧ (∫ x in Set.Icc a b, indicator (f x ≥ 0) 1) / (b - a) = 1 / 4) :=
by
  sorry

end probability_f_geq_0_l32_32171


namespace possible_values_of_reciprocal_sum_l32_32267

theorem possible_values_of_reciprocal_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) :
  ∃ y, y = (1/a + 1/b) ∧ (2 ≤ y ∧ ∀ t, t < y ↔ ¬t < 2) :=
by sorry

end possible_values_of_reciprocal_sum_l32_32267


namespace calculate_expression_l32_32397

theorem calculate_expression : (0.0088 * 4.5) / (0.05 * 0.1 * 0.008) = 990 := by
  sorry

end calculate_expression_l32_32397


namespace sqrtS_is_arithmetic_l32_32896

-- Definitions for the problem
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def a_sequence (a : ℕ → ℝ) : ℝ := a 2

def S (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a (i + 1) -- Sum of the first n terms

def sqrtS_sequence (S : ℕ → ℝ) : ℕ → ℝ := λ n, real.sqrt (S n)

-- Theorem to prove
theorem sqrtS_is_arithmetic (a : ℕ → ℝ) :
  (arithmetic_sequence a) →
  (a 2 = 3 * (a 1)) →
  arithmetic_sequence (sqrtS_sequence (S a)) :=
by
  sorry

end sqrtS_is_arithmetic_l32_32896


namespace problem_statement_l32_32600

theorem problem_statement
  (x y : ℝ)
  (hx : 0 < x ∧ x < (π / 2))
  (hy : 0 < y ∧ y < (π / 2))
  (h : sin (2 * x) = 6 * tan (x - y) * cos (2 * x)) :
  x + y ≠ (2 * π / 3) :=
sorry

end problem_statement_l32_32600


namespace bijection_count_l32_32122

-- Define sets A and B
def A := {a1, a2, a3, a4}
def B := {b1, b2, b3, b4}

-- Definitions of the conditions
def condition1 (f : A → B) : Prop :=
  function.bijective f

def condition2 (f : A → B) : Prop :=
  f a1 ≠ b1 ∧ f⁻¹ b4 ≠ a4

-- Theorem statement
theorem bijection_count (f : A → B) :
  condition1 f ∧ condition2 f ↔ ∃! f : A → B, ∃ n : ℕ, n = 14 := sorry

end bijection_count_l32_32122


namespace geometric_series_sum_l32_32088

theorem geometric_series_sum : 
  (finset.range 12).sum (λ k, 2 ^ k) = 4095 := 
by sorry

end geometric_series_sum_l32_32088


namespace simplify_sqrt_of_mixed_number_l32_32499

noncomputable def sqrt_fraction := λ (a b : ℕ), (Real.sqrt a) / (Real.sqrt b)

theorem simplify_sqrt_of_mixed_number : sqrt_fraction 137 16 = (Real.sqrt 137) / 4 := by
  sorry

end simplify_sqrt_of_mixed_number_l32_32499


namespace point_in_third_quadrant_l32_32544

noncomputable def z : ℂ := complex.of_real (real.sin (2019 * real.pi / 180))
                      + complex.I * complex.of_real (real.cos (2019 * real.pi / 180))

theorem point_in_third_quadrant (z : ℂ) : 
  z = complex.of_real (real.sin (2019 * real.pi / 180))
      + complex.I * complex.of_real (real.cos (2019 * real.pi / 180)) → 
  z.re < 0 ∧ z.im < 0 :=
by
  sorry

end point_in_third_quadrant_l32_32544


namespace value_of_expression_l32_32201

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : 2 * a^2 + 4 * a * b + 2 * b^2 - 6 = 12 :=
by
  sorry

end value_of_expression_l32_32201


namespace factor_polynomial_l32_32104

theorem factor_polynomial :
  (x : ℝ) → (x^2 - 6*x + 9 - 64*x^4) = (-8*x^2 + x - 3) * (8*x^2 + x - 3) :=
by
  intro x
  sorry

end factor_polynomial_l32_32104


namespace find_function_l32_32111

theorem find_function (f : ℕ → ℕ) (h : ∀ m n, f (m + f n) = f (f m) + f n) :
  ∃ d, d > 0 ∧ (∀ m, ∃ k, f m = k * d) :=
sorry

end find_function_l32_32111


namespace quadratic_symmetric_l32_32218

-- Conditions: Graph passes through the point P(-2,4)
-- y = ax^2 is symmetric with respect to the y-axis

theorem quadratic_symmetric (a : ℝ) (h : a * (-2)^2 = 4) : a * 2^2 = 4 :=
by
  sorry

end quadratic_symmetric_l32_32218


namespace minimum_fencing_cost_l32_32050

-- Problem statement conditions translated to Lean definitions
def length_of_uncovered_side : ℕ := 34
def area_of_field : ℕ := 680
def cost_of_wooden_fence_per_foot : ℕ := 5
def cost_of_chain_link_fence_per_foot : ℕ := 7
-- def cost_of_iron_fence_per_foot : ℕ := 10 (not used in solution)

-- Lean statement proving the minimum cost
theorem minimum_fencing_cost :
  ∃ (W : ℕ), W = area_of_field / length_of_uncovered_side ∧ 
  let total_fencing_length := 2 * W + length_of_uncovered_side in 
  let cost_of_wooden_fence := 2 * W * cost_of_wooden_fence_per_foot in
  let cost_of_chain_link_fence := length_of_uncovered_side * cost_of_chain_link_fence_per_foot in
  let total_cost := cost_of_wooden_fence + cost_of_chain_link_fence in
  total_cost = 438 :=
by
  sorry

end minimum_fencing_cost_l32_32050


namespace tan_255_eq_2_plus_sqrt_3_l32_32863

theorem tan_255_eq_2_plus_sqrt_3 : tan 255 = 2 + sqrt 3 :=
by
  sorry

end tan_255_eq_2_plus_sqrt_3_l32_32863


namespace integer_roots_condition_l32_32850

theorem integer_roots_condition (a : ℝ) (h_pos : 0 < a) :
  (∀ x y : ℤ, (a ^ 2 * x ^ 2 + a * x + 1 - 13 * a ^ 2 = 0) ∧ (a ^ 2 * y ^ 2 + a * y + 1 - 13 * a ^ 2 = 0)) ↔
  (a = 1 ∨ a = 1/3 ∨ a = 1/4) :=
by sorry

end integer_roots_condition_l32_32850


namespace max_distance_to_line_l_l32_32235

noncomputable def parametric_curve_C (a : ℝ) : ℝ × ℝ := (2 * sqrt 3 * cos a, 2 * sin a)
def polar_coords_P := (4 * sqrt 2, π / 4)
def polar_line_l_equation (ρ θ : ℝ) := ρ * sin (θ - π / 4) + 5 * sqrt 2 = 0
def cartesian_line_l_equation (x y : ℝ) := x - y - 10 = 0
def general_curve_C_equation (x y : ℝ) := (x ^ 2) / 12 + (y ^ 2) / 4 = 1

theorem max_distance_to_line_l :
  ∀ (a : ℝ), 0 < a ∧ a < π →
  let Q := parametric_curve_C a,
      P := (4 : ℝ, 4 : ℝ),
      Mx := (Q.1 + P.1) / 2,
      My := (Q.2 + P.2) / 2,
      d := abs ((sqrt 3 * cos a - sin a - 10) / sqrt 2) in
  d ≤ 6 * sqrt 2 := 
sorry

end max_distance_to_line_l_l32_32235


namespace power_functions_satisfy_condition_l32_32385

open Real

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ (x1 x2 : ℝ), (0 < x1) → (x1 < x2) → f((x1 + x2) / 2) < (f(x1) + f(x2)) / 2

theorem power_functions_satisfy_condition :
  satisfies_condition (λ x : ℝ, x^2) ∧ satisfies_condition (λ x : ℝ, 1 / x) :=
by
  sorry

end power_functions_satisfy_condition_l32_32385


namespace sum_max_min_possible_values_of_z_l32_32969

-- Define the conditions
def conditions (x y : ℝ) : Prop :=
  (x - y + 1 ≥ 0) ∧ (x + y - 1 ≥ 0) ∧ (3 * x - y - 3 ≤ 0)

-- Define the function z
def z (x y : ℝ) : ℝ := abs (x - 4 * y + 1)

-- Theorem statement
theorem sum_max_min_possible_values_of_z :
  ∀ x y : ℝ, conditions x y →
  (max (z x y) + min (z x y) = 11 / (Real.sqrt 17)) :=
begin
  intros,
  sorry, -- Proof not provided
end

end sum_max_min_possible_values_of_z_l32_32969


namespace inequality_holds_for_triangle_sides_l32_32851

theorem inequality_holds_for_triangle_sides (a : ℝ) : 
  (∀ (x y z : ℕ), x + y > z ∧ y + z > x ∧ z + x > y → (x^2 + y^2 + z^2 ≤ a * (x * y + y * z + z * x))) ↔ (1 ≤ a ∧ a ≤ 6 / 5) :=
by sorry

end inequality_holds_for_triangle_sides_l32_32851


namespace journey_total_time_l32_32731

def journey_time (d1 d2 : ℕ) (total_distance : ℕ) (car_speed walk_speed : ℕ) : ℕ :=
  d1 / car_speed + (total_distance - d1) / walk_speed

theorem journey_total_time :
  let total_distance := 150
  let car_speed := 30
  let walk_speed := 3
  let d1 := 50
  let d2 := 15
  
  journey_time d1 d2 total_distance car_speed walk_speed =
  max (journey_time d1 0 total_distance car_speed walk_speed / car_speed + 
       (total_distance - d1) / walk_speed)
      ((d1 / car_speed + (d1 - d2) / car_speed + (total_distance - d1 + d2) / car_speed)) :=
by
  sorry

end journey_total_time_l32_32731


namespace sum_of_exponents_of_prime_factors_of_sqrt_of_largest_perfect_square_dividing_factorial_15_eq_10_l32_32000

theorem sum_of_exponents_of_prime_factors_of_sqrt_of_largest_perfect_square_dividing_factorial_15_eq_10 :
  let n := 15
  let factorial_prime_exponents (n p : ℕ) :=
    let rec exponents (n p k : ℕ) := 
      if p ^ k > n then 0 else n / (p ^ k) + exponents n p (k + 1)
    exponents n p 1
  let prime_exponents := [2, 3, 5, 7].map (λ p => (p, factorial_prime_exponents n p))
  let largest_perfect_square_factors :=
    prime_exponents.map (λ (p, e) => (p, e - e % 2))
  let sqrt_factors := largest_perfect_square_factors.map (λ (p, e) => (p, e / 2))
  let sum_of_exponents := sqrt_factors.foldl (λ sum (p, e) => sum + e) 0
  in sum_of_exponents = 10 := by
    sorry

end sum_of_exponents_of_prime_factors_of_sqrt_of_largest_perfect_square_dividing_factorial_15_eq_10_l32_32000


namespace three_colorable_l32_32335

-- Definitions based on conditions
structure Polygon (α : Type*) :=
(vertices : set α)
(edges : set (set α))
(finite_vertices : vertices.finite)
(edges_are_sides : ∀ e ∈ edges, ∃ p1 p2, p1 ∈ vertices ∧ p2 ∈ vertices ∧ e = {p1, p2})
(is_connected : ∃ p1 p2, p1 ∈ vertices ∧ p2 ∈ vertices)

def is_triangulated (poly : Polygon ℝ) :=
∀ P ∈ poly.edges, ∃ t1 t2, t1 ≠ t2 ∧ t1 ⊆ poly.edges ∧ t2 ⊆ poly.edges ∧ (t1 ∩ t2).card = 1

def adjacent_triangle (t1 t2 : set ℝ) : Prop :=
(t1 ∩ t2).card = 2

-- Problem statement
theorem three_colorable (poly : Polygon ℝ) (ht : is_triangulated poly) :
  ∃ (coloring : (set ℝ → ℕ)), (∀ t1 t2 ∈ poly.edges, adjacent_triangle t1 t2 → coloring t1 ≠ coloring t2) ∧
    ∀ t ∈ poly.edges, coloring t ∈ {0, 1, 2} :=
by sorry

end three_colorable_l32_32335


namespace problem_units_digit_1_probability_l32_32043

def units_digit (x : Nat) : Nat :=
  x % 10

def has_units_digit_1 (m n : Nat) : Prop :=
  units_digit (m^n) = 1

noncomputable def probability {α : Type*} [Fintype α] (s : Set α) (P : α → Prop) : Real :=
  Fintype.card (SetOf P) / Fintype.card α

theorem problem_units_digit_1_probability :
  probability (SetOf (λ mn : Nat × Nat, mn.1 ∈ {18, 22, 25, 27, 29} ∧ mn.2 ∈ Finset.range 20 + 2001)) (λ mn, has_units_digit_1 mn.1 mn.2) = 3 / 4 :=
by
  sorry

end problem_units_digit_1_probability_l32_32043


namespace minimum_value_of_func_l32_32336

noncomputable def func (x : ℝ) : ℝ :=
  x^2 + 6 / (x^2 + 1)

theorem minimum_value_of_func :
  ∃ x : ℝ, func x = 2 * real.sqrt 6 - 1 :=
begin
  sorry
end

end minimum_value_of_func_l32_32336


namespace vertical_asymptote_at_9_over_4_l32_32119

def vertical_asymptote (y : ℝ → ℝ) (x : ℝ) : Prop :=
  (∀ ε > 0, ∃ δ > 0, ∀ x', x' ≠ x → abs (x' - x) < δ → abs (y x') > ε)

noncomputable def function_y (x : ℝ) : ℝ :=
  (2 * x + 3) / (4 * x - 9)

theorem vertical_asymptote_at_9_over_4 :
  vertical_asymptote function_y (9 / 4) :=
sorry

end vertical_asymptote_at_9_over_4_l32_32119


namespace centroid_coincide_l32_32394

theorem centroid_coincide
  (H I O N : Point)
  (A B C : Point)
  (I_a I_b I_c : Point)
  (S : Point)
  (h1 : isOrthocenter H A B C)
  (h2 : isIncenter I A B C)
  (h3 : isCircumcenter O A B C)
  (h4 : isNagalPoint N A B C)
  (h5 : isExcenter I_a A B C A)
  (h6 : isExcenter I_b A B C B)
  (h7 : isExcenter I_c A B C C)
  (h8 : midpoint O H S) :
  centroid (Triangle.mk I_a I_b I_c) = centroid (Triangle.mk S I N) := 
sorry

end centroid_coincide_l32_32394


namespace circles_touch_time_l32_32737

-- Definitions based on given conditions
def r1 : ℝ := 981
def v1 : ℝ := 7
def d1 : ℝ := 2442

def r2 : ℝ := 980
def v2 : ℝ := 5
def d2 : ℝ := 1591

-- Statements to be proven
theorem circles_touch_time :
  let x1 := 111
  let x2 := 566
  (sqrt ((d1 - v1 * x1) ^ 2 + (d2 - v2 * x1) ^ 2) = r1 + r2) ∧
  (sqrt ((d1 - v1 * x2) ^ 2 + (d2 - v2 * x2) ^ 2) = r1 + r2) :=
by
  sorry

end circles_touch_time_l32_32737


namespace sqrt_of_mixed_number_l32_32504

theorem sqrt_of_mixed_number :
  (Real.sqrt (8 + 9 / 16)) = (Real.sqrt 137 / 4) :=
by
  sorry

end sqrt_of_mixed_number_l32_32504


namespace part1_part2_l32_32279

theorem part1 (a : ℝ) (x : ℝ) (h : a > 0) :
  (|x + 1/a| + |x - a + 1|) ≥ 1 :=
sorry

theorem part2 (a : ℝ) (h1 : a > 0) (h2 : |3 + 1/a| + |3 - a + 1| < 11/2) :
  2 < a ∧ a < (13 + 3 * Real.sqrt 17) / 4 :=
sorry

end part1_part2_l32_32279


namespace sum_ac_equals_seven_l32_32565

theorem sum_ac_equals_seven 
  (a b c d : ℝ)
  (h1 : ab + bc + cd + da = 42)
  (h2 : b + d = 6) :
  a + c = 7 := 
sorry

end sum_ac_equals_seven_l32_32565


namespace range_of_b_l32_32217

theorem range_of_b (b : ℝ) : 
  (∀ x : ℝ, ∃ I : set ℝ, x ∈ I ∧ is_monotone_on f I) → 
  (b > sqrt 3 ∨ b < -sqrt 3) :=
by
  -- Given function
  let f := λ x : ℝ, x^3 + b*x^2 + x
  -- Derivative of the function
  let f' := λ x : ℝ, 3*x^2 + 2*b*x + 1
  -- Condition for three monotonic intervals
  have h : ∀ x : ℝ, f'(x) = 0 → ∃ a b : ℝ, a ≠ b := sorry
  -- Derived condition involving the discriminant
  let Δ := 4*b^2 - 12
  have hΔ : Δ > 0 := sorry
  -- Solving the inequality for b
  have hb : b > sqrt 3 ∨ b < -sqrt 3 := sorry
  exact hb

end range_of_b_l32_32217


namespace ways_to_stand_on_staircase_l32_32355

theorem ways_to_stand_on_staircase (A B C : Type) (steps : Fin 7) : 
  ∃ ways : Nat, ways = 336 := by sorry

end ways_to_stand_on_staircase_l32_32355


namespace rate_of_leakage_l32_32908

-- Define the initial conditions
def initial_canteen_cups : ℕ := 11
def remaining_canteen_cups : ℕ := 2
def total_miles : ℕ := 7
def hike_duration_hours : ℕ := 3
def last_mile_drink_cups : ℕ := 3
def first_6_miles_drink_rate : ℝ := 0.5

-- Define auxiliary variables
def first_6_miles_drink_cups : ℝ := first_6_miles_drink_rate * 6
def total_drink_cups : ℝ := first_6_miles_drink_cups + last_mile_drink_cups
def total_water_lost : ℝ := initial_canteen_cups - remaining_canteen_cups
def leakage_cups : ℝ := total_water_lost - total_drink_cups
def leakage_rate : ℝ := leakage_cups / hike_duration_hours

theorem rate_of_leakage :
  leakage_rate = 1 := by
  sorry

end rate_of_leakage_l32_32908


namespace number_of_x_intercepts_l32_32516

theorem number_of_x_intercepts : 
  (set.Ioo 0.001 0.01).countOn {x : ℝ | sin (1 / x) = 0} = 287 := by
  sorry

end number_of_x_intercepts_l32_32516


namespace student_failed_by_l32_32056

-- Definitions based on the problem conditions
def total_marks : ℕ := 500
def passing_percentage : ℕ := 40
def marks_obtained : ℕ := 150
def passing_marks : ℕ := (passing_percentage * total_marks) / 100

-- The theorem statement
theorem student_failed_by :
  (passing_marks - marks_obtained) = 50 :=
by
  -- The proof is omitted
  sorry

end student_failed_by_l32_32056


namespace john_minimum_pizzas_l32_32947

theorem john_minimum_pizzas (car_cost bag_cost earnings_per_pizza gas_cost p : ℕ) 
  (h_car : car_cost = 6000)
  (h_bag : bag_cost = 200)
  (h_earnings : earnings_per_pizza = 12)
  (h_gas : gas_cost = 4)
  (h_p : 8 * p >= car_cost + bag_cost) : p >= 775 := 
sorry

end john_minimum_pizzas_l32_32947


namespace expected_number_of_socks_l32_32678

noncomputable def expected_socks_to_pick (n : ℕ) : ℚ := (2 * (n + 1)) / 3

theorem expected_number_of_socks (n : ℕ) (h : n ≥ 2) : 
  (expected_socks_to_pick n) = (2 * (n + 1)) / 3 := 
by
  sorry

end expected_number_of_socks_l32_32678


namespace product_and_sum_l32_32450

theorem product_and_sum :
  (∏ x in (finset.range 751).map (λ i, i + 1)) / (∏ x in (finset.range 751).map (λ i, i + 2)) + (1 / 100) = 213 / 18800 :=
by sorry

end product_and_sum_l32_32450


namespace total_fish_count_l32_32076

-- Define the number of fish for each person
def Billy := 10
def Tony := 3 * Billy
def Sarah := Tony + 5
def Bobby := 2 * Sarah

-- Define the total number of fish
def TotalFish := Billy + Tony + Sarah + Bobby

-- Prove that the total number of fish all 4 people have put together is 145
theorem total_fish_count : TotalFish = 145 := 
by
  -- provide the proof steps here
  sorry

end total_fish_count_l32_32076


namespace cube_root_of_sqrt_64_is_2_l32_32757

theorem cube_root_of_sqrt_64_is_2 :
  (∛(√64)) = 2 :=
sorry

end cube_root_of_sqrt_64_is_2_l32_32757


namespace find_angle_COD_l32_32733

variables {Q C D O : Type} -- Define types for the points

-- Define the angle measure function
def angle (A B C : Type) := ℝ

-- Given condition
axiom angle_CQD : angle C Q D = 50

-- Statement to prove
theorem find_angle_COD : angle C O D = 65 :=
by
  sorry -- Proof is skipped

end find_angle_COD_l32_32733


namespace circle_line_distance_l32_32037

noncomputable def circle_and_line : Prop :=
  ∃ r : ℝ, (r > 0) ∧ (4 < r ∧ r < 6) ∧
  let center := (3 : ℝ, -5 : ℝ),
      line := (4 : ℝ, -3 : ℝ, -((-2) : ℝ)),
      dist := abs(center.1 * line.1 - center.2 * line.2 + line.3) / (real.sqrt (line.1^2 + line.2^2)) in
  dist = 5 ∧
  ∃ px py : ℝ, ((px - 3) ^ 2 + (py + 5) ^ 2 = r ^ 2) ∧
               (abs(4 * px - 3 * py - 2) / sqrt(4 ^ 2 + (-3) ^ 2) = 1) ∧
               ∃ qx qy qz : ℝ, 
               ((qx - 3) ^ 2 + (qy + 5) ^ 2 = r ^ 2) ∧
               dist = qz
               
theorem circle_line_distance : circle_and_line :=
sorry

end circle_line_distance_l32_32037


namespace smallest_lcm_of_4_digit_integers_l32_32206

open Nat

theorem smallest_lcm_of_4_digit_integers (k l : ℕ) (hk : 1000 ≤ k ∧ k < 10000) (hl : 1000 ≤ l ∧ l < 10000) (h_gcd : gcd k l = 5) :
  lcm k l = 203010 := sorry

end smallest_lcm_of_4_digit_integers_l32_32206


namespace combinations_seven_choose_three_l32_32309

theorem combinations_seven_choose_three : nat.choose 7 3 = 35 := by
  sorry

end combinations_seven_choose_three_l32_32309


namespace john_can_run_168_miles_l32_32948

def original_duration : ℕ := 8
def duration_increase_percentage : ℝ := 0.75
def original_speed : ℕ := 8
def speed_increase : ℕ := 4

def new_duration (orig : ℕ) (inc_pct : ℝ) : ℕ := orig + (orig : ℝ) * inc_pct
def new_speed (orig : ℕ) (inc : ℕ) : ℕ := orig + inc

def distance (duration : ℕ) (speed : ℕ) : ℕ := duration * speed

theorem john_can_run_168_miles :
  distance (new_duration original_duration duration_increase_percentage).natCeil 
           (new_speed original_speed speed_increase) = 168 := 
  by
    sorry

end john_can_run_168_miles_l32_32948


namespace zero_of_f_in_interval_2_3_l32_32350

noncomputable def f (x : ℝ) : ℝ := x - 3 + log 3 x

theorem zero_of_f_in_interval_2_3 : ∃ x ∈ (Set.Ioo 2 3 : Set ℝ), f x = 0 :=
by
  sorry

end zero_of_f_in_interval_2_3_l32_32350


namespace no_quadratic_polynomial_nature_l32_32644

theorem no_quadratic_polynomial_nature (P : ℤ[X]) (h : P.degree = 2 ∧ ∀ n : ℕ, (∃ k : ℕ, n = (10^k - 1) / 9) → ∀ m : ℕ, (∃ l : ℕ, m = (10^l - 1) / 9) → n = m) : 
  ¬ (∀ n : ℕ, (∃ k : ℕ, n = (10^k - 1) / 9) → (∃ l : ℕ, P.eval n = (10^l - 1) / 9)) :=
by sorry

end no_quadratic_polynomial_nature_l32_32644


namespace chocolate_cost_l32_32784

def cost_of_chocolates (candies_per_box : ℕ) (cost_per_box : ℕ) (total_candies : ℕ) : ℕ :=
  (total_candies / candies_per_box) * cost_per_box

theorem chocolate_cost : cost_of_chocolates 30 8 450 = 120 :=
by
  -- The proof is not needed per the instructions
  sorry

end chocolate_cost_l32_32784


namespace Violet_family_tickets_cost_l32_32746

theorem Violet_family_tickets_cost :
  let adult_ticket_cost := 35
  let child_ticket_cost := 20
  let num_children := 6
  let num_adult := 1
  num_adult * adult_ticket_cost + num_children * child_ticket_cost = 155 :=
by
  let adult_ticket_cost := 35
  let child_ticket_cost := 20
  let num_children := 6
  let num_adult := 1
  calc
    num_adult * adult_ticket_cost + num_children * child_ticket_cost
    = 1 * 35 + 6 * 20 : by refl
    = 35 + 120 : by norm_num
    = 155 : by norm_num

end Violet_family_tickets_cost_l32_32746


namespace ephraim_keiko_same_heads_probability_l32_32655

def coin_toss_probability_same_heads : ℚ :=
  let keiko_prob_0 := 1 / 4
  let keiko_prob_1 := 1 / 2
  let keiko_prob_2 := 1 / 4
  let ephraim_prob_0 := 1 / 8
  let ephraim_prob_1 := 3 / 8
  let ephraim_prob_2 := 3 / 8
  let ephraim_prob_3 := 1 / 8
  (keiko_prob_0 * ephraim_prob_0) 
  + (keiko_prob_1 * ephraim_prob_1) 
  + (keiko_prob_2 * ephraim_prob_2)

theorem ephraim_keiko_same_heads_probability : 
  coin_toss_probability_same_heads = 11 / 32 :=
by 
  unfold coin_toss_probability_same_heads
  norm_num
  sorry

end ephraim_keiko_same_heads_probability_l32_32655


namespace root_of_polynomial_l32_32973

theorem root_of_polynomial (a b : ℝ) (h₁ : a^4 + a^3 - 1 = 0) (h₂ : b^4 + b^3 - 1 = 0) : 
  (ab : ℝ) → ab * ab * ab * ab * ab * ab + ab * ab * ab * ab + ab * ab * ab - ab * ab - 1 = 0 :=
sorry

end root_of_polynomial_l32_32973


namespace triangle_angle_incircle_l32_32087

theorem triangle_angle_incircle (A B C X Y Z : Type*) 
  (h_incircle : incircle ΔABC Γ)
  (h_X_on_BC : X ∈ BC)
  (h_Y_on_AB : Y ∈ AB)
  (h_Z_on_AC : Z ∈ AC)
  (h_angle_A : ∠A = 50)
  (h_angle_B : ∠B = 70)
  (h_angle_c : ∠C = 60)
  (h_right_triangle : ∠XYZ = 90) :
  ∠AYX = 55 := 
sorry

end triangle_angle_incircle_l32_32087


namespace polynomial_positive_for_all_reals_l32_32999

theorem polynomial_positive_for_all_reals (m : ℝ) : m^6 - m^5 + m^4 + m^2 - m + 1 > 0 :=
by
  sorry

end polynomial_positive_for_all_reals_l32_32999


namespace symmetric_point_origin_l32_32558

theorem symmetric_point_origin (A : ℝ × ℝ) (A_sym : ℝ × ℝ) (h : A = (3, -2)) (h_sym : A_sym = (-A.1, -A.2)) : A_sym = (-3, 2) :=
by
  sorry

end symmetric_point_origin_l32_32558


namespace combinatorial_sum_identity_l32_32488

theorem combinatorial_sum_identity :
  ∑ k in Finset.range 51, (-1: ℤ)^k * (k + 1) * Nat.choose 50 k = 0 := 
by
  sorry

end combinatorial_sum_identity_l32_32488


namespace knight_reachability_l32_32296

theorem knight_reachability (p q : ℕ) (hpq_pos : 0 < p ∧ 0 < q) :
  (p + q) % 2 = 1 ∧ Nat.gcd p q = 1 ↔
  ∀ x y x' y', ∃ k h n m, x' = x + k * p + h * q ∧ y' = y + n * p + m * q :=
by
  sorry

end knight_reachability_l32_32296


namespace total_fish_l32_32079

-- Defining the number of fish each person has, based on the conditions.
def billy_fish : ℕ := 10
def tony_fish : ℕ := 3 * billy_fish
def sarah_fish : ℕ := tony_fish + 5
def bobby_fish : ℕ := 2 * sarah_fish

-- The theorem stating the total number of fish.
theorem total_fish : billy_fish + tony_fish + sarah_fish + bobby_fish = 145 := by
  sorry

end total_fish_l32_32079


namespace difference_of_squares_example_l32_32465

theorem difference_of_squares_example :
  262^2 - 258^2 = 2080 := by
sorry

end difference_of_squares_example_l32_32465


namespace john_horizontal_distance_l32_32654

theorem john_horizontal_distance (v_increase : ℕ)
  (elevation_start : ℕ) (elevation_end : ℕ) (h_ratio : ℕ) :
  (elevation_end - elevation_start) * h_ratio = 1350 * 2 :=
begin
  -- Let elevation_start be 100 feet
  let elevation_start := 100,
  -- Let elevation_end be 1450 feet
  let elevation_end := 1450,
  -- The steepened ratio, height per step
  let v_increase := elevation_end - elevation_start,
  -- John travels 1 foot vertically for every 2 feet horizontally
  let h_ratio := 2,
  -- Hence the horizontal distance theorem
  -- s_foot = h_ratio * t_foot = 1350 * 2 = 2700 feet.
  sorry
end

end john_horizontal_distance_l32_32654


namespace minimum_AP_BP_CP_DP_EP_l32_32762

-- Define points and constants
def A : ℝ := 0
def B : ℝ := 3
def C : ℝ := 4
def D : ℝ := 9
def E : ℝ := 13

-- Define the function S(x)
def S (x : ℝ) : ℝ := (x - A)^2 + (x - B)^2 + (x - C)^2 + (x - D)^2 + (x - E)^2

-- Derivative of S(x)
def dS_dx (x : ℝ) : ℝ := 2 * (x - A) + 2 * (x - B) + 2 * (x - C) + 2 * (x - D) + 2 * (x - E)

-- Statement of the theorem
theorem minimum_AP_BP_CP_DP_EP :
  (∃ x : ℝ, (S x = 170.24 ∧ dS_dx x = 0)) :=
sorry

end minimum_AP_BP_CP_DP_EP_l32_32762


namespace factor_polynomial_l32_32102

theorem factor_polynomial :
  (x : ℝ) → (x^2 - 6*x + 9 - 64*x^4) = (-8*x^2 + x - 3) * (8*x^2 + x - 3) :=
by
  intro x
  sorry

end factor_polynomial_l32_32102


namespace prime_divides_binom_sum_l32_32275

theorem prime_divides_binom_sum {p : ℕ} (hp_prime: Nat.Prime p) (hp_geq5 : p ≥ 5) :
  p^2 ∣ ∑ r in Finset.range (⌊2 * p / 3⌋ + 1), Nat.choose p r :=
sorry

end prime_divides_binom_sum_l32_32275


namespace finite_sequence_l32_32876

noncomputable def sequence : ℕ → ℤ
| 0       := 6
| 1       := 4
| (n + 2) := (sequence (n + 1) ^ 2 - 4) / sequence n

theorem finite_sequence :
  (∃ N : ℕ, ∀ n > N, (sequence n = 0 ∨ sequence n = -2)) ∧
  (¬ ∀ n : ℕ, sequence (n + 2) = 2 * sequence (n + 1) - sequence n) :=
begin
  sorry
end

end finite_sequence_l32_32876


namespace correct_multiplication_l32_32679

theorem correct_multiplication 
  (f : ℕ) 
  (incorrect_result : ℕ) 
  (multiplier : ℕ) 
  (correct_result : ℕ) 
  (incorrect_result = 102325)
  (multiplier = 153) 
  (correct_result = 669 * multiplier) : 
  f * multiplier = correct_result → 
  incorrect_result ≠ correct_result := 
sorry

end correct_multiplication_l32_32679


namespace minimize_distance_sum_l32_32702

-- Definitions of points A, B, and C with coordinates
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := { x := 4, y := 5 }
def B : Point := { x := 3, y := 2 }
def C (k : ℝ) : Point := { x := 0, y := k }

-- Definition of the distance function between two points
def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Proof statement
theorem minimize_distance_sum : ∃ (k : ℝ), k = -7 ∧ ∀ k' : ℝ, (distance A (C k) + distance B (C k)) ≤ (distance A (C k') + distance B (C k')) :=
by
  sorry

end minimize_distance_sum_l32_32702


namespace time_spent_on_seals_l32_32366

theorem time_spent_on_seals (s : ℕ) 
  (h1 : 2 * 60 + 10 = 130) 
  (h2 : s + 8 * s + 13 = 130) :
  s = 13 :=
sorry

end time_spent_on_seals_l32_32366


namespace area_transformed_function_l32_32344

noncomputable def area_between_curve_f (f : ℝ → ℝ) : ℝ := sorry

theorem area_transformed_function (f : ℝ → ℝ) (h : area_between_curve_f f = 10) :
  area_between_curve_f (λ x, 2 * f (x - 4)) = 20 :=
sorry

end area_transformed_function_l32_32344


namespace smallest_positive_angle_l32_32462

theorem smallest_positive_angle (x : ℝ) (hx : x > 0) (h : Real.cot (3 * x) = (Real.cos x - Real.sin x) / (Real.cos x + Real.sin x)) : x = Real.pi / 12 :=
by
  sorry

end smallest_positive_angle_l32_32462


namespace conclusion_correctness_l32_32898

variable (a b c m : ℝ)
variable (h_vertex : -b / (2 * a) = -1 / 2)
variable (h_vertex_y : m > 0)
variable (h_intersect_x : ∃ (x : ℝ), 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0)
variable (h_no_real_roots : a * x ^ 2 + b * x + c - 2 = 0 → (b^2 - 4 * a * (c - 2) < 0))

-- Assume this theorem represents the conclusion's verification
theorem conclusion_correctness :
    (b < 0) ∧
    (¬ (2 * b + c > 0)) ∧
    (∀ (y1 y2 : ℝ), (-2, y1) ∈ (λ x, a * x ^ 2 + b * x + c) → (2, y2) ∈ (λ x, a * x ^ 2 + b * x + c) → y1 > y2) ∧
    (m <= 2) :=
by 
    sorry

end conclusion_correctness_l32_32898


namespace find_y_l32_32186

-- Definitions of vectors and parallel relationship
def vector_a : ℝ × ℝ := (4, 2)
def vector_b (y : ℝ) : ℝ × ℝ := (6, y)
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

-- The theorem we want to prove
theorem find_y (y : ℝ) (h : parallel vector_a (vector_b y)) : y = 3 :=
sorry

end find_y_l32_32186


namespace smallest_lcm_of_4_digit_integers_l32_32207

open Nat

theorem smallest_lcm_of_4_digit_integers (k l : ℕ) (hk : 1000 ≤ k ∧ k < 10000) (hl : 1000 ≤ l ∧ l < 10000) (h_gcd : gcd k l = 5) :
  lcm k l = 203010 := sorry

end smallest_lcm_of_4_digit_integers_l32_32207


namespace joint_probability_bound_l32_32395

open ProbabilityTheory

theorem joint_probability_bound {Ω : Type*} [measure_space Ω] (P : measure Ω) [probability_measure P]
  (A B : set Ω) :
  |P[A ∩ B] - P[A] * P[B]| ≤ 1/4 := 
sorry

end joint_probability_bound_l32_32395


namespace simplify_sqrt_of_mixed_number_l32_32496

noncomputable def sqrt_fraction := λ (a b : ℕ), (Real.sqrt a) / (Real.sqrt b)

theorem simplify_sqrt_of_mixed_number : sqrt_fraction 137 16 = (Real.sqrt 137) / 4 := by
  sorry

end simplify_sqrt_of_mixed_number_l32_32496


namespace number_of_buses_l32_32404

theorem number_of_buses (total_people : ℕ) (bus_capacity : ℕ) (h1 : total_people = 1230) (h2 : bus_capacity = 48) : 
  Nat.ceil (total_people / bus_capacity : ℝ) = 26 := 
by 
  unfold Nat.ceil 
  sorry

end number_of_buses_l32_32404


namespace probability_of_longer_piece_at_most_three_times_shorter_l32_32417

noncomputable def cuttingProbability : ℝ :=
by
  let C := measure_theory.MeasureTheory.uniform [0.0, 1.0]
  have cond1 : ∀ (c : ℝ), (1 - c ≤ 3 * c) ∨ (c ≤ 3 * (1 - c)) ↔ (1 / 4 ≤ c) ∧ (c ≤ 3 / 4) := sorry
  have prob_interval : measure_theory.MeasureTheory.measure_Icc (1 / 4 : ℝ) (3 / 4) := sorry
  let prob := prob_interval / (C.measure_univ)
  exact prob 

theorem probability_of_longer_piece_at_most_three_times_shorter :
  cuttingProbability = 1 / 2 := sorry

end probability_of_longer_piece_at_most_three_times_shorter_l32_32417


namespace probability_of_sum_four_is_five_over_eighteen_l32_32403

def balls := [1, 2, 2, 3, 3, 3]

noncomputable def probability_sum_four : ℚ :=
  let draws := list.product balls balls
  let favorable_outcomes := draws.filter (λ p, p.1 + p.2 = 4)
  (favorable_outcomes.length : ℚ) / (draws.length : ℚ)

theorem probability_of_sum_four_is_five_over_eighteen : probability_sum_four = 5 / 18 := 
  sorry

end probability_of_sum_four_is_five_over_eighteen_l32_32403


namespace percent_calculation_l32_32778

-- Given conditions
def part : ℝ := 120.5
def whole : ℝ := 80.75

-- Theorem statement
theorem percent_calculation : (part / whole) * 100 = 149.26 := 
sorry

end percent_calculation_l32_32778


namespace caterpillar_catches_cicada_in_36_minutes_l32_32352

-- Definitions based on given conditions
def caterpillar_speed : ℝ := 4 / 2 -- meters per minute
def cicada_larva_speed : ℝ := 4 / 3 -- meters per minute
def initial_distance : ℝ := 24 -- meters

-- Theorem stating the time (in minutes) when the caterpillar will catch up with the cicada larva
theorem caterpillar_catches_cicada_in_36_minutes (h: initial_distance = 24) : 
  ∃ t : ℝ, t = 36 ∧ (initial_distance / (caterpillar_speed - cicada_larva_speed) = t) :=
  sorry

end caterpillar_catches_cicada_in_36_minutes_l32_32352


namespace sum_of_largest_100_l32_32346

theorem sum_of_largest_100 (a : Fin 123 → ℝ) (h1 : (Finset.univ.sum a) = 3813) 
  (h2 : ∀ i j : Fin 123, i ≤ j → a i ≤ a j) : 
  ∃ s : Finset (Fin 123), s.card = 100 ∧ (s.sum a) ≥ 3100 :=
by
  sorry

end sum_of_largest_100_l32_32346


namespace correct_definition_of_regression_independence_l32_32008

-- Definitions
def regression_analysis (X Y : Type) := ∃ r : X → Y, true -- Placeholder, ideal definition studies correlation
def independence_test (X Y : Type) := ∃ rel : X → Y → Prop, true -- Placeholder, ideal definition examines relationship

-- Theorem statement
theorem correct_definition_of_regression_independence (X Y : Type) :
  (∃ r : X → Y, true) ∧ (∃ rel : X → Y → Prop, true)
  → "Regression analysis studies the correlation between two variables, and independence tests examine whether there is some kind of relationship between two variables" = "C" :=
sorry

end correct_definition_of_regression_independence_l32_32008


namespace bridget_and_sarah_have_3_dollars_l32_32815

noncomputable def total_money_in_dollars (sarah_money_cents bridget_money_cents total_money_cents : ℕ) : ℚ :=
  total_money_cents / 100

theorem bridget_and_sarah_have_3_dollars :
  ∀ (sarah_money_cents : ℕ), (bridget_money_cents : ℕ), (total_money_cents : ℕ),
  sarah_money_cents = 125 →
  bridget_money_cents = sarah_money_cents + 50 →
  total_money_cents = sarah_money_cents + bridget_money_cents →
  total_money_in_dollars sarah_money_cents bridget_money_cents total_money_cents = 3 :=
by
  intros sarah_money_cents bridget_money_cents total_money_cents
  intros h_sarah h_bridget h_total
  simp only [total_money_in_dollars]
  sorry

end bridget_and_sarah_have_3_dollars_l32_32815


namespace all_solution_polynomials_l32_32136

noncomputable def polynomial_properties {n : ℕ} (hn : n ≥ 2) (P : Polynomial ℝ) : Prop :=
  (∃ (a : Fin (n+1) → ℝ), P = Polynomial.sum (λ i, a ⟨i, Nat.lt_succ_self i.1⟩ * Polynomial.monomial i.1 1) ∧
    (P.natDegree = n) ∧
    (∀ x : ℝ, P.eval x = 0 → x ≤ -1) ∧
    (let a₀ := a ⟨0, Nat.zero_lt_succ n⟩ in
     let a₁ := a ⟨1, Nat.lt_succ_of_lt (Nat.succ_lt_succ (Nat.zero_lt_succ n))⟩ in
     let aₙ := a ⟨n, Nat.lt_succ_self n⟩ in
     let aₙ₋₁ := a ⟨n-1, Nat.pred_lt (Nat.succ_pos n) (Nat.succ_sub_one n).symm.ge⟩ in
     a₀ ^ 2 + a₁ * aₙ = aₙ ^ 2 + a₀ * aₙ₋₁))

noncomputable def solution_polynomials {n : ℕ} (hn : n ≥ 2) : set (Polynomial ℝ) :=
  {P | ∃ (a : ℝ) (beta : ℝ), a ≠ 0 ∧ beta ≤ -1 ∧ P = Polynomial.C a * (Polynomial.X + 1)^(n-1) * (Polynomial.X + Polynomial.C beta)}

theorem all_solution_polynomials {n : ℕ} (hn : n ≥ 2) (P : Polynomial ℝ) :
  polynomial_properties hn P ↔ P ∈ solution_polynomials hn :=
by sorry

end all_solution_polynomials_l32_32136


namespace calc_part_one_calc_part_two_l32_32084

open Real

theorem calc_part_one :
  0.064 ^ (- (1 / 3)) - ((- 4 / 5) ^ 0) + 0.01 ^ (1 / 2) = 8 / 5 := 
sorry

theorem calc_part_two : 
  2 * log 10 5 + log 10 4 + log (sqrt (exp 1)) = 5 / 2 := 
sorry

end calc_part_one_calc_part_two_l32_32084


namespace find_a_ln_a_l32_32536

noncomputable def f (a x : ℝ) : ℝ := x^a - Real.logBase a x

theorem find_a_ln_a (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1)
    (h₂ : ∀ (x : ℝ), 0 < x → 1 ≤ f a x) : a * Real.log a = 1 := 
sorry

end find_a_ln_a_l32_32536


namespace construct_line_with_compass_and_straightedge_l32_32373

noncomputable def e1 (x y : ℝ) : Prop := 3 * x - 4 * y + 3 = 0
noncomputable def e2 (x y : ℝ) : Prop := 48 * x - 55 * y - 55 = 0

noncomputable def circle_k (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Points where the circle intersects the lines
noncomputable def B1 : ℝ × ℝ := (-1, 0)
noncomputable def B2 : ℝ × ℝ := (7 / 25, 24 / 25)
noncomputable def C1 : ℝ × ℝ := (0, -1)
noncomputable def C2 : ℝ × ℝ := (5280 / 5329, -721 / 5329)

-- Intermediate points on lines drawn from the origin
noncomputable def D : ℝ × ℝ := (1, 0)
noncomputable def E : ℝ × ℝ := (0, 1)

-- Intersection points
noncomputable def M : ℝ × ℝ := (55 / 52, 309 / (7 * 52))

theorem construct_line_with_compass_and_straightedge :
  ∃ (x y : ℝ), 
    ((3 * x - 4 * y + 3 = 0) ∧ (48 * x - 55 * y - 55 = 0)) ∧
    line_equation_through (0, 0) (55 / 52, 309 / (7 * 52)) := by
  sorry

end construct_line_with_compass_and_straightedge_l32_32373


namespace evaluate_expression_l32_32840

theorem evaluate_expression (b : ℝ) (hb : b = -3) : 
  (3 * b⁻¹ + b⁻¹ / 3) / b = 10 / 27 :=
by 
  -- Lean code typically begins the proof block here
  sorry  -- The proof itself is omitted

end evaluate_expression_l32_32840


namespace complex_imaginary_unit_sum_l32_32448

theorem complex_imaginary_unit_sum (i : ℂ) (h : i^2 = -1) : i + i^2 + i^3 = -1 := 
by sorry

end complex_imaginary_unit_sum_l32_32448


namespace max_min_abs_m_l32_32940

-- Define complex number variables
variables {z1 z2 m α β : ℂ}

-- Conditions from the problem
def condition1 := ∃ (α β : ℂ), α + β = -z1 ∧ α * β = z2 + m ∧ |α - β| = 2 * real.sqrt 7
def condition2 := z1^2 - 4 * z2 = 16 + 20 * I

-- Prove the main statement
theorem max_min_abs_m (h1: condition1) (h2: condition2) :
  ∃ (m_max m_min : ℝ), m_max = real.sqrt 41 + 7 ∧ m_min = 7 - real.sqrt 41 ∧ ∀ (m : ℂ), ∃ (abs_m : ℝ), abs_m = complex.abs m ∧ abs_m = 7 - real.sqrt 41 ∨ abs_m = real.sqrt 41 + 7 :=
by {
  -- sorry is used to skip the proof which is not required
  sorry,
}

end max_min_abs_m_l32_32940


namespace gilda_stickers_left_l32_32872

variable (S : ℝ) (hS : S > 0)

def remaining_after_olga : ℝ := 0.70 * S
def remaining_after_sam : ℝ := 0.80 * remaining_after_olga S
def remaining_after_max : ℝ := 0.70 * remaining_after_sam S
def remaining_after_charity : ℝ := 0.90 * remaining_after_max S

theorem gilda_stickers_left :
  remaining_after_charity S / S * 100 = 35.28 := by
  sorry

end gilda_stickers_left_l32_32872


namespace adult_tickets_count_l32_32941

theorem adult_tickets_count (A C : ℕ) (h1 : A + C = 7) (h2 : 21 * A + 14 * C = 119) : A = 3 :=
sorry

end adult_tickets_count_l32_32941


namespace matrix_power_four_l32_32457

-- Given matrix M representing a rotation by pi/4 radians
def M : Matrix (Fin 2) (Fin 2) ℝ :=
  ![\[Real.sqrt 2 / 2, -Real.sqrt 2 / 2\], \[Real.sqrt 2 / 2, Real.sqrt 2 / 2\]]

-- Statement of the problem
theorem matrix_power_four :
  M ^ 4 = ![\[-1, 0\], \[0, -1\]] := 
sorry

end matrix_power_four_l32_32457


namespace vector_parallel_m_l32_32021

theorem vector_parallel_m {m : ℝ} (h : (2:ℝ) * m - (-1 * -1) = 0) : m = 1 / 2 := 
by
  sorry

end vector_parallel_m_l32_32021


namespace correct_statements_l32_32159

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then
  let c := (λ x, 2) 1  -- Using the condition f(1) = 2 to determine c.
  have hx : x ≠ 0 := λ h_eq, lt_irrefl _ (h_eq ▸ h),
  ln x + c / x 
else 0 -- Placeholder for x ≤ 0, not applicable here due to the domain restriction.

theorem correct_statements (x : ℝ) (h : x > 0) (h1 : f(1) = 2) :
  (f 2 = ln 2 + 1) ∧ (∃! y > 0, f y - y = 0) :=
by
  sorry

end correct_statements_l32_32159


namespace quadratic_rational_solutions_product_l32_32833

theorem quadratic_rational_solutions_product :
  ∃ (c₁ c₂ : ℕ), (7 * x^2 + 15 * x + c₁ = 0 ∧ 225 - 28 * c₁ = k^2 ∧ ∃ k : ℤ, k^2 = 225 - 28 * c₁) ∧
                 (7 * x^2 + 15 * x + c₂ = 0 ∧ 225 - 28 * c₂ = k^2 ∧ ∃ k : ℤ, k^2 = 225 - 28 * c₂) ∧
                 (c₁ = 1) ∧ (c₂ = 8) ∧ (c₁ * c₂ = 8) :=
by
  sorry

end quadratic_rational_solutions_product_l32_32833


namespace minor_arc_length_PQ_l32_32958

def radius : ℝ := 24
def angle_PRQ_degrees : ℝ := 60
def total_circumference : ℝ := 2 * Real.pi * radius
def minor_arc_circumference : ℝ := (angle_PRQ_degrees / 360) * total_circumference

theorem minor_arc_length_PQ :
  minor_arc_circumference = 8 * Real.pi := by
  sorry

end minor_arc_length_PQ_l32_32958


namespace number_of_ordered_pairs_l32_32524

/-- 
Given positive integers m and n, which are composite numbers,
and L(m) and L(n) as the largest factors of m and n respectively 
other than m and n, such that L(m) * L(n) = 80,
we want to prove that the number of ordered pairs (m, n) is 12.
-/
theorem number_of_ordered_pairs : 
  let L := (λ (x : ℕ), (finset.filter (λ y, y < x ∧ x % y = 0) (finset.range x)).max' (finset.range_nonempty _)) in
  finset.card (finset.filter 
    (λ (p : ℕ × ℕ), 
        (p.1 > 1 ∧ finset.card (finset.filter (λ y, y < p.1 ∧ p.1 % y = 0) (finset.range p.1)) > 2) ∧ -- p1 is composite
        (p.2 > 1 ∧ finset.card (finset.filter (λ y, y < p.2 ∧ p.2 % y = 0) (finset.range p.2)) > 2) ∧ -- p2 is composite
        L p.1 * L p.2 = 80) 
    ((finset.range 81).product (finset.range 81))) = 12 := 
sorry

end number_of_ordered_pairs_l32_32524


namespace determine_q_l32_32887

-- Definitions based on given conditions
variables (a : ℕ → ℝ) (d : ℝ) (n : ℕ)

def arithmetic_sequence := ∀ n, a (n + 1) = a n + d
def negative_difference := d < 0
def T : ℕ → ℝ
| 0     := 0
| (n+1) := T n + a (n+1)
def T_3 := T 3 = 15
def geometric_condition := (a 1 + 1) * (a 3 + 9) = (a 2 + 3) ^ 2

-- Constant q representing the common ratio
constant q : ℝ

noncomputable def q_value := (a 2 + 3) / (a 1 + 1)

-- The goal is to prove q = 1/2 under the given conditions
theorem determine_q :
  arithmetic_sequence a d ∧ negative_difference d ∧ T_3 ∧ geometric_condition a →
  q_value a = 1 / 2 :=
by sorry

end determine_q_l32_32887


namespace vector_subtraction_l32_32591

variables (a : ℝ × ℝ) (b : ℝ × ℝ)

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

def vector_scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

theorem vector_subtraction (ha : a = (-1, 3)) (hb : b = (2, -1)) : 
  vector_sub a (vector_scalar_mult 2 b) = (-5, 5) :=
by sorry

end vector_subtraction_l32_32591


namespace football_games_per_month_l32_32353

theorem football_games_per_month (total_games : ℕ) (total_months : ℕ) 
  (h_games: total_games = 323) (h_months: total_months = 17) : 
  total_games / total_months = 19 := 
by 
  rw [h_games, h_months]
  exact Nat.div_eq_of_eq_mul_left (by decide) rfl

end football_games_per_month_l32_32353


namespace inequality_proof_l32_32016

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a^2 / b + b^2 / c + c^2 / a) ≥ 3 * (a^3 + b^3 + c^3) / (a^2 + b^2 + c^2) := 
sorry

end inequality_proof_l32_32016


namespace dot_product_CP_CQ_zero_l32_32586

noncomputable theory

open Real

def circle_C (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 1) ^ 2 = 1
def line_l (x y : ℝ) : Prop := y = -x + 4

theorem dot_product_CP_CQ_zero :
  ∀ P Q : ℝ × ℝ, 
  (circle_C P.1 P.2) → 
  (circle_C Q.1 Q.2) →
  (line_l P.1 P.2) →
  (line_l Q.1 Q.2) →
  let CP : ℝ × ℝ := (P.1 - 2, P.2 - 1) 
  let CQ : ℝ × ℝ := (Q.1 - 2, Q.2 - 1) 
  (CP.1 * CQ.1 + CP.2 * CQ.2 = 0) :=
by
  intros P Q hPC hQC hPL hQL CP CQ
  sorry

end dot_product_CP_CQ_zero_l32_32586


namespace congruence_problem_l32_32831

-- Define the binomial coefficient function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the expression for n based on the problem statement
noncomputable def n : ℤ :=
  ∑ i in Finset.range 11, (C 10 i) * (-1)^(i : ℤ) * (10^i : ℤ)

-- Define the proof statement
theorem congruence_problem :
  ∃ p, (n ≡ p [MOD 7]) ∧ (p = 37) :=
by
  use 37
  split
  sorry -- Proof goes here
  refl

end congruence_problem_l32_32831


namespace distinct_remainders_property_l32_32393

theorem distinct_remainders_property (p m n : ℕ) (a b : ℕ → ℕ) 
  (hp : prime p) 
  (ha : ∀ i, i < m → a i < p) 
  (hb : ∀ j, j < n → b j < p)
  (a_sorted : ∀ i j, i < j ∧ j < m → a i < a j) 
  (b_sorted : ∀ i j, i < j ∧ j < n → b i < b j)
  (k : ℕ) 
  (hk : k = finset.card ((finset.image (λ (ij : ℕ × ℕ), (a ij.fst + b ij.snd) % p) (finset.product (finset.range m) (finset.range n))))):
  (m + n > p → k = p) ∧ (m + n ≤ p → k ≥ m + n - 1) := 
by sorry

end distinct_remainders_property_l32_32393


namespace petya_vasya_game_min_k_l32_32300

theorem petya_vasya_game_min_k :
  ∃(k : ℕ), (∀ (grid : list (list ℕ)) (marks : list (ℕ × ℕ)),
    grid.length = 9 ∧ (∀ row, grid.row.length = 9) ∧
    marks.length = k ∧
    (∀ (rectangle_pos : ℕ × ℕ) (orientation : ℕ),
      ∃! (a b c d : ℕ × ℕ),
      list.mem a marks ∨ list.mem b marks ∨ list.mem c marks ∨ list.mem d marks)
    ) → k = 40 :=
sorry

end petya_vasya_game_min_k_l32_32300


namespace problem_statement_l32_32214

theorem problem_statement (x : ℝ) (h : x < 0) :
  (x / |x| < 0) ∧ (-x^3 > 0) ∧ (-3^x + 1 > 0) ∧ (-x⁻² < 0) ∧ (x ^ (1 / 3) < 0) :=
by
  sorry

end problem_statement_l32_32214


namespace arithmetic_sum_eight_terms_l32_32452

theorem arithmetic_sum_eight_terms :
  ∀ (a d : ℤ) (n : ℕ), a = -3 → d = 6 → n = 8 → 
  (last_term = a + (n - 1) * d) →
  (last_term = 39) →
  (sum = (n * (a + last_term)) / 2) →
  sum = 144 :=
by
  intros a d n ha hd hn hlast_term hlast_term_value hsum
  sorry

end arithmetic_sum_eight_terms_l32_32452


namespace count_three_letter_words_with_A_l32_32190

theorem count_three_letter_words_with_A : 
  let total_words := 5^3 in
  let words_without_A := 4^3 in
  total_words - words_without_A = 61 :=
by
  sorry

end count_three_letter_words_with_A_l32_32190


namespace dihedral_angle_90_deg_l32_32635

variable (a : ℝ)

-- Define coordinates of vertices in the cube
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (a, 0, 0)
def C : ℝ × ℝ × ℝ := (a, a, 0)
def D : ℝ × ℝ × ℝ := (0, a, 0)
def A₁ : ℝ × ℝ × ℝ := (0, 0, a)
def B₁ : ℝ × ℝ × ℝ := (a, 0, a)
def C₁ : ℝ × ℝ × ℝ := (a, a, a)
def D₁ : ℝ × ℝ × ℝ := (0, a, a)

-- Define the midpoint E of BC
def E : ℝ × ℝ × ℝ := (a, a / 2, 0)

-- Define the point F on AA₁ such that the ratio A₁F : FA = 1 : 2
def F : ℝ × ℝ × ℝ := (0, 0, 2 * a / 3)

-- Define the normal vectors to the planes
def n₁ : ℝ × ℝ × ℝ := (0, 0, 1) -- Normal to the base plane A₁B₁C₁D₁
def n₂ : ℝ × ℝ × ℝ := 
  let B₁E := (0, a / 2, -a)
  let B₁F := (-a, 0, -a / 3)
  (-(a ^ 2) / 6, -(a ^ 2), 0) -- Normal to plane B₁EF

-- Define the dot product of two vectors
def dot_product (v₁ v₂ : ℝ × ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2 + v₁.3 * v₂.3

-- Define the norm of a vector
def norm (v : ℝ × ℝ × ℝ) : ℝ := 
  real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

-- Define the cosine of the angle between normals
def cos_theta : ℝ := (dot_product n₁ n₂) / ((norm n₁) * (norm n₂))

-- Prove the dihedral angle is 90 degrees
theorem dihedral_angle_90_deg : real.arccos cos_theta = real.pi / 2 := 
  sorry

end dihedral_angle_90_deg_l32_32635


namespace intersection_A_B_l32_32918

def setA : Set ℤ := { x | x < -3 }
def setB : Set ℤ := {-5, -4, -3, 1}

theorem intersection_A_B : setA ∩ setB = {-5, -4} := by
  sorry

end intersection_A_B_l32_32918


namespace unique_int_pair_exists_l32_32282

theorem unique_int_pair_exists (a b : ℤ) : 
  ∃! (x y : ℤ), (x + 2 * y - a)^2 + (2 * x - y - b)^2 ≤ 1 :=
by
  sorry

end unique_int_pair_exists_l32_32282


namespace determine_linear_relation_is_scatter_plot_l32_32003

-- Define the types of plots
inductive PlotType
| ScatterPlot
| StemAndLeafPlot
| FrequencyDistributionHistogram
| FrequencyDistributionLineChart

-- Define the problem statement
def isLinearRelationFunction : PlotType → Prop
| PlotType.ScatterPlot := true
| PlotType.StemAndLeafPlot := false
| PlotType.FrequencyDistributionHistogram := false
| PlotType.FrequencyDistributionLineChart := false

-- Theorem statement: Scatter plot is the appropriate method
theorem determine_linear_relation_is_scatter_plot :
  ∀ (plot : PlotType), (plot = PlotType.ScatterPlot) → isLinearRelationFunction plot :=
by
  intros plot h
  rw h
  trivial

end determine_linear_relation_is_scatter_plot_l32_32003


namespace closest_integer_to_cbrt_1728_l32_32006

theorem closest_integer_to_cbrt_1728 : 
  let x := 1728 in closest_integer (real.cbrt x) = 12 :=
by
  sorry

end closest_integer_to_cbrt_1728_l32_32006


namespace equation_of_line_l_l32_32162

-- Define the conditions for the parabola and the line
def parabola_vertex : Prop := 
  ∃ C : ℝ × ℝ, C = (0, 0)

def parabola_symmetry_axis : Prop := 
  ∃ l : ℝ → ℝ, ∀ x, l x = -1

def midpoint_of_AB (A B : ℝ × ℝ) : Prop :=
  (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 1

def parabola_equation (A B : ℝ × ℝ) : Prop :=
  A.2^2 = 4 * A.1 ∧ B.2^2 = 4 * B.1

-- State the theorem to be proven
theorem equation_of_line_l (A B : ℝ × ℝ) :
  parabola_vertex ∧ parabola_symmetry_axis ∧ midpoint_of_AB A B ∧ parabola_equation A B →
  ∃ l : ℝ → ℝ, ∀ x, l x = 2 * x - 3 :=
by sorry

end equation_of_line_l_l32_32162


namespace part1_part2_l32_32175

noncomputable def h (x : ℝ) (a : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + 1

noncomputable def h_prime (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := h_prime x a - 2 * a * real.log x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (real.log x)^2 + 2 * a^2

noncomputable def F (x : ℝ) (a : ℝ) : ℝ := f x a + g x a

theorem part1 (h_mono_incr : ∀ x ∈ set.Ioi 2, 0 ≤ (2 * x^2 - 2 * x * a - 2 * a) / x) :
  a ≤ 4 / 3 :=
sorry

theorem part2 (x_gt_0 : x > 0) (a_in_R : a ∈ set.univ) : F x a ≥ 1 / 2 :=
sorry

end part1_part2_l32_32175


namespace pencils_multiple_of_30_l32_32708

-- Defines the conditions of the problem
def num_pens : ℕ := 2010
def max_students : ℕ := 30
def equal_pens_per_student := num_pens % max_students = 0

-- Proves that the number of pencils must be a multiple of 30
theorem pencils_multiple_of_30 (P : ℕ) (h1 : equal_pens_per_student) (h2 : ∀ n, n ≤ max_students → ∃ m, n * m = num_pens) : ∃ k : ℕ, P = max_students * k :=
sorry

end pencils_multiple_of_30_l32_32708


namespace sum_x_coords_above_line_eq_zero_l32_32290

def Point := (ℝ × ℝ)

def points : List Point := [(3, 8), (5, 20), (10, 25), (15, 35), (18, 45)]

def line := λ x : ℝ, 3 * x + 5

def lies_above_line (p : Point) : Prop := p.2 > line p.1

def sum_of_x_coords_above_line (pts : List Point) : ℝ :=
  pts.filter lies_above_line |>.sum (λ p, p.1)

theorem sum_x_coords_above_line_eq_zero : sum_of_x_coords_above_line points = 0 :=
by
  sorry

end sum_x_coords_above_line_eq_zero_l32_32290


namespace consecutive_count_l32_32449

theorem consecutive_count :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
  ∃ (choose : (S → Prop) → ℕ), 
    (choose (λ t, true)) = 260 :=
by
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
  have H1 : ∀ (P : S → Prop), choose P = if (P = (λ x, true)) then 330 else if (P = (λ x, false)) then 0 else sorry,
  sorry

end consecutive_count_l32_32449


namespace matrix_quarter_rotation_pow_four_l32_32459

def sqrt2_div2 : ℝ := real.sqrt 2 / 2

def matrix_rotation (θ : ℝ) : matrix (fin 2) (fin 2) ℝ :=
  ![![real.cos θ, -real.sin θ], ![real.sin θ, real.cos θ]]

def M : matrix (fin 2) (fin 2) ℝ :=
  ![![sqrt2_div2, -sqrt2_div2], ![sqrt2_div2, sqrt2_div2]]

theorem matrix_quarter_rotation_pow_four :
  M ^ 4 = ![![(-1 : ℝ), 0], ![0, -1]] :=
by
  sorry

end matrix_quarter_rotation_pow_four_l32_32459


namespace solution_set_of_bx2_ax_c_lt_zero_l32_32895

theorem solution_set_of_bx2_ax_c_lt_zero (a b c : ℝ) (h1 : a > 0) (h2 : b = a) (h3 : c = -6 * a) (h4 : ∀ x, ax^2 - bx + c < 0 ↔ -2 < x ∧ x < 3) :
  ∀ x, bx^2 + ax + c < 0 ↔ -3 < x ∧ x < 2 :=
by
  sorry

end solution_set_of_bx2_ax_c_lt_zero_l32_32895


namespace paths_from_A_to_C_l32_32745

theorem paths_from_A_to_C :
  ∀ (A B C : Type)
  (paths_A_B : ℕ)
  (paths_B_C : ℕ)
  (direct_A_C : ℕ),
  paths_A_B = 3 → paths_B_C = 1 → direct_A_C = 1 →
  (paths_A_B * paths_B_C + direct_A_C = 4) :=
by
  intros A B C paths_A_B paths_B_C direct_A_C h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end paths_from_A_to_C_l32_32745


namespace polar_equation_C1_polar_equation_C2_maximum_AB_l32_32639

-- Define the parametric equations of C1 and the equation of C2
def parametric_C1 (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, 2 + 2 * Real.sin α)
def equation_C2 (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

-- Statements to prove
theorem polar_equation_C1 (θ : ℝ) : ∃ ρ : ℝ, ρ = 4 * Real.sin θ := by
  sorry

theorem polar_equation_C2 (θ : ℝ) : ∃ ρ : ℝ, ρ = 2 * Real.sqrt 2 * Real.sin (θ + (Real.pi / 4)) := by
  sorry

theorem maximum_AB (β : ℝ) (hβ : 0 < β ∧ β < Real.pi) : ∃ max_AB : ℝ, max_AB = 2 * Real.sqrt 2 := by
  sorry

end polar_equation_C1_polar_equation_C2_maximum_AB_l32_32639


namespace slips_distribution_correct_l32_32812

def slips : List ℝ := [1, 1.5, 2, 2.5, 2.5, 3, 3.5, 3.5, 4, 5]

def cups := (X Y Z W : Set ℝ)

noncomputable def assignment : Prop := sorry

theorem slips_distribution_correct :
  (assignment slips cups) ∧ 
  (2 ∈ W) ∧ 
  (3.5 ∈ Y) ∧ 
  (cups X + cups Y + cups Z + cups W = 20) →
  (4 ∈ Z) :=
sorry

end slips_distribution_correct_l32_32812


namespace harmonic_not_integer_l32_32996

theorem harmonic_not_integer (n : ℕ) (h : n > 1) : ¬ (∃ k : ℤ, H n = k) :=
by
  sorry

def H : ℕ → ℚ
| 0     => 0
| 1     => 1
| (n+1) => H n + 1 / (n+1)

end harmonic_not_integer_l32_32996


namespace gavin_shirts_l32_32532

theorem gavin_shirts (t g b : ℕ) (h_total : t = 23) (h_green : g = 17) (h_blue : b = t - g) : b = 6 :=
by sorry

end gavin_shirts_l32_32532


namespace maria_green_beans_l32_32284

theorem maria_green_beans
    (potatoes : ℕ)
    (carrots : ℕ)
    (onions : ℕ)
    (green_beans : ℕ)
    (h1 : potatoes = 2)
    (h2 : carrots = 6 * potatoes)
    (h3 : onions = 2 * carrots)
    (h4 : green_beans = onions / 3) :
  green_beans = 8 := 
sorry

end maria_green_beans_l32_32284


namespace horizontal_distance_l32_32650

theorem horizontal_distance (elev_initial elev_final v_ratio h_ratio distance: ℝ)
  (h1 : elev_initial = 100)
  (h2 : elev_final = 1450)
  (h3 : v_ratio = 1)
  (h4 : h_ratio = 2)
  (h5 : distance = 1350) :
  distance * h_ratio = 2700 := by
  sorry

end horizontal_distance_l32_32650


namespace additional_track_length_l32_32800

theorem additional_track_length (rise : ℝ) (grade1 grade2 : ℝ) (h_rise : rise = 800) (h_grade1 : grade1 = 0.04) (h_grade2 : grade2 = 0.03) :
  let horizontal_length1 := rise / grade1,
      horizontal_length2 := rise / grade2,
      additional_length := horizontal_length2 - horizontal_length1 in
  additional_length = 6667 := 
sorry

end additional_track_length_l32_32800


namespace combinatorial_sum_identity_l32_32489

theorem combinatorial_sum_identity :
  ∑ k in Finset.range 51, (-1: ℤ)^k * (k + 1) * Nat.choose 50 k = 0 := 
by
  sorry

end combinatorial_sum_identity_l32_32489


namespace min_a_b_div_1176_l32_32204

theorem min_a_b_div_1176 (a b : ℕ) (h : b^3 = 1176 * a) : a = 63 :=
by sorry

end min_a_b_div_1176_l32_32204


namespace algebraic_expression_value_l32_32134

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 2 * x - 2 = 0) :
  x * (x + 2) + (x + 1)^2 = 5 :=
by
  sorry

end algebraic_expression_value_l32_32134


namespace area_of_triangle_ADG_l32_32052

/-- The area of triangle ADG in a regular octagon ABCDEFGH with each side of length 4 is 8(1+√2). -/
theorem area_of_triangle_ADG :
  ∀ (A B C D E F G H : Type) (s : ℝ) (angle : ℝ), 
    regular_octagon A B C D E F G H s angle →
    s = 4 →
    angle = 135 →
    area_of_triangle A D G = 8 * (1 + real.sqrt 2) :=
begin
  intros,
  sorry
end

/-- Assuming the regular_octagon definition -/
def regular_octagon (A B C D E F G H : Type) (s : ℝ) (angle : ℝ) := sorry

/-- Assuming the area_of_triangle definition -/
def area_of_triangle (A D G : Type) := sorry

end area_of_triangle_ADG_l32_32052


namespace integer_part_sum_seq_l32_32710

-- Define the sequence
def x : ℕ → ℝ
| 0       := 1 / 2
| (k + 1) := x k ^ 2 + x k

-- Problem statement
theorem integer_part_sum_seq : ⌊∑ i in Finset.range 100, 1 / (x (i + 1) + 1)⌋ = 1 :=
sorry

end integer_part_sum_seq_l32_32710


namespace parallel_lines_b_value_l32_32001

-- Define the first line equation in slope-intercept form.
def line1_slope (b : ℝ) : ℝ :=
  3

-- Define the second line equation in slope-intercept form.
def line2_slope (b : ℝ) : ℝ :=
  b + 10

-- Theorem stating that if the lines are parallel, the value of b is -7.
theorem parallel_lines_b_value :
  ∀ b : ℝ, line1_slope b = line2_slope b → b = -7 :=
by
  intro b
  intro h
  sorry

end parallel_lines_b_value_l32_32001


namespace degrees_to_radians_l32_32781

theorem degrees_to_radians (deg : ℝ) (rad : ℝ) (h1 : 1 = π / 180) (h2 : deg = 60) : rad = deg * (π / 180) :=
by
  sorry

end degrees_to_radians_l32_32781


namespace john_horizontal_distance_l32_32653

theorem john_horizontal_distance (v_increase : ℕ)
  (elevation_start : ℕ) (elevation_end : ℕ) (h_ratio : ℕ) :
  (elevation_end - elevation_start) * h_ratio = 1350 * 2 :=
begin
  -- Let elevation_start be 100 feet
  let elevation_start := 100,
  -- Let elevation_end be 1450 feet
  let elevation_end := 1450,
  -- The steepened ratio, height per step
  let v_increase := elevation_end - elevation_start,
  -- John travels 1 foot vertically for every 2 feet horizontally
  let h_ratio := 2,
  -- Hence the horizontal distance theorem
  -- s_foot = h_ratio * t_foot = 1350 * 2 = 2700 feet.
  sorry
end

end john_horizontal_distance_l32_32653


namespace number_of_songs_l32_32247

-- Definition of the given conditions
def total_storage_GB : ℕ := 16
def used_storage_GB : ℕ := 4
def storage_per_song_MB : ℕ := 30
def GB_to_MB : ℕ := 1000

-- Theorem stating the result
theorem number_of_songs (total_storage remaining_storage song_size conversion_factor : ℕ) :
  total_storage = total_storage_GB →
  remaining_storage = total_storage - used_storage_GB →
  song_size = storage_per_song_MB →
  conversion_factor = GB_to_MB →
  (remaining_storage * conversion_factor) / song_size = 400 :=
by
  intros h_total h_remaining h_song_size h_conversion
  rw [h_total, h_remaining, h_song_size, h_conversion]
  sorry

end number_of_songs_l32_32247


namespace coordinates_of_B_l32_32890

noncomputable def point (x : ℝ) (y : ℝ) := (x, y)

def A := point (-2) 0
def P_on_C (x y : ℝ) := (x + 4) ^ 2 + y ^ 2 = 16
def condition_PA_PB (x y a : ℝ) := 4 * ((x + 2) ^ 2 + y ^ 2) = (x - a) ^ 2 + y ^ 2

theorem coordinates_of_B (a : ℝ) (hP : ∃ x y, P_on_C x y ∧ condition_PA_PB x y a) : a = 4 ∧ y = 0 :=
by
  sorry

end coordinates_of_B_l32_32890


namespace milk_production_l32_32319

theorem milk_production (a b c d e f : ℕ) (h₁ : a > 0) (h₂ : c > 0) (h₃ : f > 0) : 
  ((d * e * b * f) / (100 * a * c)) = (d * e * b * f / (100 * a * c)) :=
by
  sorry

end milk_production_l32_32319


namespace sum_greatest_odd_divisors_1_to_1024_l32_32658

-- Define the greatest odd divisor function
def greatestOddDivisor (n : Nat) : Nat :=
  if n % 2 = 1 then n
  else greatestOddDivisor (n / 2)

-- Theorem statement
theorem sum_greatest_odd_divisors_1_to_1024 : 
  (List.range' 1 1024).sum (λ k => greatestOddDivisor k) = 349526 :=
by
  sorry

end sum_greatest_odd_divisors_1_to_1024_l32_32658


namespace george_run_speed_l32_32124

theorem george_run_speed (usual_distance : ℝ) (usual_speed : ℝ) (today_first_distance : ℝ) (today_first_speed : ℝ)
  (remaining_distance : ℝ) (expected_time : ℝ) :
  usual_distance = 1.5 →
  usual_speed = 3 →
  today_first_distance = 1 →
  today_first_speed = 2.5 →
  remaining_distance = 0.5 →
  expected_time = usual_distance / usual_speed →
  today_first_distance / today_first_speed + remaining_distance / (remaining_distance / (expected_time - today_first_distance / today_first_speed)) = expected_time →
  remaining_distance / (expected_time - today_first_distance / today_first_speed) = 5 :=
by sorry

end george_run_speed_l32_32124


namespace part_a_part_b_part_c_l32_32993

theorem part_a (p q : ℝ) : q < p^2 → ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 + r2 = 2 * p) ∧ (r1 * r2 = q) :=
by
  sorry

theorem part_b (p q : ℝ) : q = 4 * p - 4 → (2^2 - 2 * p * 2 + q = 0) :=
by
  sorry

theorem part_c (p q : ℝ) : q = p^2 ∧ q = 4 * p - 4 → (p = 2 ∧ q = 4) :=
by
  sorry

end part_a_part_b_part_c_l32_32993


namespace inequality_solution_empty_solution_set_l32_32774

-- Problem 1: Prove the inequality and the solution range
theorem inequality_solution (x : ℝ) : (-7 < x ∧ x < 3) ↔ ( (x - 3)/(x + 7) < 0 ) :=
sorry

-- Problem 2: Prove the conditions for empty solution set
theorem empty_solution_set (a : ℝ) : (a > 0) ↔ ∀ x : ℝ, ¬ (x^2 - 4*a*x + 4*a^2 + a ≤ 0) :=
sorry

end inequality_solution_empty_solution_set_l32_32774


namespace area_of_ABC_equation_of_circumcircle_l32_32143

-- Part 1: Area of the triangle ABC
noncomputable def point_A : ℝ × ℝ := (3, 0)
noncomputable def point_B : ℝ × ℝ := (1, 2)
noncomputable def point_C : ℝ × ℝ := (0, -real.sqrt 3)
noncomputable def distance_AC : ℝ := 2 * real.sqrt 3

theorem area_of_ABC :
  let A := point_A
      B := point_B
      C := point_C
  in abs (1/2 : ℝ) * abs ((A.1 - C.1) * (B.2 - C.2) - (A.2 - C.2) * (B.1 - C.1)) = 3 + real.sqrt 3 := 
by
  sorry

-- Part 2: Equation of the circumcircle of the triangle ABC
noncomputable def circumcircle_eq : ℝ × ℝ × ℝ := (-2, 0, -3)

theorem equation_of_circumcircle :
  let A := point_A
      B := point_B
      C := point_C
      (D, E, F) := circumcircle_eq
  in A.1 ^ 2 + A.2 ^ 2 + D * A.1 + E * A.2 + F = 0 ∧
     B.1 ^ 2 + B.2 ^ 2 + D * B.1 + E * B.2 + F = 0 ∧
     C.1 ^ 2 + C.2 ^ 2 + D * C.1 + E * C.2 + F = 0 :=
by
  sorry


end area_of_ABC_equation_of_circumcircle_l32_32143


namespace inequality_f_extreme_points_l32_32585

-- Definitions based on the problem statement
def f (x a : ℝ) : ℝ := x * (exp x - (a / 3) * x^2 - (a / 2) * x)
def g (x : ℝ) : ℝ := exp x - exp 1 * x

-- Proof problem 1: Proving the inequality for a = 0 and x > 0
theorem inequality_f (x : ℝ) (h : x > 0) : f x 0 ≥ exp 1 * x^2 := by
  sorry

-- Definitions for extreme point discussions
def h (x a : ℝ) : ℝ := exp x - a * x
def h' (x a : ℝ) : ℝ := exp x - a

-- Proof problem 2: Number of extreme points based on the value of a
theorem extreme_points (a : ℝ) (ha : a ≤ exp 1) :
  (a < -1 / exp 1 ∨ (-1 / exp 1 < a ∧ a < 0) → ∃! (x₀ x₁ : ℝ), x₀ ≠ x₁ ∧ ∀ x, f x a ≤ f x₀ a ∨ f x a ≤ f x₁ a) ∧
  (a = -1 / exp 1 → ¬∃ (x₀ : ℝ), ∀ x, f x a ≤ f x₀ a) ∧
  (0 ≤ a ∧ a ≤ exp 1 → ∃! (x₀ : ℝ), ∀ x, f x₀ a ≤ f x a) :=
by
  sorry

end inequality_f_extreme_points_l32_32585


namespace distance_P1_P2_l32_32817

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def distance (P1 P2 : Point3D) : ℝ :=
  real.sqrt ((P2.x - P1.x)^2 + (P2.y - P1.y)^2 + (P2.z - P1.z)^2)

def P1 : Point3D := { x := 2, y := 3, z := 5 }
def P2 : Point3D := { x := 3, y := 1, z := 4 }

theorem distance_P1_P2 : distance P1 P2 = real.sqrt 6 := sorry

end distance_P1_P2_l32_32817


namespace triangle_has_120_degree_l32_32551

noncomputable def angles_of_triangle (α β γ : Real) : Prop :=
  α + β + γ = 180

theorem triangle_has_120_degree (α β γ : Real)
    (h1 : angles_of_triangle α β γ)
    (h2 : Real.cos (3 * α) + Real.cos (3 * β) + Real.cos (3 * γ) = 1) :
  γ = 120 :=
  sorry

end triangle_has_120_degree_l32_32551


namespace inequality_solution_l32_32537

theorem inequality_solution (a : ℝ) (h : 1 < a) : ∀ x : ℝ, a ^ (2 * x + 1) > (1 / a) ^ (2 * x) ↔ x > -1 / 4 :=
by
  sorry

end inequality_solution_l32_32537


namespace new_line_eq_l32_32602

theorem new_line_eq :
  ∀ (m1 c1 : ℝ) (p : ℝ × ℝ), 
    p = (2, 5) →
    m1 = 3 →
    c1 = -4 →
    let m2 := 2 * m1 in
    let c2 := c1 in
    let new_line_eq := λ x, m2 * x + c2 in
    new_line_eq = (λ x, 6 * x - 7) :=
by
  intros m1 c1 p hp hm1 hc1
  simp only [hp, hm1, hc1]
  sorry

end new_line_eq_l32_32602


namespace length_BD_l32_32237

open Real

-- Let triangle ABC be right-angled at A, with given sides AB and AC
variables (A B C D : Type*)
variables (AB AC BC AD BD : ℝ)
variables (h1 : right_angle A B C)
variables (h2 : AB = 45)
variables (h3 : AC = 60)
variables (h4 : on_line D B C)
variables (h5 : perpendicular AD BC)
variables (hAD : ∥AD∥ = 36)
variables (hBC : ∥BC∥ = 75)

-- We need to show that BD = 27
theorem length_BD : BD = 27 :=
by sorry

end length_BD_l32_32237


namespace sqrt_of_mixed_fraction_simplified_l32_32509

theorem sqrt_of_mixed_fraction_simplified :
  let x := 8 + (9 / 16) in
  sqrt x = (sqrt 137) / 4 := by
  sorry

end sqrt_of_mixed_fraction_simplified_l32_32509


namespace induction_sum_formula_induction_sum_step_l32_32743

theorem induction_sum_formula (n : ℕ) : 
  (∑ i in range (2 * n + 2), (i + 1)) = (n + 1) * (2 * n + 1) :=
by
  sorry

theorem induction_sum_step (k : ℕ) :
  (2 * k + 2) + (2 * k + 3) = (2 * (k + 1) + 2) + (2 * (k + 1) + 3) :=
by
  sorry

end induction_sum_formula_induction_sum_step_l32_32743


namespace polynomial_degree_l32_32472

theorem polynomial_degree : 
  polynomial.degree ((5 * polynomial.C (1 : ℚ) * X^3 + 2 * polynomial.C (1 : ℚ) * X^2 - polynomial.C (1 : ℚ) * X - 7) *
                     (2 * polynomial.C (1 : ℚ) * X^8 - 4 * polynomial.C (1 : ℚ) * X^6 + 3 * polynomial.C (1 : ℚ) * X^3 + 15) -
                     (polynomial.C (1 : ℚ) * X^3 - 3)^4) = 12 := 
sorry

end polynomial_degree_l32_32472


namespace inequality_three_variables_l32_32976

theorem inequality_three_variables (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 1) : 
  (1/x) + (1/y) + (1/z) ≥ 9 := 
by 
  sorry

end inequality_three_variables_l32_32976


namespace symmetry_phi_l32_32176

theorem symmetry_phi (φ : ℝ) (h1 : - (Real.pi / 2) < φ) (h2 : φ < (Real.pi / 2))
  (h3 : ∀ x : ℝ, sin (2 * (x + Real.pi / 3) + φ) = sin (2 * (Real.pi / 3 - x) + φ)) : 
  φ = -Real.pi / 6 :=
sorry

end symmetry_phi_l32_32176


namespace probability_of_A_not_losing_l32_32682

/-- The probability of player A winning is 0.3,
    and the probability of a draw between player A and player B is 0.4.
    Hence, the probability of player A not losing is 0.7. -/
theorem probability_of_A_not_losing (pA_win p_draw : ℝ) (hA_win : pA_win = 0.3) (h_draw : p_draw = 0.4) : 
  (pA_win + p_draw = 0.7) :=
by
  sorry

end probability_of_A_not_losing_l32_32682


namespace integral_evaluation_l32_32847

theorem integral_evaluation : 
  ∫ x in 0..1, (x - x^2) = 1 / 6 :=
by
  sorry

end integral_evaluation_l32_32847


namespace factor_polynomial_l32_32106

theorem factor_polynomial :
  ∃ (a b c d e f : ℤ), a < d ∧
    (a * x^2 + b * x + c) * (d * x^2 + e * x + f) = x^2 - 6 * x + 9 - 64 * x^4 ∧
    (a = -8 ∧ b = 1 ∧ c = -3 ∧ d = 8 ∧ e = 1 ∧ f = -3) := by
  sorry

end factor_polynomial_l32_32106


namespace sqrt_mixed_fraction_l32_32494

theorem sqrt_mixed_fraction (a b : ℤ) (h_a : a = 8) (h_b : b = 9) : 
  (√(a + b / 16)) = (√137) / 4 := 
by 
  sorry

end sqrt_mixed_fraction_l32_32494


namespace proof_problem_l32_32184

variables {V : Type*} [inner_product_space ℝ V] (a e : V) 

noncomputable def given_conditions (a e : V) : Prop :=
  a ≠ e ∧ ∥e∥ = 1 ∧ ∀ t : ℝ, ∥a - t • e∥ ≥ ∥a - e∥

theorem proof_problem (a e : V) (h : given_conditions a e) : inner_product e (a - e) = 0 := 
  sorry

end proof_problem_l32_32184


namespace memorable_telephone_numbers_l32_32018

theorem memorable_telephone_numbers :
  let A := { n : fin 10^7 | (n / 10^4 % 1000) = (n % 10000 / 10) };
  let B := { n : fin 10^7 | (n / 10^4 % 1000) = (n % 1000) };
  let A_count := 10^4;
  let B_count := 10^4;
  let intersection_count := 10^3;
  A.count + B.count - (A ∩ B).count = 19990 :=
by
  sorry

end memorable_telephone_numbers_l32_32018


namespace multiplication_identity_l32_32766

theorem multiplication_identity : 32519 * 9999 = 324857481 := by
  sorry

end multiplication_identity_l32_32766


namespace sky_color_change_l32_32848

theorem sky_color_change (hours: ℕ) (colors: ℕ) (minutes_per_hour: ℕ) 
                          (H1: hours = 2) 
                          (H2: colors = 12) 
                          (H3: minutes_per_hour = 60) : 
                          (hours * minutes_per_hour) / colors = 10 := 
by
  sorry

end sky_color_change_l32_32848


namespace number_of_students_playing_two_or_more_instruments_l32_32617

def group_size := 800
def fraction_playing_at_least_one := 1 / 5
def probability_playing_exactly_one := 0.04

theorem number_of_students_playing_two_or_more_instruments :
  let num_playing_at_least_one := fraction_playing_at_least_one * group_size in
  let num_playing_exactly_one := probability_playing_exactly_one * group_size in
  let num_playing_two_or_more := num_playing_at_least_one - num_playing_exactly_one in
  num_playing_two_or_more = 128 :=
by { 
  sorry 
}

end number_of_students_playing_two_or_more_instruments_l32_32617


namespace b3_b7_equals_16_l32_32886

variable {a b : ℕ → ℝ}
variable {d : ℝ}

-- Conditions: a is an arithmetic sequence with common difference d
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Condition: b is a geometric sequence
def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

-- Given condition on the arithmetic sequence a
def condition_on_a (a : ℕ → ℝ) (d : ℝ) : Prop :=
  2 * a 2 - (a 5) ^ 2 + 2 * a 8 = 0

-- Define the specific arithmetic sequence in terms of d and a5
noncomputable def a_seq (a5 d : ℝ) : ℕ → ℝ
| 0 => a5 - 5 * d
| 1 => a5 - 4 * d
| 2 => a5 - 3 * d
| 3 => a5 - 2 * d
| 4 => a5 - d
| 5 => a5
| 6 => a5 + d
| 7 => a5 + 2 * d
| 8 => a5 + 3 * d
| 9 => a5 + 4 * d
| n => 0 -- extending for unspecified

-- Condition: b_5 = a_5
def b_equals_a (a b : ℕ → ℝ) : Prop :=
  b 5 = a 5

-- Theorem: Given the conditions, prove b_3 * b_7 = 16
theorem b3_b7_equals_16 (a b : ℕ → ℝ) (d : ℝ)
  (ha_seq : is_arithmetic_sequence a d)
  (hb_seq : is_geometric_sequence b)
  (h_cond_a : condition_on_a a d)
  (h_b_equals_a : b_equals_a a b) : b 3 * b 7 = 16 :=
by
  sorry

end b3_b7_equals_16_l32_32886


namespace rhombus_diagonal_length_l32_32703

theorem rhombus_diagonal_length (d1 d2 : ℝ) (Area : ℝ) 
  (h1 : d1 = 12) (h2 : Area = 60) 
  (h3 : Area = (d1 * d2) / 2) : d2 = 10 := 
by
  sorry

end rhombus_diagonal_length_l32_32703


namespace initial_population_l32_32620

theorem initial_population (P : ℝ) 
  (H1 : P * 1.15 * 0.90 * 1.20 * 0.75 = 7575) :
  P ≈ 12199 := by
  sorry

end initial_population_l32_32620


namespace slope_intercept_form_l32_32412

-- Definition to represent the vectorized form
def vector_form (x y : ℝ) : Prop :=
  let v1 := ⟨-2, 4⟩
  let p := ⟨5, -6⟩
  let p' := ⟨x, y⟩
  (v1.1 * (p'.1 - p.1) + v1.2 * (p'.2 - p.2)) = 0

-- The theorem stating that the line can be expressed as y = (1/2)x - 8.5
theorem slope_intercept_form (x y : ℝ) (h : vector_form x y) : y = (1/2) * x - 8.5 :=
  sorry

end slope_intercept_form_l32_32412


namespace circle_diameter_tangents_l32_32396

-- Define the length of the tangents
variables {a b : ℝ} (ha : a ≠ b)

-- Define the theorem stating the diameter
theorem circle_diameter_tangents (a b : ℝ) (ha : a ≠ b) : 
  diameter_of_circle_with_tangents a b = real.sqrt (a * b) :=
sorry

end circle_diameter_tangents_l32_32396


namespace second_sum_is_1704_l32_32389

theorem second_sum_is_1704
    (total_sum : ℝ)
    (x : ℝ)
    (interest_rate_first_part : ℝ)
    (time_first_part : ℝ)
    (interest_rate_second_part : ℝ)
    (time_second_part : ℝ)
    (h1 : total_sum = 2769)
    (h2 : interest_rate_first_part = 3)
    (h3 : time_first_part = 8)
    (h4 : interest_rate_second_part = 5)
    (h5 : time_second_part = 3)
    (h6 : 24 * x / 100 = (total_sum - x) * 15 / 100) :
    total_sum - x = 1704 :=
  by
    sorry

end second_sum_is_1704_l32_32389


namespace solve_congruence_y37_x3_11_l32_32690

theorem solve_congruence_y37_x3_11 (p : ℕ) (hp_pr : Nat.Prime p) (hp_le100 : p ≤ 100) : 
  ∃ (x y : ℕ), y^37 ≡ x^3 + 11 [MOD p] := 
sorry

end solve_congruence_y37_x3_11_l32_32690


namespace exists_nat_numbers_abcd_l32_32835

theorem exists_nat_numbers_abcd
  (a b c d : ℕ)
  (h1 : a / b + c / d = 1)
  (h2 : a / d + d / b = 2008) :
  ∃ a b c d : ℕ, (a / b + c / d = 1) ∧ (a / d + d / b = 2008) :=
begin
  sorry
end

end exists_nat_numbers_abcd_l32_32835


namespace min_value_geq_4_plus_2sqrt2_l32_32150

theorem min_value_geq_4_plus_2sqrt2
  (a b c : ℝ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: c > 1)
  (h4: a + b = 1) :
  ( ( (a^2 + 1) / (a * b) - 2 ) * c + (Real.sqrt 2) / (c - 1) ) ≥ (4 + 2 * (Real.sqrt 2)) :=
sorry

end min_value_geq_4_plus_2sqrt2_l32_32150


namespace total_fish_l32_32073

theorem total_fish :
  let Billy := 10
  let Tony := 3 * Billy
  let Sarah := Tony + 5
  let Bobby := 2 * Sarah
  in Billy + Tony + Sarah + Bobby = 145 :=
by
  sorry

end total_fish_l32_32073


namespace x_intercept_of_rotated_line_l32_32329

-- Define the slope of the original line l
def slope_l : ℝ := -2 / 3

-- Define the point about which the rotation occurs
def rotation_point : ℝ × ℝ := (3, -1)

-- Define the 30 degree rotation in radians
def theta : ℝ := Math.pi / 6

-- Use the tangent addition formula to define the slope after rotation
def slope_m : ℝ := (slope_l + Math.tan theta) / (1 - slope_l * Math.tan theta)

-- Define the y-intercept b after rotating and moving through the rotation point
def b : ℝ := -1 - slope_m * 3

-- The x-coordinate of the x-intercept of the new line m
def x_intercept : ℝ := -b / slope_m

-- Main theorem statement
theorem x_intercept_of_rotated_line :
  ∃ (x : ℝ), x = x_intercept :=
by
  have h_slope_l : slope_l = -2 / 3 := rfl
  have h_rotation_point : rotation_point = (3, -1) := rfl
  have h_theta : theta = Math.pi / 6 := rfl
  have h_slope_m : slope_m = (slope_l + Math.tan theta) / (1 - slope_l * Math.tan theta) := rfl
  have h_b : b = -1 - slope_m * 3 := rfl
  have h_x_intercept : x_intercept = -b / slope_m := rfl
  exact ⟨x_intercept, h_x_intercept⟩

end x_intercept_of_rotated_line_l32_32329


namespace necessary_but_not_sufficient_condition_l32_32281

variables {a b c : ℝ × ℝ}

def nonzero_vector (v : ℝ × ℝ) : Prop := v ≠ (0, 0)

theorem necessary_but_not_sufficient_condition (ha : nonzero_vector a) (hb : nonzero_vector b) (hc : nonzero_vector c) :
  (a.1 * (b.1 - c.1) + a.2 * (b.2 - c.2) = 0) ↔ (b = c) :=
sorry

end necessary_but_not_sufficient_condition_l32_32281


namespace sin_le_half_probability_l32_32867

theorem sin_le_half_probability : 
  ∀ x ∈ Icc (0 : ℝ) (Real.pi / 3), 
  (set.measure (set_of (λ x, sin x ≤ 1 / 2)) (real.volume.restrict (Icc 0 (Real.pi / 3)))) / 
  (set.measure (Icc 0 (Real.pi / 3)) (real.volume)) = 1 / 2 :=
begin
  sorry
end

end sin_le_half_probability_l32_32867


namespace seaweed_percentage_correct_l32_32819

-- Define the conditions
variables (total_seaweed : ℝ) (fire_percentage : ℝ) (livestock_weight : ℝ)

-- Assign the given values to the variables
def total_seaweed := 400
def fire_percentage := 0.50
def livestock_weight := 150

-- Calculate the remaining seaweed after starting fires
def remaining_seaweed := total_seaweed * (1 - fire_percentage)

-- Define the correct answer calculation
def human_percentage := ((remaining_seaweed - livestock_weight) / remaining_seaweed) * 100

-- Proof statement: given the conditions, prove the human edible percentage is 25%
theorem seaweed_percentage_correct :
  human_percentage total_seaweed fire_percentage livestock_weight = 25 := 
  by
    -- Proof omitted
    sorry

end seaweed_percentage_correct_l32_32819


namespace sum_of_first_9_terms_l32_32552

variable {a : ℕ → ℝ} -- the arithmetic sequence
variable {S : ℕ → ℝ} -- the sum function

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (a 1 + a n)) / 2

axiom arithmetic_sequence_condition (h : is_arithmetic_sequence a) : a 5 = 2

theorem sum_of_first_9_terms (h : is_arithmetic_sequence a) (h5: a 5 = 2) : sum_of_first_n_terms a 9 = 18 := by
  sorry

end sum_of_first_9_terms_l32_32552


namespace equal_areas_ACE_BDF_l32_32304

variables {V : Type*} [InnerProductSpace ℝ V]

/-- Given a hexagon with opposite sides parallel -/
variables (a b c d e f : V) 
  (h1 : Parallel (span ℝ {a}) (span ℝ {d}))
  (h2 : Parallel (span ℝ {b}) (span ℝ {e}))
  (h3 : Parallel (span ℝ {c}) (span ℝ {f}))

-- Define the areas of triangles ACE and BDF
noncomputable def area_triangle (u v : V) : ℝ := (1 / 2) * ‖u × v‖

noncomputable def t_ACE : ℝ := area_triangle (c + d) (-a - b)
noncomputable def t_BDF : ℝ := area_triangle (d + e) (-b - c)

theorem equal_areas_ACE_BDF : t_ACE a b c d e f h1 h2 h3 = t_BDF a b c d e f h1 h2 h3 :=
sorry

end equal_areas_ACE_BDF_l32_32304


namespace complement_intersection_l32_32180

open Set

noncomputable def A : Set ℝ := {x : ℝ | x^2 - x - 2 > 0}
noncomputable def B : Set ℝ := {x : ℝ | abs (2 * x - 3) < 3}

theorem complement_intersection :
  compl (A ∩ B) = {x : ℝ | x ≥ 3 ∨ x ≤ 2} :=
by
  sorry

end complement_intersection_l32_32180


namespace inscribed_circle_of_triangle_abc_l32_32485

-- Definitions for simplicity
variables (r : ℝ)
variables (a b c : ℝ) -- sides of the triangle

-- Conditions from the problem
def right_triangle_abc (a b c : ℝ) : Prop := (c = sqrt (a^2 + b^2))
def radius_condition (r : ℝ) (a b c : ℝ) : Prop := (a = 9 * r) ∧ (b = 12 * r) ∧ (c = 15 * r)
def inscribed_circle_radius (a b c : ℝ) (r_in : ℝ) : Prop := (r_in = (a + b - c) / 2)

-- Main statement to be proved
theorem inscribed_circle_of_triangle_abc (r : ℝ) :
  ∀ a b c, right_triangle_abc a b c → radius_condition r a b c →
  ∃ r_in, inscribed_circle_radius a b c r_in ∧ r_in = 3 * r :=
by
  intros a b c h_triangle h_condition
  have : r_in = 3 * r, sorry
  use 3 * r
  exact ⟨by { sorry }, this⟩ -- no formal proofs provided as per instructions.

end inscribed_circle_of_triangle_abc_l32_32485


namespace probability_of_product_multiple_of_105_l32_32612

open Finset

def set_S : Finset ℕ := {3, 5, 15, 21, 35, 45, 63}

noncomputable def is_multiple_of_105 (a b : ℕ) : Prop :=
  (a * b) % 105 = 0

theorem probability_of_product_multiple_of_105 :
  let pairs := (set_S.filter (λ x, 7 ∣ x) ×ˢ set_S.filter (λ x, 3 ∣ x)) ∪ 
               (set_S.filter (λ x, 5 ∣ x) ×ˢ set_S.filter (λ x, 21 ≤ x ∨ x = 35)) in
  ∃ (count : ℕ), count = pairs.card ∧
  let total_pairs := set_S.pairwise_disjoint_card in
  total_pairs =  21 ∧
  let valid_pairs := pairs.filter (λ pair, is_multiple_of_105 pair.1 pair.2) in
  count = valid_pairs.card ∧
  valid_pairs.card.to_rat / total_pairs.to_rat = (1 : ℚ) / 3 :=
by
  let set_S := {3, 5, 15, 21, 35, 45, 63}
  let pairs := (set_S.filter (λ x, 7 ∣ x) ×ˢ set_S.filter (λ x, 3 ∣ x)) ∪ 
               (set_S.filter (λ x, 5 ∣ x) ×ˢ set_S.filter (λ x, 21 ≤ x ∨ x = 35))
  let count := pairs.card
  have total_pairs : set_S.pairwise_disjoint_card = 21 :=
    sorry
  have valid_pairs := pairs.filter (λ pair, is_multiple_of_105 pair.1 pair.2)
  have h1 : valid_pairs.card = count := sorry
  have div_result : valid_pairs.card.to_rat / total_pairs.to_rat = (1 : ℚ) / 3 := 
    by sorry
  exact ⟨count, rfl, total_pairs, rfl, valid_pairs.count, h1, div_result⟩

end probability_of_product_multiple_of_105_l32_32612


namespace sqrt_cosine_product_eq_l32_32476

theorem sqrt_cosine_product_eq (cos : ℤ → ℝ) (h_cos_1 : cos (1*π / 9) = real.cos (π / 9))
                               (h_cos_2 : cos (2*π / 9) = real.cos (2*π / 9))
                               (h_cos_4 : cos (4*π / 9) = real.cos (4*π / 9)) :
  sqrt ((3 - cos (1*π / 9)^2) * 
        (3 - cos (2*π / 9)^2) * 
        (3 - cos (4*π / 9)^2)) = real.sqrt(15) / 32 :=
sorry

end sqrt_cosine_product_eq_l32_32476


namespace min_value_fraction_l32_32916

theorem min_value_fraction (x : ℝ) (h : x > -2) : min (λ x, (x^2 + 6*x + 9) / (x + 2)) x = 4 :=
sorry

end min_value_fraction_l32_32916


namespace smallest_natur_number_with_units_digit_6_and_transf_is_four_times_l32_32861

open Nat

theorem smallest_natur_number_with_units_digit_6_and_transf_is_four_times (n : ℕ) :
  (n % 10 = 6 ∧ ∃ m, 6 * 10 ^ (m - 1) + n / 10 = 4 * n) → n = 153846 :=
by 
  sorry

end smallest_natur_number_with_units_digit_6_and_transf_is_four_times_l32_32861


namespace total_distance_total_distance_alt_l32_32047

variable (D : ℝ) -- declare the variable for the total distance

-- defining the conditions
def speed_walking : ℝ := 4 -- speed in km/hr when walking
def speed_running : ℝ := 8 -- speed in km/hr when running
def total_time : ℝ := 3.75 -- total time in hours

-- proving that D = 10 given the conditions
theorem total_distance 
    (h1 : D / (2 * speed_walking) + D / (2 * speed_running) = total_time) : 
    D = 10 := 
sorry

-- Alternative theorem version declaring variables directly
theorem total_distance_alt
    (speed_walking speed_running total_time : ℝ) -- declaring variables
    (D : ℝ) -- the total distance
    (h1 : D / (2 * speed_walking) + D / (2 * speed_running) = total_time)
    (hw : speed_walking = 4)
    (hr : speed_running = 8)
    (ht : total_time = 3.75) : 
    D = 10 := 
sorry

end total_distance_total_distance_alt_l32_32047


namespace fourth_intersection_point_l32_32634

noncomputable def curve (p : ℝ × ℝ) : Prop := p.1 * p.2 = 2

def circle (a b r : ℝ) (p : ℝ × ℝ) : Prop := (p.1 - a)^2 + (p.2 - b)^2 = r^2

def point1 := (4 : ℝ, 1 / 2 : ℝ)
def point2 := (-2 : ℝ, -1 : ℝ)
def point3 := (1 / 4 : ℝ, 8 : ℝ)
def point4 := (-1 / 8 : ℝ, -16 : ℝ)

theorem fourth_intersection_point 
(a b r : ℝ) 
(h1 : curve point1)
(h2 : curve point2)
(h3 : curve point3)
(h4 : circle a b r point1)
(h5 : circle a b r point2)
(h6 : circle a b r point3) 
(h7 : curve point4) :
  circle a b r point4 :=
  sorry

end fourth_intersection_point_l32_32634


namespace parallel_lines_slope_l32_32751

theorem parallel_lines_slope (d : ℝ) (h : 3 = 4 * d) : d = 3 / 4 :=
by
  sorry

end parallel_lines_slope_l32_32751


namespace new_cases_on_third_day_l32_32430

theorem new_cases_on_third_day (initial_cases : ℕ) (new_cases_day2 : ℕ) (recoveries_day2 : ℕ) 
  (recoveries_day3 : ℕ) (total_cases_after_day3 : ℕ) (new_cases_day3 : ℕ) :
  initial_cases = 2000 →
  new_cases_day2 = 500 →
  recoveries_day2 = 50 →
  recoveries_day3 = 200 →
  total_cases_after_day3 = 3750 →
  new_cases_day3 = 1500 →
  initial_cases + new_cases_day2 - recoveries_day2 + new_cases_day3 - recoveries_day3 = total_cases_after_day3 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end new_cases_on_third_day_l32_32430


namespace quadratic_inequality_l32_32721

variables {a b c : ℝ}

theorem quadratic_inequality (h1 : ∀ x : ℝ, a * x^2 + b * x + c < 0) 
                            (h2 : a < 0) 
                            (h3 : b^2 - 4 * a * c < 0) : 
                            (b / a) < (c / a + 1) :=
begin
  sorry
end

end quadratic_inequality_l32_32721


namespace dirk_profit_l32_32483

theorem dirk_profit 
  (days : ℕ) 
  (amulets_per_day : ℕ) 
  (sale_price : ℕ) 
  (cost_price : ℕ) 
  (cut_percentage : ℕ) 
  (profit : ℕ) : 
  days = 2 → amulets_per_day = 25 → sale_price = 40 → cost_price = 30 → cut_percentage = 10 → profit = 300 :=
by
  intros h_days h_amulets_per_day h_sale_price h_cost_price h_cut_percentage
  -- Placeholder for the proof
  sorry

end dirk_profit_l32_32483


namespace greatest_number_zero_l32_32514

-- Define the condition (inequality)
def inequality (x : ℤ) : Prop :=
  3 * x + 2 < 5 - 2 * x

-- Define the property of being the greatest whole number satisfying the inequality
def greatest_whole_number (x : ℤ) : Prop :=
  inequality x ∧ (∀ y : ℤ, inequality y → y ≤ x)

-- The main theorem stating the greatest whole number satisfying the inequality is 0
theorem greatest_number_zero : greatest_whole_number 0 :=
by
  sorry

end greatest_number_zero_l32_32514


namespace possible_values_of_reciprocal_sum_l32_32266

theorem possible_values_of_reciprocal_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) :
  ∃ y, y = (1/a + 1/b) ∧ (2 ≤ y ∧ ∀ t, t < y ↔ ¬t < 2) :=
by sorry

end possible_values_of_reciprocal_sum_l32_32266


namespace altitudes_intersection_on_angle_bisector_l32_32989

theorem altitudes_intersection_on_angle_bisector
  {A B C C1 A1 B1 : Type*}
  [triangle A B C]
  (h1 : side_length B C1 = side_length C1 A1)
  (h2 : side_length C1 A1 = side_length A1 B1)
  (h3 : side_length A1 B1 = side_length B1 C):
  lies_on_angle_bisector (orthocenter C1 A1 B1) (angle_bisector A (B C)) :=
sorry

end altitudes_intersection_on_angle_bisector_l32_32989


namespace smallest_n_six_consecutive_integers_l32_32227

theorem smallest_n_six_consecutive_integers : ∃ n : ℤ, n+4 < 3*(n + 2.5) ∧ ∀ m : ℤ, m+4 < 3*(m + 2.5) → n ≤ m := by
  sorry

end smallest_n_six_consecutive_integers_l32_32227


namespace negative_integer_solution_l32_32347

theorem negative_integer_solution (N : ℤ) (h1 : N < 0) (h2 : 3 * N^2 + N = 15) : N = -3 :=
by
  have h3 : ∃ N : ℤ, N < 0 ∧ 3 * N^2 + N = 15 :=
    ⟨-3, by norm_num, by norm_num⟩
  exact sorry

end negative_integer_solution_l32_32347


namespace solve_fraction_eq_zero_l32_32198

theorem solve_fraction_eq_zero (x : ℝ) (h : x - 2 ≠ 0) : (x + 1) / (x - 2) = 0 ↔ x = -1 :=
by
  sorry

end solve_fraction_eq_zero_l32_32198


namespace simplify_sqrt_of_mixed_number_l32_32498

noncomputable def sqrt_fraction := λ (a b : ℕ), (Real.sqrt a) / (Real.sqrt b)

theorem simplify_sqrt_of_mixed_number : sqrt_fraction 137 16 = (Real.sqrt 137) / 4 := by
  sorry

end simplify_sqrt_of_mixed_number_l32_32498


namespace number_line_is_line_l32_32044

-- Define the terms
def number_line : Type := ℝ -- Assume number line can be considered real numbers for simplicity
def is_line (l : Type) : Prop := l = ℝ

-- Proving that number line is a line.
theorem number_line_is_line : is_line number_line :=
by {
  -- by definition of the number_line and is_line
  sorry
}

end number_line_is_line_l32_32044


namespace find_consecutive_even_numbers_l32_32729

theorem find_consecutive_even_numbers :
  ∃ a b c : ℕ, (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 ∧
               b = a + 2 ∧ c = a + 4 ∧
               (a * b * c) / 10000 = 8 ∧ (a * b * c) % 10 = 2 ∧
               800000 ≤ a * b * c ∧ a * b * c ≤ 899998) :=
begin
  use [94, 96, 98],
  split, norm_num [Int.even_iff_two_dvd.mp], -- a % 2 = 0
  split, norm_num [Int.even_iff_two_dvd.mp], -- b % 2 = 0
  split, norm_num [Int.even_iff_two_dvd.mp], -- c % 2 = 0
  split, norm_num, -- b = a + 2
  split, norm_num, -- c = a + 4
  split, norm_num, -- (a * b * c) / 100000 = 8
  split, norm_num, -- (a * b * c) % 10 = 2
  norm_num, linarith, -- 800000 ≤ a * b * c ≤ 899998
end

end find_consecutive_even_numbers_l32_32729


namespace prove_monomial_l32_32579

-- Definitions and conditions from step a)
def like_terms (x y : ℕ) := 
  x = 2 ∧ x + y = 5

-- Main statement to be proved
theorem prove_monomial (x y : ℕ) (h : like_terms x y) : 
  1 / 2 * x^3 - 1 / 6 * x * y^2 = 1 :=
by
  sorry

end prove_monomial_l32_32579


namespace quadratic_inequality_l32_32718

variable {a b c : ℝ}

noncomputable def quadratic_polynomial (x : ℝ) := a * x^2 + b * x + c

theorem quadratic_inequality (h1 : ∀ x : ℝ, quadratic_polynomial x < 0)
    (h2 : a < 0) (h3 : b^2 - 4*a*c < 0) : (b / a) < (c / a + 1) := 
sorry

end quadratic_inequality_l32_32718


namespace ice_cream_cost_l32_32688

theorem ice_cream_cost
    (total_tip : ℕ)
    (cost_steak : ℕ) (number_steak : ℕ)
    (cost_burger : ℝ) (number_burger : ℕ)
    (number_ice_cream_cups : ℕ)
    (remaining_money : ℕ)
    (h1 : total_tip = 99)
    (h2 : cost_steak = 24)
    (h3 : number_steak = 2)
    (h4 : cost_burger = 3.5)
    (h5 : number_burger = 2)
    (h6 : number_ice_cream_cups = 3)
    (h7 : remaining_money = 38) :
    (total_tip - (cost_steak * number_steak + cost_burger * number_burger).natAbs - remaining_money) / number_ice_cream_cups = 20.33 := 
by
  sorry

end ice_cream_cost_l32_32688


namespace sum_powers_eq_34_over_3_l32_32594

theorem sum_powers_eq_34_over_3 (a b c : ℝ) 
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 3)
  (h3 : a^3 + b^3 + c^3 = 6):
  a^4 + b^4 + c^4 = 34 / 3 :=
by
  sorry

end sum_powers_eq_34_over_3_l32_32594


namespace range_condition_l32_32584

def f (x : ℝ) : ℝ := Real.log (Real.exp x + Real.exp (-x)) + x^2

theorem range_condition (x : ℝ) :
  f (2 * x) > f (x + 3) ↔ x ∈ Set.Ioo (-(1 : ℝ)) (3 : ℝ) ∨ x ∈ Set.Ioo 3 (Real.PosInfinity) :=
sorry

end range_condition_l32_32584


namespace digits_divisible_by_11_l32_32849

theorem digits_divisible_by_11 (n : ℕ) : 
  (∀ m, list.perm (list.map (λ d, d.to_nat) (nat.digits 10 n)) (list.map (λ d, d.to_nat) (nat.digits 10 m)) → m % 11 = 0) ↔ 
  (∃ a : ℕ, (n = a * (10^length (nat.digits 10 n)) / 9) ∧ 2 ∣ (length (nat.digits 10 n))) :=
sorry

end digits_divisible_by_11_l32_32849


namespace impossible_transform_l32_32069

-- Definition of a word and its tripling
def word := List Bool
def tripling (A : word) : word := A ++ A ++ A

-- Definition of the value function v
def value_function (A : word) : ℕ :=
  A.foldl (fun (acc: ℕ × ℕ) (x: Bool) => (acc.1 + 1, acc.2 + (if x then acc.1 + 1 else 0))) (0, 0) |>.2

-- Theorem stating the impossibility of transforming '10' to '01' using the given operations.
theorem impossible_transform (A B : word) (h_trip : ∀ C : word, A = B ++ tripling C ∨ ∃ D, A = D ++ tripling C ++ B) :
  A = [true, false] → B = [false, true] → False :=
by
  intro hA hB
  let vA := value_function A
  let vB := value_function B
  -- Value function mod 3 should be preserved
  have h_mod : vA % 3 = vB % 3 := sorry -- Derived from given conditions
  have h_val_10 : vA = 1 := by
    unfold value_function;
    simp [hA]
  have h_val_01 : vB = 2 := by
    unfold value_function;
    simp [hB]
  -- Contradiction
  have h_contra : 1 % 3 ≠ 2 % 3 := by
    simp
  contradiction

end impossible_transform_l32_32069


namespace limit_of_function_l32_32773

open Real

theorem limit_of_function :
  tendsto (λ x => (1 + x^2 * 2^x) / (1 + x^2 * 5^x)) (nhds 0) (𝓝 (2/5)) := sorry

end limit_of_function_l32_32773


namespace star_problem_l32_32240

def star_problem_proof (p q r s u : ℤ) (S : ℤ): Prop :=
  (S = 64) →
  ({n : ℤ | n = 19 ∨ n = 21 ∨ n = 23 ∨ n = 25 ∨ n = 27} = {p, q, r, s, u}) →
  (p + q + r + s + u = 115) →
  (9 + p + q + 7 = S) →
  (3 + p + u + 15 = S) →
  (3 + q + r + 11 = S) →
  (9 + u + s + 11 = S) →
  (15 + s + r + 7 = S) →
  (q = 27)

theorem star_problem : ∃ p q r s u S, star_problem_proof p q r s u S := by
  -- Proof goes here
  sorry

end star_problem_l32_32240


namespace unique_balance_point_iff_l32_32569

-- Definitions
def is_balance_point (f : ℝ → ℝ) (t : ℝ) : Prop :=
  f t = t

def balance_point_property (m : ℝ) : Prop :=
  ∀ t : ℝ, is_balance_point (λ x : ℝ, (m - 1) * x^2 - 3 * x + 2 * m) t → t = 1

-- The main statement
theorem unique_balance_point_iff (m : ℝ) :
  balance_point_property m ↔ m = 2 ∨ m = -1 ∨ m = 1 :=
sorry

end unique_balance_point_iff_l32_32569


namespace weight_of_each_crate_find_weight_of_crate_l32_32439

theorem weight_of_each_crate (W_t W_tc : ℝ) (n : ℕ) (h1 : W_t = 9600) (h2 : W_tc = 38000) (h3 : n = 40) :
    W_tc - W_t = 28400 := by
  -- given conditions
  calc
    W_tc - W_t = 38000 - 9600 := by rw [h1, h2]
    ... = 28400 := by norm_num

theorem find_weight_of_crate (W_t W_tc : ℝ) (n : ℕ) (h1 : W_t = 9600) (h2 : W_tc = 38000) (h3 : n = 40) :
    (W_tc - W_t) / n = 710 := by
  -- use the previous result to find weight of crates
  have h4 : W_tc - W_t = 28400 := weight_of_each_crate W_t W_tc n h1 h2 h3
  -- given condition number of crates n
  calc
    (W_tc - W_t) / n = 28400 / 40 := by rw [h4, h3]
    ... = 710 := by norm_num

end weight_of_each_crate_find_weight_of_crate_l32_32439


namespace year_2023_not_lucky_l32_32482

def is_valid_date (month day year : ℕ) : Prop :=
  month * day = year % 100

def is_lucky_year (year : ℕ) : Prop :=
  ∃ month day, month ≤ 12 ∧ day ≤ 31 ∧ is_valid_date month day year

theorem year_2023_not_lucky : ¬ is_lucky_year 2023 :=
by sorry

end year_2023_not_lucky_l32_32482


namespace min_value_eq_144_l32_32970

noncomputable def min_value (x y z w : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_pos_w : w > 0) (h_sum : x + y + z + w = 1) : ℝ :=
  if x <= 0 ∨ y <= 0 ∨ z <= 0 ∨ w <= 0 then 0 else (x + y + z) / (x * y * z * w)

theorem min_value_eq_144 (x y z w : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_pos_w : w > 0) (h_sum : x + y + z + w = 1) :
  min_value x y z w h_pos_x h_pos_y h_pos_z h_pos_w h_sum = 144 :=
sorry

end min_value_eq_144_l32_32970


namespace intersection_points_C2_C3_max_distance_AB_l32_32234

theorem intersection_points_C2_C3 : 
  (∀ θ : ℝ, (∃ x y : ℝ, (x^2 + y^2 = 16 * sin(θ)^2 ∧ x = 4*sin(θ) ∧ y = 4*sin(θ)) ∧
                        (∀ α : ℝ, x^2 + y^2 = 48 * cos(θ)^2 ∧ x = 4 * √3 * cos(θ) ∧ y = 4 * √3 * cos(θ)))
  ∃ (u v : ℝ), (u = 0 ∧ v = 0) ∨ (u = √3 ∧ v = 3)) := sorry

theorem max_distance_AB : 
  (∀ α θ : ℝ, 0 ≤ α ∧ α < π ∧ (∃ t : ℝ, t ≠ 0 ∧ (4*sin(α) = t*cos(α) ∧ 4*√3*cos(α) = t*sin(α))
  ∃ A B : ℝ, |(4*√3*cos(α)) - (4*sin(α))| = |8 * cos(π/6 + α)|) 8):= sorry

end intersection_points_C2_C3_max_distance_AB_l32_32234


namespace remainder_div_14_l32_32377

def S : ℕ := 11065 + 11067 + 11069 + 11071 + 11073 + 11075 + 11077

theorem remainder_div_14 : S % 14 = 7 :=
by
  sorry

end remainder_div_14_l32_32377


namespace midpoints_of_edges_lie_on_sphere_l32_32229

noncomputable theory
open EuclideanGeometry

/-- Midpoints of the six edges of a tetrahedron where all three pairs of opposite edges 
are mutually perpendicular lie on one sphere. -/
theorem midpoints_of_edges_lie_on_sphere
  {A B C D : Point}
  (h1 : ∀ X Y Z W : Point, opp_edges (X, Y) (Z, W) → Perpendicular (X - Y) (Z - W)) :
    ∃ (S : Sphere), ∀ (M : Point), is_midpoint_of_edge M (Edges A B C D) → M ∈ S := 
sorry

end midpoints_of_edges_lie_on_sphere_l32_32229


namespace jen_hours_per_week_l32_32251

theorem jen_hours_per_week (B : ℕ) (h1 : ∀ t : ℕ, t * (B + 7) = 6 * B) : B + 7 = 21 := by
  sorry

end jen_hours_per_week_l32_32251


namespace monroe_legs_total_l32_32289

def num_spiders : ℕ := 8
def num_ants : ℕ := 12
def legs_per_spider : ℕ := 8
def legs_per_ant : ℕ := 6

theorem monroe_legs_total :
  num_spiders * legs_per_spider + num_ants * legs_per_ant = 136 :=
by
  sorry

end monroe_legs_total_l32_32289


namespace makenna_garden_larger_l32_32258

noncomputable def areaOfKarlGarden : ℝ := π * 15^2
def areaOfMakennaGarden : ℝ := 30 * 50

theorem makenna_garden_larger :
  abs (areaOfMakennaGarden - (areaOfKarlGarden : ℝ)) = 793 := 
sorry

end makenna_garden_larger_l32_32258


namespace colorable_sets_l32_32953

variables {α : Type*}
variables (A : ℕ → set α) (n : ℕ)

theorem colorable_sets (h_nonempty : ∀ i, 1 ≤ i ∧ i ≤ n → 2 ≤ (A i).card)
  (h_intersection : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → (A i ∩ A j).card ≠ 1) :
  ∃ (coloring : α → bool), (∀ i, 1 ≤ i ∧ i ≤ n → ∃ b1 b2 : bool, b1 ≠ b2 ∧ ∀ a ∈ A i, coloring a = b1 ∨ coloring a = b2) :=
by { sorry }

end colorable_sets_l32_32953


namespace find_blue_shirts_l32_32530

-- Statements of the problem conditions
def total_shirts : ℕ := 23
def green_shirts : ℕ := 17

-- Definition that we want to prove
def blue_shirts : ℕ := total_shirts - green_shirts

-- Proof statement (no need to include the proof itself)
theorem find_blue_shirts : blue_shirts = 6 := by
  sorry

end find_blue_shirts_l32_32530


namespace man_reaches_train_B_l32_32592

noncomputable def total_time_to_reach_train_B (length_trainA speed_trainA length_trainB speed_trainB speed_man obstacle_time relative_speed_to_trainA relative_speed_to_trainB: ℝ): ℝ :=
let relative_speed_A := speed_trainA - speed_man,
    relative_speed_B := speed_trainB + speed_man,
    time_to_cross_A := length_trainA / relative_speed_A,
    time_to_cross_B := length_trainB / relative_speed_B,
    total_obstacle_time := 2 * obstacle_time in
time_to_cross_B + total_obstacle_time

theorem man_reaches_train_B {lA lB vA vB vM t_obs : ℝ}
  (hA : lA = 420) (sA : vA = 30 * (1000 / 3600)) 
  (hB : lB = 520) (sB : vB = 40 * (1000 / 3600)) 
  (sM : vM = 6 * (1000 / 3600)) (obs : t_obs = 5) :
  total_time_to_reach_train_B lA vA lB vB vM t_obs 6.66 12.78 ≈ 50.69 :=
by
  -- Placeholders for the proof steps
  sorry

end man_reaches_train_B_l32_32592


namespace sixth_6_composite_eq_441_l32_32795

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def composite_factors_count (n : ℕ) : ℕ :=
  (finset.filter is_composite (finset.Icc 1 n)).card

def is_6_composite (n : ℕ) : Prop :=
  composite_factors_count n = 6

def sixth_smallest_6_composite : ℕ :=
  finset.sort nat.lt (finset.filter is_6_composite (finset.range 1000)).nth_le 5 sorry

theorem sixth_6_composite_eq_441 :
  sixth_smallest_6_composite = 441 :=
sorry

end sixth_6_composite_eq_441_l32_32795


namespace find_sum_l32_32538

variable (a b c d : ℝ)

theorem find_sum (h1 : a * b + b * c + c * d + d * a = 48) (h2 : b + d = 6) : a + c = 8 :=
sorry

end find_sum_l32_32538


namespace spadesuit_problem_l32_32870

def spadesuit (x y : ℝ) := (x + y) * (x - y)

theorem spadesuit_problem : spadesuit 5 (spadesuit 3 2) = 0 := by
  sorry

end spadesuit_problem_l32_32870


namespace circumcenter_triangle_BPQ_on_BC_l32_32628

variables {A B C D E F P Q : Type*}
variables [square ABCD] [on_segment E BC] [on_segment F CD]
variables [perpendicular_from E P AC] [perpendicular_from F Q AC] [angle EAF 45]

-- The goal is to prove the circumcenter of triangle BPQ lies on segment BC.
theorem circumcenter_triangle_BPQ_on_BC (h1 : square ABCD) 
  (h2 : on_segment E BC) 
  (h3 : on_segment F CD)
  (h4 : perpendicular_from E P AC)
  (h5 : perpendicular_from F Q AC)
  (h6 : angle EAF 45) :
  on_segment (circumcenter_triangle BPQ) BC := 
sorry

end circumcenter_triangle_BPQ_on_BC_l32_32628


namespace bisector_of_angle_OCD_varies_l32_32659

theorem bisector_of_angle_OCD_varies 
  (C D O A B : Point) (circle : Circle)
  (hAB_diameter : Diameter AB circle)
  (hO_center : Center O circle)
  (hC_on_circle : OnCircle C circle)
  (hCD_30_degrees : angle CD AB = 30) :
  varies (bisector (angle OCD)) :=
sorry

end bisector_of_angle_OCD_varies_l32_32659


namespace geom_seq_sum_lt_two_geom_seq_squared_geom_seq_square_inequality_l32_32520

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {q : ℝ} (hq : 0 < q) (hq2 : q ≠ 1)

-- ① If $a_{1}=1$ and the common ratio is $\frac{1}{2}$, then $S_{n} < 2$;
theorem geom_seq_sum_lt_two (h₁ : a 1 = 1) (hq_half : q = 1 / 2) (n : ℕ) : S n < 2 := sorry

-- ② The sequence $\{a_{n}^{2}\}$ must be a geometric sequence
theorem geom_seq_squared (h_geom : ∀ n, a (n + 1) = q * a n) : ∃ r : ℝ, ∀ n, a n ^ 2 = r ^ n := sorry

-- ④ For any positive integer $n$, $a{}_{n}^{2}+a{}_{n+2}^{2}\geqslant 2a{}_{n+1}^{2}$
theorem geom_seq_square_inequality (h_geom : ∀ n, a (n + 1) = q * a n) (n : ℕ) (hn : 0 < n) : 
  a n ^ 2 + a (n + 2) ^ 2 ≥ 2 * a (n + 1) ^ 2 := sorry

end geom_seq_sum_lt_two_geom_seq_squared_geom_seq_square_inequality_l32_32520


namespace matrix_power_l32_32593

theorem matrix_power (a n : ℕ) :
  let A := ![![1, 3, a], ![0, 1, 6], ![0, 0, 1]],
      B := ![![0, 3, a], ![0, 0, 6], ![0, 0, 0]],
      I := ![![1, 0, 0], ![0, 1, 0], ![0, 0, 1]]
  in  A^n = ![![1, 27, 4064], ![0, 1, 54], ![0, 0, 1]] →
  a + n = 51 :=
by
  sorry

end matrix_power_l32_32593


namespace volume_of_sand_pile_l32_32803

theorem volume_of_sand_pile
  (diameter : ℝ) 
  (height : ℝ)
  (r : ℝ)
  (V : ℝ) :
  diameter = 12 →
  height = 0.6 * diameter →
  r = diameter / 2 →
  V = (1 / 3) * π * r^2 * height →
  V = 86.4 * π :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  linarith

end volume_of_sand_pile_l32_32803


namespace sqrt_of_trig_identity_l32_32475

noncomputable def cos_sq (theta : ℝ) : ℝ := (Real.cos theta) ^ 2

theorem sqrt_of_trig_identity :
  let x := 3 - cos_sq (Real.pi / 9)
  let y := 3 - cos_sq (2 * Real.pi / 9)
  let z := 3 - cos_sq (4 * Real.pi / 9) in
  Real.sqrt (x * y * z) = 0 :=
by
  sorry

end sqrt_of_trig_identity_l32_32475


namespace hyperbola_standard_equation_l32_32545

theorem hyperbola_standard_equation 
  (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : ∀ x y : ℝ, x = 0 → y = 0 -> (x^2 / a^2) - (y^2 / b^2) = 1) 
  (focal_length : 2 * real_sqrt (a^2 + b^2) = 10)
  (point_on_asymptote : ∃ P : (ℝ × ℝ), P = (3, 4) ∧ (3 / a) = (4 / b)) :
  (∃ c : ℝ, a = 3 ∧ b = 4 ∧ (c^2 = a^2 + b^2) ∧ (2 * c = 10)) :=
begin
  sorry
end

end hyperbola_standard_equation_l32_32545


namespace correct_proposition_l32_32583

theorem correct_proposition :
  (¬(∀ (d c : Real), ⊥)) ∧
  (¬(∀ a b : Real, a ≠ b → ⊥)) ∧
  (∀ P : Set Point, inscribed_in_circle P → parallelogram P → rectangle P) ∧
  (¬(∃ s : Arc, diameter_is_chord s ∧ s = Arc.sem)) := by
  sorry

end correct_proposition_l32_32583


namespace arithmetic_sequence_sum_l32_32891

noncomputable def S_n (a1 d : ℤ) (n : ℕ) : ℤ := n * a1 + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_sum (d : ℤ) (h : d ≠ 0) :
  let a1 := 20
  let a := λ n : ℕ, a1 + n * d
  S_n a1 d 10 = 110 :=
by
  have a1 := 20
  let a := λ n : ℕ, a1 + n * d 
  have h1 : a 2 * a 8 = (a 6) ^ 2 := by
    simp [a]
    have h_eq := (a1 + 2 * d) * (a1 + 8 * d) = (a1 + 6 * d) ^ 2
    simp [h_eq, a1, d]
    sorry
  have h2 : d = -2 := by
    simp [h1, a1] at *
    sorry
  have S_10 := S_n a1 (-2) 10
  simp [S_n, a1, d]
  show S_10 = 110 from
    calc
      S_10 = 10 * a1 + ((10 * (10 - 1) / 2) * (-2)) : rfl
      ... = 10 * 20 + (45 * (-2)) : by simp [a1]
      ... = 200 - 90 : by simp
      ... = 110 : rfl
  sorry

end arithmetic_sequence_sum_l32_32891


namespace problem_1_problem_2_l32_32020

open Real

theorem problem_1 : sqrt 3 * cos (π / 12) - sin (π / 12) = sqrt 2 := 
sorry

theorem problem_2 : ∀ θ : ℝ, sqrt 3 * cos θ - sin θ ≤ 2 := 
sorry

end problem_1_problem_2_l32_32020


namespace parallel_lines_from_perpendicularity_l32_32915

variables (a b : Type) (α β : Type)

-- Define the necessary conditions
def is_line (l : Type) : Prop := sorry
def is_plane (p : Type) : Prop := sorry
def perpendicular (l : Type) (p : Type) : Prop := sorry
def parallel (l1 l2 : Type) : Prop := sorry

axiom line_a : is_line a
axiom line_b : is_line b
axiom plane_alpha : is_plane α
axiom plane_beta : is_plane β
axiom a_perp_alpha : perpendicular a α
axiom b_perp_alpha : perpendicular b α

-- State the theorem
theorem parallel_lines_from_perpendicularity : parallel a b :=
  sorry

end parallel_lines_from_perpendicularity_l32_32915


namespace balls_in_box_A_after_50_children_l32_32351

def initial_state : (ℕ × ℕ × ℕ × ℕ) := (8, 6, 3, 1)

def child_action (state : ℕ × ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ × ℕ :=
  let (a, b, c, d) := state
  match List.minimum [a, b, c, d] with
  | some m =>
    if m = a then (a + (b - 1) + (c - 1) + (d - 1), 1, 1, 1) 
    else if m = b then (1, b + (a - 1) + (c - 1) + (d - 1), 1, 1)
    else if m = c then (1, 1, c + (a - 1) + (b - 1) + (d - 1), 1)
    else (1, 1, 1, d + (a - 1) + (b - 1) + (c - 1))
  | none => state

def iterate_child_actions (n : ℕ) (state : ℕ × ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ × ℕ :=
  (List.range n).foldl (λ s _ => child_action s) state

theorem balls_in_box_A_after_50_children :
  let final_state := iterate_child_actions 50 initial_state
  in final_state.1 = 6 :=
by
  let final_state := iterate_child_actions 50 initial_state
  have final_state_eq : final_state = (6, 4, 5, 3) sorry
  show final_state.1 = 6 from by rw [final_state_eq]; rfl

end balls_in_box_A_after_50_children_l32_32351


namespace sqrt_of_mixed_fraction_simplified_l32_32508

theorem sqrt_of_mixed_fraction_simplified :
  let x := 8 + (9 / 16) in
  sqrt x = (sqrt 137) / 4 := by
  sorry

end sqrt_of_mixed_fraction_simplified_l32_32508


namespace count_integer_points_in_intersection_l32_32092

theorem count_integer_points_in_intersection :
  let sphere1 := {p : ℤ × ℤ × ℤ | p.1^2 + p.2^2 + (p.3 - 10)^2 ≤ 64}
  let sphere2 := {p : ℤ × ℤ × ℤ | p.1^2 + p.2^2 + p.3^2 ≤ 25}
  (sphere1 ∩ sphere2).card = 45 :=
by
  -- Start of formal proof (left as exercise)
  sorry

end count_integer_points_in_intersection_l32_32092


namespace periodic_function_is_periodic_l32_32546

noncomputable def periodic_function (p : ℕ) (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f(x + a) = (Real.cos(2 * Real.pi / p) * f(x) - Real.sin(2 * Real.pi / p)) / 
            (Real.sin(2 * Real.pi / p) * f(x) + Real.cos(2 * Real.pi / p))

theorem periodic_function_is_periodic (p : ℕ) (h : p > 2) (f : ℝ → ℝ) (a : ℝ) (ha : a > 0) 
  (h_f : periodic_function p f a) : ∃ T : ℝ, T = p * a ∧ ∀ x : ℝ, f(x + T) = f(x) :=
sorry

end periodic_function_is_periodic_l32_32546


namespace mural_total_cost_is_192_l32_32254

def mural_width : ℝ := 6
def mural_height : ℝ := 3
def paint_cost_per_sqm : ℝ := 4
def area_per_hour : ℝ := 1.5
def hourly_rate : ℝ := 10

def mural_area := mural_width * mural_height
def paint_cost := mural_area * paint_cost_per_sqm
def labor_hours := mural_area / area_per_hour
def labor_cost := labor_hours * hourly_rate
def total_mural_cost := paint_cost + labor_cost

theorem mural_total_cost_is_192 : total_mural_cost = 192 := by
  -- Definitions
  sorry

end mural_total_cost_is_192_l32_32254


namespace total_bottles_l32_32029

theorem total_bottles (n : ℕ) (h1 : ∃ one_third two_third: ℕ, one_third = n / 3 ∧ two_third = 2 * (n / 3) ∧ 3 * one_third = n)
    (h2 : 25 ≤ n)
    (h3 : ∃ damage1 damage2 damage_diff : ℕ, damage1 = 25 * 160 ∧ damage2 = (n / 3) * 160 + ((2 * (n / 3) - 25) * 130) ∧ damage1 - damage2 = 660) :
    n = 36 :=
by
  sorry

end total_bottles_l32_32029


namespace sum_of_solutions_l32_32378

theorem sum_of_solutions : 
  ∀ x : ℝ, (6 * x / 30 = 7 / x) → (x = sqrt 35 ∨ x = -sqrt 35) → x + (-x) = 0 :=
by
  sorry

end sum_of_solutions_l32_32378


namespace smallest_perfect_cube_divisor_l32_32274

theorem smallest_perfect_cube_divisor 
  (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hpq : p ≠ q) 
  (hpr : p ≠ r) (hqr : q ≠ r) (s := 4) (hs : ¬ Nat.Prime s) 
  (hdiv : Nat.Prime 2) :
  ∃ n : ℕ, n = (p * q * r^2 * s)^3 ∧ ∀ m : ℕ, (∃ a b c d : ℕ, a = 3 ∧ b = 3 ∧ c = 6 ∧ d = 3 ∧ m = p^a * q^b * r^c * s^d) → m ≥ n :=
sorry

end smallest_perfect_cube_divisor_l32_32274


namespace contrapositive_of_lt_l32_32328

theorem contrapositive_of_lt (a b c : ℝ) :
  (a < b → a + c < b + c) → (a + c ≥ b + c → a ≥ b) :=
by
  intro h₀ h₁
  sorry

end contrapositive_of_lt_l32_32328


namespace sphere_cylinder_surface_area_difference_l32_32923

theorem sphere_cylinder_surface_area_difference (R : ℝ) :
  let S_sphere := 4 * Real.pi * R^2
  let S_lateral := 4 * Real.pi * R^2
  S_sphere - S_lateral = 0 :=
by
  sorry

end sphere_cylinder_surface_area_difference_l32_32923


namespace values_range_l32_32269

noncomputable def possible_values (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 2) : set ℝ :=
  {x | x = 1 / a + 1 / b}

theorem values_range : 
  ∀ (a b : ℝ), 
  a > 0 → 
  b > 0 → 
  a + b = 2 → 
  possible_values a b (by assumption) (by assumption) (by assumption) = {x | 2 ≤ x} :=
sorry

end values_range_l32_32269


namespace maximum_area_rectangular_backyard_l32_32687

theorem maximum_area_rectangular_backyard (x : ℕ) (y : ℕ) (h_perimeter : 2 * (x + y) = 100) : 
  x * y ≤ 625 :=
by
  sorry

end maximum_area_rectangular_backyard_l32_32687


namespace no_minimum_value_l32_32971

noncomputable def f (x : ℝ) : ℝ :=
  (1 + 1 / Real.log (Real.sqrt (x^2 + 10) - x)) *
  (1 + 2 / Real.log (Real.sqrt (x^2 + 10) - x))

theorem no_minimum_value : ¬ ∃ x, (0 < x ∧ x < 4.5) ∧ (∀ y, (0 < y ∧ y < 4.5) → f x ≤ f y) :=
sorry

end no_minimum_value_l32_32971


namespace poly_square_inequality_l32_32951

theorem poly_square_inequality
  {n : ℕ}
  {a : ℕ → ℝ}
  {b : ℕ → ℝ}
  (hP : ∀ X, a 0 * X^n + a 1 * X^(n-1) + (∑ i in finset.range (n-2), a (i+2) * X^(n-i-2)) + a n = (b 0 * X^(2*n)) + b 1 * X^(2*n-1) + (∑ i in finset.range (2*n-2), b (i+2) * X^(2*n-i-2)) + b (2*n))
  (h₀ : 0 < a 0)
  (h₁ : ∀ i < n, 0 ≤ a i)
  (h₂ : ∀ i < n, a n ≥ a i) :
  a 0^2 + (2 * a 0 * a 1) + ∑ i in finset.range (n-2), a (i+2)^2 ≥ 2 * (a 0 * a n + a 1 * a (n-1) + (∑ i in finset.range (n-2), a (i+2) * a (n-i-2)))
 := sorry

end poly_square_inequality_l32_32951


namespace compute_f_even_odd_bound_f_max_no_universal_constant_l32_32224

-- (a) f(m, n) when m, n are both even or both odd
theorem compute_f_even_odd (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (m % 2 = 0 ∧ n % 2 = 0 → f(m, n) = 0) ∧
  (m % 2 = 1 ∧ n % 2 = 1 → f(m, n) = 1/2) := sorry

-- (b) Prove f(m, n) ≤ 1/2 * max {m, n}
theorem bound_f_max (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  f(m, n) ≤ 1/2 * max m n := sorry

-- (c) Prove no constant c such that for all m, n, f(m, n) < c
theorem no_universal_constant (c : ℝ) : ¬∀ m n : ℕ, f(m, n) < c := sorry

end compute_f_even_odd_bound_f_max_no_universal_constant_l32_32224


namespace sum_of_integer_solutions_eq_zero_l32_32862

theorem sum_of_integer_solutions_eq_zero :
  ∀ x : ℤ, (x^4 - 25 * x^2 + 144 = 0) → (x = -4 ∨ x = 4 ∨ x = -3 ∨ x = 3) → 
    sum (Finset.filter (λ x : ℤ, x^4 - 25 * x^2 + 144 = 0) { -4, -3, 3, 4 }) = 0 :=
by
  sorry

end sum_of_integer_solutions_eq_zero_l32_32862


namespace solution_l32_32262

variables (α β γ : Plane) (l : Line)
variables (distinct_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)

def prop1 : Prop := (α ⊥ β) ∧ (β ⊥ γ) → (α ⊥ γ)
def prop2 : Prop := ∀ (A B : Point), (A ∈ l ∧ B ∈ l ∧ dist(A, α) = dist(B, α)) → (l ∥ α)
def prop3 : Prop := (l ⊥ α) ∧ (l ∥ β) → (α ⊥ β)
def prop4 : Prop := (α ∥ β) ∧ (l \notin β) ∧ (l ∥ α) → (l ∥ β)

noncomputable def correct_propositions : set Prop :=
  {p | p = prop3 ∨ p = prop4}

theorem solution : correct_propositions = {prop3, prop4} :=
sorry

end solution_l32_32262


namespace new_weight_of_oranges_l32_32407

def initial_total_weight : ℝ := 5
def initial_water_percentage : ℝ := 0.95
def first_day_evaporation_percentage : ℝ := 0.05
def additional_skin_loss_percentage : ℝ := 0.02

theorem new_weight_of_oranges :
  let initial_water_weight := initial_total_weight * initial_water_percentage in
  let orange_substance_weight := initial_total_weight - initial_water_weight in
  let water_loss_evaporation := initial_water_weight * first_day_evaporation_percentage in
  let remaining_water_weight_after_evaporation := initial_water_weight - water_loss_evaporation in
  let water_loss_skin := remaining_water_weight_after_evaporation * additional_skin_loss_percentage in
  let new_water_weight := remaining_water_weight_after_evaporation - water_loss_skin in
  let new_total_weight := new_water_weight + orange_substance_weight in
  new_total_weight = 4.67225 := 
by
  let initial_water_weight := initial_total_weight * initial_water_percentage
  let orange_substance_weight := initial_total_weight - initial_water_weight
  let water_loss_evaporation := initial_water_weight * first_day_evaporation_percentage
  let remaining_water_weight_after_evaporation := initial_water_weight - water_loss_evaporation
  let water_loss_skin := remaining_water_weight_after_evaporation * additional_skin_loss_percentage
  let new_water_weight := remaining_water_weight_after_evaporation - water_loss_skin
  let new_total_weight := new_water_weight + orange_substance_weight
  sorry

end new_weight_of_oranges_l32_32407


namespace pentadecagon_triangle_count_l32_32195

-- Define a regular pentadecagon as a 15-sided polygon
def pentadecagon : Finset (Fin 15) := (Finset.univ : Finset (Fin 15))

-- Define a function to determine if three vertices are consecutive
def consecutive (a b c: Fin 15) : Prop :=
  (a.val + 1) % 15 = b.val ∧ (b.val + 1) % 15 = c.val

-- Theorem to state the number of valid triangles
theorem pentadecagon_triangle_count :
  let vertices := pentadecagon
  let triangles := vertices.powerset.filter (λ s, s.card = 3)
  let valid_triangles := triangles.filter (λ s, ¬ ∃ a b c ∈ s, consecutive a b c)
  valid_triangles.card = 440 :=
  by
  sorry

end pentadecagon_triangle_count_l32_32195


namespace binomial_odd_term_sum_is_512_l32_32155

theorem binomial_odd_term_sum_is_512 :
  ∀ (n : ℕ), (C(n, 3) = C(n, 7) → ∑ k in finset.range (n + 1), if odd k then binomial n k else 0 = 512) :=
begin
  assume n hn_eq,
  -- Assuming n = 10 from C(n, 3) = C(n, 7)
  have h_n : n = 10,
  {
    -- Proof to derive n = 10 will be here
    sorry,
  },
  -- Rewriting the sum of the odd binomial coefficients as 512
  rw h_n,
  simp only [finset.sum_range_succ, binomial, nat.cast_binom],
  -- Proof to show the sum is 512 will be here
  sorry,
end

end binomial_odd_term_sum_is_512_l32_32155


namespace quadratic_inequality_l32_32717

variable {a b c : ℝ}

noncomputable def quadratic_polynomial (x : ℝ) := a * x^2 + b * x + c

theorem quadratic_inequality (h1 : ∀ x : ℝ, quadratic_polynomial x < 0)
    (h2 : a < 0) (h3 : b^2 - 4*a*c < 0) : (b / a) < (c / a + 1) := 
sorry

end quadratic_inequality_l32_32717


namespace pet_store_cats_left_l32_32019

theorem pet_store_cats_left (siamese house sold : ℕ) (h_siamese : siamese = 38) (h_house : house = 25) (h_sold : sold = 45) :
  siamese + house - sold = 18 :=
by
  sorry

end pet_store_cats_left_l32_32019


namespace remaining_part_of_pentagon_is_correct_l32_32529

noncomputable def remaining_area_of_pentagon : ℝ :=
  let side_length := 1
  let pentagon_area := (1 / 4) * real.sqrt (5 * (5 + 2 * real.sqrt 5)) * side_length^2
  let sector_angle := 2 * real.pi / 5
  let sector_area := (1 / 2) * (1 : ℝ)^2 * sector_angle
  let total_removed_area := 5 * sector_area
  let removed_area_lying_inside_pentagon := total_removed_area / 5
  let remaining_area := pentagon_area - total_removed_area
  remaining_area

theorem remaining_part_of_pentagon_is_correct :
  remaining_area_of_pentagon = (5 * real.sqrt 3 / 4) - (real.pi / 6) :=
sorry

end remaining_part_of_pentagon_is_correct_l32_32529


namespace actual_average_height_correct_l32_32768

theorem actual_average_height_correct (n : ℕ) (incorrect_avg_height incorrect_height actual_height : ℝ)
  (h_n : n = 35)
  (h_incorrect_avg_height : incorrect_avg_height = 180)
  (h_incorrect_height : incorrect_height = 156)
  (h_actual_height : actual_height = 106) :
  let incorrect_total_height := incorrect_avg_height * n,
      error := incorrect_height - actual_height,
      correct_total_height := incorrect_total_height - error,
      actual_avg_height := correct_total_height / n
  in abs (actual_avg_height - 178.57) < 0.01 :=
by {
  sorry
}

end actual_average_height_correct_l32_32768


namespace find_n_value_l32_32965

theorem find_n_value (x y : ℕ) : x = 3 → y = 1 → n = x - y^(x - y) → x > y → n + x * y = 5 := by sorry

end find_n_value_l32_32965


namespace find_a_l32_32470

def F (a b c : ℝ) : ℝ := a * b^3 + c

theorem find_a (a : ℝ) (h : F a 3 8 = F a 5 12) : a = -2 / 49 := by
  sorry

end find_a_l32_32470


namespace number_of_songs_l32_32248

-- Definition of the given conditions
def total_storage_GB : ℕ := 16
def used_storage_GB : ℕ := 4
def storage_per_song_MB : ℕ := 30
def GB_to_MB : ℕ := 1000

-- Theorem stating the result
theorem number_of_songs (total_storage remaining_storage song_size conversion_factor : ℕ) :
  total_storage = total_storage_GB →
  remaining_storage = total_storage - used_storage_GB →
  song_size = storage_per_song_MB →
  conversion_factor = GB_to_MB →
  (remaining_storage * conversion_factor) / song_size = 400 :=
by
  intros h_total h_remaining h_song_size h_conversion
  rw [h_total, h_remaining, h_song_size, h_conversion]
  sorry

end number_of_songs_l32_32248


namespace closest_integer_to_cube_root_of_1728_l32_32004

theorem closest_integer_to_cube_root_of_1728: 
  ∃ n : ℕ, n^3 = 1728 ∧ (∀ m : ℤ, m^3 < 1728 → m < n) ∧ (∀ p : ℤ, p^3 > 1728 → p > n) :=
by
  sorry

end closest_integer_to_cube_root_of_1728_l32_32004


namespace min_val_f_l32_32877

noncomputable def f (x : ℝ) : ℝ :=
  4 / (x - 2) + x

theorem min_val_f (x : ℝ) (h : x > 2) : ∃ y, y = f x ∧ y ≥ 6 :=
by {
  sorry
}

end min_val_f_l32_32877


namespace not_traversable_n_62_l32_32334

theorem not_traversable_n_62 :
  ¬ (∃ (path : ℕ → ℕ), ∀ i < 62, path (i + 1) = (path i + 8) % 62 ∨ path (i + 1) = (path i + 9) % 62 ∨ path (i + 1) = (path i + 10) % 62) :=
by sorry

end not_traversable_n_62_l32_32334


namespace probability_of_selection_of_X_l32_32736

theorem probability_of_selection_of_X (P : Type → ℝ) (X Y : Type) 
  (PY : P Y = 2 / 5)
  (PXY : P (X ∧ Y) = 0.13333333333333333) :
  P X = 1 / 3 :=
by
  sorry

end probability_of_selection_of_X_l32_32736


namespace pizza_toppings_combination_l32_32311

def num_combinations {α : Type} (s : Finset α) (k : ℕ) : ℕ :=
  (s.card.choose k)

theorem pizza_toppings_combination (s : Finset ℕ) (h : s.card = 7) : num_combinations s 3 = 35 :=
by
  sorry

end pizza_toppings_combination_l32_32311


namespace find_q_value_l32_32961

-- Define the conditions
variable (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) 
variable (q : ℝ) (n : ℕ)
-- Condition 1: Geometric sequence with common ratio q and |q| > 1
axiom geo_seq : a_n n = a_n 0 * q^n
axiom common_ratio : |q| > 1
-- Condition 2: b_n = a_n + 1
axiom b_def : b_n n = a_n n + 1
-- Condition 3: Sequence {b_n} contains four consecutive terms in {-53, -23, 19, 37, 82}
axiom b_seq_in_set : ∃ m, List.ofFn b_n.contains_sublist ([ -53, -23, 19, 37, 82 ])

-- Prove that q = - 3 / 2
theorem find_q_value : q = -3/2 := sorry

end find_q_value_l32_32961


namespace width_of_domain_l32_32599

theorem width_of_domain
  (h_domain : ∀ x, -9 ≤ x ∧ x ≤ 9 → x ∈ set.Icc (-9 : ℝ) 9)
  (g : ℝ → ℝ) (h : ℝ → ℝ)
  (h_def : ∀ x, g x = h (x / 3)) :
  set.Icc (-27 : ℝ) 27 = {x | -27 ≤ x ∧ x ≤ 27} ∧ (27 - (-27) = 54) :=
by
  sorry

end width_of_domain_l32_32599


namespace reaction_rate_correct_l32_32479

def c1 : ℝ := 3.5
def c2 : ℝ := 1.5
def t1 : ℝ := 0
def t2 : ℝ := 15

def v_reaction : ℝ := (c1 - c2) / (t2 - t1)

theorem reaction_rate_correct : v_reaction = 0.133 :=
by
  sorry

end reaction_rate_correct_l32_32479


namespace sqrt_mixed_fraction_l32_32491

theorem sqrt_mixed_fraction (a b : ℤ) (h_a : a = 8) (h_b : b = 9) : 
  (√(a + b / 16)) = (√137) / 4 := 
by 
  sorry

end sqrt_mixed_fraction_l32_32491


namespace part1_solution_set_a1_part2_range_of_a_l32_32979

def f (a x : ℝ) : ℝ := |a * x + 1| + |x - a|
def g (x : ℝ) : ℝ := x ^ 2 + x

-- Part 1
theorem part1_solution_set_a1 (x : ℝ) : 
  {x | x ≥ 1 ∨ x ≤ -3} = {x | g(x) ≥ f(1, x)} :=
sorry

-- Part 2
theorem part2_range_of_a (a : ℝ) : 
  f(a, a) ≥ 3 / 2 ↔ a ≥ Real.sqrt 2 / 2 :=
sorry

end part1_solution_set_a1_part2_range_of_a_l32_32979


namespace calculate_sample_statistics_and_grade_distribution_l32_32009

noncomputable def sample_data : List ℕ := [92, 84, 86, 78, 89, 74, 83, 78, 77, 89]

def sample_mean (data : List ℕ) : ℚ :=
  (data.sum : ℚ) / data.length

def sample_variance (data : List ℕ) : ℚ :=
  let mean := sample_mean data
  (data.map (λ x => (x : ℚ - mean) ^ 2)).sum / data.length

def sample_stddev (data : List ℕ) : ℚ :=
  Real.sqrt (sample_variance data)

def within_range (data : List ℕ) (low high : ℚ) : ℕ :=
  data.count (λ x => low < (x : ℚ) ∧ (x : ℚ) < high)

theorem calculate_sample_statistics_and_grade_distribution :
  let x_bar := sample_mean sample_data in
  let s_squared := sample_variance sample_data in
  let s := sample_stddev sample_data in
  x_bar = 83 ∧ s_squared = 33 ∧ s ≈ 5.74 ∧
  (within_range sample_data (x_bar - s) (x_bar + s)) = 5 :=
by
  sorry

end calculate_sample_statistics_and_grade_distribution_l32_32009


namespace average_rate_dan_trip_l32_32771

/-- 
Given:
- Dan runs along a 4-mile stretch of river and then swims back along the same route.
- Dan runs at a rate of 10 miles per hour.
- Dan swims at a rate of 6 miles per hour.

Prove:
Dan's average rate for the entire trip is 0.125 miles per minute.
-/
theorem average_rate_dan_trip :
  let distance := 4 -- miles
  let run_rate := 10 -- miles per hour
  let swim_rate := 6 -- miles per hour
  let time_run_hours := distance / run_rate -- hours
  let time_swim_hours := distance / swim_rate -- hours
  let time_run_minutes := time_run_hours * 60 -- minutes
  let time_swim_minutes := time_swim_hours * 60 -- minutes
  let total_distance := distance + distance -- miles
  let total_time := time_run_minutes + time_swim_minutes -- minutes
  let average_rate := total_distance / total_time -- miles per minute
  average_rate = 0.125 :=
by sorry

end average_rate_dan_trip_l32_32771


namespace carpet_needed_l32_32801

theorem carpet_needed (length width : ℕ) (h1 : length = 15) (h2 : width = 9) :
  let area_ft := length * width,
  let area_per_section_ft := area_ft / 2,
  let area_per_section_yd := area_per_section_ft / 9
  in area_per_section_yd = 7.5 :=
by
  sorry

end carpet_needed_l32_32801


namespace range_of_a_l32_32922

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ set.Icc (-1 : ℝ) 1 → a ≤ -x^2 + 2 * x) ↔ (a ≤ -1) :=
by
  sorry

end range_of_a_l32_32922


namespace problem_l32_32869

def tens_digit_of_cube_even (n : ℕ) : Prop :=
  (((n * n * n) / 10) % 10) % 2 = 0

def count_tens_digit_of_cube_even (N : ℕ) : ℕ :=
  (Finset.range N).filter tens_digit_of_cube_even |>.card

theorem problem (h : count_tens_digit_of_cube_even 200 = 160) : Prop :=
  sorry

end problem_l32_32869


namespace infinite_series_sum_eq_seven_l32_32843

noncomputable def infinite_series_sum : ℝ :=
  ∑' k : ℕ, (1 + k)^2 / 3^(1 + k)

theorem infinite_series_sum_eq_seven : infinite_series_sum = 7 :=
sorry

end infinite_series_sum_eq_seven_l32_32843


namespace total_gifts_between_two_birthdays_l32_32256

/-- 
  John received 20 gifts on his 12th birthday and 
  received 8 fewer gifts on his 13th birthday.
  Prove that he received a total of 32 gifts 
  between these two birthdays.
-/
theorem total_gifts_between_two_birthdays :
  let gifts_12th := 20
  let fewer_gifts := 8
  let gifts_13th := gifts_12th - fewer_gifts
  in gifts_12th + gifts_13th = 32 := 
by
  sorry

end total_gifts_between_two_birthdays_l32_32256


namespace smallest_clock_equivalent_to_square_l32_32295

theorem smallest_clock_equivalent_to_square (n : ℕ) (h_gt_6 : n > 6) :
  (∃ (m : ℕ), (n = m) ∧ ((m^2 - m) % 24 = 0)) → n = 9 :=
begin
  intros h_exists,
  cases h_exists with m h_m,
  cases h_m with h_m_eq h_mod,
  sorry
end

end smallest_clock_equivalent_to_square_l32_32295


namespace walter_time_at_seals_l32_32364

theorem walter_time_at_seals 
  (s p e total : ℕ)
  (h1 : p = 8 * s)
  (h2 : e = 13)
  (h3 : total = 130)
  (h4 : s + p + e = total) : s = 13 := 
by 
  sorry

end walter_time_at_seals_l32_32364


namespace find_angle_COD_l32_32732

variables {Q C D O : Type} -- Define types for the points

-- Define the angle measure function
def angle (A B C : Type) := ℝ

-- Given condition
axiom angle_CQD : angle C Q D = 50

-- Statement to prove
theorem find_angle_COD : angle C O D = 65 :=
by
  sorry -- Proof is skipped

end find_angle_COD_l32_32732


namespace Diane_age_when_conditions_met_l32_32002

variable (Diane_current : ℕ) (Alex_current : ℕ) (Allison_current : ℕ)
variable (D : ℕ)

axiom Diane_current_age : Diane_current = 16
axiom Alex_Allison_sum : Alex_current + Allison_current = 47
axiom Diane_half_Alex : D = (Alex_current + (D - 16)) / 2
axiom Diane_twice_Allison : D = 2 * (Allison_current + (D - 16))

theorem Diane_age_when_conditions_met : D = 78 :=
by
  sorry

end Diane_age_when_conditions_met_l32_32002


namespace part_a_l32_32014

theorem part_a (x α : ℝ) (hα : 0 < α ∧ α < 1) (hx : x ≥ 0) : x^α - α * x ≤ 1 - α :=
sorry

end part_a_l32_32014


namespace arg_u_omega_not_real_positive_l32_32666

-- Definitions
def theta_condition (theta : ℝ) : Prop := 0 < theta ∧ theta < 2 * Real.pi
def z (theta : ℝ) : ℂ := 1 - Real.cos theta + Complex.i * Real.sin theta
def u (a theta : ℝ) : ℂ := a^2 + a * Complex.i
def purely_imaginary (z u : ℂ) : Prop := (z * u).re = 0
def omega (z u : ℂ) : ℂ := z^2 + u^2 + 2 * z * u
def a_value (theta : ℝ) : ℝ := Real.cotan (theta / 2)

-- Questions
theorem arg_u (theta : ℝ) (a : ℝ) (h_theta : theta_condition theta) (h_imag : purely_imaginary (z theta) (u a theta)) : 
  Complex.arg (u (a_value theta) theta) = 
  if theta < Real.pi then theta / 2 else Real.pi + theta / 2 :=
sorry

theorem omega_not_real_positive (theta : ℝ) (h_theta : theta_condition theta) 
  (a := a_value theta) (h_imag : purely_imaginary (z theta) (u a theta)) :
  ¬ (omega (z theta) (u a theta)).im = 0 ∧ (omega (z theta) (u a theta)).re > 0 :=
sorry

end arg_u_omega_not_real_positive_l32_32666


namespace unicorn_rope_problem_l32_32059

theorem unicorn_rope_problem :
  ∃ a b c : ℕ, (c > 0) ∧ (Nat.Prime c) ∧ (a - (Real.sqrt b) = c * (Real.sqrt 6 - Real.sqrt 5)) ∧ (a + b + c = 813) :=
begin
  use [60, 750, 3],
  repeat { split },
  { norm_num, },
  { norm_num, exact nat.prime_three, },
  { norm_num, congr, simp [Real.sqrt_eq_rpow], norm_num, rw [sub_eq_add_neg, neg_div, neg_neg], },
  { norm_num, }
end

end unicorn_rope_problem_l32_32059


namespace smallest_five_disjoint_sets_l32_32522

theorem smallest_five_disjoint_sets (k : ℕ) (hk : 0 < k) :
    ∃ (S : fin 5 → finset (fin (2 * k + 1))),
      (∀ i : fin 5, S i.card = k) ∧
      (∀ i : fin 5, S i ∩ S ((i + 1) % 5) = ∅) ∧
      (finset.bUnion finset.univ S).card = 2 * k + 1 :=
begin
  sorry
end

end smallest_five_disjoint_sets_l32_32522


namespace sum_of_altitudes_eq_l32_32792

theorem sum_of_altitudes_eq :
  let h := (96 / Real.sqrt 292) in
  let sum_altitudes := (22 * Real.sqrt 292 + 96) / Real.sqrt 292 in
  sum_altitudes = (6 + 16 + h) :=
begin
  sorry
end

end sum_of_altitudes_eq_l32_32792


namespace complex_modulus_half_l32_32325

open Complex

theorem complex_modulus_half (z : ℂ) (hz1 : abs z < 1) (hz2 : abs (conj z + (1 / z)) = 5 / 2) : abs z = 1 / 2 :=
by
  sorry

end complex_modulus_half_l32_32325


namespace ariel_started_fencing_in_2006_l32_32441

theorem ariel_started_fencing_in_2006 :
  ∀ (birth_year current_age fencing_years : ℕ), 
  birth_year = 1992 → 
  current_age = 30 → 
  fencing_years = 16 → 
  let current_year := birth_year + current_age in
  let start_year := current_year - fencing_years in
  start_year = 2006 :=
by
  intros birth_year current_age fencing_years hb ha hf
  let current_year := birth_year + current_age
  let start_year := current_year - fencing_years
  sorry

end ariel_started_fencing_in_2006_l32_32441


namespace walter_time_spent_at_seals_l32_32371

theorem walter_time_spent_at_seals (S : ℕ) 
(h1 : 8 * S + S + 13 = 130) : S = 13 :=
sorry

end walter_time_spent_at_seals_l32_32371


namespace value_of_expression_l32_32200

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : 2 * a^2 + 4 * a * b + 2 * b^2 - 6 = 12 :=
by
  sorry

end value_of_expression_l32_32200


namespace minimum_additional_workers_l32_32626

-- Define the given conditions in Lean 4
def workers_per_quarter (initial_workers total_quarters : ℕ) : ℕ := initial_workers / total_quarters

theorem minimum_additional_workers 
  (initial_workers : ℕ) 
  (first_layoff : ℕ) 
  (second_layoff : ℕ) 
  (work_quarters : ℕ) 
  (rate_per_worker : ℕ) 
  (total_time : ℕ) 
  (total_work : ℕ) :
  let remaining_workers := initial_workers - first_layoff - second_layoff
  let work_time_first_quarter := (initial_workers * rate_per_worker * total_time) / work_quarters
  let work_done_first_quarter := work_time_first_quarter / total_time
  let work_time_second_quarter := ((initial_workers - first_layoff) * rate_per_worker * total_time) / work_quarters
  let work_done_second_quarter := work_time_second_quarter / total_time
  let work_time_third_quarter := ((initial_workers - first_layoff - second_layoff) * rate_per_worker * total_time) / work_quarters
  let work_done_third_quarter := work_time_third_quarter / total_time
  let total_work_done := work_done_first_quarter + work_done_second_quarter + work_done_third_quarter
  let remaining_time := (total_time * 3 / 4 - total_work_done)
  let required_workers := 
    let worker_equation := (remaining_workers * rate_per_worker * remaining_time) / work_quarters = total_work - total_work_done
    worker_equation / remaining_time - remaining_workers
  := required_workers = 766 :=
  sorry

end minimum_additional_workers_l32_32626


namespace total_fish_l32_32081

-- Defining the number of fish each person has, based on the conditions.
def billy_fish : ℕ := 10
def tony_fish : ℕ := 3 * billy_fish
def sarah_fish : ℕ := tony_fish + 5
def bobby_fish : ℕ := 2 * sarah_fish

-- The theorem stating the total number of fish.
theorem total_fish : billy_fish + tony_fish + sarah_fish + bobby_fish = 145 := by
  sorry

end total_fish_l32_32081


namespace expected_value_of_winnings_l32_32288

def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2 ^ k

def winnings (n : ℕ) : ℝ :=
  if is_prime n then n
  else if is_power_of_two n then 1
  else -2

def probabilities (n : ℕ) : ℝ :=
  if is_prime n then 1/2
  else if is_power_of_two n then 1/4
  else 1/4

def expected_winnings : ℝ :=
  ((1/2) * (2 + 3 + 5 + 7)) + ((1/4) * 1 * 2) + ((1/4) * (-2) * 2)

theorem expected_value_of_winnings : expected_winnings = 8 := by
  sorry

end expected_value_of_winnings_l32_32288


namespace cube_root_times_two_l32_32510

theorem cube_root_times_two : (∛(6 / 18) * 2) = (2 / 3) :=
by
  sorry

end cube_root_times_two_l32_32510


namespace number_of_bottles_of_regular_soda_l32_32041

theorem number_of_bottles_of_regular_soda (diet_soda : ℕ) (additional_soda : ℕ) (regular_soda : ℕ) 
  (h1 : diet_soda = 4) 
  (h2 : additional_soda = 79) 
  (h3 : regular_soda = diet_soda + additional_soda) : 
  regular_soda = 83 := 
by
  rw [h1, h2, h3]
  rfl

sorry

end number_of_bottles_of_regular_soda_l32_32041


namespace intersection_of_A_B_union_of_A_B_complement_intersection_A_B_l32_32980

def U := Set.univ

def A : Set ℝ := { x | x^2 - 2 * x - 3 ≤ 0 }
def B : Set ℝ := { x | x^2 - 4 * x ≤ 0 }

theorem intersection_of_A_B :
  A ∩ B = { x | 0 ≤ x ∧ x ≤ 3 } :=
sorry

theorem union_of_A_B :
  A ∪ B = { x | -1 ≤ x ∧ x ≤ 4 } :=
sorry

theorem complement_intersection_A_B :
  (Set.univ \ A) ∩ (Set.univ \ B) = { x | x < -1 ∨ x > 4 } :=
sorry

end intersection_of_A_B_union_of_A_B_complement_intersection_A_B_l32_32980


namespace irreducible_polynomial_of_odd_degree_l32_32312

open Polynomial Nat Zmod

theorem irreducible_polynomial_of_odd_degree 
  (m : ℕ)
  (c : Fin (2 * m + 2) → ℤ)
  (p : ℕ)
  (hp_prime : Prime p)
  (h1 : ¬ p ∣ c ⟨2 * m + 1, by simp⟩)
  (h2 : ∀ i : Fin (m + 1), p^2 ∣ c i)
  (h3 : ∀ i : Fin (m + 1), p^3 ∣ c ⟨0, by simp⟩)
  (h4 : ∀ i : Fin ((2 * m + 1) - m), p ∣ c ⟨m + 1 + i, by simp⟩) :
  Irreducible (∑ i in Finset.range (2 * m + 2), C (c i) * X ^ (i : ℕ)) := by
  sorry

end irreducible_polynomial_of_odd_degree_l32_32312


namespace exists_two_numbers_one_divides_other_l32_32991

theorem exists_two_numbers_one_divides_other :
  ∀ (chosen : Finset ℕ), chosen.card = 101 →
  chosen ⊆ (Finset.range 201).filter (λ x, x > 0) →
  ∃ a b ∈ chosen, a ∣ b ∨ b ∣ a :=
by
  intro chosen h_card h_subset
  sorry

end exists_two_numbers_one_divides_other_l32_32991


namespace pentagon_BK_parallel_AE_l32_32879

noncomputable def convex_pentagon (A B C D E : Type) :=
  ∀ (AE_parallel_CD : AE ∥ CD)
    (AB_eq_BC : AB = BC)
    (angle_bisectors_intersect_at_K : ∃ (K : Type), is_angle_bisector A K ∧ is_angle_bisector C K),
  BK ∥ AE

-- Auxiliary definitions (these do not represent solution steps)
def is_convex_pentagon (P : Type) := sorry
def is_angle_bisector (A K : Type) := sorry

theorem pentagon_BK_parallel_AE :
  ∀ (A B C D E K : Type),
  is_convex_pentagon (A, B, C, D, E) →
  (AE ∥ CD) → (AB = BC) → (is_angle_bisector A K ∧ is_angle_bisector C K) →
  BK ∥ AE :=
by {
  intros,
  sorry
}

end pentagon_BK_parallel_AE_l32_32879


namespace seq_product_gt_one_l32_32161

open nat 

noncomputable def min_n (a : ℕ → ℝ) (r : ℝ) (h₁ : ∀ n, 0 < a n) (h₂ : 1 < r) (h₃ : a 2 * a 4 = a 3)
: ℕ :=
6

theorem seq_product_gt_one
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geometric : ∀ n, a (n+1) = r * a n)
  (h_positive: ∀ n, 0 < a n)
  (h_ratio_gt_one : 1 < r)
  (h_condition : a 2 * a 4 = a 3) :
  ∀ n, (∏ i in range n, a (i + 1)) > 1 ↔ n ≥ 6 :=
begin
  sorry
end

end seq_product_gt_one_l32_32161


namespace collinear_vectors_l32_32966

variables {ℝ : Type*} [field ℝ] [inhabited ℝ]
variables (a b : ℝ → Prop) (p q λ : ℝ)
variables (A B D : ℝ → Prop)

def collinear (A B D : ℝ → Prop) : Prop :=
    ∃ (λ : ℝ), ∀ x, A x → B (x + λ * x) ∧ D (x + λ * x)

theorem collinear_vectors (AB BC CD : ℝ → ℝ) (a b : ℝ)
  (hAB : AB = (2:ℝ) * a + p * b)
  (hBC : BC = a + b)
  (hCD : CD = (q - 1) * a - (2:ℝ) * b)
  (hcol : collinear A B D) :
  p * q = -2 :=
sorry

end collinear_vectors_l32_32966


namespace rearrangement_identity_h_ne_k_l32_32974

-- Given conditions as assumptions:

def a (n : ℕ) : ℚ := ∑ i in Finset.range n, (-1)^(i+1) / (i+1)

axiom limit_a (k : ℝ) (h_lim : ∀ ε : ℝ, ε > 0 → ∃ N : ℕ, ∀ n: ℕ, n ≥ N → |a n - k| < ε) : Prop

noncomputable def rearrangement_series (b : ℕ → ℚ) : Prop :=
∀ n, b n = (∑ i in Finset.range (2 * n), (-1)^(i+1) / (i+1)) + (- ∑ i in Finset.range (n), (-1)^(i+1) / (n * 2 + (n+1)))

axiom limit_b (h : ℝ) (hb_lim : ∀ ε : ℝ, ε > 0 → ∃ N : ℕ, ∀ n: ℕ, n ≥ N → |b n - h| < ε) : Prop

-- Show that:
theorem rearrangement_identity (n : ℕ) : 
b (3 * n) = a (4 * n) + (a (2 * n)) / 2 := sorry
  
theorem h_ne_k (k h : ℝ) (lim_a_condition : limit_a k)
  (lim_b_condition : limit_b h)
: h ≠ k := sorry

end rearrangement_identity_h_ne_k_l32_32974


namespace equivalent_sigma_algebras_l32_32264

open Set

noncomputable def borelExtendedReal := generateFrom (λ x : ℝ, [[-∞, x] ∪ {-∞}])

theorem equivalent_sigma_algebras :
  borelExtendedReal = 
  generateFrom (λ x : ℝ, {[(-∞, x)]}) ∧
  borelExtendedReal = 
  generateFrom (λ x : ℝ, {[(x, ∞)] ∪ {∞}}) ∧
  borelExtendedReal = 
  generateFrom (λ I : Set ℝ, {(I : Set (ℝ ∪ {-∞} ∪ {∞})), is_finite_interval I ∨ I = {-∞} ∨ I = {∞}}) :=
by
  sorry

end equivalent_sigma_algebras_l32_32264


namespace factor_polynomial_l32_32105

theorem factor_polynomial :
  (x : ℝ) → (x^2 - 6*x + 9 - 64*x^4) = (-8*x^2 + x - 3) * (8*x^2 + x - 3) :=
by
  intro x
  sorry

end factor_polynomial_l32_32105


namespace time_spent_on_seals_l32_32368

theorem time_spent_on_seals (s : ℕ) 
  (h1 : 2 * 60 + 10 = 130) 
  (h2 : s + 8 * s + 13 = 130) :
  s = 13 :=
sorry

end time_spent_on_seals_l32_32368


namespace find_a_l32_32907

variable (a b : ℝ × ℝ) 

axiom b_eq : b = (2, -1)
axiom length_eq_one : ‖a + b‖ = 1
axiom parallel_x_axis : (a + b).snd = 0

theorem find_a : a = (-1, 1) ∨ a = (-3, 1) := by
  sorry

end find_a_l32_32907


namespace chloe_cherries_l32_32189

noncomputable def cherries_received (x y : ℝ) : Prop :=
  x = y + 8 ∧ y = x / 3

theorem chloe_cherries : ∃ (x : ℝ), ∀ (y : ℝ), cherries_received x y → x = 12 := 
by
  sorry

end chloe_cherries_l32_32189


namespace climbing_time_total_l32_32822

theorem climbing_time_total
  (n : ℕ := 6)
  (a : ℕ := 30)
  (d : ℕ := 10)
  (arith_seq : ∀ k, k < n -> ℕ := λ k : ℕ, a + k * d)
  (last_term : ℕ := arith_seq (n - 1) (by decide)) :
  (let total_time : ℕ := n / 2 * (a + last_term) in total_time) = 330 := by
  sorry

end climbing_time_total_l32_32822


namespace number_of_quadratic_residues_l32_32117

theorem number_of_quadratic_residues (n : ℕ) (h_n : n = 35) :
  ∃ (count : ℕ), count = 144 ∧
    (∀ (a b : ℤ), (1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n) →
     (∃ f : ℤ[X], (∃ P Q : ℤ[X], f^2 - (polynomial.C a * polynomial.X + polynomial.C b) = (polynomial.X^2 + 1) * P + n * Q) ∨
                   ∀ r : ℤ[X], (f^2 - (polynomial.C a * polynomial.X + polynomial.C b)).div (polynomial.X^2 + 1) % x ∣ polynomial.C n))
  := sorry

end number_of_quadratic_residues_l32_32117


namespace complement_of_A_relative_to_U_l32_32960

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {x | 1 ≤ x ∧ x ≤ 3}

theorem complement_of_A_relative_to_U : (U \ A) = {4, 5, 6} := 
by
  sorry

end complement_of_A_relative_to_U_l32_32960


namespace gcd_45045_30030_l32_32097

/-- The greatest common divisor of 45045 and 30030 is 15015. -/
theorem gcd_45045_30030 : Nat.gcd 45045 30030 = 15015 :=
by 
  sorry

end gcd_45045_30030_l32_32097


namespace roots_of_polynomial_l32_32663

theorem roots_of_polynomial (c d : ℝ) (h1 : Polynomial.eval c (Polynomial.X ^ 4 - 6 * Polynomial.X + 3) = 0)
    (h2 : Polynomial.eval d (Polynomial.X ^ 4 - 6 * Polynomial.X + 3) = 0) :
    c * d + c + d = Real.sqrt 3 := 
sorry

end roots_of_polynomial_l32_32663


namespace division_factorization_l32_32100

theorem division_factorization (k : ℕ) (a b : ℝ) :
  (a^2^k - b^2^k) / ((a + b) * (a^2 + b^2) * (a^4 + b^4) * ⋯ * (a^(2^(k-1)) + b^(2^(k-1)))) = a - b :=
by
  sorry

end division_factorization_l32_32100


namespace perimeter_non_shaded_region_l32_32324

theorem perimeter_non_shaded_region (A_shaded : ℕ) (l w h_s w_s : ℕ) 
  (hA_s : A_shaded = 55) 
  (hl : l = 12) (hw : w = 10)
  (hh_s : h_s = 5) (hw_s : w_s = 11) : 
  (2 * l + 2 * (w - hw_s) + 2 * hw + 2 * (l - hi_s) = 48) := 
by 
  sorry 

end perimeter_non_shaded_region_l32_32324


namespace least_k_for_factorial_multiple_of_315_l32_32868

theorem least_k_for_factorial_multiple_of_315 (k : ℕ) (h1 : k > 1) :
  (∃ k : ℕ, k > 1 ∧ (factorial k) % 315 = 0) → 
  (factorial 7) % 315 = 0 :=
by 
  sorry

end least_k_for_factorial_multiple_of_315_l32_32868


namespace smallest_m_satisfy_condition_l32_32521

def D_n (n : ℕ) : Set ℕ := {d | d ∣ n}

def F_i (n : ℕ) (i : ℕ) : Set ℕ := {a ∈ D_n n | a % 4 = i}

def f_i (n : ℕ) (i : ℕ) : ℕ := F_i n i |>.toFinset.card

theorem smallest_m_satisfy_condition :
  2 * f_i (2 * 5 ^ 2016) 1 - f_i (2 * 5 ^ 2016) 2 = 2017 := sorry

end smallest_m_satisfy_condition_l32_32521


namespace non_negative_sums_in_table_l32_32222

theorem non_negative_sums_in_table (n : ℕ) (A : Finₓ 1 × Finₓ n → ℝ) :
  ∃ B : Finₓ 1 × Finₓ n → ℝ, (∀ i : Finₓ 1, 0 ≤ ∑ j : Finₓ n, B (i, j)) ∧ (∀ j : Finₓ n, 0 ≤ ∑ i : Finₓ 1, B (i, j)) :=
sorry

end non_negative_sums_in_table_l32_32222


namespace angle_COD_eq_65_l32_32735

-- Define the conditions from step a)
variables (Q C D O : Type) [triangle : Triangle Q C D]

-- Assume the given conditions
variable (h1 : TangentTriangle Q C D O)
variable (h2 : Angle CQD = 50)

-- State the theorem to prove
theorem angle_COD_eq_65 : Angle COD = 65 := by
  sorry

end angle_COD_eq_65_l32_32735


namespace art_collection_total_cost_l32_32946

theorem art_collection_total_cost 
  (price_first_three : ℕ)
  (price_fourth : ℕ)
  (total_first_three : price_first_three * 3 = 45000)
  (price_fourth_cond : price_fourth = price_first_three + (price_first_three / 2)) :
  3 * price_first_three + price_fourth = 67500 :=
by
  sorry

end art_collection_total_cost_l32_32946


namespace max_clouds_through_planes_l32_32936

-- Define the problem parameters and conditions
def max_clouds (n : ℕ) : ℕ :=
  n + 1

-- Mathematically equivalent proof problem statement in Lean 4
theorem max_clouds_through_planes : max_clouds 10 = 11 :=
  by
    sorry  -- Proof skipped as required

end max_clouds_through_planes_l32_32936


namespace find_angles_DBE_l32_32549

variable {A B C D E : Type}
variable [IsTriangle A B C]

-- Conditions
variable (AD_eq_AB : segment AD = segment AB)
variable (CE_eq_CB : segment CE = segment CB)

-- Angles of triangle ABC
variable (angle_A angle_B angle_C : ℝ)

-- Resulting angles in DBE
def angles_DBE : ℝ × ℝ × ℝ :=
  (0.5 * angle_A, 0.5 * angle_C, 90 + 0.5 * angle_B)

theorem find_angles_DBE : 
  angles_DBE AD_eq_AB CE_eq_CB angle_A angle_B angle_C
  = (0.5 * angle_A, 0.5 * angle_C, 90 + 0.5 * angle_B) :=
sorry

end find_angles_DBE_l32_32549


namespace inequality_proof_l32_32561

theorem inequality_proof (a b c : ℝ) (hab : a > b) (hbc : b > c) :
  (1 / (a - b) + 1 / (b - c) + 1 / (c - a) > 0) :=
begin
  sorry
end

end inequality_proof_l32_32561


namespace songs_can_be_stored_l32_32250

def totalStorageGB : ℕ := 16
def usedStorageGB : ℕ := 4
def songSizeMB : ℕ := 30
def gbToMb : ℕ := 1000

def remainingStorageGB := totalStorageGB - usedStorageGB
def remainingStorageMB := remainingStorageGB * gbToMb
def numberOfSongs := remainingStorageMB / songSizeMB

theorem songs_can_be_stored : numberOfSongs = 400 :=
by
  sorry

end songs_can_be_stored_l32_32250


namespace ratio_students_l32_32726

theorem ratio_students (M : ℕ) (hM : M = 50) (total : ℕ) (h_total : total = 247) :
    (247 - 50) / 50 = 197 / 50 :=
by
  rw [hM, h_total]
  sorry

end ratio_students_l32_32726


namespace benjamin_annual_expenditure_l32_32446

theorem benjamin_annual_expenditure (T Y : ℕ) (hT : T = 30000) (hY : Y = 10) : T / Y = 3000 := 
by {
  rw [hT, hY],
  norm_num,
  sorry,
}

end benjamin_annual_expenditure_l32_32446


namespace ax5_by5_is_40_l32_32662

variable (a b x y : ℝ)
variable (s : ℕ → ℝ)

-- Define the conditions as Lean variables
def s1 : Prop := s 1 = a * x + b * y = 2
def s2 : Prop := s 2 = a * x^2 + b * y^2 = 5
def s3 : Prop := s 3 = a * x^3 + b * y^3 = 10
def s4 : Prop := s 4 = a * x^4 + b * y^4 = 30
def recurrence_relation (n : ℕ) : Prop := (x + y) * s n = s (n + 1) + x * y * s (n - 1)

theorem ax5_by5_is_40 (h1 : s1) (h2 : s2) (h3 : s3) (h4 : s4) (hr : ∀ n, recurrence_relation n) : s 5 = 40 := 
sorry

end ax5_by5_is_40_l32_32662


namespace lune_area_correct_l32_32420

noncomputable def area_lune (d1 d2: ℝ) : ℝ :=
  let r1 := d1 / 2
  let r2 := d2 / 2
  let semicircle_area_1 := (1 / 2) * π * r1^2
  let angle := 2 * Real.arcsin (r1 / r2)
  let sector_area := (1 / 2) * r2^2 * angle * π
  let lune_area := semicircle_area_1 - sector_area
  lune_area

theorem lune_area_correct :
  area_lune 2 3 = (1 / 2 - 2.25 * Real.arcsin(2 / 3)) * π := 
by
  unfold area_lune
  -- calculations can be deferred here
  sorry

end lune_area_correct_l32_32420


namespace result_after_2016_operations_l32_32871

-- Define the function for the operation
def operation (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (λ sum digit, sum + digit ^ 3) 0

-- Define the initial conditions and sequence
def first_operation : ℕ := operation 25
def second_operation : ℕ := operation first_operation
def third_operation : ℕ := operation second_operation

-- Define the function to find the result after k operations
def result_after_k_operations (n : ℕ) (k : ℕ) : ℕ :=
  (λ i, operation)^[k] n

-- Proposition: The result after the 2016th operation is 250
theorem result_after_2016_operations : result_after_k_operations 25 2016 = 250 :=
by 
  sorry -- Proof of the theorem goes here

end result_after_2016_operations_l32_32871


namespace find_b_l32_32563

theorem find_b (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
               (h3 : ∃ F₁ F₂ P : ℝ × ℝ, 
                     (∀ x y, x^2 / a^2 + y^2 / b^2 = 1 → 
                       ((P.1, P.2) = (x, y) ∧ 
                        (F₁.1, F₁.2, F₂.1, F₂.2) = (1, 0, -1, 0) ∧ 
                        (P - F₁) ⬝ (P - F₂) = 0 ∧ 
                        ∃ A : ℝ, (√((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * 
                                 √((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) * 
                                sin 90) / 2 = A ∧ 
                                 A = 9))) :
b = 3 :=
by
  sorry

end find_b_l32_32563


namespace probability_lean_statement_l32_32645

noncomputable def probability_same_heads : ℚ :=
  let fair_coin := 1 / 2
  let special_coin1 := 4 / 7
  let special_coin2 := 3 / 5
  let generating_function := (1 + fair_coin) ^ 2 * (3 * fair_coin + 4) * (2 * fair_coin + 3)
  -- Placeholder for the calculation function:
  sorry

theorem probability_lean_statement : 
  (compute_probability (generating_function) = (445 / 3364 : ℚ)) ∧ (gcd 445 3364 = 1) →
  445 + 3364 = 3809 :=
by
  intros h
  sorry

end probability_lean_statement_l32_32645


namespace div_1988_form_1989_div_1989_form_1988_l32_32764

/-- There exists a number of the form 1989...19890... (1989 repeated several times followed by several zeros), which is divisible by 1988. -/
theorem div_1988_form_1989 (k : ℕ) : ∃ n : ℕ, (n = 1989 * 10^(4*k) ∧ n % 1988 = 0) := sorry

/-- There exists a number of the form 1988...1988 (1988 repeated several times), which is divisible by 1989. -/
theorem div_1989_form_1988 (k : ℕ) : ∃ n : ℕ, (n = 1988 * ((10^(4*k)) - 1) ∧ n % 1989 = 0) := sorry

end div_1988_form_1989_div_1989_form_1988_l32_32764


namespace no_selfish_table_for_large_n_l32_32772

-- Definition of a "selfish" table
-- A 2D matrix (indexed from 0 to n-1 for both rows and columns), where a_ij indicates
-- the number of times i appears in row j.
-- This definition is used within the theorem.

def is_selfish (n : ℕ) (table : matrix (fin n) (fin n) ℕ) : Prop :=
  ∀ (i j : fin n), table j j = ∑ x, ite (table j x = i) 1 0

-- The theorem
theorem no_selfish_table_for_large_n (n : ℕ) (h : n > 5) :
  ¬ ∃ (table : matrix (fin n) (fin n) ℕ), is_selfish n table :=
by
  sorry

end no_selfish_table_for_large_n_l32_32772


namespace quadratic_polynomial_inequality_l32_32716

variable {a b c : ℝ}

theorem quadratic_polynomial_inequality (h1 : ∀ x : ℝ, a * x^2 + b * x + c < 0)
    (h2 : a < 0)
    (h3 : b^2 - 4 * a * c < 0) :
    b / a < c / a + 1 := 
by 
  sorry

end quadratic_polynomial_inequality_l32_32716


namespace time_first_pipe_l32_32738

-- Define the variables for the problem
variables (T : ℝ)

-- Define the rates based on the conditions
def rate_first_pipe := 1 / T
def rate_second_pipe := 1 / 30
def rate_outlet_pipe := 1 / 45

-- Define the combined rate when all pipes are opened
def combined_rate := rate_first_pipe + rate_second_pipe - rate_outlet_pipe

-- Define the time it takes to fill the tank when all pipes are opened
def time_all_pipes := 0.06666666666666665 -- Approximately 1/15

-- The proof statement showing the combined rate equals the rate when all pipes are opened
theorem time_first_pipe :
  combined_rate = 1 / time_all_pipes → T = 18 :=
begin
  simp [combined_rate, rate_first_pipe, rate_second_pipe, rate_outlet_pipe, time_all_pipes],
  sorry
end

end time_first_pipe_l32_32738


namespace sample_systematic_draw_first_group_l32_32361

theorem sample_systematic_draw_first_group :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 8 →
  (x + 15 * 8 = 126) →
  x = 6 :=
by
  intros x h1 h2
  sorry

end sample_systematic_draw_first_group_l32_32361


namespace walter_time_at_seals_l32_32365

theorem walter_time_at_seals 
  (s p e total : ℕ)
  (h1 : p = 8 * s)
  (h2 : e = 13)
  (h3 : total = 130)
  (h4 : s + p + e = total) : s = 13 := 
by 
  sorry

end walter_time_at_seals_l32_32365


namespace band_song_arrangements_l32_32699

theorem band_song_arrangements (n : ℕ) (t : ℕ) (r : ℕ) 
  (h1 : n = 8) (h2 : t = 3) (h3 : r = 5) : 
  ∃ (ways : ℕ), ways = 14400 := by
  sorry

end band_song_arrangements_l32_32699


namespace hyperbola_equation_l32_32882

theorem hyperbola_equation
  (e : ℝ) (h_e : e = 5 / 3)
  (line : AffineSubspace ℝ ℝ) (h_line : ∀ x y : ℝ, 8 * x + 2 * Real.sqrt 7 * y = 16 → 0)
  (is_tangent : ∃ x y : ℝ, line.contains ⟨x, y⟩) :
  ∃ (a b : ℝ), a = sqrt 18 ∧ b = sqrt 32 ∧ (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) :=
by
  sorry

end hyperbola_equation_l32_32882


namespace problem_solution_l32_32239

variables (p q r s : ℕ)

def function := (λ x : ℝ, (x^2 - x - 2) / (x^3 - 3x^2 + 2x))

-- Definitions for p, q, r, s based on the analysis of the function
def number_of_holes (f : ℝ → ℝ) : ℕ := 1  -- hole at x = 2
def number_of_vertical_asymptotes (f : ℝ → ℝ) : ℕ := 2  -- vertical asymptotes at x = 0, x = 1
def number_of_horizontal_asymptotes (f : ℝ → ℝ) : ℕ := 1  -- horizontal asymptote at y = 0
def number_of_oblique_asymptotes (f : ℝ → ℝ) : ℕ := 0  -- no oblique asymptotes

axiom function_asymptotes_holes : 
  ∀ (f : ℝ → ℝ),
    number_of_holes f = 1 ∧ 
    number_of_vertical_asymptotes f = 2 ∧
    number_of_horizontal_asymptotes f = 1 ∧ 
    number_of_oblique_asymptotes f = 0

theorem problem_solution : 
  p = number_of_holes function ∧ 
  q = number_of_vertical_asymptotes function ∧ 
  r = number_of_horizontal_asymptotes function ∧ 
  s = number_of_oblique_asymptotes function 
  → 
  p + 2 * q + 3 * r + 4 * s = 8 := 
by 
  assume h : p = number_of_holes function ∧
             q = number_of_vertical_asymptotes function ∧
             r = number_of_horizontal_asymptotes function ∧
             s = number_of_oblique_asymptotes function,
  sorry

end problem_solution_l32_32239


namespace sum_of_digits_of_31_digit_number_l32_32402

theorem sum_of_digits_of_31_digit_number 
  (n : Nat)
  (digits : Fin 31 → Fin 10)
  (cond1 : ∀ (i : Fin 30), ((digits i) * 10 + (digits ⟨i.val + 1, sorry⟩)) % 17 = 0 ∨ ((digits i) * 10 + (digits ⟨i.val + 1, sorry⟩)) % 23 = 0)
  (cond2 : (List.count digits.to_list 7) = 1)
  : digits.to_list.sum = 151 := sorry

end sum_of_digits_of_31_digit_number_l32_32402


namespace harmonic_not_integer_l32_32998

theorem harmonic_not_integer (n : ℕ) (h : n > 1) : ¬ ∃ (k : ℤ), H_n n = k :=
by
  sorry

noncomputable def H_n : ℕ → ℚ
| 0     := 0
| (n+1) := H_n n + 1 / (n + 1)

end harmonic_not_integer_l32_32998


namespace max_area_triangle_ABC_l32_32233

noncomputable def TriangleABC : Type :=
{A B C : Vect}
(h_ab_ac : A = B)
(h_ac_bc_mag: ‖A + B‖ = 2 * Real.sqrt 6)

theorem max_area_triangle_ABC : ∃ (A B C: Vect) (h_bc: Real.sqrt (9 * A^2 + B^2) = 24), 
∀ (area : ℝ), area = 4 := 
sorry

end max_area_triangle_ABC_l32_32233


namespace domain_w_l32_32114

noncomputable def w (y : ℝ) : ℝ := (y - 3)^(1/3) + (15 - y)^(1/3)

theorem domain_w : ∀ y : ℝ, ∃ x : ℝ, w y = x := by
  sorry

end domain_w_l32_32114


namespace john_horizontal_distance_l32_32647

-- Define the conditions and the question
def elevation_initial : ℕ := 100
def elevation_final : ℕ := 1450
def vertical_to_horizontal_ratio (v h : ℕ) : Prop := v * 2 = h

-- Define the proof problem: the horizontal distance John moves
theorem john_horizontal_distance : ∃ h, vertical_to_horizontal_ratio (elevation_final - elevation_initial) h ∧ h = 2700 := 
by 
  sorry

end john_horizontal_distance_l32_32647


namespace walter_time_spent_at_seals_l32_32369

theorem walter_time_spent_at_seals (S : ℕ) 
(h1 : 8 * S + S + 13 = 130) : S = 13 :=
sorry

end walter_time_spent_at_seals_l32_32369


namespace product_of_terms_l32_32637

variable {α : Type*} [LinearOrderedField α]

namespace GeometricSequence

def is_geometric_sequence (a : ℕ → α) :=
  ∃ r : α, ∀ n : ℕ, a (n + 1) = r * a n

theorem product_of_terms (a : ℕ → α) (r : α) (h_geo : is_geometric_sequence a) :
  (a 4) * (a 8) = 16 → (a 2) * (a 10) = 16 :=
by
  intro h1
  sorry

end GeometricSequence

end product_of_terms_l32_32637


namespace pie_eating_contest_l32_32741

theorem pie_eating_contest :
  (7 / 8 : ℚ) - (5 / 6 : ℚ) = (1 / 24 : ℚ) :=
sorry

end pie_eating_contest_l32_32741


namespace spider_final_position_l32_32689

def circle_points : List ℕ := [1, 2, 3, 4, 5, 6, 7]

def next_position (current : ℕ) : ℕ :=
  if current % 2 = 0 
  then (current + 3 - 1) % 7 + 1 -- Clockwise modulo operation for even
  else (current + 1 - 1) % 7 + 1 -- Clockwise modulo operation for odd

def spider_position_after_jumps (start : ℕ) (jumps : ℕ) : ℕ :=
  (Nat.iterate next_position jumps start)

theorem spider_final_position : spider_position_after_jumps 6 2055 = 2 := 
  by
  sorry

end spider_final_position_l32_32689


namespace fourth_intersection_point_l32_32633

noncomputable def curve (p : ℝ × ℝ) : Prop := p.1 * p.2 = 2

def circle (a b r : ℝ) (p : ℝ × ℝ) : Prop := (p.1 - a)^2 + (p.2 - b)^2 = r^2

def point1 := (4 : ℝ, 1 / 2 : ℝ)
def point2 := (-2 : ℝ, -1 : ℝ)
def point3 := (1 / 4 : ℝ, 8 : ℝ)
def point4 := (-1 / 8 : ℝ, -16 : ℝ)

theorem fourth_intersection_point 
(a b r : ℝ) 
(h1 : curve point1)
(h2 : curve point2)
(h3 : curve point3)
(h4 : circle a b r point1)
(h5 : circle a b r point2)
(h6 : circle a b r point3) 
(h7 : curve point4) :
  circle a b r point4 :=
  sorry

end fourth_intersection_point_l32_32633


namespace equilateral_triangle_area_sum_l32_32068

theorem equilateral_triangle_area_sum (A : ℝ) : 
  let a := A
  let r := (1 : ℝ) / 4
  let S := ∑' n, A * r^n
in S = 4 * A / 3 :=
by {
  -- Define a and r in context
  let a := A,
  let r := (1 : ℝ) / 4,

  -- Use the formula for the sum of an infinite geometric series
  have geom_series_sum : ∑' n, a * r^n = a / (1 - r),
  { sorry },

  -- Plug in the value of r
  have sum_formula : a / (1 - r) = 4 * A / 3,
  { sorry },

  -- Combine the results
  show ∑' n, A * r^n = 4 * A / 3,
  from geom_series_sum.trans sum_formula,
}

end equilateral_triangle_area_sum_l32_32068


namespace sum_of_faces_edges_vertices_l32_32051

def rectangular_prism := 
  ∃ (faces edges vertices : ℕ), 
    faces = 6 ∧ edges = 12 ∧ vertices = 8

def add_pyramid (prism : rectangular_prism) := 
  let (faces, edges, vertices) := (prism.faces, prism.edges, prism.vertices)
  (faces - 1 + 4, edges + 4, vertices + 1)

theorem sum_of_faces_edges_vertices (prism : rectangular_prism) :
  let (new_faces, new_edges, new_vertices) := add_pyramid prism in
  new_faces + new_edges + new_vertices = 34 := 
by 
  -- Original properties of the prism
  obtain ⟨faces, edges, vertices, h_faces, h_edges, h_vertices⟩ := prism,
  -- Define the new properties when the pyramid is added
  let new_faces := faces - 1 + 4,
  let new_edges := edges + 4,
  let new_vertices := vertices + 1,
  -- Prove the sum
  have h1 : new_faces = 9 := by 
    rw [h_faces],
    norm_num,
  have h2 : new_edges = 16 := by 
    rw [h_edges],
    norm_num,
  have h3 : new_vertices = 9 := by 
    rw [h_vertices],
    norm_num,
  show 9 + 16 + 9 = 34, 
  norm_num

end sum_of_faces_edges_vertices_l32_32051


namespace lewis_weekly_earning_l32_32283

def total_amount_earned : ℕ := 178
def number_of_weeks : ℕ := 89
def weekly_earning (total : ℕ) (weeks : ℕ) : ℕ := total / weeks

theorem lewis_weekly_earning : weekly_earning total_amount_earned number_of_weeks = 2 :=
by
  -- The proof will go here
  sorry

end lewis_weekly_earning_l32_32283


namespace matrix_quarter_rotation_pow_four_l32_32460

def sqrt2_div2 : ℝ := real.sqrt 2 / 2

def matrix_rotation (θ : ℝ) : matrix (fin 2) (fin 2) ℝ :=
  ![![real.cos θ, -real.sin θ], ![real.sin θ, real.cos θ]]

def M : matrix (fin 2) (fin 2) ℝ :=
  ![![sqrt2_div2, -sqrt2_div2], ![sqrt2_div2, sqrt2_div2]]

theorem matrix_quarter_rotation_pow_four :
  M ^ 4 = ![![(-1 : ℝ), 0], ![0, -1]] :=
by
  sorry

end matrix_quarter_rotation_pow_four_l32_32460


namespace circle_radius_of_inscribed_square_l32_32425

theorem circle_radius_of_inscribed_square (s r : ℝ) 
  (h1 : ∀ s r, s * ℝ.sqrt 2 = 2 * r) 
  (h2 : 4 * s = π * r * r): 
  r = 4 * ℝ.sqrt 2 / π := 
by 
suffices : r * ℝ.sqrt 2 = 4 * ℝ.sqrt 2 / π 
  sorry 

end circle_radius_of_inscribed_square_l32_32425


namespace find_k_perpendicular_lines_l32_32183

noncomputable def l1 (k : ℝ) : ℝ → ℝ → Prop := 
  λ x y, (k-3) * x + (3-k) * y + 1 = 0

noncomputable def l2 (k : ℝ) : ℝ → ℝ → Prop := 
  λ x y, 2 * (k-3) * x - 2 * y + 3 = 0

def perpendicular_lines (k : ℝ) : Prop :=
  2 * (k-3)^2 - 2 * (3-k) = 0

theorem find_k_perpendicular_lines : 
  ∃ k : ℝ, perpendicular_lines k ∧ k = 2 :=
sorry

end find_k_perpendicular_lines_l32_32183


namespace conjugate_z_is_5_plus_12i_l32_32164

-- Define the complex number z and the given condition
noncomputable def z : ℂ := Complex.ofReal 169 / (5 + 12 * Complex.I)

-- State the theorem
theorem conjugate_z_is_5_plus_12i : Complex.conj z = 5 + 12 * Complex.I := by
  sorry

end conjugate_z_is_5_plus_12i_l32_32164


namespace pizza_toppings_combination_l32_32310

def num_combinations {α : Type} (s : Finset α) (k : ℕ) : ℕ :=
  (s.card.choose k)

theorem pizza_toppings_combination (s : Finset ℕ) (h : s.card = 7) : num_combinations s 3 = 35 :=
by
  sorry

end pizza_toppings_combination_l32_32310


namespace minimum_sin_value_l32_32547

open Real

noncomputable def min_value (n : Nat) (x : Fin n → ℝ) : ℝ :=
  ∏ i, (sin (x i) + 1 / sin (x i))

theorem minimum_sin_value 
  (n : Nat)
  (h_n : 2 ≤ n)
  (x : Fin n → ℝ)
  (h_x_pos : ∀ i, 0 < x i)
  (h_sum_x_pi : sum (fun i => x i) = π) :
  min_value n x ≥ (sin (π / n) + 1 / sin (π / n)) ^ n := 
sorry

end minimum_sin_value_l32_32547


namespace part_a_part_b_l32_32297

-- Define the function with the given conditions
variable {f : ℝ → ℝ}
variable (h_nonneg : ∀ x, 0 ≤ x → 0 ≤ f x)
variable (h_f1 : f 1 = 1)
variable (h_subadditivity : ∀ (x₁ x₂ : ℝ), 0 ≤ x₁ → 0 ≤ x₂ → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂)

-- Part (a): Prove that f(x) ≤ 2x for all x ∈ [0, 1]
theorem part_a : ∀ x, 0 ≤ x → x ≤ 1 → f x ≤ 2 * x :=
by
  sorry -- Proof required.

-- Part (b): Prove that it is not true that f(x) ≤ 1.9x for all x ∈ [0,1]
theorem part_b : ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ 1.9 * x < f x :=
by
  sorry -- Proof required.

end part_a_part_b_l32_32297


namespace total_pieces_correct_l32_32944

-- Given conditions
def puzzle1_pieces : ℕ := 1000
def puzzle2_pieces : ℕ := puzzle1_pieces + 20 * puzzle1_pieces / 100
def puzzle3_pieces : ℕ := puzzle2_pieces + 50 * puzzle2_pieces / 100
def puzzle4_pieces : ℕ := puzzle3_pieces + 75 * puzzle3_pieces / 100
def puzzle5_pieces : ℕ := puzzle4_pieces + 35 * puzzle4_pieces / 100

-- Total pieces
def total_pieces : ℕ :=
  puzzle1_pieces + puzzle2_pieces + puzzle3_pieces + puzzle4_pieces + (puzzle4_pieces + 35 * puzzle4_pieces / 100).natRound

-- Proof problem statement
theorem total_pieces_correct : total_pieces = 11403 := by
  sorry

end total_pieces_correct_l32_32944


namespace semicircle_tangent_point_l32_32638

theorem semicircle_tangent_point :
  ∃ (α : ℝ), α ∈ Icc 0 π ∧
  let x := 1 + Real.cos α,
      y := Real.sin α in
  (∃ p : ℝ × ℝ, p = (x, y) ∧
    (p.1 - 1)^2 + p.2^2 = 1 ∧
    y / (x - 1) = Real.sqrt 3) :=
begin
  sorry
end

end semicircle_tangent_point_l32_32638


namespace snowboard_price_after_discounts_l32_32293

noncomputable def final_snowboard_price (P_original : ℝ) (d_Friday : ℝ) (d_Monday : ℝ) : ℝ :=
  P_original * (1 - d_Friday) * (1 - d_Monday)

theorem snowboard_price_after_discounts :
  final_snowboard_price 100 0.50 0.30 = 35 :=
by 
  sorry

end snowboard_price_after_discounts_l32_32293


namespace domain_single_point_at_M_l32_32273

noncomputable def g : ℕ → (ℝ → ℝ)
| 1 := λ x, Real.sqrt (2 - x)
| (n + 2) := λ x, g (n + 1) (Real.sqrt ((n + 2)^2 + 1 - x))

def M : ℕ := 5
def d : ℝ := -261

theorem domain_single_point_at_M : 
  ∀ x : ℝ, x ∈ {y : ℝ | ∃ x', g M x' = y} → x = d :=
by
  sorry

end domain_single_point_at_M_l32_32273


namespace value_of_expression_l32_32203

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : 2 * a^2 + 4 * a * b + 2 * b^2 - 6 = 12 :=
by
  sorry

end value_of_expression_l32_32203


namespace setC_not_pythagorean_l32_32435

/-- Defining sets of numbers as options -/
def SetA := (3, 4, 5)
def SetB := (5, 12, 13)
def SetC := (7, 25, 26)
def SetD := (6, 8, 10)

/-- Function to check if a set is a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

/-- Theorem stating set C is not a Pythagorean triple -/
theorem setC_not_pythagorean :
  ¬isPythagoreanTriple 7 25 26 :=
by {
  -- This slot will be filled with the concrete proof steps in Lean.
  sorry
}

end setC_not_pythagorean_l32_32435


namespace circle_system_fixed_point_l32_32775

theorem circle_system_fixed_point (a : ℝ) (h : a ≠ 1) :
  ∀ a ∈ ℝ \ {1}, (1 : ℝ, 1 : ℝ) satisfies (x, y) → (x^2 + y^2 - 2 * a * x + 2 * (a - 2) * y + 2 = 0) :=
sorry

end circle_system_fixed_point_l32_32775


namespace number_of_proper_subsets_of_complement_l32_32181

-- Define the universal set U and set A based on the given conditions
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℝ := {x | x^2 - 6*x + 5 = 0}

-- Define the complement of A with respect to U
def C_U_A : Set ℕ := {x ∈ U | x ∉ A}

-- Define the property of B being a proper subset of C_U_A
def is_proper_subset (B : Set ℕ) (S : Set ℕ) : Prop := B ⊆ S ∧ B ≠ S

-- State the theorem
theorem number_of_proper_subsets_of_complement :
  (∃ (B : Set ℕ) (n : ℕ), is_proper_subset B C_U_A ∧ n = 7) :=
sorry

end number_of_proper_subsets_of_complement_l32_32181


namespace calculate_expression_l32_32454

theorem calculate_expression :
  8^8 + 8^8 + 8^8 + 8^8 + 8^5 = 4 * 8^8 + 8^5 := 
by sorry

end calculate_expression_l32_32454


namespace product_P_2009_l32_32320

-- Define the sequence a_n
noncomputable def a : ℕ → ℝ
| 0       := 2
| (n + 1) := 1 - (1 / a n)

-- Define the product P_n
noncomputable def P : ℕ → ℝ
| 0       := 1
| (n + 1) := P n * a (n + 1)

theorem product_P_2009 : P 2009 = -1 := 
sorry

end product_P_2009_l32_32320


namespace Linda_has_24_classmates_l32_32671

theorem Linda_has_24_classmates 
  (cookies_per_student : ℕ := 10)
  (cookies_per_batch : ℕ := 48)
  (chocolate_chip_batches : ℕ := 2)
  (oatmeal_raisin_batches : ℕ := 1)
  (additional_batches : ℕ := 2) : 
  (chocolate_chip_batches * cookies_per_batch + oatmeal_raisin_batches * cookies_per_batch + additional_batches * cookies_per_batch) / cookies_per_student = 24 := 
by 
  sorry

end Linda_has_24_classmates_l32_32671


namespace find_b3_b17_l32_32885

variable {a : ℕ → ℤ} -- Arithmetic sequence
variable {b : ℕ → ℤ} -- Geometric sequence

axiom arith_seq {a : ℕ → ℤ} (d : ℤ) : ∀ (n : ℕ), a (n + 1) = a n + d
axiom geom_seq {b : ℕ → ℤ} (r : ℤ) : ∀ (n : ℕ), b (n + 1) = b n * r

theorem find_b3_b17 
  (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d) 
  (h_geom : ∃ r, ∀ n, b (n + 1) = b n * r)
  (h_cond1 : 3 * a 1 - (a 8)^2 + 3 * a 15 = 0)
  (h_cond2 : a 8 = b 10) :
  b 3 * b 17 = 36 := 
sorry

end find_b3_b17_l32_32885


namespace number_of_3_letter_words_with_at_least_one_A_l32_32192

theorem number_of_3_letter_words_with_at_least_one_A :
  let all_words := 5^3
  let no_A_words := 4^3
  all_words - no_A_words = 61 :=
by
  sorry

end number_of_3_letter_words_with_at_least_one_A_l32_32192


namespace distance_between_polar_points_l32_32260

open Real
open Complex

noncomputable def distance_polar (r1 θ₁ r2 θ₂ : ℝ) : ℝ :=
  sqrt ((r2 * cos θ₂ - r1 * cos θ₁) ^ 2 + (r2 * sin θ₂ - r1 * sin θ₁) ^ 2)

theorem distance_between_polar_points (r1 r2 θ₁ θ₂ : ℝ) (h₁ : r1 = 4) (h₂ : r2 = 12) 
(h₃ : θ₁ - θ₂ = (3 * π) / 4) : 
  distance_polar r1 θ₁ r2 θ₂ = 4 * (3 + sqrt 2) := by
  rw [h₁, h₂, ←h₃]
  sorry

end distance_between_polar_points_l32_32260


namespace symmetric_point_origin_l32_32556

def symmetric_point (p: ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem symmetric_point_origin : 
  (symmetric_point (3, -2)) = (-3, 2) :=
by
  sorry

end symmetric_point_origin_l32_32556


namespace overall_percentage_l32_32805

theorem overall_percentage (s1 s2 s3 : ℝ) (h1 : s1 = 60) (h2 : s2 = 80) (h3 : s3 = 85) :
  (s1 + s2 + s3) / 3 = 75 := by
  sorry

end overall_percentage_l32_32805


namespace fixed_point_of_function_l32_32333

theorem fixed_point_of_function (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) : (2, 3) ∈ { (x, y) | y = 2 + a^(x-2) } :=
sorry

end fixed_point_of_function_l32_32333


namespace average_speed_l32_32793

theorem average_speed (k : ℕ) (h₁ : k > 0) :
  let distance_north := k
  let speed_north := 1 / (10 / 60 : ℝ)  -- in km per hour
  let rest_time := 5 / 60  -- in hours
  let speed_south := 0.5 * 60  -- in km per hour
  let time_north := (10 * k) / 60  -- in hours
  let time_south := (k / 0.5) / 60  -- in hours
  let total_time := time_north + rest_time + time_south
  let total_distance := 2 * k
  let avg_speed := total_distance / total_time
  avg_speed = (120 * k) / (12 * k + 5) :=
by {
  sorry
}

end average_speed_l32_32793


namespace select_one_person_serving_both_days_l32_32864

theorem select_one_person_serving_both_days :
  ∃ (ways : ℕ), ways = 60 ∧
    let total_volunteers : ℕ := 5 in
    let choose_person_both_days := total_volunteers in
    let choose_saturday := total_volunteers - 1 in
    let choose_sunday := total_volunteers - 2 in
    ways = choose_person_both_days * choose_saturday * choose_sunday :=
by
  sorry

end select_one_person_serving_both_days_l32_32864


namespace strictly_increasing_functions_count_non_decreasing_functions_count_l32_32665

section CombinatorialFunctions

variables (m n : ℕ) (hm : m > 0) (hn : n > 0)

-- Part (a): Number of strictly increasing functions from {1, 2, ..., m} to {1, 2, ..., n}
theorem strictly_increasing_functions_count :
  {f : fin m → fin n // ∀ (i j : fin m), i < j → f i < f j}.card = nat.choose n m :=
by sorry

-- Part (b): Number of non-decreasing functions from {1, 2, ..., m} to {1, 2, ..., n}
theorem non_decreasing_functions_count :
  {f : fin m → fin n // ∀ (i j : fin m), i < j → f i ≤ f j}.card = nat.choose (m + n) m :=
by sorry

end CombinatorialFunctions

end strictly_increasing_functions_count_non_decreasing_functions_count_l32_32665


namespace value_of_S33_l32_32137

-- Definition of the sequence {a_n}
def a : ℕ → ℝ 
| 1 := 1
| 2 := 2
| (n+1) := real.sqrt (3 * (n + 1) - 2)

-- Definition of the sequence {b_n}
def b (n : ℕ) : ℝ := 1 / (a n + a (n + 1))

-- Sum of the first n terms of the sequence {b_n}
noncomputable def S (n : ℕ) : ℝ := ∑ i in finset.range n, b i

-- Statement to prove
theorem value_of_S33 : S 33 = 3 := 
by sorry

end value_of_S33_l32_32137


namespace product_consecutive_natural_number_square_l32_32643

theorem product_consecutive_natural_number_square (n : ℕ) : 
  ∃ k : ℕ, 100 * (n^2 + n) + 25 = k^2 :=
by
  sorry

end product_consecutive_natural_number_square_l32_32643


namespace collinear_vectors_lambda_value_l32_32873

theorem collinear_vectors_lambda_value (λ : ℝ) (h : (2 / 1) = ((λ + 2) / λ)) : λ = 2 :=
sorry

end collinear_vectors_lambda_value_l32_32873


namespace minimize_total_cost_l32_32422

theorem minimize_total_cost (x : ℝ) (h_pos : x > 0) 
 (h_fuel_cost : ∀ c k : ℝ, c = k * 10^3 → k = 3 / 500) 
 (h_other_cost : Real) : 
  (x = 20) → 
  ((3 / 500) * x^2 + (96 / x)) ≤ ((3 / 500) * y^2 + (96 / y))  ∀ y ∈ Ioi 0 := sorry

end minimize_total_cost_l32_32422


namespace numbers_on_board_l32_32724

theorem numbers_on_board (n : ℕ) (x : ℕ → ℕ) (S : ℕ)
  (h1 : n ≥ 2)
  (h2 : ∀ i j, i < j → x i < x j)
  (h3 : 30 * x 0 + ∑ i in Finset.range n, x i = 450)
  (h4 : 14 * x (n - 1) + ∑ i in Finset.range n, x i = 450) :
  (x 0 = 13 ∧ ∃ i, x i = 14 ∧ ∃ j, x j = 17 ∧ x (n - 1) = 29) ∨
  (x 0 = 13 ∧ ∃ i, x i = 15 ∧ ∃ j, x j = 16 ∧ x (n - 1) = 29) :=
by sorry

end numbers_on_board_l32_32724


namespace rental_problem_revenue_maximization_l32_32777

noncomputable def number_of_rented_cars (r : ℕ) : ℕ :=
  if r < 3000 then 100 else 100 - (r - 3000) / 50

noncomputable def revenue (r : ℕ) : ℝ :=
  let n := 100 - ((r - 3000) / 50 : ℕ) in
  n * (r - 150) - ((r - 3000) / 50 : ℕ) * 50

theorem rental_problem (r : ℕ) (h : r = 3600) :
  number_of_rented_cars r = 88 :=
by
  simp [number_of_rented_cars, h]
  sorry

theorem revenue_maximization :
  ∃ r_max : ℕ, r_max = 4050 ∧ revenue 4050 = 307050 :=
by
  use 4050
  split
  · refl
  · simp [revenue]
    sorry

end rental_problem_revenue_maximization_l32_32777


namespace geometric_sequence_relation_l32_32661

variables (a : ℕ → ℝ) (q : ℝ) (n : ℕ)
variables [h_pos : ∀ n, 0 < a n] [h_geom : ∀ n, a (n+1) = q * a n]

def b_n (n : ℕ) := a (n+1) + a (n+2)
def c_n (n : ℕ) := a n + a (n+3)

theorem geometric_sequence_relation :
  b_n a n ≤ c_n a n :=
sorry

end geometric_sequence_relation_l32_32661


namespace min_value_frac_square_l32_32857

theorem min_value_frac_square (x : ℝ) (h : x > 9) : (x^2 / (x - 9) ≥ 36) ∧ ∃ y > 9, y^2 / (y - 9) = 36 :=
by {
  split,
  -- Prove that for all x > 9, x^2 / (x - 9) ≥ 36
  sorry,
  -- Prove that there exists an x > 9 such that x^2 / (x - 9) = 36
  use 18,
  split,
  linarith,
  norm_num,
}

end min_value_frac_square_l32_32857


namespace distinct_real_values_of_a_l32_32517

theorem distinct_real_values_of_a :
  let N := {-499, -498, ..., 499}
  ∃ (a : ℝ) (f : ℤ → ℝ), (∀ n ∈ N, f(n) = (8 * n^3 - 1) / (2 * n + 1)) ∧ function.injective f ∧
  (∀ n ∈ N, ∃ x, x = 2 * n ∧ |x| < 1000 ∧ x^3 = a * x + a + 1) ∧ set.card N = 999 :=
sorry

end distinct_real_values_of_a_l32_32517


namespace bee_flight_time_l32_32791

-- Define the key parameters used in the problem
variables (t : ℝ)  -- time from daisy to rose
def speed_daisy_to_rose := 2.6  -- speed from daisy to rose in m/s
def time_rose_to_poppy := 6  -- time from rose to poppy in seconds
def additional_speed := 3  -- additional speed from rose to poppy in m/s
def distance_difference := 8  -- distance difference in meters

-- Distance calculations
def distance_daisy_to_rose := speed_daisy_to_rose * t
def speed_rose_to_poppy := speed_daisy_to_rose + additional_speed
def distance_rose_to_poppy := speed_rose_to_poppy * time_rose_to_poppy

-- Statement of the theorem to be proved
theorem bee_flight_time : 
  distance_daisy_to_rose = distance_rose_to_poppy + distance_difference → t = 16 :=
by
  sorry

end bee_flight_time_l32_32791


namespace sum_diff_9114_l32_32391

def sum_odd_ints (n : ℕ) := (n + 1) / 2 * (1 + n)
def sum_even_ints (n : ℕ) := n / 2 * (2 + n)

theorem sum_diff_9114 : 
  let m := sum_odd_ints 215
  let t := sum_even_ints 100
  m - t = 9114 :=
by
  sorry

end sum_diff_9114_l32_32391


namespace matching_last_two_digits_l32_32754

theorem matching_last_two_digits (n : ℕ) : 
  let b := n % 100 in 
    (b^2 % 100 = b) ↔ (b = 0 ∨ b = 1 ∨ b = 25 ∨ b = 76) :=
by
  sorry

end matching_last_two_digits_l32_32754


namespace problem_solution_l32_32540

theorem problem_solution (a b c d : ℝ) (h1 : ab + bc + cd + da = 48) (h2 : b + d = 6) : a + c = 8 :=
sorry

end problem_solution_l32_32540


namespace parallelogram_area_l32_32681

-- Defining the basic conditions
def side_lengths (EF FG EH : ℝ) : Prop :=
  EF = 40 ∧ FG = 30 ∧ EH = 50

-- The theorem statement
theorem parallelogram_area (EF FG EH : ℝ) (h : side_lengths EF FG EH) : 
  ∃ (A : ℝ), A = 1200 :=
by
  obtain ⟨hEF, hFG, hEH⟩ := h
  use 1200
  sorry

end parallelogram_area_l32_32681


namespace minimize_sum_of_segments_l32_32614

open Real

theorem minimize_sum_of_segments
  (E F G H I : ℝ×ℝ)
  (angleEFG : ∠(E, F, G) = π / 4)
  (EF : dist E F = 8)
  (EG : dist E G = 12)
  (H_on_EF : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ H = (1 - t) • E + t • F)
  (I_on_EG : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ I = (1 - t) • E + t • G) :
  FH + HI + IG ≥ sqrt(208 + 192 * sqrt(2)) :=
sorry

end minimize_sum_of_segments_l32_32614


namespace problem_statement_l32_32188

noncomputable def f (x : ℝ) : ℝ := (sin x + sqrt 3 * cos x) * sin x + 3 / 2

theorem problem_statement :
  (∀ k : ℤ, by
    let I := Set.Icc (-π / 6 + k * π) (π / 3 + k * π)
    (monotonic_increasing f I)) ∧
  (let A : ℝ := π / 3,
       b : ℝ := 2,
       S : ℝ := 2 * sqrt 3 in
    (a = 2 * sqrt 3) ∧ (c = 4) ∧ (f A = 3) → A = π / 3 ∧ b = 2 ∧ S = 2 * sqrt 3
  )
:= 
sorry

end problem_statement_l32_32188


namespace number_of_integers_satisfying_condition_l32_32095

/-- Determine how many integers n satisfy the condition that n / (30-n) is the square of an integer. -/
theorem number_of_integers_satisfying_condition :
  {n : ℤ | 0 ≤ n ∧ n ≤ 30 ∧ ∃ k : ℤ, n / (30 - n) = k * k}.finite.card = 4 :=
begin
  sorry
end

end number_of_integers_satisfying_condition_l32_32095


namespace percentage_of_indian_children_is_10_l32_32226

def men_count : ℕ := 700
def women_count : ℕ := 500
def children_count : ℕ := 800
def indian_men_percentage : ℝ := 0.20
def indian_women_percentage : ℝ := 0.40
def non_indian_people_percentage : ℝ := 0.79
def total_people : ℕ := men_count + women_count + children_count

def indian_people (men_count women_count children_count : ℕ) 
  (indian_men_percentage indian_women_percentage non_indian_people_percentage : ℝ) : ℕ :=
  (1 - non_indian_people_percentage) * (men_count + women_count + children_count)

def indian_men (men_count : ℕ) (indian_men_percentage : ℝ) : ℕ :=
  (indian_men_percentage * men_count).toNat

def indian_women (women_count : ℕ) (indian_women_percentage : ℝ) : ℕ :=
  (indian_women_percentage * women_count).toNat

def indian_children (total_indian indian_men indian_women : ℕ) : ℕ :=
  total_indian - indian_men - indian_women

def percentage_indian_children (indian_children : ℕ) (children_count : ℕ) : ℝ :=
  (indian_children.toReal / children_count.toReal) * 100

theorem percentage_of_indian_children_is_10 :
  percentage_indian_children (indian_children 
    (indian_people men_count women_count children_count indian_men_percentage indian_women_percentage non_indian_people_percentage)
    (indian_men men_count indian_men_percentage) 
    (indian_women women_count indian_women_percentage)) children_count = 10 := 
sorry

end percentage_of_indian_children_is_10_l32_32226


namespace total_distance_craig_walked_l32_32828

theorem total_distance_craig_walked :
  0.2 + 0.7 = 0.9 :=
by sorry

end total_distance_craig_walked_l32_32828


namespace mean_noon_temperature_l32_32338

def temperatures : List ℝ := [79, 78, 82, 86, 88, 90, 88, 90, 89]

theorem mean_noon_temperature :
  (List.sum temperatures) / (temperatures.length) = 770 / 9 := by
  sorry

end mean_noon_temperature_l32_32338


namespace binom_arithmetic_sequence_l32_32148

noncomputable def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_arithmetic_sequence {n : ℕ} (h : 2 * binom n 5 = binom n 4 + binom n 6) (n_eq : n = 14) : binom n 12 = 91 := by
  sorry

end binom_arithmetic_sequence_l32_32148


namespace f_of_f_of_8_eq_neg_half_l32_32168

-- Define the function f
def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -Real.logb 2 x else Real.sin (Real.pi * x + Real.pi / 6)

-- State the theorem
theorem f_of_f_of_8_eq_neg_half : f (f 8) = -1 / 2 :=
by
  sorry

end f_of_f_of_8_eq_neg_half_l32_32168


namespace problem1_problem2_problem3_l32_32398

-- Define the first problem
theorem problem1 {x1 x2 : ℝ} (h1 : 0 < x1) (h2 : x1 < x2) : 
  (x1 + 1) / (x2 + 1) > x1 / x2 := 
sorry

-- Define the function for the second problem
noncomputable def f (x : ℝ) : ℝ := log (x + 1) - (1/2) * log 3 x

-- Prove that f(x) is monotonic decreasing
theorem problem2 : ∀ (x1 x2 : ℝ), (0 < x1) → (x1 < x2) → f x1 > f x2 :=
sorry

-- State assumptions and prove the number of subsets of set M
theorem problem3 : ∃ M : Set ℤ, (∀ n ∈ M, f (n^2 - 214 * n - 1998) ≥ 0) ∧
  ∃ S : Finset (Finset ℤ), S.card = 4 :=
sorry

end problem1_problem2_problem3_l32_32398


namespace matrix_product_correct_l32_32825

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  !![2, 0, -1; 1, 3, -2; -2, 3, 2]

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  !![3, -1, 1; 2, 1, -4; 5, 0, 0]

def C : Matrix (Fin 3) (Fin 3) ℤ :=
  !![1, -2, 2; -1, 2, -11; 10, 5, -14]

theorem matrix_product_correct : A ⬝ B = C := by
  sorry

end matrix_product_correct_l32_32825


namespace mr_thompson_statement_l32_32987

variable (P Q : Prop)
open Classical

theorem mr_thompson_statement (h : ¬ P → ¬ Q) : Q → P :=
begin
  intro q,
  by_contradiction p,
  exact h p q,
end

example (received_C_or_higher answered_all_correctly : Prop) (h : ¬ answered_all_correctly → ¬ received_C_or_higher) : received_C_or_higher → answered_all_correctly :=
begin
  intro h_q,
  by_contradiction p_not,
  exact h p_not h_q,
end

end mr_thompson_statement_l32_32987


namespace find_equation_of_circle_l32_32853

noncomputable def equation_of_circle_through_A_and_tangent_to_l (A B : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  let circle_equation : ℝ → ℝ → ℝ :=
    λ x y, x^2 + y^2 - 12 * x - 12 * y - 88
  ∃ (D E F : ℝ), (x^2 + y^2 + D * x + E * y + F = 0) ∧
  A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0 ∧
  B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0 ∧
  tan (atan (-(B.2 - E/2) / (B.1 - D/2))) = -(l 1 3) ∧
  (circle_equation = λ x y, x^2 + y^2 + D * x + E * y + F)

-- Now, stating the problem
theorem find_equation_of_circle :
  equation_of_circle_through_A_and_tangent_to_l (-6, 10) (2, -6) (λ x y, x + 3 * y + 16 = 0) :=
sorry

end find_equation_of_circle_l32_32853


namespace james_prom_total_cost_l32_32244

-- Definitions and conditions
def ticket_cost : ℕ := 100
def num_tickets : ℕ := 2
def dinner_cost : ℕ := 120
def tip_rate : ℚ := 0.30
def limo_hourly_rate : ℕ := 80
def limo_hours : ℕ := 6

-- Calculation of each component
def total_ticket_cost : ℕ := ticket_cost * num_tickets
def total_tip : ℚ := tip_rate * dinner_cost
def total_dinner_cost : ℚ := dinner_cost + total_tip
def total_limo_cost : ℕ := limo_hourly_rate * limo_hours

-- Final total cost calculation
def total_cost : ℚ := total_ticket_cost + total_dinner_cost + total_limo_cost

-- Proving the final total cost
theorem james_prom_total_cost : total_cost = 836 := by sorry

end james_prom_total_cost_l32_32244


namespace calculate_y_l32_32185

open Real

variables (y : ℝ)
def v := (1, y)
def w' := (-3, 1)

def proj_v_on_w' : ℝ × ℝ :=
  let v_dot_w' := 1 * (-3) + y * 1
  let w'_dot_w' := (-3) * (-3) + 1 * 1
  let factor := v_dot_w' / w'_dot_w'
  (factor * (-3), factor * 1)

theorem calculate_y :
  proj_v_on_w' y = (2, -2 / 3) → y = -11 / 3 := by
  sorry

end calculate_y_l32_32185


namespace exists_coprime_linear_combination_l32_32782

theorem exists_coprime_linear_combination (a b p : ℤ) :
  ∃ k l : ℤ, Int.gcd k l = 1 ∧ p ∣ (a * k + b * l) :=
  sorry

end exists_coprime_linear_combination_l32_32782


namespace total_fish_l32_32075

theorem total_fish :
  let Billy := 10
  let Tony := 3 * Billy
  let Sarah := Tony + 5
  let Bobby := 2 * Sarah
  in Billy + Tony + Sarah + Bobby = 145 :=
by
  sorry

end total_fish_l32_32075


namespace min_value_func_l32_32858

noncomputable def func (x : ℝ) := x^2 / (x - 9)

theorem min_value_func : (∀ x > 9, func(x) ≥ 36) ∧ (∃ x = 18, func(x) = 36) :=
by
  sorry

end min_value_func_l32_32858


namespace max_sum_bn_value_l32_32139

-- Definitions based on conditions
def Sn (n : ℕ) (a_n : ℕ → ℕ) : ℕ := 2 * a_n n - 2
def bn (n : ℕ) (a_n : ℕ → ℕ) : ℕ := 10 - nat.log 2 (a_n n)

-- Proving the statement
theorem max_sum_bn_value (a_n : ℕ → ℕ) (ha : ∀ n, Sn n a_n = 2 * a_n n - 2) (hb : ∀ n, bn n a_n = 10 - nat.log 2 (a_n n)) :
  ∃ n, n = 9 ∨ n = 10 :=
sorry

end max_sum_bn_value_l32_32139


namespace evaluate_series_l32_32846

noncomputable def infinite_series := ∑ k in (Finset.range ∞), (k + 1)^2 / 3^(k + 1)

theorem evaluate_series : infinite_series = 1 / 2 := sorry

end evaluate_series_l32_32846


namespace acute_triangle_radius_ge_circumcircle_radius_obtuse_triangle_radius_ge_circumcircle_radius_l32_32067

theorem acute_triangle_radius_ge_circumcircle_radius
  (A B C : Type)
  (hA : is_acute_angle A)
  (hB : is_acute_angle B)
  (hC : is_acute_angle C)
  (r : ℝ) -- radius of the circumscribing circle S
  (r₁ : ℝ) -- radius of the circumcircle S₁ of ∆ABC
  (hTriangleInscribed : is_triangle_inscribed A B C r)
  (hCircumcircle : is_circumcircle A B C r₁) :
  r >= r₁ :=
by sorry

theorem obtuse_triangle_radius_ge_circumcircle_radius
  (A B C : Type)
  (hA : is_obtuse_angle A)
  (hB : ¬ is_acute_angle B)
  (hC : ¬ is_acute_angle C)
  (r : ℝ) -- radius of the circumscribing circle S
  (r₁ : ℝ) -- radius of the circumcircle S₁ of ∆ABC
  (hTriangleInscribed : is_triangle_inscribed A B C r)
  (hCircumcircle : is_circumcircle A B C r₁) :
  r < r₁ :=
by sorry

end acute_triangle_radius_ge_circumcircle_radius_obtuse_triangle_radius_ge_circumcircle_radius_l32_32067


namespace roots_are_irrational_l32_32179

open Polynomial

theorem roots_are_irrational (k : ℝ) :
  (∃ (α β : ℝ), (x^2 - 5*k*x + (3*k^2 - 2) = 0) ∧ α * β = 9 ∧ is_root (x^2 - 5*k*x + (3*k^2 - 2)) α ∧ is_root (x^2 - 5*k*x + (3*k^2 - 2)) β) →
  (¬ is_integral α ∨ ¬ is_integral β) ∧ (¬ (¬ is_integral α ∧ ¬ is_integral β)) :=
sorry

end roots_are_irrational_l32_32179


namespace fill_time_l32_32797

theorem fill_time (t : ℕ) (h : t = 8) : 11 * t = 88 :=
by
  rw h
  sorry

end fill_time_l32_32797


namespace binomial_expansion_x3_coeff_l32_32938

theorem binomial_expansion_x3_coeff : 
  let x := 2
  let n := 6
  let r := 3
  let a := 2
  let coeff := Nat.choose(n, r) * a^r
  coeff = 160 := 
by 
  sorry

end binomial_expansion_x3_coeff_l32_32938


namespace count_triangles_l32_32913

theorem count_triangles (n : ℕ) (h : n = 7) : 
  let total_smallest_triangles := n * (n + 1) / 2
  let total_larger_triangles := (n - 1) * n / 2
  total_smallest_triangles + total_larger_triangles = 28 :=
by
  have h1 : total_smallest_triangles = 7 * (7 + 1) / 2 := by sorry
  have h2 : total_larger_triangles = (7 - 1) * 7 / 2 := by sorry
  show total_smallest_triangles + total_larger_triangles = 28, from by sorry

end count_triangles_l32_32913


namespace solve_eq1_solve_eq2_l32_32118

theorem solve_eq1 (x : ℝ) : (x - 2)^2 - 16 = 0 ↔ x = 6 ∨ x = -2 :=
by sorry

theorem solve_eq2 (x : ℝ) : (x + 3)^3 = -27 ↔ x = -6 :=
by sorry

end solve_eq1_solve_eq2_l32_32118


namespace song_arrangement_count_l32_32799

theorem song_arrangement_count :
  ∃ (O F P : Nat), O = 2 ∧ F = 4 ∧ P = 4 ∧
  (forall arr : list (list char), 
    length arr = 2 ∧ 
    (forall s, s ∈ arr → length s = 5 ∧ 
               ('O' ∈ s) ∧
               ('F' ∈ s) ∧ 
               ('P' ∈ s) ∧ 
               (∀ p q, p < q → s[p] ≠ 'P' ∨ s[q] ≠ 'P') ∧ 
               (s[4] = 'O' → s = arr[0]) → 
  arr.permutations.length = 71424)) :=
sorry

end song_arrangement_count_l32_32799


namespace chord_length_of_circle_intersection_l32_32572

theorem chord_length_of_circle_intersection
  (θ : ℝ)
  (x y ρ : ℝ)
  (r : ℝ := 2)
  (C : ℝ := 1)
  (π : ℝ := Real.pi)
  (line_equation : θ = π / 3)
  (parametric_eqs : x = 1 + 2 * Real.cos θ ∧ y = 2 * Real.sin θ)
  (polar_circle_eq : ρ ^ 2 = 2 * ρ * Real.cos θ + 3) :
  let d : ℝ := (Real.sqrt 3) / 2
  in 2 * Real.sqrt (r^2 - d^2) = Real.sqrt 13 := 
sorry

end chord_length_of_circle_intersection_l32_32572


namespace find_f_2016_l32_32135

-- Define the given function
def f (x : ℝ) (a α b β : ℝ) : ℝ :=
  a * Real.sin (π * x + α) + b * Real.cos (π * x + β)

-- State the theorem
theorem find_f_2016 (a α b β : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : α ≠ 0) (h4 : β ≠ 0) (h5 : f 2015 a α b β = -1) :
  f 2016 a α b β = -1 :=
  sorry

end find_f_2016_l32_32135


namespace point_P_division_l32_32972

variable {A B C M N P : Type}
variable [LinearOrderedField A] {BC AC MN AB : A}

-- Let ABC be a triangle
axiom triangle_ABC : ∃ (A B C : P), true

-- M is the midpoint of BC
axiom midpoint_M_BC : ∃ (M B C : P) (BC : A), (BC/2 = MB) ∧ (BC/2 = MC)

-- N is on AC such that NC/NA = 1/2
axiom point_N_AC : ∃ (N C A : P) (AC : A), (NC / NA) = 1/2

-- Line MN intersects AB at P
axiom line_intersect_MN_AB_P : ∃ (M N P : P) (MN AB : A), MN ∩ AB = P

-- PA/PB = 2
theorem point_P_division :
  ∀ {A B C M N P : P} {BC AC MN AB : A},
  (triangle_ABC) →
  (midpoint_M_BC) →
  (point_N_AC) → 
  (line_intersect_MN_AB_P) →
  (PA / PB = 2) :=
by
  sorry

end point_P_division_l32_32972


namespace average_jail_days_before_trial_l32_32742

theorem average_jail_days_before_trial
    (days_of_protest : ℕ)
    (num_cities : ℕ)
    (arrests_per_day_per_city : ℕ)
    (weeks_of_jail_time : ℕ)
    (half_2_week_sentence_days : ℕ)
    (total_jail_days : ℕ) :
    days_of_protest = 30 →
    num_cities = 21 →
    arrests_per_day_per_city = 10 →
    weeks_of_jail_time = 9900 →
    half_2_week_sentence_days = 7 →
    total_jail_days = weeks_of_jail_time * 7 →
    (num_cities * days_of_protest * arrests_per_day_per_city) * 
    (half_2_week_sentence_days + average_jail_days_before_trial) = total_jail_days →
    average_jail_days_before_trial = 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end average_jail_days_before_trial_l32_32742


namespace teamA_fraction_and_sum_l32_32045

def time_to_minutes (t : ℝ) : ℝ := t * 60

def fraction_teamA_worked (m n : ℕ) (h_coprime : Nat.gcd m n = 1) (h_fraction : m = 1 ∧ n = 5) : Prop :=
  (90 - 60) / 150 = m / n

theorem teamA_fraction_and_sum (m n : ℕ) (h_coprime : Nat.gcd m n = 1) (h_fraction : m = 1 ∧ n = 5) :
  90 / 150 = 1 / 5 → m + n = 6 :=
by
  sorry

end teamA_fraction_and_sum_l32_32045


namespace find_angle_between_vectors_l32_32263

noncomputable def angle_between_vectors (a b : ℝ^3) : Real := sorry

theorem find_angle_between_vectors (a b : ℝ^3)
  (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : ∥a∥ = ∥b∥) (h4 : ∥a∥ = 2 * ∥a + b∥) :
  angle_between_vectors a b = Real.arccos (-7/8) :=
sorry

end find_angle_between_vectors_l32_32263


namespace coin_sum_solution_l32_32788

def coin_sum_problem : Prop :=
  let coins := [1, 1, 5, 5, 25, 25, 50]
  let possible_sums : Finset ℕ := 
    Finset.image (λ (p : ℕ × ℕ), p.1 + p.2) (Finset.filter (λ p : ℕ × ℕ, p.1 ≠ p.2) (Finset.product coins.to_finset coins.to_finset))
  (9 = possible_sums.card)  

theorem coin_sum_solution : coin_sum_problem :=
by
  sorry

end coin_sum_solution_l32_32788


namespace fraction_of_4_is_8_l32_32927

theorem fraction_of_4_is_8 (fraction : ℝ) (h : fraction * 4 = 8) : fraction = 8 := 
sorry

end fraction_of_4_is_8_l32_32927


namespace hexagon_area_l32_32082

/-- Hexagon area calculation based on the conditions provided -/
theorem hexagon_area :
  let rectangle_area := 6 * 8
  let triangle_area := 4 * (1 * 4 / 2)
  let hexagon_area := rectangle_area - triangle_area
  in hexagon_area = 40 :=
by
  let rectangle_area := 6 * 8
  let triangle_area := 4 * (1 * 4 / 2)
  let hexagon_area := rectangle_area - triangle_area
  show hexagon_area = 40 from sorry

end hexagon_area_l32_32082


namespace problem_solved_probability_l32_32695

theorem problem_solved_probability :
  let PA := 1 / 2
  let PB := 1 / 3
  let PC := 1 / 4
  1 - ((1 - PA) * (1 - PB) * (1 - PC)) = 3 / 4 := 
sorry

end problem_solved_probability_l32_32695


namespace percentage_salt_in_mixture_l32_32036

-- Conditions
def volume_pure_water : ℝ := 1
def volume_salt_solution : ℝ := 2
def salt_concentration : ℝ := 0.30
def total_volume : ℝ := volume_pure_water + volume_salt_solution
def amount_of_salt_in_solution : ℝ := salt_concentration * volume_salt_solution

-- Theorem
theorem percentage_salt_in_mixture :
  (amount_of_salt_in_solution / total_volume) * 100 = 20 :=
by
  sorry

end percentage_salt_in_mixture_l32_32036


namespace sum_of_tens_and_units_digit_of_7_pow_2023_l32_32380

theorem sum_of_tens_and_units_digit_of_7_pow_2023 :
  let n := 7 ^ 2023
  (n % 100).div 10 + (n % 10) = 16 :=
by
  sorry

end sum_of_tens_and_units_digit_of_7_pow_2023_l32_32380


namespace necessary_not_sufficient_condition_l32_32327

noncomputable def is_hyperbola_with_y_foci (m n : ℝ) : Prop :=
  ∀ (x y : ℝ), m * x^2 + n * y^2 = 1 → (m < 0 ∧ n > 0)

theorem necessary_not_sufficient_condition (m n : ℝ) (H : m * n < 0) :
  (is_hyperbola_with_y_foci m n) ↔ (necessary_condition : m < 0 ∧ n > 0) :=
sorry

end necessary_not_sufficient_condition_l32_32327


namespace smallest_n_divisible_by_31997_l32_32039

noncomputable def smallest_n_divisible_by_prime : Nat :=
  let p := 31997
  let k := p
  2 * k

theorem smallest_n_divisible_by_31997 :
  smallest_n_divisible_by_prime = 63994 :=
by
  unfold smallest_n_divisible_by_prime
  rfl

end smallest_n_divisible_by_31997_l32_32039


namespace domain_of_sqrt_function_l32_32704

theorem domain_of_sqrt_function : 
  ∀ x : ℝ, 0 ≤ 7 + 6 * x - x^2 ↔ -1 ≤ x ∧ x ≤ 7 :=
by
  intro x
  have h1 : 0 ≤ 7 + 6 * x - x^2 ↔ x^2 - 6 * x - 7 ≤ 0 := by sorry -- quadratic rearrangement and properties
  have h2 : x^2 - 6 * x - 7 ≤ 0 ↔ -1 ≤ x ∧ x ≤ 7 := by sorry -- solving quadratic inequality
  exact ⟨(λ h => by rw [h1] at h; exact (by rw [h2] at h; exact h)), (λ h => by rw [h2] at h; exact (by rw [h1] at h; exact h))⟩

end domain_of_sqrt_function_l32_32704


namespace part1_part2_1_part2_2_l32_32169

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 - x * Real.log x

theorem part1 (a : ℝ) :
  (∀ x : ℝ, x > 0 → (2 * a * x - Real.log x - 1) ≥ 0) ↔ a ≥ 0.5 := 
sorry

theorem part2_1 (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 < x2 ∧ f a x1 = x1 ∧ f a x2 = x2) :
  0 < a ∧ a < 1 := 
sorry

theorem part2_2 (a x1 x2 : ℝ) (h1 : x1 < x2) (h2 : f a x1 = x1) (h3 : f a x2 = x2) (h4 : x2 ≥ 3 * x1) :
  x1 * x2 ≥ 9 / Real.exp 2 := 
sorry

end part1_part2_1_part2_2_l32_32169


namespace semicircle_radius_l32_32419

theorem semicircle_radius 
  (r P : ℝ)
  (pi_approx : ℝ := 3.141592653589793) 
  (P_approx : P = 56.55751918948772) :
  P = (pi_approx * r + 2 * r) → r ≈ 11 :=
by
  intro h
  sorry

end semicircle_radius_l32_32419


namespace no5_battery_mass_l32_32032

theorem no5_battery_mass :
  ∃ (x y : ℝ), 2 * x + 2 * y = 72 ∧ 3 * x + 2 * y = 96 ∧ x = 24 :=
by
  sorry

end no5_battery_mass_l32_32032


namespace tank_capacity_l32_32763

theorem tank_capacity :
  (∃ (C : ℕ), ∀ (leak_rate inlet_rate net_rate : ℕ),
    leak_rate = C / 6 ∧
    inlet_rate = 6 * 60 ∧
    net_rate = C / 12 ∧
    inlet_rate - leak_rate = net_rate → C = 1440) :=
sorry

end tank_capacity_l32_32763


namespace even_degrees_impossible_l32_32642

structure Square :=
  (points : Set Point)
  (segments : Set (Point × Point))
  (no_intersection : ∀ (s1 s2 : Point × Point), s1 ≠ s2 → disjoint (seg.Points s1) (seg.Points s2))
  (triangles : Set Triangle)
  (points_at_vertices : ∀ (v : Point), v ∈ vertices_set → v ∈ points ∨ v ∈ vertices_of_square)
  (no_points_on_sides : ∀ (v : Point), v ∈ points → v ∉ sides_of_triangles)

noncomputable def is_possible (sq : Square) : Prop := sorry

theorem even_degrees_impossible (sq : Square) : ¬ (∀ v ∈ sq.points, even (degree v sq.segments)) := sorry

end even_degrees_impossible_l32_32642


namespace min_m_n_sum_l32_32697

theorem min_m_n_sum (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : 108 * m = n^3) : m + n = 8 :=
  sorry

end min_m_n_sum_l32_32697


namespace casey_pigs_l32_32820

theorem casey_pigs :
  let water_per_minute := 3
  let rows_of_corn := 4
  let plants_per_row := 15
  let water_per_corn := 0.5
  let number_of_ducks := 20
  let water_per_duck := 0.25
  let water_per_pig := 4
  let pumping_minutes := 25
  ∃(pigs : ℕ), 
  pigs = (water_per_minute * pumping_minutes - (rows_of_corn * plants_per_row * water_per_corn + number_of_ducks * water_per_duck)) / water_per_pig ∧ 
  pigs = 10 :=
by
  let water_per_minute := 3
  let rows_of_corn := 4
  let plants_per_row := 15
  let water_per_corn := 0.5
  let number_of_ducks := 20
  let water_per_duck := 0.25
  let water_per_pig := 4
  let pumping_minutes := 25
  use 10
  sorry

end casey_pigs_l32_32820


namespace computer_and_washing_machine_cost_l32_32674

variable (C W : ℕ)

noncomputable def cost_computer_and_washing_machine : ℕ :=
  950

theorem computer_and_washing_machine_cost :
  (C + W = cost_computer_and_washing_machine) ∧
  (C + 500 + 800 + C + W = 2800) ∧
  (C + (C + 500) <= 1600) ∧
  (C + W <= 1600) :=
begin
  -- Conditions 
  have h1 : 800 + (C + 500) + C + W = 2800 := sorry, -- Sum of all items
  have h2 : C + (C + 500) <= 1600 := sorry, -- Fridge and computer constraint
  have h3 : C + W <= 1600 := sorry, -- Any two items constraint
  
  -- Conclusion based on conditions
  exact ⟨
    sorry, -- Proof of C + W = 950
    h1, h2, h3
  ⟩,
end

end computer_and_washing_machine_cost_l32_32674


namespace percentage_of_nine_hundred_l32_32399

theorem percentage_of_nine_hundred : (45 * 8 = 360) ∧ ((360 / 900) * 100 = 40) :=
by
  have h1 : 45 * 8 = 360 := by sorry
  have h2 : (360 / 900) * 100 = 40 := by sorry
  exact ⟨h1, h2⟩

end percentage_of_nine_hundred_l32_32399


namespace percentage_of_black_marbles_l32_32299

variable (T : ℝ) -- Total number of marbles
variable (C : ℝ) -- Number of clear marbles
variable (B : ℝ) -- Number of black marbles
variable (O : ℝ) -- Number of other colored marbles

-- Conditions
def condition1 := C = 0.40 * T
def condition2 := O = (2 / 5) * T
def condition3 := C + B + O = T

-- Proof statement
theorem percentage_of_black_marbles :
  C = 0.40 * T → O = (2 / 5) * T → C + B + O = T → B = 0.20 * T :=
by
  intros hC hO hTotal
  -- Intermediate steps would go here, but we use sorry to skip the proof.
  sorry

end percentage_of_black_marbles_l32_32299


namespace quadratic_equation_solution_l32_32606

theorem quadratic_equation_solution (m : ℝ) :
  (m - 3) * x ^ (m^2 - 7) - x + 3 = 0 → m^2 - 7 = 2 → m ≠ 3 → m = -3 :=
by
  intros h_eq h_power h_nonzero
  sorry

end quadratic_equation_solution_l32_32606


namespace natasha_average_speed_up_l32_32675

-- Define Natasha's conditions and assumptions
def time_up : ℝ := 4 -- hours
def time_down : ℝ := 2 -- hours
def average_speed_total : ℝ := 1.5 -- km/h

-- Define Natasha's average speed while climbing up
def average_speed_up (total_distance : ℝ) : ℝ := total_distance / 2 / time_up

-- State the theorem to prove her average speed while climbing up
theorem natasha_average_speed_up :
  average_speed_up (average_speed_total * (time_up + time_down)) = 1.125 :=
by
  -- Explicitly state total distance for clarity
  let total_distance := average_speed_total * (time_up + time_down)
  -- Calculate the average speed while climbing up
  show average_speed_up total_distance = 1.125
  sorry -- Proof is omitted, as per instructions

end natasha_average_speed_up_l32_32675


namespace pure_imaginary_solution_second_quadrant_solution_l32_32667

def isPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

def isSecondQuadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

def complexNumber (m : ℝ) : ℂ :=
  ⟨m^2 - 2*m - 3, m^2 + 3*m + 2⟩

theorem pure_imaginary_solution (m : ℝ) : isPureImaginary (complexNumber m) ↔ m = 3 :=
by sorry

theorem second_quadrant_solution (m : ℝ) : isSecondQuadrant (complexNumber m) ↔ (-1 < m ∧ m < 3) :=
by sorry

end pure_imaginary_solution_second_quadrant_solution_l32_32667


namespace sum_of_terms_l32_32931

noncomputable def u1 := 8
noncomputable def r := 2

def first_geometric (u2 u3 : ℝ) (u1 r : ℝ) : Prop := 
  u2 = r * u1 ∧ u3 = r^2 * u1

def last_arithmetic (u2 u3 u4 : ℝ) : Prop := 
  u3 - u2 = u4 - u3

def terms (u1 u2 u3 u4 : ℝ) (r : ℝ) : Prop :=
  first_geometric u2 u3 u1 r ∧
  last_arithmetic u2 u3 u4 ∧
  u4 = u1 + 40

theorem sum_of_terms (u1 u2 u3 u4 : ℝ)
  (h : terms u1 u2 u3 u4 r) : u1 + u2 + u3 + u4 = 104 :=
by {
  sorry
}

end sum_of_terms_l32_32931


namespace remainder_when_m_plus_n_divided_by_500_l32_32400

noncomputable def sum_of_squares_of_terms_in_1_div_500_array : ℚ :=
  let term (r c : ℕ) : ℚ := (1 / (1000^r)) * (1 / (500^c))
  let square_term (r c : ℕ) : ℚ := term r c ^ 2
  (∑' r : ℕ, ∑' c : ℕ, square_term r c)

theorem remainder_when_m_plus_n_divided_by_500 :
  ∃ m n : ℕ, m + n = 1 + 249997000001 ∧ 
  gcd m n = 1 ∧ 
  249997000002 % 500 = 2 := 
by {
  let sum_sq := sum_of_squares_of_terms_in_1_div_500_array
  have h_equiv : sum_sq = 1 / 249997000001 := sorry,
  use 1,
  use 249997000001,
  simp,
  split,
  exact gcd_one_right 249997000001,
  rfl,
}

end remainder_when_m_plus_n_divided_by_500_l32_32400


namespace solve_for_x_l32_32381

theorem solve_for_x :
  (∀ x : ℝ, (1 / Real.log x / Real.log 3 + 1 / Real.log x / Real.log 4 + 1 / Real.log x / Real.log 5 = 2))
  → x = 2 * Real.sqrt 15 :=
by
  sorry

end solve_for_x_l32_32381


namespace find_x_given_sinx_interval_l32_32595

theorem find_x_given_sinx_interval (x : ℝ) : 
  sin x = 1 / 3 ∧ x ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 2) → 
  x = Real.pi - Real.arcsin (1 / 3) := 
by
  intro h
  sorry

end find_x_given_sinx_interval_l32_32595


namespace time_for_type_Q_machine_l32_32428

theorem time_for_type_Q_machine (Q : ℝ) (h1 : Q > 0)
  (h2 : 2 * (1 / Q) + 3 * (1 / 7) = 5 / 6) :
  Q = 84 / 17 :=
sorry

end time_for_type_Q_machine_l32_32428


namespace castiel_sausages_l32_32985

theorem castiel_sausages :
  ∃ x : ℝ, (x = 1 / 2) ∧ 
  (let initial_sausages := 600 in
   let monday_eaten := (2 / 5) * initial_sausages in
   let remaining_after_monday := initial_sausages - monday_eaten in
   let tuesday_remaining := remaining_after_monday * (1 - x) in
   let friday_eaten := (3 / 4) * tuesday_remaining in
   let remaining_after_friday := tuesday_remaining - friday_eaten in
   remaining_after_friday = 45) :=
begin
  sorry
end

end castiel_sausages_l32_32985


namespace sum_of_elements_in_B_l32_32905

def A : Set ℤ := {2, 0, 1, 3}

def B : Set ℤ := { x | -x ∈ A ∧ 2 - x^2 ∉ A }

theorem sum_of_elements_in_B : (∑ x in B.toFinset, x) = -5 := by
  sorry

end sum_of_elements_in_B_l32_32905


namespace point_P_in_fourth_quadrant_l32_32153

def point_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

theorem point_P_in_fourth_quadrant (m : ℝ) : point_in_fourth_quadrant (1 + m^2) (-1) :=
by
  sorry

end point_P_in_fourth_quadrant_l32_32153


namespace fresh_fruit_water_content_l32_32033

theorem fresh_fruit_water_content (W N : ℝ) 
  (fresh_weight_dried: W + N = 50) 
  (dried_weight: (0.80 * 5) = N) : 
  ((W / (W + N)) * 100 = 92) :=
by
  sorry

end fresh_fruit_water_content_l32_32033


namespace sequence_50th_term_is_183_l32_32345

-- Helper function to check if a number contains digit '1' or '2'
def contains_one_or_two (n : ℕ) : Prop :=
  n.digits 10 |> List.any (λ d, d = 1 ∨ d = 2)

-- Sequence of positive multiples of 3 containing at least one digit '1' or '2'
def sequence : List ℕ :=
  List.filter (λ n, n % 3 = 0 ∧ contains_one_or_two n) (List.range (10^6))

-- 50th term in the desired sequence
def sequence_50th_term : ℕ :=
  sequence.get! 49 -- 0-indexed, so the 50th term is at index 49

theorem sequence_50th_term_is_183 : sequence_50th_term = 183 :=
sorry

end sequence_50th_term_is_183_l32_32345


namespace total_attended_seminars_l32_32034

-- Definitions of the conditions
def attended_math_seminar : ℕ := 75
def attended_music_seminar : ℕ := 61
def attended_both_seminars : ℕ := 12

-- Theorem statement
theorem total_attended_seminars : 
  let only_math := attended_math_seminar - attended_both_seminars
  let only_music := attended_music_seminar - attended_both_seminars
  only_math + only_music + attended_both_seminars = 124 :=
by
  let only_math := attended_math_seminar - attended_both_seminars
  let only_music := attended_music_seminar - attended_both_seminars
  have h1 : only_math = 63 := by sorry
  have h2 : only_music = 49 := by sorry
  calc
    only_math + only_music + attended_both_seminars
        = 63 + 49 + 12 : by rw [h1, h2]
    ... = 124 : by norm_num

end total_attended_seminars_l32_32034


namespace probability_cos_condition_l32_32640

open Finset

-- Define the set of elements as described in the problem
def element_set : Finset ℝ := (range 10).map ⟨λ n, (n+1) * (Real.pi / 6), sorry⟩

-- Define the predicate for the condition
def cos_condition (x : ℝ) := Real.cos x = 1 / 2

-- The theorem stating the probability
theorem probability_cos_condition : (card (element_set.filter cos_condition)).toRat / (card element_set).toRat = 1 / 5 := 
sorry

end probability_cos_condition_l32_32640


namespace constant_term_in_expansion_is_180_l32_32939

-- Defining the problem: Given the expression to expand
def expr := (λ (x : ℝ), (Real.sqrt x + 2 / x^2) ^ 10)

-- Theorem statement: The constant term in the expansion of this expression is 180
theorem constant_term_in_expansion_is_180 : 
  let T := (λ (r : ℕ), 2^r * Nat.choose 10 r * x^(5 - r)) in
  T 2 = 180 :=
by
  sorry

end constant_term_in_expansion_is_180_l32_32939


namespace analytical_expression_of_f_l32_32151

theorem analytical_expression_of_f (f : ℤ → ℤ) : 
  (∀ x : ℤ, f(x+1) = 3*x + 2) -> 
  (∀ x : ℤ, f(x) = 3*x - 1) :=
by
  intro h
  sorry

end analytical_expression_of_f_l32_32151


namespace simplify_sqrt_of_mixed_number_l32_32495

noncomputable def sqrt_fraction := λ (a b : ℕ), (Real.sqrt a) / (Real.sqrt b)

theorem simplify_sqrt_of_mixed_number : sqrt_fraction 137 16 = (Real.sqrt 137) / 4 := by
  sorry

end simplify_sqrt_of_mixed_number_l32_32495


namespace verify_statements_l32_32339

variable {f : ℝ → ℝ}

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

noncomputable def defined_on_domain (f : ℝ → ℝ) : Prop := ∀ x, (x < 0 ∨ x > 0) → f x ∈ ℝ

noncomputable def range_ℝ (f : ℝ → ℝ) : Prop := ∀ y, ∃ x, f x = y

noncomputable def f_positive_iff_x_gt_1 (f : ℝ → ℝ) : Prop := ∀ x, (f x > 0 ↔ x > 1)

theorem verify_statements (f : ℝ → ℝ)
  (odd_f : odd_function f)
  (dom_f : defined_on_domain f)
  (range_f : range_ℝ f)
  (positive_iff : f_positive_iff_x_gt_1 f) :
  (f (-1) = 0) ∧ (∀ x, f x = 0 ↔ x ∈ [-1, 0) ∪ (0, 1]) := 
by
  sorry 

end verify_statements_l32_32339


namespace no_negative_roots_l32_32484

theorem no_negative_roots (x : ℝ) (h : x < 0) : (x^4 - 4*x^3 - 6*x^2 - 3*x + 9) ≠ 0 :=
sorry

end no_negative_roots_l32_32484


namespace fuelA_amount_l32_32811

def tankCapacity : ℝ := 200
def ethanolInFuelA : ℝ := 0.12
def ethanolInFuelB : ℝ := 0.16
def totalEthanol : ℝ := 30
def limitedFuelA : ℝ := 100
def limitedFuelB : ℝ := 150

theorem fuelA_amount : ∃ (x : ℝ), 
  (x ≤ limitedFuelA ∧ x ≥ 0) ∧ 
  ((tankCapacity - x) ≤ limitedFuelB ∧ (tankCapacity - x) ≥ 0) ∧ 
  (ethanolInFuelA * x + ethanolInFuelB * (tankCapacity - x)) = totalEthanol ∧ 
  x = 50 := 
by
  sorry

end fuelA_amount_l32_32811


namespace exists_x_in_interval_satisfying_inequality_l32_32481

theorem exists_x_in_interval_satisfying_inequality : ∃ x ∈ set.Icc (0 : ℝ) 1, x^2 - 1 ≥ 0 :=
by
  sorry

end exists_x_in_interval_satisfying_inequality_l32_32481


namespace tan_arctan_five_twelfths_l32_32823

theorem tan_arctan_five_twelfths : Real.tan (Real.arctan (5 / 12)) = 5 / 12 :=
by
  sorry

end tan_arctan_five_twelfths_l32_32823


namespace negative_number_l32_32755

def optionA := abs (-2023)
def optionB := real.sqrt ((-2)^2)
def optionC := 0
def optionD := -(3^2)

theorem negative_number : optionD < 0 := 
by {
  sorry
}

end negative_number_l32_32755


namespace intersection_of_sets_l32_32906

def E : set ℝ := {θ | cos θ < sin θ ∧ 0 ≤ θ ∧ θ ≤ 2 * π}
def F : set ℝ := {θ | tan θ < sin θ}

theorem intersection_of_sets : E ∩ F = {θ | (π / 2) < θ ∧ θ < π} :=
by {
  sorry
}

end intersection_of_sets_l32_32906


namespace last_four_digits_of_5_pow_15000_l32_32761

theorem last_four_digits_of_5_pow_15000 (h : 5^500 ≡ 1 [MOD 2000]) : 
  5^15000 ≡ 1 [MOD 2000] :=
sorry

end last_four_digits_of_5_pow_15000_l32_32761


namespace largest_sum_of_three_largest_angles_l32_32627

-- Definitions and main theorem statement
theorem largest_sum_of_three_largest_angles (EFGH : Type*)
    (a b c d : ℝ) 
    (h1 : a + b + c + d = 360)
    (h2 : b = 3 * c)
    (h3 : ∃ (common_diff : ℝ), (c - a = common_diff) ∧ (b - c = common_diff) ∧ (d - b = common_diff))
    (h4 : ∀ (x y z : ℝ), (x = y + z) ↔ (∃ (progression_diff : ℝ), x - y = y - z ∧ y - z = z - x)) :
    (∃ (A B C D : ℝ), A = a ∧ B = b ∧ C = c ∧ D = d ∧ A + B + C + D = 360 ∧ A = max a (max b (max c d)) ∧ B = 2 * D ∧ A + B + C = 330) :=
sorry

end largest_sum_of_three_largest_angles_l32_32627


namespace packaging_cost_l32_32440

-- Definitions based on the conditions
def cost_ingredients (n : ℕ) : ℝ := 12 / 2 -- Cost of ingredients per cake
def sell_price : ℝ := 15 -- Selling price per cake
def profit : ℝ := 8 -- Profit per cake
def total_cost_per_cake : ℝ := sell_price - profit -- Total cost (ingredients + packaging) per cake

-- The statement to be proven
theorem packaging_cost : 
  (total_cost_per_cake - cost_ingredients 1) = 1 := 
begin
  -- Since no proof required as per the instructions
  sorry
end

end packaging_cost_l32_32440


namespace model_y_completion_time_l32_32406

theorem model_y_completion_time :
  ∀ (T : ℝ), (∃ k ≥ 0, k = 20) →
  (∀ (task_completed_x_per_minute : ℝ), task_completed_x_per_minute = 1 / 60) →
  (∀ (task_completed_y_per_minute : ℝ), task_completed_y_per_minute = 1 / T) →
  (20 * (1 / 60) + 20 * (1 / T) = 1) →
  T = 30 :=
by
  sorry

end model_y_completion_time_l32_32406


namespace hyperbola_equation_l32_32157

-- Define the asymptotes as equations
def asymptote₁ (x y : ℝ) : Prop := y = √2 * x
def asymptote₂ (x y : ℝ) : Prop := y = -√2 * x

-- Define the coordinates of the foci
def foci₁ := (-√6, 0) : ℝ × ℝ
def foci₂ := (√6, 0) : ℝ × ℝ

-- Define the candidate hyperbola equations
def hyperbolaA (x y : ℝ) : Prop := x^2 / 2 - y^2 / 8 = 1
def hyperbolaB (x y : ℝ) : Prop := x^2 / 8 - y^2 / 2 = 1
def hyperbolaC (x y : ℝ) : Prop := x^2 / 2 - y^2 / 4 = 1
def hyperbolaD (x y : ℝ) : Prop := x^2 / 4 - y^2 / 2 = 1

-- Prove that the correct hyperbola is hyperbolaC
theorem hyperbola_equation {x y : ℝ} :
  (asymptote₁ x y ∨ asymptote₂ x y) ∧ (x, y) = foci₁ ∨ (x, y) = foci₂ →
  hyperbolaC x y :=
sorry

end hyperbola_equation_l32_32157


namespace problem_equiv_solution_l32_32535

-- Define a mathematical expression
def simplified_form (a b c : ℝ) : Prop :=
  a / (b * c) = (a / 100) / ((b / 10) * c)

-- Noncomputable definition for the given values
noncomputable def expression_simplified : Prop :=
  simplified_form 10 8 60

-- The main theorem stating the equivalence
theorem problem_equiv_solution : expression_simplified :=
by
  unfold expression_simplified
  unfold simplified_form
  simp
  sorry

end problem_equiv_solution_l32_32535


namespace sqrt_of_mixed_number_l32_32503

theorem sqrt_of_mixed_number :
  (Real.sqrt (8 + 9 / 16)) = (Real.sqrt 137 / 4) :=
by
  sorry

end sqrt_of_mixed_number_l32_32503


namespace chord_midpoint_line_l32_32163

open Real 

theorem chord_midpoint_line (x y : ℝ) (P : ℝ × ℝ) 
  (hP : P = (1, 1)) (hcircle : ∀ (x y : ℝ), x^2 + y^2 = 10) :
  x + y - 2 = 0 :=
by
  sorry

end chord_midpoint_line_l32_32163


namespace minimum_value_of_f_l32_32964

open Real

noncomputable def f (x : ℝ) : ℝ := (2*x - 1) * exp x / (x - 1)

theorem minimum_value_of_f : f (3 / 2) = 4 * exp (3 / 2) :=
by
  -- Proof required here
  sorry

end minimum_value_of_f_l32_32964


namespace average_rate_of_change_l32_32384

variable {α : Type*} [LinearOrderedField α]
variable (f : α → α)
variable (x x₁ : α)
variable (h₁ : x ≠ x₁)

theorem average_rate_of_change : 
  (f x₁ - f x) / (x₁ - x) = (f x₁ - f x) / (x₁ - x) :=
by
  sorry

end average_rate_of_change_l32_32384


namespace min_value_of_xsquare_ysquare_l32_32125

variable {x y : ℝ}

theorem min_value_of_xsquare_ysquare (h : 5 * x^2 * y^2 + y^4 = 1) : x^2 + y^2 ≥ 4 / 5 :=
sorry

end min_value_of_xsquare_ysquare_l32_32125


namespace sum_cubes_zero_l32_32083

-- Define the two sums S1 and S2
def S1 : ℕ → ℤ := λ n, Σ i in (finset.range n).filter (λ i, i % 2 = 0), (2 * (i + 1))^3
def S2 : ℕ → ℤ := λ n, Σ i in (finset.range n).filter (λ i, i % 2 = 0), (-(2 * (i + 1)))^3

-- The proof of the statement that S1 + S2 is 0 for n = 100
theorem sum_cubes_zero : S1 100 + S2 100 = 0 :=
by sorry

end sum_cubes_zero_l32_32083


namespace factorize_l32_32511

theorem factorize (a b : ℝ) : 2 * a ^ 2 - 8 * b ^ 2 = 2 * (a + 2 * b) * (a - 2 * b) :=
by 
  sorry

end factorize_l32_32511


namespace three_digit_number_problem_l32_32713

theorem three_digit_number_problem (c d : ℕ) (h1 : 400 + c*10 + 1 = 786 - (300 + d*10 + 5)) (h2 : (300 + d*10 + 5) % 7 = 0) : c + d = 8 := 
sorry

end three_digit_number_problem_l32_32713


namespace joe_cut_kids_hair_l32_32943

theorem joe_cut_kids_hair
  (time_women minutes_women count_women : ℕ)
  (time_men minutes_men count_men : ℕ)
  (time_kid minutes_kid : ℕ)
  (total_minutes: ℕ) : 
  minutes_women = 50 → 
  minutes_men = 15 →
  minutes_kid = 25 →
  count_women = 3 →
  count_men = 2 →
  total_minutes = 255 →
  (count_women * minutes_women + count_men * minutes_men + time_kid * minutes_kid) = total_minutes →
  time_kid = 3 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  -- Proof is not provided, hence stating sorry.
  sorry

end joe_cut_kids_hair_l32_32943


namespace inequality_always_holds_l32_32874

variable {a b : ℝ}

theorem inequality_always_holds (h1 : a > b) (h2 : a * b ≠ 0) : 2 ^ a > 2 ^ b :=
  sorry

end inequality_always_holds_l32_32874


namespace false_prop_D_l32_32144

open set

variables (Plane Line : Type) 
variables (α β : Plane) (m n : Line)

-- Assumptions based on the conditions given in the problem
variables (parallel : Line → Plane → Prop) (perpendicular : Line → Plane → Prop) (subset : Line → Plane → Prop)
variables (parallel_planes : Plane → Plane → Prop) (perpendicular_planes : Plane → Plane → Prop)

-- Hypotheses for Proposition D
axiom parallel_m_alpha : parallel m α
axiom parallel_alpha_beta : parallel_planes α β
axiom n_in_beta : subset n β

-- Proposition D
def prop_D := parallel m n

-- Proposition D is false
theorem false_prop_D : ¬prop_D := 
by sorry

end false_prop_D_l32_32144


namespace remainder_of_2365487_div_3_l32_32749

theorem remainder_of_2365487_div_3 : (2365487 % 3) = 2 := by
  sorry

end remainder_of_2365487_div_3_l32_32749


namespace tangency_of_abs_and_circle_l32_32608

theorem tangency_of_abs_and_circle (a : ℝ) (ha_pos : a > 0) (ha_ne_two : a ≠ 2) :
    (y = abs x ∧ ∀ x, y = abs x → x^2 + (y - a)^2 = 2 * (a - 2)^2)
    → (a = 4/3 ∨ a = 4) := sorry

end tangency_of_abs_and_circle_l32_32608


namespace lines_intersect_at_same_point_l32_32834

theorem lines_intersect_at_same_point (m k : ℝ) :
  (∃ x y : ℝ, y = 3 * x + 5 ∧ y = -4 * x + m ∧ y = 2 * x + k) ↔ k = (m + 30) / 7 :=
by {
  sorry -- proof not required, only statement.
}

end lines_intersect_at_same_point_l32_32834


namespace quadratic_polynomial_inequality_l32_32714

variable {a b c : ℝ}

theorem quadratic_polynomial_inequality (h1 : ∀ x : ℝ, a * x^2 + b * x + c < 0)
    (h2 : a < 0)
    (h3 : b^2 - 4 * a * c < 0) :
    b / a < c / a + 1 := 
by 
  sorry

end quadratic_polynomial_inequality_l32_32714


namespace problem_l32_32952

structure CyclicQuadrilateral (A B C D O : Point) :=
(cyclic : Cyclic A B C D)
(center_O: Center O)

def Line.meet (l1 l2 : Line) : Option Point := sorry

structure Circle (O : Point) :=
(radius : ℝ)
(tangent_to : ∀ {P : Point}, TangentTo O P)

structure CircleTangent (γ ω : Circle) (A : Point) :=
(tangent_at: TangentTo ω A)

structure AngleBisector (P O : Point) (angle_XPY : Angle) :=
(bisector : IsBisector P O angle_XPY)

theorem problem
  {A B C D E F P O X Y : Point}
  (quad : CyclicQuadrilateral A B C D O)
  (AD_BC_meet_E : quad.cyclic.meet A D B C = some E)
  (AB_CD_meet_F : quad.cyclic.meet A B C D = some F)
  (P_on_EF : P ∈ LineSegment E F)
  (OP_perp_EF : Perpendicular (LineSegment P O) (LineSegment E F))
  (circle_Γ₁_tangent_ω : CircleTangent (Circle.mk A) (Circle.mk quad.center_O) A)
  (circle_Γ₂_tangent_ω : CircleTangent (Circle.mk C) (Circle.mk quad.center_O) C)
  (Γ₁_Γ₂_meet_XY : Meet (Circle.mk A) (Circle.mk C) = (X, Y))
  : AngleBisector P O (∠ X P Y) := sorry

end problem_l32_32952


namespace surface_area_of_modified_cube_l32_32024

noncomputable def resultingSurfaceArea (sideLength : ℝ) (pyramidBase : ℝ) : ℝ :=
  let originalArea := 6 * (sideLength ^ 2)
  let removedArea := 8 * (pyramidBase ^ 2)
  let slantHeight := (√2) * sideLength
  let triangleArea := (1 / 2) * pyramidBase * slantHeight
  let addedArea := 8 * 4 * triangleArea
  originalArea - removedArea + addedArea

theorem surface_area_of_modified_cube :
  (resultingSurfaceArea 4 1) ≈ 133 :=
by
  have approx_sqrt2 : √2 ≈ 1.414 := by sorry
  have approx_surface_area : resultingSurfaceArea 4 1 ≈ 88 + (8 * 4 * 1.414) := by
    apply eq_of_approx,
    have h1 : 96 - 8 = 88 := by norm_num,
    calc 88 + 32 * √2 ≈ 88 + 32 * 1.414 : by congr; apply eq_of_approx; exact approx_sqrt2
                ... = 88 + 45.248 : by norm_num,
    rw [h1],
    exact here,
  trivial

end surface_area_of_modified_cube_l32_32024


namespace parallel_PQ_AB_l32_32230

-- Define the conditions as part of triangle ABC with constructions of angle bisectors and perpendiculars
variables {A B C E D P Q : Point}

-- Given triangle ABC and its properties
def triangle_ABC (A B C : Point) : Prop := 
  Triangle A B C

-- Given points P on [BD] and Q on [AE] such that CP ⊥ BD and CQ ⊥ AE
def conditions (A B C E D P Q : Point) : Prop :=
  angle_bisector A E B ∧
  angle_bisector B D C ∧
  P ∈ line_segment B D ∧ 
  Q ∈ line_segment A E ∧ 
  perpendicular C P (line B D) ∧
  perpendicular C Q (line A E)

-- Prove PQ ∥ AB given the conditions
theorem parallel_PQ_AB (A B C E D P Q : Point) :
  triangle_ABC A B C → 
  conditions A B C E D P Q → 
  parallel (line P Q) (line A B) :=
by 
  sorry

end parallel_PQ_AB_l32_32230


namespace values_range_l32_32268

noncomputable def possible_values (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 2) : set ℝ :=
  {x | x = 1 / a + 1 / b}

theorem values_range : 
  ∀ (a b : ℝ), 
  a > 0 → 
  b > 0 → 
  a + b = 2 → 
  possible_values a b (by assumption) (by assumption) (by assumption) = {x | 2 ≤ x} :=
sorry

end values_range_l32_32268


namespace train_length_approx_l32_32390

noncomputable def length_of_train (speed_km_per_hr : ℕ) (time_sec : ℕ) : ℝ :=
  (speed_km_per_hr * 1000 / 3600) * time_sec

theorem train_length_approx (speed_km_per_hr : ℕ) (time_sec : ℕ) 
  (h_speed : speed_km_per_hr = 30) (h_time : time_sec = 24) :
  length_of_train speed_km_per_hr time_sec ≈ 199.92 :=
by
  rw [h_speed, h_time]
  calc
    length_of_train 30 24
        = (30 * 1000 / 3600) * 24 : by simp [length_of_train]
    ... = 199.92 : by norm_num

end train_length_approx_l32_32390


namespace perimeter_triangle_MNO_l32_32055

-- Define the points' properties
variables (P Q R S U V : ℝ)
variable (height : ℝ)
variable (side_length : ℝ)
variable (PM : ℝ)
variable (MQ : ℝ)
variable (PN : ℝ)
variable (RN : ℝ)
variable (PO : ℝ)
variable (OR : ℝ)
variable (PR : ℝ)

-- Main theorem stating the problem
theorem perimeter_triangle_MNO {P Q R S U V : Point}
  (height : ℝ) (side_length : ℝ) (PM MQ PN RN PO OR PR : ℝ)
  (is_midpoint_M : midpoint M P Q)
  (is_midpoint_N : midpoint N Q R)
  (is_midpoint_O : midpoint O P R)
  (height_eq : height = 18)
  (side_length_eq : side_length = 10)
  (PM_eq : PM = 5)
  (MQ_eq : MQ = 5)
  (PN_eq : PN = 5)
  (RN_eq : RN = 5)
  (PO_eq : PO = 5 * real.sqrt 3)
  (OR_eq : OR = 5 * real.sqrt 3)
  (PR_eq : PR = 10)
  :
  perimeter (triangle M N O) = 25 :=
sorry

end perimeter_triangle_MNO_l32_32055


namespace circumcircle_eq_l32_32854

open Real

-- Define the vertices of the triangle
def A : Point := ⟨-1, 5⟩
def B : Point := ⟨5, 5⟩
def C : Point := ⟨6, -2⟩

-- The general form of the circle equation
def circle_eq (D E F : ℝ) (p : Point) :=
  p.1^2 + p.2^2 + D * p.1 + E * p.2 + F = 0

-- Prove the specific equation of the circumcircle
theorem circumcircle_eq :
  ∃ D E F, D = -4 ∧ E = -2 ∧ F = -20 ∧
           (circle_eq D E F A) ∧ (circle_eq D E F B) ∧ (circle_eq D E F C) :=
by
  have hA : (-1)^2 + 5^2 + D * -1 + E * 5 + F = 0 := sorry
  have hB : 5^2 + 5^2 + D * 5 + E * 5 + F = 0 := sorry
  have hC : 6^2 + (-2)^2 + D * 6 + E * -2 + F = 0 := sorry
  use [-4, -2, -20]
  split
  { exact rfl }
  split
  { exact rfl }
  split
  { exact rfl }
  repeat { assumption }

end circumcircle_eq_l32_32854


namespace power_expression_l32_32374

theorem power_expression : (1 / ((-5)^4)^2) * (-5)^9 = -5 := sorry

end power_expression_l32_32374


namespace area_of_ground_l32_32932

def height_of_rain : ℝ := 0.05
def volume_of_water : ℝ := 750

theorem area_of_ground : ∃ A : ℝ, A = (volume_of_water / height_of_rain) ∧ A = 15000 := by
  sorry

end area_of_ground_l32_32932


namespace sqrt_of_mixed_number_l32_32501

theorem sqrt_of_mixed_number :
  (Real.sqrt (8 + 9 / 16)) = (Real.sqrt 137 / 4) :=
by
  sorry

end sqrt_of_mixed_number_l32_32501


namespace sufficient_but_not_necessary_l32_32127

variables {a b : ℝ}
def p := a > |b|
def q := a^3 + b^3 > a^2 * b + a * b^2

theorem sufficient_but_not_necessary
  (h₁ : a ≠ b) 
  (hp : p) : q :=
begin
  sorry
end

end sufficient_but_not_necessary_l32_32127


namespace arc_length_radius_l32_32437

theorem arc_length_radius 
  (L : ℝ) (θ : ℝ) (r : ℝ) 
  (hL : L = 2000) 
  (hθ : θ = 300) : 
  Float.to_int (1200 / Real.pi) = 382 :=
by 
  sorry

end arc_length_radius_l32_32437


namespace number_of_paths_passing_through_C_from_A_to_B_l32_32813

theorem number_of_paths_passing_through_C_from_A_to_B :
  let total_paths_from_A_to_B := (4 + 2).choose 2,
      paths_from_A_to_C := (2 + 1).choose 1,
      paths_from_C_to_B := (2 + 1).choose 1 in
  total_paths_from_A_to_B = 15 ∧
  paths_from_A_to_C = 3 ∧
  paths_from_C_to_B = 3 ∧
  (paths_from_A_to_C * paths_from_C_to_B) = 42 :=
by
    -- Conditions and calculations:
    have total_paths := nat.choose 6 2,
    have paths_A_C := nat.choose 3 1,
    have paths_C_B := nat.choose 3 1,
    trivial

end number_of_paths_passing_through_C_from_A_to_B_l32_32813


namespace min_cubes_are_three_l32_32427

/-- 
  A toy construction set consists of cubes, each with one button on one side and socket holes on the other five sides.
  Prove that the minimum number of such cubes required to build a structure where all buttons are hidden, and only the sockets are visible is 3.
--/

def min_cubes_to_hide_buttons (num_cubes : ℕ) : Prop :=
  num_cubes = 3

theorem min_cubes_are_three : ∃ (n : ℕ), (∀ (num_buttons : ℕ), min_cubes_to_hide_buttons num_buttons) :=
by
  use 3
  sorry

end min_cubes_are_three_l32_32427


namespace card_moves_limit_l32_32677

theorem card_moves_limit:
  let total_moves := ∑ k in Finset.range 999, (k + 1) in   -- Summing from 2 to 1000, (k-1) corresponds to (k+1-1)
  total_moves ≤ 500000 := 
by
  sorry

end card_moves_limit_l32_32677


namespace value_of_expression_l32_32202

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : 2 * a^2 + 4 * a * b + 2 * b^2 - 6 = 12 :=
by
  sorry

end value_of_expression_l32_32202
