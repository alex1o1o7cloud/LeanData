import Mathlib

namespace probability_less_than_one_third_l586_586130

def interval := set.Ioo (0 : ℝ) (1 / 2)

def desired_interval := set.Ioo (0 : ℝ) (1 / 3)

theorem probability_less_than_one_third :
  (∃ p : ℝ, p = (volume desired_interval / volume interval) ∧ p = 2 / 3) :=
  sorry

end probability_less_than_one_third_l586_586130


namespace exterior_angle_polygon_l586_586171

theorem exterior_angle_polygon (n : ℕ) (h : 360 / n = 72) : n = 5 :=
by
  sorry

end exterior_angle_polygon_l586_586171


namespace inequality_solution_l586_586644

theorem inequality_solution :
  { x : ℝ | (x-1)/(x+4) ≤ 0 } = { x : ℝ | (-4 < x ∧ x ≤ 0) ∨ (x = 1) } :=
by 
  sorry

end inequality_solution_l586_586644


namespace sum_all_x_values_eq_nine_l586_586991

theorem sum_all_x_values_eq_nine (x y z : ℂ) :
  (x + yz = 9) → (y + xz = 14) → (z + xy = 14) → 
  let solutions := {(x, y, z) | ∃ y z, x + yz = 9 ∧ y + xz = 14 ∧ z + xy = 14} in
  (∑ (p : ℂ × ℂ × ℂ) in solutions, p.1) = 9 :=
sorry

end sum_all_x_values_eq_nine_l586_586991


namespace shop_dimension_is_100_l586_586297

-- Given conditions
def monthly_rent : ℕ := 1300
def annual_rent_per_sqft : ℕ := 156

-- Define annual rent
def annual_rent : ℕ := monthly_rent * 12

-- Define dimension to prove
def dimension_of_shop : ℕ := annual_rent / annual_rent_per_sqft

-- The theorem statement
theorem shop_dimension_is_100 :
  dimension_of_shop = 100 :=
by
  sorry

end shop_dimension_is_100_l586_586297


namespace boats_people_distribution_l586_586366

theorem boats_people_distribution (boats people : ℕ) (h_boats : boats = 5) (h_people : people = 15) :
  ¬∃ (x : Fin 5 → ℕ), (∀ i, x i ≥ 0) ∧ (∑ i, x i = people) ∧ (∀ i j, (i ≠ j) → x i ≠ x j) ↔ 
  ∃ (x : Fin 5 → ℕ), (∀ i, x i ≥ 0) ∧ (∑ i, x i = people) :=
begin
  sorry
end

end boats_people_distribution_l586_586366


namespace milk_water_ratio_l586_586693

theorem milk_water_ratio (x : ℝ) (h1 : x > 0) :
  let v1 := 3 * x,
      v2 := 5 * x,
      v3 := 7 * x,
      m1 := 1 / 3 * v1,
      w1 := 2 / 3 * v1,
      m2 := 3 / 5 * v2,
      w2 := 2 / 5 * v2,
      m3 := 2 / 5 * v3,
      w3 := 3 / 5 * v3,
      total_milk := m1 + m2 + m3,
      total_water := w1 + w2 + w3
  in total_milk / total_water = 34 / 41 := sorry

end milk_water_ratio_l586_586693


namespace symmetric_angles_fraction_l586_586075

theorem symmetric_angles_fraction
  (θ : ℝ)
  (x y : ℝ)
  (h1 : 0 < θ ∧ θ < 360)
  (h2 : (θ = 60) ∨ (θ = 60 + 180))
  (h3 : y = √3 * x ∧ (x ≠ 0 ∨ y ≠ 0)) :
  (x * y) / (x^2 + y^2) = √3 / 4 :=
by
  sorry

end symmetric_angles_fraction_l586_586075


namespace domain_of_f_f_even_l586_586012

def f (x : ℝ) : ℝ := log (3 + x) + log (3 - x)

theorem domain_of_f : {x : ℝ | -3 < x ∧ x < 3} = set_of (λ x, -3 < x ∧ x < 3) := by
  sorry

theorem f_even (x : ℝ) (h : -3 < x ∧ x < 3) : f x = f (-x) := by
  sorry

end domain_of_f_f_even_l586_586012


namespace slips_with_number_three_l586_586943

theorem slips_with_number_three : 
  ∀ (total_slips : ℕ) (number3 number8 : ℕ) (E : ℚ), 
  total_slips = 15 → 
  E = 5.6 → 
  number3 + number8 = total_slips → 
  (number3 : ℚ) / total_slips * 3 + (number8 : ℚ) / total_slips * 8 = E →
  number3 = 8 :=
by
  intros total_slips number3 number8 E h1 h2 h3 h4
  sorry

end slips_with_number_three_l586_586943


namespace initial_money_l586_586351

theorem initial_money (M : ℝ)
  (clothes : M * (1 / 3) = M - M * (2 / 3))
  (food : (M - M * (1 / 3)) * (1 / 5) = (M - M * (1 / 3)) - ((M - M * (1 / 3)) * (4 / 5)))
  (travel : ((M - M * (1 / 3)) - ((M - M * (1 / 3)) * (1 / 5))) * (1 / 4) = ((M - M * (1 / 3)) - ((M - M * (1 / 3)) * (1 / 5))) - (((M - M * (1 / 3)) - ((M - M * (1 / 3)) * (1 / 5)))) * (3 / 4))
  (left : ((M - M * (1 / 3)) - (((M - M * (1 / 3)) - ((M - M * (1 / 3)) * (1 / 5))) * (1 / 4))) = 400)
  : M = 1000 := 
sorry

end initial_money_l586_586351


namespace set_subtraction_cardinality_l586_586977

open Set

variable {α : Type} {κ : Type}
variable (𝓐 : κ → Set α)

theorem set_subtraction_cardinality (𝓐 : κ → Set α) :
  (fintype.card (⋃ (A B : κ), {A - B | A ∈ 𝓐 A ∧ B ∈ 𝓐 B})) ≥ fintype.card (⋃ A : κ, 𝓐 A) :=
sorry

end set_subtraction_cardinality_l586_586977


namespace probability_merlin_dismissed_l586_586211

variable {p q : ℝ} (hpq : p + q = 1) (hp : 0 < p ∧ p < 1)

theorem probability_merlin_dismissed : 
  let decision_prob : ℝ := if h : p = 1 then 1 else p in
  if h : decision_prob = p then (1 / 2) else 0 := 
begin
  sorry
end

end probability_merlin_dismissed_l586_586211


namespace parallel_lines_slope_l586_586717

theorem parallel_lines_slope (k : ℝ) :
  (∀ x : ℝ, 15 * x + 5 = (5 * k) * x - 7) → k = 3 :=
by
  intro h
  let m1 := 15
  let m2 := 5 * k
  have h_slope : m1 = m2,
  { sorry }  -- Placeholder for actual proof, since slopes must be equal
  rw [← h_slope]  -- Assuming h_slope gives us the equation 15 = 5 * k
  sorry -- Placeholder for the final proof steps to show k = 3

end parallel_lines_slope_l586_586717


namespace find_m_l586_586531

def vec_a : ℝ × ℝ := (2, 1)
def vec_b : ℝ × ℝ := (1, -1)
def diff := (vec_a.1 - vec_b.1, vec_a.2 - vec_b.2)
def lin_comb (m : ℝ) := (2 * m + 1, m - 1)

theorem find_m (m : ℝ) (h : diff.1 * lin_comb m.1 + diff.2 * lin_comb m.2 = 0) : 
  m = 1/4 := 
by 
  -- the proof would go here
  sorry

end find_m_l586_586531


namespace interval_probability_l586_586150

theorem interval_probability (x y z : ℝ) (h0x : 0 < x) (hy : y > 0) (hz : z > 0) (hx : x < y) (hy_mul_z_eq_half : y * z = 1 / 2) (hx_eq_third_y : x = 1 / 3 * y) :
  (x / y) = (2 / 3) :=
by
  -- Here you should provide the proof
  sorry

end interval_probability_l586_586150


namespace monotonicity_and_extrema_l586_586990

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + (1 + a) * x - x^2 - x^3

theorem monotonicity_and_extrema (a : ℝ) (h_a : 0 < a) :
  (∀ x, f a x < 0) → (f a x > 0) ∧ 
  (∀ x ∈ Icc (0 : ℝ) (1 : ℝ), 
    if a ≥ 4 then (x = 0 → f a x = min_val) ∧ (x = 1 → f a x = max_val)
    else if 0 < a ∧ a < 1 then (x = 1 → f a x = min_val) ∧
    else if a = 1 then (x = 0 → (f a x = min_val) ∧ (x = 1 → f a x = min_val))
    else if 1 < a ∧ a < 4 then (x = 0 → f a x = min_val)
    else (x = (\frac{-1 + sqrt(4 + 3 * a)}{3}) → f a x = max_val)) :=
sorry

end monotonicity_and_extrema_l586_586990


namespace problem_statement_l586_586005

def is_prime (p : ℕ) : Prop := ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

noncomputable def prime_stream (n : ℕ) : ℕ := (nat.primes.drop n).head

-- Function to get the nth prime number.
noncomputable def p_i (i : ℕ) : ℕ := prime_stream (i - 1)

def all_products_sum (k : ℕ) : ℕ :=
  (finset.powerset (finset.range k)).sum (λ s, s.prod (λ i, p_i (i+1)))

theorem problem_statement (k : ℕ) (hk : k > 5) :
  ∃ S : ℕ, S = all_products_sum k ∧ (S + 1) has more than 2k distinct prime factors :=
sorry

end problem_statement_l586_586005


namespace max_unit_digit_2015_l586_586986

def is_divisor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

def unit_digit (n : ℕ) : ℕ :=
  n % 10

theorem max_unit_digit_2015 (d : ℕ) (hd : is_divisor 2015 d) :
  let exp := 2005 / d in
  d > 0 → 
  let unit_d := unit_digit (d ^ exp) in 
        unit_d = 1 ∨
        unit_d = 3 ∨
        unit_d = 5 ∨
        unit_d = 7 ∨
        unit_d = 9 ∧
  7 = List.maximum [unit_digit (1 ^ (2005 / 1)),
                    unit_digit (5 ^ (2005 / 5)),
                    unit_digit (13 ^ (2005 / 13)),
                    unit_digit (31 ^ (2005 / 31)),
                    unit_digit (65 ^ (2005 / 65)),
                    unit_digit (155 ^ (2005 / 155)),
                    unit_digit (403 ^ (2005 / 403)),
                    unit_digit (2015 ^ (2005 / 2015))] := 
sorry

end max_unit_digit_2015_l586_586986


namespace pentagon_ratio_d_a_l586_586471

theorem pentagon_ratio_d_a (a b c d : ℝ) (h1 : c = 2 * a)
  (h2 : -- Property representing the pentagon with required segment lengths and parallel sides
        ∃ (pentagon : ℝ → ℝ → ℝ → Prop), 
        (pentagon a b d) ∧
        ( -- additional geometric properties ensuring ∃ such pentagon with sides of length a, b, c, d
          -- considering all segments lengths and the property about parallel sides, each of length b
        )):
  d / a = Real.sqrt 2 :=
sorry

end pentagon_ratio_d_a_l586_586471


namespace conner_day3_tie_l586_586284

def sydney_initial := 837
def conner_initial := 723

def sydney_day1 := 4
def conner_day1 := 8 * sydney_day1

def sydney_day2 := 0
def conner_day2 := 123

def sydney_day3 := 2 * conner_day1

def sydney_total := sydney_initial + sydney_day1 + sydney_day2 + sydney_day3
def conner_total_day2 := conner_initial + conner_day1 + conner_day2

def conner_day3_required := sydney_total - conner_total_day2

theorem conner_day3_tie (c : ℕ) :
  sydney_initial + sydney_day1 + sydney_day2 + sydney_day3 = 
  conner_initial + conner_day1 + conner_day2 + c 
  ↔ c = 27 := by
  intro h
  sorry

end conner_day3_tie_l586_586284


namespace digit_6_appears_300_times_l586_586724

def appears_in_unit_place (n : ℕ) : Prop :=
  n % 10 = 6

def appears_in_tens_place (n : ℕ) : Prop :=
  (n / 10) % 10 = 6

def appears_in_hundreds_place (n : ℕ) : Prop :=
  (n / 100) % 10 = 6

def count_digit_6 (n : ℕ) : ℕ :=
  (List.range (n + 1)).count appears_in_unit_place + 
  (List.range (n + 1)).count appears_in_tens_place +
  (List.range (n + 1)).count appears_in_hundreds_place

theorem digit_6_appears_300_times : ∃ n : ℕ, count_digit_6 n = 300 := 
  ∃ n : ℕ, n = 1000 ∧ count_digit_6 n = 300

end digit_6_appears_300_times_l586_586724


namespace problem_statement_l586_586889

open Real

noncomputable def f (a x : ℝ) : ℝ := (1/2) * a * x^2 - (2 * a + 1) * x + 2 * log x
def g (x : ℝ) : ℝ := x^2 - 2 * x

theorem problem_statement (a : ℝ) :
  (∀ x1 ∈ Ioc 0 2, ∃ x2 ∈ Ioc 0 2, f a x1 < g x2) → a > log 2 - 1 :=
sorry

end problem_statement_l586_586889


namespace chenny_friends_count_l586_586431

def initial_candies := 10
def additional_candies := 4
def candies_per_friend := 2

theorem chenny_friends_count :
  (initial_candies + additional_candies) / candies_per_friend = 7 := by
  sorry

end chenny_friends_count_l586_586431


namespace find_r_l586_586780

noncomputable def parabola_vertex : (ℝ × ℝ) := (0, -1)

noncomputable def intersection_points (r : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let x1 := (r - Real.sqrt (r^2 + 4)) / 2
  let y1 := r * x1
  let x2 := (r + Real.sqrt (r^2 + 4)) / 2
  let y2 := r * x2
  ((x1, y1), (x2, y2))

noncomputable def triangle_area (r : ℝ) : ℝ :=
  let base := Real.sqrt (r^2 + 4)
  let height := 2
  1/2 * base * height

theorem find_r (r : ℝ) (h : r > 0) : triangle_area r = 32 → r = Real.sqrt 1020 := 
by
  sorry

end find_r_l586_586780


namespace min_value_of_f_l586_586314

open Classical

noncomputable def min_intersection_points (n : ℕ) : ℕ := 4 * n

theorem min_value_of_f (n : ℕ) (h : 0 < n) : 
  ∃ (C : Fin 4n → Circle), (∀ i j, i ≠ j → ¬(C i).IsTangent (C j)) ∧ 
  (∀ i, ∃ j₁ j₂ j₃, j₁ ≠ i ∧ j₂ ≠ i ∧ j₃ ≠ i ∧ 
    (C i).Intersects (C j₁) ∧ (C i).Intersects (C j₂) ∧ (C i).Intersects (C j₃)) ∧
  (∀ f : ℕ, f = ∑ P ∈ intersection_points C, 1) → 
  min_intersection_points n = 4n :=
by sorry

end min_value_of_f_l586_586314


namespace series_sum_l586_586829

theorem series_sum (x : ℝ) (h : x < 1) : 
  ∃ S : ℕ → ℝ, (λ n, S n) ⟶ (1 / (1 - x)^2) at_top :=
sorry

end series_sum_l586_586829


namespace cot_trig_identity_l586_586979

noncomputable def cot (x : Real) : Real :=
  Real.cos x / Real.sin x

theorem cot_trig_identity (a b c α β γ : Real) 
  (habc : a^2 + b^2 = 2021 * c^2) 
  (hα : α = Real.arcsin (a / c)) 
  (hβ : β = Real.arcsin (b / c)) 
  (hγ : γ = Real.arccos ((2021 * c^2 - a^2 - b^2) / (2 * 2021 * c^2))) 
  (h_triangle : a^2 = b^2 + c^2 - 2 * b * c * Real.cos α) :
  cot α / (cot β + cot γ) = 1010 :=
by
  sorry

end cot_trig_identity_l586_586979


namespace probability_of_interval_l586_586118

theorem probability_of_interval (a b x : ℝ) (h : 0 < a ∧ a < b ∧ 0 < x) : 
  (x < b) → (b = 1/2) → (x = 1/3) → (0 < x) → (x - 0) / (b - 0) = 2/3 := 
by 
  sorry

end probability_of_interval_l586_586118


namespace area_of_gray_region_l586_586193

variables (r : ℝ) (π : ℝ) -- radius of the inner circle, constant pi

-- Definitions based on conditions
def outer_radius := 3 * r
def width := outer_radius - r

-- Statement we need to prove
theorem area_of_gray_region (h1 : width = 4) (h2 : π > 0) : 
  let inner_circle_area := π * r ^ 2 in
  let outer_circle_area := π * outer_radius ^ 2 in
  outer_circle_area - inner_circle_area = 32 * π :=
by
  sorry

end area_of_gray_region_l586_586193


namespace sin_C_eq_l586_586494

-- Definitions based on conditions given
def a : ℝ := 1
def b : ℝ := Real.sqrt 2
def A (B C : ℝ) : ℝ := π - B - C
def C (A B : ℝ) : ℝ := 2 * B - A

-- The theorem we want to prove
theorem sin_C_eq : ∀ (B : ℝ), A B (C (A B (π - B - C B)) B) + C (A B (π - B - C B)) B = 2 * B →
                      A B (C (A B (π - B - C B)) B) + B + C (A B (π - B - C B)) B = π →
                      Real.sin (C (A B (π - B - C B)) B) = 1 / Real.sqrt 2 :=
by sorry

end sin_C_eq_l586_586494


namespace regular_polygon_exterior_angle_l586_586167

theorem regular_polygon_exterior_angle (n : ℕ) (h : 72 = 360 / n) : n = 5 :=
by
  sorry

end regular_polygon_exterior_angle_l586_586167


namespace milk_concentration_l586_586445

variable {V_initial V_removed V_total : ℝ}

theorem milk_concentration (h1 : V_initial = 20) (h2 : V_removed = 2) (h3 : V_total = 20) :
    (V_initial - V_removed) / V_total * 100 = 90 := 
by 
  sorry

end milk_concentration_l586_586445


namespace orthocenter_fixed_l586_586867

-- Given a circle ω with center O, and two distinct points A and C on it.
variable {ω : Type*} [MetricSpace ω] {O A C P : ω}
variable [MetricSpace.ball ω]

-- Assume A and C are distinct points on the circle ω and P is an arbitrary point on the circle ω.
variable (hA : A ∈ MetricSpace.ball ω O 1)
variable (hC : C ∈ MetricSpace.ball ω O 1)
variable (hAC : A ≠ C)
variable (hP : P ∈ MetricSpace.ball ω O 1)

-- Let X be the midpoint of AP and Y be the midpoint of CP.
def midpoint (p1 p2 : ω) : ω := MetricSpace.midpoint p1 p2

noncomputable def X := midpoint A P
noncomputable def Y := midpoint C P

-- Define the orthocenter of triangle OXY.
def orthocenter (O X Y : ω) : ω := sorry

-- Define the midpoint of AC.
noncomputable def midpoint_AC := midpoint A C

-- Theorem: The orthocenter H of triangle OXY is always the midpoint of AC.
theorem orthocenter_fixed (O A C P : ω) (hA : A ∈ MetricSpace.ball ω O 1) (hC : C ∈ MetricSpace.ball ω O 1)
  (hAC : A ≠ C) (hP : P ∈ MetricSpace.ball ω O 1) :
  orthocenter O (midpoint A P) (midpoint C P) = midpoint A C := 
sorry

end orthocenter_fixed_l586_586867


namespace second_order_arithmetic_progression_a100_l586_586560

theorem second_order_arithmetic_progression_a100 :
  ∀ (a : ℕ → ℕ), 
    a 1 = 2 → 
    a 2 = 3 → 
    a 3 = 5 → 
    (∀ n, a (n + 1) - a n = n) → 
    a 100 = 4952 :=
by
  intros a h1 h2 h3 hdiff
  sorry

end second_order_arithmetic_progression_a100_l586_586560


namespace range_estimate_of_expression_l586_586447

theorem range_estimate_of_expression : 
  6 < (2 * Real.sqrt 2 + Real.sqrt 3) * Real.sqrt 2 ∧ 
       (2 * Real.sqrt 2 + Real.sqrt 3) * Real.sqrt 2 < 7 :=
by
  sorry

end range_estimate_of_expression_l586_586447


namespace probability_of_interval_l586_586119

theorem probability_of_interval (a b x : ℝ) (h : 0 < a ∧ a < b ∧ 0 < x) : 
  (x < b) → (b = 1/2) → (x = 1/3) → (0 < x) → (x - 0) / (b - 0) = 2/3 := 
by 
  sorry

end probability_of_interval_l586_586119


namespace octagon_dissection_l586_586813

theorem octagon_dissection :
  ∃ (O : Point) (S T : Point) (P R Q1 Q2 P1 S1 T1 R1 P2 S2 T2 R2 : Point),
  (is_regular_octagon O [S, T]) ∧ 
  (is_side_of_octagon O S T) ∧ 
  (is_perpendicular O ST P) ∧
  (is_perpendicular O (adjacent_side S) R) ∧
  (is_square O P Q R) ∧
  (intersection PR OQ = Q1) ∧
  (intersection ST OQ = Q2) ∧
  (are_parallel_lines PQ (through Q1)) ∧
  (are_parallel_lines RQ (through Q2)) ∧
  (line_intersects Q1 [OP, OS, OT, OR] [P1, S1, T1, R1]) ∧
  (line_intersects Q2 [OP, OS, OT, OR] [P2, S2, T2, R2]) ∧
  (seg_constructed [O, P1, S1, S, T, T1, Q2]) ∧
  (rotates_by O 45 ([O, P1, S1, S, T, T1, Q2])) ∧
  (symmetry_preserved_through_rotations 45) ∧
  (segment_lengths_equal [OPS, ST, P1S1, S1T1]) ∧
  (seg_valid [T2S1, T2Q2, T2R1]) ∧
  (octagon_reconstructable (8 * [segment]))  :=
sorry

end octagon_dissection_l586_586813


namespace probability_no_correct_letter_five_l586_586316

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | n + 1 => (n + 1) * factorial n

-- Definition of derangements
def derangements (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => 0
  | n + 1 => n * (derangements n + (-1)^n * factorial n)

-- Probability calculation
def probability_no_correct_letter (n : ℕ) : ℚ :=
  if n = 5 then (44 : ℚ) / factorial n else 0

theorem probability_no_correct_letter_five : probability_no_correct_letter 5 = 11 / 30 :=
  by
    -- Using the definitions and known values of derangements and factorial
    -- but not writing the proof steps
    sorry

end probability_no_correct_letter_five_l586_586316


namespace probability_of_interval_l586_586139

theorem probability_of_interval {x : ℝ} (h₀ : 0 < x ∧ x < 1/2) :
    Pr (λ x, x < 1/3) = 2/3 :=
sorry

end probability_of_interval_l586_586139


namespace problem_statement_l586_586008

noncomputable def a : ℝ := 2 * Real.log 0.99
noncomputable def b : ℝ := Real.log 0.98
noncomputable def c : ℝ := Real.sqrt 0.96 - 1

theorem problem_statement : a > b ∧ b > c := by
  sorry

end problem_statement_l586_586008


namespace num_false_statements_l586_586880

-- Definitions and conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

-- Theorem to prove the number of false statements among original, converse, inverse, and contrapositive
theorem num_false_statements (a : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q →
  (¬ (q > 1 → is_increasing_sequence a)) ∧
  (¬ (is_increasing_sequence a → q > 1)) ∧
  (¬ (q ≤ 1 → ¬ (is_increasing_sequence a))) ∧
  (¬ (¬ (is_increasing_sequence a) → q ≤ 1)) →
  4 :=
by { intros, exact 4 }

end num_false_statements_l586_586880


namespace number_of_sides_l586_586178

-- Define the given conditions as Lean definitions

def exterior_angle := 72
def sum_of_exterior_angles := 360

-- Now state the theorem based on these conditions

theorem number_of_sides (n : ℕ) (h1 : exterior_angle = 72) (h2 : sum_of_exterior_angles = 360) : n = 5 :=
by
  sorry

end number_of_sides_l586_586178


namespace geometric_probability_l586_586108

theorem geometric_probability :
  let total_length := (1/2 - 0)
  let favorable_length := (1/3 - 0)
  total_length > 0 ∧ favorable_length > 0 →
  (favorable_length / total_length) = 2 / 3 :=
by
  intros _ _ _
  sorry

end geometric_probability_l586_586108


namespace green_tractor_price_l586_586705

-- Define the conditions
def salary_based_on_sales (r_ct : Nat) (r_price : ℝ) (g_ct : Nat) (g_price : ℝ) : ℝ :=
  0.1 * r_ct * r_price + 0.2 * g_ct * g_price

-- Define the problem's Lean statement
theorem green_tractor_price
  (r_ct : Nat) (g_ct : Nat)
  (r_price : ℝ) (total_salary : ℝ)
  (h_rct : r_ct = 2)
  (h_gct : g_ct = 3)
  (h_rprice : r_price = 20000)
  (h_salary : total_salary = 7000) :
  ∃ g_price : ℝ, salary_based_on_sales r_ct r_price g_ct g_price = total_salary ∧ g_price = 5000 :=
by
  sorry

end green_tractor_price_l586_586705


namespace perfect_square_factors_of_420_l586_586067

theorem perfect_square_factors_of_420 : 
  let p := 420 
  ∧ (∀ d, d ∣ p ↔ (∃ a b c d, d = 2^a * 3^b * 5^c * 7^d 
                      ∧ 0 ≤ a ≤ 2 
                      ∧ 0 ≤ b ≤ 1 
                      ∧ 0 ≤ c ≤ 1 
                      ∧ 0 ≤ d ≤ 1))
  ∧ (∀ d, d ∣ p → (∀ e, d = e * e)) 
  → fintype.card {d // d ∣ p ∧ is_square d} = 2 :=
by sorry

end perfect_square_factors_of_420_l586_586067


namespace find_values_l586_586851

-- Define the conditions as Lean hypotheses
variables (A B : ℝ)

-- State the problem conditions
def condition1 := 30 - (4 * A + 5) = 3 * B
def condition2 := B = 2 * A

-- State the main theorem to be proved
theorem find_values (h1 : condition1 A B) (h2 : condition2 A B) : A = 2.5 ∧ B = 5 :=
by { sorry }

end find_values_l586_586851


namespace real_solutions_equation_l586_586603

theorem real_solutions_equation :
  ∃! x : ℝ, 9 * x^2 - 90 * ⌊ x ⌋ + 99 = 0 :=
sorry

end real_solutions_equation_l586_586603


namespace find_y_l586_586656

theorem find_y :
  ∀ (y : ℚ), (∑ i in Finset.range 100, i + y) / 100 = 50 * y → y = 4950 / 4999 :=
by
  intro y
  sorry

end find_y_l586_586656


namespace cos_double_sum_l586_586074

variable (α β : ℝ)

-- Define the conditions as hypotheses
def condition_1 := sin α + sin β = 1
def condition_2 := cos α + cos β = 0

-- State the theorem that we need to prove
theorem cos_double_sum (h1 : condition_1 α β) (h2 : condition_2 α β) : cos (2 * α) + cos (2 * β) = 1 := 
sorry

end cos_double_sum_l586_586074


namespace shopkeeper_profit_percentage_l586_586390

def cost_price := 100 -- We denote the cost price as a constant for simplicity

def marked_price (cp : ℝ) := cp * 1.30

def discount_percentage := 18.461538461538467 / 100

def selling_price (mp : ℝ) := mp * (1 - discount_percentage)

def profit_percentage (cp sp : ℝ) := ((sp - cp) / cp) * 100

theorem shopkeeper_profit_percentage : 
  profit_percentage cost_price (selling_price (marked_price cost_price)) = 6 := 
by
  sorry

end shopkeeper_profit_percentage_l586_586390


namespace no_positive_real_roots_l586_586249

noncomputable section

def P (x : ℝ) (a : List ℤ) (M : ℤ) (k : ℤ) : ℝ :=
  M * (x + 1)^k - (a.map (λ ai, x + ai)).prod

theorem no_positive_real_roots 
  (a : List ℤ) (k : ℤ) (M : ℤ) 
  (h_sum : (a.map (λ ai, 1/ai.to_float)).sum = k) 
  (h_prod : (a.prod id = M))
  (h_M_gt_1 : M > 1) : 
  ¬ ∃ x : ℝ, x > 0 ∧ P x a M k = 0 :=
by
  sorry

end no_positive_real_roots_l586_586249


namespace number_of_common_tangents_l586_586062

-- Define the conditions of the problem
def C1_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1
def C2_eq (x y : ℝ) : Prop := (x - 2)^2 + (y - 5)^2 = 9

-- Define the centers and radii for circles
def center_C1 : ℝ × ℝ := (1, 2)
def radius_C1 : ℝ := 1

def center_C2 : ℝ × ℝ := (2, 5)
def radius_C2 : ℝ := 3

-- Define the distance between centers function
def distance (A B : ℝ × ℝ) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Statement of the problem
theorem number_of_common_tangents : 
  distance center_C1 center_C2 < radius_C1 + radius_C2 →
  2 = 2 :=
by 
  sorry

end number_of_common_tangents_l586_586062


namespace fixed_point_rotation_l586_586527

noncomputable def reflection_point (P : Point) (e : Line) : Point := sorry

theorem fixed_point_rotation
  (e₁ e₂ : Line) 
  (P A : Point)
  (k : Circle)
  (H₁ : e₁ ≠ e₂)
  (H₂ : k.contains A)
  (H₃ : incidence e₁ A)
  (H₄ : incidence e₂ A)
  (H₅ : ∀ (Q : Point), k.contains Q → k.intersects e₁ Q)
  (H₆ : ∀ (Q : Point), k.contains Q → k.intersects e₂ Q) :
  ∃ B : Point, ∀ (P : Point), k.contains P →
  let P₁ := reflection_point P e₁,
      P₂ := reflection_point P e₂
  in collinear P₁ P₂ B :=
sorry

end fixed_point_rotation_l586_586527


namespace max_value_pq_qr_rs_sp_l586_586312

/-- The values of p, q, r, and s are 2, 3, 4, and 5. Determine the maximum value of pq + qr + rs + sp.--/
theorem max_value_pq_qr_rs_sp 
  (p q r s : ℕ) 
  (hpqrs : (p, q, r, s) ∈ {(2,3,4,5), (2,3,5,4), (2,4,3,5), (2,4,5,3), (2,5,3,4), (2,5,4,3), 
                          (3,2,4,5), (3,2,5,4), (3,4,2,5), (3,4,5,2), (3,5,2,4), (3,5,4,2),
                          (4,2,3,5), (4,2,5,3), (4,3,2,5), (4,3,5,2), (4,5,2,3), (4,5,3,2),
                          (5,2,3,4), (5,2,4,3), (5,3,2,4), (5,3,4,2), (5,4,2,3), (5,4,3,2)}) :
  pq + qr + rs + sp ≤ 49 := 
by 
  sorry

end max_value_pq_qr_rs_sp_l586_586312


namespace prob_0_le_ξ_le_1_l586_586040

noncomputable def ξ : ℝ → MeasureTheory.ProbabilityMeasure ℝ := sorry

axiom normal_distribution : ξ ∼ MeasureTheory.probabilityMeasure (MeasureTheory.NormalDist.mk 1 σ^2)

axiom probability_greater_than_2 : MeasureTheory.Probability ξ {x : ℝ | x > 2} = 0.15

theorem prob_0_le_ξ_le_1 : MeasureTheory.Probability ξ {x : ℝ | 0 ≤ x ∧ x ≤ 1} = 0.35 :=
by
  sorry

end prob_0_le_ξ_le_1_l586_586040


namespace total_wheels_at_station_l586_586687

/--
There are 4 trains at a train station.
Each train has 4 carriages.
Each carriage has 3 rows of wheels.
Each row of wheels has 5 wheels.
The total number of wheels at the train station is 240.
-/
theorem total_wheels_at_station : 
    let number_of_trains := 4
    let carriages_per_train := 4
    let rows_per_carriage := 3
    let wheels_per_row := 5
    number_of_trains * carriages_per_train * rows_per_carriage * wheels_per_row = 240 := 
by
    sorry

end total_wheels_at_station_l586_586687


namespace pool_depth_equivalent_l586_586794

noncomputable def depth_of_pool : ℝ :=
  let d := 60 -- Diameter in feet
  let V := 16964.600329384884 -- Volume in cubic feet
  let r := d / 2 -- Radius in feet
  V / (π * r^2)

theorem pool_depth_equivalent :
  depth_of_pool = 6 := by
  sorry

end pool_depth_equivalent_l586_586794


namespace positive_integer_n_exists_positive_integer_k_l586_586456

theorem positive_integer_n_exists_positive_integer_k 
  (n : ℕ) (h₁ : 0 < n) (h₂ : (log 10 n) ∉ ℚ) : 
  ∃ k : ℕ, 0 < k ∧ (∃ d : ℕ, (n^k % 10 = d ∧ nat.digits 10 (n^k) = d :: _ :: d :: _)) :=
sorry

end positive_integer_n_exists_positive_integer_k_l586_586456


namespace find_other_parallel_side_l586_586840

theorem find_other_parallel_side
  (a b h A : ℝ)
  (h_a : a = 18)
  (h_h : h = 10)
  (h_A : A = 190) :
  b = 20 :=
by
  have h_area : A = (1/2) * (a + b) * h,
    sorry
  rw [h_area, h_a, h_h, h_A] at *,
  -- Continue proving this leads to b = 20
  sorry

end find_other_parallel_side_l586_586840


namespace probability_less_than_third_l586_586110

def interval_real (a b : ℝ) := set.Ioo a b

def probability_space (a b : ℝ) := {
  length := b - a,
  sample_space := interval_real a b
}

def probability_of_event {a b c : ℝ} (h : a < b) (h1 : a < c) (h2 : c < b) : Prop :=
  (c - a) / (b - a)

theorem probability_less_than_third :
  probability_of_event (by norm_num : (0:ℝ) < (1 / 2))
                       (by norm_num : (0:ℝ) < (1 / 3))
                       (by norm_num : (1 / 3) < (1 / 2))
  = (2 / 3) := sorry

end probability_less_than_third_l586_586110


namespace probability_intervals_l586_586087

-- Definitions: interval (0, 1/2), and the condition interval (0, 1/3)
def interval1 := set.Ioo (0:ℝ) (1/2)
def interval2 := set.Ioo (0:ℝ) (1/3)

-- The proof statement: probability of randomly selecting a number from (0, 1/2), being within (0, 1/3), is 2/3.
theorem probability_intervals : (interval2.measure / interval1.measure = 2 / 3) :=
sorry

end probability_intervals_l586_586087


namespace trig_identity_example_l586_586442

theorem trig_identity_example :
  cos (-225 * real.pi / 180) + sin (-225 * real.pi / 180) = 0 :=
by
  sorry

end trig_identity_example_l586_586442


namespace inequality_true_l586_586542

variable (a b : ℝ)

theorem inequality_true (h : a > b ∧ b > 0) : (b^2 / a) < (a^2 / b) := by
  sorry

end inequality_true_l586_586542


namespace table_tennis_matches_l586_586586

def num_players : ℕ := 8

def total_matches (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

theorem table_tennis_matches : total_matches num_players = 28 := by
  sorry

end table_tennis_matches_l586_586586


namespace verify_generalized_distance_l586_586936

def non_neg (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f(x, y) ≥ 0 ∧ (f(x, y) = 0 ↔ (x = 0 ∧ y = 0))

def symmetric (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f(x, y) = f(y, x)

def triangle_ineq (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y z, f(x, y) ≤ f(x, z) + f(z, y)

def generalized_distance (f : ℝ → ℝ → ℝ) : Prop :=
  non_neg f ∧ symmetric f ∧ triangle_ineq f

theorem verify_generalized_distance :
  generalized_distance (λ x y, x^2 + y^2) ∧
  ¬ generalized_distance (λ x y, (x-y)^2) ∧
  ¬ generalized_distance (λ x y, sqrt (x-y)) ∧
  ¬ generalized_distance (λ x y, sin (x-y)) := by
  sorry

end verify_generalized_distance_l586_586936


namespace maximum_value_z_l586_586868

theorem maximum_value_z (u v x y : ℝ) 
    (h1 : u^2 + v^2 = 1) 
    (h2 : x + y - 1 ≥ 0)
    (h3 : x - 2y + 2 ≥ 0)
    (h4 : x ≤ 2) : 
    (∃ z, z = ux + vy ∧ ∀ z', z' = ux + vy → z' ≤ 2 * Real.sqrt 2) := 
begin
  sorry
end

end maximum_value_z_l586_586868


namespace fourth_term_expansion_l586_586725

theorem fourth_term_expansion (b x : ℝ) : 
  ( ∑ k in finset.range (8), (finset.binom 7 k) * (b/x)^(7-k) * (-x^2/b^3)^k ) = -35 * x^2 / b^5 :=
by
  sorry

end fourth_term_expansion_l586_586725


namespace cycle_powers_of_i_l586_586448

open Complex

theorem cycle_powers_of_i :
  (Complex.I ^ 23456) + (Complex.I ^ 23457) + (Complex.I ^ 23458) + (Complex.I ^ 23459) = 0 :=
by
  have h1 : Complex.I ^ 4 = 1 := by sorry
  have h2 : Complex.I ^ 23456 = 1 := by sorry
  have h3 : Complex.I ^ 23457 = Complex.I := by sorry
  have h4 : Complex.I ^ 23458 = -1 := by sorry
  have h5 : Complex.I ^ 23459 = -Complex.I := by sorry
  calc (Complex.I ^ 23456) + (Complex.I ^ 23457) + (Complex.I ^ 23458) + (Complex.I ^ 23459)
       = 1 + Complex.I + (-1) + (-Complex.I) : by rw [h2, h3, h4, h5]
   ... = 0 : by ring

end cycle_powers_of_i_l586_586448


namespace probability_of_interval_l586_586142

theorem probability_of_interval {x : ℝ} (h₀ : 0 < x ∧ x < 1/2) :
    Pr (λ x, x < 1/3) = 2/3 :=
sorry

end probability_of_interval_l586_586142


namespace prob_1_lt_X_lt_3_l586_586039

noncomputable def X : ℝ → ℝ := sorry

axiom X_normal_dist : ∀ x, X x = pdf_normal 3 \(σ^2)
axiom prob_X_less_5 : ∫ (x : ℝ) from -∞ to 5, X x = 0.8

theorem prob_1_lt_X_lt_3 : (∫ (x : ℝ) from 1 to 3, X x) = 0.3 := 
by {
    sorry
}

end prob_1_lt_X_lt_3_l586_586039


namespace problem_statement_l586_586891

noncomputable def line_equation (t α : ℝ) : (ℝ × ℝ) :=
(1 + t * Real.cos α, t * Real.sin α)

def polar_eq (θ : ℝ) : ℝ := Real.cos θ / (Real.sin θ * Real.sin θ)

noncomputable def rectangular_eq (x y : ℝ) : Prop := y^2 = x

def focus_F : (ℝ × ℝ) := (1/4, 0)

noncomputable def area_triangle (A B F : ℝ × ℝ) : ℝ :=
1/2 * Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.abs(F.2 - (A.2 + B.2) / 2)

theorem problem_statement
  (α π : ℝ)
  (t : ℝ)
  (P A B : ℝ × ℝ)
  (H1 : (polar_eq π / 2) = 1)
  (H2 : Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)
         = 2 * (Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) * Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)))
  (H3 : line_equation t α = A)
  (H4 : line_equation t α = B)
  (H5 : rectangular_eq 1 1)
  (H6 : focus_F = (1/4, 0))
  (d : ℝ := 1 - 1/4)
  (AB_dist : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2))
  : area_triangle A B focus_F = 3 / 4 := 
sorry

end problem_statement_l586_586891


namespace goldfish_count_15_weeks_l586_586996

def goldfish_count_after_weeks (initial : ℕ) (weeks : ℕ) : ℕ :=
  let deaths := λ n => 10 + 2 * (n - 1)
  let purchases := λ n => 5 + 2 * (n - 1)
  let rec update_goldfish (current : ℕ) (week : ℕ) :=
    if week = 0 then current
    else 
      let new_count := current - deaths week + purchases week
      update_goldfish new_count (week - 1)
  update_goldfish initial weeks

theorem goldfish_count_15_weeks : goldfish_count_after_weeks 35 15 = 15 :=
  by
  sorry

end goldfish_count_15_weeks_l586_586996


namespace height_of_screen_is_100_l586_586672

-- Definitions for the conditions and the final proof statement
def side_length_of_square_paper := 20 -- cm

def perimeter_of_square_paper (s : ℕ) : ℕ := 4 * s

def height_of_computer_screen (P : ℕ) := P + 20

theorem height_of_screen_is_100 :
  let s := side_length_of_square_paper in
  let P := perimeter_of_square_paper s in
  height_of_computer_screen P = 100 :=
by
  sorry

end height_of_screen_is_100_l586_586672


namespace union_in_lambda_system_l586_586237

variable (Ω : Type) (L : Set (Set Ω))

def is_lambda_system := 
  ∀ (S ∈ L), 
    (Ω ∈ L) ∧
    (∀ A ∈ L, Ω \ A ∈ L) ∧
    (∀ A B ∈ L, A ∩ B = ∅ → A ∪ B ∈ L)

theorem union_in_lambda_system 
  {A B : Set Ω}
  (hL : is_lambda_system Ω L)
  (hA : A ∈ L)
  (hB : B ∈ L)
  (hAB : A ∩ B = ∅) :
  A ∪ B ∈ L :=
by
  sorry

end union_in_lambda_system_l586_586237


namespace radius_of_circle_l586_586694

theorem radius_of_circle (a b : ℝ) (R : ℝ) : 
  ∃ R, R = sqrt(a^2 - a * b + b^2) :=
by
  use sqrt(a^2 - a * b + b^2)
  sorry

end radius_of_circle_l586_586694


namespace probability_merlin_dismissed_l586_586222

variable (p : ℝ) (q : ℝ := 1 - p)
variable (r : ℝ := 1 / 2)

theorem probability_merlin_dismissed :
  (0 < p ∧ p ≤ 1) →
  (q = 1 - p) →
  (r = 1 / 2) →
  (probability_king_correct_two_advisors = p) →
  (strategy_merlin_minimizes_dismissal = true) →
  (strategy_percival_honest = true) →
  probability_merlin_dismissed = 1 / 2 := by
  sorry

end probability_merlin_dismissed_l586_586222


namespace number_of_valid_functions_l586_586970

-- Definitions based on conditions
def A : Finset ℕ := (Finset.range 2013).erase 0  -- {1, 2, ..., 2012}
def B : Finset ℕ := (Finset.range 20).erase 0    -- {1, 2, ..., 19}
def S : Finset (Finset ℕ) := (Finset.powerset A)  -- Set of all subsets of A

-- Function property: f(A1 ∩ A2) = min(f(A1), f(A2))
def validFunction (f : Finset ℕ → ℕ) : Prop :=
  ∀ (A1 A2 : Finset ℕ), f (A1 ∩ A2) = min (f A1) (f A2)

-- Main theorem to state the required number of functions
theorem number_of_valid_functions : 
  ∃ (count : ℕ), count =
    ∑ n in B, n ^ 2012 :=
sorry

end number_of_valid_functions_l586_586970


namespace proof_of_problem_l586_586532

variables {V : Type*} [inner_product_space ℝ V] [normed_group V] [normed_space ℝ V]

noncomputable def problem_statement (a b c : V) : Prop :=
  ∥a∥ = 1 ∧ ∥b∥ = 1 ∧ ∥a + b∥ = 2 ∧ (c - a - 2 • b = 4 • (a ×ₗ b)) ∧ (b ⬝ c = 3)

theorem proof_of_problem (a b c : V) (h : ∥a∥ = 1 ∧ ∥b∥ = 1 ∧ ∥a + b∥ = 2 ∧ (c - a - 2 • b = 4 • (a ×ₗ b))) : 
  b ⬝ c = 3 :=
sorry

end proof_of_problem_l586_586532


namespace two_numbers_equal_l586_586473

theorem two_numbers_equal 
  (a b c : ℝ) 
  (n : ℕ) 
  (h : ∀ n : ℕ, (a^n, b^n, c^n) forms_triangle) :
  a = b ∨ b = c ∨ a = c :=
sorry

end two_numbers_equal_l586_586473


namespace range_of_a_l586_586504

variables {x a : ℝ}

def p := x^2 + 2*x - 3 > 0
def q := x > a

theorem range_of_a
  (H1 : ∀ x, ¬(q x) → ¬(p x))
  (H2 : ¬q = (λ x, x ≤ a)) 
  (H3 : ¬p = (λ x, -3 ≤ x ∧ x ≤ 1)) : 
  a ≥ 1 := 
sorry

end range_of_a_l586_586504


namespace construct_remaining_vertices_l586_586692

-- Define the vertices of the hexagon and their projections.
structure Hexagon (V : Type) :=
(A B C D E F : V)

structure Projection (V : Type) :=
(A1 B1 C1 D1 E1 F1 : V)

-- Define the properties of the hexagon and its parallel projection.
def isRegularHexagon {V : Type} [euclideanSpace V] (hex : Hexagon V) : Prop :=
  let ⟨A, B, C, D, E, F⟩ := hex in
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D E ∧
  dist D E = dist E F ∧ dist E F = dist F A ∧
  (∀ {u v w : V}, (u - v) ∧ (v - w) ∧ ∠u v w = π / 3)

def isParallelProjection {V : Type} [euclideanSpace V] (proj : Projection V) (hex : Hexagon V) : Prop :=
  let ⟨A1, B1, C1, D1, E1, F1⟩ := proj in
  let ⟨A, B, C, D, E, F⟩ := hex in
  -- Conditions mirroring the construction of projections should translate to:
  (∃ (P : AffineSubspace ℝ V), 
    parallel (A1 - B1) (A - B) ∧ parallel (B1 - C1) (B - C) ∧ 
    parallel (C1 - D1) (C - D) ∧ parallel (D1 - E1) (D - E) ∧ 
    parallel (E1 - F1) (E - F) ∧ parallel (F1 - A1) (F - A) ∧
    dist A1 B1 = dist B1 C1 ∧ dist B1 C1 = dist C1 D1 ∧
    dist C1 D1 = dist D1 E1 ∧ dist D1 E1 = dist E1 F1 ∧
    dist E1 F1 = dist F1 A1)

-- The main theorem that ties together the question and conditions.
theorem construct_remaining_vertices {V : Type} [euclideanSpace V]
  (proj : Projection V) (hex : Hexagon V)
  (h_reg_hex : isRegularHexagon hex) 
  (h_proj : isParallelProjection ⟨proj.A1, proj.B1, proj.C1, (0 : V), (0 : V), (0 : V)⟩ hex) :
  ∃ (proj_full : Projection V), isParallelProjection proj_full hex :=
sorry

end construct_remaining_vertices_l586_586692


namespace interval_probability_l586_586153

theorem interval_probability (x y z : ℝ) (h0x : 0 < x) (hy : y > 0) (hz : z > 0) (hx : x < y) (hy_mul_z_eq_half : y * z = 1 / 2) (hx_eq_third_y : x = 1 / 3 * y) :
  (x / y) = (2 / 3) :=
by
  -- Here you should provide the proof
  sorry

end interval_probability_l586_586153


namespace orthocenter_fixed_position_l586_586864

-- Definitions of circle, midpoints, and orthocenter
structure Circle :=
  (center : Point)
  (radius : ℝ)

structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def midpoint (p1 p2 : Point) : Point :=
  { x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2 }

noncomputable def orthocenter (o p1 p2 : Point) := sorry 

theorem orthocenter_fixed_position (O A C P X Y H : Point) (ω : Circle) 
  (hA_on_circle : A ∈ ω)
  (hC_on_circle : C ∈ ω)
  (hP_on_circle : P ∈ ω)
  (hX_is_midpoint : X = midpoint A P)
  (hY_is_midpoint : Y = midpoint C P)
  (hH_is_orthocenter : H = orthocenter O X Y) :
  H = midpoint A C :=
sorry

end orthocenter_fixed_position_l586_586864


namespace binary_multiplication_correct_l586_586845

theorem binary_multiplication_correct :
  (0b1101 : ℕ) * (0b1011 : ℕ) = (0b10011011 : ℕ) :=
by
  sorry

end binary_multiplication_correct_l586_586845


namespace part1_part2_l586_586960

-- Definitions corresponding to the problem conditions
variables {A B C O H D E F M N: Point}
variable {triangle_ABC : Triangle}
variable h1 : IsCircumcenter O triangle_ABC
variable h2 : IsOrthocenter H triangle_ABC
variable h3 : IsAltitude AD triangle_ABC -- Assumes altitudes are defined this way
variable h4 : IsAltitude BE triangle_ABC
variable h5 : IsAltitude CF triangle_ABC
variable h6 : Line_inter :: ed AB M -- Representation of ED intersecting AB at M
variable h7 : Line_inter :: fd AC N -- Representation of FD intersecting AC at N

-- Statements to be proved
theorem part1 : Perpendicular OB DF ∧ Perpendicular OC DE := 
sorry

theorem part2 : Perpendicular OH MN := 
sorry

end part1_part2_l586_586960


namespace probability_less_than_third_l586_586112

def interval_real (a b : ℝ) := set.Ioo a b

def probability_space (a b : ℝ) := {
  length := b - a,
  sample_space := interval_real a b
}

def probability_of_event {a b c : ℝ} (h : a < b) (h1 : a < c) (h2 : c < b) : Prop :=
  (c - a) / (b - a)

theorem probability_less_than_third :
  probability_of_event (by norm_num : (0:ℝ) < (1 / 2))
                       (by norm_num : (0:ℝ) < (1 / 3))
                       (by norm_num : (1 / 3) < (1 / 2))
  = (2 / 3) := sorry

end probability_less_than_third_l586_586112


namespace second_term_is_44_l586_586189

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, n ≥ 3 → a n = (1 / (n - 1) * (∑ i in range (n - 1), a i))

theorem second_term_is_44
  (a : ℕ → ℕ)
  (h₁ : a 1 = 8)
  (h₂ : a 10 = 26)
  (h₃ : sequence a) :
  a 2 = 44 :=
sorry

end second_term_is_44_l586_586189


namespace probability_less_than_one_third_l586_586134

def interval := set.Ioo (0 : ℝ) (1 / 2)

def desired_interval := set.Ioo (0 : ℝ) (1 / 3)

theorem probability_less_than_one_third :
  (∃ p : ℝ, p = (volume desired_interval / volume interval) ∧ p = 2 / 3) :=
  sorry

end probability_less_than_one_third_l586_586134


namespace only_n_equal_one_l586_586814

theorem only_n_equal_one (n : ℕ) (hn : 0 < n) : 
  (5 ^ (n - 1) + 3 ^ (n - 1)) ∣ (5 ^ n + 3 ^ n) → n = 1 := by
  intro h_div
  sorry

end only_n_equal_one_l586_586814


namespace chenny_friends_l586_586424

noncomputable def num_friends (initial_candies add_candies candies_per_friend) : ℕ :=
  (initial_candies + add_candies) / candies_per_friend

theorem chenny_friends :
  num_friends 10 4 2 = 7 :=
by
  sorry

end chenny_friends_l586_586424


namespace range_of_x_l586_586050

noncomputable def f : ℝ → ℝ
| x := if x ≤ 0 then x^3 else Real.log (x + 1)

theorem range_of_x (x : ℝ) : 
  f (2 - x^2) > f x ↔ -2 < x ∧ x < 1 := by
  sorry

end range_of_x_l586_586050


namespace find_c_l586_586933

def f (x : ℤ) : ℤ := x - 2

def F (x y : ℤ) : ℤ := y^2 + x

theorem find_c : ∃ c, c = F 3 (f 16) ∧ c = 199 :=
by
  use F 3 (f 16)
  sorry

end find_c_l586_586933


namespace sequence_nonzero_l586_586985

def a : ℕ → ℤ
| 0     := 1
| 1     := 2
| (n+2) := if a (n+1) * a n % 2 = 0 then 5 * a (n+1) - 3 * a n else a (n+1) - a n

theorem sequence_nonzero (n : ℕ) : a n ≠ 0 := by
sorry

end sequence_nonzero_l586_586985


namespace asymptotes_of_hyperbola_l586_586459

def hyperbola := { x : ℝ, y : ℝ // x^2 / 4 - y^2 / 3 = 1 }

theorem asymptotes_of_hyperbola :
  ∀ (h : hyperbola), ∃ m : ℝ, (m = ℝ.sqrt 3 / 2 ∨ m = -ℝ.sqrt 3 / 2) → 
  (h.1 ≠ 0 → h.2 = m * h.1) := 
sorry

end asymptotes_of_hyperbola_l586_586459


namespace overall_germination_percentage_l586_586852

theorem overall_germination_percentage :
  let seeds_planted := [300, 200, 400, 150, 250]
  let germ_rates := [0.30, 0.35, 0.25, 0.40, 0.20]
  let seeds_germinated := List.map2 (λ (s : ℕ) (r : ℝ) => s * r) seeds_planted germ_rates
  let total_seeds_planted := (300 + 200 + 400 + 150 + 250 : ℕ)
  let total_seeds_germinated := List.sum seeds_germinated
  in (total_seeds_germinated / total_seeds_planted) * 100 = 28.46 := 
sorry

end overall_germination_percentage_l586_586852


namespace part1_l586_586512

noncomputable def problem1 (b : ℝ) :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  deriv (λ x, x^2 + b * Real.exp x) x₁ = 0 ∧ 
  deriv (λ x, x^2 + b * Real.exp x) x₂ = 0

theorem part1 :
  ∀ b : ℝ, problem1 b ↔ -2/(Real.exp 1) < b ∧ b < 0 :=
by sorry

end part1_l586_586512


namespace line1_line2_line3_l586_586461

-- Line 1: Through (-1, 3), parallel to x - 2y + 3 = 0.
theorem line1 (x y : ℝ) : (x - 2 * y + 3 = 0) ∧ (x = -1) ∧ (y = 3) →
                              (x - 2 * y + 7 = 0) :=
by sorry

-- Line 2: Through (3, 4), perpendicular to 3x - y + 2 = 0.
theorem line2 (x y : ℝ) : (3 * x - y + 2 = 0) ∧ (x = 3) ∧ (y = 4) →
                              (x + 3 * y - 15 = 0) :=
by sorry

-- Line 3: Through (1, 2), with equal intercepts on both axes.
theorem line3 (x y : ℝ) : (x = y) ∧ (x = 1) ∧ (y = 2) →
                              (x + y - 3 = 0) :=
by sorry

end line1_line2_line3_l586_586461


namespace seating_solution_l586_586559

/-- 
Imagine Abby, Bret, Carl, and Dana are seated in a row of four seats numbered from 1 to 4.
Joe observes them and declares:

- "Bret is sitting next to Dana" (False)
- "Carl is between Abby and Dana" (False)

Further, it is known that Abby is in seat #2.

Who is seated in seat #3? 
-/

def seating_problem : Prop :=
  ∃ (seats : ℕ → ℕ),
  (¬ (seats 1 = 1 ∧ seats 1 = 4 ∨ seats 4 = 1 ∧ seats 4 = 4)) ∧
  (¬ (seats 3 > seats 1 ∧ seats 3 < seats 2 ∨ seats 3 > seats 2 ∧ seats 3 < seats 1)) ∧
  (seats 2 = 2) →
  (seats 3 = 3)

theorem seating_solution : seating_problem :=
sorry

end seating_solution_l586_586559


namespace correct_options_l586_586023

def data : List ℕ := [0, 1, 2, 4]

noncomputable def variance (l : List ℕ) : ℚ :=
  let mean := l.sum / l.length
  (l.map (λ x => (x - mean) ^ 2)).sum / l.length

def even_pairs_probability (l : List ℕ) : ℚ :=
  let pairs := l.product l
  let valid_pairs := pairs.filter (λ p => p.1 ≠ p.2)
  let two_digit_nums := valid_pairs.map (λ p => p.1 * 10 + p.2)
  let even_count := (two_digit_nums.filter (λ x => x % 2 = 0)).length
  ↑even_count / valid_pairs.length

theorem correct_options (l : List ℕ) (h : l = data) :
  variance l = 2.1875 ∧ even_pairs_probability l = 7 / 9 := by
  sorry

end correct_options_l586_586023


namespace sales_notebooks_or_markers_l586_586651

noncomputable def percentage_sales (N M O : ℕ) : Prop :=
  N = 42 ∧ M = 25 ∧ O = 100 - (N + M)

theorem sales_notebooks_or_markers :
  percentage_sales 42 25 33 :=
by {
  unfold percentage_sales,
  sorry
}

end sales_notebooks_or_markers_l586_586651


namespace number_of_friends_l586_586427

def initial_candies : ℕ := 10
def additional_candies : ℕ := 4
def total_candies : ℕ := initial_candies + additional_candies
def candies_per_friend : ℕ := 2

theorem number_of_friends : total_candies / candies_per_friend = 7 :=
by
  sorry

end number_of_friends_l586_586427


namespace mixed_oil_rate_is_correct_l586_586355

def rate_of_mixed_oil (volume1 : ℕ) (price1 : ℕ) (volume2 : ℕ) (price2 : ℕ) : ℕ :=
  (volume1 * price1 + volume2 * price2) / (volume1 + volume2)

theorem mixed_oil_rate_is_correct :
  rate_of_mixed_oil 10 50 5 68 = 56 := by
  sorry

end mixed_oil_rate_is_correct_l586_586355


namespace balls_in_boxes_count_l586_586266

/--
There are five balls numbered 1, 2, 3, 4, 5 and three boxes numbered 1, 2, 3.
We need to place the balls into the boxes such that:
1. Each box contains at least one ball.
2. Balls numbered 1 and 2 cannot be in the same box.

The total number of different ways to do this is 114.
-/
theorem balls_in_boxes_count :
  ∃ (f : ℕ → ℕ), (∀ i, 1 ≤ f i ∧ f i ≤ 3) ∧ -- f maps each ball to a box
    ∀ x y, (f 1 ≠ f 2) ∧ (1 ≤ f 1) ∧ (f 1 ≤ 3) ∧ (1 ≤ f 2) ∧ (f 2 ≤ 3) ∧ (∃ box, (1 ≤ box) ∧ (box ≤ 3) ∧ (box ≠ f 1) ∧ (box ≠ f 2) ∧ -- at least one ball in each box
    (∃ b, 1 ≤ b ∧ b ≤ 5 ∧ f b = box)) ∧ -- each box has at least one ball
    (∃ b1 b2 b3 b4 b5, ∀ i, 1 ≤ i ∧ i ≤ 3 → ∃ j, 1 ≤ j ∧ j ≤ 5 ∧ f j = i) -- all 5 balls are placed such that each box has at least one ball
    ∧ (card {x : set (ℕ × ℕ) | set.pairwise_disjoint (set.nonempty x)} = 114) := -- count total ways to place balls
sorry

end balls_in_boxes_count_l586_586266


namespace zeros_of_quadratic_l586_586313

theorem zeros_of_quadratic : ∃ x : ℝ, x^2 - x - 2 = 0 -> (x = -1 ∨ x = 2) :=
by
  sorry

end zeros_of_quadratic_l586_586313


namespace find_f_31_over_2_l586_586792

def f : ℝ → ℝ := sorry   -- Definition to match the problem's conditions

-- Conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f (x)
axiom even_f_x_plus_1 : ∀ x : ℝ, f (-x + 1) = f (x + 1)
axiom f_definition : ∀ x : ℝ, x ∈ set.Icc 0 1 → f (x) = x * (3 - 2 * x)

-- Proof statement
theorem find_f_31_over_2 : f (31 / 2) = -1 :=
by
  sorry

end find_f_31_over_2_l586_586792


namespace green_tractor_price_l586_586704

-- Define the conditions
def salary_based_on_sales (r_ct : Nat) (r_price : ℝ) (g_ct : Nat) (g_price : ℝ) : ℝ :=
  0.1 * r_ct * r_price + 0.2 * g_ct * g_price

-- Define the problem's Lean statement
theorem green_tractor_price
  (r_ct : Nat) (g_ct : Nat)
  (r_price : ℝ) (total_salary : ℝ)
  (h_rct : r_ct = 2)
  (h_gct : g_ct = 3)
  (h_rprice : r_price = 20000)
  (h_salary : total_salary = 7000) :
  ∃ g_price : ℝ, salary_based_on_sales r_ct r_price g_ct g_price = total_salary ∧ g_price = 5000 :=
by
  sorry

end green_tractor_price_l586_586704


namespace find_f_31_over_2_l586_586791

def f : ℝ → ℝ := sorry   -- Definition to match the problem's conditions

-- Conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f (x)
axiom even_f_x_plus_1 : ∀ x : ℝ, f (-x + 1) = f (x + 1)
axiom f_definition : ∀ x : ℝ, x ∈ set.Icc 0 1 → f (x) = x * (3 - 2 * x)

-- Proof statement
theorem find_f_31_over_2 : f (31 / 2) = -1 :=
by
  sorry

end find_f_31_over_2_l586_586791


namespace probability_merlin_dismissed_l586_586224

variable (p : ℝ) (q : ℝ) (coin_flip : ℝ)

axiom h₁ : q = 1 - p
axiom h₂ : 0 ≤ p ∧ p ≤ 1
axiom h₃ : 0 ≤ q ∧ q ≤ 1
axiom h₄ : coin_flip = 0.5

theorem probability_merlin_dismissed : coin_flip = 0.5 := by
  sorry

end probability_merlin_dismissed_l586_586224


namespace length_AD_find_AD_length_l586_586636

-- Definition of the trapezoid and given lengths
structure Trapezoid (A B C D M H : Type) :=
(parallel_AD_BC : A D -> B C -> Prop)
(point_M_on_CD : C D -> M -> Prop)
(perpendicular_AH_BM : A H -> B M -> Prop)
(equal_AD_HD : A D -> H D -> Prop)
(BC_len : B C -> ℝ)
(CM_len : C M -> ℝ)
(MD_len : M D -> ℝ)

-- The given trapezoid with the specific lengths and conditions
def myTrapezoid : Trapezoid :=
{ parallel_AD_BC := sorry,
  point_M_on_CD := sorry,
  perpendicular_AH_BM := sorry,
  equal_AD_HD := sorry,
  BC_len := 16,
  CM_len := 8,
  MD_len := 9 }

-- The theorem to prove: the length of segment AD
theorem length_AD (t : Trapezoid) : ℝ :=
  if t.BC_len = 16 ∧ t.CM_len = 8 ∧ t.MD_len = 9 ∧ (∀ A D H, t.equal_AD_HD A D H) 
  then 18 
  else 0

-- Prove that the length of AD is 18
theorem find_AD_length : length_AD myTrapezoid = 18 := 
sorry

end length_AD_find_AD_length_l586_586636


namespace fillTime_is_57_hours_l586_586645

/-- Steve has a pool that holds 36,000 gallons. --/
def poolCapacity : ℕ := 36000

/-- The first hose has a flow rate of 3 gallons per minute. --/
def firstHoseRate : ℕ := 3

/-- The second hose has a flow rate of 3.5 gallons per minute. --/
noncomputable def secondHoseRate : ℝ := 3.5

/-- The third hose has a flow rate of 4 gallons per minute. --/
def thirdHoseRate : ℕ := 4

/--
  Prove that it takes 57 hours to fill the pool using the described hoses.
--/
theorem fillTime_is_57_hours : 
  ((poolCapacity : ℝ) / ((firstHoseRate + secondHoseRate + thirdHoseRate) * 60)) ≈ 57 :=
by
  sorry

end fillTime_is_57_hours_l586_586645


namespace smallest_period_pi_max_min_values_l586_586052

noncomputable def f (x : ℝ) : ℝ := sin x * cos x - sqrt 3 * (cos x)^2

theorem smallest_period_pi : 
  ∀ x : ℝ, 
  f (x + π) = f x := 
by
  sorry

theorem max_min_values :
  ∀ x : ℝ, 
  0 ≤ x ∧ x ≤ π/2 → 
  -sqrt 3 ≤ f x ∧ f x ≤ 1 - sqrt 3/2 :=
by
  sorry

end smallest_period_pi_max_min_values_l586_586052


namespace cone_base_radius_proof_l586_586663

-- Mathematical definitions based on the conditions stated
def lateral_surface_unfolded_to_semicircle (radius_semi : ℝ) : Prop :=
  radius_semi = 4

def cone_radius_base (radius_base : ℝ) : Prop :=
  radius_base = 4

-- The mathematically equivalent proof problem statement in Lean 4
theorem cone_base_radius_proof (radius_semi : ℝ) :
  lateral_surface_unfolded_to_semicircle radius_semi →
  ∃ radius_base : ℝ, cone_radius_base radius_base :=
begin
  intro h,
  use 4,
  exact h,
end

end cone_base_radius_proof_l586_586663


namespace correct_fraction_evaluation_l586_586738

theorem correct_fraction_evaluation : (1 : ℂ) / (3 * complex.i + 1) = (1 / 10) + (3 * complex.i / 10) :=
by
  sorry

end correct_fraction_evaluation_l586_586738


namespace multiplication_value_l586_586718

theorem multiplication_value : 725143 * 999999 = 725142274857 :=
by
  sorry

end multiplication_value_l586_586718


namespace unit_prices_min_number_of_A_l586_586696

theorem unit_prices (x y : ℝ)
  (h1 : 3 * x + 4 * y = 580)
  (h2 : 6 * x + 5 * y = 860) :
  x = 60 ∧ y = 100 :=
by
  sorry

theorem min_number_of_A (x y a : ℝ)
  (x_h : x = 60)
  (y_h : y = 100)
  (h1 : 3 * x + 4 * y = 580)
  (h2 : 6 * x + 5 * y = 860)
  (trash_can_condition : a + 200 - a = 200)
  (cost_condition : 60 * a + 100 * (200 - a) ≤ 15000) :
  a ≥ 125 :=
by
  sorry

end unit_prices_min_number_of_A_l586_586696


namespace Alyssa_puppies_l586_586784

theorem Alyssa_puppies (initial_puppies : ℕ) (given_puppies : ℕ)
  (h_initial : initial_puppies = 7) (h_given : given_puppies = 5) :
  initial_puppies - given_puppies = 2 :=
by
  sorry

end Alyssa_puppies_l586_586784


namespace number_of_westbound_vehicles_l586_586681

-- Definitions based on conditions
def speed : ℝ := 60 -- constant speed in miles per hour in both directions
def time_interval : ℝ := 5 / 60 -- five minutes interval in hours
def vehicles_passed : ℕ := 20 -- vehicles passed by the eastbound driver in the interval
def section_length : ℝ := 100 -- length of the highway section in miles

-- The Lean statement
theorem number_of_westbound_vehicles :
  let relative_speed := speed + speed in
  let distance_encountered := relative_speed * time_interval in
  let density := vehicles_passed / distance_encountered in
  (density * section_length : ℝ) = 200 :=
by
  sorry

end number_of_westbound_vehicles_l586_586681


namespace probability_right_angled_triangle_l586_586935

-- Definition of valid pairs that form a right-angled triangle with points (0,0) and (1,-1).
def is_valid_pair (m n : ℕ) : Prop :=
  (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 2) ∨ (m = 3 ∧ n = 3) ∨ (m = 4 ∧ n = 4) ∨
  (m = 5 ∧ n = 5) ∨ (m = 6 ∧ n = 6) ∨ (m = 3 ∧ n = 1) ∨ (m = 4 ∧ n = 2) ∨
  (m = 5 ∧ n = 3) ∨ (m = 6 ∧ n = 4)

-- Set of all possible pairs (m, n) where m, n ∈ {1, 2, ..., 6}
def all_pairs : Finset (ℕ × ℕ) := 
  Finset.product (Finset.range 6).map (λ x, x + 1) (Finset.range 6).map (λ x, x + 1)

-- Set of valid pairs (m, n) that form a right-angled triangle with points (0,0) and (1,-1)
def valid_pairs : Finset (ℕ × ℕ) := all_pairs.filter (λ ⟨m, n⟩, is_valid_pair m n)

-- Probability of forming a right-angled triangle
def triangle_probability : ℝ := ((valid_pairs.card : ℝ) / (all_pairs.card : ℝ))

theorem probability_right_angled_triangle : triangle_probability = 5 / 18 := by
  sorry

end probability_right_angled_triangle_l586_586935


namespace quadratic_roots_sum_l586_586454

theorem quadratic_roots_sum :
  ∃ a b c d : ℤ, (x^2 + 23 * x + 132 = (x + a) * (x + b)) ∧ (x^2 - 25 * x + 168 = (x - c) * (x - d)) ∧ (a + c + d = 42) :=
by {
  sorry
}

end quadratic_roots_sum_l586_586454


namespace john_avg_speed_l586_586968

/-- John's average speed problem -/
theorem john_avg_speed (d : ℕ) (total_time : ℕ) (time1 : ℕ) (speed1 : ℕ) 
  (time2 : ℕ) (speed2 : ℕ) (time3 : ℕ) (x : ℕ) :
  d = 144 ∧ total_time = 120 ∧ time1 = 40 ∧ speed1 = 64 
  ∧ time2 = 40 ∧ speed2 = 70 ∧ time3 = 40 
  → (d = time1 * speed1 + time2 * speed2 + time3 * x / 60)
  → x = 82 := 
by
  intros h1 h2
  sorry

end john_avg_speed_l586_586968


namespace sum_of_odd_non_prime_between_60_and_100_l586_586469

-- Conditions:
def is_odd (n : ℤ) : Prop := n % 2 = 1
def is_prime (n : ℤ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n
def between_60_and_100 (n : ℤ) : Prop := 60 < n ∧ n < 100

-- Definition of the numbers we are interested in
def odd_non_prime_numbers : List ℤ :=
  List.filter (λ n => is_odd n ∧ ¬ is_prime n ∧ between_60_and_100 n) (List.range' 61 98)

-- Sum of elements in the list
def sum_odd_non_prime_numbers : ℤ :=
  List.sum odd_non_prime_numbers

-- The theorem we want to prove
theorem sum_of_odd_non_prime_between_60_and_100 :
  sum_odd_non_prime_numbers = 880 :=
by 
  -- The proof would go here.
  sorry

end sum_of_odd_non_prime_between_60_and_100_l586_586469


namespace molecular_weight_C4H10_l586_586714

theorem molecular_weight_C4H10 (molecular_weight_six_moles : ℝ) (h : molecular_weight_six_moles = 390) :
  molecular_weight_six_moles / 6 = 65 :=
by
  -- proof to be filled in here
  sorry

end molecular_weight_C4H10_l586_586714


namespace inverse_implications_l586_586576

-- Definitions used in the conditions
def not_coplanar (points : set (set point)) : Prop :=
  ∃ p1 p2 p3 p4 ∈ points, ¬ (affine_span ℝ {p1, p2, p3, p4} ≤ affine_span ℝ {p1, p2, p3})

def not_collinear (points : set point) : Prop :=
  ∀ p1 p2 p3 ∈ points, ¬ (affine_span ℝ {p1, p2, p3} ≤ line ℝ p1 p2)

def skew_lines (l1 l2 : line ℝ) : Prop :=
  ¬ ∃ p, p ∈ l1 ∧ p ∈ l2

-- Problem statement
theorem inverse_implications :
  (¬ ∀ points : set point, not_collinear points → not_coplanar (insert points))
  ∧ 
  (∀ l1 l2 : line ℝ, skew_lines l1 l2 → ¬ ∃ p, p ∈ l1 ∧ p ∈ l2) :=
sorry

end inverse_implications_l586_586576


namespace sufficient_condition_for_q_l586_586480

open Classical
local attribute [instance] propDecidable

variable {x m : ℝ}

theorem sufficient_condition_for_q (p : x^2 - 8x - 20 > 0) (q : (x - (1 - m)) * (x - (1 + m)) > 0) (h : p → q) (h' : ¬ (q → p)) : 0 < m ∧ m ≤ 3 :=
by
  have h_pos : m > 0 := sorry
  have h1 : 1 - m ≥ -2 := sorry
  have h2 : 1 + m ≤ 10 := sorry
  constructor
  exact h_pos
  linarith [h1, h2]
  sorry -- Proof details


end sufficient_condition_for_q_l586_586480


namespace chenny_friends_l586_586426

noncomputable def num_friends (initial_candies add_candies candies_per_friend) : ℕ :=
  (initial_candies + add_candies) / candies_per_friend

theorem chenny_friends :
  num_friends 10 4 2 = 7 :=
by
  sorry

end chenny_friends_l586_586426


namespace range_of_k_l586_586028

theorem range_of_k (k m n : ℝ) (A : (-3, m)) (B : (-2, n)) 
  (h1: m = (k-1) / -3) (h2: n = (k-1) / -2) (h3: m > n) : k > 1 := 
sorry

end range_of_k_l586_586028


namespace sequence_proof_l586_586020

noncomputable def seqA (a : ℕ → ℝ) (n : ℕ) : ℝ := 
if n = 0 then 1 else a (n - 1) * 2

def seqB (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) : ℝ := 
3 * n - 2

def T (a b : ℕ → ℝ) (n : ℕ) : ℝ :=
(finsum (λ k, a k * b k) (range n)).val

theorem sequence_proof : 
∀ (n : ℕ) (h : 0 < n) 
(a : ℕ → ℝ) 
(b : ℕ → ℝ),
(a 0 = 1) →
(∀ (k : ℕ), a (k + 1) = seqA a k) →
(b 0 = seqB a b 0) →
(∀ (k : ℕ), b (k + 1) = seqB a b (k + 1)) →
T a b n = (3 * n - 5) * 2^n + 5 :=
by 
  sorry

end sequence_proof_l586_586020


namespace problem_l586_586716

-- Define number1 as 999...999 with 20089 digits
def num1 : ℕ := nat.of_digits 10 (list.repeat 9 20089)
-- Define number2 as 333...333 with 20083 digits
def num2 : ℕ := nat.of_digits 10 (list.repeat 3 20083)

theorem problem (n : ℕ) (m : ℕ) (k : ℕ) : 
  (nat.pow num1 2007 - nat.pow num2 2007) % 11 = 0 := 
by
  sorry

end problem_l586_586716


namespace probability_less_than_one_third_l586_586098

theorem probability_less_than_one_third : 
  (∃ p : ℚ, p = (measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ)) (set.Ioo (0 : ℚ) (1 / 3 : ℚ)) / 
                  measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ))) ∧ p = 2 / 3) :=
by {
  -- The proof will be added here.
  sorry
}

end probability_less_than_one_third_l586_586098


namespace sheela_monthly_income_eq_l586_586269

-- Defining the conditions
def sheela_deposit : ℝ := 4500
def percentage_of_income : ℝ := 0.28

-- Define Sheela's monthly income as I
variable (I : ℝ)

-- The theorem to prove
theorem sheela_monthly_income_eq : (percentage_of_income * I = sheela_deposit) → (I = 16071.43) :=
by
  sorry

end sheela_monthly_income_eq_l586_586269


namespace value_ne_one_l586_586928

theorem value_ne_one (a b: ℝ) (h : a * b ≠ 0) : (|a| / a) + (|b| / b) ≠ 1 := 
by 
  sorry

end value_ne_one_l586_586928


namespace problem_part_1_problem_part_2_l586_586059

noncomputable def set_A : Set ℝ := { x | x^2 - 2*x - 3 < 0 }

noncomputable def set_B (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < a + 1 }

noncomputable def set_C : Set ℝ := { a | 0 ≤ a ∧ a ≤ 2 }

noncomputable def f (x : ℝ) : ℝ := 4 * sin x

noncomputable def set_x0 : Set ℝ :=
  { x0 | ∃ k : ℤ, (2*k*π ≤ x0 ∧ x0 ≤ 2*k*π + π/6) ∨ (2*k*π + 5*π/6 ≤ x0 ∧ x0 ≤ 2*k*π + π) }

theorem problem_part_1 (a : ℝ) (h : set_B a ⊆ set_A) : a ∈ set_C :=
  sorry

theorem problem_part_2 (x0 : ℝ) (hx0 : f x0 ∈ set_C) : x0 ∈ set_x0 :=
  sorry

end problem_part_1_problem_part_2_l586_586059


namespace complex_division_l586_586872

theorem complex_division (i : ℂ) (hi : i^2 = -1) : (2 * i) / (1 - i) = -1 + i :=
by sorry

end complex_division_l586_586872


namespace gen_term_arithmetic_seq_min_lambda_leq_l586_586488

-- Definitions for conditions
def isArithmeticSeq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sumOfFirstFourTerms (a : ℕ → ℝ) (S : ℝ) : Prop :=
  a 1 + a 2 + a 3 + a 4 = S

def formsGeometricSeq (a : ℕ → ℝ) : Prop :=
  a 3 ^ 2 = a 1 * a 7

-- Part (I) General term formula for the sequence {a_n}
theorem gen_term_arithmetic_seq :
  ∃ a : ℕ → ℝ, 
    isArithmeticSeq a 1 ∧ sumOfFirstFourTerms a 14 ∧ formsGeometricSeq a ∧ 
    ∀ n, a n = n + 1 :=
sorry

-- Definitions for part (II) conditions
def sumOfSeq (a : ℕ → ℝ) (T : ℕ → ℝ) : Prop :=
  ∀ n, T n = ∑ k in finset.range n, 1 / (a (k + 1) * a (k + 2))

def geqLambda (T : ℕ → ℝ) (a : ℕ → ℝ) (lambda : ℝ) : Prop :=
  ∀ n > 0, T n ≤ lambda * a (n + 1)

-- Part (II) Minimum value of lambda
theorem min_lambda_leq :
  ∃ λ : ℝ, λ = 1 / 16 ∧
    (∃ a : ℕ → ℝ, isArithmeticSeq a 1 ∧ sumOfFirstFourTerms a 14 ∧ formsGeometricSeq a ∧
      ∃ T : ℕ → ℝ, sumOfSeq a T ∧ geqLambda T a λ) :=
sorry

end gen_term_arithmetic_seq_min_lambda_leq_l586_586488


namespace geometric_sequence_common_ratio_eq_one_third_l586_586038

variable {a_n : ℕ → ℝ}
variable {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio_eq_one_third
  (h_geom : geometric_sequence a_n q)
  (h_increasing : ∀ n, a_n n < a_n (n + 1))
  (h_a1 : a_n 1 = -2)
  (h_recurrence : ∀ n, 3 * (a_n n + a_n (n + 2)) = 10 * a_n (n + 1)) :
  q = 1 / 3 :=
by
  sorry

end geometric_sequence_common_ratio_eq_one_third_l586_586038


namespace exterior_angle_of_regular_polygon_l586_586166

theorem exterior_angle_of_regular_polygon (n : ℕ) (h : 1 ≤ n) (angle : ℝ) 
  (h_angle : angle = 72) (sum_ext_angles : ∀ (polygon : Type) [fintype polygon], (∑ i, angle) = 360) : 
  n = 5 := sorry

end exterior_angle_of_regular_polygon_l586_586166


namespace probability_intervals_l586_586089

-- Definitions: interval (0, 1/2), and the condition interval (0, 1/3)
def interval1 := set.Ioo (0:ℝ) (1/2)
def interval2 := set.Ioo (0:ℝ) (1/3)

-- The proof statement: probability of randomly selecting a number from (0, 1/2), being within (0, 1/3), is 2/3.
theorem probability_intervals : (interval2.measure / interval1.measure = 2 / 3) :=
sorry

end probability_intervals_l586_586089


namespace calculate_fourth_quarter_shots_l586_586833

-- Definitions based on conditions
def first_quarters_shots : ℕ := 20
def first_quarters_successful_shots : ℕ := 12
def third_quarter_shots : ℕ := 10
def overall_accuracy : ℚ := 46 / 100
def total_shots (n : ℕ) : ℕ := first_quarters_shots + third_quarter_shots + n
def total_successful_shots (n : ℕ) : ℚ := first_quarters_successful_shots + 3 + (4 / 10 * n)


-- Main theorem to prove
theorem calculate_fourth_quarter_shots (n : ℕ) (h : (total_successful_shots n) / (total_shots n) = overall_accuracy) : 
  n = 20 :=
by {
  sorry
}

end calculate_fourth_quarter_shots_l586_586833


namespace no_integer_solutions_x2_plus_3xy_minus_2y2_eq_122_l586_586265

theorem no_integer_solutions_x2_plus_3xy_minus_2y2_eq_122 :
  ¬ ∃ x y : ℤ, x^2 + 3 * x * y - 2 * y^2 = 122 := sorry

end no_integer_solutions_x2_plus_3xy_minus_2y2_eq_122_l586_586265


namespace hypotenuse_length_of_rotated_right_triangle_l586_586342

theorem hypotenuse_length_of_rotated_right_triangle (a b : ℝ)
  (h1 : (1 / 3) * π * a * b^2 = 640 * π)
  (h2 : (1 / 3) * π * b * a^2 = 1536 * π) : 
  real.sqrt (a^2 + b^2) = 32.5 :=
  sorry

end hypotenuse_length_of_rotated_right_triangle_l586_586342


namespace volume_formula_l586_586190

-- Define the problem
variable (a b c : ℝ) (α : ℝ)

namespace Tetrahedron

-- Define semi-perimeter
def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

-- Define Heron's formula for the area of a triangle
def area (a b c : ℝ) (p : ℝ) : ℝ := 
  Real.sqrt (p * (p - a) * (p - b) * (p - c))

-- Define volume of the tetrahedron
def volume (a b c : ℝ) (α : ℝ) : ℝ :=
  let p := semi_perimeter a b c in
  ( (p - a) * (p - b) * (p - c) / 3 ) * Real.tan α

-- Theorem to prove that the volume has the required form
theorem volume_formula (a b c α : ℝ) :
  let p := semi_perimeter a b c in
  volume a b c α = ( (p - a) * (p - b) * (p - c) / 3 ) * Real.tan α :=
by
  sorry

end Tetrahedron

end volume_formula_l586_586190


namespace litter_patrol_total_pieces_l586_586650

theorem litter_patrol_total_pieces :
  let glass_bottles := 25
  let aluminum_cans := 18
  let plastic_bags := 12
  let paper_cups := 7
  let cigarette_packs := 5
  let discarded_face_masks := 3
  glass_bottles + aluminum_cans + plastic_bags + paper_cups + cigarette_packs + discarded_face_masks = 70 :=
by
  sorry

end litter_patrol_total_pieces_l586_586650


namespace area_of_circle_l586_586637

-- Point definitions for P and Q
def P : (ℝ × ℝ) := (3, 7)
def Q : (ℝ × ℝ) := (7, 5)

-- Tangent lines to the circle at P and Q intersect at a point on the x-axis
def intersectionOnXAxis : Prop := ∃ R : ℝ × ℝ, R.2 = 0 ∧ tangentAt P R ∧ tangentAt Q R

-- Circle with points P and Q, and tangent line intersections on x-axis
def circleOmega : Prop := ∃ c : ℝ × ℝ, (P.1 - c.1)^2 + (P.2 - c.2)^2 = r^2 ∧ (Q.1 - c.1)^2 + (Q.2 - c.2)^2 = r^2

-- The main theorem stating the area of the circle
theorem area_of_circle : circleOmega ∧ intersectionOnXAxis → 
                          ∃ r : ℝ, π * r^2 = 50 * π :=
by
  sorry

end area_of_circle_l586_586637


namespace find_probability_greater_than_three_l586_586063

noncomputable def X : ℝ := Binomial 4 (1 / 4)

noncomputable def Y (μ σ : ℝ) : Measure ℝ := Normal μ σ^2

axiom E_X (X : ℝ) : E[X] = 1
axiom E_Y (Y : ℝ) (μ : ℝ) : E[Y] = μ

axiom P_abs_Y_lt_1 (Y : ℝ) (σ : ℝ) : P (|Y| < 1) = 0.4

theorem find_probability_greater_than_three (μ σ : ℝ) (hσ : σ > 0) :
  (∃ μ, E[X] = E[Y μ σ] ∧ P_abs_Y_lt_1 (Y μ σ) σ) → P (Y μ σ > 3) = 0.1 :=
by
  sorry

end find_probability_greater_than_three_l586_586063


namespace line_does_not_pass_through_second_quadrant_l586_586558

theorem line_does_not_pass_through_second_quadrant (t : ℝ) :
  (∀ (x y : ℝ), ¬((2t - 3) * x + 2 * y + t = 0 ∧ x < 0 ∧ y > 0)) ↔ 0 ≤ t ∧ t ≤ (3 / 2) :=
by
  sorry

end line_does_not_pass_through_second_quadrant_l586_586558


namespace v_combination_value_l586_586409

def v (x : ℝ) : ℝ := 2 * x + 4 * Real.sin (x * Real.pi / 4)

theorem v_combination_value : 
  v (-3.42) + v (-1.27) + v (1.27) + v (3.42) = 0 := by
  sorry

end v_combination_value_l586_586409


namespace min_students_l586_586256

theorem min_students (b g : ℕ) 
  (h1 : 3 * b = 2 * g) 
  (h2 : (b + g) % 5 = 2) : 
  b + g = 57 :=
sorry

end min_students_l586_586256


namespace minimum_value_of_functions_l586_586526

def linear_fn (a b c: ℝ) := a ≠ 0 
def f (a b: ℝ) (x: ℝ) := a * x + b 
def g (a c: ℝ) (x: ℝ) := a * x + c

theorem minimum_value_of_functions (a b c: ℝ) (hx: linear_fn a b c) :
  (∀ x: ℝ, 3 * (f a b x)^2 + 2 * g a c x ≥ -19 / 6) → (∀ x: ℝ, 3 * (g a c x)^2 + 2 * f a b x ≥ 5 / 2) :=
by
  sorry

end minimum_value_of_functions_l586_586526


namespace total_length_of_belt_belt_sufficient_l586_586737

-- Definitions only, assuming all necessary geometric and trigonometric operations are imported

-- Given conditions
def radius : ℝ := 2 -- radius of each pulley in cm
def O1O2 : ℝ := 12 -- distance between O1 and O2 in cm
def O1O3 : ℝ := 10 -- distance between O1 and O3 in cm
def distance_plane_O3 : ℝ := 8 -- perpendicular distance from O3 to the plane containing O1 and O2 in cm

-- Prove the total length of the belt
theorem total_length_of_belt : 
  let O2O3 := Real.sqrt (O1O2^2 + O1O3^2 - 2 * O1O2 * O1O3 * (distance_plane_O3 / O1O3)) in
  let straight_segment_length := O1O2 + O1O3 + O2O3 in
  let arc_length := 2 * Real.pi * radius in
  straight_segment_length + arc_length = 41.77 :=
  
begin
  sorry
end

-- Confirmation if the length is sufficient against 54 cm
theorem belt_sufficient : 
  let O2O3 := Real.sqrt (O1O2^2 + O1O3^2 - 2 * O1O2 * O1O3 * (distance_plane_O3 / O1O3)) in
  let straight_segment_length := O1O2 + O1O3 + O2O3 in
  let arc_length := 2 * Real.pi * radius in
  let total_length := straight_segment_length + arc_length in
  total_length < 54 :=
  
begin
  sorry
end

end total_length_of_belt_belt_sufficient_l586_586737


namespace maxwell_walking_speed_l586_586255

theorem maxwell_walking_speed (v : ℝ) (h1 : ∀ t : ℝ, t = 3 → 3 * v = 24 - 2 * 6) : v = 4 :=
by
  have h2: 2 * 6 = 12 := by norm_num
  have h3: 24 - 12 = 12 := by norm_num
  have h4: 3 * v = 12 := by rw [h2, h3]; exact h1 3 rfl
  linarith

-- Dummy variable
variable (dummy : ℝ)

end maxwell_walking_speed_l586_586255


namespace inequality_solution_maximum_expression_l586_586739

-- Problem 1: Inequality for x
theorem inequality_solution (x : ℝ) : |x + 1| + 2 * |x - 1| < 3 * x + 5 ↔ x > -1/2 :=
by
  sorry

-- Problem 2: Maximum value for expression within [0, 1]
theorem maximum_expression (a b : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1) (hb : 0 ≤ b) (hb1 : b ≤ 1) : 
  ab + (1 - a - b) * (a + b) ≤ 1/3 :=
by
  sorry

end inequality_solution_maximum_expression_l586_586739


namespace tree_height_at_two_years_l586_586779

variable (h : ℕ → ℕ)

-- Given conditions
def condition1 := h 4 = 81
def condition2 := ∀ t : ℕ, h (t + 1) = 3 * h t

theorem tree_height_at_two_years
  (h_tripled : ∀ t : ℕ, h (t + 1) = 3 * h t)
  (h_at_four : h 4 = 81) :
  h 2 = 9 :=
by
  -- Formal proof will be provided here
  sorry

end tree_height_at_two_years_l586_586779


namespace probability_calc_l586_586253

noncomputable def probability_x_plus_y_less_4 : ℝ :=
  let square_area := 9
  let excluded_triangle_area := 2
  let shaded_area := square_area - excluded_triangle_area
  shaded_area / square_area

theorem probability_calc
  (x y : ℝ)
  (h1 : 0 ≤ x) (h2 : x ≤ 3)
  (h3 : 0 ≤ y) (h4 : y ≤ 3)
  (uniform_dist : uniform_of square_area)
  : probability_x_plus_y_less_4 = 7 / 9 :=
by
  exact sorry

end probability_calc_l586_586253


namespace probability_less_than_one_third_l586_586133

def interval := set.Ioo (0 : ℝ) (1 / 2)

def desired_interval := set.Ioo (0 : ℝ) (1 / 3)

theorem probability_less_than_one_third :
  (∃ p : ℝ, p = (volume desired_interval / volume interval) ∧ p = 2 / 3) :=
  sorry

end probability_less_than_one_third_l586_586133


namespace probability_less_than_one_third_l586_586161

-- Define the intervals and the associated lengths
def interval_total : ℝ := 1 / 2
def interval_condition : ℝ := 1 / 3

-- Define the probability calculation
def probability : ℝ :=
  interval_condition / interval_total

-- Statement of the problem
theorem probability_less_than_one_third : probability = 2 / 3 := by
  sorry

end probability_less_than_one_third_l586_586161


namespace real_part_fraction_l586_586498

theorem real_part_fraction {i : ℂ} (h : i^2 = -1) : (
  let numerator := 1 - i
  let denominator := (1 + i) ^ 2
  let fraction := numerator / denominator
  let real_part := (fraction.re)
  real_part
) = -1/2 := sorry

end real_part_fraction_l586_586498


namespace diagonals_in_decagon_l586_586821

theorem diagonals_in_decagon :
  let n := 10
  let d := n * (n - 3) / 2
  d = 35 :=
by
  sorry

end diagonals_in_decagon_l586_586821


namespace probability_merlin_dismissed_l586_586225

variable (p : ℝ) (q : ℝ) (coin_flip : ℝ)

axiom h₁ : q = 1 - p
axiom h₂ : 0 ≤ p ∧ p ≤ 1
axiom h₃ : 0 ≤ q ∧ q ≤ 1
axiom h₄ : coin_flip = 0.5

theorem probability_merlin_dismissed : coin_flip = 0.5 := by
  sorry

end probability_merlin_dismissed_l586_586225


namespace ratio_diagonals_to_sides_l586_586299

-- Definition of the number of diagonals in a polygon with n sides
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- Definition of the condition
def n : ℕ := 5

-- Proof statement that the ratio of the number of diagonals to the number of sides is 1
theorem ratio_diagonals_to_sides (n_eq_5 : n = 5) : 
  (number_of_diagonals n) / n = 1 :=
by {
  -- Proof would go here, but is omitted
  sorry
}

end ratio_diagonals_to_sides_l586_586299


namespace sin_theta_l586_586240

-- Defining the conditions
def line_equation (x y z : ℝ) : Prop := ∃ k : ℝ, x = 2 * k - 1 ∧ y = 3 * k ∧ z = 6 * k + 3
def plane_equation (x y z : ℝ) : Prop := -10 * x - 2 * y + 11 * z = 3

-- Direction vector of the line
def direction_vector : ℝ × ℝ × ℝ := (2, 3, 6)

-- Normal vector of the plane
def normal_vector : ℝ × ℝ × ℝ := (-10, -2, 11)

-- Magnitudes of the vectors
def vector_magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  match v with
  | (vx, vy, vz) => Real.sqrt (vx ^ 2 + vy ^ 2 + vz ^ 2)

-- Dot product of the vectors
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  match v1, v2 with
  | (vx1, vy1, vz1), (vx2, vy2, vz2) => vx1 * vx2 + vy1 * vy2 + vz1 * vz2

-- Define the theorem statement
theorem sin_theta :
  let θ := 90 - (Real.arccos (dot_product direction_vector normal_vector / (vector_magnitude direction_vector * vector_magnitude normal_vector))) in
  sin θ = 8 / 21 :=
by
  sorry

end sin_theta_l586_586240


namespace equation_of_line_l_l586_586666

noncomputable def point := ℝ × ℝ

def l1 : point → Prop := λ p, p.1 + p.2 - 2 = 0
def l2 : point → Prop := λ p, p.1 - p.2 - 4 = 0

def A : point := (-1, 3)
def B : point := (5, 1)

def midpoint (a b : point) : point := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

theorem equation_of_line_l :
  let P := (λ p, l1 p ∧ l2 p)
  let Q := midpoint A B
  ∃ (a b c : ℝ), (λ p, a * p.1 + b * p.2 + c = 0) ∧ a = 3 ∧ b = 1 ∧ c = -8 :=
sorry

end equation_of_line_l_l586_586666


namespace solve_line_eq_l586_586667

theorem solve_line_eq (a b x : ℝ) (h1 : (0 : ℝ) * a + b = 2) (h2 : -3 * a + b = 0) : x = -3 :=
by
  sorry

end solve_line_eq_l586_586667


namespace tangent_line_circle_l586_586026

theorem tangent_line_circle (k : ℝ) (h1 : k = Real.sqrt 3) (h2 : ∃ (x y : ℝ), y = k * x + 2 ∧ x^2 + y^2 = 1) :
  (k = Real.sqrt 3 → (∃ (x y : ℝ), y = k * x + 2 ∧ x^2 + y^2 = 1)) ∧ (¬ (∀ (k : ℝ), (∃ (x y : ℝ), y = k * x + 2 ∧ x^2 + y^2 = 1) → k = Real.sqrt 3)) :=
  sorry

end tangent_line_circle_l586_586026


namespace complete_task_in_3_days_l586_586773

theorem complete_task_in_3_days (x y z w v : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) (hv : v > 0)
  (h1 : 1 / x + 1 / y + 1 / z = 1 / 7.5)
  (h2 : 1 / x + 1 / z + 1 / v = 1 / 5)
  (h3 : 1 / x + 1 / z + 1 / w = 1 / 6)
  (h4 : 1 / y + 1 / w + 1 / v = 1 / 4) :
  1 / (1 / x + 1 / z + 1 / v + 1 / w + 1 / y) = 3 :=
sorry

end complete_task_in_3_days_l586_586773


namespace length_more_than_breadth_l586_586665

theorem length_more_than_breadth :
  ∀ (b l : ℕ) (x : ℕ) (cost_per_meter total_cost : ℝ),
  l = 63 ∧ total_cost = 5300 ∧ cost_per_meter = 26.50 ∧ l = b + x ∧ Perimeter = 2 * (l + b)
  → x = 26 := by
  -- Perimeter is calculated using the cost and the rate
  have perimeter_eq : Perimeter = total_cost / cost_per_meter := sorry
  
  -- From the perimeter equation of the rectangle
  rw [perimeter_eq]
  derive 200 = 2 * (63 + b) := sorry
  
  -- Solving for breadth
  have b_eq : b = 37 := sorry
  
  -- Proving that x = 26
  calc
    x = l - b := sorry
    ... = 63 - 37 := sorry
    ... = 26 := by norm_num

end length_more_than_breadth_l586_586665


namespace modulo_sum_remainder_l586_586208

theorem modulo_sum_remainder (a b: ℤ) (k j: ℤ) 
  (h1 : a = 84 * k + 77) 
  (h2 : b = 120 * j + 113) :
  (a + b) % 42 = 22 := by
  sorry

end modulo_sum_remainder_l586_586208


namespace cyclic_quadrilateral_AN_NC_eq_CD_BN_l586_586599

-- Definitions provided by conditions
variables (A B C D M N I : Type)
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited M] [Inhabited N] [Inhabited I]

-- Assume a cyclic quadrilateral and necessary points
variables {cyclic_quad : Prop} (intersection_diag_M : Prop)
variables (circle_passing_BM_and_incenter_BCM : Prop) (second_intersection_N : Prop)
variables (AN NC CD BN : ℝ)

-- Proof problem
theorem cyclic_quadrilateral_AN_NC_eq_CD_BN :
  cyclic_quad →
  intersection_diag_M →
  circle_passing_BM_and_incenter_BCM →
  second_intersection_N →
  |AD| = |BD| →
  AN * NC = CD * BN :=
by sorry

end cyclic_quadrilateral_AN_NC_eq_CD_BN_l586_586599


namespace probability_less_than_one_third_l586_586127

def interval := set.Ioo (0 : ℝ) (1 / 2)

def desired_interval := set.Ioo (0 : ℝ) (1 / 3)

theorem probability_less_than_one_third :
  (∃ p : ℝ, p = (volume desired_interval / volume interval) ∧ p = 2 / 3) :=
  sorry

end probability_less_than_one_third_l586_586127


namespace probability_merlin_dismissed_l586_586227

variable (p : ℝ) (q : ℝ) (coin_flip : ℝ)

axiom h₁ : q = 1 - p
axiom h₂ : 0 ≤ p ∧ p ≤ 1
axiom h₃ : 0 ≤ q ∧ q ≤ 1
axiom h₄ : coin_flip = 0.5

theorem probability_merlin_dismissed : coin_flip = 0.5 := by
  sorry

end probability_merlin_dismissed_l586_586227


namespace triangle_is_isosceles_l586_586380

theorem triangle_is_isosceles
  (O A B C : Point)
  (circle : Circle O)
  (chord_AB : Chord circle A B)
  (tangent_C : Tangent circle C)
  (parallel_tangent_chord : Parallel tangent_C chord_AB) : IsoscelesTriangle A B C := 
sorry

end triangle_is_isosceles_l586_586380


namespace unique_solution_l586_586839

theorem unique_solution (n : ℕ) (h : n > 1) :
  (∀ p : ℕ, nat.prime p → p ∣ n^6 - 1 → p ∣ (n^3 - 1) * (n^2 - 1)) ↔ n = 2 := 
sorry

end unique_solution_l586_586839


namespace angle_ODQ_ninety_degrees_l586_586951

-- Definitions for the geometric elements and conditions
variables {ABC : Type} [triangle ABC] [acute_triangle ABC] [¬isosceles ABC]
variables {O : Point} [circumcenter O ABC]
variables {H : Point} [orthocenter H ABC]
variables {A B C D Q P : Point}
variables {HQ : Line} {BC : Line}
variables [extends HQ Q P] (AB_gt_AC : B > C)
variables (BD_eq_DP : dist B D = dist D P) -- D is the foot of perpendicular from A to BC

-- The theorem to be proved
theorem angle_ODQ_ninety_degrees :
  ∠ O D Q = 90 := 
sorry

end angle_ODQ_ninety_degrees_l586_586951


namespace rob_grapes_solution_l586_586783

-- Define the problem condition
def rob_grapes (R A B : ℕ) : Prop :=
  A = R + 2 ∧ B = R + 6 ∧ R + A + B = 83

-- Define the theorem with the correct answer
theorem rob_grapes_solution : ∃ R A B, rob_grapes R A B ∧ R = 25 :=
by
  use 25
  use 27
  use 31
  unfold rob_grapes
  simp
  sorry -- Skipping the proof

end rob_grapes_solution_l586_586783


namespace function_neither_even_nor_odd_l586_586590

noncomputable def f (x : ℝ) : ℝ := Real.log (x + Real.sqrt (1 + x^3))

theorem function_neither_even_nor_odd :
  ¬(∀ x, f x = f (-x)) ∧ ¬(∀ x, f x = -f (-x)) := by
  sorry

end function_neither_even_nor_odd_l586_586590


namespace problem_statement_l586_586191

-- Define the equiangular hexagon and the inscribed square
structure EquiangularHexagon where
  AB BC CD DE EF FA : ℝ
  equiangular : True

structure InscribedSquareInHexagon where
  MNPQ : (EquiangularHexagon → (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ))

noncomputable def side_length_of_square : ℝ :=
  let hexagon := EquiangularHexagon.mk 50 0 0 0 (50 * (Real.sqrt 3 - 1)) 0 trivial
  let square := InscribedSquareInHexagon.mk ((hex) => ((0, 0), (0, 0), (0, 0)))
  25

theorem problem_statement : 
  ∀ (hex : EquiangularHexagon) (square : InscribedSquareInHexagon),
  hex.AB = 50 → hex.EF = 50 * (Real.sqrt 3 - 1) → side_length_of_square = 25 :=
by sorry

end problem_statement_l586_586191


namespace number_of_equilateral_triangles_in_expanded_lattice_l586_586379

-- Define the structure of the hexagonal lattice and related properties
structure Point where
  x : ℝ
  y : ℝ

-- Define property for the given hexagonal lattice
def in_lattice (p : Point) : Prop :=
  -- This will represent the specific condition that forms a hexagonal lattice with 13 points
  sorry

-- Define the distance condition (one unit distance from nearest neighbors)
def unit_distance (p1 p2 : Point) : Prop :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2) = 1

-- Define the property for equilateral triangle
def equilateral_triangle (p1 p2 p3 : Point) : Prop :=
  unit_distance p1 p2 ∧ unit_distance p2 p3 ∧ unit_distance p3 p1

-- The final theorem statement to be proven
theorem number_of_equilateral_triangles_in_expanded_lattice : 
  ∃ (points : list Point), 
  (∀ p ∈ points, in_lattice p) ∧ 
  ∀ (t : finset (fin 3) → Point), 
  (t 0 ∈ points → t 1 ∈ points → t 2 ∈ points → equilateral_triangle (t 0) (t 1) (t 2)) →
  finset.univ.card (finset.filter (λ t, 
    ∃ (a b c : Point), 
      a ∈ points ∧ 
      b ∈ points ∧ 
      c ∈ points ∧ 
      t = {a, b, c} ∧
      equilateral_triangle a b c
  ) finset.univ) = 18 :=
sorry

end number_of_equilateral_triangles_in_expanded_lattice_l586_586379


namespace rachel_bought_3_tables_l586_586267

-- Definitions from conditions
def chairs := 7
def minutes_per_furniture := 4
def total_minutes := 40

-- Define the number of tables Rachel bought
def number_of_tables (chairs : ℕ) (minutes_per_furniture : ℕ) (total_minutes : ℕ) : ℕ :=
  (total_minutes - (chairs * minutes_per_furniture)) / minutes_per_furniture

-- Lean theorem stating the proof problem
theorem rachel_bought_3_tables : number_of_tables chairs minutes_per_furniture total_minutes = 3 :=
by
  sorry

end rachel_bought_3_tables_l586_586267


namespace decagon_diagonals_l586_586827

-- Number of diagonals calculation definition
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Proving the number of diagonals in a decagon
theorem decagon_diagonals : num_diagonals 10 = 35 := by
  sorry

end decagon_diagonals_l586_586827


namespace angle_PAQ_lt_60_l586_586961

-- Define the regular tetrahedron with vertices A, B, C, D
variable (A B C D P Q : Point)
variable (tetrahedron_regular : RegularTetrahedron A B C D)
variable (P_inside : PointInsideTetrahedron P A B C D)
variable (Q_inside : PointInsideTetrahedron Q A B C D)

theorem angle_PAQ_lt_60 : angle P A Q < 60 :=
sorry

end angle_PAQ_lt_60_l586_586961


namespace find_two_digit_number_l586_586571

-- Define the problem conditions and statement
theorem find_two_digit_number (a b n : ℕ) (h1 : a = 2 * b) (h2 : 10 * a + b + a^2 = n^2) : 
  10 * a + b = 21 :=
sorry

end find_two_digit_number_l586_586571


namespace min_xy_value_l586_586035

theorem min_xy_value (x y : ℝ) (hx : 1 < x) (hy : 1 < y) (hlog : Real.log x / Real.log 2 * Real.log y / Real.log 2 = 1) : x * y = 4 :=
by sorry

end min_xy_value_l586_586035


namespace find_common_difference_l586_586947

theorem find_common_difference (a a_n S_n : ℝ) (h1 : a = 3) (h2 : a_n = 50) (h3 : S_n = 318) : 
  ∃ d n, (a + (n - 1) * d = a_n) ∧ (n / 2 * (a + a_n) = S_n) ∧ (d = 47 / 11) := 
by
  sorry

end find_common_difference_l586_586947


namespace cot_thirty_deg_l586_586415

theorem cot_thirty_deg : ∃ (θ : ℝ), θ = 30 ∧ cot θ = sqrt 3 :=
by
  sorry

end cot_thirty_deg_l586_586415


namespace quadratic_union_nonempty_l586_586897

theorem quadratic_union_nonempty (a : ℝ) :
  (∃ x : ℝ, x^2 - (a-2)*x - 2*a + 4 = 0) ∨ (∃ y : ℝ, y^2 + (2*a-3)*y + 2*a^2 - a - 3 = 0) ↔
    a ≤ -6 ∨ (-7/2) ≤ a ∧ a ≤ (3/2) ∨ a ≥ 2 :=
sorry

end quadratic_union_nonempty_l586_586897


namespace cot_theta_in_terms_of_x_l586_586239

theorem cot_theta_in_terms_of_x (θ : ℝ) (x : ℝ) 
  (h_obtuse : π / 2 < θ ∧ θ < π)
  (h_cos : cos (θ / 2) = sqrt ((x - 1) / (2 * x))) :
  cot θ = -1 / sqrt (x^2 - 1) := 
sorry

end cot_theta_in_terms_of_x_l586_586239


namespace exterior_angle_of_regular_polygon_l586_586164

theorem exterior_angle_of_regular_polygon (n : ℕ) (h : 1 ≤ n) (angle : ℝ) 
  (h_angle : angle = 72) (sum_ext_angles : ∀ (polygon : Type) [fintype polygon], (∑ i, angle) = 360) : 
  n = 5 := sorry

end exterior_angle_of_regular_polygon_l586_586164


namespace at_least_one_is_one_l586_586736

theorem at_least_one_is_one
  (a : Fin (2 ^ 2016) → ℕ)
  (h1 : ∀ n, 1 ≤ a n)
  (h2 : ∀ n, a n ≤ 2016)
  (h3 : ∀ n, ∃ k : ℕ, (∏ i in Finset.range (n + 1), a i) + 1 = k ^ 2) :
  ∃ i, a i = 1 :=
sorry

end at_least_one_is_one_l586_586736


namespace probability_of_interval_l586_586138

theorem probability_of_interval {x : ℝ} (h₀ : 0 < x ∧ x < 1/2) :
    Pr (λ x, x < 1/3) = 2/3 :=
sorry

end probability_of_interval_l586_586138


namespace part1_part2_l586_586992

def U := {1, 2, 3, 4, 5, 6}
def N := {2, 3, 6}
def B := {5, 6}
def string_representation (S : Set Nat) : List Bool := List.map (λ n => n ∈ S) [1, 2, 3, 4, 5, 6]

theorem part1 :
  string_representation ((U \ N : Set Nat)) = [true, false, false, true, true, false] :=
by
  sorry

theorem part2 (A : Set Nat) :
  (string_representation (A ∪ B) = [false, true, true, false, true, true]) →
  (A ⊆ {2, 3, 5, 6} ∧ ∀ x ∈ A, x ∉ B) →
  (4 = {S : Set Nat | 
        (string_representation (S ∪ B) = [false, true, true, false, true, true]) ∧
        (S ⊆ {2, 3, 5, 6} ∧ ∀ x ∈ S, x ∉ B) }.card) :=
by
  sorry

end part1_part2_l586_586992


namespace find_number_l586_586204

theorem find_number (x : ℝ) : 
  (x + 72 = (2 * x) / (2 / 3)) → x = 36 :=
by
  intro h
  sorry

end find_number_l586_586204


namespace arithmetic_and_geometric_sequences_l586_586044

noncomputable def Sn (n : ℕ) : ℚ := (3/2) * n^2 + (1/2) * n
noncomputable def an (n : ℕ) : ℚ := 3 * n - 1
noncomputable def bn (n : ℕ) : ℚ := if n = 1 then 2 else an (n + 1)
noncomputable def Tn (n : ℕ) : ℚ := (2/3) * (4^n - 1)

theorem arithmetic_and_geometric_sequences {n : ℕ} (h1 : Sn = λ n, (3/2) * n^2 + (1/2) * n) (h2 : bn 1 = an 1) (h3 : bn 2 = an 3) :
  (an = λ n, 3 * n - 1) ∧ (Tn = λ n, (2/3) * (4^n - 1)) := 
sorry

end arithmetic_and_geometric_sequences_l586_586044


namespace green_tractor_price_is_5000_l586_586700

-- Definitions based on the given conditions
def red_tractor_price : ℝ := 20000
def green_tractor_commission_rate : ℝ := 0.20
def red_tractor_commission_rate : ℝ := 0.10
def red_tractors_sold : ℕ := 2
def green_tractors_sold : ℕ := 3
def salary : ℝ := 7000

-- The theorem statement
theorem green_tractor_price_is_5000 
  (rtp : ℝ := red_tractor_price)
  (gtcr : ℝ := green_tractor_commission_rate)
  (rtcr : ℝ := red_tractor_commission_rate)
  (rts : ℕ := red_tractors_sold)
  (gts : ℕ := green_tractors_sold)
  (s : ℝ := salary) :
  let earnings_red := rts * (rtcr * rtp) in
  let earnings_green := s - earnings_red in
  let green_tractor_price := (earnings_green / gts) / gtcr in
  green_tractor_price = 5000 := sorry

end green_tractor_price_is_5000_l586_586700


namespace total_books_sold_l586_586997

theorem total_books_sold (tuesday_books wednesday_books thursday_books : Nat) 
  (h1 : tuesday_books = 7) 
  (h2 : wednesday_books = 3 * tuesday_books) 
  (h3 : thursday_books = 3 * wednesday_books) : 
  tuesday_books + wednesday_books + thursday_books = 91 := 
by 
  sorry

end total_books_sold_l586_586997


namespace solve_system_l586_586362

theorem solve_system (x y : ℝ) (h1 : x ≥ 2 / 3) (h2 : y ≥ 2 / 3)
  (h3 : x^2 - 4 * real.sqrt (3 * x - 2) + 6 = y)
  (h4 : y^2 - 4 * real.sqrt (3 * y - 2) + 6 = x) : x = 2 ∧ y = 2 :=
  sorry

end solve_system_l586_586362


namespace Maggie_earnings_l586_586627

theorem Maggie_earnings
    (price_per_subscription : ℕ)
    (subscriptions_parents : ℕ)
    (subscriptions_grandfather : ℕ)
    (subscriptions_nextdoor : ℕ)
    (subscriptions_another : ℕ)
    (total_subscriptions : ℕ)
    (total_earnings : ℕ) :
    subscriptions_parents = 4 →
    subscriptions_grandfather = 1 →
    subscriptions_nextdoor = 2 →
    subscriptions_another = 2 * subscriptions_nextdoor →
    total_subscriptions = subscriptions_parents + subscriptions_grandfather + subscriptions_nextdoor + subscriptions_another →
    price_per_subscription = 5 →
    total_earnings = price_per_subscription * total_subscriptions →
    total_earnings = 55 :=
by
  intros
  sorry

end Maggie_earnings_l586_586627


namespace probability_of_interval_l586_586137

theorem probability_of_interval {x : ℝ} (h₀ : 0 < x ∧ x < 1/2) :
    Pr (λ x, x < 1/3) = 2/3 :=
sorry

end probability_of_interval_l586_586137


namespace average_marks_110_l586_586566

def marks_problem (P C M B E : ℕ) : Prop :=
  (C = P + 90) ∧
  (M = P + 140) ∧
  (P + C + M + B + E = P + 350) ∧
  (B = E) ∧
  (P ≥ 40) ∧
  (C ≥ 40) ∧
  (M ≥ 40) ∧
  (B ≥ 40) ∧
  (E ≥ 40)

theorem average_marks_110 (P C M B E : ℕ) (h : marks_problem P C M B E) : 
    (B + C + M) / 3 = 110 := 
by
  sorry

end average_marks_110_l586_586566


namespace smallest_perimeter_scalene_triangle_l586_586336

theorem smallest_perimeter_scalene_triangle (a b c : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > 0)
  (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) :
  a + b + c = 9 := 
sorry

end smallest_perimeter_scalene_triangle_l586_586336


namespace number_of_friends_l586_586428

def initial_candies : ℕ := 10
def additional_candies : ℕ := 4
def total_candies : ℕ := initial_candies + additional_candies
def candies_per_friend : ℕ := 2

theorem number_of_friends : total_candies / candies_per_friend = 7 :=
by
  sorry

end number_of_friends_l586_586428


namespace average_weight_of_whole_class_l586_586361

theorem average_weight_of_whole_class :
  let students_A := 50
  let students_B := 50
  let avg_weight_A := 60
  let avg_weight_B := 80
  let total_students := students_A + students_B
  let total_weight_A := students_A * avg_weight_A
  let total_weight_B := students_B * avg_weight_B
  let total_weight := total_weight_A + total_weight_B
  let avg_weight := total_weight / total_students
  avg_weight = 70 := 
by 
  sorry

end average_weight_of_whole_class_l586_586361


namespace log_expression_l586_586337

theorem log_expression :
  [Real.log10 (10 * Real.log10 1000)] ^ 2 = (1 + Real.log10 3) ^ 2 :=
by
  sorry

end log_expression_l586_586337


namespace percent_difference_l586_586411

theorem percent_difference : 0.12 * 24.2 - 0.10 * 14.2 = 1.484 := by
  sorry

end percent_difference_l586_586411


namespace max_min_sum_l586_586374

noncomputable def f : ℝ → ℝ := sorry

theorem max_min_sum (M N : ℝ) (h1 : ∀ x1 x2 : ℝ, x1 ∈ Icc (-2014) 2014 → x2 ∈ Icc (-2014) 2014 → 
  f(x1 + x2) = f(x1) + f(x2) - 2012)
  (h2 : ∀ x : ℝ, 0 < x → f(x) > 2012)
  (hM : M = Sup (f '' Icc (-2014) 2014))
  (hN : N = Inf (f '' Icc (-2014) 2014)) :
  M + N = 4024 :=
sorry

end max_min_sum_l586_586374


namespace interval_probability_l586_586151

theorem interval_probability (x y z : ℝ) (h0x : 0 < x) (hy : y > 0) (hz : z > 0) (hx : x < y) (hy_mul_z_eq_half : y * z = 1 / 2) (hx_eq_third_y : x = 1 / 3 * y) :
  (x / y) = (2 / 3) :=
by
  -- Here you should provide the proof
  sorry

end interval_probability_l586_586151


namespace value_of_product_of_sums_of_roots_l586_586606

theorem value_of_product_of_sums_of_roots 
    (a b c : ℂ)
    (h1 : a + b + c = 15)
    (h2 : a * b + b * c + c * a = 22)
    (h3 : a * b * c = 8) :
    (1 + a) * (1 + b) * (1 + c) = 46 := by
  sorry

end value_of_product_of_sums_of_roots_l586_586606


namespace cotangent_30_degrees_l586_586418

theorem cotangent_30_degrees :
  ∀ (x : ℝ), x = 30 → tan (real.to_radians x) = 1 / real.sqrt 3 →
    cot (real.to_radians x) = real.sqrt 3 :=
by
  intros x hx htan
  sorry

end cotangent_30_degrees_l586_586418


namespace find_unknown_number_l586_586470

-- Defining the conditions of the problem
def equation (x : ℝ) : Prop := (45 + x / 89) * 89 = 4028

-- Stating the theorem to be proved
theorem find_unknown_number : equation 23 :=
by
  -- Placeholder for the proof
  sorry

end find_unknown_number_l586_586470


namespace cafeteria_sales_comparison_l586_586942

theorem cafeteria_sales_comparison
  (S : ℝ) -- initial sales
  (a : ℝ) -- monthly increment for Cafeteria A
  (p : ℝ) -- monthly percentage increment for Cafeteria B
  (h1 : S > 0) -- initial sales are positive
  (h2 : a > 0) -- constant increment for Cafeteria A is positive
  (h3 : p > 0) -- constant percentage increment for Cafeteria B is positive
  (h4 : S + 8 * a = S * (1 + p) ^ 8) -- sales are equal in September 2013
  (h5 : S = S) -- sales are equal in January 2013 (trivially true)
  : S + 4 * a > S * (1 + p) ^ 4 := 
sorry

end cafeteria_sales_comparison_l586_586942


namespace probability_intervals_l586_586085

-- Definitions: interval (0, 1/2), and the condition interval (0, 1/3)
def interval1 := set.Ioo (0:ℝ) (1/2)
def interval2 := set.Ioo (0:ℝ) (1/3)

-- The proof statement: probability of randomly selecting a number from (0, 1/2), being within (0, 1/3), is 2/3.
theorem probability_intervals : (interval2.measure / interval1.measure = 2 / 3) :=
sorry

end probability_intervals_l586_586085


namespace recursive_sqrt_bound_l586_586934

noncomputable def x : ℝ := by 
  let seq (n : ℕ) : ℝ :=
    match n with
    | 0     => 3.0
    | (n+1) => real.sqrt (3 + seq n)
  exact seq 1000 -- choosing a large enough n for approximation

theorem recursive_sqrt_bound (x : ℝ) (h : x = real.sqrt (3 + x)) : 1 < x ∧ x < 3 := sorry

end recursive_sqrt_bound_l586_586934


namespace velocity_left_load_l586_586816

-- Define the conditions
def inextensible_and_weightless_strings : Prop := true
def rigid_lever : Prop := true
def velocity_right_load : ℝ := 1 -- in m/s

-- Define the required proof
theorem velocity_left_load 
  (h1 : inextensible_and_weightless_strings)
  (h2 : rigid_lever)
  (h3 : velocity_right_load = 1) : 
  ∃ u : ℝ, u = 3.5 ∧ u > 0 :=
by
  use 3.5
  split
  . refl
  . linarith

sorry

end velocity_left_load_l586_586816


namespace num_perfect_square_divisors_of_product_factorials_eq_1280_l586_586363

-- Define factorials from 1 to 10
def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

-- Define the product of factorials from 1! to 10!
def product_factorials : ℕ := 
  List.prod (List.map factorial [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

-- Define a function to count perfect square divisors of a natural number
def count_perfect_square_divisors (n : ℕ) : ℕ := sorry

-- We need to prove that the number of perfect square divisors of the product factorials is 1280.
theorem num_perfect_square_divisors_of_product_factorials_eq_1280 :
  count_perfect_square_divisors product_factorials = 1280 :=
by
  unfold product_factorials factorial count_perfect_square_divisors
  -- detailed proof omitted
  sorry

end num_perfect_square_divisors_of_product_factorials_eq_1280_l586_586363


namespace fibonacci_identity_l586_586288

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

theorem fibonacci_identity (n : ℕ) (hn : n ≥ 1) :
  fibonacci (n - 1) * fibonacci (n + 1) - fibonacci n ^ 2 = (-1 : ℤ)^n :=
by
  -- Base and induction steps are skipped for brevity.
  sorry

end fibonacci_identity_l586_586288


namespace cot_thirty_deg_l586_586416

theorem cot_thirty_deg : ∃ (θ : ℝ), θ = 30 ∧ cot θ = sqrt 3 :=
by
  sorry

end cot_thirty_deg_l586_586416


namespace diagonals_in_decagon_l586_586820

theorem diagonals_in_decagon :
  let n := 10
  let d := n * (n - 3) / 2
  d = 35 :=
by
  sorry

end diagonals_in_decagon_l586_586820


namespace find_b_and_range_c_find_equation_with_AB_congruent_triangles_exist_l586_586018

variable (b c x1 x2 : ℝ)

-- Given conditions
def parabola_eq := ∀ x y, y = -x^2 + b*x + c
def roots := x1 + x2 = 4
def intersection := parabola_eq (x1, 0) ∧ parabola_eq (x2, 0)
def distance_AB := ((x1 - x2)^2 = 4)

-- Prove statements
theorem find_b_and_range_c (h : intersection ∧ roots) : b = 4 ∧ c > -4 := sorry

theorem find_equation_with_AB (h1 : intersection ∧ roots ∧ distance_AB) : parabola_eq (b = 4 ∧ c = -3) := sorry

theorem congruent_triangles_exist (h2 : intersection ∧ roots ∧ distance_AB) :
  ∃ c, c = -2 ∨ c = (1 + Real.sqrt 17) / 2 ∨ c = (1 - Real.sqrt 17) / 2 := sorry

end find_b_and_range_c_find_equation_with_AB_congruent_triangles_exist_l586_586018


namespace A_n_squared_l586_586548

-- Define C(n-2)
def C_n_2 (n : ℕ) : ℕ := n * (n - 1) / 2

-- Define A_n_2
def A_n_2 (n : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - 2)

theorem A_n_squared (n : ℕ) (hC : C_n_2 n = 15) : A_n_2 n = 30 := by
  sorry

end A_n_squared_l586_586548


namespace find_m_l586_586492

theorem find_m
  (x : ℝ)
  (h1 : log 10 (tan x) + log 10 (cot x) = 0)
  (h2 : log 10 (sin x + cos x) = 1 / 2 * (log 10 m - 1)) :
  m = 4 :=
by
  sorry

end find_m_l586_586492


namespace maximum_value_of_f_on_interval_l586_586496

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 3

theorem maximum_value_of_f_on_interval :
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ 3) →
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = 57 :=
by
  sorry

end maximum_value_of_f_on_interval_l586_586496


namespace value_of_b_l586_586668

theorem value_of_b : ∀ (b : ℝ), (∀ (x y : ℝ), (y = 2 * x + b) → (x = -4 → y = 0)) → b = 8 :=
by
  intro b h
  specialize h (-4) 0 (by rfl)
  sorry

end value_of_b_l586_586668


namespace number_of_sides_l586_586175

-- Define the given conditions as Lean definitions

def exterior_angle := 72
def sum_of_exterior_angles := 360

-- Now state the theorem based on these conditions

theorem number_of_sides (n : ℕ) (h1 : exterior_angle = 72) (h2 : sum_of_exterior_angles = 360) : n = 5 :=
by
  sorry

end number_of_sides_l586_586175


namespace f_half_31_l586_586789

noncomputable def f : ℝ → ℝ :=
  sorry -- will be provided in the proof

variables {x : ℝ}

theorem f_half_31 [odd_function f] [even_function (λ x, f (x + 1))] (h_piecewise : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x * (3 - 2 * x)) :
  f (31 / 2) = -1 :=
by sorry

end f_half_31_l586_586789


namespace martin_and_martina_ages_l586_586995

-- Conditions
def martin_statement (x y : ℕ) : Prop := x = 3 * (2 * y - x)
def martina_statement (x y : ℕ) : Prop := 3 * x - y = 77

-- Proof problem
theorem martin_and_martina_ages :
  ∃ (x y : ℕ), martin_statement x y ∧ martina_statement x y ∧ x = 33 ∧ y = 22 :=
by {
  -- No proof required, just the statement
  sorry
}

end martin_and_martina_ages_l586_586995


namespace sufficient_not_necessary_l586_586855

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

noncomputable theory

def is_collinear (a b : V) : Prop :=
  ∃ (k : ℝ), a = k • b

theorem sufficient_not_necessary (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a + b = 0) → (is_collinear a b) ∧ ¬(is_collinear a b → a + b = 0) :=
by
  sorry

end sufficient_not_necessary_l586_586855


namespace intersection_points_count_l586_586828

theorem intersection_points_count :
  (∃ x : ℝ, |3 * x + 4| + 1 = -|4 * x - 3| - 2) → (Set.countable {x : ℝ | |3 * x + 4| + 1 = -|4 * x - 3| - 2}.card = 3)
:= sorry

end intersection_points_count_l586_586828


namespace cotangent_30_degrees_l586_586419

theorem cotangent_30_degrees :
  ∀ (x : ℝ), x = 30 → tan (real.to_radians x) = 1 / real.sqrt 3 →
    cot (real.to_radians x) = real.sqrt 3 :=
by
  intros x hx htan
  sorry

end cotangent_30_degrees_l586_586419


namespace rank_A_eq_k_l586_586973

noncomputable theory

open Matrix

variables {n : ℕ}
variables {k : ℂ}
variables {A : Matrix (Fin n) (Fin n) ℂ}

-- The problem statement in Lean 4
theorem rank_A_eq_k 
  (h1 : n > 0)
  (h2 : Tr A ≠ 0)
  (h3 : rank A + rank (scalar n (Tr A) - k • A) = n) : 
  rank A = k :=
sorry

end rank_A_eq_k_l586_586973


namespace line_segment_parameterization_l586_586296

theorem line_segment_parameterization :
  ∃ (a b c d : ℝ),
  b = 1 ∧ d = -3 ∧
  (a + b = 6) ∧ (c + d = 9) ∧
  ((a + b) * (c + d) = 54) :=
by
  use 5, 1, 12, -3
  split; [refl, split, refl, split]
  . rfl
  . rfl
  . sorry

end line_segment_parameterization_l586_586296


namespace probability_less_than_one_third_l586_586091

theorem probability_less_than_one_third : 
  (∃ p : ℚ, p = (measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ)) (set.Ioo (0 : ℚ) (1 / 3 : ℚ)) / 
                  measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ))) ∧ p = 2 / 3) :=
by {
  -- The proof will be added here.
  sorry
}

end probability_less_than_one_third_l586_586091


namespace probability_of_interval_l586_586125

theorem probability_of_interval (a b x : ℝ) (h : 0 < a ∧ a < b ∧ 0 < x) : 
  (x < b) → (b = 1/2) → (x = 1/3) → (0 < x) → (x - 0) / (b - 0) = 2/3 := 
by 
  sorry

end probability_of_interval_l586_586125


namespace limit_S_l586_586502

open Real Sequence BigOperators

def a (n : ℕ) : ℝ :=
if n = 0 then 1 else (0.5^n)

def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a i

theorem limit_S : 
  Tendsto (λ n, S n) atTop (𝓝 2) :=
begin
  sorry
end

end limit_S_l586_586502


namespace add_9999_seconds_to_4_45_pm_l586_586203

def time := Nat × Nat × Nat -- (hours, minutes, seconds)

def add_seconds_to_time (t : time) (s : Nat) : time :=
  let hours := t.1
  let minutes := t.2
  let seconds := t.3 + s
  let new_minutes := seconds / 60
  let remaining_seconds := seconds % 60
  let total_minutes := minutes + new_minutes
  let new_hours := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  let total_hours := (hours + new_hours) % 12 -- Assuming 12-hour format
  (total_hours, remaining_minutes, remaining_seconds)

theorem add_9999_seconds_to_4_45_pm : add_seconds_to_time (4, 45, 0) 9999 = (7, 31, 39) := 
by
  sorry

end add_9999_seconds_to_4_45_pm_l586_586203


namespace five_color_theorem_l586_586638

-- Define the Five Color Theorem (a general formalization in terms of graph theory)
theorem five_color_theorem (G : SimpleGraph V) [Fintype V] [DecidableRel G.Adj] (G_colorable : ∀ v ∈ G.support, Fintype.card (G.neighbors v) ≤ 5) :
  ∃ (coloring : V → Fin 5), G.IsProperColoring coloring :=
sorry

end five_color_theorem_l586_586638


namespace rectangle_area_192_l586_586664

variable (b l : ℝ) (A : ℝ)

-- Conditions
def length_is_thrice_breadth : Prop :=
  l = 3 * b

def perimeter_is_64 : Prop :=
  2 * (l + b) = 64

-- Area calculation
def area_of_rectangle : ℝ :=
  l * b

theorem rectangle_area_192 (h1 : length_is_thrice_breadth b l) (h2 : perimeter_is_64 b l) :
  area_of_rectangle l b = 192 := by
  sorry

end rectangle_area_192_l586_586664


namespace range_of_a_l586_586890

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x-1| + |x+2| > a^2 + a + 1) ↔ a ∈ set.Ioo (-2 : ℝ) (1 : ℝ) :=
by 
  sorry

end range_of_a_l586_586890


namespace product_of_undefined_x_l586_586846

-- Define the quadratic equation condition
def quad_eq (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- The main theorem to prove the product of all x such that the expression is undefined
theorem product_of_undefined_x :
  (∃ x₁ x₂ : ℝ, quad_eq 1 4 3 x₁ ∧ quad_eq 1 4 3 x₂ ∧ x₁ * x₂ = 3) :=
by
  sorry

end product_of_undefined_x_l586_586846


namespace cot_30_eq_sqrt3_l586_586413

theorem cot_30_eq_sqrt3 (h : Real.tan (Real.pi / 6) = 1 / Real.sqrt 3) : Real.cot (Real.pi / 6) = Real.sqrt 3 := by
  sorry

end cot_30_eq_sqrt3_l586_586413


namespace percentage_increase_of_cube_surface_area_l586_586339

-- Basic setup definitions and conditions
variable (a : ℝ)

-- Step 1: Initial surface area
def initial_surface_area : ℝ := 6 * a^2

-- Step 2: New edge length after 50% growth
def new_edge_length : ℝ := 1.5 * a

-- Step 3: New surface area after edge growth
def new_surface_area : ℝ := 6 * (new_edge_length a)^2

-- Step 4: Surface area after scaling by 1.5
def scaled_surface_area : ℝ := new_surface_area a * (1.5)^2

-- Prove the percentage increase
theorem percentage_increase_of_cube_surface_area :
  (scaled_surface_area a - initial_surface_area a) / initial_surface_area a * 100 = 406.25 := by
  sorry

end percentage_increase_of_cube_surface_area_l586_586339


namespace probability_less_than_one_third_l586_586162

-- Define the intervals and the associated lengths
def interval_total : ℝ := 1 / 2
def interval_condition : ℝ := 1 / 3

-- Define the probability calculation
def probability : ℝ :=
  interval_condition / interval_total

-- Statement of the problem
theorem probability_less_than_one_third : probability = 2 / 3 := by
  sorry

end probability_less_than_one_third_l586_586162


namespace find_a_plus_b_l586_586884

variables {a b : ℝ}
def f (x : ℝ) : ℝ := a^x
def g (x : ℝ) : ℝ := (2 - 7*b) * x

theorem find_a_plus_b (h1 : 0 < a) (h2 : a ≠ 1) 
                      (h3 : ∀ x ∈ set.Icc (-2:ℝ) 1, f(x) ≤ 4)
                      (h4 : ∀ x ∈ set.Icc (-2:ℝ) 1, b ≤ f(x))
                      (h5 : ∀ x1 x2, x1 < x2 → g(x1) > g(x2)) :
  a + b = 1 :=
sorry

end find_a_plus_b_l586_586884


namespace exists_M_on_EC_l586_586952

-- Define the geometry context in Lean
variables {A B C D E M : Type*} [is_metric_space A]
variables [has_dist A B C D E]
variables [is_isosceles_right_triangle A B C]
variables [is_isosceles_right_triangle A D E]
variables [non_congruent A B C A D E]
variables [fixed_triangle A B C]
variables [rotated_triangle A D E around A]

theorem exists_M_on_EC (A B C D E : Point)
  (ABC_isosceles_right : is_isosceles_right_triangle A B C)
  (ADE_isosceles_right : is_isosceles_right_triangle A D E)
  (non_congruent_tris : non_congruent A B C A D E)
  (fixed_ABC : fixed_triangle A B C)
  (rotated_ADE : rotated_triangle A D E around A) :
  ∃ M, M ∈ segment E C ∧ is_isosceles_right_triangle B M D :=
sorry

end exists_M_on_EC_l586_586952


namespace max_lateral_area_l586_586881

noncomputable def sphere_surface_area (s : ℝ) : ℝ := 8 * real.pi

def radius_of_sphere (s : ℝ) : ℝ := real.sqrt 2

noncomputable def height_of_prism (x : ℝ) : ℝ :=
  2 * real.sqrt (2 - (1 / 3) * x^2)

noncomputable def lateral_area (x : ℝ) : ℝ :=
  6 * real.sqrt (- (1 / 3) * (x^2 - 3)^2 + 3)

theorem max_lateral_area (x : ℝ) (h : 0 < x ∧ x < real.sqrt 6) :
  lateral_area (real.sqrt 3) = 6 * real.sqrt 3 := by
  sorry

end max_lateral_area_l586_586881


namespace fabric_problem_l586_586318

theorem fabric_problem
  (x y : ℝ)
  (h1 : y > 0)
  (cost_second_piece := x)
  (cost_first_piece := x + 126)
  (cost_per_meter_first := (x + 126) / y)
  (cost_per_meter_second := x / y)
  (h2 : 4 * cost_per_meter_first - 3 * cost_per_meter_second = 135)
  (h3 : 3 * cost_per_meter_first + 4 * cost_per_meter_second = 382.5) :
  y = 5.6 ∧ cost_per_meter_first = 67.5 ∧ cost_per_meter_second = 45 :=
sorry

end fabric_problem_l586_586318


namespace rectangular_to_polar_conversion_l586_586436

theorem rectangular_to_polar_conversion :
  let point := (2 : ℝ, 2 * Real.sqrt 3)
  let r := Real.sqrt ((point.1) ^ 2 + (point.2) ^ 2)
  let theta := Real.arctan (point.2 / point.1)
  (point = (2, 2 * Real.sqrt 3)) →
  r = 4 ∧ theta = Real.pi / 3 := 
by
  sorry

end rectangular_to_polar_conversion_l586_586436


namespace f_is_periodic_f_nat_exact_l586_586055

noncomputable def f : ℝ → ℝ := sorry

axiom f_functional_eq (x y : ℝ) : f x + f y = 2 * f ((x + y) / 2) * f ((x - y) / 2)
axiom f_0_nonzero : f 0 ≠ 0
axiom f_1_zero : f 1 = 0

theorem f_is_periodic : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x :=
  by
    use 4
    sorry

theorem f_nat_exact (n : ℕ) : f n = Real.cos (n * Real.pi / 2) :=
  by
    sorry

end f_is_periodic_f_nat_exact_l586_586055


namespace inclination_angle_of_line_3x_sqrt3y_minus1_l586_586660

noncomputable def inclination_angle_of_line (A B C : ℝ) (h : A ≠ 0 ∧ B ≠ 0) : ℝ :=
  let m := -A / B 
  if m = Real.tan (120 * Real.pi / 180) then 120
  else 0 -- This will return 0 if the slope m does not match, for simplifying purposes

theorem inclination_angle_of_line_3x_sqrt3y_minus1 :
  inclination_angle_of_line 3 (Real.sqrt 3) (-1) (by sorry) = 120 := 
sorry

end inclination_angle_of_line_3x_sqrt3y_minus1_l586_586660


namespace remainder_of_division_l586_586335

theorem remainder_of_division :
  ∃ R : ℕ, 176 = (19 * 9) + R ∧ R = 5 :=
by
  sorry

end remainder_of_division_l586_586335


namespace probability_intervals_l586_586088

-- Definitions: interval (0, 1/2), and the condition interval (0, 1/3)
def interval1 := set.Ioo (0:ℝ) (1/2)
def interval2 := set.Ioo (0:ℝ) (1/3)

-- The proof statement: probability of randomly selecting a number from (0, 1/2), being within (0, 1/3), is 2/3.
theorem probability_intervals : (interval2.measure / interval1.measure = 2 / 3) :=
sorry

end probability_intervals_l586_586088


namespace average_age_combined_l586_586653

theorem average_age_combined (avg_age_students : ℕ) (n_students : ℕ) (sum_age_students : ℕ)
                             (avg_age_guardians : ℕ) (n_guardians : ℕ) (sum_age_guardians : ℕ)
                             (total_age : ℕ) (total_people : ℕ) :
  avg_age_students = 13 → n_students = 40 → sum_age_students = 520 →
  avg_age_guardians = 40 → n_guardians = 60 → sum_age_guardians = 2400 →
  total_age = sum_age_students + sum_age_guardians →
  total_people = n_students + n_guardians →
  total_age / total_people = 29.2 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end average_age_combined_l586_586653


namespace number_of_girls_l586_586832

theorem number_of_girls (total_students : ℕ) (sample_size : ℕ) (girls_sampled_minus : ℕ) (girls_sampled_ratio : ℚ) :
  total_students = 1600 →
  sample_size = 200 →
  girls_sampled_minus = 20 →
  girls_sampled_ratio = 90 / 200 →
  (∃ x, x / (total_students : ℚ) = girls_sampled_ratio ∧ x = 720) :=
by intros _ _ _ _; sorry

end number_of_girls_l586_586832


namespace area_of_triangle_proof_l586_586333

noncomputable def area_of_triangle := 
  let curve1 := {p : ℝ × ℝ // p.1 ^ 2 + p.2 ^ 2 = 16}
  let curve2 := {p : ℝ × ℝ // (p.1 - 2) ^ 2 + 4 * p.2 ^ 2 = 36}
  let points_intersection :=
    {p : ℝ × ℝ // p ∈ curve1 ∧ p ∈ curve2}
  -- Calculate and prove the area from the intersections.
  (16 * real.sqrt 80) / 9

theorem area_of_triangle_proof :
  area_of_triangle = (16 * real.sqrt 80) / 9 :=
sorry

end area_of_triangle_proof_l586_586333


namespace temperature_difference_l586_586634

variable (T_morning : ℝ) (T_noon_increase : ℝ) (T_night_decrease : ℝ)

theorem temperature_difference :
  T_morning = 7 →
  T_noon_increase = 9 →
  T_night_decrease = 13 →
  let T_high := T_morning + T_noon_increase in
  let T_low := T_high - T_night_decrease in
  T_high - T_low = 13 :=
by
  intros h_morning h_noon h_night
  sorry

end temperature_difference_l586_586634


namespace probability_intervals_l586_586082

-- Definitions: interval (0, 1/2), and the condition interval (0, 1/3)
def interval1 := set.Ioo (0:ℝ) (1/2)
def interval2 := set.Ioo (0:ℝ) (1/3)

-- The proof statement: probability of randomly selecting a number from (0, 1/2), being within (0, 1/3), is 2/3.
theorem probability_intervals : (interval2.measure / interval1.measure = 2 / 3) :=
sorry

end probability_intervals_l586_586082


namespace correct_divisor_l586_586185

theorem correct_divisor (X D : ℕ) (h1 : X / 72 = 24) (h2 : X / D = 48) : D = 36 :=
sorry

end correct_divisor_l586_586185


namespace probability_merlin_dismissed_l586_586210

variable {p q : ℝ} (hpq : p + q = 1) (hp : 0 < p ∧ p < 1)

theorem probability_merlin_dismissed : 
  let decision_prob : ℝ := if h : p = 1 then 1 else p in
  if h : decision_prob = p then (1 / 2) else 0 := 
begin
  sorry
end

end probability_merlin_dismissed_l586_586210


namespace range_m_for_p_range_m_for_q_range_m_for_not_p_or_q_l586_586618

-- Define the propositions p and q
def p (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀^2 + 2 * m * x₀ + 2 + m = 0

def q (m : ℝ) : Prop :=
  1 - 2 * m < 0 ∧ m + 2 > 0 ∨ 1 - 2 * m > 0 ∧ m + 2 < 0 -- Hyperbola condition

-- Prove the ranges of m
theorem range_m_for_p {m : ℝ} (hp : p m) : m ≤ -2 ∨ m ≥ 1 :=
sorry

theorem range_m_for_q {m : ℝ} (hq : q m) : m < -2 ∨ m > (1 / 2) :=
sorry

theorem range_m_for_not_p_or_q {m : ℝ} (h_not_p : ¬ (p m)) (h_not_q : ¬ (q m)) : -2 < m ∧ m ≤ (1 / 2) :=
sorry

end range_m_for_p_range_m_for_q_range_m_for_not_p_or_q_l586_586618


namespace star_equiv_zero_l586_586439

-- Define the new operation for real numbers a and b
def star (a b : ℝ) : ℝ := (a^2 - b^2)^2

-- Prove that (x^2 - y^2) star (y^2 - x^2) equals 0
theorem star_equiv_zero (x y : ℝ) : star (x^2 - y^2) (y^2 - x^2) = 0 := 
by sorry

end star_equiv_zero_l586_586439


namespace time_for_six_visits_l586_586593

noncomputable def time_to_go_n_times (total_time : ℕ) (total_visits : ℕ) (n_visits : ℕ) : ℕ :=
  (total_time / total_visits) * n_visits

theorem time_for_six_visits (h : time_to_go_n_times 20 8 6 = 15) : time_to_go_n_times 20 8 6 = 15 :=
by
  exact h

end time_for_six_visits_l586_586593


namespace gaussian_vector_chi_squared_l586_586853

-- Definitions of Gaussian vector, mean, and covariance are assumed
def gaussian_vector (X : Type*) : Prop := sorry 

def covariance_matrix (X : Type*) : Matrix X X := sorry 

def expected_value (X : Type*) : X := sorry 

def rank (M : Matrix X X) : ℕ := sorry

-- Statement of the theorem
theorem gaussian_vector_chi_squared (X : Type*) (hX : gaussian_vector X) :
  let EX := expected_value X
  let DX := covariance_matrix X
  let DX_plus := pseudo_inverse DX
  let rank_DX := rank DX
  let Y := (X - EX)
  in Y^T * DX_plus * Y ∼ chi_squared rank_DX := 
sorry

end gaussian_vector_chi_squared_l586_586853


namespace compare_abs_m_n_l586_586930
noncomputable theory

theorem compare_abs_m_n (m n : ℝ) (h1 : m * n < 0) (h2 : m + n < 0) (h3 : n > 0) : |m| > |n| :=
sorry

end compare_abs_m_n_l586_586930


namespace great_grandson_age_is_36_l586_586378

-- Define the problem conditions and the required proof
theorem great_grandson_age_is_36 :
  ∃ n : ℕ, (∃ k : ℕ, k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧  2 * k * 111 = n * (n + 1)) ∧ n = 36 :=
by
  sorry

end great_grandson_age_is_36_l586_586378


namespace max_principals_in_10_years_l586_586408

theorem max_principals_in_10_years : 
  ∀ (n : ℕ), 
    (∀ (term : ℕ), term = 3) → 
    (∀ (period : ℕ), period = 10) → 
    n ≤ period → 
    ∃ p, p = 5 := 
by
  intros n term_add term_val period_val period_limit
  sorry

end max_principals_in_10_years_l586_586408


namespace trash_can_prices_and_minimum_A_can_purchase_l586_586698

theorem trash_can_prices_and_minimum_A_can_purchase 
  (x y : ℕ) 
  (h₁ : 3 * x + 4 * y = 580)
  (h₂ : 6 * x + 5 * y = 860)
  (total_trash_cans : ℕ)
  (total_cost : ℕ)
  (cond₃ : total_trash_cans = 200)
  (cond₄ : 60 * (total_trash_cans - x) + 100 * x ≤ 15000) : 
  x = 60 ∧ y = 100 ∧ x ≥ 125 := 
sorry

end trash_can_prices_and_minimum_A_can_purchase_l586_586698


namespace minimal_distance_l586_586744

-- Define points p1 and p2.
variable (p1 p2 : ℝ × ℝ)

-- Define the conditions for the problem.
-- The ball hits 3 sides of the billiard table.
def hits_three_sides (table : set (ℝ × ℝ)) (p1 p2 : ℝ × ℝ) : Prop :=
-- The specific condition on how the ball hits three sides would be more detailed, but for our purposes
-- we keep it abstract. 
sorry

-- Statement of the problem.
theorem minimal_distance (h : hits_three_sides (λ x, true) p1 p2) : 
  distance p1 p2 = 28 * real.sqrt 2 := sorry

end minimal_distance_l586_586744


namespace factor_tree_X_value_l586_586186

theorem factor_tree_X_value :
  let F := 7 * 3,
      G := 11 * 3,
      P := 7 * F,
      Q := 11 * G,
      X := P * Q in
  X = 53361 := 
by
  sorry

end factor_tree_X_value_l586_586186


namespace number_of_positive_divisors_360_l586_586915

theorem number_of_positive_divisors_360 : 
  let n := 360 
  in let prime_factors := [(2, 3), (3, 2), (5, 1)]
  in (∀ (p : ℕ) (a : ℕ), (p, a) ∈ prime_factors → p.prime) →
     (∀ m ∈ prime_factors, ∃ (p a : ℕ), m = (p, a) ∧ n = (p ^ a) * (prime_factors.filter (λ m', m ≠ m')).prod (λ m', (m'.fst ^ m'.snd))) →
     (prime_factors.foldr (λ (m : ℕ × ℕ) acc, (m.snd + 1) * acc) 1) = 24 := 
begin
  sorry
end

end number_of_positive_divisors_360_l586_586915


namespace angles_sum_to_180_l586_586587

theorem angles_sum_to_180 (A B C D E F : Point)
  (h_triangle : is_triangle A B C)
  (h_point_D : inside D A B C)
  (h_point_E_on_AD : on_line_segment E A D)
  (h_point_F_on_CD : on_line_segment F C D)
  (angle_x : Angle)
  (angle_y : Angle)
  (angle_z : Angle)
  (h_angle_x : angle_x = angle E A B)
  (h_angle_y : angle_y = angle F C B)
  (h_angle_z : angle_z = angle E B F) :
  angle_x + angle_y + angle_z = 180 :=
sorry

end angles_sum_to_180_l586_586587


namespace probability_merlin_dismissed_l586_586212

variable {p q : ℝ} (hpq : p + q = 1) (hp : 0 < p ∧ p < 1)

theorem probability_merlin_dismissed : 
  let decision_prob : ℝ := if h : p = 1 then 1 else p in
  if h : decision_prob = p then (1 / 2) else 0 := 
begin
  sorry
end

end probability_merlin_dismissed_l586_586212


namespace percentage_increase_l586_586775

theorem percentage_increase 
    (P : ℝ)
    (buying_price : ℝ) (h1 : buying_price = 0.80 * P)
    (selling_price : ℝ) (h2 : selling_price = 1.24 * P) :
    ((selling_price - buying_price) / buying_price) * 100 = 55 := by 
  sorry

end percentage_increase_l586_586775


namespace geometric_probability_l586_586102

theorem geometric_probability :
  let total_length := (1/2 - 0)
  let favorable_length := (1/3 - 0)
  total_length > 0 ∧ favorable_length > 0 →
  (favorable_length / total_length) = 2 / 3 :=
by
  intros _ _ _
  sorry

end geometric_probability_l586_586102


namespace probability_not_6_in_six_rolls_l586_586715

-- Define the probability of not rolling a 6 on a single roll of a fair die
def prob_not_6_one_roll : ℚ := 5 / 6

-- Define the probability of not rolling a 6 in 6 independent rolls
def prob_not_6_six_rolls : ℚ := prob_not_6_one_roll ^ 6

-- Define the correct approximate probability value
def approx_prob_not_6_six_rolls : ℚ := 0.3349

-- The main statement to be proved
theorem probability_not_6_in_six_rolls :
  prob_not_6_six_rolls ≈ approx_prob_not_6_six_rolls := 
by
  sorry

end probability_not_6_in_six_rolls_l586_586715


namespace share_of_y_l586_586772

theorem share_of_y (A y z : ℝ)
  (hx : y = 0.45 * A)
  (hz : z = 0.30 * A)
  (h_total : A + y + z = 140) :
  y = 36 := by
  sorry

end share_of_y_l586_586772


namespace area_EFGH_eq_l586_586583

-- Definitions/Conditions
def point := ℝ × ℝ
def length (p q : point) := real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

variable {A B C D E : point}
variable {F G H : point}

-- Given: ABCD is a square with side length 10 units
axiom AB_eq_10 : length A B = 10
axiom BC_eq_10 : length B C = 10
axiom CD_eq_10 : length C D = 10
axiom DA_eq_10 : length D A = 10

-- Given: BE = 3 units
axiom BE_eq_3 : length B E = 3

-- E lies on CD
axiom E_on_CD : ∃ a, E = (D.1 - a * (D.1 - C.1), D.2 - a * (D.2 - C.2))

-- EFGH is a square
axiom square_EFGH : length E F = length F G ∧ length G H = length H E

-- Prove the area of square EFGH is 100 - 6 * sqrt 91
theorem area_EFGH_eq : ∃ side_length, side_length = real.sqrt 91 - 3 ∧ 
  (side_length * side_length = 100 - 6 * real.sqrt 91) := by
  -- code to construct the proof
  sorry

end area_EFGH_eq_l586_586583


namespace tiles_walked_on_l586_586383

/-- 
A park has a rectangular shape with a width of 13 feet and a length of 19 feet.
Square-shaped tiles of dimension 1 foot by 1 foot cover the entire area.
The gardener walks in a straight line from one corner of the rectangle to the opposite corner.
One specific tile in the path is not to be stepped on. 
Prove that the number of tiles the gardener walks on is 30.
-/
theorem tiles_walked_on (width length gcd_width_length tiles_to_avoid : ℕ)
  (h_width : width = 13)
  (h_length : length = 19)
  (h_gcd : gcd width length = 1)
  (h_tiles_to_avoid : tiles_to_avoid = 1) : 
  (width + length - gcd_width_length - tiles_to_avoid = 30) := 
by
  sorry

end tiles_walked_on_l586_586383


namespace problem_statement_l586_586858

-- Define the conditions as a Lean statement
theorem problem_statement (a b : ℝ) (h : (a - 1)^2 + |b + 2| = 0) : 
  2 * (5 * a^2 - 7 * a * b + 9 * b^2) - 3 * (14 * a^2 - 2 * a * b + 3 * b^2) = 20 :=
begin
  -- Proof goes here
  sorry
end

end problem_statement_l586_586858


namespace ordered_pairs_count_l586_586535

open Real

theorem ordered_pairs_count :
  ∃ x : ℕ, x = 597 ∧ ∀ (a : ℝ) (b : ℕ), (0 < a) → (2 ≤ b ∧ b ≤ 200) →
    (log b a ^ 2017 = log b (a ^ 2017) → True) :=
begin
  sorry
end

end ordered_pairs_count_l586_586535


namespace eccentricity_of_ellipse_l586_586870

noncomputable def ellipse_eccentricity (a b c : ℝ) : ℝ := c / a

theorem eccentricity_of_ellipse:
  ∀ (a b : ℝ) (c : ℝ), 
    0 < b ∧ b < a ∧ a = 3 * c → 
    ellipse_eccentricity a b c = 1/3 := by
  intros a b c h
  let e := ellipse_eccentricity a b c
  have h1 : 0 < b := h.1
  have h2 : b < a := h.2.left
  have h3 : a = 3 * c := h.2.right
  simp [ellipse_eccentricity, h3]
  sorry

end eccentricity_of_ellipse_l586_586870


namespace g_f_3_eq_1476_l586_586241

def f (x : ℝ) : ℝ := x^3 - 2 * x + 1
def g (x : ℝ) : ℝ := 3 * x^2 + x + 2

theorem g_f_3_eq_1476 : g (f 3) = 1476 :=
by
  sorry

end g_f_3_eq_1476_l586_586241


namespace num_divisors_360_l586_586916

theorem num_divisors_360 :
  ∀ n : ℕ, n = 360 → (∀ (p q r : ℕ), p = 2 ∧ q = 3 ∧ r = 5 →
    (∃ (a b c : ℕ), 360 = p^a * q^b * r^c ∧ a = 3 ∧ b = 2 ∧ c = 1) →
    (3+1) * (2+1) * (1+1) = 24) :=
  sorry

end num_divisors_360_l586_586916


namespace find_arith_seq_sum_l586_586580

noncomputable def arith_seq_sum : ℕ → ℕ → ℕ
| 0, d => 2
| (n+1), d => arith_seq_sum n d + d

theorem find_arith_seq_sum :
  ∃ d : ℕ, 
    arith_seq_sum 1 d + arith_seq_sum 2 d = 13 ∧
    arith_seq_sum 3 d + arith_seq_sum 4 d + arith_seq_sum 5 d = 42 :=
by
  sorry

end find_arith_seq_sum_l586_586580


namespace eleven_to_the_fourth_l586_586731

theorem eleven_to_the_fourth : 11^4 = 14641 :=
by
  have base2: 11^2 = 121 := by sorry
  have base3: 11^3 = 1331 := by sorry
  have eq1: (10+1)^2 = 11^2 := by sorry
  have eq2: (10+1)^3 = 11^3 := by sorry
  have eq3: (10+1)^4 = 11^4 := by sorry
  -- Following from binomial expansion
  calc
  11^4 = (10+1)^4 : by rw eq3
       ... = ∑ k in finset.range 5, (nat.choose 4 k) * (10^(4-k)) * (1^k) : by sorry
       ... = 14641 : by sorry

end eleven_to_the_fourth_l586_586731


namespace quotient_of_x6_plus_8_by_x_minus_1_l586_586467

theorem quotient_of_x6_plus_8_by_x_minus_1 :
  ∀ (x : ℝ), x ≠ 1 →
  (∃ Q : ℝ → ℝ, x^6 + 8 = (x - 1) * Q x + 9 ∧ Q x = x^5 + x^4 + x^3 + x^2 + x + 1) := 
  by
    intros x hx
    sorry

end quotient_of_x6_plus_8_by_x_minus_1_l586_586467


namespace bathroom_visits_time_l586_586591

variable (t_8 : ℕ) (n8 : ℕ) (n6 : ℕ)

theorem bathroom_visits_time (h1 : t_8 = 20) (h2 : n8 = 8) (h3 : n6 = 6) :
  (t_8 / n8) * n6 = 15 := by
  sorry

end bathroom_visits_time_l586_586591


namespace total_wheels_at_station_l586_586686

theorem total_wheels_at_station (trains carriages rows wheels : ℕ) 
  (h_trains : trains = 4)
  (h_carriages : carriages = 4)
  (h_rows : rows = 3)
  (h_wheels : wheels = 5) : 
  trains * carriages * rows * wheels = 240 := 
by 
  rw [h_trains, h_carriages, h_rows, h_wheels]
  exact Nat.mul_eq_iff_eq_div.mpr rfl

end total_wheels_at_station_l586_586686


namespace option_a_option_b_option_c_option_d_l586_586959

-- Assume we have a triangle ABC
variables {A B C : Real} {a b c : Real} (h_triangle: A + B + C = π)

-- Option A
theorem option_a (h1 : A > B) : sin A > sin B := sorry

-- Option B (negation form)
theorem option_b : ¬(A = 60 ∧ c = 2 ∧ a = 1.74 ∧ 
  (∃ B C b, 
    A + B + C = π ∧ 
    b * sin C = c * sin B ∧ 
    a * sin C = c * sin A ∧ 
    sin A = sin (60:Real) ∧ 
    sin B = sin (B:Real)
    )
) := sorry

-- Option C (negation form)
theorem option_c (h2: tan A = a / b) : ¬ (A = π / 2 ∨ B = π / 2 ∨ C = π / 2) := sorry

-- Option D
theorem option_d (h_triangle: A + B + C = π) : cos A + cos B + cos C > 0 := sorry

end option_a_option_b_option_c_option_d_l586_586959


namespace find_range_of_a_l586_586506

noncomputable def range_of_a (a : ℝ) (x₀ : ℝ) : Prop :=
  1 ≤ a ∧ a ≤ 3 / 2

theorem find_range_of_a :
  ∃ x₀ ∈ set.Icc (0:ℝ) (3/2),
  let k₁ := (a * x₀ + a - 1) * Real.exp x₀,
      k₂ := (x₀ - 2) * Real.exp (-x₀)
  in k₁ * k₂ = -1 →
      range_of_a (a := (x₀ - 3) / (x₀^2 - x₀ - 2)) x₀ :=
sorry

end find_range_of_a_l586_586506


namespace gray_region_area_l586_586763

noncomputable def area_of_gray_region (length width : ℝ) (angle_deg : ℝ) : ℝ :=
  if (length = 55 ∧ width = 44 ∧ angle_deg = 45) then 10 else 0

theorem gray_region_area :
  area_of_gray_region 55 44 45 = 10 :=
by sorry

end gray_region_area_l586_586763


namespace number_of_committees_with_mixed_genders_l586_586684

def boys : ℕ := 21
def girls : ℕ := 14
def total_students : ℕ := boys + girls
def committee_size : ℕ := 4

theorem number_of_committees_with_mixed_genders : 
  (∑ n in (finset.range (total_students.choose committee_size + 1)), if n ≥ boys.choose committee_size + girls.choose committee_size then total_students.choose committee_size - (boys.choose committee_size + girls.choose committee_size) else 0) = 27285 := 
  sorry

end number_of_committees_with_mixed_genders_l586_586684


namespace geometric_probability_l586_586103

theorem geometric_probability :
  let total_length := (1/2 - 0)
  let favorable_length := (1/3 - 0)
  total_length > 0 ∧ favorable_length > 0 →
  (favorable_length / total_length) = 2 / 3 :=
by
  intros _ _ _
  sorry

end geometric_probability_l586_586103


namespace make_all_red_l586_586754

theorem make_all_red (n : ℕ) (points : Finset (ℝ × ℝ × ℝ)) 
  (h_no_four_coplanar : ∀ {a b c d : (ℝ × ℝ × ℝ)}, a ∈ points → b ∈ points → c ∈ points → d ∈ points → ¬ coplanar {a, b, c, d})
  (color : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ) → Prop) 
  (color_initial : ∀ (a b : (ℝ × ℝ × ℝ)), a ∈ points → b ∈ points → (color a b = red ∨ color a b = blue))
  (switch : (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ) → (ℝ × ℝ × ℝ))
  (h_three_points : ∀ (a b c : (ℝ × ℝ × ℝ)), a ∈ points → b ∈ points → c ∈ points → ∃ (switches : list (ℝ × ℝ × ℝ)), ∀ (p : (ℝ × ℝ × ℝ)), p ∈ switches → p ∈ points ∧ (switch p a = red ∧ switch p b = red ∧ switch p c = red)) :
  ∃ (switches : list (ℝ × ℝ × ℝ)), switches.length ≤ n / 2 ∧ ∀ (a b : (ℝ × ℝ × ℝ)), a ∈ points → b ∈ points → switch (∀ p ∈ switches, p) a = red ∧ switch (∀ p ∈ switches, p) b = red :=
sorry

end make_all_red_l586_586754


namespace ribbon_cuts_l586_586373

theorem ribbon_cuts (rolls : ℕ) (length_per_roll : ℕ) (piece_length : ℕ) (total_rolls : rolls = 5) (roll_length : length_per_roll = 50) (piece_size : piece_length = 2) : 
  (rolls * ((length_per_roll / piece_length) - 1) = 120) :=
by
  sorry

end ribbon_cuts_l586_586373


namespace compare_fractions_l586_586806

theorem compare_fractions : (- (4 / 5) < - (2 / 3)) :=
by
  sorry

end compare_fractions_l586_586806


namespace product_largest_second_smallest_l586_586317

theorem product_largest_second_smallest 
  (a b c : Nat)
  (h1 : a = 10) 
  (h2 : b = 11) 
  (h3 : c = 12) : 
  (max (max a b) c) * (min (max a b) (max b c)) = 132 :=
by
  -- Now let's verify whether the conditions are properly set up and evaluated.
  have max_ab : max a b = 11 := by rw [h1, h2]; exact rfl
  have max_bc : max b c = 12 := by rw [h2, h3]; exact rfl
  have max_abc : max (max a b) c = 12 := by rw [max_ab, max_bc]; exact rfl
  have min_max_bc_max_ab : min (max a b) (max b c) = 11 := by rw [max_ab, max_bc]; exact rfl
  rw [max_abc, min_max_bc_max_ab]
  -- Compute the product
  show 12 * 11 = 132 from rfl

end product_largest_second_smallest_l586_586317


namespace number_of_divisors_360_l586_586924

theorem number_of_divisors_360 : 
  ∃ (e1 e2 e3 : ℕ), e1 = 3 ∧ e2 = 2 ∧ e3 = 1 ∧ (∏ e in [e1, e2, e3], e + 1) = 24 := by
    use 3, 2, 1
    split
    { exact rfl }
    split
    { exact rfl }
    split
    { exact rfl }
    simp
    norm_num

end number_of_divisors_360_l586_586924


namespace determinant_cot_l586_586246

variable {A B C : ℝ}
variable (h_non_obtuse : A + B + C = π)
variable (h_A_pos : A > 0) (h_B_pos : B > 0) (h_C_pos : C > 0)

def cot (x : ℝ) := 1 / tan x

theorem determinant_cot (h_tri : A + B + C = π) (hA : A > 0) (hB : B > 0) (hC : C > 0) :
\[
    \begin{vmatrix} cot A & 1 & 1 \\ 1 & cot B & 1 \\ 1 & 1 & cot C \end{vmatrix} = 2
\]
:= sorry

end determinant_cot_l586_586246


namespace Maggie_earnings_l586_586629

theorem Maggie_earnings
    (price_per_subscription : ℕ)
    (subscriptions_parents : ℕ)
    (subscriptions_grandfather : ℕ)
    (subscriptions_nextdoor : ℕ)
    (subscriptions_another : ℕ)
    (total_subscriptions : ℕ)
    (total_earnings : ℕ) :
    subscriptions_parents = 4 →
    subscriptions_grandfather = 1 →
    subscriptions_nextdoor = 2 →
    subscriptions_another = 2 * subscriptions_nextdoor →
    total_subscriptions = subscriptions_parents + subscriptions_grandfather + subscriptions_nextdoor + subscriptions_another →
    price_per_subscription = 5 →
    total_earnings = price_per_subscription * total_subscriptions →
    total_earnings = 55 :=
by
  intros
  sorry

end Maggie_earnings_l586_586629


namespace diagonals_of_decagon_l586_586822

theorem diagonals_of_decagon : 
  let n := 10 in 
  (n * (n - 3)) / 2 = 35 := 
by
  let n := 10
  show (n * (n - 3)) / 2 = 35
  sorry

end diagonals_of_decagon_l586_586822


namespace translated_graph_min_point_l586_586658

theorem translated_graph_min_point :
  let f := λ x : ℝ, |x| - 4
  let g := λ x : ℝ, |x - 3| - 8
  ∃ x₀ y₀, (∀ x y, y = g x → y ≥ y₀) ∧ g x₀ = y₀ ∧ (x₀, y₀) = (3, -8) :=
by
  intros
  sorry

end translated_graph_min_point_l586_586658


namespace trishul_investment_less_than_raghu_l586_586331

noncomputable def VishalInvestment (T : ℝ) : ℝ := 1.10 * T

noncomputable def TotalInvestment (T : ℝ) (R : ℝ) : ℝ :=
  T + VishalInvestment T + R

def RaghuInvestment : ℝ := 2100

def TotalSumInvested : ℝ := 6069

theorem trishul_investment_less_than_raghu :
  ∃ T : ℝ, TotalInvestment T RaghuInvestment = TotalSumInvested → (RaghuInvestment - T) / RaghuInvestment * 100 = 10 := by
  sorry

end trishul_investment_less_than_raghu_l586_586331


namespace area_of_parallelogram_l586_586841

/-- The vectors a and b used in the problem -/
def vector_a : ℝ^3 := ⟨2, 4, -1⟩
def vector_b : ℝ^3 := ⟨3, -1, 5⟩

/-- The area of the parallelogram formed by vectors a and b is sqrt(726) -/
theorem area_of_parallelogram : 
  let cross_product := λ u v : ℝ^3, ⟨
    u.2 * v.3 - u.3 * v.2, 
    u.3 * v.1 - u.1 * v.3, 
    u.1 * v.2 - u.2 * v.1 
  ⟩ in
  ∥cross_product vector_a vector_b∥ = Real.sqrt 726 :=
by
  sorry

end area_of_parallelogram_l586_586841


namespace afternoon_pear_sales_l586_586765

theorem afternoon_pear_sales (morning_sales afternoon_sales total_sales : ℕ)
  (h1 : afternoon_sales = 2 * morning_sales)
  (h2 : total_sales = morning_sales + afternoon_sales)
  (h3 : total_sales = 420) : 
  afternoon_sales = 280 :=
by {
  -- placeholders for the proof
  sorry 
}

end afternoon_pear_sales_l586_586765


namespace y_incr_decr_on_domains_range_of_a_l586_586292

noncomputable theory
open Function Real

-- Definition and conditions
variable (f : ℝ → ℝ)
variable (domain_f : ∀ x : ℝ, ∃ y : ℝ, f x = y)
variable (additivity : ∀ x y : ℝ, f (x + y) = f x + f y)
variable (increasing_ge_0 : ∀ x y : ℝ, 0 ≤ x ∧ x ≤ y → f x ≤ f y)

-- Question 1:
theorem y_incr_decr_on_domains :
  ∀ x : ℝ, (x ∈ Iio 0 → ∀ y : ℝ, y = -(f x) ^ 2 → y ≤ y) ∧
           (x ∈ Ici 0 → ∀ y : ℝ, y = -(f x) ^ 2 → y ≥ y) :=
sorry

-- Additional condition for Question 2:
variable (f1 : f 1 = 2)

-- Question 2:
theorem range_of_a (a : ℝ) :
  f (2 * a ^ 2 - 1) + 2 * f a - 6 < 0 → -2 < a ∧ a < 1 :=
sorry

end y_incr_decr_on_domains_range_of_a_l586_586292


namespace min_value_f_solve_inequality_f_l586_586053

def f (x : ℝ) : ℝ := |x - 5/2| + |x - 1/2|

theorem min_value_f : ∃ x, ∀ y, f y ≥ f x ∧ f x = 2 := 
sorry

theorem solve_inequality_f : {x : ℝ | f x ≤ x + 4} = {x : ℝ | -1/3 ≤ x ∧ x ≤ 7} := 
sorry

end min_value_f_solve_inequality_f_l586_586053


namespace distance_of_ellipse_points_is_2_sqrt_5_l586_586808

noncomputable def distance_CD : ℝ :=
  let ellipse_eq : (ℝ × ℝ) → Prop := λ (x y), 16 * (x + 2)^2 + 4 * y^2 = 64
  let center := (-2, 0)
  let major_axis_end := (-2, 4)
  let minor_axis_end := (0, 0)
  real.sqrt (((-2 - 0)^2 + (4 - 0)^2) : ℝ)

theorem distance_of_ellipse_points_is_2_sqrt_5 :
  distance_CD = 2 * real.sqrt 5 := by
  sorry

end distance_of_ellipse_points_is_2_sqrt_5_l586_586808


namespace prime_solution_l586_586457

theorem prime_solution (p : ℕ) (hp : Nat.Prime p) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ (p = 2 ∨ p = 3 ∨ p = 7) :=
by
  sorry

end prime_solution_l586_586457


namespace probability_merlin_dismissed_l586_586231

-- Define the conditions
variables (p : ℝ) (q : ℝ) (hpq : p + q = 1) (hp_pos : 0 < p) (hq_pos : 0 < q)

/--
Given advisor Merlin is equally likely to dismiss as Percival
since they are equally likely to give the correct answer independently,
prove that the probability of Merlin being dismissed is \( \frac{1}{2} \).
-/
theorem probability_merlin_dismissed : (1/2 : ℝ) = 1/2 :=
by 
  sorry

end probability_merlin_dismissed_l586_586231


namespace polynomial_equivalence_l586_586614

-- Define the polynomial T in terms of x.
def T (x : ℝ) : ℝ := (x-2)^5 + 5 * (x-2)^4 + 10 * (x-2)^3 + 10 * (x-2)^2 + 5 * (x-2) + 1

-- Define the target polynomial.
def target (x : ℝ) : ℝ := (x-1)^5

-- State the theorem that T is equivalent to target.
theorem polynomial_equivalence (x : ℝ) : T x = target x :=
by
  sorry

end polynomial_equivalence_l586_586614


namespace f_recursive_l586_586543

noncomputable def f (k : ℕ) : ℚ :=
  Finset.sum (Finset.range k) (λ n, if even n then -1 / (n + 1 : ℚ) else 1 / (n + 1 : ℚ))

theorem f_recursive (x : ℕ) : f (x + 1) = f x + (1 / (2 * x + 1 : ℚ) - 1 / (2 * x + 2 : ℚ)) :=
by
  sorry

end f_recursive_l586_586543


namespace simplify_expression_l586_586274

theorem simplify_expression :
  (3 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7) =
  6 * Real.sqrt 6 + 6 * Real.sqrt 10 - 6 * Real.sqrt 14 :=
sorry

end simplify_expression_l586_586274


namespace probability_of_interval_l586_586144

theorem probability_of_interval {x : ℝ} (h₀ : 0 < x ∧ x < 1/2) :
    Pr (λ x, x < 1/3) = 2/3 :=
sorry

end probability_of_interval_l586_586144


namespace probability_less_than_third_l586_586113

def interval_real (a b : ℝ) := set.Ioo a b

def probability_space (a b : ℝ) := {
  length := b - a,
  sample_space := interval_real a b
}

def probability_of_event {a b c : ℝ} (h : a < b) (h1 : a < c) (h2 : c < b) : Prop :=
  (c - a) / (b - a)

theorem probability_less_than_third :
  probability_of_event (by norm_num : (0:ℝ) < (1 / 2))
                       (by norm_num : (0:ℝ) < (1 / 3))
                       (by norm_num : (1 / 3) < (1 / 2))
  = (2 / 3) := sorry

end probability_less_than_third_l586_586113


namespace person_a_mistake_person_b_mistake_correct_solution_l586_586343

def equation_system_person_a (x y a : ℝ) : Prop :=
  (2 * a * x + y = 5) ∧ (2 * x - b * y = 13)

def equation_system_person_b (x y b : ℝ) : Prop :=
  (2 * a * x + y = 5) ∧ (2 * x - b * y = 13)

theorem person_a_mistake (x y : ℝ) (a b : ℝ) :
  x = 7 / 2 → y = -2 → 2 * a * x + y = 5 → 2 * x - b * y = 13 → a = 1 ∧ b = 3 :=
by
  intros h1 h2 h3 h4
  sorry

theorem person_b_mistake (x y : ℝ) (a b : ℝ) :
  x = 3 → y = -7 → 2 * a * x + y = 5 → 2 * x - b * y = 13 → a = 2 ∧ b = 1 :=
by
  intros h1 h2 h3 h4
  sorry

theorem correct_solution (a b x y : ℝ) :
  a = 2 → b = 3 → 4 * x + y = 5 → 2 * x - 3 * y = 13 → x = 2 ∧ y = -3 :=
by
  intros h1 h2 h3 h4 
  sorry

end person_a_mistake_person_b_mistake_correct_solution_l586_586343


namespace train_cross_pole_time_l586_586589

theorem train_cross_pole_time (L : ℝ) (S : ℝ) (hL : L = 50) (hS : S = 360) : 
  let conversion_factor := (1000 : ℝ) / (3600 : ℝ) in
  let S_m_s := S * conversion_factor in
  let t := L / S_m_s in
  t = 0.5 :=
by
  sorry

end train_cross_pole_time_l586_586589


namespace range_of_a_l586_586847

theorem range_of_a (a : ℝ) : 
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ π / 2 → 
    (x + 3 + 2 * (Real.sin θ) * (Real.cos θ))^2 + (x + a * (Real.sin θ) + a * (Real.cos θ))^2 ≥ 1 / 8) → 
  a ≥ 7 / 2 ∨ a ≤ Real.sqrt 6 :=
sorry

end range_of_a_l586_586847


namespace num_pos_ints_satisfying_conditions_l586_586070

theorem num_pos_ints_satisfying_conditions : 
  {n : ℤ | 9 < n ∧ n < 150}.card = 140 := 
by
  sorry

end num_pos_ints_satisfying_conditions_l586_586070


namespace total_books_sold_l586_586998

theorem total_books_sold (tuesday_books wednesday_books thursday_books : Nat) 
  (h1 : tuesday_books = 7) 
  (h2 : wednesday_books = 3 * tuesday_books) 
  (h3 : thursday_books = 3 * wednesday_books) : 
  tuesday_books + wednesday_books + thursday_books = 91 := 
by 
  sorry

end total_books_sold_l586_586998


namespace geometric_S5_l586_586033

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := 
  ∀ n, a (n + 1) = a n * q

def Sn (a : ℕ → ℝ) (n : ℕ) := ∑ i in range n, a i

theorem geometric_S5 {a : ℕ → ℝ} {q : ℝ} 
  (h_seq : geometric_sequence a q)
  (h1 : a 2 * a 4 = (1/4) * a 0)
  (h2 : (a 3 + a 6) / 2 = 9 / 8):
  Sn a 5 = 31 :=
sorry

end geometric_S5_l586_586033


namespace problem_part_one_problem_part_two_l586_586201

variables {a b c : ℝ} {A B C : ℝ}

def m_vector (C : ℝ) : ℝ × ℝ :=
  (2 * Real.cos (C / 2), -Real.sin C)

def n_vector (C : ℝ) : ℝ × ℝ :=
  (Real.cos (C / 2), 2 * Real.sin C)

def perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem problem_part_one (C : ℝ) (h : perpendicular (m_vector C) (n_vector C)) : C = Real.pi / 3 :=
  sorry

theorem problem_part_two (a b c A : ℝ) (h1 : a ^ 2 = 2 * b ^ 2 + c ^ 2) : Real.tan A = -3 * Real.sqrt 3 :=
  sorry

end problem_part_one_problem_part_two_l586_586201


namespace part1_part2_l586_586804

open Real

theorem part1:
  sqrt 12 + sqrt 15 / sqrt 5 = 3 * sqrt 3 :=
sorry

theorem part2:
  (3 / sqrt 3) - (π - 3)^0 + ((-1 / 2)^(-1)) - sqrt 27 + abs (sqrt 3 - 2) = -1 - 3 * sqrt 3 :=
sorry

end part1_part2_l586_586804


namespace interval_probability_l586_586152

theorem interval_probability (x y z : ℝ) (h0x : 0 < x) (hy : y > 0) (hz : z > 0) (hx : x < y) (hy_mul_z_eq_half : y * z = 1 / 2) (hx_eq_third_y : x = 1 / 3 * y) :
  (x / y) = (2 / 3) :=
by
  -- Here you should provide the proof
  sorry

end interval_probability_l586_586152


namespace max_zoo_area_l586_586329

theorem max_zoo_area (length width x y : ℝ) (h1 : length = 16) (h2 : width = 8 - x) (h3 : y = x * (8 - x)) : 
  ∃ M, ∀ x, 0 < x ∧ x < 8 → y ≤ M ∧ M = 16 :=
by
  sorry

end max_zoo_area_l586_586329


namespace sheila_hourly_wage_l586_586270

/-- Sheila works 8 hours per day on Monday, Wednesday, and Friday. -/
axiom hours_MWF : ℕ := 8
/-- Sheila works 6 hours per day on Tuesday and Thursday. -/
axiom hours_TT : ℕ := 6
/-- Number of days Sheila works 8 hours per day. -/
axiom days_MWF : ℕ := 3
/-- Number of days Sheila works 6 hours per day. -/
axiom days_TT : ℕ := 2
/-- Sheila's weekly earnings in dollars. -/
axiom weekly_earnings : ℕ := 504

def total_hours_worked : ℕ := (hours_MWF * days_MWF) + (hours_TT * days_TT)

def hourly_wage (weekly_earnings : ℕ) (total_hours_worked : ℕ) : ℕ :=
  weekly_earnings / total_hours_worked

theorem sheila_hourly_wage : hourly_wage weekly_earnings total_hours_worked = 14 := by
  sorry

end sheila_hourly_wage_l586_586270


namespace concyclic_points_B_P_Q_X_l586_586395

universe u

variable {α : Type u} 

-- Define the points A, B, C, M, P, Q, R, X and the circle ω
variables (A B C M P Q R X : α)
variable (ω : set α) -- ω represents the circle

-- Conditions
variable [inhabited α] [linear_order α]

--- Assume necessary structural properties for the points and ω
variable (triangle_ABC_inscribed : ω ⊆ {A, B, C})
variable (external_angle_bisector_B_intersects_ω_at_M : M ∈ ω)
variable (line_parallel_BM_intersects : ∃ P Q R, P ∈ line_segment(B, C) ∧ Q ∈ line_segment(A, B) ∧ R ∈ extension(A, C))
variable (line_MR_intersects_ω_at_X : X ∈ ω ∧ X ∈ line_segment(M, R))

-- The statement to prove
theorem concyclic_points_B_P_Q_X :
  ∃ (circle : set α), {B, P, Q, X} ⊆ circle :=
sorry

end concyclic_points_B_P_Q_X_l586_586395


namespace complex_series_sum_is_1004_l586_586243

noncomputable def complex_series_sum (x : ℂ) (h1: x ^ 2011 = 1) (h2: x ≠ 1) : ℂ :=
  ∑ k in finset.range (2010) + 1, (x ^ (2 * k)) / (x ^ k - 1)

theorem complex_series_sum_is_1004 (x : ℂ) (h1: x ^ 2011 = 1) (h2: x ≠ 1) : 
  complex_series_sum x h1 h2 = 1004 := by
  sorry

end complex_series_sum_is_1004_l586_586243


namespace problem_statement_l586_586009

noncomputable def a : ℝ := 2 * Real.log 0.99
noncomputable def b : ℝ := Real.log 0.98
noncomputable def c : ℝ := Real.sqrt 0.96 - 1

theorem problem_statement : a > b ∧ b > c := by
  sorry

end problem_statement_l586_586009


namespace abc_sumsq_correct_l586_586493

noncomputable def abc_sumsq (a b c d e f : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) (h6 : f > 0)
  (h7 : bcdef = \frac{a}{1}{2\frac{1}{2}}) (h8 : acdef = \frac{b}{1}{4\frac{1}{4}}) 
  (h9 : abdef = \frac{c}{1}{8\frac{1}{8}}) (h10 : abcef = \frac{d}{2}) 
  (h11 : abcdf = 4e) (h12 : abcde = 8f) : ℝ := 
(a^2 + b^2 + c^2 + d^2 + e^2 + f^2)

theorem abc_sumsq_correct (a b c d e f : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) (h6 : f > 0)
  (h7 : bcdef = \frac{a}{1}{2\frac{1}{2}}) (h8 : acdef = \frac{b}{1}{4\frac{1}{4}}) 
  (h9 : abdef = \frac{c}{1}{8\frac{1}{8}}) (h10 : abcef = \frac{d}{2}) 
  (h11 : abcdf = 4e) (h12 : abcde = 8f) : 
a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = \frac{119}{8} := 
sorry

end abc_sumsq_correct_l586_586493


namespace condition_necessary_but_not_sufficient_l586_586987

variable (a : ℝ)

theorem condition_necessary_but_not_sufficient (h : a^2 < 1) : (a < 1) ∧ (¬(a < 1 → a^2 < 1)) := sorry

end condition_necessary_but_not_sufficient_l586_586987


namespace probability_not_hearing_favorite_song_l586_586788

open Nat

def song_length (n : ℕ) : ℕ := 30 * (n + 1)

def favorite_song_length : ℕ := 3 * 60 + 30

def total_play_time : ℕ := 4 * 60 + 30

def total_songs : ℕ := 10

-- Probability that she hears the first 4 minutes, 30 seconds of music
-- without hearing every second of her favorite song
theorem probability_not_hearing_favorite_song :
  let total_ways := fact total_songs,
      favorable_ways := fact 9 + 2 * fact 8 in
  (1 - favorable_ways / total_ways) = 79 / 90 := sorry

end probability_not_hearing_favorite_song_l586_586788


namespace number_of_positive_divisors_of_360_is_24_l586_586906

theorem number_of_positive_divisors_of_360_is_24 :
  ∀ n : ℕ, n = 360 → n = 2^3 * 3^2 * 5^1 → 
  (n_factors : {p : ℕ × ℕ // p.1 ∈ [2, 3, 5] ∧ p.2 ∈ [3, 2, 1]} )
    → (n_factors.val.snd + 1).prod = 24 :=
by
  intro n hn h_factors
  rw hn at *
  have factors := h_factors.val
  cases factors with p_k q_l r_m
  have hpq : p_k.1 = 2 ∧ p_k.2 = 3 :=
    And.intro sorry sorry,
  have hqr : q_l.1 = 3 ∧ q_l.2 = 2 :=
    And.intro sorry sorry,
  have hr : r_m.1 = 5 ∧ r_m.2 = 1 :=
    And.intro sorry sorry,
  -- The proof would continue, but we'll skip it
  sorry

end number_of_positive_divisors_of_360_is_24_l586_586906


namespace largest_prime_factor_sum_of_four_digit_numbers_l586_586027

theorem largest_prime_factor_sum_of_four_digit_numbers 
  (a b c d : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) 
  (h3 : 1 ≤ b) (h4 : b ≤ 9) 
  (h5 : 1 ≤ c) (h6 : c ≤ 9) 
  (h7 : 1 ≤ d) (h8 : d ≤ 9) 
  (h_diff : (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d))
  : Nat.gcd 6666 (a + b + c + d) = 101 :=
sorry

end largest_prime_factor_sum_of_four_digit_numbers_l586_586027


namespace probability_of_interval_l586_586126

theorem probability_of_interval (a b x : ℝ) (h : 0 < a ∧ a < b ∧ 0 < x) : 
  (x < b) → (b = 1/2) → (x = 1/3) → (0 < x) → (x - 0) / (b - 0) = 2/3 := 
by 
  sorry

end probability_of_interval_l586_586126


namespace difference_of_sums_is_minus_18_l586_586458

theorem difference_of_sums_is_minus_18 : (3 + (-4) + (-5)) - (|3| + | -4 | + | -5 |) = -18 := by
  sorry

end difference_of_sums_is_minus_18_l586_586458


namespace opposite_of_3_is_neg3_l586_586669

theorem opposite_of_3_is_neg3 :
  ∃ y, 3 + y = 0 ∧ y = -3 :=
by
  use -3
  split
  · exact rfl
  · exact rfl

end opposite_of_3_is_neg3_l586_586669


namespace algebraic_expression_value_l586_586721

theorem algebraic_expression_value (x : ℝ) (h : x = Real.sqrt 19 - 1) : x^2 + 2 * x + 2 = 20 := by
  sorry

end algebraic_expression_value_l586_586721


namespace number_of_satisfying_elements_l586_586609

open Set

def f (x : ℤ) : ℤ := x^2 + 3*x + 2

def S : Finset ℤ := Finset.range 26

theorem number_of_satisfying_elements :
  (Finset.filter (λ s, f s % 6 = 0) S).card = 17 :=
by
  sorry

end number_of_satisfying_elements_l586_586609


namespace max_value_of_ratio_PA_PB_is_correct_l586_586192

noncomputable def max_ratio_PA_PB (A B : ℝ × ℝ) (r : ℝ) : ℝ :=
  let circle := {P : ℝ × ℝ | P.1^2 + P.2^2 = r^2}
  let PA := λ P : ℝ × ℝ, real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  let PB := λ P : ℝ × ℝ, real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)
  real.cSup (set.image (λ P, PB P / PA P) circle) (λ x hx, set.mem_image_of_mem _ hx)

theorem max_value_of_ratio_PA_PB_is_correct :
  max_ratio_PA_PB (0, -2) (1, -1) (real.sqrt 2) = 3 * real.sqrt 2 / 2 :=
by
  sorry

end max_value_of_ratio_PA_PB_is_correct_l586_586192


namespace part1_l586_586029

def A (a : ℝ) : Set ℝ := {x | 2 * a - 3 < x ∧ x < a + 1}
def B : Set ℝ := {x | (x + 1) * (x - 3) < 0}

theorem part1 (a : ℝ) (h : a = 0) : A a ∩ B = {x | -1 < x ∧ x < 1} :=
by
  -- Proof here
  sorry

end part1_l586_586029


namespace probability_of_interval_l586_586120

theorem probability_of_interval (a b x : ℝ) (h : 0 < a ∧ a < b ∧ 0 < x) : 
  (x < b) → (b = 1/2) → (x = 1/3) → (0 < x) → (x - 0) / (b - 0) = 2/3 := 
by 
  sorry

end probability_of_interval_l586_586120


namespace max_area_triangle_OAB_l586_586025

noncomputable theory

-- Define the basic setup and conditions
def ellipse (a b x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def parabola_focal_point : ℝ × ℝ := (-1, 0)
def eccentricity (a c : ℝ) : ℝ := c / a

-- Main theorem to prove
theorem max_area_triangle_OAB :
  ∃ (a b : ℝ), 
    (a > b ∧ b > 0 ∧
    (∃ c : ℝ, parabola_focal_point = (-c, 0) ∧ c = 1 ∧ eccentricity a c = (real.sqrt 2) / 2 ∧
    ellipse a b x y)) ∧
  (∃ (m : ℝ), m > 3/4 ∧
  ∃ (x1 x2 y1 y2 t : ℝ),
    let l := λ y : ℝ, t * y + m in
    (y1 + y2 = (-2 * t * m) / (t^2 + 2) ∧ 
     y1 * y2 = (m^2 - 2) / (t^2 + 2) ∧ 
     ∃ P_x P_y : ℝ, (P_x, P_y) = (5/4, 0) ∧
     let PA := (x1 - P_x, y1), PB := (x2 - P_x, y2) in
     (PA.1 * PB.1 + y1 * y2).constant) ∧
  sorry

-- Show that the maximum area of triangle OAB is (sqrt 2)/2
  (area_OAB := real.sqrt 2 / 2) :=
sorry

end max_area_triangle_OAB_l586_586025


namespace cannot_obtain_2015_stacks_of_1_token_l586_586280

theorem cannot_obtain_2015_stacks_of_1_token :
  ∀ (initial_tokens : ℕ), initial_tokens = 2014 →
  (∀ (operations : list (list ℕ → list ℕ)),
    (∀ op ∈ operations, 
      (∀ stacks, list.sum (op stacks) = list.sum stacks)) →
    false) :=
begin
  intros initial_tokens h_initial operations h_operations,
  have h_total : initial_tokens = 2014 := h_initial,
  cases h_initial,
  have h_goal : 2015 ≠ 2014 := by linarith,
  sorry
end

end cannot_obtain_2015_stacks_of_1_token_l586_586280


namespace maggie_earnings_proof_l586_586626

def rate_per_subscription : ℕ := 5
def subscriptions_to_parents : ℕ := 4
def subscriptions_to_grandfather : ℕ := 1
def subscriptions_to_neighbor1 : ℕ := 2
def subscriptions_to_neighbor2 : ℕ := 2 * subscriptions_to_neighbor1
def total_subscriptions : ℕ := subscriptions_to_parents + subscriptions_to_grandfather + subscriptions_to_neighbor1 + subscriptions_to_neighbor2
def total_earnings : ℕ := total_subscriptions * rate_per_subscription

theorem maggie_earnings_proof : total_earnings = 55 := by
  sorry

end maggie_earnings_proof_l586_586626


namespace tetromino_symmetry_and_distinctness_l586_586065

-- Define tetrominoes with rotational symmetry and distinctness properties
structure Tetromino where
  has_90_deg_rotation_symmetry : Bool
  has_180_deg_rotation_symmetry : Bool

-- Example tetrominoes as described in the problem
def square_tetromino : Tetromino :=
  { has_90_deg_rotation_symmetry := true, has_180_deg_rotation_symmetry := true }

def L_shaped_tetromino : Tetromino :=
  { has_90_deg_rotation_symmetry := false, has_180_deg_rotation_symmetry := false }

def T_shaped_tetromino : Tetromino :=
  { has_90_deg_rotation_symmetry := false, has_180_deg_rotation_symmetry := true }

-- The given set of tetrominoes
def tetrominoes : List Tetromino := [square_tetromino, square_tetromino, L_shaped_tetromino, T_shaped_tetromino, square_tetromino]

-- Helper function to count tetrominoes with specified properties
def count_with_symmetry (tetrominoes : List Tetromino) (rotation_90 : Bool) (rotation_180 : Bool) : Nat :=
  tetrominoes.count (λ t => t.has_90_deg_rotation_symmetry = rotation_90 || t.has_180_deg_rotation_symmetry = rotation_180)

-- Helper function to count distinct tetrominoes
def count_distinct_tetrominoes (tetrominoes : List Tetromino) : Nat :=
  -- Assuming a simplistic approach here, more sophisticated logic may be required for real scenarios
  tetrominoes.dedup.length

theorem tetromino_symmetry_and_distinctness :
  count_with_symmetry tetrominoes true true = 3 ∧ count_distinct_tetrominoes tetrominoes = 3 :=
by
  sorry

end tetromino_symmetry_and_distinctness_l586_586065


namespace probability_less_than_one_third_l586_586159

-- Define the intervals and the associated lengths
def interval_total : ℝ := 1 / 2
def interval_condition : ℝ := 1 / 3

-- Define the probability calculation
def probability : ℝ :=
  interval_condition / interval_total

-- Statement of the problem
theorem probability_less_than_one_third : probability = 2 / 3 := by
  sorry

end probability_less_than_one_third_l586_586159


namespace interval_probability_l586_586146

theorem interval_probability (x y z : ℝ) (h0x : 0 < x) (hy : y > 0) (hz : z > 0) (hx : x < y) (hy_mul_z_eq_half : y * z = 1 / 2) (hx_eq_third_y : x = 1 / 3 * y) :
  (x / y) = (2 / 3) :=
by
  -- Here you should provide the proof
  sorry

end interval_probability_l586_586146


namespace factorial_has_100_zeros_l586_586962

def count_factors_of_5 (n : ℕ) : ℕ :=
  n / 5 + n / 25 + n / 125 + n / 625 + n / 3125 + n / 15625 + n / 78125 + n / 390625 + n / 1953125 + n / 9765625

theorem factorial_has_100_zeros (n : ℕ) (h : n ∈ {405, 406, 407, 408, 409}) :
  count_factors_of_5 n = 100 :=
sorry

end factorial_has_100_zeros_l586_586962


namespace f_1987_is_3_l586_586076

noncomputable def f : ℕ → ℕ :=
sorry

axiom f_is_defined : ∀ x : ℕ, f x ≠ 0
axiom f_initial : f 1 = 3
axiom f_functional_equation : ∀ (a b : ℕ), f (a + b) = f a + f b - 2 * f (a * b) + 1

theorem f_1987_is_3 : f 1987 = 3 :=
by
  -- Here we would provide the mathematical proof
  sorry

end f_1987_is_3_l586_586076


namespace forestry_third_year_l586_586756

noncomputable def acres_planted_in_third_year :=
    let a := 10000 in
    let r := 1.2 in
    a * r^2

theorem forestry_third_year :
    acres_planted_in_third_year = 14400 :=
by
    sorry

end forestry_third_year_l586_586756


namespace subset_sum_inequality_l586_586972

variable (Z : Set ℂ) (n : ℕ) [fact (2 ≤ n)]
variable (m : ℕ) [fact (m ≤ n / 2)]

theorem subset_sum_inequality {Z : Set ℂ} (hZ : Z.card = n) :
  ∃ U ⊆ Z, U.card = m ∧ (∥(∑ z in U, z)∥ ≤ ∥(∑ z in (Z \ U), z)∥) := sorry

end subset_sum_inequality_l586_586972


namespace probability_less_than_one_third_l586_586095

theorem probability_less_than_one_third : 
  (∃ p : ℚ, p = (measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ)) (set.Ioo (0 : ℚ) (1 / 3 : ℚ)) / 
                  measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ))) ∧ p = 2 / 3) :=
by {
  -- The proof will be added here.
  sorry
}

end probability_less_than_one_third_l586_586095


namespace exists_z0_complex_l586_586014

theorem exists_z0_complex (n : ℕ) (a : Fin (n + 1) → ℂ) (ha : ∀ k, k ≤ n → |a k| = 1) :
  ∃ z0 : ℂ, |z0| = 1 ∧ |(∑ k in Finset.range (n + 1), a k * z0 ^ k)| ≥ sqrt (n + 3) :=
begin
  sorry
end

end exists_z0_complex_l586_586014


namespace hyperbola_eqn_correct_l586_586440

def parabola_focus : ℝ × ℝ := (1, 0)

def hyperbola_vertex := parabola_focus

def hyperbola_eccentricity : ℝ := 2

def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - (y^2 / 3) = 1

theorem hyperbola_eqn_correct (x y : ℝ) :
  hyperbola_equation x y :=
sorry

end hyperbola_eqn_correct_l586_586440


namespace triangle_area_ratio_l586_586200

theorem triangle_area_ratio {A B C : ℝ} {a b c : ℝ} 
  (h : 2 * Real.sin A * Real.cos (B - C) + Real.sin (2 * A) = 2 / 3) 
  (S1 : ℝ) (S2 : ℝ) :
  S1 / S2 = 1 / (3 * Real.pi) :=
sorry

end triangle_area_ratio_l586_586200


namespace find_x_perpendicular_l586_586064

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 3)
def b (x : ℝ) : ℝ × ℝ := (-3, x)

-- Define the condition that the dot product of vectors a and b is zero
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

-- Statement we need to prove
theorem find_x_perpendicular (x : ℝ) (h : perpendicular a (b x)) : x = -1 :=
by sorry

end find_x_perpendicular_l586_586064


namespace projection_a_on_b_is_minus_3_l586_586859

-- Constants representing vector a and vector b
def vec_a : ℝ × ℝ := (- (7 * Real.sqrt 2) / 2, Real.sqrt 2 / 2)
def vec_b : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)

-- The function that projects vector a onto vector b
def projection (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (b.1^2 + b.2^2))

theorem projection_a_on_b_is_minus_3 :
  projection vec_a vec_b = -3 :=
by
  sorry

end projection_a_on_b_is_minus_3_l586_586859


namespace interval_probability_l586_586145

theorem interval_probability (x y z : ℝ) (h0x : 0 < x) (hy : y > 0) (hz : z > 0) (hx : x < y) (hy_mul_z_eq_half : y * z = 1 / 2) (hx_eq_third_y : x = 1 / 3 * y) :
  (x / y) = (2 / 3) :=
by
  -- Here you should provide the proof
  sorry

end interval_probability_l586_586145


namespace quadratic_inequality_positive_l586_586002

theorem quadratic_inequality_positive (r : ℝ) : (r > 1) → ∀ x : ℝ, ((r^2 - 1) * x^2 + 2 * (r - 1) * x + 1) > 0 :=
by 
  assume h : r > 1
  sorry

end quadratic_inequality_positive_l586_586002


namespace line_equation_correct_l586_586484

-- Definitions for the conditions
def point := ℝ × ℝ
def vector := ℝ × ℝ

-- Given the line has a direction vector and passes through a point
def line_has_direction_vector (l : point → Prop) (v : vector) : Prop :=
  ∀ p₁ p₂ : point, l p₁ → l p₂ → (p₂.1 - p₁.1, p₂.2 - p₁.2) = v

def line_passes_through_point (l : point → Prop) (p : point) : Prop :=
  l p

-- The line equation in point-direction form
def line_equation (x y : ℝ) : Prop :=
  (x - 1) / 2 = y / -3

-- Main statement
theorem line_equation_correct :
  ∃ l : point → Prop, 
    line_has_direction_vector l (2, -3) ∧
    line_passes_through_point l (1, 0) ∧
    ∀ x y, l (x, y) ↔ line_equation x y := 
sorry

end line_equation_correct_l586_586484


namespace find_A_l586_586722

theorem find_A (A B : ℕ) (A_digit : A < 10) (B_digit : B < 10) :
  let fourteenA := 100 * 1 + 10 * 4 + A
  let Bseventy3 := 100 * B + 70 + 3
  fourteenA + Bseventy3 = 418 → A = 5 :=
by
  sorry

end find_A_l586_586722


namespace probability_of_two_black_balls_relationship_x_y_l586_586572

-- Conditions
def initial_black_balls : ℕ := 3
def initial_white_balls : ℕ := 2

variable (x y : ℕ)

-- Given relationship
def total_white_balls := x + 2
def total_black_balls := y + 3
def white_ball_probability := (total_white_balls x) / (total_white_balls x + total_black_balls y + 5)

-- Proof goals
theorem probability_of_two_black_balls :
  (3 / 5) * (2 / 4) = 3 / 10 := by sorry

theorem relationship_x_y :
  white_ball_probability x y = 1 / 3 → y = 2 * x + 1 := by sorry

end probability_of_two_black_balls_relationship_x_y_l586_586572


namespace range_of_a_l586_586293

-- Define a function that is decreasing on \(\mathbb{R}\)
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f y ≤ f x

-- Statement translating the problem conditions and conclusion
theorem range_of_a 
  (f : ℝ → ℝ) 
  (h_decreasing : is_decreasing f) 
  (h : f (3 * a) < f (-2 * a + 10)) 
  : a > 2 :=
begin
  sorry
end

end range_of_a_l586_586293


namespace ryegrass_percentage_l586_586640

theorem ryegrass_percentage (x_ryegrass_percent : ℝ) (y_ryegrass_percent : ℝ) (mixture_x_percent : ℝ)
  (hx : x_ryegrass_percent = 0.40)
  (hy : y_ryegrass_percent = 0.25)
  (hmx : mixture_x_percent = 0.8667) :
  (x_ryegrass_percent * mixture_x_percent + y_ryegrass_percent * (1 - mixture_x_percent)) * 100 = 38 :=
by
  sorry

end ryegrass_percentage_l586_586640


namespace probability_of_interval_l586_586136

theorem probability_of_interval {x : ℝ} (h₀ : 0 < x ∧ x < 1/2) :
    Pr (λ x, x < 1/3) = 2/3 :=
sorry

end probability_of_interval_l586_586136


namespace original_number_l586_586345

theorem original_number (x y : ℕ) (h1 : x + y = 859560) (h2 : y = 859560 % 456) : x = 859376 ∧ 456 ∣ x :=
by
  sorry

end original_number_l586_586345


namespace parabola_x_intercepts_l586_586905

theorem parabola_x_intercepts : 
  let eqn := fun (y : ℝ) => -3 * y^2 + 4 * y + 2
  in (∃ y : ℝ, eqn y = 0) → (∀ x, ∃! y, eqn y = x) :=
by
  sorry

end parabola_x_intercepts_l586_586905


namespace find_value_of_expression_l586_586795

noncomputable def x1 : ℝ := sorry
noncomputable def x2 : ℝ := sorry
noncomputable def x3 : ℝ := sorry
noncomputable def x4 : ℝ := sorry
noncomputable def x5 : ℝ := sorry
noncomputable def x6 : ℝ := sorry

def condition1 : Prop := x1 + 3 * x2 + 5 * x3 + 7 * x4 + 9 * x5 + 11 * x6 = 2
def condition2 : Prop := 3 * x1 + 5 * x2 + 7 * x3 + 9 * x4 + 11 * x5 + 13 * x6 = 15
def condition3 : Prop := 5 * x1 + 7 * x2 + 9 * x3 + 11 * x4 + 13 * x5 + 15 * x6 = 52

theorem find_value_of_expression : condition1 → condition2 → condition3 → (7 * x1 + 9 * x2 + 11 * x3 + 13 * x4 + 15 * x5 + 17 * x6 = 65) :=
by
  intros h1 h2 h3
  sorry

end find_value_of_expression_l586_586795


namespace Matias_sales_l586_586999

def books_sold (Tuesday Wednesday Thursday : Nat) : Prop :=
  Tuesday = 7 ∧ 
  Wednesday = 3 * Tuesday ∧ 
  Thursday = 3 * Wednesday ∧ 
  Tuesday + Wednesday + Thursday = 91

theorem Matias_sales
  (Tuesday Wednesday Thursday : Nat) :
  books_sold Tuesday Wednesday Thursday := by
  sorry

end Matias_sales_l586_586999


namespace part_one_part_two_l586_586032

noncomputable def problem_conditions (θ : ℝ) : Prop :=
  let sin_theta := Real.sin θ
  let cos_theta := Real.cos θ
  ∃ m : ℝ, (∀ x : ℝ, x^2 - (Real.sqrt 3 - 1) * x + m = 0 → (x = sin_theta ∨ x = cos_theta))

theorem part_one (θ: ℝ) (h: problem_conditions θ) : 
  let sin_theta := Real.sin θ
  let cos_theta := Real.cos θ
  let m := sin_theta * cos_theta
  m = (3 - 2 * Real.sqrt 3) / 2 :=
sorry

theorem part_two (θ: ℝ) (h: problem_conditions θ) : 
  let sin_theta := Real.sin θ
  let cos_theta := Real.cos θ
  let tan_theta := sin_theta / cos_theta
  (cos_theta - sin_theta * tan_theta) / (1 - tan_theta) = Real.sqrt 3 - 1 :=
sorry

end part_one_part_two_l586_586032


namespace greatest_number_of_market_women_l586_586760

def shillings_to_pence (s: ℕ) : ℕ := s * 12
def farthings_to_pence (f: ℚ) : ℚ := f / 4
def total_pence (sh: ℕ) (far: ℚ) : ℚ := shillings_to_pence sh + farthings_to_pence far
def pence_to_farthings (p: ℚ) : ℚ := p * 4
def divisors (n: ℕ) : list ℕ := (list.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0)

theorem greatest_number_of_market_women : (pence_to_farthings (total_pence 2 (9/4))).natAbs = 105 → (divisors 105).length = 8 := by
  intros h
  sorry

end greatest_number_of_market_women_l586_586760


namespace fifth_rectangle_more_tiles_than_fourth_l586_586768

theorem fifth_rectangle_more_tiles_than_fourth :
  let n := 5
  let L := λ (n : ℕ), (2 * n - 1)
  let number_of_tiles (n : ℕ) := (L n) * (L n)
  (number_of_tiles n) - (number_of_tiles (n-1)) = 32 :=
by
  let n := 5
  let L := λ (n : ℕ), (2 * n - 1)
  let number_of_tiles := λ (n : ℕ), (L n) * (L n)
  calc
    (number_of_tiles n) - (number_of_tiles (n-1)) = 32 : sorry

end fifth_rectangle_more_tiles_than_fourth_l586_586768


namespace green_tractor_price_l586_586708

variable (S : ℕ) (r g R G : ℕ) (R_price G_price : ℝ)
variable (h1 : S = 7000)
variable (h2 : r = 2)
variable (h3 : g = 3)
variable (h4 : R_price = 20000)
variable (h5 : G_price = 5000)
variable (h6 : ∀ (x : ℝ), Tobias_earning_red : ℝ) (h7 : ∀ (y : ℝ), Tobias_earning_green : ℝ)

def earning_percentage_red : ℝ := 0.10
def earning_percentage_green : ℝ := 0.20

theorem green_tractor_price :
  S = Tobias_earning_red + Tobias_earning_green →
  Tobias_earning_red = (earning_percentage_red * R_price) * r →
  Tobias_earning_green = (earning_percentage_green * G_price) * g →
  G_price = 5000 :=
  sorry

end green_tractor_price_l586_586708


namespace kids_difference_l586_586597

def kidsPlayedOnMonday : Nat := 11
def kidsPlayedOnTuesday : Nat := 12

theorem kids_difference :
  kidsPlayedOnTuesday - kidsPlayedOnMonday = 1 := by
  sorry

end kids_difference_l586_586597


namespace missing_number_is_6630_l586_586734

theorem missing_number_is_6630 (x : ℕ) (h : 815472 / x = 123) : x = 6630 :=
by {
  sorry
}

end missing_number_is_6630_l586_586734


namespace arithmetic_sequence_problem_l586_586578

variable {a : ℕ → ℕ}

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n m : ℕ, a (n+1) = a n + d

theorem arithmetic_sequence_problem
  (h_arith : is_arithmetic_sequence a)
  (h1 : a 1 + a 2 + a 3 = 32)
  (h2 : a 11 + a 12 + a 13 = 118) :
  a 4 + a 10 = 50 :=
sorry

end arithmetic_sequence_problem_l586_586578


namespace total_books_count_l586_586690

theorem total_books_count (books_read : ℕ) (books_unread : ℕ) (h1 : books_read = 13) (h2 : books_unread = 8) : books_read + books_unread = 21 := 
by
  -- Proof omitted
  sorry

end total_books_count_l586_586690


namespace maggie_earnings_proof_l586_586625

def rate_per_subscription : ℕ := 5
def subscriptions_to_parents : ℕ := 4
def subscriptions_to_grandfather : ℕ := 1
def subscriptions_to_neighbor1 : ℕ := 2
def subscriptions_to_neighbor2 : ℕ := 2 * subscriptions_to_neighbor1
def total_subscriptions : ℕ := subscriptions_to_parents + subscriptions_to_grandfather + subscriptions_to_neighbor1 + subscriptions_to_neighbor2
def total_earnings : ℕ := total_subscriptions * rate_per_subscription

theorem maggie_earnings_proof : total_earnings = 55 := by
  sorry

end maggie_earnings_proof_l586_586625


namespace find_k_l586_586902

-- Define the vectors a and b, and the perpendicular condition
def vector_a (k : ℝ) : ℝ × ℝ := (2 * k - 3, -6)
def vector_b : ℝ × ℝ := (2, 1)

-- Define perpendicularity condition
def perpendicular {α : Type} [Add α] [Mul α] :=
  λ (u v : α × α), u.1 * v.1 + u.2 * v.2 = 0

-- Main theorem statement to prove k = 3
theorem find_k : ∃ k : ℝ, perpendicular (vector_a k) vector_b ∧ k = 3 :=
by
  sorry

end find_k_l586_586902


namespace items_per_crate_l586_586835

theorem items_per_crate (novels comics documentaries albums crates : ℕ)
  (h_novels : novels = 145)
  (h_comics : comics = 271)
  (h_documentaries : documentaries = 419)
  (h_albums : albums = 209)
  (h_crates : crates = 116) :
  (novels + comics + documentaries + albums) / crates = 9 :=
by
  rw [h_novels, h_comics, h_documentaries, h_albums, h_crates]
  norm_num
  sorry

end items_per_crate_l586_586835


namespace new_person_weight_l586_586735

-- Define the total number of persons and their average weight increase
def num_persons : ℕ := 9
def avg_increase : ℝ := 1.5

-- Define the weight of the person being replaced
def weight_of_replaced_person : ℝ := 65

-- Define the total increase in weight
def total_increase_in_weight : ℝ := num_persons * avg_increase

-- Define the weight of the new person
def weight_of_new_person : ℝ := weight_of_replaced_person + total_increase_in_weight

-- Theorem to prove the weight of the new person is 78.5 kg
theorem new_person_weight : weight_of_new_person = 78.5 := by
  -- proof is omitted
  sorry

end new_person_weight_l586_586735


namespace perfect_square_factors_of_420_l586_586066

theorem perfect_square_factors_of_420 : 
  let p := 420 
  ∧ (∀ d, d ∣ p ↔ (∃ a b c d, d = 2^a * 3^b * 5^c * 7^d 
                      ∧ 0 ≤ a ≤ 2 
                      ∧ 0 ≤ b ≤ 1 
                      ∧ 0 ≤ c ≤ 1 
                      ∧ 0 ≤ d ≤ 1))
  ∧ (∀ d, d ∣ p → (∀ e, d = e * e)) 
  → fintype.card {d // d ∣ p ∧ is_square d} = 2 :=
by sorry

end perfect_square_factors_of_420_l586_586066


namespace distinct_paths_in_grid_l586_586187

/-- In a 6x5 grid, there are exactly 462 distinct paths from the bottom-left corner to the top-right corner, when only moving right or up, with exactly 6 right steps and 5 up steps. -/
theorem distinct_paths_in_grid : @nat.choose 11 5 = 462 := 
by sorry

end distinct_paths_in_grid_l586_586187


namespace part_I_part_II_l586_586863

open Real

-- Definitions of the functions f and g
def f (x : ℝ) (m : ℝ) : ℝ := x * log x + m * x
def g (x : ℝ) (a : ℝ) : ℝ := -x^2 + a * x - 3

-- Constants we are trying to solve for
variable {m a : ℝ}

-- Hypothesized conditions and results
theorem part_I (h_m : ∀ x : ℝ, 1 < x → deriv (λ x, f x m) x ≥ 0) : m ≤ -1 := 
sorry

theorem part_II (m_eq_zero : m = 0) (h_cond : ∀ x : ℝ, 0 < x → 2 * f x 0 ≥ g x a) : a ≤ 4 :=
sorry

#check part_I
#check part_II

end part_I_part_II_l586_586863


namespace middle_term_and_sum_of_odd_coeffs_l586_586251

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem middle_term_and_sum_of_odd_coeffs :
  (∀ x : ℕ, x ∈ [0, 8] →
    (x + 2) ^ 8 = binomial 8 x * 2 ^ (8 - x) * x ^ x) →
  (let a0 := (2 ^ 8) in
   let a1 := (8 * 2 ^ 7) in
   let a2 := Nat.div (8 * 7 * 2 ^ 6) 2 in
   a0, a1, a2 are an arithmetic sequence) →
  (let t5 := binomial 8 4 * 2 ^ 4 in
   t5 = 1120) ∧
  ((3 ^ 8 - 1) / 2 = 3280) :=
begin
  sorry
end

end middle_term_and_sum_of_odd_coeffs_l586_586251


namespace problem_solution_l586_586398

def is_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x < f y

noncomputable def option_A := fun x : ℝ => (1/3)^x
noncomputable def option_B := fun x : ℝ => -2 * x + 5
noncomputable def option_C := fun x : ℝ => Real.log x
noncomputable def option_D := fun x : ℝ => 3 / x

theorem problem_solution :
  is_increasing option_A (set.Ioi 0) = false ∧
  is_increasing option_B (set.Ioi 0) = false ∧
  is_increasing option_C (set.Ioi 0) = true ∧
  is_increasing option_D (set.Ioi 0) = false := 
  sorry

end problem_solution_l586_586398


namespace equilateral_triangle_vertex_distance_l586_586712

noncomputable def distance_vertex_to_center (l r : ℝ) : ℝ :=
  Real.sqrt (r^2 + (l^2 / 4))

theorem equilateral_triangle_vertex_distance
  (l r : ℝ)
  (h1 : l > 0)
  (h2 : r > 0) :
  distance_vertex_to_center l r = Real.sqrt (r^2 + (l^2 / 4)) :=
sorry

end equilateral_triangle_vertex_distance_l586_586712


namespace robot_trajectory_no_intersection_l586_586770

noncomputable def parabola_equation (x y : ℝ) : Prop := y^2 = 4 * x
noncomputable def line_equation (x y k : ℝ) : Prop := y = k * (x + 1)

theorem robot_trajectory_no_intersection (k : ℝ) :
  (∀ x y : ℝ, parabola_equation x y → ¬ line_equation x y k) →
  (k > 1 ∨ k < -1) :=
by
  sorry

end robot_trajectory_no_intersection_l586_586770


namespace car_second_hour_speed_l586_586679

theorem car_second_hour_speed (x : ℝ) 
  (first_hour_speed : ℝ := 20)
  (average_speed : ℝ := 40) 
  (total_time : ℝ := 2)
  (total_distance : ℝ := first_hour_speed + x) 
  : total_distance / total_time = average_speed → x = 60 :=
by
  intro h
  sorry

end car_second_hour_speed_l586_586679


namespace max_value_f_1994_values_n_f_1994_l586_586740

-- Defining the function f based on the given conditions
def f : ℕ → ℕ
| 1       := 1
| (2 * n) := f n
| (2 * n + 1) := f (2 * n) + 1

theorem max_value_f_1994 : (∃ n : ℕ, 1 ≤ n ∧ n ≤ 1994 ∧ f n = 10) ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 1994 → f n ≤ 10) :=
by
  sorry

theorem values_n_f_1994 : {n : ℕ | 1 ≤ n ∧ n ≤ 1994 ∧ f n = 10} = {1023, 1535, 1791, 1919, 1983} :=
by
  sorry

end max_value_f_1994_values_n_f_1994_l586_586740


namespace chenny_friends_l586_586425

noncomputable def num_friends (initial_candies add_candies candies_per_friend) : ℕ :=
  (initial_candies + add_candies) / candies_per_friend

theorem chenny_friends :
  num_friends 10 4 2 = 7 :=
by
  sorry

end chenny_friends_l586_586425


namespace smallest_positive_number_l586_586539

/-- Given the expressions in the list, we need to identify the smallest positive number. -/
theorem smallest_positive_number :
  let a := 8 - 3 * Real.sqrt 10
  let b := 3 * Real.sqrt 10 - 8
  let c := 23 - 6 * Real.sqrt 15
  let d := 58 - 12 * Real.sqrt 30
  let e := 12 * Real.sqrt 30 - 58
  (min {x : ℝ | x > 0 ∧ (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e)} = b) :=
by
  let a := 8 - 3 * Real.sqrt 10
  let b := 3 * Real.sqrt 10 - 8
  let c := 23 - 6 * Real.sqrt 15
  let d := 58 - 12 * Real.sqrt 30
  let e := 12 * Real.sqrt 30 - 58
  sorry

end smallest_positive_number_l586_586539


namespace max_distinct_numbers_in_grid_l586_586632

/-- 
Given an infinite grid where each cell contains a natural number and 
the number in each cell is equal to the number of its adjacent cells 
sharing a common vertex having the same number, prove that the maximum 
number of distinct natural numbers in such a grid is 5.
-/
theorem max_distinct_numbers_in_grid : 
  ∀ (grid : ℕ → ℕ → ℕ),
  (∀ (x y : ℕ), grid x y = (finset.univ.filter (λ p, grid p.1 p.2 = grid x y)).card) → 
  ∃ S : finset ℕ, S.card = 5 ∧ ∀ (x y : ℕ), grid x y ∈ S :=
sorry

end max_distinct_numbers_in_grid_l586_586632


namespace conner_day3_tie_l586_586283

def sydney_initial := 837
def conner_initial := 723

def sydney_day1 := 4
def conner_day1 := 8 * sydney_day1

def sydney_day2 := 0
def conner_day2 := 123

def sydney_day3 := 2 * conner_day1

def sydney_total := sydney_initial + sydney_day1 + sydney_day2 + sydney_day3
def conner_total_day2 := conner_initial + conner_day1 + conner_day2

def conner_day3_required := sydney_total - conner_total_day2

theorem conner_day3_tie (c : ℕ) :
  sydney_initial + sydney_day1 + sydney_day2 + sydney_day3 = 
  conner_initial + conner_day1 + conner_day2 + c 
  ↔ c = 27 := by
  intro h
  sorry

end conner_day3_tie_l586_586283


namespace probability_of_interval_l586_586123

theorem probability_of_interval (a b x : ℝ) (h : 0 < a ∧ a < b ∧ 0 < x) : 
  (x < b) → (b = 1/2) → (x = 1/3) → (0 < x) → (x - 0) / (b - 0) = 2/3 := 
by 
  sorry

end probability_of_interval_l586_586123


namespace find_initial_quarters_l586_586621

-- Define the initial number of dimes, nickels, and quarters (unknown)
def initial_dimes : ℕ := 2
def initial_nickels : ℕ := 5
def initial_quarters (Q : ℕ) := Q

-- Define the additional coins given by Linda’s mother
def additional_dimes : ℕ := 2
def additional_quarters : ℕ := 10
def additional_nickels : ℕ := 2 * initial_nickels

-- Define the total number of each type of coin after Linda receives the additional coins
def total_dimes : ℕ := initial_dimes + additional_dimes
def total_quarters (Q : ℕ) : ℕ := additional_quarters + initial_quarters Q
def total_nickels : ℕ := initial_nickels + additional_nickels

-- Define the total number of coins
def total_coins (Q : ℕ) : ℕ := total_dimes + total_quarters Q + total_nickels

theorem find_initial_quarters : ∃ Q : ℕ, total_coins Q = 35 ∧ Q = 6 := by
  -- Provide the corresponding proof here
  sorry

end find_initial_quarters_l586_586621


namespace product_of_more_than_2k_prime_factors_l586_586004

theorem product_of_more_than_2k_prime_factors
  {k : ℕ} (h_k : 5 < k) (p : ℕ → ℕ) (h_prime : ∀ i, 1 ≤ i → i ≤ k → Nat.Prime (p i))
  (h_p_seq : p 1 = 2 ∧ p 2 = 3 ∧ p 3 = 5 ∧ ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → p i < p j) :
  ∃ S : ℕ, (∑ i in (Finset.powerset (Finset.range k)).map (λ s, ∏ j in s, p (j+1))) + 1
           ∣ (∏ i in Finset.range k, (p (i+1) + 1)) ∧
           (Nat.factors ((∑ i in (Finset.powerset (Finset.range k)).map (λ s, ∏ j in s, p (j+1))) + 1)).length > 2 * k :=
sorry

end product_of_more_than_2k_prime_factors_l586_586004


namespace quadrilateral_area_proof_l586_586695

noncomputable def area_quadrilateral_OMCD (area_parallelogram : ℝ) (midpoint_BC : Prop)
  (line_intersects_BD_at_O : Prop) : ℝ :=
if area_parallelogram = 1 ∧ midpoint_BC ∧ line_intersects_BD_at_O then
  5 / 12
else
  0  -- degenerate case or error

theorem quadrilateral_area_proof : 
  ∀ (A B C D M O : Type) [geometry A B C D M O],
  (area_parallelogram A B C D = 1) →
  (is_midpoint B C M) →
  (intersects (line_through A) (BD_diagonal B D) = O) →
  area_quadrilateral_OMCD 1 (is_midpoint B C M) (intersects (line_through A) (BD_diagonal B D) = O) = 5 / 12 :=
by
  sorry

end quadrilateral_area_proof_l586_586695


namespace tables_can_hold_guests_l586_586332

theorem tables_can_hold_guests (tables guests : ℕ) (h_tables : tables = 252) (h_guests : guests = 1008) :
  guests / tables = 4 :=
by
  rw [h_tables, h_guests]
  norm_num -- divisibility check and computation

end tables_can_hold_guests_l586_586332


namespace calculate_shaded_area_l586_586801

def point := (ℝ × ℝ)

def line_through_points (p1 p2 : point) : ℝ → ℝ := 
  let m := (p2.2 - p1.2) / (p2.1 - p1.1) -- slope
  λ x, m * x + (p1.2 - m * p1.1)          -- y = mx + b

def area_of_shaded_region (l1 l2 l3 : ℝ → ℝ) : ℝ :=
  let x_intersect := 45 / 8
  (3 * x_intersect + (1 / 18) * (x_intersect * x_intersect))

theorem calculate_shaded_area :
  let l1 := line_through_points (0, 3) (9, 2)
  let l2 := line_through_points (2, 6) (9, -1)
  let l3 := line_through_points (0, 6) (6, 0)
  area_of_shaded_region l1 l2 l3 = 21465 / 1152 :=
by sorry

end calculate_shaded_area_l586_586801


namespace circle_intersection_range_l586_586554

noncomputable def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 = 25
noncomputable def circle2_eq (x y r : ℝ) : Prop := (x - 7)^2 + y^2 = r^2

theorem circle_intersection_range (r : ℝ) (h : r > 0) :
  (∃ x y : ℝ, circle1_eq x y ∧ circle2_eq x y r) ↔ 2 < r ∧ r < 12 :=
sorry

end circle_intersection_range_l586_586554


namespace f_has_two_zeros_iff_l586_586885

open Real

noncomputable def f (x a : ℝ) : ℝ := (x - 2) * exp x + a * (x - 1)^2

theorem f_has_two_zeros_iff (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0) ↔ 0 < a :=
sorry

end f_has_two_zeros_iff_l586_586885


namespace coloring_15_segments_impossible_l586_586202

theorem coloring_15_segments_impossible :
  ¬ ∃ (colors : Fin 15 → Fin 3) (adj : Fin 15 → Fin 2),
    ∀ i j, adj i = adj j → i ≠ j → colors i ≠ colors j :=
by
  sorry

end coloring_15_segments_impossible_l586_586202


namespace f_f1_eq_1_solution_set_of_f_gt_2_l586_586616

def f : ℝ → ℝ := λ x, if x < 2 then 2 * Real.exp(x - 1) else Real.logb 3 (x^2 - 1)

theorem f_f1_eq_1 : f (f 1) = 1 := by
  sorry

theorem solution_set_of_f_gt_2 : {x : ℝ | f x > 2} = (Set.Ioo 1 2 ∪ Set.Ioi (Real.sqrt 10)) := by
  sorry

end f_f1_eq_1_solution_set_of_f_gt_2_l586_586616


namespace rectangle_sections_max_5_lines_l586_586585

theorem rectangle_sections_max_5_lines (R : Type) (rectangle : R) (MN : R → Prop) :
  ∃ max_sections : ℕ, 
  (∀ n : ℕ, n ≤ 5 → 
    (∃ lines : fin n.succ → (R → Prop), 
    ∀ m : fin (n + 1), partitions (lines m) rectangle) → 
    partitions (max_sections) rectangle) 
  ∧ max_sections = 16 := 
sorry

end rectangle_sections_max_5_lines_l586_586585


namespace probability_less_than_one_third_l586_586132

def interval := set.Ioo (0 : ℝ) (1 / 2)

def desired_interval := set.Ioo (0 : ℝ) (1 / 3)

theorem probability_less_than_one_third :
  (∃ p : ℝ, p = (volume desired_interval / volume interval) ∧ p = 2 / 3) :=
  sorry

end probability_less_than_one_third_l586_586132


namespace main_theorem_l586_586945

variable (students : Type) [Fintype students] [DecidableEq students]
variable (friends : students → students → Prop) [DecidableRel friends]

axiom friends_symmetric : symmetric friends
axiom friends_irreflexive : irreflexive friends
axiom every_student_has_friend : ∀ s : students, ∃ t : students, friends s t

def exists_group_with_friends (n : ℕ) : Prop :=
  ∃ (S : Finset students), S.card = n ∧ ∀ s ∈ S, ∃ t ∈ S, friends s t

theorem main_theorem : ∀ n, 1 < n ∧ n < 101 → exists_group_with_friends students friends n :=
by
  sorry

end main_theorem_l586_586945


namespace great_grandson_age_is_36_l586_586377

-- Define the problem conditions and the required proof
theorem great_grandson_age_is_36 :
  ∃ n : ℕ, (∃ k : ℕ, k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧  2 * k * 111 = n * (n + 1)) ∧ n = 36 :=
by
  sorry

end great_grandson_age_is_36_l586_586377


namespace coordinates_A_100_l586_586407

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem coordinates_A_100 : 
  (let A_n (n : ℕ) := (sum_first_n n, sum_first_n n) 
    in A_n 100) = (5050, 5050) :=
by
  sorry

end coordinates_A_100_l586_586407


namespace smallest_prime_factor_in_C_l586_586641

def setC := {34, 35, 37, 41, 43}

def smallest_prime_factor (n : ℕ) : ℕ :=
  if n = 34 then 2
  else if n = 35 then 5
  else if n = 37 then 37
  else if n = 41 then 41
  else if n = 43 then 43
  else n

theorem smallest_prime_factor_in_C : ∃ n ∈ setC, smallest_prime_factor n = 2 := 
by {
  sorry
}

end smallest_prime_factor_in_C_l586_586641


namespace large_painting_area_l586_586811

theorem large_painting_area :
  ∃ (large_painting : ℕ),
  (3 * (6 * 6) + 4 * (2 * 3) + large_painting = 282) → large_painting = 150 := by
  sorry

end large_painting_area_l586_586811


namespace trash_can_prices_and_minimum_A_can_purchase_l586_586699

theorem trash_can_prices_and_minimum_A_can_purchase 
  (x y : ℕ) 
  (h₁ : 3 * x + 4 * y = 580)
  (h₂ : 6 * x + 5 * y = 860)
  (total_trash_cans : ℕ)
  (total_cost : ℕ)
  (cond₃ : total_trash_cans = 200)
  (cond₄ : 60 * (total_trash_cans - x) + 100 * x ≤ 15000) : 
  x = 60 ∧ y = 100 ∧ x ≥ 125 := 
sorry

end trash_can_prices_and_minimum_A_can_purchase_l586_586699


namespace regular_polygon_exterior_angle_l586_586170

theorem regular_polygon_exterior_angle (n : ℕ) (h : 72 = 360 / n) : n = 5 :=
by
  sorry

end regular_polygon_exterior_angle_l586_586170


namespace unique_ab_for_interval_condition_l586_586001

theorem unique_ab_for_interval_condition : 
  ∃! (a b : ℝ), (∀ x, (0 ≤ x ∧ x ≤ 1) → |x^2 - a * x - b| ≤ 1 / 8) ∧ a = 1 ∧ b = -1 / 8 := by
  sorry

end unique_ab_for_interval_condition_l586_586001


namespace orthocenter_fixed_l586_586866

-- Given a circle ω with center O, and two distinct points A and C on it.
variable {ω : Type*} [MetricSpace ω] {O A C P : ω}
variable [MetricSpace.ball ω]

-- Assume A and C are distinct points on the circle ω and P is an arbitrary point on the circle ω.
variable (hA : A ∈ MetricSpace.ball ω O 1)
variable (hC : C ∈ MetricSpace.ball ω O 1)
variable (hAC : A ≠ C)
variable (hP : P ∈ MetricSpace.ball ω O 1)

-- Let X be the midpoint of AP and Y be the midpoint of CP.
def midpoint (p1 p2 : ω) : ω := MetricSpace.midpoint p1 p2

noncomputable def X := midpoint A P
noncomputable def Y := midpoint C P

-- Define the orthocenter of triangle OXY.
def orthocenter (O X Y : ω) : ω := sorry

-- Define the midpoint of AC.
noncomputable def midpoint_AC := midpoint A C

-- Theorem: The orthocenter H of triangle OXY is always the midpoint of AC.
theorem orthocenter_fixed (O A C P : ω) (hA : A ∈ MetricSpace.ball ω O 1) (hC : C ∈ MetricSpace.ball ω O 1)
  (hAC : A ≠ C) (hP : P ∈ MetricSpace.ball ω O 1) :
  orthocenter O (midpoint A P) (midpoint C P) = midpoint A C := 
sorry

end orthocenter_fixed_l586_586866


namespace probability_merlin_dismissed_l586_586232

-- Define the conditions
variables (p : ℝ) (q : ℝ) (hpq : p + q = 1) (hp_pos : 0 < p) (hq_pos : 0 < q)

/--
Given advisor Merlin is equally likely to dismiss as Percival
since they are equally likely to give the correct answer independently,
prove that the probability of Merlin being dismissed is \( \frac{1}{2} \).
-/
theorem probability_merlin_dismissed : (1/2 : ℝ) = 1/2 :=
by 
  sorry

end probability_merlin_dismissed_l586_586232


namespace series_sum_eq_l586_586327

open Nat

theorem series_sum_eq (n : ℕ) (hn : n > 0) : 
  (∑ i in range n, 1 / ((i + 1) * (i + 2))) = n / (n + 1) := 
sorry

end series_sum_eq_l586_586327


namespace Maggie_earnings_l586_586628

theorem Maggie_earnings
    (price_per_subscription : ℕ)
    (subscriptions_parents : ℕ)
    (subscriptions_grandfather : ℕ)
    (subscriptions_nextdoor : ℕ)
    (subscriptions_another : ℕ)
    (total_subscriptions : ℕ)
    (total_earnings : ℕ) :
    subscriptions_parents = 4 →
    subscriptions_grandfather = 1 →
    subscriptions_nextdoor = 2 →
    subscriptions_another = 2 * subscriptions_nextdoor →
    total_subscriptions = subscriptions_parents + subscriptions_grandfather + subscriptions_nextdoor + subscriptions_another →
    price_per_subscription = 5 →
    total_earnings = price_per_subscription * total_subscriptions →
    total_earnings = 55 :=
by
  intros
  sorry

end Maggie_earnings_l586_586628


namespace find_N_l586_586455

noncomputable def N (p q : ℕ) : ℕ := p * q

noncomputable def φ (N : ℕ) : ℕ := (N - 1)

def twin_primes (p q : ℕ) : Prop := p.prime ∧ q.prime ∧ (abs (p - q) = 2)

theorem find_N (p q : ℕ) (hpq : twin_primes p q) (hφ : φ (N p q) = 120) : N p q = 143 := 
sorry

end find_N_l586_586455


namespace find_n_l586_586620

open Nat

def count_trailing_zeros (n : ℕ) : ℕ :=
  let rec ct_aux : ℕ → ℕ → ℕ
    | 0, _ => 0
    | num, base =>
      let p := num / base
      in
      if p = 0 then 0 else p + ct_aux p base

in ct_aux n 5

theorem find_n (n k : ℕ) (hn : n > 4)
  (h1 : count_trailing_zeros n = k)
  (h2 : count_trailing_zeros (2 * n) = 3 * k) :
  n = 8 ∨ n = 9 ∨ n = 13 ∨ n = 14 :=
sorry

end find_n_l586_586620


namespace probability_less_than_one_third_l586_586094

theorem probability_less_than_one_third : 
  (∃ p : ℚ, p = (measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ)) (set.Ioo (0 : ℚ) (1 / 3 : ℚ)) / 
                  measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ))) ∧ p = 2 / 3) :=
by {
  -- The proof will be added here.
  sorry
}

end probability_less_than_one_third_l586_586094


namespace line_AM_bisects_KA_0_l586_586370

variables {ABC : Triangle} {ω ω' : Circle} {A B C K A_0 M : Point}

-- Define the conditions
variable (h1 : ω.is_inscribed_in_triangle ABC ∧ ω.is_tangent_at BC K)
variable (h2 : ω'.is_symmetrical_to ω A)
variable (h3 : touches ω' (segment B A_0) ∧ touches ω' (segment C A_0))
variable (h4 : is_midpoint M (segment B C))

-- Define the proof statement
theorem line_AM_bisects_KA_0 :
  bisects (line A M) (segment K A_0) :=
sorry

end line_AM_bisects_KA_0_l586_586370


namespace value_of_f_at_1000_l586_586244

theorem value_of_f_at_1000 :
  (∃ f : ℝ → ℝ, (∀ x > 0, f x - (1 / 2) * f (1 / x) = real.log x) ∧ f 1000 = 2) :=
begin
  sorry
end

end value_of_f_at_1000_l586_586244


namespace grasshoppers_on_daisy_plant_l586_586205

theorem grasshoppers_on_daisy_plant (baby_grasshoppers: ℕ) (total_grasshoppers: ℕ) : baby_grasshoppers = 24 → total_grasshoppers = 31 → total_grasshoppers - baby_grasshoppers = 7 :=
by
  assume H1 : baby_grasshoppers = 24,
  assume H2 : total_grasshoppers = 31,
  calc
    total_grasshoppers - baby_grasshoppers = 31 - 24 : by rw [H2, H1]
    ... = 7 : by norm_num

end grasshoppers_on_daisy_plant_l586_586205


namespace perfect_square_factors_of_420_l586_586069

-- Define the prime factorization of 420
def factor_420 : ℕ := 420
def prime_factors_420 : ℕ × ℕ × ℕ × ℕ := (2, 2, 3, 1, 5, 1, 7, 1)

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∀ p k, nat.prime p → n = p^k → even k

-- Cardinality of the set of perfect square factors of 420
def num_perfect_square_factors (n : ℕ) : ℕ :=
  { m | m ∣ n ∧ is_perfect_square m }.to_finset.card

-- The main theorem to be proven
theorem perfect_square_factors_of_420 : num_perfect_square_factors factor_420 = 2 :=
sorry

end perfect_square_factors_of_420_l586_586069


namespace probability_sum_six_l586_586341

theorem probability_sum_six : 
  let die_outcomes := {1, 2, 3, 4, 5, 6} in
  let all_outcomes := { (x, y) | x ∈ die_outcomes ∧ y ∈ die_outcomes } in
  let favorable_outcomes := {
    (1, 5), (2, 4), (3, 3), (4, 2), (5, 1)
  } in
  fintype.card favorable_outcomes = 5 →
  fintype.card all_outcomes = 36 →
  5 / 36 ∈ {(fintype.card favorable_outcomes)/(fintype.card all_outcomes)} →
  5 / 36 = 5/36 := by sorry

end probability_sum_six_l586_586341


namespace max_modulus_l586_586873

theorem max_modulus (z : ℂ) (h : |z| = 1) : max (abs (z - (3 - 4 * Complex.I))) = 6 :=
sorry

end max_modulus_l586_586873


namespace total_students_in_class_l586_586563

theorem total_students_in_class (R : ℕ) :
  let S := 25 + R
  (0 * 5 + 1 * 12 + 2 * 8 + 3 * R) / S = 2 → S = 47 :=
by
  intro h
  have h1: 28 + 3 * R = 2*S, from sorry
  have h2: S = 25 + R, from sorry
  have h3: 28 + 3 * R = 50 + 2 * R, from sorry
  have h4: R = 22, from sorry
  show S = 47, from sorry

end total_students_in_class_l586_586563


namespace value_difference_max_min_l586_586513

noncomputable def f : ℝ → ℝ := λ x => x^3 - 12 * x + 8

def interval : Set ℝ := { x | -3 ≤ x ∧ x ≤ 3 }

theorem value_difference_max_min :
  let M := superset f interval,
  let m := infimum f interval in
  M - m = 32 := 
sorry

end value_difference_max_min_l586_586513


namespace max_min_k_max_min_2x_y_l586_586485

variables {x y k : Real}

def on_circle (x y : Real) : Prop := x^2 + (y - 1)^2 = 1

def slope_k (x y : Real) : Real := (y - 1) / (x - 2)

def linear_comb (x : Real) (y : Real) : Real := 2 * x + y

theorem max_min_k {x y : Real} (h : on_circle x y) :
  ∃ k_max k_min, k_max = Real.sqrt 2 ∧ k_min = -Real.sqrt 2 ∧ 
  (∀ k, (slope_k x y = k → k ≤ k_max ∧ k ≥ k_min)) :=
sorry

theorem max_min_2x_y {x y : Real} (h : on_circle x y) :
  ∃ m_max m_min, m_max = 1 + Real.sqrt 5 ∧ m_min = 1 - Real.sqrt 5 ∧ 
  (∀ m, (linear_comb x y = m → m ≤ m_max ∧ m ≥ m_min)) :=
sorry

end max_min_k_max_min_2x_y_l586_586485


namespace tangent_line_equation_l586_586604

noncomputable def tangent_line_at_origin (a : ℝ) : ℝ → ℝ := 
λ x, x^3 + a * x^2 + (a - 2) * x

theorem tangent_line_equation (a : ℝ) (h : ∀ x : ℝ, 3 * x^2 + 2 * a * x + (a - 2) = 3 * x^2 - 2 * a * x + (a - 2)) : 
  tangent_line_at_origin a 0 = -2 :=
by 
  -- The proof steps would go here
  sorry

end tangent_line_equation_l586_586604


namespace count_four_digit_numbers_greater_than_3410_l586_586534

-- Define the set of digits and conditions
def digits : List ℕ := [0, 1, 2, 3, 4, 5]

-- Define the condition for forming a four-digit number greater than 3410 without repetition.
def valid_numbers (n : ℕ) : Prop :=
  let d := n.digits 10 in
  (digits.filter (· ∈ d)).length = 4 ∧
  d.length = 4 ∧
  n > 3410 ∧
  ∀ i j, i ≠ j → d.nth i ≠ d.nth j

-- Using a noncomputable definition to state the theorem
noncomputable def count_valid_numbers : ℕ :=
  (Finset.range 10000).filter valid_numbers).card

-- The theorem to be proven
theorem count_four_digit_numbers_greater_than_3410 : count_valid_numbers = 132 := by
  sorry

end count_four_digit_numbers_greater_than_3410_l586_586534


namespace product_of_two_numbers_l586_586291

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 205) : x * y = 42 :=
by
  sorry

end product_of_two_numbers_l586_586291


namespace probability_of_first_spade_or_ace_and_second_ace_l586_586711

theorem probability_of_first_spade_or_ace_and_second_ace :
  let deck_size := 52
  let aces := 4
  let spades := 13
  let non_ace_spades := spades - 1
  let other_aces := aces - 1
  let prob_first_non_ace_spade := non_ace_spades / deck_size
  let prob_first_ace_not_spade := (aces - 1) / deck_size
  let prob_first_ace_spade := 1 / deck_size
  let prob_second_ace_after_non_ace_spade := aces / (deck_size - 1)
  let prob_second_ace_after_ace_not_spade := (aces - 1) / (deck_size - 1)
  let prob_second_ace_after_ace_spade := (aces - 1) / (deck_size - 1)
  ((prob_first_non_ace_spade * prob_second_ace_after_non_ace_spade) +
   (prob_first_ace_not_spade * prob_second_ace_after_ace_not_spade) +
   (prob_first_ace_spade * prob_second_ace_after_ace_spade)) = 5 / 221 :=
by
  let deck_size := 52
  let aces := 4
  let spades := 13
  let non_ace_spades := spades - 1
  let other_aces := aces - 1
  let prob_first_non_ace_spade := non_ace_spades / deck_size
  let prob_first_ace_not_spade := (aces - 1) / deck_size
  let prob_first_ace_spade := 1 / deck_size
  let prob_second_ace_after_non_ace_spade := aces / (deck_size - 1)
  let prob_second_ace_after_ace_not_spade := (aces - 1) / (deck_size - 1)
  let prob_second_ace_after_ace_spade := (aces - 1) / (deck_size - 1)
  sorry

end probability_of_first_spade_or_ace_and_second_ace_l586_586711


namespace new_pyramid_volume_l586_586387

-- Definitions related to the volume of the original pyramid
def original_volume (l w h : ℝ) : ℝ := (1/3) * l * w * h

-- Given conditions in the problem
def given_original_volume : ℝ := 60
def length_tripled (l : ℝ) : ℝ := 3 * l
def width_quadrupled (w : ℝ) : ℝ := 4 * w
def height_doubled (h : ℝ) : ℝ := 2 * h

-- Statement to prove the new volume of the pyramid
theorem new_pyramid_volume (l w h : ℝ) :
  original_volume l w h = given_original_volume →
  original_volume (length_tripled l) (width_quadrupled w) (height_doubled h) = 1440 :=
by
  intros h₁
  -- Proof is skipped with sorry
  sorry

end new_pyramid_volume_l586_586387


namespace measure_angle_BPC_is_60_degrees_l586_586581

-- Define the setting of the problem
variables (A B C D G H E P Q : Type) [regular_hexagon ABCDGH]
variables (length_AB : length AB = 10) (equilateral_triangle_ABE : equilateral_triangle A B E)
variables (intersect_BE_AD_at_P : ∃ P, BE ∩ AD = {P})
variables (perpendicular_PQ_BD : PQ ⊥ BD) (length_PQ : length PQ = x)

-- Define the main proposition to prove
theorem measure_angle_BPC_is_60_degrees
  (hABCDGH : regular_hexagon ABCDGH)
  (h_length : length AB = 10)
  (hAEB : equilateral_triangle A B E)
  (h_intersect : ∃ P, BE ∩ AD = {P})
  (h_perpendicular : PQ ⊥ BD)
  (h_length_PQ : length PQ = x) :
  measure_of_angle B P C = 60 :=
sorry

end measure_angle_BPC_is_60_degrees_l586_586581


namespace inequality_solution_l586_586860

section
variables (a x : ℝ)

theorem inequality_solution (h : a < 0) :
  (ax^2 + (1 - a) * x - 1 > 0 ↔
     (-1 < a ∧ a < 0 ∧ 1 < x ∧ x < -1/a) ∨
     (a = -1 ∧ false) ∨
     (a < -1 ∧ -1/a < x ∧ x < 1)) :=
by sorry

end inequality_solution_l586_586860


namespace probability_less_than_third_l586_586117

def interval_real (a b : ℝ) := set.Ioo a b

def probability_space (a b : ℝ) := {
  length := b - a,
  sample_space := interval_real a b
}

def probability_of_event {a b c : ℝ} (h : a < b) (h1 : a < c) (h2 : c < b) : Prop :=
  (c - a) / (b - a)

theorem probability_less_than_third :
  probability_of_event (by norm_num : (0:ℝ) < (1 / 2))
                       (by norm_num : (0:ℝ) < (1 / 3))
                       (by norm_num : (1 / 3) < (1 / 2))
  = (2 / 3) := sorry

end probability_less_than_third_l586_586117


namespace simplify_expression_l586_586642

theorem simplify_expression (w : ℝ) : 2 * w + 4 * w + 6 * w + 8 * w + 10 * w + 12 = 30 * w + 12 :=
by
  sorry

end simplify_expression_l586_586642


namespace ratio_of_sizes_l586_586404

-- Defining Anna's size
def anna_size : ℕ := 2

-- Defining Becky's size as three times Anna's size
def becky_size : ℕ := 3 * anna_size

-- Defining Ginger's size
def ginger_size : ℕ := 8

-- Defining the goal statement
theorem ratio_of_sizes : (ginger_size : ℕ) / (becky_size : ℕ) = 4 / 3 :=
by
  sorry

end ratio_of_sizes_l586_586404


namespace length_PF_eq_eight_by_three_l586_586892

def parabola : Prop :=
  ∃ y x (:ℝ), y^2 = 8 * (x + 2)

def inclination_of_60_degrees (F : Inhabited ℝ) : Prop :=
  ∃ a b (:ℝ), b = (60 : ℝ) * a / 180 * π

def focus_F (x y : ℝ) : Prop :=
  x = 0 ∧ y = 0

def intersection_points (A B : Inhabited ℝ) : Prop :=
  ∃ x_1 x_2 x_3 x_4 (:ℝ), A = (x_1, x_3) ∧ B = (x_2, x_4) ∧ (x_1 ≥ 0 ∨ x_2 ≤ 0)

def perpendicular_bisector_intersect (P : Inhabited ℝ) (AB : Inhabited ℝ) : Prop :=
  ∃ x_1 x_2 x_3 x_4 (:ℝ), AB = (x_1, x_3) ∧ P = (x_2, x_4) ∧ (x_3 = 0 ∨ x_4 = 0)

theorem length_PF_eq_eight_by_three (P F : ℝ) :
  parabola →
  focus_F 0 0 → 
  inclination_of_60_degrees (F : Inhabited ℝ) →
  intersection_points (A : Inhabited ℝ) (B : Inhabited ℝ) →
  perpendicular_bisector_intersect (P : Inhabited ℝ) (AB : Inhabited ℝ) →
  P - F = (8 / 3) :=
begin
  sorry,
end

end length_PF_eq_eight_by_three_l586_586892


namespace num_divisors_360_l586_586920

theorem num_divisors_360 :
  ∀ n : ℕ, n = 360 → (∀ (p q r : ℕ), p = 2 ∧ q = 3 ∧ r = 5 →
    (∃ (a b c : ℕ), 360 = p^a * q^b * r^c ∧ a = 3 ∧ b = 2 ∧ c = 1) →
    (3+1) * (2+1) * (1+1) = 24) :=
  sorry

end num_divisors_360_l586_586920


namespace parabola_translation_l586_586710

theorem parabola_translation :
  ∀ (x : ℝ), (let y := 3 * (x - 4) ^ 2 + 2 in y = 3 * ((x - 1) - 4) ^ 2 - 1) := 
  by
    sorry

end parabola_translation_l586_586710


namespace ring_cost_l586_586533

theorem ring_cost (total_cost : ℕ) (rings : ℕ) (h1 : total_cost = 24) (h2 : rings = 2) : total_cost / rings = 12 :=
by
  sorry

end ring_cost_l586_586533


namespace quadratic_inequality_solution_l586_586900

theorem quadratic_inequality_solution
  (a b : ℝ)
  (h1 : ∀ x : ℝ, x^2 + a * x + b > 0 ↔ (x < -2 ∨ -1/2 < x)) :
  ∀ x : ℝ, b * x^2 + a * x + 1 < 0 ↔ -2 < x ∧ x < -1/2 :=
by
  sorry

end quadratic_inequality_solution_l586_586900


namespace bisecting_unit_vector_l586_586236

noncomputable def vector_example : ℝ :=
  15 -- using the precision

def a : ℝ × ℝ × ℝ := (4, 3, 0)
def b : ℝ × ℝ × ℝ := (1, 2, 1)
def v : ℝ × ℝ × ℝ := (-22 / 15, -14 / 15, 2 / 15)

theorem bisecting_unit_vector :
  b = 6 • (1 / 2 • (a + 5 • v)) :=
by {
  sorry
}

end bisecting_unit_vector_l586_586236


namespace part_I_part_II_part_III_l586_586519

noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log (x + 1)
def g (x : ℝ) : ℝ := x^3

theorem part_I :
  ∃ a b : ℝ, a = 1 ∧ b = 0 := by 
  sorry

theorem part_II (x : ℝ) :
  0 < x → f x < g x := by 
  sorry

theorem part_III (n : ℕ) (h : n > 0) :
  1 + ∑ i in Finset.range (n - 1), 1 / Real.exp ((i - 1) * (i ^ 2 + 1)) < n * (n + 3) / 2 := by 
  sorry

end part_I_part_II_part_III_l586_586519


namespace exists_point_P_l586_586894

noncomputable def point_exists (A B C D : Point) (P : Point) : Prop :=
  P ∈ LineSegment C D ∧ ∠ (A P D) = 2 * ∠ (B P C)

variables {A B C D P : Point}
variables (h1 : ∠C = 90°) (h2 : ∠D = 90°) (h3 : C ≠ D)

theorem exists_point_P (hcd_gt_bc : dist C D > dist B C) :
  ∃ P : Point, point_exists A B C D P := sorry

end exists_point_P_l586_586894


namespace probability_less_than_one_third_l586_586097

theorem probability_less_than_one_third : 
  (∃ p : ℚ, p = (measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ)) (set.Ioo (0 : ℚ) (1 / 3 : ℚ)) / 
                  measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ))) ∧ p = 2 / 3) :=
by {
  -- The proof will be added here.
  sorry
}

end probability_less_than_one_third_l586_586097


namespace calculate_CF_l586_586953

-- Definitions
variable (A B C D F: Type) [AddGroup F] [HasSmul A F]
variables (AD BC AB DC DE DF CF: F)
variables (isosceles_trapezoid : A → B → C → D → Prop)
variables (on_line : C → D → F → Prop)
variables (one_third : B → F → E → Prop)

-- Conditions
axiom h1 : isosceles_trapezoid A B C D
axiom h2 : AD = 6
axiom h3 : BC = 6
axiom h4 : AB = 3
axiom h5 : DC = 12
axiom h6 : on_line C D F
axiom h7 : one_third B D E

-- Theorem to be proved
theorem calculate_CF : CF = 10.5 := 
by
  -- Proof goes here
  sorry

end calculate_CF_l586_586953


namespace product_of_more_than_2k_prime_factors_l586_586003

theorem product_of_more_than_2k_prime_factors
  {k : ℕ} (h_k : 5 < k) (p : ℕ → ℕ) (h_prime : ∀ i, 1 ≤ i → i ≤ k → Nat.Prime (p i))
  (h_p_seq : p 1 = 2 ∧ p 2 = 3 ∧ p 3 = 5 ∧ ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → p i < p j) :
  ∃ S : ℕ, (∑ i in (Finset.powerset (Finset.range k)).map (λ s, ∏ j in s, p (j+1))) + 1
           ∣ (∏ i in Finset.range k, (p (i+1) + 1)) ∧
           (Nat.factors ((∑ i in (Finset.powerset (Finset.range k)).map (λ s, ∏ j in s, p (j+1))) + 1)).length > 2 * k :=
sorry

end product_of_more_than_2k_prime_factors_l586_586003


namespace real_mul_eq_zero_iff_l586_586344

theorem real_mul_eq_zero_iff (a b : ℝ) (h : a * b = 0) : a = 0 ∨ b = 0 :=
sorry

end real_mul_eq_zero_iff_l586_586344


namespace semi_circle_shaded_area_ratio_l586_586349

noncomputable def shaded_area_ratio (AB AC CB CD : ℝ) (h : ∀ (r : ℝ), AB = 2*r ∧ AC = r ∧ CB = r) : Prop :=
  let r := AC in
  let shaded_area := (1/2) * π * r^2 - 2 * (1/2) * π * (r/2)^2
  let circle_area := π * CD^2 in
  shaded_area / circle_area = 1 / 4

-- The final theorem stating the proof problem
theorem semi_circle_shaded_area_ratio (AB AC CB CD : ℝ) (h : ∀ (r : ℝ), AB = 2*r ∧ AC = r ∧ CB = r) (h1 : CD = AC) :
  shaded_area_ratio AB AC CB CD h :=
by
  sorry

end semi_circle_shaded_area_ratio_l586_586349


namespace number_of_solution_pairs_l586_586525

def integer_solutions_on_circle : Set (Int × Int) := {
  (1, 7), (1, -7), (-1, 7), (-1, -7),
  (5, 5), (5, -5), (-5, 5), (-5, -5),
  (7, 1), (7, -1), (-7, 1), (-7, -1) 
}

def system_of_equations_has_integer_solution (a b : ℝ) : Prop :=
  ∃ (x y : ℤ), a * ↑x + b * ↑y = 1 ∧ (↑x ^ 2 + ↑y ^ 2 = 50)

theorem number_of_solution_pairs : ∃ (n : ℕ), n = 72 ∧
  (∀ (a b : ℝ), system_of_equations_has_integer_solution a b → n = 72) := 
sorry

end number_of_solution_pairs_l586_586525


namespace simplify_f_find_value_of_f_l586_586031

variable {α : ℝ}

-- Given that α is an angle in the third quadrant.
def in_third_quadrant (α : ℝ) : Prop := π < α ∧ α < 3 * π / 2

-- Given that f(α) is defined as:
def f (α : ℝ) : ℝ :=
  (sin (α - π / 2) * cos (3 * π / 2 + α) * tan (π - α)) /
  (tan (-α - π) * sin (-α - π))

-- Proof problem 1: Simplify f(α) and show f(α) = -cos α.
theorem simplify_f : in_third_quadrant α → f(α) = -cos α :=
by
  sorry

-- Given that cos (α - 3π/2) = 1/5
axiom cos_alpha_condition : α → in_third_quadrant α → cos (α - 3 * π / 2) = 1/5

-- Proof problem 2: Given cos(α - 3π/2) = 1/5, find the value of f(α).
theorem find_value_of_f : in_third_quadrant α → cos (α - 3 * π / 2) = 1 / 5 → f(α) = 2 * sqrt 6 / 5 :=
by
  sorry

end simplify_f_find_value_of_f_l586_586031


namespace average_velocity_instantaneous_velocity_l586_586762

noncomputable def s (t : ℝ) : ℝ := 8 - 3 * t^2

theorem average_velocity {Δt : ℝ} (h : Δt ≠ 0) :
  (s (1 + Δt) - s 1) / Δt = -6 - 3 * Δt :=
sorry

theorem instantaneous_velocity :
  deriv s 1 = -6 :=
sorry

end average_velocity_instantaneous_velocity_l586_586762


namespace luke_money_by_march_l586_586622

variable (initial_amount spent_amount received_amount : ℕ)
variable (h_initial : initial_amount = 48)
variable (h_spent : spent_amount = 11)
variable (h_received : received_amount = 21)

theorem luke_money_by_march :
  initial_amount - spent_amount + received_amount = 58 :=
by {
  rw [h_initial, h_spent, h_received],
  norm_num,
  sorry
}

end luke_money_by_march_l586_586622


namespace matrix_product_zero_l586_586807

variable {R : Type*} [CommRing R]

def matrix_R (d e f : R) : Matrix (Fin 3) (Fin 3) R := 
  ![![0, d, -e],
    ![-d, 0, f],
    ![e, -f, 0]]

def matrix_S (x y z : R) : Matrix (Fin 3) (Fin 3) R := 
  ![![x^2, x * y, x * z],
    ![x * y, y^2, y * z],
    ![x * z, y * z, z^2]]

theorem matrix_product_zero (d e f x y z : R) (h1 : d * y = e * z) (h2 : d * x = f * z) (h3 : e * x = f * y) :
  matrix_R d e f ⬝ matrix_S x y z = 0 :=
by
  sorry

end matrix_product_zero_l586_586807


namespace λ_plus_μ_l586_586573

variables (AB AD BE BC DF DC AE AF CE CF : Vector ℝ)
variables (λ μ : ℝ)
variables (AB_length AD_length angle_BAD : ℝ)
variables (E_on_BC : BE = λ • BC)
variables (F_on_CD : DF = μ • DC)
variables (dot_AE_AF : AE ⬝ AF = 1)
variables (dot_CE_CF : CE ⬝ CF = -(3/2))

-- Given conditions:
#check AB_length = 2
#check AD_length = 2
#check angle_BAD = 120 * π / 180
#check ∀ x, x ⬝ x = 2 * 2 * Real.cos angle_BAD
#check E_on_BC
#check F_on_CD
#check dot_AE_AF
#check dot_CE_CF

-- Main theorem:
theorem λ_plus_μ : λ + μ = 5 / 4 :=
sorry

end λ_plus_μ_l586_586573


namespace oliver_ratio_l586_586258

theorem oliver_ratio (v a b : ℝ) (h1 : 5 * v = v * 5) 
  (h2 : b / v = (a / v) + (a + b) / (5 * v)) : 
  a / b = 2 / 3 :=
by
  -- key equations derived from the problem statement
  have h3 : b / v = (6 * a + b) / (5 * v) 
    from h2,
  have h4 : 5 * b = 6 * a + b 
    from (mul_eq_mul_right_iff.mpr (or.intro_left _ (ne_of_gt (by norm_num)))).mpr h3,
  have h5 : 4 * b = 6 * a 
    from sub_eq_zero.mpr h4,
  -- solve for the ratio
  exact eq_inv_mul_of_mul_eq_mul_left (by norm_num) h5

end oliver_ratio_l586_586258


namespace min_square_sum_l586_586303

theorem min_square_sum (a b m n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 15 * a + 16 * b = m * m) (h4 : 16 * a - 15 * b = n * n) : 481 ≤ min (m * m) (n * n) :=
sorry

end min_square_sum_l586_586303


namespace probability_merlin_dismissed_l586_586216

noncomputable def merlin_dismissal_probability {p : ℝ} (hp : 0 ≤ p ∧ p ≤ 1) : ℝ :=
  let q := 1 - p in
  -- Assuming the coin flip is fair, the probability Merlin is dismissed
  1 / 2

theorem probability_merlin_dismissed (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  merlin_dismissal_probability hp = 1 / 2 :=
by
  -- Placeholder for the proof
  sorry

end probability_merlin_dismissed_l586_586216


namespace determine_M_l586_586767

def sequence (M : ℝ) (u : ℕ → ℝ) : Prop :=
  u 0 = M + 1/2 ∧ ∀ n, u (n + 1) = u n * (⌊u n⌋)

def takes_integer_value (u : ℕ → ℝ) : Prop :=
  ∃ n, ∃ m : ℤ, u n = m

theorem determine_M (M : ℝ) (h1 : 1 ≤ M) (h2 : ∀ u, sequence M u → takes_integer_value u) : 1 < M :=
  sorry

end determine_M_l586_586767


namespace roots_of_quadratic_l586_586306

theorem roots_of_quadratic (x : ℝ) (h : x^2 = x) : x = 0 ∨ x = 1 :=
by
  sorry

end roots_of_quadratic_l586_586306


namespace factorize_expression_l586_586838

theorem factorize_expression (a x : ℝ) : -a*x^2 + 2*a*x - a = -a*(x - 1)^2 := 
sorry

end factorize_expression_l586_586838


namespace sum_of_odd_integers_l586_586680

theorem sum_of_odd_integers (n : ℕ) (h : n * (n + 1) = 4970) : (n * n = 4900) :=
by sorry

end sum_of_odd_integers_l586_586680


namespace system_inequalities_1_l586_586276

theorem system_inequalities_1 (x : ℝ) (h1 : 2 * x ≥ x - 1) (h2 : 4 * x + 10 > x + 1) :
  x ≥ -1 :=
sorry

end system_inequalities_1_l586_586276


namespace place_balls_l586_586261

def labelSet := {1, 2, 3, 4, 5, 6}

def ballPlacementCount : ℕ :=
  let box1Choices := 3
  let remainingBallSet := {3, 4, 5, 6}
  let box2Choices := (remainingBallSet.card.choose 2)
  let totalChoices := box1Choices * box2Choices
  totalChoices

-- Noncomputable because we use combinatorial calculation 
noncomputable def numberOfWaysToPlaceBalls : ℕ := 18

theorem place_balls : ballPlacementCount = numberOfWaysToPlaceBalls := 
by 
  have h_card : remainingBallSet.card = 4 := rfl
  rw [Nat.choose]
  have h_comb : 4.choose 2 = 6 := rfl
  have box1_choices: 3 = 3 := rfl
  rw [h_comb]
  have result: 3 * 6 = 18 := rfl
  exact result

end place_balls_l586_586261


namespace hyperbola_focal_length_range_l586_586057

theorem hyperbola_focal_length_range (m : ℝ) (h1 : m > 0)
    (h2 : ∀ x y, x^2 - y^2 / m^2 ≠ 1 → y ≠ m * x ∧ y ≠ -m * x)
    (h3 : ∀ x y, x^2 + (y + 2)^2 = 1 → x^2 + y^2 / m^2 ≠ 1) :
    ∃ c : ℝ, 2 < 2 * Real.sqrt (1 + m^2) ∧ 2 * Real.sqrt (1 + m^2) < 4 :=
by
  sorry

end hyperbola_focal_length_range_l586_586057


namespace find_angle_A_l586_586495

variables {a b c : ℝ} -- sides of the triangle
variables (A : ℝ) -- angle A
variables (S : ℝ) -- area of the triangle

-- Define the conditions as assumptions
def condition1 : Prop :=
  b > 0 ∧ c > 0 ∧ S = 1 / (4 * Real.sqrt 3) * (b^2 + c^2 - a^2)

-- State the theorem to be proved
theorem find_angle_A (h : condition1) : A = Real.pi / 6 :=
by sorry

end find_angle_A_l586_586495


namespace zero_in_interval_l586_586662

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x + 3 * x

-- Prove that 0 is in the interval (-1, 0)
theorem zero_in_interval : 0 ∈ set.Ioo (-1 : ℝ) (0 : ℝ) := sorry

end zero_in_interval_l586_586662


namespace bathroom_visits_time_l586_586592

variable (t_8 : ℕ) (n8 : ℕ) (n6 : ℕ)

theorem bathroom_visits_time (h1 : t_8 = 20) (h2 : n8 = 8) (h3 : n6 = 6) :
  (t_8 / n8) * n6 = 15 := by
  sorry

end bathroom_visits_time_l586_586592


namespace solve_for_nabla_l586_586079

theorem solve_for_nabla (nabla : ℤ) (h : 3 * (-2) = nabla + 2) : nabla = -8 := 
by
  sorry

end solve_for_nabla_l586_586079


namespace geom_seq_ratio_l586_586015

-- Definition of a geometric sequence
noncomputable def geometric_seq (a₁ q : ℕ) : ℕ → ℕ
| 0     => a₁
| (n + 1) => geometric_seq q n * q

-- Conditions given in the problem
def a₃ (a₁ q : ℕ) := geometric_seq a₁ q 2 = 2
def a₄a₆ (a₁ q : ℕ) := (geometric_seq a₁ q 3) * (geometric_seq a₁ q 5) = 16

-- Proof goal
theorem geom_seq_ratio (a₁ q : ℕ) (h₁ : a₃ a₁ q) (h₂ : a₄a₆ a₁ q) :
  (geometric_seq a₁ q 8 - geometric_seq a₁ q 10) / (geometric_seq a₁ q 4 - geometric_seq a₁ q 6) = 4 :=
sorry

end geom_seq_ratio_l586_586015


namespace area_of_living_room_floor_l586_586365

-- Definitions for conditions
def carpet_length : ℝ := 4
def carpet_width : ℝ := 9
def carpet_area := carpet_length * carpet_width -- 4 * 9 = 36
def carpet_coverage : ℝ := 0.3 -- 30%

-- Define the total area of the living room floor
def total_area := carpet_area / carpet_coverage -- 36 / 0.3 = 120

-- Theorem stating that the total area is 120 square feet
theorem area_of_living_room_floor : total_area = 120 := by
  sorry

end area_of_living_room_floor_l586_586365


namespace parabola_properties_l586_586036

noncomputable def parabola_definition (x y : ℝ) : Prop :=
  y^2 = 16 * x

def midpoint (x1 y1 x2 y2 mx my : ℝ) : Prop :=
  (mx = (x1 + x2) / 2 ∧ my = (y1 + y2) / 2)

def distance (x1 y1 x2 y2 d : ℝ) : Prop :=
  d = Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def triangle_area (x1 y1 x2 y2 x3 y3 area : ℝ) : Prop :=
  area = (1/2) * Real.abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

theorem parabola_properties :
  (parabola_definition 4 0) ∧
  (∃ t : ℝ, parabola_definition 2 (t / 2) ∧ midpoint 4 0 0 t 2 (t / 2)) ∧ 
  (distance 4 0 0 t 12) ∧
  (triangle_area 0 0 0 t 4 0 (16 * Real.sqrt 2)) :=
by
  sorry

end parabola_properties_l586_586036


namespace exponent_in_right_side_l586_586937

theorem exponent_in_right_side (m : ℤ) (h1 : (-2)^(2 * m) = 2^(12 - m)) (h2 : m = 4) : 12 - m = 8 :=
by
  sorry

end exponent_in_right_side_l586_586937


namespace range_of_e1e2_l586_586617

open Real

theorem range_of_e1e2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) :
  (∀ x y : ℝ, (x^2/a^2 + y^2/b^2 = 1) ∧ (x^2/a^2 - y^2/b^2 = 1)) → 
  (2 * sqrt (a^2 - b^2) > 2 * b) →
  (sqrt (1 - (b/a)^4) ∈ set.Ioo (sqrt 3 / 2) 1) := by
  intros
  sorry

end range_of_e1e2_l586_586617


namespace area_equal_l586_586975

-- Define the semiperimeter function
def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

-- Define Heron's formula for the area of a triangle
def heron_area (a b c : ℝ) : ℝ :=
  let s := semiperimeter a b c
  in real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Define the sides of the triangles
def triangle1_sides : ℝ × ℝ × ℝ := (25, 25, 30)
def triangle2_sides : ℝ × ℝ × ℝ := (25, 25, 40)

-- Calculate the areas using Heron's formula
def area_triangle1 : ℝ := heron_area triangle1_sides.1 triangle1_sides.2 triangle1_sides.3
def area_triangle2 : ℝ := heron_area triangle2_sides.1 triangle2_sides.2 triangle2_sides.3

-- The actual proof statement that the areas are equal
theorem area_equal : area_triangle1 = area_triangle2 :=
by {
  -- skip the proof itself by replacing with sorry
  sorry
}

end area_equal_l586_586975


namespace find_m_value_l586_586396

theorem find_m_value (m : ℝ) : 
    let original_line := λ x : ℝ, 2 * x,
        translated_line := λ x : ℝ, 2 * x - 3 in
    translated_line (m + 2) = -5 ↔ m = -3 := 
by
    simp
    sorry

end find_m_value_l586_586396


namespace number_of_divisors_360_l586_586925

theorem number_of_divisors_360 : 
  ∃ (e1 e2 e3 : ℕ), e1 = 3 ∧ e2 = 2 ∧ e3 = 1 ∧ (∏ e in [e1, e2, e3], e + 1) = 24 := by
    use 3, 2, 1
    split
    { exact rfl }
    split
    { exact rfl }
    split
    { exact rfl }
    simp
    norm_num

end number_of_divisors_360_l586_586925


namespace part1_part2_part3_l586_586516

noncomputable def f (a x : ℝ) := a * Real.log x - a * x + 1
noncomputable def g (a x : ℝ) := f a x + ½ * x^2 - 1

theorem part1 (a : ℝ) (h : ∀ k, k = (f a 2) / 2 → k = (- a / 2)) : a = 1 / (1 - Real.log 2) :=
sorry

theorem part2 (a : ℝ) 
  (decrease_cond : ∀ x, (3 / 2 < x) ∧ (x < 4) → (f a x + x / 2 - a) ≤ 0) : a > 16 / 3 :=
sorry

theorem part3 (a λ : ℝ) 
  (crit_cond : ∀ x1 x2, x1 ≠ x2 → (f a x1 + x1 / 2 - a = 0) ∧ (f a x2 + x2 / 2 - a = 0) → 
    (f a x1 + ½ * x1^2 - 1 + f a x2 + ½ * x2^2 - 1 < λ * (x1 + x2))) : λ > 2 * Real.log 2 - 3 :=
sorry

end part1_part2_part3_l586_586516


namespace monotonically_increasing_implies_non_negative_derivative_non_negative_derivative_not_implies_monotonically_increasing_l586_586441

variables {f : ℝ → ℝ}

-- Definition that f is monotonically increasing
def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 ≤ f x2

-- Definition of the derivative being non-negative everywhere
def non_negative_derivative (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ (deriv f) x

theorem monotonically_increasing_implies_non_negative_derivative (f : ℝ → ℝ) :
  monotonically_increasing f → non_negative_derivative f :=
sorry

theorem non_negative_derivative_not_implies_monotonically_increasing (f : ℝ → ℝ) :
  non_negative_derivative f → ¬ monotonically_increasing f :=
sorry

end monotonically_increasing_implies_non_negative_derivative_non_negative_derivative_not_implies_monotonically_increasing_l586_586441


namespace probability_less_than_one_third_l586_586158

-- Define the intervals and the associated lengths
def interval_total : ℝ := 1 / 2
def interval_condition : ℝ := 1 / 3

-- Define the probability calculation
def probability : ℝ :=
  interval_condition / interval_total

-- Statement of the problem
theorem probability_less_than_one_third : probability = 2 / 3 := by
  sorry

end probability_less_than_one_third_l586_586158


namespace probability_merlin_dismissed_l586_586223

variable (p : ℝ) (q : ℝ := 1 - p)
variable (r : ℝ := 1 / 2)

theorem probability_merlin_dismissed :
  (0 < p ∧ p ≤ 1) →
  (q = 1 - p) →
  (r = 1 / 2) →
  (probability_king_correct_two_advisors = p) →
  (strategy_merlin_minimizes_dismissal = true) →
  (strategy_percival_honest = true) →
  probability_merlin_dismissed = 1 / 2 := by
  sorry

end probability_merlin_dismissed_l586_586223


namespace team_d_final_probability_correct_l586_586743

noncomputable def probability_d_entering_final (pDC pDA pDB pAB : ℝ) : ℝ :=
  let pBA := 1 - pAB in
  let p_second_match := pAB * pDA + pBA * pDB in
  pDC * p_second_match

theorem team_d_final_probability_correct :
  probability_d_entering_final 0.6 0.5 0.4 0.6 = 0.276 :=
by
  unfold probability_d_entering_final
  simp
  sorry

end team_d_final_probability_correct_l586_586743


namespace chenny_friends_count_l586_586432

def initial_candies := 10
def additional_candies := 4
def candies_per_friend := 2

theorem chenny_friends_count :
  (initial_candies + additional_candies) / candies_per_friend = 7 := by
  sorry

end chenny_friends_count_l586_586432


namespace number_of_floors_in_building_l586_586234

theorem number_of_floors_in_building 
  (kolya_floor : ℕ) (kolya_apt : ℕ)
  (vasya_floor : ℕ) (vasya_apt : ℕ)
  (apts_per_floor : ℕ)
  (h1 : kolya_floor = 5)
  (h2 : kolya_apt = 83)
  (h3 : vasya_floor = 3)
  (h4 : vasya_apt = 169)
  (h5 : apts_per_floor = 4) : 
  ∃ n : ℕ, n = 4 :=
by
  use 4
  sorry

end number_of_floors_in_building_l586_586234


namespace probability_less_than_third_l586_586109

def interval_real (a b : ℝ) := set.Ioo a b

def probability_space (a b : ℝ) := {
  length := b - a,
  sample_space := interval_real a b
}

def probability_of_event {a b c : ℝ} (h : a < b) (h1 : a < c) (h2 : c < b) : Prop :=
  (c - a) / (b - a)

theorem probability_less_than_third :
  probability_of_event (by norm_num : (0:ℝ) < (1 / 2))
                       (by norm_num : (0:ℝ) < (1 / 3))
                       (by norm_num : (1 / 3) < (1 / 2))
  = (2 / 3) := sorry

end probability_less_than_third_l586_586109


namespace range_of_m_inequality_l586_586877

noncomputable def f (a b x : ℝ) : ℝ := (a * 2^x - 1) / (2^x + b)

theorem range_of_m_inequality (a b m : ℝ) (h_odd : ∀x : ℝ, f a b (-x) = -f a b x) :
  ∀ x ∈ set.Icc 1 2, 2 + m * f a b x + 2^x > 0 ↔ m > -2 * real.sqrt 6 - 5 :=
sorry

end range_of_m_inequality_l586_586877


namespace tree_height_after_two_years_l586_586777

theorem tree_height_after_two_years :
  (∀ (f : ℕ → ℝ), (∀ (n : ℕ), f (n + 1) = 3 * f n) → f 4 = 81 → f 2 = 9) :=
begin
  sorry
end

end tree_height_after_two_years_l586_586777


namespace exterior_angle_polygon_l586_586173

theorem exterior_angle_polygon (n : ℕ) (h : 360 / n = 72) : n = 5 :=
by
  sorry

end exterior_angle_polygon_l586_586173


namespace min_a_decreasing_range_a_condition_l586_586883

noncomputable def f (x a : ℝ) : ℝ := x / Real.log x - a * x
noncomputable def f' (x a : ℝ) : ℝ := (Real.log x - 1) / (Real.log x)^2 - a

theorem min_a_decreasing (a : ℝ) (h : ∀ x > 1, f' x a ≤ 0) : 
  a ≥ 1 / 4 :=
sorry

theorem range_a_condition (a : ℝ) 
  (h : ∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc Real.exp (Real.exp 1 2) ∧ x₂ ∈ Set.Icc Real.exp (Real.exp 1 2) ∧ f x₁ a ≤ f' x₂ a + a) : 
  a ≥ (1 / 2) - (1 / (4 * Real.exp 1 2)) :=
sorry

end min_a_decreasing_range_a_condition_l586_586883


namespace geometric_sequence_sum_l586_586757

theorem geometric_sequence_sum (a r : ℚ) (h_a : a = 1/3) (h_r : r = 1/2) (S_n : ℚ) (h_S_n : S_n = 80/243) : ∃ n : ℕ, S_n = a * ((1 - r^n) / (1 - r)) ∧ n = 4 := by
  sorry

end geometric_sequence_sum_l586_586757


namespace count_three_digit_prime_with_all_prime_digits_l586_586536

open Nat

-- Define the set of single-digit primes
def single_digit_primes : set ℕ := {2, 3, 5, 7}

-- Define the predicate for three-digit prime number with all digits being primes
def three_digit_prime_with_all_prime_digits (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧
  (n / 100) ∈ single_digit_primes ∧
  ((n % 100) / 10) ∈ single_digit_primes ∧
  (n % 10) ∈ single_digit_primes ∧
  prime n

-- Define the theorem to prove the count
theorem count_three_digit_prime_with_all_prime_digits : 
  Finset.card (Finset.filter three_digit_prime_with_all_prime_digits (Finset.range 900 \ Finset.range 100)) = 12 := 
sorry

end count_three_digit_prime_with_all_prime_digits_l586_586536


namespace cyclic_inequality_l586_586854

theorem cyclic_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (cyclic_sum (λ x y z, (y + z) * (x^4 - y^2 * z^2) / (x * y + 2 * y * z + z * x)) a b c) ≥ 0 :=
begin
  sorry
end

end cyclic_inequality_l586_586854


namespace polynomial_root_abs_sum_eq_80_l586_586474

theorem polynomial_root_abs_sum_eq_80 (a b c : ℤ) (m : ℤ) 
  (h1 : a + b + c = 0) 
  (h2 : ab + bc + ac = -2023) 
  (h3 : ∃ m, ∀ x : ℤ, x^3 - 2023 * x + m = (x - a) * (x - b) * (x - c)) : 
  |a| + |b| + |c| = 80 := 
by {
  sorry
}

end polynomial_root_abs_sum_eq_80_l586_586474


namespace number_of_divisors_360_l586_586922

theorem number_of_divisors_360 : 
  ∃ (e1 e2 e3 : ℕ), e1 = 3 ∧ e2 = 2 ∧ e3 = 1 ∧ (∏ e in [e1, e2, e3], e + 1) = 24 := by
    use 3, 2, 1
    split
    { exact rfl }
    split
    { exact rfl }
    split
    { exact rfl }
    simp
    norm_num

end number_of_divisors_360_l586_586922


namespace basketball_game_total_first_half_points_l586_586565

theorem basketball_game_total_first_half_points
  (a r b d : ℝ) 
  (h1 : a * (1 + r + r^2) = b + (b + d) + (b + 2 * d)) 
  (S_M = a * (1 + r + r^2 + r^3)) 
  (S_F = 4 * b + 6 * d) 
  (h2 : S_M = S_F + 2) 
  (h3 : S_M + S_F < 80) : 
  a + a * r + (b + b + d) = 20.4 := 
sorry

end basketball_game_total_first_half_points_l586_586565


namespace number_of_positive_divisors_360_l586_586912

theorem number_of_positive_divisors_360 : 
  let n := 360 
  in let prime_factors := [(2, 3), (3, 2), (5, 1)]
  in (∀ (p : ℕ) (a : ℕ), (p, a) ∈ prime_factors → p.prime) →
     (∀ m ∈ prime_factors, ∃ (p a : ℕ), m = (p, a) ∧ n = (p ^ a) * (prime_factors.filter (λ m', m ≠ m')).prod (λ m', (m'.fst ^ m'.snd))) →
     (prime_factors.foldr (λ (m : ℕ × ℕ) acc, (m.snd + 1) * acc) 1) = 24 := 
begin
  sorry
end

end number_of_positive_divisors_360_l586_586912


namespace find_theta_range_l586_586530

-- Definitions used in Lean statement should appear directly in the conditions problem.
variables {a b : EuclideanSpace ℝ (Fin 3)}
variable θ : ℝ
variable (cosθ : ℝ := inner_product_space.angle a b)

-- Conditions in Lean
axiom length_sum : ∥a + b∥ = 2 * sqrt 3
axiom length_diff : ∥a - b∥ = 2

-- Main theorem statement (with conditions)
theorem find_theta_range :
  0 ≤ θ ∧ θ ≤ Real.arccos (1/2) :=
sorry

end find_theta_range_l586_586530


namespace ellipse_range_l586_586507

variable (m : ℝ)

def ellipse_condition (m : ℝ) :=
  2 + m > 0 ∧ -(m + 1) > 0 ∧ 2 + m ≠ -(m + 1)

theorem ellipse_range (m : ℝ) (h : ellipse_condition m) : m ∈ set.Ioo (-2 : ℝ) (-3 / 2) ∪ set.Ioo (-3 / 2) (-1) :=
  sorry

end ellipse_range_l586_586507


namespace isosceles_triangle_perimeter_l586_586490

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 3 ∨ b = 3) (h2 : a = 6 ∨ b = 6) 
(h_isosceles : a = b ∨ b = a) : 
  a + b + a = 15 ∨ b + a + b = 15 :=
by sorry

end isosceles_triangle_perimeter_l586_586490


namespace conner_day3_rocks_to_tie_l586_586286

def initial_rocks_sydney : ℕ := 837
def initial_rocks_conner : ℕ := 723
def day1_rocks_sydney : ℕ := 4
def day1_rocks_conner : ℕ := 8 * day1_rocks_sydney
def day2_rocks_sydney : ℕ := 0
def day2_rocks_conner : ℕ := 123
def day3_rocks_sydney : ℕ := 2 * day1_rocks_conner

theorem conner_day3_rocks_to_tie :
  ∃ x : ℕ, initial_rocks_conner + day1_rocks_conner + day2_rocks_conner + x = 
            initial_rocks_sydney + day1_rocks_sydney + day2_rocks_sydney + day3_rocks_sydney :=
begin
  use 27,
  sorry
end

end conner_day3_rocks_to_tie_l586_586286


namespace b_horses_months_l586_586353

theorem b_horses_months (x : ℕ) :
  let a_horses := 12
  let a_months := 8
  let b_horses := 16
  let b_cost := 360
  let c_horses := 18
  let c_months := 6
  let total_cost := 870
  (
    b_cost * (a_horses * a_months + b_horses * x + c_horses * c_months) 
    = total_cost * b_horses * x
  ) → x = 9 := 
by {
  intros,
  sorry
}

end b_horses_months_l586_586353


namespace probability_merlin_dismissed_l586_586213

variable {p q : ℝ} (hpq : p + q = 1) (hp : 0 < p ∧ p < 1)

theorem probability_merlin_dismissed : 
  let decision_prob : ℝ := if h : p = 1 then 1 else p in
  if h : decision_prob = p then (1 / 2) else 0 := 
begin
  sorry
end

end probability_merlin_dismissed_l586_586213


namespace value_of_a_l586_586181

theorem value_of_a (a : ℝ) : (∀ x : ℝ, |x - a| < 1 ↔ 2 < x ∧ x < 4) → a = 3 :=
by
  intro h
  have h1 : 2 = a - 1 := sorry
  have h2 : 4 = a + 1 := sorry
  have h3 : a = 3 := sorry
  exact h3

end value_of_a_l586_586181


namespace color_of_face_opposite_silver_is_yellow_l586_586834

def Face : Type := String

def Color : Type := String

variable (B Y O Bl S V : Color)

-- Conditions based on views
variable (cube : Face → Color)
variable (top front_right_1 right_1 front_right_2 front_right_3 : Face)
variable (back : Face)

axiom view1 : cube top = B ∧ cube front_right_1 = Y ∧ cube right_1 = O
axiom view2 : cube top = B ∧ cube front_right_2 = Bl ∧ cube right_1 = O
axiom view3 : cube top = B ∧ cube front_right_3 = V ∧ cube right_1 = O

-- Additional axiom based on the fact that S is not visible and deduced to be on the back face
axiom silver_back : cube back = S

-- The problem: Prove that the color of the face opposite the silver face is yellow.
theorem color_of_face_opposite_silver_is_yellow :
  (∃ front : Face, cube front = Y) :=
by
  sorry

end color_of_face_opposite_silver_is_yellow_l586_586834


namespace find_width_of_brick_l586_586849

theorem find_width_of_brick (l h : ℝ) (SurfaceArea : ℝ) (w : ℝ) :
  l = 8 → h = 2 → SurfaceArea = 152 → 2*l*w + 2*l*h + 2*w*h = SurfaceArea → w = 6 :=
by
  intro l_value
  intro h_value
  intro SurfaceArea_value
  intro surface_area_equation
  sorry

end find_width_of_brick_l586_586849


namespace locus_of_M_l586_586562

-- Definitions based on conditions
variables {circle : Type} [metric_space circle]
variables (O A B C D P : circle)
-- Assumptions based on the problem conditions
variables (diam_AC : line circle)
variables (diam_BD : line circle)
variables (orthogonal : are_orthogonal diam_AC diam_BD)
variables (on_circle_P : is_on_circumference circle P)
variables (PA : line circle)
variables (BD : line circle)
variables (intersection_E : point_of_intersection PA BD)
variables (line_through_E : line circle)
variables (parallel_to_AC : is_parallel_to line_through_E diam_AC)
variables (intersects_PB_at_M : intersects_at line_through_E (line_through P B) M)

-- Lean theorem statement
theorem locus_of_M :
  ∀ P, is_on_circumference circle P →
       ∃ M, intersects_at (parallel_through intersection_E diam_AC) (line_through P B) M ∧
            lies_on_line M (line_through D C) :=
sorry

end locus_of_M_l586_586562


namespace sum_of_roots_ln_abs_eq_l586_586611

theorem sum_of_roots_ln_abs_eq (m : ℝ) (x1 x2 : ℝ) (hx1 : Real.log (|x1|) = m) (hx2 : Real.log (|x2|) = m) : x1 + x2 = 0 :=
sorry

end sum_of_roots_ln_abs_eq_l586_586611


namespace range_of_a_l586_586518

open Real

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f(f(x)) > x) :
  1 - (sqrt 3) / 2 < a ∧ a < 1 + (sqrt 3) / 2 :=
sorry

end range_of_a_l586_586518


namespace cot_thirty_deg_l586_586417

theorem cot_thirty_deg : ∃ (θ : ℝ), θ = 30 ∧ cot θ = sqrt 3 :=
by
  sorry

end cot_thirty_deg_l586_586417


namespace necessary_but_not_sufficient_l586_586182

variables {a b : ℝ}

theorem necessary_but_not_sufficient (h : a > 0) (h₁ : a > b) (h₂ : a⁻¹ > b⁻¹) : 
  b < 0 :=
sorry

end necessary_but_not_sufficient_l586_586182


namespace polynomial_with_shifted_roots_l586_586982

open Polynomial 

noncomputable def given_polynomial : Polynomial ℝ :=
  X^3 - 6 * X^2 + 11 * X - 6

noncomputable def required_polynomial : Polynomial ℝ :=
  X^3 - 15 * X^2 + 74 * X - 120

theorem polynomial_with_shifted_roots :
  let a b c : ℝ in
  (is_root given_polynomial a) ∧ (is_root given_polynomial b) ∧ (is_root given_polynomial c) ∧
  (required_polynomial = (C 1) * (X - (a + 3)) * (X - (b + 3)) * (X - (c + 3))) :=
sorry

end polynomial_with_shifted_roots_l586_586982


namespace purchase_price_of_shares_l586_586350

variable (P : ℝ)

-- Conditions
def dividend_rate : ℝ := 12.5 / 100
def share_value : ℝ := 40
def roi : ℝ := 25 / 100
def dividend_per_share : ℝ := dividend_rate * share_value := by sorry

-- Proof statement
theorem purchase_price_of_shares : dividend_per_share = 5 → 5 / roi = P → P = 20 := by
  sorry

end purchase_price_of_shares_l586_586350


namespace interval_probability_l586_586149

theorem interval_probability (x y z : ℝ) (h0x : 0 < x) (hy : y > 0) (hz : z > 0) (hx : x < y) (hy_mul_z_eq_half : y * z = 1 / 2) (hx_eq_third_y : x = 1 / 3 * y) :
  (x / y) = (2 / 3) :=
by
  -- Here you should provide the proof
  sorry

end interval_probability_l586_586149


namespace all_options_valid_l586_586830

-- Definitions for the given conditions
def line_equation (x y : ℝ) : Prop := y = 2 * x + 1
def parametric_eq (a b t : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (a.1 + d * b.1, a.2 + d * b.2)

-- The options as parameteric equations
def option_A (t : ℝ) : ℝ × ℝ := parametric_eq (1, 3) (-2, -4) t
def option_B (t : ℝ) : ℝ × ℝ := parametric_eq (0, 1) (5, 10) t
def option_C (t : ℝ) : ℝ × ℝ := parametric_eq (-1, -1) (1, 2) t
def option_D (t : ℝ) : ℝ × ℝ := parametric_eq (2, 5) (0.5, 1) t
def option_E (t : ℝ) : ℝ × ℝ := parametric_eq (3, 7) (-1, -2) t

-- The proof problem to be stated in Lean
theorem all_options_valid:
  (∀ t, line_equation (option_A t).fst (option_A t).snd) ∧
  (∀ t, line_equation (option_B t).fst (option_B t).snd) ∧
  (∀ t, line_equation (option_C t).fst (option_C t).snd) ∧
  (∀ t, line_equation (option_D t).fst (option_D t).snd) ∧
  (∀ t, line_equation (option_E t).fst (option_E t).snd) :=
sorry

end all_options_valid_l586_586830


namespace quadratic_to_standard_form_l586_586304

theorem quadratic_to_standard_form (a b c : ℝ) (x : ℝ) :
  (20 * x^2 + 240 * x + 3200 = a * (x + b)^2 + c) → (a + b + c = 2506) :=
  sorry

end quadratic_to_standard_form_l586_586304


namespace part_a_part_b_l586_586733

-- Define natural number and prime number
variables (N : ℕ) (p : ℕ) (k : ℕ)

-- Assume N is a natural number and p is a prime number
-- Defining the valuation function
def val_p (n : ℕ) (p : ℕ) : ℕ :=
  if p.prime then n.factors.count p else 0

-- Assuming conditions: N is natural, p is prime, and the given formulas
theorem part_a (h1 : Nat.prime p) :
  val_p (N.factorial) p = ∑ i in Finset.range (N + 1), N / (p ^ i) :=
sorry

theorem part_b (h1 : Nat.prime p) (H : N ≥ k) (hN : ∃ n : ℕ, N = p ^ n) : 
  val_p (Nat.choose N k) p = val_p N p - val_p k p :=
sorry

end part_a_part_b_l586_586733


namespace product_equals_odd_sequence_l586_586328

theorem product_equals_odd_sequence (n : ℕ) (h : 0 < n) :
  ((n+1) * (n+2) * ... * (n+n)) = 2^n * (finset.range n).product (λ i, 2*i + 1) := 
sorry

end product_equals_odd_sequence_l586_586328


namespace consistent_2_config_subset_l586_586272

-- Definitions corresponding to conditions in a)
variables {A : Type*} [fintype A]

/-- A definition of consistent 2-configuration of given order on A -/
def consistent_2_config (n : ℕ) (A : Type*) [fintype A] := 
  ∀ (pairs : finset (finset A)), pairs.card = n / 2

-- Translating the proof statement in Lean
theorem consistent_2_config_subset {A : Type*} [finite A] :
  ∀ (config4 : consistent_2_config 4 A), 
  ∃ (B : finset A), consistent_2_config 2 B :=
sorry

end consistent_2_config_subset_l586_586272


namespace point_in_first_quadrant_l586_586301

def complex_to_point (z : ℂ) : ℝ × ℝ := (z.re, z.im)

def is_first_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

theorem point_in_first_quadrant :
  let z : ℂ := 1 + 2 * complex.I
  let p := complex_to_point z
  is_first_quadrant p :=
by
  let z : ℂ := 1 + 2 * complex.I
  let p := complex_to_point z
  show is_first_quadrant p
  sorry

end point_in_first_quadrant_l586_586301


namespace probability_of_interval_l586_586122

theorem probability_of_interval (a b x : ℝ) (h : 0 < a ∧ a < b ∧ 0 < x) : 
  (x < b) → (b = 1/2) → (x = 1/3) → (0 < x) → (x - 0) / (b - 0) = 2/3 := 
by 
  sorry

end probability_of_interval_l586_586122


namespace water_added_l586_586367

-- Definitions for given conditions
def original_volume : ℝ := 440
def original_sugar_percentage : ℝ := 0.04
def added_sugar : ℝ := 3.2
def added_kola : ℝ := 6.8
def new_sugar_percentage : ℝ := 0.04521739130434784

-- Derived facts from conditions
def original_sugar : ℝ := original_sugar_percentage * original_volume
def total_sugar_after_addition : ℝ := original_sugar + added_sugar
def new_total_volume : ℝ := total_sugar_after_addition / new_sugar_percentage

-- Theorem: Proving the amount of water added
theorem water_added : (new_total_volume - original_volume - added_sugar - added_kola) = 10 := 
by sorry

end water_added_l586_586367


namespace students_both_sports_l586_586746

-- Define the sets and their cardinalities based on the given conditions
variable (total_students : ℕ := 38)
variable (students_running : ℕ := 21)
variable (students_basketball : ℕ := 15)
variable (students_neither : ℕ := 10)

-- Theorem stating the number of students who enjoy both sports
theorem students_both_sports (total_students students_running students_basketball students_neither : ℕ) (h : total_students = 38) (h1 : students_running = 21) (h2 : students_basketball = 15) (h3 : students_neither = 10) :
  let students_both := students_running + students_basketball - (total_students - students_neither) in
  students_both = 8 :=
by {
  -- Lean proof would go here.
  sorry
}

end students_both_sports_l586_586746


namespace sin_C_of_right_triangle_l586_586574

theorem sin_C_of_right_triangle (A B C : Type) [triangle : Triangle A B C]
  (right_angle : ∠ B = 90) (sin_A : sin A = 7/25) : sin C = 24/25 := 
sorry

end sin_C_of_right_triangle_l586_586574


namespace sum_of_b_l586_586613

theorem sum_of_b (S : ℤ) : 
  (S = ∑ b in {b : ℤ | ∃ r s : ℤ, r + s = -b ∧ r * s = 2023 * b ∧ r + 2023 ≥ 0 ∧ r + 2023 ≤ 2023^2}, b) → |S| = 121380 :=
sorry

end sum_of_b_l586_586613


namespace area_of_rectangular_field_l586_586300

-- Define the conditions
variables (l w : ℝ)

def perimeter_condition : Prop := 2 * l + 2 * w = 100
def length_width_relation : Prop := l = 3 * w

-- Define the area
def area : ℝ := l * w

-- Prove the area given the conditions
theorem area_of_rectangular_field (h1 : perimeter_condition l w) (h2 : length_width_relation l w) : area l w = 468.75 :=
by sorry

end area_of_rectangular_field_l586_586300


namespace initial_amount_celine_had_l586_586769

-- Define the costs and quantities
def laptop_cost : ℕ := 600
def smartphone_cost : ℕ := 400
def num_laptops : ℕ := 2
def num_smartphones : ℕ := 4
def change_received : ℕ := 200

-- Calculate costs and total amount
def cost_laptops : ℕ := num_laptops * laptop_cost
def cost_smartphones : ℕ := num_smartphones * smartphone_cost
def total_cost : ℕ := cost_laptops + cost_smartphones
def initial_amount : ℕ := total_cost + change_received

-- The statement to prove
theorem initial_amount_celine_had : initial_amount = 3000 := by
  sorry

end initial_amount_celine_had_l586_586769


namespace geometric_probability_l586_586106

theorem geometric_probability :
  let total_length := (1/2 - 0)
  let favorable_length := (1/3 - 0)
  total_length > 0 ∧ favorable_length > 0 →
  (favorable_length / total_length) = 2 / 3 :=
by
  intros _ _ _
  sorry

end geometric_probability_l586_586106


namespace area_of_inscribed_square_l586_586308

theorem area_of_inscribed_square (s : ℝ) : 
  let r := (sqrt 3 / 6) * s in 
  let diagonal := 2 * r in
  let a := diagonal / sqrt 2 in
  a * a = s^2 / 6 :=
by
  let r := (sqrt 3 / 6) * s
  let diagonal := 2 * r
  let a := diagonal / sqrt 2
  have ha : a * a = (s^2 / 6), from sorry
  exact ha

end area_of_inscribed_square_l586_586308


namespace stratified_sampling_l586_586766

theorem stratified_sampling (students_to_select grade10_students grade11_students grade12_students : ℕ)
  (h_total : students_to_select = 18)
  (h_grade10 : grade10_students = 600)
  (h_grade11 : grade11_students = 800)
  (h_grade12 : grade12_students = 400) :
  let total_students := grade10_students + grade11_students + grade12_students in
  let grade10_select := (students_to_select * grade10_students) / total_students in
  let grade11_select := (students_to_select * grade11_students) / total_students in
  let grade12_select := (students_to_select * grade12_students) / total_students in
  grade10_select = 6 ∧ grade11_select = 8 ∧ grade12_select = 4 :=
by
  sorry

end stratified_sampling_l586_586766


namespace collinear_points_l586_586238

variables {V : Type*} [add_comm_group V] [module ℝ V]
variables (e1 e2 : V) (k : ℝ)
variables (A B C D : V)
variables (AB CB CD BD : V)

-- Conditions
def AB_def : AB = e1 - k • e2 := sorry
def CB_def : CB = 4 • e1 - 2 • e2 := sorry
def CD_def : CD = 3 • e1 - 3 • e2 := sorry
def BD_def : BD = CD - CB := sorry

theorem collinear_points (hAB : AB = e1 - k • e2)
                        (hCB : CB = 4 • e1 - 2 • e2)
                        (hCD : CD = 3 • e1 - 3 • e2)
                        (hBD : BD = CD - CB)
                        (collinear : ∃ (λ : ℝ), AB = λ • BD) :
  k = -1 := sorry

end collinear_points_l586_586238


namespace gift_arrangement_l586_586948

theorem gift_arrangement (n k : ℕ) (h_n : n = 5) (h_k : k = 4) : 
  (n * Nat.factorial k) = 120 :=
by
  sorry

end gift_arrangement_l586_586948


namespace construct_points_X_Y_l586_586024

variable {α : Type*}
variables (A B C X Y : EuclideanGeometry.Point α)
variables (triangle : EuclideanGeometry.Triangle A B C)

def acute_triangle (A B C : EuclideanGeometry.Point α) : Prop :=
  EuclideanGeometry.angle A B C < 90 ∧ EuclideanGeometry.angle B C A < 90 ∧ EuclideanGeometry.angle C A B < 90

def on_segment (P Q R : EuclideanGeometry.Point α) : Prop :=
  EuclideanGeometry.Between P Q R ∨ EuclideanGeometry.Between Q R P ∨ EuclideanGeometry.Between R P Q

axiom acute_ABC : acute_triangle A B C
axiom X_on_AB : on_segment A X B
axiom Y_on_BC : on_segment B Y C
axiom equal_segments : EuclideanGeometry.distance A X = EuclideanGeometry.distance X Y ∧ EuclideanGeometry.distance X Y = EuclideanGeometry.distance Y C

theorem construct_points_X_Y (A B C : EuclideanGeometry.Point α) (X Y : EuclideanGeometry.Point α) :
  acute_triangle A B C → on_segment A X B → on_segment B Y C → EuclideanGeometry.distance A X = EuclideanGeometry.distance X Y ∧ EuclideanGeometry.distance X Y = EuclideanGeometry.distance Y C :=
begin
  intros,
  exact equal_segments,
end

end construct_points_X_Y_l586_586024


namespace solve_for_x_l586_586443

theorem solve_for_x :
  (∃ x : ℝ, 5 ^ (3 * x^2 - 8 * x + 5) = 5 ^ (3 * x^2 + 4 * x - 7)) → 1 = 1 :=
by
  sorry

end solve_for_x_l586_586443


namespace find_number_of_cows_l586_586564

-- Definitions from the conditions
def number_of_ducks : ℕ := sorry
def number_of_cows : ℕ := sorry

-- Define the number of legs and heads
def legs := 2 * number_of_ducks + 4 * number_of_cows
def heads := number_of_ducks + number_of_cows

-- Given condition from the problem
def condition := legs = 2 * heads + 32

-- Assert the number of cows
theorem find_number_of_cows (h : condition) : number_of_cows = 16 :=
sorry

end find_number_of_cows_l586_586564


namespace simplify_expression_l586_586643

theorem simplify_expression : 
  (4 * 6 / (12 * 14)) * (8 * 12 * 14 / (4 * 6 * 8)) = 1 := by
  sorry

end simplify_expression_l586_586643


namespace probability_merlin_dismissed_l586_586228

variable (p : ℝ) (q : ℝ) (coin_flip : ℝ)

axiom h₁ : q = 1 - p
axiom h₂ : 0 ≤ p ∧ p ≤ 1
axiom h₃ : 0 ≤ q ∧ q ≤ 1
axiom h₄ : coin_flip = 0.5

theorem probability_merlin_dismissed : coin_flip = 0.5 := by
  sorry

end probability_merlin_dismissed_l586_586228


namespace eval_i_powers_l586_586451

noncomputable def i : ℂ := complex.I

theorem eval_i_powers : 
  i^(23456:ℕ) + i^(23457:ℕ) + i^(23458:ℕ) + i^(23459:ℕ) = 0 :=
by
  sorry

end eval_i_powers_l586_586451


namespace two_digit_integers_l586_586242

theorem two_digit_integers (x y m : ℕ) (h1 : 10 ≤ x ∧ x < 100)
                           (h2 : 10 ≤ y ∧ y < 100)
                           (h3 : y = (x % 10) * 10 + x / 10)
                           (h4 : x^2 - y^2 = 9 * m^2) :
  x + y + 2 * m = 143 :=
sorry

end two_digit_integers_l586_586242


namespace f_odd_function_f_increasing_on_R_t_range_l586_586514

-- Definition of the function f
def f (x : ℝ) : ℝ := (exp x - exp (-x)) / (exp x + exp (-x))

-- (I) Prove that f is an odd function
theorem f_odd_function : ∀ x : ℝ, f (-x) = -f x := 
sorry

-- (II) Prove that f is increasing on ℝ
theorem f_increasing_on_R : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := 
sorry

-- (III) Prove the range of t such that f(x-t) + f(x^2 - t^2) ≥ 0 for x ∈ [1, 2]
theorem t_range : ∀ x : ℝ, x ∈ set.Icc (1 : ℝ) (2 : ℝ) → 
  ∀ t : ℝ, (f(x - t) + f(x^2 - t^2) ≥ 0) ↔ (t ∈ set.Icc (-2 : ℝ) (1 : ℝ)) := 
sorry

end f_odd_function_f_increasing_on_R_t_range_l586_586514


namespace tree_height_at_two_years_l586_586778

variable (h : ℕ → ℕ)

-- Given conditions
def condition1 := h 4 = 81
def condition2 := ∀ t : ℕ, h (t + 1) = 3 * h t

theorem tree_height_at_two_years
  (h_tripled : ∀ t : ℕ, h (t + 1) = 3 * h t)
  (h_at_four : h 4 = 81) :
  h 2 = 9 :=
by
  -- Formal proof will be provided here
  sorry

end tree_height_at_two_years_l586_586778


namespace roots_polynomial_product_l586_586608

theorem roots_polynomial_product (a b c : ℝ) (h₁ : Polynomial.eval a (Polynomial.Coeff (Polynomial.coeff 3) - 15 * Polynomial.coeff 2 + 22 * Polynomial.coeff 1 - 8 * Polynomial.X 0) = 0)
(h₂ : Polynomial.eval b (Polynomial.Coeff (Polynomial.coeff 3) - 15 * Polynomial.coeff 2 + 22 * Polynomial.coeff 1 - 8 * Polynomial.X 0) = 0)
(h₃ : Polynomial.eval c (Polynomial.Coeff (Polynomial.coeff 3) - 15 * Polynomial.coeff 2 + 22 * Polynomial.coeff 1 - 8 * Polynomial.X 0) = 0) :
(1 + a) * (1 + b) * (1 + c) = 46 :=
sorry

end roots_polynomial_product_l586_586608


namespace sum_of_integers_with_prime_factors_2_3_5_and_no_perfect_cubes_gt_1_l586_586468

theorem sum_of_integers_with_prime_factors_2_3_5_and_no_perfect_cubes_gt_1 :
  (∑ n in { n | ∃ (a b c : ℕ), (n = 2^a * 3^b * 5^c) ∧ (0 ≤ a ∧ a < 3) ∧ (0 ≤ b ∧ b < 3) ∧ (0 ≤ c ∧ c < 3) ∧ n > 1 }, n) = 2820 :=
by
  sorry

end sum_of_integers_with_prime_factors_2_3_5_and_no_perfect_cubes_gt_1_l586_586468


namespace xiaohua_total_time_l586_586347

def time_in_minutes := λ (hrs mins : ℕ), hrs * 60 + mins

-- Condition definitions
def arrival_time_morning := time_in_minutes 7 20
def leave_time_morning := time_in_minutes 11 45
def arrival_time_afternoon := time_in_minutes 13 50
def leave_time_afternoon := time_in_minutes 17 15

-- Calculate time spent in morning and afternoon
def time_spent_morning := leave_time_morning - arrival_time_morning
def time_spent_afternoon := leave_time_afternoon - arrival_time_afternoon

-- Total time spent at school
def total_time_spent := time_spent_morning + time_spent_afternoon

-- Expected time in minutes (7 hours 50 minutes)
def expected_time := time_in_minutes 7 50

-- The theorem to prove
theorem xiaohua_total_time :
  total_time_spent = expected_time :=
by sorry

end xiaohua_total_time_l586_586347


namespace pictures_at_museum_l586_586346

variable (M : ℕ)

-- Definitions from conditions
def pictures_at_zoo : ℕ := 50
def pictures_deleted : ℕ := 38
def pictures_left : ℕ := 20

-- Theorem to prove the total number of pictures taken including the museum pictures
theorem pictures_at_museum :
  pictures_at_zoo + M - pictures_deleted = pictures_left → M = 8 :=
by
  sorry

end pictures_at_museum_l586_586346


namespace total_volume_is_correct_l586_586764

def radius : ℝ := 3
def height : ℝ := 10

def volume_cone (r h : ℝ) : ℝ := (1/3) * π * r^2 * h
def volume_hemisphere (r : ℝ) : ℝ := (2/3) * π * r^3

def total_volume_ice_cream (r h : ℝ) : ℝ :=
  volume_cone r h + volume_hemisphere r

theorem total_volume_is_correct :
  total_volume_ice_cream radius height = 48 * π := by
  sorry

end total_volume_is_correct_l586_586764


namespace square_combinations_l586_586545

theorem square_combinations (n : ℕ) (h : n * (n - 1) = 30) : n * (n - 1) = 30 :=
by sorry

end square_combinations_l586_586545


namespace dad_eyes_l586_586966

def mom_eyes : ℕ := 1
def kids_eyes : ℕ := 3 * 4
def total_eyes : ℕ := 16

theorem dad_eyes :
  mom_eyes + kids_eyes + (total_eyes - (mom_eyes + kids_eyes)) = total_eyes :=
by 
  -- The proof part is omitted as per instructions
  sorry

example : (total_eyes - (mom_eyes + kids_eyes)) = 3 :=
by 
  -- The proof part is omitted as per instructions
  sorry

end dad_eyes_l586_586966


namespace smallest_positive_base_l586_586500

theorem smallest_positive_base 
  (N : ℕ) 
  (h0 : ∃ n : ℕ, N = n^4)
  (h1 : ∃ k : ℤ, 2 * 50 * N = k)
  (h2 : ∀ b : ℕ, 777_b = 2 * 50 * N) : 
  ∃ b : ℕ, b = 18 :=
sorry

end smallest_positive_base_l586_586500


namespace wu_xing_arrangement_l586_586782

-- Define the elements
inductive Element
| metal
| wood
| earth
| water
| fire
deriving DecidableEq

open Element

-- Define the "overcomes" relationship as a function
def overcomes : Element → Element → Prop
| metal, wood  => true
| wood, earth  => true
| earth, water => true
| water, fire  => true
| fire, metal  => true
| _, _         => false

-- Define the problem statement
theorem wu_xing_arrangement : 
  ∃ (arrangements : Finset (List Element)), 
    arrangements.card = 10 ∧ 
    ∀ (l : List Element), l ∈ arrangements → 
      ∀ (i : ℕ), i < l.length - 1 → ¬ overcomes (l.nth_le i sorry) (l.nth_le (i + 1) sorry) := 
begin
  sorry
end

end wu_xing_arrangement_l586_586782


namespace problem_conditions_l586_586874

-- Define the ellipse and foci
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 2) = 1
def F1 : ℝ × ℝ := (-√2, 0)
def F2 : ℝ × ℝ := (√2, 0)
def M : ℝ × ℝ := (0, 2)
def dist (A B : ℝ × ℝ) : ℝ := (sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2))

theorem problem_conditions (P : ℝ × ℝ) (hP : ellipse P.1 P.2) :
  (dist P F1 + dist P F2 = 4) ∧
  (∀ P, ellipse P.1 P.2 → dist P F1 - dist P F2 <= 2 * √2) ∧
  (¬ ∃ P, ellipse P.1 P.2 ∧ ∃ θ, θ = arctan (F2.2 - P.2) (F2.1 - P.1) - arctan (F1.2 - P.2) (F1.1 - P.1) ∧ θ = π/3) ∧
  (∀ θ, ∃ max_dist : ℝ, P = (2 * cos θ, √2 * sin θ) ∧ max_dist = dist P M ∧ max_dist ≤ 2 + √2) :=
sorry

end problem_conditions_l586_586874


namespace line_curve_hyperbola_l586_586862

theorem line_curve_hyperbola (a b : ℝ) (h : a * b ≠ 0) : 
  ∃ (x y : ℝ), bx^2 + ay^2 = ab ∧ ax - y + b = 0 :=
by
  sorry

end line_curve_hyperbola_l586_586862


namespace triangle_not_right_triangle_l586_586730

theorem triangle_not_right_triangle (a b c : ℝ) (angle_A angle_B angle_C : ℝ) :
  ¬ (b^2 = (a + c) * (a - c)) →
  ¬ (a:b:c = 1:real.sqrt 3:2) →
  ¬ (angle_C = angle_A - angle_B) →
  (angle_A : angle_B : angle_C = 3:4:5) →
  angle_C < 90 :=
by
  intros hA hB hC hD
  sorry

end triangle_not_right_triangle_l586_586730


namespace cot_30_eq_sqrt3_l586_586414

theorem cot_30_eq_sqrt3 (h : Real.tan (Real.pi / 6) = 1 / Real.sqrt 3) : Real.cot (Real.pi / 6) = Real.sqrt 3 := by
  sorry

end cot_30_eq_sqrt3_l586_586414


namespace telephone_subset_exists_l586_586793

theorem telephone_subset_exists (num_telephones : ℕ)
  (wires : finset (fin 2004 × fin 2004))
  (wire_colors : wires → fin 4)
  (h_telephones : num_telephones = 2004)
  (h_all_colors_used : ∀ c : fin 4, ∃ (p : wires), wire_colors p = c) :
  ∃ (subset : finset (fin 2004)),
    (subset.card > 0) ∧ (∃ colors : finset (fin 4), colors.card = 3 ∧ 
    ∀ (x y : fin 2004), x ∈ subset → y ∈ subset → (x ≠ y) → 
    wire_colors (min x y, max x y) ∈ colors) :=
  sorry

end telephone_subset_exists_l586_586793


namespace decagon_diagonals_l586_586825

-- Number of diagonals calculation definition
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Proving the number of diagonals in a decagon
theorem decagon_diagonals : num_diagonals 10 = 35 := by
  sorry

end decagon_diagonals_l586_586825


namespace intersection_A_B_l586_586899

def is_log2 (y x : ℝ) : Prop := y = Real.log x / Real.log 2

def set_A (y : ℝ) : Set ℝ := { x | ∃ y, is_log2 y x}
def set_B : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

theorem intersection_A_B : (set_A 1) ∩ set_B = { x | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_A_B_l586_586899


namespace monotonic_interval_fx_l586_586886

noncomputable def fx (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ) - 1

theorem monotonic_interval_fx :
  ∀ (φ : ℝ), |φ| < Real.pi →
  ∀ (k : ℤ),
  ∃ (ω : ℝ),
    ω > 0 ∧ 
    (ω = 2 / 3) →
    ∀ x, 
      (fx ω φ x = 0 → fx ω φ (π / 3) = 0) ∧ 
      (Real.sin (ω * -π/6 + φ) = Real.sin π/2) →
      (-5 * π / 3 + 3 * k * π ≤ x ∧ x ≤ -π / 6 + 3 * k * π) :=
begin
  sorry
end

end monotonic_interval_fx_l586_586886


namespace kelvin_can_win_l586_586598

-- Defining the game conditions
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Game Strategy
def kelvin_always_wins : Prop :=
  ∀ (n : ℕ), ∀ (d : ℕ), (d ∈ (List.range 10)) → 
    ∃ (k : ℕ), k ∈ [3, 7] ∧ ¬is_perfect_square (10 * n + k)

theorem kelvin_can_win : kelvin_always_wins :=
by {
  sorry -- Proof based on strategy of adding 3 or 7 modulo 10 and modulo 100 analysis
}

end kelvin_can_win_l586_586598


namespace polynomial_with_shifted_roots_l586_586983

open Polynomial 

noncomputable def given_polynomial : Polynomial ℝ :=
  X^3 - 6 * X^2 + 11 * X - 6

noncomputable def required_polynomial : Polynomial ℝ :=
  X^3 - 15 * X^2 + 74 * X - 120

theorem polynomial_with_shifted_roots :
  let a b c : ℝ in
  (is_root given_polynomial a) ∧ (is_root given_polynomial b) ∧ (is_root given_polynomial c) ∧
  (required_polynomial = (C 1) * (X - (a + 3)) * (X - (b + 3)) * (X - (c + 3))) :=
sorry

end polynomial_with_shifted_roots_l586_586983


namespace rearrangement_plans_l586_586371

theorem rearrangement_plans : 
  let n := 7
  let k := 3
  (nat.choose n k) * 2 = 70 := 
by
  sorry

end rearrangement_plans_l586_586371


namespace probability_merlin_dismissed_l586_586233

-- Define the conditions
variables (p : ℝ) (q : ℝ) (hpq : p + q = 1) (hp_pos : 0 < p) (hq_pos : 0 < q)

/--
Given advisor Merlin is equally likely to dismiss as Percival
since they are equally likely to give the correct answer independently,
prove that the probability of Merlin being dismissed is \( \frac{1}{2} \).
-/
theorem probability_merlin_dismissed : (1/2 : ℝ) = 1/2 :=
by 
  sorry

end probability_merlin_dismissed_l586_586233


namespace time_to_pass_platform_l586_586732

-- Definitions
def length_of_train : ℤ := 140
def length_of_platform : ℤ := 260
def speed_of_train_kmph : ℤ := 60

-- Statement to prove:
theorem time_to_pass_platform : 
  let distance := length_of_train + length_of_platform in
  let speed_m_per_s := (speed_of_train_kmph * 1000) / 3600 in
  (distance: ℝ) / (speed_m_per_s: ℝ) ≈ 24.0 := 
by
  let distance := length_of_train + length_of_platform
  let speed_m_per_s := (speed_of_train_kmph * 1000) / 3600
  sorry

end time_to_pass_platform_l586_586732


namespace side_length_of_second_triangle_l586_586402

theorem side_length_of_second_triangle :
  ∀ (a : ℝ), 
    a = 60 → 
    3 * (a + a / 2 + a / 4 + a / 8 + ...) = 360 →
    (a / 2) = 30 :=
by
  sorry

end side_length_of_second_triangle_l586_586402


namespace vector_equalities_l586_586958

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b : V)

-- Define the trisection points on AB
def D := 1/3 • (2 • b - 3 • a)
def E := 2/3 • (2 • b - 3 • a)

-- Given Conditions
def CA := 3 • a
def CB := 2 • b

-- The main theorem we want to prove
theorem vector_equalities :
  (1/3 • (2 • b - 3 • a)) - (2/3 • (2 • b - 3 • a)) = (2/3 • b - a) ∧
  (-(3 • a) + 3 • a + 1/3 • (2 • b - 3 • a)) = (2 • a + 2/3 • b) ∧
  (-(3 • a) + 3 • a + 2/3 • (2 • b - 3 • a)) = (a + 4/3 • b) :=
by
  sorry

end vector_equalities_l586_586958


namespace expression_evaluation_l586_586452

noncomputable def evaluate_expression (b : ℝ) : ℝ :=
  (1 / 8) * b^0 + (1 / (9 * b))^0 - 27 ^ (-1/3) - (-81) ^ (-1/4) + (1 / 4) ^ (-2)

theorem expression_evaluation (b : ℝ) (hb : b ≠ 0) : evaluate_expression b = 143 / 8 :=
by
  unfold evaluate_expression
  -- solve it in Lean
  sorry

end expression_evaluation_l586_586452


namespace range_of_m_l586_586893

theorem range_of_m (m : ℝ) : 
  (m^2 - 4 < 0 ∨ m > 2) ∧ ¬(m^2 - 4 < 0 ∧ m > 2) → (-2 < m ∧ m < 2) ∨ (m > 2) :=
by
  intro h
  cases h with hp_not_q hboth
  cases hp_not_q
  case inl hp =>
    left
    exact hp
  case inr hq =>
    right
    exact hq
  sorry

end range_of_m_l586_586893


namespace locus_of_distance_difference_l586_586567

-- Define the data needed for the problem
variables {X : Type} [EuclideanSpace ℝ X]
variables (l1 l2 : affine_subspace ℝ X) (a : ℝ)

-- Define the distance function d
def d (P : X) (l : affine_subspace ℝ X) :=
  infi (λ Q : l, dist P Q)

-- Define the main theorem statement
theorem locus_of_distance_difference :
  (∃ (X : X), |d X l1 - d X l2| = a) ↔
  (X ∈ extended_sides_of_rectangle_parallel_to l1 l2 a) := sorry

end locus_of_distance_difference_l586_586567


namespace value_of_m_l586_586988

noncomputable def has_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

noncomputable def has_no_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c < 0

theorem value_of_m (m : ℝ) :
  (has_distinct_real_roots 1 m 1 ∧ has_no_real_roots 4 (4 * (m + 2)) 1) ↔ (-3 < m ∧ m < -2) :=
by
  sorry

end value_of_m_l586_586988


namespace binomial_x3_coefficient_l586_586939

open Classical

theorem binomial_x3_coefficient 
  (a : ℝ) 
  (h : 2 * a - 1 = 0) 
  : 
  let e := (1/2 : ℝ) in
  let binom := (2 * x - 1 / x)^5 in
  term_coeff binom x 3 = -80 := 
by
  sorry

end binomial_x3_coefficient_l586_586939


namespace option_A_correct_option_B_correct_option_C_correct_option_D_correct_l586_586081

-- Define two conditions, tangent and cutting through.
def is_tangent (l C : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ (slope : ℝ), deriv C P.1 = slope ∧ deriv l P.1 = slope

def cuts_through (l C : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
  is_tangent l C P ∧ ∃ ϵ > 0, ∀ x, abs (x - P.1) < ϵ → (C x - l x) * (C P.1 - l P.1) ≤ 0

-- Define points P that correspond to each problem statement.
def P0 : ℝ × ℝ := (0, 0)
def P1 : ℝ × ℝ := (-1, 0)

-- Define line and curve functions for each part of the problem.
def lA (x : ℝ) := 0
def CA (x : ℝ) := x^3
def lB (x : ℝ) := x -- Representing x=-1 through line equation 'lB'
def CB (x : ℝ) := (x + 1)^2
def lC (x : ℝ) := x
def CC (x : ℝ) := sin x
def lD (x : ℝ) := x
def CD (x : ℝ) := tan x

-- Define the propositions for each option being correct.
theorem option_A_correct : cuts_through lA CA P0 := sorry
theorem option_B_correct : ¬cuts_through lB CB P1 := sorry
theorem option_C_correct : cuts_through lC CC P0 := sorry
theorem option_D_correct : cuts_through lD CD P0 := sorry

end option_A_correct_option_B_correct_option_C_correct_option_D_correct_l586_586081


namespace green_tractor_price_l586_586706

variable (S : ℕ) (r g R G : ℕ) (R_price G_price : ℝ)
variable (h1 : S = 7000)
variable (h2 : r = 2)
variable (h3 : g = 3)
variable (h4 : R_price = 20000)
variable (h5 : G_price = 5000)
variable (h6 : ∀ (x : ℝ), Tobias_earning_red : ℝ) (h7 : ∀ (y : ℝ), Tobias_earning_green : ℝ)

def earning_percentage_red : ℝ := 0.10
def earning_percentage_green : ℝ := 0.20

theorem green_tractor_price :
  S = Tobias_earning_red + Tobias_earning_green →
  Tobias_earning_red = (earning_percentage_red * R_price) * r →
  Tobias_earning_green = (earning_percentage_green * G_price) * g →
  G_price = 5000 :=
  sorry

end green_tractor_price_l586_586706


namespace volume_sphere_gt_cube_l586_586723

theorem volume_sphere_gt_cube (a r : ℝ) (h : 6 * a^2 = 4 * π * r^2) : 
  (4 / 3) * π * r^3 > a^3 :=
by sorry

end volume_sphere_gt_cube_l586_586723


namespace value_of_product_of_sums_of_roots_l586_586605

theorem value_of_product_of_sums_of_roots 
    (a b c : ℂ)
    (h1 : a + b + c = 15)
    (h2 : a * b + b * c + c * a = 22)
    (h3 : a * b * c = 8) :
    (1 + a) * (1 + b) * (1 + c) = 46 := by
  sorry

end value_of_product_of_sums_of_roots_l586_586605


namespace necessary_not_sufficient_condition_l586_586553

variable {α : Type} (M N : Set α) (a : α)
variable [NonEmpty α]

theorem necessary_not_sufficient_condition (hM_nonempty : M.Nonempty) (hM_notin_N : M ∉ N) :
  (a ∈ M ∪ N) ↔ ¬(a ∈ M ∩ N) → (a ∈ M ∩ N) → (a ∈ M ∨ a ∈ N) :=
by
  sorry

end necessary_not_sufficient_condition_l586_586553


namespace complement_intersection_l586_586898

noncomputable def A : Set ℝ := {x : ℝ | x^2 - 3 * x - 4 > 0}

def B : Set ℕ := {x : ℕ | x ≤ 2}

theorem complement_intersection (complement_A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 4}) :
  (({x : ℝ | -1 ≤ x ∧ x ≤ 4}) ∩ (B : Set ℝ)) = ({0, 1, 2} : Set ℝ) := 
sorry

end complement_intersection_l586_586898


namespace max_height_of_ball_l586_586742

def h (t : ℝ) : ℝ := -20 * t^2 + 70 * t + 20

theorem max_height_of_ball : ∃ t₀, h t₀ = 81.25 ∧ ∀ t, h t ≤ 81.25 :=
by
  sorry

end max_height_of_ball_l586_586742


namespace median_salary_is_30000_l586_586749

structure EmployeeDistribution :=
  (president_count : Nat) (president_salary : Nat)
  (vice_president_count : Nat) (vice_president_salary : Nat)
  (director_count : Nat) (director_salary : Nat)
  (associate_director_count : Nat) (associate_director_salary : Nat)
  (admin_specialist_count : Nat) (admin_specialist_salary : Nat)

def company_distribution : EmployeeDistribution :=
  { president_count := 1, president_salary := 150000,
    vice_president_count := 7, vice_president_salary := 100000,
    director_count := 10, director_salary := 82000,
    associate_director_count := 10, associate_director_salary := 55000,
    admin_specialist_count := 50, admin_specialist_salary := 30000 }

theorem median_salary_is_30000 : 
  let total_employees := company_distribution.president_count + company_distribution.vice_president_count + company_distribution.director_count + company_distribution.associate_director_count + company_distribution.admin_specialist_count in
  let median_index1 := total_employees / 2 in
  let median_index2 := median_index1 + 1 in
  total_employees = 78 ∧ median_index1 = 39 ∧ median_index2 = 40 ∧ 
  (median_index1 + median_index2) / 2 = 39.5 ∧ 
  company_distribution.admin_specialist_salary = 30000 ∧ 
  company_distribution.president_count + company_distribution.vice_president_count + company_distribution.director_count + company_distribution.associate_director_count < median_index1 ∧ 
  company_distribution.president_count + company_distribution.vice_president_count + company_distribution.director_count + company_distribution.associate_director_count + company_distribution.admin_specialist_count ≥ median_index2 :=
by 
  sorry

end median_salary_is_30000_l586_586749


namespace inequality_y1_y2_y3_l586_586612

def y_1 : ℝ := Real.log 0.8 / Real.log 0.7
def y_2 : ℝ := Real.log 0.9 / Real.log 1.1
def y_3 : ℝ := 1.1^0.9

theorem inequality_y1_y2_y3 : y_3 > y_1 ∧ y_1 > y_2 := by
  sorry

end inequality_y1_y2_y3_l586_586612


namespace quadratic_roots_distinct_real_l586_586818

theorem quadratic_roots_distinct_real (a b c : ℝ) (h : a = 1 ∧ b = -2 ∧ c = 0)
    (Δ : ℝ := b^2 - 4 * a * c) (hΔ : Δ > 0) :
    (∀ r1 r2 : ℝ, r1 ≠ r2) :=
by
  sorry

end quadratic_roots_distinct_real_l586_586818


namespace equilateral_triangle_area_in_circle_l586_586433

theorem equilateral_triangle_area_in_circle (r : ℝ) (h : r = 9) :
  let s := 2 * r * Real.sin (π / 3)
  let A := (Real.sqrt 3 / 4) * s^2
  A = (243 * Real.sqrt 3) / 4 := by
  sorry

end equilateral_triangle_area_in_circle_l586_586433


namespace volume_of_new_cube_l586_586338

noncomputable def volume_of_cube (s : ℝ) : ℝ :=
  s ^ 3

noncomputable def surface_area_of_cube (s : ℝ) : ℝ :=
  6 * s ^ 2

theorem volume_of_new_cube :
  ∃ (V_new : ℝ),
    (∀ (s_ref Vol_ref Area_new : ℝ),
      Vol_ref = volume_of_cube s_ref ∧ 
      Vol_ref = 8 ∧ 
      Area_new = 4 * (surface_area_of_cube s_ref) 
    → V_new = volume_of_cube (real.sqrt (Area_new / 6))
    ∧ V_new = 64) :=
begin
  use 64,
  intros s_ref Vol_ref Area_new h,
  rcases h with ⟨hvref, h8, harea⟩,
  have h_side_ref: s_ref = real.cbrt 8,
  { rw ←hvref, rw h8, apply real.cbrt_eq_rat },
  rw [surface_area_of_cube, ←harea, h_side_ref, real.cbrt_eq_rat (8 : ℝ)] at *,
  field_simp,
  have hs_ref_pos : 8 = real.cbrt 8 ^ 3 := by simp [real.cbrt_eq_rat (8 : ℝ)],
  have side: real.cbrt 8 = 2 := by simp [real.cbrt_eq_rat (8 : ℝ)],
  simp at side,
  have s_new : 4 = real.sqrt(96/6),
  { field_simp, norm_num, rw side },
  rw ←side,
  rw volume_of_cube,
  norm_num,
  ring_nf,
  rw s_new,
  norm_num,
end

end volume_of_new_cube_l586_586338


namespace initial_mixture_volume_l586_586759

variable (p q : ℕ) (x : ℕ)

theorem initial_mixture_volume :
  (3 * x) + (2 * x) = 5 * x →
  (3 * x) / (2 * x + 12) = 3 / 4 →
  5 * x = 30 :=
by
  sorry

end initial_mixture_volume_l586_586759


namespace tan_ratio_sum_l586_586540

theorem tan_ratio_sum 
  (x y : ℝ) 
  (h1 : sin x / cos y + sin y / cos x = 2) 
  (h2 : cos x / sin y + cos y / sin x = 4) : 
  tan x / tan y + tan y / tan x = 9 := 
sorry

end tan_ratio_sum_l586_586540


namespace seating_arrangements_dog_in_car_l586_586631

-- Define the data for the problem
structure Family where
  mr_lopez : Bool
  mrs_lopez : Bool
  child1 : Bool
  child2 : Bool
  dog : Bool

-- Define the seating constraints
def driver (f : Family) : Bool :=
  f.mr_lopez = true ∨ f.mrs_lopez = true

def front_passenger (f : Family) : Bool :=
  f.child1 = true ∨ f.child2 = true ∨ (f.mr_lopez = false ∧ f.mrs_lopez = false)

def back_with_dog (f : Family) : Bool :=
  f.dog = true → f.child1 = true ∨ f.child2 = true

-- Define the main proposition
theorem seating_arrangements_dog_in_car (f : Family) :
  driver f = true →
  back_with_dog f = true →
  (∃ mr_driver mrs_driver child1_passenger child2_passenger : Bool,
    ((mr_driver = true ∧ mrs_driver = false) ∨ (mr_driver = false ∧ mrs_driver = true)) ∧
    (child1_passenger = true ∨ child2_passenger = true ∨ 
     ((mr_driver = false ∨ mrs_driver = false) ∧ (child1_passenger = false ∧ child2_passenger = false))) ∧
    (f.dog = true → f.child1 = true ∨ f.child2 = true) ∧
    (if f.dog = true then ((mr_driver ∨ mrs_driver) → (child1_passenger ∨ child2_passenger) 
     ∧ not (child1_passenger ∧ child2_passenger)) else False)) :=
begin
  sorry
end

end seating_arrangements_dog_in_car_l586_586631


namespace great_grandson_age_l586_586375

theorem great_grandson_age (n : ℕ) : 
  ∃ n, (n * (n + 1)) / 2 = 666 :=
by
  -- Solution steps would go here
  sorry

end great_grandson_age_l586_586375


namespace probability_less_than_one_third_l586_586135

def interval := set.Ioo (0 : ℝ) (1 / 2)

def desired_interval := set.Ioo (0 : ℝ) (1 / 3)

theorem probability_less_than_one_third :
  (∃ p : ℝ, p = (volume desired_interval / volume interval) ∧ p = 2 / 3) :=
  sorry

end probability_less_than_one_third_l586_586135


namespace number_of_divisors_360_l586_586923

theorem number_of_divisors_360 : 
  ∃ (e1 e2 e3 : ℕ), e1 = 3 ∧ e2 = 2 ∧ e3 = 1 ∧ (∏ e in [e1, e2, e3], e + 1) = 24 := by
    use 3, 2, 1
    split
    { exact rfl }
    split
    { exact rfl }
    split
    { exact rfl }
    simp
    norm_num

end number_of_divisors_360_l586_586923


namespace number_of_positive_divisors_of_360_is_24_l586_586907

theorem number_of_positive_divisors_of_360_is_24 :
  ∀ n : ℕ, n = 360 → n = 2^3 * 3^2 * 5^1 → 
  (n_factors : {p : ℕ × ℕ // p.1 ∈ [2, 3, 5] ∧ p.2 ∈ [3, 2, 1]} )
    → (n_factors.val.snd + 1).prod = 24 :=
by
  intro n hn h_factors
  rw hn at *
  have factors := h_factors.val
  cases factors with p_k q_l r_m
  have hpq : p_k.1 = 2 ∧ p_k.2 = 3 :=
    And.intro sorry sorry,
  have hqr : q_l.1 = 3 ∧ q_l.2 = 2 :=
    And.intro sorry sorry,
  have hr : r_m.1 = 5 ∧ r_m.2 = 1 :=
    And.intro sorry sorry,
  -- The proof would continue, but we'll skip it
  sorry

end number_of_positive_divisors_of_360_is_24_l586_586907


namespace number_of_positive_divisors_of_360_is_24_l586_586908

theorem number_of_positive_divisors_of_360_is_24 :
  ∀ n : ℕ, n = 360 → n = 2^3 * 3^2 * 5^1 → 
  (n_factors : {p : ℕ × ℕ // p.1 ∈ [2, 3, 5] ∧ p.2 ∈ [3, 2, 1]} )
    → (n_factors.val.snd + 1).prod = 24 :=
by
  intro n hn h_factors
  rw hn at *
  have factors := h_factors.val
  cases factors with p_k q_l r_m
  have hpq : p_k.1 = 2 ∧ p_k.2 = 3 :=
    And.intro sorry sorry,
  have hqr : q_l.1 = 3 ∧ q_l.2 = 2 :=
    And.intro sorry sorry,
  have hr : r_m.1 = 5 ∧ r_m.2 = 1 :=
    And.intro sorry sorry,
  -- The proof would continue, but we'll skip it
  sorry

end number_of_positive_divisors_of_360_is_24_l586_586908


namespace total_wheels_at_station_l586_586688

/--
There are 4 trains at a train station.
Each train has 4 carriages.
Each carriage has 3 rows of wheels.
Each row of wheels has 5 wheels.
The total number of wheels at the train station is 240.
-/
theorem total_wheels_at_station : 
    let number_of_trains := 4
    let carriages_per_train := 4
    let rows_per_carriage := 3
    let wheels_per_row := 5
    number_of_trains * carriages_per_train * rows_per_carriage * wheels_per_row = 240 := 
by
    sorry

end total_wheels_at_station_l586_586688


namespace find_b_values_l586_586434

noncomputable def solution_set_b : Set ℝ :=
  {b : ℝ | ∃ x y : ℝ, (sqrt (x * y) = b ^ (2 * b)) ∧ (log b (x ^ (log b y)) + log b (y ^ (log b x)) = 3 * b^3)}

theorem find_b_values : solution_set_b = {b | 0 ≤ b ∧ b ≤ 4 / 3} := sorry

end find_b_values_l586_586434


namespace no_equilateral_triangle_on_grid_l586_586673

-- defining what it means for vertices to lie on grid points
def is_grid_point (x y : Int) : Prop := x ∈ Int ∧ y ∈ Int

-- defining the distance computation between points
def distance_squared (x1 y1 x2 y2 : Int) : Int := (x2 - x1) ^ 2 + (y2 - y1) ^ 2

-- a predicate to check if three points form an equilateral triangle
def is_equilateral (x1 y1 x2 y2 x3 y3 : Int) : Prop := 
  distance_squared x1 y1 x2 y2 = distance_squared x2 y2 x3 y3 ∧ 
  distance_squared x2 y2 x3 y3 = distance_squared x3 y3 x1 y1

-- stating the impossibility of forming an equilateral triangle with vertices at grid points
theorem no_equilateral_triangle_on_grid : ∀ (x1 y1 x2 y2 x3 y3 : Int),
  is_grid_point x1 y1 →
  is_grid_point x2 y2 →
  is_grid_point x3 y3 →
  ¬ is_equilateral x1 y1 x2 y2 x3 y3 := 
begin
  -- proof is not included
  sorry
end

end no_equilateral_triangle_on_grid_l586_586673


namespace probability_less_than_one_third_l586_586131

def interval := set.Ioo (0 : ℝ) (1 / 2)

def desired_interval := set.Ioo (0 : ℝ) (1 / 3)

theorem probability_less_than_one_third :
  (∃ p : ℝ, p = (volume desired_interval / volume interval) ∧ p = 2 / 3) :=
  sorry

end probability_less_than_one_third_l586_586131


namespace parabola_tangents_focus_area_l586_586522

theorem parabola_tangents_focus_area
  (p : ℝ) (h : p > 0)
  (K : ℝ × ℝ) (hK : K = (-4, 0))
  (A B : ℝ × ℝ)
  (tangentA : is_tangent_to_parabola A (λ x y, y^2 = 2 * p * x))
  (tangentB : is_tangent_to_parabola B (λ x y, y^2 = 2 * p * x))
  (AB_through_focus : passes_through_focus (line_through A B) (focus_of_parabola (λ x y, y^2 = 2 * p * x)))
  (area_triangle_KAB : area_of_triangle K A B = 24) :
  p = 4 :=
sorry

end parabola_tangents_focus_area_l586_586522


namespace chenny_friends_count_l586_586430

def initial_candies := 10
def additional_candies := 4
def candies_per_friend := 2

theorem chenny_friends_count :
  (initial_candies + additional_candies) / candies_per_friend = 7 := by
  sorry

end chenny_friends_count_l586_586430


namespace green_tractor_price_l586_586707

variable (S : ℕ) (r g R G : ℕ) (R_price G_price : ℝ)
variable (h1 : S = 7000)
variable (h2 : r = 2)
variable (h3 : g = 3)
variable (h4 : R_price = 20000)
variable (h5 : G_price = 5000)
variable (h6 : ∀ (x : ℝ), Tobias_earning_red : ℝ) (h7 : ∀ (y : ℝ), Tobias_earning_green : ℝ)

def earning_percentage_red : ℝ := 0.10
def earning_percentage_green : ℝ := 0.20

theorem green_tractor_price :
  S = Tobias_earning_red + Tobias_earning_green →
  Tobias_earning_red = (earning_percentage_red * R_price) * r →
  Tobias_earning_green = (earning_percentage_green * G_price) * g →
  G_price = 5000 :=
  sorry

end green_tractor_price_l586_586707


namespace min_cone_volume_l586_586392

theorem min_cone_volume : 
  ∃ (α : ℝ), (α > 0) ∧ (α < π / 4) ∧ 
  let R := Real.cot α,
      h := 2 / (1 - (Real.tan α) ^ 2),
      V := (2 * Real.pi / 3) * (1 / ((Real.tan α) ^ 2 * (1 - (Real.tan α) ^ 2))) 
  in V = (8 * Real.pi / 3) := 
sorry

end min_cone_volume_l586_586392


namespace exterior_angle_polygon_l586_586174

theorem exterior_angle_polygon (n : ℕ) (h : 360 / n = 72) : n = 5 :=
by
  sorry

end exterior_angle_polygon_l586_586174


namespace remaining_amount_to_be_paid_l586_586552

variable (d : ℝ) (p : ℝ)

theorem remaining_amount_to_be_paid {d : ℝ} {p : ℝ} (h₁ : d = 105) (h₂ : p = 0.10) : 
  let total_cost := d / p in
  let remaining_amount := total_cost - d in
  remaining_amount = 945 :=
by
  sorry

end remaining_amount_to_be_paid_l586_586552


namespace inclination_angle_range_l586_586305

theorem inclination_angle_range (θ α : ℝ) (h : ∃ α, ∀ θ, x * sin θ + (sqrt 3) * y + 2 = 0 ∧ tan α = - (sin θ / sqrt 3)) :
  α ∈ set.Icc 0 (π / 6) ∪ set.Ico (5 * π / 6) π :=
sorry

end inclination_angle_range_l586_586305


namespace variance_transformation_l586_586940

def variance (s : List ℝ) : ℝ := sorry -- Assume a variance function is defined elsewhere

theorem variance_transformation (x : List ℝ) (h : x.length = 8) 
  (hv : variance x = 3) :
  variance (x.map (λ a, 2 * a)) = 12 :=
by
  sorry

end variance_transformation_l586_586940


namespace find_5_star_10_l586_586529

section problem

variable (a b : ℝ)
variable (star : ℝ → ℝ → ℝ)

axiom A1 : ∀ a b > 0, (a star b) star b = a * (b star b)
axiom A2 : ∀ a > 0, (a star 1) star a = a star 1
axiom A3 : 1 star 1 = 2

theorem find_5_star_10 : 5 star 10 = 100 :=
by sorry

end problem

end find_5_star_10_l586_586529


namespace point_on_curve_l586_586400

-- Define the equation of the curve
def curve (x y : ℝ) := x^2 - x * y + 2 * y + 1 = 0

-- State that point (3, 10) satisfies the given curve equation
theorem point_on_curve : curve 3 10 :=
by
  -- this is where the proof would go but we will skip it for now
  sorry

end point_on_curve_l586_586400


namespace range_of_f_l586_586938

noncomputable def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

noncomputable def is_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop := a ≤ b → f a ≥ f b

theorem range_of_f (f : ℝ → ℝ) (h1 : is_even f) (h2 : is_decreasing f (-∞) 0) (h3 : f 2 = 0) : 
  ∀ x, f x < 0 ↔ -2 < x ∧ x < 2 :=
by
  sorry

end range_of_f_l586_586938


namespace circumradius_of_triangle_l586_586876

theorem circumradius_of_triangle
  (A B C : Type)
  (AB BC Area : ℝ)
  (AB_pos : AB = 2)
  (BC_pos : BC = 4)
  (Area_pos : Area = 2 * Real.sqrt 3)
  (is_acute : ∀ angle, 0 < angle ∧ angle < π / 2) : 
  let AC := Real.sqrt (AB^2 + BC^2 - 2 * AB * BC * (1 / 2)) in  -- Cos(B) = 1/2
  let R := (AC / (Real.sin (π / 3))) / 2 in  -- Sin(B) = sqrt(3)/2
  R = 2 :=
by
  sorry

end circumradius_of_triangle_l586_586876


namespace base_side_length_l586_586652

theorem base_side_length (area : ℝ) (slant_height : ℝ) (s : ℝ)
  (h_area : area = 50)
  (h_slant_height : slant_height = 20)
  (h_base_squared : ∃ b, b = s * s) :
  s = 5 :=
by 
  have h_triangle_area : ½ * s * slant_height = area :=
    calc
      ½ * s * slant_height = ½ * s * 20 : by rw h_slant_height
      ... = 10 * s : by norm_num
      ... = area    : by rw h_area
  have hs : s = 5 := by
    linarith
  sorry

end base_side_length_l586_586652


namespace probability_of_interval_l586_586143

theorem probability_of_interval {x : ℝ} (h₀ : 0 < x ∧ x < 1/2) :
    Pr (λ x, x < 1/3) = 2/3 :=
sorry

end probability_of_interval_l586_586143


namespace number_of_sides_l586_586176

-- Define the given conditions as Lean definitions

def exterior_angle := 72
def sum_of_exterior_angles := 360

-- Now state the theorem based on these conditions

theorem number_of_sides (n : ℕ) (h1 : exterior_angle = 72) (h2 : sum_of_exterior_angles = 360) : n = 5 :=
by
  sorry

end number_of_sides_l586_586176


namespace trucks_meeting_distance_l586_586326

theorem trucks_meeting_distance
  (speed1 speed2 : ℝ)
  (start_delay : ℝ)
  (distance head_start : ℝ)
  (speed1_eq : speed1 = 30)
  (speed2_eq : speed2 = 40)
  (start_delay_eq : start_delay = 12 / 60)
  (head_start_eq : head_start = speed1 * start_delay)
  (speeds_diff_eq : speed2 - speed1 = 10)
  (catchup_time_eq : head_start / 10 = 0.6)
  (distance_eq : speed2 * 0.6 = distance)
  : distance = 24 := 
by
  have speed1_val : speed1 = 30 := by rw [speed1_eq]
  have speed2_val : speed2 = 40 := by rw [speed2_eq]
  have start_delay_val : start_delay = 0.2 := by 
    rw [start_delay_eq]
    norm_num
  have head_start_val : head_start = 6 := by
    rw [head_start_eq, speed1_val, start_delay_val]
    norm_num
  have speeds_diff_val : speed2 - speed1 = 10 := by
    rw [speed2_val, speed1_val]
    norm_num
  have catchup_time_val: head_start / 10 = 0.6 := by
    rw [head_start_val]
    norm_num
  have distance_val : distance = 24 := by
    rw [distance_eq, speed2_val]
    norm_num
  exact distance_val

end trucks_meeting_distance_l586_586326


namespace complex_conjugate_problem_l586_586252

-- Define the imaginary unit.
def i := complex.I

-- Define the hypothesis condition.
def condition := ∃ z : ℂ, (2 * i / z = 1 - i)

-- Define the target statement.
def target_statement := ∀ (z : ℂ), (2 * i / z = 1 - i) → complex.conj z = -1 - i

-- Put everything together in a theorem statement.
theorem complex_conjugate_problem : target_statement :=
by sorry

end complex_conjugate_problem_l586_586252


namespace train_passes_through_tunnel_in_3_minutes_l586_586394

-- Definitions of the constants
def speed_of_train := 75 -- in mph
def length_of_tunnel := 3.5 -- in miles
def length_of_train := 0.25 -- in miles

-- Definition of the question
def time_to_pass_through_tunnel (speed : ℝ) (tunnel_length : ℝ) (train_length : ℝ) : ℝ :=
  let total_distance := tunnel_length + train_length
  let time_in_hours := total_distance / speed
  time_in_hours * 60 -- converting hours to minutes

-- Statement of the theorem to prove
theorem train_passes_through_tunnel_in_3_minutes :
  time_to_pass_through_tunnel speed_of_train length_of_tunnel length_of_train = 3 :=
by
  -- Actual proof would go here
  sorry

end train_passes_through_tunnel_in_3_minutes_l586_586394


namespace find_remainder_l586_586298

theorem find_remainder (a : ℕ) :
  (a ^ 100) % 73 = 2 ∧ (a ^ 101) % 73 = 69 → a % 73 = 71 :=
by
  sorry

end find_remainder_l586_586298


namespace theta_in_3rd_quadrant_l586_586478

noncomputable def theta_in_third_quadrant (theta : ℝ) : Prop :=
  (sin theta < 0) ∧ (tan theta > 0) → (π < theta) ∧ (theta < 3 * π / 2)

theorem theta_in_3rd_quadrant 
  (theta : ℝ) 
  (h₁ : sin theta < 0) 
  (h₂ : tan theta > 0) : 
  (π < theta) ∧ (theta < 3 * π / 2) :=
by sorry

end theta_in_3rd_quadrant_l586_586478


namespace convert_angle_l586_586810

theorem convert_angle (deg_angle : ℝ) (k : ℤ) (alpha : ℝ) : 
  deg_angle = -765 ∧ deg_angle = (2 * k) * Real.pi + alpha ∧ 0 ≤ alpha ∧ alpha < 2 * Real.pi → 
  alpha = (7 * Real.pi / 4) ∧ k = -3 :=
begin
  intros h,
  sorry
end

end convert_angle_l586_586810


namespace line_parallel_through_point_line_perpendicular_through_point_line_with_equal_intercepts_through_point_l586_586463

-- Define the line passing through a given point and parallel to another line
theorem line_parallel_through_point 
  (p : Point) (line : Line) (a b c : ℝ) (hp : p = (-1,3)) (hline : line = Equation (Coeff a b c) 
  (a = 1) (b = -2) (c = 3)) : Line := 
Proof
  sorry

-- Define the line passing through a given point and perpendicular to another line
theorem line_perpendicular_through_point 
  (p : Point) (line : Line) (a b c : ℝ) (hp : p = (3,4)) (hline : line = Equation (Coeff a b c) 
  (a = 3) (b = -1) (c = 2)) : Line := 
Proof
  sorry

-- Define the line passing through a given point with equal intercepts on both axes
theorem line_with_equal_intercepts_through_point 
  (p : Point) (intercept : ℝ) (hp : p = (1,2)) (hintercept : intercept = intercept): Line := 
Proof
  sorry

noncomputable def Point : Type* := ℝ × ℝ

noncomputable def Coeff (a b c : ℝ) : Type := ∀ (x y z: ℤ), a * x + b * y + c * z = 0

noncomputable def Equation (c : Coeff) : Type := Refl c

end line_parallel_through_point_line_perpendicular_through_point_line_with_equal_intercepts_through_point_l586_586463


namespace custom_operation_evaluation_l586_586929

theorem custom_operation_evaluation (m n : ℤ) : m * n = m - n + 1 := sorry

lemma calculate_custom_operation : ((2 * 3) * 2) = -1 :=
by {
  have h1 : (2 * 3) = 2 - 3 + 1 := by rw custom_operation_evaluation,
  rw h1,
  have h2 : (0 * 2) = 0 - 2 + 1 := by rw custom_operation_evaluation,
  rw h2,
  exact rfl
}

end custom_operation_evaluation_l586_586929


namespace num_divisors_360_l586_586918

theorem num_divisors_360 :
  ∀ n : ℕ, n = 360 → (∀ (p q r : ℕ), p = 2 ∧ q = 3 ∧ r = 5 →
    (∃ (a b c : ℕ), 360 = p^a * q^b * r^c ∧ a = 3 ∧ b = 2 ∧ c = 1) →
    (3+1) * (2+1) * (1+1) = 24) :=
  sorry

end num_divisors_360_l586_586918


namespace range_of_S_l586_586508

-- The function representing the area of triangle ABC in terms of m.
def S (a : ℝ) (m : ℝ) : ℝ :=
  1 / 2 * log a (1 + 4 / (m * (m + 4)))

-- The main theorem that we need to prove.
theorem range_of_S (a : ℝ) (h : 1 < a) : 
  ∀ m : ℝ, 1 < m →
  ∃ l k : ℝ, 
    l = (0 : ℝ) ∧ k = (1 / 2 * log a (9 / 5)) ∧ 
    (∀ ε > 0, ∃ δ > 0, abs (S a m - l) < ε) ∧
    (∀ ε > 0, ∃ δ > 0, abs (S a (m + δ) - k) < ε) :=
sorry

end range_of_S_l586_586508


namespace prove_hyperbola_propositions_l586_586510

noncomputable def hyperbola_trajectory (x y : ℝ) : Prop :=
  (y / (x + 3)) * (y / (x - 3)) = (16 / 9)

theorem prove_hyperbola_propositions (M : ℝ × ℝ)
  (cond : hyperbola_trajectory M.1 M.2) :
  ((M.1 = -5 ∧ M.2 = 0) ∨ (M.1 = 5 ∧ M.2 = 0))
  ∧ (¬ x < 0 → M.1 = -3) :=
sorry

end prove_hyperbola_propositions_l586_586510


namespace probability_less_than_one_third_l586_586128

def interval := set.Ioo (0 : ℝ) (1 / 2)

def desired_interval := set.Ioo (0 : ℝ) (1 / 3)

theorem probability_less_than_one_third :
  (∃ p : ℝ, p = (volume desired_interval / volume interval) ∧ p = 2 / 3) :=
  sorry

end probability_less_than_one_third_l586_586128


namespace max_perimeter_value_l586_586957

-- Definitions of the conditions
def A : ℝ := Real.pi / 3
def BC : ℝ := 2 * Real.sqrt 3
def B : ℝ := x
def perimeter (x : ℝ) : ℝ := 4 * Real.sin x + 4 * Real.sin (2 * Real.pi / 3 - x) + 2 * Real.sqrt 3

-- Lean statement for the maximum value problem
theorem max_perimeter_value:
  ∃ x ∈ Ioo 0 (2 * Real.pi / 3), perimeter x = 6 * Real.sqrt 3 :=
sorry

end max_perimeter_value_l586_586957


namespace integral_f_eq_l586_586049

def f (x : ℝ) : ℝ :=
  if (0 < x ∧ x ≤ 1) then sqrt (1 - x^2)
  else if (-1 ≤ x ∧ x ≤ 0) then x + 1
  else 0

theorem integral_f_eq : ∫ x in -1..1, f x = (1 + Real.pi) / 4 :=
by
  sorry

end integral_f_eq_l586_586049


namespace harry_angular_momentum_l586_586903

theorem harry_angular_momentum 
  (r : ℝ) (m : ℝ) (g_force : ℝ) (g : ℝ) 
  (a_R : ℝ) (I : ℝ) (ω : ℝ) (L : ℝ) 
  (hrad : r = 2.0)
  (hmass : m = 50.0)
  (hg : g = 9.8)
  (hg_force : g_force = 5.0)
  (ha_R : a_R = g_force * g)
  (hω : ω^2 * r = a_R)
  (hI : I = m * r^2)
  (hL : L = I * ω) :
  L = 1000 := 
begin
  -- Proof steps are skipped
  sorry
end

end harry_angular_momentum_l586_586903


namespace original_proposition_true_converse_false_l586_586058

-- Lean 4 statement for the equivalent proof problem
theorem original_proposition_true_converse_false (a b : ℝ) : 
  (a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1)) ∧ ¬((a ≥ 1 ∨ b ≥ 1) → a + b ≥ 2) :=
sorry

end original_proposition_true_converse_false_l586_586058


namespace tangent_line_at_zero_l586_586843

noncomputable def curve := λ x : ℝ, Real.exp x + 3 * x

theorem tangent_line_at_zero :
  ∃ (m b : ℝ), (m = 4) ∧ (b = 1) ∧ (∀ x : ℝ, tangent_line curve 0 x = m * x + b) :=
begin
  sorry
end

def tangent_line (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ :=
  f a + (derivative f a) * (x - a)

end tangent_line_at_zero_l586_586843


namespace probability_less_than_third_l586_586111

def interval_real (a b : ℝ) := set.Ioo a b

def probability_space (a b : ℝ) := {
  length := b - a,
  sample_space := interval_real a b
}

def probability_of_event {a b c : ℝ} (h : a < b) (h1 : a < c) (h2 : c < b) : Prop :=
  (c - a) / (b - a)

theorem probability_less_than_third :
  probability_of_event (by norm_num : (0:ℝ) < (1 / 2))
                       (by norm_num : (0:ℝ) < (1 / 3))
                       (by norm_num : (1 / 3) < (1 / 2))
  = (2 / 3) := sorry

end probability_less_than_third_l586_586111


namespace cot_30_eq_sqrt3_l586_586412

theorem cot_30_eq_sqrt3 (h : Real.tan (Real.pi / 6) = 1 / Real.sqrt 3) : Real.cot (Real.pi / 6) = Real.sqrt 3 := by
  sorry

end cot_30_eq_sqrt3_l586_586412


namespace range_of_a_if_inequality_holds_for_any_x_l586_586557

theorem range_of_a_if_inequality_holds_for_any_x :
  (∀ x : ℝ, 0 < x ∧ x < 3 → x^2 - a * x + 4 ≥ 0) → a ≤ 4 :=
begin
  sorry
end

end range_of_a_if_inequality_holds_for_any_x_l586_586557


namespace area_of_segment_solution_max_sector_angle_solution_l586_586486
open Real

noncomputable def area_of_segment (α R : ℝ) : ℝ :=
  let l := (R * α)
  let sector := 0.5 * R * l
  let triangle := 0.5 * R^2 * sin α
  sector - triangle

theorem area_of_segment_solution : area_of_segment (π / 3) 10 = 50 * ((π / 3) - (sqrt 3 / 2)) :=
by sorry

noncomputable def max_sector_angle (c : ℝ) (hc : c > 0) : ℝ :=
  2

theorem max_sector_angle_solution (c : ℝ) (hc : c > 0) : max_sector_angle c hc = 2 :=
by sorry

end area_of_segment_solution_max_sector_angle_solution_l586_586486


namespace part_i_part_ii_part_iii_l586_586051

section ProofProblem

variable {a x t : ℝ} (f : ℝ → ℝ) (a_equation : f(x) = (x + a - 1) / (x^2 + 1))
variable (odd_f : ∀ x, x ∈ Set.Icc (-1) 1 → f(-x) = -f(x))
variable (a_is_one : a = 1)

-- Define f using the derived value of a
def g (x : ℝ) : ℝ := (x + 1 - 1) / (x^2 + 1)

-- (Ⅰ) Prove a = 1
theorem part_i : a = 1 :=
  a_is_one

-- (Ⅱ) Prove f(x) is increasing on [-1, 1]
theorem part_ii (increasing_f : ∀ x1 x2, x1, x2 ∈ Set.Icc (-1) 1 → x1 < x2 → g(x1) < g(x2)) :
  ∀ x1 x2, x1, x2 ∈ Set.Icc (-1) 1 → x1 < x2 → f(x1) < f(x2) :=
  by
  sorry

-- (Ⅲ) Find t's range such that f(t-1) + f(2t) < 0
theorem part_iii (f_inequality : ∀ t, 0 ≤ t ∧ t < 1/3 → g(t-1) + g(2*t) < 0) :
  ∀ t, 0 ≤ t ∧ t < 1/3 → f(t-1) + f(2*t) < 0 :=
  by
  sorry

end ProofProblem

end part_i_part_ii_part_iii_l586_586051


namespace conner_day3_rocks_to_tie_l586_586285

def initial_rocks_sydney : ℕ := 837
def initial_rocks_conner : ℕ := 723
def day1_rocks_sydney : ℕ := 4
def day1_rocks_conner : ℕ := 8 * day1_rocks_sydney
def day2_rocks_sydney : ℕ := 0
def day2_rocks_conner : ℕ := 123
def day3_rocks_sydney : ℕ := 2 * day1_rocks_conner

theorem conner_day3_rocks_to_tie :
  ∃ x : ℕ, initial_rocks_conner + day1_rocks_conner + day2_rocks_conner + x = 
            initial_rocks_sydney + day1_rocks_sydney + day2_rocks_sydney + day3_rocks_sydney :=
begin
  use 27,
  sorry
end

end conner_day3_rocks_to_tie_l586_586285


namespace tangent_parabola_points_l586_586675

theorem tangent_parabola_points (a b : ℝ) (h_circle : a^2 + b^2 = 1) (h_discriminant : a^2 - 4 * b * (b - 1) = 0) :
    (a = 0 ∧ b = 1) ∨ 
    (a = 2 * Real.sqrt 6 / 5 ∧ b = -1 / 5) ∨ 
    (a = -2 * Real.sqrt 6 / 5 ∧ b = -1 / 5) := sorry

end tangent_parabola_points_l586_586675


namespace sparrows_and_swallows_equations_l586_586194

-- Definitions for the weights
def weight_sparrow (x : ℝ) (y : ℝ) : Prop := 5 * x + 6 * y = 16
def exchange_condition (x : ℝ) (y : ℝ) : Prop := 4 * x + y = 5 * y + x

-- The theorem statement
theorem sparrows_and_swallows_equations (x y : ℝ) :
  weight_sparrow x y ∧ exchange_condition x y ↔ (5 * x + 6 * y = 16) ∧ (4 * x + y = 5 * y + x) :=
by
  apply Iff.intro
  assumption
  sorry

end sparrows_and_swallows_equations_l586_586194


namespace integral_of_derivative_eq_l586_586728

theorem integral_of_derivative_eq (f : ℝ → ℝ) (a b : ℝ)
  (h1 : ContinuousOn f (Set.Icc a b))
  (h2 : IntegrableOn (deriv f) (Set.Icc a b)) :
  (∫ x in a..b, deriv f x) = f b - f a :=
sorry

end integral_of_derivative_eq_l586_586728


namespace least_integer_x_l586_586464

theorem least_integer_x (x : ℤ) (h : 3 * |x| - 2 * x + 8 < 23) : x = -3 :=
sorry

end least_integer_x_l586_586464


namespace false_proposition_among_ABCD_l586_586927

theorem false_proposition_among_ABCD :
  (∃ x : ℝ, log x = 0) ∧
  (∃ x : ℝ, tan x = 1) ∧
  ∀ x : ℝ, ¬ (x ^ 3 > 0) ∧
  ∀ x : ℝ, 2 ^ x > 0 :=
begin
  sorry
end

end false_proposition_among_ABCD_l586_586927


namespace sequence_100th_term_l586_586648

/-- Define the sequence where the number k appears k times -/
def sequence : ℕ → ℕ
| n :=
  let ⟨k, _, _⟩ := Nat.exists_eq_add_of_le (λ m, (m * (m + 1)) / 2) n
  in k + 1

theorem sequence_100th_term : sequence 100 = 14 := 
by 
  -- To explain why the sequence's 100th term evaluates to 14, we use the properties of the sequence.
  -- We'll break it down by identifying how the sequence is structured.
  -- Sequence structure: 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, ...
  -- Here, we use the sum of the first k natural numbers up to 13, which sum to 91 terms.
  -- The next 14 terms are all '14's.
  -- Hence, the 100th term is 14.
  sorry

end sequence_100th_term_l586_586648


namespace concurrent_or_parallel_l586_586971

-- Definitions for the conditions
variables (A B C A' B' C' H : Type)

-- Triangles ABC and A'B'C' are similar
axiom similar_triangles : similar (triangle A B C) (triangle A' B' C')

-- Triangles ABC and A'B'C' have different orientations
axiom different_orientation : ¬ same_orientation (triangle A B C) (triangle A' B' C')

-- Orthocenters of ABC and A'B'C' coincide
axiom common_orthocenter : orthocenter (triangle A B C) = orthocenter (triangle A' B' C')

-- Prove that lines AA', BB', and CC' are concurrent or parallel
theorem concurrent_or_parallel : concurrent_or_parallel (line A A') (line B B') (line C C') :=
sorry

end concurrent_or_parallel_l586_586971


namespace no_solution_for_k_eq_six_l586_586000

theorem no_solution_for_k_eq_six :
  ∀ x k : ℝ, k = 6 → (x ≠ 2 ∧ x ≠ 7) → (x - 1) / (x - 2) = (x - k) / (x - 7) → false :=
by 
  intros x k hk hnx_eq h_eq
  sorry

end no_solution_for_k_eq_six_l586_586000


namespace analytical_expression_for_f_l586_586011

noncomputable def f : ℝ → Prop := λ x, x ≠ -1

theorem analytical_expression_for_f (x : ℝ) : f x = (x ≠ -1) :=
sorry

end analytical_expression_for_f_l586_586011


namespace particle_paths_count_l586_586761

theorem particle_paths_count :
  let paths (start finish : (ℕ × ℕ)) (moves : ((ℕ × ℕ) → (ℕ × ℕ) → Prop)) :=
    {p : list (ℕ × ℕ) // p.head = start ∧ p.last = finish ∧ ∀ i < p.length - 1, moves (p.get i) (p.get (i + 1)) }.card =
      963 :=
  sorry

end particle_paths_count_l586_586761


namespace pages_with_same_units_digit_count_l586_586382

theorem pages_with_same_units_digit_count {n : ℕ} (h1 : n = 67) :
  ∃ k : ℕ, k = 13 ∧
  (∀ x : ℕ, 1 ≤ x ∧ x ≤ n → 
    (x ≡ (n + 1 - x) [MOD 10] ↔ 
     (x % 10 = 4 ∨ x % 10 = 9))) :=
by
  sorry

end pages_with_same_units_digit_count_l586_586382


namespace firework_height_l586_586446

noncomputable def initial_velocity := 90 -- m/s
noncomputable def total_time := 5 -- seconds
noncomputable def speed_of_sound := 340 -- m/s
noncomputable def gravity := 9.8 -- m/s^2

theorem firework_height :
  let x := (total_time - sqrt((total_time * total_time * speed_of_sound * speed_of_sound) - (2 * total_time * speed_of_sound * initial_velocity)) / (gravity + speed_of_sound)) in
  let height := initial_velocity * x - 0.5 * gravity * x * x in
  height = 289 := 
sorry

end firework_height_l586_586446


namespace probability_sum_gt_seven_l586_586325

theorem probability_sum_gt_seven : 
  let outcomes := [(i, j) | i in Finset.range 1 (6+1), j in Finset.range 1 (6+1)] in
  let favorable := Finset.filter (λ (p : ℕ × ℕ), p.1 + p.2 > 7) (Finset.product (Finset.range 1 (6 + 1)) (Finset.range 1 (6 + 1))) in
  (favorable.card : ℚ) / (outcomes.card : ℚ) = 5 / 12 := 
by sorry

end probability_sum_gt_seven_l586_586325


namespace fourth_equation_pattern_l586_586633

theorem fourth_equation_pattern :
  36^2 + 37^2 + 38^2 + 39^2 + 40^2 = 41^2 + 42^2 + 43^2 + 44^2 :=
by
  sorry

end fourth_equation_pattern_l586_586633


namespace determine_c_value_l586_586984

theorem determine_c_value (a b c x y : ℝ)
    (h1 : a * x + b * y = c - 1)
    (h2 : (a + 5) * x + (b + 3) * y = c + 1) :
    c = (2 * a + 5) / 5 :=
begin
    sorry, -- This is where the proof would go
end

end determine_c_value_l586_586984


namespace find_a_b_find_k_l586_586054

noncomputable def g (x a b : ℝ) : ℝ :=
  a * x^2 - 2 * a * x + 1 + b

def f (x : ℝ) (a b : ℝ) : ℝ :=
  g x a b / x

theorem find_a_b
  (a : ℝ) (b : ℝ)
  (h_max : g 3 a b = 4)
  (h_min : g 2 a b = 1)
  (h_a_pos : 0 < a) :
  a = 2 / 3 ∧ b = 1 / 3 := sorry

theorem find_k
  (f : ℝ → ℝ)
  (a : ℝ := 2 / 3)
  (b : ℝ := 1 / 3)
  (h_f : f = λ x, g x a b / x)
  (h_k : ∀ x ∈ set.Icc (-1:ℝ) 1, f (2^x) - k * 2^x ≥ 0) :
  k ∈ set.Iic (1 : ℝ) := sorry

end find_a_b_find_k_l586_586054


namespace E_is_subspace_of_R3X_l586_586060

-- Define the set E as specified in the problem
def E : set (polynomial ℝ) := {P : polynomial ℝ | degree P ≤ 3 ∧ P.eval 1 = 0}

-- Define the set of polynomials of degree ≤ 3 as R3X
def R3X : submodule ℝ (polynomial ℝ) := {
  carrier := {P : polynomial ℝ | degree P ≤ 3},
  zero_mem' := by {
    sorry
  },
  add_mem' := by {
    sorry
  },
  smul_mem' := by {
    sorry
  }
}

-- Prove that E is a subspace of R3X
theorem E_is_subspace_of_R3X : submodule ℝ (polynomial ℝ) :=
{
  carrier := E,
  zero_mem' := by {
    show (0 : polynomial ℝ) ∈ E,
    rw [set.mem_set_of_eq, polynomial.eval_zero],
    split,
    { exact polynomial.degree_zero.le },
    { refl }
  },
  add_mem' := by {
    intros P Q hP hQ,
    rw [set.mem_set_of_eq] at *,
    split,
    { exact polynomial.degree_add_le _ _ },
    { rw [polynomial.eval_add, hP.2, hQ.2, add_zero] }
  },
  smul_mem' := by {
    intros c P hP,
    rw [set.mem_set_of_eq] at *,
    split,
    { exact polynomial.degree_smul_le _ _ },
    { rw [polynomial.eval_smul, hP.2, mul_zero] }
  }
}

end E_is_subspace_of_R3X_l586_586060


namespace probability_merlin_dismissed_l586_586217

noncomputable def merlin_dismissal_probability {p : ℝ} (hp : 0 ≤ p ∧ p ≤ 1) : ℝ :=
  let q := 1 - p in
  -- Assuming the coin flip is fair, the probability Merlin is dismissed
  1 / 2

theorem probability_merlin_dismissed (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  merlin_dismissal_probability hp = 1 / 2 :=
by
  -- Placeholder for the proof
  sorry

end probability_merlin_dismissed_l586_586217


namespace proof_goal_l586_586043

open BigOperators

-- Define the condition that sum of all binomial coefficients in the expansion is 256
def sum_of_binomial_coefficients_is_256 (n : ℕ) : Prop :=
  2^n = 256

-- Define the property that the term with the largest binomial coefficient is the 5th term (middle term for n=8)
def term_with_largest_binomial_coefficient_is_5th (n : ℕ) : Prop :=
  n = 8 ∧ 5 = n / 2 + 1

-- Define the property that the sum of all coefficients in the expansion is 1
def sum_of_all_coefficients_is_1 (n : ℕ) (x : ℝ) : Prop :=
  (x - (2 / sqrt x))^n = 1

-- The final proof goal
theorem proof_goal : ∃ n : ℕ, sum_of_binomial_coefficients_is_256 n ∧ term_with_largest_binomial_coefficient_is_5th n ∧
  ∀ x : ℝ, sum_of_all_coefficients_is_1 n x :=
by
  sorry

end proof_goal_l586_586043


namespace interval_probability_l586_586148

theorem interval_probability (x y z : ℝ) (h0x : 0 < x) (hy : y > 0) (hz : z > 0) (hx : x < y) (hy_mul_z_eq_half : y * z = 1 / 2) (hx_eq_third_y : x = 1 / 3 * y) :
  (x / y) = (2 / 3) :=
by
  -- Here you should provide the proof
  sorry

end interval_probability_l586_586148


namespace perimeter_of_square_B_l586_586277

theorem perimeter_of_square_B
  (perimeter_A : ℝ)
  (h_perimeter_A : perimeter_A = 36)
  (area_ratio : ℝ)
  (h_area_ratio : area_ratio = 1 / 3)
  : ∃ (perimeter_B : ℝ), perimeter_B = 12 * Real.sqrt 3 :=
by
  sorry

end perimeter_of_square_B_l586_586277


namespace triangle_area_ratio_l586_586199

theorem triangle_area_ratio {A B C : ℝ} {a b c : ℝ} 
  (h : 2 * Real.sin A * Real.cos (B - C) + Real.sin (2 * A) = 2 / 3) 
  (S1 : ℝ) (S2 : ℝ) :
  S1 / S2 = 1 / (3 * Real.pi) :=
sorry

end triangle_area_ratio_l586_586199


namespace probability_merlin_dismissed_l586_586209

variable {p q : ℝ} (hpq : p + q = 1) (hp : 0 < p ∧ p < 1)

theorem probability_merlin_dismissed : 
  let decision_prob : ℝ := if h : p = 1 then 1 else p in
  if h : decision_prob = p then (1 / 2) else 0 := 
begin
  sorry
end

end probability_merlin_dismissed_l586_586209


namespace find_arith_seq_sum_l586_586579

noncomputable def arith_seq_sum : ℕ → ℕ → ℕ
| 0, d => 2
| (n+1), d => arith_seq_sum n d + d

theorem find_arith_seq_sum :
  ∃ d : ℕ, 
    arith_seq_sum 1 d + arith_seq_sum 2 d = 13 ∧
    arith_seq_sum 3 d + arith_seq_sum 4 d + arith_seq_sum 5 d = 42 :=
by
  sorry

end find_arith_seq_sum_l586_586579


namespace quartic_polynomial_unique_solution_l586_586466

noncomputable def p (x : ℝ) : ℝ := -x^4 + 2 * x^2 - 5 * x + 1

theorem quartic_polynomial_unique_solution :
  p 1 = -3 ∧ p 2 = -5 ∧ p 3 = -11 ∧ p 4 = -27 ∧ p 5 = -59 :=
by
  have h1 : p 1 = -3 := by
    calc
      p 1 = -1^4 + 2 * 1^2 - 5 * 1 + 1 : rfl
         ... = -1 + 2 - 5 + 1 : by norm_num
         ... = -3 : rfl
  have h2 : p 2 = -5 := by
    calc
      p 2 = -2^4 + 2 * 2^2 - 5 * 2 + 1 : rfl
         ... = -16 + 8 - 10 + 1 : by norm_num
         ... = -5 : rfl
  have h3 : p 3 = -11 := by
    calc
      p 3 = -3^4 + 2 * 3^2 - 5 * 3 + 1 : rfl
         ... = -81 + 18 - 15 + 1 : by norm_num
         ... = -11 : rfl
  have h4 : p 4 = -27 := by
    calc
      p 4 = -4^4 + 2 * 4^2 - 5 * 4 + 1 : rfl
         ... = -256 + 32 - 20 + 1 : by norm_num
         ... = -27 : rfl
  have h5 : p 5 = -59 := by
    calc
      p 5 = -5^4 + 2 * 5^2 - 5 * 5 + 1 : rfl
         ... = -625 + 50 - 25 + 1 : by norm_num
         ... = -59 : rfl
  exact ⟨h1, h2, h3, h4, h5⟩

end quartic_polynomial_unique_solution_l586_586466


namespace one_circle_contains_center_of_another_l586_586275

noncomputable def circles_contain_center (r : ℝ) (c : ℕ → ℝ × ℝ) := 
  ∀ i, ∃ t, (t ∈ (set.range c)) ∧ (∥t - c i∥ < r)

theorem one_circle_contains_center_of_another:
  let c := λ n : fin 6, (n : ℝ, n : ℝ) in
  let r := 1 in
  circles_contain_center r c := by 
    sorry

end one_circle_contains_center_of_another_l586_586275


namespace exists_lattice_point_l586_586235

-- Definitions for convexity, symmetry, and area are assumed to be part of Lean's Mathlib
-- Minkowski's theorem is also assumed to be a known fact within the context of Lean's library

variable {R : Type} [convex R] [symmetrical_about_origin R] [measure_space R]

theorem exists_lattice_point (h_convex : is_convex R) 
                            (h_symmetric : is_symmetric R (0 : ℝ × ℝ)) 
                            (h_area : measure_area R > 4) : 
  ∃ p : ℝ × ℝ, p ≠ (0, 0) ∧ is_lattice_point p :=
by {
  sorry -- Proof follows from Minkowski's theorem
}

end exists_lattice_point_l586_586235


namespace triangle_area_le_half_parallelogram_area_l586_586264

theorem triangle_area_le_half_parallelogram_area 
    (P Q R S A B C : Point) 
    (parallelogram : Parallelogram P Q R S) 
    (vertex_on_side1 : IsPointOnSide A parallelogram)
    (vertex_on_side2 : IsPointOnSide B parallelogram)
    (vertex_on_side3 : IsPointOnSide C parallelogram) :
    AreaTriangle A B C ≤ 1/2 * AreaParallelogram P Q R S :=
by
  sorry

end triangle_area_le_half_parallelogram_area_l586_586264


namespace sum_of_distances_A_D_B_between_21_and_22_l586_586949

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A : ℝ × ℝ := (20, 0)
def D : ℝ × ℝ := (6, 6)

def AD : ℝ := distance A D
def BD : ℝ := distance D A

theorem sum_of_distances_A_D_B_between_21_and_22 :
  21 < AD + BD ∧ AD + BD < 22 :=
by
  sorry

end sum_of_distances_A_D_B_between_21_and_22_l586_586949


namespace exterior_angle_polygon_l586_586172

theorem exterior_angle_polygon (n : ℕ) (h : 360 / n = 72) : n = 5 :=
by
  sorry

end exterior_angle_polygon_l586_586172


namespace grazing_months_for_b_l586_586354

/-
  We define the problem conditions and prove that b put his oxen for grazing for 5 months.
-/

theorem grazing_months_for_b (x : ℕ) :
  let a_oxen := 10
  let a_months := 7
  let b_oxen := 12
  let c_oxen := 15
  let c_months := 3
  let total_rent := 210
  let c_share := 54
  let a_ox_months := a_oxen * a_months
  let b_ox_months := b_oxen * x
  let c_ox_months := c_oxen * c_months
  let total_ox_months := a_ox_months + b_ox_months + c_ox_months
  (c_share : ℚ) / total_rent = (c_ox_months : ℚ) / total_ox_months →
  x = 5 :=
by
  sorry

end grazing_months_for_b_l586_586354


namespace no_one_signs_up_for_aviation_modeling_l586_586073

/-- There are 4 students and 4 interest groups (math, physics, computer, and aviation modeling).
    Each student signs up for exactly 1 group. Prove that there are 36 ways in which no one signs up
    for the aviation modeling group. -/
theorem no_one_signs_up_for_aviation_modeling :
  let students := 4 in
  let groups := 4 in
  let groups_ex_avation := 3 in
  ∀ (signups : Fin students → Fin groups), 
    (∀ student, signups student ≠ 3) →
    ∃ (combinations : Nat), combinations = 36 :=
by
  sorry

end no_one_signs_up_for_aviation_modeling_l586_586073


namespace great_grandson_age_l586_586376

theorem great_grandson_age (n : ℕ) : 
  ∃ n, (n * (n + 1)) / 2 = 666 :=
by
  -- Solution steps would go here
  sorry

end great_grandson_age_l586_586376


namespace find_two_digit_number_l586_586570

-- Define the problem conditions and statement
theorem find_two_digit_number (a b n : ℕ) (h1 : a = 2 * b) (h2 : 10 * a + b + a^2 = n^2) : 
  10 * a + b = 21 :=
sorry

end find_two_digit_number_l586_586570


namespace range_of_k_l586_586521

theorem range_of_k (k : ℝ) : 
  (∃ (x1 x2 : ℝ), (3 * x1^2 - (kx1 + 1)^2 = 3) ∧ (3 * x2^2 - (kx2 + 1)^2 = 3) ∧ x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0) 
  ↔ (-2 < k) ∧ (k < -sqrt 3) :=
by
  sorry

end range_of_k_l586_586521


namespace probability_less_than_one_third_l586_586096

theorem probability_less_than_one_third : 
  (∃ p : ℚ, p = (measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ)) (set.Ioo (0 : ℚ) (1 / 3 : ℚ)) / 
                  measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ))) ∧ p = 2 / 3) :=
by {
  -- The proof will be added here.
  sorry
}

end probability_less_than_one_third_l586_586096


namespace solve_x_l586_586850

theorem solve_x (x : ℝ) (h : (real.sqrt (2 * x + 7)) / (real.sqrt (4 * x + 7)) = (real.sqrt 7) / 2) : x = -21 / 20 :=
sorry

end solve_x_l586_586850


namespace proof_problem_l586_586196

-- Definitions of the sides, angles, and vectors
variables {a b c : ℝ} (A B C : ℝ)
def is_triangle (ABC : Prop) := 
  ABC = (a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ A < π ∧ B > 0 ∧ B < π ∧ C > 0 ∧ C < π ∧ A + B + C = π)

-- Given condition a^2 + b^2 - c^2 = ab
def given_condition := a^2 + b^2 - c^2 = ab

-- Collinearity of vectors
def vector_a := (2 * sin A, 1)
def vector_b := (cos C, 1 / 2)
def collinear := vector_a.1 * vector_b.2 = vector_a.2 * vector_b.1

-- Theorem to find measure of angle C
noncomputable def find_angle_C : Prop :=
  given_condition → cos C = 1 / 2 ∧ C = π / 3

-- Theorem to determine the shape of the triangle
noncomputable def determine_triangle_shape : Prop :=
  collinear → is_triangle A B C → A = π / 6 ∧ B = π / 2 ∧ C = π / 3 ∧ (A + B + C = π) ∧ (B = π / 2)

-- Stating both proofs
theorem proof_problem :
  (given_condition → cos C = 1 / 2 ∧ C = π / 3) ∧
  (collinear → is_triangle A B C → A = π / 6 ∧ B = π / 2 ∧ C = π / 3 ∧ (A + B + C = π) ∧ (B = π / 2)) := 
sorry

end proof_problem_l586_586196


namespace log_xy_gt_two_l586_586477

open real -- Assumes a logarithm base is real for simplicity

variables {a x y : ℝ}
hypotheses (h1 : 0 < x)
           (h2 : x < y)
           (h3 : y < 1)
           (ha : 1 < a)

theorem log_xy_gt_two : log a (x * y) > 2 := sorry

end log_xy_gt_two_l586_586477


namespace ratio_AC_AB_constant_l586_586324

-- Define the points, and circles' radii and centers
variables (O1 O2 A B C : Type) [point A] [point B] [point C]
variable [real R1 R2 : ℝ]

-- Assumptions
variable (h1 : ext_circle_touch A O1 R1)
variable (h2 : ext_circle_touch A O2 R2)
variable (secant : secant_through_A B C)

-- Theorem statement
theorem ratio_AC_AB_constant : 
  (AC / AB) = (R1 / R2) :=
sorry

end ratio_AC_AB_constant_l586_586324


namespace craig_distance_ridden_farther_l586_586320

/-- Given that Craig rode the bus for 3.83 miles and walked for 0.17 miles,
    prove that the distance he rode farther than he walked is 3.66 miles. -/
theorem craig_distance_ridden_farther :
  let distance_bus := 3.83
  let distance_walked := 0.17
  distance_bus - distance_walked = 3.66 :=
by
  let distance_bus := 3.83
  let distance_walked := 0.17
  show distance_bus - distance_walked = 3.66
  sorry

end craig_distance_ridden_farther_l586_586320


namespace placing_balls_in_boxes_l586_586926

theorem placing_balls_in_boxes :
  let balls := 7
  let boxes := 3
  (boxes^balls = 2187) := 
begin
    sorry
end

end placing_balls_in_boxes_l586_586926


namespace geometric_series_sum_l586_586802

-- Define the geometric series
def geometricSeries (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r ^ n) / (1 - r)

-- Define the conditions
def a : ℚ := 1 / 4
def r : ℚ := 1 / 4
def n : ℕ := 5

-- Define the sum of the first n terms using the provided formula
def S_n := geometricSeries a r n

-- State the theorem: the sum S_5 equals the given answer
theorem geometric_series_sum :
  S_n = 1023 / 3072 :=
by
  sorry

end geometric_series_sum_l586_586802


namespace neznaika_expression_l586_586257

theorem neznaika_expression (a b c : ℝ) (h1 : a = 20) (h2 : b = 2) (h3 : c = 2) :
  ∃ (x : ℝ), x = a / (b - sqrt c) ∧ x > 30 := 
by
  sorry

end neznaika_expression_l586_586257


namespace geometric_probability_l586_586100

theorem geometric_probability :
  let total_length := (1/2 - 0)
  let favorable_length := (1/3 - 0)
  total_length > 0 ∧ favorable_length > 0 →
  (favorable_length / total_length) = 2 / 3 :=
by
  intros _ _ _
  sorry

end geometric_probability_l586_586100


namespace find_max_min_find_angle_C_l586_586501

open Real

noncomputable def f (x : ℝ) : ℝ :=
  12 * sin (x + π / 6) * cos x - 3

theorem find_max_min (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 4) :
  let fx := f x 
  (∀ a, a = abs (fx - 6)) -> (∀ b, b = abs (fx - 3)) -> fx = 6 ∨ fx = 3 := sorry

theorem find_angle_C (AC BC CD : ℝ) (hAC : AC = 6) (hBC : BC = 3) (hCD : CD = 2 * sqrt 2) :
  ∃ C : ℝ, C = π / 2 := sorry

end find_max_min_find_angle_C_l586_586501


namespace cost_of_kerosene_in_cents_l586_586188

-- Define the conditions
def cost_per_pound_of_rice : ℝ := 0.36
def cost_per_dozen_eggs : ℝ := cost_per_pound_of_rice
def cost_per_half_liter_kerosene_in_eggs : ℝ := (8 / 12) * cost_per_dozen_eggs  -- Cost of 8 eggs

-- Statement of the problem to be proven
theorem cost_of_kerosene_in_cents :
  let cost_per_half_liter_kerosene := cost_per_half_liter_kerosene_in_eggs in
  let cost_per_liter_kerosene := 2 * cost_per_half_liter_kerosene in
  (cost_per_liter_kerosene * 100) = 48 :=
by
  sorry

end cost_of_kerosene_in_cents_l586_586188


namespace magnitude_of_complex_number_l586_586481

theorem magnitude_of_complex_number : ∀ (z : ℂ), z = 3 - complex.i → complex.abs z = real.sqrt 10 :=
by
  intro z
  intro hz
  rw hz
  sorry

end magnitude_of_complex_number_l586_586481


namespace lowest_score_l586_586438

theorem lowest_score (max_mark : ℕ) (n_tests : ℕ) (avg_mark : ℕ) (h_avg : n_tests * avg_mark = 352) (h_max : ∀ k, k < n_tests → k ≤ max_mark) :
  ∃ x, (x ≤ max_mark ∧ (3 * max_mark + x) = 352) ∧ x = 52 :=
by
  sorry

end lowest_score_l586_586438


namespace exterior_angle_of_regular_polygon_l586_586163

theorem exterior_angle_of_regular_polygon (n : ℕ) (h : 1 ≤ n) (angle : ℝ) 
  (h_angle : angle = 72) (sum_ext_angles : ∀ (polygon : Type) [fintype polygon], (∑ i, angle) = 360) : 
  n = 5 := sorry

end exterior_angle_of_regular_polygon_l586_586163


namespace bens_average_speed_l586_586799

theorem bens_average_speed (total_distance : ℕ) (total_time : ℕ) (h1 : total_distance = 45) (h2 : total_time = 9) :
  total_distance / total_time = 5 :=
by
  rw [h1, h2]
  norm_num

end bens_average_speed_l586_586799


namespace center_of_circle_eq_l586_586509

theorem center_of_circle_eq {x y : ℝ} : (x - 2)^2 + (y - 3)^2 = 1 → (x, y) = (2, 3) :=
by
  intro h
  sorry

end center_of_circle_eq_l586_586509


namespace part1_part2_l586_586887

def f (x a : ℝ) : ℝ := abs (x - a)

theorem part1 (a : ℝ) (h : a = 2) : 
  {x : ℝ | f x a ≥ 4 - abs (x - 4)} = {x : ℝ | x ≤ 1} ∪ {x : ℝ | x ≥ 5} :=
by
  sorry

theorem part2 (set_is : {x : ℝ | 1 ≤ x ∧ x ≤ 2}) : 
  ∃ a : ℝ, 
    (∀ x : ℝ, abs (f (2*x + a) a - 2*f x a) ≤ 2 → (1 ≤ x ∧ x ≤ 2)) ∧ 
    a = 3 :=
by
  sorry

end part1_part2_l586_586887


namespace train_crossing_time_l586_586358

theorem train_crossing_time 
    (length : ℝ) (speed_kmph : ℝ) 
    (conversion_factor: ℝ) (speed_mps: ℝ) 
    (time : ℝ) :
  length = 400 ∧ speed_kmph = 144 ∧ conversion_factor = 1000 / 3600 ∧ speed_mps = speed_kmph * conversion_factor ∧ time = length / speed_mps → time = 10 := 
by 
  sorry

end train_crossing_time_l586_586358


namespace evaluate_expression_l586_586423

theorem evaluate_expression : (-2 : ℝ)^0 - 3 * real.tan (real.pi / 6) - |real.sqrt 3 - 2| = -1 :=
by
  sorry

end evaluate_expression_l586_586423


namespace numIncorrectPropsIsOne_l586_586511

def proposition1 : Prop :=
  ∀ (f : ℝ → ℝ) (x0 : ℝ), (f x0 = 0) → (∃ y, f' y = 0)

def proposition2 : Prop :=
  ∀ (a b : ℝ), (a < 0 ∧ a ≠ 0) → (∠(a, b) = 180)

def proposition3 : Prop :=
  ∀ (x : ℝ), (1 / (x - 1) > 0) ↔ ¬ (1 / (x - 1) ≤ 0)

def proposition4 : Prop :=
  ∀ (x : ℝ), ¬ (∃ x, x^2 + x + 1 ≤ 0) ↔ (∀ x, x^2 + x + 1 ≥ 0)

def numberOfIncorrectPropositions : Nat :=
  let incorrectProps := [proposition1, proposition2, proposition3, proposition4].filter (λ p, p = false)
  incorrectProps.length

theorem numIncorrectPropsIsOne : numberOfIncorrectPropositions = 1 :=
  by
    sorry

end numIncorrectPropsIsOne_l586_586511


namespace cone_volume_l586_586503

theorem cone_volume (l : ℝ) (S_side : ℝ) (h r V : ℝ)
  (hl : l = 10)
  (hS : S_side = 60 * Real.pi)
  (hr : S_side = π * r * l)
  (hh : h = Real.sqrt (l^2 - r^2))
  (hV : V = (1/3) * π * r^2 * h) :
  V = 96 * Real.pi := 
sorry

end cone_volume_l586_586503


namespace false_statement_among_options_l586_586729

open Classical

theorem false_statement_among_options :
  (A : (∀ x : ℝ, (x ^ 2 - 4 * x + 3 = 0) → (x = 3)) ↔ (∀ x : ℝ, (x ≠ 3) → (x ^ 2 - 4 * x + 3 ≠ 0))) ∧
  (B : ∀ p q : Prop, (p ∧ q) → (p ∧ q)) ∧
  (C : ∀ x : ℝ, (x > -1) → (x ^ 2 + 4 * x + 3 > 0)) ∧
  (D : ¬(¬(∃ x : ℝ, x ^ 2 - x + 2 > 0) ↔ (∀ x : ℝ, x ^ 2 - x + 2 ≤ 0))) :=
by
  sorry

end false_statement_among_options_l586_586729


namespace intersection_M_N_l586_586896

-- Define set M
def M : Set Int := {-2, -1, 0, 1}

-- Define set N using the given condition
def N : Set Int := {n : Int | -1 <= n ∧ n <= 3}

-- State that the intersection of M and N is the set {-1, 0, 1}
theorem intersection_M_N :
  M ∩ N = {-1, 0, 1} := by
  sorry

end intersection_M_N_l586_586896


namespace incorrect_statement_C_l586_586517

theorem incorrect_statement_C (m x : ℝ) :
  (∀ (m = 0), y = x - 1 → increasing y x) ∧
  (m = 1/2 → (∃ (h k : ℝ), h = 1/2 ∧ k = -1/4 ∧ (y = (x-h)^2 + k)) → has_vertex y (1/2) (-1/4)) ∧
  (m = -1 → (x < 5/4) → decreasing_increasing y x) ∧
  (∀ m, passes_through y 1 0) →
  ¬ statement_C_incorrect_properties
:=
begin
  sorry
end

end incorrect_statement_C_l586_586517


namespace probability_intervals_l586_586083

-- Definitions: interval (0, 1/2), and the condition interval (0, 1/3)
def interval1 := set.Ioo (0:ℝ) (1/2)
def interval2 := set.Ioo (0:ℝ) (1/3)

-- The proof statement: probability of randomly selecting a number from (0, 1/2), being within (0, 1/3), is 2/3.
theorem probability_intervals : (interval2.measure / interval1.measure = 2 / 3) :=
sorry

end probability_intervals_l586_586083


namespace sequence_periodic_l586_586956

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = -2 ∧ ∀ n, a (n + 1) = (1 + a n) / (1 - a n)

theorem sequence_periodic :
  ∃ a : ℕ → ℝ, sequence a ∧ a 2016 = 3 :=
by
  sorry

end sequence_periodic_l586_586956


namespace total_wheels_at_station_l586_586685

theorem total_wheels_at_station (trains carriages rows wheels : ℕ) 
  (h_trains : trains = 4)
  (h_carriages : carriages = 4)
  (h_rows : rows = 3)
  (h_wheels : wheels = 5) : 
  trains * carriages * rows * wheels = 240 := 
by 
  rw [h_trains, h_carriages, h_rows, h_wheels]
  exact Nat.mul_eq_iff_eq_div.mpr rfl

end total_wheels_at_station_l586_586685


namespace probability_merlin_dismissed_l586_586230

-- Define the conditions
variables (p : ℝ) (q : ℝ) (hpq : p + q = 1) (hp_pos : 0 < p) (hq_pos : 0 < q)

/--
Given advisor Merlin is equally likely to dismiss as Percival
since they are equally likely to give the correct answer independently,
prove that the probability of Merlin being dismissed is \( \frac{1}{2} \).
-/
theorem probability_merlin_dismissed : (1/2 : ℝ) = 1/2 :=
by 
  sorry

end probability_merlin_dismissed_l586_586230


namespace ellipses_and_orthocenter_collinear_l586_586787

-- Definitions for the problem
structure Triangle :=
(A B C : Point)

structure Ellipse :=
(foci : Point × Point)
passing_point : Point

def midpoint (p1 p2 : Point) : Point := sorry

def orthocenter (t : Triangle) : Point := sorry

def common_points (e1 e2 : Ellipse) : set Point := sorry

-- Given condition: construction of the ellipses
variable (t : Triangle)

def ellipse_Gamma1 : Ellipse :=
{ foci := (midpoint t.A t.B, midpoint t.A t.C),
  passing_point := t.A }

def ellipse_Gamma2 : Ellipse :=
{ foci := (midpoint t.A t.C, midpoint t.B t.C),
  passing_point := t.C }

-- The goal to prove
theorem ellipses_and_orthocenter_collinear : 
  ∀ (X Y : Point), X ∈ common_points (ellipse_Gamma1 t) (ellipse_Gamma2 t) →
  Y ∈ common_points (ellipse_Gamma1 t) (ellipse_Gamma2 t) →
  collinear {X, Y, orthocenter t} :=
by {
  intros X Y commonX commonY,
  -- The proof would go here, but we skip it with sorry
  sorry
}

end ellipses_and_orthocenter_collinear_l586_586787


namespace probability_of_pq_is_seven_twentieths_l586_586538

noncomputable def probability_pq (p q : ℤ) : Prop :=
  pq - 6 * p - 4 * q = 8

noncomputable def valid_p_values := {p : ℤ | ∃ q : ℤ, (p - 4) * (q - 6) = 32}

noncomputable def valid_p_values_in_range := {p : ℤ | (1 ≤ p ∧ p ≤ 20) ∧ p ∈ valid_p_values}

noncomputable def result := (valid_p_values_in_range.card : ℚ) / 20

theorem probability_of_pq_is_seven_twentieths :
  result = 7 / 20 := by
  sorry

end probability_of_pq_is_seven_twentieths_l586_586538


namespace sum_log2_a_n_eq_sum_b_n_eq_l586_586487

-- Define the sequence a_n according to the given conditions
def a (n : ℕ) : ℝ := 2^(n-1)

-- Define the sequence log2_a_n and S_n
def log2_a (n : ℕ) : ℝ := Real.log 2 (a n)
def S (n : ℕ) : ℝ := (Finset.range n).sum (λ k, log2_a (k+1))

-- Define the sequence b_n and T_n
def b (n : ℕ) : ℝ := a n * log2_a n
def T (n : ℕ) : ℝ := (Finset.range n).sum (λ k, b (k+1))

-- The statement of the first part
theorem sum_log2_a_n_eq (n : ℕ) : S n = n-1 := sorry

-- The statement of the second part
theorem sum_b_n_eq (n : ℕ) : T n = n * 2^(n+2) := sorry

end sum_log2_a_n_eq_sum_b_n_eq_l586_586487


namespace line1_line2_line3_l586_586460

-- Line 1: Through (-1, 3), parallel to x - 2y + 3 = 0.
theorem line1 (x y : ℝ) : (x - 2 * y + 3 = 0) ∧ (x = -1) ∧ (y = 3) →
                              (x - 2 * y + 7 = 0) :=
by sorry

-- Line 2: Through (3, 4), perpendicular to 3x - y + 2 = 0.
theorem line2 (x y : ℝ) : (3 * x - y + 2 = 0) ∧ (x = 3) ∧ (y = 4) →
                              (x + 3 * y - 15 = 0) :=
by sorry

-- Line 3: Through (1, 2), with equal intercepts on both axes.
theorem line3 (x y : ℝ) : (x = y) ∧ (x = 1) ∧ (y = 2) →
                              (x + y - 3 = 0) :=
by sorry

end line1_line2_line3_l586_586460


namespace line_parallel_through_point_line_perpendicular_through_point_line_with_equal_intercepts_through_point_l586_586462

-- Define the line passing through a given point and parallel to another line
theorem line_parallel_through_point 
  (p : Point) (line : Line) (a b c : ℝ) (hp : p = (-1,3)) (hline : line = Equation (Coeff a b c) 
  (a = 1) (b = -2) (c = 3)) : Line := 
Proof
  sorry

-- Define the line passing through a given point and perpendicular to another line
theorem line_perpendicular_through_point 
  (p : Point) (line : Line) (a b c : ℝ) (hp : p = (3,4)) (hline : line = Equation (Coeff a b c) 
  (a = 3) (b = -1) (c = 2)) : Line := 
Proof
  sorry

-- Define the line passing through a given point with equal intercepts on both axes
theorem line_with_equal_intercepts_through_point 
  (p : Point) (intercept : ℝ) (hp : p = (1,2)) (hintercept : intercept = intercept): Line := 
Proof
  sorry

noncomputable def Point : Type* := ℝ × ℝ

noncomputable def Coeff (a b c : ℝ) : Type := ∀ (x y z: ℤ), a * x + b * y + c * z = 0

noncomputable def Equation (c : Coeff) : Type := Refl c

end line_parallel_through_point_line_perpendicular_through_point_line_with_equal_intercepts_through_point_l586_586462


namespace number_of_positive_divisors_of_360_is_24_l586_586910

theorem number_of_positive_divisors_of_360_is_24 :
  ∀ n : ℕ, n = 360 → n = 2^3 * 3^2 * 5^1 → 
  (n_factors : {p : ℕ × ℕ // p.1 ∈ [2, 3, 5] ∧ p.2 ∈ [3, 2, 1]} )
    → (n_factors.val.snd + 1).prod = 24 :=
by
  intro n hn h_factors
  rw hn at *
  have factors := h_factors.val
  cases factors with p_k q_l r_m
  have hpq : p_k.1 = 2 ∧ p_k.2 = 3 :=
    And.intro sorry sorry,
  have hqr : q_l.1 = 3 ∧ q_l.2 = 2 :=
    And.intro sorry sorry,
  have hr : r_m.1 = 5 ∧ r_m.2 = 1 :=
    And.intro sorry sorry,
  -- The proof would continue, but we'll skip it
  sorry

end number_of_positive_divisors_of_360_is_24_l586_586910


namespace num_divisors_360_l586_586917

theorem num_divisors_360 :
  ∀ n : ℕ, n = 360 → (∀ (p q r : ℕ), p = 2 ∧ q = 3 ∧ r = 5 →
    (∃ (a b c : ℕ), 360 = p^a * q^b * r^c ∧ a = 3 ∧ b = 2 ∧ c = 1) →
    (3+1) * (2+1) * (1+1) = 24) :=
  sorry

end num_divisors_360_l586_586917


namespace average_side_length_of_squares_l586_586857

theorem average_side_length_of_squares :
  let s1 := Real.sqrt 25
      s2 := Real.sqrt 64
      s3 := Real.sqrt 121
      s4 := Real.sqrt 196 in
  (s1 + s2 + s3 + s4) / 4 = 9.5 :=
by
  sorry

end average_side_length_of_squares_l586_586857


namespace probability_less_than_one_third_l586_586156

-- Define the intervals and the associated lengths
def interval_total : ℝ := 1 / 2
def interval_condition : ℝ := 1 / 3

-- Define the probability calculation
def probability : ℝ :=
  interval_condition / interval_total

-- Statement of the problem
theorem probability_less_than_one_third : probability = 2 / 3 := by
  sorry

end probability_less_than_one_third_l586_586156


namespace arithmetic_sequence_sum_l586_586034

theorem arithmetic_sequence_sum :
  ∀ {a : ℕ → ℕ} {S : ℕ → ℕ},
  (∀ n, a (n + 1) - a n = a 1 - a 0) →
  (∀ n, S n = n * (a 1 + a n) / 2) →
  a 1 + a 9 = 18 →
  a 4 = 7 →
  S 8 = 64 :=
by
  intros a S h_arith_seq h_sum_formula h_a1_a9 h_a4
  sorry

end arithmetic_sequence_sum_l586_586034


namespace chicken_cost_l586_586747
noncomputable def chicken_cost_per_plate
  (plates : ℕ) 
  (rice_cost_per_plate : ℝ) 
  (total_cost : ℝ) : ℝ :=
  let total_rice_cost := plates * rice_cost_per_plate
  let total_chicken_cost := total_cost - total_rice_cost
  total_chicken_cost / plates

theorem chicken_cost
  (hplates : plates = 100)
  (hrice_cost_per_plate : rice_cost_per_plate = 0.10)
  (htotal_cost : total_cost = 50) :
  chicken_cost_per_plate 100 0.10 50 = 0.40 :=
by
  sorry

end chicken_cost_l586_586747


namespace eccentricity_of_ellipse_l586_586489

-- Given the conditions for the ellipse and the point on the ellipse
variables {m : ℝ} (h_m : m > 1)
def ellipse_eq (x y : ℝ) := (x^2 / m^2) + (y^2 / (m^2 - 1)) = 1
def point_distance_left_focus {x y : ℝ} (h_point : ellipse_eq m x y) : ℝ := 3
def point_distance_right_focus {x y : ℝ} (h_point : ellipse_eq m x y) : ℝ := 1
def major_axis_length := point_distance_left_focus h_m + point_distance_right_focus h_m
def a := major_axis_length / 2

theorem eccentricity_of_ellipse :
  ∃ e : ℝ, e = 1 / 2 :=
begin
  have h_ellipse_def : (forall x y : ℝ, ellipse_eq m x y -> point_distance_left_focus h_m + point_distance_right_focus h_m = 4), 
  { sorry },
  let c := 1,
  have e_def : e = c / a,
  { sorry },
  use e,
  have h_e : e = 1 / 2,
  { sorry },
  exact h_e,
end

end eccentricity_of_ellipse_l586_586489


namespace probability_merlin_dismissed_l586_586214

noncomputable def merlin_dismissal_probability {p : ℝ} (hp : 0 ≤ p ∧ p ≤ 1) : ℝ :=
  let q := 1 - p in
  -- Assuming the coin flip is fair, the probability Merlin is dismissed
  1 / 2

theorem probability_merlin_dismissed (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  merlin_dismissal_probability hp = 1 / 2 :=
by
  -- Placeholder for the proof
  sorry

end probability_merlin_dismissed_l586_586214


namespace bird_families_away_to_Africa_l586_586689

theorem bird_families_away_to_Africa (total_families original_families asia_families left_families : ℕ) :
  original_families = 85 →
  asia_families = 37 →
  left_families = 25 →
  total_families = original_families - left_families →
  (total_families - asia_families) = 23 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  exact rfl

end bird_families_away_to_Africa_l586_586689


namespace orthocenter_fixed_position_l586_586865

-- Definitions of circle, midpoints, and orthocenter
structure Circle :=
  (center : Point)
  (radius : ℝ)

structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def midpoint (p1 p2 : Point) : Point :=
  { x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2 }

noncomputable def orthocenter (o p1 p2 : Point) := sorry 

theorem orthocenter_fixed_position (O A C P X Y H : Point) (ω : Circle) 
  (hA_on_circle : A ∈ ω)
  (hC_on_circle : C ∈ ω)
  (hP_on_circle : P ∈ ω)
  (hX_is_midpoint : X = midpoint A P)
  (hY_is_midpoint : Y = midpoint C P)
  (hH_is_orthocenter : H = orthocenter O X Y) :
  H = midpoint A C :=
sorry

end orthocenter_fixed_position_l586_586865


namespace number_of_juniors_in_sample_l586_586389

theorem number_of_juniors_in_sample
  (total_students : ℕ)
  (num_freshmen : ℕ)
  (num_freshmen_sampled : ℕ)
  (num_sophomores_exceeds_num_juniors_by : ℕ)
  (num_sophomores num_juniors num_juniors_sampled : ℕ)
  (h_total : total_students = 1290)
  (h_num_freshmen : num_freshmen = 480)
  (h_num_freshmen_sampled : num_freshmen_sampled = 96)
  (h_exceeds : num_sophomores_exceeds_num_juniors_by = 30)
  (h_equation : total_students - num_freshmen = num_sophomores + num_juniors)
  (h_num_sophomores : num_sophomores = num_juniors + num_sophomores_exceeds_num_juniors_by)
  (h_fraction : num_freshmen_sampled / num_freshmen = 1 / 5)
  (h_num_juniors_sampled : num_juniors_sampled = num_juniors * (num_freshmen_sampled / num_freshmen)) :
  num_juniors_sampled = 78 := by
  sorry

end number_of_juniors_in_sample_l586_586389


namespace line_equation_l586_586878

open Real

theorem line_equation (x y : ℝ) :
  let l := λ p : ℝ × ℝ, ∃ k : ℝ, k ≠ 0 ∧ p.2 + 4 = k * (p.1 + 5) in
  let area := λ a b : ℝ, (1 / 2) * abs (a * b) in
  (-5, -4) ∈ {p | l p} →
  l (a, 0) →
  l (0, b) →
  area a b = 5 →
  (∃ k : ℝ, l (x, y) ∧ (x * 8 - y * 5 + 20 = 0 ∨ x * 2 - y * 5 - 10 = 0)) :=
by
  sorry

end line_equation_l586_586878


namespace find_k_value_l586_586022

variable {a : ℕ → ℕ} {S : ℕ → ℕ} 

axiom sum_of_first_n_terms (n : ℕ) (hn : n > 0) : S n = a n / n
axiom exists_Sk_inequality (k : ℕ) (hk : k > 0) : 1 < S k ∧ S k < 9

theorem find_k_value 
  (k : ℕ) (hk : k > 0) (hS : S k = a k / k) (hSk : 1 < S k ∧ S k < 9)
  (h_cond : ∀ n > 0, S n = n * S n ∧ S (n - 1) = S n * (n - 1)) : 
  k = 4 :=
sorry

end find_k_value_l586_586022


namespace corey_needs_more_golf_balls_l586_586437

-- Defining the constants based on the conditions
def goal : ℕ := 48
def found_on_saturday : ℕ := 16
def found_on_sunday : ℕ := 18

-- The number of golf balls Corey has found over the weekend
def total_found : ℕ := found_on_saturday + found_on_sunday

-- The number of golf balls Corey still needs to find to reach his goal
def remaining : ℕ := goal - total_found

-- The desired theorem statement
theorem corey_needs_more_golf_balls : remaining = 14 := 
by 
  sorry

end corey_needs_more_golf_balls_l586_586437


namespace smallest_x_solution_l586_586848

def smallest_x_condition (x : ℝ) : Prop :=
  (x^2 - 5 * x - 84 = (x - 12) * (x + 7)) ∧
  (x ≠ 9) ∧
  (x ≠ -7) ∧
  ((x^2 - 5 * x - 84) / (x - 9) = 4 / (x + 7))

theorem smallest_x_solution :
  ∃ x : ℝ, smallest_x_condition x ∧ ∀ y : ℝ, smallest_x_condition y → x ≤ y :=
sorry

end smallest_x_solution_l586_586848


namespace derivative_f_l586_586544

noncomputable def f (x : ℝ) : ℝ := exp (-x) * (cos x + sin x)

theorem derivative_f (x : ℝ) : deriv f x = -2 * exp (-x) * sin x := by
  sorry

end derivative_f_l586_586544


namespace cube_root_of_x_plus_y_l586_586042

theorem cube_root_of_x_plus_y (x y : ℝ) 
  (h1 : x + 2 = 9)
  (h2 : -y = -1) 
  : real.cbrt (x + y) = 2 :=
sorry

end cube_root_of_x_plus_y_l586_586042


namespace incorrect_propositions_l586_586575

variables {α : Type*} [plane α]
variables {m n : line α} {m1 n1 : line α}

-- Define projections of lines m and n onto plane α.
def projection (l : line α) : line α := sorry

-- Define conditions based on the problem statement.
axiom not_on_plane (m n : line α) (α : Type*) : ¬ (m ⊆ α) ∧ ¬ (n ⊆ α)
axiom projections (m n : line α) (α : Type*) : projection m = m1 ∧ projection n = n1

-- Define propositions we need to verify.
def proposition1 : Prop := m1 ⟂ n1 → m ⟂ n
def proposition2 : Prop := m ⟂ n → m1 ⟂ n1
def proposition3 : Prop := m1 ∩ n1 ≠ ∅ → m ∩ n ≠ ∅ ∧ m ≠ n
def proposition4 : Prop := parallel m1 n1 → (parallel m n ∨ m = n)

-- Prove that all propositions are incorrect.
theorem incorrect_propositions : ¬ proposition1 ∧ ¬ proposition2 ∧ ¬ proposition3 ∧ ¬ proposition4 := sorry

end incorrect_propositions_l586_586575


namespace number_of_friends_l586_586429

def initial_candies : ℕ := 10
def additional_candies : ℕ := 4
def total_candies : ℕ := initial_candies + additional_candies
def candies_per_friend : ℕ := 2

theorem number_of_friends : total_candies / candies_per_friend = 7 :=
by
  sorry

end number_of_friends_l586_586429


namespace problem1_problem2_l586_586901

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (-1, 2)

-- Define the first proof problem: 2a + b = (1, 2)
theorem problem1 : 2 • a + b = (1, 2) := 
sorry

-- Define the second proof problem: |a - b| = 2√2
theorem problem2 : ∥a - b∥ = 2 * Real.sqrt 2 := 
sorry

end problem1_problem2_l586_586901


namespace laura_running_speed_is_correct_l586_586969

noncomputable def laura_runs_at_correct_speed (x : ℝ) : Prop :=
  let motion_time := 140 - 10
  let motion_hours := motion_time / 60
  let biking_time := 25 / (3 * x + 2)
  let running_time := 6 / x
  (biking_time + running_time ≈ motion_hours)

theorem laura_running_speed_is_correct :
  ∃ x : ℝ, laura_runs_at_correct_speed x ∧ 0 ≤ x := sorry

end laura_running_speed_is_correct_l586_586969


namespace length_AB_l586_586262

-- Definitions of points and segments
variables (A B C D E F : Type) [Dist : ∀ XY: Type, XY → XY → ℕ] 
          [Midpoint : ∀ XY: Type, XY → XY → XY]

-- Conditions
axiom C_midpoint : Midpoint A B C
axiom D_midpoint : Midpoint A C D
axiom E_midpoint : Midpoint A D E
axiom F_midpoint : Midpoint A E F
axiom AF_length : Dist A F = 3

-- Proof problem statement: Length of AB is 48 units
theorem length_AB : Dist A B = 48 := 
by sorry

end length_AB_l586_586262


namespace proof_greatest_n_l586_586601

noncomputable def greatest_n_less_1000 : ℕ := 956

def floor_sqrt (n : ℕ) : ℕ := nat.floor (real.sqrt n)

theorem proof_greatest_n :
  0 < greatest_n_less_1000 ∧ 
  greatest_n_less_1000 < 1000 ∧ 
  (floor_sqrt greatest_n_less_1000 - 2) ∣ (greatest_n_less_1000 - 4) ∧ 
  (floor_sqrt greatest_n_less_1000 + 2) ∣ (greatest_n_less_1000 + 4) :=
by
  -- prove all the conditions for n = 956
  sorry

end proof_greatest_n_l586_586601


namespace unit_prices_min_number_of_A_l586_586697

theorem unit_prices (x y : ℝ)
  (h1 : 3 * x + 4 * y = 580)
  (h2 : 6 * x + 5 * y = 860) :
  x = 60 ∧ y = 100 :=
by
  sorry

theorem min_number_of_A (x y a : ℝ)
  (x_h : x = 60)
  (y_h : y = 100)
  (h1 : 3 * x + 4 * y = 580)
  (h2 : 6 * x + 5 * y = 860)
  (trash_can_condition : a + 200 - a = 200)
  (cost_condition : 60 * a + 100 * (200 - a) ≤ 15000) :
  a ≥ 125 :=
by
  sorry

end unit_prices_min_number_of_A_l586_586697


namespace building_shadow_length_l586_586755

theorem building_shadow_length (H_f: ℝ) (L_f: ℝ) (H_b: ℝ): ∃ L_b: ℝ, L_b = 70 :=
  assume (hf: H_f = 18) (lf: L_f = 45) (hb: H_b = 28),
  have proportion: H_f / L_f = H_b / L_b, from sorry,
  have L_b: ℝ := (28 * 45) / 18,
  have L_b_eq: L_b = 70, from sorry,
  exists.intro L_b L_b_eq

end building_shadow_length_l586_586755


namespace sampling_is_systematic_l586_586944

noncomputable def students : List ℕ := List.range' 1 60

def isSystematicSampling (sample_set : List ℕ) (interval : ℕ) : Prop :=
  ∀ i, i < sample_set.length - 1 → sample_set[i + 1] - sample_set[i] = interval

theorem sampling_is_systematic :
  (∃ sample_set, (∀ n, n ∈ sample_set ↔ (n ∈ students ∧ n % 5 = 0)) ∧ isSystematicSampling sample_set 5) :=
begin
  sorry, -- proof to be provided
end

end sampling_is_systematic_l586_586944


namespace withdrawal_amount_in_2008_l586_586279

noncomputable def total_withdrawal (a : ℕ) (p : ℝ) : ℝ :=
  (a / p) * ((1 + p) - (1 + p)^8)

theorem withdrawal_amount_in_2008 (a : ℕ) (p : ℝ) (h_pos : 0 < p) (h_neg_one_lt : -1 < p) :
  total_withdrawal a p = (a / p) * ((1 + p) - (1 + p)^8) :=
by
  -- Conditions
  -- Starting from May 10th, 2001, multiple annual deposits.
  -- Annual interest rate p > 0 and p > -1.
  sorry

end withdrawal_amount_in_2008_l586_586279


namespace incenter_equidistant_l586_586781

theorem incenter_equidistant {A B C : Point} (h : Triangle A B C) : 
  ∃ I : Point, (incenter A B C = I) ∧ 
  (distance I (Line A B) = distance I (Line B C) ∧ distance I (Line B C) = distance I (Line C A)) := 
sorry

end incenter_equidistant_l586_586781


namespace set_intersection_complement_l586_586524

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}
noncomputable def A : Set ℕ := {1, 3}
noncomputable def B : Set ℕ := {2, 3}

theorem set_intersection_complement :
  A ∩ (U \ B) = {1} :=
sorry

end set_intersection_complement_l586_586524


namespace complex_div_pure_imaginary_l586_586555

theorem complex_div_pure_imaginary (a : ℝ) : (a - 1 = 0) → (a + 1 ≠ 0) → (im ((a + 1) * I / 2) = (a + 1) / 2 * I) :=
by
  sorry

end complex_div_pure_imaginary_l586_586555


namespace geometric_probability_l586_586105

theorem geometric_probability :
  let total_length := (1/2 - 0)
  let favorable_length := (1/3 - 0)
  total_length > 0 ∧ favorable_length > 0 →
  (favorable_length / total_length) = 2 / 3 :=
by
  intros _ _ _
  sorry

end geometric_probability_l586_586105


namespace regular_polygon_exterior_angle_l586_586168

theorem regular_polygon_exterior_angle (n : ℕ) (h : 72 = 360 / n) : n = 5 :=
by
  sorry

end regular_polygon_exterior_angle_l586_586168


namespace triangle_area_circumcircle_area_ratio_l586_586197

theorem triangle_area_circumcircle_area_ratio {A B C a b c : ℝ} (h1 : 2 * Real.sin A * Real.cos (B - C) + Real.sin (2 * A) = 2 / 3) :
  let S₁ := (1 / 2) * a * b * Real.sin C in
  let S₂ := Real.pi * (b / (2 * Real.sin B)) ^ 2 in
  S₁ / S₂ = 1 / (3 * Real.pi) :=
by
  sorry

end triangle_area_circumcircle_area_ratio_l586_586197


namespace probability_merlin_dismissed_l586_586229

-- Define the conditions
variables (p : ℝ) (q : ℝ) (hpq : p + q = 1) (hp_pos : 0 < p) (hq_pos : 0 < q)

/--
Given advisor Merlin is equally likely to dismiss as Percival
since they are equally likely to give the correct answer independently,
prove that the probability of Merlin being dismissed is \( \frac{1}{2} \).
-/
theorem probability_merlin_dismissed : (1/2 : ℝ) = 1/2 :=
by 
  sorry

end probability_merlin_dismissed_l586_586229


namespace largest_power_of_3_factor_of_exp_q_l586_586610

noncomputable def q : ℝ := ∑ k in (Finset.range 5).map (λ k, k + 1), (k^2 * Real.log k)

theorem largest_power_of_3_factor_of_exp_q : 
  ∃ (n : ℕ), (3^n ∤ Real.exp q) ∧ (3^(n + 1) | Real.exp q) := by
  sorry

end largest_power_of_3_factor_of_exp_q_l586_586610


namespace perimeter_of_square_B_l586_586278

theorem perimeter_of_square_B
  (perimeter_A : ℝ)
  (h_perimeter_A : perimeter_A = 36)
  (area_ratio : ℝ)
  (h_area_ratio : area_ratio = 1 / 3)
  : ∃ (perimeter_B : ℝ), perimeter_B = 12 * Real.sqrt 3 :=
by
  sorry

end perimeter_of_square_B_l586_586278


namespace pie_crust_flour_amount_l586_586967

-- Step 1: Define the context and main variables.
def total_flour_old := 30 * (1 / 6 : ℚ)
def total_flour_new := 5 : ℚ

-- Step 2: Define the condition for the new pie crusts
def new_pie_crusts := 25

-- Step 3: Define the target amount of flour per new crust
def flour_per_new_crust := total_flour_new / new_pie_crusts

-- Step 4: Define the theorem to be proved
theorem pie_crust_flour_amount :
  flour_per_new_crust = 1 / 5 :=
by
  -- proof skipped
  sorry

end pie_crust_flour_amount_l586_586967


namespace total_cost_of_feeding_pets_for_one_week_l586_586287

-- Definitions based on conditions
def turtle_food_per_weight : ℚ := 1 / (1 / 2)
def turtle_weight : ℚ := 30
def turtle_food_qty_per_jar : ℚ := 15
def turtle_food_cost_per_jar : ℚ := 3

def bird_food_per_weight : ℚ := 2
def bird_weight : ℚ := 8
def bird_food_qty_per_bag : ℚ := 40
def bird_food_cost_per_bag : ℚ := 5

def hamster_food_per_weight : ℚ := 1.5 / (1 / 2)
def hamster_weight : ℚ := 3
def hamster_food_qty_per_box : ℚ := 20
def hamster_food_cost_per_box : ℚ := 4

-- Theorem stating the equivalent proof problem
theorem total_cost_of_feeding_pets_for_one_week :
  let turtle_food_needed := (turtle_weight * turtle_food_per_weight)
  let turtle_jars_needed := turtle_food_needed / turtle_food_qty_per_jar
  let turtle_cost := turtle_jars_needed * turtle_food_cost_per_jar
  let bird_food_needed := (bird_weight * bird_food_per_weight)
  let bird_bags_needed := bird_food_needed / bird_food_qty_per_bag
  let bird_cost := if bird_bags_needed < 1 then bird_food_cost_per_bag else bird_bags_needed * bird_food_cost_per_bag
  let hamster_food_needed := (hamster_weight * hamster_food_per_weight)
  let hamster_boxes_needed := hamster_food_needed / hamster_food_qty_per_box
  let hamster_cost := if hamster_boxes_needed < 1 then hamster_food_cost_per_box else hamster_boxes_needed * hamster_food_cost_per_box
  turtle_cost + bird_cost + hamster_cost = 21 :=
by
  sorry

end total_cost_of_feeding_pets_for_one_week_l586_586287


namespace simplify_expression_l586_586037

-- Define the given condition as a hypothesis
theorem simplify_expression (a b c : ℝ) (h : a + b + c = 0) :
  a * (1 / b + 1 / c) + b * (1 / c + 1 / a) + c * (1 / a + 1 / b) + 3 = 0 :=
by
  sorry -- Proof will be provided here.

end simplify_expression_l586_586037


namespace abc_eq_zero_l586_586600

variable (a b c : ℝ) (n : ℕ)

theorem abc_eq_zero
  (h1 : a^n + b^n = c^n)
  (h2 : a^(n+1) + b^(n+1) = c^(n+1))
  (h3 : a^(n+2) + b^(n+2) = c^(n+2)) :
  a * b * c = 0 :=
sorry

end abc_eq_zero_l586_586600


namespace effective_CAGR_l586_586334

/-- CAGR problem given conditions -/
theorem effective_CAGR (R1 R2 R3 R4 I C : ℝ)
    (h1 : 0 ≤ R1)
    (h2 : 0 ≤ R2)
    (h3 : 0 ≤ R3)
    (h4 : 0 ≤ R4)
    (hI : 0 ≤ I)
    (hC : 0 ≤ C) :
    let investment_growth := 
        ((1 + R1 / 100) ^ (2.5) *
        (1 + R2 / 100) ^ (2.5) *
        (1 + R3 / 100) ^ (2.5) *
        (1 + R4 / 100) ^ (2.5)) /
        ((1 + I / 100) ^ 10 * (1 + C / 100) ^ 10)
    in 
    CAGR = investment_growth ^ (1 / 10) - 1 := 
sorry

end effective_CAGR_l586_586334


namespace find_cos_gamma_l586_586602

-- Definitions corresponding to the conditions
variables {x y z : ℝ} -- Coordinates of the point R
variable (R : ℝ × ℝ × ℝ) -- Point R in coordinate space
variable (α β γ : ℝ) -- Angles with the axes

-- Given conditions
axiom pos_coords : 0 < x ∧ 0 < y ∧ 0 < z
axiom cos_alpha : α = real.arccos (1/4)
axiom cos_beta : β = real.arccos (1/8)

-- Theorem we need to prove
theorem find_cos_gamma : 
  (γ = real.arccos (z / (real.sqrt (x^2 + y^2 + z^2)))) → 
  γ = real.arccos (real.sqrt 59 / 8) := sorry

end find_cos_gamma_l586_586602


namespace probability_less_than_one_third_l586_586099

theorem probability_less_than_one_third : 
  (∃ p : ℚ, p = (measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ)) (set.Ioo (0 : ℚ) (1 / 3 : ℚ)) / 
                  measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ))) ∧ p = 2 / 3) :=
by {
  -- The proof will be added here.
  sorry
}

end probability_less_than_one_third_l586_586099


namespace focus_of_parabola_y_eq_9x2_plus_6_l586_586817

noncomputable def focus_of_parabola (a b : ℝ) : ℝ × ℝ :=
  (0, b + (1 / (4 * a)))

theorem focus_of_parabola_y_eq_9x2_plus_6 :
  focus_of_parabola 9 6 = (0, 217 / 36) :=
by
  sorry

end focus_of_parabola_y_eq_9x2_plus_6_l586_586817


namespace sand_overflow_l586_586410

theorem sand_overflow 
  (C_A : ℝ) 
  (C_B : ℝ) 
  (C_C : ℝ) 
  (hCB : C_B = C_A / 2) 
  (hCC : C_C = 2 * C_A) 
  (sand_A : ℝ) 
  (sand_B : ℝ) 
  (sand_C : ℝ) 
  (hSand_A : sand_A = C_A / 4) 
  (hSand_B : sand_B = 3 / 8 * C_B) 
  (hSand_C : sand_C = C_C / 3) :
  let sand_B_after_pour := sand_B + sand_C in
  let overflow_B := sand_B_after_pour - C_B in
  let sand_in_A := sand_A + min C_B sand_B_after_pour in
  ∃ overflow : ℝ, overflow = max overflow_B 0 ∧ overflow = 17 / 48 * C_A := 
by
  sorry

end sand_overflow_l586_586410


namespace probability_merlin_dismissed_l586_586218

noncomputable def merlin_dismissal_probability {p : ℝ} (hp : 0 ≤ p ∧ p ≤ 1) : ℝ :=
  let q := 1 - p in
  -- Assuming the coin flip is fair, the probability Merlin is dismissed
  1 / 2

theorem probability_merlin_dismissed (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  merlin_dismissal_probability hp = 1 / 2 :=
by
  -- Placeholder for the proof
  sorry

end probability_merlin_dismissed_l586_586218


namespace one_belt_one_road_l586_586017

theorem one_belt_one_road (m n : ℝ) :
  (∀ x y : ℝ, y = x^2 - 2 * x + n ↔ (x, y) ∈ { p : ℝ × ℝ | p.1 = 0 ∧ p.2 = 1 }) →
  (∀ x y : ℝ, y = m * x + 1 ↔ (x, y) ∈ { q : ℝ × ℝ | q.1 = 0 ∧ q.2 = 1 }) →
  (∀ x y : ℝ, y = x^2 - 2 * x + 1 → y = 0) →
  m = -1 ∧ n = 1 :=
by
  intros h1 h2 h3
  sorry

end one_belt_one_road_l586_586017


namespace area_of_shaded_region_l586_586582

def radius_of_larger_circle : ℝ := 10
def radius_of_smaller_circle : ℝ := radius_of_larger_circle / 2
def area_of_circle (r : ℝ) : ℝ := π * r^2

theorem area_of_shaded_region :
  let rₗ := radius_of_larger_circle
  let rₛ := radius_of_smaller_circle
  area_of_circle rₗ - 2 * area_of_circle rₛ = 50 * π :=
by
  sorry

end area_of_shaded_region_l586_586582


namespace solve_prime_square_sum_l586_586856

theorem solve_prime_square_sum (p q n : ℕ) (hp : p.prime) (hq : q.prime) 
    (hn : n ≥ 0) (h : n ^ 2 = p ^ 2 + q ^ 2 + p ^ 2 * q ^ 2) :
    (p = 2 ∧ q = 3 ∧ n = 7) ∨ (p = 3 ∧ q = 2 ∧ n = 7) :=
    sorry

end solve_prime_square_sum_l586_586856


namespace distance_between_midpoints_is_root_10_l586_586528

-- Definitions based on the conditions provided
variables (a b c d : ℝ)

def A : ℝ × ℝ := (a, b)
def B : ℝ × ℝ := (c, d)

def M : ℝ × ℝ := ((a + c) / 2, (b + d) / 2)

def A' : ℝ × ℝ := (a + 3, b + 10)
def B' : ℝ × ℝ := (c - 5, d - 4)

def M' : ℝ × ℝ := ((a + 3 + (c - 5)) / 2, (b + 10 + (d - 4)) / 2)

-- Proving the distance between M and M' is √10
theorem distance_between_midpoints_is_root_10 :
  let m := (a + c) / 2 in let n := (b + d) / 2 in
  let M' := (m - 1, n + 3) in
  ∥(m, n) - (m - 1, n + 3)∥ = real.sqrt 10 :=
by
  sorry

end distance_between_midpoints_is_root_10_l586_586528


namespace meal_cost_for_group_l586_586797

def cost_of_group_meal 
  (total_people : ℕ)
  (kids : ℕ)
  (adult_meal_cost : ℕ)
  (adults := total_people - kids)
  (total_cost := adults * adult_meal_cost) : ℕ :=
  total_cost

theorem meal_cost_for_group (total_people : ℕ) (kids : ℕ) (adult_meal_cost : ℕ) :
  total_people = 12 → 
  kids = 7 → 
  adult_meal_cost = 3 → 
  cost_of_group_meal total_people kids adult_meal_cost = 15 :=
by 
  intros h1 h2 h3
  simp [cost_of_group_meal, h1, h2, h3]
  exact h3

end meal_cost_for_group_l586_586797


namespace geometric_sequence_condition_l586_586021

theorem geometric_sequence_condition (A B q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ)
  (hSn_def : ∀ n, S n = A * q^n + B) (hq_ne_zero : q ≠ 0) :
  (∀ n, a n = S n - S (n-1)) → (A = -B) ↔ (∀ n, a n = A * (q - 1) * q^(n-1)) := 
sorry

end geometric_sequence_condition_l586_586021


namespace alice_prob_after_three_turns_l586_586397

/-
Definition of conditions:
 - Alice starts with the ball.
 - If Alice has the ball, there is a 1/3 chance that she will toss it to Bob and a 2/3 chance that she will keep the ball.
 - If Bob has the ball, there is a 1/4 chance that he will toss it to Alice and a 3/4 chance that he keeps the ball.
-/

def alice_to_bob : ℚ := 1/3
def alice_keeps : ℚ := 2/3
def bob_to_alice : ℚ := 1/4
def bob_keeps : ℚ := 3/4

theorem alice_prob_after_three_turns :
  alice_to_bob * bob_keeps * bob_to_alice +
  alice_keeps * alice_keeps * alice_keeps +
  alice_to_bob * bob_to_alice * alice_keeps = 179/432 :=
by
  sorry

end alice_prob_after_three_turns_l586_586397


namespace Li_Minghui_should_go_to_supermarket_B_for_300_yuan_cost_equal_for_500_yuan_l586_586281

def cost_supermarket_A (x : ℝ) : ℝ :=
  200 + 0.8 * (x - 200)

def cost_supermarket_B (x : ℝ) : ℝ :=
  100 + 0.85 * (x - 100)

theorem Li_Minghui_should_go_to_supermarket_B_for_300_yuan :
  cost_supermarket_B 300 < cost_supermarket_A 300 := by
  sorry

theorem cost_equal_for_500_yuan :
  cost_supermarket_A 500 = cost_supermarket_B 500 := by
  sorry

end Li_Minghui_should_go_to_supermarket_B_for_300_yuan_cost_equal_for_500_yuan_l586_586281


namespace min_area_triangle_inradius_tan_l586_586879

theorem min_area_triangle_inradius_tan
  (a b c : ℝ) (inradius : ℝ) (tanA : ℝ)
  (h_inradius : inradius = 2)
  (h_tanA : tanA = -4/3)
  (s : ℝ := (a + b + c) / 2) :
  min_area_triangle a b c inradius tanA = 18 + 8 * Real.sqrt 5 := 
sorry

end min_area_triangle_inradius_tan_l586_586879


namespace exists_infinite_coprime_set_l586_586444

theorem exists_infinite_coprime_set :
  ∃ (S : Set ℕ), S.Countable ∧ (∀ (x y z w ∈ S), x < y → z < w → (x, y) ≠ (z, w) → Nat.gcd (x * y + 2022) (z * w + 2022) = 1) :=
sorry

end exists_infinite_coprime_set_l586_586444


namespace bus_speed_excluding_stoppages_l586_586836

theorem bus_speed_excluding_stoppages (v : ℝ) 
  (speed_including_stoppages : ℝ := 45) 
  (stoppage_time : ℝ := 1/6) 
  (h : v * (1 - stoppage_time) = speed_including_stoppages) : 
  v = 54 := 
by 
  sorry

end bus_speed_excluding_stoppages_l586_586836


namespace part1_part2_part3_l586_586056

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 1
noncomputable def g (x a : ℝ) : ℝ := x - a

theorem part1 (a : ℝ) : (∀ x : ℝ, f x a > g x a) → (-real.sqrt 3 / 2 < a ∧ a < real.sqrt 3 / 2) :=
sorry

theorem part2 (a : ℝ) : (∀ (x1 x2 : ℝ), x1 ∈ set.Icc (-2) (-1) → x2 ∈ set.Icc 2 4 → f x1 a > g x2 a) → (a > 2 / 3) :=
sorry

theorem part3 (a : ℝ) : (∃ (x1 x2 : ℝ), x1 ∈ set.Icc (-2) (-1) ∧ x2 ∈ set.Icc 2 4 ∧ f x1 a > g x2 a) → (a > 0) :=
sorry

end part1_part2_part3_l586_586056


namespace glasses_in_smaller_box_l586_586798

variable (x : ℕ)

theorem glasses_in_smaller_box (h : (x + 16) / 2 = 15) : x = 14 :=
by
  sorry

end glasses_in_smaller_box_l586_586798


namespace constant_term_proof_l586_586931

noncomputable def constant_term_in_expansion : ℝ :=
  let n := ∫ x in 0..2, 2 * x
  n

theorem constant_term_proof : 
  let n := ∫ x in 0..2, 2 * x
  (n = 4) → constant_term_in_expansion = (3 / 2) := by
  sorry

end constant_term_proof_l586_586931


namespace maggie_earnings_proof_l586_586624

def rate_per_subscription : ℕ := 5
def subscriptions_to_parents : ℕ := 4
def subscriptions_to_grandfather : ℕ := 1
def subscriptions_to_neighbor1 : ℕ := 2
def subscriptions_to_neighbor2 : ℕ := 2 * subscriptions_to_neighbor1
def total_subscriptions : ℕ := subscriptions_to_parents + subscriptions_to_grandfather + subscriptions_to_neighbor1 + subscriptions_to_neighbor2
def total_earnings : ℕ := total_subscriptions * rate_per_subscription

theorem maggie_earnings_proof : total_earnings = 55 := by
  sorry

end maggie_earnings_proof_l586_586624


namespace solve_for_a_l586_586941

theorem solve_for_a (S P Q R : Type) (a b c d : ℝ) 
  (h1 : a + b + c + d = 360)
  (h2 : ∀ (PSQ : Type), d = 90) :
  a = 270 - b - c :=
by
  sorry

end solve_for_a_l586_586941


namespace eta_expectation_and_variance_l586_586019

noncomputable def E_of_eta (η : ℝ) := 2
noncomputable def D_of_eta (η : ℝ) := 2.4

theorem eta_expectation_and_variance (ξ : ℝ) (η : ℝ) (n : ℕ) (p : ℝ) 
  (ξ_dist : ξ ~ binomial n p)
  (η_def : η = 8 - ξ)
  (np_eq_10 : n = 10)
  (p_eq_0_6 : p = 0.6) :
  E η = E_of_eta η ∧ D η = D_of_eta η := sorry

end eta_expectation_and_variance_l586_586019


namespace area_of_triangle_tangent_to_curve_correct_l586_586950

noncomputable def area_of_triangle_tangent_to_curve (e : ℝ) : ℝ :=
  let f := λ x : ℝ, x * Real.log x
  let f' := λ x : ℝ, Real.log x + 1
  let tangent_at_e := λ x : ℝ, (f e) + f' e * (x - e)
  let x_intercept := e / 2
  let y_intercept := -e
  (1/2) * (x_intercept * -y_intercept)

theorem area_of_triangle_tangent_to_curve_correct :
  area_of_triangle_tangent_to_curve Real.exp = (Real.exp ^ 2) / 4 :=
  by
    sorry

end area_of_triangle_tangent_to_curve_correct_l586_586950


namespace find_interest_rate_second_account_l586_586348

def interest_rate_second_account 
  (total_investment : ℝ) 
  (interest_first_account_rate : ℝ) 
  (total_interest : ℝ) 
  (amount_invested_first_account : ℝ) 
  : ℝ := 
  let interest_first_account := amount_invested_first_account * interest_first_account_rate
  let amount_invested_second_account := total_investment - amount_invested_first_account
  let interest_second_account := total_interest - interest_first_account
  interest_second_account / amount_invested_second_account

theorem find_interest_rate_second_account : 
  interest_rate_second_account 8000 0.08 490 3000 = 0.05 :=
by 
  -- The actual proof would be here, but according to instructions, we skip it.
  sorry

end find_interest_rate_second_account_l586_586348


namespace f_has_neither_maximum_nor_minimum_l586_586989

noncomputable def f : ℝ → ℝ :=
sorry

axiom f_domain : ∀ x : ℝ, x > 0 → f x = f x

axiom f_derivative_eq : ∀ x : ℝ, x > 0 → x ^ 4 * (deriv^[2] f x) + 3 * x ^ 3 * f x = Real.exp x

axiom f_at_3 : f 3 = Real.exp 3 / 81

theorem f_has_neither_maximum_nor_minimum :
  (¬ ∃ x : ℝ, x > 0 ∧ is_maximum (f x)) ∧ (¬ ∃ x : ℝ, x > 0 ∧ is_minimum (f x)) :=
sorry

end f_has_neither_maximum_nor_minimum_l586_586989


namespace fractional_part_of_students_who_walk_home_l586_586796

theorem fractional_part_of_students_who_walk_home 
  (students_by_bus : ℚ)
  (students_by_car : ℚ)
  (students_by_bike : ℚ)
  (students_by_skateboard : ℚ)
  (h_bus : students_by_bus = 1/3)
  (h_car : students_by_car = 1/5)
  (h_bike : students_by_bike = 1/8)
  (h_skateboard : students_by_skateboard = 1/15)
  : 1 - (students_by_bus + students_by_car + students_by_bike + students_by_skateboard) = 11/40 := 
by
  sorry

end fractional_part_of_students_who_walk_home_l586_586796


namespace diagonals_in_decagon_l586_586819

theorem diagonals_in_decagon :
  let n := 10
  let d := n * (n - 3) / 2
  d = 35 :=
by
  sorry

end diagonals_in_decagon_l586_586819


namespace range_of_a_satisfying_f_a_ge_2_l586_586048

def f (x : ℝ) : ℝ :=
if x ≤ -1 then 2^(-2*x) else 2*x + 2

theorem range_of_a_satisfying_f_a_ge_2 :
  {a : ℝ | f a ≥ 2} = {a : ℝ | a ≤ -1} ∪ {a : ℝ | a ≥ 0} :=
by sorry

end range_of_a_satisfying_f_a_ge_2_l586_586048


namespace fraction_zero_value_x_l586_586180

theorem fraction_zero_value_x (x : ℝ) (h1 : (x - 2) / (1 - x) = 0) (h2 : 1 - x ≠ 0) : x = 2 := 
sorry

end fraction_zero_value_x_l586_586180


namespace geometric_sequence_common_ratio_l586_586871

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * q)
  (h_arith : -a 5 + a 6 = 2 * a 4) :
  q = -1 ∨ q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l586_586871


namespace stock_return_to_original_l586_586771

theorem stock_return_to_original (x : ℝ) (h : x * 1.3 * (1 - 3 / 13) = x) :
  ((3 / 13) * 100 ≈ 23.08) :=
sorry

end stock_return_to_original_l586_586771


namespace minimum_grid_sum_l586_586319

-- Define the problem conditions in Lean
def grid_label (i j : ℕ) : ℝ :=
  1 / (i + j - 1)

-- Assert the main theorem (we will skip the proof)
theorem minimum_grid_sum : ∃ selected_cells : Fin 10 → Fin 10,
  (∀ k1 k2 : Fin 10, k1 ≠ k2 → selected_cells k1 ≠ selected_cells k2) ∧
  (∑ i, grid_label (selected_cells i) (i + 1)) = 1 :=
by
  sorry

end minimum_grid_sum_l586_586319


namespace sum_of_digits_1000_to_2000_l586_586071

-- Define the sum of the digits of the numbers from 1000 to 2000
def sum_of_digits (n : ℕ) : ℕ :=
  if n >= 1000 ∧ n <= 2000 then (n / 1000) + (n / 100 % 10) + (n / 10 % 10) + (n % 10)
  else 0

def total_sum_of_digits : ℕ :=
  (List.range' 1000 (2001 - 1000)).sum sum_of_digits

theorem sum_of_digits_1000_to_2000 : total_sum_of_digits = 14502 := by
  sorry

end sum_of_digits_1000_to_2000_l586_586071


namespace probability_intervals_l586_586084

-- Definitions: interval (0, 1/2), and the condition interval (0, 1/3)
def interval1 := set.Ioo (0:ℝ) (1/2)
def interval2 := set.Ioo (0:ℝ) (1/3)

-- The proof statement: probability of randomly selecting a number from (0, 1/2), being within (0, 1/3), is 2/3.
theorem probability_intervals : (interval2.measure / interval1.measure = 2 / 3) :=
sorry

end probability_intervals_l586_586084


namespace probability_less_than_third_l586_586116

def interval_real (a b : ℝ) := set.Ioo a b

def probability_space (a b : ℝ) := {
  length := b - a,
  sample_space := interval_real a b
}

def probability_of_event {a b c : ℝ} (h : a < b) (h1 : a < c) (h2 : c < b) : Prop :=
  (c - a) / (b - a)

theorem probability_less_than_third :
  probability_of_event (by norm_num : (0:ℝ) < (1 / 2))
                       (by norm_num : (0:ℝ) < (1 / 3))
                       (by norm_num : (1 / 3) < (1 / 2))
  = (2 / 3) := sorry

end probability_less_than_third_l586_586116


namespace term_2018_is_neg_1_l586_586649

def sequence : ℕ → ℤ
| 0 := 1
| 1 := -1
| 2 := 2
| 3 := 0
| (n + 4) := sequence n

theorem term_2018_is_neg_1 : sequence 2018 = -1 :=
by
  sorry

end term_2018_is_neg_1_l586_586649


namespace angle_normal_vector_y_axis_90_deg_l586_586381

-- Define a vector representing the normal vector of the plane α
def normal_vector : Vector ℝ 3 := ![1, -1, 0]

-- Define the y-axis unit vector
def y_axis : Vector ℝ 3 := ![0, 1, 0]

-- Formalize the problem statement: prove the angle between normal vector and y-axis is 90 degrees
theorem angle_normal_vector_y_axis_90_deg :
  let dot_product := normal_vector ⬝ y_axis in
  let y_axis_len := ‖y_axis‖ in
  let normal_vector_len := ‖normal_vector‖ in
  dot_product = 0 ∧ (normal_vector_len ≠ 0 ∧ y_axis_len ≠ 0) →
  arccos (dot_product / (normal_vector_len * y_axis_len)) = π / 2 :=
by
  sorry

end angle_normal_vector_y_axis_90_deg_l586_586381


namespace problem_g3_1_l586_586080

theorem problem_g3_1 (a : ℝ) : 
  (2002^3 + 4 * 2002^2 + 6006) / (2002^2 + 2002) = a ↔ a = 2005 := 
sorry

end problem_g3_1_l586_586080


namespace system_of_equations_correct_l586_586184

def question_statement (x y : ℕ) : Prop :=
  x + y = 12 ∧ 6 * x = 3 * 4 * y

theorem system_of_equations_correct
  (x y : ℕ)
  (h1 : x + y = 12)
  (h2 : 6 * x = 3 * 4 * y)
: question_statement x y :=
by
  unfold question_statement
  exact ⟨h1, h2⟩

end system_of_equations_correct_l586_586184


namespace find_value_of_f_l586_586479

def f (α : ℝ) : ℝ :=
  (cos (π / 2 + α) * sin (3 * π / 2 - α)) / (cos (-π - α) * tan (π - α))

theorem find_value_of_f : f (-25 * π / 3) = 1 / 2 :=
by
  sorry

end find_value_of_f_l586_586479


namespace max_distance_l586_586311

noncomputable def starting_cost : ℝ := 10
noncomputable def additional_cost_per_km : ℝ := 1.5
noncomputable def round_up : ℝ := 1
noncomputable def total_fare : ℝ := 19

theorem max_distance (x : ℝ) : (starting_cost + additional_cost_per_km * (x - 4)) = total_fare → x = 10 :=
by sorry

end max_distance_l586_586311


namespace decagon_diagonals_l586_586826

-- Number of diagonals calculation definition
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Proving the number of diagonals in a decagon
theorem decagon_diagonals : num_diagonals 10 = 35 := by
  sorry

end decagon_diagonals_l586_586826


namespace rectangular_garden_area_l586_586294

theorem rectangular_garden_area (w l : ℝ) 
  (h1 : l = 3 * w + 30) 
  (h2 : 2 * (l + w) = 800) : w * l = 28443.75 := 
by
  sorry

end rectangular_garden_area_l586_586294


namespace probability_intervals_l586_586086

-- Definitions: interval (0, 1/2), and the condition interval (0, 1/3)
def interval1 := set.Ioo (0:ℝ) (1/2)
def interval2 := set.Ioo (0:ℝ) (1/3)

-- The proof statement: probability of randomly selecting a number from (0, 1/2), being within (0, 1/3), is 2/3.
theorem probability_intervals : (interval2.measure / interval1.measure = 2 / 3) :=
sorry

end probability_intervals_l586_586086


namespace proof_problem_l586_586974

noncomputable theory

-- We define our variables and assumptions
variables {x y z a : ℝ}
variables (hx : x ≠ y) (hy : y ≠ z) (hz : z ≠ x)
variables (h1 : x^2 - 1/y = a) (h2 : y^2 - 1/z = a) (h3 : z^2 - 1/x = a)

-- Our goal is to prove the given statement
theorem proof_problem : (x + y + z) * x * y * z = -a^2 :=
by
  -- Here we would include the proof steps, but we replace it with sorry to denote it is not included
  sorry

end proof_problem_l586_586974


namespace luke_total_points_l586_586623

theorem luke_total_points (points_per_round rounds_played : ℕ) (h1 : points_per_round = 11) (h2 : rounds_played = 14) : points_per_round * rounds_played = 154 :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end luke_total_points_l586_586623


namespace height_of_screen_is_100_l586_586671

-- Definitions for the conditions and the final proof statement
def side_length_of_square_paper := 20 -- cm

def perimeter_of_square_paper (s : ℕ) : ℕ := 4 * s

def height_of_computer_screen (P : ℕ) := P + 20

theorem height_of_screen_is_100 :
  let s := side_length_of_square_paper in
  let P := perimeter_of_square_paper s in
  height_of_computer_screen P = 100 :=
by
  sorry

end height_of_screen_is_100_l586_586671


namespace arrangement_count_l586_586904

theorem arrangement_count : 
  (∑ m in Finset.range 7, (Nat.choose 6 m) ^ 3) =
  ∑ k in Finset.range 7, (Nat.choose 6 k) ^ 3 :=
begin
  sorry,
end

end arrangement_count_l586_586904


namespace number_of_positive_divisors_of_360_is_24_l586_586909

theorem number_of_positive_divisors_of_360_is_24 :
  ∀ n : ℕ, n = 360 → n = 2^3 * 3^2 * 5^1 → 
  (n_factors : {p : ℕ × ℕ // p.1 ∈ [2, 3, 5] ∧ p.2 ∈ [3, 2, 1]} )
    → (n_factors.val.snd + 1).prod = 24 :=
by
  intro n hn h_factors
  rw hn at *
  have factors := h_factors.val
  cases factors with p_k q_l r_m
  have hpq : p_k.1 = 2 ∧ p_k.2 = 3 :=
    And.intro sorry sorry,
  have hqr : q_l.1 = 3 ∧ q_l.2 = 2 :=
    And.intro sorry sorry,
  have hr : r_m.1 = 5 ∧ r_m.2 = 1 :=
    And.intro sorry sorry,
  -- The proof would continue, but we'll skip it
  sorry

end number_of_positive_divisors_of_360_is_24_l586_586909


namespace probability_merlin_dismissed_l586_586215

noncomputable def merlin_dismissal_probability {p : ℝ} (hp : 0 ≤ p ∧ p ≤ 1) : ℝ :=
  let q := 1 - p in
  -- Assuming the coin flip is fair, the probability Merlin is dismissed
  1 / 2

theorem probability_merlin_dismissed (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) :
  merlin_dismissal_probability hp = 1 / 2 :=
by
  -- Placeholder for the proof
  sorry

end probability_merlin_dismissed_l586_586215


namespace perimeter_of_plot_l586_586677

variable (length breadth : ℝ)
variable (h_ratio : length / breadth = 7 / 5)
variable (h_area : length * breadth = 5040)

theorem perimeter_of_plot (h_ratio : length / breadth = 7 / 5) (h_area : length * breadth = 5040) : 
  (2 * length + 2 * breadth = 288) :=
sorry

end perimeter_of_plot_l586_586677


namespace goods_train_length_is_280_meters_l586_586758

def speed_of_man_train_kmph : ℝ := 80
def speed_of_goods_train_kmph : ℝ := 32
def time_to_pass_seconds : ℝ := 9

theorem goods_train_length_is_280_meters :
  let relative_speed_kmph := speed_of_man_train_kmph + speed_of_goods_train_kmph
  let relative_speed_mps := relative_speed_kmph * (1000 / 3600)
  let length_of_goods_train := relative_speed_mps * time_to_pass_seconds
  abs (length_of_goods_train - 280) < 1 :=
by
  -- skipping the proof
  sorry

end goods_train_length_is_280_meters_l586_586758


namespace num_divisors_360_l586_586919

theorem num_divisors_360 :
  ∀ n : ℕ, n = 360 → (∀ (p q r : ℕ), p = 2 ∧ q = 3 ∧ r = 5 →
    (∃ (a b c : ℕ), 360 = p^a * q^b * r^c ∧ a = 3 ∧ b = 2 ∧ c = 1) →
    (3+1) * (2+1) * (1+1) = 24) :=
  sorry

end num_divisors_360_l586_586919


namespace divisibility_l586_586615

theorem divisibility (a : ℤ) : (5 ∣ a^3) ↔ (5 ∣ a) := 
by sorry

end divisibility_l586_586615


namespace f_half_31_l586_586790

noncomputable def f : ℝ → ℝ :=
  sorry -- will be provided in the proof

variables {x : ℝ}

theorem f_half_31 [odd_function f] [even_function (λ x, f (x + 1))] (h_piecewise : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x * (3 - 2 * x)) :
  f (31 / 2) = -1 :=
by sorry

end f_half_31_l586_586790


namespace portion_divided_equally_for_efforts_l586_586709

-- Definitions of conditions
def tom_investment : ℝ := 700
def jerry_investment : ℝ := 300
def tom_more_than_jerry : ℝ := 800
def total_profit : ℝ := 3000

-- Theorem stating what we need to prove
theorem portion_divided_equally_for_efforts (T J R E : ℝ) 
  (h1 : T = tom_investment)
  (h2 : J = jerry_investment)
  (h3 : total_profit = R)
  (h4 : (E / 2) + (7 / 10) * (R - E) - (E / 2 + (3 / 10) * (R - E)) = tom_more_than_jerry) 
  : E = 1000 :=
by
  sorry

end portion_divided_equally_for_efforts_l586_586709


namespace sum_possible_values_l586_586861

def abs_eq_2023 (a : ℤ) : Prop := abs a = 2023
def abs_eq_2022 (b : ℤ) : Prop := abs b = 2022
def greater_than (a b : ℤ) : Prop := a > b

theorem sum_possible_values (a b : ℤ) (h1 : abs_eq_2023 a) (h2 : abs_eq_2022 b) (h3 : greater_than a b) :
  a + b = 1 ∨ a + b = 4045 := 
sorry

end sum_possible_values_l586_586861


namespace solution_l586_586646

-- Define the variable a and b
variable (a b : ℝ)

-- Define the constant k which relates a and b
def k := a * b^3

-- Given the first condition: a = 8 when b = 2
def condition1 := (8 : ℝ) * (2 : ℝ)^3 = k

-- Set k to the computed value
def value_of_k := (64 : ℝ)

-- Given the second condition: a * b^3 = 64
def condition2 := a * (4 : ℝ)^3 = value_of_k

-- Prove that a = 1 when b = 4
theorem solution : condition1 → condition2 → a = 1 := by
  sorry

end solution_l586_586646


namespace probability_less_than_third_l586_586115

def interval_real (a b : ℝ) := set.Ioo a b

def probability_space (a b : ℝ) := {
  length := b - a,
  sample_space := interval_real a b
}

def probability_of_event {a b c : ℝ} (h : a < b) (h1 : a < c) (h2 : c < b) : Prop :=
  (c - a) / (b - a)

theorem probability_less_than_third :
  probability_of_event (by norm_num : (0:ℝ) < (1 / 2))
                       (by norm_num : (0:ℝ) < (1 / 3))
                       (by norm_num : (1 / 3) < (1 / 2))
  = (2 / 3) := sorry

end probability_less_than_third_l586_586115


namespace find_b_l586_586290

theorem find_b (b : ℕ) (h : (5 * b + 17) / (7 * b + 12) = 0.85) : b = 7 :=
sorry

end find_b_l586_586290


namespace A_n_squared_l586_586547

-- Define C(n-2)
def C_n_2 (n : ℕ) : ℕ := n * (n - 1) / 2

-- Define A_n_2
def A_n_2 (n : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - 2)

theorem A_n_squared (n : ℕ) (hC : C_n_2 n = 15) : A_n_2 n = 30 := by
  sorry

end A_n_squared_l586_586547


namespace probability_two_tails_l586_586007

theorem probability_two_tails :
  let head_prob := 1/4 in
  let tail_prob := 3/4 in
  let num_flips := 7 in
  let num_tails := 2 in
  (Nat.choose num_flips num_tails) * (tail_prob ^ num_tails) * (head_prob ^ (num_flips - num_tails)) = 189 / 16384 :=
sorry

end probability_two_tails_l586_586007


namespace soccer_substitution_modulo_l586_586391

def number_of_substitutions (max_subs : ℕ) : ℕ := 
  match max_subs with
  | 0 => 1
  | n + 1 => (11 * (13 - (n + 1)) * number_of_substitutions n)

theorem soccer_substitution_modulo :
  let n := number_of_substitutions 0 + number_of_substitutions 1 + number_of_substitutions 2 
            + number_of_substitutions 3 + number_of_substitutions 4 in
  (n % 1000) = 25 := 
by
  sorry

end soccer_substitution_modulo_l586_586391


namespace total_amount_is_4200_l586_586359

variables (p q r : ℕ)
variable (total_amount : ℕ)
variable (r_has_two_thirds : total_amount / 3 * 2 = 2800)
variable (r_value : r = 2800)

theorem total_amount_is_4200 (h1 : total_amount / 3 * 2 = 2800) (h2 : r = 2800) : total_amount = 4200 :=
by
  sorry

end total_amount_is_4200_l586_586359


namespace eval_i_powers_l586_586450

noncomputable def i : ℂ := complex.I

theorem eval_i_powers : 
  i^(23456:ℕ) + i^(23457:ℕ) + i^(23458:ℕ) + i^(23459:ℕ) = 0 :=
by
  sorry

end eval_i_powers_l586_586450


namespace jenny_hours_left_l586_586206

theorem jenny_hours_left 
    (h_research : ℕ := 10)
    (h_proposal : ℕ := 2)
    (h_visual_aids : ℕ := 5)
    (h_editing : ℕ := 3)
    (h_total : ℕ := 25) :
    h_total - (h_research + h_proposal + h_visual_aids + h_editing) = 5 := by
  sorry

end jenny_hours_left_l586_586206


namespace szilveszter_age_l586_586647

theorem szilveszter_age :
  ∃ (a b : ℕ), a ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
               b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
               11 * a + 2 * b = 89 ∧
               1999 - (1900 + 10 * a + b) = 23 :=
by
  sorry

end szilveszter_age_l586_586647


namespace stratified_sampling_selection_l586_586372

theorem stratified_sampling_selection
  (total_employees : ℕ)
  (middle_managers : ℕ)
  (senior_managers : ℕ)
  (selection_total : ℕ)
  (N_e : ℕ := total_employees - middle_managers - senior_managers)
  (N_m : ℕ := middle_managers)
  (N_s : ℕ := senior_managers)
  (N_total : ℕ := N_e + N_m + N_s)
  (P_e : ℚ := (N_e : ℚ) / N_total)
  (P_m : ℚ := (N_m : ℚ) / N_total)
  (P_s : ℚ := (N_s : ℚ) / N_total)
  (n_e : ℕ := int.to_nat (int.of_nat P_e * selection_total))
  (n_m : ℕ := int.to_nat (int.of_nat P_m * selection_total))
  (n_s : ℕ := int.to_nat (int.of_nat P_s * selection_total))
  (adjusted_n_m : ℕ := n_m + 1)
  (final_n_m : ℕ := if n_e + adjusted_n_m + n_s > selection_total then n_m else adjusted_n_m)
  (final_n_e : ℕ := n_e)
  (final_n_s : ℕ := n_s) :
  total_employees = 160 ∧ middle_managers = 30 ∧ senior_managers = 10 ∧ selection_total = 20 →
  final_n_e = 15 ∧ final_n_m = 3 ∧ final_n_s = 1 := 
by
  intros h
  cases h
  sorry

end stratified_sampling_selection_l586_586372


namespace island_of_misfortune_l586_586259

theorem island_of_misfortune (inhabitant1: ℕ) (inhabitant2: ℕ) (inhabitant3: ℕ) (inhabitant4: ℕ) (inhabitant5: ℕ) 
  (H1: inhabitant1 = 1) (H2: inhabitant2 = 2) (H3: inhabitant3 = 3) (H4: inhabitant4 = 4) (H5: inhabitant5 = 5) : 
  ∃ k, k = 4 :=
by
  have : (inhabitant4 = 4) := H4
  use 4
  assumption
  sorry

end island_of_misfortune_l586_586259


namespace find_AC_l586_586307

noncomputable def AC := Classical.some (exists_of_singleton (3 : ℝ))

theorem find_AC (AB BC DE : ℝ) (h1 : AB = 3) (h2 : BC = 2 * AC) (h3 : DE = 1) (h4 : true): AC = Real.sqrt 3 :=
by 
  sorry

end find_AC_l586_586307


namespace KDH_is_isosceles_l586_586491

noncomputable def is_parallelogram (A B C D : Type) : Prop :=
  -- Definition of a parallelogram
  sorry

noncomputable def is_isosceles (A B C : Type) (a b : Prop) : Prop :=
  -- Definition of an isosceles triangle
  sorry

noncomputable def triangles_isosceles_KDH (A B C D H K : Type) (parallelogram: is_parallelogram A B C D) (isosceles1: is_isosceles K A B K A) (isosceles2: is_isosceles H C B C B) : Prop :=
  -- Prove that triangle KDH is isosceles
  sorry

theorem KDH_is_isosceles : ∀ (A B C D H K : Type),
  is_parallelogram A B C D →
  is_isosceles K A B K A →
  is_isosceles H C B C B →
  triangles_isosceles_KDH A B C D H K :=
by
  intros
  sorry

end KDH_is_isosceles_l586_586491


namespace rectangle_is_axisymmetric_and_central_symmetric_l586_586727

structure Figure where
  is_axisymmetric : Prop
  is_central_symmetric : Prop

def rectangle : Figure := {
  is_axisymmetric := true,
  is_central_symmetric := true
}

theorem rectangle_is_axisymmetric_and_central_symmetric :
  rectangle.is_axisymmetric ∧ rectangle.is_central_symmetric :=
by
  split
  repeat
    sorry

end rectangle_is_axisymmetric_and_central_symmetric_l586_586727


namespace least_area_triangle_DEF_correct_l586_586310

noncomputable def least_area_triangle_DEF : ℝ :=
  let z := 2^(1/2) in
  let vertex (k : ℕ) : ℂ := z * complex.exp (2 * real.pi * complex.I * k / 12) in
  let z₀ := vertex 0 in
  let z₁ := vertex 1 in
  let z₂ := vertex 2 in
  let area_triangle (a b c : ℂ) : ℝ := 
    abs (a.re * (b.im - c.im) + b.re * (c.im - a.im) + c.re * (a.im - b.im)) / 2 in
  area_triangle z₀ z₁ z₂

theorem least_area_triangle_DEF_correct :
  least_area_triangle_DEF = (real.sqrt (18 - 6 * real.sqrt 3)) / 2 :=
sorry

end least_area_triangle_DEF_correct_l586_586310


namespace average_weight_l586_586657

/-- 
Given the following conditions:
1. (A + B) / 2 = 40
2. (B + C) / 2 = 41
3. B = 27
Prove that the average weight of a, b, and c is 45 kg.
-/
theorem average_weight (A B C : ℝ) 
  (h1 : (A + B) / 2 = 40)
  (h2 : (B + C) / 2 = 41)
  (h3 : B = 27): 
  (A + B + C) / 3 = 45 :=
by
  sorry

end average_weight_l586_586657


namespace probability_merlin_dismissed_l586_586220

variable (p : ℝ) (q : ℝ := 1 - p)
variable (r : ℝ := 1 / 2)

theorem probability_merlin_dismissed :
  (0 < p ∧ p ≤ 1) →
  (q = 1 - p) →
  (r = 1 / 2) →
  (probability_king_correct_two_advisors = p) →
  (strategy_merlin_minimizes_dismissal = true) →
  (strategy_percival_honest = true) →
  probability_merlin_dismissed = 1 / 2 := by
  sorry

end probability_merlin_dismissed_l586_586220


namespace square_combinations_l586_586546

theorem square_combinations (n : ℕ) (h : n * (n - 1) = 30) : n * (n - 1) = 30 :=
by sorry

end square_combinations_l586_586546


namespace parabola_solution_l586_586523

def parabola := {p : ℝ // p > 0}

noncomputable def focus (p : parabola) := (p.1 / 2, 0)

def line_through_focus (x y : ℝ) := k : ℝ → F : suplines z L := temp

theorem parabola_solution :
  ∃ p : parabola, p.1 = 8 ∧ 
  ∀ (line l : (ℝ × ℝ) → Prop), 
    (l (4,0)) ∧ 
    ({M N : ℝ × ℝ | l M ∧ l N ∧ on_parabola p M ∧ on_parabola p N}) → 
    (min_value_formula M N = 1 / 3) :=
begin
  sorry
end

def on_parabola (p : parabola) (point : ℝ × ℝ) :=
  point.2 ^ 2 = 2 * p.1 * point.1

def min_value_formula (M N : ℝ × ℝ) : ℝ :=
  |(N.1, N.2).dist (4,0)| / 9 - 4 / |(M.1, M.2).dist (4,0)|

open_locale classical

noncomputable theory

lemma on_vertical_line_through_focus (p : parabola) (M N : ℝ × ℝ) :
  M.1 = 4 ∧ N.1 = 4 → 
  min_value_formula M N = 7 / 18 :=
begin
  sorry
end

end parabola_solution_l586_586523


namespace scientific_notation_of_12400_l586_586315

theorem scientific_notation_of_12400 :
  12400 = 1.24 * 10^4 :=
sorry

end scientific_notation_of_12400_l586_586315


namespace cousins_distribution_l586_586254

theorem cousins_distribution : 
  ∃ (n : ℕ), (∀ (c c1 c2 c3 c4 : set ℕ), 
  c = {1, 2, 3, 4, 5} ∧
  c1 ∪ c2 ∪ c3 ∪ c4 = c ∧ 
  c1 ∩ c2 = ∅ ∧ c2 ∩ c3 = ∅ ∧ c3 ∩ c4 = ∅ ∧ c1 ∩ c3 = ∅ ∧ c1 ∩ c4 = ∅ ∧ c2 ∩ c4 = ∅ → 
  {c1, c2, c3, c4}.size = n) ∧ n = 51 :=
by
  sorry

end cousins_distribution_l586_586254


namespace question1_question2_l586_586041

variable (m : ℝ)

def p : set ℝ := { x | -1 ≤ x ∧ x ≤ 2}
def q : set ℝ := { x | -2 * m - 1 < x ∧ x < m + 1 }

theorem question1 (h : m > -2 / 3) :
  (p m ⊆ q m) → (m > 1) :=
by sorry

theorem question2 (h : m > -2 / 3) :
  (q m ⊆ p m) → (-2 / 3 < m ∧ m ≤ 0) :=
by sorry

end question1_question2_l586_586041


namespace roots_polynomial_product_l586_586607

theorem roots_polynomial_product (a b c : ℝ) (h₁ : Polynomial.eval a (Polynomial.Coeff (Polynomial.coeff 3) - 15 * Polynomial.coeff 2 + 22 * Polynomial.coeff 1 - 8 * Polynomial.X 0) = 0)
(h₂ : Polynomial.eval b (Polynomial.Coeff (Polynomial.coeff 3) - 15 * Polynomial.coeff 2 + 22 * Polynomial.coeff 1 - 8 * Polynomial.X 0) = 0)
(h₃ : Polynomial.eval c (Polynomial.Coeff (Polynomial.coeff 3) - 15 * Polynomial.coeff 2 + 22 * Polynomial.coeff 1 - 8 * Polynomial.X 0) = 0) :
(1 + a) * (1 + b) * (1 + c) = 46 :=
sorry

end roots_polynomial_product_l586_586607


namespace probability_less_than_one_third_l586_586093

theorem probability_less_than_one_third : 
  (∃ p : ℚ, p = (measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ)) (set.Ioo (0 : ℚ) (1 / 3 : ℚ)) / 
                  measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ))) ∧ p = 2 / 3) :=
by {
  -- The proof will be added here.
  sorry
}

end probability_less_than_one_third_l586_586093


namespace graph_shift_sin_2x_l586_586322

theorem graph_shift_sin_2x :
  ∀ (x : ℝ), (sin (2 * (x - π/12) + π/6) = sin (2 * x)) :=
by 
  sorry

end graph_shift_sin_2x_l586_586322


namespace find_A_l586_586655

variable (p q r s A : ℝ)

theorem find_A (H1 : (p + q + r + s) / 4 = 5) (H2 : (p + q + r + s + A) / 5 = 8) : A = 20 := 
by
  sorry

end find_A_l586_586655


namespace coefficient_x2y3z2_in_expansion_l586_586815

/-- Define the polynomial expansion terms and conditions. -/
def polynomial_expansion := (x y z : ℚ) → (x - y) * (x + 2 * y + z)^6

/-- Prove that the coefficient of x^2 y^3 z^2 is 120. -/
theorem coefficient_x2y3z2_in_expansion (x y z : ℚ) :
  coeff (monomial (single 2 1 + single 3 1 + single 2 1) 1) (polynomial_expansion x y z) = 120 :=
sorry

end coefficient_x2y3z2_in_expansion_l586_586815


namespace probability_cheryl_same_color_l586_586369

def num_ways_to_draw_marbles : ℕ := (Nat.choose 9 3) * (Nat.choose 6 3) * (Nat.choose 3 3)

def favorable_outcomes_for_cheryl : ℕ := 3 * (Nat.choose 6 3)

theorem probability_cheryl_same_color : 
  Real.of_nat favorable_outcomes_for_cheryl / Real.of_nat num_ways_to_draw_marbles = 1 / 28 := by
  sorry

end probability_cheryl_same_color_l586_586369


namespace hyperbola_range_m_l586_586505

theorem hyperbola_range_m (m : ℝ) (hC : ∀ x y : ℝ, mx^2 + (2 - m)y^2 = 1) : m ∈ set.Ioi 2 := sorry

end hyperbola_range_m_l586_586505


namespace preimage_of_20_is_4_l586_586619

def f (n : ℕ) : ℕ := 2^n + n

theorem preimage_of_20_is_4 : ∃ n : ℕ, f(n) = 20 ∧ n = 4 :=
by {
  sorry
}

end preimage_of_20_is_4_l586_586619


namespace product_distance_inequality_l586_586406

open Real

variables {A B C O A' B' C' : Point}
variables [isAcuteTriangle : acute_triangle A B C]

variables (circumcenter_O : is_circumcenter O A B C)
variables (R : circumradius O A B C)
variables (p1 : extend AO intersects circumcircle BOC at A')
variables (p2 : extend BO intersects circumcircle AOC at B')
variables (p3 : extend CO intersects circumcircle AOB at C')

theorem product_distance_inequality :
  (dist O A') * (dist O B') * (dist O C') ≥ 8 * R ^ 3 := 
sorry

end product_distance_inequality_l586_586406


namespace valid_license_plates_l586_586195

def letters := 26
def digits := 10
def totalPlates := letters^3 * digits^4

theorem valid_license_plates : totalPlates = 175760000 := by
  sorry

end valid_license_plates_l586_586195


namespace range_of_k_l586_586179

theorem range_of_k (k : ℝ) : (x^2 + k * y^2 = 2) ∧ (k > 0) ∧ (k < 1) ↔ (0 < k ∧ k < 1) :=
by
  sorry

end range_of_k_l586_586179


namespace log_base_2_of_3_l586_586010

theorem log_base_2_of_3 (a b : ℝ) (h1 : a = Real.log 6) (h2 : b = Real.log 20) : 
  Real.log₂ 3 = (a - b + 1) / (b - 1) :=
sorry

end log_base_2_of_3_l586_586010


namespace BC_parallel_AD_l586_586289

-- Define the convex quadrilateral ABCD
structure ConvexQuadrilateral where
  A B C D : Type
  convex : Convex α

-- Assume the conditions from the problem
variables {A B C D : Point}
variables {ABCircle CDCircle : Circle}
variables (h1 : ABCircle.diameter = distance A B)
variables (h2 : CDCircle.diameter = distance C D)
variables (h3 : ABCircle.tangentCD = true)
variables (h4 : CDCircle.tangentAB = true)

-- Prove that BC is parallel to AD
theorem BC_parallel_AD (quad : ConvexQuadrilateral A B C D) 
    (h1 : ABCircle.diameter = distance A B)
    (h2 : CDCircle.diameter = distance C D)
    (h3 : ABCircle.tangentCD = true)
    (h4 : CDCircle.tangentAB = true) : BC ∥ AD := 
begin
  sorry,
end

end BC_parallel_AD_l586_586289


namespace max_sum_china_l586_586682

noncomputable theory
open_locale classical

def char_val := Fin 25

def sum_except (s : Finset char_val) (exclude: Finset char_val) : ℕ :=
  ∑ i in (s \ exclude), i.val

theorem max_sum_china (A : Finset char_val) (S: ℕ) (avg : S = 34 * 12 + 14) 
  (Hdistinct : (∀ a b ∈ A, a ≠ b → a.val ≠ b.val)) :
  (∃ 中 华 : char_val , 中 ≠ 华 ∧ 中.val + 华.val = 46) :=
by
  sorry

end max_sum_china_l586_586682


namespace num_lines_with_integer_chord_length_l586_586295

-- Definitions based on the given problem
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 25

def line_eq (m x y : ℝ) : Prop := m * x - y - 4 * m + 1 = 0

def chord_length (m : ℝ) : ℝ := 2 * (Real.sqrt (25 - (Real.abs (4 * m - 1) / (Real.sqrt (1 + m^2))) ^ 2))

-- The statement to be proved
theorem num_lines_with_integer_chord_length : 
  (∃ n : ℕ, n = 9 ∧ ∀ m : ℝ, line_eq m 4 1 → circle_eq 4 1 → 
    chord_length m ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) :=
sorry

end num_lines_with_integer_chord_length_l586_586295


namespace divides_14_pow_n_minus_27_for_all_natural_numbers_l586_586475

theorem divides_14_pow_n_minus_27_for_all_natural_numbers :
  ∀ n : ℕ, 13 ∣ 14^n - 27 :=
by sorry

end divides_14_pow_n_minus_27_for_all_natural_numbers_l586_586475


namespace unique_four_digit_products_l586_586676

theorem unique_four_digit_products :
  ∃ l : List ℕ, 
    l = [1, 625, 2401, 4096, 6561] ∧ 
    ∀ n : ℕ, 
      (n ∈ l ↔ ∃ d : ℕ, 1 ≤ d ∧ d ≤ 9 ∧ 
                          (1000 * d + 100 * d + 10 * d + d).digits.prod = n ∧ 
                          (∀ m : ℕ, (1000 * m + 100 * m + 10 * m + m).digits.prod = n → 
                            m = d)) :=
by 
  -- Add proof steps here
  sorry

end unique_four_digit_products_l586_586676


namespace scientists_ratio_correct_l586_586260

noncomputable def scientists_proj (total_scientists : ℕ) (perc_Germany perc_Japan perc_USA : ℝ) : Prop :=
  let Germany := (perc_Germany * total_scientists).to_nat in
  let Japan := (perc_Japan * total_scientists).to_nat in
  let USA := (perc_USA * total_scientists).to_nat in
  Germany = 27 ∧ Japan = 18 ∧ USA = 12 ∧ Germany / 3 = 9 ∧ Japan / 3 = 6 ∧ USA / 3 = 4

theorem scientists_ratio_correct :
  scientists_proj 150 0.18 0.12 0.08 := by
  sorry

end scientists_ratio_correct_l586_586260


namespace fiona_working_hours_l586_586561

theorem fiona_working_hours (F : ℕ) 
  (John_hours_per_week : ℕ := 30) 
  (Jeremy_hours_per_week : ℕ := 25) 
  (pay_rate : ℕ := 20) 
  (monthly_total_pay : ℕ := 7600) : 
  4 * (John_hours_per_week * pay_rate + Jeremy_hours_per_week * pay_rate + F * pay_rate) = monthly_total_pay → 
  F = 40 :=
by sorry

end fiona_working_hours_l586_586561


namespace sunflower_mix_is_50_percent_l586_586753

-- Define the proportions and percentages given in the problem
def prop_A : ℝ := 0.60 -- 60% of the mix is Brand A
def prop_B : ℝ := 0.40 -- 40% of the mix is Brand B
def sf_A : ℝ := 0.60 -- Brand A is 60% sunflower
def sf_B : ℝ := 0.35 -- Brand B is 35% sunflower

-- Define the final percentage of sunflower in the mix
noncomputable def sunflower_mix_percentage : ℝ :=
  (sf_A * prop_A) + (sf_B * prop_B)

-- Statement to prove that the percentage of sunflower in the mix is 50%
theorem sunflower_mix_is_50_percent : sunflower_mix_percentage = 0.50 :=
by
  sorry

end sunflower_mix_is_50_percent_l586_586753


namespace can_tile_with_1_crosses_can_tile_with_2_crosses_cannot_tile_with_k_crosses_l586_586659

-- Definitions based on the conditions
def k_cross_squares (k : ℕ) : ℕ := 6 * k + 1

-- (a) Prove that space can be tiled with 1-crosses.
theorem can_tile_with_1_crosses :
  ∃ (tile : ℤ × ℤ × ℤ → bool), ∀ (point : ℤ × ℤ × ℤ),
  (∃ offset, tile (point.1 + offset, point.2, point.3)) ∨
  (∃ offset, tile (point.1 - offset, point.2, point.3)) ∨
  (∃ offset, tile (point.1, point.2 + offset, point.3)) ∨
  (∃ offset, tile (point.1, point.2 - offset, point.3)) ∨
  (∃ offset, tile (point.1, point.2, point.3 + offset)) ∨
  (∃ offset, tile (point.1, point.2, point.3 - offset)) :=
sorry

-- (b) Prove that space can be tiled with 2-crosses.
theorem can_tile_with_2_crosses :
  ∃ (tile : ℤ × ℤ × ℤ → bool), ∀ (point : ℤ × ℤ × ℤ),
  (∃ offset, tile (point.1 + 2 * offset, point.2, point.3)) ∨
  (∃ offset, tile (point.1 - 2 * offset, point.2, point.3)) ∨
  (∃ offset, tile (point.1, point.2 + 2 * offset, point.3)) ∨
  (∃ offset, tile (point.1, point.2 - 2 * offset, point.3)) ∨
  (∃ offset, tile (point.1, point.2, point.3 + 2 * offset)) ∨
  (∃ offset, tile (point.1, point.2, point.3 - 2 * offset)) :=
sorry

-- (c) Prove that for k ≥ 5, space cannot be tiled with k-crosses.
theorem cannot_tile_with_k_crosses (k : ℕ) (hk : k ≥ 5) :
  ¬ ∃ (tile : ℤ × ℤ × ℤ → bool), ∀ (point : ℤ × ℤ × ℤ),
  (∃ offset, tile (point.1 + k * offset, point.2, point.3)) ∨
  (∃ offset, tile (point.1 - k * offset, point.2, point.3)) ∨
  (∃ offset, tile (point.1, point.2 + k * offset, point.3)) ∨
  (∃ offset, tile (point.1, point.2 - k * offset, point.3)) ∨
  (∃ offset, tile (point.1, point.2, point.3 + k * offset)) ∨
  (∃ offset, tile (point.1, point.2, point.3 - k * offset)) :=
sorry

end can_tile_with_1_crosses_can_tile_with_2_crosses_cannot_tile_with_k_crosses_l586_586659


namespace total_cats_after_sales_and_arrivals_l586_586385

-- The counts before any sale
def initial_siamese : Nat := 12
def initial_house : Nat := 20
def initial_persian : Nat := 8
def initial_sphynx : Nat := 18

-- The amounts sold during the first sale
def first_sale_siamese : Nat := 6
def first_sale_house : Nat := 4
def first_sale_persian : Nat := 5

-- The amounts sold during the second sale
def second_sale_sphynx : Nat := 10
def second_sale_house : Nat := 15

-- The additional shipments
def shipment_siamese : Nat := 5
def shipment_persian : Nat := 3

-- The total cats left after sales and arrivals
def total_cats_left : Nat :=
    (initial_siamese - first_sale_siamese + shipment_siamese) +
    (initial_house - first_sale_house - second_sale_house) +
    (initial_persian - first_sale_persian + shipment_persian) +
    (initial_sphynx - second_sale_sphynx)

theorem total_cats_after_sales_and_arrivals : total_cats_left = 26 :=
by
  -- Calculation of each category
  let remaining_siamese := initial_siamese - first_sale_siamese + shipment_siamese
  let remaining_house := initial_house - first_sale_house - second_sale_house
  let remaining_persian := initial_persian - first_sale_persian + shipment_persian
  let remaining_sphynx := initial_sphynx - second_sale_sphynx
  -- Calculation of total
  have total := remaining_siamese + remaining_house + remaining_persian + remaining_sphynx
  -- Show total equals to 26
  show total = 26 from sorry

end total_cats_after_sales_and_arrivals_l586_586385


namespace checkers_diff_colors_exists_l586_586635

def checkerboard := (Fin₁₀ × Fin₁₀) → Bool -- Define a 10x10 checkerboard where cells can be True (white) or False (black)

def initial_checkers : checkerboard :=
  fun (i, j) => if (i + j < 91) then true else false -- Initial state with 91 white checkers

def adjacent (c1 c2 : Fin₁₀ × Fin₁₀) : Bool :=
  (c1.1 = c2.1 ∧ (c1.2 = c2.2 + 1 ∨ c1.2 = c2.2 - 1)) ∨
  (c1.2 = c2.2 ∧ (c1.1 = c2.1 + 1 ∨ c1.1 = c2.1 - 1)) -- Function to determine if two cells are adjacent

theorem checkers_diff_colors_exists : ∀ current_board : checkerboard,
  (∃ c1 c2 : Fin₁₀ × Fin₁₀, adjacent c1 c2 = true ∧ current_board c1 ≠ current_board c2) :=
by
  sorry

end checkers_diff_colors_exists_l586_586635


namespace compressor_station_distances_compressor_station_distances_when_a_is_30_l586_586691

theorem compressor_station_distances (a : ℝ) (h : 0 < a ∧ a < 60) :
  ∃ x y z : ℝ, x + y = 3 * z ∧ z + y = x + a ∧ x + z = 60 :=
sorry

theorem compressor_station_distances_when_a_is_30 :
  ∃ x y z : ℝ, 
  (x + y = 3 * z) ∧ (z + y = x + 30) ∧ (x + z = 60) ∧ 
  (x = 35) ∧ (y = 40) ∧ (z = 25) :=
sorry

end compressor_station_distances_compressor_station_distances_when_a_is_30_l586_586691


namespace express_x2_y2_z2_in_terms_of_sigma1_sigma2_l586_586453

variable (x y z : ℝ)
def sigma1 := x + y + z
def sigma2 := x * y + y * z + z * x

theorem express_x2_y2_z2_in_terms_of_sigma1_sigma2 :
  x^2 + y^2 + z^2 = sigma1 x y z ^ 2 - 2 * sigma2 x y z := by
  sorry

end express_x2_y2_z2_in_terms_of_sigma1_sigma2_l586_586453


namespace reducible_iff_form_l586_586800

def isReducible (a : ℕ) : Prop :=
  ∃ d : ℕ, d ≠ 1 ∧ d ∣ (2 * a + 5) ∧ d ∣ (3 * a + 4)

theorem reducible_iff_form (a : ℕ) : isReducible a ↔ ∃ k : ℕ, a = 7 * k + 1 := by
  sorry

end reducible_iff_form_l586_586800


namespace M_subset_N_l586_586061

def M (x : ℝ) : Prop := ∃ k : ℤ, x = (↑k * Real.pi / 2) + (Real.pi / 4)
def N (x : ℝ) : Prop := ∃ k : ℤ, x = (↑k * Real.pi / 4) + (Real.pi / 2)

theorem M_subset_N : ∀ x, M x → N x := 
by
  sorry

end M_subset_N_l586_586061


namespace probability_of_given_condition_l586_586639

section
  -- Definitions of the conditions
  def total_lamps := 8
  def red_lamps := 4
  def blue_lamps := 4
  def total_turn_on := 4

  -- Binomial coefficient
  def choose (n k : ℕ) := Nat.choose n k

  -- Probability calculation
  def probability_condition := 
    -- Total ways to choose placements and turn ons
    let total_arrangements := choose total_lamps red_lamps
    let total_turning_on := choose total_lamps total_turn_on
  
    -- Condition-specific calculations
    let remaining_positions := 5
    let remaining_red := 2
    let remaining_blue := 3
    let remaining_turn_on := 3
  
    let ways_to_place_r := choose remaining_positions remaining_red
    let ways_to_turn_on := choose remaining_positions remaining_turn_on
    
    -- Probability
    (ways_to_place_r * ways_to_turn_on) / (total_arrangements * total_turning_on : ℝ)

  -- The proof problem stating that the result equals 1/49
  theorem probability_of_given_condition : probability_condition = 1 / 49 :=
  by
    sorry
end

end probability_of_given_condition_l586_586639


namespace maximize_profit_l586_586750

noncomputable def profit (x : ℝ) : ℝ :=
  (x - 8) * (100 - 10 * (x - 10))

theorem maximize_profit :
  let max_price := 14
  let max_profit := 360
  (∀ x > 10, profit x ≤ profit max_price) ∧ profit max_price = max_profit :=
by
  let max_price := 14
  let max_profit := 360
  sorry

end maximize_profit_l586_586750


namespace inequality_range_of_a_l586_586340

theorem inequality_range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → (1 / 2) * x ^ 2 + (1 - a) * x - a * log x > 2 * a - (3 / 2) * a ^ 2) →
  a ∈ Icc 0 1 ∪ Ioi 1 :=
sorry

end inequality_range_of_a_l586_586340


namespace probability_of_interval_l586_586140

theorem probability_of_interval {x : ℝ} (h₀ : 0 < x ∧ x < 1/2) :
    Pr (λ x, x < 1/3) = 2/3 :=
sorry

end probability_of_interval_l586_586140


namespace infinitely_many_multiples_of_7_l586_586273

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n ≥ 2, a n = a (n - 1) + a (n / 2)

theorem infinitely_many_multiples_of_7
  (a : ℕ → ℕ) 
  (h : sequence a) : 
  ∃ (infinitely_many : ℕ → Prop), 
  (forall k, infinitely_many k -> k ∉ set.range a -> false) ∧ 
  (forall k, ((k % 7 = 0) -> infinitely_many k)) :=
sorry

end infinitely_many_multiples_of_7_l586_586273


namespace probability_of_interval_l586_586124

theorem probability_of_interval (a b x : ℝ) (h : 0 < a ∧ a < b ∧ 0 < x) : 
  (x < b) → (b = 1/2) → (x = 1/3) → (0 < x) → (x - 0) / (b - 0) = 2/3 := 
by 
  sorry

end probability_of_interval_l586_586124


namespace not_all_unique_names_l586_586954

noncomputable def unique_names_possible (A Y I E : Type) [monoid A] [monoid Y] [monoid I] [monoid E] 
  (E_eq : E → E) (A_repeated_seven : A → A) (Y_repeated_seven : Y → Y) (I_repeated_seven : I → I) :
  Prop :=
∀ (num_tribe_members : ℕ), num_tribe_members = 400 → 343 < 400

theorem not_all_unique_names : unique_names_possible A Y I E sorry sorry sorry sorry :=
sorry

end not_all_unique_names_l586_586954


namespace thirty_ninth_digit_l586_586556

theorem thirty_ninth_digit (digits_list : List ℕ) (h : digits_list = [3, 0, 2, 9, 2, 8, 2, 7, 2, 6, 2, 5, 2, 4, 2, 3, 2, 2, 2, 1, 2, 0, 1, 9, 1, 8, 1, 7, 1, 6, 1, 5, 1, 4, 1, 3, 1, 2, 1, 1, 9, 8, 7, 6, 5, 4, 3, 2, 1]) : 
  digits_list.nth 38 = some 1 :=
sorry

end thirty_ninth_digit_l586_586556


namespace total_arrangements_l586_586405

-- Define the conditions
def student_count : ℕ := 5
def class_count : ℕ := 3
def class_with_student_A : ℕ := 1

-- Define the question as a theorem
theorem total_arrangements : 
  (∃ (arrangements : set (fin student_count → fin class_count)), 
    (∀ (student : fin student_count), ∃ (class : fin class_count), arrangements student = class) ∧
    (∀ (class : fin class_count), ∃ (student : fin student_count), arrangements student = class) ∧
    arrangements 0 = class_with_student_A) 
    → 50 :=
sorry

end total_arrangements_l586_586405


namespace exterior_angle_of_regular_polygon_l586_586165

theorem exterior_angle_of_regular_polygon (n : ℕ) (h : 1 ≤ n) (angle : ℝ) 
  (h_angle : angle = 72) (sum_ext_angles : ∀ (polygon : Type) [fintype polygon], (∑ i, angle) = 360) : 
  n = 5 := sorry

end exterior_angle_of_regular_polygon_l586_586165


namespace angle_equality_l586_586831

-- Define the geometric setup
variables {Point Line Angle : Type}
variables [Geometry Point Line Angle]

-- Given conditions
variables (M P Q D B C A : Point)
variables (PQ BC MQ AD : Line)
variables (angle_BAM angle_BDM : Angle)

-- Define properties from the conditions
def DrawThroughParallel (M PQ : Line) : Line := sorry -- Line through M parallel to PQ intersects BC at D

def MidlineProperties (PQ : Line) (DC : Line) : Prop :=
  PQ ∥ DC ∧ midpoint DC ∈ PQ

def EqualAngles (angle1 angle2 : Angle) := angle1 = angle2

-- Main statement to prove the result
theorem angle_equality
    (h1 : ∃ D, DrawThroughParallel M PQ = (BC : Line)) -- Define point D
    (h2 : MidlineProperties PQ (line DC)) -- PQ bisects DC
    (h3 : \angle BAM = 180 - \angle BQP) -- angle BAM relationship
    (h4 : \angle BAM = 180 - \angle BDM) -- angle BAM relationship step 2
    (h5 : QP ∥ AD) -- MQ is a midline in ADC
    (h6 : EqualAngles \angle ADM \angle MQP) -- ADM == MQP
    : EqualAngles \angle ABM \angle ADM -- To be proven
:= 
sorry -- Proof is omitted as per the instruction

end angle_equality_l586_586831


namespace polynomial_transformation_correct_l586_586980

-- Define the original polynomial
def original_polynomial (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

-- Define the polynomial transformation
def transformed_polynomial (y : ℝ) : ℝ := (y - 3)^3 - 6 * (y - 3)^2 + 11 * (y - 3) - 6

-- Define the target polynomial
def target_polynomial (x : ℝ) : ℝ := x^3 - 15*x^2 + 74*x - 120

-- Main theorem
theorem polynomial_transformation_correct :
  (∀ a b c : ℝ, (original_polynomial a = 0) ∧ (original_polynomial b = 0) ∧ (original_polynomial c = 0) →
  transformed_polynomial y = original_polynomial x) ≡ target_polynomial y := by
  sorry

end polynomial_transformation_correct_l586_586980


namespace probability_merlin_dismissed_l586_586226

variable (p : ℝ) (q : ℝ) (coin_flip : ℝ)

axiom h₁ : q = 1 - p
axiom h₂ : 0 ≤ p ∧ p ≤ 1
axiom h₃ : 0 ≤ q ∧ q ≤ 1
axiom h₄ : coin_flip = 0.5

theorem probability_merlin_dismissed : coin_flip = 0.5 := by
  sorry

end probability_merlin_dismissed_l586_586226


namespace license_plate_difference_l586_586282

theorem license_plate_difference :
  let california_plates := 26^4 * 10^3
  let texas_plates := 26^3 * 10^4
  california_plates - texas_plates = 281216000 :=
by
  let california_plates := 26^4 * 10^3
  let texas_plates := 26^3 * 10^4
  have h1 : california_plates = 456976 * 1000 := by sorry
  have h2 : texas_plates = 17576 * 10000 := by sorry
  have h3 : 456976000 - 175760000 = 281216000 := by sorry
  exact h3

end license_plate_difference_l586_586282


namespace work_hours_needed_per_week_l586_586537

theorem work_hours_needed_per_week (
  planned_hours_per_week : ℕ,
  planned_weeks : ℕ,
  financial_goal : ℕ,
  missed_weeks : ℕ
) : planned_hours_per_week = 20 ∧ planned_weeks = 12 ∧ financial_goal = 3000 ∧ missed_weeks = 3 → 
    let remaining_weeks := planned_weeks - missed_weeks in
    let required_ratio := (planned_weeks : ℚ) / remaining_weeks in
    let new_workload := required_ratio * planned_hours_per_week in
    new_workload.ceil.to_nat = 27 :=
by
  intros h
  sorry

end work_hours_needed_per_week_l586_586537


namespace function_properties_l586_586888

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x + 3

theorem function_properties :
  (∃ x1 x2, x1 ≠ x2 ∧ f x1 = f x2) ∧             -- corresponds to having two extreme points
  (∀ x, f(x) + f(-x) = 6) ∧                      -- point (0,3) is the center of symmetry
  (¬ (∃ x1 x2 x3, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0)) ∧ -- does not have three zeros
  (∀ k, (∃ x1 x2, x1 ≠ x2 ∧ f x1 = k ∧ f x2 = k) → (k = 1 ∨ k = 5)) := 
by
  sorry

end function_properties_l586_586888


namespace probability_less_than_one_third_l586_586129

def interval := set.Ioo (0 : ℝ) (1 / 2)

def desired_interval := set.Ioo (0 : ℝ) (1 / 3)

theorem probability_less_than_one_third :
  (∃ p : ℝ, p = (volume desired_interval / volume interval) ∧ p = 2 / 3) :=
  sorry

end probability_less_than_one_third_l586_586129


namespace coeff_x4_of_expr_l586_586421

noncomputable def expr := 
  4 * (x^4 - 2 * x^3 + x^2) +
  2 * (3 * x^4 + x^3 - 2 * x^2 + x) -
  6 * (2 * x^2 - x^4 + 3 * x^3)

theorem coeff_x4_of_expr : 
  coefficient (expr, 4) = 4 := 
by sorry

end coeff_x4_of_expr_l586_586421


namespace probability_area_less_than_circumference_is_zero_l586_586323

noncomputable def diceRollToDiameters : List ℕ :=
  List.bind (List.range 8) (λ x => List.map (λ y => x + y + 2) (List.range 8))

def validDiameters : List ℕ :=
  diceRollToDiameters.filter (λ d => d ≥ 5)

def diameterCondition (d : ℕ) : Prop :=
  0 < d ∧ d < 4

theorem probability_area_less_than_circumference_is_zero :
  ∀ d ∈ validDiameters, ¬ diameterCondition d :=
by
  intros d hd
  simp [diceRollToDiameters, validDiameters, diameterCondition] at hd
  sorry

end probability_area_less_than_circumference_is_zero_l586_586323


namespace problem_statement_l586_586006

def is_prime (p : ℕ) : Prop := ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

noncomputable def prime_stream (n : ℕ) : ℕ := (nat.primes.drop n).head

-- Function to get the nth prime number.
noncomputable def p_i (i : ℕ) : ℕ := prime_stream (i - 1)

def all_products_sum (k : ℕ) : ℕ :=
  (finset.powerset (finset.range k)).sum (λ s, s.prod (λ i, p_i (i+1)))

theorem problem_statement (k : ℕ) (hk : k > 5) :
  ∃ S : ℕ, S = all_products_sum k ∧ (S + 1) has more than 2k distinct prime factors :=
sorry

end problem_statement_l586_586006


namespace sum_2015_terms_of_sequence_eq_l586_586047

/-- The given function f -/
def f (x : ℕ) : ℕ := 4 * x^2 - 1

/-- The sequence terms -/
def sequence_term (n : ℕ) : ℕ := 1 / (f n)

/-- The sum S_n of the first n terms of the sequence -/
noncomputable def S (n : ℕ) : ℚ := ∑ i in Finset.range n, sequence_term (i + 1)

/-- The main theorem -/
theorem sum_2015_terms_of_sequence_eq :
  S 2015 = (2015 : ℚ) / 4031 :=
sorry

end sum_2015_terms_of_sequence_eq_l586_586047


namespace three_planes_intersection_l586_586183

noncomputable def number_of_intersection_lines (P₁ P₂ P₃ : Set (Point3D)) : ℕ :=
by
  sorry

theorem three_planes_intersection
  (P₁ P₂ P₃ : Set (Point3D))
  (h₁₂ : ∃ l₁₂ : Line3D, P₁ ∩ P₂ = l₁₂)
  (h₂₃ : ∃ l₂₃ : Line3D, P₂ ∩ P₃ = l₂₃)
  (h₁₃ : ∃ l₁₃ : Line3D, P₁ ∩ P₃ = l₁₃) :
  number_of_intersection_lines P₁ P₂ P₃ = 1 ∨ number_of_intersection_lines P₁ P₂ P₃ = 3 :=
by
  sorry

end three_planes_intersection_l586_586183


namespace geometric_probability_l586_586104

theorem geometric_probability :
  let total_length := (1/2 - 0)
  let favorable_length := (1/3 - 0)
  total_length > 0 ∧ favorable_length > 0 →
  (favorable_length / total_length) = 2 / 3 :=
by
  intros _ _ _
  sorry

end geometric_probability_l586_586104


namespace average_of_4_quantities_l586_586654

theorem average_of_4_quantities (a b c d e f : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 8) 
  (h2 : ((e + f) / 2) = 14) : 
  ((a + b + c + d) / 4) = 5 :=
begin
  sorry
end

end average_of_4_quantities_l586_586654


namespace total_shaded_area_correct_l586_586786

-- Definitions based on the conditions
def tile_side : ℝ := 1 -- Each tile is 1 foot by 1 foot
def tile_area : ℝ := tile_side * tile_side -- Area of one tile

def radius : ℝ := tile_side / 2 -- Radius of each quarter circle
def quarter_circle_area : ℝ := (real.pi * radius^2) / 4
def white_area : ℝ := 4 * quarter_circle_area -- Four quarter circles make a full circle
def shaded_area_per_tile : ℝ := tile_area - white_area -- Shaded area of one tile

def floor_length : ℝ := 8 -- The floor is 8 feet by 10 feet
def floor_width : ℝ := 10
def number_of_tiles : ℝ := (floor_length * floor_width) / tile_area -- Total number of tiles

def total_shaded_area : ℝ := number_of_tiles * shaded_area_per_tile -- Total shaded area

-- The theorem to prove that the total shaded area is 80 - 20π square feet
theorem total_shaded_area_correct : total_shaded_area = 80 - 20 * real.pi :=
  by
  -- Proof left as an exercise
  sorry

end total_shaded_area_correct_l586_586786


namespace geometric_probability_l586_586101

theorem geometric_probability :
  let total_length := (1/2 - 0)
  let favorable_length := (1/3 - 0)
  total_length > 0 ∧ favorable_length > 0 →
  (favorable_length / total_length) = 2 / 3 :=
by
  intros _ _ _
  sorry

end geometric_probability_l586_586101


namespace interval_monotonic_increase_l586_586515

def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Math.sin (ω * x + φ)

theorem interval_monotonic_increase :
  ∀ (x₁ x₂ ω φ : ℝ) (k : ℤ),
    ω > 0 →
    0 < φ ∧ φ < π / 2 →
    f x₁ ω φ = 1 →
    f x₂ ω φ = 0 →
    |x₁ - x₂| = 1 / 2 →
    f (1 / 2) ω φ = 1 / 2 →
    (f (x) ω φ = sin (π * x + π / 3) →
      -5/6 + 2 * (k : ℝ) ≤ x ∧ x ≤ 1/6 + 2 * (k : ℝ)) :=
begin
  intros,
  sorry
end

end interval_monotonic_increase_l586_586515


namespace determine_real_coins_l586_586330

def has_fake_coin (coins : List ℝ) : Prop :=
  ∃ fake_coin ∈ coins, (∀ coin ∈ coins, coin ≠ fake_coin)

theorem determine_real_coins (coins : List ℝ) (h : has_fake_coin coins) (h_length : coins.length = 101) :
  ∃ real_coins : List ℝ, ∀ r ∈ real_coins, r ∈ coins ∧ real_coins.length ≥ 50 :=
by
  sorry

end determine_real_coins_l586_586330


namespace probability_of_interval_l586_586121

theorem probability_of_interval (a b x : ℝ) (h : 0 < a ∧ a < b ∧ 0 < x) : 
  (x < b) → (b = 1/2) → (x = 1/3) → (0 < x) → (x - 0) / (b - 0) = 2/3 := 
by 
  sorry

end probability_of_interval_l586_586121


namespace binomial_coefficients_sum_and_seventh_term_l586_586045

noncomputable def binomial_expansion (x : ℝ) (n : ℕ) : ℝ := (x^2 + 1/(2*sqrt(x)))^n

theorem binomial_coefficients_sum_and_seventh_term 
  (n : ℕ) (h : binomial_expansion x n = binomial_expansion x 10)
  (h1 : 1 + n + 1/2 * n * (n - 1) = 56) : n = 10 ∧ (binomial_expansion x n).coeff 7 = 105/32 * x^5 :=
by
  sorry

end binomial_coefficients_sum_and_seventh_term_l586_586045


namespace min_value_B_A_l586_586483

noncomputable def geom_sequence (a₀ r : ℚ) : ℕ → ℚ
| 0       := a₀
| (n + 1) := geom_sequence n * r

def first_term : ℚ := 4 / 3
def common_ratio : ℚ := -1 / 3

def S (n : ℕ) : ℚ := if n = 0 then 0
                      else let t := common_ratio ^ n in
                           1 - t

def A : ℚ := - 17 / 72
def B : ℚ := 7 / 12

theorem min_value_B_A :
  ∀n : ℕ, n > 0 → 
  (A ≤ S n - (1 / S n) ∧ S n - (1 / S n) ≤ B) →
  B - A = 59 / 72 :=
sorry

end min_value_B_A_l586_586483


namespace perfect_square_factors_of_420_l586_586068

-- Define the prime factorization of 420
def factor_420 : ℕ := 420
def prime_factors_420 : ℕ × ℕ × ℕ × ℕ := (2, 2, 3, 1, 5, 1, 7, 1)

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∀ p k, nat.prime p → n = p^k → even k

-- Cardinality of the set of perfect square factors of 420
def num_perfect_square_factors (n : ℕ) : ℕ :=
  { m | m ∣ n ∧ is_perfect_square m }.to_finset.card

-- The main theorem to be proven
theorem perfect_square_factors_of_420 : num_perfect_square_factors factor_420 = 2 :=
sorry

end perfect_square_factors_of_420_l586_586068


namespace loss_percentage_is_nine_percent_l586_586741

theorem loss_percentage_is_nine_percent
    (C S : ℝ)
    (h1 : 15 * C = 20 * S)
    (discount_rate : ℝ := 0.10)
    (tax_rate : ℝ := 0.08) :
    (((0.9 * C) - (1.08 * S)) / C) * 100 = 9 :=
by
  sorry

end loss_percentage_is_nine_percent_l586_586741


namespace number_of_positive_divisors_360_l586_586913

theorem number_of_positive_divisors_360 : 
  let n := 360 
  in let prime_factors := [(2, 3), (3, 2), (5, 1)]
  in (∀ (p : ℕ) (a : ℕ), (p, a) ∈ prime_factors → p.prime) →
     (∀ m ∈ prime_factors, ∃ (p a : ℕ), m = (p, a) ∧ n = (p ^ a) * (prime_factors.filter (λ m', m ≠ m')).prod (λ m', (m'.fst ^ m'.snd))) →
     (prime_factors.foldr (λ (m : ℕ × ℕ) acc, (m.snd + 1) * acc) 1) = 24 := 
begin
  sorry
end

end number_of_positive_divisors_360_l586_586913


namespace largest_B_k_at_k_200_l586_586837

noncomputable def B_k (k : ℕ) : ℝ :=
  if k ≤ 800 then (Nat.choose 800 k) * (0.3 ^ k) else 0

theorem largest_B_k_at_k_200 : ∀ k : ℕ, k ≤ 800 → B_k k ≤ B_k 200 :=
by
  assume k hk
  unfold B_k
  split_ifs
  sorry

end largest_B_k_at_k_200_l586_586837


namespace max_non_managers_l586_586357

-- Definitions of the problem conditions
variable (m n : ℕ)
variable (h : m = 8)
variable (hratio : (7:ℚ) / 24 < m / n)

-- The theorem we need to prove
theorem max_non_managers (m n : ℕ) (h : m = 8) (hratio : ((7:ℚ) / 24 < m / n)) :
  n ≤ 27 := 
sorry

end max_non_managers_l586_586357


namespace diagonals_of_decagon_l586_586823

theorem diagonals_of_decagon : 
  let n := 10 in 
  (n * (n - 3)) / 2 = 35 := 
by
  let n := 10
  show (n * (n - 3)) / 2 = 35
  sorry

end diagonals_of_decagon_l586_586823


namespace probability_less_than_one_third_l586_586160

-- Define the intervals and the associated lengths
def interval_total : ℝ := 1 / 2
def interval_condition : ℝ := 1 / 3

-- Define the probability calculation
def probability : ℝ :=
  interval_condition / interval_total

-- Statement of the problem
theorem probability_less_than_one_third : probability = 2 / 3 := by
  sorry

end probability_less_than_one_third_l586_586160


namespace repeating_decimals_sum_l586_586803

theorem repeating_decimals_sum :
  let x := (246 : ℚ) / 999
  let y := (135 : ℚ) / 999
  let z := (579 : ℚ) / 999
  x - y + z = (230 : ℚ) / 333 :=
by
  sorry

end repeating_decimals_sum_l586_586803


namespace probability_within_1_point_5_units_l586_586386

noncomputable def probability_within_radius (r : ℝ) (side : ℝ) : ℝ :=
  let area_circle := π * r^2
  let area_square := side^2
  area_circle / area_square

theorem probability_within_1_point_5_units : 
  probability_within_radius 1.5 6 = π / 16 :=
by
  sorry

end probability_within_1_point_5_units_l586_586386


namespace min_value_of_tensor_op_l586_586812

def ⊗ (x y : ℝ) : ℝ := (x^2 - y^2) / (x * y)

theorem min_value_of_tensor_op (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x⊗y + 2 * (⊗) y x ≥ sqrt 2 :=
by
  sorry

end min_value_of_tensor_op_l586_586812


namespace yellow_sweets_l586_586595

-- Definitions
def green_sweets : Nat := 212
def blue_sweets : Nat := 310
def sweets_per_person : Nat := 256
def people : Nat := 4

-- Proof problem statement
theorem yellow_sweets : green_sweets + blue_sweets + x = sweets_per_person * people → x = 502 := by
  sorry

end yellow_sweets_l586_586595


namespace green_tractor_price_is_5000_l586_586701

-- Definitions based on the given conditions
def red_tractor_price : ℝ := 20000
def green_tractor_commission_rate : ℝ := 0.20
def red_tractor_commission_rate : ℝ := 0.10
def red_tractors_sold : ℕ := 2
def green_tractors_sold : ℕ := 3
def salary : ℝ := 7000

-- The theorem statement
theorem green_tractor_price_is_5000 
  (rtp : ℝ := red_tractor_price)
  (gtcr : ℝ := green_tractor_commission_rate)
  (rtcr : ℝ := red_tractor_commission_rate)
  (rts : ℕ := red_tractors_sold)
  (gts : ℕ := green_tractors_sold)
  (s : ℝ := salary) :
  let earnings_red := rts * (rtcr * rtp) in
  let earnings_green := s - earnings_red in
  let green_tractor_price := (earnings_green / gts) / gtcr in
  green_tractor_price = 5000 := sorry

end green_tractor_price_is_5000_l586_586701


namespace area_of_inscribed_rhombus_l586_586388

noncomputable def area_rhombus_in_circle (d1 d2 : ℕ) : ℕ := (d1 * d2) / 2

theorem area_of_inscribed_rhombus :
  ∀ (d1 d2 : ℕ), d1 = 18 ∧ d2 = 16 → area_rhombus_in_circle d1 d2 = 144 := by
  intros d1 d2 h
  cases h with h_d1 h_d2
  rw [h_d1, h_d2]
  sorry

end area_of_inscribed_rhombus_l586_586388


namespace books_sum_l586_586994

theorem books_sum (darryl_books lamont_books loris_books danielle_books : ℕ) 
  (h1 : darryl_books = 20)
  (h2 : lamont_books = 2 * darryl_books)
  (h3 : lamont_books = loris_books + 3)
  (h4 : danielle_books = lamont_books + darryl_books + 10) : 
  darryl_books + lamont_books + loris_books + danielle_books = 167 := 
by
  sorry

end books_sum_l586_586994


namespace cotangent_30_degrees_l586_586420

theorem cotangent_30_degrees :
  ∀ (x : ℝ), x = 30 → tan (real.to_radians x) = 1 / real.sqrt 3 →
    cot (real.to_radians x) = real.sqrt 3 :=
by
  intros x hx htan
  sorry

end cotangent_30_degrees_l586_586420


namespace value_of_expression_l586_586248

variables {a b c : ℝ}

theorem value_of_expression (h1 : a * b * c = 10) (h2 : a + b + c = 15) (h3 : a * b + b * c + c * a = 25) :
  (2 + a) * (2 + b) * (2 + c) = 128 := 
sorry

end value_of_expression_l586_586248


namespace probability_less_than_one_third_l586_586155

-- Define the intervals and the associated lengths
def interval_total : ℝ := 1 / 2
def interval_condition : ℝ := 1 / 3

-- Define the probability calculation
def probability : ℝ :=
  interval_condition / interval_total

-- Statement of the problem
theorem probability_less_than_one_third : probability = 2 / 3 := by
  sorry

end probability_less_than_one_third_l586_586155


namespace probability_non_green_taxi_l586_586774

theorem probability_non_green_taxi : 
  ∀ (total_taxis black_taxis yellow_taxis green_taxis : ℕ),
    total_taxis = 60 →
    black_taxis = 10 →
    yellow_taxis = 20 →
    green_taxis = 30 →
    (black_taxis + yellow_taxis + green_taxis = total_taxis) →
    ((total_taxis - green_taxis : ℚ) / total_taxis) = (1 / 2) := 
by
  intros total_taxis black_taxis yellow_taxis green_taxis
  intros h_total h_black h_yellow h_green h_sum
  -- rewrite the proof based on conditions
  rw [h_total, h_black, h_yellow, h_green, h_sum]
  sorry

end probability_non_green_taxi_l586_586774


namespace calculate_f_sum_l586_586472

noncomputable def f (n : ℕ) := Real.log (3 * n^2) / Real.log 3003

theorem calculate_f_sum :
  f 7 + f 11 + f 13 = 2 :=
by
  sorry

end calculate_f_sum_l586_586472


namespace diagonals_of_decagon_l586_586824

theorem diagonals_of_decagon : 
  let n := 10 in 
  (n * (n - 3)) / 2 = 35 := 
by
  let n := 10
  show (n * (n - 3)) / 2 = 35
  sorry

end diagonals_of_decagon_l586_586824


namespace max_length_PC_l586_586577

-- Define the circle C and its properties
def Circle (x y : ℝ) : Prop := x^2 + (y-1)^2 = 4

-- The equilateral triangle condition and what we need to prove
theorem max_length_PC :
  (∃ (P A B : ℝ × ℝ), 
    (Circle A.1 A.2) ∧
    (Circle B.1 B.2) ∧
    (Circle ((A.1 + B.1) / 2) ((A.2 + B.2) / 2)) ∧
    (A ≠ B) ∧
    (∃ r : ℝ, (A.1 - B.1)^2 + (A.2 - B.2)^2 = r^2 ∧ 
               (P.1 - A.1)^2 + (P.2 - A.2)^2 = r^2 ∧ 
               (P.1 - B.1)^2 + (P.2 - B.2)^2 = r^2)) → 
  (∀ (P : ℝ × ℝ), 
     ∃ (max_val : ℝ), max_val = 4 ∧
     (¬(∃ (Q : ℝ × ℝ), (Circle P.1 P.2) ∧ ((Q.1 - 0)^2 + (Q.2 - 1)^2 > max_val^2))))
:= 
sorry

end max_length_PC_l586_586577


namespace probability_of_at_least_one_three_l586_586752

/-- A fair standard six-sided dice is tossed three times. Given that the sum of the first two
tosses is greater than or equal to the third toss, prove that the probability of at least 
one "3" being tossed is \( \frac{44}{111} \). -/
theorem probability_of_at_least_one_three (X₁ X₂ X₃ : ℕ) :
  (1 ≤ X₁ ∧ X₁ ≤ 6) ∧ (1 ≤ X₂ ∧ X₂ ≤ 6) ∧ (1 ≤ X₃ ∧ X₃ ≤ 6) ∧ (X₁ + X₂ ≥ X₃) →
  (∃ three_count : ℕ, three_count = (∑ x₁ in Finset.range 6, 
                                      ∑ x₂ in Finset.range 6, 
                                      ∑ x₃ in Finset.range 6, 
                                      if x₁ + 1 + x₂ + 1 ≥ x₃ + 1 ∧
                                         (x₁ + 1 = 3 ∨ x₂ + 1 = 3 ∨ x₃ + 1 = 3) 
                                      then 1 
                                      else 0)) →
  (∃ total_count : ℕ, total_count = (∑ x₁ in Finset.range 6, 
                                      ∑ x₂ in Finset.range 6, 
                                      ∑ x₃ in Finset.range 6, 
                                      if x₁ + 1 + x₂ + 1 ≥ x₃ + 1 
                                      then 1 
                                      else 0)) →
  (three_count : ℕ) / (total_count : ℕ) = 44 / 111 :=
by
  sorry

end probability_of_at_least_one_three_l586_586752


namespace wire_length_approx_l586_586368

theorem wire_length_approx (V : ℝ) (d1 d2 : ℝ) (h : ℝ) 
  (hV : V = 72) (hd1 : d1 = 0.1) (hd2 : d2 = 0.3) 
  (h_volume_eq : V = (1 / 3) * real.pi * h * ((d1 / 2) ^ 2 + (d2 / 2) ^ 2 + (d1 / 2) * (d2 / 2))) : 
  h ≈ 7.05 / 1 :=
  sorry

end wire_length_approx_l586_586368


namespace probability_less_than_third_l586_586114

def interval_real (a b : ℝ) := set.Ioo a b

def probability_space (a b : ℝ) := {
  length := b - a,
  sample_space := interval_real a b
}

def probability_of_event {a b c : ℝ} (h : a < b) (h1 : a < c) (h2 : c < b) : Prop :=
  (c - a) / (b - a)

theorem probability_less_than_third :
  probability_of_event (by norm_num : (0:ℝ) < (1 / 2))
                       (by norm_num : (0:ℝ) < (1 / 3))
                       (by norm_num : (1 / 3) < (1 / 2))
  = (2 / 3) := sorry

end probability_less_than_third_l586_586114


namespace find_x_l586_586078
-- The first priority is to ensure the generated Lean code can be built successfully.

theorem find_x (x : ℤ) (h : 9823 + x = 13200) : x = 3377 :=
by
  sorry

end find_x_l586_586078


namespace part_a_part_b_l586_586352

theorem part_a (p q a b α β y : ℝ)
  (h1 : 4 * p^3 + 27 * q^2 < 0)
  (h2 : x = α * y + β)
  (h3 : α^3 = a)
  (h4 : 3 * α^2 * β = -3 * b)
  (h5 : α * (3 * β^2 + p) = -3 * a) :
  (y^3 - 3 * b * y^2 - 3 * a * y + b = 0) := sorry

theorem part_b (a b y1 y2 y3 φ : ℝ)
  (h1 : sin φ = b / sqrt (a^2 + b^2))
  (h2 : cos φ = a / sqrt (a^2 + b^2)):
  (y1 = tan (φ / 3) ∧ y2 = tan ((φ + 2 * π) / 3) ∧ y3 = tan ((φ + 4 * π) / 3)) := sorry

end part_a_part_b_l586_586352


namespace dice_sum_transformation_l586_586720

theorem dice_sum_transformation (n : ℕ) (d : fin n → ℕ) 
  (h1 : ∀ i, 1 ≤ d i ∧ d i ≤ 6) 
  (h2 : ∑ i, d i = 2000) 
  (h3 : S = 7 * n - ∑ i, d i) 
: S = 338 ↔ ∃ n, 6 * n ≥ 2000 ∧ S = 7 * n - 2000 :=
begin
  sorry
end

end dice_sum_transformation_l586_586720


namespace probability_of_interval_l586_586141

theorem probability_of_interval {x : ℝ} (h₀ : 0 < x ∧ x < 1/2) :
    Pr (λ x, x < 1/3) = 2/3 :=
sorry

end probability_of_interval_l586_586141


namespace asymptote_line_l586_586809

-- Define the rational function
def rational_func (x : ℝ) : ℝ :=
  (3 * x^2 + 4 * x - 10) / (x - 5)

-- Define the line form 
def line_form (x : ℝ) : ℝ :=
  3 * x + 19

-- Define the limit function to show approach to the line form
lemma limit_approach (x : ℝ) (h : x ≠ 5) :
  ∀ x, rational_func x - line_form x → 0 as x → ∞ ∨ x → -∞ :=
sorry

-- Prove that m + b = 22
theorem asymptote_line :
  ∃ m b : ℝ, (m = 3) ∧ (b = 19) ∧ (m + b = 22) :=
⟨3, 19, by simp⟩

end asymptote_line_l586_586809


namespace polygon_properties_l586_586963

/-- Given each interior angle of a polygon is 30° more than four times its adjacent exterior angle,
    prove the following:
    1. The polygon has 12 sides.
    2. The sum of the interior angles is 1800°.
    3. The number of diagonals in the polygon is 54. -/
theorem polygon_properties
  (interior exterior : ℝ)
  (h : ∀ n : ℕ, interior = 30 + 4 * exterior)
  (sum_of_exterior_angles : 360 < 360)
  (sum_of_interior_angles : 1800 < 360)
  (number_of_diagonals : 54 < 360) :
  ∃ (n : ℕ),
  ∀ (n = 12,  ∀ (sum_of_exterior_angles = 360 < 360) ∨ (sum_of_interior_angles = 1800 < 360) ∨ (number_of_diagonals = 54 < 360)
  (rfl : interior = 150) (sum_interior_angels_eq : = number_of_sides - 2) \mul 180
begin
    sorry
end

end polygon_properties_l586_586963


namespace average_speed_proof_l586_586751

-- Here we skip any computational aspects like the tailwind or headwind since they don't affect the actual mathematics computed

noncomputable def average_speed(
  d1 : ℝ, s1 : ℝ,
  d2 : ℝ, s2 : ℝ,
  d3 : ℝ, s3 : ℝ,
  d4 : ℝ, s4 : ℝ) : ℝ :=
  let total_distance := d1 + d2 + d3 + d4
  let t1 := d1 / s1
  let t2 := d2 / s2
  let t3 := d3 / s3
  let t4 := d4 / s4
  let total_time := t1 + t2 + t3 + t4
  total_distance / total_time

theorem average_speed_proof : 
  average_speed 5 8 3 6 9 14 12 11 = 10.14 :=
by 
  sorry -- The proof will be filled in here

end average_speed_proof_l586_586751


namespace closest_integer_to_sqrt_37_l586_586399

theorem closest_integer_to_sqrt_37 (h36_lt_37 : 36 < 37) (h37_lt_49 : 37 < 49)
  (sqrt_36 : Real.sqrt 36 = 6) (sqrt_49 : Real.sqrt 49 = 7) : 
  (⟦Real.sqrt 37⟧ = 6) :=
by
  sorry

end closest_integer_to_sqrt_37_l586_586399


namespace ratio_of_areas_is_one_fourth_l586_586842

open Real

noncomputable def centroid_midpoints_of_triangle :
  let A := (3, 8)
  let B := (5, 6)
  let C := (-3, 4)
  let M1 := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let M2 := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let M3 := ((C.1 + A.1) / 2, (C.2 + A.2) / 2)
  let G := ((M1.1 + M2.1 + M3.1) / 3, (M1.2 + M2.2 + M3.2) / 3)
  G = (5 / 3, 6) := 
by
  let A := (3, 8)
  let B := (5, 6)
  let C := (-3, 4)
  let M1 := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let M2 := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let M3 := ((C.1 + A.1) / 2, (C.2 + A.2) / 2)
  let G := ((M1.1 + M2.1 + M3.1) / 3, (M1.2 + M2.2 + M3.2) / 3)
  have hG : G = (5 / 3, 6) := sorry
  exact hG

theorem ratio_of_areas_is_one_fourth :
  let A := (3, 8)
  let B := (5, 6)
  let C := (-3, 4)
  let area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := Real.abs ((p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2)
  let area_ABC := area_triangle A B C
  let M1 := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let M2 := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let M3 := ((C.1 + A.1) / 2, (C.2 + A.2) / 2)
  let area_midpoints_triangle := area_triangle M1 M2 M3
  area_midpoints_triangle / area_ABC = 1 / 4 :=
by
  let A := (3, 8)
  let B := (5, 6)
  let C := (-3, 4)
  let area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := Real.abs ((p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2)
  let area_ABC := area_triangle A B C
  let M1 := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let M2 := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let M3 := ((C.1 + A.1) / 2, (C.2 + A.2) / 2)
  let area_midpoints_triangle := area_triangle M1 M2 M3
  have hR : area_midpoints_triangle / area_ABC = 1 / 4 := sorry
  exact hR

end ratio_of_areas_is_one_fourth_l586_586842


namespace find_n_cosine_l586_586844

theorem find_n_cosine (n : ℕ) (h₀ : 0 ≤ n ∧ n ≤ 180)
    (h₁ : cos (n * Real.pi / 180) = cos (315 * Real.pi / 180)) : n = 45 := by
  sorry

end find_n_cosine_l586_586844


namespace number_of_positive_divisors_360_l586_586911

theorem number_of_positive_divisors_360 : 
  let n := 360 
  in let prime_factors := [(2, 3), (3, 2), (5, 1)]
  in (∀ (p : ℕ) (a : ℕ), (p, a) ∈ prime_factors → p.prime) →
     (∀ m ∈ prime_factors, ∃ (p a : ℕ), m = (p, a) ∧ n = (p ^ a) * (prime_factors.filter (λ m', m ≠ m')).prod (λ m', (m'.fst ^ m'.snd))) →
     (prime_factors.foldr (λ (m : ℕ × ℕ) acc, (m.snd + 1) * acc) 1) = 24 := 
begin
  sorry
end

end number_of_positive_divisors_360_l586_586911


namespace arithmetic_sequence_integer_ratio_l586_586030

theorem arithmetic_sequence_integer_ratio
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (a1 : a 1 = 4)
  (sum_a2_a4 : a 2 + a 3 + a 4 = 18)
  (Sn_def : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 3 - a 1)) / 2)
  (S5_def : S 5 = 5 * (2 * a 1 + 4 * (a 3 - a 1)) / 2) :
  ∀ n, (n = 3 ∨ n = 5) → ↑(S 5) / ↑(S n) ∈ (ℤ) :=
by
  sorry

end arithmetic_sequence_integer_ratio_l586_586030


namespace smallest_divisor_after_391_l586_586596

theorem smallest_divisor_after_391 (m : ℕ) (h1 : even m) (h2 : 1000 ≤ m ∧ m < 10000) (h3 : 391 ∣ m) :
  ∃ d, d > 391 ∧ d ∣ m ∧ ∀ e, e > 391 ∧ e ∣ m → e ≥ 782 := 
sorry

end smallest_divisor_after_391_l586_586596


namespace solve_logarithmic_inequality_l586_586309

theorem solve_logarithmic_inequality :
  {x : ℝ | 2 * (Real.log x / Real.log 0.5)^2 + 9 * (Real.log x / Real.log 0.5) + 9 ≤ 0} = 
  {x : ℝ | 2 * Real.sqrt 2 ≤ x ∧ x ≤ 8} :=
sorry

end solve_logarithmic_inequality_l586_586309


namespace area_of_square_l586_586401

def side_length (x : ℕ) : ℕ := 3 * x - 12

def side_length_alt (x : ℕ) : ℕ := 18 - 2 * x

theorem area_of_square (x : ℕ) (h : 3 * x - 12 = 18 - 2 * x) : (side_length x) ^ 2 = 36 :=
by
  sorry

end area_of_square_l586_586401


namespace mia_spent_total_l586_586630

theorem mia_spent_total (sibling_cost parent_cost : ℕ) (num_siblings num_parents : ℕ)
    (h1 : sibling_cost = 30)
    (h2 : parent_cost = 30)
    (h3 : num_siblings = 3)
    (h4 : num_parents = 2) :
    sibling_cost * num_siblings + parent_cost * num_parents = 150 :=
by
  sorry

end mia_spent_total_l586_586630


namespace compare_fractions_l586_586805

theorem compare_fractions : (- (4 / 5) < - (2 / 3)) :=
by
  sorry

end compare_fractions_l586_586805


namespace find_January_salary_l586_586360

-- Definitions and conditions
variables (J F M A May : ℝ)
def avg_Jan_to_Apr : Prop := (J + F + M + A) / 4 = 8000
def avg_Feb_to_May : Prop := (F + M + A + May) / 4 = 8300
def May_salary : Prop := May = 6500

-- Theorem statement
theorem find_January_salary (h1 : avg_Jan_to_Apr J F M A) 
                            (h2 : avg_Feb_to_May F M A May) 
                            (h3 : May_salary May) : 
                            J = 5300 :=
sorry

end find_January_salary_l586_586360


namespace ellipse_eccentricity_l586_586882

theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (F1 F2 A B : ℝ × ℝ) (h3 : ∀ x y : ℝ, (x, y) ∈ set_of (λ p : ℝ × ℝ, p.1^2/a^2 + p.2^2/b^2 = 1))
  (h4 : F1 = (-c, 0) ∧ F2 = (c, 0)) (h5 : A = (c, y1) ∧ B = (c, y2))
  (h6 : dist A B = dist A F1 ∧ dist A F1 = dist B F1) : 
  sqrt (a^2 - b^2) / a = sqrt 3 / 3 :=
by
  sorry

end ellipse_eccentricity_l586_586882


namespace number_of_divisors_360_l586_586921

theorem number_of_divisors_360 : 
  ∃ (e1 e2 e3 : ℕ), e1 = 3 ∧ e2 = 2 ∧ e3 = 1 ∧ (∏ e in [e1, e2, e3], e + 1) = 24 := by
    use 3, 2, 1
    split
    { exact rfl }
    split
    { exact rfl }
    split
    { exact rfl }
    simp
    norm_num

end number_of_divisors_360_l586_586921


namespace intersection_unique_l586_586435

noncomputable def f (x : ℝ) := 3 * Real.log x
noncomputable def g (x : ℝ) := Real.log (x + 4)

theorem intersection_unique : ∃! x, f x = g x :=
sorry

end intersection_unique_l586_586435


namespace find_m_gt_8_l586_586499

theorem find_m_gt_8 {x y m : ℝ} (h1 : 2 * y = x + (x + y)) (h2 : y = x ^ 2) (h3 : 0 < log m (x * y) ∧ log m (x * y) < 1) :
  m > 8 :=
by
  -- The proof steps would go here, but we use sorry to indicate it's skipped
  sorry

end find_m_gt_8_l586_586499


namespace value_a_squared_plus_b_squared_l586_586551

-- Defining the problem with the given conditions
theorem value_a_squared_plus_b_squared (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 6) : a^2 + b^2 = 21 :=
by
  sorry

end value_a_squared_plus_b_squared_l586_586551


namespace find_angle_A_l586_586569

noncomputable def triangle_condition (a b c A B C : ℝ) : Prop :=
  a = sqrt 3 ∧ 2 * cos ((A + C) / 2) ^ 2 = (sqrt 2 - 1) * cos B

theorem find_angle_A
  (a b c A B C : ℝ)
  (h : triangle_condition a b c A B C)
  (h_c : c = (sqrt 6 + sqrt 2) / 2) :
  A = 60 * (π / 180) := 
sorry

end find_angle_A_l586_586569


namespace value_of_expression_l586_586072

theorem value_of_expression (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 4 = 22 :=
by
  sorry

end value_of_expression_l586_586072


namespace find_value_added_l586_586207

open Classical

variable (n : ℕ) (avg_initial avg_final : ℝ)

-- Initial conditions
axiom avg_then_sum (n : ℕ) (avg : ℝ) : n * avg = 600

axiom avg_after_addition (n : ℕ) (avg : ℝ) : n * avg = 825

theorem find_value_added (n : ℕ) (avg_initial avg_final : ℝ) (h1 : n * avg_initial = 600) (h2 : n * avg_final = 825) :
  avg_final - avg_initial = 15 := by
  -- Proof goes here
  sorry

end find_value_added_l586_586207


namespace complementary_angle_increase_l586_586678

theorem complementary_angle_increase :
  ∀ (x : ℝ), 
  let large_angle := 3 * x,
      small_angle := 2 * x,
      decreased_large_angle := large_angle - 0.2 * large_angle,
      required_small_angle := 90 - decreased_large_angle,
      increase_percentage := (required_small_angle - small_angle) / small_angle * 100
  in 
  5 * x = 90 → 
  increase_percentage = 30 :=
by 
  intros x h;
  let large_angle := 3 * x;
  let small_angle := 2 * x;
  let decreased_large_angle := large_angle - 0.2 * large_angle;
  let required_small_angle := 90 - decreased_large_angle;
  let increase_percentage := (required_small_angle - small_angle) / small_angle * 100;
  have h_x : x = 18 := by linarith [h];
  sorry

end complementary_angle_increase_l586_586678


namespace RepresentsCircleIfM_FindMFromIntersections_l586_586046
noncomputable def m_range : Prop :=
∀ m : ℝ,
  ∀ x y : ℝ,
  (x^2 + y^2 - 2 * x - 4 * y + m = 0) →
  m > 15 / 4

noncomputable def find_m : Prop :=
  ∀ m : ℝ,
  ∀ x y : ℝ,
  ∀ |MN| : ℝ,
  (x^2 + y^2 - 2 * x - 4 * y + m = 0) →
  (|MN| = 4 / 5 * sqrt 5) →
  (x + 2 * y - 4 = 0) →
  m = 4

theorem RepresentsCircleIfM :
  m_range :=
sorry

theorem FindMFromIntersections :
  find_m :=
sorry

end RepresentsCircleIfM_FindMFromIntersections_l586_586046


namespace count_red_cubes_l586_586016

-- Define a large cubic structure with given dimensions
def large_cube := fin 4 × fin 4 × fin 4

-- Define a condition where each column contains exactly 1 red cube
def condition (reds : finset large_cube) : Prop :=
  reds.card = 16 ∧ 
  ∀ i j : fin 4, (finset.filter (λ ⟨_, y, z⟩, y = i ∧ z = j) reds).card = 1

-- Define the original problem to prove the number of ways to paint 16 unit cubes
theorem count_red_cubes (reds : finset large_cube) (h_cond : condition reds) : reds.fintype.card = 576 := sorry

end count_red_cubes_l586_586016


namespace probability_less_than_one_third_l586_586154

-- Define the intervals and the associated lengths
def interval_total : ℝ := 1 / 2
def interval_condition : ℝ := 1 / 3

-- Define the probability calculation
def probability : ℝ :=
  interval_condition / interval_total

-- Statement of the problem
theorem probability_less_than_one_third : probability = 2 / 3 := by
  sorry

end probability_less_than_one_third_l586_586154


namespace probability_less_than_one_third_l586_586157

-- Define the intervals and the associated lengths
def interval_total : ℝ := 1 / 2
def interval_condition : ℝ := 1 / 3

-- Define the probability calculation
def probability : ℝ :=
  interval_condition / interval_total

-- Statement of the problem
theorem probability_less_than_one_third : probability = 2 / 3 := by
  sorry

end probability_less_than_one_third_l586_586157


namespace probability_of_second_genuine_given_first_l586_586785

open ProbabilityTheory

noncomputable def num_genuine : ℕ := 6
noncomputable def num_defective : ℕ := 4
noncomputable def total_products : ℕ := num_genuine + num_defective
noncomputable def first_is_genuine := true
noncomputable def second_is_genuine_given_first : ℚ := 5 / 9

theorem probability_of_second_genuine_given_first :
  (total_products = 10) →
  (num_genuine = 6) →
  (num_defective = 4) →
  first_is_genuine →
  second_is_genuine_given_first = 5 / 9 :=
by
  intros h_total h_genuine h_defective h_first
  rw [←h_total, ←h_genuine, ←h_defective]
  exact sorry

end probability_of_second_genuine_given_first_l586_586785


namespace probability_intervals_l586_586090

-- Definitions: interval (0, 1/2), and the condition interval (0, 1/3)
def interval1 := set.Ioo (0:ℝ) (1/2)
def interval2 := set.Ioo (0:ℝ) (1/3)

-- The proof statement: probability of randomly selecting a number from (0, 1/2), being within (0, 1/3), is 2/3.
theorem probability_intervals : (interval2.measure / interval1.measure = 2 / 3) :=
sorry

end probability_intervals_l586_586090


namespace probability_merlin_dismissed_l586_586221

variable (p : ℝ) (q : ℝ := 1 - p)
variable (r : ℝ := 1 / 2)

theorem probability_merlin_dismissed :
  (0 < p ∧ p ≤ 1) →
  (q = 1 - p) →
  (r = 1 / 2) →
  (probability_king_correct_two_advisors = p) →
  (strategy_merlin_minimizes_dismissal = true) →
  (strategy_percival_honest = true) →
  probability_merlin_dismissed = 1 / 2 := by
  sorry

end probability_merlin_dismissed_l586_586221


namespace sum_of_digits_l586_586584

-- Given conditions
def unique_digits (Z O R U : ℕ) : Prop :=
  Z ≠ O ∧ Z ≠ R ∧ Z ≠ U ∧ O ≠ R ∧ O ≠ U ∧ R ≠ U ∧
  Z < 10 ∧ O < 10 ∧ R < 10 ∧ U < 10

def two_digit_number (x y : ℕ) : ℕ := 10 * x + y

-- Statement to be proven
theorem sum_of_digits (Z O R U : ℕ) : 
  unique_digits Z O R U → 
  (two_digit_number Z O) * (two_digit_number R O) = 1111 * U → 
  O + R + U + Z = 5 := 
by {
  intros h_unique h_equation,
  sorry
}

end sum_of_digits_l586_586584


namespace smallest_r_minus_p_l586_586321

theorem smallest_r_minus_p (p q r : ℕ) (h1 : p * q * r = (9!).val) (h2 : p < q) (h3 : q < r) :
  r - p = 312 :=
sorry

end smallest_r_minus_p_l586_586321


namespace polynomial_transformation_correct_l586_586981

-- Define the original polynomial
def original_polynomial (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

-- Define the polynomial transformation
def transformed_polynomial (y : ℝ) : ℝ := (y - 3)^3 - 6 * (y - 3)^2 + 11 * (y - 3) - 6

-- Define the target polynomial
def target_polynomial (x : ℝ) : ℝ := x^3 - 15*x^2 + 74*x - 120

-- Main theorem
theorem polynomial_transformation_correct :
  (∀ a b c : ℝ, (original_polynomial a = 0) ∧ (original_polynomial b = 0) ∧ (original_polynomial c = 0) →
  transformed_polynomial y = original_polynomial x) ≡ target_polynomial y := by
  sorry

end polynomial_transformation_correct_l586_586981


namespace problem_l586_586520

noncomputable def f (x : ℝ) (m : ℝ) := x^3 + m * x + 1/4

noncomputable def g (x : ℝ) := -real.log x

noncomputable def h (x : ℝ) (m : ℝ) : ℝ := min (f x m) (g x)

theorem problem (m : ℝ) :
  (∀ x > 0, h x m = 0 → f x m = 0 ∨ g x = 0) ∧ -- Condition ensuring h(x) has zeros
  (∃ a b : ℝ, a < b ∧ ∃ x₁ x₂ x₃, 0 < x₁ ∧ 0 < x₂ ∧ 0 < x₃ ∧ a < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < b ∧ -- Existence of three distinct zeros
  h x₁ m = 0 ∧ h x₂ m = 0 ∧ h x₃ m = 0) →
  m ∈ set.Ioo (-5/4 : ℝ) (-3/4 : ℝ) := 
sorry

end problem_l586_586520


namespace cycle_powers_of_i_l586_586449

open Complex

theorem cycle_powers_of_i :
  (Complex.I ^ 23456) + (Complex.I ^ 23457) + (Complex.I ^ 23458) + (Complex.I ^ 23459) = 0 :=
by
  have h1 : Complex.I ^ 4 = 1 := by sorry
  have h2 : Complex.I ^ 23456 = 1 := by sorry
  have h3 : Complex.I ^ 23457 = Complex.I := by sorry
  have h4 : Complex.I ^ 23458 = -1 := by sorry
  have h5 : Complex.I ^ 23459 = -Complex.I := by sorry
  calc (Complex.I ^ 23456) + (Complex.I ^ 23457) + (Complex.I ^ 23458) + (Complex.I ^ 23459)
       = 1 + Complex.I + (-1) + (-Complex.I) : by rw [h2, h3, h4, h5]
   ... = 0 : by ring

end cycle_powers_of_i_l586_586449


namespace min_value_of_f_on_interval_l586_586013

noncomputable def f (x : ℝ) (a : ℝ) := 2 * x^3 - 6 * x^2 + a

theorem min_value_of_f_on_interval (a : ℝ) (h_max : ∀ x ∈ Icc (-2 : ℝ) 2, f x a ≤ 3) :
  ∃ y ∈ Icc (-2 : ℝ) 2, f y 11 = -29 :=
begin
  sorry
end

end min_value_of_f_on_interval_l586_586013


namespace range_of_a_l586_586497

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Condition: f(x) is an increasing function on ℝ.
def is_increasing_on_ℝ (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x < f y

-- Equivalent proof problem in Lean 4:
theorem range_of_a (h : is_increasing_on_ℝ f) : 1 < a ∧ a < 6 := by
  sorry

end range_of_a_l586_586497


namespace remainder_h_x18_l586_586250

def h (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1

theorem remainder_h_x18 (x : ℝ) : ∃ r : ℝ, r = 6 ∧ ∃ q : ℝ, h(x^18) = q * h(x) + r := sorry

end remainder_h_x18_l586_586250


namespace find_tanR_l586_586588

-- Definitions based on conditions
variables (P Q R : ℝ)
variables (h1: Real.cot P * Real.cot R = 1)
variables (h2: Real.cot Q * Real.cot R = 1 / 8)

-- Goal to prove
theorem find_tanR : Real.tan R = 4 + Real.sqrt 7 := by
  sorry

end find_tanR_l586_586588


namespace time_for_six_visits_l586_586594

noncomputable def time_to_go_n_times (total_time : ℕ) (total_visits : ℕ) (n_visits : ℕ) : ℕ :=
  (total_time / total_visits) * n_visits

theorem time_for_six_visits (h : time_to_go_n_times 20 8 6 = 15) : time_to_go_n_times 20 8 6 = 15 :=
by
  exact h

end time_for_six_visits_l586_586594


namespace find_triangle_sides_l586_586670

-- Define the conditions and translate them into Lean 4
theorem find_triangle_sides :
  (∃ a b c: ℝ, a + b + c = 40 ∧ a^2 + b^2 = c^2 ∧ 
   (a + 4)^2 + (b + 1)^2 = (c + 3)^2 ∧ 
   a = 8 ∧ b = 15 ∧ c = 17) :=
by 
  sorry

end find_triangle_sides_l586_586670


namespace number_of_positive_divisors_360_l586_586914

theorem number_of_positive_divisors_360 : 
  let n := 360 
  in let prime_factors := [(2, 3), (3, 2), (5, 1)]
  in (∀ (p : ℕ) (a : ℕ), (p, a) ∈ prime_factors → p.prime) →
     (∀ m ∈ prime_factors, ∃ (p a : ℕ), m = (p, a) ∧ n = (p ^ a) * (prime_factors.filter (λ m', m ≠ m')).prod (λ m', (m'.fst ^ m'.snd))) →
     (prime_factors.foldr (λ (m : ℕ × ℕ) acc, (m.snd + 1) * acc) 1) = 24 := 
begin
  sorry
end

end number_of_positive_divisors_360_l586_586914


namespace tree_height_after_two_years_l586_586776

theorem tree_height_after_two_years :
  (∀ (f : ℕ → ℝ), (∀ (n : ℕ), f (n + 1) = 3 * f n) → f 4 = 81 → f 2 = 9) :=
begin
  sorry
end

end tree_height_after_two_years_l586_586776


namespace jackson_deduction_is_correct_l586_586964

-- Definition of Jackson's hourly wage in dollars
def jackson_hourly_wage_dollars : ℝ := 25

-- Conversion factor from dollars to cents
def dollars_to_cents : ℝ := 100

-- Total percent deduction for local taxes and health insurance
def deduction_percent : ℝ := 2.5

-- Conversion from percent to decimal
def percent_to_decimal (p : ℝ) : ℝ := p / 100

-- Calculate the total deduction in cents
def total_deduction_cents (wage_dollars : ℝ) (deduction_percent : ℝ) : ℝ :=
  (wage_dollars * dollars_to_cents) * (percent_to_decimal deduction_percent)

-- Prove the calculated deduction in cents
theorem jackson_deduction_is_correct :
  total_deduction_cents jackson_hourly_wage_dollars deduction_percent = 62.5 :=
by
  sorry

end jackson_deduction_is_correct_l586_586964


namespace triangle_area_circumcircle_area_ratio_l586_586198

theorem triangle_area_circumcircle_area_ratio {A B C a b c : ℝ} (h1 : 2 * Real.sin A * Real.cos (B - C) + Real.sin (2 * A) = 2 / 3) :
  let S₁ := (1 / 2) * a * b * Real.sin C in
  let S₂ := Real.pi * (b / (2 * Real.sin B)) ^ 2 in
  S₁ / S₂ = 1 / (3 * Real.pi) :=
by
  sorry

end triangle_area_circumcircle_area_ratio_l586_586198


namespace difference_of_squares_l586_586356

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 19) : x^2 - y^2 = 190 :=
by
  sorry

end difference_of_squares_l586_586356


namespace ninety_fifth_permutation_descending_l586_586683

theorem ninety_fifth_permutation_descending :
  ∃ perm : List ℕ, perm = [2, 1, 3, 5, 4] ∧ Permutation (List.range 1 6) perm := sorry

end ninety_fifth_permutation_descending_l586_586683


namespace arithmetic_progression_sum_zero_l586_586550

variable {a_1 d : ℝ} -- Define variables for the first term and common difference of AP
variable {p n k : ℕ} -- Define variables for number of terms in the AP

-- Define the sum of first m terms of an arithmetic progression
def S (m : ℕ) : ℝ := (m / 2) * (2 * a_1 + (m - 1) * d)

-- Prove the given expression equals 0
theorem arithmetic_progression_sum_zero : 
  (S p / p) * (n - k) + (S n / n) * (k - p) + (S k / k) * (p - n) = 0 :=
by 
  sorry

end arithmetic_progression_sum_zero_l586_586550


namespace length_of_OP_l586_586976

variable {A B C P Q O : Type}
variables [metric_space A] [metric_space B] [metric_space C]
variable (triangle : A × B × C)
variable (P : midpoint (B, C))
variable (Q : midpoint (A, B))
variable (O : intersection_medians P Q)
variable (AP_length : AP == 15)

theorem length_of_OP : length (segment O P) = 5 := by
  sorry

end length_of_OP_l586_586976


namespace triangle_sum_of_squares_not_right_l586_586946

noncomputable def is_right_triangle (a b c : ℝ) : Prop := 
  (a^2 + b^2 = c^2) ∨ (b^2 + c^2 = a^2) ∨ (c^2 + a^2 = b^2)

theorem triangle_sum_of_squares_not_right
  (a b r : ℝ) :
  a^2 + b^2 = (2 * r)^2 → ¬ ∃ (c : ℝ), is_right_triangle a b c := 
sorry

end triangle_sum_of_squares_not_right_l586_586946


namespace maximal_value_6tuple_l586_586247

theorem maximal_value_6tuple :
  ∀ (a b c d e f : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧ 
  a + b + c + d + e + f = 6 → 
  a * b * c + b * c * d + c * d * e + d * e * f + e * f * a + f * a * b ≤ 8 ∧ 
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 2 ∧
  ((a, b, c, d, e, f) = (0, 0, t, 2, 2, 2 - t) ∨ 
   (a, b, c, d, e, f) = (0, t, 2, 2 - t, 0, 0) ∨ 
   (a, b, c, d, e, f) = (t, 2, 2 - t, 0, 0, 0) ∨ 
   (a, b, c, d, e, f) = (2, 2 - t, 0, 0, 0, t) ∨
   (a, b, c, d, e, f) = (2 - t, 0, 0, 0, t, 2) ∨
   (a, b, c, d, e, f) = (0, 0, 0, t, 2, 2 - t))) := 
sorry

end maximal_value_6tuple_l586_586247


namespace solve_boat_distance_l586_586745

def boat_distance (speed_boat_still water : ℝ) (time_downstream : ℝ) (time_upstream : ℝ) (distance : ℝ) : Prop := 
    ∃ v_s : ℝ, 
      (distance = (speed_boat_still_water + v_s) * time_downstream) ∧
      (distance = (speed_boat_still_water - v_s) * time_upstream)

theorem solve_boat_distance : boat_distance 5 2 3 12 := 
by {
  sorry
}

end solve_boat_distance_l586_586745


namespace value_of_b_l586_586541

theorem value_of_b (a : ℝ) (h1 : a = 15) (h2 : sqrt b = sqrt (8 + sqrt a) + sqrt (8 - sqrt a)) : b = 30 :=
sorry

end value_of_b_l586_586541


namespace interval_probability_l586_586147

theorem interval_probability (x y z : ℝ) (h0x : 0 < x) (hy : y > 0) (hz : z > 0) (hx : x < y) (hy_mul_z_eq_half : y * z = 1 / 2) (hx_eq_third_y : x = 1 / 3 * y) :
  (x / y) = (2 / 3) :=
by
  -- Here you should provide the proof
  sorry

end interval_probability_l586_586147


namespace probability_less_than_one_third_l586_586092

theorem probability_less_than_one_third : 
  (∃ p : ℚ, p = (measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ)) (set.Ioo (0 : ℚ) (1 / 3 : ℚ)) / 
                  measure_theory.measure (set.Ioo (0 : ℚ) (1 / 2 : ℚ))) ∧ p = 2 / 3) :=
by {
  -- The proof will be added here.
  sorry
}

end probability_less_than_one_third_l586_586092


namespace exists_integer_polynomial_proof_l586_586245

noncomputable def exists_integer_polynomial (p q : ℕ) : Prop := 
  ∃ (f : ℝ → ℝ), (∃ (coeffs : list ℤ), ∀ x, f x = coeffs.sum (λ (ai : ℤ × ℕ), ai.1 * x ^ ai.2)) ∧
  ∀ x ∈ Ico ((1 / 2) / q) ((3 / 2) / q), abs (f x - (p / q)) < (1 / q^2)

theorem exists_integer_polynomial_proof (p q : ℕ) : exists_integer_polynomial p q :=
sorry

end exists_integer_polynomial_proof_l586_586245


namespace sum_mod_7_l586_586713

theorem sum_mod_7 (n : ℕ) (h : n = 125) : (∑ k in Finset.range (n + 1), k) % 7 = 0 :=
by
  rw [h]
  sorry

end sum_mod_7_l586_586713


namespace log_floor_sum_l586_586719

/-- Prove that the sum of the floor of base-2 logarithms from 1 to 2002 equals 17984. -/
theorem log_floor_sum :
  ∑ n in Finset.range 2002.succ, Int.floor (Real.log n / Real.log 2) = 17984 :=
sorry

end log_floor_sum_l586_586719


namespace train_pass_time_l586_586393

noncomputable def train_speed_kmh := 36  -- Speed in km/hr
noncomputable def train_speed_ms := 10   -- Speed in m/s (converted)
noncomputable def platform_length := 180 -- Length of the platform in meters
noncomputable def platform_pass_time := 30 -- Time in seconds to pass platform
noncomputable def train_length := 120    -- Train length derived from conditions

theorem train_pass_time 
  (speed_in_kmh : ℕ) (speed_in_ms : ℕ) (platform_len : ℕ) (pass_platform_time : ℕ) (train_len : ℕ)
  (h1 : speed_in_kmh = 36)
  (h2 : speed_in_ms = 10)
  (h3 : platform_len = 180)
  (h4 : pass_platform_time = 30)
  (h5 : train_len = 120) :
  (train_len / speed_in_ms) = 12 := by
  sorry

end train_pass_time_l586_586393


namespace circle_radius_l586_586748

theorem circle_radius (X Y : ℝ) (h1 : X = real.pi * r^2) (h2 : Y = 2 * real.pi * r) (h3 : X / Y = 10) : r = 20 :=
sorry

end circle_radius_l586_586748


namespace find_current_velocity_l586_586384

-- Given definitions
def rowing_speed : ℝ := 10  -- The person's rowing speed in still water in kmph
def distance : ℝ := 72      -- The distance to the place in km

-- Definition of the total round trip time
def total_time (v : ℝ) : ℝ := (distance / (rowing_speed - v)) + (distance / (rowing_speed + v))

-- The theorem we want to state and prove
theorem find_current_velocity (v : ℝ) (h : total_time v = 15) : v = 2 :=
by sorry

end find_current_velocity_l586_586384


namespace obtuse_triangle_partition_l586_586263

theorem obtuse_triangle_partition (n : ℕ) (h : 0 < n) :
  ∃ (T : Finset (ℕ × ℕ × ℕ)), 
    T.card = n ∧ 
    (∀ (t ∈ T), 
      t.1 ∈ Finset.Icc 2 (3 * n + 1) ∧ 
      t.2 ∈ Finset.Icc 2 (3 * n + 1) ∧ 
      t.3 ∈ Finset.Icc 2 (3 * n + 1) ∧ 
      t.1 < t.2 ∧ t.2 < t.3 ∧ 
      t.1 + t.2 > t.3 ∧ 
      t.1 * t.1 + t.2 * t.2 < t.3 * t.3) :=
sorry

end obtuse_triangle_partition_l586_586263


namespace transform_to_opposite_l586_586364

theorem transform_to_opposite (n : ℕ) (A : matrix (fin n) (fin n) ℤ)
    (h1 : ∀ i : fin n, ∃! j : fin n, A i j = 1)
    (h2 : ∀ i : fin n, ∃! j : fin n, A i j = -1)
    (h3 : ∀ j : fin n, ∃! i : fin n, A i j = 1)
    (h4 : ∀ j : fin n, ∃! i : fin n, A i j = -1) :
    ∃ P Q : matrix (fin n) (fin n) ℤ, is_permutation_matrix P ∧ is_permutation_matrix Q ∧ P ⬝ A ⬝ Q = -A :=
begin
    sorry
end

end transform_to_opposite_l586_586364


namespace polynomial_divisibility_l586_586993

theorem polynomial_divisibility (
  p q r s : ℝ
) :
  (x^5 + 5 * x^4 + 10 * p * x^3 + 10 * q * x^2 + 5 * r * x + s) % (x^4 + 4 * x^3 + 6 * x^2 + 4 * x + 1) = 0 ->
  (p + q + r) * s = -2 :=
by {
  sorry
}

end polynomial_divisibility_l586_586993


namespace number_of_sides_l586_586177

-- Define the given conditions as Lean definitions

def exterior_angle := 72
def sum_of_exterior_angles := 360

-- Now state the theorem based on these conditions

theorem number_of_sides (n : ℕ) (h1 : exterior_angle = 72) (h2 : sum_of_exterior_angles = 360) : n = 5 :=
by
  sorry

end number_of_sides_l586_586177


namespace ratio_of_first_term_to_common_ratio_l586_586726

variable (a r : ℝ)
variable (ha : a ≠ 0)
variable (hr : r ≠ 1)
variable (S : ℕ → ℝ)

-- Definition of the sum of first n terms of a geometric progression
def geometric_sum (n : ℕ) : ℝ := a * (1 - r ^ n) / (1 - r)

-- Given conditions
axiom sum_eqn1 : S 3 = geometric_sum a r 3
axiom sum_eqn2 : S 6 = geometric_sum a r 6
axiom condition : S 6 = 3 * S 3

theorem ratio_of_first_term_to_common_ratio :
  a / r = 1 / Real.cbrt 2 :=
by
  sorry

end ratio_of_first_term_to_common_ratio_l586_586726


namespace honzik_total_cost_l586_586568

theorem honzik_total_cost : 
  ∃ (x z : ℕ), (3 * x = 24) ∧ -- Price of one lollipop
               (4 * x + (109 - 4 * x) = 109) ∧ -- Price of four lollipops plus cost of ice creams
               (109 - 4 * x) % z = 0 ∧ -- Total ice cream cost is divisible by price of one ice cream
               (1 < (109 - 4 * x) / z) ∧ -- More than one ice cream
               ((109 - 4 * x) / z < 10) ∧ -- Less than ten ice creams
               (x + z = 19) := -- Honzík's total cost
by {
  let x := 8,
  let z := 11,
  have h1 : 3 * x = 24 := by norm_num,
  have h2 : 4 * x + (109 - 4 * x) = 109 := by norm_num,
  have h3 : (109 - 4 * x) % z = 0 := by norm_num,
  have h4 : 1 < (109 - 4 * x) / z := by norm_num,
  have h5 : ((109 - 4 * x) / z < 10) := by norm_num,
  have h6 : x + z = 19 := by norm_num,
  exact ⟨x, z, h1, h2, h3, h4, h5, h6⟩,
}

end honzik_total_cost_l586_586568


namespace regular_polygon_exterior_angle_l586_586169

theorem regular_polygon_exterior_angle (n : ℕ) (h : 72 = 360 / n) : n = 5 :=
by
  sorry

end regular_polygon_exterior_angle_l586_586169


namespace geometric_probability_l586_586107

theorem geometric_probability :
  let total_length := (1/2 - 0)
  let favorable_length := (1/3 - 0)
  total_length > 0 ∧ favorable_length > 0 →
  (favorable_length / total_length) = 2 / 3 :=
by
  intros _ _ _
  sorry

end geometric_probability_l586_586107


namespace regular_price_of_pony_jeans_l586_586476

-- Define the regular price of fox jeans
def fox_jeans_price := 15

-- Define the given conditions
def pony_discount_rate := 0.18
def total_savings := 9
def total_discount_rate := 0.22

-- State the problem: Prove the regular price of pony jeans
theorem regular_price_of_pony_jeans : 
  ∃ P, P * pony_discount_rate = 3.6 :=
by
  sorry

end regular_price_of_pony_jeans_l586_586476


namespace value_of_square_l586_586077

variable (x y : ℝ)

theorem value_of_square (h1 : x * (x + y) = 30) (h2 : y * (x + y) = 60) :
  (x + y) ^ 2 = 90 := sorry

end value_of_square_l586_586077


namespace fencing_calculation_l586_586674

noncomputable def perimeter_square (side: ℝ) : ℝ := 4 * side

noncomputable def perimeter_rectangle (length width: ℝ) : ℝ := 2 * (length + width)

noncomputable def circumference_circle (radius: ℝ) : ℝ := 2 * Real.pi * radius

noncomputable def perimeter_triangle (a b c: ℝ) : ℝ := a + b + c

noncomputable def perimeter_irregular (p: ℝ) : ℝ := p

noncomputable def perimeter_l_shaped (l1 w1 l2 w2 shared: ℝ) : ℝ :=
  l1 + w1 + l2 + w2 + shared

def total_fencing_needed : ℝ :=
  let playground := perimeter_square 27
  let veg_garden1 := perimeter_rectangle 12 9
  let flower_bed := circumference_circle 5
  let sandpit := perimeter_triangle 7 10 13
  let irregular := perimeter_irregular 18
  let veg_garden2 := perimeter_l_shaped 6 4 3 4 6
  playground + veg_garden1 + flower_bed + sandpit + irregular + veg_garden2

theorem fencing_calculation : total_fencing_needed ≈ 256.42 := 
  by sorry

end fencing_calculation_l586_586674


namespace y_m_odd_exists_infinite_subseq_b_n_l586_586869

-- Definition of x_m, y_m, and the sequence a_n
def seq_x_y (m : ℕ) (x y : ℕ) : Prop :=
  (sqrt 2 + 1) ^ m = sqrt 2 * x + y

def a_n (n : ℕ) : ℕ := ⌊sqrt 2 * n⌋

-- Main theorem statements
theorem y_m_odd (m : ℕ) (x y : ℕ) (h : seq_x_y m x y) (h1 : ∃ y1, y1 = 1 ∧ Odd y1) :
  Odd y :=
  sorry

theorem exists_infinite_subseq_b_n (a_n : ℕ → ℕ) (x y : ℕ) (h : seq_x_y x y) :
  ∃ (b_n : ℕ → ℕ), (∀ n, b_n n % 4 = 1) ∧ ∀ N, ∃ n ≥ N, a_n n = b_n n :=
  sorry

end y_m_odd_exists_infinite_subseq_b_n_l586_586869


namespace prod_x_y_congruence_l586_586271

theorem prod_x_y_congruence (p : ℕ) (hp : Nat.Prime p) (h_mod4 : p % 4 = 3) :
  ∏ x in Finset.Ico 1 ((p - 1) / 2 + 1), ∏ y in Finset.Ico (x + 1) ((p - 1) / 2 + 1), (x^2 + y^2) % p ≡ (-1) ^ (Nat.floor ((p + 1) / 8)) % p :=
by
  sorry

end prod_x_y_congruence_l586_586271


namespace geometric_series_common_ratio_l586_586403

theorem geometric_series_common_ratio (a S r : ℝ) (ha : a = 512) (hS : S = 2048) (h_sum : S = a / (1 - r)) : r = 3 / 4 :=
by
  rw [ha, hS] at h_sum 
  sorry

end geometric_series_common_ratio_l586_586403


namespace ideal_listening_state_duration_optimal_start_time_for_question_l586_586268

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 8 then 2 * x + 68
  else if 8 ≤ x ∧ x ≤ 40 then -1 / 8 * (x^2 - 32 * x - 480)
  else 0

theorem ideal_listening_state_duration : 
  (∃ a b : ℝ, 0 ≤ a ∧ b ≤ 40 ∧ (∀ x, a ≤ x ∧ x ≤ b → f x ≥ 80) ∧ (b - a ≈ 20)) :=
sorry

theorem optimal_start_time_for_question : 
  (∃ t : ℝ, 0 ≤ t ≤ 6 ∧ (∀ x, t ≤ x ∧ x ≤ t + 24 → f x ≥ (f (t + 24)))) :=
sorry

end ideal_listening_state_duration_optimal_start_time_for_question_l586_586268


namespace sum_of_coefficients_of_g_equals_9_l586_586302

noncomputable def g (x : ℂ) : ℂ := x^4 + (-2 : ℂ) * x^3 + (11 : ℂ) * x^2 + (-18 : ℂ) * x + 18

theorem sum_of_coefficients_of_g_equals_9 :
  g(1 + complex.I) = 0 ∧ g(3 * complex.I) = 0 → 
  let a := -2 
  let b := 11 
  let c := -18 
  let d := 18 
  a + b + c + d = 9 :=
by
  intro h
  let a := -2 
  let b := 11 
  let c := -18 
  let d := 18 
  have sum := a + b + c + d 
  show sum = 9
  sorry

end sum_of_coefficients_of_g_equals_9_l586_586302


namespace binomial_coeff_sum_l586_586422

theorem binomial_coeff_sum : 
  (Nat.choose 3 2) + (Nat.choose 4 2) + (Nat.choose 5 2) + (Nat.choose 6 2) + (Nat.choose 7 2) + (Nat.choose 8 2) = 83 := by
  sorry

end binomial_coeff_sum_l586_586422


namespace green_tractor_price_l586_586703

-- Define the conditions
def salary_based_on_sales (r_ct : Nat) (r_price : ℝ) (g_ct : Nat) (g_price : ℝ) : ℝ :=
  0.1 * r_ct * r_price + 0.2 * g_ct * g_price

-- Define the problem's Lean statement
theorem green_tractor_price
  (r_ct : Nat) (g_ct : Nat)
  (r_price : ℝ) (total_salary : ℝ)
  (h_rct : r_ct = 2)
  (h_gct : g_ct = 3)
  (h_rprice : r_price = 20000)
  (h_salary : total_salary = 7000) :
  ∃ g_price : ℝ, salary_based_on_sales r_ct r_price g_ct g_price = total_salary ∧ g_price = 5000 :=
by
  sorry

end green_tractor_price_l586_586703


namespace probability_merlin_dismissed_l586_586219

variable (p : ℝ) (q : ℝ := 1 - p)
variable (r : ℝ := 1 / 2)

theorem probability_merlin_dismissed :
  (0 < p ∧ p ≤ 1) →
  (q = 1 - p) →
  (r = 1 / 2) →
  (probability_king_correct_two_advisors = p) →
  (strategy_merlin_minimizes_dismissal = true) →
  (strategy_percival_honest = true) →
  probability_merlin_dismissed = 1 / 2 := by
  sorry

end probability_merlin_dismissed_l586_586219


namespace order_x_y_z_l586_586482

noncomputable def x : ℝ := 0.82 ^ 0.5
noncomputable def y : ℝ := Real.sin 1
noncomputable def z : ℝ := Real.log (7 ^ (1 / 3 : ℝ))

theorem order_x_y_z : y < z ∧ z < x :=
by
  -- sorry placeholder for the proof itself
  sorry

end order_x_y_z_l586_586482


namespace green_tractor_price_is_5000_l586_586702

-- Definitions based on the given conditions
def red_tractor_price : ℝ := 20000
def green_tractor_commission_rate : ℝ := 0.20
def red_tractor_commission_rate : ℝ := 0.10
def red_tractors_sold : ℕ := 2
def green_tractors_sold : ℕ := 3
def salary : ℝ := 7000

-- The theorem statement
theorem green_tractor_price_is_5000 
  (rtp : ℝ := red_tractor_price)
  (gtcr : ℝ := green_tractor_commission_rate)
  (rtcr : ℝ := red_tractor_commission_rate)
  (rts : ℕ := red_tractors_sold)
  (gts : ℕ := green_tractors_sold)
  (s : ℝ := salary) :
  let earnings_red := rts * (rtcr * rtp) in
  let earnings_green := s - earnings_red in
  let green_tractor_price := (earnings_green / gts) / gtcr in
  green_tractor_price = 5000 := sorry

end green_tractor_price_is_5000_l586_586702


namespace marble_difference_l586_586965

theorem marble_difference :
  ∃ (A B C : ℕ), A = 28 ∧ B > 28 ∧ C = 2 * B ∧ A + B + C = 148 ∧ (B - A = 12) :=
by
  use 28, 40, 80
  split
  -- A = 28
  { refl }
  split
  -- B > 28
  { norm_num }
  split
  -- C = 2B
  { norm_num }
  split
  -- A + B + C = 148
  { norm_num }
  -- B - A = 12
  norm_num
  done

end marble_difference_l586_586965


namespace max_sin_a_l586_586978

theorem max_sin_a (a b : ℝ)
  (h1 : b = Real.pi / 2 - a)
  (h2 : Real.cos (a + b) = Real.cos a + Real.cos b) :
  Real.sin a ≤ Real.sqrt 2 / 2 :=
sorry

end max_sin_a_l586_586978


namespace min_surface_area_of_sphere_l586_586875

noncomputable def cuboid_diagonal (x y : ℝ) := sqrt (x^2 + y^2 + 4)

theorem min_surface_area_of_sphere (r x y : ℝ) 
  (h1 : 2 * r = cuboid_diagonal x y)
  (h2 : x * y = 6) 
  (AB_eq_2 : 2 = 2) :
  4 * π * r^2 = 16 * π :=
by
  sorry

end min_surface_area_of_sphere_l586_586875


namespace problem_statement_l586_586932

theorem problem_statement (P Q : ℝ) (h1 : P = 3^2000 + 3^(-2000)) (h2 : Q = 3^2000 - 3^(-2000)) : P^2 - Q^2 = 4 :=
by
  sorry

end problem_statement_l586_586932


namespace inclination_angle_l586_586661

theorem inclination_angle (α : ℝ) (h : α ∈ Ioo 0 180):
  (∃ k : ℝ, ∀ x y : ℝ, x - y + 1 = 0 → y = k * x + 1) →
  tan α = 1 →
  α = 45 :=
by {
  intros h_line h_tan,
  sorry
}

end inclination_angle_l586_586661


namespace seq_sum_first_20_terms_l586_586955

theorem seq_sum_first_20_terms (a : ℕ → ℤ) :
  (∀ n : ℕ, n > 0 → a (n + 1) + a n = (-1)^n * n) →
  ∑ i in Finset.range 20, a i = -100 :=
by sorry

end seq_sum_first_20_terms_l586_586955


namespace max_cos_sin_l586_586465

theorem max_cos_sin (h : 0 < θ ∧ θ < π) : 
  ∃ θ : ℝ, 0 < θ ∧ θ < π ∧ cos (θ / 2) * (1 + sin θ) = sqrt 2 :=
by
  use π / 2
  have : cos (π / 4) = sqrt 2 / 2 := sorry
  have : sin (π / Π)= 1 := sorry
  calc
    cos (π / 4) * (1 + sin (π / Π))
    = sqrt 2 / 2 * 2 : by rw [this, this]
    = sqrt 2 : by ring

end max_cos_sin_l586_586465


namespace SSR_calculation_l586_586895

noncomputable def SSR (f : ℝ → ℝ) (data : List (ℝ × ℝ)) : ℝ :=
  data.foldl (λ acc (x, y), acc + (y - f x)^2) 0

theorem SSR_calculation : 
  let f := λ x, 2 * x + 1,
      data := [(2, 4.9), (3, 7.1), (4, 9.1)]
  in SSR f data = 0.03 :=
by
  let f := λ x, 2 * x + 1
  let data := [(2, 4.9), (3, 7.1), (4, 9.1)]
  sorry

end SSR_calculation_l586_586895


namespace math_problem_l586_586549

theorem math_problem (x t : ℝ) (h1 : 6 * x + t = 4 * x - 9) (h2 : t = 7) : x + 4 = -4 := by
  sorry

end math_problem_l586_586549
