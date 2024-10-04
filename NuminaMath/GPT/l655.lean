import Mathlib

namespace simplest_form_option_l655_655200

theorem simplest_form_option (x y : ℚ) :
  (∀ (a b : ℚ), (a ≠ 0 ∧ b ≠ 0 → (12 * (x - y) / (15 * (x + y)) ≠ 4 * (x - y) / 5 * (x + y))) ∧
   ∀ (a b : ℚ), (a ≠ 0 ∧ b ≠ 0 → (x^2 + y^2) / (x + y) = a / b) ∧
   ∀ (a b : ℚ), (a ≠ 0 ∧ b ≠ 0 → (x^2 - y^2) / ((x + y)^2) ≠ (x - y) / (x + y)) ∧
   ∀ (a b : ℚ), (a ≠ 0 ∧ b ≠ 0 → (x^2 - y^2) / (x + y) ≠ x - y)) := sorry

end simplest_form_option_l655_655200


namespace largest_prime_factor_of_factorial_sum_is_5_l655_655588

-- Define factorial function
def fact (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * fact (n - 1)

-- Define the specific factorial sums in the problem
def n_factorial_sum : ℕ := fact 6 + fact 7

-- Lean statement to assert the largest prime factor of the sum
theorem largest_prime_factor_of_factorial_sum_is_5 :
  (∀ p : ℕ, (nat.prime p ∧ (p ∣ n_factorial_sum)) → p ≤ 5) ∧
  (∃ p : ℕ, nat.prime p ∧ (p ∣ n_factorial_sum) ∧ p = 5) :=
by
  -- Proof of the theorem
  sorry

end largest_prime_factor_of_factorial_sum_is_5_l655_655588


namespace find_AE_l655_655876

-- Definitions of points and distances
def intersecting_circles_with_tangents (S1 S2 : Type) (A B D C E : S1) :=
  -- Conditions from the problem
  ∃ (S1_intersect_S2 : S1) (A B S1_intersect_S2 : S1),
  ∃ (line_through_B_intersect_S1 : S1) (D B line_through_B_intersect_S1 : S1),
  ∃ (line_through_B_intersect_S2 : S1) (C B line_through_B_intersect_S2 : S1),
  ∃ (tangent_to_S1_at_D : S1) (D tangent_to_S1_at_D : S1),
  ∃ (tangent_to_S2_at_C : S1) (C tangent_to_S2_at_C : S1),
  ∃ (meet_of_tangents_at_E : S1) (E meet_of_tangents_at_E : S1),
  -- Given distances
  dist A D = 15 ∧
  dist A C = 16 ∧
  dist A B = 10

-- Theorem to prove that |AE| = 24
theorem find_AE (S1 S2 : Type) (A B D C E : S1) 
  (h : intersecting_circles_with_tangents S1 S2 A B D C E) : 
  dist A E = 24 :=
sorry

end find_AE_l655_655876


namespace solution_set_l655_655530

def solve_inequalities (x : ℝ) : Prop :=
  (3 * x - 2) / (x - 6) ≤ 1 ∧ 2 * x ^ 2 - x - 1 > 0

theorem solution_set : { x : ℝ | solve_inequalities x } = { x : ℝ | (-2 ≤ x ∧ x < 1/2) ∨ (1 < x ∧ x < 6) } :=
by sorry

end solution_set_l655_655530


namespace largest_prime_factor_of_6_factorial_plus_7_factorial_l655_655591

theorem largest_prime_factor_of_6_factorial_plus_7_factorial :
  ∃ p : ℕ, Nat.Prime p ∧ p = 5 ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ (6! + 7!) → q ≤ 5 :=
by
  sorry

end largest_prime_factor_of_6_factorial_plus_7_factorial_l655_655591


namespace limit_d_n_63_l655_655097

/-- Define the function d -/
def d : ℕ → ℤ
| 1          := 0
| n @ (p ::  []) := if is_prime p then 1 else 0
| (m * n)    := m * d n + n * d m

/-- Property about primes and number of primes -/
def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m l : ℕ, n = m * l → m = 1 ∨ l = 1

/-- Define the iterated function d_n -/
noncomputable def d_n : ℕ → ℕ → ℤ
| 1 m       := d m
| (n + 1) m := d (d_n n m)

/-- State the final limit property -/
theorem limit_d_n_63 : ∀ N, ∃ n ≥ N, d_n n 63 > 48 := 
sorry -- leaving the proof as an exercise

end limit_d_n_63_l655_655097


namespace problem_part_one_problem_part_two_l655_655307

/-- Define sequence a_n and summation T_n based on given conditions -/
def seq (n : ℕ) : ℕ := 
  if n = 1 then 1 
  else 3^(n-1) + seq (n-1)

def sum_first_n_terms (n : ℕ) : ℝ := 
  (∑ i in Finset.range n, seq (i + 1))

/-- Prove the specific values a_2 and a_3 and the general formula for a_n -/
theorem problem_part_one : 
  seq 2 = 4 ∧ seq 3 = 13 ∧ ∀ n : ℕ, n ≥ 1 → (seq n : ℝ) = (3^n / 2) - 1 / 2 :=
by sorry

/-- Define the sum T_n and prove the given formula for T_n -/
theorem problem_part_two :
  ∀ n : ℕ, (sum_first_n_terms n : ℝ) = (3 * (seq n : ℝ) - n) / 2 :=
by sorry

end problem_part_one_problem_part_two_l655_655307


namespace negation_of_proposition_l655_655158

theorem negation_of_proposition :
  ¬(∃ x₀ : ℝ, 0 < x₀ ∧ Real.log x₀ = x₀ - 1) ↔ ∀ x : ℝ, 0 < x → Real.log x ≠ x - 1 :=
by
  sorry

end negation_of_proposition_l655_655158


namespace find_ellipse_eq_l655_655719

noncomputable def ellipse_eq (a b c : ℝ) := 
  ∃ x y : ℝ, (c^2 = a^2 - b^2) ∧
             ((x^2 / (a^2) + y^2 / (b^2) = 1) ∨ 
              (x^2 / (4*b^2) + y^2 / b^2 = 1) ∨ 
              (x^2 / b^2 + y^2 / (4*b^2) = 1))

theorem find_ellipse_eq : 
  (ellipse_eq 6 (sqrt 20) 4 ∧ ellipse_eq (2*sqrt 17) (sqrt 17) (sqrt 68) ∧ ellipse_eq (sqrt 8) 2 (sqrt 32)) :=
sorry

end find_ellipse_eq_l655_655719


namespace exists_disjoint_subset_of_unit_circles_l655_655239

theorem exists_disjoint_subset_of_unit_circles
  (U : Set (Set (ℝ × ℝ)))
  (S : ℝ)
  (hU : ∀ u ∈ U, is_unit_circle u)
  (h_area : measure_theory.measure_union U = S):
  ∃ V ⊆ U, mutual_disjoint V ∧ measure_theory.measure_union V > 25 / 9 := 
sorry

end exists_disjoint_subset_of_unit_circles_l655_655239


namespace find_d_vector_l655_655754

theorem find_d_vector : 
  ∃ (d : ℝ × ℝ), 
  ∀ (x y t : ℝ), 
  (y = (2 * x - 4) / 3) →
  (x ≥ 2) →
  (∥(x - 2, y - 0)∥ = t) →
  (x, y) = (2, 0) + t • d :=
begin
  use (3 / real.sqrt 13, 2 / real.sqrt 13),
  intros x y t hline hgeq hdist,
  sorry
end

end find_d_vector_l655_655754


namespace bus_time_in_motion_l655_655118

theorem bus_time_in_motion :
  ∃ t : ℕ, 
    let avg_speed_without_stops := 600 / t,
        avg_speed_with_stops := 600 / (t + 6) in
    avg_speed_without_stops = avg_speed_with_stops + 5 ∧ t = 24 :=
sorry

end bus_time_in_motion_l655_655118


namespace polynomial_no_positive_roots_l655_655491

/- Definitions based on the conditions in the problem -/
variables {a : ℕ → ℕ} {n k M : ℕ}

-- Assumptions
def positive_integers : Prop := ∀ i, 1 ≤ i → i ≤ n → a i > 0

def sum_reciprocals_eq_k : Prop := (finset.sum (finset.range n) (λ i, 1 / (a i))) = k

def product_eq_M : Prop := (finset.prod (finset.range n) (λ i, a i)) = M

def M_gt_1 : Prop := M > 1

/- Theorem Statement -/
theorem polynomial_no_positive_roots
  (h1 : positive_integers)
  (h2 : sum_reciprocals_eq_k)
  (h3 : product_eq_M)
  (h4 : M_gt_1) :
  ∀ (X : ℝ), X > 0 → M * (1 + X)^k ≠ (finset.prod (finset.range n) (λ i, X + a i)) := 
by 
  sorry

end polynomial_no_positive_roots_l655_655491


namespace gcd_lcm_336_1260_l655_655339

def gcd_lcm_example (a b : ℕ) : Prop :=
  gcd a b = 84 ∧ lcm a b = 5040

theorem gcd_lcm_336_1260: gcd_lcm_example 336 1260 :=
by
  unfold gcd_lcm_example
  split
  sorry
  sorry

end gcd_lcm_336_1260_l655_655339


namespace number_of_zeros_l655_655096

theorem number_of_zeros (f : ℝ → ℝ) (h1 : f = λ x, Real.sin (Real.pi * x)) (h2 : ∀ x₀, f x₀ = 0 → |x₀| + f (x₀ + 1 / 2) < 11) : 
  (∃ n, n = 21 ∧ ∃ s : Finset ℝ, s.card = n ∧ ∀ x₀ ∈ s, f x₀ = 0 ∧ |x₀| + f (x₀ + 1 / 2) < 11) :=
by
  sorry

end number_of_zeros_l655_655096


namespace sin_alpha_cases_l655_655036

theorem sin_alpha_cases (α : ℝ) 
  (h : ∃ (x y : ℝ), (x = cos (3 * π / 4)) ∧ (y = sin (3 * π / 4)) ∧ (x, y) = (cos α, sin α)) : 
  sin α = sqrt 2 / 2 ∨ sin α = -sqrt 2 / 2 :=
by
  sorry

end sin_alpha_cases_l655_655036


namespace base_seven_sum_digits_of_product_l655_655162

def base_seven_to_decimal (n : ℕ) : ℕ :=
  -- Function to convert base seven to decimal
  (n / 10) * 7 + (n % 10)

def product_in_base_seven (n1 n2 : ℕ) : ℕ :=
  -- Function to compute product in decimal and convert back to base seven
  let product_decimal := base_seven_to_decimal n1 * base_seven_to_decimal n2
  (product_decimal / 343) * 1000 + ((product_decimal % 343) / 49) * 100 +
  ((product_decimal % 49) / 7) * 10 + (product_decimal % 7)

def sum_of_digits_base_seven (n : ℕ) : ℕ :=
  -- Function to sum the digits of a base seven number
  (n / 1000) + ((n % 1000) / 100) + ((n % 100) / 10) + (n % 10)

theorem base_seven_sum_digits_of_product (n1 n2 : ℕ) :
  let product_base_seven := product_in_base_seven n1 n2 in
  sum_of_digits_base_seven product_base_seven = 15 :=
by {
  sorry
}

end base_seven_sum_digits_of_product_l655_655162


namespace train_length_l655_655653

theorem train_length
  (cross_time : ℝ) 
  (bridge_length : ℝ) 
  (train_speed : ℝ) 
  (cross_time_eq : cross_time = 36)
  (bridge_length_eq : bridge_length = 200)
  (train_speed_eq : train_speed = 29) :
  ∃ (L : ℝ), L + bridge_length = train_speed * cross_time ∧ L = 844 :=
by
  use 844
  simp [cross_time_eq, bridge_length_eq, train_speed_eq]
  norm_num
  sorry

end train_length_l655_655653


namespace smallest_n_l655_655916

theorem smallest_n (n : ℕ) : (n > 0) ∧ (2^n % 30 = 1) → n = 4 :=
by
  intro h
  sorry

end smallest_n_l655_655916


namespace initial_bananas_proof_l655_655569

def initial_bananas (total bananas_added : ℕ) : ℕ := total - bananas_added

theorem initial_bananas_proof (total bananas_added : ℕ) (h_total : total = 9) (h_added : bananas_added = 7) : initial_bananas total bananas_added = 2 := by
  have h1 : initial_bananas 9 7 = 2 := by
    unfold initial_bananas
    simp
  exact h1

end initial_bananas_proof_l655_655569


namespace find_y_l655_655255

theorem find_y (y : ℝ) (hy_pos : y > 0) (hy_prop : y^2 / 100 = 9) : y = 30 := by
  sorry

end find_y_l655_655255


namespace original_price_computer_l655_655129

noncomputable def first_store_price (P : ℝ) : ℝ := 0.94 * P

noncomputable def second_store_price (exchange_rate : ℝ) : ℝ := (920 / 0.95) * exchange_rate

theorem original_price_computer 
  (exchange_rate : ℝ)
  (h : exchange_rate = 1.1) 
  (H : (first_store_price P - second_store_price exchange_rate = 19)) :
  P = 1153.47 :=
by
  sorry

end original_price_computer_l655_655129


namespace domain_of_g_l655_655981

noncomputable def g (x : ℝ) : ℝ := log 3 (log 4 (log 5 (log 6 x)))

theorem domain_of_g : {x : ℝ | x > 7776} = {x : ℝ | real.logb 3 (real.logb 4 (real.logb 5 (real.logb 6 x)))} :=
sorry

end domain_of_g_l655_655981


namespace triangle_perimeter_l655_655944

-- Define the ratios
def ratio1 : ℚ := 1 / 2
def ratio2 : ℚ := 1 / 3
def ratio3 : ℚ := 1 / 4

-- Define the longest side
def longest_side : ℚ := 48

-- Compute the perimeter given the conditions
theorem triangle_perimeter (ratio1 ratio2 ratio3 : ℚ) (longest_side : ℚ) 
  (h_ratio1 : ratio1 = 1 / 2) (h_ratio2 : ratio2 = 1 / 3) (h_ratio3 : ratio3 = 1 / 4)
  (h_longest_side : longest_side = 48) : 
  (longest_side * 6/ (ratio1 * 12 + ratio2 * 12 + ratio3 * 12)) = 104 := by
  sorry

end triangle_perimeter_l655_655944


namespace is_incorrect_B_l655_655731

variable {a b c : ℝ}

theorem is_incorrect_B :
  ¬ ((a > b ∧ b > c) → (1 / (b - c)) < (1 / (a - c))) :=
sorry

end is_incorrect_B_l655_655731


namespace sample_capacity_n_l655_655263

theorem sample_capacity_n
  (n : ℕ) 
  (engineers technicians craftsmen : ℕ) 
  (total_population : ℕ)
  (stratified_interval systematic_interval : ℕ) :
  engineers = 6 →
  technicians = 12 →
  craftsmen = 18 →
  total_population = engineers + technicians + craftsmen →
  total_population = 36 →
  (∃ n : ℕ, n ∣ total_population ∧ 6 ∣ n ∧ 35 % (n + 1) = 0) →
  n = 6 :=
by
  sorry

end sample_capacity_n_l655_655263


namespace domain_of_g_l655_655977

noncomputable def g (x : ℝ) : ℝ := Real.logb 3 (Real.logb 4 (Real.logb 5 (Real.logb 6 x)))

theorem domain_of_g : setOf (λ x, g x) = set.Ioi (6^625) :=
by
  sorry

end domain_of_g_l655_655977


namespace quadratic_b_value_l655_655678

theorem quadratic_b_value (b : ℝ) (n : ℝ) (h_b_neg : b < 0) 
  (h_equiv : ∀ x : ℝ, (x + n)^2 + 1 / 16 = x^2 + b * x + 1 / 4) : 
  b = - (Real.sqrt 3) / 2 := 
sorry

end quadratic_b_value_l655_655678


namespace midpoint_on_circumcircle_of_triangle_ADZ_l655_655104

theorem midpoint_on_circumcircle_of_triangle_ADZ
  {A B C D Z : Point}
  (h_triangle: Triangle A B C)
  (h_AB_lt_AC : dist A B < dist A C)
  (h_D_on_angle_bisector : D lies_on_bisector BAC A circumcircle A B C)
  (h_Z_on_perpendicular_bisector_external_bisector : Z lies_on_perpendicular_bisector_both_angle AC and_external_bisector A B C) :
  midpoint A B lies_on_circumcircle A D Z :=
begin
  sorry
end

end midpoint_on_circumcircle_of_triangle_ADZ_l655_655104


namespace arithmetic_and_geometric_means_of_reciprocals_primes_l655_655335

theorem arithmetic_and_geometric_means_of_reciprocals_primes :
  let primes := [2, 3, 5, 7]
  let reciprocals := [1/2, 1/3, 1/5, 1/7]
  (∀ r ∈ reciprocals, r = 1 / primes[nth r]) →
  (arithmetic_mean : ℚ := (reciprocals.sum) / primes.length) →
  (geometric_mean : ℚ := (reciprocals.prod) ^ (1 / primes.length)) →
  arithmetic_mean = 247 / 840 ∧ geometric_mean = 1 / 210^(1/4) := 
by
  sorry

end arithmetic_and_geometric_means_of_reciprocals_primes_l655_655335


namespace cells_covered_by_exactly_two_squares_l655_655613

/-- 
  Given three 5x5 squares on a piece of graph paper where some cells are covered by exactly two squares,
  prove that the number of cells covered by exactly two squares is 13.
-/
theorem cells_covered_by_exactly_two_squares : 
  let squares : list (set (ℕ × ℕ)) := 
    [((λ x y => (x, y)), (λ x => ∀ y, x ≤ y ∧ y ≤ x + 4)),
     ((λ x y => (x + 2, y + 2)), (λ x => ∀ y, x ≤ y ∧ y ≤ x + 4)),
     ((λ x y => (x + 4, y + 1)), (λ x => ∀ y, x ≤ y ∧ y ≤ x + 4))] in
  let countCoveredByTwoSquares := 
    λ (squares : list (set (ℕ × ℕ))) => 
      |{cell : ℕ × ℕ | (∃ sq1 ∈ squares, cell ∈ sq1) ∧ 
                      (∃ sq2 ∈ squares, cell ∈ sq2 ∧ sq1 ≠ sq2) ∧ 
                      ¬(∃ sq3 ∈ squares, sq3 ≠ sq1 ∧ sq3 ≠ sq2 ∧ cell ∈ sq3)}| in 
  countCoveredByTwoSquares squares = 13 := 
sorry

end cells_covered_by_exactly_two_squares_l655_655613


namespace muffin_cost_ratio_l655_655533

theorem muffin_cost_ratio (m b : ℝ) 
  (h1 : 5 * m + 4 * b = 20)
  (h2 : 3 * (5 * m + 4 * b) = 60)
  (h3 : 3 * m + 18 * b = 60) :
  m / b = 13 / 4 :=
by
  sorry

end muffin_cost_ratio_l655_655533


namespace sum_of_ratios_eq_one_l655_655654

open Triangle Circle

variable {α β γ: ℝ}

-- Conditions
def triangle_inscribed_in_circle (A B C O: Point) (R: ℝ): Prop :=
  ∃ (C : Circle), C.radius = R ∧ C.contains A ∧ C.contains B ∧ C.contains C

def ratio_radii_circles_tangent (O: Point) (R: ℝ): Prop :=
  ∃ (α β γ < 1), true  -- Here, we assume α, β, and γ exist and are < 1

-- Main statement to prove
theorem sum_of_ratios_eq_one (A B C O: Point) (R: ℝ) (α β γ: ℝ)
  (h1: triangle_inscribed_in_circle A B C O R)
  (h2: ratio_radii_circles_tangent O R):
  α + β + γ = 1 :=
sorry

end sum_of_ratios_eq_one_l655_655654


namespace length_of_platform_l655_655623

-- Define the relevant constants and variables
def train_length : ℝ := 500 -- in meters
def acceleration : ℝ := 2 -- in meters per second squared
def time_to_cross_platform : ℝ := 50 -- in seconds
def time_to_cross_signal : ℝ := 25 -- in seconds

-- The kinematic equations used in the problem
def speed_at_signal_pole : ℝ :=
  acceleration * time_to_cross_signal

def distance_covered_in_50_seconds : ℝ :=
  (0 * time_to_cross_platform) + (1 / 2 * acceleration * (time_to_cross_platform)^2)

-- The final calculation for the length of the platform
def platform_length : ℝ := 
  distance_covered_in_50_seconds - train_length

-- Prove that platform_length is 2000 meters under given conditions
theorem length_of_platform : platform_length = 2000 := by
  sorry

end length_of_platform_l655_655623


namespace medals_awarded_condition_l655_655821

theorem medals_awarded_condition :
  let total_sprinters := 10
  let american_sprinters := 4
  let nonamerican_sprinters := total_sprinters - american_sprinters
  let medals_awarded := 3
  let no_americans := nonamerican_sprinters * (nonamerican_sprinters - 1) * (nonamerican_sprinters - 2)
  let one_american := (american_sprinters.choose 1) * medals_awarded * (nonamerican_sprinters * (nonamerican_sprinters - 1)) 
  let two_americans := (american_sprinters.choose 2) * (medals_awarded * (medals_awarded - 1)) * nonamerican_sprinters 
  in no_americans + one_american + two_americans = 696 :=
by
  let total_sprinters := 10
  let american_sprinters := 4
  let nonamerican_sprinters := total_sprinters - american_sprinters
  let medals_awarded := 3
  let no_americans := nonamerican_sprinters * (nonamerican_sprinters - 1) * (nonamerican_sprinters - 2)
  let one_american := (american_sprinters.choose 1) * medals_awarded * (nonamerican_sprinters * (nonamerican_sprinters - 1))
  let two_americans := (american_sprinters.choose 2) * (medals_awarded * (medals_awarded - 1)) * nonamerican_sprinters
  have no_am_ways : no_americans = 120 := sorry
  have one_am_ways : one_american = 360 := sorry
  have two_am_ways : two_americans = 216 := sorry
  rw [no_am_ways, one_am_ways, two_am_ways]
  exact calc 
    120 + 360 + 216 = 120 + 576 := by sorry
    ... = 576 := by sorry
    ... = 696 := by sorry

end medals_awarded_condition_l655_655821


namespace sine_variance_half_l655_655767

def sin_variance (a0 a1 a2 a3 : ℝ) : ℝ :=
  (sin (a1 - a0))^2 + (sin (a2 - a0))^2 + (sin (a3 - a0))^2 / 3

theorem sine_variance_half (a0 : ℝ) : 
  sin_variance a0 (π / 2) (5 * π / 6) (7 * π / 6) = 1 / 2 :=
sorry

end sine_variance_half_l655_655767


namespace cos_double_angle_shift_l655_655377

noncomputable def α := sorry -- α is implicitly defined by the conditions.

theorem cos_double_angle_shift
  (h1 : cos (α + π/4) = 3/5)
  (h2 : π/2 < α ∧ α < 3 * π / 2) :
  cos (2 * α + π / 4) = - 31 * Real.sqrt 2 / 50 :=
by
  sorry

end cos_double_angle_shift_l655_655377


namespace sum_f_geq_n_f_m_l655_655480

variable {n : ℕ}
variable {b c d : ℝ}
variable {x : Fin n → ℝ}

-- Define the cubic polynomial f
def f (x : ℝ) : ℝ := x^3 + b * x^2 + c * x + d

-- Define the average of the numbers x_i
def m (x : Fin n → ℝ) : ℝ := (∑ i, x i) / n

-- State the theorem to be proved
theorem sum_f_geq_n_f_m
  (h_nonneg : ∀ i, 0 ≤ x i)
  (h_avg : m x ≥ -b / 2) :
  (∑ i, f (x i)) ≥ n * f (m x) :=
by
  sorry

end sum_f_geq_n_f_m_l655_655480


namespace sally_oscillation_distance_l655_655304

noncomputable def C : ℝ := 5 / 4
noncomputable def D : ℝ := 11 / 4

theorem sally_oscillation_distance :
  abs (C - D) = 3 / 2 :=
by
  sorry

end sally_oscillation_distance_l655_655304


namespace find_positive_integer_solutions_l655_655704

def is_solution (x y : ℕ) : Prop :=
  4 * x^3 + 4 * x^2 * y - 15 * x * y^2 - 18 * y^3 - 12 * x^2 + 6 * x * y + 36 * y^2 + 5 * x - 10 * y = 0

theorem find_positive_integer_solutions :
  ∀ x y : ℕ, 0 < x ∧ 0 < y → (is_solution x y ↔ (x = 1 ∧ y = 1) ∨ (∃ y', y = y' ∧ x = 2 * y' ∧ 0 < y')) :=
by
  intros x y hxy
  sorry

end find_positive_integer_solutions_l655_655704


namespace minimum_points_l655_655574

-- Define the conditions of the problem
def points (M : Type) := set M
def circle (M : Type) := set M

def passes_through (C : circle M) (pts : list (points M)) : Prop :=
  ∀ p, p ∈ pts → p ∈ C

-- The circles with specific points they pass through
def C1 (M : Type) := { p : points M | passes_through p [p] }
def C2 (M : Type) := { p : points M | passes_through p [p1, p2] }
def C3 (M : Type) := { p : points M | passes_through p [p1, p2, p3] }
def C4 (M : Type) := { p : points M | passes_through p [p1, p2, p3, p4] }
def C5 (M : Type) := { p : points M | passes_through p [p1, p2, p3, p4, p5] }
def C6 (M : Type) := { p : points M | passes_through p [p1, p2, p3, p4, p5, p6] }
def C7 (M : Type) := { p : points M | passes_through p [p1, p2, p3, p4, p5, p6, p7] }

-- The theorem stating the condition
theorem minimum_points (M : Type) (C1 C2 C3 C4 C5 C6 C7 : circle M) :
  ∃ (pts : points M), passes_through C1 [pt1] ∧
                      passes_through C2 [pt1, pt2] ∧
                      passes_through C3 [pt1, pt2, pt3] ∧
                      passes_through C4 [pt1, pt2, pt3, pt4] ∧
                      passes_through C5 [pt1, pt2, pt3, pt4, pt5] ∧
                      passes_through C6 [pt1, pt2, pt3, pt4, pt5, pt6] ∧
                      passes_through C7 [pt1, pt2, pt3, pt4, pt5, pt6, pt7] ∧
                      card pts = 12 :=
sorry

end minimum_points_l655_655574


namespace circle_outside_square_area_l655_655483

open Real

-- Definitions and conditions
def square_side_length : ℝ := 10
def circle_radius : ℝ := square_side_length / 2
def area_of_square : ℝ := square_side_length^2
def area_of_circle : ℝ := π * circle_radius^2

-- Statement to prove
theorem circle_outside_square_area : area_of_circle - area_of_square ≤ 0 := by
  sorry

end circle_outside_square_area_l655_655483


namespace handshake_count_l655_655288

theorem handshake_count {teams : Fin 4 → Fin 2 → Prop}
    (h_teams_disjoint : ∀ (i j : Fin 4) (x y : Fin 2), i ≠ j → teams i x → teams j y → x ≠ y)
    (unique_partner : ∀ (i : Fin 4) (x1 x2 : Fin 2), teams i x1 → teams i x2 → x1 = x2) : 
    24 = (∑ i : Fin 8, (∑ j : Fin 8, if i ≠ j ∧ ¬(∃ k : Fin 4, teams k i ∧ teams k j) then 1 else 0)) / 2 :=
by sorry

end handshake_count_l655_655288


namespace transform_sets_l655_655571

theorem transform_sets :
  ∀ (S1 S2 : List ℤ), 
  (∀ x ∈ S1, x = 1 ∨ x = -1) → 
  (∀ x ∈ S2, x = 1 ∨ x = -1) → 
  S1.length = 1958 → 
  S2.length = 1958 → 
  (∃ n : ℕ, transformable S1 S2 n) :=
sorry

-- Definition of the transformation operation in terms of allowed steps
def transformable (S1 S2 : List ℤ) (n : ℕ) : Prop :=
  ∃ steps : List (Fin 1958 → List ℤ → List ℤ),
    List.length steps = n ∧
    steps.apply_all S1 = S2

-- Definition of the application of all steps
def List.apply_all {α : Type} (steps : List (α → α)) (v : α) : α :=
  steps.foldl (λ x f => f x) v

end transform_sets_l655_655571


namespace range_of_a_l655_655878

noncomputable def f (x : ℝ) := (1 / 2) * x ^ 2 - 16 * Real.log x

theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, a - 1 ≤ x ∧ x ≤ a + 2 → (fderiv ℝ f x) x < 0)
  ↔ (1 < a) ∧ (a ≤ 2) :=
by
  sorry

end range_of_a_l655_655878


namespace six_arts_probability_l655_655450

theorem six_arts_probability :
  let n := 720 in
  let m := 120 in
    m / n = 1 / 6 :=
by
  let n := 720
  let m := 120
  show m / n = 1 / 6
  sorry

end six_arts_probability_l655_655450


namespace represents_26_as_sum_of_naturals_rec_sum_eq_one_l655_655067

noncomputable def sum_reciprocals_equals_one : Prop :=
  ∃ (a b c d e : ℕ), a + b + c + d + e = 26 ∧ (1 / (a : ℝ) + 1 / (b : ℝ) + 1 / (c : ℝ) + 1 / (d : ℝ) + 1 / (e : ℝ) = 1)

theorem represents_26_as_sum_of_naturals_rec_sum_eq_one :
  sum_reciprocals_equals_one :=
by
  use [6, 6, 6, 4, 4]
  norm_num
  norm_num
  sorry

end represents_26_as_sum_of_naturals_rec_sum_eq_one_l655_655067


namespace common_points_count_l655_655359

def f (x : ℝ) : ℝ := 2 * log x - x
def g (t x : ℝ) : ℝ := - (1 / 2) * t * x^2 + 2 * t * x

theorem common_points_count (t : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = g t x₁ ∧ f x₂ = g t x₂) ↔ t < log 2 - 1 :=
sorry

end common_points_count_l655_655359


namespace harry_water_per_mile_l655_655211

noncomputable def water_per_mile_during_first_3_miles (initial_water : ℝ) (remaining_water : ℝ) (leak_rate : ℝ) (hike_time : ℝ) (water_drunk_last_mile : ℝ) (first_3_miles : ℝ) : ℝ :=
  let total_leak := leak_rate * hike_time
  let remaining_before_last_mile := remaining_water + water_drunk_last_mile
  let start_last_mile := remaining_before_last_mile + total_leak
  let water_before_leak := initial_water - total_leak
  let water_drunk_first_3_miles := water_before_leak - start_last_mile
  water_drunk_first_3_miles / first_3_miles

theorem harry_water_per_mile :
  water_per_mile_during_first_3_miles 10 2 1 2 3 3 = 1 / 3 :=
by
  have initial_water := 10
  have remaining_water := 2
  have leak_rate := 1
  have hike_time := 2
  have water_drunk_last_mile := 3
  have first_3_miles := 3
  let total_leak := leak_rate * hike_time
  let remaining_before_last_mile := remaining_water + water_drunk_last_mile
  let start_last_mile := remaining_before_last_mile + total_leak
  let water_before_leak := initial_water - total_leak
  let water_drunk_first_3_miles := water_before_leak - start_last_mile
  let result := water_drunk_first_3_miles / first_3_miles
  exact sorry

end harry_water_per_mile_l655_655211


namespace smallest_square_area_with_five_lattice_points_l655_655317

theorem smallest_square_area_with_five_lattice_points : 
    ∃ (s : ℝ), s^2 = 32 ∧
                (∃ (c : ℤ × ℤ), ∀ v : ℤ × ℤ,
                 v = (c.1 + ⌊s/√2⌋, c.2 + ⌊s/√2⌋) ∨ 
                 v = (c.1 - ⌊s/√2⌋, c.2 + ⌊s/√2⌋) ∨  
                 v = (c.1 + ⌊s/√2⌋, c.2 - ⌊s/√2⌋) ∨ 
                 v = (c.1 - ⌊s/√2⌋, c.2 - ⌊s/√2⌋)) → 
                (∃ p : ℤ × ℤ, p ∈ boundary_of_square s (c)) := 
sorry

end smallest_square_area_with_five_lattice_points_l655_655317


namespace intersection_M_N_l655_655412

noncomputable def M : Set ℕ := { x | 0 < x ∧ x < 8 }
def N : Set ℕ := { x | ∃ n : ℕ, x = 2 * n + 1 }
def K : Set ℕ := { 1, 3, 5, 7 }

theorem intersection_M_N : M ∩ N = K :=
by sorry

end intersection_M_N_l655_655412


namespace sea_turtle_vs_whale_l655_655659

def convert_base8_to_decimal(n : Nat) : Nat :=
  let d0 := n % 10
  let n1 := n / 10
  let d1 := n1 % 10
  let n2 := n1 / 10
  let d2 := n2 % 10
  d0 + d1 * 8 + d2 * (8^2)

theorem sea_turtle_vs_whale :
  let a := 724
  let b := 560
  let t := convert_base8_to_decimal(a)
  let w := convert_base8_to_decimal(b)
  let d := t - w
  d = 100 := by
  sorry

end sea_turtle_vs_whale_l655_655659


namespace game_winnable_iff_l655_655470

-- Definitions based on conditions
def cards (n : ℕ) := {i : ℕ // 1 ≤ i ∧ i ≤ n}
noncomputable def strategy_winnable (n k : ℕ) (h1 : 2 ≤ k) (h2 : k < n) : Prop :=
  ∃ m : ℕ, ∃ s : (fin m → fin (2 * n)), ∀ (perm : fin (2 * n) → fin (2 * n)),
    (∀ x : fin (2 * n), x < k → perm x < k) →
    ∃ i j : fin k, i ≠ j ∧ s i = s j

-- Theorem statement: The game is winnable if and only if 2 ≤ k < n
theorem game_winnable_iff (n k : ℕ) (h : 2 ≤ n) : ((2 ≤ k ∧ k < n) ↔ strategy_winnable n k) :=
sorry

end game_winnable_iff_l655_655470


namespace part_I_1_part_I_2_part_II_1_part_II_2_i_part_II_2_ii_l655_655618

-- Part (I)(1)
def j (a x : ℝ) : ℝ := a / (x + 1)
def f (a x : ℝ) : ℝ := Real.log x + j a x

theorem part_I_1 :
  ∀ (a : ℝ), a = 9 / 2 →
    (∀ x, f a x > Real.log x → (x ∈ (Ioo 0 (1 / 2) ∪ Ioi 2))) := by sorry

-- Part (I)(2)
def g (a x : ℝ) : ℝ := abs (Real.log x) + j a x

theorem part_I_2 :
  ∀ (a : ℝ), (∀ (x1 x2 : ℝ), x1 ∈ Ioc 0 2 → x2 ∈ Ioc 0 2 → x1 ≠ x2 →
    (g a x2 - g a x1) / (x2 - x1) < -1) → a ≥ 27 / 2 := by sorry

-- Part (II)(1)
def f' (a x : ℝ) : ℝ := -a * x + Real.log x + (1 - a) / x - 1

theorem part_II_1 :
  ∀ (a : ℝ), a ≥ 0 →
    (∀ x, f' a x > 0 ↔ (x ∈ Ioi 1 - Real.log 1 ∪ Ioi (1 / a - 1))) := by sorry

-- Part (II)(2)(i)
def g2 (b x : ℝ) : ℝ := x^2 - 2 * b * x + 4

theorem part_II_2_i :
  ∀ (b : ℝ), (∀ x1 ∈ Ioo 0 2, ∃ x2 ∈ Icc 1 2, f 1/4 x1 ≥ g2 b x2) → b ≥ 17 / 8 := by sorry

-- Part (II)(2)(ii)
theorem part_II_2_ii :
  ∀ (l : ℝ), (∀ (x1 x2 : ℝ), x1 ∈ Ioc 1 2 → x2 ∈ Ioc 1 2 →
    abs (f 1/4 x1 - f 1/4 x2) ≤ l * abs (1 / x1 - 1 / x2)) → l ≥ 1 / 4 := by sorry

end part_I_1_part_I_2_part_II_1_part_II_2_i_part_II_2_ii_l655_655618


namespace a_99_value_l655_655361

def f1(x : ℝ) := 2 / (x + 1)
def f (n : ℕ) : (ℝ → ℝ) :=
  match n with
  | 0 => id
  | n+1 => f1 ∘ f n

def a (n : ℕ) := (f n 2 - 1) / (f n 2 + 2)

theorem a_99_value : a 99 = - (1 / (2 ^ 101)) := 
by
  sorry

end a_99_value_l655_655361


namespace probability_of_x_lt_2y_l655_655254

noncomputable def point_probability : ℝ :=
  let rectangle_vertices : List (ℝ × ℝ) := [(0, 0), (5, 0), (5, 2), (0, 2)]
  let rectangle_area := 5 * 2
  let triangle_area := (1 / 2) * 4 * 2
  in triangle_area / rectangle_area

theorem probability_of_x_lt_2y : point_probability = 2 / 5 :=
  by
    -- (proof would be here but is not needed as per instructions)
    sorry

end probability_of_x_lt_2y_l655_655254


namespace ratio_of_p_q_l655_655967

theorem ratio_of_p_q (b : ℝ) (p q : ℝ) (h1 : p = -b / 8) (h2 : q = -b / 12) : p / q = 3 / 2 := 
by
  sorry

end ratio_of_p_q_l655_655967


namespace total_number_of_roads_l655_655460

theorem total_number_of_roads (n a r : ℕ) (h1 : ∀ e, exists C, e ∈ C ∧ (∀ v ∈ C, even (degree v))) :
  (∑ e in (road_set n a), count_in_systems e r) = (a * r) / 2 := 
sorry

end total_number_of_roads_l655_655460


namespace boat_distance_along_stream_in_one_hour_l655_655045

theorem boat_distance_along_stream_in_one_hour :
  ∀ (v_b v_s d_up t : ℝ),
  v_b = 7 →
  d_up = 3 →
  t = 1 →
  (t * (v_b - v_s) = d_up) →
  t * (v_b + v_s) = 11 :=
by
  intros v_b v_s d_up t Hv_b Hd_up Ht Hup
  sorry

end boat_distance_along_stream_in_one_hour_l655_655045


namespace concave_sequence_inequality_l655_655130

noncomputable def concave_sequence_bound : ℝ :=
  (1389 * 1388) / (2 * (2 * 1389 + 1))

theorem concave_sequence_inequality (a : Fin 1390 → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ a i)
  (h_concave : ∀ i : Fin 1388, i.val < 1388 → a (i + 1) ≥ (a i + a (i + 2)) / 2) :
  ∑ i, i.val * (a i)^2 ≥ concave_sequence_bound * ∑ i, (a i)^2 :=
by
  sorry

end concave_sequence_inequality_l655_655130


namespace max_value_of_expression_l655_655103

-- Define the variables and condition.
variable (x y z : ℝ)
variable (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1)

-- State the theorem.
theorem max_value_of_expression :
  (8 * x + 5 * y + 15 * z) ≤ 4.54 :=
sorry

end max_value_of_expression_l655_655103


namespace latest_time_with_digits_2_0_2_2_l655_655697

def is_latest_time_with_digits (hour minute second : ℕ) : Prop :=
  (hour = 23) ∧ (minute = 50) ∧ (second = 22)

theorem latest_time_with_digits_2_0_2_2 :
  ∃ (hour minute second : ℕ),
    is_latest_time_with_digits hour minute second :=
by
  use 23, 50, 22
  repeat { exact rfl }

end latest_time_with_digits_2_0_2_2_l655_655697


namespace side_length_of_square_l655_655791

theorem side_length_of_square (P : ℝ) (h1 : P = 12 / 25) : 
  P / 4 = 0.12 := 
by
  sorry

end side_length_of_square_l655_655791


namespace length_of_stripe_l655_655229

-- Define the conditions given in the problem as Lean definitions
def circumference_of_base (base: ℝ) : Prop := base = 18
def height_of_can (height: ℝ) : Prop := height = 8
def stripes_wind (wind: ℕ) : Prop := wind = 2

-- Define the Pythagorean theorem in Lean
theorem length_of_stripe (base: ℝ) (height: ℝ) (wind: ℕ) 
  [circumference_of_base base] 
  [height_of_can height] 
  [stripes_wind wind] : 
  (wind * base)^2 + height^2 = 1360 := 
  by { sorry }

end length_of_stripe_l655_655229


namespace four_couples_arrangement_l655_655699

def seating_arrangements := 6144

theorem four_couples_arrangement :
  ∀ (E H W1 H1 W2 H2 W3 H3 : Type), 
  -- Emily (E), Her Husband (H), Wife1 (W1), Husband1 (H1), Wife2 (W2), Husband2 (H2), 
  -- Wife3 (W3), Husband3 (H3)
  (rotations E H W1 H1 W2 H2 W3 H3 = arrangements) 
  ∧ (reflections E H W1 H1 W2 H2 W3 H3 = arrangements)
  → seating_arrangements = 6144 
  := by
  sorry

end four_couples_arrangement_l655_655699


namespace smallest_positive_number_is_correct_l655_655018

noncomputable def smallest_positive_number : ℝ := 20 - 5 * Real.sqrt 15

theorem smallest_positive_number_is_correct :
  ∀ n,
    (n = 12 - 3 * Real.sqrt 12 ∨ n = 3 * Real.sqrt 12 - 11 ∨ n = 20 - 5 * Real.sqrt 15 ∨ n = 55 - 11 * Real.sqrt 30 ∨ n = 11 * Real.sqrt 30 - 55) →
    n > 0 → smallest_positive_number ≤ n :=
by
  sorry

end smallest_positive_number_is_correct_l655_655018


namespace find_m_l655_655413

def vector := { x : ℝ // true } × { y : ℝ // true }

def a : vector :=
  (⟨-1, trivial⟩, ⟨2, trivial⟩)

def b (m : ℝ) : vector :=
  (⟨m, trivial⟩, ⟨1, trivial⟩)

def add_vectors (v1 v2 : vector) : vector :=
  (⟨v1.1.1 + v2.1.1, trivial⟩, ⟨v1.2.1 + v2.2.1, trivial⟩)

def dot_product (v1 v2 : vector) : ℝ :=
  (v1.1.1 * v2.1.1) + (v1.2.1 * v2.2.1)

theorem find_m (m : ℝ) (h : dot_product (add_vectors a (b m)) a = 0) : m = 7 :=
by
  sorry

end find_m_l655_655413


namespace perp_condition_line_eq_condition_l655_655403

variable (m k : ℝ)
variable (x y : ℝ)

-- Definition for lines l1 and l2
def line1 (x y : ℝ) := (m + 2) * x + m * y - 6 = 0
def line2 (x y : ℝ) := m * x + y - 3 = 0

-- Part 1: Perpendicular condition
theorem perp_condition (hm : m * m + m * 2 = 0) :
    m = 0 ∨ m = -3 :=
sorry

-- Part 2: Point P and intercepts condition 
def point_P (x y : ℝ) := x = 1 ∧ y = 2 * m

def line_l (x y : ℝ) := y - 2 = k * (x - 1)

-- Line intercept conditions
def intercepts (x_intercept y_intercept : ℝ) :=
  x_intercept = 0 ∧ y_intercept = 2 - k ∧ (x = (k - 2) / k) ∧ y = 0 ∧
  (x_intercept = - y_intercept)

theorem line_eq_condition (hP : point_P 1 (2 * m)) (hIntercept : intercepts (- k) (k - 2)) :
    (line_l x y = 0) ∧ (x - y + 1 = 0) ∨ (2 * x - y = 0) :=
sorry

end perp_condition_line_eq_condition_l655_655403


namespace exists_two_positive_integers_dividing_3003_l655_655351

theorem exists_two_positive_integers_dividing_3003 : 
  ∃ (m1 m2 : ℕ), m1 > 0 ∧ m2 > 0 ∧ m1 ≠ m2 ∧ (3003 % (m1^2 + 2) = 0) ∧ (3003 % (m2^2 + 2) = 0) :=
by
  sorry

end exists_two_positive_integers_dividing_3003_l655_655351


namespace common_tangent_exist_l655_655295

theorem common_tangent_exist (A B C D : Point)
  (semicircle : Circle) (K1 K2 K3 : Circle)
  (on_semicircle : C ∈ semicircle)
  (between_AB : A ≠ C ∧ B ≠ C)
  (perpendicular_to_AB : ⟂ (C - D) AB)
  (K1_incircle_ABC : incircle K1 (triangle A B C))
  (K2_condition : touches_at K2 CD ∧ touches_at K2 DA ∧ touches_at K2 semicircle)
  (K3_condition : touches_at K3 CD ∧ touches_at K3 DB ∧ touches_at K3 semicircle) :
  common_tangent_exists K1 K2 K3 ∧ ¬ common_tangent_exists K1 K2 K3 AB := 
sorry

end common_tangent_exist_l655_655295


namespace fraction_of_students_l655_655040

variable (G B T : ℕ)
variable (F : ℚ)

-- Conditions
def ratio_condition : Prop := B = 1.5 * G
def total_students_condition : Prop := T = B + G
def fraction_condition : Prop := (1 / 2 : ℚ) * G = F * T

-- Theorem statement
theorem fraction_of_students (h1 : ratio_condition G B)
                            (h2 : total_students_condition G B T)
                            (h3 : fraction_condition G T F) :
  F = (1 / 5 : ℚ) := sorry

end fraction_of_students_l655_655040


namespace total_revenue_increased_by_1_5_l655_655232

-- Define the initial conditions
variables (P U : ℝ) (h_price_decrease : P' = 0.75 * P) 
          (h_increase_ratio : ∀ (x y : ℝ), x / y = 4) 
          (h_units_increase : U' = 2 * U)

-- Definition of both revenues before and after the changes
def revenue_before := P * U
def revenue_after := 0.75 * P * 2 * U

-- Required Statement
theorem total_revenue_increased_by_1_5 :
    revenue_after = 1.5 * revenue_before :=
sorry

end total_revenue_increased_by_1_5_l655_655232


namespace find_a_l655_655331

theorem find_a (a : ℝ) (m n : ℤ)
  (h1 : a + 2 / 3 = m)
  (h2 : 1 / a - 3 / 4 = n) :
  a = 4 / 3 :=
begin
  sorry
end

end find_a_l655_655331


namespace largest_prime_factor_of_6_factorial_plus_7_factorial_l655_655590

theorem largest_prime_factor_of_6_factorial_plus_7_factorial :
  ∃ p : ℕ, Nat.Prime p ∧ p = 5 ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ (6! + 7!) → q ≤ 5 :=
by
  sorry

end largest_prime_factor_of_6_factorial_plus_7_factorial_l655_655590


namespace see_segment_at_angle_l655_655180

-- Define the basic elements of the problem: given line MN, segment AB, and angle θ
variables {Point Line Angle : Type}
variables (MN : Line) (A B : Point) (θ : Angle)

-- The main theorem stating the equivalence
theorem see_segment_at_angle (P : Point) (intersect1 intersect2 : Point) :
  (are_points_intersection MN (circumcircle (A, B, P)) ∧ are_points_intersection MN (circumcircle (A, B, reflect P))) →
  sees_at_angle P A B θ ↔ P = intersect1 ∨ P = intersect2 :=
sorry

end see_segment_at_angle_l655_655180


namespace john_lift_total_weight_l655_655079

-- Define the conditions as constants
def initial_weight : ℝ := 135
def weight_increase : ℝ := 265
def bracer_factor : ℝ := 6

-- Define a theorem to prove the total weight John can lift
theorem john_lift_total_weight : initial_weight + weight_increase + (initial_weight + weight_increase) * bracer_factor = 2800 := by
  -- proof here
  sorry

end john_lift_total_weight_l655_655079


namespace solve_for_x_l655_655420

theorem solve_for_x (x : ℝ) (h1 : 3 * x^2 - 5 * x = 0) (h2 : x ≠ 0) : x = 5 / 3 :=
by
  sorry

end solve_for_x_l655_655420


namespace bottle_caps_difference_l655_655686

theorem bottle_caps_difference :
  ∀ (found thrown_away : ℕ), 
  found = 50 → 
  thrown_away = 6 → 
  (found - thrown_away) = 44 :=
by
  intros found thrown_away found_eq thrown_eq
  rw [found_eq, thrown_eq]
  -- Proof not required
  sorry

end bottle_caps_difference_l655_655686


namespace fibonacci_matrix_identity_fibonacci_determinant_identity_l655_655068

theorem fibonacci_matrix_identity (n : ℕ) (hn : 0 < n) :
  let F : ℕ → ℕ := sorry in
  let matrix_pow := sorry in
  matrix_pow 
  (\begin
    pmatrix
    1 & 1 
    1 & 0
  end)^n 
  = 
  \begin
    pmatrix
    F (n + 1) & F n 
    F n & F (n - 1) 
  end := sorry

theorem fibonacci_determinant_identity ():
  let F : ℕ → ℕ := sorry in
  F (1002) * F (1004) - F (1003) ^ 2 = -1 := sorry

end fibonacci_matrix_identity_fibonacci_determinant_identity_l655_655068


namespace seating_arrangements_count_l655_655656

open Finset

-- Let A, B, C, D, and E represent Alice, Bob, Carla, Derek, and Eric respectively.
inductive Person
| A | B | C | D | E 
open Person

-- Function to check if a seating arrangement meets the constraints
def valid_seating (seating : Vector Person 5) : Prop :=
  (¬ (seating.anyp (= A 
   ∧ (seating.get! 0 = B ∨ seating.get! 1 = B ∨
       seating.get! 2 = D ∨ seating.get! 3 = D ∨ seating.get! 4 = D))))
  ∧ (¬ (seating.anyp (= C ∧ seating.anyp (= D ∧ (seating.get! 0 = D ∨ seating.get! 4 = D)))))

noncomputable def seating_arrangements : Finset (Vector Person 5) :=
  filter valid_seating (univ : Finset (Vector Person 5))

theorem seating_arrangements_count : seating_arrangements.card = 15 :=
sorry

end seating_arrangements_count_l655_655656


namespace limit_S_n_over_n_squared_l655_655531

noncomputable def S (n : ℕ) : ℝ := (n * a₁ + (n * (n - 1) / 2) * d)

theorem limit_S_n_over_n_squared
  (a₁ d : ℝ)
  (h : (S 3) / 3 = (S 2) / 2 + 5) :
  (tendsto (λ n, (S n) / (n * n)) at_top (𝓝 5)) :=
begin
  sorry,
end

end limit_S_n_over_n_squared_l655_655531


namespace triangle_area_with_perpendicular_medians_l655_655836

theorem triangle_area_with_perpendicular_medians
  (P Q R M N S : Type) [linear_ordered_semiring S]
  (PM QN : P → S) (medians_perpendicular : PM ⟂ QN) 
  (PM_eq_18 : PM = 18) (QN_eq_24 : QN = 24) :
  area (triangle P Q R) = 288 :=
sorry

end triangle_area_with_perpendicular_medians_l655_655836


namespace percentage_of_tulips_is_55_l655_655241

-- Define the parameters and necessary conditions
variables (F : ℕ) 
def pink_flowers := (7 / 12 : ℚ) * F
def pink_daisies := (1 / 2) * pink_flowers
def pink_tulips := (1 / 2) * pink_flowers
def red_flowers := F - pink_flowers
def red_tulips := (6 / 10 : ℚ) * red_flowers
def total_tulips := pink_tulips + red_tulips

-- Define the target theorem
theorem percentage_of_tulips_is_55 :
  ((total_tulips / F) * 100).round = (55 : ℚ) :=
by
  sorry

end percentage_of_tulips_is_55_l655_655241


namespace sin_value_l655_655749

variable (x a : ℝ)

theorem sin_value 
(h1 : x ∈ Ioo (-π/2) 0) 
(h2 : cos (2 * x) = a) : 
  sin x = -real.sqrt ((1 - a) / 2) := 
sorry

end sin_value_l655_655749


namespace three_digit_non_multiples_of_3_or_11_l655_655780

theorem three_digit_non_multiples_of_3_or_11 : 
  ∃ (n : ℕ), n = 546 ∧ 
  (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → 
    ¬ (x % 3 = 0 ∨ x % 11 = 0) → 
    n = (900 - (300 + 81 - 27))) := 
by 
  sorry

end three_digit_non_multiples_of_3_or_11_l655_655780


namespace doughnuts_left_l655_655323

/-- 
  During a staff meeting, 50 doughnuts were served. If each of the 19 staff ate 2 doughnuts,
  prove that there are 12 doughnuts left. 
-/
theorem doughnuts_left (total_doughnuts : ℕ) (staff_count : ℕ) (doughnuts_per_staff : ℕ) :
  total_doughnuts = 50 → staff_count = 19 → doughnuts_per_staff = 2 →
  total_doughnuts - (staff_count * doughnuts_per_staff) = 12 :=
by
  intros h_total h_staff h_per_staff
  rw [h_total, h_staff, h_per_staff]
  norm_num
  sorry

end doughnuts_left_l655_655323


namespace particle_speed_l655_655645
-- Import the entire library to ensure all necessary tools are available

-- The position function of the particle as described in the problem
def position (t : ℝ) : ℝ × ℝ := (3 * t + 8, 5 * t - 15)

-- Define the speed calculation
def speed : ℝ :=
  let deltaX := 3
  let deltaY := 5
  Real.sqrt (deltaX ^ 2 + deltaY ^ 2)

-- Statement to prove
theorem particle_speed : speed = Real.sqrt 34 := by
  sorry

end particle_speed_l655_655645


namespace ellipse_range_of_k_l655_655802

theorem ellipse_range_of_k (k : ℝ) :
  (4 - k > 0) → (k - 1 > 0) → (4 - k ≠ k - 1) → (1 < k ∧ k < 4 ∧ k ≠ 5 / 2) :=
by
  intros h1 h2 h3
  sorry

end ellipse_range_of_k_l655_655802


namespace annual_interest_rate_l655_655139

noncomputable def compound_interest (P A : ℝ) (n t r : ℝ) : Prop :=
  A = P * (1 + r / n) ^ (n * t)

theorem annual_interest_rate :
  ∃ r : ℝ, compound_interest 1000 1157.625 2 1.5 r ∧ r ≈ 0.101 :=
by
  sorry

end annual_interest_rate_l655_655139


namespace find_2023rd_letter_l655_655971

def seq : List Char := ['A', 'B', 'C', 'D', 'D', 'C', 'B', 'A']

theorem find_2023rd_letter : seq.get! ((2023 % seq.length) - 1) = 'B' :=
by
  sorry

end find_2023rd_letter_l655_655971


namespace length_segment_BP_l655_655814

theorem length_segment_BP (A B C D P : Type*) [Parallel AB CD] 
  (h_AD : length AD = 52) (h_BC : length BC = 20) (h_CP : length CP = 35) 
  (BP_x : length BP = 20) : True :=
sorry

end length_segment_BP_l655_655814


namespace minimum_value_correct_l655_655712

noncomputable def minimum_value (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) : ℝ :=
  a^3 + b^3 + (1 / (a + b)^3)

theorem minimum_value_correct 
  (a b : ℝ) 
  (h₁ : 0 < a) 
  (h₂ : 0 < b) 
  : ∃ (s : ℝ), 
    minimum_value a b h₁ h₂ = real.sqrt (2 * real.sqrt 2) := 
sorry

end minimum_value_correct_l655_655712


namespace A_gets_half_optimal_play_R1_optimal_play_R2_optimal_play_R3_l655_655225

-- We define the context of the game and the rules
def game (N : Nat) : Type :=
  { piles : List Nat // (∀ n ∈ piles, n ≥ 1) ∧ (∃ a b, a + b = N ∧ a ≥ 2 ∧ b ≥ 2) }

def rule_R1 (piles : List Nat) : List Nat :=
  let sorted := List.sort (· ≤ ·) piles
  [sorted.head!, sorted.getLast!]

def rule_R2 (piles : List Nat) : List Nat :=
  let sorted := List.sort (· ≤ ·) piles
  [sorted.get! 1, sorted.get! 2]

def rule_R3 (piles : List Nat) : List Nat :=
  let sorted := List.sort (· ≤ ·) piles
  if some_condition then [sorted.head!, sorted.getLast!] else [sorted.get! 1, sorted.get! 2]
  -- 'some_condition' determines B's choice criteria based on another strategy or condition

-- Main proof problem

theorem A_gets_half (N : Nat) (hN : N ≥ 4) (g : game N) :
  (N / 2).floor = g.piles.updateNth 0 0 |>.sum
:= by sorry

#check A_gets_half -- Check the theorem definition outline

theorem optimal_play_R1 (N : Nat) (hN : N ≥ 4) (g : game N) :
  let B_piles := rule_R1 g.piles
  (N / 2).floor = g.piles.sum - B_piles.sum
:= by sorry

#check optimal_play_R1

theorem optimal_play_R2 (N : Nat) (hN : N ≥ 4) (g : game N) :
  let B_piles := rule_R2 g.piles
  (N / 2).floor = g.piles.sum - B_piles.sum
:= by sorry

#check optimal_play_R2

theorem optimal_play_R3 (N : Nat) (hN : N ≥ 4) (g : game N) :
  let B_piles := rule_R3 g.piles
  (N / 2).floor = g.piles.sum - B_piles.sum
:= by sorry

#check optimal_play_R3

end A_gets_half_optimal_play_R1_optimal_play_R2_optimal_play_R3_l655_655225


namespace marketing_strategy_increases_sales_of_B_l655_655455

-- Definitions of the products with their characteristics
structure Product where
  quality : ℤ
  price : ℤ
  likelihood_of_purchase : ℤ

-- Instances for each product
def product_A : Product := {
  quality := 10,    -- Placeholder value for high quality
  price := 100,    -- Placeholder value for high price
  likelihood_of_purchase := 1 -- Placeholder value for low likelihood of purchase
}

def product_B : Product := {
  quality := 8,    -- Placeholder value for slightly inferior quality
  price := 60,     -- Placeholder value for significantly lower price
  likelihood_of_purchase := 0 -- Placeholder to be updated based on conditions
}

def product_C : Product := {
  quality := 5,    -- Placeholder value for economy quality
  price := 30,     -- Placeholder value for economy price
  likelihood_of_purchase := 0 -- Placeholder, not relevant for this proof
}

-- Hypothesis based on the problem statement
axiom H1 : product_A.likelihood_of_purchase < product_B.likelihood_of_purchase
axiom H2 : product_B.price < product_A.price
axiom H3 : product_C.price < product_B.price
axiom H4 : product_B.price > average_market_price

-- The theorem that represents the proof problem
theorem marketing_strategy_increases_sales_of_B :
  product_A.likelihood_of_purchase < product_B.likelihood_of_purchase ∧
  (product_C.price < product_B.price ∨ product_B.price > average_market_price) →
  product_B.likelihood_of_purchase > product_A.likelihood_of_purchase :=
by sorry

end marketing_strategy_increases_sales_of_B_l655_655455


namespace problem_statement_l655_655375

theorem problem_statement (M : ℕ) 
  (h : (1 / 3.factorial * 18.factorial + 
        1 / 4.factorial * 17.factorial + 
        1 / 5.factorial * 16.factorial + 
        1 / 6.factorial * 15.factorial + 
        1 / 7.factorial * 14.factorial + 
        1 / 8.factorial * 13.factorial + 
        1 / 9.factorial * 12.factorial + 
        1 / 10.factorial * 11.factorial = M / 2.factorial * 19.factorial)) : 
  float.floor (M / 100.0) = 498 :=
sorry

end problem_statement_l655_655375


namespace otimes_identity_l655_655313

def otimes (x y : ℝ) : ℝ := x^2 - y^2

theorem otimes_identity (h : ℝ) : otimes h (otimes h h) = h^2 :=
by
  sorry

end otimes_identity_l655_655313


namespace derivative_of_f_l655_655429

-- Define the function
def f (x : ℝ) : ℝ := 2 * x + Real.cos x

-- State the theorem: The derivative of f is 2 - sin x
theorem derivative_of_f : ∀ x : ℝ, deriv f x = 2 - Real.sin x := 
by
  sorry

end derivative_of_f_l655_655429


namespace find_p_l655_655194

noncomputable def vector_on_line (a : ℝ) : ℝ × ℝ :=
  ⟨a, (3 / 2) * a + 5⟩

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * w.1 + v.2 * w.2
  let norm_square := w.1 * w.1 + w.2 * w.2
  let factor := dot_product / norm_square
  in ⟨factor * w.1, factor * w.2⟩

theorem find_p (w : ℝ × ℝ) (hw : w = ⟨-3 / 2 * w.2, w.2⟩) :
  projection (vector_on_line 1) w = ⟨-30 / 13, 20 / 13⟩ :=
sorry

end find_p_l655_655194


namespace jed_change_l655_655841

theorem jed_change :
  ∀ (num_games : ℕ) (cost_per_game : ℕ) (payment : ℕ) (bill_value : ℕ),
  num_games = 6 →
  cost_per_game = 15 →
  payment = 100 →
  bill_value = 5 →
  (payment - num_games * cost_per_game) / bill_value = 2 :=
by
  intros num_games cost_per_game payment bill_value
  sorry

end jed_change_l655_655841


namespace machine_rate_ratio_l655_655502

theorem machine_rate_ratio (A B : ℕ) (h1 : ∃ A : ℕ, 8 * A = 8 * A)
  (h2 : ∃ W : ℕ, W = 8 * A)
  (h3 : ∃ W1 : ℕ, W1 = 6 * A)
  (h4 : ∃ W2 : ℕ, W2 = 2 * A)
  (h5 : ∃ B : ℕ, 8 * B = 2 * A) :
  (B:ℚ) / (A:ℚ) = 1 / 4 :=
by sorry

end machine_rate_ratio_l655_655502


namespace mark_not_inviting_five_l655_655326

def Classmates : Type := Fin 25

def Friends (M : Classmates) (x y : Classmates) : Prop := sorry -- friendship relation

def Invited (M : Classmates) (x : Classmates) : Prop :=
  Friends M M x ∨ ∃ y : Classmates, Friends M M y ∧ Friends M y x ∨
  ∃ y z : Classmates, Friends M M y ∧ Friends M y z ∧ Friends M z x

noncomputable def count_non_invited (M : Classmates) : ℕ :=
  (Finset.univ.filter (λ x, ¬ Invited M x)).card

theorem mark_not_inviting_five (M : Classmates) :
  count_non_invited M = 5 :=
begin
  -- With the given conditions of isolation
  sorry
end

end mark_not_inviting_five_l655_655326


namespace transform_sets_l655_655570

theorem transform_sets :
  ∀ (S1 S2 : List ℤ), 
  (∀ x ∈ S1, x = 1 ∨ x = -1) → 
  (∀ x ∈ S2, x = 1 ∨ x = -1) → 
  S1.length = 1958 → 
  S2.length = 1958 → 
  (∃ n : ℕ, transformable S1 S2 n) :=
sorry

-- Definition of the transformation operation in terms of allowed steps
def transformable (S1 S2 : List ℤ) (n : ℕ) : Prop :=
  ∃ steps : List (Fin 1958 → List ℤ → List ℤ),
    List.length steps = n ∧
    steps.apply_all S1 = S2

-- Definition of the application of all steps
def List.apply_all {α : Type} (steps : List (α → α)) (v : α) : α :=
  steps.foldl (λ x f => f x) v

end transform_sets_l655_655570


namespace bullet_count_in_first_120_circles_l655_655261

theorem bullet_count_in_first_120_circles : 
  (count_bullets_in_sequence (generate_circle_sequence 120) = 14) := 
sorry

def generate_circle_sequence : ℕ → list char :=
/- Generates the circle sequence up to n terms -/
sorry

def count_bullets_in_sequence : list char → ℕ :=
/- Counts the number of bullets (●) in the given sequence -/
sorry

end bullet_count_in_first_120_circles_l655_655261


namespace domain_of_g_l655_655979

noncomputable def g (x : ℝ) : ℝ := log 3 (log 4 (log 5 (log 6 x)))

theorem domain_of_g : {x : ℝ | x > 7776} = {x : ℝ | real.logb 3 (real.logb 4 (real.logb 5 (real.logb 6 x)))} :=
sorry

end domain_of_g_l655_655979


namespace house_spirits_elevator_l655_655812

-- Define the given conditions
def first_floor_domovoi := 1
def middle_floor_domovoi := 2
def last_floor_domovoi := 1
def total_floors := 7
def spirits_per_cycle := first_floor_domovoi + 5 * middle_floor_domovoi + last_floor_domovoi

-- Prove the statement
theorem house_spirits_elevator (n : ℕ) (floor : ℕ) (h1 : total_floors = 7) (h2 : spirits_per_cycle = 12) (h3 : n = 1000) :
  floor = 4 :=
by
  sorry

end house_spirits_elevator_l655_655812


namespace soccer_balls_added_l655_655575

theorem soccer_balls_added :
  ∀ (initial removed final added : ℕ),
    initial = 6 →
    removed = 3 →
    final = 24 →
    added = final - (initial - removed) →
    added = 21 := 
by
  intros initial removed final added h_initial h_removed h_final h_added
  rw [h_added, h_initial, h_removed, h_final]
  norm_num

end soccer_balls_added_l655_655575


namespace radius_of_tangent_circle_l655_655632

theorem radius_of_tangent_circle :
  ∃ r : ℝ, abs (r - 2.37) < 0.01 ∧
  (∀ (x y : ℝ), x = r → y = r →
  ∃ k b : ℝ, k = sqrt 3 ∧ b = -sqrt 3 ∧
  (∀ (x : ℝ), y = k * x + b → (abs ((sqrt 3 * r + r - sqrt 3) / sqrt (3 + 1) - r) = 0)) ∧ dist (r, r) (r / sqrt 3 + r - sqrt 3) = r) :=
begin
  sorry
end

end radius_of_tangent_circle_l655_655632


namespace wheel_center_travel_distance_l655_655655

theorem wheel_center_travel_distance (r : ℝ) (h : r = 1) : 
  let C := 2 * Real.pi * r in
  C = 2 * Real.pi :=
by
  intros
  exact_mod_cast h
  sorry

end wheel_center_travel_distance_l655_655655


namespace max_value_f_l655_655873

theorem max_value_f (x y z u v : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0) (hv : v > 0) :
  let f := (x * y + y * z + z * u + u * v) / (2 * x^2 + y^2 + 2 * z^2 + u^2 + 2 * v^2) in
  f ≤ 1 / 2 :=
by
  sorry

end max_value_f_l655_655873


namespace joao_claudia_grades_l655_655081

variable (star : ℕ)

theorem joao_claudia_grades (h1 : João's grade = star) 
                            (h2 : Cláudia's grade = star + 13) 
                            (h3 : star + (star + 13) = 149) :
  (João's grade = 68) ∧ (Cláudia's grade = 81) :=
by
  sorry

end joao_claudia_grades_l655_655081


namespace number_of_combinations_l655_655790

def is_multiple_of_12 (n : ℕ) : Prop :=
  (2 ∣ n) ∧ (3 ∣ n)

def list_of_numbers : List ℕ := [1, 5, 8, 21, 22, 27, 30, 33, 37, 39, 46, 50]

def valid_triplets (l : List ℕ) : List (ℕ × ℕ × ℕ) :=
  l.toFinset.subsetsOfCard 3 |>.toList.map (fun s => match s.toList with
    | [a, b, c] => (a, b, c)
    | _ => (0, 0, 0))

def count_valid_triplets (l : List ℕ) : ℕ :=
  valid_triplets l |>.count (fun (a, b, c) => a < b ∧ b < c ∧ is_multiple_of_12 (a * b * c))

theorem number_of_combinations : count_valid_triplets list_of_numbers = 76 :=
by
  sorry

end number_of_combinations_l655_655790


namespace value_of_expression_l655_655355

theorem value_of_expression (m : ℝ) (h : 1 / (m - 2) = 1) : (2 / (m - 2)) - m + 2 = 1 :=
sorry

end value_of_expression_l655_655355


namespace peanuts_in_box_l655_655804

theorem peanuts_in_box (p₀ : ℕ) (p_add : ℕ) : p₀ = 4 → p_add = 12 → p₀ + p_add = 16 :=
by
  intros h₀ h₁
  rw [h₀, h₁]
  norm_num

end peanuts_in_box_l655_655804


namespace inversion_base_to_arc_of_circumcircle_l655_655514

theorem inversion_base_to_arc_of_circumcircle
  (A B C : Point)
  (h_iso : dist A B = dist A C)
  (P : Point → Point := λ P, reflect_point A (dist A B^2 / (dist A P^2))) :
  maps_to_circumcircle P A B C BC (arc A B C) := by
  sorry

end inversion_base_to_arc_of_circumcircle_l655_655514


namespace nearest_integer_x_sub_y_l655_655027

theorem nearest_integer_x_sub_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : |x| + y = 5) (h2 : |x| * y + x^2 = 5) :
  int.nearest (x - y) = -3 :=
sorry

end nearest_integer_x_sub_y_l655_655027


namespace sum_of_real_solutions_eq_zero_l655_655720

noncomputable def geometric_series (y : ℝ) : ℝ :=
  if |y| < 1 then 2 - y^2 + y^4 - y^6 + y^8 - y^10 + ∑' (n : ℕ), (-1)^n * y^(2 * n) else 0

theorem sum_of_real_solutions_eq_zero : 
  ∑ y in {y : ℝ | geometric_series y = y}, y = 0 :=
by 
  sorry

end sum_of_real_solutions_eq_zero_l655_655720


namespace max_value_exponential_and_power_functions_l655_655376

variable (a b : ℝ)

-- Given conditions
axiom condition : 0 < b ∧ b < a ∧ a < 1

-- Problem statement
theorem max_value_exponential_and_power_functions : 
  a^b = max (max (a^b) (b^a)) (max (a^a) (b^b)) :=
by
  sorry

end max_value_exponential_and_power_functions_l655_655376


namespace sum_of_all_valid_primes_l655_655223

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_substring (n m : ℕ) : Prop :=
  ∃ (a b : ℕ), m = a * 10 ^ b + n

def all_substrings_prime (p : ℕ) : Prop :=
  ∀ q : ℕ, is_substring q p → is_prime q

def sum_valid_primes : ℕ :=
  (∑ p in {2, 3, 5, 7, 23, 37, 53, 73, 373}, p)

theorem sum_of_all_valid_primes : sum_valid_primes = 576 := 
  by sorry

end sum_of_all_valid_primes_l655_655223


namespace MarlySoupBags_l655_655115

theorem MarlySoupBags :
  ∀ (milk chicken_stock vegetables bag_capacity total_soup total_bags : ℚ),
    milk = 6 ∧
    chicken_stock = 3 * milk ∧
    vegetables = 3 ∧
    bag_capacity = 2 ∧
    total_soup = milk + chicken_stock + vegetables ∧
    total_bags = total_soup / bag_capacity ∧
    total_bags.ceil = 14 :=
by
  intros
  sorry

end MarlySoupBags_l655_655115


namespace number_of_integer_points_in_intersection_l655_655681

def sphere1 (x y z : ℤ) : Prop :=
  x^2 + y^2 + (z - 11)^2 ≤ 49

def sphere2 (x y z : ℤ) : Prop :=
  x^2 + y^2 + z^2 ≤ 64

theorem number_of_integer_points_in_intersection : 
  (setOf (λ (p : ℤ × ℤ × ℤ), sphere1 p.1 p.2 p.3 ∧ sphere2 p.1 p.2 p.3)).card = 37 := 
by sorry

end number_of_integer_points_in_intersection_l655_655681


namespace total_time_spent_l655_655884

def timeDrivingToSchool := 20
def timeAtGroceryStore := 15
def timeFillingGas := 5
def timeAtParentTeacherNight := 70
def timeAtCoffeeShop := 30
def timeDrivingHome := timeDrivingToSchool

theorem total_time_spent : 
  timeDrivingToSchool + timeAtGroceryStore + timeFillingGas + timeAtParentTeacherNight + timeAtCoffeeShop + timeDrivingHome = 160 :=
by
  sorry

end total_time_spent_l655_655884


namespace quadratic_equal_roots_l655_655755

theorem quadratic_equal_roots (k : ℝ) :
  (∀ x : ℝ, x^2 + k * x + 1 = 0 → x = -k / 2) ↔ (k = 2 ∨ k = -2) :=
by
  sorry

end quadratic_equal_roots_l655_655755


namespace num_solutions_eq_sin3x_cosx_l655_655773

noncomputable def eq_solutions_sin3x_cosx (x : ℝ) := sin (3 * x) = cos x

theorem num_solutions_eq_sin3x_cosx :
  ∃ (s : Finset ℝ), (∀ x ∈ s, x ∈ Set.Icc 0 (2 * Real.pi) ∧ eq_solutions_sin3x_cosx x) ∧ s.card = 6 := by
  sorry

end num_solutions_eq_sin3x_cosx_l655_655773


namespace value_of_p10_l655_655292

def p (d e f x : ℝ) : ℝ := d * x^2 + e * x + f

theorem value_of_p10 (d e f : ℝ) 
  (h1 : p d e f 3 = p d e f 4)
  (h2 : p d e f 2 = p d e f 5)
  (h3 : p d e f 0 = 2) :
  p d e f 10 = 2 :=
by
  sorry

end value_of_p10_l655_655292


namespace compute_fraction_l655_655863

theorem compute_fraction (x y z : ℝ) (h : x + y + z = 1) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = (1 - (x^2 + y^2 + z^2)) / (2 * (x^2 + y^2 + z^2)) :=
by 
  sorry

end compute_fraction_l655_655863


namespace tourists_originally_in_group_l655_655651

theorem tourists_originally_in_group (x : ℕ) (h₁ : 220 / x - 220 / (x + 1) = 2) : x = 10 := 
by
  sorry

end tourists_originally_in_group_l655_655651


namespace irrational_number_among_given_l655_655658

noncomputable def num1 := 3 * Real.pi
def num2 := 0.333
def num3 := 0
def num4 := Real.sqrt 4

theorem irrational_number_among_given :
  irrational num1 ∧ ¬irrational num2 ∧ ¬irrational num3 ∧ ¬irrational num4 :=
by
  -- Proof goes here
  sorry

end irrational_number_among_given_l655_655658


namespace total_import_value_l655_655059

-- Define the given conditions
def export_value : ℝ := 8.07
def additional_amount : ℝ := 1.11
def factor : ℝ := 1.5

-- Define the import value to be proven
def import_value : ℝ := 46.4

-- Main theorem statement
theorem total_import_value :
  export_value = factor * import_value + additional_amount → import_value = 46.4 :=
by sorry

end total_import_value_l655_655059


namespace problem_solution_l655_655850

noncomputable def A : ℝ := 
 ∑' m : ℕ+, ∑' n : ℕ+, (m^n * n^m) * (Nat.choose (m + n) m) / (↑(m + n) ^ (m + n))

theorem problem_solution :
  ∀ (A : ℝ), 
  A = ∑' m : ℕ+, ∑' n : ℕ+, (m^(n-1) * n^(m-1)) * (Nat.choose (m + n) m) / (↑(m + n) ^ (m + n)) →
  (Real.floor (1000 * A) = 1289) :=
begin
  assume A hA,
  sorry
end

end problem_solution_l655_655850


namespace henry_walks_l655_655010

theorem henry_walks (
  (gym_distance : ℝ) (h_gym_distance : gym_distance = 3)
  (walk_fraction : ℝ) (h_walk_fraction : walk_fraction = 2/3)
  (A B : ℝ)) 
  (h_limit_A : A = gym_distance * walk_fraction) 
  (h_limit_B : B = gym_distance * (1 - walk_fraction)) :
  |A - B| = 1 := by
  sorry

end henry_walks_l655_655010


namespace PZ_perpendicular_BX_l655_655745

-- Definitions
def midpoint (A B : Point) : Point := sorry
def diameter (O : Point) (A B : Point) : Circle := sorry
def intersection (L : Line) (C : Circle) : Point := sorry
def tangent_line (X : Point) (C : Circle) : Line := sorry
def perpendicular (L1 L2 : Line) : Prop := sorry

theorem PZ_perpendicular_BX 
  (O A B X Y Z P : Point) 
  (H1 : O = midpoint A B) 
  (H2 : diameter O A B) 
  (H3 : Z inside half_circle A O B) 
  (H4 : X = intersection (line_through O Z) half_circle)
  (H5 : Y = intersection (line_through A Z) half_circle)
  (H6 : P = intersection (line_through B Y) (tangent_line X half_circle)) : 
  perpendicular (line_through P Z) (line_through B X) := 
  sorry

end PZ_perpendicular_BX_l655_655745


namespace additional_savings_l655_655551

def initial_price : Float := 30
def discount1 : Float := 5
def discount2_percent : Float := 0.25

def price_after_discount1_then_discount2 : Float := 
  (initial_price - discount1) * (1 - discount2_percent)

def price_after_discount2_then_discount1 : Float := 
  initial_price * (1 - discount2_percent) - discount1

theorem additional_savings :
  price_after_discount1_then_discount2 - price_after_discount2_then_discount1 = 1.25 := by
  sorry

end additional_savings_l655_655551


namespace parking_fines_l655_655644

theorem parking_fines (total_citations littering_citations offleash_dog_citations parking_fines : ℕ) 
  (h1 : total_citations = 24) 
  (h2 : littering_citations = 4) 
  (h3 : offleash_dog_citations = 4) 
  (h4 : total_citations = littering_citations + offleash_dog_citations + parking_fines) : 
  parking_fines = 16 := 
by 
  sorry

end parking_fines_l655_655644


namespace part1_part2_l655_655690

def op (a b : ℤ) := 2 * a - 3 * b

theorem part1 : op (-2) 3 = -13 := 
by
  -- Proof omitted
  sorry

theorem part2 (x : ℤ) : 
  let A := op (3 * x - 2) (x + 1)
  let B := op (-3 / 2 * x + 1) (-1 - 2 * x)
  B > A :=
by
  -- Proof omitted
  sorry

end part1_part2_l655_655690


namespace x_y_sum_eq_l655_655063

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D E F : V)
variables {x y : ℝ}

-- Given conditions
def midpoint (E : V) (C D : V) : Prop := E = (C + D) / 2
def af_eq_2fd (A F D : V) : Prop := A - F = 2 * (F - D)
def ef_eq_linear_comb (E F A B C : V) (x y : ℝ) : Prop := E - F = x * (C - A) + y * (B - A)

-- Theorem statement
theorem x_y_sum_eq : midpoint E C D ∧ af_eq_2fd A F D ∧ ef_eq_linear_comb E F A B C x y → x + y = -1/2 :=
sorry

end x_y_sum_eq_l655_655063


namespace number_of_real_solutions_l655_655485

theorem number_of_real_solutions (x : ℝ) : 
  ∃ n, (3 * x^2 - 30 * (⌊x⌋) + 28 = 0) ∧ 
       (∀ m, (m : ℝ) * (3 * m^2 - 30 * (⌊m⌋) + 28 = 0) → m = x) → 
       n = 3 :=
sorry

end number_of_real_solutions_l655_655485


namespace num_elements_begin_with_2_l655_655090

def elem_begin_with_digit (d : Nat) (n : Nat) : Prop :=
  let num_str := toString (3 ^ n)
  num_str.head == Char.ofNat (48 + d)

noncomputable def count_elements_begin_with_2 (n : Nat) : Nat :=
  (List.range (n + 1)).filter (elem_begin_with_digit 2).length

theorem num_elements_begin_with_2 :
  let T := {k | k ∈ {0, ..., 1500}}
  let digits := 716
  (count_elements_begin_with_2 1500) = 784
:= by
  -- proof would go here
  sorry

end num_elements_begin_with_2_l655_655090


namespace domain_of_g_l655_655978

noncomputable def g (x : ℝ) : ℝ := log 3 (log 4 (log 5 (log 6 x)))

theorem domain_of_g : {x : ℝ | x > 7776} = {x : ℝ | real.logb 3 (real.logb 4 (real.logb 5 (real.logb 6 x)))} :=
sorry

end domain_of_g_l655_655978


namespace slope_angle_of_parametric_line_l655_655762

theorem slope_angle_of_parametric_line :
  (∃ t : ℝ, ∀ x y : ℝ, (x = 1 + 2 * real.sqrt 3 * t ∧ y = 3 - 2 * t) → 
    let θ := real.arctan (real.abs ((-1) / real.sqrt 3 / 3)) 
    in θ = (5 * real.pi) / 6) :=
sorry

end slope_angle_of_parametric_line_l655_655762


namespace coeff_x4_in_expansion_l655_655973

open Nat

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def coefficient_x4_term : ℕ := binom 9 4

noncomputable def constant_term : ℕ := 243 * 4

theorem coeff_x4_in_expansion : coefficient_x4_term * 972 * Real.sqrt 2 = 122472 * Real.sqrt 2 :=
by
  sorry

end coeff_x4_in_expansion_l655_655973


namespace weaving_sequence_l655_655048

-- Define the arithmetic sequence conditions
def day1_weaving := 5
def total_cloth := 390
def days := 30

-- Mathematical statement to be proved
theorem weaving_sequence : 
    ∃ d : ℚ, 30 * day1_weaving + (days * (days - 1) / 2) * d = total_cloth ∧ d = 16 / 29 :=
by 
  sorry

end weaving_sequence_l655_655048


namespace hexagon_chord_problem_solution_l655_655243

noncomputable def hexagon_chord_length {p q : ℕ} (hpq_coprime : Nat.coprime p q) : ℕ :=
  let a := 4
  let b := 6
  let circ_hex := inscribed_hexagon_in_circle a b
  let chord := circ_hex.divides_into_trapezoids
  if (chord.len = p / q) then p + q else 0

theorem hexagon_chord_problem_solution :
  ∃ (p q : ℕ), Nat.coprime p q ∧ hexagon_chord_length Nat.Coprime p q = 799 :=
begin
  sorry
end

end hexagon_chord_problem_solution_l655_655243


namespace age_ratio_l655_655558

theorem age_ratio 
    (a m s : ℕ) 
    (h1 : m = 60) 
    (h2 : m = 3 * a) 
    (h3 : s = 40) : 
    (m + a) / s = 2 :=
by
    sorry

end age_ratio_l655_655558


namespace vectors_not_coplanar_l655_655661

-- Defining the vectors a, b, and c
def a : (Fin 3 → ℤ) := ![3, 1, 0]
def b : (Fin 3 → ℤ) := ![-5, -4, -5]
def c : (Fin 3 → ℤ) := ![4, 2, 4]

-- Statement of the proof
theorem vectors_not_coplanar : 
  Matrix.det !![a, b, c] = -18 := by
  sorry

end vectors_not_coplanar_l655_655661


namespace condition_suff_not_necessary_condition_not_necessary_l655_655356

theorem condition_suff_not_necessary (a b : ℝ) (h : 1 < b ∧ b < a) : a - 1 > |b - 1| :=
by {
  cases h with h1 h2,
  rw abs_lt,
  split,
  { linarith },
  { exact h2 }
}

theorem condition_not_necessary (a b : ℝ) (h : a - 1 > |b - 1|) : ¬(1 < b ∧ b < a) → True :=
by {
  intros h1,
  exact trivial
}

end condition_suff_not_necessary_condition_not_necessary_l655_655356


namespace problem_solution_l655_655744

-- Definitions based on conditions
def p (a b : ℝ) : Prop := a > b → a^2 > b^2
def neg_p (a b : ℝ) : Prop := a > b → a^2 ≤ b^2
def disjunction (p q : Prop) : Prop := p ∨ q
def suff_but_not_nec (x : ℝ) : Prop := x > 2 → x > 1 ∧ ¬(x > 1 → x > 2)
def congruent_triangles (T1 T2 : Prop) : Prop := T1 → T2
def neg_congruent_triangles (T1 T2 : Prop) : Prop := ¬(T1 → T2)

-- Mathematical problem as Lean statements
theorem problem_solution :
  ( (∀ a b : ℝ, p a b = (a > b → a^2 > b^2) ∧ neg_p a b = (a > b → a^2 ≤ b^2)) ∧
    (∀ p q : Prop, (disjunction p q) = false → p = false ∧ q = false) ∧
    (∀ x : ℝ, suff_but_not_nec x = (x > 2 → x > 1 ∧ ¬(x > 1 → x > 2))) ∧
    (∀ T1 T2 : Prop, (neg_congruent_triangles T1 T2) = true ↔ ¬(T1 → T2)) ) →
  ( (∀ a b : ℝ, neg_p a b = (a > b → a^2 ≤ b^2)) ∧
    (∀ p q : Prop, (disjunction p q) = false → p = false ∧ q = false) ∧
    (∀ x : ℝ, suff_but_not_nec x = (x > 2 → x > 1 ∧ ¬(x > 1 → x > 2))) ∧
    (∀ T1 T2 : Prop, (neg_congruent_triangles T1 T2) = false) ) :=
sorry

end problem_solution_l655_655744


namespace lcm_of_5_6_8_9_l655_655187

theorem lcm_of_5_6_8_9 : Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 9 = 360 := 
by 
  sorry

end lcm_of_5_6_8_9_l655_655187


namespace fractional_equation_solution_l655_655131

theorem fractional_equation_solution (x : ℝ) (h : x = 7) : (3 / (x - 3)) - 1 = 1 / (3 - x) := by
  sorry

end fractional_equation_solution_l655_655131


namespace second_caterer_is_cheaper_at_41_l655_655119

def cost1 (x : ℕ) : ℕ := 150 + 17 * x

def cost2 (x : ℕ) : ℕ :=
  if x ≤ 40 then 250 + 15 * x else 250 + 13 * x

theorem second_caterer_is_cheaper_at_41 : 
  ∃ x : ℕ, x >= 41 ∧ cost2 x < cost1 x :=
by
  use 41
  split
  . exact Nat.le_refl 41
  . unfold cost1 cost2
    rw if_neg (Nat.not_le_of_gt (Nat.succ_pos 40))
    sorry

end second_caterer_is_cheaper_at_41_l655_655119


namespace halfway_between_one_third_and_one_fifth_l655_655713

theorem halfway_between_one_third_and_one_fifth : (1/3 + 1/5) / 2 = 4/15 := 
by 
  sorry

end halfway_between_one_third_and_one_fifth_l655_655713


namespace three_digit_non_multiples_of_3_or_11_l655_655778

theorem three_digit_non_multiples_of_3_or_11 : 
  ∃ (n : ℕ), n = 546 ∧ 
  (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → 
    ¬ (x % 3 = 0 ∨ x % 11 = 0) → 
    n = (900 - (300 + 81 - 27))) := 
by 
  sorry

end three_digit_non_multiples_of_3_or_11_l655_655778


namespace small_slices_sold_l655_655209

theorem small_slices_sold (S L : ℕ) 
  (h1 : S + L = 5000) 
  (h2 : 150 * S + 250 * L = 1050000) : 
  S = 2000 :=
by
  sorry

end small_slices_sold_l655_655209


namespace mass_percentage_C_in_C6HxO6_indeterminate_l655_655340

-- Definition of conditions
def mass_percentage_C_in_C6H8O6 : ℚ := 40.91 / 100
def molar_mass_C : ℚ := 12.01
def molar_mass_H : ℚ := 1.01
def molar_mass_O : ℚ := 16.00

-- Formula for molar mass of C6H8O6
def molar_mass_C6H8O6 : ℚ := 6 * molar_mass_C + 8 * molar_mass_H + 6 * molar_mass_O

-- Mass of carbon in C6H8O6 is 40.91% of the total molar mass
def mass_of_C_in_C6H8O6 : ℚ := mass_percentage_C_in_C6H8O6 * molar_mass_C6H8O6

-- Hypothesis: mass percentage of carbon in C6H8O6 is given
axiom hyp_mass_percentage_C_in_C6H8O6 : mass_of_C_in_C6H8O6 = 72.06

-- Proof that we need the value of x to determine the mass percentage of C in C6HxO6
theorem mass_percentage_C_in_C6HxO6_indeterminate (x : ℚ) :
  (molar_mass_C6H8O6 = 176.14) → (mass_of_C_in_C6H8O6 = 72.06) → False :=
by
  sorry

end mass_percentage_C_in_C6HxO6_indeterminate_l655_655340


namespace train_speed_problem_l655_655224

noncomputable def speed_of_second_train := ∀ 
  (L1 : ℝ) (S1 : ℝ) (L2 : ℝ) (t : ℝ), 
  L1 = 270 →
  S1 = 120 →
  L2 = 230 →
  t = 9 →
  ∃ V2 : ℝ, V2 = 80.01

-- Setup the conditions as given in the problem
theorem train_speed_problem :
  speed_of_second_train 270 120 230 9 :=
by
  intros L1 S1 L2 t hL1 hS1 hL2 ht
  use 80.01
  sorry

end train_speed_problem_l655_655224


namespace sum_positive_integers_satisfy_lcm_gcd_l655_655994

theorem sum_positive_integers_satisfy_lcm_gcd (n : ℕ) :
  (∃ n : ℕ, lcm n 180 = gcd n 180 + 360) ∧ (∑ n in { n : ℤ | lcm n 180 = gcd n 180 + 360}.toFinset, n) = 450 :=
by
  sorry

end sum_positive_integers_satisfy_lcm_gcd_l655_655994


namespace exterior_angle_DEF_l655_655907

-- Definitions relating to regular polygons and their angles
def interior_angle (n : ℕ) : ℝ := (180 * (n - 2) / n)

-- Conditions
def angle_DEA : ℝ := interior_angle 8
def angle_FEA : ℝ := interior_angle 10

-- Theorem to prove
theorem exterior_angle_DEF : angle_DEA = 135 ∧ angle_FEA = 144 → (360 - angle_DEA - angle_FEA = 81) :=
by intros h 
   cases h with h1 h2
   simp [angle_DEA, angle_FEA, interior_angle] at h1 h2
   rw [h1, h2]
   norm_num

end exterior_angle_DEF_l655_655907


namespace point_P_in_first_quadrant_l655_655459

def pointInFirstQuadrant (x y : Int) : Prop := x > 0 ∧ y > 0

theorem point_P_in_first_quadrant : pointInFirstQuadrant 2 3 :=
by
  sorry

end point_P_in_first_quadrant_l655_655459


namespace tennis_handshakes_l655_655284

theorem tennis_handshakes :
  ∀ (teams : Fin 4 → Fin 2 → ℕ),
    (∀ i, teams i 0 ≠ teams i 1) ∧ (∀ i j a b, i ≠ j → teams i a ≠ teams j b) →
    ∃ handshakes : ℕ, handshakes = 24 :=
begin
  sorry
end

end tennis_handshakes_l655_655284


namespace sine_function_point_value_l655_655401

theorem sine_function_point_value 
  (x : ℝ) 
  (h : x = 7 * Real.pi / 3) : 
  ∃ m : ℝ, (sin x = m) ∧ (m = Real.sqrt 3 / 2) := by
  use sin x
  field_simp [h]
  sorry

end sine_function_point_value_l655_655401


namespace length_of_each_piece_cm_l655_655417

theorem length_of_each_piece_cm 
  (total_length : ℝ) 
  (number_of_pieces : ℕ) 
  (htotal : total_length = 17) 
  (hpieces : number_of_pieces = 20) : 
  (total_length / number_of_pieces) * 100 = 85 := 
by
  sorry

end length_of_each_piece_cm_l655_655417


namespace cost_of_gas_used_l655_655843

theorem cost_of_gas_used (initial_odometer final_odometer fuel_efficiency cost_per_gallon : ℝ)
  (h₀ : initial_odometer = 82300)
  (h₁ : final_odometer = 82335)
  (h₂ : fuel_efficiency = 22)
  (h₃ : cost_per_gallon = 3.80) :
  (final_odometer - initial_odometer) / fuel_efficiency * cost_per_gallon = 6.04 :=
by
  sorry

end cost_of_gas_used_l655_655843


namespace arithmetic_sequence_property_l655_655108

-- Define the arithmetic sequence {an}
variable {α : Type*} [LinearOrderedField α]

def is_arith_seq (a : ℕ → α) := ∃ (d : α), ∀ (n : ℕ), a (n+1) = a n + d

-- Define the condition
def given_condition (a : ℕ → α) : Prop := a 5 / a 3 = 5 / 9

-- Main theorem statement
theorem arithmetic_sequence_property (a : ℕ → α) (h : is_arith_seq a) 
  (h_condition : given_condition a) : 1 = 1 :=
by
  sorry

end arithmetic_sequence_property_l655_655108


namespace grunters_win_all_games_l655_655534

open ProbabilityTheory

theorem grunters_win_all_games :
  let win_probability : ℚ := 3 / 4 in
  (win_probability ^ 4) = 81 / 256 :=
by
  sorry

end grunters_win_all_games_l655_655534


namespace total_distance_traveled_l655_655249

noncomputable def V_m : ℝ := 10
noncomputable def V_r : ℝ := 2.4
noncomputable def V_r_new : ℝ := V_r + 0.6
noncomputable def T_total : ℝ := 3

theorem total_distance_traveled :
  let V_up := V_m - V_r,
      V_down_new := V_m + V_r_new,
      D := ((T_total * (V_up * V_down_new)) / (V_up + V_down_new))
  in 2 * D ≈ 28.78 :=
by
  -- placeholder for proof
  sorry

end total_distance_traveled_l655_655249


namespace seven_n_form_l655_655425

theorem seven_n_form (n : ℤ) (a b : ℤ) (h : 7 * n = a^2 + 3 * b^2) : 
  ∃ c d : ℤ, n = c^2 + 3 * d^2 :=
by {
  sorry
}

end seven_n_form_l655_655425


namespace heat_engine_efficiency_l655_655936

noncomputable def ideal_monoatomic_gas_engine_efficiency : Real :=
  let T_max := 2 * T_min
  let T_min : Real := sorry -- assign a value to T_min for concrete calculation
  let n : Real := sorry -- amount of gas in moles
  let R : Real := sorry -- ideal gas constant
  let Q_12 := (3 / 2) * n * R * T_min
  let Q_23 := n * R * (2 * T_min) * Real.log (Real.sqrt 2)
  let Q_31 := -2 * n * R * T_min
  let Q_H := Q_12 + Q_23
  let Q_C := -Q_31
  (Q_H - Q_C) / Q_H * 100

theorem heat_engine_efficiency :
  ideal_monoatomic_gas_engine_efficiency = 8.8 :=
by
  sorry

end heat_engine_efficiency_l655_655936


namespace find_b_l655_655028

variable (x : ℝ)

theorem find_b (a b: ℝ) (h1 : x + 1/x = a) (h2 : x^3 + 1/x^3 = b) (ha : a = 3): b = 18 :=
by
  sorry

end find_b_l655_655028


namespace Bryan_total_amount_l655_655664

theorem Bryan_total_amount (n_stones : ℕ) (price_per_stone : ℕ) (hn : n_stones = 8) (hp : price_per_stone = 1785) : n_stones * price_per_stone = 14280 := by
  rw [hn, hp]
  norm_num
  sorry

end Bryan_total_amount_l655_655664


namespace total_time_spent_l655_655072

-- Define the conditions
def t1 : ℝ := 2.5
def t2 : ℝ := 3 * t1

-- Define the theorem to prove
theorem total_time_spent : t1 + t2 = 10 := by
  sorry

end total_time_spent_l655_655072


namespace probability_both_defective_l655_655598

-- Probability of selecting a defective tube on the first draw
def P_first_defective (total_tubes defective_tubes : ℕ) : ℚ :=
  defective_tubes / total_tubes

-- Probability of selecting a defective tube on the second draw given the first was defective
def P_second_defective (remaining_tubes remaining_defective : ℕ) : ℚ :=
  remaining_defective / remaining_tubes

-- Probability of both tubes being defective
def P_both_defective (total_tubes defective_tubes : ℕ) : ℚ :=
  P_first_defective total_tubes defective_tubes * P_second_defective (total_tubes - 1) (defective_tubes - 1)

-- Given conditions
axiom total_tubes : ℕ := 20
axiom defective_tubes : ℕ := 5

-- Theorem statement
theorem probability_both_defective : P_both_defective total_tubes defective_tubes = 1 / 19 :=
by
  sorry

end probability_both_defective_l655_655598


namespace midpoint_on_circumcircle_of_triangle_ADZ_l655_655105

theorem midpoint_on_circumcircle_of_triangle_ADZ
  {A B C D Z : Point}
  (h_triangle: Triangle A B C)
  (h_AB_lt_AC : dist A B < dist A C)
  (h_D_on_angle_bisector : D lies_on_bisector BAC A circumcircle A B C)
  (h_Z_on_perpendicular_bisector_external_bisector : Z lies_on_perpendicular_bisector_both_angle AC and_external_bisector A B C) :
  midpoint A B lies_on_circumcircle A D Z :=
begin
  sorry
end

end midpoint_on_circumcircle_of_triangle_ADZ_l655_655105


namespace c_is_11_years_younger_than_a_l655_655214

variable (A B C : ℕ) (h : A + B = B + C + 11)

theorem c_is_11_years_younger_than_a (A B C : ℕ) (h : A + B = B + C + 11) : C = A - 11 := by
  sorry

end c_is_11_years_younger_than_a_l655_655214


namespace solution_set_f_2_minus_x_gt_0_l655_655638

noncomputable def f(x : ℝ) (a : ℝ) (b : ℝ) := (x - 2) * (a * x + b)

-- Given: f(x) is an even function and f(x) is monotonically increasing on (0, +∞)
variables (a b : ℝ)
axiom even_function : ∀ x : ℝ, f x a b = f (-x) a b
axiom monotonic_increasing : ∀ x y : ℝ, (0 < x ∧ x < y) → f x a b ≤ f y a b

theorem solution_set_f_2_minus_x_gt_0 (x : ℝ) :
  (∀ x : ℝ, b = 2 * a) ∧ (∀ x : ℝ, a > 0) →
  (f(2 - x) a b > 0 ↔ x < 0 ∨ x > 4) :=
sorry

end solution_set_f_2_minus_x_gt_0_l655_655638


namespace intersection_M_N_l655_655498

def is_M (x : ℝ) : Prop := x^2 + x - 6 < 0
def is_N (x : ℝ) : Prop := abs (x - 1) <= 2

theorem intersection_M_N : {x : ℝ | is_M x} ∩ {x : ℝ | is_N x} = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_M_N_l655_655498


namespace total_pencils_l655_655077

-- Defining the number of pencils each person has.
def jessica_pencils : ℕ := 8
def sandy_pencils : ℕ := 8
def jason_pencils : ℕ := 8

-- Theorem stating the total number of pencils
theorem total_pencils : jessica_pencils + sandy_pencils + jason_pencils = 24 := by
  sorry

end total_pencils_l655_655077


namespace abc_sum_zero_l655_655423

theorem abc_sum_zero
  (a b c : ℝ)
  (h1 : ∀ x: ℝ, (a * (c * x^2 + b * x + a)^2 + b * (c * x^2 + b * x + a) + c = x)) :
  (a + b + c = 0) :=
by
  sorry

end abc_sum_zero_l655_655423


namespace scaled_multiplication_l655_655751

theorem scaled_multiplication (h : 268 * 74 = 19832) : 2.68 * 0.74 = 1.9832 :=
by
  -- proof steps would go here
  sorry

end scaled_multiplication_l655_655751


namespace total_animals_correct_l655_655892

def initial_cows : ℕ := 2
def initial_pigs : ℕ := 3
def initial_goats : ℕ := 6

def added_cows : ℕ := 3
def added_pigs : ℕ := 5
def added_goats : ℕ := 2

def total_cows : ℕ := initial_cows + added_cows
def total_pigs : ℕ := initial_pigs + added_pigs
def total_goats : ℕ := initial_goats + added_goats

def total_animals : ℕ := total_cows + total_pigs + total_goats

theorem total_animals_correct : total_animals = 21 := by
  sorry

end total_animals_correct_l655_655892


namespace complement_U_A_l655_655880

def U := {x : ℝ | x < 2}
def A := {x : ℝ | x^2 < x}

theorem complement_U_A :
  (U \ A) = {x : ℝ | x ≤ 0 ∨ (1 ≤ x ∧ x < 2)} :=
sorry

end complement_U_A_l655_655880


namespace three_digit_numbers_not_multiple_of_3_or_11_l655_655784

theorem three_digit_numbers_not_multiple_of_3_or_11 : 
  let total_three_digit_numbers := 999 - 100 + 1 in
  let multiples_of_3 := 333 - 34 + 1 in
  let multiples_of_11 := 90 - 10 + 1 in
  let multiples_of_33 := 30 - 4 + 1 in
  let multiples_of_3_or_11 := multiples_of_3 + multiples_of_11 - multiples_of_33 in
  total_three_digit_numbers - multiples_of_3_or_11 = 546 :=
by
  let total_three_digit_numbers := 999 - 100 + 1
  let multiples_of_3 := 333 - 34 + 1
  let multiples_of_11 := 90 - 10 + 1
  let multiples_of_33 := 30 - 4 + 1
  let multiples_of_3_or_11 := multiples_of_3 + multiples_of_11 - multiples_of_33
  show total_three_digit_numbers - multiples_of_3_or_11 = 546 from sorry

end three_digit_numbers_not_multiple_of_3_or_11_l655_655784


namespace primes_equal_l655_655576

def is_prime (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_equal (p q r n : ℕ) (h_prime_p : is_prime p) (h_prime_q : is_prime q)
(h_prime_r : is_prime r) (h_pos_n : 0 < n)
(h1 : (p + n) % (q * r) = 0)
(h2 : (q + n) % (r * p) = 0)
(h3 : (r + n) % (p * q) = 0) : p = q ∧ q = r := by
  sorry

end primes_equal_l655_655576


namespace even_function_composition_is_even_l655_655858

-- Let's define what it means for a function to be even
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- The main theorem stating the evenness of the composition of an even function
theorem even_function_composition_is_even {f : ℝ → ℝ} (h : even_function f) :
  even_function (λ x, f (f x)) :=
by
  intros x
  have : f (-x) = f x := h x
  rw [←this, h (-x)]
  sorry

end even_function_composition_is_even_l655_655858


namespace smallest_value_8335_l655_655002

noncomputable def smallest_value (x y z : ℂ) : ℝ :=
  abs x * abs x + abs y * abs y + abs z * abs z

theorem smallest_value_8335 {x y z : ℂ} :
  (x + 5) * (y - 5) = 0 ∧ (y + 5) * (z - 5) = 0 ∧ (z + 5) * (x - 5) = 0 ∧
  (∀ a b c : ℂ, a ≠ b + 1 ∨ b ≠ c + 1 ∨ c ≠ a + 1) →
  smallest_value x y z = 83.75 :=
by
  sorry

end smallest_value_8335_l655_655002


namespace tennis_handshakes_l655_655281

theorem tennis_handshakes :
  ∀ (teams : Fin 4 → Fin 2 → ℕ),
    (∀ i, teams i 0 ≠ teams i 1) ∧ (∀ i j a b, i ≠ j → teams i a ≠ teams j b) →
    ∃ handshakes : ℕ, handshakes = 24 :=
begin
  sorry
end

end tennis_handshakes_l655_655281


namespace domain_of_g_l655_655975

noncomputable def g (x : ℝ) : ℝ := Real.logb 3 (Real.logb 4 (Real.logb 5 (Real.logb 6 x)))

theorem domain_of_g : setOf (λ x, g x) = set.Ioi (6^625) :=
by
  sorry

end domain_of_g_l655_655975


namespace sphere_radius_invariant_l655_655517

noncomputable def reflect_point (Q P: Point3) (tetra: Tetrahedron) (Q1 Q2 Q3 Q4 P1 P2 P3 P4: Point3) : Prop :=
∀ i ∈ {1, 2, 3, 4},
reflect(Q, face(tetra, i)) = Q_i ∧ reflect(P, face(tetra, i)) = P_i

theorem sphere_radius_invariant {Q P: Point3} (tetra: Tetrahedron) (Q1 Q2 Q3 Q4 P1 P2 P3 P4: Point3)
  (hQ1 : reflect(Q, face(tetra, 1)) = Q1)
  (hQ2 : reflect(Q, face(tetra, 2)) = Q2)
  (hQ3 : reflect(Q, face(tetra, 3)) = Q3)
  (hQ4 : reflect(Q, face(tetra, 4)) = Q4)
  (hP1 : reflect(P, face(tetra, 1)) = P1)
  (hP2 : reflect(P, face(tetra, 2)) = P2)
  (hP3 : reflect(P, face(tetra, 3)) = P3)
  (hP4 : reflect(P, face(tetra, 4)) = P4)
  (distinct_Qi : ∀ i j, i ≠ j → Qi i ≠ Qi j)
  (not_on_face: ∀ i, Q ∉ face(tetra, i))
  (not_on_circumsphere: Q ∉ circumsphere(tetra))
  (sphere_center_radius: sphere_center_radius(Q1, Q2, Q3, Q4) = (P, r)) :
  sphere_center_radius(P1, P2, P3, P4) = (P, r) := sorry

end sphere_radius_invariant_l655_655517


namespace problem1_l655_655909

variable (x y : ℝ)
variable (h1 : x = Real.sqrt 3 + Real.sqrt 5)
variable (h2 : y = Real.sqrt 3 - Real.sqrt 5)

theorem problem1 : 2 * x^2 - 4 * x * y + 2 * y^2 = 40 :=
by sorry

end problem1_l655_655909


namespace area_of_triangle_l655_655708

-- Define points
def Point := ℝ × ℝ

-- Define lines using points
def Line1 : Point × Point := ((0, 5), (10, 2))
def Line2 : Point × Point := ((2, 6), (8, 1))
def Line3 : Point × Point := ((0, 2), (8, 6))

-- Expected area of the triangle
def TriangleArea : ℝ := 0.48

theorem area_of_triangle :
  let l1 := Line1 in
  let l2 := Line2 in
  let l3 := Line3 in
  ∃ A B C : Point, 
    (A = (4, 3.7) ∧ B = (4.2, 4.1) ∧ C = (5, 4.5)) ∧ 
    let area := (1 / 2) * (abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))) in
    area = TriangleArea := 
by
  sorry

end area_of_triangle_l655_655708


namespace three_digit_numbers_not_multiple_of_3_or_11_l655_655783

theorem three_digit_numbers_not_multiple_of_3_or_11 : 
  let total_three_digit_numbers := 999 - 100 + 1 in
  let multiples_of_3 := 333 - 34 + 1 in
  let multiples_of_11 := 90 - 10 + 1 in
  let multiples_of_33 := 30 - 4 + 1 in
  let multiples_of_3_or_11 := multiples_of_3 + multiples_of_11 - multiples_of_33 in
  total_three_digit_numbers - multiples_of_3_or_11 = 546 :=
by
  let total_three_digit_numbers := 999 - 100 + 1
  let multiples_of_3 := 333 - 34 + 1
  let multiples_of_11 := 90 - 10 + 1
  let multiples_of_33 := 30 - 4 + 1
  let multiples_of_3_or_11 := multiples_of_3 + multiples_of_11 - multiples_of_33
  show total_three_digit_numbers - multiples_of_3_or_11 = 546 from sorry

end three_digit_numbers_not_multiple_of_3_or_11_l655_655783


namespace max_area_percentage_hexagon_square_l655_655583

theorem max_area_percentage_hexagon_square :
  ∀ (r : ℝ), r > 0 →
  let hexagon_area := (3 * Real.sqrt 3 / 2) * r^2 in
  let cos_15 := (Real.sqrt 6 + Real.sqrt 2) / 4 in
  let square_side := (Real.sqrt 3 * r) / (Real.sqrt 2 * cos_15) in
  let square_area := square_side^2 in
  (square_area / hexagon_area) * 100 ≈ 62 := sorry

end max_area_percentage_hexagon_square_l655_655583


namespace intersection_points_count_l655_655928

noncomputable def log_base_change (b a : ℝ) : ℝ := real.log a / real.log b

theorem intersection_points_count :
  let f := λ x : ℝ, real.log x / real.log 2
  let g := λ x : ℝ, 1 / (real.log x / real.log 2)
  let h := λ x : ℝ, -(real.log x / real.log 2)
  let k := λ x : ℝ, -1 / (real.log x / real.log 2)
  ∃ (s : finset (ℝ × ℝ)), 
  (∀ p ∈ s, (∃ x > 0, (f x = p.2 ∧ g x = p.2) ∨ (f x = p.2 ∧ h x = p.2) ∨ (f x = p.2 ∧ k x = p.2) ∨ (g x = p.2 ∧ h x = p.2) ∨ (g x = p.2 ∧ k x = p.2) ∨ (h x = p.2 ∧ k x = p.2)))
  ∧ s.card = 3 :=
by
  sorry

end intersection_points_count_l655_655928


namespace Jenny_older_than_Rommel_l655_655176

theorem Jenny_older_than_Rommel :
  ∃ t r j, t = 5 ∧ r = 3 * t ∧ j = t + 12 ∧ (j - r = 2) := 
by
  -- We insert the proof here using sorry to skip the actual proof part.
  sorry

end Jenny_older_than_Rommel_l655_655176


namespace width_first_sheet_l655_655256

-- Given Conditions
def length_first_sheet : ℝ := 13
def area_second_sheet : ℝ := 6.5 * 11
def combined_area_greater : ℝ := 100

-- Calculate the width of the first sheet
theorem width_first_sheet :
  ∃ w : ℝ, 2 * (w * length_first_sheet) = area_second_sheet + combined_area_greater ∧ w ≈ 6.6 := by
  sorry

end width_first_sheet_l655_655256


namespace direction_vector_is_2_1_l655_655932

noncomputable def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  !![3 / 5, 4 / 5; 4 / 5, -3 / 5]

def is_direction_vector (a b : ℤ) : Prop :=
  ∃ a b : ℤ,
    reflection_matrix.mul_vec !![a.to_rat, b.to_rat] =
    !![a.to_rat, b.to_rat] ∧ 
    a > 0 ∧ gcd a.nat_abs b.nat_abs = 1

theorem direction_vector_is_2_1 : is_direction_vector 2 1 :=
  sorry

end direction_vector_is_2_1_l655_655932


namespace max_value_ab_cd_l655_655934

theorem max_value_ab_cd : 
  ∃ (a b c d : ℕ), a ∈ {2, 3, 4, 5} ∧ b ∈ {2, 3, 4, 5} ∧ c ∈ {2, 3, 4, 5} ∧ d ∈ {2, 3, 4, 5} ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (a + b + c + d = 14) ∧ ((a + b) * (c + d) = 49) :=
sorry

end max_value_ab_cd_l655_655934


namespace line_param_proof_chord_length_proof_l655_655832

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 16

-- Define the parametric equation of the line in polar coordinates
def line_param_polar (θ : ℝ) : Prop := 
  let x := 4 * Real.cos θ
  let y := 2 + 4 * Real.sin θ
  circle_eq x y

-- Define the parametric equation of the original line (rectangular coordinates)
def line_param_rect (t : ℝ) : Prop := 
  let x := 1 + 2 * t
  let y := 1 + t
  true

-- Define the general equation of the line l
def line_general (x y : ℝ) : Prop := 2 * x - y - 3 = 0

-- Distance of the line from the center of the circle
def distance_from_center : ℝ := (| (-2) - 3 |) / Real.sqrt 5

-- Length of the chord cut off by the line
def chord_length (r d : ℝ) : ℝ := 2 * Real.sqrt (r^2 - d^2)

-- Theorem statements
theorem line_param_proof : ∀ θ : ℝ, line_param_polar θ :=
by
  intro θ
  sorry

theorem chord_length_proof : chord_length 4 distance_from_center = 2 * Real.sqrt 11 :=
by
  sorry

end line_param_proof_chord_length_proof_l655_655832


namespace ratio_AH_HD_zero_l655_655468

-- Declaring the triangle with given sides and angles
variables {A B C H D : Type} [Nonempty A]

axiom side_BC : ℝ := 6
axiom side_AC : ℝ := 3
axiom angle_C : ℝ := 30

-- Assume the orthocenter conditions
axiom orthocenter_of_triangle 
  (BC_len : ℝ) (AC_len : ℝ) (angle_C : ℝ) (H : A) (D : A) :
  ∃ (AH : ℝ) (HD : ℝ), BC_len = 6 ∧ AC_len = 3 ∧ angle_C = 30 ∧ (AH / HD = 0)

-- The theorem statement
theorem ratio_AH_HD_zero
  (BC_len : ℝ) (AC_len : ℝ) (angle_C : ℝ) (H : A) (D : A)
  (H_orthocenter : orthocenter_of_triangle BC_len AC_len angle_C H D) :
  (AH H D) / (HD H D) = 0 :=
sorry

end ratio_AH_HD_zero_l655_655468


namespace sin_2B_value_l655_655370

-- Define the triangle's internal angles and the tangent of angles
variables (A B C : ℝ) 

-- Given conditions from the problem
def tan_sequence (tanA tanB tanC : ℝ) : Prop :=
  tanA = (1/2) * tanB ∧
  tanC = (3/2) * tanB ∧
  2 * tanB = tanC + tanB + (tanC - tanA)

-- The statement to be proven
theorem sin_2B_value (h : tan_sequence (Real.tan A) (Real.tan B) (Real.tan C)) :
  Real.sin (2 * B) = 4 / 5 :=
sorry

end sin_2B_value_l655_655370


namespace simplify_expression_l655_655525

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
    ( (3 * x / 2)⁻² - ((x ^ 3 / 4) * (2 * x)) ) = (8 - 9 * x ^ 6) / (18 * x ^ 2) :=
by
  sorry

end simplify_expression_l655_655525


namespace distance_PB_l655_655217

theorem distance_PB (A B C D P : Point) (h_square : is_square A B C D) (hPA : dist P A = 5)
  (hPD : dist P D = 2) (hPC : dist P C = 7) : dist P B = 2 * Real.sqrt 7 :=
by
  sorry

end distance_PB_l655_655217


namespace trig_identity_l655_655721

theorem trig_identity : ∀ (θ : ℝ), 2 * real.cos (10 * real.pi / 180) - real.sin (20 * real.pi / 180) / real.cos (20 * real.pi / 180) = real.sqrt 3 :=
by
  sorry

end trig_identity_l655_655721


namespace distance_center_circle_to_line_l655_655763

-- Definition of the parametric equations for line l
def line_parametric (t : ℝ) : ℝ × ℝ :=
  let x := -2 - Real.sqrt 2 * t
  let y := 3 + Real.sqrt 2 * t
  (x, y)

-- Convert parametric equation to Cartesian form
def line_cartesian (x y : ℝ) : Prop :=
  x + y = 1

-- Polar equation of circle C and conversion to Cartesian coordinates
noncomputable def circle_cartesian := {p : ℝ × ℝ // p.1^2 - 4 * p.1 + p.2^2 = 0}

-- Center of the circle
def center_circle : ℝ × ℝ := (2, 0)

-- Distance formula from a point to a line
noncomputable def distance_to_line (x1 y1 : ℝ) : ℝ :=
  abs (x1 + y1 - 1) / Real.sqrt (1^2 + 1^2)

-- Proof that the distance from the center of circle C to line l is sqrt(2) / 2
theorem distance_center_circle_to_line : distance_to_line center_circle.1 center_circle.2 = Real.sqrt 2 / 2 :=
  by sorry

end distance_center_circle_to_line_l655_655763


namespace decreasing_on_interval_l655_655219

variable {α : Type*}[LinearOrder α]

def f (x: α) : ℝ

variable (a b : ℝ) (h : ∀ (x1 x2 : ℝ), a < x1 ∧ x1 < b → a < x2 ∧ x2 < b → (x1 - x2) * (f x1 - f x2) < 0)

theorem decreasing_on_interval (h : ∀ (x1 x2 : ℝ), a < x1 ∧  x1 < b → a < x2 ∧ x2 < b → (x1 - x2) * (f x1 - f x2) < 0) :
  ∀ (x1 x2 : ℝ), a < x1 ∧ x1 < b → a < x2 ∧ x2 < b → x1 < x2 → f(x1) > f(x2) :=
by
  sorry

end decreasing_on_interval_l655_655219


namespace tennis_tournament_handshakes_l655_655278

theorem tennis_tournament_handshakes :
  ∃ (number_of_handshakes : ℕ),
    let total_women := 8 in
    let handshakes_per_woman := 6 in
    let total_handshakes_counted_twice := total_women * handshakes_per_woman in
    number_of_handshakes = total_handshakes_counted_twice / 2 :=
begin
  use 24,
  unfold total_women handshakes_per_woman total_handshakes_counted_twice,
  norm_num,
end

end tennis_tournament_handshakes_l655_655278


namespace triangle_area_formula_l655_655315

theorem triangle_area_formula (x1 y1 x2 y2 x3 y3 : ℝ) :
  counter_clockwise_order (x1, y1) (x2, y2) (x3, y3) →
  (1 / 2) * |x1 * y2 - x2 * y1 + x2 * y3 - x3 * y2 + x3 * y1 - x1 * y3| = 
    (1 / 2) * abs ((x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y1 - x1 * y3)) :=
by sorry

end triangle_area_formula_l655_655315


namespace quadratic_inequality_l655_655408

theorem quadratic_inequality (a b c : ℝ)
  (h1 : ∀ x : ℝ, x = -2 → y = 8)
  (h2 : ∀ x : ℝ, x = -1 → y = 3)
  (h3 : ∀ x : ℝ, x = 0 → y = 0)
  (h4 : ∀ x : ℝ, x = 1 → y = -1)
  (h5 : ∀ x : ℝ, x = 2 → y = 0)
  (h6 : ∀ x : ℝ, x = 3 → y = 3)
  : ∀ x : ℝ, (y - 3 > 0) ↔ x < -1 ∨ x > 3 :=
sorry

end quadratic_inequality_l655_655408


namespace angle_between_vectors_l655_655007

variables {G : Type*} [InnerProductSpace ℝ G]

/-- Given vectors a and b such that |a| = 1, |b| = 2, and a ⋅ (a + b) = 0,
    prove that the angle between a and b is 2π/3. -/
theorem angle_between_vectors (a b : G) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 2) (h : inner a (a + b) = 0) : 
  real.angle a b = 2 * real.pi / 3 := 
by sorry

end angle_between_vectors_l655_655007


namespace percentage_problem_l655_655801

theorem percentage_problem (P : ℝ) :
  (P / 100) * 600 = (40 / 100) * 1050 → P = 70 :=
by
  intro h
  sorry

end percentage_problem_l655_655801


namespace a_n_found_S_n_found_l655_655856

noncomputable def a_n (n : ℕ) : ℕ := 2 ^ n

def b_n (n : ℕ) : ℕ := 1 + 2 * n

def S_n (n : ℕ) : ℕ := (Finset.range n).sum (λ k, a_n k + b_n k)

theorem a_n_found : ∀ n : ℕ, a_n n = 2 ^ n := by
  intro n
  sorry

theorem S_n_found : ∀ n : ℕ, S_n n = 2 ^ (n + 1) + n ^ 2 - 2 := by
  intro n
  sorry

end a_n_found_S_n_found_l655_655856


namespace incorrect_statement_l655_655312

def vector_mult (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.2 - a.2 * b.1

theorem incorrect_statement (a b : ℝ × ℝ) : vector_mult a b ≠ vector_mult b a :=
by
  sorry

end incorrect_statement_l655_655312


namespace number_of_sets_containing_4_summing_to_16_l655_655568

theorem number_of_sets_containing_4_summing_to_16 :
  let S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  ∃! (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = 16 ∧ a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a = 4 :=
  2 :=
sorry

end number_of_sets_containing_4_summing_to_16_l655_655568


namespace weight_of_replaced_person_l655_655538

def new_person_weight : ℝ := 98.6
def avg_weight_increase_per_person : ℝ := 4.2
def num_persons : ℕ := 8

theorem weight_of_replaced_person : 
  ∃ W : ℝ, W = new_person_weight - avg_weight_increase_per_person * num_persons :=
begin
  use 65,
  sorry
end

end weight_of_replaced_person_l655_655538


namespace equalize_balls_after_addition_l655_655567

-- Problem statement and definitions
variables (a : fin 10 → ℕ) -- a vector representing the number of balls removed from each color
variables (n : fin 10 → ℕ) -- n is the initial number of balls of each color

theorem equalize_balls_after_addition (h : ∑ i, a i = 100) (k : ℕ) (h₁ : ∀ i, n i = k + a i) :
  ∃ b, (∀ i, b i = 100 - a i) ∧ (∑ i, b i = 900) :=
begin
  -- The proof will eventually show that adding 900 balls in the way described makes each color equal
  sorry
end

end equalize_balls_after_addition_l655_655567


namespace speed_boat_upstream_l655_655227

-- Define the conditions provided in the problem
def V_b : ℝ := 8.5  -- Speed of the boat in still water (in km/hr)
def V_downstream : ℝ := 13 -- Speed of the boat downstream (in km/hr)
def V_s : ℝ := V_downstream - V_b  -- Speed of the stream (in km/hr), derived from V_downstream and V_b
def V_upstream (V_b : ℝ) (V_s : ℝ) : ℝ := V_b - V_s  -- Speed of the boat upstream (in km/hr)

-- Statement to prove: the speed of the boat upstream is 4 km/hr
theorem speed_boat_upstream :
  V_upstream V_b V_s = 4 :=
by
  -- This line is for illustration, replace with an actual proof
  sorry

end speed_boat_upstream_l655_655227


namespace three_digit_numbers_not_multiple_of_3_or_11_l655_655786

-- Proving the number of three-digit numbers that are multiples of neither 3 nor 11 is 547
theorem three_digit_numbers_not_multiple_of_3_or_11 : (finset.Icc 100 999).filter (λ n, ¬(3 ∣ n) ∧ ¬(11 ∣ n)).card = 547 :=
by
  -- The steps to reach the solution will be implemented here
  sorry

end three_digit_numbers_not_multiple_of_3_or_11_l655_655786


namespace first_trial_addition_amounts_l655_655391

-- Define the range and conditions for the biological agent addition amount.
def lower_bound : ℝ := 20
def upper_bound : ℝ := 30
def golden_ratio_method : ℝ := 0.618
def first_trial_addition_amount_1 : ℝ := lower_bound + (upper_bound - lower_bound) * golden_ratio_method
def first_trial_addition_amount_2 : ℝ := upper_bound - (upper_bound - lower_bound) * golden_ratio_method

-- Prove that the possible addition amounts for the first trial are 26.18g or 23.82g.
theorem first_trial_addition_amounts :
  (first_trial_addition_amount_1 = 26.18 ∨ first_trial_addition_amount_2 = 23.82) :=
by
  -- Placeholder for the proof.
  sorry

end first_trial_addition_amounts_l655_655391


namespace evaluate_expression_l655_655380

theorem evaluate_expression (a b c d m : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 1) :
  ∃ v : ℝ, v ∈ {0, -2} ∧ m + 2024 * (a + b) / 2023 - (c * d)^2 = v :=
by
  sorry

end evaluate_expression_l655_655380


namespace tennis_handshakes_l655_655274

-- Define the participants and the condition
def number_of_participants := 8
def handshakes_no_partner (n : ℕ) := n * (n - 2) / 2

-- Prove that the number of handshakes is 24
theorem tennis_handshakes : handshakes_no_partner number_of_participants = 24 := by
  -- Since we are skipping the proof for now
  sorry

end tennis_handshakes_l655_655274


namespace triangle_area_98_l655_655709

theorem triangle_area_98 (B C : ℝ) (AB : ℝ) (hB : B = 45) (hC : C = 45) (hAB : AB = 14) :
  let BC := AB,
      area := (1 / 2) * AB * BC in
  area = 98 := by
  sorry

end triangle_area_98_l655_655709


namespace angle_parallel_l655_655727

-- Definitions of the vectors a and b and the condition that they are parallel
variables (α : ℝ)

def vec_a := (3 / 2, Real.sin α)
def vec_b := (Real.cos α, 1 / 3)

def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- The theorem to prove
theorem angle_parallel (h : parallel (vec_a α) (vec_b α)) : α = π / 4 :=
by
  -- proof steps will go here
  sorry

end angle_parallel_l655_655727


namespace value_of_m_perfect_square_l655_655484

noncomputable def F (x : ℝ) (m : ℝ) : ℝ := (6 * x^2 + 16 * x + 3 * m) / 6

theorem value_of_m_perfect_square :
  ∀ (m : ℝ), (∃ (a b : ℝ), F x m = (a * x + b) ^ 2) → m = 32 / 9 :=
by
  assume m,
  assume h,
  sorry

end value_of_m_perfect_square_l655_655484


namespace number_of_days_same_l655_655230

-- Defining volumes as given in the conditions.
def volume_project1 : ℕ := 100 * 25 * 30
def volume_project2 : ℕ := 75 * 20 * 50

-- The mathematical statement we want to prove.
theorem number_of_days_same : volume_project1 = volume_project2 → ∀ d : ℕ, d > 0 → d = d :=
by
  sorry

end number_of_days_same_l655_655230


namespace round_robin_arrangement_exists_l655_655125

structure Participant (α : Type) :=
  (id : α)
  (defeated_by : α → Prop)

def round_robin_tournament {α : Type} (players : List (Participant α)) : Prop :=
  ∀ p1 p2 : Participant α, p1 ≠ p2 → p1.defeated_by p2.id ∨ p2.defeated_by p1.id

theorem round_robin_arrangement_exists {α : Type} (players : List (Participant α)) 
  (H : round_robin_tournament players) : 
  ∃ arrangement : List (Participant α), 
  ∀ i : ℕ,
  (i < arrangement.length - 1) → 
  arrangement[i+1].defeated_by arrangement[i].id := 
sorry

end round_robin_arrangement_exists_l655_655125


namespace valid_range_a_unique_point_angle_dihedral_angle_proof_l655_655826

-- Definition and conditions for Part 1
def is_valid_range_a (a : ℝ) : Prop :=
  a ≥ 2 * Real.sqrt 3

-- Proof goal for Part 1
theorem valid_range_a (a : ℝ) : is_valid_range_a a ↔ (∃ t : ℝ, ∀ Q : ℝ, Q = t ∧ Q ∈ set.Icc 0 a ∧ PQ Q b c PQ_perp Q_perp QD) sorry

-- Definition and conditions for Part 2
def is_angle_between_skew_lines (angle : ℝ) : Prop :=
  angle = Real.arccos (Real.sqrt 42 / 14)

-- Proof goal for Part 2
theorem unique_point_angle (angle : ℝ) : is_angle_between_skew_words a ↔ angle sorry

-- Definition and conditions for Part 3
def is_dihedral_angle (a : ℝ) (angle1 angle2 : ℝ) : Prop :=
  angle1 = Real.arccos (Real.sqrt 15 / 5) ∧ angle2 = Real.arccos (Real.sqrt 7 / 7)

-- Proof goal for Part 3
theorem dihedral_angle_proof (a : ℝ) (angle1 angle2 : ℝ) : is_dihedral_angle a angle1 angle2 → angle1 sorry

end valid_range_a_unique_point_angle_dihedral_angle_proof_l655_655826


namespace max_value_a_has_three_solutions_l655_655341

theorem max_value_a_has_three_solutions :
  ∃ a, (∀ x, (|x - 2| + 2 * a)^2 - 3 * (|x - 2| + 2 * a) + 4 * a * (3 - 4 * a) = 0 ∧
            ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) → a ≤ 0.5 :=
sorry

end max_value_a_has_three_solutions_l655_655341


namespace find_x_l655_655728

theorem find_x (x : ℝ) (h : (x^2 - x - 6) / (x + 1) = (x^2 - 2*x - 3) * (0 : ℂ).im) : x = 3 :=
sorry

end find_x_l655_655728


namespace sqrt_combination_l655_655798

theorem sqrt_combination (a : ℝ) (h₁ : sqrt 12 = 5 * sqrt (a + 1)) : 
  a = 2 ∧ sqrt 12 + 5 * sqrt (a + 1) = 7 * sqrt 3 := by
  sorry

end sqrt_combination_l655_655798


namespace range_of_a_l655_655031

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - (a - 1) * x + 1

theorem range_of_a (a : ℝ) (h : ∀ x ∈ set.Icc 0 1, f' a x ≤ 0) : 
  a ∈ set.Ici (Real.exp 1 + 1) :=
by
  have h_deriv : ∀ x ∈ set.Icc 0 1, Real.exp x - (a - 1) ≤ 0 := 
    by 
      sorry
  have exp_increasing : ∀ x ∈ set.Icc 0 1, Real.exp x + 1 ≤ Real.exp 1 + 1 :=
    by 
      sorry
  sorry

where f' (a x : ℝ) : ℝ := Real.exp x - (a - 1)

end range_of_a_l655_655031


namespace tennis_tournament_handshakes_l655_655279

theorem tennis_tournament_handshakes :
  ∃ (number_of_handshakes : ℕ),
    let total_women := 8 in
    let handshakes_per_woman := 6 in
    let total_handshakes_counted_twice := total_women * handshakes_per_woman in
    number_of_handshakes = total_handshakes_counted_twice / 2 :=
begin
  use 24,
  unfold total_women handshakes_per_woman total_handshakes_counted_twice,
  norm_num,
end

end tennis_tournament_handshakes_l655_655279


namespace countDistinguishedDigitsTheorem_l655_655013

-- Define a function to count numbers with four distinct digits where leading zeros are allowed
def countDistinguishedDigits : Nat :=
  10 * 9 * 8 * 7

-- State the theorem we need to prove
theorem countDistinguishedDigitsTheorem :
  countDistinguishedDigits = 5040 := 
by
  sorry

end countDistinguishedDigitsTheorem_l655_655013


namespace equal_angles_given_conditions_l655_655161

open Classical

noncomputable theory

variables {α : Type*} [MetricSpace α] [NormedGroup α] [NormedSpace ℝ α]
variables (A B K L X P Q T : α)
variables (Γ : Sphere α)
variables (C1 C2 : Subset α)

-- Assume all points A, B, K, L, X lie on the circle Γ
def points_on_circle : Prop := A ∈ Γ ∧ B ∈ Γ ∧ K ∈ Γ ∧ L ∈ Γ ∧ X ∈ Γ

-- Assume arcs are equal
def equal_arcs : Prop := MeasurableSpace.measure (Arc B K Γ) = MeasurableSpace.measure (Arc K L Γ)

-- Circle tangent and intersection conditions
def circle_tangent_and_intersections : Prop :=
  (∃ C1 : Circle α, A ∈ C1 ∧ Circle.Tangent C1 (LineSegment B K) ∧ Circle.Intersects C1 (LineSegment K X) P ∧ Circle.Intersects C1 (LineSegment K X) Q) ∧
  (∃ C2 : Circle α, A ∈ C2 ∧ Circle.Tangent C2 (LineSegment B L) ∧ Circle.Intersects C2 (LineSegment B X) T)

-- Angle condition to prove
def desired_angle : Prop := ∠PTB = ∠XTQ

-- The theorem statement
theorem equal_angles_given_conditions :
  points_on_circle A B K L X Γ →
  equal_arcs B K L Γ →
  circle_tangent_and_intersections A B K L X P Q T C1 C2 →
  desired_angle P T B X Q T :=
sorry

end equal_angles_given_conditions_l655_655161


namespace tan_105_eq_neg2_sub_sqrt3_l655_655346

theorem tan_105_eq_neg2_sub_sqrt3 :
  let A := 60
  let B := 45
  ∠ A + ∠ B = 105
  let tan_A := Real.tan (A * Real.pi / 180)
  let tan_B := Real.tan (B * Real.pi / 180)
  tan_A = Real.sqrt 3
  tan_B = 1
  Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l655_655346


namespace parabola_intersection_l655_655898

theorem parabola_intersection (h1 : ℝ) (h2 : ℝ) (t : ℝ):
  (h1 = (10 + 13) / 2) →
  (h2 = 2 * h1) →
  (h2 = (13 + t) / 2) →
  t = 33 :=
by
  intros h1_eq h2_eq h2_t_eq
  rw [←h1_eq, ←h2_eq] at h2_t_eq
  linarith
  sorry

end parabola_intersection_l655_655898


namespace largest_median_l655_655698

-- Defining the initial list of given numbers
def given_numbers : List ℕ := [3, 7, 2, 5, 9, 6]

-- A definition for the problem condition that includes these numbers within a list of eleven positive integers.
def contains_given_numbers (l : List ℕ) : Prop :=
  given_numbers ⊆ l ∧ l.length = 11 ∧ ∀ x ∈ l, x > 0

-- A definition for the median of a list of eleven positive integers.
def median (l : List ℕ) : ℕ :=
  let sorted_list := l.qsort (· ≤ ·) in
  sorted_list.nthLe 5 sorry  -- nthLe with proof that index 5 is within bounds of 11-element list

-- The statement to prove that the largest possible value of the median is 8
theorem largest_median :
  ∃ l : List ℕ, contains_given_numbers l ∧ median l = 8 :=
sorry

end largest_median_l655_655698


namespace cats_to_dogs_ratio_l655_655153

noncomputable def num_dogs : ℕ := 18
noncomputable def num_cats : ℕ := num_dogs - 6
noncomputable def ratio (a b : ℕ) : ℚ := a / b

theorem cats_to_dogs_ratio (h1 : num_dogs = 18) (h2 : num_cats = num_dogs - 6) : ratio num_cats num_dogs = 2 / 3 :=
by
  sorry

end cats_to_dogs_ratio_l655_655153


namespace commodity_price_change_l655_655041

theorem commodity_price_change (x : ℝ)
    (P0 : ℝ) (P1 : P1 = P0 * (1 - 0.15))
    (P2 : P2 = P1 * (1 + 0.30))
    (P3 : P3 = P2 * (1 - 0.25))
    (P4 : P4 = P3 * (1 + 0.10))
    (P5 : P5 = P4 * (1 - x))
    (H : P5 = P0) :
    x = 0.10 :=
by 
  sorry

end commodity_price_change_l655_655041


namespace areas_equal_l655_655056

noncomputable def midpoint (A B : Point) : Point :=
  (A + B) / 2

def orthocenter (A B C : Triangle) : Point :=
  -- The orthocenter is the intersection of the altitudes

theorem areas_equal (A B C D : Point) (circABC : is_cyclic_quadrilateral A B C D) :
  let E := midpoint A B
  let F := midpoint B C
  let G := midpoint C D
  let H := midpoint D A
  let W := orthocenter A H E
  let X := orthocenter B E F
  let Y := orthocenter C F G
  let Z := orthocenter D G H
  area (quadrilateral A B C D) = area (quadrilateral W X Y Z) :=
sorry

end areas_equal_l655_655056


namespace find_abs_sum_l655_655872

noncomputable def x : ℝ := 1 + (Real.sqrt 3) / (1 + (Real.sqrt 3) / (1 + ...))

theorem find_abs_sum :
  let A : ℤ := -2
  let B : ℤ := 3
  let C : ℤ := -1
  in |A| + |B| + |C| = 6 := 
by 
  sorry

end find_abs_sum_l655_655872


namespace common_root_eqn_l655_655902

variables {a b c A B C x1 : ℝ}

-- Defining the conditions
def eq1 : Prop := a * x1^2 + b * x1 + c = 0
def eq2 : Prop := A * x1^2 + B * x1 + C = 0
def other_roots_different : Prop := ∀ (x2 x3 : ℝ), 
  (a * x2^2 + b * x2 + c = 0) ∧ (A * x3^2 + B * x3 + C = 0) → x2 ≠ x3

-- The statement to be proved
theorem common_root_eqn (h1 : eq1) (h2 : eq2) (h3 : other_roots_different) :
  (A * c - C * a)^2 = (A * b - B * a) * (B * c - C * b) :=
by sorry

end common_root_eqn_l655_655902


namespace max_value_of_f_sin_theta_l655_655761

variable {a b : ℝ} (h_a_pos : a > 0) (f : ℝ → ℝ)
  (hf : ∀ x, f x = 2 * a * x ^ 2 - 2 * b * x - a + b)

theorem max_value_of_f_sin_theta :
  ∀ θ ∈ Icc 0 (Real.pi / 2), f (Real.sin θ) ≤ |a - b| :=
sorry

end max_value_of_f_sin_theta_l655_655761


namespace num_three_digit_numbers_divisible_by_5_and_6_with_digit_6_l655_655016

theorem num_three_digit_numbers_divisible_by_5_and_6_with_digit_6 : 
  ∃ S : Finset ℕ, (∀ n ∈ S, 100 ≤ n ∧ n < 1000 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ (6 ∈ n.digits 10)) ∧ S.card = 6 :=
by
  sorry

end num_three_digit_numbers_divisible_by_5_and_6_with_digit_6_l655_655016


namespace roots_and_discriminant_l655_655555

theorem roots_and_discriminant (a b c : ℝ) (h_eq : a = 1 ∧ b = 5 ∧ c = 0):
  (roots : Finset ℝ, Δ : ℝ) → roots = {0, -5} ∧ Δ = 25 := by
  sorry

end roots_and_discriminant_l655_655555


namespace repeated_root_value_l655_655428

theorem repeated_root_value (m : ℝ) :
  (∃ x : ℝ, x ≠ 1 ∧ (2 / (x - 1) + 3 = m / (x - 1)) ∧ 
            ∀ y : ℝ, y ≠ 1 ∧ (2 / (y - 1) + 3 = m / (y - 1)) → y = x) →
  m = 2 :=
by
  sorry

end repeated_root_value_l655_655428


namespace intersection_point_of_lateral_sides_l655_655507

theorem intersection_point_of_lateral_sides 
  (a b : ℝ) (k : ℝ) 
  (h1 : k > 0) 
  (h2 : 4 * a * b = k):
  ∃ q : ℝ × ℝ, (∀ (x1 x2 : ℝ), (x1 = ±a ∨ x1 = ±b) → (x2 = ±a ∨ x2 = ±b) → (x1 ≠ x2) → q ∈ [pair (x1, x1 ^ 2), pair (x2, x2 ^ 2)]) ∧ q = (0, -k / 4) := by
  sorry

end intersection_point_of_lateral_sides_l655_655507


namespace simplify_log_expression_l655_655910

-- Given values are non-zero
variables (a b c d x y : ℝ)
variables (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)

-- The mathematical proof problem in Lean 4
theorem simplify_log_expression :
  log ((2 * a) / (3 * b)) + log ((5 * b) / (4 * c)) + log ((6 * c) / (7 * d)) - log ((20 * a * y) / (21 * d * x)) = log (3 * x / (4 * y)) :=
by {
  sorry
}

end simplify_log_expression_l655_655910


namespace find_n_if_2017_exists_in_sequence_l655_655887

theorem find_n_if_2017_exists_in_sequence (n : ℕ) (h : 2017 ∈ (list.range n).map (λ k, n^2 - n + 2*k + 1)) : n = 45 := 
by sorry

end find_n_if_2017_exists_in_sequence_l655_655887


namespace tennis_handshakes_l655_655282

theorem tennis_handshakes :
  ∀ (teams : Fin 4 → Fin 2 → ℕ),
    (∀ i, teams i 0 ≠ teams i 1) ∧ (∀ i j a b, i ≠ j → teams i a ≠ teams j b) →
    ∃ handshakes : ℕ, handshakes = 24 :=
begin
  sorry
end

end tennis_handshakes_l655_655282


namespace arithmetic_sequence_problem_l655_655379

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (a1 : ℝ)
  (d : ℝ)
  (h1 : d = 2)
  (h2 : ∀ n : ℕ, a n = a1 + (n - 1) * d)
  (h3 :  ∀ n : ℕ, S n = (n * (2 * a1 + (n - 1) * d)) / 2)
  (h4 : S 6 = 3 * S 3) :
  a 9 = 20 :=
by sorry

end arithmetic_sequence_problem_l655_655379


namespace area_triangle_BXD_l655_655964

-- Define the geometric entities and conditions
variables (AB CD AC BD X : Type) [Trapezoid AB CD AC BD X] (area_ABCD : ℝ)
variables (base_AB base_CD : ℝ)
variables (A B C D : Point) 
variables (h : ℝ) (area_BXD : ℝ)

-- Given conditions
def conditions : Prop :=
  base_AB = 24 ∧ 
  base_CD = 36 ∧ 
  area_ABCD = 360

-- The proof problem
theorem area_triangle_BXD (h : ℝ) (height_DXC : ℝ) (height_BXD : ℝ) (base_BXD : ℝ) :
  conditions AB CD AC BD X area_ABCD base_AB base_CD → 
  h = 12 →
  height_DXC = (3/5) * h →
  height_BXD = h - height_DXC →
  base_BXD = base_AB →
  area_BXD = (1/2) * base_BXD * height_BXD →
  area_BXD = 57.6 :=
by {
  intros h_eq h_val height_DXC_eq height_DXC_val height_BXD_eq height_BXD_val base_BXD_eq base_BXD_val area_BXD_eq,
  rw [h_val, height_DXC_val, height_BXD_val, base_BXD_val, area_BXD_eq],
  norm_num,
  exact area_BXD_val,
}

end area_triangle_BXD_l655_655964


namespace integral_value_l655_655676

noncomputable def given_integral : ℝ := ∫ x in (-14 / 15 : ℝ)..(-7 / 8 : ℝ), (6 * real.sqrt (x+2)) / ((x+2)^2 * real.sqrt (x+1))

theorem integral_value : given_integral = 1 := by
  sorry

end integral_value_l655_655676


namespace problem_statement_l655_655792

def integer_part (a : ℝ) : ℤ := Int.floor a

theorem problem_statement : integer_part (1 / Real.sqrt (16 - 6 * Real.sqrt 7)) = 2 :=
by
  sorry

end problem_statement_l655_655792


namespace monochromatic_triangle_in_K9_l655_655679

theorem monochromatic_triangle_in_K9 (n : ℕ) (E : Finset (Fin 9 × Fin 9)) 
  (color : (Fin 9 × Fin 9) → ℕ) :
  (E.card = n) ∧ (∀ e ∈ E, color e = 1 ∨ color e = 2) → (∃ T : Finset (Fin 9 × Fin 9), T.card = 3 ∧ T ⊆ E ∧ ∀ e ∈ T, color e = 1 ∨ color e = 2) → monochromatic_triangle.

end monochromatic_triangle_in_K9_l655_655679


namespace greatest_two_digit_with_product_9_l655_655990

theorem greatest_two_digit_with_product_9 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ a b : ℕ, n = 10 * a + b ∧ a * b = 9) ∧ (∀ m : ℕ, 10 ≤ m ∧ m < 100 ∧ (∃ c d : ℕ, m = 10 * c + d ∧ c * d = 9) → m ≤ 91) :=
by
  sorry

end greatest_two_digit_with_product_9_l655_655990


namespace roses_formula_l655_655808

open Nat

def total_roses (n : ℕ) : ℕ := 
  (choose n 4) + (choose (n - 1) 2)

theorem roses_formula (n : ℕ) (h : n ≥ 4) : 
  total_roses n = (choose n 4) + (choose (n - 1) 2) := 
by
  sorry

end roses_formula_l655_655808


namespace sum_of_primes_is_prime_l655_655937

open Nat

theorem sum_of_primes_is_prime (C D : ℕ) (hC_prime : Prime C) (hD_prime : Prime D) (hCD_prime : Prime (C - D)) (hCP_prime : Prime (C + D)) : 
  Prime (C + D + (C - D) + (C + D)) := 
sorry 

end sum_of_primes_is_prime_l655_655937


namespace polynomial_div_remainder_l655_655342

theorem polynomial_div_remainder (x : ℝ) : 
  (x^4 % (x^2 + 7*x + 2)) = -315*x - 94 := 
by
  sorry

end polynomial_div_remainder_l655_655342


namespace digit_inequality_l655_655532

theorem digit_inequality (d : ℕ) (h : d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) : 
  ( (3 + d / 100 + 1 / 1000) > 3.01) ↔ d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} :=
sorry

end digit_inequality_l655_655532


namespace net_pay_is_correct_l655_655805

-- Define the gross pay and taxes paid as constants
def gross_pay : ℕ := 450
def taxes_paid : ℕ := 135

-- Define net pay as a function of gross pay and taxes paid
def net_pay (gross : ℕ) (taxes : ℕ) : ℕ := gross - taxes

-- The proof statement
theorem net_pay_is_correct : net_pay gross_pay taxes_paid = 315 := by
  sorry -- The proof goes here

end net_pay_is_correct_l655_655805


namespace left_hand_derivative_at_zero_right_hand_derivative_at_zero_l655_655968

def f (x : ℝ) : ℝ := real.sqrt (1 - real.exp (-x^2))

theorem left_hand_derivative_at_zero :
  filter.tendsto (λ Δx : ℝ, (f Δx - f 0) / Δx) (filter.at_bot) (nhds (-1)) := 
sorry

theorem right_hand_derivative_at_zero :
  filter.tendsto (λ Δx : ℝ, (f Δx - f 0) / Δx) (filter.at_top) (nhds 1) := 
sorry

end left_hand_derivative_at_zero_right_hand_derivative_at_zero_l655_655968


namespace star_perimeter_is_four_l655_655087

noncomputable def regular_hexagon_perimeter : ℝ := 1

def hexagon_side_length (perimeter : ℝ) := perimeter / 6

def star_side_length (side_length : ℝ) := 2 * side_length

def star_perimeter (star_side_length : ℝ) := 12 * star_side_length

theorem star_perimeter_is_four :
  star_perimeter (star_side_length (hexagon_side_length regular_hexagon_perimeter)) = 4 :=
by
  -- this is where the actual proof would go
  sorry

end star_perimeter_is_four_l655_655087


namespace annie_purchases_l655_655269

theorem annie_purchases (x y z : ℕ) 
  (h1 : x + y + z = 50) 
  (h2 : 20 * x + 400 * y + 500 * z = 5000) :
  x = 40 :=
by sorry

end annie_purchases_l655_655269


namespace num_routes_A_to_B_l655_655300

-- Define types for Cities and Roads
inductive City
| A | B | C | D | F

inductive Road
| AB | AD | AF | BC | BD | CD | DF

-- Define a structure for conditions and roads between cities
structure ProblemConditions where
  cities : List City
  roads  : List (City × City)

-- Given conditions from the problem
def conditions : ProblemConditions :=
  { cities := [City.A, City.B, City.C, City.D, City.F],
    roads := [
      (City.A, City.B),
      (City.A, City.D),
      (City.A, City.F),
      (City.B, City.C),
      (City.B, City.D),
      (City.C, City.D),
      (City.D, City.F)
    ] }

-- Statement of the proof problem
theorem num_routes_A_to_B (conds : ProblemConditions) : 
  conds = conditions →
  16 = count_routes_A_to_B_using_all_roads conds :=
sorry

-- Dummy definition for count_routes_A_to_B_using_all_roads for now
noncomputable def count_routes_A_to_B_using_all_roads (conds : ProblemConditions) : ℕ := 
  sorry

end num_routes_A_to_B_l655_655300


namespace tetrahedron_altitudes_intersect_l655_655044

theorem tetrahedron_altitudes_intersect (A B C D : Point) (AA₁ BB₁ CC₁ DD₁ : Point) 
  (h1 : Altitude A A₁ B C D) 
  (h2 : Altitude B B₁ A C D) 
  (h3 : Altitude C C₁ A B D) 
  (h4 : Altitude D D₁ A B C) 
  (h_intersect : ∃ P, P ∈ Line A A₁ ∧ P ∈ Line D D₁ ∧ P ∈ Line B B₁)
  : ∃ P, ∀ E. Altitude E P (other vertices excluding E), belongs to every altitude line sorry

end tetrahedron_altitudes_intersect_l655_655044


namespace price_of_table_l655_655600

variable (C T : ℝ)

theorem price_of_table :
  2 * C + T = 0.6 * (C + 2 * T) ∧
  C + T = 96 →
  T = 84 := by
sorry

end price_of_table_l655_655600


namespace solution_set_inequality_l655_655389

-- Define the function f
def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x - 3 else -2^(-x) + 3

-- Given that f is an odd function
axiom f_odd (x : ℝ) : f (-x) = -f x

-- Prove the solution set for the inequality
theorem solution_set_inequality : {x : ℝ | f x ≤ -5} = {x : ℝ | x ≤ -3} :=
  sorry

end solution_set_inequality_l655_655389


namespace min_area_triangle_PJ1J2_l655_655854

theorem min_area_triangle_PJ1J2 {P Q R Y J1 J2 : Type} 
  [Incenter PQR P Q R Y J1 J2]
  (hPQ : dist P Q = 26)
  (hQR : dist Q R = 28)
  (hPR : dist P R = 30)
  (hY_in_QR : Y ∈ interior_segment Q R)
  (hJ1_incenter : is_incenter J1 (triangle P Q Y))
  (hJ2_incenter : is_incenter J2 (triangle P R Y)) :
  minimum_area (triangle P J1 J2) = 51 :=
sorry

end min_area_triangle_PJ1J2_l655_655854


namespace max_cardinality_valid_A_l655_655083

-- Define the conditions of the set A
def is_valid_A (A : Set ℕ) : Prop :=
  (∀ n ∈ A, n ≤ 2018) ∧
  (∀ S ⊆ A, S.card = 3 → ∃ m n ∈ S, |n - m| ≥ real.sqrt n + real.sqrt m)

-- Define the function to determine the maximum cardinality of the set A
def max_cardinality_A : ℕ := 44

-- The main theorem statement
theorem max_cardinality_valid_A (A : Set ℕ) : is_valid_A A → A.card ≤ max_cardinality_A := 
sorry

end max_cardinality_valid_A_l655_655083


namespace steve_oranges_count_l655_655294

variable (Brian_oranges Marcie_oranges Shawn_oranges Steve_oranges : ℝ)

def oranges_conditions : Prop :=
  (Marcie_oranges = 12) ∧
  (Brian_oranges = Marcie_oranges) ∧
  (Shawn_oranges = 1.075 * (Brian_oranges + Marcie_oranges)) ∧
  (Steve_oranges = 3 * (Marcie_oranges + Brian_oranges + Shawn_oranges))

theorem steve_oranges_count (h : oranges_conditions Brian_oranges Marcie_oranges Shawn_oranges Steve_oranges) :
  Steve_oranges = 149.4 :=
sorry

end steve_oranges_count_l655_655294


namespace fraction_problem_l655_655147

theorem fraction_problem (b : ℕ) (h₀ : 0 < b) (h₁ : (b : ℝ) / (b + 35) = 0.869) : b = 232 := 
by
  sorry

end fraction_problem_l655_655147


namespace a_2009_value_correct_l655_655411

def seq (n : ℕ) : ℕ :=
  if n = 1 then 0
  else seq (n - 1) + 1 + 2 * Int.sqrt (1 + seq (n - 1))

theorem a_2009_value_correct :
  seq 2009 = 4036080 := by
  sorry

end a_2009_value_correct_l655_655411


namespace product_of_palindromic_primes_less_than_100_l655_655897

def is_palindromic_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime (p % 10 * 10 + p / 10)

def palindromic_primes : List ℕ :=
  [11, 13, 17, 31, 37, 71, 73]

def product_of_palindromic_primes : ℕ :=
  List.prod palindromic_primes

theorem product_of_palindromic_primes_less_than_100 :
  product_of_palindromic_primes = 14458972331 := by
  sorry

end product_of_palindromic_primes_less_than_100_l655_655897


namespace solve_for_y_l655_655127

theorem solve_for_y (x y : ℝ) (h : 5 * x - y = 6) : y = 5 * x - 6 :=
sorry

end solve_for_y_l655_655127


namespace general_formula_for_seq_sum_of_first_n_terms_of_cn_l655_655366

open List

variables {a b c : ℕ → ℤ} (lg : ℤ → ℤ)
variables {T : ℕ → ℚ}

-- Conditions
axiom pos_seq_a (n : ℕ) : a n > 0
axiom recurrence_relation (n : ℕ) (hn : 2 ≤ n) : (a (n+1) : ℚ) / (a (n-1) : ℚ) + (a (n-1) : ℚ) / (a (n+1) : ℚ) = (4 * (a n)^2 : ℚ) / ((a (n+1) * a (n-1)) : ℚ) - 2
axiom a6_value : a 6 = 11
axiom sum_first_9_terms : (∑ i in range 9, a (i + 1)) = 81
axiom sum_lg_b (n : ℕ) : (∑ i in range n, lg (b (i + 1))) = lg (2 * n + 1)

-- Definition of c_n sequence
noncomputable def cn (n : ℕ) : ℚ := (a n * b n) / (2^(n+1))

-- Given condition for sum of c_n sequence
noncomputable def T_n (n : ℕ) : ℚ := (5 / 2 : ℚ) - (2 * ↑n + 5) / (2^(n+1))

-- The math proof problems
theorem general_formula_for_seq :
  ∀ n, a n = 2 * n - 1 := sorry

theorem sum_of_first_n_terms_of_cn :
  ∀ n, (∑ i in range n, cn i) = T_n n := sorry

end general_formula_for_seq_sum_of_first_n_terms_of_cn_l655_655366


namespace part1_part2_part3_l655_655970

section Part1

variables (a b : Real)

theorem part1 : 2 * (a + b)^2 - 8 * (a + b)^2 + 3 * (a + b)^2 = -3 * (a + b)^2 :=
by
  sorry

end Part1

section Part2

variables (x y : Real)

theorem part2 (h : x^2 + 2 * y = 4) : -3 * x^2 - 6 * y + 17 = 5 :=
by
  sorry

end Part2

section Part3

variables (a b c d : Real)

theorem part3 (h1 : a - 3 * b = 3) (h2 : 2 * b - c = -5) (h3 : c - d = 9) :
  (a - c) + (2 * b - d) - (2 * b - c) = 7 :=
by
  sorry

end Part3

end part1_part2_part3_l655_655970


namespace frequency_in_range_l655_655409

def sample_data_1 : List ℕ := [10, 8, 6, 10, 13, 8, 10, 12, 11, 7]
def sample_data_2 : List ℕ := [8, 9, 11, 9, 12, 9, 10, 11, 12, 12]
def sample_data : List ℕ := sample_data_1 ++ sample_data_2

def count_in_range (data : List ℕ) (lower upper : ℕ) : ℕ :=
  data.filter (λ x, lower <= x ∧ x <= upper).length

def freq : ℕ := ((sample_data.length) * 3 / 10).toNat

theorem frequency_in_range :
  count_in_range sample_data 8 9 = freq :=
sorry

end frequency_in_range_l655_655409


namespace sam_bought_nine_books_l655_655520

-- Definitions based on the conditions
def initial_money : ℕ := 79
def cost_per_book : ℕ := 7
def money_left : ℕ := 16

-- The amount spent on books
def money_spent_on_books : ℕ := initial_money - money_left

-- The number of books bought
def number_of_books (spent : ℕ) (cost : ℕ) : ℕ := spent / cost

-- Let x be the number of books bought and prove x = 9
theorem sam_bought_nine_books : number_of_books money_spent_on_books cost_per_book = 9 :=
by
  sorry

end sam_bought_nine_books_l655_655520


namespace filter_price_l655_655628

theorem filter_price (x : ℝ) :
  let kit_price := 72.50
  let price1 := 12.45
  let price3 := 11.50
  let saved_percentage := 0.1103448275862069
  let total_individual_price := 2 * price1 + 2 * x + price3
  let saved_amount := saved_percentage * total_individual_price
  (total_individual_price - kit_price = saved_amount) → (x = 22.55) := 
begin
  intros,
  sorry,
end

end filter_price_l655_655628


namespace concert_revenue_l655_655920

/-- Define problem conditions and variables -/
variables (attendance : ℕ) (cost_adult cost_child : ℝ) (num_adults : ℕ)

/-- Define the number of children -/
def num_children (attendance num_adults : ℕ) : ℕ := attendance - num_adults

/-- Define total money collected from adults -/
def total_from_adults (num_adults : ℕ) (cost_adult : ℝ) : ℝ := num_adults * cost_adult

/-- Define total money collected from children -/
def total_from_children (num_children : ℕ) (cost_child : ℝ) : ℝ := num_children * cost_child

/-- Define total money collected -/
def total_collected (attendance num_adults : ℕ) (cost_adult cost_child : ℝ) : ℝ :=
    total_from_adults num_adults cost_adult + total_from_children (num_children attendance num_adults) cost_child

/-- Prove that total money collected equals $1038 -/
theorem concert_revenue : total_collected 578 342 2.00 1.50 = 1038.00 :=
by
    sorry

end concert_revenue_l655_655920


namespace general_integral_of_differential_eqn_l655_655338

theorem general_integral_of_differential_eqn :
  ∃ C : ℝ, ∀ x y : ℝ, x = C * exp (y^2 / (2 * x^2)) ↔
    (x^2 + y^2) * (y' x = x * y) :=
begin
  sorry
end

end general_integral_of_differential_eqn_l655_655338


namespace number_with_exact_3_odd_factors_greater_than_1_l655_655564

noncomputable def odd_factors_greater_than_1 (n : ℕ) : ℕ := 
  (finset.filter (λ x, x > 1 ∧ x % 2 = 1) (finset.divisors n)).card

theorem number_with_exact_3_odd_factors_greater_than_1 : odd_factors_greater_than_1 9 = 2 :=
by
  sorry

end number_with_exact_3_odd_factors_greater_than_1_l655_655564


namespace find_question_mark_l655_655210

theorem find_question_mark : ∃ (x : ℕ), (√x / 19 = 4) ∧ (x = 5776) := by
  sorry

end find_question_mark_l655_655210


namespace tennis_handshakes_l655_655283

theorem tennis_handshakes :
  ∀ (teams : Fin 4 → Fin 2 → ℕ),
    (∀ i, teams i 0 ≠ teams i 1) ∧ (∀ i j a b, i ≠ j → teams i a ≠ teams j b) →
    ∃ handshakes : ℕ, handshakes = 24 :=
begin
  sorry
end

end tennis_handshakes_l655_655283


namespace external_tangent_length_l655_655178

-- Definitions
variables {x y R a : ℝ}

-- Conditions
axiom circle_tangency_conditions (hx : x > 0) (hy : y > 0) (hR : R > 0) (ha : a > 0) : True

-- Main theorem part a: 
theorem external_tangent_length 
  (hx : x > 0) (hy : y > 0) (hR : R > 0) (ha : a > 0) : 
  (((a / R) ^ 2) * (R + x) * (R + y) = 
  ((a / R) ^ 2) * (R - x) * (R - y) ∨ 
  ((a / R) ^ 2) * (R - y) * (R + x)) := 
by 
  sorry

end external_tangent_length_l655_655178


namespace unique_polynomial_value_l655_655958

noncomputable def Q (x : ℚ) : ℚ := x^4 - 12 * x^2 + 16

theorem unique_polynomial_value :
  (∀ Q : ℚ → ℚ, 
    (degree Q = 4) ∧ 
    (leading_coeff Q = 1) ∧
    (∀ x : ℚ, (x - (sqrt 3 + sqrt 7)) ∣ Q(x) ∧ (x - (sqrt 3 - sqrt 7)) ∣ Q(x)) ∧
    (∀ x : ℚ, Q(x) ∈ ℚ)) → 
  Q(1) = 5 :=
sorry

end unique_polynomial_value_l655_655958


namespace head_start_fraction_of_length_l655_655208

-- Define the necessary variables and assumptions.
variables (Va Vb L H : ℝ)

-- Given conditions
def condition_speed_relation : Prop := Va = (22 / 19) * Vb
def condition_dead_heat : Prop := (L / Va) = ((L - H) / Vb)

-- The statement to be proven
theorem head_start_fraction_of_length (h_speed_relation: condition_speed_relation Va Vb) (h_dead_heat: condition_dead_heat L Va H Vb) : 
  H = (3 / 22) * L :=
sorry

end head_start_fraction_of_length_l655_655208


namespace problem_divisibility_l655_655614

theorem problem_divisibility (k : ℤ) : 
  let n := 4 * k + 3 in 
  ∃ m : ℤ, 937^n + 11^(2*n) - 2^(n+1) * 69^n = 1955 * m :=
by {
  -- initial setup
  sorry
}

end problem_divisibility_l655_655614


namespace sum_of_distinct_prime_factors_of_2016_l655_655192

-- Define 2016 and the sum of its distinct prime factors
def n : ℕ := 2016
def sumOfDistinctPrimeFactors (n : ℕ) : ℕ :=
  if n = 2016 then 2 + 3 + 7 else 0  -- Capture the problem-specific condition

-- The main theorem to prove the sum of the distinct prime factors of 2016 is 12
theorem sum_of_distinct_prime_factors_of_2016 :
  sumOfDistinctPrimeFactors 2016 = 12 :=
by
  -- Since this is beyond the obvious steps, we use a sorry here
  sorry

end sum_of_distinct_prime_factors_of_2016_l655_655192


namespace inclination_angle_of_line_l655_655585

theorem inclination_angle_of_line (α : ℝ) (h : α ∈ set.Ico 0 real.pi) (slope : ℝ) (h_slope : slope = 1) (h_tan : real.tan α = slope) :
  α = real.pi / 4 :=
begin
  sorry
end

end inclination_angle_of_line_l655_655585


namespace minimum_trips_l655_655259

theorem minimum_trips (students buses capacity : ℕ) (h_students : students = 520) (h_buses : buses = 5) (h_capacity : capacity = 45) :
  let trips := (students + buses * capacity - 1) / (buses * capacity) in
  trips = 3 :=
by 
    -- Given conditions
    have h1 : students = 520 := h_students,
    have h2 : buses = 5 := h_buses,
    have h3 : capacity = 45 := h_capacity,

    -- Calculating per-trip capacity
    let per_trip_capacity := 5 * 45,

    -- Determining minimum trips
    let trips := (520 + per_trip_capacity - 1) / per_trip_capacity,

    -- Proving the required trips
    show trips = 3,

    sorry

end minimum_trips_l655_655259


namespace min_good_pairs_correct_l655_655251

def is_good (sheet : ℕ → ℕ → ℕ) (colors : Fin 23) (i j : ℕ × ℕ) : Prop :=
  (sheet i.fst i.snd = colors) ∧
  (i ≠ j ∧ ((abs (i.fst - j.fst) + abs (i.snd - j.snd)) = 1))

def min_good_pairs : ℕ :=
  22

theorem min_good_pairs_correct (sheet : ℕ → ℕ → ℕ) (colors : Fin 23) :
  (∃ pairs, (∀ p ∈ pairs, ∃ i j, p = (i, j) ∧ is_good sheet colors i j) ∧
    ∀i j, i ≠ j → is_good sheet colors i j → (i, j) ∈ pairs) →
  pairs.card = min_good_pairs := 
sorry

end min_good_pairs_correct_l655_655251


namespace problem_proof_l655_655758

def R (x : ℝ) : ℝ := 
if x = 0 then 1 else if ∃ (p : ℕ) (q : ℤ), (p > 0) ∧ (q ≠ 0) ∧ (nat.gcd p q.nat_abs = 1) ∧ (x = (q : ℚ) / p) 
then 1 / (λ ⟨p, q, _⟩, (p : ℝ)).val else 0

theorem problem_proof :
  (R (1 / 4) = R (3 / 4)) ∧ 
  (R (1 / 5) = R (6 / 5)) ∧ 
  (∀ x : ℝ, R (-x) = R x) ∧ 
  (∀ x : ℝ, R (x + 1) = R x) :=
by {
  sorry -- proof goes here
}

end problem_proof_l655_655758


namespace divisors_180_6_l655_655014

theorem divisors_180_6 : 
  let n := 180
  let a := 2
  let b := 3
  let c := 5 
  let n_exp := (a*a) * (b*b) * c
  let N := n_exp ^ 6
  (N = (2^12) * (3^12) * (5^6)) →
  let perfect_squares := 7 * 7 * 4
  let perfect_cubes := 5 * 5 * 3
  let perfect_sixths := 3 * 3 * 2
  253 = perfect_squares + perfect_cubes - perfect_sixths :=
by
  assume hN_exp,
  sorry

end divisors_180_6_l655_655014


namespace percentage_closest_to_25_l655_655885

-- Definitions of prices and total bill
def item_prices : List ℝ := [2.50, 3.75, 6.25, 9.50, 10.25, 5.00]
def total_bill : ℝ := 50.00

-- Definition of total cost of items
def total_cost : ℝ := item_prices.foldl (λ acc price => acc + price) 0

-- Definition of change
def change : ℝ := total_bill - total_cost

-- Definition of change percentage
def change_percentage : ℝ := (change / total_bill) * 100

-- Statement to prove that the percentage of the $50.00 that Mary receives in change is closest to $25\%
theorem percentage_closest_to_25 (h_total_valid: total_cost = 37.25) (h_change_valid: change = 12.75) : 
  ((change_percentage - 25).abs ≤ (change_percentage - x).abs ∀ x ∈ [20, 22, 25, 26, 28] := 
sorry

end percentage_closest_to_25_l655_655885


namespace value_of_k_l655_655938

theorem value_of_k (k : ℕ) (n : ℕ) (h : 8 * (10^k - 1) / 9) 
  (hsum : n = 7 + 9 * (k - 2) + 9 + 2) : 
  n = 2000 → k = 221 := 
by
  sorry

end value_of_k_l655_655938


namespace find_d_e_f_l655_655490

noncomputable def y : ℝ := Real.sqrt ((Real.sqrt 77) / 2 + 5 / 2)
def y_squared : ℝ := (Real.sqrt 77) / 2 + 5 / 2
def y_fourth_power : ℝ := 5 * y_squared + 13

theorem find_d_e_f :
  ∃ d e f : ℕ,
    (y ^ 100 = 2 * y ^ 98 + 18 * y ^ 96 + 15 * y ^ 94 - y ^ 50 + d * y ^ 46 + e * y ^ 44 + f * y ^ 40) ∧
    (d + e + f = 242) := sorry

end find_d_e_f_l655_655490


namespace problem_theorem_l655_655386

-- Assuming we have a geometric sequence {a_n} with common ratio q > 0,
-- and T_n denotes the product of the first n terms, we aim to prove:
-- 0 < q < 1 and T_13 > 1 > T_14 under the given condition T_7 > T_6 > T_8.

variable (a : ℕ → ℝ) (q : ℝ) (T : ℕ → ℝ) (h_q_pos : q > 0)

-- Definition of geometric sequence
def is_geometric_sequence : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Definition of T_n as the product of the first n terms
def product_first_n_terms (n : ℕ) : Prop :=
  T n = ∏ i in finset.range n, a (i + 1)

-- Condition given in the problem
def condition : Prop :=
  T 7 > T 6 ∧ T 6 > T 8

-- Theorem to prove the required conditions
theorem problem_theorem 
  (geo_seq : is_geometric_sequence a q) 
  (prod_terms : ∀ n, product_first_n_terms a T n)
  (cond : condition T) :
  0 < q ∧ q < 1 ∧ T 13 > 1 ∧ T 14 < 1 :=
  by
    sorry

end problem_theorem_l655_655386


namespace current_length_of_highway_l655_655639

def total_length : ℕ := 650
def miles_first_day : ℕ := 50
def miles_second_day : ℕ := 3 * miles_first_day
def miles_still_needed : ℕ := 250
def miles_built : ℕ := miles_first_day + miles_second_day

theorem current_length_of_highway :
  total_length - miles_still_needed = 400 :=
by
  sorry

end current_length_of_highway_l655_655639


namespace water_charge_rel_water_usage_from_charge_l655_655438

-- Define the conditions and functional relationship
theorem water_charge_rel (x : ℝ) (hx : x > 5) : y = 3.5 * x - 7.5 :=
  sorry

-- Prove the specific case where the charge y is 17 yuan
theorem water_usage_from_charge (h : 17 = 3.5 * x - 7.5) :
  x = 7 :=
  sorry

end water_charge_rel_water_usage_from_charge_l655_655438


namespace projectile_hits_ground_at_5_l655_655544

-- Definition of the height function for the projectile
def height (t : ℝ) : ℝ := -6.1 * t^2 + 7 * t + 10

-- The statement of the problem: Prove that the projectile hits the ground at t = 5 seconds
theorem projectile_hits_ground_at_5 : (∃ t : ℝ, height t = 0 ∧ t = 5) :=
  sorry

end projectile_hits_ground_at_5_l655_655544


namespace digits_of_product_l655_655416

def number_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else (Nat.log10 (n + 1) + 1)

theorem digits_of_product :
  number_of_digits (6 ^ 3 * 3 ^ 9) = 7 :=
by
  sorry

end digits_of_product_l655_655416


namespace proper_subsets_cardinality_l655_655556

theorem proper_subsets_cardinality : 
  let s := {1, 2, 3}
  in { t // t ⊂ s }.to_finset.card = 7 :=
by
  sorry

end proper_subsets_cardinality_l655_655556


namespace arrangements_count_l655_655954

-- Definitions to set up the problem:
def boys : set ℕ := {1, 2, 3}
def girls : set ℕ := {4, 5, 6}
def A : ℕ := 1
def B : ℕ := 4
def is_boy (x : ℕ) : Prop := x ∈ boys
def is_girl (x : ℕ) : Prop := x ∈ girls
def is_adjacent (x y : ℕ) (lst : list ℕ) : Prop := (x ∈ lst) ∧ (y ∈ lst) ∧ 
                                              (list.index_of x lst = list.index_of y lst + 1 ∨ 
                                               list.index_of x lst = list.index_of y lst - 1)

-- Conditions:
def conditions (lst : list ℕ) : Prop := 
    lst.length = 6 ∧
    ∀ i < 5, (is_boy (lst.nth i).iget ∧ is_girl (lst.nth (i+1)).iget) ∨
             (is_girl (lst.nth i).iget ∧ is_boy (lst.nth (i+1)).iget) ∧
    is_adjacent A B lst

-- The proof statement:
theorem arrangements_count : (∃ lst : list ℕ, conditions lst) ↔ 40 :=
by sorry

end arrangements_count_l655_655954


namespace minimum_value_l655_655489

noncomputable def min_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : ℝ :=
  (x + y + z) * (1 / (x + y) + 1 / (x + z) + 1 / (y + z))

theorem minimum_value : ∀ x y z : ℝ, 0 < x → 0 < y → 0 < z →
  (x + y + z) * (1 / (x + y) + 1 / (x + z) + 1 / (y + z)) ≥ 9 / 2 :=
by
  intro x y z hx hy hz
  sorry

end minimum_value_l655_655489


namespace triangle_overlap_angle_is_30_l655_655965

noncomputable def triangle_rotation_angle (hypotenuse : ℝ) (overlap_ratio : ℝ) :=
  if hypotenuse = 10 ∧ overlap_ratio = 0.5 then 30 else sorry

theorem triangle_overlap_angle_is_30 :
  triangle_rotation_angle 10 0.5 = 30 :=
sorry

end triangle_overlap_angle_is_30_l655_655965


namespace price_increase_needed_to_restore_l655_655159

-- Let's define the original price
def original_price : ℝ := 100

-- The series of reductions and increases
def reduced_once (P : ℝ) : ℝ := P - (P * 0.25)
def reduced_twice (P : ℝ) : ℝ := reduced_once P - (reduced_once P * 0.25)
def increased_after_sale (P : ℝ) : ℝ := reduced_twice P + (reduced_twice P * 0.15)
def final_price_with_tax (P : ℝ) : ℝ := increased_after_sale P + (increased_after_sale P * 0.1)

-- Define the final price after all operations
def final_price := final_price_with_tax original_price

-- To restore to original price
def required_percentage_increase (F P: ℝ) : ℝ := ((P / F) - 1) * 100

-- Statement to prove that approximately 40.5% increase is required to restore the original price
theorem price_increase_needed_to_restore :
  abs (required_percentage_increase final_price original_price - 40.5) < 0.1 :=
sorry

end price_increase_needed_to_restore_l655_655159


namespace lcm_5_6_8_9_l655_655189

theorem lcm_5_6_8_9 : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 9) = 360 := by
  sorry

end lcm_5_6_8_9_l655_655189


namespace initial_sum_simple_interest_l655_655650

theorem initial_sum_simple_interest :
  ∃ P : ℝ, (P * (3/100) + P * (5/100) + P * (4/100) + P * (6/100) = 100) ∧ (P = 5000 / 9) :=
by
  sorry

end initial_sum_simple_interest_l655_655650


namespace part1_part2_l655_655693

def op (a b : ℝ) : ℝ := 2 * a - 3 * b

theorem part1 : op (-2) 3 = -13 :=
by
  -- sorry step to skip proof
  sorry

theorem part2 (x : ℝ) :
  let A := op (3 * x - 2) (x + 1)
  let B := op (-3 / 2 * x + 1) (-1 - 2 * x)
  B > A :=
by
  -- sorry step to skip proof
  sorry

end part1_part2_l655_655693


namespace locus_of_N_l655_655631

theorem locus_of_N (x y : ℝ) :
  let O := (0, 0) in
  ∃ A : ℝ × ℝ, 
    (O.1 - A.1)^2 + (O.2 - A.2)^2 = (x - A.1)^2 + (y - A.2)^2 ∧
    A.1 * A.1 + A.2 * A.2 - 8 * A.1 = 0 →
  x^2 + y^2 - 16 * x = 0 :=
by 
  intro O A hA hcircle
  sorry

end locus_of_N_l655_655631


namespace evaluate_expression_1_evaluate_expression_2_l655_655621

-- Problem 1
def expression_1 (a b : Int) : Int :=
  2 * a + 3 * b - 2 * a * b - a - 4 * b - a * b

theorem evaluate_expression_1 : expression_1 6 (-1) = 25 :=
by
  sorry

-- Problem 2
def expression_2 (m n : Int) : Int :=
  m^2 + 2 * m * n + n^2

theorem evaluate_expression_2 (m n : Int) (hm : |m| = 3) (hn : |n| = 2) (hmn : m < n) : expression_2 m n = 1 :=
by
  sorry

end evaluate_expression_1_evaluate_expression_2_l655_655621


namespace constant_term_in_expansion_l655_655336

theorem constant_term_in_expansion :
  let f (x : ℂ) := (√x + (1 : ℂ) / (2 * √x))^8 in
  constant_term (f x) = 35 / 8 :=
sorry

end constant_term_in_expansion_l655_655336


namespace cot_pi_over_18_minus_3_sin_2pi_over_9_eq_zero_l655_655675

theorem cot_pi_over_18_minus_3_sin_2pi_over_9_eq_zero :
  (Real.cot (Real.pi / 18) - 3 * Real.sin (2 * Real.pi / 9)) = 0 :=
by
  sorry

end cot_pi_over_18_minus_3_sin_2pi_over_9_eq_zero_l655_655675


namespace math_club_team_selection_l655_655154

def num_combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

def num_ways_to_form_team_with_at_least_4_girls (boys girls team_size : ℕ) : ℕ :=
  num_combinations girls 4 * num_combinations boys 4 +
  num_combinations girls 5 * num_combinations boys 3 +
  num_combinations girls 6 * num_combinations boys 2 +
  num_combinations girls 7 * num_combinations boys 1 +
  num_combinations girls 8

theorem math_club_team_selection : 
  num_ways_to_form_team_with_at_least_4_girls 10 12 8 = 245985 :=
by sorry

end math_club_team_selection_l655_655154


namespace john_adds_and_subtracts_l655_655577

theorem john_adds_and_subtracts :
  (41^2 = 40^2 + 81) ∧ (39^2 = 40^2 - 79) :=
by {
  sorry
}

end john_adds_and_subtracts_l655_655577


namespace sin_double_theta_l655_655793

theorem sin_double_theta (θ : ℝ) (h₁ : cos (π / 4 - θ) * cos (π / 4 + θ) = sqrt 2 / 6) (h₂ : 0 < θ ∧ θ < π / 2) :
  sin (2 * θ) = sqrt 7 / 3 :=
by
  sorry

end sin_double_theta_l655_655793


namespace number_with_exact_3_odd_factors_greater_than_1_l655_655563

noncomputable def odd_factors_greater_than_1 (n : ℕ) : ℕ := 
  (finset.filter (λ x, x > 1 ∧ x % 2 = 1) (finset.divisors n)).card

theorem number_with_exact_3_odd_factors_greater_than_1 : odd_factors_greater_than_1 9 = 2 :=
by
  sorry

end number_with_exact_3_odd_factors_greater_than_1_l655_655563


namespace hexagon_chord_length_valid_l655_655246

def hexagon_inscribed_chord_length : ℚ := 48 / 49

theorem hexagon_chord_length_valid : 
    ∃ (p q : ℕ), gcd p q = 1 ∧ hexagon_inscribed_chord_length = p / q ∧ p + q = 529 :=
sorry

end hexagon_chord_length_valid_l655_655246


namespace degree_inequality_l655_655482

theorem degree_inequality (n : ℕ) (hn : n ≥ 2)
  (P : Fin n → Polynomial ℝ)
  (hpos : ∀ i, (P i).leadingCoeff > 0)
  (hnot_all_same : ¬ (∀ i j, P i = P j)) :
  ∃ i, (Polynomial.degree (Finset.sum Finset.univ (λ i, (P i) ^ n) - n * Polynomial.prod P)) ≥
    (n - 2) * (Polynomial.degree (P i)) :=
sorry

end degree_inequality_l655_655482


namespace ellipse_eccentricity_l655_655393

theorem ellipse_eccentricity (a : ℝ) :
  (∀ x y : ℝ, (x^2) / (a^2) + (y^2) / 16 = 1) ∧ (∃ e : ℝ, e = 3 / 4) ∧ (∀ c : ℝ, c = 3 / 4)
   → a = 7 :=
by
  sorry

end ellipse_eccentricity_l655_655393


namespace no_solution_sqrt_eq_l655_655596

theorem no_solution_sqrt_eq (x : ℝ) (h1 : x + 1 ≥ 0) (h2 : 3 - x ≥ 0) : √(x + 1) + √(3 - x) ≠ 17 := 
by sorry

end no_solution_sqrt_eq_l655_655596


namespace point_in_second_quadrant_l655_655050

theorem point_in_second_quadrant (a : ℝ) : ∃ y : ℝ, y = a^2 + 1 ∧ (-2 < 0 ∧ y > 0) := 
by
  exists a^2 + 1
  constructor
  rfl
  constructor
  apply neg_lt_zero
  exact lt_add_of_pos_right (a^2) zero_lt_one
  sorry

end point_in_second_quadrant_l655_655050


namespace probability_increasing_function_l655_655748

def f (x m : ℝ) : ℝ := (1 / 3) * x ^ 3 - 2 * x ^ 2 + m ^ 2 * x + 3

def f_prime (x m : ℝ) : ℝ := x ^ 2 - 4 * x + m ^ 2

theorem probability_increasing_function :
  let I := set.Icc 0 4,
      S := { m : ℝ | 2 ≤ m ∧ m ≤ 4 }
  in
  (set.volume S) / (set.volume I) = 1 / 2 :=
sorry

end probability_increasing_function_l655_655748


namespace f_geq_expression_l655_655395

noncomputable def f (x a : ℝ) : ℝ := x^2 + (2 * a - 1 / a) * x - Real.log x

theorem f_geq_expression (a x : ℝ) (h : a < 0) : f x a ≥ (1 - 2 * a) * (a + 1) := 
  sorry

end f_geq_expression_l655_655395


namespace functional_equation_solution_l655_655702

open Function

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f (x ^ 2 + f y) = y + f x ^ 2) → (∀ x : ℝ, f x = x) :=
by
  sorry

end functional_equation_solution_l655_655702


namespace hyperbola_asymptotes_l655_655402

variable (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)

theorem hyperbola_asymptotes (e : ℝ) (h_ecc : e = (Real.sqrt 5) / 2)
  (h_hyperbola : e = Real.sqrt (1 + (b^2 / a^2))) :
  (∀ x : ℝ, y = x * (b / a) ∨ y = -x * (b / a)) :=
by
  -- Here, the proof would follow logically from the given conditions.
  sorry

end hyperbola_asymptotes_l655_655402


namespace sally_lost_two_balloons_l655_655519

-- Condition: Sally originally had 9 orange balloons.
def original_orange_balloons := 9

-- Condition: Sally now has 7 orange balloons.
def current_orange_balloons := 7

-- Problem: Prove that Sally lost 2 orange balloons.
theorem sally_lost_two_balloons : original_orange_balloons - current_orange_balloons = 2 := by
  sorry

end sally_lost_two_balloons_l655_655519


namespace inner_rectangle_length_l655_655809

theorem inner_rectangle_length :
  ∃ x : ℝ, 
    let A_inner := 2 * x,
        A_middle := 4 * (x + 2),
        A_outer := 6 * (x + 4) in
    ((A_middle - A_inner) = (A_outer - A_middle)) ∧ x = 8 :=
by
  sorry

end inner_rectangle_length_l655_655809


namespace find_b_of_sin_l655_655663

theorem find_b_of_sin (a b c d : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
                       (h_period : (2 * Real.pi) / b = Real.pi / 2) : b = 4 := by
  sorry

end find_b_of_sin_l655_655663


namespace option_B_is_one_variable_quadratic_l655_655198

theorem option_B_is_one_variable_quadratic :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x : ℝ, 2 * (x - x^2) - 1 = a * x^2 + b * x + c) :=
by
  sorry

end option_B_is_one_variable_quadratic_l655_655198


namespace hexagon_chord_problem_solution_l655_655244

noncomputable def hexagon_chord_length {p q : ℕ} (hpq_coprime : Nat.coprime p q) : ℕ :=
  let a := 4
  let b := 6
  let circ_hex := inscribed_hexagon_in_circle a b
  let chord := circ_hex.divides_into_trapezoids
  if (chord.len = p / q) then p + q else 0

theorem hexagon_chord_problem_solution :
  ∃ (p q : ℕ), Nat.coprime p q ∧ hexagon_chord_length Nat.Coprime p q = 799 :=
begin
  sorry
end

end hexagon_chord_problem_solution_l655_655244


namespace price_per_gallon_of_milk_l655_655117

theorem price_per_gallon_of_milk 
  (daily_production : ℕ)
  (days_in_june : ℕ)
  (total_income : ℕ)
  (total_gallons : ℕ) :
  daily_production = 200 →
  days_in_june = 30 →
  total_income = 18300 →
  total_gallons = daily_production * days_in_june →
  total_income / total_gallons = 3.05 := 
by
  sorry

end price_per_gallon_of_milk_l655_655117


namespace length_of_smallest_room_l655_655929

theorem length_of_smallest_room :
  ∀ (width_large length_large width_small : ℕ) (area_difference : ℕ),
    width_large = 45 →
    length_large = 30 →
    width_small = 15 →
    area_difference = 1230 →
    let area_large := width_large * length_large in
    let area_small := area_large - area_difference in
    let length_small := area_small / width_small in
    length_small = 8 := by
  intros width_large length_large width_small area_difference
  intros h1 h2 h3 h4
  let area_large := width_large * length_large
  let area_small := area_large - area_difference
  let length_small := area_small / width_small
  sorry

end length_of_smallest_room_l655_655929


namespace marketing_strategy_increases_sales_of_B_l655_655454

-- Definitions of the products with their characteristics
structure Product where
  quality : ℤ
  price : ℤ
  likelihood_of_purchase : ℤ

-- Instances for each product
def product_A : Product := {
  quality := 10,    -- Placeholder value for high quality
  price := 100,    -- Placeholder value for high price
  likelihood_of_purchase := 1 -- Placeholder value for low likelihood of purchase
}

def product_B : Product := {
  quality := 8,    -- Placeholder value for slightly inferior quality
  price := 60,     -- Placeholder value for significantly lower price
  likelihood_of_purchase := 0 -- Placeholder to be updated based on conditions
}

def product_C : Product := {
  quality := 5,    -- Placeholder value for economy quality
  price := 30,     -- Placeholder value for economy price
  likelihood_of_purchase := 0 -- Placeholder, not relevant for this proof
}

-- Hypothesis based on the problem statement
axiom H1 : product_A.likelihood_of_purchase < product_B.likelihood_of_purchase
axiom H2 : product_B.price < product_A.price
axiom H3 : product_C.price < product_B.price
axiom H4 : product_B.price > average_market_price

-- The theorem that represents the proof problem
theorem marketing_strategy_increases_sales_of_B :
  product_A.likelihood_of_purchase < product_B.likelihood_of_purchase ∧
  (product_C.price < product_B.price ∨ product_B.price > average_market_price) →
  product_B.likelihood_of_purchase > product_A.likelihood_of_purchase :=
by sorry

end marketing_strategy_increases_sales_of_B_l655_655454


namespace ammunition_explosion_probability_l655_655624

theorem ammunition_explosion_probability :
  (let P_A : ℝ := 0.025,
       P_B : ℝ := 0.1,
       P_C : ℝ := 0.1,
       P_D := 1 - (1 - P_A) * (1 - P_B) * (1 - P_C) in
   P_D = 0.21025) :=
by sorry

end ammunition_explosion_probability_l655_655624


namespace find_remainder_mod_105_l655_655123

-- Define the conditions as a set of hypotheses
variables {n a b c : ℕ}
variables (hn : n > 0)
variables (ha : a < 3) (hb : b < 5) (hc : c < 7)
variables (h3 : n % 3 = a) (h5 : n % 5 = b) (h7 : n % 7 = c)
variables (heq : 4 * a + 3 * b + 2 * c = 30)

-- State the theorem
theorem find_remainder_mod_105 : n % 105 = 29 :=
by
  -- Hypotheses block for documentation
  have ha_le : 0 ≤ a := sorry
  have hb_le : 0 ≤ b := sorry
  have hc_le : 0 ≤ c := sorry
  sorry

end find_remainder_mod_105_l655_655123


namespace triangle_AEF_area_l655_655680

open Set

-- Define the points on the plane
structure Point := (x : ℝ) (y : ℝ)

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)

-- Define the area of a right triangle given base and height
def area_right_triangle (base height : ℝ) : ℝ :=
  (1/2) * base * height

-- Given the points and segments in the problem
def A : Point := ⟨0, 12⟩
def B : Point := ⟨0, 0⟩
def C : Point := ⟨8, 0⟩
def D : Point := ⟨0, 8⟩
def E : Point := ⟨0, 4⟩ -- E is the midpoint of BD
def F : Point := ⟨4, 0⟩

-- Verify the lengths and definitions match the problem description
-- (assumptions, no real checking/asserting done here)
def AB_height := distance A B = 12
def BD_height := distance B D = 8
def BF_height := distance B F = 8
def BE_length  := distance B E = 4
def EF_base := distance E F = 4

-- Define the area of triangle AEF
def area_triangle_AEF :=
  area_right_triangle 4 12

theorem triangle_AEF_area : 
  area_triangle_AEF = 24 := by
  sorry

end triangle_AEF_area_l655_655680


namespace find_norm_b_l655_655382

variable {a b : EuclideanSpace ℝ}

noncomputable def angle_between (a b : EuclideanSpace ℝ) : ℝ :=
  real.acos ((dot_product a b) / (norm a * norm b))

theorem find_norm_b (ha : ‖a‖ = 2) (dot_ab : dot_product a b = 1) (theta : angle_between a b = real.pi / 3) :
  ‖b‖ = 1 := by
  sorry

end find_norm_b_l655_655382


namespace technicians_count_l655_655816

theorem technicians_count (
    workers_total : ℕ,
    avg_salary_all : ℕ,
    avg_salary_tech : ℕ,
    avg_salary_rest : ℕ,
    total_salary : ℕ,
    technicians : ℕ,
    rest : ℕ,
    h1 : workers_total = 28,
    h2 : avg_salary_all = 8000,
    h3 : avg_salary_tech = 14000,
    h4 : avg_salary_rest = 6000,
    h5 : total_salary = avg_salary_all * workers_total,
    h6 : total_salary = avg_salary_tech * technicians + avg_salary_rest * rest,
    h7 : technicians + rest = workers_total
  ) : technicians = 7 :=
by 
    sorry

end technicians_count_l655_655816


namespace count_primes_1021_eq_one_l655_655726

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_primes_1021_eq_one :
  (∃ n : ℕ, 3 ≤ n ∧ is_prime (n^3 + 2*n + 1) ∧
  ∀ m : ℕ, (3 ≤ m ∧ m ≠ n) → ¬ is_prime (m^3 + 2*m + 1)) :=
sorry

end count_primes_1021_eq_one_l655_655726


namespace initial_balloons_blown_up_l655_655924
-- Import necessary libraries

-- Define the statement
theorem initial_balloons_blown_up (x : ℕ) (hx : x + 13 = 60) : x = 47 :=
by
  sorry

end initial_balloons_blown_up_l655_655924


namespace perimeter_of_square_l655_655221

theorem perimeter_of_square
  (s : ℝ) -- s is the side length of the square
  (h_divided_rectangles : ∀ r, r ∈ {r : ℝ × ℝ | r = (s, s / 6)} → true) -- the square is divided into six congruent rectangles
  (h_perimeter_rect : 2 * (s + s / 6) = 42) -- the perimeter of each of these rectangles is 42 inches
  : 4 * s = 72 := 
sorry

end perimeter_of_square_l655_655221


namespace double_even_l655_655861

-- Define even function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Lean statement of the mathematically equivalent proof problem
theorem double_even (f : ℝ → ℝ) (h : is_even_function f) : is_even_function (f ∘ f) :=
by
  sorry

end double_even_l655_655861


namespace integer_values_count_of_x_l655_655797

theorem integer_values_count_of_x (x : ℝ) (h : ⌈real.sqrt x⌉ = 17) : ∃ n : ℕ, n = 33 :=
by
  let lower_bound := 256
  let upper_bound := 289
  have h1 : lower_bound < x,
  have h2 : x ≤ upper_bound,
  let count := upper_bound - lower_bound + 1
  use count
  sorry

end integer_values_count_of_x_l655_655797


namespace typical_dosage_l655_655636

theorem typical_dosage (prescribed_dosage : ℝ) (body_weight : ℝ) (percentage : ℝ) (ratio : ℝ) :
  (prescribed_dosage = 12) →
  (body_weight = 120) →
  (percentage = 0.75) →
  (ratio = 15) →
  let D := (prescribed_dosage / (percentage * (body_weight / ratio))) in
  D = 2 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  let D := (12 / (0.75 * (120 / 15)))
  have : D = 2 := by sorry
  exact this

end typical_dosage_l655_655636


namespace total_pages_in_book_l655_655602

theorem total_pages_in_book (P : ℕ) 
  (h1 : 7 / 13 * P = P - 96 - 5 / 9 * (P - 7 / 13 * P))
  (h2 : 96 = 4 / 9 * (P - 7 / 13 * P)) : 
  P = 468 :=
 by 
    sorry

end total_pages_in_book_l655_655602


namespace log_inequality_condition_l655_655747

theorem log_inequality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a > b ↔ ln a > ln b :=
begin
  split,
  { intro h,
    exact Real.log_lt_log_out ha hb h, },
  { intro h,
    exact Real.log_lt_log_out ha hb h, }
end

end log_inequality_condition_l655_655747


namespace dihedral_angle_of_cube_l655_655827

-- Given definitions and conditions
def cube (A B C D A' B' C' D' : ℝ^3) : Prop := sorry
def dihedral_angle (plane1 plane2 : set ℝ^3) : ℝ := sorry

-- Lean statement
theorem dihedral_angle_of_cube (A B C D A' B' C' D' : ℝ^3)
  (h_cube : cube A B C D A' B' C' D')
  : dihedral_angle {p | ∃ (α β γ : ℝ), p = α • A' + β • B + γ • D} {p | ∃ (μ ν : ℝ), p = μ • C'} = Real.arccos (1/3) :=
sorry

end dihedral_angle_of_cube_l655_655827


namespace technicians_count_l655_655817

theorem technicians_count (
    workers_total : ℕ,
    avg_salary_all : ℕ,
    avg_salary_tech : ℕ,
    avg_salary_rest : ℕ,
    total_salary : ℕ,
    technicians : ℕ,
    rest : ℕ,
    h1 : workers_total = 28,
    h2 : avg_salary_all = 8000,
    h3 : avg_salary_tech = 14000,
    h4 : avg_salary_rest = 6000,
    h5 : total_salary = avg_salary_all * workers_total,
    h6 : total_salary = avg_salary_tech * technicians + avg_salary_rest * rest,
    h7 : technicians + rest = workers_total
  ) : technicians = 7 :=
by 
    sorry

end technicians_count_l655_655817


namespace fifteenth_term_arithmetic_sequence_l655_655951

theorem fifteenth_term_arithmetic_sequence (a d : ℤ) : 
  (a + 20 * d = 17) ∧ (a + 21 * d = 20) → (a + 14 * d = -1) := by
  sorry

end fifteenth_term_arithmetic_sequence_l655_655951


namespace time_to_ascend_non_working_escalator_l655_655581

-- Define the variables as given in the conditions
def V := 1 / 60 -- Speed of the moving escalator in units per minute
def U := (1 / 24) - (1 / 60) -- Speed of Gavrila running relative to the escalator

-- Theorem stating that the time to ascend a non-working escalator is 40 seconds
theorem time_to_ascend_non_working_escalator : 
  (1 : ℚ) = U * (40 / 60) := 
by sorry

end time_to_ascend_non_working_escalator_l655_655581


namespace part1_part2_l655_655692

def op (a b : ℝ) : ℝ := 2 * a - 3 * b

theorem part1 : op (-2) 3 = -13 :=
by
  -- sorry step to skip proof
  sorry

theorem part2 (x : ℝ) :
  let A := op (3 * x - 2) (x + 1)
  let B := op (-3 / 2 * x + 1) (-1 - 2 * x)
  B > A :=
by
  -- sorry step to skip proof
  sorry

end part1_part2_l655_655692


namespace circle_intersection_distance_l655_655923

-- Define the properties of the first circle
def circle1_center : ℝ × ℝ := (2, 0)
def circle1_radius : ℝ := 3

-- Define the properties of the second circle
def circle2_center : ℝ × ℝ := (5, 0)
def circle2_radius : ℝ := 5

-- Define the square of the distance between the intersection points C and D
def CD_squared (C D : ℝ × ℝ) : ℝ := (C.1 - D.1)^2 + (C.2 - D.2)^2

-- The proof statement that needs to be proven
theorem circle_intersection_distance :
  ∀ (C D : ℝ × ℝ),
  (C ≠ D) →
  ((C.1 - circle1_center.1)^2 + (C.2 - circle1_center.2)^2 = circle1_radius^2) →
  ((C.1 - circle2_center.1)^2 + (C.2 - circle2_center.2)^2 = circle2_radius^2) →
  ((D.1 - circle1_center.1)^2 + (D.2 - circle1_center.2)^2 = circle1_radius^2) →
  ((D.1 - circle2_center.1)^2 + (D.2 - circle2_center.2)^2 = circle2_radius^2) →
  CD_squared C D = 50 :=
begin
  sorry -- proof goes here
end

end circle_intersection_distance_l655_655923


namespace lcm_5_6_8_9_l655_655188

theorem lcm_5_6_8_9 : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 9) = 360 := by
  sorry

end lcm_5_6_8_9_l655_655188


namespace calculate_expression_l655_655669

variable (a : ℝ)

theorem calculate_expression : 2 * a - 7 * a + 4 * a = -a := by
  sorry

end calculate_expression_l655_655669


namespace bill_due_in_months_l655_655949

noncomputable def true_discount_time (TD A R : ℝ) : ℝ :=
  let P := A - TD
  let T := TD / (P * R / 100)
  12 * T

theorem bill_due_in_months :
  ∀ (TD A R : ℝ), TD = 189 → A = 1764 → R = 16 →
  abs (true_discount_time TD A R - 10.224) < 1 :=
by
  intros TD A R hTD hA hR
  sorry

end bill_due_in_months_l655_655949


namespace concurrency_of_inscribed_square_centers_l655_655865

theorem concurrency_of_inscribed_square_centers
  (A B C A₁ B₁ C₁ : Point)
  (hA₁ : is_center_in_square_on_triangle_side A₁ A B C)
  (hB₁ : is_center_in_square_on_triangle_side B₁ B C A)
  (hC₁ : is_center_in_square_on_triangle_side C₁ C A B) :
  concurrent AA₁ BB₁ CC₁ :=
sorry

end concurrency_of_inscribed_square_centers_l655_655865


namespace hexagon_perimeter_l655_655213

-- Definitions of the conditions
def side_length : ℕ := 5
def number_of_sides : ℕ := 6

-- The perimeter of the hexagon
def perimeter : ℕ := side_length * number_of_sides

-- Proof statement
theorem hexagon_perimeter : perimeter = 30 :=
by
  sorry

end hexagon_perimeter_l655_655213


namespace Chris_age_l655_655537

theorem Chris_age 
  (a b c : ℝ)
  (h1 : a + b + c = 36)
  (h2 : c - 5 = a)
  (h3 : b + 4 = (3 / 4) * (a + 4)) :
  c = 15.5454545454545 :=
by
  sorry

end Chris_age_l655_655537


namespace volume_of_extended_region_equals_l655_655308

def volume_of_parallelepiped (length width height : ℝ) : ℝ :=
  length * width * height

def volume_of_extended_region (length width height : ℝ) : ℝ := 
  let original_volume := volume_of_parallelepiped length width height
  let external_slabs_volume := 2 * (length * width + length * height + width * height)
  let eighth_spheres_volume := (8 * (1 / 8) * (4 / 3) * Real.pi * (1 ^ 3))
  let quarter_cylinders_volume := 
    (4 * Real.pi) + 
    (4 * (width / 2) * Real.pi) + 
    (4 * height * Real.pi / 2)
  original_volume + external_slabs_volume + 
    eighth_spheres_volume + 
    quarter_cylinders_volume

theorem volume_of_extended_region_equals :
  volume_of_extended_region 2 3 4 = 76 + (31 * Real.pi / 6) := by
  sorry

end volume_of_extended_region_equals_l655_655308


namespace find_both_artifacts_total_time_l655_655074

variables (months_in_year : Nat) (expedition_first_years : Nat) (artifact_factor : Nat)

noncomputable def total_time (research_months : Nat) (expedition_first_years : Nat) :=
  let research_first_years := float_of_nat research_months / float_of_nat months_in_year
  let total_first := research_first_years + float_of_nat expedition_first_years
  let total_second := artifact_factor * total_first
  total_first + total_second

theorem find_both_artifacts_total_time :
  forall (months_in_year : Nat) (expedition_first_years : Nat) (artifact_factor : Nat),
    months_in_year = 12 → 
    expedition_first_years = 2 → 
    artifact_factor = 3 → 
    total_time 6 expedition_first_years = 10 :=
by intros months_in_year expedition_first_years artifact_factor hm he hf
   unfold total_time 
   sorry

end find_both_artifacts_total_time_l655_655074


namespace centroid_area_trace_l655_655112

theorem centroid_area_trace {circle : Type} [metric_space circle] 
    (O A B C : circle)
    (h1 : dist A B = 36)
    (h2 : dist O A = dist O B)
    (h3 : ∃ r : ℝ, dist O C = r ∧ r ≠ 0 ∧ r ≠ dist O A)
    (h_condition : ∃ θ : ℝ, θ < π ∧ sin θ = dist O C / dist O A ∧ cos θ = (dist O A - dist O C) / dist O A) :
    ∃ traced_area : ℝ, traced_area = 18 * real.pi :=
by
  sorry

end centroid_area_trace_l655_655112


namespace bananas_oranges_equivalence_l655_655135

noncomputable def bananasOrangesEquivalence (bananas oranges : ℝ) : Prop :=
  (3/4 * 12 * bananas = 9 * oranges) → 
  (2/5 * 15 * bananas = 6 * oranges)

theorem bananas_oranges_equivalence {bananas oranges : ℝ} : 
  bananasOrangesEquivalence bananas oranges :=
begin
  sorry
end

end bananas_oranges_equivalence_l655_655135


namespace village_population_l655_655815

theorem village_population (initial_population : ℕ) (died_fraction fear_fraction : ℚ) 
  (h1 : initial_population = 4500) 
  (h2 : died_fraction = 0.10) 
  (h3 : fear_fraction = 0.20) : 
  let died := (died_fraction * initial_population).to_nat in
  let remaining_after_died := initial_population - died in
  let left_due_to_fear := (fear_fraction * remaining_after_died).to_nat in
  let current_population := remaining_after_died - left_due_to_fear in
  current_population = 3240 := 
  sorry

end village_population_l655_655815


namespace cost_of_eight_memory_cards_l655_655961

theorem cost_of_eight_memory_cards (total_cost_of_three: ℕ) (h: total_cost_of_three = 45) : 8 * (total_cost_of_three / 3) = 120 := by
  sorry

end cost_of_eight_memory_cards_l655_655961


namespace select_at_least_one_first_class_l655_655953

def numFirstClassParts : Nat := 6
def numSecondClassParts : Nat := 4
def totalParts : Nat := numFirstClassParts + numSecondClassParts
def numSelectedParts : Nat := 3

noncomputable def binom : ℕ → ℕ → ℕ := λ n k, Nat.choose n k

theorem select_at_least_one_first_class :
  (binom totalParts numSelectedParts) - (binom numSecondClassParts numSelectedParts) = 116 := 
by
  sorry -- Proof is omitted

end select_at_least_one_first_class_l655_655953


namespace shape_with_2_axes_of_symmetry_l655_655622

-- Definitions for the shapes and their axes of symmetry
inductive Shape
| A | B | C | D

def axes_of_symmetry : Shape → ℕ
| Shape.A => 4
| Shape.B => 0
| Shape.C => 2
| Shape.D => 1

-- The theorem we want to prove: Shape C has exactly 2 lines of symmetry
theorem shape_with_2_axes_of_symmetry : ∃ s : Shape, axes_of_symmetry s = 2 :=
by
  apply Exists.intro Shape.C
  simp [axes_of_symmetry]
  sorry

end shape_with_2_axes_of_symmetry_l655_655622


namespace courses_differ_by_at_least_one_l655_655611

theorem courses_differ_by_at_least_one :
  (let courses := ({1, 2, 3, 4} : Finset ℕ)
       m := 2
       choose := Nat.choose
       total_ways := (choose 4 2) + (choose 4 1 * choose 3 1 * choose 2 1)
   in total_ways) = 30 :=
by 
  sorry

end courses_differ_by_at_least_one_l655_655611


namespace decreasing_interval_f_l655_655390

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 4 * x - 3

-- Statement to prove that the interval where f is monotonically decreasing is [2, +∞)
theorem decreasing_interval_f : (∀ x₁ x₂ : ℝ, 2 ≤ x₁ ∧ x₁ ≤ x₂ → f x₁ ≥ f x₂) :=
by
  sorry

end decreasing_interval_f_l655_655390


namespace conjugate_z_l655_655381

-- Define the imaginary unit
notation "i" => Complex.I

-- Define the given complex number z
def z : ℂ := 2 * i / (1 + i)

-- State the proof problem
theorem conjugate_z : conj z = 1 - i :=
by
  sorry

end conjugate_z_l655_655381


namespace different_colors_probability_l655_655566

noncomputable def differentColorProbability : ℚ :=
  let redChips := 7
  let greenChips := 5
  let totalChips := redChips + greenChips
  let probRedThenGreen := (redChips / totalChips) * (greenChips / totalChips)
  let probGreenThenRed := (greenChips / totalChips) * (redChips / totalChips)
  (probRedThenGreen + probGreenThenRed)

theorem different_colors_probability :
  differentColorProbability = 35 / 72 :=
by sorry

end different_colors_probability_l655_655566


namespace number_of_elements_in_union_l655_655768

definition A : Set ℕ := {4, 5, 6}
definition B : Set ℕ := {2, 3, 4}

theorem number_of_elements_in_union :
  (A ∪ B).size = 5 :=
by
  sorry

end number_of_elements_in_union_l655_655768


namespace scalene_triangle_angles_l655_655448

theorem scalene_triangle_angles (x y z : ℝ) (h1 : x + y + z = 180) (h2 : x ≠ y ∧ y ≠ z ∧ x ≠ z)
(h3 : x = 36 ∨ y = 36 ∨ z = 36) (h4 : x = 2 * y ∨ y = 2 * x ∨ z = 2 * x ∨ x = 2 * z ∨ y = 2 * z ∨ z = 2 * y) :
(x = 36 ∧ y = 48 ∧ z = 96) ∨ (x = 18 ∧ y = 36 ∧ z = 126) ∨ (x = 36 ∧ z = 48 ∧ y = 96) ∨ (y = 18 ∧ x = 36 ∧ z = 126) :=
sorry

end scalene_triangle_angles_l655_655448


namespace reconstruct_quadrilateral_l655_655367

variables (V : Type) [AddCommGroup V] [Module ℝ V]
variables (E E' F F' G G' H H' : V)
variables (u v w x : ℝ)

-- Conditions
def F_midpoint := F = (1/2) • E + (1/2) • E'
def G_midpoint := G = (1/2) • F + (1/2) • F'
def H_midpoint := H = (1/2) • G + (1/2) • G'

-- The statement to be proved
theorem reconstruct_quadrilateral
    (hF : F_midpoint)
    (hG : G_midpoint)
    (hH : H_midpoint) :
    ∃ u v w x, E = u • E' + v • F' + w • G' + x • H' :=
sorry

end reconstruct_quadrilateral_l655_655367


namespace compare_y_l655_655750

theorem compare_y (y1 y2 y3 : ℝ)
  (h1 : y1 = 4 ^ 0.9) 
  (h2 : y2 = 8 ^ 0.48)
  (h3 : y3 = (1 / 2) ^ (-1.5)) :
  y1 > y3 ∧ y3 > y2 :=
by {
  sorry
}

end compare_y_l655_655750


namespace cos_double_angle_l655_655021

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 1/3) : Real.cos (2 * α) = 7/9 :=
by
    sorry

end cos_double_angle_l655_655021


namespace equal_distances_of_projected_points_l655_655875

theorem equal_distances_of_projected_points
  (α β : ℝ)
  (hα_pos : 0 < α)
  (hβ_pos : 0 < β)
  (hαβ_sum : α + β < π) :
  ( ∀ (A B : ℝ), ∃ (A' B' : ℝ),
    (A'B' * Math.cos α = A * Math.cos β * Math.cos α) ∧ 
    (B'A' * Math.cos β = A * Math.cos α * Math.cos β) )
:=
sorry

end equal_distances_of_projected_points_l655_655875


namespace decimal_repeat_digit_2023_l655_655584

theorem decimal_repeat_digit_2023 :
  (∀ (n : ℕ), (n > 0) → has_substring ((17 : ℚ) / 53).decimalString "32075471698113207547") →
  natMod ((2023 : ℕ)) 20 = 3 →
  string.get ((17 : ℚ) / 53).decimalString 2023 = '0' :=
by sorry

end decimal_repeat_digit_2023_l655_655584


namespace quadratic_function_value_when_x_is_zero_l655_655407

theorem quadratic_function_value_when_x_is_zero :
  (∃ h : ℝ, (∀ x : ℝ, x < -3 → (-(x + h)^2 < -(x + h + 1)^2)) ∧
            (∀ x : ℝ, x > -3 → (-(x + h)^2 > -(x + h - 1)^2)) ∧
            (y = -(0 + h)^2) → y = -9) := 
sorry

end quadratic_function_value_when_x_is_zero_l655_655407


namespace product_of_two_numbers_l655_655926

theorem product_of_two_numbers
  (x y : ℝ)
  (h1 : x - y = 12)
  (h2 : x^2 + y^2 = 106) :
  x * y = 32 := by 
  sorry

end product_of_two_numbers_l655_655926


namespace arithmetic_geometric_sequence_problem_l655_655378

theorem arithmetic_geometric_sequence_problem :
  (∃ (a : ℕ → ℕ) (b : ℕ → ℕ), a 1 = 1 ∧ b 1 = 1 ∧ a 3 * b 2 = 14 ∧ a 3 - b 2 = 5 ∧
    (∀ n, a n = 3 * n - 2) ∧ (∀ n, b n = 2^(n - 1)) ∧
    (∀ n, (finset.range n).sum (λ i, a i.succ + b i.succ) = (3 * n * n - n) / 2 + 2^n - 1)) :=
begin
  sorry
end

end arithmetic_geometric_sequence_problem_l655_655378


namespace greatest_possible_value_of_n_greatest_possible_value_of_10_l655_655601

theorem greatest_possible_value_of_n (n : ℤ) (h : 101 * n^2 ≤ 12100) : n ≤ 10 :=
by
  sorry

theorem greatest_possible_value_of_10 (n : ℤ) (h : 101 * n^2 ≤ 12100) : n = 10 → n = 10 :=
by
  sorry

end greatest_possible_value_of_n_greatest_possible_value_of_10_l655_655601


namespace correct_calculation_B_l655_655594

theorem correct_calculation_B (a b : ℝ) (ha : a = 3 * real.sqrt 7) (hb : b = real.sqrt 7) : 
  a - b = 2 * real.sqrt 7 :=
by
  sorry

end correct_calculation_B_l655_655594


namespace minimum_area_of_triangle_l655_655008

theorem minimum_area_of_triangle (a : ℝ) (h : a > 0) :
  let C1 := { p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / (2 * a^2) = 1 }
  let F1_x := -sqrt(3) * a
  let F1 := (F1_x, 0)
  let C2 := { p : ℝ × ℝ | p.2^2 = -4 * sqrt(3) * a * p.1 }
  let lineAB := { p : ℝ × ℝ | p.1 = -sqrt(3) * a }
  let triangle_area (p q : ℝ × ℝ) : ℝ := 
    let (x1, y1) := p in
    let (x2, y2) := q in
    abs ((x1 * y2 - x2 * y1) / 2)
  let S_triangle_AOB (A B : ℝ × ℝ) : ℝ := 
    triangle_area A F1 + triangle_area F1 B
  (∃ A B : ℝ × ℝ, A ∈ C2 ∧ B ∈ C2 ∧ F1 ∈ lineAB ∧
    S_triangle_AOB A B = 6 * a^2 ∧ lineAB = { p : ℝ × ℝ | p.1 = -sqrt(3) * a }) := sorry

end minimum_area_of_triangle_l655_655008


namespace triangle_angles_side_c_range_l655_655743

-- Definitions for the given problem
variables (A B C a b c : ℝ)
variable S : ℝ := (sqrt 3 / 2) * a * c * cos B

-- Equivalent proof problem for question 1
theorem triangle_angles (h1 : c = 2 * a) : A = π / 6 ∧ B = π / 3 ∧ C = π / 2 :=
sorry

-- Equivalent proof problem for question 2
theorem side_c_range (h1 : a = 2) (h2 : π / 4 ≤ A) (h3 : A ≤ π / 3) : 2 ≤ c ∧ c ≤ sqrt 3 + 1 :=
sorry

end triangle_angles_side_c_range_l655_655743


namespace graph_intersection_points_l655_655134

noncomputable def g : ℝ → ℝ := sorry

theorem graph_intersection_points 
  (hg : function.injective g) : 
  ∃ S : finset ℝ, (∀ x, x ∈ S ↔ g (x^3) = g (x^6)) ∧ S.card = 3 := 
by 
  sorry

end graph_intersection_points_l655_655134


namespace cannot_become_10_after_33_moves_l655_655179

theorem cannot_become_10_after_33_moves :
  ∀ (n : ℤ), n = 20 → (∀ i, 0 ≤ i ∧ i < 33 → n + i - (33 - i) ≠ 10) := 
by
  intros n h initial_move_parity
  rw h at initial_move_parity
  sorry

end cannot_become_10_after_33_moves_l655_655179


namespace number_of_sides_multiple_of_n_l655_655015

-- Define n as the number of lines of symmetry
def has_n_symmetry_axes (polygon : Type u) [has_symmetry polygon] (n : ℕ) : Prop :=
  -- condition: polygon has exactly n symmetry lines
  count_symmetry_lines polygon = n

-- Define the statement that the number of sides of the polygon is a multiple of n
theorem number_of_sides_multiple_of_n (polygon : Type u) [has_symmetry polygon] [has_sides polygon] (n : ℕ)
  (h : has_n_symmetry_axes polygon n) : 
  ∃ m : ℕ, number_of_sides polygon = n * m := 
sorry

end number_of_sides_multiple_of_n_l655_655015


namespace add_three_to_both_sides_l655_655730

variable {a b : ℝ}

theorem add_three_to_both_sides (h : a < b) : 3 + a < 3 + b :=
by
  sorry

end add_three_to_both_sides_l655_655730


namespace walter_zoo_time_l655_655582

theorem walter_zoo_time (S: ℕ) (H1: S + 8 * S + 13 = 130) : S = 13 :=
by sorry

end walter_zoo_time_l655_655582


namespace sum_of_divisors_630_l655_655191

theorem sum_of_divisors_630 :
  let n := 630 in
  let factors := [(2, 1), (3, 2), (5, 1), (7, 1)] in
  ∑ d in (finset.divisors n), d = 1872 :=
by sorry

end sum_of_divisors_630_l655_655191


namespace domain_of_g_l655_655989

def g (x : ℝ) := Real.log 3 (Real.log 4 (Real.log 5 (Real.log 6 x)))

theorem domain_of_g (x : ℝ) : g x = Real.log 3 (Real.log 4 (Real.log 5 (Real.log 6 x))) → x > 6 ^ 625 :=
by
  intros hg -- Introduce the hypothesis that g x is defined
  sorry -- Proof to be completed

end domain_of_g_l655_655989


namespace number_of_odd_subsets_correct_total_sum_of_odd_subsets_correct_l655_655499

def is_odd_subset (A : Set ℕ) (X : Finset ℕ) : Prop :=
  let odd_count := (A ∩ { x | x % 2 = 1 }).card
  let even_count := (A ∩ { x | x % 2 = 0 }).card
  odd_count > even_count

noncomputable def num_odd_subsets (n : ℕ) : ℕ :=
  if even n then
    let k := n / 2
    2^(2*k-1) - (2^k / 2) * finset.card finset.nat_binom.k
  else
    2^(n-1)
  
noncomputable def total_sum (n : ℕ) : ℕ :=
  if even n then
    let k := n / 2
    n*(n+1)*2^(n-3) - (n/2)*(n/2+1)*finset.card finset.nat_binom.k
  else
    n*(n+1)*2^(n-3) + (n+1)*finset.card finset.nat_binom.(n/2)

theorem number_of_odd_subsets_correct (n : ℕ) :
  ∃ (X : Finset ℕ), X = finset.range (n+1) ∧
  (num_odd_subsets n = (if even n then (let k := n / 2 in 2^(2*k-1) - ((2^k/2) * finset.card finset.nat_binom.k)) else 2^(n-1))) := sorry

theorem total_sum_of_odd_subsets_correct (n : ℕ) :
  ∃ (X : Finset ℕ), X = finset.range (n+1) ∧
  (total_sum n = (if even n then (let k := n / 2 in n*(n+1)*2^(n-3) - (n/2)*(n/2+1)*finset.card finset.nat_binom.k) else n*(n+1)*2^(n-3) + (n+1)*finset.card finset.nat_binom.(n/2))) := sorry

end number_of_odd_subsets_correct_total_sum_of_odd_subsets_correct_l655_655499


namespace car_b_speed_correct_l655_655298

noncomputable def car_b_speed (v_A1 v_A2 t1 t2 d1 d2 : ℝ) (h1 : v_A1 = 50) (h2 : v_A2 = 80) (h3 : t1 = 6) (h4 : t2 = 2) (h5 : d1 = v_A1 * t1) (h6 : d2 = v_A2 * t2) (h7 : d1 - d2 = 140) : ℝ :=
  let delta_t := t1 - t2 in
  140 / delta_t

theorem car_b_speed_correct : car_b_speed 50 80 6 2 (50 * 6) (80 * 2) rfl rfl rfl rfl rfl rfl = 35 := 
by {
  sorry
}

end car_b_speed_correct_l655_655298


namespace simplify_cot45_tan30_l655_655524

-- Conditions for the problem
def cot_45 : ℝ := 1
def tan_30 : ℝ := 1 / (Math.sqrt 3)

-- The proof problem
theorem simplify_cot45_tan30 : cot_45 + tan_30 = (Math.sqrt 18 + Math.sqrt 6) / 6 := 
begin
  -- proof goes here
  sorry
end

end simplify_cot45_tan30_l655_655524


namespace nested_factorial_inequality_l655_655839

theorem nested_factorial_inequality : 
  ∀ k : ℕ, k > 0 → ¬ (factorial_n_times 4 k > factorial_n_times 3 (k + 1)) :=
by sorry

def factorial_n_times (n k : ℕ) : ℕ :=
  match k with
  | 0     => n
  | k + 1 => (factorial_n_times n k)!

end nested_factorial_inequality_l655_655839


namespace minimum_shift_l655_655149

noncomputable def f (x : ℝ) : ℝ := sin x * cos x - sqrt(3) * cos x ^ 2

noncomputable def g (x : ℝ) : ℝ := sin (2 * x + π / 3) - sqrt(3) / 2

def shifted_g (x k : ℝ) : ℝ := sin (2 * x - 2 * k + π / 3) - sqrt(3) / 2

theorem minimum_shift (k : ℝ) (h : k > 0) :
  (∀ x, f x = g (x - k)) → k = π / 3 :=
sorry

end minimum_shift_l655_655149


namespace avg_growth_rate_selling_price_reduction_l655_655137

open Real

-- Define the conditions for the first question
def sales_volume_aug : ℝ := 50000
def sales_volume_oct : ℝ := 72000

-- Define the conditions for the second question
def cost_price_per_unit : ℝ := 40
def initial_selling_price_per_unit : ℝ := 80
def initial_sales_volume_per_day : ℝ := 20
def additional_units_per_half_dollar_decrease : ℝ := 4
def desired_daily_profit : ℝ := 1400

-- First proof: monthly average growth rate
theorem avg_growth_rate (x : ℝ) :
  sales_volume_aug * (1 + x)^2 = sales_volume_oct → x = 0.2 :=
by {
  sorry
}

-- Second proof: reduction in selling price for daily profit
theorem selling_price_reduction (y : ℝ) :
  (initial_selling_price_per_unit - y - cost_price_per_unit) * (initial_sales_volume_per_day + additional_units_per_half_dollar_decrease * y / 0.5) = desired_daily_profit → y = 30 :=
by {
  sorry
}

end avg_growth_rate_selling_price_reduction_l655_655137


namespace no_equilateral_triangle_on_square_grid_l655_655673

theorem no_equilateral_triangle_on_square_grid :
  ¬ ∃ (A B C : ℤ × ℤ),
    let d := (λ (P Q : ℤ × ℤ), rat.cast ((P.2 - Q.2) : ℚ) / rat.cast ((P.1 - Q.1) : ℚ)) in
    d A B = /-(rat.cast 3).sqrt ∧ d B C = /-(rat.cast 3).sqrt ∧ d C A = /-(rat.cast 3).sqrt :=
by
  sorry

end no_equilateral_triangle_on_square_grid_l655_655673


namespace subsets_inequality_l655_655478

variable {U : Type} {m n : ℕ}
variable (A : Fin m → Set U)

theorem subsets_inequality (h_univ : ∀ i, (A i).finite) (h_size : (⋃ i, A i).finite.card = n) :
  ∑ i : Fin m, ∑ j : Fin m, (A i).toFinset.card * (A i ∩ A j).toFinset.card ≥
  (1 / (m * n)) * (∑ i : Fin m, (A i).toFinset.card) ^ 3 :=
by
  sorry

end subsets_inequality_l655_655478


namespace domain_of_g_l655_655986

def g (x : ℝ) := Real.log 3 (Real.log 4 (Real.log 5 (Real.log 6 x)))

theorem domain_of_g (x : ℝ) : g x = Real.log 3 (Real.log 4 (Real.log 5 (Real.log 6 x))) → x > 6 ^ 625 :=
by
  intros hg -- Introduce the hypothesis that g x is defined
  sorry -- Proof to be completed

end domain_of_g_l655_655986


namespace averagePrice_is_20_l655_655516

-- Define the conditions
def books1 : Nat := 32
def cost1 : Nat := 1500

def books2 : Nat := 60
def cost2 : Nat := 340

-- Define the total books and total cost
def totalBooks : Nat := books1 + books2
def totalCost : Nat := cost1 + cost2

-- Define the average price calculation
def averagePrice : Nat := totalCost / totalBooks

-- The statement to prove
theorem averagePrice_is_20 : averagePrice = 20 := by
  -- Sorry is used here as a placeholder for the actual proof.
  sorry

end averagePrice_is_20_l655_655516


namespace three_digit_non_multiples_of_3_or_11_l655_655779

theorem three_digit_non_multiples_of_3_or_11 : 
  ∃ (n : ℕ), n = 546 ∧ 
  (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → 
    ¬ (x % 3 = 0 ∨ x % 11 = 0) → 
    n = (900 - (300 + 81 - 27))) := 
by 
  sorry

end three_digit_non_multiples_of_3_or_11_l655_655779


namespace axis_of_symmetry_l655_655927

def f (x : ℝ) : ℝ := 
  Real.sin (3 * x + Real.pi / 3) * Real.cos (x - Real.pi / 6) + 
  Real.cos (3 * x + Real.pi / 3) * Real.sin (x - Real.pi / 6)

theorem axis_of_symmetry : 
  ∃ k : ℤ, ∀ x : ℝ, f(x) = Real.sin (4 * x + Real.pi / 6) 
  {x = k * Real.pi / 4 + Real.pi / 12} → x = Real.pi / 12 := 
sorry

end axis_of_symmetry_l655_655927


namespace total_height_of_three_buildings_l655_655234

theorem total_height_of_three_buildings :
  let h1 := 600
  let h2 := 2 * h1
  let h3 := 3 * (h1 + h2)
  h1 + h2 + h3 = 7200 :=
by
  sorry

end total_height_of_three_buildings_l655_655234


namespace find_m_l655_655037

-- Define the quadratic equation's roots.
def roots_eq (m : ℚ) : Prop :=
  ( -17 + (1:ℝ) * complex.I * real.sqrt 471 ) / 8 = complex.I * real.sqrt (289 - 16 * m) / 8 ∧
  ( -17 - (1:ℝ) * complex.I * real.sqrt 471 ) / 8 = - complex.I * real.sqrt (289 - 16 * m) / 8

-- Define the condition that expresses the relation derived from the given roots.
def quadratic_condition (m : ℚ) : Prop :=
  289 - 16 * m = -471

-- The proof problem in Lean.
theorem find_m (m : ℚ) : quadratic_condition m → m = 47.5 :=
by
  intro h
  sorry

end find_m_l655_655037


namespace locus_points_isosceles_triangles_l655_655946

theorem locus_points_isosceles_triangles {A B C : Point}
  (hABC_equilateral : is_equilateral_triangle A B C) :
  ∃ (circumcircle_ABC : Circle),
    (∀ (M : Point), (is_isosceles_triangle A B M) ∧ (is_isosceles_triangle A C M) ↔ M ∈ circumcircle_ABC) :=
sorry

end locus_points_isosceles_triangles_l655_655946


namespace geometric_sequence_eighth_term_l655_655305

theorem geometric_sequence_eighth_term 
  (a : ℕ) (r : ℕ) (h1 : a = 4) (h2 : r = 16 / 4) :
  a * r^(7) = 65536 :=
by
  sorry

end geometric_sequence_eighth_term_l655_655305


namespace steven_falls_correct_l655_655915

/-
  We will model the problem where we are given the conditions about the falls of Steven, Stephanie,
  and Sonya, and then prove that the number of times Steven fell is 3.
-/

variables (S : ℕ) -- Steven's falls

-- Conditions
def stephanie_falls := S + 13
def sonya_falls := 6 
def sonya_condition := 6 = (stephanie_falls / 2) - 2

-- Theorem statement
theorem steven_falls_correct : S = 3 :=
by {
  -- Note: the actual proof steps would go here, but are omitted per instructions
  sorry
}

end steven_falls_correct_l655_655915


namespace simplest_common_denominator_l655_655166

theorem simplest_common_denominator (x a : ℕ) :
  let d1 := 3 * x
  let d2 := 6 * x^2
  lcm d1 d2 = 6 * x^2 := 
by
  let d1 := 3 * x
  let d2 := 6 * x^2
  show lcm d1 d2 = 6 * x^2
  sorry

end simplest_common_denominator_l655_655166


namespace locus_of_centers_l655_655550

variables {A B O : Euclidean.Geometry.Point}
variable {r : ℝ}

def is_circle (O : Euclidean.Geometry.Point) (r : ℝ) (A B : Euclidean.Geometry.Point) :=
  Euclidean.Geometry.dist O A = r ∧ Euclidean.Geometry.dist O B = r

def midpoint (A B : Euclidean.Geometry.Point) : Euclidean.Geometry.Point :=
  Euclidean.Geometry.midpoint A B

def perpendicular_bisector (A B : Euclidean.Geometry.Point) : Set Euclidean.Geometry.Point :=
  {O | Euclidean.Geometry.dist O A = Euclidean.Geometry.dist O B}

theorem locus_of_centers (A B : Euclidean.Geometry.Point) (r : ℝ) :
  let d := Euclidean.Geometry.dist A B in
  if d = 2 * r then
    {O | O = midpoint A B}
  else if d < 2 * r then
    let mid := midpoint A B in
    let h_dist := sqrt (r^2 - (d/2)^2) in
    {O | O ∈ perpendicular_bisector A B ∧ Euclidean.Geometry.dist O mid = h_dist}
  else
    ∅ :=
by sorry

end locus_of_centers_l655_655550


namespace total_height_of_buildings_l655_655235

theorem total_height_of_buildings :
  let height_first_building := 600
  let height_second_building := 2 * height_first_building
  let height_third_building := 3 * (height_first_building + height_second_building)
  height_first_building + height_second_building + height_third_building = 7200 := by
    let height_first_building := 600
    let height_second_building := 2 * height_first_building
    let height_third_building := 3 * (height_first_building + height_second_building)
    show height_first_building + height_second_building + height_third_building = 7200
    sorry

end total_height_of_buildings_l655_655235


namespace range_of_func_l655_655541

noncomputable def func (x : ℝ) : ℝ := 1 / (x - 1)

theorem range_of_func :
  (∀ y : ℝ, 
    (∃ x : ℝ, (x < 1 ∨ (2 ≤ x ∧ x < 5)) ∧ y = func x) ↔ 
    (y < 0 ∨ (1/4 < y ∧ y ≤ 1))) :=
by
  sorry

end range_of_func_l655_655541


namespace exist_m_eq_l655_655101

theorem exist_m_eq (n b : ℕ) (p : ℕ) (hp_prime : Nat.Prime p) (hp_odd : p % 2 = 1) (hn_zero : n ≠ 0) (hb_zero : b ≠ 0)
  (h_div : p ∣ (b^(2^n) + 1)) :
  ∃ m : ℕ, p = 2^(n+1) * m + 1 :=
by
  sorry

end exist_m_eq_l655_655101


namespace cutting_four_pieces_l655_655311

/--
Given that cutting a wooden stick into three pieces takes 6 minutes, 
prove that cutting a wooden stick into four pieces will take 9 minutes.
-/
theorem cutting_four_pieces (t : ℕ) (n1 n2 : ℕ) : 
  (n1 - 1) * t = 6 → (n2 - 1) = 3 → n2 * t = 9 :=
by
  assume h1 h2
  sorry

end cutting_four_pieces_l655_655311


namespace calc_result_l655_655991

theorem calc_result : (377 / 13 / 29 * 1 / 4 / 2) = 0.125 := 
by sorry

end calc_result_l655_655991


namespace sum_of_bases_is_20_l655_655822

theorem sum_of_bases_is_20
  (B1 B2 : ℕ)
  (G1 : ℚ)
  (G2 : ℚ)
  (hG1_B1 : G1 = (4 * B1 + 5) / (B1^2 - 1))
  (hG2_B1 : G2 = (5 * B1 + 4) / (B1^2 - 1))
  (hG1_B2 : G1 = (3 * B2) / (B2^2 - 1))
  (hG2_B2 : G2 = (6 * B2) / (B2^2 - 1)) :
  B1 + B2 = 20 :=
sorry

end sum_of_bases_is_20_l655_655822


namespace number_of_people_adopting_cats_l655_655473

theorem number_of_people_adopting_cats 
    (initial_cats : ℕ)
    (monday_kittens : ℕ)
    (tuesday_injured_cat : ℕ)
    (final_cats : ℕ)
    (cats_per_person_adopting : ℕ)
    (h_initial : initial_cats = 20)
    (h_monday : monday_kittens = 2)
    (h_tuesday : tuesday_injured_cat = 1)
    (h_final: final_cats = 17)
    (h_cats_per_person: cats_per_person_adopting = 2) :
    ∃ (people_adopting : ℕ), people_adopting = 3 :=
by
  sorry

end number_of_people_adopting_cats_l655_655473


namespace domain_of_g_l655_655987

def g (x : ℝ) := Real.log 3 (Real.log 4 (Real.log 5 (Real.log 6 x)))

theorem domain_of_g (x : ℝ) : g x = Real.log 3 (Real.log 4 (Real.log 5 (Real.log 6 x))) → x > 6 ^ 625 :=
by
  intros hg -- Introduce the hypothesis that g x is defined
  sorry -- Proof to be completed

end domain_of_g_l655_655987


namespace sum_of_first_n_terms_l655_655879

noncomputable def a (n : ℕ) : ℕ := 2 * n - 1
def S (n : ℕ) : ℕ := (n * (a 1 + a n)) / 2

theorem sum_of_first_n_terms (n : ℕ) : S n = n^2 := by
  sorry

end sum_of_first_n_terms_l655_655879


namespace three_digit_numbers_not_multiple_of_3_or_11_l655_655787

-- Proving the number of three-digit numbers that are multiples of neither 3 nor 11 is 547
theorem three_digit_numbers_not_multiple_of_3_or_11 : (finset.Icc 100 999).filter (λ n, ¬(3 ∣ n) ∧ ¬(11 ∣ n)).card = 547 :=
by
  -- The steps to reach the solution will be implemented here
  sorry

end three_digit_numbers_not_multiple_of_3_or_11_l655_655787


namespace arithmetic_mean_roots_f2015_l655_655215

def f (x : ℝ) : ℝ := x^2 - x + 1

noncomputable def f_iter : ℕ → (ℝ → ℝ)
| 0     := id
| (n+1) := f ∘ (f_iter n)

noncomputable def r (n : ℕ) : ℝ := 
  let roots := { x : ℝ | f_iter n x = 0 } 
  in (roots.to_finset.card ≠ 0) → (roots.to_finset.sum id) / (roots.to_finset.card)

theorem arithmetic_mean_roots_f2015 : r 2015 = 1 / 2 := sorry

end arithmetic_mean_roots_f2015_l655_655215


namespace intersection_S_T_eq_U_l655_655110

def S : Set ℝ := {x | abs x < 5}
def T : Set ℝ := {x | (x + 7) * (x - 3) < 0}
def U : Set ℝ := {x | -5 < x ∧ x < 3}

theorem intersection_S_T_eq_U : (S ∩ T) = U := 
by 
  sorry

end intersection_S_T_eq_U_l655_655110


namespace technicians_count_l655_655818

theorem technicians_count 
    (total_workers : ℕ) (avg_salary_all : ℕ) (avg_salary_technicians : ℕ) (avg_salary_rest : ℕ)
    (h_workers : total_workers = 28) (h_avg_all : avg_salary_all = 8000) 
    (h_avg_tech : avg_salary_technicians = 14000) (h_avg_rest : avg_salary_rest = 6000) : 
    ∃ T R : ℕ, T + R = total_workers ∧ (avg_salary_technicians * T + avg_salary_rest * R = avg_salary_all * total_workers) ∧ T = 7 :=
by
  sorry

end technicians_count_l655_655818


namespace find_line_equation_l655_655457

theorem find_line_equation (l : ℝ → ℝ) (a : ℝ) (C₁_eq : ∀ x y : ℝ, x^2 + y^2 = a ↔ C₁ x y)
                             (C₂_eq : ∀ x y : ℝ, x^2 + y^2 + 2x - 2a*y + 3 = 0 ↔ C₂ x y)
                             (a_eq : a = 2)
                             (symmetry_condition : ∀ x y : ℝ, C₁ x y ↔ C₂ (l x) y) :
  ∀ x y : ℝ, 2x - 4y + 5 = 0 :=
by
  sorry

end find_line_equation_l655_655457


namespace exists_k_with_n_distinct_prime_factors_l655_655901

theorem exists_k_with_n_distinct_prime_factors (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  ∃ k : ℕ, k > 0 ∧ (2^k - m).natAbs.primeFactors.length ≥ n :=
by
  sorry

end exists_k_with_n_distinct_prime_factors_l655_655901


namespace find_lines_through_intersection_and_conditions_l655_655769

theorem find_lines_through_intersection_and_conditions :
  let l1 := λ (x y : ℝ), 3 * x + 4 * y - 2 = 0,
      l2 := λ (x y : ℝ), 2 * x + y + 2 = 0,
      l3 := λ (x y : ℝ), x - 2 * y - 1 = 0,
      P := (-2, 2) in
  (l1 P.1 P.2) ∧ (l2 P.1 P.2) →
  (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ a * 0 + b * 0 + c = 0 ∧ a * (-2) + b * 2 + c = 0) ∧
  (∃ a b c : ℝ, a * 2 = -2 ∧ a * 2 - 2 * b = 0 ∧ a * (-2) + b * 2 + 2 = 0)

end find_lines_through_intersection_and_conditions_l655_655769


namespace butterfly_eq_roots_l655_655314

theorem butterfly_eq_roots (a b c : ℝ) (h1 : a ≠ 0) (h2 : a - b + c = 0)
    (h3 : (a + c)^2 - 4 * a * c = 0) : a = c :=
by
  sorry

end butterfly_eq_roots_l655_655314


namespace bill_due_in_months_l655_655950

noncomputable def true_discount_time (TD A R : ℝ) : ℝ :=
  let P := A - TD
  let T := TD / (P * R / 100)
  12 * T

theorem bill_due_in_months :
  ∀ (TD A R : ℝ), TD = 189 → A = 1764 → R = 16 →
  abs (true_discount_time TD A R - 10.224) < 1 :=
by
  intros TD A R hTD hA hR
  sorry

end bill_due_in_months_l655_655950


namespace complex_number_in_fourth_quadrant_l655_655055

theorem complex_number_in_fourth_quadrant :
  let z := (1:ℂ) / (3 + (1:ℂ) * complex.I) in
  z.re > 0 ∧ z.im < 0 := by
sorry

end complex_number_in_fourth_quadrant_l655_655055


namespace sum_of_sequence_inverse_l655_655309

-- Definitions of sequences
def a_seq (n : ℕ) : ℕ := 3 * n
def b_seq (n : ℕ) : ℕ := 3 ^ (n - 1)

-- Definition of sequence sum S_n
def S_n (n : ℕ) : ℕ := (n * (3 + 3 * n)) / 2

-- Definition of T_n as per the given telescoping sum
def T_n (n : ℕ) : ℚ := 2 * n / (3 * (n + 1))

-- Example conditions based on provided problem details
lemma arithmetic_geometric_sequences :
  (a_seq 1 = 3) ∧
  (S_n 2 + b_seq 2 = 12) ∧
  (a_seq 3 = b_seq 3) ∧
  (∀ n, a_seq n = 3 * n) ∧
  (∀ n, b_seq n = 3^(n-1)) :=
begin
  sorry
end

-- The main theorem. No need to prove it here, just stating based on the problem.
theorem sum_of_sequence_inverse (n : ℕ) :
  (T_n n) = 2 * n / (3 * (n + 1)) :=
begin
  sorry
end

end sum_of_sequence_inverse_l655_655309


namespace musicians_performed_at_conservatory_l655_655289

theorem musicians_performed_at_conservatory :
  let quartets := 4 in
  let duets := 5 in
  let trios := 6 in
  let musicians_in_quartet := 4 in
  let musicians_in_duet := 2 in
  let musicians_in_trio := 3 in
  let absent_quartets := 1 in
  let absent_duets := 2 in
  let absent_soloists := 1 in
  let original_musicians := (quartets * musicians_in_quartet) +
                            (duets * musicians_in_duet) +
                            (trios * musicians_in_trio) in
  let absent_musicians := (absent_quartets * musicians_in_quartet) +
                          (absent_duets * musicians_in_duet) +
                          absent_soloists in
  let actual_musicians := original_musicians - absent_musicians in
  actual_musicians = 35 :=
by
  let quartets := 4
  let duets := 5
  let trios := 6
  let musicians_in_quartet := 4
  let musicians_in_duet := 2
  let musicians_in_trio := 3
  let absent_quartets := 1
  let absent_duets := 2
  let absent_soloists := 1
  let original_musicians := (quartets * musicians_in_quartet) +
                            (duets * musicians_in_duet) +
                            (trios * musicians_in_trio)
  let absent_musicians := (absent_quartets * musicians_in_quartet) +
                          (absent_duets * musicians_in_duet) +
                          absent_soloists
  let actual_musicians := original_musicians - absent_musicians
  show actual_musicians = 35 from sorry

end musicians_performed_at_conservatory_l655_655289


namespace maximum_tied_teams_in_tournament_l655_655446

theorem maximum_tied_teams_in_tournament : 
  ∀ (n : ℕ), n = 8 →
  (∀ (wins : ℕ), wins = (n * (n - 1)) / 2 →
   ∃ (k : ℕ), k ≤ n ∧ (k > 7 → false) ∧ 
               (∃ (w : ℕ), k * w = wins)) :=
by
  intros n hn wins hw
  use 7
  split
  · exact (by linarith)
  · intro h
    exfalso
    exact h (by linarith)
  · use 4
    calc
      7 * 4 = 28 : by norm_num
      ... = 28 : by rw hw; linarith
  
-- The proof is omitted as per instructions ("sorry" can be used to indicate this).

end maximum_tied_teams_in_tournament_l655_655446


namespace log_2_625_is_k_log_8_5_l655_655486

theorem log_2_625_is_k_log_8_5 (y k : ℝ) (h1 : log 8 5 = y) (h2 : log 2 625 = k * y) : k = 12 :=
by
  sorry

end log_2_625_is_k_log_8_5_l655_655486


namespace solve_problem_l655_655741

noncomputable def quadratic_function (f : ℝ → ℝ) : Prop :=
    (∀ x, f x = x^2 + 2 * x) ∧
    (∀ x ∈ set.Icc (1 : ℝ) 2, f x ≥ 3) ∧
    (∃ a > 0, f (a - 2) = 0 ∧ f a = 0)

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
    a 1 = 9 ∧
    (∀ n, a (n + 1) = a n^2 + 2 * a n) ∧
    (∀ n, ∃ r e, (0 < r ∧ ∀ m, log (1 + a (m + 1)) = 2 * log (1 + a m)) ∧ r = 2)

noncomputable def find_minimum [f : ℕ → ℝ] (C : ℕ → ℝ) : Prop :=
    (∀ n, C n = (2 * log (1 + a n)) / (n^2) ) ∧
    (∃ n₀ > 0, ∀ n, C n₀ ≤ C n)

theorem solve_problem (f : ℝ → ℝ) (a : ℕ → ℝ) (C : ℕ → ℝ) : 
    quadratic_function f ∧ geometric_sequence a ∧ find_minimum C :=
    sorry

end solve_problem_l655_655741


namespace least_total_distance_traveled_l655_655724
noncomputable def total_travel_distance (radius : ℝ) (friends : ℕ) : ℝ :=
  let d := 2 * radius * (Real.sqrt ((1 - ((1 + Real.sqrt 5) / 4)))] in
  400 + 10 * d

theorem least_total_distance_traveled :
  least_total_distance_traveled 20 5 = 400 + 10 * (40 / 3) :=
by
  sorry

end least_total_distance_traveled_l655_655724


namespace quotient_of_even_and_odd_composites_l655_655716

theorem quotient_of_even_and_odd_composites:
  (4 * 6 * 8 * 10 * 12) / (9 * 15 * 21 * 25 * 27) = 512 / 28525 := by
sorry

end quotient_of_even_and_odd_composites_l655_655716


namespace age_of_X_today_l655_655175

variable (x y z : ℕ) 

-- Conditions
def condition1 := x - 3 = 2 * (y - 3)
def condition2 := y - 3 = 3 * (z - 3)
def condition3 := (x + 7) + (y + 7) + (z + 7) = 130

theorem age_of_X_today 
  (h1 : condition1 x y z) 
  (h2 : condition2 y z) 
  (h3 : condition3 x y z) : 
  x = 63 := 
sorry

end age_of_X_today_l655_655175


namespace three_digit_numbers_not_multiples_of_3_or_11_l655_655775

def count_multiples (a b : ℕ) (lower upper : ℕ) : ℕ :=
  (upper - lower) / b + 1

theorem three_digit_numbers_not_multiples_of_3_or_11 : 
  let total := 900
  let multiples_3 := count_multiples 3 3 102 999
  let multiples_11 := count_multiples 11 11 110 990
  let multiples_33 := count_multiples 33 33 132 990
  let multiples_3_or_11 := multiples_3 + multiples_11 - multiples_33
  total - multiples_3_or_11 = 546 := 
by 
  sorry

end three_digit_numbers_not_multiples_of_3_or_11_l655_655775


namespace pm_star_eq_6_l655_655688

open Set

-- Definitions based on the conditions
def universal_set : Set ℕ := univ
def M : Set ℕ := {1, 2, 3, 4, 5}
def P : Set ℕ := {2, 3, 6}
def star (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- The theorem to prove
theorem pm_star_eq_6 : star P M = {6} :=
sorry

end pm_star_eq_6_l655_655688


namespace terminal_side_in_second_quadrant_l655_655746

theorem terminal_side_in_second_quadrant (α : ℝ) (h : (Real.tan α < 0) ∧ (Real.cos α < 0)) :
  (2 < α / (π / 2)) ∧ (α / (π / 2) < 3) :=
by
  sorry

end terminal_side_in_second_quadrant_l655_655746


namespace handshake_count_l655_655285

theorem handshake_count {teams : Fin 4 → Fin 2 → Prop}
    (h_teams_disjoint : ∀ (i j : Fin 4) (x y : Fin 2), i ≠ j → teams i x → teams j y → x ≠ y)
    (unique_partner : ∀ (i : Fin 4) (x1 x2 : Fin 2), teams i x1 → teams i x2 → x1 = x2) : 
    24 = (∑ i : Fin 8, (∑ j : Fin 8, if i ≠ j ∧ ¬(∃ k : Fin 4, teams k i ∧ teams k j) then 1 else 0)) / 2 :=
by sorry

end handshake_count_l655_655285


namespace exists_infinitely_many_n_l655_655739

theorem exists_infinitely_many_n (D : ℕ) (hD1 : D ≥ 2) (hD2 : ¬∃ k : ℕ, k^2 ∣ D) (m : ℕ) (hm : m > 0) :
  ∃ᶠ n in at_top, (n.gcd (⌊real.sqrt (D : ℝ) * n⌋) = m) :=
sorry

end exists_infinitely_many_n_l655_655739


namespace no_closed_path_exists_l655_655823

theorem no_closed_path_exists (n : ℕ) (h : n = 2017) :
  ¬∃ path : list (ℕ × ℕ), (∀ c ∈ fin 2017 × fin 2017, (∃ i, i ∈ path ∧ i = c)) ∧ (is_closed_path path) := 
sorry

end no_closed_path_exists_l655_655823


namespace xyz_neg_of_ineq_l655_655174

variables {x y z : ℝ}

theorem xyz_neg_of_ineq
  (h1 : 2 * x - y < 0)
  (h2 : 3 * y - 2 * z < 0)
  (h3 : 4 * z - 3 * x < 0) :
  x < 0 ∧ y < 0 ∧ z < 0 :=
sorry

end xyz_neg_of_ineq_l655_655174


namespace smallest_n_gt_15_l655_655082

theorem smallest_n_gt_15 (n : ℕ) : n ≡ 4 [MOD 6] → n ≡ 3 [MOD 7] → n > 15 → n = 52 :=
by
  sorry

end smallest_n_gt_15_l655_655082


namespace integer_divisibility_l655_655617

theorem integer_divisibility
  (x y z : ℤ)
  (h : 11 ∣ (7 * x + 2 * y - 5 * z)) :
  11 ∣ (3 * x - 7 * y + 12 * z) :=
sorry

end integer_divisibility_l655_655617


namespace teacherZhangAge_in_5_years_correct_l655_655960

variable (a : ℕ)

def teacherZhangAgeCurrent := 3 * a - 2

def teacherZhangAgeIn5Years := teacherZhangAgeCurrent a + 5

theorem teacherZhangAge_in_5_years_correct :
  teacherZhangAgeIn5Years a = 3 * a + 3 := by
  sorry

end teacherZhangAge_in_5_years_correct_l655_655960


namespace geometric_sequence_eighth_term_is_correct_l655_655440

noncomputable def geometric_sequence_eighth_term : ℚ :=
  let a1 := 2187
  let a5 := 960
  let r := (960 / 2187)^(1/4)
  let a8 := a1 * r^7
  a8

theorem geometric_sequence_eighth_term_is_correct :
  let a1 := 2187
  let a5 := 960
  let r := (960 / 2187)^(1/4)
  let a8 := a1 * r^7
  a8 = 35651584 / 4782969 := by
    sorry

end geometric_sequence_eighth_term_is_correct_l655_655440


namespace evaluate_expression_l655_655700

theorem evaluate_expression :
  (let odd_sum := (∑ k in (finset.range 1012), 2 * k + 1) in 
   let even_sum := (∑ k in (finset.range 1011), 2 * (k + 1)) in 
   odd_sum - even_sum + 7 - 8) = 47 :=
by sorry

end evaluate_expression_l655_655700


namespace number_of_hexagonal_faces_geq_2_l655_655521

noncomputable def polyhedron_condition (P H : ℕ) : Prop :=
  ∃ V E : ℕ, 
    V - E + (P + H) = 2 ∧ 
    3 * V = 2 * E ∧ 
    E = (5 * P + 6 * H) / 2 ∧
    P > 0 ∧ H > 0

theorem number_of_hexagonal_faces_geq_2 (P H : ℕ) (h : polyhedron_condition P H) : H ≥ 2 :=
sorry

end number_of_hexagonal_faces_geq_2_l655_655521


namespace rhombus_perimeter_l655_655607

-- Define the rhombus and its properties
def rhombus_diagonals (d1 d2 : ℝ) := ∀ {A B C D : ℝ}, 
  d1 = A + C ∧ d2 = B + D ∧ A = C ∧ B = D

-- Prove that given specific diagonal lengths, the perimeter is a certain value
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 72) (h2 : d2 = 30) :
  rhombus_diagonals d1 d2 → (4 * (Math.sqrt ((d1 / 2)^2 + (d2 / 2)^2))) = 156 :=
by
  intro h
  sorry

end rhombus_perimeter_l655_655607


namespace necessary_but_not_sufficient_l655_655026

-- Definitions of lines, plane, and perpendicularity
variable (Point : Type)
variable (Line : Type)
variable (Plane : Type)
variable (⊥ : Line → Plane → Prop) -- Line perpendicular to Plane
variable (⟂ : Line → Line → Prop) -- Line perpendicular to Line
variable (∥ : Line → Plane → Prop) -- Line parallel to Plane

-- The given conditions
variable {l m : Line}
variable {α : Plane}
variable (l_ne_m : l ≠ m) -- l and m are two different lines
variable (m_perp_alpha : m ⊥ α) -- m is perpendicular to Alfa

-- Statement we aim to prove
theorem necessary_but_not_sufficient :
  (l ⟂ m) → (l ∥ α) ∧ ¬(l ∥ α → l ⟂ m) :=
sorry

end necessary_but_not_sufficient_l655_655026


namespace domain_of_g_l655_655974

noncomputable def g (x : ℝ) : ℝ := Real.logb 3 (Real.logb 4 (Real.logb 5 (Real.logb 6 x)))

theorem domain_of_g : setOf (λ x, g x) = set.Ioi (6^625) :=
by
  sorry

end domain_of_g_l655_655974


namespace cone_radius_given_lateral_area_and_angle_l655_655362

theorem cone_radius_given_lateral_area_and_angle (R : ℝ) 
  (lateral_area_eq : 50 * real.pi = real.pi * R * (2 * R)) 
  (angle_eq : real.angle.cos (real.angle.of_deg 60) = real.sqrt (3) / 2) : 
  R = 5 :=
by
  sorry

end cone_radius_given_lateral_area_and_angle_l655_655362


namespace three_digit_numbers_not_multiples_of_3_or_11_l655_655774

def count_multiples (a b : ℕ) (lower upper : ℕ) : ℕ :=
  (upper - lower) / b + 1

theorem three_digit_numbers_not_multiples_of_3_or_11 : 
  let total := 900
  let multiples_3 := count_multiples 3 3 102 999
  let multiples_11 := count_multiples 11 11 110 990
  let multiples_33 := count_multiples 33 33 132 990
  let multiples_3_or_11 := multiples_3 + multiples_11 - multiples_33
  total - multiples_3_or_11 = 546 := 
by 
  sorry

end three_digit_numbers_not_multiples_of_3_or_11_l655_655774


namespace incenter_of_ABC_is_tangent_point_of_centroid_to_Omega_l655_655066

noncomputable theory

variables {A B C : Point}
variables (BC AB CA : ℝ)
variables (centroid incenter : Point)
variables (Omega : Circle)

-- Define the conditions
def is_triangle (A B C : Point) : Prop := -- Assume this definition exists
  true

def side_eq (BC : ℝ) (AB CA : ℝ) : Prop :=
  BC = (AB + CA) / 2

def passes_through (Omega : Circle) (P Q R : Point) : Prop := -- Assume this definition exists
  true

def tangents_from_centroid (G : Point) (Omega : Circle) (I : Point) : Prop := -- Assume this definition exists
  true

-- The main theorem
theorem incenter_of_ABC_is_tangent_point_of_centroid_to_Omega
  (h_triangle : is_triangle A B C)
  (h_side_eq : side_eq BC AB CA)
  (h_passes_through : passes_through Omega A B' C')
  (h_tangents_from_centroid : tangents_from_centroid centroid Omega incenter) :
  ∃ I, is_incenter I A B C ∧ tangent_point G I Omega :=
sorry

end incenter_of_ABC_is_tangent_point_of_centroid_to_Omega_l655_655066


namespace minimum_value_l655_655106

noncomputable def problem_statement (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 27) : ℝ :=
  a^2 + 6 * a * b + 9 * b^2 + 4 * c^2

theorem minimum_value : ∃ (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 27), 
  problem_statement a b c h = 180 :=
sorry

end minimum_value_l655_655106


namespace exponential_rule_l655_655595

theorem exponential_rule (a : ℝ) : (a ^ 3) ^ 2 = a ^ 6 :=  
  sorry

end exponential_rule_l655_655595


namespace three_digit_non_multiples_of_3_or_11_l655_655781

theorem three_digit_non_multiples_of_3_or_11 : 
  ∃ (n : ℕ), n = 546 ∧ 
  (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → 
    ¬ (x % 3 = 0 ∨ x % 11 = 0) → 
    n = (900 - (300 + 81 - 27))) := 
by 
  sorry

end three_digit_non_multiples_of_3_or_11_l655_655781


namespace largest_prime_factor_of_factorial_sum_is_5_l655_655587

-- Define factorial function
def fact (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * fact (n - 1)

-- Define the specific factorial sums in the problem
def n_factorial_sum : ℕ := fact 6 + fact 7

-- Lean statement to assert the largest prime factor of the sum
theorem largest_prime_factor_of_factorial_sum_is_5 :
  (∀ p : ℕ, (nat.prime p ∧ (p ∣ n_factorial_sum)) → p ≤ 5) ∧
  (∃ p : ℕ, nat.prime p ∧ (p ∣ n_factorial_sum) ∧ p = 5) :=
by
  -- Proof of the theorem
  sorry

end largest_prime_factor_of_factorial_sum_is_5_l655_655587


namespace range_of_f_l655_655939

def f (x : ℝ) : ℝ := (3 * x + 1) / (2 - x)

theorem range_of_f :
  ∀ y : ℝ, y ∈ (set.range f) ↔ y ≠ -3 :=
by
  sorry

end range_of_f_l655_655939


namespace goldie_total_earnings_l655_655414

-- Define weekly earnings based on hours and rates
def earnings_first_week (hours_dog_walking hours_medication : ℕ) : ℕ :=
  (hours_dog_walking * 5) + (hours_medication * 8)

def earnings_second_week (hours_feeding hours_cleaning hours_playing : ℕ) : ℕ :=
  (hours_feeding * 6) + (hours_cleaning * 4) + (hours_playing * 3)

-- Given conditions for hours worked each task in two weeks
def hours_dog_walking : ℕ := 12
def hours_medication : ℕ := 8
def hours_feeding : ℕ := 10
def hours_cleaning : ℕ := 15
def hours_playing : ℕ := 5

-- Proof statement: Total earnings over two weeks equals $259
theorem goldie_total_earnings : 
  (earnings_first_week hours_dog_walking hours_medication) + 
  (earnings_second_week hours_feeding hours_cleaning hours_playing) = 259 :=
by
  sorry

end goldie_total_earnings_l655_655414


namespace max_a_monotonic_f_l655_655729

theorem max_a_monotonic_f {a : ℝ} (h1 : 0 < a)
  (h2 : ∀ x ≥ 1, 0 ≤ (3 * x^2 - a)) : a ≤ 3 := by
  -- Proof to be provided
  sorry

end max_a_monotonic_f_l655_655729


namespace interval_of_monotonic_increase_range_of_f_l655_655500

def m (x : ℝ) : ℝ × ℝ := (Real.sin x, -1)
def n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, -1 / 2)
def f (x : ℝ) : ℝ := let mx := m x; let nx := n x in
  (mx.1 + nx.1) * mx.1 + (mx.2 + nx.2) * mx.2

theorem interval_of_monotonic_increase (k : ℤ) : 
  ∀ x, f x = Real.sin (2 * x - Real.pi / 6) + 2 →
  (k * Real.pi - Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 3) :=
sorry

theorem range_of_f (x : ℝ) :
  0 < x ∧ x < Real.pi / 2 → 
  3 / 2 < f x ∧ f x ≤ 3 :=
sorry

end interval_of_monotonic_increase_range_of_f_l655_655500


namespace land_plot_side_length_l655_655609

theorem land_plot_side_length (area : ℝ) (h : area = real.sqrt 900) : real.sqrt area = 30 :=
by {
  sorry
}

end land_plot_side_length_l655_655609


namespace T_at_far_right_end_l655_655723

structure Rectangle where
  w : ℕ
  x : ℕ
  y : ℕ
  z : ℕ

def P : Rectangle := {w := 3, x := 0, y := 9, z := 5}
def Q : Rectangle := {w := 6, x := 1, y := 0, z := 8}
def R : Rectangle := {w := 0, x := 3, y := 2, z := 7}
def S : Rectangle := {w := 8, x := 5, y := 4, z := 1}
def T : Rectangle := {w := 5, x := 2, y := 6, z := 9}

theorem T_at_far_right_end : ∀ (rectangles : List Rectangle), 
  (rectangles = [R, S, P, Q, T]) → 
  List.getLast rectangles T = T :=
by
  intros rectangles h
  rw h
  sorry

end T_at_far_right_end_l655_655723


namespace smallest_area_circle_l655_655337

noncomputable def circle1 : set (ℝ × ℝ) := { p | p.1^2 + p.2^2 + 4 * p.1 + p.2 = -1 }
noncomputable def circle2 : set (ℝ × ℝ) := { p | p.1^2 + p.2^2 + 2 * p.1 + 2 * p.2 + 1 = 0 }

theorem smallest_area_circle :
  ∃ c : set (ℝ × ℝ),
    (∀ p : ℝ × ℝ, p ∈ c ↔ p.1^2 + p.2^2 + (6 / 5) * p.1 + (12 / 5) * p.2 + 1 = 0) ∧
    (∀ p : ℝ × ℝ, p ∈ circle1 ∧ p ∈ circle2 → p ∈ c) :=
sorry

end smallest_area_circle_l655_655337


namespace triangle_segments_possible_l655_655469

theorem triangle_segments_possible (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) : 
  (∠ABC = 120) → 
  (a + b > b + c) → 
  (b + (b + c) > a) → 
  (a + b > b + c) → 
  ∃ (x y z : ℝ), x = a ∧ y = b ∧ z = b + c ∧ (x + y > z ∧ y + z > x ∧ z + x > y) :=
by
  sorry

end triangle_segments_possible_l655_655469


namespace max_triangles_formed_l655_655966

-- Represent the two equilateral triangles and their configurations
structure EquilateralTriangle where
  side_length : ℝ
  midpoint_segment : (ℝ × ℝ) × (ℝ × ℝ) -- Segment joining midpoints of two sides

-- Define the problem statement
theorem max_triangles_formed (A B : EquilateralTriangle)
  (initially_non_overlapping : (A ≠ B))
  (move_right : ∀ t : ℝ, t ≥ 0 → A.moved_horizontally_to (B, t)):
  (∃ k : ℕ, max_number_of_triangles_during_movement A B k ∧ k = 11) :=
sorry

end max_triangles_formed_l655_655966


namespace krista_egg_sales_l655_655844

-- Define the conditions
def hens : ℕ := 10
def eggs_per_hen_per_week : ℕ := 12
def price_per_dozen : ℕ := 3
def weeks : ℕ := 4

-- Define the total money made as the value we want to prove
def total_money_made : ℕ := 120

-- State the theorem
theorem krista_egg_sales : 
  (hens * eggs_per_hen_per_week * weeks / 12) * price_per_dozen = total_money_made :=
by
  sorry

end krista_egg_sales_l655_655844


namespace problem1_problem2_problem3_problem4_l655_655671

-- Problem 1
theorem problem1 : (-23) - (-58) + (-17) = 18 := by
  sorry

-- Problem 2
theorem problem2 : (-8) / (-1 - 1/9) * 0.125 = 9/10 := by
  sorry

-- Problem 3
theorem problem3 : (-1/3 - 1/4 + 1/15) * (-60) = 31 := by
  sorry

-- Problem 4
theorem problem4 : -2^2 + (-4)^3 + |0.8 - 1| * (2 + 1/2)^2 = -66 - 3/4 := by
  sorry

end problem1_problem2_problem3_problem4_l655_655671


namespace transform_sets_l655_655572

-- Define the sets and the conditions
def Set1958 := Fin 1958 → ℤ 

def valid_elements (s : Set1958) := ∀ i, s i = 1 ∨ s i = -1

def transformable (s1 s2 : Set1958) : Prop := 
  ∃ f : ℕ → (Fin 1958 → ℤ), 
    (f 0 = s1) ∧ (∃ n, f n = s2) ∧ ∀ k, ∃ indices : Finset (Fin 1958), 
    indices.card = 11 → f (k+1) = f k ∘ (λ i, if i ∈ indices then -(f k i) else f k i)

-- The main theorem statement
theorem transform_sets (s1 s2 : Set1958) (h1 : valid_elements s1) (h2 : valid_elements s2) :
  transformable s1 s2 :=
sorry

end transform_sets_l655_655572


namespace slope_angle_is_pi_div_4_l655_655759

-- Define the function f(x) = x * ln(x)
def f (x : ℝ) : ℝ := x * Real.log x

-- Define the point of interest x = 1
def x₀ : ℝ := 1

-- Define the slope of the tangent line at x = 1
def slope_at_x₀ : ℝ := Real.log x₀ + 1  -- since the derivative f'(x) = ln(x) + 1

-- Define the correct slope angle we want to prove
def correct_slope_angle : ℝ := Real.pi / 4

-- Prove that the slope angle of the tangent line to the curve y = f(x) at x₀ is π/4
theorem slope_angle_is_pi_div_4 : 
  ∀ x, f x = x * Real.log x → slope_at_x₀ = 1 → atan slope_at_x₀ = correct_slope_angle := 
by
  sorry

end slope_angle_is_pi_div_4_l655_655759


namespace permutation_arrangement_count_l655_655320

theorem permutation_arrangement_count :
  let total_letters := 11
  let num_T := 2
  let num_A := 2
  (11.factorial / (2.factorial * 2.factorial)) = 9979200 :=
by
  sorry

end permutation_arrangement_count_l655_655320


namespace positive_integers_satisfy_l655_655142

theorem positive_integers_satisfy (n : ℕ) (h1 : 25 - 5 * n > 15) : n = 1 :=
by sorry

end positive_integers_satisfy_l655_655142


namespace domain_of_g_l655_655982

noncomputable def g (x : ℝ) : ℝ := log 3 (log 4 (log 5 (log 6 x)))

theorem domain_of_g :
  { x : ℝ | x > 1296 } = { x : ℝ | g x = real.log 3 (real.log 4 (real.log 5 (real.log 6 x))) ∧ ∀ x > 1296, x ∈ ℝ } :=
sorry

end domain_of_g_l655_655982


namespace ronaldo_messi_tie_in_june_l655_655543

theorem ronaldo_messi_tie_in_june :
  let goals_r : List ℕ := [2, 9, 14, 8, 7, 11, 12] in
  let goals_m : List ℕ := [5, 8, 18, 6, 10, 9, 9] in
  let cumulative (l : List ℕ) : List ℕ := l.scanl (+) 0 in
  let R := cumulative goals_r in
  let M := cumulative goals_m in
  ∃ n, n ∈ {1, 2, 3, 4, 5, 6} ∧ R.get! n = M.get! n :=
begin
  sorry
end

end ronaldo_messi_tie_in_june_l655_655543


namespace maximum_tied_teams_in_tournament_l655_655447

theorem maximum_tied_teams_in_tournament : 
  ∀ (n : ℕ), n = 8 →
  (∀ (wins : ℕ), wins = (n * (n - 1)) / 2 →
   ∃ (k : ℕ), k ≤ n ∧ (k > 7 → false) ∧ 
               (∃ (w : ℕ), k * w = wins)) :=
by
  intros n hn wins hw
  use 7
  split
  · exact (by linarith)
  · intro h
    exfalso
    exact h (by linarith)
  · use 4
    calc
      7 * 4 = 28 : by norm_num
      ... = 28 : by rw hw; linarith
  
-- The proof is omitted as per instructions ("sorry" can be used to indicate this).

end maximum_tied_teams_in_tournament_l655_655447


namespace find_area_difference_l655_655633

open Real

noncomputable def area_circle (r : ℝ) : ℝ := π * r^2

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ := (sqrt 3 / 4) * s^2

theorem find_area_difference :
  let r := 3,
  let s := 6,
  let area_c := area_circle r,
  let area_t := area_equilateral_triangle s
  in
  area_c - area_t = 9 * (π - sqrt 3) :=
by
  let r := 3
  let s := 6
  let area_c := area_circle r
  let area_t := area_equilateral_triangle s
  have area_c_eq : area_c = 9 * π,
  { simp [area_circle, r] },
  have area_t_eq : area_t = 9 * sqrt 3,
  { simp [area_equilateral_triangle, s] },
  calc
    area_c - area_t
        = 9 * π - 9 * sqrt 3 : by rw [area_c_eq, area_t_eq]
    ... = 9 * (π - sqrt 3) : by ring

end find_area_difference_l655_655633


namespace doughnuts_left_l655_655322

/-- 
  During a staff meeting, 50 doughnuts were served. If each of the 19 staff ate 2 doughnuts,
  prove that there are 12 doughnuts left. 
-/
theorem doughnuts_left (total_doughnuts : ℕ) (staff_count : ℕ) (doughnuts_per_staff : ℕ) :
  total_doughnuts = 50 → staff_count = 19 → doughnuts_per_staff = 2 →
  total_doughnuts - (staff_count * doughnuts_per_staff) = 12 :=
by
  intros h_total h_staff h_per_staff
  rw [h_total, h_staff, h_per_staff]
  norm_num
  sorry

end doughnuts_left_l655_655322


namespace perp_condition_line_eq_condition_l655_655404

variable (m k : ℝ)
variable (x y : ℝ)

-- Definition for lines l1 and l2
def line1 (x y : ℝ) := (m + 2) * x + m * y - 6 = 0
def line2 (x y : ℝ) := m * x + y - 3 = 0

-- Part 1: Perpendicular condition
theorem perp_condition (hm : m * m + m * 2 = 0) :
    m = 0 ∨ m = -3 :=
sorry

-- Part 2: Point P and intercepts condition 
def point_P (x y : ℝ) := x = 1 ∧ y = 2 * m

def line_l (x y : ℝ) := y - 2 = k * (x - 1)

-- Line intercept conditions
def intercepts (x_intercept y_intercept : ℝ) :=
  x_intercept = 0 ∧ y_intercept = 2 - k ∧ (x = (k - 2) / k) ∧ y = 0 ∧
  (x_intercept = - y_intercept)

theorem line_eq_condition (hP : point_P 1 (2 * m)) (hIntercept : intercepts (- k) (k - 2)) :
    (line_l x y = 0) ∧ (x - y + 1 = 0) ∨ (2 * x - y = 0) :=
sorry

end perp_condition_line_eq_condition_l655_655404


namespace no_integer_roots_l655_655099

theorem no_integer_roots (f g : ℤ[X]) (hf : f = 6 * (X^2 + 1)^2 + 5 * g * (X^2 + 1) - 21 * g^2) : 
  ¬ ∃ x : ℤ, eval x f = 0 :=
sorry

end no_integer_roots_l655_655099


namespace marketing_strategy_increases_mid_sales_l655_655453

structure Product :=
  (name : String)
  (quality : ℕ)
  (price : ℕ)

inductive Category
| premium | mid | economy

structure Store :=
  (products : List Product)
  (category : Product → Category)

def is_premium (p : Product) : Prop := p.name = "A" ∧ ∃ s : Store, s.category p = Category.premium
def is_mid (p : Product) : Prop := p.name = "B" ∧ ∃ s : Store, s.category p = Category.mid
def is_economy (p : Product) : Prop := p.name = "C" ∧ ∃ s : Store, s.category p = Category.economy

theorem marketing_strategy_increases_mid_sales
  (s : Store)
  (hA : ∃ pA : Product, is_premium pA ∧ (pA ∈ s.products) ∧ (pA.price > 0) ∧ (∃ q, q > pA.quality))
  (hB : ∃ pB : Product, is_mid pB ∧ (pB ∈ s.products) ∧ (pB.price < pA.price) ∧ (pB.price > 0))
  (hC : ∃ pC : Product, is_economy pC ∧ (pC ∈ s.products) ∧ (pC.price < pB.price) ∧ (pC.price > 0)) :
  (∃ pB : Product, is_mid pB ∧ pB ∈ s.products → sales pB > 0) :=
sorry

end marketing_strategy_increases_mid_sales_l655_655453


namespace bianca_ate_candy_l655_655348

theorem bianca_ate_candy (original_candies : ℕ) (pieces_per_pile : ℕ) 
                         (number_of_piles : ℕ) 
                         (remaining_candies : ℕ) 
                         (h_original : original_candies = 78) 
                         (h_pieces_per_pile : pieces_per_pile = 8) 
                         (h_number_of_piles : number_of_piles = 6) 
                         (h_remaining : remaining_candies = pieces_per_pile * number_of_piles) :
  original_candies - remaining_candies = 30 := by
  subst_vars
  sorry

end bianca_ate_candy_l655_655348


namespace necessary_sufficient_condition_for_coprimeness_l655_655111

open Nat

theorem necessary_sufficient_condition_for_coprimeness
    (a b c d : ℕ) (h₁ : gcd a b = 1) (h₂ : gcd a c = 1) (h₃ : gcd a d = 1) (h₄ : gcd b c = 1) 
    (h₅ : gcd b d = 1) (h₆ : gcd c d = 1)
    : (∀ n : ℕ, gcd (a * n + b) (c * n + d) = 1) ↔ 
      (∀ p : ℕ, prime p → p ∣ (a * d - b * c) → p ∣ a ∧ p ∣ c) := 
sorry

end necessary_sufficient_condition_for_coprimeness_l655_655111


namespace pastries_selection_l655_655301

/--
Clara wants to purchase six pastries from an ample supply of five types: muffins, eclairs, croissants, scones, and turnovers. 
Prove that there are 210 possible selections using the stars and bars theorem.
-/
theorem pastries_selection : ∃ (selections : ℕ), selections = (Nat.choose (6 + 5 - 1) (5 - 1)) ∧ selections = 210 := by
  sorry

end pastries_selection_l655_655301


namespace largest_prime_factor_of_factorial_sum_is_5_l655_655586

-- Define factorial function
def fact (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * fact (n - 1)

-- Define the specific factorial sums in the problem
def n_factorial_sum : ℕ := fact 6 + fact 7

-- Lean statement to assert the largest prime factor of the sum
theorem largest_prime_factor_of_factorial_sum_is_5 :
  (∀ p : ℕ, (nat.prime p ∧ (p ∣ n_factorial_sum)) → p ≤ 5) ∧
  (∃ p : ℕ, nat.prime p ∧ (p ∣ n_factorial_sum) ∧ p = 5) :=
by
  -- Proof of the theorem
  sorry

end largest_prime_factor_of_factorial_sum_is_5_l655_655586


namespace tan_product_range_l655_655829

theorem tan_product_range (α β γ : ℝ) (h : cos α^2 + cos β^2 + cos γ^2 = 1) :
  2 * sqrt 2 ≤ (tan α * tan β * tan γ) :=
sorry

end tan_product_range_l655_655829


namespace transform_sets_l655_655573

-- Define the sets and the conditions
def Set1958 := Fin 1958 → ℤ 

def valid_elements (s : Set1958) := ∀ i, s i = 1 ∨ s i = -1

def transformable (s1 s2 : Set1958) : Prop := 
  ∃ f : ℕ → (Fin 1958 → ℤ), 
    (f 0 = s1) ∧ (∃ n, f n = s2) ∧ ∀ k, ∃ indices : Finset (Fin 1958), 
    indices.card = 11 → f (k+1) = f k ∘ (λ i, if i ∈ indices then -(f k i) else f k i)

-- The main theorem statement
theorem transform_sets (s1 s2 : Set1958) (h1 : valid_elements s1) (h2 : valid_elements s2) :
  transformable s1 s2 :=
sorry

end transform_sets_l655_655573


namespace gene_segregation_error_in_secondary_spermatocyte_l655_655054

-- Definitions based on the conditions
def dominant : Type := "E"
def recessive : Type := "e"
def genotype : Type := List Type

def male_parent_genotype : genotype := [dominant, recessive] -- Ee
def female_parent_genotype : genotype := [recessive, recessive] -- ee
def offspring_genotype : genotype := [dominant, dominant, recessive] -- EEe

-- The Lean proposition to prove
theorem gene_segregation_error_in_secondary_spermatocyte :
  (∃ (male_parent female_parent offspring : genotype),
  male_parent = male_parent_genotype ∧ 
  female_parent = female_parent_genotype ∧ 
  offspring = offspring_genotype) →
  offspring_genotype = [dominant, dominant, recessive] → 
  "Error occurred in secondary spermatocyte" :=
by
  sorry

end gene_segregation_error_in_secondary_spermatocyte_l655_655054


namespace fg2_eq_14_l655_655492

def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x
def g (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 1

theorem fg2_eq_14 : f (g 2) = 14 := by
  sorry

end fg2_eq_14_l655_655492


namespace sum_of_ages_l655_655842

def brother_age : ℕ := 8
def john_age (B : ℕ) : ℕ := 6 * B - 4

theorem sum_of_ages :
  let B := brother_age in
  let J := john_age B in
  B + J = 52 :=
by
  sorry

end sum_of_ages_l655_655842


namespace random_sampling_cannot_prove_inequality_l655_655201

-- Define the methods as Lean types
inductive Method
| comparison
| random_sampling
| synthetic_analytic
| contradiction_scaling

-- Specify a predicate 'can_prove_inequality' indicating that a method can prove inequality
def can_prove_inequality : Method → Prop
| Method.comparison := true
| Method.random_sampling := false
| Method.synthetic_analytic := true
| Method.contradiction_scaling := true

-- The main theorem proving the random sampling method cannot be used to prove inequalities
theorem random_sampling_cannot_prove_inequality : 
  ¬ can_prove_inequality Method.random_sampling :=
by {
  sorry
}

end random_sampling_cannot_prove_inequality_l655_655201


namespace Lakers_win_probability_championship_l655_655918

theorem Lakers_win_probability_championship :
  let prob_win := (2 / 3 : ℝ)
  let prob_lose := (1 / 3 : ℝ)
  let prob_game k :=
    (choose (4 + k) k) * (prob_win^5) * (prob_lose^k)
  in
  (Enum.sum [prob_game 0, prob_game 1, prob_game 2, prob_game 3, prob_game 4]) = 0.85 :=
by
  sorry

end Lakers_win_probability_championship_l655_655918


namespace island_knight_problem_l655_655959

theorem island_knight_problem :
  ∃ k : ℕ, (k = 0 ∨ k = 10) ∧ (∀ i ∈ {1, 2, ..., 100}, (i % k = 0 ↔ (i is a position of a knight))) := sorry

end island_knight_problem_l655_655959


namespace hexagon_chord_length_valid_l655_655245

def hexagon_inscribed_chord_length : ℚ := 48 / 49

theorem hexagon_chord_length_valid : 
    ∃ (p q : ℕ), gcd p q = 1 ∧ hexagon_inscribed_chord_length = p / q ∧ p + q = 529 :=
sorry

end hexagon_chord_length_valid_l655_655245


namespace six_digit_number_not_divisible_by_15_and_2_l655_655432

-- Define the six digits
def digits := [1, 2, 3, 5, 7, 8]

-- Define a function to check divisibility by n
def divisible_by (n : ℕ) (k : ℕ) : Prop := k % n = 0

-- Define the probability of forming a number divisible by both 15 and 2
theorem six_digit_number_not_divisible_by_15_and_2 : 
  ∀ (perm : List ℕ), perm.permutations → perm = digits →
  ¬ (divisible_by 15 (digits.sum) ∧ divisible_by 2 (digits.head)) :=
by {
  sorry
}

end six_digit_number_not_divisible_by_15_and_2_l655_655432


namespace intersection_of_sets_l655_655089

theorem intersection_of_sets :
  let S := {x : ℝ | x > -1/2}
  let T := {x : ℝ | 2^(3 * x - 1) < 1}
  S ∩ T = {x : ℝ | -1/2 < x ∧ x < 1/3} :=
by
  sorry

end intersection_of_sets_l655_655089


namespace correct_propositions_l655_655004

-- Definitions for the conditions
variables {m n l : Type} {α : Type}

-- Propositions
def proposition1 (hm: m ∥ l) (hn: n ∥ l) : Prop := m ∥ n
def proposition2 (hmn: m ∥ n) (hma: m ∥ α) : Prop := n ∥ α
def proposition3 (hnm: n ⊥ m) (hma: m ⊆ α) : Prop := n ⊥ α
def proposition4 (hma: m ⊥ α) (hna: n ⊆ α) : Prop := m ⊥ n

-- The proof problem statement
theorem correct_propositions {m n l : Type} {α : Type}
  (prop1 : m ∥ l → n ∥ l → m ∥ n)
  (prop2 : m ∥ n → m ∥ α → n ∥ α)
  (prop3 : n ⊥ m → m ⊆ α → n ⊥ α)
  (prop4 : m ⊥ α → n ⊆ α → m ⊥ n) :
  ∃ (n_correct : ℕ), n_correct = 2 := 
by
  -- Placeholder for the actual proof logic
  sorry

end correct_propositions_l655_655004


namespace sum_of_matching_numbers_correct_l655_655630

def f (i j : ℕ) : ℕ := 17 * (i - 1) + j
def g (i j : ℕ) : ℕ := 13 * (j - 1) + i

def valid_pair (i j : ℕ) : Prop := f i j = g i j

def valid_pairs := {(1, 1), (4, 5), (7, 9), (10, 13), (13, 17)}

def sum_of_matching_numbers : ℕ :=
  valid_pairs.to_finset.fold (λ p acc, acc + f p.1 p.2) 0

theorem sum_of_matching_numbers_correct : sum_of_matching_numbers = 555 := 
  sorry

end sum_of_matching_numbers_correct_l655_655630


namespace sum_of_possible_values_of_x_l655_655019

noncomputable def cubic_sum_of_roots : ℝ :=
  let f := (λ x : ℝ, (x + 2)^2 * (x - 3) - 40) in
  let a := 1 in
  let b := 1 in
  -b / a

theorem sum_of_possible_values_of_x : 
  let f := (λ x : ℝ, (x + 2)^2 * (x - 3) - 40) in
  cubic_sum_of_roots = -1 := 
by
  let f := (λ x : ℝ, (x + 2)^2 * (x - 3) - 40) 
  have H : f = (λ x, x^3 + x^2 - 8x - 52) :=
    sorry
  have roots_known : 
    ∀ x : ℝ, (x^3 + x^2 - 8x - 52 = 0) ↔ ((x + 2)^2 * (x - 3) - 40 = 0) := 
    sorry
  exact cubic_sum_of_roots = -1

end sum_of_possible_values_of_x_l655_655019


namespace line_tangent_to_curve_at_point_l655_655931

theorem line_tangent_to_curve_at_point :
  ∃ (b : ℝ), ∃ (a k : ℝ), ∃ (x y : ℝ),
    y = k * x + b ∧
    y = x^3 + a * x + 1 ∧
    x = 2 ∧ y = 3 ∧
    (∀ (x : ℝ), y = x^3 + a * x + 1 → deriv (λ x, x^3 + a * x + 1) 2 = k) ∧
    b = -15 :=
sorry

end line_tangent_to_curve_at_point_l655_655931


namespace parabola_properties_l655_655738

structure Point :=
  (x : ℚ)
  (y : ℚ)

def parabola (a b c : ℚ) (p : Point) : Prop :=
  p.y = a * p.x ^ 2 + b * p.x + c

theorem parabola_properties :
  ∃ a b c, 
    (∀ p : Point, p = ⟨-2, 0⟩ → parabola a b c p) ∧
    (∀ p : Point, p = ⟨1, 0⟩ → parabola a b c p) ∧
    (∀ p : Point, p = ⟨2, 8⟩ → parabola a b c p) ∧
    (parabola a b c ⟨0, -(b^2 - 4 * a * c) / (4 * a)⟩ ∧ 
    (let h := -(b : ℚ) / (2 * a) in ∃ y, parabola a b c ⟨h, y⟩ ∧ 
        ⟨h, y⟩ = ⟨-1/2, -9/2⟩)) :=
sorry

end parabola_properties_l655_655738


namespace area_of_enclosed_region_l655_655334

noncomputable def enclosedRegionArea : ℝ :=
  let f : ℝ → ℝ → Prop := λ x y, |x - 80| + |y| = |x / 2|
  let kite_diagonal1 : ℝ := 160 - 160 / 3
  let kite_diagonal2 : ℝ := 40 - (-40)
  1 / 2 * kite_diagonal1 * kite_diagonal2

theorem area_of_enclosed_region : enclosedRegionArea = 12800 / 3 := 
by 
  sorry

end area_of_enclosed_region_l655_655334


namespace scientific_notation_of_diameter_l655_655540

def diameter_H1N1_influenza_virus : ℝ := 0.00000011

theorem scientific_notation_of_diameter : diameter_H1N1_influenza_virus = 1.1 * 10^(-7) := by
  sorry

end scientific_notation_of_diameter_l655_655540


namespace quadratic_expression_value_l655_655857

-- Given conditions
variables (a : ℝ) (h : 2 * a^2 + 3 * a - 2022 = 0)

-- Prove the main statement
theorem quadratic_expression_value :
  2 - 6 * a - 4 * a^2 = -4042 :=
sorry

end quadratic_expression_value_l655_655857


namespace plane_divides_diagonal_l655_655899

variables (A B C D A1 B1 C1 D1 M : Type) [Point A B C D A1 B1 C1 D1 M] 
open_locale classical

noncomputable def CM_MD_ratio : ℚ := 1 / 2

theorem plane_divides_diagonal (hM : M ∈ segment C D) (hCM_MD : CM_MD_ratio = 1 / 2) 
  (h_plane : ∀ (P Q : Type) [Point P Q], parallel (plane P Q) (line (D B)) ∧ parallel (plane P Q) (line (A1 C1))) : 
  divides_ratio (line (A1 C)) M 1 11 :=
sorry

end plane_divides_diagonal_l655_655899


namespace range_of_a_l655_655384

-- Definitions of position conditions in the 4th quadrant
def PosInFourthQuad (x y : ℝ) : Prop := (x > 0) ∧ (y < 0)

-- Statement to prove
theorem range_of_a (a : ℝ) (h : PosInFourthQuad (2 * a + 4) (3 * a - 6)) : -2 < a ∧ a < 2 :=
  sorry

end range_of_a_l655_655384


namespace trajectory_of_moving_circle_l655_655641

noncomputable def circle_eq_form (x y k: ℝ) := x^2 + y^2 + k = 0

theorem trajectory_of_moving_circle :
  let C : ℝ → ℝ → Prop := λ x y, circle_eq_form (x - 3) y 1
  let O : ℝ → ℝ → Prop := λ x y, circle_eq_form x y 1
  let r : ℝ
  MO (x y : ℝ) (M : ℝ×ℝ) := (M.fst - x)^2 + (M.snd - y)^2 = (r + 1)^2
  MC (x y : ℝ) (M : ℝ×ℝ) := (M.fst - x)^2 + (M.snd - y)^2 = (r - 1)^2
  in ∃ M : ℝ × ℝ, (MO 0 0 M → MC 3 0 M → (trajectory M = Hyperbola)) :=
sorry

end trajectory_of_moving_circle_l655_655641


namespace largest_prime_factor_of_n_l655_655487

theorem largest_prime_factor_of_n 
  (n : ℕ) 
  (hn_div_36 : 36 ∣ n)
  (hn2_cube : ∃ k, n^2 = k^3)
  (hn3_square : ∃ m, n^3 = m^2)
  (hn_min : ∀ m, (36 ∣ m ∧ ∃ k, m^2 = k^3 ∧ ∃ p, m^3 = p^2) → n ≤ m) : 
  nat.prime 3 ∧ (¬ ∃ p, nat.prime p ∧ p > 3 ∧ p ∣ n) :=
by sorry

end largest_prime_factor_of_n_l655_655487


namespace acute_triangle_sine_inequality_l655_655091

theorem acute_triangle_sine_inequality
  (α β γ : ℝ)
  (h1 : ∀ θ, θ ∈ Set.ofList [α, β, γ] → θ < Real.pi / 2)
  (h2 : 0 < α ∧ 0 < β ∧ 0 < γ)
  (h3 : α + β + γ = Real.pi)
  (h_condition : α < β ∧ β < γ) :
  Real.sin (2 * α) > Real.sin (2 * β) ∧ Real.sin (2 * β) > Real.sin (2 * γ) :=
by
  sorry

end acute_triangle_sine_inequality_l655_655091


namespace holder_inequality_l655_655612

noncomputable def problem_statement (p q : ℝ) (h : 1/p + 1/q = 1)
    (a b : Fin n.succ → ℝ) : Prop :=
    (∀ i, 0 < a i ∧ 0 < b i) →
    (∑ i, (a i) * (b i) ≤ ((∑ i, (a i) ^ p) ^ (1 / p)) * ((∑ i, (b i) ^ q) ^ (1 / q)))

variables {n : ℕ}

theorem holder_inequality (p q : ℝ) (h : 1/p + 1/q = 1)
    (a b : Fin n.succ → ℝ) (positivity : ∀ i, 0 < a i ∧ 0 < b i) :
    (∑ i, (a i) * (b i))
    ≤ ((∑ i, (a i) ^ p) ^ (1 / p)) * ((∑ i, (b i) ^ q) ^ (1 / q)) :=
sorry

end holder_inequality_l655_655612


namespace angle_equality_l655_655084

-- Define the points and midpoints as described in the problem
variables {A B C K L B1 C1 : Type} [pseudo_metric_space A] 

-- Conditions: Let B1 and C1 be the midpoints of AC and AB of triangle ABC respectively.
def midpoint (A B : Type) (C1 : Type) := sorry -- Assume some definition for midpoint

-- Tangents to the circumcircle at B and C meet the rays CC1 and BB1 at points K and L respectively.
def tangent_points (B C : Type) (K B1 C1 L : Type) := sorry -- Assume some definition for tangent points

-- The angles to be proved as equal
variables (BAK CAL: Type)

-- Prove angles BAK and CAL are equal given the conditions
theorem angle_equality 
  (h1: midpoint AC B1) 
  (h2: midpoint AB C1) 
  (h3: tangent_points B C K B1 C1 L) : 
  (BAK = CAL) :=
sorry

end angle_equality_l655_655084


namespace find_value_of_a3_l655_655364

noncomputable def geometric_sequence (n : ℕ) : ℕ → ℝ 
| 1 => 2
| 5 => 8
| n+1 => geometric_sequence n * r -- r stands for the ratio

theorem find_value_of_a3 (a : ℕ → ℝ) (h₁ : a 1 = 2) (h₅ : a 5 = 8) : a 3 = 4 := 
by
  sorry

end find_value_of_a3_l655_655364


namespace rational_sum_of_cubes_l655_655124

theorem rational_sum_of_cubes (t : ℚ) : 
    ∃ (a b c : ℚ), t = (a^3 + b^3 + c^3) :=
by
  sorry

end rational_sum_of_cubes_l655_655124


namespace no_integer_roots_l655_655102

theorem no_integer_roots (n : ℕ) (a : Fin (n + 1) → ℤ)
    (h₀ : odd (a 0)) 
    (h₁ : odd (∑ i : Fin (n + 1), a i)) :
    ∀ x : ℤ, x ≠ 0 → p x = a.foldr (λ (i : Fin (n + 1)) (acc : ℤ), a i * x^i + acc) 0 ≠ 0 :=
by 
  intros x hx
  sorry

end no_integer_roots_l655_655102


namespace hall_volume_proof_l655_655242

-- Define the given conditions.
def hall_length (l : ℝ) : Prop := l = 18
def hall_width (w : ℝ) : Prop := w = 9
def floor_ceiling_area_eq_wall_area (h l w : ℝ) : Prop := 
  2 * (l * w) = 2 * (l * h) + 2 * (w * h)

-- Define the volume calculation.
def hall_volume (l w h V : ℝ) : Prop := 
  V = l * w * h

-- The main theorem stating that the volume is 972 cubic meters.
theorem hall_volume_proof (l w h V : ℝ) 
  (length : hall_length l) 
  (width : hall_width w) 
  (fc_eq_wa : floor_ceiling_area_eq_wall_area h l w) 
  (volume : hall_volume l w h V) : 
  V = 972 :=
  sorry

end hall_volume_proof_l655_655242


namespace howard_items_l655_655418

theorem howard_items (a b c : ℕ) (h1 : a + b + c = 40) (h2 : 40 * a + 300 * b + 400 * c = 5000) : a = 20 :=
by
  sorry

end howard_items_l655_655418


namespace compare_a_b_l655_655270

variable (c : ℝ)
variable (h : c > 1)
def a : ℝ := (Real.sqrt (c + 1)) - (Real.sqrt c)
def b : ℝ := (Real.sqrt c) - (Real.sqrt (c - 1))

theorem compare_a_b : a h < b h := by
  sorry

end compare_a_b_l655_655270


namespace add_and_multiply_l655_655136

def num1 : ℝ := 0.0034
def num2 : ℝ := 0.125
def num3 : ℝ := 0.00678
def sum := num1 + num2 + num3

theorem add_and_multiply :
  (sum * 2) = 0.27036 := by
  sorry

end add_and_multiply_l655_655136


namespace decreasing_interval_of_g_l655_655523

theorem decreasing_interval_of_g :
  ∀ x, 
    (∀ x₁ x₂ : ℝ, -π / 12 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 5 * π / 12 → cos (2 * x₁ + π / 6) ≥ cos (2 * x₂ + π / 6)) :=
by
  sorry

end decreasing_interval_of_g_l655_655523


namespace domain_of_g_l655_655546

theorem domain_of_g :
  (∀ (g : ℝ → ℝ), (∀ x : ℝ, x ≠ 0 → (∃ z : ℝ, x = 1/z)) →
  (∀ x : ℝ, x ≠ 0 → g(x) + g(1/x) = 2 * x) →
  (∀ x : ℝ, x ≠ 0 → g(1/x) + g(x) = 2/x) →
  ∀ x : ℝ, (x = -1 ∨ x = 1)) :=
by
  sorry

end domain_of_g_l655_655546


namespace sin_value_l655_655421

theorem sin_value (A : ℝ) (h : real.cot A + real.csc A = 2) : real.sin A = 4 / 5 :=
sorry

end sin_value_l655_655421


namespace two_aces_or_at_least_one_king_probability_l655_655038

theorem two_aces_or_at_least_one_king_probability :
  let total_cards := 52
  let total_aces := 5
  let total_kings := 4
  let prob_two_aces := (total_aces / total_cards) * ((total_aces - 1) / (total_cards - 1))
  let prob_exactly_one_king := ((total_kings / total_cards) * ((total_cards - total_kings) / (total_cards - 1))) +
                               (((total_cards - total_kings) / total_cards) * (total_kings / (total_cards - 1)))
  let prob_two_kings := (total_kings / total_cards) * ((total_kings - 1) / (total_cards - 1))
  let prob_at_least_one_king := prob_exactly_one_king + prob_two_kings
  let prob_question := prob_two_aces + prob_at_least_one_king
in prob_question = 104 / 663 := by
  let total_cards := 52
  let total_aces := 5
  let total_kings := 4
  let prob_two_aces := (total_aces / total_cards) * ((total_aces - 1) / (total_cards - 1))
  let prob_exactly_one_king := ((total_kings / total_cards) * ((total_cards - total_kings) / (total_cards - 1))) +
                               (((total_cards - total_kings) / total_cards) * (total_kings / (total_cards - 1)))
  let prob_two_kings := (total_kings / total_cards) * ((total_kings - 1) / (total_cards - 1))
  let prob_at_least_one_king := prob_exactly_one_king + prob_two_kings
  let prob_question := prob_two_aces + prob_at_least_one_king
  have h_prob_two_aces : prob_two_aces = 10 / 1326 := sorry
  have h_prob_exactly_one_king : prob_exactly_one_king = 32 / 221 := sorry
  have h_prob_two_kings : prob_two_kings = 1 / 221 := sorry
  have h_prob_at_least_one_king : prob_at_least_one_king = (32 / 221) + (1 / 221) := sorry
  have h_prob_at_least_one_king := prob_at_least_one_king = 33 / 221 := sorry 
  have h_final := prob_question = (10 / 1326) + (33 / 221) := sorry
  have h_final := prob_question = (10 / 1326) + (198 / 1326) := sorry
  have h_final := prob_question = 208 / 1326 := sorry 
  exact h_final = 104 / 663 sorry 

end two_aces_or_at_least_one_king_probability_l655_655038


namespace sum_powers_div_5_iff_l655_655515

theorem sum_powers_div_5_iff (n : ℕ) (h : n > 0) : (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := 
sorry

end sum_powers_div_5_iff_l655_655515


namespace problem1_problem2_l655_655372

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 + 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x^2 - 6 * x - 72 ≤ 0) ∧ (x^2 + x - 6 > 0)

-- Problem 1: Proving the range of x
theorem problem1 (x : ℝ) (h₁ : a = -1) (h₂ : ∀ (x : ℝ), p x a → q x) : 
  x ∈ {x : ℝ | -6 ≤ x ∧ x < -3} ∨ x ∈ {x : ℝ | 1 < x ∧ x ≤ 12} := sorry

-- Problem 2: Proving the range of a
theorem problem2 (a : ℝ) (h₃ : (∀ x, q x → p x a) ∧ ¬ (∀ x, ¬q x → ¬p x a)) : 
  -4 ≤ a ∧ a ≤ -2 := sorry

end problem1_problem2_l655_655372


namespace number_of_questionnaire_C_l655_655181

theorem number_of_questionnaire_C (n total selected initial interval : ℕ)
  (A B C : set ℕ)
  (h1 : total = 960)
  (h2 : selected = 32)
  (h3 : initial = 9)
  (h4 : interval = total / selected)
  (h5 : A = {x : ℕ | 1 ≤ x ∧ x ≤ 450})
  (h6 : B = {x : ℕ | 451 ≤ x ∧ x ≤ 750})
  (h7 : C = {x : ℕ | x ≥ 751}) :
  let num_questionnaire_C := selected - (750 - initial) / interval in
  num_questionnaire_C = 8 :=
by
  sorry

end number_of_questionnaire_C_l655_655181


namespace no_root_in_interval_l655_655696

noncomputable def f (x : ℝ) : ℝ := sin (2 * x) + 5 * sin x + 5 * cos x + 1

theorem no_root_in_interval : ¬ ∃ x : ℝ, 0 ≤ x ∧ x ≤ (π / 4) ∧ f x = 0 := 
by
  sorry

end no_root_in_interval_l655_655696


namespace expected_value_problem_l655_655935

noncomputable def expected_larger_neighbors : ℚ :=
  17 / 3

theorem expected_value_problem :
  ∃ (arrangement : list ℕ), arrangement.perm ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) ∧
  (expected_larger_neighbors = 17 / 3) :=
sorry

end expected_value_problem_l655_655935


namespace domain_and_range_of_f_l655_655542

noncomputable def f (x : Real) : Real := (sin (2 * x) * cos x) / (1 - sin x)

theorem domain_and_range_of_f : 
  (∀ x : Real, x ∉ { k : Int | x = 2 * k * Real.pi + Real.pi / 2 }) ∧ 
  (∀ y : Real, y ∈ Set.Ico (-1 / 2 : Real) 4) := 
sorry

end domain_and_range_of_f_l655_655542


namespace stationery_store_loss_l655_655260

-- Condition 1: Selling price of each calculator is 120 yuan.
def selling_price := 120

-- Condition 2: One calculator sold at a 20% profit.
def cost_price_profit := 100
def profit_rate := 0.20

-- Condition 3: The other calculator sold at a 20% loss.
def cost_price_loss := 150
def loss_rate := 0.20

-- Prove that the total loss is 10 yuan.
theorem stationery_store_loss :
  (selling_price - cost_price_profit) + (selling_price - cost_price_loss) = -10 :=
by
  sorry

end stationery_store_loss_l655_655260


namespace definite_integral_value_l655_655952

theorem definite_integral_value :
  ∫ x in 0..1, (Real.exp x + 2 * x) = Real.exp 1 := 
by
  sorry

end definite_integral_value_l655_655952


namespace sum_products_of_chords_l655_655811

variable {r x y u v : ℝ}

theorem sum_products_of_chords (h1 : x * y = u * v) (h2 : 4 * r^2 = (x + y)^2 + (u + v)^2) :
  x * (x + y) + u * (u + v) = 4 * r^2 := by
sorry

end sum_products_of_chords_l655_655811


namespace comm_ring_of_center_condition_l655_655620

variable {R : Type*} [Ring R]

def in_center (x : R) : Prop := ∀ y : R, (x * y = y * x)

def is_commutative (R : Type*) [Ring R] : Prop := ∀ a b : R, a * b = b * a

theorem comm_ring_of_center_condition (h : ∀ x : R, in_center (x^2 - x)) : is_commutative R :=
sorry

end comm_ring_of_center_condition_l655_655620


namespace john_total_amount_l655_655475

def grandpa_amount : ℕ := 30
def grandma_amount : ℕ := 3 * grandpa_amount
def aunt_amount : ℕ := 3 / 2 * grandpa_amount
def uncle_amount : ℕ := 2 / 3 * grandma_amount

def total_amount : ℕ :=
  grandpa_amount + grandma_amount + aunt_amount + uncle_amount

theorem john_total_amount : total_amount = 225 := by sorry

end john_total_amount_l655_655475


namespace handshake_count_l655_655287

theorem handshake_count {teams : Fin 4 → Fin 2 → Prop}
    (h_teams_disjoint : ∀ (i j : Fin 4) (x y : Fin 2), i ≠ j → teams i x → teams j y → x ≠ y)
    (unique_partner : ∀ (i : Fin 4) (x1 x2 : Fin 2), teams i x1 → teams i x2 → x1 = x2) : 
    24 = (∑ i : Fin 8, (∑ j : Fin 8, if i ≠ j ∧ ¬(∃ k : Fin 4, teams k i ∧ teams k j) then 1 else 0)) / 2 :=
by sorry

end handshake_count_l655_655287


namespace chord_intersects_inner_circle_l655_655579

-- Define the radii of the circles
def r_inner : ℝ := 1
def r_outer : ℝ := 2

-- Define the circle centers (concentric circles have the same center)
def center : ℝ × ℝ := (0, 0)

-- Noncomputable definition to state the probability calculation
noncomputable def chord_intersects_inner_circle_probability : ℝ :=
  -- This represents the given probability which needs to be proved
  1 / 3

-- The theorem to prove
theorem chord_intersects_inner_circle :
  ∀ (A B : ℝ × ℝ), 
  -- A and B are points on the outer circle, uniformly random points are assumed to be on the circle of radius 2
  dist center A = r_outer ∧ dist center B = r_outer → 
  -- Probability of the chord AB intersecting the inner circle with radius 1
  (chord_intersects_inner_circle_probability = 1 / 3) :=
begin
  sorry
end

end chord_intersects_inner_circle_l655_655579


namespace pies_can_be_made_l655_655922

def total_apples : Nat := 51
def apples_handout : Nat := 41
def apples_per_pie : Nat := 5

theorem pies_can_be_made :
  ((total_apples - apples_handout) / apples_per_pie) = 2 := by
  sorry

end pies_can_be_made_l655_655922


namespace complex_inequality_equivalence_l655_655838

open Complex

theorem complex_inequality_equivalence {z a c : ℂ} :
  (|z| - z.re ≤ 1 / 2) ↔ (z = a * c ∧ |conj c - a| ≤ 1) :=
sorry

end complex_inequality_equivalence_l655_655838


namespace probability_sum_three_l655_655535

-- Definitions based on problem conditions
def rounded_interval (x : ℝ) : ℕ :=
  if x < 0.5 then 0
  else if x < 1.5 then 1
  else if x <= 2.5 then 2
  else 0  -- This case won't happen given the problem conditions

def sum_three_probability : ℝ :=
  let interval1 := set.Ico 0.5 1.5
  let interval2 := set.Ico 1.5 2.5
  (interval1.measure + interval2.measure) / 2.5

-- Statement to be proved
theorem probability_sum_three : sum_three_probability = 3 / 5 :=
  sorry

end probability_sum_three_l655_655535


namespace cannot_form_right_triangle_l655_655203

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem cannot_form_right_triangle : ¬ is_right_triangle 40 50 60 := 
by
  sorry

end cannot_form_right_triangle_l655_655203


namespace percentage_cut_is_50_l655_655560

-- Conditions
def yearly_subscription_cost : ℝ := 940.0
def reduction_amount : ℝ := 470.0

-- Assertion to be proved
theorem percentage_cut_is_50 :
  (reduction_amount / yearly_subscription_cost) * 100 = 50 :=
by
  sorry

end percentage_cut_is_50_l655_655560


namespace length_KD_l655_655160

-- Given data
variables (A B C D K L : Type) [AddGroup A] [Module ℝ A]

-- Conditions
-- A trapezoid ABCD
axiom trapezoid (ABCD : Set (ℝ × ℝ))

-- K is the midpoint of side AB
axiom midpoint (A B K : A) (h_midpoint : K = (A + B) / 2)

-- The perpendicular from K to side CD intersects at L
axiom perpendicular (CD KL : Set (ℝ × ℝ)) (h_perp : ∃ L, L ∈ CD ∧ K ∈ KL ∧ ∀ p ∈ KL, ⟪K - p, L - p⟫ = 0)

-- Area condition
axiom area_condition {S_AKLD S_BKLC : ℝ}
  (h_area : S_AKLD = 5 * S_BKLC)

-- Given segment lengths
variable (CL DL KC KD : ℝ)
axiom h_lengths1 : CL = 3
axiom h_lengths2 : DL = 15
axiom h_lengths3 : KC = 4

-- The proof statement
theorem length_KD : KD = 20 := sorry

end length_KD_l655_655160


namespace circle_equation_l655_655753

/-- Given that the center of the circle is C at (t, 2/t), where t ∈ ℝ, t ≠ 0, 
and the circle intersects the x-axis at points O and A, and the y-axis at points 
O and B (where O is the origin). Line 2x+y-4=0 intersects the circle at points 
M and N such that OM=ON. Prove that the equation of the circle is (x-2)² + (y-1)² = 5. -/
theorem circle_equation (t : ℝ) (h : t ≠ 0) :
  let C := ⟨t, 2 / t⟩ in
  let equation := "(x - 2)^2 + (y - 1)^2 = 5" in
  (∃ M N : ℝ × ℝ,
    let line_eq := 2 * M.1 + M.2 - 4 = 0 ∧ 2 * N.1 + N.2 - 4 = 0 in
    let circle_eq := (M.1 - t) ^ 2 + (M.2 - 2 / t) ^ 2 = (N.1 - t) ^ 2 + (N.2 - 2 / t) ^ 2 in
    line_eq ∧ circle_eq)
→ equation = "(x - 2)^2 + (y - 1)^2 = 5" := sorry

end circle_equation_l655_655753


namespace quadratic_function_determination_l655_655764

noncomputable def f : ℝ → ℝ := λ x, a * x^2 + b * x + 1

theorem quadratic_function_determination 
    (a b : ℝ)
    (h1 : ∀ x : ℝ, f x = a * x^2 + b * x + 1)
    (h2 : f (-1) = 0)
    (h3 : ∀ x : ℝ, f x = a * (x + 1)^2) :
    f x = x^2 + 2 * x + 1 :=
by 
    sorry

end quadratic_function_determination_l655_655764


namespace ramu_profit_correct_l655_655212

open Real

def ramu_profit_percent (P R S : ℝ) : ℝ :=
  let total_cost := P + R
  let profit := S - total_cost
  (profit / total_cost) * 100

theorem ramu_profit_correct :
  ramu_profit_percent 42000 10000 64900 ≈ 24.81 :=
by
  sorry

end ramu_profit_correct_l655_655212


namespace intercept_sum_l655_655248

theorem intercept_sum (x y : ℝ) :
  (y - 3 = 6 * (x - 5)) →
  (∃ x_intercept, (y = 0) ∧ (x_intercept = 4.5)) →
  (∃ y_intercept, (x = 0) ∧ (y_intercept = -27)) →
  (4.5 + (-27) = -22.5) :=
by
  intros h_eq h_xint h_yint
  sorry

end intercept_sum_l655_655248


namespace hexagon_angle_sum_l655_655012

theorem hexagon_angle_sum
  (a1 a2 a3 a4 a5 : ℝ)
  (h_sum : a1 + a2 + a3 + a4 + a5 = 537)
  (total_sum : 720 = 4 * 180):
  let Q := 720 - (a1 + a2 + a3 + a4 + a5)
  in Q = 183 :=
by
  have eq : Q = 720 - 537 := rfl
  rw [h_sum] at eq
  simp at eq
  exact eq

end hexagon_angle_sum_l655_655012


namespace non_congruent_rectangles_l655_655646

theorem non_congruent_rectangles (h w : ℕ) (hp : 2 * (h + w) = 80) :
  ∃ n, n = 20 := by
  sorry

end non_congruent_rectangles_l655_655646


namespace f_behavior_l655_655760

-- Define the function and conditions in the problem
def y_decreasing_on_domain (a : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → a^x - a^(-x) > a^y - a^(-y)

noncomputable def f (a x : ℝ) := a^(sqrt (-x^2 + 4*x - 3))

theorem f_behavior (a : ℝ) :
  (0 < a) → (a ≠ 1) → y_decreasing_on_domain a →
  (∀ x, 1 < x ∧ x < 2 ↔ ∀ {x1 x2 : ℝ}, x1 ∈ Ioo 1 2 → x2 ∈ Ioo 1 2 → x1 < x2 → f a x1 < f a x2) ∧
  (∀ x, 2 < x ∧ x < 3 ↔ ∀ {x1 x2 : ℝ}, x1 ∈ Ioo 2 3 → x2 ∈ Ioo 2 3 → x1 < x2 → f a x2 < f a x1) :=
by
  sorry

end f_behavior_l655_655760


namespace curve_equations_and_intersection_distance_l655_655049

theorem curve_equations_and_intersection_distance (α t₁ t₂ : ℝ) (hα : α ≠ π/2) 
(h₁ : ∃ t, (2 + t * cos α, 1 + t * sin α) ∈ (set_of (λ (x, y : ℝ), (x - 3) ^ 2 + y ^ 2 = 5))) 
(h₂ : t₁ + t₂ = 0) :
  (∀ (t : ℝ), (2 + t * cos α, 1 + t * sin α) ∈ (set_of (λ (x, y : ℝ), y = (x - 2) * tan α + 1))) ∧
  (∀ (ρ θ : ℝ), ρ^2 - 6 * cos θ + 4 = 0 → (ρ * cos θ - 3)^2 + (ρ * sin θ)^2 = 5) ∧
  abs (sqrt (((2 + t₁ * cos α) - (2 + t₂ * cos α))^2 + ((1 + t₁*sin α) - (1 + t₂*sin α))^2)) = 2 * sqrt 3 :=
by
  sorry

end curve_equations_and_intersection_distance_l655_655049


namespace find_b_l655_655092

open Matrix

def a : ℝ^3 := ![3, 2, 4]
def b (x y z : ℝ) : ℝ^3 := ![x, y, z]

theorem find_b (x y z : ℝ) :
  (a.dot_product (b x y z) = 20) ∧
  (a.cross_product (b x y z) = ![-8, 16, -2]) :=
sorry

end find_b_l655_655092


namespace paul_earns_from_license_plates_l655_655121

theorem paul_earns_from_license_plates
  (plates_from_40_states : ℕ)
  (total_50_states : ℕ)
  (reward_per_percentage_point : ℕ)
  (h1 : plates_from_40_states = 40)
  (h2 : total_50_states = 50)
  (h3 : reward_per_percentage_point = 2) :
  (40 / 50) * 100 * 2 = 160 := 
sorry

end paul_earns_from_license_plates_l655_655121


namespace probability_reach_2_3_in_7_steps_l655_655914

theorem probability_reach_2_3_in_7_steps (q : ℚ) (m n : ℕ) (h_rel_prime : Nat.coprime m n)
  (h_q : q = 179 / 8192) (h_frac : Rat.mk_nat m n = q) :
  m + n = 8371 := by
  sorry

end probability_reach_2_3_in_7_steps_l655_655914


namespace line_perpendicular_to_AB_l655_655152

-- Define the points A, B, C, D, P, K, K1
variables {A B C D P K K1 : Type}

-- In a general Euclidean space
variable [euclidean_space ℝ (fin 2)]

-- Circle with diameter AB
def circle_diameter (A B : Type) := sorry -- A formal definition will be here

-- Tangents and intersections
axiom tangents (P C D : Type) : Prop := sorry -- A formal definition will be here
axiom intersection_of_lines (A C B D : Type) (K: Type) : Prop := sorry -- A formal definition will be here
axiom perpendicular (K K1 AB : Type) : Prop := sorry -- A formal definition will be here

-- Given the conditions
variables (h1 : circle_diameter A B) 
          (h2 : tangents P C D) 
          (h3 : intersection_of_lines A C B D K)
          (h4 : intersection_of_lines B C A D K1)
          (h5 : perpendicular K K1 AB)

-- The theorem to prove
theorem line_perpendicular_to_AB :
  let L := intersection_of_lines AC BD in
  ∃ L, line P L ⊥ line A B :=
begin
  sorry -- proof goes here
end

end line_perpendicular_to_AB_l655_655152


namespace total_legs_correct_l655_655436

variable (a b : ℕ)

def total_legs (a b : ℕ) : ℕ := 2 * a + 4 * b

theorem total_legs_correct (a b : ℕ) : total_legs a b = 2 * a + 4 * b :=
by sorry

end total_legs_correct_l655_655436


namespace positive_integer_solutions_count_l655_655143

theorem positive_integer_solutions_count : 
  (∃! (n : ℕ), n > 0 ∧ 25 - 5 * n > 15) :=
sorry

end positive_integer_solutions_count_l655_655143


namespace total_height_of_three_buildings_l655_655233

theorem total_height_of_three_buildings :
  let h1 := 600
  let h2 := 2 * h1
  let h3 := 3 * (h1 + h2)
  h1 + h2 + h3 = 7200 :=
by
  sorry

end total_height_of_three_buildings_l655_655233


namespace lcm_of_5_6_8_9_l655_655186

theorem lcm_of_5_6_8_9 : Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 9 = 360 := 
by 
  sorry

end lcm_of_5_6_8_9_l655_655186


namespace bug_paths_A_to_B_l655_655627

noncomputable def distinct_paths_from_A_to_B (lattice : Type) [finite_lattice : fintype lattice] (segment : lattice → lattice → Prop) (direction : lattice → lattice → Prop) : ℕ :=
sorry

theorem bug_paths_A_to_B (lattice : Type) [finite_lattice : fintype lattice] (segment : lattice → lattice → Prop) (direction : lattice → lattice → Prop) :
  distinct_paths_from_A_to_B lattice segment direction = 2880 :=
sorry

end bug_paths_A_to_B_l655_655627


namespace value_of_fg2_l655_655025

def g(x : ℕ) : ℕ := x^3
def f(x : ℕ) : ℕ := 2 * x + 3

theorem value_of_fg2 : f(g(2)) = 19 :=
by
  let g_2 := g 2
  have h1 : g_2 = 8 := by
    simp [g, pow_succ, mul_assoc]
  let f_8 := f g_2
  have h2 : f_8 = 19 := by
    simp [f, h1]
  exact h2

end value_of_fg2_l655_655025


namespace mitzi_money_left_in_dollars_l655_655116

def initial_amount_yen : ℝ := 10000
def ticket_cost : ℝ := 3000
def food_cost : ℝ := 2500
def tshirt_cost : ℝ := 1500
def souvenir_cost : ℝ := 2200
def exchange_rate : ℝ := 110
def total_spent : ℝ := ticket_cost + food_cost + tshirt_cost + souvenir_cost
def yen_left : ℝ := initial_amount_yen - total_spent
def dollars_left : ℝ := yen_left / exchange_rate

theorem mitzi_money_left_in_dollars : dollars_left ≈ 7.27 := 
by
  sorry

end mitzi_money_left_in_dollars_l655_655116


namespace total_movies_in_series_l655_655565

def book_count := 4
def total_books_read := 19
def movies_watched := 7
def movies_to_watch := 10

theorem total_movies_in_series : movies_watched + movies_to_watch = 17 := by
  sorry

end total_movies_in_series_l655_655565


namespace sequence_a_11_is_58_l655_655410

/-- Definition of the sequence -/
def a : ℕ → ℕ
| 0       := 3
| (n + 1) := a n + n

/-- Statement of the theorem we want to prove -/
theorem sequence_a_11_is_58 : a 10 = 58 := sorry

end sequence_a_11_is_58_l655_655410


namespace root_iff_coeff_sum_zero_l655_655905

theorem root_iff_coeff_sum_zero (a b c : ℝ) :
    (a * 1^2 + b * 1 + c = 0) ↔ (a + b + c = 0) := sorry

end root_iff_coeff_sum_zero_l655_655905


namespace continuous_at_2_iff_b_eq_neg5_l655_655352

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x > 2 then x - 1 else 3 * x + b

theorem continuous_at_2_iff_b_eq_neg5 (b : ℝ) : continuous_at (f x b) 2 ↔ b = -5 := by
  sorry

end continuous_at_2_iff_b_eq_neg5_l655_655352


namespace mass_Ω_eq_4pi_l655_655711

noncomputable def mass_of_body_Ω :=
  ∫∫∫ (x y z : ℝ) in (set.Icc 0 1) × (set.Icc 0 (Real.arctan 2)) × (set.Icc 0 (2 * Real.pi)),
      20 * z

theorem mass_Ω_eq_4pi : mass_of_body_Ω = 4 * Real.pi :=
  sorry

end mass_Ω_eq_4pi_l655_655711


namespace equal_costs_l655_655684

theorem equal_costs (x : ℕ) : 
    (22 + 0.13 * x = 8 + 0.18 * x) → x = 280 := by
  sorry

end equal_costs_l655_655684


namespace evaluate_expressions_l655_655199

theorem evaluate_expressions : (∀ (a b c d : ℤ), a = -(-3) → b = -(|-3|) → c = -(-(3^2)) → d = ((-3)^2) → b < 0) :=
by
  sorry

end evaluate_expressions_l655_655199


namespace categorize_numbers_l655_655330

def numbers := {x | x = 20 ∨ x = -4.8 ∨ x = 0 ∨ x = -13 ∨ x = (2/7) ∨ x = (86 / 100) ∨
                    x = -2008 ∨ x = 0.020020002 ∨ x = 0.010010001 ∨ x = 0.1212121212}

def negative_numbers := {-4.8, -13, -2008}
def fraction_numbers := {-4.8, (2 / 7), (86 / 100), 0.020020002, 0.1212121212}
def positive_integers := {20}
def irrational_numbers := {0.010010001}

theorem categorize_numbers : 
  (negative_numbers = {-4.8, -13, -2008}) ∧
  (fraction_numbers = {-4.8, (2 / 7), (86 / 100), 0.020020002, 0.1212121212}) ∧
  (positive_integers = {20}) ∧
  (irrational_numbers = {0.010010001}) :=
by
  -- the actual proof would go here
  sorry

end categorize_numbers_l655_655330


namespace domain_of_g_l655_655976

noncomputable def g (x : ℝ) : ℝ := Real.logb 3 (Real.logb 4 (Real.logb 5 (Real.logb 6 x)))

theorem domain_of_g : setOf (λ x, g x) = set.Ioi (6^625) :=
by
  sorry

end domain_of_g_l655_655976


namespace correct_proposition_l655_655267

-- Define the propositions and necessary logic
def prop1 (x : ℝ) : Prop := (x^2 = 1) → (x = 1)
def neg_prop1 (x : ℝ) : Prop := (x^2 ≠ 1) → (x ≠ 1)

def eq_suff_nec (x : ℝ) : Prop := (x = -1) → (x^2 - 5*x - 6 = 0)

def exists_prop (x : ℝ) : Prop := ∃ x, (x^2 + x + 1 = 0)
def neg_exists_prop (x : ℝ) : Prop := ∀ x, (x^2 + x + 1 ≠ 0)

def sin_contrapos (x y : ℝ) : Prop := (x = y) → (sin x = sin y)

-- Formal statement verifying the correctness (or falseness) of each proposition
theorem correct_proposition :
  (¬ ∀ x, prop1 x ↔ ¬ neg_prop1 x) ∧
  (¬ ∃ x, eq_suff_nec x) ∧
  (¬ ∀ x, exists_prop x ↔ ¬ neg_exists_prop x) ∧
  (∀ x y, sin_contrapos x y) :=
by
  sorry

end correct_proposition_l655_655267


namespace handshake_count_l655_655286

theorem handshake_count {teams : Fin 4 → Fin 2 → Prop}
    (h_teams_disjoint : ∀ (i j : Fin 4) (x y : Fin 2), i ≠ j → teams i x → teams j y → x ≠ y)
    (unique_partner : ∀ (i : Fin 4) (x1 x2 : Fin 2), teams i x1 → teams i x2 → x1 = x2) : 
    24 = (∑ i : Fin 8, (∑ j : Fin 8, if i ≠ j ∧ ¬(∃ k : Fin 4, teams k i ∧ teams k j) then 1 else 0)) / 2 :=
by sorry

end handshake_count_l655_655286


namespace product_of_y_coordinates_on_line_l655_655512

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem product_of_y_coordinates_on_line (y1 y2 : ℝ) (h1 : distance (4, -1) (-2, y1) = 8) (h2 : distance (4, -1) (-2, y2) = 8) :
  y1 * y2 = -27 :=
sorry

end product_of_y_coordinates_on_line_l655_655512


namespace margie_driving_distance_l655_655883

-- Define the constants given in the conditions
def mileage_per_gallon : ℝ := 40
def cost_per_gallon : ℝ := 5
def total_money : ℝ := 25

-- Define the expected result/answer
def expected_miles : ℝ := 200

-- The theorem that needs to be proved
theorem margie_driving_distance :
  (total_money / cost_per_gallon) * mileage_per_gallon = expected_miles :=
by
  -- proof goes here
  sorry

end margie_driving_distance_l655_655883


namespace sum_of_coefficients_in_binomial_expansion_l655_655539

theorem sum_of_coefficients_in_binomial_expansion :
  ∀ (n : ℕ), (n = 9) →
  let f := (x - (2 / sqrt x)) in 
  (∑ i in (range (n + 1)), (choose n i) * (coe_fn f i)) = -1 :=
sorry

end sum_of_coefficients_in_binomial_expansion_l655_655539


namespace inequality_proof_l655_655481

variable (k : ℕ) (a b c : ℝ)
variables (hk : 0 < k) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem inequality_proof (hk : k > 0) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * (1 - a^k) + b * (1 - (a + b)^k) + c * (1 - (a + b + c)^k) < k / (k + 1) :=
sorry

end inequality_proof_l655_655481


namespace simplify_sqrt_expression_l655_655526

theorem simplify_sqrt_expression : sqrt 12 + 3 * sqrt (1 / 3) = 3 * sqrt 3 := 
by sorry

end simplify_sqrt_expression_l655_655526


namespace neither_sufficient_nor_necessary_l655_655023

noncomputable def a_b_conditions (a b: ℝ) : Prop :=
∃ (a b: ℝ), ¬((a - b > 0) → (a^2 - b^2 > 0)) ∧ ¬((a^2 - b^2 > 0) → (a - b > 0))

theorem neither_sufficient_nor_necessary (a b: ℝ) : a_b_conditions a b :=
sorry

end neither_sufficient_nor_necessary_l655_655023


namespace graph_always_passes_fixed_point_l655_655548

theorem graph_always_passes_fixed_point (a : ℝ) (h_a_pos : a > 0) (h_a_not_one : a ≠ 1) :
    ∃ (x y : ℝ), x = 2 ∧ y = 3 ∧ y = a^(2 - x) + 2 :=
by
  use [2, 3]
  split
  { reflexivity }
  split
  { reflexivity }
  { rw [pow_zero]
    reflexivity }
sorry

end graph_always_passes_fixed_point_l655_655548


namespace shooting_competition_sequences_l655_655043

-- To prove the number of sequences is exactly 560

theorem shooting_competition_sequences :
  let leftmost_column := 3
      middle_column := 2
      rightmost_column := 3
      total_targets := 8
  in (Nat.choose total_targets leftmost_column) * (Nat.choose (total_targets - leftmost_column) middle_column) = 560 :=
by
  sorry

end shooting_competition_sequences_l655_655043


namespace cyclic_quad_diagonals_perpendicular_sumsq_l655_655140

noncomputable def cyclic_quadrilateral (A B C D : Type) : Prop := sorry -- cyclic quadrilateral predicate

theorem cyclic_quad_diagonals_perpendicular_sumsq (A B C D : Type) (r : ℝ) 
  (cyclic : cyclic_quadrilateral A B C D) 
  (perpendicular : ∠(A, C, O) = 90 ∧ ∠(B, D, O) = 90) :
  let a := dist A B in
  let c := dist C D in
  a^2 + c^2 = (2 * r)^2 :=
sorry

end cyclic_quad_diagonals_perpendicular_sumsq_l655_655140


namespace tennis_tournament_handshakes_l655_655280

theorem tennis_tournament_handshakes :
  ∃ (number_of_handshakes : ℕ),
    let total_women := 8 in
    let handshakes_per_woman := 6 in
    let total_handshakes_counted_twice := total_women * handshakes_per_woman in
    number_of_handshakes = total_handshakes_counted_twice / 2 :=
begin
  use 24,
  unfold total_women handshakes_per_woman total_handshakes_counted_twice,
  norm_num,
end

end tennis_tournament_handshakes_l655_655280


namespace expression_value_range_l655_655086

theorem expression_value_range (a b c d e : ℝ) (h₁ : 0 ≤ a) (h₂ : a ≤ 1) (h₃ : 0 ≤ b) (h₄ : b ≤ 1) (h₅ : 0 ≤ c) (h₆ : c ≤ 1) (h₇ : 0 ≤ d) (h₈ : d ≤ 1) (h₉ : 0 ≤ e) (h₁₀ : e ≤ 1) :
  4 * Real.sqrt (2 / 3) ≤ (Real.sqrt (a^2 + (1 - b)^2 + e^2) + Real.sqrt (b^2 + (1 - c)^2 + e^2) + Real.sqrt (c^2 + (1 - d)^2 + e^2) + Real.sqrt (d^2 + (1 - a)^2 + e^2)) ∧ 
  (Real.sqrt (a^2 + (1 - b)^2 + e^2) + Real.sqrt (b^2 + (1 - c)^2 + e^2) + Real.sqrt (c^2 + (1 - d)^2 + e^2) + Real.sqrt (d^2 + (1 - a)^2 + e^2)) ≤ 8 :=
sorry

end expression_value_range_l655_655086


namespace angle_between_a_and_c_is_120_l655_655770

variables (a b c : ℝ × ℝ)
def vector_a : ℝ × ℝ := (-2, 0)
def vector_b : ℝ × ℝ := (1, 2)
def vector_c : ℝ × ℝ := (1, real.sqrt 3)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2)
def cos_angle (u v : ℝ × ℝ) : ℝ := dot_product u v / (magnitude u * magnitude v)

theorem angle_between_a_and_c_is_120 :
  cos_angle vector_a vector_c = -1/2 :=
sorry

end angle_between_a_and_c_is_120_l655_655770


namespace ac_inequalities_l655_655800

theorem ac_inequalities (a c : ℝ) (h : a * c < 0) : 
  ((a / c < 0) ∧ ¬(a * c^2 < 0) ∧ ¬(a^2 * c < 0) ∧ (c^3 * a < 0) ∧ (c * a^3 < 0)) → 
  count_true [a / c < 0, ¬(a * c^2 < 0), ¬(a^2 * c < 0), c^3 * a < 0, c * a^3 < 0] = 3 :=
begin
  sorry
end

-- Helper function to count true conditions
noncomputable def count_true (conds : list Prop) : ℕ :=
conds.count (λ p, p)

end ac_inequalities_l655_655800


namespace select_parents_l655_655247

theorem select_parents : 
  let n := 12 
  let couples := 6 
  let total_pairs := (n / 2)
  let select_couple : ℕ := 6
  let select_2_parents := Nat.choose 10 2
  in couples = total_pairs ∧ couples = select_couple ∧ select_2_parents = 45 → (select_couple * select_2_parents) = 270 := 
by
  intros n couples total_pairs select_couple select_2_parents h
  sorry

end select_parents_l655_655247


namespace common_ratio_of_geometric_sequence_l655_655925

theorem common_ratio_of_geometric_sequence (a : ℝ) :
  let x1 := -a + Real.log 2017 / Real.log 3,
      x2 := 2a + Real.log 2017 / Real.log 2,
      x3 := -4a + Real.log 2017 / Real.log 21 in
  ((x3 + 2 * x2) / (x2 + 2 * x1)) = (8 / 15) := by
  sorry

end common_ratio_of_geometric_sequence_l655_655925


namespace pete_ate_percentage_l655_655132

-- Definitions of the conditions
def total_slices : ℕ := 2 * 12
def stephen_ate_slices : ℕ := (25 * total_slices) / 100
def remaining_slices_after_stephen : ℕ := total_slices - stephen_ate_slices
def slices_left_after_pete : ℕ := 9

-- The statement to be proved
theorem pete_ate_percentage (h1 : total_slices = 24)
                            (h2 : stephen_ate_slices = 6)
                            (h3 : remaining_slices_after_stephen = 18)
                            (h4 : slices_left_after_pete = 9) :
  ((remaining_slices_after_stephen - slices_left_after_pete) * 100 / remaining_slices_after_stephen) = 50 :=
sorry

end pete_ate_percentage_l655_655132


namespace part1_part2_l655_655691

def op (a b : ℤ) := 2 * a - 3 * b

theorem part1 : op (-2) 3 = -13 := 
by
  -- Proof omitted
  sorry

theorem part2 (x : ℤ) : 
  let A := op (3 * x - 2) (x + 1)
  let B := op (-3 / 2 * x + 1) (-1 - 2 * x)
  B > A :=
by
  -- Proof omitted
  sorry

end part1_part2_l655_655691


namespace tennis_handshakes_l655_655273

-- Define the participants and the condition
def number_of_participants := 8
def handshakes_no_partner (n : ℕ) := n * (n - 2) / 2

-- Prove that the number of handshakes is 24
theorem tennis_handshakes : handshakes_no_partner number_of_participants = 24 := by
  -- Since we are skipping the proof for now
  sorry

end tennis_handshakes_l655_655273


namespace greatest_distance_is_not_opposite_corner_l655_655737

-- Assumptions about the dimensions of the rectangle and other conditions
def box_dimensions : ℝ × ℝ × ℝ := (2, 2, 12)
def ant_initial_position : (ℝ × ℝ × ℝ) := (0, 0, 0)
def ant_opposite_position : (ℝ × ℝ × ℝ) := (2, 2, 12)

-- We will state the proposition that the longest distance on the surface is not necessarily to the opposite corner
theorem greatest_distance_is_not_opposite_corner :
  ¬ is_greatest_distance_on_surface box_dimensions ant_initial_position ant_opposite_position := sorry

end greatest_distance_is_not_opposite_corner_l655_655737


namespace octagon_area_inscribed_in_square_l655_655258

noncomputable def side_length_of_square (perimeter : ℝ) : ℝ :=
  perimeter / 4

noncomputable def trisected_segment_length (side_length : ℝ) : ℝ :=
  side_length / 3

noncomputable def area_of_removed_triangle (segment_length : ℝ) : ℝ :=
  (segment_length * segment_length) / 2

noncomputable def total_area_removed_by_triangles (area_of_triangle : ℝ) : ℝ :=
  4 * area_of_triangle

noncomputable def area_of_square (side_length : ℝ) : ℝ :=
  side_length * side_length

noncomputable def area_of_octagon (area_of_square : ℝ) (total_area_removed : ℝ) : ℝ :=
  area_of_square - total_area_removed

theorem octagon_area_inscribed_in_square (perimeter : ℝ) (H : perimeter = 144) :
  area_of_octagon (area_of_square (side_length_of_square perimeter))
    (total_area_removed_by_triangles (area_of_removed_triangle (trisected_segment_length (side_length_of_square perimeter))))
  = 1008 :=
by
  rw [H]
  -- Intermediate steps would contain calculations for side_length_of_square, trisected_segment_length, area_of_removed_triangle, total_area_removed_by_triangles, and area_of_square based on the given perimeter.
  sorry

end octagon_area_inscribed_in_square_l655_655258


namespace maximum_temperature_range_l655_655921

theorem maximum_temperature_range (temps : Fin 5 → ℕ) (h_avg : (1 / 5 : ℚ) * (Finset.univ.sum (λ i, temps i)) = 40)
  (h_min : ∃ i, temps i = 30) : (Finset.univ.max' (by decide) temps - Finset.univ.min' (by decide) temps) = 50 :=
sorry

end maximum_temperature_range_l655_655921


namespace sheep_count_l655_655553

/-- The ratio between the number of sheep and the number of horses at the Stewart farm is 2 to 7.
    Each horse is fed 230 ounces of horse food per day, and the farm needs a total of 12,880 ounces
    of horse food per day. -/
theorem sheep_count (S H : ℕ) (h_ratio : S = (2 / 7) * H)
    (h_food : H * 230 = 12880) : S = 16 :=
sorry

end sheep_count_l655_655553


namespace subset_no_family_members_l655_655615

variable (X : Type) [Fintype X] (F : Finset (Finset X))

theorem subset_no_family_members (n : ℕ) 
  (hX_card : Fintype.card X = n)
  (hF_size : ∀ S ∈ F, S.card = 3)
  (hF_intersection : ∀ S₁ S₂ ∈ F, S₁ ≠ S₂ → (S₁ ∩ S₂).card ≤ 1) :
  ∃ Y : Finset X, Y.card ≥ ⌈Real.sqrt (2 * n)⌉₊ ∧ ∀ S ∈ F, ¬ S ⊆ Y := sorry

end subset_no_family_members_l655_655615


namespace find_X_l655_655807

def tax_problem (X I T : ℝ) (income : ℝ) (total_tax : ℝ) :=
  (income = 56000) ∧ (total_tax = 8000) ∧ (T = 0.12 * X + 0.20 * (I - X))

theorem find_X :
  ∃ X : ℝ, ∀ I T : ℝ, tax_problem X I T 56000 8000 → X = 40000 := 
  by
    sorry

end find_X_l655_655807


namespace length_of_CD_l655_655472

-- Definitions corresponding to the conditions
def isosceles_triangle (ABE : Type) (area_ABE : ℝ) (height_A : ℝ) : Prop :=
  area_ABE = 180 ∧ height_A = 24

def isosceles_trapezoid_cut (ABE : Type) (CD : ABE) (area_trap : ℝ) : Prop :=
  area_trap = 135

def similar_triangles (k : ℝ) : Prop :=
  k ^ 2 = 1 / 4

-- Main theorem statement
theorem length_of_CD 
  (ABE : Type) 
  (area_ABE : ℝ) 
  (height_A : ℝ) 
  (area_trap : ℝ) 
  (CD : ABE) 
  (k : ℝ) 
  (asg_id1 : isosceles_triangle ABE area_ABE height_A) 
  (asg_id2 : isosceles_trapezoid_cut ABE CD area_trap) 
  (asg_id3 : similar_triangles k) : 
    CD = 7.5 :=
by
  sorry

end length_of_CD_l655_655472


namespace exists_Q_fixed_angle_segment_l655_655085

noncomputable def parallel_lines_point_angle (l m : Line) (P : Point) (E F : Point) (α : Real) : Prop :=
  Parallel l m ∧
  (∃ A B : Line, A ≠ l ∧ B ≠ m ∧ Parallel A l ∧ Parallel B m ∧
    between P A B ∧ lies_on E l ∧ lies_on F m ∧ Angle E P F = α ∧
    0 < α ∧ α < (π / 2))

theorem exists_Q_fixed_angle_segment (l m : Line) (P : Point) (α : Real) (hα : 0 < α ∧ α < (π / 2)) (E F : Point)
  (hl : lies_on E l) (hm : lies_on F m) (hAngle : Angle E P F = α) :
  ∃ Q : Point, Angle E Q F = π - α := sorry

end exists_Q_fixed_angle_segment_l655_655085


namespace symmetry_axis_of_sine_function_l655_655399

theorem symmetry_axis_of_sine_function (φ : ℝ) (h : cos (2 * Real.pi / 3 - φ) = cos φ) :
  ∃ k : ℤ, ∃ x : ℝ, x = k * Real.pi + 5 * Real.pi / 6 :=
by
  sorry

end symmetry_axis_of_sine_function_l655_655399


namespace triangle_inequality_l655_655495

theorem triangle_inequality (a b c p q r : ℝ) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_sum_zero : p + q + r = 0) : 
  a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 := 
sorry

end triangle_inequality_l655_655495


namespace part_I_part_II_l655_655458

noncomputable def curve_M (θ : ℝ) : ℝ := 4 * cos θ

structure Line :=
  (m t α : ℝ)
  (eqns : ∀ t : ℝ, (x y : ℝ) → x = m + t * cos α ∧ y = t * sin α)

variables
  (θ φ : ℝ)
  (α : ℝ) (hα : 0 ≤ α ∧ α < π)
  (m : ℝ)
  (t : ℝ)
  (O : ℝ) -- Origin
  (A B C : ℝ → ℝ)

def OA : ℝ := curve_M φ
def OB : ℝ := curve_M (φ + π / 4)
def OC : ℝ := curve_M (φ - π / 4)

-- Part (I)
theorem part_I : OB + OC = sqrt 2 * OA := sorry

-- Part (II)
theorem part_II (φ : ℝ) (hφ : φ  = π / 12) (l : Line) (B C : ℝ × ℝ) :
  (curve_M (φ + π / 3) = 2) →
  (curve_M (-(π / 6)) = 2 * sqrt 3) →
  B = (1, sqrt 3) →
  C = (3, -sqrt 3) →
  ∃ (m α : ℝ), (m = 2 ∧ α = 2 * π / 3) := sorry

end part_I_part_II_l655_655458


namespace arithmetic_sums_l655_655945

theorem arithmetic_sums (d : ℤ) (p q : ℤ) (S : ℤ → ℤ)
  (hS : ∀ n, S n = p * n^2 + q * n)
  (h_eq : S 20 = S 40) : S 60 = 0 :=
by
  sorry

end arithmetic_sums_l655_655945


namespace range_of_a_l655_655497

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x > 2 then 2^x + a else x + a^2

theorem range_of_a (a : ℝ) : (∀ y : ℝ, ∃ x : ℝ, f x a = y) ↔ (a ≤ -1 ∨ a ≥ 2) :=
by
  sorry

end range_of_a_l655_655497


namespace checker_move_10_cells_checker_move_11_cells_l655_655231

noncomputable def F : ℕ → Nat 
| 0 => 1
| 1 => 1
| n + 2 => F (n + 1) + F n

theorem checker_move_10_cells : F 10 = 89 := by
  sorry

theorem checker_move_11_cells : F 11 = 144 := by
  sorry

end checker_move_10_cells_checker_move_11_cells_l655_655231


namespace angle_acb_is_35_l655_655462

-- Define the angles PQR and the properties of the triangle.
variables (ABC ABD BAC ACB : ℝ)

-- Known conditions
axiom angle_abc_abd : ABC + ABD = 180
axiom angle_abd : ABD = 120
axiom angle_bac : BAC = 85

-- The hypothesis that angles in triangle ABC sum up to 180 degrees
axiom triangle_abc_sum : ABC + BAC + ACB = 180

-- Statement to prove
def measure_angle_acb : Prop :=
  ACB = 35

-- Proof statement (without the proof itself)
theorem angle_acb_is_35 : measure_angle_acb ABC ABD BAC ACB :=
  by
    sorry

end angle_acb_is_35_l655_655462


namespace k_configurations_of_set_l655_655220

theorem k_configurations_of_set (n k : ℕ) (h : k ≤ n) : 
  ∃ count : ℕ, count = 2^(Nat.binomial n k) :=
by
  sorry

end k_configurations_of_set_l655_655220


namespace four_points_plane_l655_655456

theorem four_points_plane (P1 P2 P3 P4 : Point)
  (h1 : ¬Collinear P1 P2 P3)
  (h2 : ¬Collinear P1 P2 P4)
  (h3 : ¬Collinear P1 P3 P4)
  (h4 : ¬Collinear P2 P3 P4) :
  (∀ P5, ∃! P1 P2 P3 (Plane P1 P2 P3 P1 P2 P3) :
  ∃! P5, ∃! P4 ∨ ∃! P3) ∨
  ∃! (Plane P1 P2 P3 P1 P4):
  sorry

end four_points_plane_l655_655456


namespace find_cherry_pasty_no_more_than_three_attempts_l655_655660

def pasties := List (String) -- Define the type of pasties, where "cabbage", "meat" and "cherries" are elements
def arrangement : pasties := ["cabbage", "meat", "cherries"].cycle 5.take 15 -- Circular arrangement of 15 pasties
def rotate {α : Type*} (n : ℕ) (l : List α) : List α := (l.drop n) ++ (l.take n) -- Definition to rotate the list

theorem find_cherry_pasty_no_more_than_three_attempts :
  ∃ n m : ℕ, ∀ (rotation : ℕ), n ≤ 2 ∧ m ≤ 2 ∧
  let rotated := rotate rotation arrangement in
  (rotated !! n = some "cherries") ∨
  (rotated !! m = some "cherries") ∨
  (rotated !! ((n + 1) % 15) = some "cherries") ∨
  (rotated !! ((m + 1) % 15) = some "cherries")
  := by
  sorry

end find_cherry_pasty_no_more_than_three_attempts_l655_655660


namespace exists_non_triangle_triplet_l655_655848

theorem exists_non_triangle_triplet
  (a b c d e : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ d ∧ b ≠ e ∧ c ≠ e)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e)
  (h_eq : a^2 + b^2 + c^2 + d^2 + e^2 = ab + ac + ad + ae + bc + bd + be + cd + ce + de) :
  ∃ x y z, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ ¬ (x + y > z ∧ y + z > x ∧ z + x > y) :=
begin
  sorry
end

end exists_non_triangle_triplet_l655_655848


namespace min_value_inequality_l655_655735

theorem min_value_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 3^x + 9^y ≥ 2 * Real.sqrt 3 := 
by
  sorry

end min_value_inequality_l655_655735


namespace second_number_in_sequence_is_23_l655_655165

theorem second_number_in_sequence_is_23 :
  ∃ (x2 : ℕ),
    let diff := [36, 48, 60, 72, 84, 96, 108, 120] in
    let seq := [11, x2, 47, 83, 131, 191, 263, 347, 443, 551, 671] in
    (∀ i : ℕ, 1 ≤ i → i < seq.length - 1 → seq[i+1] - seq[i] = diff[i-1])
    → x2 = 23 := by
  sorry

end second_number_in_sequence_is_23_l655_655165


namespace domain_of_g_l655_655984

noncomputable def g (x : ℝ) : ℝ := log 3 (log 4 (log 5 (log 6 x)))

theorem domain_of_g :
  { x : ℝ | x > 1296 } = { x : ℝ | g x = real.log 3 (real.log 4 (real.log 5 (real.log 6 x))) ∧ ∀ x > 1296, x ∈ ℝ } :=
sorry

end domain_of_g_l655_655984


namespace find_k_l655_655430

theorem find_k (k r d : ℤ) 
  (h1 : 36 + k = r^2) 
  (h2 : 300 + k = (r + d)^2) 
  (h3 : 596 + k = (r + 2d)^2) : 
  k = 925 :=
by
  sorry

end find_k_l655_655430


namespace coach_mike_change_l655_655302

theorem coach_mike_change (cost amount_given change : ℕ) 
    (h_cost : cost = 58) (h_amount_given : amount_given = 75) : 
    change = amount_given - cost → change = 17 := by
    sorry

end coach_mike_change_l655_655302


namespace calculate_S_l655_655736

noncomputable def z : ℂ := 1/2 + (real.sqrt 3 / 2)*complex.I
def S : ℂ := z + 2*z^2 + 3*z^3 + 4*z^4 + 5*z^5 + 6*z^6

-- The problem is to prove that S = 3 - 3*sqrt(3)*I given that z = 1/2 + sqrt(3)/2*I and z^6 = 1
theorem calculate_S :
    S = 3 - 3*(real.sqrt 3)*complex.I :=
begin
  sorry
end

end calculate_S_l655_655736


namespace problem_statement_l655_655406

section
variables {m : ℝ} {x y : ℝ}

/-- Define the line l1 -/
def l1 (m x y : ℝ) : Prop := (m + 2) * x + m * y - 6 = 0

/-- Define the line l2 -/
def l2 (m x y : ℝ) : Prop := m * x + y - 3 = 0

/-- Condition (1) - l1 is perpendicular to l2 -/
def perp (m : ℝ) : Prop :=
  if m = 0 then True else 
  let slope_l1 := -(m + 2) / m in
  let slope_l2 := -m in
  slope_l1 * slope_l2 = -1

/-- Condition (2) - point P(1, 2m) lies on l2, and intercepts are negatives of each other -/
def passes_through_p_and_intercepts (m : ℝ) (x : ℝ) : Prop :=
  l2 m 1 (2 * m) ∧ ∃ k, (x = 0 → 2 - k = 0) ∧ (2 = k * (x - 1)) ∧ 
  (1 + (2 - k) / k = -(2 - k))

/-- Main theorem combining both conditions and conclusions -/
theorem problem_statement :
  (perp m → m = 0 ∨ m = -3) ∧ 
  (passes_through_p_and_intercepts 1 2 → ∃ (l : ℝ → ℝ → Prop), 
    (l = (2 * x - y = 0) ∨ l = (x - y + 1 = 0))) :=
by sorry
end

end problem_statement_l655_655406


namespace stars_crossed_out_l655_655216

theorem stars_crossed_out {n : ℕ} (h : n > 0) (h3n_stars : ∃ stars : fin(2 * n) → fin(2 * n) → Prop, (∑ i, ∑ j, if stars i j then 1 else 0) = 3 * n) :
  ∃ rows cols, (rows ⊆ finset.range (2 * n) ∧ cols ⊆ finset.range (2 * n) ∧ rows.card = n ∧ cols.card = n ∧ (∀ i j, stars i j → i ∈ rows ∨ j ∈ cols)) :=
sorry

end stars_crossed_out_l655_655216


namespace geometric_series_sum_l655_655303

variable (a r : ℤ) (n : ℕ) 

theorem geometric_series_sum :
  a = -1 ∧ r = 2 ∧ n = 10 →
  (a * (r^n - 1) / (r - 1)) = -1023 := 
by
  intro h
  rcases h with ⟨ha, hr, hn⟩
  sorry

end geometric_series_sum_l655_655303


namespace area_triangle_PQR_l655_655173

-- Declaring the centers of the three circles
variables {P Q R : Point}
-- Radii of the circles
variables (rP : ℝ) (rQ : ℝ) (rR : ℝ)
-- Additional condition: circles lie on the same side of a line and tangent to it
variables (l : Line)
variables {P' Q' R' : Point} -- Tangent points on line l
-- Condition the distances
variables (hPQ : dist P Q = rP + rQ)
variables (hQR : dist Q R = rQ + rR)
variables (hQ_between : point_between l Q' P' R')

-- Declaring the coordinates of the centers as stated
axiom hP_coord : coordinates P = (-5, 2)
axiom hQ_coord : coordinates Q = (0, 3)
axiom hR_coord : coordinates R = (7, 4)

-- The proof that the area of triangle PQR is 1
theorem area_triangle_PQR : 
  area_triangle P Q R = 1 :=
sorry

end area_triangle_PQR_l655_655173


namespace rolls_combination_problem_l655_655226

theorem rolls_combination_problem
  (n k : ℕ)
  (h_n : n = 4)
  (h_k : k = 4)
  (at_least_one_each : ℕ) :
  (at_least_one_each = 1) →
  (n + k - 1).choose (k - 1) = 35 :=
begin
  intros,
  rw [h_n, h_k],
  rw add_comm,
  norm_num, -- simplifies (4 + 4 - 1).choose(4 - 1) = 7.choose 3
  norm_num,
end

end rolls_combination_problem_l655_655226


namespace positive_integers_satisfy_l655_655141

theorem positive_integers_satisfy (n : ℕ) (h1 : 25 - 5 * n > 15) : n = 1 :=
by sorry

end positive_integers_satisfy_l655_655141


namespace nicky_run_time_before_catch_up_l655_655603

-- Define the conditions
def nicky_speed : ℕ := 3  -- Nicky's speed in meters per second
def cristina_speed : ℕ := 5  -- Cristina's speed in meters per second
def head_start_time : ℕ := 12  -- Nicky's head start time in seconds

-- Define the problem as a statement to be proven
theorem nicky_run_time_before_catch_up : ∃ t : ℕ, 12 + t = 30 :=
by {
  -- Calculate the distance Nicky covers during his 12-second head start
  let distance_nicky_head_start := nicky_speed * head_start_time,
  -- Define the time it takes Cristina to catch up after Nicky's head start
  let t := (distance_nicky_head_start) / (cristina_speed - nicky_speed),
  use t + head_start_time,
  sorry
}

end nicky_run_time_before_catch_up_l655_655603


namespace functions_eq_of_conditions_l655_655845

open Set

variables {I J : Set ℝ} {f g : ℝ → ℝ} {φ : ℝ → ℝ}

theorem functions_eq_of_conditions (hJ : J ⊆ ℝ)
  (hI : I ⊆ ℝ)
  (hφ_continuous : Continuous φ)
  (hφ_nonzero : ∀ y ∈ J, φ y ≠ 0)
  (hf_diff : ∀ x ∈ I, DifferentiableAt ℝ f x)
  (hg_diff : ∀ x ∈ I, DifferentiableAt ℝ g x)
  (hf_prime : ∀ x ∈ I, deriv f x = φ (f x))
  (hg_prime : ∀ x ∈ I, deriv g x = φ (g x))
  (h_image_contains_zero : ∃ x ∈ I, f x = g x) :
  f = g :=
sorry

end functions_eq_of_conditions_l655_655845


namespace average_of_combined_results_l655_655606

theorem average_of_combined_results {avg1 avg2 n1 n2 : ℝ} (h1 : avg1 = 28) (h2 : avg2 = 55) (h3 : n1 = 55) (h4 : n2 = 28) :
  ((n1 * avg1) + (n2 * avg2)) / (n1 + n2) = 37.11 :=
by sorry

end average_of_combined_results_l655_655606


namespace tennis_handshakes_l655_655275

-- Define the participants and the condition
def number_of_participants := 8
def handshakes_no_partner (n : ℕ) := n * (n - 2) / 2

-- Prove that the number of handshakes is 24
theorem tennis_handshakes : handshakes_no_partner number_of_participants = 24 := by
  -- Since we are skipping the proof for now
  sorry

end tennis_handshakes_l655_655275


namespace max_good_numbers_are_1346_l655_655552

def max_good_numbers : ℕ :=
  2021

def is_good_number (i : ℕ) : Prop :=
  ∃ (a b c : ℕ), ((a = i ∨ b = i ∨ c = i) ∧ (b = i + 1)) ∧ (b ≠ (min a c) + x)

def count_good_numbers (n : ℕ) : ℕ :=
  sorry

theorem max_good_numbers_are_1346 :
  count_good_numbers max_good_numbers = 1346 :=
sorry

end max_good_numbers_are_1346_l655_655552


namespace find_values_l655_655357

noncomputable def a : ℝ := Real.sqrt 5 + Real.sqrt 3
noncomputable def b : ℝ := Real.sqrt 5 - Real.sqrt 3

theorem find_values :
  (a + b = 2 * Real.sqrt 5) ∧
  (a * b = 2) ∧
  (a^2 + a * b + b^2 = 18) := by
  sorry

end find_values_l655_655357


namespace juggling_sequences_l655_655318

theorem juggling_sequences (n : ℕ) : 
  let time_steps := finset.range n in
  (2 ^ n - 1 = (time_steps.powerset.filter (λ s, s.nonempty)).card) := by
  sorry

end juggling_sequences_l655_655318


namespace value_of_4_ampersand_neg3_l655_655689

-- Define the operation '&'
def ampersand (x y : Int) : Int :=
  x * (y + 2) + x * y

-- State the theorem
theorem value_of_4_ampersand_neg3 : ampersand 4 (-3) = -16 :=
by
  sorry

end value_of_4_ampersand_neg3_l655_655689


namespace tennis_tournament_handshakes_l655_655277

theorem tennis_tournament_handshakes :
  ∃ (number_of_handshakes : ℕ),
    let total_women := 8 in
    let handshakes_per_woman := 6 in
    let total_handshakes_counted_twice := total_women * handshakes_per_woman in
    number_of_handshakes = total_handshakes_counted_twice / 2 :=
begin
  use 24,
  unfold total_women handshakes_per_woman total_handshakes_counted_twice,
  norm_num,
end

end tennis_tournament_handshakes_l655_655277


namespace regression_equation_represents_real_relationship_maximized_l655_655940

-- Definitions from the conditions
def regression_equation (y x : ℝ) := ∃ (a b : ℝ), y = a * x + b

def represents_real_relationship_maximized (y x : ℝ) := regression_equation y x

-- The proof problem statement
theorem regression_equation_represents_real_relationship_maximized 
: ∀ (y x : ℝ), regression_equation y x → represents_real_relationship_maximized y x :=
by
  sorry

end regression_equation_represents_real_relationship_maximized_l655_655940


namespace cone_from_sector_l655_655998

theorem cone_from_sector
  (r : ℝ) (slant_height : ℝ)
  (radius_circle : ℝ := 10)
  (angle_sector : ℝ := 252) :
  (r = 7 ∧ slant_height = 10) :=
by
  sorry

end cone_from_sector_l655_655998


namespace three_digit_numbers_not_multiple_of_3_or_11_l655_655785

theorem three_digit_numbers_not_multiple_of_3_or_11 : 
  let total_three_digit_numbers := 999 - 100 + 1 in
  let multiples_of_3 := 333 - 34 + 1 in
  let multiples_of_11 := 90 - 10 + 1 in
  let multiples_of_33 := 30 - 4 + 1 in
  let multiples_of_3_or_11 := multiples_of_3 + multiples_of_11 - multiples_of_33 in
  total_three_digit_numbers - multiples_of_3_or_11 = 546 :=
by
  let total_three_digit_numbers := 999 - 100 + 1
  let multiples_of_3 := 333 - 34 + 1
  let multiples_of_11 := 90 - 10 + 1
  let multiples_of_33 := 30 - 4 + 1
  let multiples_of_3_or_11 := multiples_of_3 + multiples_of_11 - multiples_of_33
  show total_three_digit_numbers - multiples_of_3_or_11 = 546 from sorry

end three_digit_numbers_not_multiple_of_3_or_11_l655_655785


namespace impossible_to_have_all_polynomials_with_37_distinct_positive_roots_l655_655889

open polynomial

noncomputable def is_deg (p : polynomial ℝ) (n : ℕ) : Prop :=
p.degree = n

noncomputable def leading_coeff_is_one (p : polynomial ℝ) : Prop :=
leading_coeff p = 1

noncomputable def nonnegative_coeffs (p : polynomial ℝ) : Prop :=
∀ (i : ℕ), 0 ≤ coeff p i

-- The main statement to prove.
theorem impossible_to_have_all_polynomials_with_37_distinct_positive_roots
  (P : list (polynomial ℝ))
  (hdeg : ∀ p ∈ P, is_deg p 37)
  (hleading : ∀ p ∈ P, leading_coeff_is_one p)
  (hnonneg : ∀ p ∈ P, nonnegative_coeffs p)
  (moves : ∀ f g ∈ P, ∃ f1 g1, is_deg f1 37 ∧ is_deg g1 37 ∧
    leading_coeff_is_one f1 ∧ leading_coeff_is_one g1 ∧
    (nonnegative_coeffs f1 ∧ nonnegative_coeffs g1 ∧ ((f1 + g1 = f + g) ∨ (f1 * g1 = f * g)))) :
  ¬ ∀ p ∈ P, ∃ r : set ℝ, (∀ x ∈ r, p.eval x = 0) ∧ r.card = 37 := by
    sorry

end impossible_to_have_all_polynomials_with_37_distinct_positive_roots_l655_655889


namespace dividend_rate_l655_655640

theorem dividend_rate (investment face_value premium dividend_received : ℝ) 
  (h1 : investment = 14400)
  (h2 : face_value = 100)
  (h3 : premium = 0.25)
  (h4 : dividend_received = 576) :
  let cost_per_share := face_value * (1 + premium),
      number_of_shares := (investment / cost_per_share).toInt,
      dividend_per_share := dividend_received / number_of_shares in
  (dividend_per_share / face_value) * 100 = 5 :=
by
  sorry

end dividend_rate_l655_655640


namespace gcd_bc_2023_l655_655917

theorem gcd_bc_2023 (a b c : ℕ) (hab : Nat.gcd a b) (hac : Nat.gcd a c) (hbc : Nat.gcd b c) :
  hab + hac + hbc = b + c + 2023 → hbc = 2023 := by
  sorry

end gcd_bc_2023_l655_655917


namespace super_champion_games_l655_655504

theorem super_champion_games (n : ℕ) (h1 : n = 24) : 
  let games_for_initial_tournament := 12 + 6 + 3 + 2 in
  let games_to_determine_champion := games_for_initial_tournament in
  let total_games := games_to_determine_champion + 1 in
  total_games = 24 :=
by {
  sorry
}

end super_champion_games_l655_655504


namespace hyperbola_range_of_k_l655_655030

theorem hyperbola_range_of_k (k : ℝ) :
  (∃ x y : ℝ, (x^2)/(k + 4) + (y^2)/(k - 1) = 1) → -4 < k ∧ k < 1 :=
by 
  sorry

end hyperbola_range_of_k_l655_655030


namespace log3_div_eq_l655_655900

-- The conditions are defined as follows:
def x (x : ℝ) := x ≠ 1
def y (y : ℝ) := y ≠ 1
def log_eq (x y : ℝ) := log x 3 = log 81 y
def xy_eq (x y : ℝ) := x * y^2 = 729

-- We want to prove that under these conditions:
-- (\log_3(x / y))^2 = (206 - 90 * sqrt 5) / 4

theorem log3_div_eq (x y : ℝ) (hx : x ≠ 1) (hy : y ≠ 1) (hxy2 : x * y^2 = 729) (hlog : log 3 x = log y 81) :
  (log 3 (x / y))^2 = (206 - 90 * real.sqrt 5) / 4 :=
sorry

end log3_div_eq_l655_655900


namespace new_number_is_greater_l655_655933

-- Define the original number as a rational number
def orig_num : ℚ := 1 / 42

-- Define the repeating sequence for the decimal expansion of 1/42
def repeating_seq : List ℕ := [0, 2, 3, 8, 0, 9]

-- Function to get the nth digit after the decimal point in the repeating sequence
def nth_digit (n : ℕ) : ℕ :=
  repeating_seq.get! (n % repeating_seq.length)

-- Function to compute the new number after removing the nth digit and shifting
def new_num (orig_num : ℚ) (n : ℕ) : ℚ :=
  let digits := List.range (n + 1) ++ List.range (n + 1).tail
  let new_digits := digits.map nth_digit
  let decimal_part := new_digits.join.toList.foldr (λ (d acc : ℕ), d + acc / 10) 0
  orig_num.toInt + decimal_part -- Convert the list of digits back to a number

-- The main statement to be proven
theorem new_number_is_greater : new_num orig_num 1997 > orig_num := by
  sorry

end new_number_is_greater_l655_655933


namespace hortense_flower_production_l655_655011

def daisy_seeds_planted : ℕ := 25
def sunflower_seeds_planted : ℕ := 25
def daisy_germination_rate : ℝ := 0.60
def sunflower_germination_rate : ℝ := 0.80
def flower_production_rate : ℝ := 0.80

noncomputable def plants_that_produce_flowers : ℕ :=
  let daisy_plants := daisy_germination_rate * daisy_seeds_planted
  let sunflower_plants := sunflower_germination_rate * sunflower_seeds_planted
  let daisy_flowers := flower_production_rate * daisy_plants
  let sunflower_flowers := flower_production_rate * sunflower_plants
  daisy_flowers.to_nat + sunflower_flowers.to_nat

theorem hortense_flower_production : plants_that_produce_flowers = 28 := 
by trivial

end hortense_flower_production_l655_655011


namespace max_tied_teams_round_robin_l655_655442

theorem max_tied_teams_round_robin (n : ℕ) (h: n = 8) :
  ∃ k, (k <= n) ∧ (∀ m, m > k → k * m < n * (n - 1) / 2) :=
by
  sorry

end max_tied_teams_round_robin_l655_655442


namespace parallelogram_angles_sum_l655_655047

variable {A B C D : Type}

structure IsParallelogram (A B C D : Type) : Prop where
  side_parallel1 : AD ∥ BC
  side_parallel2 : AB ∥ CD

theorem parallelogram_angles_sum
  (A B C D : Type)
  (h1 : IsParallelogram A B C D)
  (h2 : AD ∥ BC)
  : $\angle B + \angle C = 180^{\circ}$ := sorry

end parallelogram_angles_sum_l655_655047


namespace range_of_s_triangle_l655_655431

theorem range_of_s_triangle (inequalities_form_triangle : Prop) : 
  (0 < s ∧ s ≤ 2) ∨ (s ≥ 4) ↔ inequalities_form_triangle := 
sorry

end range_of_s_triangle_l655_655431


namespace probability_one_at_least_one_even_product_l655_655580

structure Spinner where
  values : Set ℕ

def SpinnerA : Spinner := { values := {3, 5, 7} }
def SpinnerB : Spinner := { values := {2, 6, 8, 10} }

def is_even (n : ℕ) : Prop := n % 2 = 0

def product_even (x y : ℕ) : Prop := is_even (x * y)

def at_least_one_even_product (a1 a2 b1 b2 : ℕ) : Prop :=
  product_even a1 b1 ∨ product_even a1 b2 ∨ product_even a2 b1 ∨ product_even a2 b2

theorem probability_one_at_least_one_even_product :
  ∀ (a1 a2 ∈ SpinnerA.values) (b1 b2 ∈ SpinnerB.values),
  at_least_one_even_product a1 a2 b1 b2 :=
by
  intros a1 ha1 a2 ha2 b1 hb1 b2 hb2
  unfold at_least_one_even_product product_even is_even
  have ha1_odd : ¬is_even a1 := by cases ha1; norm_num
  have ha2_odd : ¬is_even a2 := by cases ha2; norm_num
  have hb1_even : is_even b1 := by cases hb1; norm_num
  have hb2_even : is_even b2 := by cases hb2; norm_num
  left; exact hb1_even sorry
  -- The proof that the left disjunction will hold because a1 * b1 is even, due to b1 being always even.
sorry

end probability_one_at_least_one_even_product_l655_655580


namespace area_N1N2N3_eq_one_ninth_area_ABC_l655_655463

/-- In a triangle ABC, points N1, N2, N3 divide segments such that:
     - CD, AE, and BF are one-fourth of their respective sides.
     - AN2 : N2N1 : N1D = 4 : 4 : 1.
  This ensures that the area of triangle N1N2N3 is 1/9 the area of triangle ABC. -/
theorem area_N1N2N3_eq_one_ninth_area_ABC
  (K : ℝ) -- Placeholder for the area of triangle ABC
  (ratio_CD : ℝ := 1 / 4) (ratio_AE : ℝ := 1 / 4) (ratio_BF : ℝ := 1 / 4)
  (ratio_AN2_N2N1_N1D : ℝ := 9 / (4 + 4 + 1)) :
  area N1N2N3 = (1 / 9) * area ABC :=
by sorry

end area_N1N2N3_eq_one_ninth_area_ABC_l655_655463


namespace three_digit_numbers_not_multiple_of_3_or_11_l655_655789

-- Proving the number of three-digit numbers that are multiples of neither 3 nor 11 is 547
theorem three_digit_numbers_not_multiple_of_3_or_11 : (finset.Icc 100 999).filter (λ n, ¬(3 ∣ n) ∧ ¬(11 ∣ n)).card = 547 :=
by
  -- The steps to reach the solution will be implemented here
  sorry

end three_digit_numbers_not_multiple_of_3_or_11_l655_655789


namespace find_S_l655_655996

theorem find_S (x y : ℝ) (h : x + y = 4) : 
  ∃ S, (∀ x y, x + y = 4 → 3*x^2 + y^2 = 12) → S = 6 := 
by 
  sorry

end find_S_l655_655996


namespace solve_t_l655_655911

theorem solve_t (t : ℝ) (h : 5 * 3^t + (25 * 9^t) ^ (1/2) = 50) : t = real.log 5 / real.log 3 :=
sorry

end solve_t_l655_655911


namespace mean_lt_median_eq_mode_l655_655810

def data_set : List ℕ := [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5]

def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def median (l : List ℕ) : ℚ :=
  if l.length % 2 = 0 then
    (l.nth (l.length / 2 - 1) + l.nth (l.length / 2)).getD 0 / 2
  else
    (l.nth (l.length / 2)).getD 0

def mode (l : List ℕ) : List ℕ :=
  match l.groupBy id with
  | [] => []
  | (x :: xs) => let g := x.length in
    let modes := (x :: xs).filter (λ y => y.length = g) in
    modes.map List.head!.init

theorem mean_lt_median_eq_mode :
  mean data_set < median data_set ∧ median data_set = 3 :=
by
  sorry

end mean_lt_median_eq_mode_l655_655810


namespace largest_integer_value_of_t_l655_655877

def is_D_t_function (f : ℝ → ℝ) (D : Set ℝ) (t : ℝ) : Prop :=
  ∀ x ∈ D, f x ≥ t

noncomputable def f (t : ℝ) : ℝ → ℝ :=
  λ x, (x - t) * Real.exp x

theorem largest_integer_value_of_t :
  (∃ t : ℤ, ∀ x : ℝ, (f t x) ≥ ↑t) ∧
  (∀ t' : ℤ, (∀ x : ℝ, (f t' x) ≥ ↑t') → t' ≤ -1) :=
by
  sorry

end largest_integer_value_of_t_l655_655877


namespace functional_interval_l655_655157

variable (x : ℝ)

def f (x : ℝ) : ℝ := real.sqrt (3 * x ^ 2 - x - 2)

def domain (x : ℝ) : Prop := x ≥ 1 ∨ x ≤ -2 / 3

lemma increasing_sqrt {t : ℝ} (ht : 0 < t) : monotone (λ t, real.sqrt t) := sorry

lemma increasing_inner (h : x > 1 / 6) : monotone (λ x, 3 * x ^ 2 - x - 2) := sorry

theorem functional_interval :
  ∀ x, domain x → monotone f :=
begin
  intro x,
  intro hx,
  cases hx,
  { -- Case x ≥ 1
    sorry },
  { -- Case x ≤ - 2 / 3
    sorry }
end

end functional_interval_l655_655157


namespace initial_even_distance_l655_655265

variables {d1 d2 d3 total_length remaining_length x : ℕ}

-- Conditions:
axiom condition1 : total_length = 5000
axiom condition2 : remaining_length = 3890
axiom condition3 : d1 = 300
axiom condition4 : d2 = 170
axiom condition5 : d3 = 440

-- Based on these conditions, define the actual question.
theorem initial_even_distance :
  (∃ x : ℕ, (x + (d1 - d2) + d3 = total_length - remaining_length) ∧ x = 540) :=
begin
  let x := total_length - remaining_length - (d1 - d2 + d3),
  use x,
  split,
  { sorry },
  { sorry }
end

end initial_even_distance_l655_655265


namespace percentage_y_less_than_x_l655_655207

variable (x y : ℝ)
variable (h : x = 12 * y)

theorem percentage_y_less_than_x :
  (11 / 12) * 100 = 91.67 := by
  sorry

end percentage_y_less_than_x_l655_655207


namespace sum_geometric_sequence_l655_655766

noncomputable def sequence (a : ℕ → ℝ) := ∀ n : ℕ, n > 0 → a (n + 1) = 2 * a n

lemma geometric_mean_condition (a : ℕ → ℝ) (h_sequence : sequence a) : (a 2 + a 4) / 2 = 5 := sorry

theorem sum_geometric_sequence (a : ℕ → ℝ) (n : ℕ) (h_sequence : sequence a)
  (h_mean : (a 2 + a 4) / 2 = 5) : (∑ i in finset.range n, a (i + 1)) = 2^n - 1 := sorry

end sum_geometric_sequence_l655_655766


namespace ratio_of_digits_l655_655128

variable (Carlos Sam Mina : ℕ)

def number_of_digits (Carlos_digits Sam_digits Mina_digits : ℕ) : Prop :=
  Sam_digits = Carlos_digits + 6 ∧
  Mina_digits = 24 ∧
  Sam_digits = 10 ∧
  ∃ k : ℕ, Mina_digits = k * Carlos_digits

theorem ratio_of_digits (h : number_of_digits Carlos Sam Mina) : (24 / Carlos) = 6 :=
by
  have h1 := h.1
  cases h1 with hSam hMinaMina
  have h2 := h1.1
  have h3 := h1.2
  have h4 := h.2.1
  have h5 := h.2.2
  sorry

end ratio_of_digits_l655_655128


namespace sin_cos_sum_eq_half_l655_655559

theorem sin_cos_sum_eq_half (α β : ℝ) (hα : α = 11) (hβ : β = 19) :
  sin (α * (π / 180) : ℝ) * cos (β * (π / 180) : ℝ) + cos (α * (π / 180) : ℝ) * sin (β * (π / 180) : ℝ) = 1 / 2 := by
  sorry

end sin_cos_sum_eq_half_l655_655559


namespace total_height_of_buildings_l655_655236

theorem total_height_of_buildings :
  let height_first_building := 600
  let height_second_building := 2 * height_first_building
  let height_third_building := 3 * (height_first_building + height_second_building)
  height_first_building + height_second_building + height_third_building = 7200 := by
    let height_first_building := 600
    let height_second_building := 2 * height_first_building
    let height_third_building := 3 * (height_first_building + height_second_building)
    show height_first_building + height_second_building + height_third_building = 7200
    sorry

end total_height_of_buildings_l655_655236


namespace length_CG_l655_655058
noncomputable def determine_length_CG (AE BF DH : ℝ) (AE_eq : AE = 10) (BF_eq : BF = 20) (DH_eq : DH = 30) : ℝ :=
  let GC := 20 * Real.sqrt 3 in
  GC

-- The theorem statement
theorem length_CG :
  ∀ AE BF DH : ℝ, AE = 10 → BF = 20 → DH = 30 → determine_length_CG AE BF DH AE_eq BF_eq DH_eq = 20 * Real.sqrt 3 :=
by
  intros AE BF DH AE_eq BF_eq DH_eq
  sorry

end length_CG_l655_655058


namespace continuity_at_2_l655_655107

noncomputable def f (x b : ℝ) : ℝ :=
if x ≤ 2 then 3 * x^2 - 7 else b * (x - 2)^2 + 5

theorem continuity_at_2 : ∀ b : ℝ, continuous_at (λ x : ℝ, f x b) 2 := by
  sorry

end continuity_at_2_l655_655107


namespace three_digit_numbers_not_multiple_of_3_or_11_l655_655782

theorem three_digit_numbers_not_multiple_of_3_or_11 : 
  let total_three_digit_numbers := 999 - 100 + 1 in
  let multiples_of_3 := 333 - 34 + 1 in
  let multiples_of_11 := 90 - 10 + 1 in
  let multiples_of_33 := 30 - 4 + 1 in
  let multiples_of_3_or_11 := multiples_of_3 + multiples_of_11 - multiples_of_33 in
  total_three_digit_numbers - multiples_of_3_or_11 = 546 :=
by
  let total_three_digit_numbers := 999 - 100 + 1
  let multiples_of_3 := 333 - 34 + 1
  let multiples_of_11 := 90 - 10 + 1
  let multiples_of_33 := 30 - 4 + 1
  let multiples_of_3_or_11 := multiples_of_3 + multiples_of_11 - multiples_of_33
  show total_three_digit_numbers - multiples_of_3_or_11 = 546 from sorry

end three_digit_numbers_not_multiple_of_3_or_11_l655_655782


namespace construct_quadrilateral_l655_655682

open EuclideanGeometry

variables (a c e f : ℝ) (ε : ℝ)

theorem construct_quadrilateral :
  ∃ (A B C D M : Point), 
  (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ A ≠ C ∧ B ≠ D) ∧
  (distance A B = a) ∧
  (distance C D = c) ∧
  (distance A C = e) ∧
  (distance B D = f) ∧
  let M := midpoint A C in
  let V := (D - A).angle (C - A) in
  (V = ε) :=
sorry

end construct_quadrilateral_l655_655682


namespace pam_number_of_bags_l655_655511

-- Definitions of the conditions
def apples_in_geralds_bag : Nat := 40
def pam_bags_ratio : Nat := 3
def total_pam_apples : Nat := 1200

-- Problem statement (Theorem)
theorem pam_number_of_bags :
  Pam_bags == total_pam_apples / (pam_bags_ratio * apples_in_geralds_bag) :=
by 
  sorry

end pam_number_of_bags_l655_655511


namespace mobot_coloring_six_colorings_l655_655725

theorem mobot_coloring_six_colorings (n m : ℕ) (h : n ≥ 3 ∧ m ≥ 3) :
  (∃ mobot, mobot = (1, 1)) ↔ (∃ colorings : ℕ, colorings = 6) :=
sorry

end mobot_coloring_six_colorings_l655_655725


namespace marble_pairs_count_l655_655177

theorem marble_pairs_count :
  let red := 1
  let green := 1
  let blue := 1
  let orange := 1
  let yellow := 5 in (yellow * (yellow - 1) / 2) + (red + green + blue + orange) * (red + green + blue + orange - 1) / 2 = 7 :=
by
  sorry

end marble_pairs_count_l655_655177


namespace star_polygon_net_of_pyramid_l655_655171

theorem star_polygon_net_of_pyramid (R r : ℝ) (H : R > 2 * r) :
  ∃ (polygon : ℕ → ℝ × ℝ), star_polygon_condition polygon R r :=
begin
  -- The proof goes here
  sorry
end

-- Definition of the condition of the star-shaped polygon forming the net of a pyramid
def star_polygon_condition (polygon : ℕ → ℝ × ℝ) (R r : ℝ) : Prop :=
  -- define the necessary properties of the polygon
  sorry

end star_polygon_net_of_pyramid_l655_655171


namespace min_diff_of_composite_sum_91_l655_655222

def is_composite (n : ℕ) : Prop :=
  2 ≤ (nat.factors n).length

theorem min_diff_of_composite_sum_91 : ∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ a + b = 91 ∧ (b - a = 1) :=
sorry

end min_diff_of_composite_sum_91_l655_655222


namespace max_tied_teams_round_robin_l655_655443

theorem max_tied_teams_round_robin (n : ℕ) (h: n = 8) :
  ∃ k, (k <= n) ∧ (∀ m, m > k → k * m < n * (n - 1) / 2) :=
by
  sorry

end max_tied_teams_round_robin_l655_655443


namespace general_term_formula_of_a_l655_655392

def S (n : ℕ) : ℚ := (3 / 2) * n^2 - 2 * n

def a (n : ℕ) : ℚ :=
  if n = 1 then (3 / 2) - 2
  else 2 * (3 / 2) * n - (3 / 2) - 2

theorem general_term_formula_of_a :
  ∀ n : ℕ, n > 0 → a n = 3 * n - (7 / 2) :=
by
  intros n hn
  sorry

end general_term_formula_of_a_l655_655392


namespace raw_score_is_correct_l655_655508

-- Define the conditions
def points_per_correct : ℝ := 1
def points_subtracted_per_incorrect : ℝ := 0.25
def total_questions : ℕ := 85
def answered_questions : ℕ := 82
def correct_answers : ℕ := 70

-- Define the number of incorrect answers
def incorrect_answers : ℕ := answered_questions - correct_answers
-- Calculate the raw score
def raw_score : ℝ := 
  (correct_answers * points_per_correct) - (incorrect_answers * points_subtracted_per_incorrect)

-- Prove the raw score is 67
theorem raw_score_is_correct : raw_score = 67 := 
by
  sorry

end raw_score_is_correct_l655_655508


namespace Aunt_Wang_Earnings_l655_655662

-- Definitions based on conditions
def P : ℝ := 50000
def r : ℝ := 4.5 / 100
def t : ℝ := 2

-- Define the simple interest calculation
def interest (P r t : ℝ) : ℝ := P * r * t

-- Lean 4 theorem statement
theorem Aunt_Wang_Earnings : interest P r t = 4500 :=
by
  sorry

end Aunt_Wang_Earnings_l655_655662


namespace numerical_trick_result_l655_655415

-- Let x be the chosen number.
variable (x : ℤ)

-- Define the steps of the numerical trick.
def step_sequence (x : ℤ) : ℤ := 
  let step1 := x * 6
  let step2 := step1 - 21
  let step3 := step2 / 3
  let final_result := step3 - (2 * x)
  final_result

theorem numerical_trick_result : step_sequence x = -7 := by sorry

end numerical_trick_result_l655_655415


namespace tea_intake_exceeds_900_l655_655509

theorem tea_intake_exceeds_900 :
  ∃ (n : ℕ), (n > 0) ∧ (∑ i in finset.range(n+1), i + ∑ i in finset.range(n+1), i^2) > 900 ∧
  ∀ m, m < n → (∑ i in finset.range(m+1), i + ∑ i in finset.range(m+1), i^2) ≤ 900 :=
by 
  sorry

end tea_intake_exceeds_900_l655_655509


namespace bryce_received_12_raisins_l655_655009

-- Defining the main entities for the problem
variables {x y z : ℕ} -- number of raisins Bryce, Carter, and Emma received respectively

-- Conditions:
def condition1 (x y : ℕ) : Prop := y = x - 8
def condition2 (x y : ℕ) : Prop := y = x / 3
def condition3 (y z : ℕ) : Prop := z = 2 * y

-- The goal is to prove that Bryce received 12 raisins
theorem bryce_received_12_raisins (x y z : ℕ) 
  (h1 : condition1 x y) 
  (h2 : condition2 x y) 
  (h3 : condition3 y z) : 
  x = 12 :=
sorry

end bryce_received_12_raisins_l655_655009


namespace gcd_polynomial_common_factor_l655_655138

def polynomial := 12 * a * b ^ 3 * c + 8 * a ^ 3 * b

theorem gcd_polynomial_common_factor (a b c : ℕ) :
  (∀ (a b c : ℕ), polynomial = 12 * a * b ^ 3 * c + 8 * a ^ 3 * b) → 
  gcd_polynomial_common_factor = 4 * a * b := 
by
  sorry

end gcd_polynomial_common_factor_l655_655138


namespace speed_of_trainA_l655_655963

-- Definitions based on conditions
def trainBSpeed : ℝ := 109.071  -- Speed of Train B in miles per hour
def totalDistance : ℝ := 240     -- Total distance between Austin and Houston in miles
def timeToPass : ℝ := 1.25542053973  -- Time in which the trains pass each other in hours

-- Defining the distance traveled by Train B
def distanceTrainB : ℝ := trainBSpeed * timeToPass

-- Defining the distance traveled by Train A
def distanceTrainA : ℝ := totalDistance - distanceTrainB

-- Defining the speed of Train A
def trainASpeed : ℝ := distanceTrainA / timeToPass

theorem speed_of_trainA :
  trainASpeed = 82.07 := by
  sorry

end speed_of_trainA_l655_655963


namespace intersection_points_l655_655398

def f(x : ℝ) : ℝ := x^2 + 3*x + 2
def g(x : ℝ) : ℝ := 4*x^2 + 6*x + 2

theorem intersection_points : {p : ℝ × ℝ | ∃ x, f x = p.2 ∧ g x = p.2 ∧ p.1 = x} = { (0, 2), (-1, 0) } := 
by {
  sorry
}

end intersection_points_l655_655398


namespace three_digit_numbers_not_multiples_of_3_or_11_l655_655776

def count_multiples (a b : ℕ) (lower upper : ℕ) : ℕ :=
  (upper - lower) / b + 1

theorem three_digit_numbers_not_multiples_of_3_or_11 : 
  let total := 900
  let multiples_3 := count_multiples 3 3 102 999
  let multiples_11 := count_multiples 11 11 110 990
  let multiples_33 := count_multiples 33 33 132 990
  let multiples_3_or_11 := multiples_3 + multiples_11 - multiples_33
  total - multiples_3_or_11 = 546 := 
by 
  sorry

end three_digit_numbers_not_multiples_of_3_or_11_l655_655776


namespace find_m_n_l655_655461

noncomputable def vector_magnitudes_and_angles (OA OB OC : ℝ) (AOC_tan BOC_angle : ℝ) : Prop :=
  OA = 2 ∧
  OB = 2 ∧
  OC = Real.sqrt 2 ∧
  AOC_tan = 3 ∧
  BOC_angle = π / 4

theorem find_m_n
  (OA OB OC : ℝ) (AOC_tan BOC_angle : ℝ)
  (cond : vector_magnitudes_and_angles OA OB OC AOC_tan BOC_angle) :
  ∃ m n : ℝ, OC = m * OA + n * OB ∧ (m, n) = (5 / 12, 1 / 3) :=
begin
  -- Sorry to skip the proof
  sorry
end

end find_m_n_l655_655461


namespace min_value_of_ratio_l655_655866

variables (A B C P : Type)
variables (a b c r1 r2 r3 S : ℝ)
variables [triangle : A] [triangle : B] [triangle : C]
variables (P : Type) [point : P]

-- conditions
def inside_triangle (P : Type) : Prop :=
  ∃ (a : ℝ) (r1 r2 r3 : ℝ), P ∈ triangleₐₐ a r1 r2 r3

noncomputable def is_incenter (P : Type) : Prop :=
  ∀ (a b c r1 r2 r3 S : ℝ),  P = incenter_triangleₐₐ a b c r1 r2 r3 S

noncomputable def minimum_value {P : Type} [point : P] (a b c r1 r2 r3 S : ℝ) : ℝ :=
  (a + b + c)^2 / (2 * S)

-- The proof problem
theorem min_value_of_ratio (a b c r1 r2 r3 S : ℝ) (P : incenter_triangleₐₐ a b c r1 r2 r3 S) :
  (a / r1 + b / r2 + c / r3) = minimum_value a b c r1 r2 r3 S :=
sorry


end min_value_of_ratio_l655_655866


namespace number_of_cute_integers_l655_655707

def is_cute (n : ℕ) : Prop :=
  let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  (digits.perm [1, 2, 3, 4]) ∧
  ((n / 1000) % 1 = 0) ∧  -- Divisible by 1 (trivially true)
  ((n / 100) % 2 = 0) ∧   -- First 2 digits divisible by 2
  ((n / 10) % 3 = 0) ∧    -- First 3 digits divisible by 3
  (n % 4 = 0)             -- All 4 digits divisible by 4

theorem number_of_cute_integers : ∃! (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ is_cute n :=
sorry

end number_of_cute_integers_l655_655707


namespace tan_of_pi_div_4_sub_alpha_eq_l655_655354

theorem tan_of_pi_div_4_sub_alpha_eq :
  ∀ (α : ℝ), α ∈ Ioo π (3 * π / 2) → cos α = -4 / 5 → tan (π / 4 - α) = 1 / 7 :=
by
  intro α hα hcos
  sorry

end tan_of_pi_div_4_sub_alpha_eq_l655_655354


namespace probability_all_co_captains_l655_655955

-- Define the number of students in each team
def students_team1 : ℕ := 4
def students_team2 : ℕ := 6
def students_team3 : ℕ := 7
def students_team4 : ℕ := 9

-- Define the probability of selecting each team
def prob_selecting_team : ℚ := 1 / 4

-- Define the probability of selecting three co-captains from each team
def prob_team1 : ℚ := 1 / Nat.choose students_team1 3
def prob_team2 : ℚ := 1 / Nat.choose students_team2 3
def prob_team3 : ℚ := 1 / Nat.choose students_team3 3
def prob_team4 : ℚ := 1 / Nat.choose students_team4 3

-- Define the total probability
def total_prob : ℚ :=
  prob_selecting_team * (prob_team1 + prob_team2 + prob_team3 + prob_team4)

theorem probability_all_co_captains :
  total_prob = 59 / 1680 := by
  sorry

end probability_all_co_captains_l655_655955


namespace dave_long_sleeve_shirts_l655_655687

def ss := 9
def washed := 20
def not_washed := 16

theorem dave_long_sleeve_shirts : 
  let total_shirts := washed + not_washed in
  let ls := total_shirts - ss in
  ls = 27 :=
by
  sorry

end dave_long_sleeve_shirts_l655_655687


namespace geometric_series_sum_eq_l655_655667

theorem geometric_series_sum_eq :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 5
  (∀ S_n, S_n = a * (1 - r^n) / (1 - r) → S_n = 1 / 3) :=
by
  intro a r n S_n
  sorry

end geometric_series_sum_eq_l655_655667


namespace douglas_percent_in_X_l655_655449

-- Define conditions: Douglas total percent, ratio of voters, percent in county Y
variables (V : ℝ) -- The number of voters in county Y
constants (P : ℝ) -- The percentage of votes Douglas won in county X

-- The given conditions
axiom h1 : (2 * P / 100 * V + 46 / 100 * V = 58 / 100 * (2 * V + V))

-- Prove the statement: P = 64
theorem douglas_percent_in_X : P = 64 :=
by
  -- The proof will be here
  sorry

end douglas_percent_in_X_l655_655449


namespace AI_eq_FI_l655_655051

variable {Point : Type} [EuclideanGeometry Point]

open EuclideanGeometry

-- Definitions of the vertices and points involved.
variable {A B C E D F G H I : Point}

-- Definitions provided in the problem
def is_midpoint (F : Point) (B C : Point) : Prop := dist F B = dist F C
def is_altitude (BE : Line) (CD : Line) (B C E D : Point) : Prop := 
  BE.perpendicular (Line.mk B E) ∧ CD.perpendicular (Line.mk C D)
def is_parallel (l1 l2 : Line) : Prop := l1.parallel l2

-- Conditions from the problem
variables 
  (hAcute : ∀ (X : Point), X ≠ A → X ≠ B → X ≠ C → ∠ A B X < 90 ∧ ∠ A C X < 90)
  (hAngle : ∠ A B C > ∠ A C B)
  (hMidF : is_midpoint F B C)
  (hAltitudes : is_altitude (Line.mk B E) (Line.mk C D) B C E D)
  (hMidG : is_midpoint G F D)
  (hMidH : is_midpoint H F E)
  (hParallel : is_parallel (Line.mk A (some_point_parallel_to_BC)) (Line.mk G H))

theorem AI_eq_FI : dist A I = dist F I :=
sorry

end AI_eq_FI_l655_655051


namespace total_animals_after_addition_l655_655890

def current_cows := 2
def current_pigs := 3
def current_goats := 6

def added_cows := 3
def added_pigs := 5
def added_goats := 2

def total_current_animals := current_cows + current_pigs + current_goats
def total_added_animals := added_cows + added_pigs + added_goats
def total_animals := total_current_animals + total_added_animals

theorem total_animals_after_addition : total_animals = 21 := by
  sorry

end total_animals_after_addition_l655_655890


namespace quadratic_roots_shifted_l655_655867

theorem quadratic_roots_shifted (a b c : ℚ) (r s : ℚ)
  (h1_eq: 5*r^2 + 2*r - 4 = 0)
  (h2_eq: 5*s^2 + 2*s - 4 = 0)
  (h_roots: (a = 1) ∧ (r-3)*(s-3) = c) :
  c = 47/5 :=
by
  sorry

end quadratic_roots_shifted_l655_655867


namespace max_tied_teams_round_robin_l655_655444

theorem max_tied_teams_round_robin (n : ℕ) (h: n = 8) :
  ∃ k, (k <= n) ∧ (∀ m, m > k → k * m < n * (n - 1) / 2) :=
by
  sorry

end max_tied_teams_round_robin_l655_655444


namespace domain_of_g_l655_655980

noncomputable def g (x : ℝ) : ℝ := log 3 (log 4 (log 5 (log 6 x)))

theorem domain_of_g : {x : ℝ | x > 7776} = {x : ℝ | real.logb 3 (real.logb 4 (real.logb 5 (real.logb 6 x)))} :=
sorry

end domain_of_g_l655_655980


namespace smallest_nat_ending_in_4_quadruples_l655_655718

theorem smallest_nat_ending_in_4_quadruples (x : ℕ) (a k : ℕ)
  (h₁ : x = 10 * a + 4)
  (h₂ : 4 * 10^k + a = 4 * x) :
  x = 102564 :=
by {
  have h₃ : 4 * 10^k + a = 40 * a + 16, from by rwa [h₁] at h₂,
  have h₄ : 4 * 10^k - 16 = 39 * a, from by linarith [h₁, h₃],
  have h₅ : a = (4 * 10^k - 16) / 39, from by rw [h₄, Nat.div_eq_of_lt],
  sorry
}

end smallest_nat_ending_in_4_quadruples_l655_655718


namespace largest_prime_factor_of_6_factorial_plus_7_factorial_l655_655589

theorem largest_prime_factor_of_6_factorial_plus_7_factorial :
  ∃ p : ℕ, Nat.Prime p ∧ p = 5 ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ (6! + 7!) → q ≤ 5 :=
by
  sorry

end largest_prime_factor_of_6_factorial_plus_7_factorial_l655_655589


namespace min_a_value_l655_655020

theorem min_a_value {a b : ℕ} (h : 1998 * a = b^4) : a = 1215672 :=
sorry

end min_a_value_l655_655020


namespace malvina_coins_l655_655114

variable (m n : ℕ)

axiom (h1 : m + n < 40)
axiom (h2 : n < 8 * m)
axiom (h3 : n ≥ 4 * m + 15)

theorem malvina_coins : n = 31 := by
  sorry

end malvina_coins_l655_655114


namespace find_both_artifacts_total_time_l655_655073

variables (months_in_year : Nat) (expedition_first_years : Nat) (artifact_factor : Nat)

noncomputable def total_time (research_months : Nat) (expedition_first_years : Nat) :=
  let research_first_years := float_of_nat research_months / float_of_nat months_in_year
  let total_first := research_first_years + float_of_nat expedition_first_years
  let total_second := artifact_factor * total_first
  total_first + total_second

theorem find_both_artifacts_total_time :
  forall (months_in_year : Nat) (expedition_first_years : Nat) (artifact_factor : Nat),
    months_in_year = 12 → 
    expedition_first_years = 2 → 
    artifact_factor = 3 → 
    total_time 6 expedition_first_years = 10 :=
by intros months_in_year expedition_first_years artifact_factor hm he hf
   unfold total_time 
   sorry

end find_both_artifacts_total_time_l655_655073


namespace angle_EMF_90_l655_655881

theorem angle_EMF_90
  (S1 S2 : Type) [circle S1] [circle S2]
  (A B C D M N K E F : Type)
  {h_inters : S1 ∩ S2 = {A, B}}
  {h_line_A_C : line_through A C ∈ S1}
  {h_line_A_D : line_through A D ∈ S2}
  {h_M_CD : M ∈ segment C D}
  {h_N_BC : N ∈ segment B C}
  {h_K_BD : K ∈ segment B D}
  {h_MN_parallel_BD : MN ∥ BD}
  {h_MK_parallel_BC : MK ∥ BC}
  {h_E_on_arc_BC : E ∈ arc B C S1 \ {A}}
  {h_F_on_arc_BD : F ∈ arc B D S2 \ {A}}
  {h_EN_perp_BC : EN ⊥ BC}
  {h_FK_perp_BD : FK ⊥ BD} :  
  ∠EMF = 90° := sorry

end angle_EMF_90_l655_655881


namespace distance_from_origin_to_projection_is_five_l655_655427

-- Define the 3D point structure
structure Point3D where
  x : ℕ
  y : ℕ
  z : ℕ

-- Define a specific instance of point P
def P : Point3D := {x := 3, y := -4, z := 5}

-- Define the projection of point P onto the xoy plane
def projectToXOY (p : Point3D) : Point3D :=
  {x := p.x, y := p.y, z := 0}

-- Define the origin point
def O : Point3D := {x := 0, y := 0, z := 0}

-- Function to compute the distance between two 3D points
def distance (a b : Point3D) : ℚ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2 + (a.z - b.z)^2)

-- Assert that the distance from the origin to the projection of point P is 5
theorem distance_from_origin_to_projection_is_five : distance O (projectToXOY P) = 5 := by
  sorry

end distance_from_origin_to_projection_is_five_l655_655427


namespace minimum_oranges_to_profit_l655_655625

/-- 
A boy buys 4 oranges for 12 cents and sells 6 oranges for 25 cents. 
Calculate the minimum number of oranges he needs to sell to make a profit of 150 cents.
--/
theorem minimum_oranges_to_profit (cost_oranges : ℕ) (cost_cents : ℕ)
  (sell_oranges : ℕ) (sell_cents : ℕ) (desired_profit : ℚ) :
  cost_oranges = 4 → cost_cents = 12 →
  sell_oranges = 6 → sell_cents = 25 →
  desired_profit = 150 →
  (∃ n : ℕ, n = 129) :=
by
  sorry

end minimum_oranges_to_profit_l655_655625


namespace tan_B_in_triangle_BCD_l655_655834

theorem tan_B_in_triangle_BCD (C B D : Type*) [EuclideanGeometry C B D]
  (angle_C_eq_90 : ∠BCD = 90)
  (CD_eq_3 : distance C D = 3)
  (BD_eq_sqrt13 : distance B D = Real.sqrt 13) :
  tan_ B = 3 / 2 := 
begin
  sorry
end

end tan_B_in_triangle_BCD_l655_655834


namespace intersection_value_l655_655831

-- Define the parametric equation of line l
def parametric_line_l (t : ℝ) : ℝ × ℝ := (2 + 1/2 * t, (√3 / 2) * t)

-- Define the polar equation of curve C in rectangular coordinates
def polar_to_rectangular (rho theta : ℝ) : ℝ × ℝ := (rho * cos theta, rho * sin theta)

def curve_C (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the point M
def M : ℝ × ℝ := (2, 0)

lemma ordinary_equations_line_l_curve_C 
  (t : ℝ) 
  (x y : ℝ)
  : (∃ t, parametric_line_l t = (x, y) → y = √3 * (x - 2)) ∧ (curve_C x y → y^2 = 4 * x)
:= sorry

theorem intersection_value 
  (t1 t2 : ℝ)
  (x1 y1 x2 y2 : ℝ)
  (h1 : parametric_line_l t1 = (x1, y1))
  (h2 : parametric_line_l t2 = (x2, y2))
  (hC1 : curve_C x1 y1)
  (hC2 : curve_C x2 y2)
  (hM : M = (2, 0))
  : ∣(1 / |((2 : ℝ) - x1) ^ 2 + (0 - y1) ^ 2| - 1 / |((2 : ℝ) - x2) ^ 2 + (0 - y2) ^ 2|)∣ = 1/4 
:= sorry

end intersection_value_l655_655831


namespace chessboard_marking_checkerboard_l655_655503

theorem chessboard_marking_checkerboard (board : fin 8 × fin 8) (marked_cells : finset (fin 8 × fin 8)) :
  (∀ cell : fin 8 × fin 8, 
    marked_cells.card ≤ 32 ∧ 
    ∃! adjacent : fin 8 × fin 8, 
    adjacent ∈ marked_cells ∧ 
    (abs (cell.1 - adjacent.1) = 1 ∧ cell.2 = adjacent.2 ∨ 
     abs (cell.2 - adjacent.2) = 1 ∧ cell.1 = adjacent.1)) :=
sorry

end chessboard_marking_checkerboard_l655_655503


namespace problem_statement_l655_655405

section
variables {m : ℝ} {x y : ℝ}

/-- Define the line l1 -/
def l1 (m x y : ℝ) : Prop := (m + 2) * x + m * y - 6 = 0

/-- Define the line l2 -/
def l2 (m x y : ℝ) : Prop := m * x + y - 3 = 0

/-- Condition (1) - l1 is perpendicular to l2 -/
def perp (m : ℝ) : Prop :=
  if m = 0 then True else 
  let slope_l1 := -(m + 2) / m in
  let slope_l2 := -m in
  slope_l1 * slope_l2 = -1

/-- Condition (2) - point P(1, 2m) lies on l2, and intercepts are negatives of each other -/
def passes_through_p_and_intercepts (m : ℝ) (x : ℝ) : Prop :=
  l2 m 1 (2 * m) ∧ ∃ k, (x = 0 → 2 - k = 0) ∧ (2 = k * (x - 1)) ∧ 
  (1 + (2 - k) / k = -(2 - k))

/-- Main theorem combining both conditions and conclusions -/
theorem problem_statement :
  (perp m → m = 0 ∨ m = -3) ∧ 
  (passes_through_p_and_intercepts 1 2 → ∃ (l : ℝ → ℝ → Prop), 
    (l = (2 * x - y = 0) ∨ l = (x - y + 1 = 0))) :=
by sorry
end

end problem_statement_l655_655405


namespace inequality_proof_l655_655846

theorem inequality_proof (x y : ℝ) (hx : -1 ≤ x ∧ x ≤ 1) (hy : -1 ≤ y ∧ y ≤ 1) :
  2 * sqrt((1 - x^2) * (1 - y^2)) ≤ 2 * (1 - x) * (1 - y) + 1 := 
by
  sorry

end inequality_proof_l655_655846


namespace incenter_l655_655477

variable (a b c : ℝ)
variable {A B C P E F : Type}
variable [linear_ordered_field ℝ]

variables [affine_space P ℝ] [affine_space E ℝ] [affine_space F ℝ] 
  [affine_space A ℝ] [affine_space B ℝ] [affine_space C ℝ]

def triangle_ABC (BC AC AB: ℝ) := BC = a ∧ AC = b ∧ AB = c

def point_property
  (P : P)
  (any_line_through_P : p → {AB AF : P} → Prop)
  (intersects_E : AB → AE)
  (intersects_F : AF → AF) :=
  ∀ P : P, ∀ (line_through_P : p) 
  , intersects_E (AB) → intersects_F (AF)
  , 1/AE + 1/AF = (a + b + c) / (b * c)

def is_incenter 
  (P : P) :=
  ∀ {A B C : P}, 
  ∀ {p},
  1/AE + 1/AF = (a + b + c) / (b * c)

theorem incenter (a b c: ℝ) {A B C P E F : Type}
  [affine_space P ℝ] [affine_space E ℝ] [affine_space F ℝ]
  [affine_space A ℝ] [affine_space B ℝ] [affine_space C ℝ]
  (h_triangle : triangle_ABC a b c)
  (h_property : point_property P any_line_through_P intersects_E intersects_F) :
  is_incenter P :=
begin
  sorry
end

end incenter_l655_655477


namespace even_function_composition_is_even_l655_655859

-- Let's define what it means for a function to be even
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- The main theorem stating the evenness of the composition of an even function
theorem even_function_composition_is_even {f : ℝ → ℝ} (h : even_function f) :
  even_function (λ x, f (f x)) :=
by
  intros x
  have : f (-x) = f x := h x
  rw [←this, h (-x)]
  sorry

end even_function_composition_is_even_l655_655859


namespace daniel_video_games_l655_655685

/--
Daniel has a collection of some video games. 80 of them, Daniel bought for $12 each.
Of the rest, 50% were bought for $7. All others had a price of $3 each.
Daniel spent $2290 on all the games in his collection.
Prove that the total number of video games in Daniel's collection is 346.
-/
theorem daniel_video_games (n : ℕ) (r : ℕ)
    (h₀ : 80 * 12 = 960)
    (h₁ : 2290 - 960 = 1330)
    (h₂ : r / 2 * 7 + r / 2 * 3 = 1330):
    n = 80 + r → n = 346 :=
by
  intro h_total
  have r_eq : r = 266 := by sorry
  rw [r_eq] at h_total
  exact h_total

end daniel_video_games_l655_655685


namespace mean_of_transformed_data_l655_655369

variable {α : Type*}

-- Define the data set and the given variance condition
def data_set_var (x : Fin 6 → α) [Add α] [Mul α] [Inv α] [OfNat α 1] [OfNat α 6] [OfNat α 24] [MulOfNat α 6] (S_squared : α) : Prop :=
  S_squared = (1 / 6) * ((x 0)^2 + (x 1)^2 + (x 2)^2 + (x 3)^2 + (x 4)^2 + (x 5)^2 - 24)

-- The proof problem defined in Lean 4
theorem mean_of_transformed_data (x : Fin 6 → ℝ) (S_squared : ℝ) (h : data_set_var x S_squared) :
  let x_bar := sqrt 4 in (x_bar + 1 = 3) ∨ (x_bar - 1 = -1) :=
sorry

end mean_of_transformed_data_l655_655369


namespace sin_A_value_l655_655434

noncomputable def solve_triangle : Prop :=
  ∃ (a b C A : ℝ), 
  a = 2 ∧ b = 3 ∧ C = Real.pi * 2 / 3 ∧ 
  sin (Real.arcsin (2 * sin Real.pi * 2 / 3 / sqrt 19)) = (sqrt 57) / 19

theorem sin_A_value : solve_triangle :=
by
  sorry

end sin_A_value_l655_655434


namespace dog_to_dropped_ratio_l655_655505

-- Definitions based on the conditions
def initial_matches := 70
def dropped_matches := 10
def remaining_matches := 40

-- Definition of the derived amounts
def total_lost_matches := initial_matches - remaining_matches
def dog_ate_matches := total_lost_matches - dropped_matches

-- Theorem stating the required ratio
theorem dog_to_dropped_ratio
  (initial_matches = 70)
  (dropped_matches = 10)
  (remaining_matches = 40)
  (total_lost_matches = initial_matches - remaining_matches)
  (dog_ate_matches = total_lost_matches - dropped_matches) :
  (dog_ate_matches : ℚ) / (dropped_matches : ℚ) = 2 :=
by
  sorry

end dog_to_dropped_ratio_l655_655505


namespace evaluate_modulus_l655_655488

noncomputable def q (x : ℕ) : ℤ := (List.range (2010 / 2 + 1)).sum (λ k, x^(2010 - 2 * k))

noncomputable def r : ℕ → ℤ := λ x, x^5 + x^4 + x^3 + 2*x^2 + x + 1

noncomputable def s : ℕ → ℤ := λ x, (q x) %ₘ (r x)

theorem evaluate_modulus : |s 2010| % 1000 = 11 := 
by
  sorry

end evaluate_modulus_l655_655488


namespace exist_positive_abc_with_nonzero_integer_roots_l655_655321

theorem exist_positive_abc_with_nonzero_integer_roots :
  ∃ (a b c : ℤ), 0 < a ∧ 0 < b ∧ 0 < c ∧
  (∀ x y : ℤ, (a * x^2 + b * x + c = 0 → x ≠ 0 ∧ y ≠ 0)) ∧
  (∀ x y : ℤ, (a * x^2 + b * x - c = 0 → x ≠ 0 ∧ y ≠ 0)) ∧
  (∀ x y : ℤ, (a * x^2 - b * x + c = 0 → x ≠ 0 ∧ y ≠ 0)) ∧
  (∀ x y : ℤ, (a * x^2 - b * x - c = 0 → x ≠ 0 ∧ y ≠ 0)) :=
sorry

end exist_positive_abc_with_nonzero_integer_roots_l655_655321


namespace part1_part2_l655_655374

open Set

variable {ι : Type*} (R : Type*) [LinearOrder R] [TopologicalSpace R]

def A : Set R := {x : R | -3 < 2 * x + 1 ∧ 2 * x + 1 < 11}
def B (m : R) : Set R := {x : R | m - 1 ≤ x ∧ x ≤ 2 * m + 1}

theorem part1 (m : R) (h : m = 3) : A R ∩ (Set.compl (B R m)) = {x : R | -2 < x ∧ x < 2} := by sorry

theorem part2 (m : R) : (A R ∪ B R m = A R) ↔ (m < -2 ∨ -1 < m ∧ m < 2) := by sorry

end part1_part2_l655_655374


namespace technicians_count_l655_655819

theorem technicians_count 
    (total_workers : ℕ) (avg_salary_all : ℕ) (avg_salary_technicians : ℕ) (avg_salary_rest : ℕ)
    (h_workers : total_workers = 28) (h_avg_all : avg_salary_all = 8000) 
    (h_avg_tech : avg_salary_technicians = 14000) (h_avg_rest : avg_salary_rest = 6000) : 
    ∃ T R : ℕ, T + R = total_workers ∧ (avg_salary_technicians * T + avg_salary_rest * R = avg_salary_all * total_workers) ∧ T = 7 :=
by
  sorry

end technicians_count_l655_655819


namespace count_valid_m_values_l655_655310

theorem count_valid_m_values :
  let m_values := {m : ℕ | ∃ n : ℕ, n ∣ 420 ∧ m ∣ 420 ∧ n > 1 ∧ m > 1 ∧ even n ∧ m * n = 420}
  m_values.to_finset.card = 9 :=
by
  sorry

end count_valid_m_values_l655_655310


namespace sum_of_number_and_reverse_is_perfect_square_iff_l655_655167

def is_two_digit (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100

def reverse_of (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  10 * b + a

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem sum_of_number_and_reverse_is_perfect_square_iff :
  ∀ n : ℕ, is_two_digit n →
    is_perfect_square (n + reverse_of n) ↔
      n = 29 ∨ n = 38 ∨ n = 47 ∨ n = 56 ∨ n = 65 ∨ n = 74 ∨ n = 83 ∨ n = 92 :=
by
  sorry

end sum_of_number_and_reverse_is_perfect_square_iff_l655_655167


namespace isosceles_triangle_perimeter_l655_655033

theorem isosceles_triangle_perimeter :
  (∃ x y : ℝ, x^2 - 6*x + 8 = 0 ∧ y^2 - 6*y + 8 = 0 ∧ (x = 2 ∧ y = 4) ∧ 2 + 4 + 4 = 10) :=
by
  sorry

end isosceles_triangle_perimeter_l655_655033


namespace no_real_roots_of_compositions_l655_655148

theorem no_real_roots_of_compositions {a b c d e f : ℝ}
  (f g : ℝ → ℝ)
  (hf : ∀ x, f(x) = a * x^2 + b * x + c)
  (hg : ∀ x, g(x) = d * x^2 + e * x + f)
  (hfg : ∀ x, f(g(x)) ≠ 0)
  (hgf : ∀ x, g(f(x)) ≠ 0) :
  (∀ x, f(f(x)) ≠ 0) ∨ (∀ x, g(g(x)) ≠ 0) :=
by
  sorry

end no_real_roots_of_compositions_l655_655148


namespace find_equation_of_line_parallel_and_area_l655_655710

theorem find_equation_of_line_parallel_and_area :
  ∃ (C : ℝ), (2 * x + 5 * y + C = 0) ∧ 
    (∃ (C1 : C = 10) ∨ ∃ (C2 : C = -10)) ∧ 
    ∀ (A : ℝ), A = (|C|^2) / 20 → A = 5 :=
begin
  sorry
end

end find_equation_of_line_parallel_and_area_l655_655710


namespace find_x_parallel_l655_655005

-- Define vector tuples
def vector_a := (3 : ℝ, 1 : ℝ)
def vector_b (x : ℝ) := (x, -1 : ℝ)

-- Define the parallel condition for vectors
def parallel {a b c d : ℝ} (v1 : (ℝ × ℝ)) (v2 : (ℝ × ℝ)) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

-- The conjecture we need to prove
theorem find_x_parallel: ∃ x : ℝ, parallel vector_a (vector_b x) ∧ x = -3 :=
by
  sorry -- proof to be filled in

end find_x_parallel_l655_655005


namespace rectangle_area_l655_655972

open Real

def point := (ℝ × ℝ)

noncomputable def distance (p q : point) : ℝ :=
sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

noncomputable def area_of_rectangle (P Q R S : point) (h1 : distance P Q = distance S R)
  (hPQ_side : distance P Q = sqrt 17)
  (hPR_diag : distance P R = sqrt 29) : ℝ :=
distance P Q * sqrt (distance P R ^ 2 - distance P Q ^ 2)

theorem rectangle_area
  (P Q R S : point)
  (hP : P = (1, 1))
  (hQ : Q = (-3, 2))
  (hR : R = (-1, 6))
  (hS : S = (3, 5))
  (hPQ_side : distance P Q = sqrt 17)
  (hPR_diag : distance P R = sqrt 29) :
  area_of_rectangle P Q R S (by simp [hP, hQ, hR, hS, distance]) hPQ_side hPR_diag = 2 * sqrt 51 := by
  sorry

end rectangle_area_l655_655972


namespace eqn_abs_3x_minus_2_solution_l655_655332

theorem eqn_abs_3x_minus_2_solution (x : ℝ) :
  (|x + 5| = 3 * x - 2) ↔ x = 7 / 2 :=
by
  sorry

end eqn_abs_3x_minus_2_solution_l655_655332


namespace cone_apex_angle_l655_655803

theorem cone_apex_angle (R : ℝ) 
  (h1 : ∀ (θ : ℝ), (∃ (r : ℝ), r = R / 2 ∧ 2 * π * r = π * R)) :
  ∀ (θ : ℝ), θ = π / 3 :=
by
  sorry

end cone_apex_angle_l655_655803


namespace find_a_l655_655358

theorem find_a (a : ℝ) (h₀ : a > 0) 
  (p : ∀ x : ℝ, monotone_decreasing (λ x, a^x)) 
  (q : ∀ x : ℝ, x^2 - 3 * a * x + 1 > 0) : 
  (p ∨ q) ↔ (2 / 3 ≤ a ∧ a < 1) :=
sorry

end find_a_l655_655358


namespace find_X_l655_655806

-- Define the variables for income, tax, and the variable X
def income := 58000
def tax := 8000

-- Define the tax formula as per the problem
def tax_formula (X : ℝ) : ℝ :=
  0.11 * X + 0.20 * (income - X)

-- The theorem we want to prove
theorem find_X :
  ∃ X : ℝ, tax_formula X = tax ∧ X = 40000 :=
sorry

end find_X_l655_655806


namespace cos_of_sin_condition_l655_655734

-- Define the condition as a hypothesis
theorem cos_of_sin_condition (α : ℝ) (h : sin (α + π / 6) = sqrt 3 / 3) : cos (2 * π / 3 - 2 * α) = -1 / 3 := 
by {
  sorry  -- Proof can be filled in here
}

end cos_of_sin_condition_l655_655734


namespace remainder_101_pow_47_mod_100_l655_655190

theorem remainder_101_pow_47_mod_100 : (101 ^ 47) % 100 = 1 := by 
  sorry

end remainder_101_pow_47_mod_100_l655_655190


namespace empty_two_thirds_l655_655268

theorem empty_two_thirds (eight_minutes : ℕ)
  (empties_one_third_in_eight : ∀ t, t = 8 → t = eight_minutes → 1 / 3) :
  ∀ t, ((t = eight_minutes * 2) → (2 / 3)) := by
  sorry

end empty_two_thirds_l655_655268


namespace value_of_a_l655_655397

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem value_of_a (a : ℝ) : 
  (∀ x, deriv (f a) x = 6 * x + 3 * a * x^2) →
  deriv (f a) (-1) = 6 → a = 4 :=
by
  -- Proof will be filled in here
  sorry

end value_of_a_l655_655397


namespace number_of_correct_conclusions_l655_655046

open List

def data : List ℝ := [9.80, 9.70, 9.55, 9.54, 9.48, 9.42, 9.41, 9.35, 9.30, 9.25]

def mean (l : List ℝ) : ℝ := (l.sum) / (l.length)

def median (l : List ℝ) : ℝ :=
  let sorted_l := l.qsort (λ x y => x < y)
  if hlen : sorted_l.length % 2 = 0 then
    let mid1 := sorted_l.nthLe (sorted_l.length / 2 - 1) (by simp [List.length, hlen])
    let mid2 := sorted_l.nthLe (sorted_l.length / 2) (by simp [List.length, hlen])
    (mid1 + mid2) / 2
  else
    sorted_l.nthLe (sorted_l.length / 2) sorry -- should never happen

def range (l : List ℝ) : ℝ :=
  l.maximum (by simp) - l.minimum (by simp)

theorem number_of_correct_conclusions : 
  (mean data = 9.48) ∧ (median data = 9.45) ∧ (range data = 0.55) → 3 = 3 :=
by 
  intros h
  sorry

end number_of_correct_conclusions_l655_655046


namespace tan_of_second_quadrant_l655_655383

theorem tan_of_second_quadrant (α : ℝ) (h₁ : α > π / 2 ∧ α < π) (h₂ : cos α = -12 / 13) : tan α = -5 / 12 :=
by sorry

end tan_of_second_quadrant_l655_655383


namespace find_x_squared_l655_655703

theorem find_x_squared :
  ∃ x : ℕ, (x^2 >= 2525 * 10^8) ∧ (x^2 < 2526 * 10^8) ∧ (x % 100 = 17 ∨ x % 100 = 33 ∨ x % 100 = 67 ∨ x % 100 = 83) ∧
    (x = 502517 ∨ x = 502533 ∨ x = 502567 ∨ x = 502583) :=
sorry

end find_x_squared_l655_655703


namespace handshake_problem_l655_655272

theorem handshake_problem (n : ℕ) (hn : n = 11) (H : n * (n - 1) / 2 = 55) : 10 = n - 1 :=
by
  sorry

end handshake_problem_l655_655272


namespace double_even_l655_655860

-- Define even function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Lean statement of the mathematically equivalent proof problem
theorem double_even (f : ℝ → ℝ) (h : is_even_function f) : is_even_function (f ∘ f) :=
by
  sorry

end double_even_l655_655860


namespace length_AE_correct_l655_655833

noncomputable def right_triangle_length_AE (A B C D M E : Point) (AB AC BC AD AM AE: ℝ) : Prop :=
  right_angle B A C ∧ 
  AB = 3 ∧ 
  AC = 3 * Real.sqrt 3 ∧ 
  AD_perpendicular_to_BC A D C ∧ 
  AM_is_median A M B C ∧ 
  AM_intersects_AD_at E A M D →
  AE = Real.sqrt 3

-- Here we assume several geometric properties as placeholders
axiom right_angle : Point → Point → Point → Prop
axiom AD_perpendicular_to_BC : Point → Point → Point → Prop
axiom AM_is_median : Point → Point → Point → Point → Prop
axiom AM_intersects_AD_at : Point → Point → Point → Point → Prop

theorem length_AE_correct (A B C D M E : Point) (AB AC BC AD AM AE : ℝ) :
  right_triangle_length_AE A B C D M E AB AC BC AD AM AE := 
by {
  sorry -- Proof required here
}

end length_AE_correct_l655_655833


namespace angle_380_in_first_quadrant_l655_655204

theorem angle_380_in_first_quadrant : ∃ n : ℤ, 380 - 360 * n = 20 ∧ 0 ≤ 20 ∧ 20 ≤ 90 :=
by
  use 1 -- We use 1 because 380 = 20 + 360 * 1
  sorry

end angle_380_in_first_quadrant_l655_655204


namespace cakes_left_correct_l655_655291

def number_of_cakes_left (total_cakes sold_cakes : ℕ) : ℕ :=
  total_cakes - sold_cakes

theorem cakes_left_correct :
  number_of_cakes_left 54 41 = 13 :=
by
  sorry

end cakes_left_correct_l655_655291


namespace number_of_2_configs_l655_655864

def set_A : Set (Set Char) := 
  { {'V', 'W'}, {'W', 'X'}, {'X', 'Y'}, {'Y', 'Z'}, {'Z', 'V'}, 
    {'v', 'x'}, {'v', 'y'}, {'w', 'y'}, {'w', 'z'}, {'x', 'z'}, 
    {'V', 'v'}, {'W', 'w'}, {'X', 'x'}, {'Y', 'y'}, {'Z', 'z'} }

def is_2_config_of_order_1 (pairs : Set (Set Char)) : Prop := 
  ∀ a ∈ (set_A.flatten), (a ∈ pairs.flatten → (Set.card (pairs.filter (λ p, a ∈ p)) = 1))

theorem number_of_2_configs (pairs : Set (Set Char)) : 
  is_2_config_of_order_1 pairs → (Set.card pairs = 6) := by sorry

end number_of_2_configs_l655_655864


namespace second_piece_cost_l655_655075

theorem second_piece_cost
  (total_spent : ℕ)
  (num_pieces : ℕ)
  (single_piece1 : ℕ)
  (single_piece2 : ℕ)
  (remaining_piece_count : ℕ)
  (remaining_piece_cost : ℕ)
  (total_cost : total_spent = 610)
  (number_of_items : num_pieces = 7)
  (first_item_cost : single_piece1 = 49)
  (remaining_piece_item_cost : remaining_piece_cost = 96)
  (first_item_total_cost : remaining_piece_count = 5)
  (sum_equation : single_piece1 + single_piece2 + (remaining_piece_count * remaining_piece_cost) = total_spent) :
  single_piece2 = 81 := 
  sorry

end second_piece_cost_l655_655075


namespace number_in_scientific_notation_l655_655264

-- Definition of the given number and scientific notation representation conditions
def number := 380180000000

def scientific_notation (a : ℝ) (b : ℤ) : ℝ := a * (10 ^ b)

-- The theorem that states the number in scientific notation
theorem number_in_scientific_notation : scientific_notation 3.8018 11 = number :=
by
  sorry

end number_in_scientific_notation_l655_655264


namespace actual_number_of_children_l655_655605

-- Define the conditions of the problem
def condition1 (C B : ℕ) : Prop := B = 2 * C
def condition2 : ℕ := 320
def condition3 (C B : ℕ) : Prop := B = 4 * (C - condition2)

-- Define the statement to be proved
theorem actual_number_of_children (C B : ℕ) 
  (h1 : condition1 C B) (h2 : condition3 C B) : C = 640 :=
by 
  -- Proof will be added here
  sorry

end actual_number_of_children_l655_655605


namespace no_line_bisected_by_P_exists_l655_655464

theorem no_line_bisected_by_P_exists (P : ℝ × ℝ) (H : ∀ x y : ℝ, (x / 3)^2 - (y / 2)^2 = 1) : 
  P ≠ (2, 1) := 
sorry

end no_line_bisected_by_P_exists_l655_655464


namespace number_of_digits_in_x_l655_655371

open Real

theorem number_of_digits_in_x
  (x y : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y)
  (hxy_inequality : x > y)
  (hxy_prod : x * y = 490)
  (hlog_cond : (log x - log 7) * (log y - log 7) = -143/4) :
  ∃ n : ℕ, n = 8 ∧ (10^(n - 1) ≤ x ∧ x < 10^n) :=
by
  sorry

end number_of_digits_in_x_l655_655371


namespace total_pencils_l655_655078

-- Defining the number of pencils each person has.
def jessica_pencils : ℕ := 8
def sandy_pencils : ℕ := 8
def jason_pencils : ℕ := 8

-- Theorem stating the total number of pencils
theorem total_pencils : jessica_pencils + sandy_pencils + jason_pencils = 24 := by
  sorry

end total_pencils_l655_655078


namespace sector_area_given_angle_radius_sector_max_area_perimeter_l655_655742

open Real

theorem sector_area_given_angle_radius :
  ∀ (α : ℝ) (R : ℝ), α = 60 * (π / 180) ∧ R = 10 →
  (α / 360 * 2 * π * R) = 10 * π / 3 ∧ 
  (α * π * R^2 / 360) = 50 * π / 3 :=
by
  intros α R h
  rcases h with ⟨hα, hR⟩
  sorry

theorem sector_max_area_perimeter :
  ∀ (r α: ℝ), (2 * r + r * α) = 8 →
  α = 2 →
  r = 2 :=
by
  intros r α h ha
  sorry

end sector_area_given_angle_radius_sector_max_area_perimeter_l655_655742


namespace simplify_expression_l655_655670

theorem simplify_expression (x : ℝ) : x^2 * x^4 + x * x^2 * x^3 = 2 * x^6 := by
  sorry

end simplify_expression_l655_655670


namespace geometric_progression_sum_l655_655903

theorem geometric_progression_sum (a_1 q : ℝ) (hq : 0 < q) : 
  let S1 := a_1 * (1 + q + q^2),
      S2 := a_1 * q^3 * (1 + q + q^2),
      S3 := a_1 * q^6 * (1 + q + q^2)
  in S2 = sqrt (S1 * S3) :=
by
  let S1 := a_1 * (1 + q + q^2)
  let S2 := a_1 * q^3 * (1 + q + q^2)
  let S3 := a_1 * q^6 * (1 + q + q^2)
  sorry

end geometric_progression_sum_l655_655903


namespace valid_parameterizations_l655_655930

def line_equation (x y : ℝ) : Prop := 
  y = -3 * x + 1

def parameterization_A (t : ℝ) : ℝ × ℝ :=
  (1 + t, -2 - 3 * t)

def parameterization_B (t : ℝ) : ℝ × ℝ :=
  ((1/3) + t, 0 + 3 * t)

def parameterization_C (t : ℝ) : ℝ × ℝ :=
  (0 + 3 * t, 1 - 9 * t)

def parameterization_D (t : ℝ) : ℝ × ℝ :=
  (-1 + t, 4 + 3 * t)

def parameterization_E (t : ℝ) : ℝ × ℝ :=
  (0 + (t / 3), 1 - t)

theorem valid_parameterizations : 
  (∀ t, line_equation (parameterization_A t).1 (parameterization_A t).2) ∧
  (∀ t, line_equation (parameterization_B t).1 (parameterization_B t).2) ∧
  (∀ t, line_equation (parameterization_C t).1 (parameterization_C t).2) ∧
  (∀ t, line_equation (parameterization_D t).1 (parameterization_D t).2) ∧
  ¬ (∀ t, line_equation (parameterization_E t).1 (parameterization_E t).2) := 
sorry

end valid_parameterizations_l655_655930


namespace solve_for_x_l655_655912

theorem solve_for_x : (∃ x : ℚ, x = (√(3^2 + 4^2)) / (√(20 + 5)) ∧ x = 1) :=
by
  sorry

end solve_for_x_l655_655912


namespace cube_root_twentyseven_sixth_root_sixty_four_square_root_nine_l655_655183

theorem cube_root_twentyseven_sixth_root_sixty_four_square_root_nine :
  (∛27) * (∛64^(1/6)) * (√9) = 18 :=
by
  sorry

end cube_root_twentyseven_sixth_root_sixty_four_square_root_nine_l655_655183


namespace anvil_factory_journeymen_percentage_l655_655604

theorem anvil_factory_journeymen_percentage:
  let total_employees := 20210
  let fraction_journeymen := 2 / 7
  let initial_journeymen := fraction_journeymen * total_employees
  let laid_off_journeymen := initial_journeymen / 2
  let remaining_journeymen := initial_journeymen - laid_off_journeymen
  let total_remaining_employees := total_employees - laid_off_journeymen
  let percentage_journeymen := (remaining_journeymen / total_remaining_employees) * 100
  percentage_journeymen ≈ 16.57 :=
sorry

end anvil_factory_journeymen_percentage_l655_655604


namespace value_of_a_l655_655034

theorem value_of_a (a : ℕ) (h_odd : ¬ even (88 * a)) (h_multiple_of_3 : 3 ∣ (88 * a)) : a = 5 :=
sorry

end value_of_a_l655_655034


namespace flagstaff_shadow_length_l655_655240

theorem flagstaff_shadow_length :
  ∀ (flagstaff_height building_height building_shadow_flagstaff_shadow : ℝ),
    flagstaff_height = 17.5 →
    building_height = 12.5 →
    building_shadow_flagstaff_shadow = 28.75 →
    (flagstaff_height / building_shadow_flagstaff_shadow = building_height / 28.75) →
    building_shadow_flagstaff_shadow = 40.25 :=
begin
  intros,
  sorry
end

end flagstaff_shadow_length_l655_655240


namespace find_p_l655_655466

def sequence (a : ℤ → ℤ) : Prop :=
∀ n, a (n - 2) + a (n - 1) = a n

theorem find_p :
  ∃ p q r s t u a : ℤ → ℤ,
    a (-3) = 3 ∧
    a (-2) = 5 ∧
    a (-1) = 8 ∧
    a 0 = 13 ∧
    a 1 = 21 ∧
    sequence a ∧
    a (-6) = p ∧
    a (-6) = -1 :=
begin
  sorry
end

end find_p_l655_655466


namespace circle_center_x_coordinate_eq_l655_655169

theorem circle_center_x_coordinate_eq (a : ℝ) (h : (∃ k : ℝ, ∀ x y : ℝ, x^2 + y^2 - a * x = k) ∧ (1 = a / 2)) : a = 2 :=
sorry

end circle_center_x_coordinate_eq_l655_655169


namespace smallest_x_abs_eq_18_l655_655343

theorem smallest_x_abs_eq_18 : 
  ∃ x : ℝ, (|2 * x + 5| = 18) ∧ (∀ y : ℝ, (|2 * y + 5| = 18) → x ≤ y) :=
sorry

end smallest_x_abs_eq_18_l655_655343


namespace pet_store_combinations_l655_655252

theorem pet_store_combinations : 
  (20 * 10 * 12 * 5) * (nat.factorial 4) = 288000 := by
  sorry

end pet_store_combinations_l655_655252


namespace parallelogram_vertex_D_l655_655830

-- Coordinates of the points A, B, and C
def A := (4, 1, 3)
def B := (2, -5, 1)
def C := (3, 7, -5)

-- Definition of vector subtraction
def vec_sub (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

-- Definition of the coordinates of vertex D
def D := (5, 13, -3)

-- Theorem to prove that D is correct given the parallelogram condition
theorem parallelogram_vertex_D :
  let AB := vec_sub A B
  let DC := vec_sub D C
  AB = DC := 
sorry

end parallelogram_vertex_D_l655_655830


namespace math_grades_logic_l655_655610

theorem math_grades_logic
  (students : Fin 4) (grades : students → Prop) (excellent good : Prop)
  (A B C D : students)
  (H1 : ∃! x, grades x = excellent)  -- There are exactly 2 students with excellent grades
  (H2 : ∃! x, grades x = good)      -- There are exactly 2 students with good grades
  (H3 : ¬ (grades A ∧ grades B ∧ grades C))   -- A does not know their own grade
  (H4 : ∀ x, x ≠ A → grades x = grades B ∨ grades x = grades C)  -- A knows B and C have one excellent, one good grade
  (H5 : ∀ x, x ≠ D → grades x = grades A ∨ grades x = grades D)  -- D knows A and D have one excellent, one good grade
  (H6 : ∃ x, x = A ∨ x = B ∨ x = C)      -- A has seen the grades of B and C
  (H7 : ∃ y, y = D ∨ y = A)      -- D has seen the grade of A  
  : (grades B → grades D) :=
sorry

end math_grades_logic_l655_655610


namespace sum_first10PrimesGT50_eq_732_l655_655193

def first10PrimesGT50 := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

theorem sum_first10PrimesGT50_eq_732 :
  first10PrimesGT50.sum = 732 := by
  sorry

end sum_first10PrimesGT50_eq_732_l655_655193


namespace concurrence_of_BP_DQ_DR_l655_655765

-- Given points and conditions
variables {A B C D E F P Q R : Point}
variable [quadrilateral ABCD]
variable (E_on_BC : lies_on E BC)
variable (F_on_CD : lies_on F CD)
variable (midpoint_P : midpoint P A E)
variable (midpoint_Q : midpoint Q E F)
variable (midpoint_R : midpoint R A F)

-- Required theorem to prove
theorem concurrence_of_BP_DQ_DR : concurrent BP DQ DR :=
sorry

end concurrence_of_BP_DQ_DR_l655_655765


namespace perpendicular_lines_l655_655616

theorem perpendicular_lines (k : ℝ) (h : k = 5) :
  (∃ k, kx + 5y - 2 = 0 ∧ (4 - k)x + y - 7 = 0 ∧ (k = 5 → perpendicular)) ∧
  (perpendicular → k = 5 ∨ k = -1) :=
by
  sorry

end perpendicular_lines_l655_655616


namespace polynomial_coeff_sum_l655_655426

theorem polynomial_coeff_sum {a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} : ℝ} :
  (∀ x : ℝ, (1 - 2 * x) ^ 10 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 +
                    a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 + a_{10} * x^10) →
  (a_0 + a_1) + (a_0 + a_2) + (a_0 + a_3) + (a_0 + a_4) + (a_0 + a_5) +
  (a_0 + a_6) + (a_0 + a_7) + (a_0 + a_8) + (a_0 + a_9) + (a_0 + a_{10}) = 10 :=
by
  sorry

end polynomial_coeff_sum_l655_655426


namespace angle_between_vectors_length_two_a_minus_b_l655_655360

open Real
open ComplexConjugate
open Complex

noncomputable def vec_a : ℝ × ℝ × ℝ := sorry
noncomputable def vec_b : ℝ × ℝ × ℝ := sorry

def len_vec (v : ℝ × ℝ × ℝ) := sqrt (v.1^2 + v.2^2 + v.3^2)
def dot_product (v1 v2 : ℝ × ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

axiom cond1 : len_vec vec_a = 2
axiom cond2 : len_vec vec_b = 1
axiom cond3 : dot_product (vec_a.1 - vec_b.1, vec_a.2 - vec_b.2, vec_a.3 - vec_b.3) 
                           ((2 * vec_a.1 + vec_b.1), (2 * vec_a.2 + vec_b.2), (2 * vec_a.3 + vec_b.3)) = 8

theorem angle_between_vectors : let a := len_vec vec_a in
                                let b := len_vec vec_b in
                                let ab := dot_product vec_a vec_b in
                                a = 2 → b = 1 → ab = -1 → 
                                acos (ab / (a * b)) = 2 * real.pi / 3 :=
by intros; sorry

theorem length_two_a_minus_b : let a := vec_a in
                               let b := vec_b in
                               let len := len_vec (2 * a.1 - b.1, 2 * a.2 - b.2, 2 * a.3 - b.3) in
                               len = sqrt 21 :=
by intros; sorry

end angle_between_vectors_length_two_a_minus_b_l655_655360


namespace triangle_ABC_proof_l655_655835

noncomputable def sin2C_eq_sqrt3sinC (C : ℝ) : Prop := Real.sin (2 * C) = Real.sqrt 3 * Real.sin C

theorem triangle_ABC_proof (C a b c : ℝ) 
  (H1 : sin2C_eq_sqrt3sinC C) 
  (H2 : 0 < Real.sin C)
  (H3 : b = 6) 
  (H4 : a + b + c = 6*Real.sqrt 3 + 6) :
  (C = π/6) ∧ (1/2 * a * b * Real.sin C = 6*Real.sqrt 3) :=
sorry

end triangle_ABC_proof_l655_655835


namespace max_value_of_f_l655_655476

noncomputable def f (x : ℝ) : ℝ :=
  real.sqrt (x * (75 - x)) + real.sqrt (x * (3 - x))

theorem max_value_of_f :
  ∃ (x₀ M : ℝ), (0 ≤ x₀ ∧ x₀ ≤ 3) ∧
  (∀ x ∈ Icc 0 3, f x ≤ M) ∧
  (f x₀ = M) ∧
  (x₀ = 25 / 8 ∧ M = 15) :=
by sorry

end max_value_of_f_l655_655476


namespace line_between_circle_centers_l655_655145

noncomputable def circle_center (a b c : ℝ) : ℝ × ℝ :=
  (a, b)

theorem line_between_circle_centers : 
  ∀ {x y : ℝ}, 
  circle_center 2 (-3) (-13) = (2, -3) ∧ 
  circle_center 3 0 (-9) = (3, 0) → 
  3 * x - y - 9 = 0 :=
begin
  sorry
end

end line_between_circle_centers_l655_655145


namespace victor_percentage_80_l655_655969

def percentage_of_marks (marks_obtained : ℕ) (maximum_marks : ℕ) : ℕ :=
  (marks_obtained * 100) / maximum_marks

theorem victor_percentage_80 :
  percentage_of_marks 240 300 = 80 := by
  sorry

end victor_percentage_80_l655_655969


namespace find_monthly_salary_l655_655599

variables (S : ℝ) (E : ℝ)

-- Define that the man saves 10% of his salary
def initial_savings : Prop := (S = E + 0.10 * S)

-- Define that after increasing his expenses by 5%, he saves Rs. 400
def increased_expenses_savings : Prop := (S = 1.05 * E + 400)

-- Define the man's salary
def man_monthly_salary : Prop := S ≈ 7272.73

theorem find_monthly_salary (h1 : initial_savings S E) (h2 : increased_expenses_savings S E) : man_monthly_salary S :=
sorry

end find_monthly_salary_l655_655599


namespace part_a_part_b_l655_655400

section

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - x
def g (x : ℝ) : ℝ := x^2 + x

-- Define their derivatives
noncomputable def f' (x : ℝ) := (deriv f) x
noncomputable def g' (x : ℝ) := (deriv g) x

-- a) prove that f'(-3/2) = 2 * g'(-3/2)
theorem part_a : ∃ x : ℝ, f'(x) = 2 * g'(x) ↔ x = -3/2 :=
by sorry 

-- b) prove that there is no x such that f'(x) = g'(x)
theorem part_b : ∀ x : ℝ, f'(x) ≠ g'(x) :=
by sorry

end

end part_a_part_b_l655_655400


namespace trigonometric_solution_count_l655_655017

theorem trigonometric_solution_count : 
  ∃ (θs : set ℝ), (∀ θ ∈ θs, 0 < θ ∧ θ ≤ 2 * Real.pi ∧ 
  2 - 4 * Real.sin θ + 6 * Real.cos (2 * θ) + Real.sin (3 * θ) = 0) ∧ θs.card = 8 :=
sorry

end trigonometric_solution_count_l655_655017


namespace integer_values_count_of_x_l655_655796

theorem integer_values_count_of_x (x : ℝ) (h : ⌈real.sqrt x⌉ = 17) : ∃ n : ℕ, n = 33 :=
by
  let lower_bound := 256
  let upper_bound := 289
  have h1 : lower_bound < x,
  have h2 : x ≤ upper_bound,
  let count := upper_bound - lower_bound + 1
  use count
  sorry

end integer_values_count_of_x_l655_655796


namespace correct_calculation_l655_655999

theorem correct_calculation (x : ℝ) : (2 * x^5) / (-x)^3 = -2 * x^2 :=
by sorry

end correct_calculation_l655_655999


namespace find_n_values_l655_655705

theorem find_n_values (n : ℕ) (h_n_pos: n > 0) :
  (∃ k : ℕ, k ≥ 2 ∧ ∃ a : ℕ → ℚ, (∀ i, 1 ≤ i ∧ i ≤ k → 0 < a i) ∧
    (∑ i in finset.range k, a i = n) ∧ (∏ i in finset.range k, a i = n)) ↔ (n = 4 ∨ n = 6 ∨ n = 7 ∨ n ≥ 9) :=
sorry

end find_n_values_l655_655705


namespace shells_total_l655_655113

variable (x y : ℝ)

theorem shells_total (h1 : y = x + (x + 32)) : y = 2 * x + 32 :=
sorry

end shells_total_l655_655113


namespace total_animals_after_addition_l655_655891

def current_cows := 2
def current_pigs := 3
def current_goats := 6

def added_cows := 3
def added_pigs := 5
def added_goats := 2

def total_current_animals := current_cows + current_pigs + current_goats
def total_added_animals := added_cows + added_pigs + added_goats
def total_animals := total_current_animals + total_added_animals

theorem total_animals_after_addition : total_animals = 21 := by
  sorry

end total_animals_after_addition_l655_655891


namespace Pasha_solution_l655_655120

noncomputable def PashaGame : Prop :=
  ∃ (a : Fin 2017 → ℕ),
    let add_stones_in_43_moves (moves: ℕ): Fin 2017 → ℕ :=
      λ i, moves * a i
    in
    ∀ moves : ℕ,
      if moves = 43 then
        add_stones_in_43_moves moves = λ i, 86
      else if 0 < moves ∧ moves < 43 then
        ¬(∀ i, add_stones_in_43_moves moves i = 86)

theorem Pasha_solution : PashaGame :=
  sorry

end Pasha_solution_l655_655120


namespace max_area_BAD_l655_655847

-- Definitions
variables (A B C D E F : Point)
-- Assuming appropriate definitions for 'Point' and properties regarding midpoints and areas.

-- Given conditions
axiom midpoint_E : midpoint E B C
axiom midpoint_F : midpoint F C D

-- Segments cutting the quadrilaterals into triangles with areas being consecutive integers
axiom areas_consecutive : ∃ (x : ℕ), 
  area (triangle A E B) = x ∧ 
  area (triangle A F C) = x + 1 ∧ 
  area (triangle E F C) = x + 2 ∧ 
  area (triangle F E D) = x + 3

-- Desired proof
theorem max_area_BAD : 
  let total_area := (4 * (nat.find areas_consecutive)) + 6 in
  ∃x, area (triangle A B D) = total_area - (4 * x) → 
  x < 6 ∧ x = 6 :=
begin
  sorry
end

end max_area_BAD_l655_655847


namespace product_divisibility_l655_655493

noncomputable def c (r : ℕ) : ℕ := r * (r + 1)

theorem product_divisibility (k m n : ℕ) (h1 : 0 < k) (h2 : 0 < m) (h3 : 0 < n)
  (h_prime : Nat.Prime (m + k + 1)) (h_greater : m + k + 1 > n + 1) :
  (∏ i in finset.range n, c (m + 1 + i) - c k) ∣ (∏ j in finset.range n, c (j + 1)) := by
  sorry

end product_divisibility_l655_655493


namespace sufficient_but_not_necessary_l655_655218

variable (p q : Prop)

theorem sufficient_but_not_necessary (h : p ∧ q) : ¬¬p :=
  by sorry -- Proof not required

end sufficient_but_not_necessary_l655_655218


namespace bicycle_trip_length_l655_655076

def total_distance (days1 day1 miles1 day2 miles2: ℕ) : ℕ :=
  days1 * miles1 + day2 * miles2

theorem bicycle_trip_length :
  total_distance 12 12 1 6 = 150 :=
by
  sorry

end bicycle_trip_length_l655_655076


namespace Harriet_found_5_pennies_l655_655995

-- Define the values of different coins
def value_quarter : ℝ := 0.25
def value_dime : ℝ := 0.10
def value_nickel : ℝ := 0.05
def value_penny : ℝ := 0.01

-- Define the number of coins found
def num_quarters : ℕ := 10
def num_dimes : ℕ := 3
def num_nickels : ℕ := 3

-- Define the total amount of money found
def total_money : ℝ := 3.00

-- Define the total value calculated for the found quarters, dimes, and nickels
def total_other_coins_value := 
  (num_quarters * value_quarter) + 
  (num_dimes * value_dime) + 
  (num_nickels * value_nickel)

-- Define the remaining amount of money to be counted in pennies
def remaining_money := total_money - total_other_coins_value

-- Define the number of pennies
def num_pennies := remaining_money / value_penny

-- The theorem to be proven
theorem Harriet_found_5_pennies : num_pennies = 5 :=
by {
  sorry
}

end Harriet_found_5_pennies_l655_655995


namespace angle_AOB_constant_l655_655365

-- Define the Hyperbola with given conditions
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ (x^2 / a^2 - y^2 / b^2 = 1)

-- Define the condition on eccentricity
def eccentricity (c a : ℝ) : Prop :=
  c / a = √3

-- Define the right directrix
def right_directrix (x : ℝ) : Prop :=
  x = √3 / 3

-- Specify the hyperbola parameters
def hyperbola_params (a b c : ℝ) : Prop :=
  a = 1 ∧ c = √3 ∧ b^2 = c^2 - a^2

-- Define the circle
def circle (radius : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = radius^2

-- Define the tangent to the circle
def tangent (m n : ℝ) : (ℝ × ℝ) → Prop
| (x, y) := m * x + n * y = 2

-- Define intersection points and prove the angle ∠AOB is constant
theorem angle_AOB_constant (a b c : ℝ) (x1 y1 x2 y2 : ℝ) (m n : ℝ) (h_hyperbola : hyperbola a b)
  (h_eccentricity : eccentricity c a) (h_directrix : right_directrix (√3 / 3))
  (h_params : hyperbola_params a b c) (h_circle : circle 2 m n) (h_tangent : tangent m n (x1, y1)) :
  ∠AOB (x1, y1) (x2, y2) = 90 :=
sorry

end angle_AOB_constant_l655_655365


namespace hundreds_digit_of_factorial_difference_l655_655185

theorem hundreds_digit_of_factorial_difference : 
  (25! - 20!) % 1000 / 100 = 0 :=
by
  sorry

end hundreds_digit_of_factorial_difference_l655_655185


namespace real_numbers_a_b_l655_655496

theorem real_numbers_a_b (a b : ℝ) (h1 : 3^a + 13^b = 17^a) (h2 : 5^a + 7^b = 11^b) : a < b :=
by
  sorry

end real_numbers_a_b_l655_655496


namespace cost_per_box_l655_655474

theorem cost_per_box 
  (homes_A : ℕ) (boxes_per_home_A : ℕ) 
  (homes_B : ℕ) (boxes_per_home_B : ℕ) 
  (total_revenue : ℝ) 
  (total_boxes_A : ℕ) 
  (total_boxes_B : ℕ) 
  (more_boxes : total_boxes_B > total_boxes_A) 
  (revenue_max : total_revenue = 50) : 
  total_boxes_B = 25 → (total_revenue / total_boxes_B) = 2 := by 
  intros h
  rw h
  simp
  done

end cost_per_box_l655_655474


namespace exists_natural_sum_of_squares_l655_655471

theorem exists_natural_sum_of_squares : ∃ n : ℕ, n^2 = 0^2 + 7^2 + 24^2 + 312^2 + 48984^2 :=
by {
  sorry
}

end exists_natural_sum_of_squares_l655_655471


namespace trigonometric_identity_in_triangle_l655_655522

variables {a b c : ℝ} {α β γ : ℝ}

theorem trigonometric_identity_in_triangle
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : α + β + γ = real.pi)
  (h5 : a * real.sin γ = c * real.sin α)
  (h6 : b = a * real.cos γ + c * real.cos α) :
  b * real.sin β + a * real.cos β * real.sin γ = c * real.sin γ + a * real.cos γ * real.sin β :=
sorry

end trigonometric_identity_in_triangle_l655_655522


namespace customer_payment_difference_l655_655238

theorem customer_payment_difference (P : ℝ) (h1 : P > 0) :
  let original_gbp := (1.20 * P * 1.08 * 0.70) / 1.4,
      new_gbp := (1.20 * P * 1.08 * 0.90) / 1.35,
      difference := new_gbp - original_gbp
  in difference = 0.216 * P :=
by
  sorry

end customer_payment_difference_l655_655238


namespace bus_full_after_12_stops_l655_655228

theorem bus_full_after_12_stops :
  ∃ n : ℕ, (∑ k in Finset.range (n + 1), k) = 78 ∧ n = 12 :=
by
  use 12
  sorry

end bus_full_after_12_stops_l655_655228


namespace probability_of_white_crows_remain_same_l655_655172

theorem probability_of_white_crows_remain_same (a b c d : ℕ) (h1 : a + b = 50) (h2 : c + d = 50) 
  (ha1 : a > 0) (h3 : b ≥ a) (h4 : d ≥ c - 1) :
  ((b - a) * (d - c) + a + b) / (50 * 51) > (bc + ad) / (50 * 51)
:= by
  -- We need to show that the probability of the number of white crows on the birch remaining the same 
  -- is greater than the probability of it changing.
  sorry

end probability_of_white_crows_remain_same_l655_655172


namespace max_conspiratorial_subset_len_l655_655672

def is_pairwise_rel_prime (a b c : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1

def is_conspiratorial (S : Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → ¬ is_pairwise_rel_prime a b c

theorem max_conspiratorial_subset_len {S : Finset ℕ} (h : ∀ x ∈ S, 1 ≤ x ∧ x ≤ 16) :
  ∃ C : Finset ℕ, is_conspiratorial C ∧ C.card = 11 :=
begin
  sorry
end

end max_conspiratorial_subset_len_l655_655672


namespace weekend_price_is_correct_l655_655271

-- Define the parameters and conditions
variables {P : ℝ} (h_weekday_price : P = 18) (h_weekend_markup : P * 1.5 = 27)

-- Define the theorem to be proved
theorem weekend_price_is_correct : ∃ P, P = 18 ∧ P * 1.5 = 27 :=
by
  use 18
  split
  · exact h_weekday_price
  · exact h_weekend_markup
  sorry

end weekend_price_is_correct_l655_655271


namespace find_function_l655_655852

theorem find_function (f : ℕ → ℕ) (k : ℕ) :
  (∀ n : ℕ, f n < f (n + 1)) →
  (∀ n : ℕ, f (f n) = n + 2 * k) →
  ∀ n : ℕ, f n = n + k := 
by
  intro h1 h2
  sorry

end find_function_l655_655852


namespace union_sets_l655_655003

theorem union_sets {U : Set ℝ} (P Q : Set ℝ) 
  (hU : U = Set.univ) 
  (hP : P = Set.Ioc 0 1) 
  (hQ : Q = { x | 2 ^ x ≤ 1 }) : 
  P ∪ Q = Set.Iic 1 := 
by sorry

end union_sets_l655_655003


namespace count_ways_to_complete_20160_l655_655062

noncomputable def waysToComplete : Nat :=
  let choices_for_last_digit := 5
  let choices_for_first_three_digits := 9^3
  choices_for_last_digit * choices_for_first_three_digits

theorem count_ways_to_complete_20160 (choices : Fin 9 → Fin 9) : waysToComplete = 3645 := by
  sorry

end count_ways_to_complete_20160_l655_655062


namespace ellipse_equation_correct_G_lies_on_fixed_line_l655_655394

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

theorem ellipse_equation_correct (a b c : ℝ) (ha : a = 2) (hb : b = sqrt 3) (hc : c = 1) : 
  ∀ x y, ellipse_equation x y ↔ (x^2 / a^2 + y^2 / b^2 = 1) := 
by
  sorry

theorem G_lies_on_fixed_line (M N F_2 G B : Type) 
  (ellipse_eq : ellipse_equation)
  (intersects_at : ∀ k : ℝ, k ≠ 0 → ∃ M N : ℝ × ℝ, 
    ∀ (x : ℝ) (y : ℝ), (x, y) ∈ {M, N})
  (intersects_at_lines : ∃ G, intersect (line A_1 M) (line A_2 N))
  (G_line_eq : G = (1, 3*sqrt 3 / 2)) : 
  ∀ G, G ∈ {F_2} → G ∈ (x = 1) :=
by
  sorry

end ellipse_equation_correct_G_lies_on_fixed_line_l655_655394


namespace triangle_area_eq_40_sqrt_3_l655_655039

open Real

theorem triangle_area_eq_40_sqrt_3 
  (a : ℝ) (A : ℝ) (b c : ℝ)
  (h1 : a = 14)
  (h2 : A = π / 3) -- 60 degrees in radians
  (h3 : b / c = 8 / 5) :
  1 / 2 * b * c * sin A = 40 * sqrt 3 :=
by
  sorry

end triangle_area_eq_40_sqrt_3_l655_655039


namespace doughnuts_left_l655_655325

theorem doughnuts_left (initial_doughnuts : ℕ) (num_staff : ℕ) (doughnuts_per_staff : ℕ)
  (initial_doughnuts = 50) (num_staff = 19) (doughnuts_per_staff = 2) :
  initial_doughnuts - (num_staff * doughnuts_per_staff) = 12 :=
by sorry

end doughnuts_left_l655_655325


namespace unwatered_rosebushes_l655_655060

/--
In a garden, Anna and Vitya had 2006 rosebushes. Vitya watered half of all the bushes,
and Anna watered half of all the bushes. Exactly three of the bushes were watered by 
both Anna and Vitya. Prove that 3 rosebushes remained unwatered.
-/
theorem unwatered_rosebushes :
  ∀ (total_bushes bushes_watered_by_anna bushes_watered_by_vitya common_bushes : ℕ),
  total_bushes = 2006 →
  bushes_watered_by_anna = total_bushes / 2 →
  bushes_watered_by_vitya = total_bushes / 2 →
  common_bushes = 3 →
  total_bushes - (bushes_watered_by_anna + bushes_watered_by_vitya - common_bushes) = 3 :=
by
  intros total_bushes bushes_watered_by_anna bushes_watered_by_vitya common_bushes
  assume h1 h2 h3 h4
  sorry

end unwatered_rosebushes_l655_655060


namespace find_n_for_stamp_problem_l655_655344

-- Define a structure to encapsulate the problem conditions and result
structure StampProblem where
  n : ℕ -- positive integer n
  maximum_unformable_postage : ℕ
  stamps : Set ℕ -- set of stamps including 4, n, and n+1

-- Instantiate the structure with the given values.
def stamp_problem : StampProblem :=
  { n := 21, 
    maximum_unformable_postage := 57,
    stamps := {4, 21, 22} }

-- Define the main problem statement to verify the solution
theorem find_n_for_stamp_problem : ∃ n, 
  (n > 0) ∧ 
  (stamp_problem.maximum_unformable_postage = 57) ∧
  ({4, n, n + 1} = stamp_problem.stamps) := 
begin
  -- Hard-coded for n = 21
  use 21,
  split,
  -- Positive integer condition
  exact nat.succ_pos' 20,
  split,
  -- Maximum unformable postage is 57
  exact rfl,
  -- Stamps set verification
  simp [stamp_problem, StampProblem.n, StampProblem.stamps],
end

end find_n_for_stamp_problem_l655_655344


namespace area_of_circular_pool_l655_655647

theorem area_of_circular_pool :
  ∀ (A B C D : Point) (R : ℝ), 
  distance A B = 20 ∧ 
  distance D C = 12 ∧ 
  midpoint D A B ∧ 
  center C ∧
  R^2 = 244 → 
  area_circle C = 244 * π :=
by
  intros A B C D R h,
  have h1 : distance A B = 20, from h.1,
  have h2 : distance D C = 12, from h.2,
  have h3 : midpoint D A B, from h.3,
  have h4 : center C, from h.4,
  have h5 : R^2 = 244, from h.5,
  sorry

end area_of_circular_pool_l655_655647


namespace survey_total_students_l655_655813

theorem survey_total_students
    (A B AB : ℕ)
    (cond1 : AB = 0.20 * (A + AB))
    (cond2 : AB = 0.25 * (B + AB))
    (cond3 : A - B = 100) :
    A + B + AB = 800 :=
by
  -- theoretical proof should be inserted here
  sorry

end survey_total_students_l655_655813


namespace income_ratio_l655_655163

-- Definitions of the conditions
variable (I1 I2 E1 E2 : ℝ)
variable (P1_income P2_income : ℝ)

-- Given conditions
axiom exp_ratio : E1 / E2 = 3 / 2
axiom save_amount : 1800
axiom income_P1 : P1_income = 4500
axiom exp_eq_income_P1 : E1 = P1_income - save_amount
axiom exp_eq_income_P2 : E2 = P2_income - save_amount

-- Theorem stating the ratio of their incomes
theorem income_ratio (h1 : E1 / E2 = 3 / 2)
                     (h2 : E1 = P1_income - save_amount)
                     (h3 : E2 = P2_income - save_amount)
                     (h4 : P1_income = 4500)
                     (h5 : save_amount = 1800) : 
                     P1_income / P2_income = 5 / 4 := by
  sorry

end income_ratio_l655_655163


namespace problem_I_problem_II_l655_655109

section 

variables {x y m : ℝ}

-- Defining p and q based on given conditions
def p (m x : ℝ) : Prop := x > m ∧ ¬ (2 * x - 5 > 0 → x > m)
def q (m x y : ℝ) : Prop := (x^2 / (m - 1) + y^2 / (2 - m) = 1)

-- Proving (Ⅰ) and (Ⅱ)
theorem problem_I (h : ∃ x y, p m x ∧ q m x y) : m ∈ (-∞, 1) ∪ (2, 5/2] := 
sorry

theorem problem_II (h : ∃ x y, ¬(p m x ∧ q m x y) ∧ (p m x ∨ q m x y)) : m ∈ [1, 2] ∪ (5/2, ∞) := 
sorry

end

end problem_I_problem_II_l655_655109


namespace maximum_tied_teams_in_tournament_l655_655445

theorem maximum_tied_teams_in_tournament : 
  ∀ (n : ℕ), n = 8 →
  (∀ (wins : ℕ), wins = (n * (n - 1)) / 2 →
   ∃ (k : ℕ), k ≤ n ∧ (k > 7 → false) ∧ 
               (∃ (w : ℕ), k * w = wins)) :=
by
  intros n hn wins hw
  use 7
  split
  · exact (by linarith)
  · intro h
    exfalso
    exact h (by linarith)
  · use 4
    calc
      7 * 4 = 28 : by norm_num
      ... = 28 : by rw hw; linarith
  
-- The proof is omitted as per instructions ("sorry" can be used to indicate this).

end maximum_tied_teams_in_tournament_l655_655445


namespace solutions_x_y_l655_655706

-- Helper definitions for the rational solutions
noncomputable def rational_sol (p : ℚ) :=
  ( (1 + 1/p) ^ (p + 1), (1 + 1/p) ^ p )

-- Main theorem statement
theorem solutions_x_y (x y : ℚ) :

  -- Natural numbers case
  ((x ∈ ℕ ∧ y ∈ ℕ) → (x = y ∨ (x = 4 ∧ y = 2)) ) ∧

  -- Rational numbers case
  ((x ∈ ℚ ∧ y ∈ ℚ) → ( ∃ p : ℚ, p ≠ 0 ∧ (x, y) = rational_sol p )) :=
sorry

end solutions_x_y_l655_655706


namespace inverse_proportional_fraction_l655_655657

theorem inverse_proportional_fraction (N : ℝ) (d f : ℝ) (h : N ≠ 0):
  d * f = N :=
sorry

end inverse_proportional_fraction_l655_655657


namespace squared_greater_abs_greater_l655_655732

theorem squared_greater_abs_greater {a b : ℝ} : a^2 > b^2 ↔ |a| > |b| :=
by sorry

end squared_greater_abs_greater_l655_655732


namespace possible_integer_values_of_x_l655_655794

theorem possible_integer_values_of_x : 
  (∃! x : ℕ, 256 < x ∧ x ≤ 289) ↔ fintype.card {x : ℕ | 256 < x ∧ x ≤ 289} = 33 := 
by sorry

end possible_integer_values_of_x_l655_655794


namespace isosceles_triangle_possible_values_l655_655943

theorem isosceles_triangle_possible_values (x : ℝ) (hx0 : x > 0) (hx180 : x < 180) :
  let a := sin (x * π / 180)
  let b := sin (5 * x * π / 180)
  a = b ↔ x = 90 / 7 ∨ x = 270 / 7 :=
by sorry

end isosceles_triangle_possible_values_l655_655943


namespace calculate_S2017_l655_655164

def a : ℕ → ℚ
| 0       := 1
| (n + 1) := -1 / (1 + a n)

def S : ℕ → ℚ
| 0       := 0
| (n + 1) := S n + a (n + 1)

theorem calculate_S2017 : S 2017 = -1007 := sorry

end calculate_S2017_l655_655164


namespace thirteen_coins_value_l655_655205

theorem thirteen_coins_value :
  ∃ (p n d q : ℕ), p + n + d + q = 13 ∧ 
                   1 * p + 5 * n + 10 * d + 25 * q = 141 ∧ 
                   2 ≤ p ∧ 2 ≤ n ∧ 2 ≤ d ∧ 2 ≤ q ∧ 
                   d = 3 :=
  sorry

end thirteen_coins_value_l655_655205


namespace find_RS_l655_655065

noncomputable def cos_inverse (x : ℝ) : ℝ := Real.arccos x

theorem find_RS :
  ∃ (RS : ℝ), 
  let d1 := 10,
      d2 := 10,
      ps := 12,
      qs := 4,
      cosPSQ := (ps ^ 2 + d1 ^ 2 - qs ^ 2) / (2 * ps * d1),
      anglePSQ := cos_inverse cosPSQ,
      angleQSR := Real.pi - anglePSQ,
      cosQSR := Real.cos angleQSR,
      RS := Real.sqrt (d2 ^ 2 - qs ^ 2 + 2 * qs * 4 * cosQSR)
  in abs (RS - 7.075) < 0.001 :=
by sorry

end find_RS_l655_655065


namespace total_animals_correct_l655_655893

def initial_cows : ℕ := 2
def initial_pigs : ℕ := 3
def initial_goats : ℕ := 6

def added_cows : ℕ := 3
def added_pigs : ℕ := 5
def added_goats : ℕ := 2

def total_cows : ℕ := initial_cows + added_cows
def total_pigs : ℕ := initial_pigs + added_pigs
def total_goats : ℕ := initial_goats + added_goats

def total_animals : ℕ := total_cows + total_pigs + total_goats

theorem total_animals_correct : total_animals = 21 := by
  sorry

end total_animals_correct_l655_655893


namespace find_a_plus_b_plus_c_l655_655547

noncomputable def parabola_satisfies_conditions (a b c : ℝ) : Prop :=
  (∀ x, a * x ^ 2 + b * x + c ≥ 61) ∧
  (a * (1:ℝ) ^ 2 + b * (1:ℝ) + c = 0) ∧
  (a * (3:ℝ) ^ 2 + b * (3:ℝ) + c = 0)

theorem find_a_plus_b_plus_c (a b c : ℝ) 
  (h_minimum : parabola_satisfies_conditions a b c) :
  a + b + c = 0 := 
sorry

end find_a_plus_b_plus_c_l655_655547


namespace num_teachers_in_Oxford_High_School_l655_655896

def classes : Nat := 15
def students_per_class : Nat := 20
def principals : Nat := 1
def total_people : Nat := 349

theorem num_teachers_in_Oxford_High_School : 
  ∃ (teachers : Nat), teachers = total_people - (classes * students_per_class + principals) :=
by
  use 48
  sorry

end num_teachers_in_Oxford_High_School_l655_655896


namespace correct_quotient_divide_8_l655_655643

theorem correct_quotient_divide_8 (N : ℕ) (Q : ℕ) 
  (h1 : N = 7 * 12 + 5) 
  (h2 : N / 8 = Q) : 
  Q = 11 := 
by
  sorry

end correct_quotient_divide_8_l655_655643


namespace proof_problem_l655_655388

variables {a : ℕ → ℝ} {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a n = a 0 * q^n
def product_first_n_terms (a : ℕ → ℝ) (n : ℕ) := ∏ i in (finset.range n), a i
def T (a : ℕ → ℝ) := λ n : ℕ, product_first_n_terms a n

-- Given conditions
variables (h_seq : geometric_sequence a q)
variables (h_q_pos : q > 0)
variables (h_ineq : T a 7 > T a 6 ∧ T a 6 > T a 8)

theorem proof_problem (h_seq : geometric_sequence a q) (h_q_pos : q > 0) (h_ineq : T a 7 > T a 6 ∧ T a 6 > T a 8) :
  (0 < q ∧ q < 1) ∧ (T a 13 > 1 ∧ 1 > T a 14) :=
sorry

end proof_problem_l655_655388


namespace triangle_AC_leq_1_AB_eq_c_A_eq_60_R_leq_1_implies_1_div_2_lt_c_lt_2_l655_655052

noncomputable def isAcuteAngledTriangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ A < 90 ∧ B < 90 ∧ C < 90

theorem triangle_AC_leq_1_AB_eq_c_A_eq_60_R_leq_1_implies_1_div_2_lt_c_lt_2
  (A B C : ℝ)
  (hAcute : isAcuteAngledTriangle A B C)
  (AC_eq_1 : AC = 1)
  (AB_eq_c : AB = c)
  (angle_A_eq_60 : ∠A = 60)
  (circumradius_le_1 : circumradius (triangle A B C) ≤ 1) :
  1 / 2 < c ∧ c < 2 := 
sorry

end triangle_AC_leq_1_AB_eq_c_A_eq_60_R_leq_1_implies_1_div_2_lt_c_lt_2_l655_655052


namespace actual_number_of_students_is_840_l655_655895

-- Define the number of students who were supposed to receive fruits
variable (total_students absent_students remaining_students : ℕ)
variable (bananas_per_student apples_per_student oranges_per_student : ℕ)
variable (extra_bananas_per_student extra_apples_per_student extra_oranges_per_student : ℕ)

-- Specific values from the conditions:
def total_students := 840
def absent_students := 420
def remaining_students := total_students - absent_students
def bananas_per_student := 2
def apples_per_student := 1
def oranges_per_student := 1
def extra_bananas_per_student := 4
def extra_apples_per_student := 2
def extra_oranges_per_student := 1

-- Total distributed
def total_bananas_supposed := bananas_per_student * total_students
def total_apples_supposed := apples_per_student * total_students
def total_oranges_supposed := oranges_per_student * total_students

-- Total actually distributed
def total_bananas_actual := bananas_per_student * total_students +
                             extra_bananas_per_student * remaining_students
def total_apples_actual := apples_per_student * total_students + 
                           extra_apples_per_student * remaining_students
def total_oranges_actual := oranges_per_student * total_students +
                            extra_oranges_per_student * remaining_students

theorem actual_number_of_students_is_840 :
  total_bananas_supposed = total_bananas_actual ∧ 
  total_apples_supposed = total_apples_actual ∧ 
  total_oranges_supposed = total_oranges_actual ∧ 
  total_students = 840 := 
by
  sorry

end actual_number_of_students_is_840_l655_655895


namespace circle_center_l655_655439

theorem circle_center (n : ℝ) (r : ℝ) (h1 : r = 7) (h2 : ∀ x : ℝ, x^2 + (x^2 - n)^2 = 49 → x^4 - x^2 * (2*n - 1) + n^2 - 49 = 0)
  (h3 : ∃! y : ℝ, y^2 + (1 - 2*n) * y + n^2 - 49 = 0) :
  (0, n) = (0, 197 / 4) := 
sorry

end circle_center_l655_655439


namespace sum_of_acute_angles_bounds_l655_655919

theorem sum_of_acute_angles_bounds (α β γ : ℝ) 
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (hγ : 0 < γ ∧ γ < π / 2)
  (h : sin α ^ 2 + sin β ^ 2 + sin γ ^ 2 = 1) :
  π / 2 < α + β + γ ∧ α + β + γ < 3 * π / 4 := 
sorry

end sum_of_acute_angles_bounds_l655_655919


namespace correct_propositions_count_l655_655757

-- Conditions (axioms)
axiom axiom1 : ∀ (A B C : Point), ¬ (collinear A B C) → determines_plane A B C 
axiom axiom2 : ∀ (P : Point) (L : Line), (¬ on_line P L) → (∃! (M : Line), (P ∈ M) ∧ (is_parallel M L))
axiom axiom3 : ∀ (α β : Plane), (∃ (A B C : Point), ¬ (collinear A B C) ∧ (is_equidistant_from_plane A β) ∧ (is_equidistant_from_plane B β) ∧ (is_equidistant_from_plane C β)) → (is_parallel α β)
axiom axiom4 : ∀ (a b c : Line), (perpendicular a b) ∧ (perpendicular a c) → (is_parallel b c)

-- Proposition count
theorem correct_propositions_count : 
  (number_of_correct_propositions = 1) :=
sorry

end correct_propositions_count_l655_655757


namespace initial_pencils_count_l655_655957

variables {pencils_taken : ℕ} {pencils_left : ℕ} {initial_pencils : ℕ}

theorem initial_pencils_count 
  (h1 : pencils_taken = 4)
  (h2 : pencils_left = 5) :
  initial_pencils = 9 :=
by 
  sorry

end initial_pencils_count_l655_655957


namespace solve_x_squared_plus_y_squared_eq_one_l655_655479

-- Define the necessary field and set up assumptions
variables (F : Type*) [Field F] (r x y : F)
noncomputable theory

-- Define the condition that characteristic of F is not 2
axiom not_char_2 : (1 : F) + 1 ≠ 0

-- Define the main theorem
theorem solve_x_squared_plus_y_squared_eq_one :
  {p : F × F | p.1^2 + p.2^2 = 1} = {(1, 0)} ∪ { (x, y) | 
  ∃ r, r^2 ≠ -1 ∧ x = (r^2 - 1) / (r^2 + 1) ∧ y = 2 * r / (r^2 + 1) } :=
sorry

end solve_x_squared_plus_y_squared_eq_one_l655_655479


namespace points_in_tetrahedron_l655_655506

theorem points_in_tetrahedron (p : Fin 9 → EuclideanGeometry.Point (Fin 3)) :
  ∃ i j : Fin 9, i ≠ j ∧ EuclideanGeometry.dist (p i) (p j) ≤ 0.5 :=
by
  -- Define the tetrahedron with edge length 1 cm
  let edge_length := 1
  let vertices := [
    EuclideanGeometry.mkPoint 0 0 0,
    EuclideanGeometry.mkPoint edge_length 0 0,
    EuclideanGeometry.mkPoint (edge_length / 2) (edge_length * real.sqrt 3 / 2) 0,
    EuclideanGeometry.mkPoint (edge_length / 2) (edge_length * real.sqrt 3 / 6) (edge_length * real.sqrt 6 / 3)
  ]
  
  -- Assume 9 points are placed on the surface of the tetrahedron
  assume (h : ∀ i, EuclideanGeometry.isOnSurface (p i) vertices),
  sorry

end points_in_tetrahedron_l655_655506


namespace multiply_square_expression_l655_655665

theorem multiply_square_expression (x : ℝ) : ((-3 * x) ^ 2) * (2 * x) = 18 * x ^ 3 := by
  sorry

end multiply_square_expression_l655_655665


namespace units_digit_of_3_7_13_product_l655_655168
noncomputable theory

def pow_mod (base : ℕ) (exp : ℕ) (modulus : ℕ) : ℕ :=
  base ^ exp % modulus

theorem units_digit_of_3_7_13_product : 
  pow_mod 3 1001 10 * pow_mod 7 1002 10 * pow_mod 13 1003 10 % 10 = 9 :=
by
  sorry

end units_digit_of_3_7_13_product_l655_655168


namespace min_max_values_of_f_l655_655156

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * (sin x)^2 + 2 * cos x

theorem min_max_values_of_f :
  ∃ (x_min x_max : ℝ), (∀ x, f x ≥ -3/2) ∧ (∀ x, f x ≤ 3) ∧ (∃ x, f x = -3/2) ∧ (∃ x, f x = 3) := by
  sorry

end min_max_values_of_f_l655_655156


namespace number_of_dogs_in_shelter_l655_655437

variables (D C R P : ℕ)

-- Conditions
axiom h1 : 15 * C = 7 * D
axiom h2 : 9 * P = 5 * R
axiom h3 : 15 * (C + 8) = 11 * D
axiom h4 : 7 * P = 5 * (R + 6)

theorem number_of_dogs_in_shelter : D = 30 :=
by sorry

end number_of_dogs_in_shelter_l655_655437


namespace unknown_number_l655_655536

theorem unknown_number (x : ℤ) (h : (20 + 40 + 60) / 3 = (10 + 70 + x) / 3 + 9) : x = 13 :=
by {
  have : (20 + 40 + 60) / 3 = 40 := by norm_num,
  rw this at h,
  have : 40 = (80 + x) / 3 + 9 := h,
  linarith,
  have : 40 - 9 = (80 + x) / 3 := by linarith,
  have : 31 * 3 = 80 + x := by linarith,
  have : 93 = 80 + x := by norm_num at this,
  linarith,
}

end unknown_number_l655_655536


namespace maximal_intersection_area_of_rectangles_l655_655894

theorem maximal_intersection_area_of_rectangles :
  ∀ (a b : ℕ), a * b = 2015 ∧ a < b →
  ∀ (c d : ℕ), c * d = 2016 ∧ c > d →
  ∃ (max_area : ℕ), max_area = 1302 ∧ ∀ intersection_area, intersection_area ≤ 1302 := 
by
  sorry

end maximal_intersection_area_of_rectangles_l655_655894


namespace projection_calc_l655_655088

def vec1 : ℝ × ℝ × ℝ := (7, 2, -1)
def proj_vec1_Q : ℝ × ℝ × ℝ := (2, -1, 7)
def vec2 : ℝ × ℝ × ℝ := (6, -3, 9)
def result_proj_vec2_Q : ℝ × ℝ × ℝ := (309 / 49, -9 / 98, 400.5 / 49)

theorem projection_calc :
  let n := ⟨-5, -3, 8⟩ in
  let v := ⟨6, -3, 9⟩ in
  let p := v - (dot v n / dot n n) • n in
  p = ⟨309 / 49, -9 / 98, 400.5 / 49⟩ :=
by
  sorry

end projection_calc_l655_655088


namespace keychain_arrangement_count_l655_655824

-- Definitions of the keys
inductive Key
| house
| car
| office
| other1
| other2

-- Function to count the number of distinct arrangements on a keychain
noncomputable def distinct_keychain_arrangements : ℕ :=
  sorry -- This will be the placeholder for the proof

-- The ultimate theorem stating the solution
theorem keychain_arrangement_count : distinct_keychain_arrangements = 2 :=
  sorry -- This will be the placeholder for the proof

end keychain_arrangement_count_l655_655824


namespace bookstore_sales_l655_655250

theorem bookstore_sales (monday_sales tuesday_sales wednesday_sales thursday_sales friday_sales saturday_sales sunday_sales : ℕ) (h_monday : monday_sales = 25)
(h_tuesday : tuesday_sales = (monday_sales + ⌊10/100 * 25⌋)) 
(h_wednesday : wednesday_sales = (tuesday_sales + ⌊10/100 * tuesday_sales⌋))
(h_thursday_base : thursday_sales = (wednesday_sales + ⌊10/100 * wednesday_sales + ⌊20/100 * (wednesday_sales + ⌊10/100 * wednesday_sales⌋)⌋))
(h_friday_base_adjustment : friday_sales = ⌊(wednesday_sales + ⌊10/100 * wednesday_sales⌋) + ⌊10/100 * (wednesday_sales + ⌊10/100 * wednesday_sales⌋)⌋⌋)
(h_saturday_base : saturday_sales = ⌊2 * friday_sales + ⌊25/100 * 2 * friday_sales⌋⌋)
(h_sunday_base : sunday_sales = ⌊0.5 * saturday_sales + ⌊15/100 * 0.5 * saturday_sales⌋⌋):
monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales + saturday_sales + sunday_sales = 286 := yes sorry 

end bookstore_sales_l655_655250


namespace part_I_part_II_part_III_l655_655771

def f (x : ℝ) := 2 * sqrt 3 * (cos x)^2 + sin (2 * x) - sqrt 3 + 1

theorem part_I : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = π := 
sorry

theorem part_II (k : ℤ) : ∀ x ∈ Set.Icc (k * π - 5 * π / 12) (k * π + π / 12), 
∃ ϵ > 0, ∀ y ∈ Set.Icc (x - ϵ) (x + ϵ), f y > f x :=
sorry

theorem part_III : ∃ S, S = Set.Icc 0 3 ∧ 
∀ y ∈ Set.Icc (-π / 4) (π / 4), f y ∈ S :=
sorry

end part_I_part_II_part_III_l655_655771


namespace three_digit_N_exists_l655_655293

def valid_N_count : Nat :=
  let valid_N := {N : Nat // 100 ≤ N ∧ N < 1000 ∧ 
    let N4 := (N / 64, (N / 16 % 4), (N % 16 % 4))
    let N7 := (N / 49, (N / 7 % 7), (N % 7 % 7))
    let S := (N / 64 + N / 49) * 100 + (N / 16 % 4 + N / 7 % 7) * 10 + (N % 16 % 4 + N % 7 % 7)
    S % 100 = (3 * N) % 100 
  }
  valid_N.toFinset.card

theorem three_digit_N_exists (h : ∃ N : Nat, 100 ≤ N ∧ N < 1000 ∧ 
    let N4 := (N / 64, (N / 16 % 4), (N % 16 % 4))
    let N7 := (N / 49, (N / 7 % 7), (N % 7 % 7))
    let S := (N / 64 + N / 49) * 100 + (N / 16 % 4 + N / 7 % 7) * 10 + (N % 16 % 4 + N % 7 % 7)
    S % 100 = (3 * N) % 100 
  ) : valid_N_count = 64 := sorry

end three_digit_N_exists_l655_655293


namespace fraction_of_integer_l655_655182

theorem fraction_of_integer :
  (5 / 6) * 30 = 25 :=
by
  sorry

end fraction_of_integer_l655_655182


namespace number_with_three_odd_factors_l655_655561

theorem number_with_three_odd_factors :
  ∃ n : ℕ, (∀ d : ℕ, d ∣ n → d > 1 → odd d → ∃ k : ℕ, n = k * k) ∧
           (∃! (d : ℕ), d > 1 ∧ odd d ∧ d ∣ n ∧ (∃! m: ℕ, d = m * m)) → n = 9 :=
sorry

end number_with_three_odd_factors_l655_655561


namespace rectangle_is_square_l655_655554

-- Define structures for rectangle and partitions
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)
  (area : ℝ)

theorem rectangle_is_square
  (ABCD_partitions : List Rectangle)
  (P5_is_square : ∃ (P5 : Rectangle), P5 ∈ ABCD_partitions ∧ P5.length = P5.width)
  (equal_areas : ∀ (P : Rectangle), P ∈ ABCD_partitions.erase P5 → P.area = 1) :
  ABCD.length = ABCD.width := 
sorry

end rectangle_is_square_l655_655554


namespace largest_reciprocal_l655_655202

def has_largest_reciprocal (x : ℚ) (s : set ℚ) : Prop :=
  x ∈ s ∧ ∀ y ∈ s, x ≤ y

theorem largest_reciprocal : 
  has_largest_reciprocal (4 : ℚ) 
    ({1 / 4, 3 / 8, 1 / 2, 4, 1000} : set ℚ) :=
sorry

end largest_reciprocal_l655_655202


namespace marketing_strategy_increases_mid_sales_l655_655452

structure Product :=
  (name : String)
  (quality : ℕ)
  (price : ℕ)

inductive Category
| premium | mid | economy

structure Store :=
  (products : List Product)
  (category : Product → Category)

def is_premium (p : Product) : Prop := p.name = "A" ∧ ∃ s : Store, s.category p = Category.premium
def is_mid (p : Product) : Prop := p.name = "B" ∧ ∃ s : Store, s.category p = Category.mid
def is_economy (p : Product) : Prop := p.name = "C" ∧ ∃ s : Store, s.category p = Category.economy

theorem marketing_strategy_increases_mid_sales
  (s : Store)
  (hA : ∃ pA : Product, is_premium pA ∧ (pA ∈ s.products) ∧ (pA.price > 0) ∧ (∃ q, q > pA.quality))
  (hB : ∃ pB : Product, is_mid pB ∧ (pB ∈ s.products) ∧ (pB.price < pA.price) ∧ (pB.price > 0))
  (hC : ∃ pC : Product, is_economy pC ∧ (pC ∈ s.products) ∧ (pC.price < pB.price) ∧ (pC.price > 0)) :
  (∃ pB : Product, is_mid pB ∧ pB ∈ s.products → sales pB > 0) :=
sorry

end marketing_strategy_increases_mid_sales_l655_655452


namespace infinite_marked_nodes_l655_655820

def is_marked (S : set (ℤ × ℤ)) (p : ℤ × ℤ) : Prop := p ∈ S

def translates_to_more_marked 
  (S : set (ℤ × ℤ)) 
  (V : set (ℤ × ℤ)) 
  (p : ℤ × ℤ) : Prop := 
  ∃ q ∈ S, ∃ v ∈ V, (q = (p.1 + v.1, p.2 + v.2))

theorem infinite_marked_nodes 
  (S : set (ℤ × ℤ)) 
  (V : set (ℤ × ℤ)) 
  (h_nonempty : set.nonempty S) 
  (h_finite_vecs : set.finite V)
  (h_condition : ∀ p ∈ S, translates_to_more_marked S V p) :
  S.infinite := 
begin
  sorry
end

end infinite_marked_nodes_l655_655820


namespace geom_seq_log_val_l655_655465

theorem geom_seq_log_val (a : ℕ → ℝ) (f : ℝ → ℝ)
  (h_geometric : ∀ n, a (n + 1) = a n * a 1)
  (h_positive : ∀ n, a n > 0)
  (h_extremum1 : f (a 1) = 0)
  (h_extremum4031 : f (a 4031) = 0)
  (h_f_derivative : ∀ x, deriv f x = x^2 - 8 * x + 6) :
  log (sqrt 6) (a 2016) = 1 := sorry

end geom_seq_log_val_l655_655465


namespace parametric_equation_of_curve_C_max_area_of_OAPB_l655_655057

theorem parametric_equation_of_curve_C (ρ θ : ℝ) :
  (ρ^2 * (1 + 3 * sin(θ)^2) = 4) ↔ ∃ α : ℝ, (ρ = 2 * cos α ∧ ρ * sin θ = sin α) :=
by sorry

theorem max_area_of_OAPB {α β γ δ : ℝ}
  (h1 : ∀ φ : ℝ, 0 < φ ∧ φ < π/2 → max_area O A P B = (cos φ + sin φ))
  (h2 : ∀ φ : ℝ, max_area(O A P B) = sqrt(2)) :
  max_area(O A P B) = sqrt(2) :=
by sorry

end parametric_equation_of_curve_C_max_area_of_OAPB_l655_655057


namespace elisa_current_dollars_l655_655328

variable (current_dollars : ℕ)
variable (earn_needed : ℕ := 16)
variable (total_dollars : ℕ := 53)

theorem elisa_current_dollars : current_dollars + earn_needed = total_dollars → current_dollars = 37 :=
by
  intro h
  have : current_dollars = total_dollars - earn_needed :=
    by linarith
  exact this.symm

end elisa_current_dollars_l655_655328


namespace bill_due_months_l655_655947

theorem bill_due_months {TD A: ℝ} (R: ℝ) : 
  TD = 189 → A = 1764 → R = 16 → 
  ∃ M: ℕ, A - TD * (1 + (R/100) * (M/12)) = 1764 - 189 * (1 + (16/100) * (10/12)) ∧ M = 10 :=
by
  intro hTD hA hR
  use 10
  sorry

end bill_due_months_l655_655947


namespace total_amount_with_tax_is_38_18_l655_655578

def price_football_game : ℝ := 14.02
def price_strategy_game : ℝ := 9.46
def price_batman_game : ℝ := 12.04
def sales_tax_rate : ℝ := 0.075

def total_before_tax : ℝ := price_football_game + price_strategy_game + price_batman_game
def sales_tax : ℝ := total_before_tax * sales_tax_rate
def total_amount_spent : ℝ := total_before_tax + sales_tax
def total_amount_spent_rounded : ℝ := Real.round (total_amount_spent * 100) / 100

theorem total_amount_with_tax_is_38_18 : total_amount_spent_rounded = 38.18 :=
  by
    -- Proof omitted
    sorry

end total_amount_with_tax_is_38_18_l655_655578


namespace sequence_equalized_l655_655368

theorem sequence_equalized (n : ℕ) (a : Fin n → ℤ) 
  (h : ∀ (k : ℕ), ∃ b : Fin n → ℤ, 
    (∀ i : Fin n, b i = (a i + a (⟨i + 1 % n, by linarith⟩)) / 2)) :
  ∃ c : ℤ, ∀ i : Fin n, b i = c :=
by
  sorry

end sequence_equalized_l655_655368


namespace tennis_handshakes_l655_655276

-- Define the participants and the condition
def number_of_participants := 8
def handshakes_no_partner (n : ℕ) := n * (n - 2) / 2

-- Prove that the number of handshakes is 24
theorem tennis_handshakes : handshakes_no_partner number_of_participants = 24 := by
  -- Since we are skipping the proof for now
  sorry

end tennis_handshakes_l655_655276


namespace students_table_tennis_not_basketball_l655_655956

variable (total_students : ℕ)
variable (students_like_basketball : ℕ)
variable (students_like_table_tennis : ℕ)
variable (students_dislike_both : ℕ)

theorem students_table_tennis_not_basketball 
  (h_total : total_students = 40)
  (h_basketball : students_like_basketball = 17)
  (h_table_tennis : students_like_table_tennis = 20)
  (h_dislike : students_dislike_both = 8) : 
  ∃ (students_table_tennis_not_basketball : ℕ), students_table_tennis_not_basketball = 15 :=
by
  sorry

end students_table_tennis_not_basketball_l655_655956


namespace sufficient_conditions_for_sum_positive_l655_655756

variable {a b : ℝ}

theorem sufficient_conditions_for_sum_positive (h₃ : a + b > 2) (h₄ : a > 0 ∧ b > 0) : a + b > 0 :=
by {
  sorry
}

end sufficient_conditions_for_sum_positive_l655_655756


namespace positive_integer_solutions_count_l655_655144

theorem positive_integer_solutions_count : 
  (∃! (n : ℕ), n > 0 ∧ 25 - 5 * n > 15) :=
sorry

end positive_integer_solutions_count_l655_655144


namespace blue_pens_count_l655_655257

variable (redPenCost bluePenCost totalCost totalPens : ℕ)
variable (numRedPens numBluePens : ℕ)

-- Conditions
axiom PriceOfRedPen : redPenCost = 5
axiom PriceOfBluePen : bluePenCost = 7
axiom TotalCost : totalCost = 102
axiom TotalPens : totalPens = 16
axiom PenCount : numRedPens + numBluePens = totalPens
axiom CostEquation : redPenCost * numRedPens + bluePenCost * numBluePens = totalCost

theorem blue_pens_count : numBluePens = 11 :=
by
  sorry

end blue_pens_count_l655_655257


namespace digit_in_421st_place_l655_655997

noncomputable def repeating_decimal : ℕ → ℕ
| 0      := 2
| 1      := 4
| 2      := 1
| 3      := 3
| 4      := 7
| 5      := 9
| 6      := 3
| 7      := 1
| 8      := 0
| 9      := 3
| 10     := 4
| 11     := 4
| 12     := 8
| 13     := 2
| 14     := 7
| 15     := 5
| 16     := 8
| 17     := 6
| 18     := 2
| 19     := 0
| 20     := 6
| 21     := 8
| 22     := 9
| 23     := 6
| 24     := 5
| 25     := 5
| 26     := 1
| 27     := 7
| n + 28 := repeating_decimal n  -- Repeat every 29 digits

theorem digit_in_421st_place :
  repeating_decimal (421 % 29) = 5 :=
by {
  -- prove that the 421 mod 29 is 26
  calc 421 % 29 = 26 : by norm_num,
  -- prove the 26th digit is 5
  show repeating_decimal 26 = 5, by norm_num
}

-- End with sorry since we are not providing the actual proof details.
sorry

end digit_in_421st_place_l655_655997


namespace second_smallest_packs_l655_655327

-- Define the number of hot dogs per pack
def hotdogs_per_pack := 12

-- Define the number of buns per pack
def buns_per_pack := 10

-- Define the number of leftover hot dogs
def leftover_hotdogs := 6

-- Define the relationship between the packs of buns and hotdogs
def buns_to_hotdogs_ratio (n m : ℕ) : Prop := m = 2 * n

-- Define the congruence condition
def congruence_condition (n : ℕ) : Prop := 12 * n % 10 = 6

-- Theorem stating the second smallest number of packs of hot dogs Phil could have bought
theorem second_smallest_packs (n : ℕ) (m : ℕ) 
  (h1 : buns_to_hotdogs_ratio n m)
  (h2 : congruence_condition n) : 
  (n = 8) :=
begin
  sorry
end

end second_smallest_packs_l655_655327


namespace cricket_bat_cost_price_A_l655_655648

noncomputable def cost_price_A (sale_price_D : ℝ) (profit_AB : ℝ) (profit_BC : ℝ) (profit_CD : ℝ) : ℝ :=
  sale_price_D / (1 + profit_AB) / (1 + profit_BC) / (1 + profit_CD)

theorem cricket_bat_cost_price_A :
  let sale_price_D := 420.90
  let profit_AB := 0.20
  let profit_BC := 0.25
  let profit_CD := 0.30
  cost_price_A sale_price_D profit_AB profit_BC profit_CD = 216.00 :=
by
  let sale_price_D := 420.90
  let profit_AB := 0.20
  let profit_BC := 0.25
  let profit_CD := 0.30
  show cost_price_A sale_price_D profit_AB profit_BC profit_CD = 216.00
  sorry

end cricket_bat_cost_price_A_l655_655648


namespace correct_statement_l655_655353

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + (Real.pi / 2))
noncomputable def g (x : ℝ) : ℝ := Real.cos (x + (3 * Real.pi / 2))

theorem correct_statement (x : ℝ) : f (x - (Real.pi / 2)) = g x :=
by sorry

end correct_statement_l655_655353


namespace inequality_correctness_l655_655022

theorem inequality_correctness (a b : ℝ) (h : a < b) (h₀ : b < 0) : - (1 / a) < - (1 / b) :=
sorry

end inequality_correctness_l655_655022


namespace find_LCM_of_three_numbers_l655_655035

noncomputable def LCM_of_three_numbers (a b c : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm a b) c

theorem find_LCM_of_three_numbers
  (a b c : ℕ)
  (h_prod : a * b * c = 1354808)
  (h_gcd : Nat.gcd (Nat.gcd a b) c = 11) :
  LCM_of_three_numbers a b c = 123164 := by
  sorry

end find_LCM_of_three_numbers_l655_655035


namespace smallest_number_condition_l655_655992

def smallest_number := 1621432330
def primes := [29, 53, 37, 41, 47, 61]
def lcm_of_primes := primes.prod

theorem smallest_number_condition :
  ∃ k : ℕ, 5 * (smallest_number + 11) = k * lcm_of_primes ∧
          (∀ y, (∃ m : ℕ, 5 * (y + 11) = m * lcm_of_primes) → smallest_number ≤ y) :=
by
  -- The proof goes here
  sorry

#print smallest_number_condition

end smallest_number_condition_l655_655992


namespace octal_to_base12_conversion_l655_655683

-- Define the computation functions required
def octalToDecimal (n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  d2 * 64 + d1 * 8 + d0

def decimalToBase12 (n : ℕ) : List ℕ :=
  let d0 := n % 12
  let n1 := n / 12
  let d1 := n1 % 12
  let n2 := n1 / 12
  let d2 := n2 % 12
  [d2, d1, d0]

-- The main theorem that combines both conversions
theorem octal_to_base12_conversion :
  decimalToBase12 (octalToDecimal 563) = [2, 6, 11] :=
sorry

end octal_to_base12_conversion_l655_655683


namespace a_2_geometric_sequence_l655_655433

theorem a_2_geometric_sequence (a : ℝ) (n : ℕ) (S : ℕ → ℝ)
  (h1 : ∀ n ≥ 2, S n = a * 3^n - 2) : S 2 = 12 :=
by 
  sorry

end a_2_geometric_sequence_l655_655433


namespace benzene_needed_for_methane_l655_655333

-- Define the objects and conditions
def methane : Type := unit
def benzene : Type := unit
def toluene : Type := unit
def hydrogen : Type := unit

def reaction (m_count b_count t_count h_count : ℕ) : Prop :=
  ∃ (m b t h : ℕ),
  m_count = m ∧
  b_count = b ∧
  t_count = t ∧
  h_count = h ∧
  m = 1 ∧
  b = 1 ∧
  t = 1 ∧
  h = 1

-- The main statement
theorem benzene_needed_for_methane :
  ∀ (m_count : ℕ), (m_count = 2) → ∃ (b_count : ℕ), (b_count = 2) ∧ reaction m_count b_count 2 2 :=
by
  intro m_count hm_count
  use 2
  sorry

end benzene_needed_for_methane_l655_655333


namespace solve_m_range_l655_655373

-- Define the propositions
def p (m : ℝ) := m + 1 ≤ 0

def q (m : ℝ) := ∀ x : ℝ, x^2 + m * x + 1 > 0

-- Provide the Lean statement for the problem
theorem solve_m_range (m : ℝ) (hpq_false : ¬ (p m ∧ q m)) (hpq_true : p m ∨ q m) :
  m ≤ -2 ∨ (-1 < m ∧ m < 2) :=
sorry

end solve_m_range_l655_655373


namespace solve_system_of_equations_solve_fractional_equation_l655_655528

noncomputable def solution1 (x y : ℚ) := (3 * x - 5 * y = 3) ∧ (x / 2 - y / 3 = 1) ∧ (x = 8 / 3) ∧ (y = 1)

noncomputable def solution2 (x : ℚ) := (x / (x - 1) + 1 = 3 / (2 * x - 2)) ∧ (x = 5 / 4)

theorem solve_system_of_equations (x y : ℚ) : solution1 x y := by
  sorry

theorem solve_fractional_equation (x : ℚ) : solution2 x := by
  sorry

end solve_system_of_equations_solve_fractional_equation_l655_655528


namespace cube_faces_numbering_impossible_l655_655297

theorem cube_faces_numbering_impossible :
  ¬(∃ (f : fin 6 → ℕ),
      (∀ i : fin 6, f i ∈ {1, 2, 3, 4, 5, 6}) ∧
      (∀ i j : fin 6, i ≠ j → f i ≠ f j) ∧
      (∀ i : fin 6, (∑ j in adjacent_faces i, f j) % f i = 0)) := 
sorry

end cube_faces_numbering_impossible_l655_655297


namespace area_of_trapezium_l655_655608

-- Definitions
def length_parallel_side_1 : ℝ := 4
def length_parallel_side_2 : ℝ := 5
def perpendicular_distance : ℝ := 6

-- Statement
theorem area_of_trapezium :
  (1 / 2) * (length_parallel_side_1 + length_parallel_side_2) * perpendicular_distance = 27 :=
by
  sorry

end area_of_trapezium_l655_655608


namespace pencil_price_in_units_l655_655424

noncomputable def price_of_pencil_in_units (base_price additional_price unit_size : ℕ) : ℝ :=
  (base_price + additional_price) / unit_size

theorem pencil_price_in_units :
  price_of_pencil_in_units 5000 200 10000 = 0.52 := 
  by 
  sorry

end pencil_price_in_units_l655_655424


namespace inverse_of_exponential_function_l655_655151

theorem inverse_of_exponential_function :
  (∀ (x : ℝ), x > 0 → 2^(1/x) = y) →
  ∃ (y : ℝ), y = 1 / (Real.log2 x) ∧ x > 1 :=
begin
  sorry
end

end inverse_of_exponential_function_l655_655151


namespace distribution_problem_l655_655635

-- Defining the problem conditions
def employees : Type := Fin 8
def departments : Type := Fin 2
def translators : Finset employees := {0, 1}  -- two English translators
def programmers : Finset employees := {2, 3, 4}  -- three computer programmers

-- A function that assigns each employee to a department
def assignment (e : employees) : departments

-- Conditions
def distinct_translators (a : assignment) : Prop :=
  a 0 ≠ a 1

def not_all_programmers_same_dept (a : assignment) : Prop :=
  ¬ ((a 2 = a 3) ∧ (a 3 = a 4))

-- Problem statement
theorem distribution_problem :
  ∃ (a : employees → departments), distinct_translators a ∧ not_all_programmers_same_dept a ∧ (number_of_valid_assignments a = 36) :=
by
  sorry

end distribution_problem_l655_655635


namespace smallest_four_digit_equivalent_6_mod_7_l655_655993

theorem smallest_four_digit_equivalent_6_mod_7 :
  (∃ (n : ℕ), n >= 1000 ∧ n < 10000 ∧ n % 7 = 6 ∧ (∀ (m : ℕ), m >= 1000 ∧ m < 10000 ∧ m % 7 = 6 → m >= n)) ∧ ∃ (n : ℕ), n = 1000 :=
sorry

end smallest_four_digit_equivalent_6_mod_7_l655_655993


namespace solve_matrix_equation_l655_655913

def A : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![1, 2],
  ![3, 4]
]

def b : Fin 2 → ℚ := ![7, 17]

def x_solution : Fin 2 → ℚ := ![3, 2]

theorem solve_matrix_equation : A.mul_vec x_solution = b := by
  sorry

end solve_matrix_equation_l655_655913


namespace ramesh_wants_to_earn_profit_l655_655126

noncomputable def labelled_price_after_discount : ℝ := 14500
noncomputable def discount_percentage : ℝ := 20
noncomputable def transport_cost : ℝ := 125
noncomputable def installation_cost : ℝ := 250
noncomputable def target_selling_price : ℝ := 20350

def percentage_profit (cost_price selling_price : ℝ) : ℝ :=
  ((selling_price - cost_price) / cost_price) * 100

theorem ramesh_wants_to_earn_profit :
  let labelled_price := labelled_price_after_discount / (1 - discount_percentage / 100)
  let total_cost := labelled_price_after_discount + transport_cost + installation_cost
  percentage_profit total_cost target_selling_price ≈ 36.81 := sorry

end ramesh_wants_to_earn_profit_l655_655126


namespace locus_of_centers_of_tangent_circles_l655_655694

structure Semicircle (O : Point) (R : ℝ) :=
(center : Point)
(radius : ℝ)
(diameter : Line)

structure TangentCircle (K : Point) (r : ℝ) :=
(center : Point)
(radius : ℝ)
(tangent_to_semi : Point → Prop)
(tangent_to_diameter : Line → Prop)

theorem locus_of_centers_of_tangent_circles (O : Point) (R : ℝ) (AB : Line) (K : Point) (r : ℝ) (E F : Point)
  (semi : Semicircle O R)
  (circle : TangentCircle K r)
  (tangent_at_E : circle.tangent_to_semi E)
  (tangent_at_F : circle.tangent_to_diameter AB)
  (collinear_OEK : collinear O E K)
  (perpendicular_KF_AB : ∠ KF AB = π / 2) :
  ∀ (K : Point), locus K ↔ distance_to_focus_equal_to_distance_to_directrix K :=
sorry

end locus_of_centers_of_tangent_circles_l655_655694


namespace relationship_a_b_l655_655363

noncomputable def f : ℝ → ℝ := sorry -- Specify a function f
noncomputable def f' (x : ℝ) : ℝ := sorry -- Specify its derivative

-- We assume f and f' satisfy the given condition for all x
axiom f_condition : ∀ x, f(x) + x * f'(x) > 0

def a : ℝ := (f 1) / 2
def b : ℝ := f 2

theorem relationship_a_b : 0 < a ∧ a < b := 
sorry -- Provide the proof here

end relationship_a_b_l655_655363


namespace symmetric_point_correct_l655_655064

noncomputable def symmetric_point (M : ℝ × ℝ) (theta : ℝ) : ℝ × ℝ :=
  let (r, alpha) := M
  (r, 2 * theta - alpha)

-- Define the given values
def M := (3, Real.pi / 2)
def theta := Real.pi / 6
def expected_symmetric_point := (3, -Real.pi / 6)

-- The proof problem statement
theorem symmetric_point_correct :
  symmetric_point M theta = expected_symmetric_point :=
  by
    sorry

end symmetric_point_correct_l655_655064


namespace ratio_BD_divides_AC_l655_655146

noncomputable def triangle_ratio (A B C H D : Point) 
(angle_A_D : real := 60) (angle_C_D : real := 90) 
(altitude_BH : Altitude BH A B C) 
(intersect_extension : IntersectExtAltitudeCirc BH D) 
(circum_circle : Circle CircABC) : Prop := 
(Divides BD AC (sqrt 3) 1)

theorem ratio_BD_divides_AC {A B C H D : Point} 
(angle_A_D : angle = 60) 
(angle_C_D : angle = 90) 
(altitude_BH : Altitude BH A B C) 
(intersect_extension : IntersectExtAltitudeCirc BH D) 
(circum_circle : CircABC) : 
(triangle_ratio A B C H D) := 
by sorry

end ratio_BD_divides_AC_l655_655146


namespace additional_pencils_l655_655329

theorem additional_pencils (original_pencils new_pencils per_container distributed_pencils : ℕ)
  (h1 : original_pencils = 150)
  (h2 : per_container = 5)
  (h3 : distributed_pencils = 36)
  (h4 : new_pencils = distributed_pencils * per_container) :
  (new_pencils - original_pencils) = 30 :=
by
  -- Proof will go here
  sorry

end additional_pencils_l655_655329


namespace steve_fraction_of_day_in_school_l655_655133

theorem steve_fraction_of_day_in_school :
  let total_hours : ℕ := 24
  let sleep_fraction : ℚ := 1 / 3
  let assignment_fraction : ℚ := 1 / 12
  let family_hours : ℕ := 10
  let sleep_hours : ℚ := sleep_fraction * total_hours
  let assignment_hours : ℚ := assignment_fraction * total_hours
  let accounted_hours : ℚ := sleep_hours + assignment_hours + family_hours
  let school_hours : ℚ := total_hours - accounted_hours
  (school_hours / total_hours) = (1 / 6) :=
by
  let total_hours : ℕ := 24
  let sleep_fraction : ℚ := 1 / 3
  let assignment_fraction : ℚ := 1 / 12
  let family_hours : ℕ := 10
  let sleep_hours : ℚ := sleep_fraction * total_hours
  let assignment_hours : ℚ := assignment_fraction * total_hours
  let accounted_hours : ℚ := sleep_hours + assignment_hours + family_hours
  let school_hours : ℚ := total_hours - accounted_hours
  have : (school_hours / total_hours) = (1 / 6) := sorry
  exact this

end steve_fraction_of_day_in_school_l655_655133


namespace find_y_l655_655695

noncomputable def y :=  2^(21/4)

theorem find_y (y : ℝ) (h : real.root 7 (y * real.root 3 (y ^ 5)) = 4): 
  y = 2^(21/4) :=
sorry

end find_y_l655_655695


namespace range_of_expression_l655_655032

theorem range_of_expression {b c : ℝ} 
  (hb : ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < 1 ∧ f x1 = 0 ∧ f x2 = 0)
  (f : ℝ → ℝ := λ x, x^2 + b * x + c) :
  0 < (1 + b) * c + c^2 ∧ (1 + b) * c + c^2 < 1 / 16 :=
sorry

end range_of_expression_l655_655032


namespace desert_area_2020_correct_desert_area_less_8_10_5_by_2023_l655_655290

-- Define initial desert area
def initial_desert_area : ℝ := 9 * 10^5

-- Define increase in desert area each year as observed
def yearly_increase (n : ℕ) : ℝ :=
  match n with
  | 1998 => 2000
  | 1999 => 4000
  | 2000 => 6001
  | 2001 => 7999
  | 2002 => 10001
  | _    => 0

-- Define arithmetic progression of increases
def common_difference : ℝ := 2000

-- Define desert area in 2020
def desert_area_2020 : ℝ :=
  initial_desert_area + 10001 + 18 * common_difference

-- Statement: Desert area by the end of 2020 is approximately 9.46 * 10^5 hm^2
theorem desert_area_2020_correct :
  desert_area_2020 = 9.46 * 10^5 :=
sorry

-- Define yearly transformation and desert increment with afforestation from 2003
def desert_area_with_afforestation (n : ℕ) : ℝ :=
  if n < 2003 then
    initial_desert_area + yearly_increase n
  else
    initial_desert_area + 10001 + (n - 2002) * (common_difference - 8000)

-- Statement: Desert area will be less than 8 * 10^5 hm^2 by the end of 2023
theorem desert_area_less_8_10_5_by_2023 :
  desert_area_with_afforestation 2023 < 8 * 10^5 :=
sorry

end desert_area_2020_correct_desert_area_less_8_10_5_by_2023_l655_655290


namespace cubic_polynomial_with_cubed_roots_l655_655941

def find_cubic_roots (a b c α β γ : ℝ) :=
  (α, β, γ) = (roots_of_cubic (a b c))

theorem cubic_polynomial_with_cubed_roots (a b c α β γ : ℝ)
  (h1 : α + β + γ = -a)
  (h2 : α * β + β * γ + γ * α = b)
  (h3 : α * β * γ = -c) :
  polynomial = polynomial.fillomorph_root (roots_of_cubic
    (a b c) (cubed_roots_of_cubic (a b c))) := 
begin
  sorry
end

end cubic_polynomial_with_cubed_roots_l655_655941


namespace customers_tried_sample_l655_655649

theorem customers_tried_sample
  {boxes_opened : ℕ} {samples_per_box : ℕ} {samples_left_over : ℕ} 
  (h1 : boxes_opened = 12) (h2 : samples_per_box = 20) (h3 : samples_left_over = 5) : 
  boxes_opened * samples_per_box - samples_left_over = 235 :=
by
  have total_samples := boxes_opened * samples_per_box
  have samples_given_out := total_samples - samples_left_over
  calc
  samples_given_out = 235 : by rw [h1, h2, h3]; exact rfl

end customers_tried_sample_l655_655649


namespace tensor_value_l655_655350

-- Define the tensor operation
def tensor (a b : ℝ) (cos_theta : ℝ) : ℝ :=
  a / b * cos_theta

-- Conditions
variables (a b : ℝ) (theta : ℝ) (n m : ℕ)
hypothesis ha_ge_hb : a ≥ b
hypothesis htheta_range : 0 < theta ∧ theta < π / 4
hypothesis haotimes : tensor a b (Real.cos theta) = n / 2
hypothesis hbotimes : tensor b a (Real.cos theta) = m / 2

-- Prove the value of a ⊗ b
theorem tensor_value : tensor a b (Real.cos theta) = 3 / 2 :=
  by
  -- proof here
  sorry

end tensor_value_l655_655350


namespace min_value_of_sum_of_squares_l655_655868

open Real

theorem min_value_of_sum_of_squares (i j k l m n o p : ℝ) 
  (h1 : i * j * k * l = 16) (h2 : m * n * o * p = 25) :
  (im^2 + jn^2 + ko^2 + lp^2 ≥ 160) := sorry

end min_value_of_sum_of_squares_l655_655868


namespace possible_start_cities_l655_655888

-- Define the cities
inductive City
| SaintPetersburg
| Tver
| Yaroslavl
| NizhnyNovgorod
| Moscow
| Kazan

open City

-- Define an edge as a pair of cities representing a bidirectional ticket
def Edge := City × City

-- Define the tickets as edges
def tickets : List Edge :=
  [(SaintPetersburg, Tver), (Tver, SaintPetersburg),
   (Yaroslavl, NizhnyNovgorod), (NizhnyNovgorod, Yaroslavl),
   (Moscow, Kazan), (Kazan, Moscow),
   (NizhnyNovgorod, Kazan), (Kazan, NizhnyNovgorod),
   (Moscow, Tver), (Tver, Moscow),
   (Moscow, NizhnyNovgorod), (NizhnyNovgorod, Moscow)]

-- Define a function to test if a journey can start in a given city
def can_start (c : City) : Prop :=
  ∀ (path : List City), 
    path.head = some c →
    (∀ p ∈ path, p ∈ [SaintPetersburg, Tver, Yaroslavl, NizhnyNovgorod, Moscow, Kazan]) →
    path.nodup →
    (path.tail.tail.length = 5) →
    (∀ {a b}, (a,b) ∈ List.zip path (path.tail) → (a,b) ∈ tickets) →
    true

-- Main theorem to be proven
theorem possible_start_cities :
  can_start SaintPetersburg ∨ can_start Yaroslavl :=
sorry

end possible_start_cities_l655_655888


namespace original_numbers_geometric_sequence_l655_655962

theorem original_numbers_geometric_sequence (a q : ℝ) :
  (2 * (a * q + 8) = a + a * q^2) →
  ((a * q + 8) ^ 2 = a * (a * q^2 + 64)) →
  (a, a * q, a * q^2) = (4, 12, 36) ∨ (a, a * q, a * q^2) = (4 / 9, -20 / 9, 100 / 9) :=
by {
  sorry
}

end original_numbers_geometric_sequence_l655_655962


namespace net_increase_stock_wealth_l655_655626

noncomputable def calc_final_investment (P : ℝ) : ℝ :=
  let after_first_year := P * 1.80 in
  let after_second_year := after_first_year * 0.70 in
  after_second_year

theorem net_increase_stock_wealth (P : ℝ) : calc_final_investment P - P = 0.26 * P :=
by
  sorry

end net_increase_stock_wealth_l655_655626


namespace arithmetic_sequence_a_m_n_zero_l655_655053

theorem arithmetic_sequence_a_m_n_zero
  (a : ℕ → ℕ)
  (m n : ℕ) 
  (hm : m > 0) (hn : n > 0)
  (h_ma_m : a m = n)
  (h_na_n : a n = m) : 
  a (m + n) = 0 :=
by 
  sorry

end arithmetic_sequence_a_m_n_zero_l655_655053


namespace quadratic_function_expression_quadratic_function_inequality_l655_655740

noncomputable def f (x : ℝ) : ℝ := x^2 - x + 1

theorem quadratic_function_expression (a b c : ℝ) (h₀ : a ≠ 0) 
  (h₁ : ∀ x : ℝ, f (x + 1) - f x = 2 * x) 
  (h₂ : f 0 = 1) : 
  (f x = x^2 - x + 1) := 
by {
  sorry
}

theorem quadratic_function_inequality (m : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x > 2 * x + m) ↔ m < -1 := 
by {
  sorry
}

end quadratic_function_expression_quadratic_function_inequality_l655_655740


namespace number_of_multiples_143_l655_655772

theorem number_of_multiples_143
  (h1 : 143 = 11 * 13)
  (h2 : ∀ i j : ℕ, 10^j - 10^i = 10^i * (10^(j-i) - 1))
  (h3 : ∀ i : ℕ, gcd (10^i) 143 = 1)
  (h4 : ∀ k : ℕ, 143 ∣ 10^k - 1 ↔ k % 6 = 0)
  (h5 : ∀ i j : ℕ, 0 ≤ i ∧ i < j ∧ j ≤ 99)
  : ∃ n : ℕ, n = 784 :=
by
  sorry

end number_of_multiples_143_l655_655772


namespace sum_of_reciprocals_of_sin_squared_sum_of_cot_squared_sum_of_reciprocals_of_squares_l655_655494

open Real
open BigOperators

theorem sum_of_reciprocals_of_sin_squared {m : ℕ} :
  ∑ k in Finset.range m + 1, (1 / (sin ((k : ℝ) * π / (2 * (m + 1) + 1))) ^ 2) = 
  2 * m * (m + 1) / 3 := 
sorry

theorem sum_of_cot_squared {m : ℕ} :
  ∑ k in Finset.range m + 1, (cot ((k : ℝ) * π / (2 * (m + 1) + 1)))^2 = 
  m * (2 * m - 1) / 3 :=
sorry

theorem sum_of_reciprocals_of_squares :
  ∑' (k : ℕ) in {1..}, 1 / (k : ℝ)^2 = π^2 / 6 :=
sorry

end sum_of_reciprocals_of_sin_squared_sum_of_cot_squared_sum_of_reciprocals_of_squares_l655_655494


namespace sonic_leads_by_19_2_meters_l655_655435

theorem sonic_leads_by_19_2_meters (v_S v_D : ℝ)
  (h1 : ∀ t, t = 200 / v_S → 200 = v_S * t)
  (h2 : ∀ t, t = 184 / v_D → 184 = v_D * t)
  (h3 : v_S / v_D = 200 / 184)
  :  240 / v_S - (200 / v_S / (200 / 184) * 240) = 19.2 := by
  sorry

end sonic_leads_by_19_2_meters_l655_655435


namespace domain_single_point_l655_655100

noncomputable def g₁ (x : ℝ) := real.sqrt (2 - x)

noncomputable def g (n : ℕ) (x : ℝ) : ℝ :=
  if n = 1 then g₁ x else g (n - 1) (real.sqrt ((n + 1) ^ 2 - x))

theorem domain_single_point {M d : ℝ} (hM : M = 4) (hd : d = 25) :
  (∀ x, (x ∈ set.univ → g M x = g 4 25) → x = d) :=
begin
  intros x hx,
  sorry
end

end domain_single_point_l655_655100


namespace rational_slope_line_l655_655306

theorem rational_slope_line (a b c : ℤ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  (∀ (lattice_point : ℤ × ℤ), ¬ (a * lattice_point.1 + b * lattice_point.2 + c = 0)) ∨ 
  (∃ (d : ℝ) (h₃ : d > 0), ∀ (lattice_point : ℤ × ℤ), 
    ((¬ (a * lattice_point.1 + b * lattice_point.2 + c = 0)) → 
    (abs (a * lattice_point.1 + b * lattice_point.2 + c) / real.sqrt (a ^ 2 + b ^ 2) ≥ d))) :=
sorry

end rational_slope_line_l655_655306


namespace fib_sequential_perfect_square_fib_divisibility_condition_l655_655000

def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

theorem fib_sequential_perfect_square :
  ∃ᶠ p q, ∃ k, k^2 = fib p * fib q - 1 := sorry

theorem fib_divisibility_condition :
  ∃ᶠ m n, fib m ∣ (fib n ^ 2 + 1) ∧ fib n ∣ (fib m ^ 2 + 1) := sorry

end fib_sequential_perfect_square_fib_divisibility_condition_l655_655000


namespace tan280_l655_655853

variable (k : ℝ)

theorem tan280 (h : Real.cos (-80 * Real.pi / 180) = k) : 
  Real.tan (280 * Real.pi / 180) = -Real.sqrt(1 - k^2) / k :=
by
  sorry

end tan280_l655_655853


namespace time_spent_watching_tv_excluding_breaks_l655_655501

-- Definitions based on conditions
def total_hours_watched : ℕ := 5
def breaks : List ℕ := [10, 15, 20, 25]

-- Conversion constants
def minutes_per_hour : ℕ := 60

-- Derived definitions
def total_minutes_watched : ℕ := total_hours_watched * minutes_per_hour
def total_break_minutes : ℕ := breaks.sum

-- The main theorem
theorem time_spent_watching_tv_excluding_breaks :
  total_minutes_watched - total_break_minutes = 230 := by
  sorry

end time_spent_watching_tv_excluding_breaks_l655_655501


namespace new_total_number_of_students_l655_655557

noncomputable def student_teacher_ratio : ℚ := 27.5
def number_of_teachers : ℕ := 42
def percentage_increase : ℚ := 0.15

theorem new_total_number_of_students : 
  let initial_students := student_teacher_ratio * number_of_teachers
  in initial_students + initial_students * percentage_increase = 1329 :=
by
  sorry

end new_total_number_of_students_l655_655557


namespace existence_of_function_values_around_k_l655_655677

-- Define the function f(n, m) with the given properties
def is_valid_function (f : ℤ × ℤ → ℤ) : Prop :=
  ∀ n m : ℤ, f (n, m) = (f (n-1, m) + f (n+1, m) + f (n, m-1) + f (n, m+1)) / 4

-- Theorem to prove the existence of such a function
theorem existence_of_function :
  ∃ (f : ℤ × ℤ → ℤ), is_valid_function f :=
sorry

-- Theorem to prove that for any k in ℤ, f(n, m) has values both greater and less than k
theorem values_around_k (k : ℤ) :
  ∃ (f : ℤ × ℤ → ℤ), is_valid_function f ∧ (∃ n1 m1 n2 m2, f (n1, m1) > k ∧ f (n2, m2) < k) :=
sorry

end existence_of_function_values_around_k_l655_655677


namespace probability_BCP_greater_l655_655253

theorem probability_BCP_greater (ABC : Type) [triangle ABC] (P : Point ABC) :
  let area_BCP_greater := measure (set_of (fun P => area (triangle BCP) > area (triangle ABP) ∧ area (triangle BCP) > area (triangle ACP)))
  (1 / 3) = measure (set_of (fun P => P ∈ ABC)).sorry

end probability_BCP_greater_l655_655253


namespace tetrahedron_angle_AB_CD_l655_655467

-- Define the points in the tetrahedron
variables {A B C D : EuclideanSpace ℝ (Fin 3)}

-- Define the length conditions
def AB := 15
def CD := 15
def BD := 20
def AC := 20
def AD := Real.sqrt 337
def BC := Real.sqrt 337

-- Define the problem statement
theorem tetrahedron_angle_AB_CD :
  ∀ (A B C D : EuclideanSpace ℝ (Fin 3)),
    dist A B = AB →
    dist C D = CD →
    dist B D = BD →
    dist A C = AC →
    dist A D = AD →
    dist B C = BC →
    ∃ (θ : ℝ), cos θ = - 7 / 25 :=
by
  intros A B C D hAB hCD hBD hAC hAD hBC
  -- skip the proof with sorry
  sorry

end tetrahedron_angle_AB_CD_l655_655467


namespace number_with_three_odd_factors_l655_655562

theorem number_with_three_odd_factors :
  ∃ n : ℕ, (∀ d : ℕ, d ∣ n → d > 1 → odd d → ∃ k : ℕ, n = k * k) ∧
           (∃! (d : ℕ), d > 1 ∧ odd d ∧ d ∣ n ∧ (∃! m: ℕ, d = m * m)) → n = 9 :=
sorry

end number_with_three_odd_factors_l655_655562


namespace bill_due_months_l655_655948

theorem bill_due_months {TD A: ℝ} (R: ℝ) : 
  TD = 189 → A = 1764 → R = 16 → 
  ∃ M: ℕ, A - TD * (1 + (R/100) * (M/12)) = 1764 - 189 * (1 + (16/100) * (10/12)) ∧ M = 10 :=
by
  intro hTD hA hR
  use 10
  sorry

end bill_due_months_l655_655948


namespace third_derivative_of_f_l655_655345

open Real

noncomputable def f (x : ℝ) := (ln (x - 1)) / (sqrt (x - 1))

theorem third_derivative_of_f (x : ℝ) (h : x > 1) :
  (deriv^[3] f) x = (46 - 15 * ln(x - 1)) / (8 * (x - 1)^(7/2)) :=
by sorry

end third_derivative_of_f_l655_655345


namespace find_x_when_fx_eq_3_l655_655095

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ -1 then x + 2 else
if x < 2 then x^2 else
2 * x

theorem find_x_when_fx_eq_3 : ∃ x : ℝ, f x = 3 ∧ x = Real.sqrt 3 := by
  sorry

end find_x_when_fx_eq_3_l655_655095


namespace profit_per_meter_correct_l655_655652

-- Define the conditions
def total_meters := 40
def total_profit := 1400

-- Define the profit per meter calculation
def profit_per_meter := total_profit / total_meters

-- Theorem stating the profit per meter is Rs. 35
theorem profit_per_meter_correct : profit_per_meter = 35 := by
  sorry

end profit_per_meter_correct_l655_655652


namespace three_digit_numbers_not_multiple_of_3_or_11_l655_655788

-- Proving the number of three-digit numbers that are multiples of neither 3 nor 11 is 547
theorem three_digit_numbers_not_multiple_of_3_or_11 : (finset.Icc 100 999).filter (λ n, ¬(3 ∣ n) ∧ ¬(11 ∣ n)).card = 547 :=
by
  -- The steps to reach the solution will be implemented here
  sorry

end three_digit_numbers_not_multiple_of_3_or_11_l655_655788


namespace points_on_curve_pass_through_center_l655_655237

theorem points_on_curve_pass_through_center {Γ : set (ℝ × ℝ)}
  {O : ℝ × ℝ} {A B : ℝ × ℝ} (h_square : is_square O)
  (h_curve : divides_square_into_equal_areas Γ) :
  (A ∈ Γ ∧ B ∈ Γ → line_through A B O) :=
sorry

end points_on_curve_pass_through_center_l655_655237


namespace perfect_square_exists_l655_655316

theorem perfect_square_exists (n : ℕ) (h : n ≥ 3) : 
  ∃ (a : Fin n → ℕ), (∀ i j : Fin n, i < j → a i < a j) ∧ 
  (∑ i, 1 / (a i) : ℚ) = 1 ∧ 
  (∏ i, a i).sqrt = ((∏ i, a i)) := 
sorry

end perfect_square_exists_l655_655316


namespace domain_of_g_l655_655985

noncomputable def g (x : ℝ) : ℝ := log 3 (log 4 (log 5 (log 6 x)))

theorem domain_of_g :
  { x : ℝ | x > 1296 } = { x : ℝ | g x = real.log 3 (real.log 4 (real.log 5 (real.log 6 x))) ∧ ∀ x > 1296, x ∈ ℝ } :=
sorry

end domain_of_g_l655_655985


namespace domain_of_g_l655_655988

def g (x : ℝ) := Real.log 3 (Real.log 4 (Real.log 5 (Real.log 6 x)))

theorem domain_of_g (x : ℝ) : g x = Real.log 3 (Real.log 4 (Real.log 5 (Real.log 6 x))) → x > 6 ^ 625 :=
by
  intros hg -- Introduce the hypothesis that g x is defined
  sorry -- Proof to be completed

end domain_of_g_l655_655988


namespace simplify_inverse_expression_l655_655196

theorem simplify_inverse_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ - y⁻¹ + z⁻¹)⁻¹ = (x * y * z) / (y * z - x * z + x * y) :=
by
  sorry

end simplify_inverse_expression_l655_655196


namespace line2_passes_through_fixed_point_l655_655752

-- Define the lines
def line1 (k : ℝ) : ℝ → ℝ := λ x, k * x + 2 - k

def symmetry_line : ℝ → ℝ := λ x, x - 1

-- Define the condition for line l2 to be symmetric to l1 with respect to y=x-1
def is_symmetric_to (l1 l2 : ℝ → ℝ) (symmetry_line : ℝ → ℝ) : Prop :=
  ∀ x y, l1 x = y ↔ l2 (x + y) = 2 * (x + y) - symmetry_line (x + y)

-- Define the fixed point
def fixed_point : ℝ × ℝ := (3, 0)

-- The main theorem statement
theorem line2_passes_through_fixed_point (k : ℝ) (l2 : ℝ → ℝ) :
  is_symmetric_to (line1 k) l2 symmetry_line →
  l2 3 = 0 :=
by
  sorry

end line2_passes_through_fixed_point_l655_655752


namespace max_cos_a_l655_655094

theorem max_cos_a (a b : ℝ) (h : Real.cos (a + b) = Real.cos a - Real.cos b) : 
  Real.cos a ≤ 1 := 
sorry

end max_cos_a_l655_655094


namespace government_subsidy_per_hour_l655_655299

-- Given conditions:
def cost_first_employee : ℕ := 20
def cost_second_employee : ℕ := 22
def hours_per_week : ℕ := 40
def weekly_savings : ℕ := 160

-- To prove:
theorem government_subsidy_per_hour (S : ℕ) : S = 2 :=
by
  -- Proof steps go here.
  sorry

end government_subsidy_per_hour_l655_655299


namespace find_a_range_l655_655093

def f (a : ℝ) (x : ℝ) := x^3 + a * x + 1 / 4

def g (x : ℝ) := - real.log x

def h (a : ℝ) (x : ℝ) := min (f a x) (g x)

theorem find_a_range (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ h a x1 = 0 ∧ h a x2 = 0 ∧ h a x3 = 0) ↔ a ∈ set.Ioo (-5/4 : ℝ) (-3/4 : ℝ) :=
by
  sorry

end find_a_range_l655_655093


namespace smallest_integer_in_odd_set_l655_655155

/-- 
Given a set of consecutive odd integers with a median of 148 and a maximum value of 159,
prove that the smallest integer in the set is 137.
-/
theorem smallest_integer_in_odd_set (S : Set ℤ) (h1 : S.nonempty)
  (h2 : ∃ m k : ℤ, (m ∈ S) ∧ (k ∈ S) ∧ (m + k = 148 * 2)) 
  (h3 : ∀ x ∈ S, odd x)
  (h4 : S.sup = 159) :
  S.inf = 137 :=
sorry

end smallest_integer_in_odd_set_l655_655155


namespace proof_problem_l655_655122

noncomputable def problem_statement : Prop :=
  ∀ (A C O B D : Type) 
    (circle : A → C → O) 
    (tangent_BA: BA → O) 
    (tangent_BC: BC → O) 
    (isosceles_ABC : ∃ (BAC BCA : ABC), BAC = BCA = 50) 
    (intersection : D ∈ BO),
  (BD / BO = 0.5774)

theorem proof_problem : problem_statement :=
by
  sorry

end proof_problem_l655_655122


namespace angle_AOD_is_120_l655_655828

variables (O A B C D : Type) [points O A B C D]
variables (angle : O → O → O → ℝ)

def perpendicular (u v : O → O) : Prop := ∃ (k : ℝ), inner (u - v) u = 0

def angle_AOD := 120

theorem angle_AOD_is_120
  (h1 : perpendicular O A C)
  (h2 : perpendicular O B D)
  (h3 : angle O A D = 2 * (angle O B C + 30)) :
  angle A O D = 120 :=
sorry

end angle_AOD_is_120_l655_655828


namespace seating_arrangements_l655_655451

theorem seating_arrangements (n : ℕ) (j w p l : ℕ) :
  n = 9 → j = 1 → w = 2 → p = 3 → l = 4 →
  (finset.card (finset.perm {1, 2, 3, 4, 5, 6, 7, 8, 9})) - 
  (finset.card (finset.perm {1, 2, 3, 4, 5, 6}) * finset.card (finset.perm {1, 2, 3, 4})) = 345600 :=
by
  intros n_eq j_eq w_eq p_eq l_eq
  sorry

end seating_arrangements_l655_655451


namespace rectangle_side_l655_655510

theorem rectangle_side
  (a b : ℕ)
  (h_a : a = 5)
  (h_b : b = 12)
  (h_diff : abs ((a * b) - (a * x)) = 25) :
  x = 7 ∨ x = 17 :=
by sorry

end rectangle_side_l655_655510


namespace JQK_base_fourteen_to_binary_has_11_digits_l655_655195

-- Conditions
def base_fourteen_to_decimal (a b c : Nat) : Nat := a * 14^2 + b * 14 + c
def to_binary (n : Nat) : List Bool := 
  if n = 0 then [false] else
  let rec loop (n l : Nat) (acc : List Bool) : List Bool :=
    if l = 0 then acc else loop (n / 2) (l - 1) (n % 2 :: acc)
  let bits := Nat.size n
  loop n bits []

-- Digit definitions for JQK
def J := 11
def Q := 12
def K := 13

-- The main statement to prove
theorem JQK_base_fourteen_to_binary_has_11_digits : 
  to_binary (base_fourteen_to_decimal J Q K) ≠ sorry.length :=
by sorry

end JQK_base_fourteen_to_binary_has_11_digits_l655_655195


namespace triangle_segment_proof_l655_655837

variables (A B C D : Type) [MetricSpace A]
variable [MetricSpace B]
variable [MetricSpace C]
variable [MetricSpace D]
variables (a b c d : A)
variables (α β γ : Real)
variables (AC BC BD AD DC : Real)

def in_triangle (A B C : Type) := ∃ D : Type, foot_of_altitude BD AC

noncomputable def angle_A_is_half_C (α γ : Real) := α = γ / 2

noncomputable def segments_difference (AD DC BC : Real) := |AD - DC| = BC

theorem triangle_segment_proof
  (α γ : Real) (angle_A_half_C : angle_A_is_half_C α γ)
  (altitude_D : in_triangle a b c) :
  segments_difference AD DC BC :=
sorry

end triangle_segment_proof_l655_655837


namespace angle_AOB_is_70_degrees_l655_655061

theorem angle_AOB_is_70_degrees 
  {A B C O : Type}  
  (h_isosceles : AB = BC) 
  (angle_ABC : ∠ B A C = 80) 
  (angle_OAC : ∠ O A C = 10)
  (angle_OCA : ∠ O C A = 30) 
  : ∠ AOB = 70 :=
sorry

end angle_AOB_is_70_degrees_l655_655061


namespace main_statement_l655_655549

variables {A B C : Type} [ordered_add_comm_group A] [ordered_add_comm_group B] [ordered_add_comm_group C]
variables {a b c S_ABC : ℝ} {δ_a δ_b δ_c : ℝ} 

def tangent_to_incircle (l : ℝ → ℝ → Prop) (r : ℝ) : Prop := 
  ∃ (I : ℝ × ℝ), ∀ (P : ℝ × ℝ), l P ↔ (dist P I = r)

def signed_distance (P : ℝ × ℝ) (l : ℝ → ℝ → Prop) : ℝ := sorry -- To be defined

variables (l : ℝ → ℝ → Prop) (r : ℝ)
variables (A_point B_point C_point : ℝ × ℝ)

def problem_condition (l : ℝ → ℝ → Prop) (A_point B_point C_point : ℝ × ℝ) (r : ℝ) (a b c : ℝ) :=
  tangent_to_incircle l r ∧
  (δ_a = signed_distance A_point l) ∧
  (δ_b = signed_distance B_point l) ∧
  (δ_c = signed_distance C_point l) ∧
  (a + b + c = 2 * (a + b + c - S_ABC) / r)

theorem main_statement (h : problem_condition l A_point B_point C_point r a b c) : 
  a * δ_a + b * δ_b + c * δ_c = 2 * S_ABC := sorry

end main_statement_l655_655549


namespace johns_profit_l655_655080

noncomputable def numberOfPuppies : ℕ :=
  let P := 8 in
  let remaining_after_giveaway := P / 2 in
  let remaining_after_keeping := remaining_after_giveaway - 1 in
  let revenue := remaining_after_keeping * 600 in
  let profit := revenue - 300 in
  P

theorem johns_profit : numberOfPuppies = 8 :=
by
  let P := 8
  have h1 : remaining_after_giveaway = P / 2 := sorry
  have h2 : remaining_after_keeping = h1 - 1 := sorry
  have h3 : revenue = h2 * 600 := sorry
  have h4 : profit = h3 - 300 := sorry
  have h5 : profit = 1500 := sorry
  exact P

end johns_profit_l655_655080


namespace three_digit_numbers_not_multiples_of_3_or_11_l655_655777

def count_multiples (a b : ℕ) (lower upper : ℕ) : ℕ :=
  (upper - lower) / b + 1

theorem three_digit_numbers_not_multiples_of_3_or_11 : 
  let total := 900
  let multiples_3 := count_multiples 3 3 102 999
  let multiples_11 := count_multiples 11 11 110 990
  let multiples_33 := count_multiples 33 33 132 990
  let multiples_3_or_11 := multiples_3 + multiples_11 - multiples_33
  total - multiples_3_or_11 = 546 := 
by 
  sorry

end three_digit_numbers_not_multiples_of_3_or_11_l655_655777


namespace angle_bisector_inequality_l655_655849

theorem angle_bisector_inequality {a b c x y z : ℝ} (habc: triangle a b c) (hx: length_bisector a b c x) (hy: length_bisector b a c y) (hz: length_bisector c a b z) :
  (1 / x) + (1 / y) + (1 / z) > (1 / a) + (1 / b) + (1 / c) :=
sorry

end angle_bisector_inequality_l655_655849


namespace lea_total_cost_example_l655_655882

/-- Léa bought one book for $16, three binders for $2 each, and six notebooks for $1 each. -/
def total_cost (book_cost binders_cost notebooks_cost : ℕ) : ℕ :=
  book_cost + binders_cost + notebooks_cost

/-- Given the individual costs, prove the total cost of Léa's purchases is $28. -/
theorem lea_total_cost_example : total_cost 16 (3 * 2) (6 * 1) = 28 := by
  sorry

end lea_total_cost_example_l655_655882


namespace limit_of_function_l655_655296

open Real

theorem limit_of_function :
  filter.tendsto (λ x, (2 - x) ^ (sin (π * x / 2) / log (2 - x))) (nhds 1) (nhds Real.exp 1) :=
  sorry

end limit_of_function_l655_655296


namespace carol_should_choose_half_l655_655266

open ProbabilityTheory

/-- Alice, Bob, and Carol play a game in which each of them chooses a number.
Alice chooses uniformly at random from [1/4, 3/4], Bob chooses uniformly at random from [1/3, 3/4],
and Carol wants to maximize her chance of winning. The winner is whose number is between the other two.
Prove that Carol should choose 1/2 for her best chance of winning. -/
theorem carol_should_choose_half : 
  ∀ (a b : ℝ),
  (uniform [1/4, 3/4] a) → 
  (uniform [1/3, 3/4] b) →
  ∃ (c : ℝ), c = 1/2 :=
begin
  assume a b h_a h_b,
  use 1/2,
  sorry
end

end carol_should_choose_half_l655_655266


namespace inclination_angle_of_line_l655_655150

-- Definitions and conditions
def line_equation (x y : ℝ) : Prop := x - y + 3 = 0

-- Theorem statement
theorem inclination_angle_of_line (x y : ℝ) (h : line_equation x y) : angle = 45 := by
  sorry

end inclination_angle_of_line_l655_655150


namespace sequences_eq_length_l655_655349

section Proof
variable (C : ℕ) (m : ℕ) (p P : ℕ → ℕ)
variable (a b : ℕ → ℕ)

-- Assume necessary conditions
axiom (h1 : ∀ m, m ≥ 2 → (p m) ∣ m ∧ ∀ d, d ∣ m → is_prime d → d ≥ p m)
axiom (h2 : ∀ m, m ≥ 2 → (P m) ∣ m ∧ ∀ d, d ∣ m → is_prime d → d ≤ P m)
axiom (h3 : a 0 = C)
axiom (h4 : b 0 = C)
axiom (h5 : ∀ k, a k ≥ 2 → a (k + 1) = a k - a k / p (a k))
axiom (h6 : ∀ k, b k ≥ 2 → b (k + 1) = b k - b k / P (b k))

-- Main theorem statement to prove both sequences terminate with the same number of terms
theorem sequences_eq_length : 
  ∃ n, a n = 1 ∧ ∃ m, b m = 1 ∧ n = m :=
sorry
end Proof

end sequences_eq_length_l655_655349


namespace infinite_positive_integer_solutions_l655_655319

theorem infinite_positive_integer_solutions :
  ∃ (k : ℕ), ∀ (n : ℕ), n > 24 → ∃ k > 24, k = n :=
sorry

end infinite_positive_integer_solutions_l655_655319


namespace unique_solution_range_l655_655717

theorem unique_solution_range :
  ∀ (m : ℝ), (∀ x : ℝ, (0 ≤ x ∧ x ≤ 2) → 2^(2*x) - (m - 1)*2^x + 2 = 0 → unique_solution_in_range x) ↔
    m ∈ set.Ioo 4 (11/2 : ℝ) ∪ {1 + 2 * real.sqrt 2} :=
by
  sorry

end unique_solution_range_l655_655717


namespace segment_bisected_by_circumcircle_l655_655904

-- Define a triangle with vertices A, B, C
variables {A B C : Point}

-- Define the incenter I of the triangle ABC
def incenter (A B C : Point) : Point := sorry

-- Define the excenter I_A opposite vertex A of the triangle ABC
def excenter (A B C : Point) : Point := sorry

-- Define the circumcenter O of the triangle ABC
def circumcenter (A B C : Point) : Point := sorry

-- Define the circumcircle with center O and radius R
def circumcircle (A B C : Point) : Circle := sorry

-- Theorem: The segment connecting the incenter I and the excenter I_A is bisected by the circumcircle of the triangle ABC
theorem segment_bisected_by_circumcircle (A B C : Point) :
  let I := incenter A B C,
      I_A := excenter A B C,
      O := circumcenter A B C,
      C := circumcircle A B C,
      D := midpoint I I_A
  in (D ∈ C) :=
  sorry

end segment_bisected_by_circumcircle_l655_655904


namespace distinct_painted_cubes_l655_655637

-- Define the context of the problem
def num_faces : ℕ := 6

def total_paintings : ℕ := num_faces.factorial

def num_rotations : ℕ := 24

-- Statement of the theorem
theorem distinct_painted_cubes (h1 : total_paintings = 720) (h2 : num_rotations = 24) : 
  total_paintings / num_rotations = 30 := by
  sorry

end distinct_painted_cubes_l655_655637


namespace doughnuts_left_l655_655324

theorem doughnuts_left (initial_doughnuts : ℕ) (num_staff : ℕ) (doughnuts_per_staff : ℕ)
  (initial_doughnuts = 50) (num_staff = 19) (doughnuts_per_staff = 2) :
  initial_doughnuts - (num_staff * doughnuts_per_staff) = 12 :=
by sorry

end doughnuts_left_l655_655324


namespace triangle_area_l655_655184

namespace MathProof

theorem triangle_area (y_eq_6 y_eq_2_plus_x y_eq_2_minus_x : ℝ → ℝ)
  (h1 : ∀ x, y_eq_6 x = 6)
  (h2 : ∀ x, y_eq_2_plus_x x = 2 + x)
  (h3 : ∀ x, y_eq_2_minus_x x = 2 - x) :
  let a := (4, 6)
  let b := (-4, 6)
  let c := (0, 2)
  let base := dist a b
  let height := (6 - 2:ℝ)
  (1 / 2 * base * height = 16) := by
    sorry

end MathProof

end triangle_area_l655_655184


namespace calculate_V3_at_2_l655_655668

def polynomial (x : ℕ) : ℕ :=
  (((((2 * x + 5) * x + 6) * x + 23) * x - 8) * x + 10) * x - 3

theorem calculate_V3_at_2 : polynomial 2 = 71 := by
  sorry

end calculate_V3_at_2_l655_655668


namespace find_m_set_l655_655001

noncomputable def A : Set ℝ := {x : ℝ | x^2 - 5*x + 6 = 0}
noncomputable def B (m : ℝ) : Set ℝ := if m = 0 then ∅ else {-1/m}

theorem find_m_set :
  { m : ℝ | A ∪ B m = A } = {0, -1/2, -1/3} :=
by
  sorry

end find_m_set_l655_655001


namespace minimum_study_tools_l655_655597

theorem minimum_study_tools (n : Nat) : n^3 ≥ 366 → n ≥ 8 := by
  intros h
  sorry

end minimum_study_tools_l655_655597


namespace domain_of_g_l655_655983

noncomputable def g (x : ℝ) : ℝ := log 3 (log 4 (log 5 (log 6 x)))

theorem domain_of_g :
  { x : ℝ | x > 1296 } = { x : ℝ | g x = real.log 3 (real.log 4 (real.log 5 (real.log 6 x))) ∧ ∀ x > 1296, x ∈ ℝ } :=
sorry

end domain_of_g_l655_655983


namespace train_crosses_lamp_post_in_30_seconds_l655_655262

open Real

/-- Prove that given a train that crosses a 2500 m long bridge in 120 s and has a length of
    833.33 m, it takes the train 30 seconds to cross a lamp post. -/
theorem train_crosses_lamp_post_in_30_seconds (L_train : ℝ) (L_bridge : ℝ) (T_bridge : ℝ) (T_lamp_post : ℝ)
  (hL_train : L_train = 833.33)
  (hL_bridge : L_bridge = 2500)
  (hT_bridge : T_bridge = 120)
  (ht : T_lamp_post = (833.33 / ((833.33 + 2500) / 120))) :
  T_lamp_post = 30 :=
by
  sorry

end train_crosses_lamp_post_in_30_seconds_l655_655262


namespace right_triangle_BC_length_l655_655518

theorem right_triangle_BC_length (ABC : Triangle)
  (right_angle_A : IsRightTriangle ABC A)
  (height_A : height_from A BC = 12)
  (cos_theta : Cos θ = 4 / 5)
  (angle_B : AngleAt B ≠ π / 2) :
  length BC = 20 :=
sorry

end right_triangle_BC_length_l655_655518


namespace emerie_quarters_l655_655206

variables (coins_zain coins_emerie kout dimes nickels : ℕ)
variable (more_coins : ℕ)

-- Definitions based on the problem's conditions
def zain_coins := 48
def emerie_dimes := 7
def emerie_nickels := 5
def more_each_coin := 10
def coin_types := 3
def emerie_non_quarters := emerie_dimes + emerie_nickels

theorem emerie_quarters : more_each_coin = 10 → 
                         coin_types = 3 → 
                         coins_zain = 48 → 
                         emerie_dimes = 7 → 
                         emerie_nickels = 5 → 
                         kout = coins_zain - (coin_types * more_each_coin) - emerie_non_quarters → 
                         kout = 6 :=
by intros; subst more_each_coin; subst coin_types; subst coins_zain; subst emerie_dimes; subst emerie_nickels; subst kout; simp [emerie_non_quarters]; sorry

end emerie_quarters_l655_655206


namespace team_combination_count_l655_655634

theorem team_combination_count (n k : ℕ) (hn : n = 7) (hk : k = 4) :
  ∃ m, m = Nat.choose n k ∧ m = 35 :=
by
  sorry

end team_combination_count_l655_655634


namespace time_per_lap_l655_655069

theorem time_per_lap 
  (total_time_minutes : ℕ := 96)
  (num_laps : ℕ := 5) :
  total_time_minutes / num_laps = 19.2 :=
by
  -- Convert 1 hour and 36 minutes to minutes
  let total_time := 60 + 36
  -- Prove that the total time in minutes is 96
  have h1 : total_time = 96 := by rfl
  -- Divide the total minutes by the number of laps
  have h2 : total_time / num_laps = 96 / 5 := by rfl
  -- Calculate 96 / 5
  calc
    (96 : ℝ) / 5 = 19.2 : by norm_num
    19.2 = 19.2 : by rfl
  -- Conclusion
  sorry

end time_per_lap_l655_655069


namespace problem_theorem_l655_655385

-- Assuming we have a geometric sequence {a_n} with common ratio q > 0,
-- and T_n denotes the product of the first n terms, we aim to prove:
-- 0 < q < 1 and T_13 > 1 > T_14 under the given condition T_7 > T_6 > T_8.

variable (a : ℕ → ℝ) (q : ℝ) (T : ℕ → ℝ) (h_q_pos : q > 0)

-- Definition of geometric sequence
def is_geometric_sequence : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Definition of T_n as the product of the first n terms
def product_first_n_terms (n : ℕ) : Prop :=
  T n = ∏ i in finset.range n, a (i + 1)

-- Condition given in the problem
def condition : Prop :=
  T 7 > T 6 ∧ T 6 > T 8

-- Theorem to prove the required conditions
theorem problem_theorem 
  (geo_seq : is_geometric_sequence a q) 
  (prod_terms : ∀ n, product_first_n_terms a T n)
  (cond : condition T) :
  0 < q ∧ q < 1 ∧ T 13 > 1 ∧ T 14 < 1 :=
  by
    sorry

end problem_theorem_l655_655385


namespace total_number_of_items_in_base10_l655_655642

theorem total_number_of_items_in_base10 : 
  let clay_tablets := (2 * 5^0 + 3 * 5^1 + 4 * 5^2 + 1 * 5^3)
  let bronze_sculptures := (1 * 5^0 + 4 * 5^1 + 0 * 5^2 + 2 * 5^3)
  let stone_carvings := (2 * 5^0 + 3 * 5^1 + 2 * 5^2)
  let total_items := clay_tablets + bronze_sculptures + stone_carvings
  total_items = 580 := by
  sorry

end total_number_of_items_in_base10_l655_655642


namespace goldbach_conjecture_contradiction_l655_655545

theorem goldbach_conjecture_contradiction : 
  (∀ n : ℕ, n > 2 → n % 2 = 0 → ∃ p1 p2 : ℕ, nat.prime p1 ∧ nat.prime p2 ∧ n = p1 + p2) ↔ 
  (¬ (∀ n : ℕ, n > 2 → n % 2 = 0 → ∃ p1 p2 : ℕ, nat.prime p1 ∧ nat.prime p2 ∧ n = p1 + p2))  :=
sorry

end goldbach_conjecture_contradiction_l655_655545


namespace vertex_of_parabola_l655_655722

theorem vertex_of_parabola :
  (∃ x y : ℝ, y = -3*x^2 + 6*x + 1 ∧ (x, y) = (1, 4)) :=
sorry

end vertex_of_parabola_l655_655722


namespace find_length_BC_l655_655441

-- Definition of points A, B, C, D
variables (A B C D : Point)
-- D is on the x-axis
variable (D_on_x_axis : D.y = 0)
-- C is below A on the x-axis
variable (C_below_A : C.x = A.x ∧ C.y < A.y)
-- Given lengths
variable (AD_len : dist A D = 26)
variable (BD_len : dist B D = 10)
variable (AC_len : dist A C = 24)

-- Required to prove BC = 24√2
theorem find_length_BC : dist B C = 24 * Real.sqrt 2 :=
by
  sorry

end find_length_BC_l655_655441


namespace min_incompatible_pairs_l655_655870

open Nat

theorem min_incompatible_pairs (n : ℕ) : 
  ∃ m, m = ⌈(prime_count ∩ Icc n (2*n) + 1) / 2⌉ :=
by
  sorry

end min_incompatible_pairs_l655_655870


namespace tetrahedron_area_property_l655_655042

-- Definitions for the areas of faces using given conditions
variable (m n l c : ℝ)
variable (OE : ℝ) (CE : ℝ)

-- Given conditions
def area_AOB := (n * m) / 2
def area_AOB_alt := (c * OE) / 2

-- Equation for OE
def OE_def := (n * m) / c

-- Equation for CE
def CE_sq := l^2 + OE^2
def CE_sq_exp := l^2 + ((n * m / c)^2)

-- Area squared equations for the required faces
def area_squared_ABC := (c * CE / 2)^2
def area_squared_COB := (l * m / 2)^2
def area_squared_COA := (l * n / 2)^2
def area_squared_AOB := (n * m / 2)^2

-- Mathematically equivalent proof statement in Lean 4
theorem tetrahedron_area_property :
  (area_squared_ABC m n l c CE) = (area_squared_COB l m + area_squared_COA l n + area_squared_AOB n m) :=
by
  sorry

end tetrahedron_area_property_l655_655042


namespace jake_peaches_l655_655070

theorem jake_peaches (jake_has_peaches : ℕ) (steven_peaches : ℕ) (h1 : jake_has_peaches = steven_peaches - 10) (h2 : steven_peaches = 13) : jake_has_peaches = 3 :=
by {
  sorry,
}

end jake_peaches_l655_655070


namespace find_f_of_three_l655_655733

variable {f : ℝ → ℝ}

theorem find_f_of_three (h : ∀ x : ℝ, f (1 - 2 * x) = x^2 + x) : f 3 = 0 :=
by
  sorry

end find_f_of_three_l655_655733


namespace area_of_right_triangle_l655_655666

theorem area_of_right_triangle (hypotenuse : ℝ) (angle : ℝ) (condition_hypotenuse : hypotenuse = 16) (condition_angle : angle = 30) : 
  let s := hypotenuse / 2 in 
  let l := s * Real.sqrt 3 in 
  (1 / 2) * s * l = 32 * Real.sqrt 3 := 
by 
  rw [condition_hypotenuse, condition_angle]
  let s := 16 / 2
  let l := s * Real.sqrt 3
  calc (1 / 2) * s * l = (1 / 2) * (16 / 2) * ((16 / 2) * Real.sqrt 3) : by sorry
  ... = 32 * Real.sqrt 3 : by sorry

end area_of_right_triangle_l655_655666


namespace simplify_sqrt5_sqrt3_simplify_series_l655_655906

theorem simplify_sqrt5_sqrt3 : (2 : ℝ) / (Real.sqrt 5 + Real.sqrt 3) = Real.sqrt 5 - Real.sqrt 3 := 
by sorry

theorem simplify_series : 
  (Finset.range 49).sum (λ n, (2 : ℝ) / (Real.sqrt (2 * n + 3) + Real.sqrt (2 * n + 1))) = Real.sqrt 99 - 1 := 
by sorry

end simplify_sqrt5_sqrt3_simplify_series_l655_655906


namespace no_rational_solution_l655_655869

theorem no_rational_solution (n : ℕ) (h : 1 < n) : ¬∃ (x : ℚ), 
  (∑ i in Finset.range (n+1), x^(n-i) * (1 / (Nat.factorial i))) = 0 := 
sorry

end no_rational_solution_l655_655869


namespace possible_integer_values_of_x_l655_655795

theorem possible_integer_values_of_x : 
  (∃! x : ℕ, 256 < x ∧ x ≤ 289) ↔ fintype.card {x : ℕ | 256 < x ∧ x ≤ 289} = 33 := 
by sorry

end possible_integer_values_of_x_l655_655795


namespace value_of_expression_at_1_l655_655593

theorem value_of_expression_at_1 : 
  let num := (1^2 - 3 * 1 - 10)
  let denom := (1^2 - 4)
  denom ≠ 0 → (num / denom = 4) :=
by
  let num := 1^2 - 3 * 1 - 10
  let denom := 1^2 - 4
  assume denom_ne_0 : denom ≠ 0
  show num / denom = 4
  sorry

end value_of_expression_at_1_l655_655593


namespace range_of_x_l655_655799

theorem range_of_x (a b : ℝ) (h1 : a > 0) (h2 : b < 0) : 
  set_of (λ x, |x - a| + |x - b| = a - b) = set.Icc b a :=
sorry

end range_of_x_l655_655799


namespace number_divisors_product_l655_655714

theorem number_divisors_product :
  ∃ N : ℕ, (∃ a b : ℕ, N = 3^a * 5^b ∧ (N^((a+1)*(b+1) / 2)) = 3^30 * 5^40) ∧ N = 3^3 * 5^4 :=
sorry

end number_divisors_product_l655_655714


namespace ratio_of_sum_of_divisors_l655_655851

noncomputable def M := 49 * 49 * 54 * 147

def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0
def not_multiple_of_3 (n : ℕ) : Prop := ¬is_multiple_of_3 n

def sum_of_divisors (n : ℕ) (p : ℕ → Prop) : ℕ :=
  Finset.sum (Finset.filter p (Finset.range (n + 1))) id

def sum_of_divisors_multiple_of_3 := sum_of_divisors M is_multiple_of_3
def sum_of_divisors_not_multiple_of_3 := sum_of_divisors M not_multiple_of_3

theorem ratio_of_sum_of_divisors :
  (∃ (r : ℕ), r = sum_of_divisors_multiple_of_3 / sum_of_divisors_not_multiple_of_3 ∧ r = 2) :=
sorry

end ratio_of_sum_of_divisors_l655_655851


namespace integer_values_of_x_in_triangle_l655_655942

theorem integer_values_of_x_in_triangle (x : ℝ) :
  (x + 14 > 38 ∧ x + 38 > 14 ∧ 14 + 38 > x) → 
  ∃ (n : ℕ), n = 27 ∧ ∀ m : ℕ, (24 < m ∧ m < 52 ↔ (m : ℝ) > 24 ∧ (m : ℝ) < 52) :=
by {
  sorry
}

end integer_values_of_x_in_triangle_l655_655942


namespace solve_n_possible_values_l655_655862

theorem solve_n_possible_values (n : ℕ) (h_pos : n > 0) :
  (∃! (S : finset (ℕ × ℕ × ℕ)), S.card = 33 ∧ 
  (∀ (x y z : ℕ), (x, y, z) ∈ S → 3 * x + 2 * y + z = n ∧ x > 0 ∧ y > 0 ∧ z > 0))
  ↔ n = 22 ∨ n = 24 ∨ n = 25 :=
begin
  sorry
end

end solve_n_possible_values_l655_655862


namespace range_of_p_is_correct_l655_655871

-- Define the function p(x) according to the problem conditions
noncomputable def p (x : ℝ) : ℝ :=
  if isPrime ⌊x⌋ then x + 2
  else p (greatestPrimeFactor ⌊x⌋) + (x + 2 - ⌊x⌋)

-- Define the interval of interest
def interval : Set ℝ := {x | 3 ≤ x ∧ x ≤ 12}

-- Definition of the range of p(x) on the interval [3, 12]
def range_p := {y | ∃ x ∈ interval, p x = y}

-- Expected range based on the solution
def expected_range_p : Set ℝ := {y | (5 ≤ y ∧ y < 6) ∨ 
                                      (7 ≤ y ∧ y < 10) ∨ 
                                      (11 ≤ y ∧ y < 14)}

-- Statement proving the range of p is equivalent to expected intervals
theorem range_of_p_is_correct : range_p = expected_range_p :=
by 
  sorry

end range_of_p_is_correct_l655_655871


namespace bus_no_loss_bus_profit_500_bus_profit_relation_l655_655629

open Real

theorem bus_no_loss (x : ℕ) : (exists x, x ≥ 300 → x * 2 - 600 ≥ 0) := by
  sorry

theorem bus_profit_500 : (2 * 500 - 600 = 400) := by
  rfl

theorem bus_profit_relation (x y : ℕ) (h : y = 2 * x - 600) 
  := (y = 2 * x - 600) := by
  rfl

end bus_no_loss_bus_profit_500_bus_profit_relation_l655_655629


namespace system_of_equations_solution_l655_655529

theorem system_of_equations_solution (a : ℝ) (x : ℕ → ℝ)
  (ha : |a| > 1)
  (h : ∀ i : ℕ, 1 ≤ i ∧ i < 1001 → (x i) ^ 2 = a * (x ((i % 1000) + 1)) + 1) :
  ∃ y, (y = (a + real.sqrt (a ^ 2 + 4)) / 2 ∨ y = (a - real.sqrt (a ^ 2 + 4)) / 2) ∧ ∀ i, 1 ≤ i ∧ i < 1001 → x i = y :=
sorry

end system_of_equations_solution_l655_655529


namespace proof_n_eq_neg2_l655_655527

theorem proof_n_eq_neg2 (n : ℤ) (h : |n + 6| = 2 - n) : n = -2 := 
by
  sorry

end proof_n_eq_neg2_l655_655527


namespace max_marked_squares_l655_655419

-- Define the problem conditions
def initial_position := (0, 0) : ℕ × ℕ
def moves : ℕ := 10
def board_size : ℕ := 20

-- Define the proposition to prove
theorem max_marked_squares : ∃ (marked_squares : ℕ), 
  (∀ i j : ℕ, 0 ≤ i → i < board_size → 0 ≤ j → j < board_size → (∃ (moves : list (ℕ × ℕ)),
  moves.length = 10 ∧ 
  list.foldl (λ (pos : ℕ × ℕ) (m : ℕ × ℕ), (pos.1 + m.1, pos.2 + m.2)) initial_position moves = (i, j)))
  → marked_squares = 36 := 
sorry

end max_marked_squares_l655_655419


namespace seth_spent_more_l655_655908

theorem seth_spent_more : 
  let ice_cream_cartons := 20
  let yogurt_cartons := 2
  let ice_cream_price := 6
  let yogurt_price := 1
  let ice_cream_discount := 0.10
  let yogurt_discount := 0.20
  let total_ice_cream_cost := ice_cream_cartons * ice_cream_price
  let total_yogurt_cost := yogurt_cartons * yogurt_price
  let discounted_ice_cream_cost := total_ice_cream_cost * (1 - ice_cream_discount)
  let discounted_yogurt_cost := total_yogurt_cost * (1 - yogurt_discount)
  discounted_ice_cream_cost - discounted_yogurt_cost = 106.40 :=
by
  sorry

end seth_spent_more_l655_655908


namespace total_time_spent_l655_655071

-- Define the conditions
def t1 : ℝ := 2.5
def t2 : ℝ := 3 * t1

-- Define the theorem to prove
theorem total_time_spent : t1 + t2 = 10 := by
  sorry

end total_time_spent_l655_655071


namespace james_makes_102_dollars_l655_655840

variables (max_capacity : ℕ) (collection_rate : ℕ)
variables (rain_monday rain_tuesday rain_wednesday : ℚ)
variables (price_monday price_tuesday price_wednesday : ℚ)

def total_money_collected (max_capacity collection_rate : ℕ) 
    (rain_monday rain_tuesday rain_wednesday price_monday price_tuesday price_wednesday : ℚ) : ℚ :=
let water_monday := rain_monday * collection_rate,
    water_tuesday := rain_tuesday * collection_rate,
    water_wednesday := rain_wednesday * collection_rate in
let total_collected := water_monday + water_tuesday + water_wednesday in
if total_collected ≤ max_capacity then 
    water_monday * price_monday + water_tuesday * price_tuesday + water_wednesday * price_wednesday
else if water_monday < max_capacity then 
    water_monday * price_monday + min (max_capacity - water_monday) water_tuesday * price_tuesday
else 
    max_capacity * price_monday

theorem james_makes_102_dollars :
total_money_collected 80 15 4 3 2.5 1.2 1.5 0.8 = 102 := 
sorry

end james_makes_102_dollars_l655_655840


namespace exists_five_consecutive_with_digit_sums_l655_655347

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := (n.to_digits 10).sum

-- State the theorem
theorem exists_five_consecutive_with_digit_sums :
  ∃ (a : ℕ), sum_of_digits a = 52 ∧ sum_of_digits (a + 4) = 20 :=
by
  -- Proof to be provided
  sorry

end exists_five_consecutive_with_digit_sums_l655_655347


namespace Carly_applications_l655_655674

theorem Carly_applications (x : ℕ) (h1 : ∀ y, y = 2 * x) (h2 : x + 2 * x = 600) : x = 200 :=
sorry

end Carly_applications_l655_655674


namespace find_y_l655_655197

theorem find_y (y : ℝ) (h : 3 * y / 7 = 21) : y = 49 := 
sorry

end find_y_l655_655197


namespace length_AD_of_circle_l655_655513

def circle_radius : ℝ := 8
def p_A : Prop := True  -- stand-in for the point A on the circle
def p_B : Prop := True  -- stand-in for the point B on the circle
def dist_AB : ℝ := 10
def p_D : Prop := True  -- stand-in for point D opposite B

theorem length_AD_of_circle 
  (r : ℝ := circle_radius)
  (A B D : Prop)
  (h_AB : dist_AB = 10)
  (h_radius : r = 8)
  (h_opposite : D)
  : ∃ AD : ℝ, AD = Real.sqrt 252.75 :=
sorry

end length_AD_of_circle_l655_655513


namespace remainder_3042_div_29_l655_655592

theorem remainder_3042_div_29 : 3042 % 29 = 26 := by
  sorry

end remainder_3042_div_29_l655_655592


namespace proof_problem_l655_655387

variables {a : ℕ → ℝ} {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a n = a 0 * q^n
def product_first_n_terms (a : ℕ → ℝ) (n : ℕ) := ∏ i in (finset.range n), a i
def T (a : ℕ → ℝ) := λ n : ℕ, product_first_n_terms a n

-- Given conditions
variables (h_seq : geometric_sequence a q)
variables (h_q_pos : q > 0)
variables (h_ineq : T a 7 > T a 6 ∧ T a 6 > T a 8)

theorem proof_problem (h_seq : geometric_sequence a q) (h_q_pos : q > 0) (h_ineq : T a 7 > T a 6 ∧ T a 6 > T a 8) :
  (0 < q ∧ q < 1) ∧ (T a 13 > 1 ∧ 1 > T a 14) :=
sorry

end proof_problem_l655_655387


namespace vector_normalization_condition_l655_655855

variables {a b : ℝ} -- Ensuring that Lean understands ℝ refers to real numbers and specifically vectors in ℝ before using it in the next parts.

-- Definitions of the vector variables
variables (a b : ℝ) (ab_non_zero : a ≠ 0 ∧ b ≠ 0)

-- Required statement
theorem vector_normalization_condition (a b : ℝ) 
(h₀ : a ≠ 0 ∧ b ≠ 0) :
  (a / abs a = b / abs b) ↔ (a = 2 * b) :=
sorry

end vector_normalization_condition_l655_655855


namespace money_conditions_l655_655024

theorem money_conditions (c d : ℝ) (h1 : 7 * c - d > 80) (h2 : 4 * c + d = 44) (h3 : d < 2 * c) :
  c > 124 / 11 ∧ d < 2 * c ∧ d = 12 :=
by
  sorry

end money_conditions_l655_655024


namespace line_plane_max_angle_l655_655029

theorem line_plane_max_angle (line_angle_with_plane: ℝ) (h : line_angle_with_plane = 720) :
  ∃ max_angle: ℝ, max_angle = 180 :=
by
  use 180
  sorry

end line_plane_max_angle_l655_655029


namespace part_a_l655_655619

variables {X : set (ℝ × ℝ)}
def f_X (X : set (ℝ × ℝ)) (n : ℕ) : ℝ := sorry
variables {m n : ℕ}

theorem part_a (X_finite : set.finite X) (m_ge_n : m ≥ n) (n_gt_2 : n > 2) :
  f_X X m + f_X X n ≥ f_X X (m+1) + f_X X (n-1) :=
sorry

end part_a_l655_655619


namespace gaokao_problem_l655_655825

noncomputable def verify_statements (total : ℕ) (boys : ℕ) (girls : ℕ) (selected : ℕ) 
                                       (selected_boys : ℕ) (selected_girls : ℕ) : Prop :=
  total = 50 ∧ boys = 30 ∧ girls = 20 ∧ selected = 5 ∧ selected_boys = 2 ∧ selected_girls = 3 →
    (¬ (selected_boys / boys.toReal > selected_girls / girls.toReal)
     ∧ ¬ (selected % 10 = 0 ∧ total % selected = 0)
     ∧ True -- statement 3 is correct but specifics are tautologically assumed here
     ∧ (selected.toReal / total.toReal = 0.1))

theorem gaokao_problem : verify_statements 50 30 20 5 2 3 := sorry

end gaokao_problem_l655_655825


namespace count_positive_integers_l655_655715

theorem count_positive_integers (n : ℕ) (x : ℝ) (h1 : n ≤ 1500) :
  (∃ x : ℝ, n = ⌊x⌋ + ⌊3*x⌋ + ⌊5*x⌋) ↔ n = 668 :=
by
  sorry

end count_positive_integers_l655_655715


namespace equality_of_a_and_b_l655_655098

theorem equality_of_a_and_b
  (a b : ℕ)
  (ha : 0 < a)
  (hb : 0 < b)
  (h : 4 * a * b - 1 ∣ (4 * a ^ 2 - 1) ^ 2) : a = b := 
sorry

end equality_of_a_and_b_l655_655098


namespace problem_condition_relationship_l655_655874

theorem problem_condition_relationship (x : ℝ) :
  (x^2 - x - 2 > 0) → (|x - 1| > 1) := 
sorry

end problem_condition_relationship_l655_655874


namespace nathan_sold_tomato_basket_for_six_dollars_l655_655886

theorem nathan_sold_tomato_basket_for_six_dollars
    (strawberry_plants : ℕ)
    (tomato_plants : ℕ)
    (strawberries_per_plant : ℕ)
    (tomatoes_per_plant : ℕ)
    (strawberries_per_basket : ℕ)
    (tomatoes_per_basket : ℕ)
    (strawberry_basket_price : ℕ)
    (total_revenue : ℕ)
    (h1 : strawberry_plants = 5)
    (h2 : tomato_plants = 7)
    (h3 : strawberries_per_plant = 14)
    (h4 : tomatoes_per_plant = 16)
    (h5 : strawberries_per_basket = 7)
    (h6 : tomatoes_per_basket = 7)
    (h7 : strawberry_basket_price = 9)
    (h8 : total_revenue = 186) : 
    let total_strawberries := strawberry_plants * strawberries_per_plant,
        total_tomatoes := tomato_plants * tomatoes_per_plant,
        strawberry_baskets := total_strawberries / strawberries_per_basket,
        tomato_baskets := total_tomatoes / tomatoes_per_basket,
        revenue_from_strawberries := strawberry_baskets * strawberry_basket_price,
        revenue_from_tomatoes := total_revenue - revenue_from_strawberries,
        tomato_basket_price := revenue_from_tomatoes / tomato_baskets
    in 
    tomato_basket_price = 6 :=
by {
  -- Defining the intermediate computations
  let total_strawberries := 5 * 14,
  let total_tomatoes := 7 * 16,
  let strawberry_baskets := total_strawberries / 7,
  let tomato_baskets := total_tomatoes / 7,
  let revenue_from_strawberries := strawberry_baskets * 9,
  let revenue_from_tomatoes := 186 - revenue_from_strawberries,
  let tomato_basket_price := revenue_from_tomatoes / tomato_baskets,
  -- The final assertion
  exact eq.refl 6,
}

end nathan_sold_tomato_basket_for_six_dollars_l655_655886


namespace function_increasing_on_interval_l655_655396

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / Real.log x

theorem function_increasing_on_interval :
  ∃ x0 > 0, ∀ x > x0, f' x > 0 :=
by
  use Real.exp 1
  sorry

end function_increasing_on_interval_l655_655396


namespace minimum_of_maximum_l655_655701

theorem minimum_of_maximum (y : ℝ) :
  (∃ (y : ℝ), ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → (minimize : ℝ)) = 1 :=
by
  sorry

end minimum_of_maximum_l655_655701


namespace value_of_expression_in_third_quadrant_l655_655422

theorem value_of_expression_in_third_quadrant (α : ℝ) (h1 : 180 < α ∧ α < 270) :
  (2 * Real.sin α) / Real.sqrt (1 - Real.cos α ^ 2) = -2 := by
  sorry

end value_of_expression_in_third_quadrant_l655_655422


namespace num_sets_with_6_sum_18_is_4_l655_655170

open Finset

def num_sets_with_6_sum_18 : ℕ :=
  let s := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  s.filter (λ t, t.card = 3 ∧ 18 ∈ {t.sum} ∧ 6 ∈ t).card

theorem num_sets_with_6_sum_18_is_4 :
  num_sets_with_6_sum_18 = 4 :=
sorry

end num_sets_with_6_sum_18_is_4_l655_655170


namespace angle_between_vectors_l655_655006

open Real

-- The conditions as stated in the problem
variables (a b : ℝ) (theta : ℝ)
hypothesis (h1 : ‖a‖ = 2)
hypothesis (h2 : ‖b‖ = 1)
hypothesis (h3 : a ∙ b = -sqrt 2)

-- The equivalent proof problem statement
theorem angle_between_vectors :
  theta = 3 * π / 4 :=
sorry

end angle_between_vectors_l655_655006
