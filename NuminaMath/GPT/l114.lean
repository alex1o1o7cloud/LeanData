import Mathlib

namespace weaving_problem_solution_l114_114882

noncomputable def daily_increase :=
  let a1 := 5
  let n := 30
  let sum_total := 390
  let d := (sum_total - a1 * n) * 2 / (n * (n - 1))
  d

theorem weaving_problem_solution :
  daily_increase = 16 / 29 :=
by
  sorry

end weaving_problem_solution_l114_114882


namespace greatest_product_obtainable_l114_114326

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l114_114326


namespace loss_eq_cost_price_of_x_balls_l114_114778

theorem loss_eq_cost_price_of_x_balls (cp ball_count sp : ℕ) (cp_ball : ℕ) 
  (hc1 : cp_ball = 60) (hc2 : cp = ball_count * cp_ball) (hs : sp = 720) 
  (hb : ball_count = 17) :
  ∃ x : ℕ, (cp - sp = x * cp_ball) ∧ x = 5 :=
by
  sorry

end loss_eq_cost_price_of_x_balls_l114_114778


namespace prob_sqrt_less_than_nine_l114_114475

/-- The probability that the square root of a randomly selected 
two-digit whole number is less than nine is 71/90. -/
theorem prob_sqrt_less_than_nine : (let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99};
                                     let A := {n : ℕ | 10 ≤ n ∧ n < 81};
                                     (A.card / S.card : ℚ) = 71 / 90) :=
by
  sorry

end prob_sqrt_less_than_nine_l114_114475


namespace greatest_divisor_condition_l114_114965

-- Define conditions
def leaves_remainder (a b k : ℕ) : Prop := ∃ q : ℕ, a = b * q + k

-- Define the greatest common divisor property
def gcd_of (a b k: ℕ) (g : ℕ) : Prop :=
  leaves_remainder a k g ∧ leaves_remainder b k g ∧ ∀ d : ℕ, (leaves_remainder a k d ∧ leaves_remainder b k d) → d ≤ g

theorem greatest_divisor_condition 
  (N : ℕ) (h1 : leaves_remainder 1657 N 6) (h2 : leaves_remainder 2037 N 5) :
  N = 127 :=
sorry

end greatest_divisor_condition_l114_114965


namespace probability_sqrt_lt_nine_two_digit_l114_114494

theorem probability_sqrt_lt_nine_two_digit :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99} in
  let T := {n : ℕ | 10 ≤ n ∧ n < 81} in
  (Fintype.card T : ℚ) / (Fintype.card S) = 71 / 90 :=
by
  sorry

end probability_sqrt_lt_nine_two_digit_l114_114494


namespace katie_candy_l114_114752

theorem katie_candy (K : ℕ) (H1 : K + 6 - 9 = 7) : K = 10 :=
by
  sorry

end katie_candy_l114_114752


namespace greatest_product_of_two_integers_with_sum_300_l114_114232

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l114_114232


namespace complement_intersection_l114_114976

noncomputable def A : set ℝ := {x | log 2 (1 / x) < 2}
noncomputable def B : set ℝ := {x | x^2 - x - 2 ≤ 0}

theorem complement_intersection (x : ℝ) :
  (x ∈ (set.univ \ A) ∩ B) ↔ (-1 ≤ x ∧ x ≤ 1/4) := by
  sorry

end complement_intersection_l114_114976


namespace hyperbolas_same_eccentricity_l114_114644

theorem hyperbolas_same_eccentricity (a : ℝ) (h : a > 0) (h' : a ≠ 1) : 
  let e1 := (sqrt (a^2 + 1)) / a
  let e2 := (sqrt (a^2 + 1)) / a
  e1 = e2 :=
by 
  sorry

end hyperbolas_same_eccentricity_l114_114644


namespace spherical_to_rectangular_coordinates_l114_114943

noncomputable def spherical_to_cartesian (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  let x := ρ * sin φ * cos θ
  let y := ρ * sin φ * sin θ
  let z := ρ * cos φ
  (x, y, z)

theorem spherical_to_rectangular_coordinates :
  spherical_to_cartesian 3 (3 * Real.pi / 2) (Real.pi / 4) = (0, -3 * Real.sqrt 2 / 2, 3 * Real.sqrt 2 / 2) := 
by
  sorry

end spherical_to_rectangular_coordinates_l114_114943


namespace no_muffin_percentage_l114_114874

/--
Each student at Cayley S.S. received exactly one snack: either a muffin, yogurt, fruit, or a granola bar.
Given that 38% of students received a muffin, prove that 62% of students did not receive a muffin.
-/
theorem no_muffin_percentage (total_students : ℕ)
  (muffin_percentage yogurt_percentage fruit_percentage granola_percentage : ℕ)
  (H1 : muffin_percentage + yogurt_percentage + fruit_percentage + granola_percentage = 100)
  (H2 : muffin_percentage = 38) :
  100 - muffin_percentage = 62 :=
by 
  rw H2
  norm_num
  sorry

end no_muffin_percentage_l114_114874


namespace percentage_increase_in_sales_l114_114517

theorem percentage_increase_in_sales (P S : ℝ) (hP : P > 0) (hS : S > 0) :
  (∃ X : ℝ, (0.8 * (1 + X / 100) = 1.44) ∧ X = 80) :=
sorry

end percentage_increase_in_sales_l114_114517


namespace probability_sqrt_lt_nine_two_digit_l114_114497

theorem probability_sqrt_lt_nine_two_digit :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99} in
  let T := {n : ℕ | 10 ≤ n ∧ n < 81} in
  (Fintype.card T : ℚ) / (Fintype.card S) = 71 / 90 :=
by
  sorry

end probability_sqrt_lt_nine_two_digit_l114_114497


namespace probability_sqrt_less_than_nine_l114_114457

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sqrt_less_than_nine (n : ℕ) : Prop := n < 81

theorem probability_sqrt_less_than_nine :
  (∃ total_num favorable_num,
    (total_num = Finset.card (Finset.filter is_two_digit (Finset.range 100)) ∧
     favorable_num = Finset.card (Finset.filter (λ n, is_two_digit n ∧ sqrt_less_than_nine n) (Finset.range 100)) ∧
     (favorable_num : ℚ) / total_num = 71 / 90)) :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114457


namespace max_product_two_integers_sum_300_l114_114154

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l114_114154


namespace max_product_300_l114_114138

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l114_114138


namespace quadratic_roots_sum_product_l114_114598

noncomputable def quadratic_sum (a b c : ℝ) : ℝ := -b / a
noncomputable def quadratic_product (a b c : ℝ) : ℝ := c / a

theorem quadratic_roots_sum_product :
  let a := 9
  let b := -45
  let c := 50
  quadratic_sum a b c = 5 ∧ quadratic_product a b c = 50 / 9 :=
by
  sorry

end quadratic_roots_sum_product_l114_114598


namespace max_product_of_two_integers_whose_sum_is_300_l114_114171

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l114_114171


namespace probability_sqrt_lt_9_l114_114397

theorem probability_sqrt_lt_9 : 
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
in probability = 71 / 90 :=
by
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  sorry

end probability_sqrt_lt_9_l114_114397


namespace tan_theta_perpendicular_value_of_f_at_alpha_l114_114041

-- Problem 1
theorem tan_theta_perpendicular {θ : ℝ} (h : Real.tan θ = -3) :
    Real.sin θ ^ 2 + Real.sin θ * Real.cos θ + 2 = 13 / 5 :=
sorry

-- Problem 2
theorem value_of_f_at_alpha {α : ℝ} :
  let f := λ α : ℝ, (Real.sin (π/2 + α) * Real.cos (π/2 - α)) / Real.cos (π + α) +
                   (2 * Real.sin (π + α) * Real.cos (π - α) - Real.cos (π + α)) /
                   (1 + Real.sin α ^ 2 + Real.cos (3 * π / 2 + α) - Real.sin (π / 2 + α) ^ 2)
  f (- 23 * π / 6) = Real.sqrt 3 - 1 / 2 :=
sorry

end tan_theta_perpendicular_value_of_f_at_alpha_l114_114041


namespace max_product_of_sum_300_l114_114271

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l114_114271


namespace g_min_value_at_0_l114_114916

noncomputable def f (x : ℝ) : ℝ := x + 1 / x
noncomputable def g (x : ℝ) : ℝ := (x^2 + 2) / (Real.sqrt (x^2 + 1))
noncomputable def h (x : ℝ) : ℝ := Real.sqrt (x^2 + 4) + 1 / (Real.sqrt (x^2 + 4))
noncomputable def j (x : ℝ) : ℝ := Real.log 3 x + Real.log x 3

theorem g_min_value_at_0 : (∃ (x : ℝ), g x = 2) :=
by
sory

end g_min_value_at_0_l114_114916


namespace inequality_equivalence_l114_114523

noncomputable def log4 (x : ℝ) : ℝ := (Real.log x / Real.log 4)

theorem inequality_equivalence (x : ℝ) (h1 : x > 0) (h2 : log4 x ≠ 0) :
    (0.2 ^ ((6 * log4 x - 3) / log4 x) > (0.008 ^ (2 * log4 x - 1)) ^ (1/3)) ↔
    (1 < x ∧ x < 2) ∨ (64 < x) :=
by sorry

end inequality_equivalence_l114_114523


namespace polynomial_irreducible_over_ZX_l114_114992

section
variables {n : ℕ} {a : Fin n → ℤ}

theorem polynomial_irreducible_over_ZX (ha : Function.Injective a) :
  Irreducible (∏ i, (X - C (a i))^2 + 1 : ℤ[X]) :=
sorry
end

end polynomial_irreducible_over_ZX_l114_114992


namespace max_product_of_sum_300_l114_114278

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l114_114278


namespace chord_division_l114_114559

theorem chord_division (P : Point) (O : Point) (A B : Point)
  (h_dist_OP : dist P O = 7)
  (h_radius : dist O (circle_center O 11) = 11)
  (h_chord_AB : dist A B = 18)
  (h_chord_passes_through_P : lies_on_line P A B) :
  ∃ (M : Point), dist A P = 12 ∧ dist P B = 6 :=
by
  sorry

end chord_division_l114_114559


namespace greatest_product_of_sum_eq_300_l114_114113

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l114_114113


namespace max_product_two_integers_l114_114188

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l114_114188


namespace points_concyclic_l114_114698

noncomputable def triangle (A B C : Point) : Prop :=
∃ I D E F : Point, 
  incircle A B C I D E F ∧ 
  ∃ K M N J : Point,
    interior_point K A B C ∧
    incircle' K B C J D M N ∧ 
    concyclic_points E F M N

theorem points_concyclic (A B C : Point) 
  (I D E F : Point) (K M N J : Point) 
  (h1 : incircle A B C I D E F)
  (h2 : interior_point K A B C)
  (h3 : incircle' K B C J D M N) : 
  concyclic_points E F M N :=
sorry

end points_concyclic_l114_114698


namespace probability_sqrt_lt_9_l114_114395

theorem probability_sqrt_lt_9 : 
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
in probability = 71 / 90 :=
by
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  sorry

end probability_sqrt_lt_9_l114_114395


namespace probability_sqrt_lt_nine_two_digit_l114_114488

theorem probability_sqrt_lt_nine_two_digit :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99} in
  let T := {n : ℕ | 10 ≤ n ∧ n < 81} in
  (Fintype.card T : ℚ) / (Fintype.card S) = 71 / 90 :=
by
  sorry

end probability_sqrt_lt_nine_two_digit_l114_114488


namespace greatest_product_obtainable_l114_114321

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l114_114321


namespace greatest_product_l114_114293

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l114_114293


namespace probability_sqrt_lt_9_of_two_digit_l114_114364

-- Define the set of two-digit whole numbers
def two_digit_whole_numbers : set ℕ := {n | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate that checks if the square root of a number is less than 9
def sqrt_lt_9 (n : ℕ) : Prop := (n : ℝ)^2 < (9 : ℝ)^2

-- Calculate the probability
theorem probability_sqrt_lt_9_of_two_digit :
  let eligible_numbers := { n ∈ two_digit_whole_numbers | sqrt_lt_9 n } in
  (eligible_numbers.to_finset.card : ℚ) / (two_digit_whole_numbers.to_finset.card : ℚ) =
  71 / 90 :=
by
  sorry

end probability_sqrt_lt_9_of_two_digit_l114_114364


namespace probability_sqrt_two_digit_lt_nine_correct_l114_114441

noncomputable def probability_sqrt_two_digit_lt_nine : ℚ :=
  let two_digit_integers := finset.Icc 10 99
  let satisfying_integers := two_digit_integers.filter (λ n => n < 81)
  let probability := (satisfying_integers.card : ℚ) / (two_digit_integers.card : ℚ)
  probability

theorem probability_sqrt_two_digit_lt_nine_correct :
  probability_sqrt_two_digit_lt_nine = 71 / 90 := by
  sorry

end probability_sqrt_two_digit_lt_nine_correct_l114_114441


namespace greatest_product_sum_300_l114_114202

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l114_114202


namespace necessary_but_not_sufficient_condition_proof_l114_114555

noncomputable def necessary_but_not_sufficient_condition (x : ℝ) : Prop :=
  2 * x ^ 2 - 5 * x - 3 ≥ 0

theorem necessary_but_not_sufficient_condition_proof (x : ℝ) :
  (x < 0 ∨ x > 2) → necessary_but_not_sufficient_condition x :=
  sorry

end necessary_but_not_sufficient_condition_proof_l114_114555


namespace simplify_sqroot_expr_l114_114794

theorem simplify_sqroot_expr : ∀ (sqrt sqrt_7 : ℝ),
  sqrt 28 = 2 * sqrt 7 ∧ sqrt 63 = 3 * sqrt 7 →
  sqrt 7 - sqrt 28 + sqrt 63 = 2 * sqrt 7 :=
by
  intros
  sorry

end simplify_sqroot_expr_l114_114794


namespace largest_square_area_contains_four_lattice_points_l114_114897

theorem largest_square_area_contains_four_lattice_points :
  ∃ s : ℝ, (⌊s⌋ = 3) ∧ (3 < s) ∧ (s^2 ≈ 9.6) := by
  sorry

end largest_square_area_contains_four_lattice_points_l114_114897


namespace card_B_l114_114002

-- Define set A
def A : Set ℕ := {2, 3, 4}

-- Define set B based on given conditions
def B : Set ℕ := {x | ∃ m n, m ∈ A ∧ n ∈ A ∧ m ≠ n ∧ x = m * n}

-- Prove that the number of elements in B is 3
theorem card_B : B.to_finset.card = 3 :=
  sorry

end card_B_l114_114002


namespace distance_between_vertices_hyperbola_l114_114964

theorem distance_between_vertices_hyperbola :
  ∀ (x y : ℝ), 16 * x^2 + 64 * x - 4 * y^2 + 8 * y + 36 = 0 → (2 * real.sqrt (3 / 2) = real.sqrt (6)) :=
by
  sorry

end distance_between_vertices_hyperbola_l114_114964


namespace perpendicular_bisector_of_intersecting_circles_l114_114699

theorem perpendicular_bisector_of_intersecting_circles :
  ∀ (C₁ C₂ : ℝ × ℝ → Prop), 
  (∀ z, C₁ z ↔ z.1^2 + z.2^2 - 6 * z.1 - 7 = 0) → 
  (∀ z, C₂ z ↔ z.1^2 + z.2^2 - 6 * z.2 - 27 = 0) → 
  ∃ (A B : ℝ × ℝ), 
    ((C₁ A ∧ C₂ A) ∧ (C₁ B ∧ C₂ B)) ∧ 
    ∀ (x y : ℝ), (x, y) ∈ line_through_centers C₁ C₂ → x + y - 3 = 0 :=
by
  sorry

end perpendicular_bisector_of_intersecting_circles_l114_114699


namespace probability_sqrt_less_than_nine_l114_114341

/-- Define the set of two-digit integers --/
def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Define the condition that the square root of the number is less than 9 --/
def sqrt_less_than_nine (n : Nat) : Prop := n < 81

/-- The number of integers from 10 to 80 --/
lemma count_satisfying_sqrt (n : Nat) : Prop :=
  is_two_digit n ∧ sqrt_less_than_nine n → n < 81

/-- Total number of two-digit integers --/
lemma count_two_digit_total (n : Nat) : Prop := is_two_digit n 

/-- The probability that a randomly selected two-digit integer's square root is less than 9. --/
theorem probability_sqrt_less_than_nine : 
  (∃ n, count_satisfying_sqrt n) / (∃ n, count_two_digit_total n) = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114341


namespace f_increasing_l114_114685

def f (x : ℝ) :=
  if x < 0 then x - Real.sin x else x^3 + 1

theorem f_increasing : StrictMono f := sorry

end f_increasing_l114_114685


namespace good_subsets_count_l114_114023

theorem good_subsets_count (n : ℕ) (hn : n ≥ 7) :
  (∃ (A : finset ℕ), A.card = 3 ∧ 
    (∀ (a₁ a₂ a₃ : ℕ), (a₁ ∈ A ∧ a₂ ∈ A ∧ a₃ ∈ A ∧ a₁ < a₂ ∧ a₂ < a₃) → 
      (a₃ ≥ a₂ + 3 ∧ a₂ ≥ a₁ + 6))) :=
  finset.card (finset.filter (λ (A : finset ℕ), 
    (∃ (a₁ a₂ a₃ : ℕ), (a₁ ∈ A ∧ a₂ ∈ A ∧ a₃ ∈ A ∧ a₁ < a₂ ∧ a₂ < a₃) ∧
        (a₃ ≥ a₂ + 3 ∧ a₂ ≥ a₁ + 6))) 
    (finset.powerset_len 3 (finset.range (n+1)))) = nat.choose (n-4) 3 :=
sorry

end good_subsets_count_l114_114023


namespace clock_angle_problem_l114_114575

-- Define the problem conditions and the required proof.
theorem clock_angle_problem (θ : ℝ) :
  (∀ t : ℝ, t > 0 → (t = 1 → (θ + 30) % 360 = θ % 360)) →
  (θ = 15 ∨ θ = 165) :=
begin
  sorry
end

end clock_angle_problem_l114_114575


namespace probability_sqrt_less_than_nine_is_correct_l114_114378

def probability_sqrt_less_than_nine : ℚ :=
  let total_two_digit_numbers := 99 - 10 + 1 in
  let satisfying_numbers := 80 - 10 + 1 in
  let probability := (satisfying_numbers : ℚ) / (total_two_digit_numbers : ℚ) in
  probability

theorem probability_sqrt_less_than_nine_is_correct :
  probability_sqrt_less_than_nine = 71 / 90 :=
by
  -- proof here
  sorry

end probability_sqrt_less_than_nine_is_correct_l114_114378


namespace probability_sqrt_lt_nine_l114_114402

theorem probability_sqrt_lt_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  let probability := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  probability = 71 / 90 :=
by
  let two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n : ℕ | 10 ≤ n ∧ n < 81}
  have h1 : two_digit_numbers.card = 90 := sorry
  have h2 : valid_numbers.card = 71 := sorry
  let probability : ℚ := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  rw [h1, h2]
  simp
  norm_num

end probability_sqrt_lt_nine_l114_114402


namespace existence_of_number_le_kr_l114_114938

variable (r : ℝ) (k : ℕ)
variable (h : r > 0)

theorem existence_of_number_le_kr :
  (2 * r ^ 2 = ∀ (a b : ℝ), a > 0 → b > 0 → ab ) →
  (∃ i : ℕ, i < k^2 → ∀ (a_i : ℝ), a_i ≤ k * r) :=
sorry

end existence_of_number_le_kr_l114_114938


namespace cubic_common_roots_l114_114627

noncomputable def roots : List ℝ := [1, -1]  -- Assume u and v can be roots 1 and -1 for simplicity

theorem cubic_common_roots (c d : ℝ) :
  (∀ u v : ℝ, u ≠ v ∧ u ∈ roots ∧ v ∈ roots →
    (u ^ 3 + c * u ^ 2 + 8 * u + 5 = 0) ∧
    (v ^ 3 + d * v ^ 2 + 10 * v + 7 = 0) ) → 
  (c = 5 ∧ d = 6) :=
by
  intros h
  sorry

end cubic_common_roots_l114_114627


namespace greatest_product_of_sum_eq_300_l114_114111

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l114_114111


namespace greatest_product_sum_300_l114_114205

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l114_114205


namespace greatest_product_of_sum_eq_300_l114_114105

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l114_114105


namespace square_area_103_l114_114082

def square_area (s : ℝ) : ℝ := s * s

theorem square_area_103 (KL PS : ℝ) 
  (h1 : KL = 5) 
  (h2 : PS = 8) : 
  square_area (sqrt (KL^2 + (KL + 5)^2)) = 103 := by
  sorry

end square_area_103_l114_114082


namespace greatest_product_sum_300_l114_114203

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l114_114203


namespace probability_of_sqrt_lt_9_l114_114354

-- Define the set of two-digit whole numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the subset of numbers for which the square root is less than 9
def valid_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 80}

-- Define the probability calculation
noncomputable def probability_sqrt_lt_9 := (valid_numbers.to_finset.card : ℝ) / (two_digit_numbers.to_finset.card : ℝ)

-- The statement we aim to prove
theorem probability_of_sqrt_lt_9 : probability_sqrt_lt_9 = 71 / 90 := 
sorry

end probability_of_sqrt_lt_9_l114_114354


namespace loss_equals_cost_price_of_some_balls_l114_114776

-- Conditions
def cost_price_per_ball := 60
def selling_price_for_17_balls := 720
def number_of_balls := 17

-- Calculations
def total_cost_price := number_of_balls * cost_price_per_ball
def loss := total_cost_price - selling_price_for_17_balls

-- Proof statement
theorem loss_equals_cost_price_of_some_balls : (loss / cost_price_per_ball) = 5 :=
by
  -- Proof would go here
  sorry

end loss_equals_cost_price_of_some_balls_l114_114776


namespace amount_of_bill_is_720_l114_114833

-- Definitions and conditions
def TD : ℝ := 360
def BD : ℝ := 428.21

-- The relationship between TD, BD, and FV
axiom relationship (FV : ℝ) : BD = TD + (TD * BD) / (FV - TD)

-- The main theorem to prove
theorem amount_of_bill_is_720 : ∃ FV : ℝ, BD = TD + (TD * BD) / (FV - TD) ∧ FV = 720 :=
by
  use 720
  sorry

end amount_of_bill_is_720_l114_114833


namespace max_product_300_l114_114145

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l114_114145


namespace find_m_l114_114899

theorem find_m (x : ℝ) (h1 : x ∈ set.Icc (-2 : ℝ) 4) (h2 : ∀ m, set.Icc (-2 : ℝ) 4 ∩ {y : ℝ | y^2 ≤ m}.measure / set.Icc (-2 : ℝ) 4.measure = 5 / 6)
  : m = 9 :=
sorry

end find_m_l114_114899


namespace loss_equals_cost_price_of_some_balls_l114_114775

-- Conditions
def cost_price_per_ball := 60
def selling_price_for_17_balls := 720
def number_of_balls := 17

-- Calculations
def total_cost_price := number_of_balls * cost_price_per_ball
def loss := total_cost_price - selling_price_for_17_balls

-- Proof statement
theorem loss_equals_cost_price_of_some_balls : (loss / cost_price_per_ball) = 5 :=
by
  -- Proof would go here
  sorry

end loss_equals_cost_price_of_some_balls_l114_114775


namespace probability_sqrt_less_than_nine_l114_114336

/-- Define the set of two-digit integers --/
def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Define the condition that the square root of the number is less than 9 --/
def sqrt_less_than_nine (n : Nat) : Prop := n < 81

/-- The number of integers from 10 to 80 --/
lemma count_satisfying_sqrt (n : Nat) : Prop :=
  is_two_digit n ∧ sqrt_less_than_nine n → n < 81

/-- Total number of two-digit integers --/
lemma count_two_digit_total (n : Nat) : Prop := is_two_digit n 

/-- The probability that a randomly selected two-digit integer's square root is less than 9. --/
theorem probability_sqrt_less_than_nine : 
  (∃ n, count_satisfying_sqrt n) / (∃ n, count_two_digit_total n) = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114336


namespace sin2θ_over_1pluscos2θ_eq_sqrt3_l114_114998

theorem sin2θ_over_1pluscos2θ_eq_sqrt3 {θ : ℝ} (h : Real.tan θ = Real.sqrt 3) :
  (Real.sin (2 * θ)) / (1 + Real.cos (2 * θ)) = Real.sqrt 3 :=
sorry

end sin2θ_over_1pluscos2θ_eq_sqrt3_l114_114998


namespace greatest_product_sum_300_l114_114240

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l114_114240


namespace find_dimes_in_jacket_l114_114520

variables (total_amount_in_dollars shorts_dimes jacket_dimes : ℕ) (value_of_dime_in_dollars : ℝ)

-- Conditions
def total_money_condition : Prop := total_amount_in_dollars = 19
def shorts_dimes_condition : Prop := shorts_dimes = 4
def value_of_dime_condition : Prop := value_of_dime_in_dollars = 0.1

-- Problem
theorem find_dimes_in_jacket :
  total_money_condition total_amount_in_dollars shorts_dimes ∧
  shorts_dimes_condition shorts_dimes ∧
  value_of_dime_condition value_of_dime_in_dollars →
  jacket_dimes = 15 :=
  sorry

end find_dimes_in_jacket_l114_114520


namespace find_variance_l114_114679

def binomial_distribution (n : ℕ) (p : ℝ) := 
  { ξ : ℕ → ℝ // ∀ k, ξ k = if k ≤ n then binom_coeff n k * p^k * (1 - p)^(n - k) else 0 }

noncomputable def expected_value (ξ : ℕ → ℝ) (n : ℕ) (p : ℝ) : ℝ := n * p
noncomputable def variance (ξ : ℕ → ℝ) (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

variable (ξ : ℕ → ℝ) (h₁ : binomial_distribution 36 (1/3) ξ) (h₂ : expected_value ξ 36 (1/3) = 12)

theorem find_variance : variance ξ 36 (1/3) = 8 := by
  sorry

end find_variance_l114_114679


namespace probability_sqrt_less_than_nine_is_correct_l114_114382

def probability_sqrt_less_than_nine : ℚ :=
  let total_two_digit_numbers := 99 - 10 + 1 in
  let satisfying_numbers := 80 - 10 + 1 in
  let probability := (satisfying_numbers : ℚ) / (total_two_digit_numbers : ℚ) in
  probability

theorem probability_sqrt_less_than_nine_is_correct :
  probability_sqrt_less_than_nine = 71 / 90 :=
by
  -- proof here
  sorry

end probability_sqrt_less_than_nine_is_correct_l114_114382


namespace expression_value_l114_114855

theorem expression_value (b : ℝ) (h : b = 1 / 3) : (3 * b^(-2) + (b^(-2) / 3)) / b = 90 := 
by
  have h₁ : b = 1 / 3 := h
  sorry

end expression_value_l114_114855


namespace mass_percentage_of_C_in_benzene_l114_114966

theorem mass_percentage_of_C_in_benzene :
  let C_molar_mass := 12.01 -- g/mol
  let H_molar_mass := 1.008 -- g/mol
  let benzene_C_atoms := 6
  let benzene_H_atoms := 6
  let C_total_mass := benzene_C_atoms * C_molar_mass
  let H_total_mass := benzene_H_atoms * H_molar_mass
  let benzene_total_mass := C_total_mass + H_total_mass
  let mass_percentage_C := (C_total_mass / benzene_total_mass) * 100
  (mass_percentage_C = 92.26) :=
by
  sorry

end mass_percentage_of_C_in_benzene_l114_114966


namespace max_product_two_integers_l114_114189

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l114_114189


namespace segment_BC_length_l114_114784

theorem segment_BC_length {A B C D : ℝ} 
  (h1 : A < B) (h2 : B < C) (h3 : C < D)
  (hAB : B - A = 12) (hCD : D - C = 32) (hAD : D - A = 62) :
  C - B = 18 :=
by
  calc
    C - B = (D - A) - (B - A) - (D - C) : sorry
    ... = 62 - 12 - 32 : sorry
    ... = 18 : sorry

end segment_BC_length_l114_114784


namespace probability_sqrt_less_than_nine_l114_114471

theorem probability_sqrt_less_than_nine :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let T := {n : ℕ | n ∈ S ∧ n < 81}
  (T.card : ℚ) / S.card = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114471


namespace vector_addition_equation_l114_114003

open_locale big_operators

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables (A B C D E F : V)

/-- D, E, and F are midpoints of sides BC, CA, and AB of triangle ABC respectively --/
def midpoint (P Q : V) := (P + Q) / 2

theorem vector_addition_equation :
  midpoint B C = D → midpoint C A = E → midpoint A B = F →
  (D - A) + 2 • (E - B) + 3 • (F - C) = (3 / 2) • (C - A) :=
by sorry

end vector_addition_equation_l114_114003


namespace hyperbola_eccentricity_l114_114811

theorem hyperbola_eccentricity
  (a b : ℝ) 
  (h_hyperbola : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1)
  (h_circle : ∀ (x y : ℝ), (x - real.sqrt 3)^2 + (y - 1)^2 = 1)
  (h_asymptote_tangent : ∀ (x y : ℝ), ∃ (k : ℝ), y = k * x ∧ k = b / a) :
  let c := real.sqrt (a^2 + b^2),
      e := c / a in
  b = real.sqrt 3 * a → e = 2 :=
sorry

end hyperbola_eccentricity_l114_114811


namespace train_length_l114_114910

theorem train_length (speed_kmph : ℕ) (time_seconds : ℕ) (length_meters : ℕ)
  (h1 : speed_kmph = 72)
  (h2 : time_seconds = 14)
  (h3 : length_meters = speed_kmph * 1000 * time_seconds / 3600)
  : length_meters = 280 := by
  sorry

end train_length_l114_114910


namespace probability_of_sqrt_lt_9_l114_114346

-- Define the set of two-digit whole numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the subset of numbers for which the square root is less than 9
def valid_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 80}

-- Define the probability calculation
noncomputable def probability_sqrt_lt_9 := (valid_numbers.to_finset.card : ℝ) / (two_digit_numbers.to_finset.card : ℝ)

-- The statement we aim to prove
theorem probability_of_sqrt_lt_9 : probability_sqrt_lt_9 = 71 / 90 := 
sorry

end probability_of_sqrt_lt_9_l114_114346


namespace man_walking_speed_l114_114553

-- This statement introduces the assumptions and goals of the proof problem.
theorem man_walking_speed
  (x : ℝ)
  (h1 : (25 * (1 / 12)) = (x * (1 / 3)))
  : x = 6.25 :=
sorry

end man_walking_speed_l114_114553


namespace min_value_c_and_d_l114_114009

theorem min_value_c_and_d (c d : ℝ) (h1 : c > 0) (h2 : d > 0)
  (h3 : c^2 - 12 * d ≥ 0)
  (h4 : 9 * d^2 - 4 * c ≥ 0) :
  c + d ≥ 5.74 :=
sorry

end min_value_c_and_d_l114_114009


namespace max_product_two_integers_l114_114190

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l114_114190


namespace max_product_of_sum_300_l114_114224

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l114_114224


namespace original_quadrilateral_area_l114_114988

theorem original_quadrilateral_area :
  let deg45 := (Real.pi / 4)
  let h := 1 * Real.sin deg45
  let base_bottom := 1 + 2 * h
  let area_perspective := 0.5 * (1 + base_bottom) * h
  let area_original := area_perspective * (2 * Real.sqrt 2)
  area_original = 2 + Real.sqrt 2 := by
  sorry

end original_quadrilateral_area_l114_114988


namespace greatest_product_sum_300_l114_114208

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l114_114208


namespace children_percentage_l114_114088

/-- 
There are 60 passengers on a bus. Children make up a certain percentage of the bus riders. 
There are 45 adults on the bus. 
Prove that the percentage of the bus riders that are children is 25%.
-/
theorem children_percentage (total_passengers adults : ℕ) (h : total_passengers = 60) (k : adults = 45) :
  (total_passengers - adults) * 100 / total_passengers = 25 :=
by
  have h_num_children : total_passengers - adults = 15 := by
    rw [h, k]
    norm_num
  rw [h_num_children]
  norm_num
  rw h
  norm_num
  sorry

end children_percentage_l114_114088


namespace greatest_product_sum_300_l114_114198

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l114_114198


namespace delta_y_over_delta_x_l114_114574

def curve (x : ℝ) : ℝ := x^2 + x

theorem delta_y_over_delta_x (Δx Δy : ℝ) 
  (hQ : (2 + Δx, 6 + Δy) = (2 + Δx, curve (2 + Δx)))
  (hP : 6 = curve 2) : 
  (Δy / Δx) = Δx + 5 :=
by
  sorry

end delta_y_over_delta_x_l114_114574


namespace simplify_expression_l114_114042

theorem simplify_expression (y : ℝ) :
  (2 * y^6 + 3 * y^5 + y^3 + 15) - (y^6 + 4 * y^5 - 2 * y^4 + 17) = 
  (y^6 - y^5 + 2 * y^4 + y^3 - 2) :=
by 
  sorry

end simplify_expression_l114_114042


namespace probability_of_sqrt_lt_9_l114_114359

-- Define the set of two-digit whole numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the subset of numbers for which the square root is less than 9
def valid_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 80}

-- Define the probability calculation
noncomputable def probability_sqrt_lt_9 := (valid_numbers.to_finset.card : ℝ) / (two_digit_numbers.to_finset.card : ℝ)

-- The statement we aim to prove
theorem probability_of_sqrt_lt_9 : probability_sqrt_lt_9 = 71 / 90 := 
sorry

end probability_of_sqrt_lt_9_l114_114359


namespace grant_received_money_l114_114568

theorem grant_received_money :
  let total_teeth := 20
  let lost_teeth := 2
  let first_tooth_amount := 20
  let other_tooth_amount_per_tooth := 2
  let remaining_teeth := total_teeth - lost_teeth - 1
  let total_amount_received := first_tooth_amount + remaining_teeth * other_tooth_amount_per_tooth
  total_amount_received = 54 :=
by  -- Start the proof mode
  sorry  -- This is where the actual proof would go

end grant_received_money_l114_114568


namespace greatest_product_l114_114289

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l114_114289


namespace difference_counts_l114_114638

noncomputable def τ (n : ℕ) : ℕ := n.factors.length + 1

noncomputable def S (n : ℕ) : ℕ := (list.range n.succ).sum (λ i, τ i)

def count_odd_S_up_to (n : ℕ) : ℕ :=
(list.range n.succ).countp (λ i, odd (S i))

def count_even_S_up_to (n : ℕ) : ℕ :=
(list.range n.succ).countp (λ i, even (S i))

theorem difference_counts {c d : ℕ} :
  c = count_odd_S_up_to 3000 →
  d = count_even_S_up_to 3000 →
  |c - d| = 1733 :=
by
  intros h_c h_d
  sorry

end difference_counts_l114_114638


namespace smaug_hoard_value_l114_114799

theorem smaug_hoard_value : 
  let gold_coins := 100
  let silver_coins := 60
  let copper_coins := 33
  let copper_per_silver := 8
  let silver_per_gold := 3
  let value_in_copper :=
    (gold_coins * silver_per_gold * copper_per_silver) +
    (silver_coins * copper_per_silver) +
    copper_coins
  in value_in_copper = 2913 := 
by 
  sorry

end smaug_hoard_value_l114_114799


namespace johns_time_to_reach_park_l114_114530

-- Define John's speed in km/hr
def speed := 7.0 -- speed in km/hr

-- Define the distance to the park in meters
def distance_m := 750.0 -- distance in meters

-- Convert the distance to kilometers
def distance_km := distance_m / 1000.0 -- distance in kilometers

-- Use the formula Time = Distance / Speed and Convert Time to Minutes
def time_minutes := (distance_km / speed) * 60.0

-- Prove the expected result
theorem johns_time_to_reach_park : abs (time_minutes - 6.43) < 0.01 := by
  sorry

end johns_time_to_reach_park_l114_114530


namespace solve_x_for_equation_l114_114959

theorem solve_x_for_equation (x : ℝ) (h : x > 1) :
  (x^2 / (x - 1)) + real.sqrt (x - 1) + (real.sqrt (x - 1) / x^2) = ((x - 1) / x^2) + (1 / real.sqrt (x - 1)) + (x^2 / real.sqrt (x - 1))
  → x = 2 := 
by { sorry }

end solve_x_for_equation_l114_114959


namespace sequence_converges_to_zero_and_N_for_epsilon_l114_114793

theorem sequence_converges_to_zero_and_N_for_epsilon :
  (∀ ε > 0, ∃ N : ℕ, ∀ n > N, |1 / (n : ℝ) - 0| < ε) ∧ 
  (∃ N : ℕ, ∀ n > N, |1 / (n : ℝ)| < 0.001) :=
by
  sorry

end sequence_converges_to_zero_and_N_for_epsilon_l114_114793


namespace find_21st_term_sequence_l114_114617

theorem find_21st_term_sequence : 
  let a n := n * (n + 1) / 2
  let t k := a k + (1..k).sum
  t 21 = 4641 :=
by
  let a (n : ℕ) := (n * (n - 1)) / 2
  let t (k : ℕ) := (Finset.range k).sum (λ i, a i + i + 1)
  suffices h: t 21 = 4641 by exact h
  sorry

end find_21st_term_sequence_l114_114617


namespace greatest_product_sum_300_l114_114197

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l114_114197


namespace max_product_of_sum_300_l114_114281

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l114_114281


namespace quadratic_rewrite_correct_a_b_c_l114_114821

noncomputable def quadratic_rewrite (x : ℝ) : ℝ := -6*x^2 + 36*x + 216

theorem quadratic_rewrite_correct_a_b_c :
  ∃ a b c : ℝ, quadratic_rewrite x = a * (x + b)^2 + c ∧ a + b + c = 261 :=
by
  sorry

end quadratic_rewrite_correct_a_b_c_l114_114821


namespace greatest_product_obtainable_l114_114324

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l114_114324


namespace probability_sqrt_lt_9_of_two_digit_l114_114373

-- Define the set of two-digit whole numbers
def two_digit_whole_numbers : set ℕ := {n | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate that checks if the square root of a number is less than 9
def sqrt_lt_9 (n : ℕ) : Prop := (n : ℝ)^2 < (9 : ℝ)^2

-- Calculate the probability
theorem probability_sqrt_lt_9_of_two_digit :
  let eligible_numbers := { n ∈ two_digit_whole_numbers | sqrt_lt_9 n } in
  (eligible_numbers.to_finset.card : ℚ) / (two_digit_whole_numbers.to_finset.card : ℚ) =
  71 / 90 :=
by
  sorry

end probability_sqrt_lt_9_of_two_digit_l114_114373


namespace smaug_hoard_value_l114_114798

def value_in_copper_coins (gold silver copper : ℕ) := gold * 24 + silver * 8 + copper

theorem smaug_hoard_value
  (gold_coins : ℕ) (silver_coins : ℕ) (copper_coins : ℕ)
  (value_silver_in_copper : ℕ) (value_gold_in_silver : ℕ) (hoarded_gold : gold_coins = 100)
  (hoarded_silver : silver_coins = 60) (hoarded_copper : copper_coins = 33)
  (value_silver : value_silver_in_copper = 8) (value_gold : value_gold_in_silver = 3) :
  value_in_copper_coins 100 60 33 = 2913 :=
by
  rw [hoarded_gold, hoarded_silver, hoarded_copper, value_silver_in_copper, value_gold]
  calc
    100 * (24:ℕ) + 60 * 8 + 33
    = 2400 + 480 + 33    : rfl
    ... = 2913           : rfl
  sorry

end smaug_hoard_value_l114_114798


namespace cost_of_socks_l114_114779

theorem cost_of_socks (x : ℝ) : 
  let initial_amount := 20
  let hat_cost := 7 
  let final_amount := 5
  let socks_pairs := 4
  let remaining_amount := initial_amount - hat_cost
  remaining_amount - socks_pairs * x = final_amount 
  -> x = 2 := 
by 
  sorry

end cost_of_socks_l114_114779


namespace compare_values_l114_114007

def a := 0.32
def b := 20.3
def c := Real.log10 20.3

theorem compare_values : b > c ∧ c > a := by
  sorry

end compare_values_l114_114007


namespace prob_sqrt_less_than_nine_l114_114479

/-- The probability that the square root of a randomly selected 
two-digit whole number is less than nine is 71/90. -/
theorem prob_sqrt_less_than_nine : (let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99};
                                     let A := {n : ℕ | 10 ≤ n ∧ n < 81};
                                     (A.card / S.card : ℚ) = 71 / 90) :=
by
  sorry

end prob_sqrt_less_than_nine_l114_114479


namespace number_of_pizzas_ordered_l114_114750

-- Define the total number of people
def total_people : ℕ := 6

-- Define the number of slices per pizza
def slices_per_pizza : ℕ := 8

-- Define the number of slices each person ate
def slices_per_person : ℕ := 4

-- Define the total number of slices eaten
def total_slices_eaten : ℕ := total_people * slices_per_person

-- Prove that the number of pizzas needed is 3
theorem number_of_pizzas_ordered : total_slices_eaten / slices_per_pizza = 3 := by
  sorry

end number_of_pizzas_ordered_l114_114750


namespace probability_sqrt_less_than_nine_l114_114448

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sqrt_less_than_nine (n : ℕ) : Prop := n < 81

theorem probability_sqrt_less_than_nine :
  (∃ total_num favorable_num,
    (total_num = Finset.card (Finset.filter is_two_digit (Finset.range 100)) ∧
     favorable_num = Finset.card (Finset.filter (λ n, is_two_digit n ∧ sqrt_less_than_nine n) (Finset.range 100)) ∧
     (favorable_num : ℚ) / total_num = 71 / 90)) :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114448


namespace greatest_product_of_sum_eq_300_l114_114118

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l114_114118


namespace ordering_of_variables_l114_114646

noncomputable def a := Real.log 3 / Real.log (1 / 2)
noncomputable def b := (1 / 3) ^ 0.2
noncomputable def c := (1 / 2) ^ (-0.5)

theorem ordering_of_variables : a < b ∧ b < c := by
  sorry

end ordering_of_variables_l114_114646


namespace number_of_integers_satisfying_inequality_l114_114706

-- Definitions based on given conditions
def satisfies_inequality (x : ℤ) : Prop :=
  (x + 2)^2 ≤ 4

-- The statement of the proof problem
theorem number_of_integers_satisfying_inequality : 
  ({x : ℤ | satisfies_inequality x}.card = 5) :=
by sorry

end number_of_integers_satisfying_inequality_l114_114706


namespace probability_sqrt_less_than_nine_l114_114333

/-- Define the set of two-digit integers --/
def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Define the condition that the square root of the number is less than 9 --/
def sqrt_less_than_nine (n : Nat) : Prop := n < 81

/-- The number of integers from 10 to 80 --/
lemma count_satisfying_sqrt (n : Nat) : Prop :=
  is_two_digit n ∧ sqrt_less_than_nine n → n < 81

/-- Total number of two-digit integers --/
lemma count_two_digit_total (n : Nat) : Prop := is_two_digit n 

/-- The probability that a randomly selected two-digit integer's square root is less than 9. --/
theorem probability_sqrt_less_than_nine : 
  (∃ n, count_satisfying_sqrt n) / (∃ n, count_two_digit_total n) = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114333


namespace range_of_f_l114_114823

def f (x : ℝ) : ℝ := real.sqrt (3 - x^2)

theorem range_of_f : set.range f = {y : ℝ | 0 ≤ y ∧ y ≤ real.sqrt 3} :=
sorry

end range_of_f_l114_114823


namespace probability_sqrt_lt_9_of_two_digit_l114_114363

-- Define the set of two-digit whole numbers
def two_digit_whole_numbers : set ℕ := {n | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate that checks if the square root of a number is less than 9
def sqrt_lt_9 (n : ℕ) : Prop := (n : ℝ)^2 < (9 : ℝ)^2

-- Calculate the probability
theorem probability_sqrt_lt_9_of_two_digit :
  let eligible_numbers := { n ∈ two_digit_whole_numbers | sqrt_lt_9 n } in
  (eligible_numbers.to_finset.card : ℚ) / (two_digit_whole_numbers.to_finset.card : ℚ) =
  71 / 90 :=
by
  sorry

end probability_sqrt_lt_9_of_two_digit_l114_114363


namespace inversion_image_l114_114099

noncomputable theory

open Real

variables {O M M' : EuclideanGeometry.Point} {R : ℝ}

def inversion (O : EuclideanGeometry.Point) (R : ℝ) (M : EuclideanGeometry.Point) : EuclideanGeometry.Point :=
if M = O + EuclideanGeometry.mkPolar R 0 then M
else O + ((R^2 / dist_sq O M) * (M - O).normalize)

theorem inversion_image (O : EuclideanGeometry.Point) (R : ℝ) (M : EuclideanGeometry.Point) :
  let M' := inversion O R M in
   if M = O + EuclideanGeometry.mkPolar R 0 then M' = M
   else dist_sq O M' = (R^2 / dist_sq O M) :=
by sorry

end inversion_image_l114_114099


namespace probability_sqrt_less_than_nine_is_correct_l114_114384

def probability_sqrt_less_than_nine : ℚ :=
  let total_two_digit_numbers := 99 - 10 + 1 in
  let satisfying_numbers := 80 - 10 + 1 in
  let probability := (satisfying_numbers : ℚ) / (total_two_digit_numbers : ℚ) in
  probability

theorem probability_sqrt_less_than_nine_is_correct :
  probability_sqrt_less_than_nine = 71 / 90 :=
by
  -- proof here
  sorry

end probability_sqrt_less_than_nine_is_correct_l114_114384


namespace correct_operations_l114_114610

theorem correct_operations : 6 * 3 + 4 + 2 = 24 := by
  -- Proof goes here
  sorry

end correct_operations_l114_114610


namespace log_8_3900_approx_eq_4_l114_114845

noncomputable def log8_round (x : ℝ) : ℤ :=
  Int.round (Real.log x / Real.log 8)

theorem log_8_3900_approx_eq_4 :
  8^3 = 512 ∧ 8^4 = 4096 ∧ 512 < 3900 ∧ 3900 < 4096 → log8_round 3900 = 4 :=
by
  intro h
  have h1 : 8^3 = 512 := h.1
  have h2 : 8^4 = 4096 := h.2.1
  have h3 : 512 < 3900 := h.2.2.1
  have h4 : 3900 < 4096 := h.2.2.2
  sorry

end log_8_3900_approx_eq_4_l114_114845


namespace pedrinho_sequence_l114_114880

theorem pedrinho_sequence (a : ℕ → ℕ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_range : ∀ i, 1 ≤ a i ∧ a i ≤ 11)
  (h_avg_int : ∀ n, 1 ≤ n ∧ n ≤ 8 → (∑ i in finRange n, a i) % n = 0) :
  a 0 = 1 ∧ a 1 = 3 ∧ a 2 = 2 ∧ a 3 = 6 ∧ a 4 = 8 ∧ a 5 = 4 ∧ a 6 = 11 ∧ a 7 = 5 ∨
  a 0 = 3 ∧ a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 6 ∧ a 4 = 8 ∧ a 5 = 4 ∧ a 6 = 11 ∧ a 7 = 5 ∨
  a 0 = 2 ∧ a 1 = 6 ∧ a 2 = 1 ∧ a 3 = 3 ∧ a 4 = 8 ∧ a 5 = 4 ∧ a 6 = 11 ∧ a 7 = 5 ∨
  a 0 = 6 ∧ a 1 = 2 ∧ a 2 = 1 ∧ a 3 = 3 ∧ a 4 = 8 ∧ a 5 = 4 ∧ a 6 = 11 ∧ a 7 = 5 ∨
  a 0 = 9 ∧ a 1 = 11 ∧ a 2 = 10 ∧ a 3 = 6 ∧ a 4 = 4 ∧ a 5 = 8 ∧ a 6 = 1 ∧ a 7 = 7 ∨
  a 0 = 11 ∧ a 1 = 9 ∧ a 2 = 10 ∧ a 3 = 6 ∧ a 4 = 4 ∧ a 5 = 8 ∧ a 6 = 1 ∧ a 7 = 7 ∨
  a 0 = 10 ∧ a 1 = 6 ∧ a 2 = 11 ∧ a 3 = 9 ∧ a 4 = 4 ∧ a 5 = 8 ∧ a 6 = 1 ∧ a 7 = 7 ∨
  a 0 = 6 ∧ a 1 = 10 ∧ a 2 = 11 ∧ a 3 = 9 ∧ a 4 = 4 ∧ a 5 = 8 ∧ a 6 = 1 ∧ a 7 = 7 :=
sorry

end pedrinho_sequence_l114_114880


namespace exists_real_polynomial_l114_114600

noncomputable def has_negative_coeff (p : Polynomial ℝ) : Prop :=
  ∃ i, (p.coeff i) < 0

noncomputable def all_positive_coeff (n : ℕ) (p : Polynomial ℝ) : Prop :=
  ∀ i, (Polynomial.derivative^[n] p).coeff i > 0

theorem exists_real_polynomial :
  ∃ p : Polynomial ℝ, has_negative_coeff p ∧ (∀ n > 1, all_positive_coeff n p) :=
sorry

end exists_real_polynomial_l114_114600


namespace greatest_product_two_ints_sum_300_l114_114269

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l114_114269


namespace greatest_product_sum_300_l114_114124

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l114_114124


namespace sin_double_angle_identity_l114_114979

theorem sin_double_angle_identity (α : ℝ) (h : Real.cos α = 1 / 4) : 
  Real.sin (π / 2 - 2 * α) = -7 / 8 :=
by 
  sorry

end sin_double_angle_identity_l114_114979


namespace probability_sqrt_lt_nine_l114_114415

theorem probability_sqrt_lt_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  let probability := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  probability = 71 / 90 :=
by
  let two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n : ℕ | 10 ≤ n ∧ n < 81}
  have h1 : two_digit_numbers.card = 90 := sorry
  have h2 : valid_numbers.card = 71 := sorry
  let probability : ℚ := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  rw [h1, h2]
  simp
  norm_num

end probability_sqrt_lt_nine_l114_114415


namespace probability_sqrt_two_digit_lt_nine_correct_l114_114435

noncomputable def probability_sqrt_two_digit_lt_nine : ℚ :=
  let two_digit_integers := finset.Icc 10 99
  let satisfying_integers := two_digit_integers.filter (λ n => n < 81)
  let probability := (satisfying_integers.card : ℚ) / (two_digit_integers.card : ℚ)
  probability

theorem probability_sqrt_two_digit_lt_nine_correct :
  probability_sqrt_two_digit_lt_nine = 71 / 90 := by
  sorry

end probability_sqrt_two_digit_lt_nine_correct_l114_114435


namespace greatest_product_sum_300_l114_114200

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l114_114200


namespace filter_replacement_in_December_l114_114608

theorem filter_replacement_in_December
  (replacement_interval : ℕ)
  (initial_month : ℕ)
  (nth_replacement : ℕ)
  (months_in_year : ℕ)
  (nth_replacement_month : ℕ) :
  replacement_interval = 7 →
  initial_month = 1 →
  nth_replacement = 18 →
  months_in_year = 12 →
  nth_replacement_month = (replacement_interval * (nth_replacement - 1) % months_in_year) + 1 →
  nth_replacement_month = 12 :=
begin
  intros h1 h2 h3 h4 h5,
  rw [h1, h2, h3, h4] at h5,
  norm_num at h5,
  exact h5,
end

end filter_replacement_in_December_l114_114608


namespace negativity_of_c_plus_b_l114_114035

variable (a b c : ℝ)

def isWithinBounds : Prop := (1 < a ∧ a < 2) ∧ (0 < b ∧ b < 1) ∧ (-2 < c ∧ c < -1)

theorem negativity_of_c_plus_b (h : isWithinBounds a b c) : c + b < 0 :=
sorry

end negativity_of_c_plus_b_l114_114035


namespace stack_logs_total_l114_114907

   theorem stack_logs_total (a l d : ℤ) (n : ℕ) (top_logs : ℕ) (h1 : a = 15) (h2 : l = 5) (h3 : d = -2) (h4 : n = ((l - a) / d).natAbs + 1) (h5 : top_logs = 5) : (n / 2 : ℤ) * (a + l) = 60 :=
   by
   sorry
   
end stack_logs_total_l114_114907


namespace find_a_l114_114763

def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem find_a (a : ℝ) 
  (h : deriv (f a) (-1) = 4) : 
  a = 10 / 3 :=
sorry

end find_a_l114_114763


namespace find_theta_l114_114622

theorem find_theta :
  ∃ θ : ℝ, 0 < θ ∧ θ < 90 ∧ θ = 80 ∧ 
  (cos 10° = sin 30° + sin θ) ∧ (sin 30° = 1 / 2) ∧ 
  (∀ x : ℝ, cos x = sin (90 - x)) ∧ (cos 10° = sin 80°) ∧ 
  (sin 80° - (1 / 2) = (sqrt 3 / 2) - (1 / 2)) := sorry

end find_theta_l114_114622


namespace greatest_product_sum_300_l114_114196

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l114_114196


namespace cyclic_quad_largest_BD_l114_114756

theorem cyclic_quad_largest_BD 
  (AB CD DA BC : ℕ) 
  (h1 : AB < 15) (h2 : BC < 15) (h3 : CD < 15) (h4 : DA < 15) 
  (distinct : List.nodup [AB, BC, CD, DA])
  (mul_eq : BC * CD = AB * DA)
  (cyclic : cyclic_quadrilateral AB CD DA BC) :
  BD = Real.sqrt (425 / 2) := sorry

end cyclic_quad_largest_BD_l114_114756


namespace prob_sqrt_less_than_nine_l114_114476

/-- The probability that the square root of a randomly selected 
two-digit whole number is less than nine is 71/90. -/
theorem prob_sqrt_less_than_nine : (let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99};
                                     let A := {n : ℕ | 10 ≤ n ∧ n < 81};
                                     (A.card / S.card : ℚ) = 71 / 90) :=
by
  sorry

end prob_sqrt_less_than_nine_l114_114476


namespace hyperbola_k_range_l114_114810

theorem hyperbola_k_range (k : ℝ) :
  (2 - k > 0) → (k - 1 < 0) → k < 1 :=
by
  intros h1 h2
  exact lt_trans h2 h1

end hyperbola_k_range_l114_114810


namespace sequence_gcd_constant_l114_114827

theorem sequence_gcd_constant (r s : ℕ) (g : ℕ) (h1 : 0 < r ∧ r % 2 = 1) (h2 : 0 < s ∧ s % 2 = 1)
  (h3 : g = Nat.gcd r s) : 
  ∃ N, ∀ n ≥ N, ∃ a : ℕ → ℕ, (a 0 = r) ∧ (a 1 = s) ∧ (∀ n ≥ 2, a n = Nat.greatest_odd_divisor (a (n - 1) + a (n - 2))) ∧ (a n = g) :=
sorry

end sequence_gcd_constant_l114_114827


namespace max_product_of_sum_300_l114_114273

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l114_114273


namespace greatest_product_two_ints_sum_300_l114_114267

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l114_114267


namespace greatest_product_l114_114286

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l114_114286


namespace arithmetic_series_sum_l114_114630

theorem arithmetic_series_sum : 
  let a1 := -41
  let d := 2
  let n := 22 in
  let an := a1 + (n - 1) * d in
  let S := n / 2 * (a1 + an) in
  an = 1 ∧ S = -440 :=
by
  sorry

end arithmetic_series_sum_l114_114630


namespace probability_sqrt_less_than_nine_l114_114332

/-- Define the set of two-digit integers --/
def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Define the condition that the square root of the number is less than 9 --/
def sqrt_less_than_nine (n : Nat) : Prop := n < 81

/-- The number of integers from 10 to 80 --/
lemma count_satisfying_sqrt (n : Nat) : Prop :=
  is_two_digit n ∧ sqrt_less_than_nine n → n < 81

/-- Total number of two-digit integers --/
lemma count_two_digit_total (n : Nat) : Prop := is_two_digit n 

/-- The probability that a randomly selected two-digit integer's square root is less than 9. --/
theorem probability_sqrt_less_than_nine : 
  (∃ n, count_satisfying_sqrt n) / (∃ n, count_two_digit_total n) = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114332


namespace probability_of_sqrt_lt_9_l114_114349

-- Define the set of two-digit whole numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the subset of numbers for which the square root is less than 9
def valid_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 80}

-- Define the probability calculation
noncomputable def probability_sqrt_lt_9 := (valid_numbers.to_finset.card : ℝ) / (two_digit_numbers.to_finset.card : ℝ)

-- The statement we aim to prove
theorem probability_of_sqrt_lt_9 : probability_sqrt_lt_9 = 71 / 90 := 
sorry

end probability_of_sqrt_lt_9_l114_114349


namespace intersection_points_l114_114595

def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 12 * x + 15
def parabola2 (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 12

theorem intersection_points :
  {p : ℝ × ℝ // (parabola1 p.1 = p.2) ∧ (parabola2 p.1 = p.2)} = {(1, 6), (3, 6)} :=
sorry

end intersection_points_l114_114595


namespace greatest_product_obtainable_l114_114329

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l114_114329


namespace derivative_tan_l114_114061

open Real

theorem derivative_tan (x : ℝ) : deriv (λ x, tan x) x = 1 / (cos x)^2 := 
by
  sorry

end derivative_tan_l114_114061


namespace greatest_product_obtainable_l114_114315

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l114_114315


namespace max_product_two_integers_sum_300_l114_114164

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l114_114164


namespace probability_sqrt_lt_9_l114_114398

theorem probability_sqrt_lt_9 : 
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
in probability = 71 / 90 :=
by
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  sorry

end probability_sqrt_lt_9_l114_114398


namespace sum_x_gx_eq_x_plus_2_l114_114069

def g (x : ℝ) : ℝ :=
  if x < -3 then (3 / 2) * x + (9 / 2) else
  if x < -1 then -3 * x + 9 else
  if x < 2 then 2 * x + 1 else
  if x < 3 then (1 / 2) * x + (7 / 2) else 
  2 * x - 2

theorem sum_x_gx_eq_x_plus_2 :
  let xs := {x : ℝ | g x = x + 2} in
  ∑ x in xs, x = -2 :=
by
  -- Using let to define xs for the set of x-coordinates where g(x) = x + 2
  let xs := {x : ℝ | g x = x + 2}
  sorry

end sum_x_gx_eq_x_plus_2_l114_114069


namespace optimal_game_order_l114_114546

theorem optimal_game_order (P_s P_w : ℝ) (h1 : 0 ≤ P_s ∧ P_s ≤ 1) (h2 : 0 ≤ P_w ∧ P_w ≤ 1) :
  let seq1 := P_s * P_w * (1 - P_s) + (1 - P_s) * P_w * P_s,
      seq2 := P_w * P_s * (1 - P_w) + (1 - P_w) * P_s * P_w in
  seq1 = seq2 :=
by
  sorry

end optimal_game_order_l114_114546


namespace probability_sqrt_less_nine_l114_114507

theorem probability_sqrt_less_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  ∃ (p : ℚ), p = (finset.card (valid_numbers.to_finset) : ℚ) / (finset.card (two_digit_numbers.to_finset) : ℚ) ∧ p = 71 / 90 :=
by
  sorry

end probability_sqrt_less_nine_l114_114507


namespace barbara_collection_value_l114_114923

/--
Barbara collects two types of ancient coins, type A and type B. She has 18 coins in total.
She finds out that 8 of these coins, which are all of type A, are worth 24 dollars in total.
Additionally, she has confirmed that 6 of the type B coins total up to 21 dollars. 
If Barbara has 12 coins of type A, how much is her entire collection worth?
-/
theorem barbara_collection_value :
  ∃ (total_value : ℝ),
    (let total_coins := 18 in
    let type_a_coins := 12 in
    let type_b_coins := total_coins - type_a_coins in
    let value_per_type_a := 24 / 8 in
    let total_value_type_a := type_a_coins * value_per_type_a in
    let value_per_type_b := 21 / 6 in
    let total_value_type_b := type_b_coins * value_per_type_b in
    total_value = total_value_type_a + total_value_type_b) ∧ total_value = 57 :=
by {
  -- The proof will be added here
  sorry
}

end barbara_collection_value_l114_114923


namespace book_pages_count_l114_114888

theorem book_pages_count (total_digits : ℕ) (h : total_digits = 552) : 
  ∃ pages_count : ℕ, pages_count = 220 :=
by {
  have : 9 * 1 + 90 * 2 = 189 := by norm_num,
  have : total_digits = 189 + 363 := by rw [h, this, add_comm, nat.add_sub_of_le],
  use 220,
  sorry
}

end book_pages_count_l114_114888


namespace max_product_of_two_integers_whose_sum_is_300_l114_114172

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l114_114172


namespace mount_pilot_snowfall_l114_114577

theorem mount_pilot_snowfall:
  ∀ (bald_mountain_snow meters billy_mountain_snow meters additional_snowfall centimeters),
    bald_mountain_snow = 150 ∧ billy_mountain_snow = 350 ∧ additional_snowfall = 326 →
    (bald_mountain_snow + billy_mountain_snow + additional_snowfall - bald_mountain_snow - billy_mountain_snow) = 326 :=
by
  intros bald_mountain_snow billy_mountain_snow additional_snowfall
  intro h
  cases h with h_bald h_rest
  cases h_rest with h_billy h_additional
  rw [h_bald, h_billy, h_additional]
  norm_num

end mount_pilot_snowfall_l114_114577


namespace sin_A_proof_l114_114734

-- Let A, B, C be points forming a right triangle with ∠C = 90°.
variables (A B C : Type) [angles : angle]

-- The triangle is a right triangle.
def is_right_triangle (A B C : Type) [angles : angle] : Prop :=
  ∡ C = π / 2

-- Angle B is given such that cos B = 1/2
def cos_B_eq_one_half (A B C : Type) [angles : angle] : Prop :=
  cos (∡ B) = 1 / 2

-- The proposition to prove: ∠A = 30° implies sin A = 1/2
theorem sin_A_proof (A B C : Type) [angles : angle] 
  (H1 : is_right_triangle A B C) (H2 : cos_B_eq_one_half A B C) : sin (∡ A) = 1 / 2 :=
by sorry

end sin_A_proof_l114_114734


namespace find_second_number_l114_114887

theorem find_second_number
  (a : ℝ) (b : ℝ)
  (h : a = 1280)
  (h_percent : 0.25 * a = 0.20 * b + 190) :
  b = 650 :=
sorry

end find_second_number_l114_114887


namespace find_values_of_a_and_b_l114_114857

variables {a b : ℤ}

theorem find_values_of_a_and_b (h : {a, 0, -1} = {4, b, 0}) : a = 4 ∧ b = -1 :=
by
  sorry

end find_values_of_a_and_b_l114_114857


namespace g_triple_is_even_l114_114663

-- Given the definition of an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define the function g
variable (g : ℝ → ℝ)

-- Assume g is an even function
axiom g_is_even : is_even_function g

-- Prove that g(g(g(x))) is an even function
theorem g_triple_is_even : is_even_function (λ x, g (g (g x))) :=
by
  -- use the axiom g_is_even to show g_triple_is_even is indeed even
  sorry

end g_triple_is_even_l114_114663


namespace greatest_product_of_sum_eq_300_l114_114112

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l114_114112


namespace greatest_product_sum_300_l114_114120

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l114_114120


namespace next_square_formed_by_4567_l114_114101

theorem next_square_formed_by_4567 :
  (∃ n : ℕ, (n ^ 2) = 5476) ∧ (Nat.digits 10 5476 = [4, 5, 6, 7]) :=
by
  sorry

end next_square_formed_by_4567_l114_114101


namespace negation_proposition_l114_114817

theorem negation_proposition:
  ¬(∃ x : ℝ, x^2 - x + 1 > 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≤ 0 :=
by
  sorry -- Proof not required as per instructions

end negation_proposition_l114_114817


namespace determine_y_l114_114950

theorem determine_y (y : ℝ) : 8^3 + 8^3 + 8^3 + 8^3 = 4^y → y = 5.5 :=
by sorry

end determine_y_l114_114950


namespace greatest_product_sum_300_l114_114127

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l114_114127


namespace probability_sqrt_less_nine_l114_114503

theorem probability_sqrt_less_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  ∃ (p : ℚ), p = (finset.card (valid_numbers.to_finset) : ℚ) / (finset.card (two_digit_numbers.to_finset) : ℚ) ∧ p = 71 / 90 :=
by
  sorry

end probability_sqrt_less_nine_l114_114503


namespace eq_solutions_l114_114958

noncomputable def solve_eq : set ℝ :=
  {x : ℝ | (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5))) = 1 / 10}

theorem eq_solutions : solve_eq = {10, -3.5} :=
by 
  sorry

end eq_solutions_l114_114958


namespace max_x_sub_y_l114_114994

theorem max_x_sub_y (x y : ℝ) 
  (h : x^2 + y^2 - 4 * x - 6 * y + 12 = 0) : 
  ∃ t : ℝ, t = x - y ∧ t ≤ 1 + Real.sqrt 2 :=
by
  sorry

end max_x_sub_y_l114_114994


namespace cubic_center_of_symmetry_l114_114975

theorem cubic_center_of_symmetry :
  let f (x : ℝ) := x^3 - 3 * x^2 + 3 * x in
  let f' (x : ℝ) := 3 * x^2 - 6 * x + 3 in
  let f'' (x : ℝ) := 6 * x - 6 in
  (∃ x0 : ℝ, f'' x0 = 0 ∧ ∀ x : ℝ, f x = x0 → f x = 1) →
  (1, 1) = (1, f 1) :=
by
  simp [f, f', f'']
  existsi (1 : ℝ)
  apply and.intro
  {
    -- f''(1) = 0
    simp,
    linarith,
  }
  {
    -- f(1) = 1
    simp,
    linarith,
  }
  
sorry

end cubic_center_of_symmetry_l114_114975


namespace no_nonconstant_polynomials_l114_114001

def P (n : ℕ) (x y : ℝ) : ℝ := x^n + x*y + y^n

theorem no_nonconstant_polynomials (n : ℕ) :
  ∀ (G H : ℝ → ℝ → ℝ), 
    (∃ (G_nonconst : ∃ x y, ∂x G != 0 ∨ ∂y G != 0) 
    ∧ ∃ (H_nonconst : ∃ x y, ∂x H != 0 ∨ ∂y H != 0),
    ∀ x y, P n x y = G x y * H x y) → false := 
sorry

end no_nonconstant_polynomials_l114_114001


namespace sum_of_products_is_70_l114_114083

theorem sum_of_products_is_70 (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 149) (h2 : a + b + c = 17) :
  a * b + b * c + c * a = 70 :=
by
  sorry 

end sum_of_products_is_70_l114_114083


namespace coeff_x2_in_PQ_is_correct_l114_114963

variable (c : ℝ)

def P (x : ℝ) : ℝ := 2 * x^3 + 4 * x^2 - 3 * x + 1
def Q (x : ℝ) : ℝ := 3 * x^3 + c * x^2 - 8 * x - 5

def coeff_x2 (x : ℝ) : ℝ := -20 - 2 * c

theorem coeff_x2_in_PQ_is_correct :
  (4 : ℝ) * (-5) + (-3) * c + c = -20 - 2 * c := by
  sorry

end coeff_x2_in_PQ_is_correct_l114_114963


namespace max_product_of_sum_300_l114_114274

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l114_114274


namespace magnitude_v_l114_114047

open Complex

theorem magnitude_v (u v : ℂ) (h1 : u * v = 20 - 15 * Complex.I) (h2 : Complex.abs u = 5) :
  Complex.abs v = 5 := by
  sorry

end magnitude_v_l114_114047


namespace average_of_xyz_l114_114710

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 15) : 
  (x + y + z) / 3 = 4 :=
sorry

end average_of_xyz_l114_114710


namespace matrix_power_2023_correct_l114_114932

noncomputable def matrix_power_2023 : Matrix (Fin 2) (Fin 2) ℤ :=
  let A := !![1, 0; 2, 1]  -- Define the matrix
  A^2023

theorem matrix_power_2023_correct :
  matrix_power_2023 = !![1, 0; 4046, 1] := by
  sorry

end matrix_power_2023_correct_l114_114932


namespace rectangles_in_5x5_grid_l114_114704

/-- 
Proof that the number of different rectangles with sides parallel to the grid 
that can be formed by connecting four of the dots in a 5x5 square array of dots 
is equal to 100.
-/
theorem rectangles_in_5x5_grid : ∃ (n : ℕ), n = 100 ∧ n = (Nat.choose 5 2 * Nat.choose 5 2) :=
by
  use 100
  split
  . rfl
  . rw [Nat.choose, Nat.choose]
  . sorry

end rectangles_in_5x5_grid_l114_114704


namespace probability_sqrt_lt_9_of_two_digit_l114_114361

-- Define the set of two-digit whole numbers
def two_digit_whole_numbers : set ℕ := {n | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate that checks if the square root of a number is less than 9
def sqrt_lt_9 (n : ℕ) : Prop := (n : ℝ)^2 < (9 : ℝ)^2

-- Calculate the probability
theorem probability_sqrt_lt_9_of_two_digit :
  let eligible_numbers := { n ∈ two_digit_whole_numbers | sqrt_lt_9 n } in
  (eligible_numbers.to_finset.card : ℚ) / (two_digit_whole_numbers.to_finset.card : ℚ) =
  71 / 90 :=
by
  sorry

end probability_sqrt_lt_9_of_two_digit_l114_114361


namespace max_product_two_integers_l114_114194

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l114_114194


namespace problem_statement_l114_114929

theorem problem_statement (a b m n : ℝ) (h1 : m = Real.sqrt 6400) 
  (h2 : n = Real.cbrt 0.064) (h3 : Real.sqrt 2 ≈ 1.4142)
  (h4 : Real.sqrt 20 ≈ 4.4721) (h5 : Real.cbrt 7 ≈ 1.9129)
  (h6 : Real.cbrt 0.7 ≈ 0.8879) (h7 : Real.sqrt a ≈ 14.142) 
  (h8 : Real.cbrt 700 ≈ b) :
  m = 80 ∧ n = 0.4 ∧ a + b ≈ 208.879 := by
  sorry

end problem_statement_l114_114929


namespace no_infinite_strictly_increasing_sequence_l114_114636

-- Definition of Sk
def S_k (k : ℕ) : Set ℝ :=
  { x | ∃ (n : Fin k → ℕ), x = (∑ i in Finset.univ, (1 : ℝ) / n i) }

-- The theorem we want to prove
theorem no_infinite_strictly_increasing_sequence (k : ℕ) (hk : 0 < k) :
  ¬ ∃ f : ℕ → ℝ, (∀ n, f n ∈ S_k k) ∧ StrictMono f := by
  sorry

end no_infinite_strictly_increasing_sequence_l114_114636


namespace probability_sqrt_lt_nine_two_digit_l114_114495

theorem probability_sqrt_lt_nine_two_digit :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99} in
  let T := {n : ℕ | 10 ≤ n ∧ n < 81} in
  (Fintype.card T : ℚ) / (Fintype.card S) = 71 / 90 :=
by
  sorry

end probability_sqrt_lt_nine_two_digit_l114_114495


namespace max_product_of_two_integers_whose_sum_is_300_l114_114173

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l114_114173


namespace average_value_l114_114711

theorem average_value (x y z : ℝ) (h : (5/4) * (x + y + z) = 15) : (x + y + z) / 3 = 4 := 
by
  have h1 : x + y + z = 15 * (4 / 5) := sorry
  have h2 : x + y + z = 12 := sorry
  have h3 : (x + y + z) / 3 = 12 / 3 := by rw [h2]
  have h4 : 12 / 3 = 4 := sorry
  rw [h4] at h3
  exact h3

end average_value_l114_114711


namespace divisors_of_two_set_divisors_of_three_set_l114_114102
noncomputable theory

-- Define the condition for a divisor
def is_divisor (d n : ℂ) : Prop := ∃ k : ℂ, k * d = n

-- Conditions for divisors of 2
def divisors_of_two (z : ℂ) : Prop :=
  (∥z∥^2 = 1 ∨ ∥z∥^2 = 2 ∨ ∥z∥^2 = 4) ∧ is_divisor z 2

-- Conditions for divisors of 3
def divisors_of_three (z : ℂ) : Prop :=
  (∥z∥^2 = 1 ∨ ∥z∥^2 = 9) ∧ is_divisor z 3

-- Theorem statements
theorem divisors_of_two_set :
  {z : ℂ | divisors_of_two z} =
  {1, -1, Complex.I, -Complex.I} ∪
  {z : ℂ | ∃ a b : ℝ, (a = 1 ∨ a = -1) ∧ (b = 1 ∨ b = -1) ∧ z = (1/√2) * (a + b * Complex.I)} ∪
  {2, -2, 2 * Complex.I, -2 * Complex.I} :=
by sorry

theorem divisors_of_three_set :
  {z : ℂ | divisors_of_three z} =
  {1, -1, Complex.I, -Complex.I, 3, -3, 3 * Complex.I, -3 * Complex.I} :=
by sorry

end divisors_of_two_set_divisors_of_three_set_l114_114102


namespace part1_part2_l114_114645

-- Conditions and Propositions
variables (a b x : ℝ)
variables (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 5 * a * b)

open Classical
noncomputable def proposition_p := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ x ≥ a + b
def proposition_q := ∀ (a b : ℝ), a > 0 ∧ b > 0 → x * a * b ≤ b^2 + 5 * a

-- Theorem statements based on the conditions.
theorem part1 : proposition_p a b x → x ∈ set.Ici (4/5) := sorry

theorem part2 : proposition_p a b x ∧ proposition_q a b x → x ∈ set.Icc (4/5) 9 := sorry

end part1_part2_l114_114645


namespace probability_sqrt_lt_nine_two_digit_l114_114490

theorem probability_sqrt_lt_nine_two_digit :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99} in
  let T := {n : ℕ | 10 ≤ n ∧ n < 81} in
  (Fintype.card T : ℚ) / (Fintype.card S) = 71 / 90 :=
by
  sorry

end probability_sqrt_lt_nine_two_digit_l114_114490


namespace max_product_sum_300_l114_114314

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l114_114314


namespace f_f_of_five_halves_l114_114068

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x - 2
  else log (x - 1) / log 2

theorem f_f_of_five_halves :
  f (f (5 / 2)) = -1 / 2 :=
by
-- Proof omitted
sorry

end f_f_of_five_halves_l114_114068


namespace graph_symmetric_about_x_eq_1_l114_114691

def f (x : ℝ) : ℝ := Real.log x + Real.log (2 - x)

theorem graph_symmetric_about_x_eq_1 : ∀ x, f x = f (2 - x) :=
by
  intro x
  dsimp [f]
  rw [add_comm]
  sorry

end graph_symmetric_about_x_eq_1_l114_114691


namespace ratio_lcm_gcf_294_490_l114_114852

theorem ratio_lcm_gcf_294_490 : ∃ (a b lcm_ab gcf_ab : ℕ), a = 294 ∧ b = 490 ∧ lcm_ab = Nat.lcm a b ∧ gcf_ab = Nat.gcd a b ∧ (lcm_ab / gcf_ab = 15) :=
by
  use [294, 490, Nat.lcm 294 490, Nat.gcd 294 490]
  simp
  sorry

end ratio_lcm_gcf_294_490_l114_114852


namespace probability_sqrt_lt_9_l114_114417

theorem probability_sqrt_lt_9 : 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 →
  ∃ p : ℚ, p = 71 / 90 ∧ 
  ∑ k in (Finset.range 100).filter (λ x, 10 ≤ x ∧ sqrt x < 9), 1 / 90 = p := 
sorry

end probability_sqrt_lt_9_l114_114417


namespace summation_values_l114_114754

theorem summation_values (x y : ℝ) (h1 : x = y * (3 - y) ^ 2) (h2 : y = x * (3 - x) ^ 2) : 
  x + y = 0 ∨ x + y = 3 ∨ x + y = 4 ∨ x + y = 5 ∨ x + y = 8 :=
sorry

end summation_values_l114_114754


namespace greatest_product_sum_300_l114_114209

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l114_114209


namespace part_i_standard_equation_part_ii_sum_distances_l114_114737

-- Given conditions
def line_parametric (t : ℝ) : (ℝ × ℝ) := 
  (1 + (real.sqrt 2 / 2) * t, 2 + (real.sqrt 2 / 2) * t)

def circle_polar (rho theta: ℝ) : Prop :=
  rho = 6 * real.sin theta

-- Prove (Ⅰ)
theorem part_i_standard_equation : ∀ (x y : ℝ),
  (∃ (t : ℝ), (x, y) = (1 + (real.sqrt 2 / 2) * t, 2 + (real.sqrt 2 / 2) * t)) →
  ((∃ (rho theta: ℝ), (x^2 + y^2 = rho^2) ∧ (y = rho * real.sin theta) ∧ circle_polar rho theta)
  ↔ (x^2 + (y - 3)^2 = 9)) :=
sorry

-- Prove (Ⅱ)
theorem part_ii_sum_distances : 
  (∃ (t1 t2 : ℝ),
    ((1 + (real.sqrt 2 / 2) * t1)^2 + (2 + (real.sqrt 2 / 2) * t1 - 3)^2 = 9)
    ∧ ((1 + (real.sqrt 2 / 2) * t2)^2 + (2 + (real.sqrt 2 / 2) * t2 - 3)^2 = 9)) 
  →
  (dist (1, 2) ((1 + (real.sqrt 2 / 2) * 1), (2 + (real.sqrt 2 / 2) * 1))
    + dist (1, 2) ((1 + (real.sqrt 2 / 2) * -1), (2 + (real.sqrt 2 / 2) * -1))
    = 14) :=
sorry

end part_i_standard_equation_part_ii_sum_distances_l114_114737


namespace probability_condition_l114_114717

noncomputable def probability_condition_satisfied : ℝ :=
  let sample_space := set.Ioo 0 4
  let interval_condition := { x : ℝ | 2 < x ∧ x < 3 }
  let measure_space := measure_theory.volume
  (measure_space interval_condition) / (measure_space sample_space)

theorem probability_condition : probability_condition_satisfied = 1 / 4 :=
  sorry -- Placeholder for the proof

end probability_condition_l114_114717


namespace max_product_of_sum_300_l114_114214

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l114_114214


namespace rate_of_markup_l114_114641

theorem rate_of_markup (S : ℝ) (hS : S = 8) (profit_percent : ℝ) (hP : profit_percent = 0.20) (expense_percent : ℝ) (hE : expense_percent = 0.10) : 
  let C := S - (profit_percent * S + expense_percent * S) in
  ((S - C) / C) * 100 = 42.857 :=
by
  sorry

end rate_of_markup_l114_114641


namespace quadrilateral_area_l114_114618

theorem quadrilateral_area {d o1 o2 : ℝ} (hd : d = 15) (ho1 : o1 = 6) (ho2 : o2 = 4) :
  (d * (o1 + o2)) / 2 = 75 := by
  sorry

end quadrilateral_area_l114_114618


namespace max_product_sum_300_l114_114309

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l114_114309


namespace initial_outlay_is_10000_l114_114037

theorem initial_outlay_is_10000 
  (I : ℝ)
  (manufacturing_cost_per_set : ℝ := 20)
  (selling_price_per_set : ℝ := 50)
  (num_sets : ℝ := 500)
  (profit : ℝ := 5000) :
  profit = (selling_price_per_set * num_sets) - (I + manufacturing_cost_per_set * num_sets) → I = 10000 :=
by
  intro h
  sorry

end initial_outlay_is_10000_l114_114037


namespace greatest_product_l114_114299

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l114_114299


namespace female_to_male_ratio_l114_114028

theorem female_to_male_ratio (total_officers_on_duty : ℕ) (female_officers_total : ℕ) (percent_female_on_duty : ℝ)
  (female_officers_on_duty : ℕ) (male_officers_on_duty : ℕ) :
  total_officers_on_duty = 200 →
  female_officers_total = 1000 →
  percent_female_on_duty = 0.10 →
  female_officers_on_duty = (percent_female_on_duty * female_officers_total).to_nat →
  total_officers_on_duty = female_officers_on_duty + male_officers_on_duty →
  (female_officers_on_duty : ℝ) / (male_officers_on_duty : ℝ) = 1 :=
begin
  sorry
end

end female_to_male_ratio_l114_114028


namespace median_of_special_list_l114_114850

theorem median_of_special_list : 
  let nums := (List.range 1250).map (λ x => x + 1) ++ (List.range 1250).map (λ x => (x + 1) ^ 3)
  ∃ m : ℝ, m = 625.5 ∧ median nums = m := 
by
  let nums := (List.range 1250).map (λ x => x + 1) ++ (List.range 1250).map (λ x => (x + 1) ^ 3)
  sorry

end median_of_special_list_l114_114850


namespace bisector_ratio_triangle_l114_114743

theorem bisector_ratio_triangle
  (X Y Z Q K L : Type)
  [EuclideanGeometry X Y Z]
  (XY : dist X Y = 8)
  (XZ : dist X Z = 6)
  (YZ : dist Y Z = 4)
  (XK_bisector : AngleBisector X Y Z K)
  (YL_bisector : AngleBisector Y X Z L)
  (Q_intersection : Intersection XK_bisector YL_bisector Q) :
  dist Y Q / dist Q L = 2 :=
sorry

end bisector_ratio_triangle_l114_114743


namespace hyperbola_asymptote_m_value_l114_114639

theorem hyperbola_asymptote_m_value (m : ℝ) :
  (∀ x y : ℝ, (x^2 / m - y^2 / 6 = 1) → (y = x)) → m = 6 :=
by
  intros hx
  sorry

end hyperbola_asymptote_m_value_l114_114639


namespace max_product_300_l114_114147

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l114_114147


namespace greatest_product_obtainable_l114_114316

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l114_114316


namespace loss_eq_cost_price_of_x_balls_l114_114777

theorem loss_eq_cost_price_of_x_balls (cp ball_count sp : ℕ) (cp_ball : ℕ) 
  (hc1 : cp_ball = 60) (hc2 : cp = ball_count * cp_ball) (hs : sp = 720) 
  (hb : ball_count = 17) :
  ∃ x : ℕ, (cp - sp = x * cp_ball) ∧ x = 5 :=
by
  sorry

end loss_eq_cost_price_of_x_balls_l114_114777


namespace geometric_extraction_from_arithmetic_l114_114031

theorem geometric_extraction_from_arithmetic (a b : ℤ) :
  ∃ k : ℕ → ℤ, (∀ n : ℕ, k n = a * (b + 1) ^ n) ∧ (∀ n : ℕ, ∃ m : ℕ, k n = a + b * m) :=
by sorry

end geometric_extraction_from_arithmetic_l114_114031


namespace total_chickens_l114_114086

-- Definitions from conditions
def ducks : ℕ := 40
def rabbits : ℕ := 30
def hens : ℕ := ducks + 20
def roosters : ℕ := rabbits - 10

-- Theorem statement: total number of chickens
theorem total_chickens : hens + roosters = 80 := 
sorry

end total_chickens_l114_114086


namespace magnitude_of_v_l114_114048

theorem magnitude_of_v (u v : ℂ) (h1 : u * v = 20 - 15 * complex.i) (h2 : complex.abs u = 5) : complex.abs v = 5 :=
sorry

end magnitude_of_v_l114_114048


namespace contestant_can_pick_three_boxes_without_zonk_l114_114774

theorem contestant_can_pick_three_boxes_without_zonk :
  let P_no_zonk (n : ℕ) := (2/3 : ℚ)^n in
  P_no_zonk 3 = 0.2962962962962963 :=
by
  let P_no_zonk := fun (n : ℕ) => (2/3 : ℚ)^n
  have h : P_no_zonk 3 = (2/3)^3 := rfl
  have h' : (8/27 : ℚ) = 0.2962962962962963 := by norm_num
  rw [h]
  rw [h']
  sorry

end contestant_can_pick_three_boxes_without_zonk_l114_114774


namespace probability_sqrt_less_than_nine_l114_114467

theorem probability_sqrt_less_than_nine :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let T := {n : ℕ | n ∈ S ∧ n < 81}
  (T.card : ℚ) / S.card = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114467


namespace find_ω_find_φ_find_b_c_l114_114684

def f (ω φ x : ℝ) : ℝ := 2 * Real.cos (ω * x + φ)

theorem find_ω (ω : ℝ) (h_period : ω > 0) (h_min_period : ∀ x, f ω 0 x = f ω 0 (x + π)) : ω = 2 := by
  sorry

theorem find_φ (φ : ℝ) (h_symmetry : 0 < φ ∧ φ < π / 2) (h_symmetry_axis : ∀ x, f 2 φ x = f 2 φ (- (π / 24) - x)) : φ = π / 12 := by
  sorry

theorem find_b_c (A b c : ℝ) (h_f_A2 : f 2 (π / 12) (- A / 2) = sqrt 2) (h_a : a = 3) (h_bc_sum : b + c = 6) (h_cos_A : A = π / 3) 
(h_law_of_cosines : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) (h_prod : a^2 = (b + c)^2 - 3 * b * c) : b = 3 ∧ c = 3 := by
  sorry

end find_ω_find_φ_find_b_c_l114_114684


namespace max_product_of_sum_300_l114_114213

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l114_114213


namespace domain_of_f_is_neg_inf_to_0_l114_114064

def domain_of_f (x : ℝ) : Prop :=
  f x = sqrt (log (1 - 2 * x))

theorem domain_of_f_is_neg_inf_to_0 : ∀ x, domain_of_f x → x ∈ set.Iic 0 := 
begin
  sorry
end

end domain_of_f_is_neg_inf_to_0_l114_114064


namespace find_number_l114_114862

theorem find_number (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 := 
sorry

end find_number_l114_114862


namespace find_p_at_8_l114_114767

noncomputable def h (x : ℝ) : ℝ := x^3 - x^2 + x - 1

noncomputable def p (x : ℝ) : ℝ :=
  let a := sorry ; -- root 1 of h
  let b := sorry ; -- root 2 of h
  let c := sorry ; -- root 3 of h
  let B := 2 / ((1 - a^3) * (1 - b^3) * (1 - c^3))
  B * (x - a^3) * (x - b^3) * (x - c^3)

theorem find_p_at_8 : p 8 = 1008 := sorry

end find_p_at_8_l114_114767


namespace stone_at_position_is_nine_l114_114840

theorem stone_at_position_is_nine (n : ℕ) : 
  (∃ k : ℕ, k < 12 ∧ (151 = (k + (n * 22) + 1))) 
  → 9 = (151 % 22) :=
by
  -- Given 12 stones count and 22 length cycle, 
  -- we need to show the stone counted as 151 is 9.
  intro h,
  show 9 = (151 % 22),
  exact Nat.mod_eq_of_lt 151 22
  sorry

end stone_at_position_is_nine_l114_114840


namespace probability_of_rain_on_at_least_one_day_is_correct_l114_114972

def rain_on_friday_probability : ℝ := 0.30
def rain_on_saturday_probability : ℝ := 0.45
def rain_on_sunday_probability : ℝ := 0.50

def rain_on_at_least_one_day_probability : ℝ := 1 - (1 - rain_on_friday_probability) * (1 - rain_on_saturday_probability) * (1 - rain_on_sunday_probability)

theorem probability_of_rain_on_at_least_one_day_is_correct :
  rain_on_at_least_one_day_probability = 0.8075 := by
sorry

end probability_of_rain_on_at_least_one_day_is_correct_l114_114972


namespace functional_relationship_profit_maximized_at_sufficient_profit_range_verified_l114_114885

noncomputable def daily_sales_profit (x : ℝ) : ℝ :=
  -5 * x^2 + 800 * x - 27500

def profit_maximized (x : ℝ) : Prop :=
  daily_sales_profit x = -5 * (80 - x)^2 + 4500

def sufficient_profit_range (x : ℝ) : Prop :=
  daily_sales_profit x >= 4000 ∧ (x - 50) * (500 - 5 * x) <= 7000

theorem functional_relationship (x : ℝ) : daily_sales_profit x = -5 * x^2 + 800 * x - 27500 :=
  sorry

theorem profit_maximized_at (x : ℝ) : profit_maximized x → x = 80 ∧ daily_sales_profit x = 4500 :=
  sorry

theorem sufficient_profit_range_verified (x : ℝ) : sufficient_profit_range x → 82 ≤ x ∧ x ≤ 90 :=
  sorry

end functional_relationship_profit_maximized_at_sufficient_profit_range_verified_l114_114885


namespace greatest_product_two_ints_sum_300_l114_114255

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l114_114255


namespace greatest_product_obtainable_l114_114328

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l114_114328


namespace total_amount_silver_l114_114739

theorem total_amount_silver (x y : ℝ) (h₁ : y = 7 * x + 4) (h₂ : y = 9 * x - 8) : y = 46 :=
by {
  sorry
}

end total_amount_silver_l114_114739


namespace greatest_product_two_ints_sum_300_l114_114259

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l114_114259


namespace mixed_repeating_decimal_denominator_divisibility_l114_114879

open Nat

theorem mixed_repeating_decimal_denominator_divisibility
  (x : ℚ)
  (m k : ℕ)
  (b a : Fin m → ℕ)
  (N M : ℕ)
  (h1 : ∀ i, b i ∈ Fin 10)
  (h2 : ∀ j, a j ∈ Fin 10)
  (h3 : 1 ≤ m)
  (h4 : b m ≠ a k)
  (hx : x = N + M / (10 ^ m * (10 ^ k - 1)))
  (hx_irreducible : ∃ p q, x = p / q ∧ Nat.gcd p q = 1) :
  ∃ (q : ℕ), hx_irreducible ∧ (2 ∣ q ∨ 5 ∣ q) :=
sorry

end mixed_repeating_decimal_denominator_divisibility_l114_114879


namespace drawings_per_neighbor_l114_114792

theorem drawings_per_neighbor (n_neighbors animals : ℕ) (h1 : n_neighbors = 6) (h2 : animals = 54) : animals / n_neighbors = 9 :=
by
  sorry

end drawings_per_neighbor_l114_114792


namespace angle_sum_180_l114_114661

-- Define the conditions and question in Lean 4

variables {α β γ : ℝ} -- representing angles

-- Additional geometric setup can be written, here simplified for clarity
variables (A B C D E F G H I J K L : Type) -- vertices and points
variables (triangle_ABC : triangle A B C)
variables (rectangle_ABDE : rectangle A B D E)
variables (rectangle_BCFG : rectangle B C F G)
variables (rectangle_CAHI : rectangle C A H I)
variables (midpoint_J : midpoint J A B)
variables (midpoint_K : midpoint K B C)
variables (midpoint_L : midpoint L C A)

-- Theorem to prove the sum of the angles
theorem angle_sum_180
  (hABDE : inscribed_on_AB rectangle_ABDE)
  (hBCFG : inscribed_on_BC rectangle_BCFG)
  (hCAHI : inscribed_on_CA rectangle_CAHI)
  (hJ_mid : midpoint J A B)
  (hK_mid : midpoint K B C)
  (hL_mid : midpoint L C A)
  (hGJH : angle G J H = α)
  (hIKD : angle I K D = β)
  (hELF : angle E L F = γ)
  (hABC : α + β + γ = 180) :
  α + β + γ = 180 := by
  -- Proof outline using geometry omitted
  sorry

end angle_sum_180_l114_114661


namespace sum_fractions_l114_114098

open Nat

theorem sum_fractions (n : ℕ) (h : 0 < n) :
  (∑ i in range n, 1 / ((2 * i + 1) * (2 * i + 3))) = n / (2 * n + 1) :=
by
  sorry

end sum_fractions_l114_114098


namespace periodic_function_l114_114786

noncomputable def is_symmetric_wrt_point (f : ℝ → ℝ) (a y₀ : ℝ) : Prop :=
  ∀ x, f(a + x) - y₀ = y₀ - f(a - x)

noncomputable def is_symmetric_wrt_line (f : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, f(b + x) = f(b - x)

theorem periodic_function {f : ℝ → ℝ} {a b y₀ : ℝ} (h₀ : a < b) 
  (h₁ : is_symmetric_wrt_point f a y₀) (h₂ : is_symmetric_wrt_line f b) 
  : ∀ x, f(x + 4 * (b - a)) = f(x) :=
sorry

end periodic_function_l114_114786


namespace points_in_triangle_area_l114_114732

/-- In a triangle with an area of 1 unit, given four points inside or on the boundary of the triangle, 
there exist three points among them that form a triangle with area at most 1/3 unit. -/
theorem points_in_triangle_area {T : Type} [linear_ordered_field T] 
  (A B C P₁ P₂ P₃ P₄ : T × T) 
  (area_ABC_eq_one : triangle_area A B C = 1)
  (P₁_in_triangle : in_triangle A B C P₁)
  (P₂_in_triangle : in_triangle A B C P₂)
  (P₃_in_triangle : in_triangle A B C P₃)
  (P₄_in_triangle : in_triangle A B C P₄) :
  ∃ (X Y Z : T × T), {X, Y, Z} ⊆ {P₁, P₂, P₃, P₄} ∧ triangle_area X Y Z ≤ 1 / 3 := 
sorry

end points_in_triangle_area_l114_114732


namespace meal_distribution_exactly_two_correct_orders_l114_114095

theorem meal_distribution_exactly_two_correct_orders :
  ∀ (persons : Fin 12 → String), 
    (∀ i, persons i ∈ ["beef", "chicken", "fish", "vegetarian"] ∧ 
          (∀ (m : String), (persons.filter (λ x, x = m)).length = 3)) →
  (∃ g : Fin 12 → Fin 12, (∀ i, g i ≠ i) ∧ (∃ pair : Fin 12 × Fin 12, 
    pair.1 ≠ pair.2 ∧ g pair.1 = pair.1 ∧ g pair.2 = pair.2) ∧
    (∑ i, if g i = i then 1 else 0) = 2) →
  (number_of_ways_to_serve_meals persons = 990) :=
by
  sorry

end meal_distribution_exactly_two_correct_orders_l114_114095


namespace value_at_x12_l114_114578

def quadratic_function (d e f x : ℝ) : ℝ :=
  d * x^2 + e * x + f

def axis_of_symmetry (d e f : ℝ) : ℝ := 10.5

def point_on_graph (d e f : ℝ) : Prop :=
  quadratic_function d e f 3 = -5

theorem value_at_x12 (d e f : ℝ)
  (Hsymm : axis_of_symmetry d e f = 10.5)
  (Hpoint : point_on_graph d e f) :
  quadratic_function d e f 12 = -5 :=
sorry

end value_at_x12_l114_114578


namespace domain_of_g_l114_114939

def g (t : ℝ) : ℝ := 1 / ((t - 2)^2 + (t - 3)^2 - 2)

theorem domain_of_g :
  ∀ t : ℝ, t ≠ (5 - Real.sqrt 3)/2 ∧ t ≠ (5 + Real.sqrt 3)/2 →
  (1 / ((t - 2)^2 + (t - 3)^2 - 2)) = g t :=
by
  assume t h,
  unfold g,
  have d := (t - 2)^2 + (t - 3)^2 - 2,
  have h_d : d ≠ 0,
  {
    cases h with h1 h2,
    intro h_d_zero,
    rcases h_d_zero with rfl,
    contradiction,
  },
  rw [h_d],
  -- conditions ensure t not equal to root points
  sorry

end domain_of_g_l114_114939


namespace greatest_product_of_two_integers_with_sum_300_l114_114229

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l114_114229


namespace solve_for_a_l114_114945

def star (a b : ℤ) : ℤ := 3 * a - b^3

theorem solve_for_a (a : ℤ) : star a 3 = 18 → a = 15 := by
  intro h₁
  sorry

end solve_for_a_l114_114945


namespace max_product_two_integers_l114_114187

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l114_114187


namespace max_product_300_l114_114144

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l114_114144


namespace probability_sqrt_less_nine_l114_114513

theorem probability_sqrt_less_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  ∃ (p : ℚ), p = (finset.card (valid_numbers.to_finset) : ℚ) / (finset.card (two_digit_numbers.to_finset) : ℚ) ∧ p = 71 / 90 :=
by
  sorry

end probability_sqrt_less_nine_l114_114513


namespace probability_of_sqrt_lt_9_l114_114352

-- Define the set of two-digit whole numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the subset of numbers for which the square root is less than 9
def valid_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 80}

-- Define the probability calculation
noncomputable def probability_sqrt_lt_9 := (valid_numbers.to_finset.card : ℝ) / (two_digit_numbers.to_finset.card : ℝ)

-- The statement we aim to prove
theorem probability_of_sqrt_lt_9 : probability_sqrt_lt_9 = 71 / 90 := 
sorry

end probability_of_sqrt_lt_9_l114_114352


namespace BigMi_gets_treadmill_l114_114602

theorem BigMi_gets_treadmill :
  ∃ (gets_toy : Σ (BigMi SecondMi ThirdMi FourthMi FifthMi : Type), BigMi),
  (∀ {ToyFish YarnBall CatTeaser Treadmill MintBall : Type}, 
  ∀ (preference : Π (Mi : Σ (BigMi SecondMi ThirdMi FourthMi FifthMi : Type), Mi),
    (preference ⟨ThirdMi, ToyFish⟩ = false) ∧
    (preference ⟨ThirdMi, MintBall⟩ = false) ∧
    (preference ⟨FifthMi, CatTeaser⟩ = false) ∧
    (preference ⟨FifthMi, MintBall⟩ = false)) ∧
  ( ( ∃ (bigmi_secondmi_1 : BigMi → Type), ∃ (bigmi_secondmi_2 : SecondMi → Type), 
  bigmi_secondmi_1 = ToyFish ∧ bigmi_secondmi_2 = Treadmill ) ∨
  ( ( ∃ (bigmi_secondmi_1 : BigMi → Type), ∃ (bigmi_secondmi_2 : SecondMi → Type), 
  bigmi_secondmi_1 = Treadmill ∧ bigmi_secondmi_2 = ToyFish ))) ∧
  ( ∀ {CatTeaser Treadmill : Type}, ∀ (insufficient_exercise : Π (Mi : Σ FourthMi), Mi),
    (insufficient_exercise ⟨FourthMi, CatTeaser⟩ = false) ∧
    (insufficient_exercise ⟨FourthMi, Treadmill⟩ = false) → 
    ( ∃ (borrow : FourthMi → Type), ∃ (borrow_from : Σ (Mi : Σ (BigMi SecondMi ThirdMi FifthMi), Mi), 
    borrow = CatTeaser ∨ borrow = Treadmill ))) ∧
  ∀ (no_share_SecondMi : Π (Mi : Σ SecondMi, Mi), true = false) ∧
  ∀ (no_share_ThirdMi : Π (Mi : Σ ThirdMi, Mi), true = false)) ∧ 
  gets_toy.1.1.1.1.1 = Treadmill := sorry

end BigMi_gets_treadmill_l114_114602


namespace max_product_sum_300_l114_114312

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l114_114312


namespace angle_OQP_90_deg_l114_114808

variables {Point : Type} [affine_space Point]
variables (A B C D O P Q : Point) (circleO : circle O) 
variables (circABP : circle (⊔ {A, B, P})) 
variables (circCDP : circle (⊔ {C, D, P}))

-- Setup the problem with the given conditions:
-- 1. Convex quadrilateral ABCD is inscribed in a circle O.
-- 2. Diagonals AC and BD intersect at P.
-- 3. Circumcircles of triangles ABP and CDP intersect at P and also at another point Q.
-- 4. Points O, P, and Q are distinct.

def solution : Prop :=
  convex_quadrilateral_inscribed circleO ⟨A, B, C, D⟩ ∧
  diagonals_intersect ⟨A, B, C, D⟩ P ∧ 
  circumcircle_intersect circABP circCDP ⟨P, Q⟩ ∧ 
  distinct_points ⟨O, P, Q⟩ ∧ ∡ O Q P = 90

-- Prove the final solution:
theorem angle_OQP_90_deg (A B C D O P Q : Point) (circleO : circle O) 
  (circABP : circle (⊔ {A, B, P})) 
  (circCDP : circle (⊔ {C, D, P})) :
  convex_quadrilateral_inscribed circleO ⟨A, B, C, D⟩ ∧
  diagonals_intersect ⟨A, B, C, D⟩ P ∧ 
  circumcircle_intersect circABP circCDP ⟨P, Q⟩ ∧ 
  distinct_points ⟨O, P, Q⟩ →
  ∡ O Q P = 90 :=
sorry

end angle_OQP_90_deg_l114_114808


namespace find_counterfeit_min_weighings_l114_114089

def genuine_weights : List ℕ := [1, 2, 3, 5]

def is_counterfeit (weights : List ℕ) : Prop :=
  ∃ i, weights ≠ genuine_weights ∧ (weights[i] != genuine_weights[i])

theorem find_counterfeit_min_weighings (weights : List ℕ) (h : is_counterfeit weights) :
  ∃ n, n = 2 :=
by
  sorry

end find_counterfeit_min_weighings_l114_114089


namespace cyclic_ABQP_iff_AC_eq_BC_or_angle_ACB_eq_90_l114_114757

-- Define the type for points and the concept of a triangle
structure Point := (x : ℝ) (y : ℝ)
structure Triangle := (A B C : Point)

-- Definition of altitude from C to AB, with H located between A and B
def is_altitude (Δ : Triangle) (H : Point) := 
  Δ.A.y = Δ.B.y ∧ Δ.C.x = H.x ∧ H.is_between Δ.A Δ.B

-- Define incenters of triangles
def is_incenter (P H C A : Point) := -- definition of incenter conditions
def is_incenter (Q H C B : Point) := -- definition of incenter conditions

-- Define the concept of a cyclic quadrilateral
def cyclic (A B Q P : Point) := 
  let α := ∡APB
  let β := ∡AQB
  let γ := ∡BQP
  let δ := ∡QPA in
  α + γ = 180 ∧ β + δ = 180

-- Define the main theorem
theorem cyclic_ABQP_iff_AC_eq_BC_or_angle_ACB_eq_90
  (A B C H P Q : Point)
  (hH : is_altitude (Triangle.mk A B C) H)
  (hP : is_incenter P H C A)
  (hQ : is_incenter Q H C B) :
  cyclic A B Q P ↔ dist A C = dist B C ∨ ∡ACB = 90 :=
sorry

end cyclic_ABQP_iff_AC_eq_BC_or_angle_ACB_eq_90_l114_114757


namespace calculate_radius_l114_114758

structure Trapezoid :=
(EF FG HE GH : ℝ)
(H_trap : EF = 8 ∧ FG = 6 ∧ HE = 6 ∧ GH = 5)

structure Circle :=
(center : Point)
(radius : ℝ)

noncomputable def is_valid_radius (s : ℝ) : Prop :=
  let a := 72
  let b := 50
  let c := 3
  let d := 26
  s = (-72 + 50 * Real.sqrt 3) / 26 ∧ (a + b + c + d = 151)

theorem calculate_radius (EF FG HE GH : ℝ) 
  (H_trap : EF = 8 ∧ FG = 6 ∧ HE = 6 ∧ GH = 5)
  (E F G H : Point)
  (C1 C2 : Circle)
  (H_circles : C1.radius = 4 ∧ C2.radius = 3 ∧
               C1.center = E ∧ C2.center = F ∧
               C1.radius = 4 ∧ C2.radius = 3 ∧
               C1.center = G ∧ C2.center = H ∧) :
  ∃ s : ℝ, is_valid_radius s :=
by {
  sorry
}

end calculate_radius_l114_114758


namespace main_theorem_l114_114616

noncomputable def f (x : ℝ) := sorry

lemma functional_equation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  f (x * y) = f x * f (3 / y) + f y * f (3 / x) := sorry

lemma initial_condition : f 1 = 1 / 2 := sorry

theorem main_theorem : ∀ x : ℝ, 0 < x → f x = 1 / 2 :=
begin
  intros x hx,
  sorry
end

end main_theorem_l114_114616


namespace sam_drives_same_rate_as_marguerite_l114_114771

variable (distanceMarguerite : ℕ) (timeMarguerite : ℕ) (timeSamHours : ℕ)

def average_speed (d : ℕ) (t : ℕ) := d / t

theorem sam_drives_same_rate_as_marguerite :
  (distanceMarguerite = 120) →
  (timeMarguerite = 3) →
  (timeSamHours = 4) →
  let speedMarguerite := average_speed distanceMarguerite timeMarguerite in
  let timeSamMinutes := timeSamHours * 60 in
  let timeSamHoursConverted := timeSamMinutes / 60 in
  speedMarguerite * timeSamHoursConverted = 160 :=
by
  intros h1 h2 h3
  simp [average_speed] at *
  -- Here we assume h1 = 120, h2 = 3, h3 = 4
  have speedMarguerite : ℕ := 40
  simp [h1, h2, average_speed] at speedMarguerite
  have timeSamMinutes : ℕ := 240
  simp [h3] at timeSamMinutes
  have timeSamHoursConverted : ℕ := 4
  simp [timeSamMinutes] at timeSamHoursConverted
  simp [speedMarguerite, timeSamHoursConverted]
  sorry

end sam_drives_same_rate_as_marguerite_l114_114771


namespace max_product_of_sum_300_l114_114223

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l114_114223


namespace greatest_product_obtainable_l114_114322

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l114_114322


namespace contrapositive_of_square_inequality_l114_114059

theorem contrapositive_of_square_inequality (x y : ℝ) :
  (x^2 > y^2 → x > y) ↔ (x ≤ y → x^2 ≤ y^2) :=
by
  sorry

end contrapositive_of_square_inequality_l114_114059


namespace complement_intersection_l114_114697

open Set

namespace UniversalSetProof

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}

theorem complement_intersection :
  (U \ A) ∩ B = {4, 5} :=
by
  sorry

end UniversalSetProof

end complement_intersection_l114_114697


namespace max_product_of_sum_300_l114_114276

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l114_114276


namespace evening_ticket_price_l114_114839

-- Defining the conditions
def evening_ticket_cost (T : ℝ) : ℝ := T
def popcorn_drink_combo : ℝ := 10
def ticket_discount : ℝ := 0.2 * evening_ticket_cost T
def combo_discount : ℝ := 0.5 * popcorn_drink_combo
def savings_by_going_early : ℝ := 7

-- Special offer calculation
def special_offer_cost (T : ℝ) : ℝ := 0.8 * evening_ticket_cost T + 0.5 * popcorn_drink_combo

-- Evening cost
def evening_cost (T : ℝ) : ℝ := evening_ticket_cost T + popcorn_drink_combo

-- Proof Statement
theorem evening_ticket_price (T : ℝ) : T = 10 :=
  have ev_cost : ℝ := evening_cost T
  have sp_offer_cost : ℝ := special_offer_cost T
  (ev_cost - sp_offer_cost = savings_by_going_early) → (T = 10) 
  sorry

end evening_ticket_price_l114_114839


namespace max_value_a_eq_1_range_of_a_l114_114687

-- Define the function f(x) = ax + ln(x) where x ∈ [1, e]
def f (a x : ℝ) : ℝ := a * x + Real.log x

-- Question (Ⅰ): Prove that if a = 1, then the maximum value of f(x) for x ∈ [1, e] is e + 1
theorem max_value_a_eq_1 : ∀ (x : ℝ), (1 <= x) ∧ (x <= Real.exp 1) → f 1 x ≤ f 1 (Real.exp 1) := 
by
  sorry

-- Question (Ⅱ): Prove that the range of values of a such that f(x) ≤ 0 for all x ∈ [1, e] is a ≤ -1/e
theorem range_of_a : ∀ (a : ℝ), (∀ (x : ℝ), (1 <= x) ∧ (x <= Real.exp 1) → f a x ≤ 0) ↔ a ≤ -1 / (Real.exp 1) :=
by
  sorry

end max_value_a_eq_1_range_of_a_l114_114687


namespace sum_base7_l114_114914

def base7_to_base10 (n : ℕ) : ℕ := 
  -- Function to convert base 7 to base 10 (implementation not shown)
  sorry

def base10_to_base7 (n : ℕ) : ℕ :=
  -- Function to convert base 10 to base 7 (implementation not shown)
  sorry

theorem sum_base7 (a b : ℕ) (ha : a = base7_to_base10 12) (hb : b = base7_to_base10 245) :
  base10_to_base7 (a + b) = 260 :=
sorry

end sum_base7_l114_114914


namespace chips_probability_l114_114541

noncomputable def probability_of_consecutive_colors : ℚ := 1 / 50400

theorem chips_probability :
  let orange_chips := 5
  let green_chips := 3
  let blue_chips := 7
  let total_chips := orange_chips + green_chips + blue_chips
  (total_chips = 15) →
  (probability_of_consecutive_colors = (factorial orange_chips * factorial green_chips * factorial blue_chips * factorial 3 
    / factorial total_chips)) :=
by
  intros
  sorry

end chips_probability_l114_114541


namespace max_type_a_workers_l114_114551

theorem max_type_a_workers (x y : ℕ) (h1 : x + y = 150) (h2 : y ≥ 3 * x) : x ≤ 37 :=
sorry

end max_type_a_workers_l114_114551


namespace triangle_inequality_l114_114744

theorem triangle_inequality {A B C : ℝ} {n : ℕ} (h : B = n * C) (hA : A + B + C = π) :
  B ≤ n * C :=
by
  sorry

end triangle_inequality_l114_114744


namespace probability_sqrt_less_than_nine_is_correct_l114_114387

def probability_sqrt_less_than_nine : ℚ :=
  let total_two_digit_numbers := 99 - 10 + 1 in
  let satisfying_numbers := 80 - 10 + 1 in
  let probability := (satisfying_numbers : ℚ) / (total_two_digit_numbers : ℚ) in
  probability

theorem probability_sqrt_less_than_nine_is_correct :
  probability_sqrt_less_than_nine = 71 / 90 :=
by
  -- proof here
  sorry

end probability_sqrt_less_than_nine_is_correct_l114_114387


namespace xiaoming_problem_count_l114_114521

theorem xiaoming_problem_count :
  ∃ (problems : Finset ℕ), 
    problems.card = 10 ∧
    ∃ (wrong_problems : Finset ℕ), 
      wrong_problems.card = 3 ∧
      ∀ x ∈ wrong_problems, 
        x < 10 ∧ 
        ∀ y ∈ wrong_problems, 
          x ≠ y →
          abs (x - y) ≠ 1 →
          (Finset.card wrong_problems = 3 ∧ 
           (Finset.card (Finset.range 10) = 10)) →
          Finset.choose (8, 3) = 56 :=
begin
  sorry
end

end xiaoming_problem_count_l114_114521


namespace probability_sqrt_lt_nine_two_digit_l114_114487

theorem probability_sqrt_lt_nine_two_digit :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99} in
  let T := {n : ℕ | 10 ≤ n ∧ n < 81} in
  (Fintype.card T : ℚ) / (Fintype.card S) = 71 / 90 :=
by
  sorry

end probability_sqrt_lt_nine_two_digit_l114_114487


namespace distinct_sums_l114_114650

theorem distinct_sums (n : ℕ) (a : Fin n → ℕ) (h_distinct : Function.Injective a) :
  ∃ S : Finset ℕ, S.card ≥ n * (n + 1) / 2 :=
by
  sorry

end distinct_sums_l114_114650


namespace greatest_product_of_sum_eq_300_l114_114116

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l114_114116


namespace probability_sqrt_less_than_nine_l114_114444

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sqrt_less_than_nine (n : ℕ) : Prop := n < 81

theorem probability_sqrt_less_than_nine :
  (∃ total_num favorable_num,
    (total_num = Finset.card (Finset.filter is_two_digit (Finset.range 100)) ∧
     favorable_num = Finset.card (Finset.filter (λ n, is_two_digit n ∧ sqrt_less_than_nine n) (Finset.range 100)) ∧
     (favorable_num : ℚ) / total_num = 71 / 90)) :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114444


namespace probability_sqrt_two_digit_lt_nine_correct_l114_114431

noncomputable def probability_sqrt_two_digit_lt_nine : ℚ :=
  let two_digit_integers := finset.Icc 10 99
  let satisfying_integers := two_digit_integers.filter (λ n => n < 81)
  let probability := (satisfying_integers.card : ℚ) / (two_digit_integers.card : ℚ)
  probability

theorem probability_sqrt_two_digit_lt_nine_correct :
  probability_sqrt_two_digit_lt_nine = 71 / 90 := by
  sorry

end probability_sqrt_two_digit_lt_nine_correct_l114_114431


namespace camping_trip_percentage_l114_114728

theorem camping_trip_percentage (students : ℕ)
  (camping_trip_ratio music_festival_ratio sports_league_ratio : ℕ → ℝ)
  (h1 : camping_trip_ratio students = 0.14)
  (h2 : music_festival_ratio students = 0.08)
  (h3 : sports_league_ratio students = 0.06)
  (camping_trip_expense : ℕ → ℝ)
  (h4_1 : ∀ x, x > 0.6 * (camping_trip_ratio students * students) → camping_trip_expense x > 100)
  (h4_2 : ∀ x, x ≤ 0.4 * (camping_trip_ratio students * students) → camping_trip_expense x < 100)
  (music_festival_expense : ℕ → ℝ)
  (h5_1 : ∀ x, x > 0.8 * (music_festival_ratio students * students) → music_festival_expense x > 90)
  (h5_2 : ∀ x, x ≤ 0.2 * (music_festival_ratio students * students) → music_festival_expense x < 90)
  (sports_league_expense : ℕ → ℝ)
  (h6_1 : ∀ x, x > 0.75 * (sports_league_ratio students * students) → sports_league_expense x > 70)
  (h6_2 : ∀ x, x ≤ 0.25 * (sports_league_ratio students * students) → sports_league_expense x < 70) :
  camping_trip_ratio students = 0.14 :=
by
  sorry

end camping_trip_percentage_l114_114728


namespace probability_sqrt_lt_nine_two_digit_l114_114491

theorem probability_sqrt_lt_nine_two_digit :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99} in
  let T := {n : ℕ | 10 ≤ n ∧ n < 81} in
  (Fintype.card T : ℚ) / (Fintype.card S) = 71 / 90 :=
by
  sorry

end probability_sqrt_lt_nine_two_digit_l114_114491


namespace probability_sqrt_less_than_nine_l114_114468

theorem probability_sqrt_less_than_nine :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let T := {n : ℕ | n ∈ S ∧ n < 81}
  (T.card : ℚ) / S.card = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114468


namespace correct_propositions_l114_114978

noncomputable def distinct_planes (α β : Type*) : Prop := 
α ≠ β

noncomputable def distinct_lines (m n : Type*) : Prop := 
m ≠ n

noncomputable def perp (a b : Type*) : Prop := sorry -- Definition for perpendicularity

noncomputable def parallel (a b : Type*) : Prop := sorry -- Definition for parallelism

noncomputable def subset (l : Type*) (p : Type*) : Prop := sorry -- Definition for subset

noncomputable def intersection (a b : Type*) : Type* := sorry -- Definition for intersection

theorem correct_propositions (α β m n : Type*) :
  distinct_planes α β →
  distinct_lines m n →
  (perp m α ∧ perp n α → parallel m n) ∧
  (subset m α ∧ subset n α ∧ parallel m β ∧ parallel n β → ¬ parallel α β) ∧
  (perp α β ∧ intersection α β = m ∧ subset n α ∧ perp n m → perp n β) ∧
  (perp m α ∧ perp α β ∧ parallel m n → ¬ parallel n β) :=
by sorry

end correct_propositions_l114_114978


namespace prob_sqrt_less_than_nine_l114_114477

/-- The probability that the square root of a randomly selected 
two-digit whole number is less than nine is 71/90. -/
theorem prob_sqrt_less_than_nine : (let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99};
                                     let A := {n : ℕ | 10 ≤ n ∧ n < 81};
                                     (A.card / S.card : ℚ) = 71 / 90) :=
by
  sorry

end prob_sqrt_less_than_nine_l114_114477


namespace trigonometric_identity_l114_114585

theorem trigonometric_identity :
  tan 70 * cos 10 * (sqrt 3 * tan 20 - 1) = -1 :=
sorry

end trigonometric_identity_l114_114585


namespace expansion_correct_l114_114609

noncomputable def expandExpression (x : ℝ) : ℝ :=
  (2 * x ^ 15 - 5 * x ^ 8 + 4 * x ^ (-3) - 9) * (-3 * x ^ 6)

theorem expansion_correct (x : ℝ) : 
  expandExpression x = -6 * x ^ 21 + 15 * x ^ 14 + 27 * x ^ 6 - 12 * x ^ 3 := 
by
  sorry

end expansion_correct_l114_114609


namespace greatest_product_two_ints_sum_300_l114_114263

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l114_114263


namespace trigonometric_identity_l114_114883

theorem trigonometric_identity :
  let deg30 := real.pi * 30 / 180
  let deg60 := real.pi * 60 / 180
  \cos (7 * real.pi / 3) + \sin (11 * real.pi / 6) = 0 :=
by
  let deg30 := real.pi * 30 / 180
  let deg60 := real.pi * 60 / 180
  have h1 : \(\cos (7 * real.pi / 3) = \cos deg60\) := by sorry
  have h2 : \(\sin (11 * real.pi / 6) = -\sin deg30\) := by sorry
  have h3 : \(\cos deg60 = 1 / 2\) := by sorry
  have h4 : \(\sin deg30 = 1 / 2\) := by sorry
  calc
    \(\cos (7 * real.pi / 3) + \sin (11 * real.pi / 6) = \(\cos deg60 + \sin (11 * real.pi / 6))\ := by rw[h1]
    ... = \(\cos deg60 - \sin deg30\ := by rw[h2]
    ... = \(\1 / 2 - 1 / 2\ := by rw[h3, h4]
    ... = 0 := by simp

end trigonometric_identity_l114_114883


namespace negation_of_p_l114_114666

-- Define the proposition p
def p : Prop := ∃ n : ℕ, 2^n > 100

-- Goal is to show the negation of p
theorem negation_of_p : (¬ p) = (∀ n : ℕ, 2^n ≤ 100) :=
by
  sorry

end negation_of_p_l114_114666


namespace max_product_two_integers_sum_300_l114_114156

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l114_114156


namespace probability_sqrt_less_nine_l114_114510

theorem probability_sqrt_less_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  ∃ (p : ℚ), p = (finset.card (valid_numbers.to_finset) : ℚ) / (finset.card (two_digit_numbers.to_finset) : ℚ) ∧ p = 71 / 90 :=
by
  sorry

end probability_sqrt_less_nine_l114_114510


namespace find_theta_l114_114621

theorem find_theta :
  ∃ θ : ℝ, 0 < θ ∧ θ < 90 ∧ θ = 80 ∧ 
  (cos 10° = sin 30° + sin θ) ∧ (sin 30° = 1 / 2) ∧ 
  (∀ x : ℝ, cos x = sin (90 - x)) ∧ (cos 10° = sin 80°) ∧ 
  (sin 80° - (1 / 2) = (sqrt 3 / 2) - (1 / 2)) := sorry

end find_theta_l114_114621


namespace point_outside_circle_OP_gt_5cm_l114_114677

theorem point_outside_circle_OP_gt_5cm (d : ℝ) (r : ℝ) (O P : Type) 
  [circle: Type] [diameter: d = 10] [radius: r = d / 2] 
  (point_outside: P → circle → Prop) : d = 10 ∧ point_outside P O → ∃ OP : ℝ, OP = 6 ∧ OP > 5 := 
by
  -- The proof would go here.
  sorry

end point_outside_circle_OP_gt_5cm_l114_114677


namespace intersection_M_N_l114_114977

noncomputable def M : set ℝ := {x | x^2 ≤ 4}
noncomputable def N : set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem intersection_M_N :
  (M ∩ N) = {x | 1 < x ∧ x ≤ 2} :=
by sorry

end intersection_M_N_l114_114977


namespace expressions_equality_l114_114707

theorem expressions_equality (x : ℝ) (h : x > 0) :
  (Count ({4*x^x} ∪ {x^(4*x)} ∪ {3*x^x} ∪ {(3*x)^(2*x)}) (λ e, e = 3*x^x + x^x)) = 1 :=
by sorry

end expressions_equality_l114_114707


namespace probability_sqrt_less_than_nine_is_correct_l114_114385

def probability_sqrt_less_than_nine : ℚ :=
  let total_two_digit_numbers := 99 - 10 + 1 in
  let satisfying_numbers := 80 - 10 + 1 in
  let probability := (satisfying_numbers : ℚ) / (total_two_digit_numbers : ℚ) in
  probability

theorem probability_sqrt_less_than_nine_is_correct :
  probability_sqrt_less_than_nine = 71 / 90 :=
by
  -- proof here
  sorry

end probability_sqrt_less_than_nine_is_correct_l114_114385


namespace max_product_of_two_integers_whose_sum_is_300_l114_114166

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l114_114166


namespace probability_sqrt_lt_9_l114_114423

theorem probability_sqrt_lt_9 : 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 →
  ∃ p : ℚ, p = 71 / 90 ∧ 
  ∑ k in (Finset.range 100).filter (λ x, 10 ≤ x ∧ sqrt x < 9), 1 / 90 = p := 
sorry

end probability_sqrt_lt_9_l114_114423


namespace max_product_two_integers_sum_300_l114_114159

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l114_114159


namespace max_product_two_integers_l114_114183

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l114_114183


namespace probability_sqrt_lt_nine_two_digit_l114_114486

theorem probability_sqrt_lt_nine_two_digit :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99} in
  let T := {n : ℕ | 10 ≤ n ∧ n < 81} in
  (Fintype.card T : ℚ) / (Fintype.card S) = 71 / 90 :=
by
  sorry

end probability_sqrt_lt_nine_two_digit_l114_114486


namespace tangent_and_normal_lines_l114_114624

-- Parametric equations
def x (t : ℝ) : ℝ := t^3 + 1
def y (t : ℝ) : ℝ := t^2

-- Given parameter value
def t0 : ℝ := -2

-- Correct answers
def tangent_line_equation : (ℝ × ℝ) → Prop := 
  λ p, p.2 = -((1:ℝ)/3) * p.1 + (5:ℝ)/3

def normal_line_equation : (ℝ × ℝ) → Prop := 
  λ p, p.2 = 3 * p.1 + 25

theorem tangent_and_normal_lines :
  ∃ (p : ℝ × ℝ), 
  (tangent_line_equation p) ∧ 
  (normal_line_equation p) ∧ 
  (p = (x t0, y t0)) :=
sorry

end tangent_and_normal_lines_l114_114624


namespace max_product_two_integers_sum_300_l114_114151

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l114_114151


namespace diamond_19_98_l114_114664

variable {R : Type} [LinearOrderedField R]

noncomputable def diamond (x y : R) : R := sorry

axiom diamond_axiom1 : ∀ (x y : R) (hx : 0 < x) (hy : 0 < y), diamond (x * y) y = x * (diamond y y)

axiom diamond_axiom2 : ∀ (x : R) (hx : 0 < x), diamond (diamond x 1) x = diamond x 1

axiom diamond_axiom3 : diamond 1 1 = 1

theorem diamond_19_98 : diamond (19 : R) (98 : R) = 19 := 
sorry

end diamond_19_98_l114_114664


namespace smallest_n_terminating_contains_9_divisible_by_3_l114_114628

theorem smallest_n_terminating_contains_9_divisible_by_3 :
  ∃ n : ℕ, (∀ m : ℕ, (∃ k1 k2 k3 : ℕ, m = 2^k1 * 5^k2 * 3^k3 ∧ m ≠ 0 ∧ (0 < k3) ∧ (∃ d : ℕ, 10 ^ d > m ∧ m * (10 ^ d) % 10^d = 0) ∧ ('9' ∈ to_digits 10 m)) → n ≤ m) ∧
  (∃ k1 k2 : ℕ, n = 2^k1 * 5^k2 * 3 ∧ '9' ∈ to_digits 10 n ∧ n = 96) := sorry

end smallest_n_terminating_contains_9_divisible_by_3_l114_114628


namespace distinct_license_plates_l114_114894

theorem distinct_license_plates :
  let digit_choices := 10
  let letter_choices := 26
  let positions := 7
  let total := positions * (digit_choices ^ 6) * (letter_choices ^ 3)
  total = 122504000 :=
by
  -- Definitions from the conditions
  let digit_choices := 10
  let letter_choices := 26
  let positions := 7
  -- Calculation
  let total := positions * (digit_choices ^ 6) * (letter_choices ^ 3)
  -- Assertion
  have h : total = 122504000 := sorry
  exact h

end distinct_license_plates_l114_114894


namespace number_of_elements_satisfying_condition_l114_114695

theorem number_of_elements_satisfying_condition :
  let A := {x : Fin 10 → ℤ | ∀ i, x i ∈ {-1, 0, 1}} in
  let S := {x ∈ A | 1 ≤ (Finset.univ.sum (λ i, |x i|)) ∧ (Finset.univ.sum (λ i, |x i|)) ≤ 9} in
  S.card = 3^10 - 2^10 - 1 :=
by
  let A := {x : Fin 10 → ℤ | ∀ i, x i ∈ {-1, 0, 1}}
  let S := {x ∈ A | 1 ≤ (Finset.univ.sum (λ i, |x i|)) ∧ (Finset.univ.sum (λ i, |x i|)) ≤ 9}
  exact sorry

end number_of_elements_satisfying_condition_l114_114695


namespace sym_diff_A_B_l114_114995

open Set

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

-- Definition of the symmetric difference
def sym_diff (A B : Set ℕ) : Set ℕ := {x | (x ∈ A ∨ x ∈ B) ∧ x ∉ (A ∩ B)}

theorem sym_diff_A_B : sym_diff A B = {0, 3} := 
by 
  sorry

end sym_diff_A_B_l114_114995


namespace min_x_y_l114_114886

theorem min_x_y (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (h_eq : 2 / x + 8 / y = 1) : x + y ≥ 18 := 
sorry

end min_x_y_l114_114886


namespace max_product_of_sum_300_l114_114282

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l114_114282


namespace calculate_festival_allowance_days_l114_114729

-- Define the given conditions as constants
constant num_staff : ℕ := 20
constant daily_rate : ℕ := 100
constant amount_given : ℕ := 65000
constant petty_cash : ℕ := 1000

-- Express the total amount available for distribution
def total_amount : ℕ := amount_given + petty_cash

-- Define the number of days for which the festival allowance is calculated
variable d : ℕ

-- The target statement to prove
theorem calculate_festival_allowance_days :
  20 * 100 * d + 1000 = total_amount → d = 32 :=
by
  sorry

end calculate_festival_allowance_days_l114_114729


namespace difference_counts_l114_114637

noncomputable def τ (n : ℕ) : ℕ := n.factors.length + 1

noncomputable def S (n : ℕ) : ℕ := (list.range n.succ).sum (λ i, τ i)

def count_odd_S_up_to (n : ℕ) : ℕ :=
(list.range n.succ).countp (λ i, odd (S i))

def count_even_S_up_to (n : ℕ) : ℕ :=
(list.range n.succ).countp (λ i, even (S i))

theorem difference_counts {c d : ℕ} :
  c = count_odd_S_up_to 3000 →
  d = count_even_S_up_to 3000 →
  |c - d| = 1733 :=
by
  intros h_c h_d
  sorry

end difference_counts_l114_114637


namespace max_product_two_integers_l114_114181

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l114_114181


namespace probability_of_sqrt_lt_9_l114_114348

-- Define the set of two-digit whole numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the subset of numbers for which the square root is less than 9
def valid_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 80}

-- Define the probability calculation
noncomputable def probability_sqrt_lt_9 := (valid_numbers.to_finset.card : ℝ) / (two_digit_numbers.to_finset.card : ℝ)

-- The statement we aim to prove
theorem probability_of_sqrt_lt_9 : probability_sqrt_lt_9 = 71 / 90 := 
sorry

end probability_of_sqrt_lt_9_l114_114348


namespace tangent_line_at_zero_l114_114812

noncomputable def f (x : ℝ) : ℝ := 2 * cos x + 3

def point : ℝ × ℝ := (0, 5)

theorem tangent_line_at_zero : ∀ x : ℝ, (f x)' 0 = 0 → ∀ y : ℝ, y - 5 = 0 :=
by
  sorry

end tangent_line_at_zero_l114_114812


namespace max_product_sum_300_l114_114308

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l114_114308


namespace infinite_p_n_eq_p_n_plus_1_l114_114033

open Int

def largest_prime_divisor (n : ℤ) : ℤ :=
  sorry

def p (n : ℤ) : ℤ := largest_prime_divisor (n^4 + n^2 + 1)

def q (n : ℤ) : ℤ := largest_prime_divisor (n^2 + n + 1)

def S := { n : ℤ | n ≥ 2 ∧ q n > q (n - 1) ∧ q n > q (n + 1) }

theorem infinite_p_n_eq_p_n_plus_1 : ∃ᶠ n in (nat.filter fun n : ℤ => n > 0), p n = p (n + 1) :=
  sorry

end infinite_p_n_eq_p_n_plus_1_l114_114033


namespace probability_sqrt_lt_9_of_two_digit_l114_114360

-- Define the set of two-digit whole numbers
def two_digit_whole_numbers : set ℕ := {n | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate that checks if the square root of a number is less than 9
def sqrt_lt_9 (n : ℕ) : Prop := (n : ℝ)^2 < (9 : ℝ)^2

-- Calculate the probability
theorem probability_sqrt_lt_9_of_two_digit :
  let eligible_numbers := { n ∈ two_digit_whole_numbers | sqrt_lt_9 n } in
  (eligible_numbers.to_finset.card : ℚ) / (two_digit_whole_numbers.to_finset.card : ℚ) =
  71 / 90 :=
by
  sorry

end probability_sqrt_lt_9_of_two_digit_l114_114360


namespace probability_sqrt_less_nine_l114_114500

theorem probability_sqrt_less_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  ∃ (p : ℚ), p = (finset.card (valid_numbers.to_finset) : ℚ) / (finset.card (two_digit_numbers.to_finset) : ℚ) ∧ p = 71 / 90 :=
by
  sorry

end probability_sqrt_less_nine_l114_114500


namespace probability_sqrt_lt_9_of_two_digit_l114_114362

-- Define the set of two-digit whole numbers
def two_digit_whole_numbers : set ℕ := {n | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate that checks if the square root of a number is less than 9
def sqrt_lt_9 (n : ℕ) : Prop := (n : ℝ)^2 < (9 : ℝ)^2

-- Calculate the probability
theorem probability_sqrt_lt_9_of_two_digit :
  let eligible_numbers := { n ∈ two_digit_whole_numbers | sqrt_lt_9 n } in
  (eligible_numbers.to_finset.card : ℚ) / (two_digit_whole_numbers.to_finset.card : ℚ) =
  71 / 90 :=
by
  sorry

end probability_sqrt_lt_9_of_two_digit_l114_114362


namespace solve_linear_system_l114_114579

theorem solve_linear_system :
  ∃ x y : ℚ, (3 * x - y = 4) ∧ (6 * x - 3 * y = 10) ∧ (x = 2 / 3) ∧ (y = -2) :=
by
  sorry

end solve_linear_system_l114_114579


namespace percentage_area_covered_by_pentagons_l114_114730

theorem percentage_area_covered_by_pentagons :
  ∀ (a : ℝ), (∃ (large_square_area small_square_area pentagon_area : ℝ),
    large_square_area = 16 * a^2 ∧
    small_square_area = a^2 ∧
    pentagon_area = 10 * small_square_area ∧
    (pentagon_area / large_square_area) * 100 = 62.5) :=
sorry

end percentage_area_covered_by_pentagons_l114_114730


namespace probability_sqrt_lt_nine_two_digit_l114_114493

theorem probability_sqrt_lt_nine_two_digit :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99} in
  let T := {n : ℕ | 10 ≤ n ∧ n < 81} in
  (Fintype.card T : ℚ) / (Fintype.card S) = 71 / 90 :=
by
  sorry

end probability_sqrt_lt_nine_two_digit_l114_114493


namespace greatest_product_two_ints_sum_300_l114_114262

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l114_114262


namespace slope_range_l114_114825

def inclination_range (θ : ℝ) : Prop := 60 * real.pi / 180 ≤ θ ∧ θ ≤ 135 * real.pi / 180

theorem slope_range (θ : ℝ) (h : inclination_range θ) : 
  let k := real.tan θ in k ∈ set.Iic (-1) ∪ set.Ici (real.sqrt 3) :=
sorry

end slope_range_l114_114825


namespace probability_of_sqrt_lt_9_l114_114350

-- Define the set of two-digit whole numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the subset of numbers for which the square root is less than 9
def valid_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 80}

-- Define the probability calculation
noncomputable def probability_sqrt_lt_9 := (valid_numbers.to_finset.card : ℝ) / (two_digit_numbers.to_finset.card : ℝ)

-- The statement we aim to prove
theorem probability_of_sqrt_lt_9 : probability_sqrt_lt_9 = 71 / 90 := 
sorry

end probability_of_sqrt_lt_9_l114_114350


namespace max_product_300_l114_114143

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l114_114143


namespace number_division_l114_114866

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end number_division_l114_114866


namespace probability_sqrt_less_than_nine_l114_114470

theorem probability_sqrt_less_than_nine :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let T := {n : ℕ | n ∈ S ∧ n < 81}
  (T.card : ℚ) / S.card = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114470


namespace total_bones_l114_114873

variable (x y z : ℕ)

def xiao_ha : Prop := z = 2 * y + 2
def xiao_shi : Prop := y = 3 * x + 3
def xiao_qi : Prop := z = 7 * x - 5

theorem total_bones (h1: xiao_ha x y z) (h2: xiao_shi x y) (h3: xiao_qi x z) : 
  x + y + z = 141 := by
  sorry

end total_bones_l114_114873


namespace probability_sqrt_lt_nine_l114_114404

theorem probability_sqrt_lt_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  let probability := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  probability = 71 / 90 :=
by
  let two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n : ℕ | 10 ≤ n ∧ n < 81}
  have h1 : two_digit_numbers.card = 90 := sorry
  have h2 : valid_numbers.card = 71 := sorry
  let probability : ℚ := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  rw [h1, h2]
  simp
  norm_num

end probability_sqrt_lt_nine_l114_114404


namespace solution1_solution2_l114_114926

open Complex

noncomputable def problem1 : Prop := 
  ((3 - I) / (1 + I)) ^ 2 = -3 - 4 * I

noncomputable def problem2 (z : ℂ) : Prop := 
  z = 1 + I → (2 / z - z = -2 * I)

theorem solution1 : problem1 := 
  by sorry

theorem solution2 : problem2 (1 + I) :=
  by sorry

end solution1_solution2_l114_114926


namespace greatest_product_of_two_integers_with_sum_300_l114_114226

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l114_114226


namespace exists_infinite_line_points_l114_114016

theorem exists_infinite_line_points {N : Type}
  (hN : infinite N)
  (f : N → ℕ)
  (hf : ∀ x : N, f x + f (x + 2) ≤ 2 * f (x + 1)) :
  ∃ c b : ℤ, ∃ (n : infinite N) (f n: N → Prop), infinite ((λ n => (n, f n)) (n ∈ N) ∈ line c b)) :=
sorry

end exists_infinite_line_points_l114_114016


namespace subtracting_negative_calculation_l114_114584

theorem subtracting_negative (a b : ℤ) : a - (-b) = a + b := by
  sorry

theorem calculation : 2 - (-3) = 5 := by
  exact subtracting_negative 2 3
  sorry

end subtracting_negative_calculation_l114_114584


namespace probability_sqrt_lt_9_l114_114418

theorem probability_sqrt_lt_9 : 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 →
  ∃ p : ℚ, p = 71 / 90 ∧ 
  ∑ k in (Finset.range 100).filter (λ x, 10 ≤ x ∧ sqrt x < 9), 1 / 90 = p := 
sorry

end probability_sqrt_lt_9_l114_114418


namespace probability_red_or_white_l114_114524

def total_marbles : ℕ := 50
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9
def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

theorem probability_red_or_white : 
  (red_marbles + white_marbles) / total_marbles = 9 / 10 := 
  sorry

end probability_red_or_white_l114_114524


namespace slope_angle_135_l114_114948

theorem slope_angle_135 (x y : ℝ) : 
  (∃ (m b : ℝ), 3 * x + 3 * y + 1 = 0 ∧ y = m * x + b ∧ m = -1) ↔ 
  (∃ α : ℝ, 0 ≤ α ∧ α < 180 ∧ Real.tan α = -1 ∧ α = 135) :=
sorry

end slope_angle_135_l114_114948


namespace focal_length_of_ellipse_l114_114813

noncomputable def focal_length (a b : ℝ) : ℝ :=
  2 * real.sqrt (a^2 - b^2)

-- Define the constants from the ellipse equation
def ellipse_a : ℝ := 3
def ellipse_b : ℝ := 2

-- Statement of the proof problem:
theorem focal_length_of_ellipse : focal_length ellipse_a ellipse_b = 2 * real.sqrt 5 := by
  unfold focal_length ellipse_a ellipse_b
  sorry

end focal_length_of_ellipse_l114_114813


namespace probability_sqrt_less_than_nine_l114_114455

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sqrt_less_than_nine (n : ℕ) : Prop := n < 81

theorem probability_sqrt_less_than_nine :
  (∃ total_num favorable_num,
    (total_num = Finset.card (Finset.filter is_two_digit (Finset.range 100)) ∧
     favorable_num = Finset.card (Finset.filter (λ n, is_two_digit n ∧ sqrt_less_than_nine n) (Finset.range 100)) ∧
     (favorable_num : ℚ) / total_num = 71 / 90)) :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114455


namespace ratio_of_side_length_to_radius_l114_114549

theorem ratio_of_side_length_to_radius (r s : ℝ) (c d : ℝ) 
  (h1 : s = 2 * r)
  (h2 : s^2 = (c / d) * (s^2 - π * r^2)) : 
  (s / r) = (Real.sqrt (c * π) / Real.sqrt (d - c)) := by
  sorry

end ratio_of_side_length_to_radius_l114_114549


namespace max_product_sum_300_l114_114310

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l114_114310


namespace find_a_max_b_num_zeros_l114_114648

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.exp x

-- Statement for part (I)
theorem find_a :
  (∀ x : ℝ, f 1 x = 1 * x * Real.exp x) ∧ 
  f (0,0).derf = (0,0).derf := b (a = 1)
:= sorry

-- Define the function g(x)
def g (a b : ℝ) (x : ℝ) : ℝ := f a x - b * (x^2 / 2 + x)

-- Statement for part (II) (i)
theorem max_b (a : ℝ) (h : ∀ x : ℝ, f a x ≥ 0):
    (∀ x ∈ set.Ici (0 : ℝ), g a 1 x ≥ 0) → 1 = max (b : ℝ)
:= sorry

 -- Statement for part (II) (ii)
theorem num_zeros (a b : ℝ) :
    (b ≤ 0 → g a b ≥ 0)
:= sorry

end find_a_max_b_num_zeros_l114_114648


namespace probability_sqrt_less_than_nine_l114_114452

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sqrt_less_than_nine (n : ℕ) : Prop := n < 81

theorem probability_sqrt_less_than_nine :
  (∃ total_num favorable_num,
    (total_num = Finset.card (Finset.filter is_two_digit (Finset.range 100)) ∧
     favorable_num = Finset.card (Finset.filter (λ n, is_two_digit n ∧ sqrt_less_than_nine n) (Finset.range 100)) ∧
     (favorable_num : ℚ) / total_num = 71 / 90)) :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114452


namespace max_product_of_sum_300_l114_114284

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l114_114284


namespace sqrt_23_point_1_approx_nearest_whole_l114_114605

theorem sqrt_23_point_1_approx_nearest_whole :
  Real.sqrt 23.1 ≈ 5 := by
  sorry

end sqrt_23_point_1_approx_nearest_whole_l114_114605


namespace max_product_two_integers_l114_114193

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l114_114193


namespace probability_sqrt_less_than_nine_l114_114345

/-- Define the set of two-digit integers --/
def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Define the condition that the square root of the number is less than 9 --/
def sqrt_less_than_nine (n : Nat) : Prop := n < 81

/-- The number of integers from 10 to 80 --/
lemma count_satisfying_sqrt (n : Nat) : Prop :=
  is_two_digit n ∧ sqrt_less_than_nine n → n < 81

/-- Total number of two-digit integers --/
lemma count_two_digit_total (n : Nat) : Prop := is_two_digit n 

/-- The probability that a randomly selected two-digit integer's square root is less than 9. --/
theorem probability_sqrt_less_than_nine : 
  (∃ n, count_satisfying_sqrt n) / (∃ n, count_two_digit_total n) = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114345


namespace greatest_product_two_ints_sum_300_l114_114264

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l114_114264


namespace probability_sqrt_less_than_nine_l114_114460

theorem probability_sqrt_less_than_nine :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let T := {n : ℕ | n ∈ S ∧ n < 81}
  (T.card : ℚ) / S.card = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114460


namespace probability_sqrt_two_digit_lt_nine_correct_l114_114434

noncomputable def probability_sqrt_two_digit_lt_nine : ℚ :=
  let two_digit_integers := finset.Icc 10 99
  let satisfying_integers := two_digit_integers.filter (λ n => n < 81)
  let probability := (satisfying_integers.card : ℚ) / (two_digit_integers.card : ℚ)
  probability

theorem probability_sqrt_two_digit_lt_nine_correct :
  probability_sqrt_two_digit_lt_nine = 71 / 90 := by
  sorry

end probability_sqrt_two_digit_lt_nine_correct_l114_114434


namespace probability_sqrt_lt_9_l114_114394

theorem probability_sqrt_lt_9 : 
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
in probability = 71 / 90 :=
by
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  sorry

end probability_sqrt_lt_9_l114_114394


namespace complex_number_purely_imaginary_l114_114721

theorem complex_number_purely_imaginary (m : ℝ) :
  (m^2 - 2 * m - 3 = 0) ∧ (m^2 - 1 ≠ 0) → m = 3 :=
by
  intros h
  sorry

end complex_number_purely_imaginary_l114_114721


namespace greatest_product_l114_114295

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l114_114295


namespace greatest_product_of_sum_eq_300_l114_114114

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l114_114114


namespace exists_infinite_set_of_lines_l114_114601

noncomputable def infinite_set_of_lines (M : Set (ℝ × ℝ → ℝ)) : Prop :=
  (∀ l₁ l₂ ∈ M, l₁ ≠ l₂ → ∃ p, l₁ p ≠ l₂ p) ∧
  (∀ p : ℤ × ℤ, ∃ l ∈ M, l (.fst p) = .snd p)

theorem exists_infinite_set_of_lines :
  ∃ M : Set (ℝ × ℝ → ℝ), infinite_set_of_lines M :=
sorry

end exists_infinite_set_of_lines_l114_114601


namespace probability_sqrt_less_than_nine_l114_114446

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sqrt_less_than_nine (n : ℕ) : Prop := n < 81

theorem probability_sqrt_less_than_nine :
  (∃ total_num favorable_num,
    (total_num = Finset.card (Finset.filter is_two_digit (Finset.range 100)) ∧
     favorable_num = Finset.card (Finset.filter (λ n, is_two_digit n ∧ sqrt_less_than_nine n) (Finset.range 100)) ∧
     (favorable_num : ℚ) / total_num = 71 / 90)) :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114446


namespace cups_per_serving_l114_114900

theorem cups_per_serving (total_cups servings : ℝ) (h1 : total_cups = 36) (h2 : servings = 18.0) :
  total_cups / servings = 2 :=
by 
  sorry

end cups_per_serving_l114_114900


namespace general_formula_sum_of_b_l114_114991

noncomputable def a (n : ℕ) : ℕ := 2 * n

def b (n : ℕ) : ℚ := 4 / (a n * a (n+1))

def S (n : ℕ) : ℚ := ∑ k in Finset.range n, b k

theorem general_formula (n : ℕ) : a n = 2 * n := by 
  sorry

theorem sum_of_b (n : ℕ) : S n = n / (n + 1) := by
  sorry

end general_formula_sum_of_b_l114_114991


namespace number_division_l114_114869

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end number_division_l114_114869


namespace probability_sqrt_less_nine_l114_114511

theorem probability_sqrt_less_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  ∃ (p : ℚ), p = (finset.card (valid_numbers.to_finset) : ℚ) / (finset.card (two_digit_numbers.to_finset) : ℚ) ∧ p = 71 / 90 :=
by
  sorry

end probability_sqrt_less_nine_l114_114511


namespace g_one_value_l114_114019

-- Define the polynomial f(x) with the given coefficients
def f (a b c d : ℝ) (x : ℝ) := x^4 + a*x^3 + b*x^2 + c*x + d

-- Define the polynomial g(x) with leading coefficient 1 and roots being the reciprocals of the roots of f(x)
def g (p q r s : ℝ) (x : ℝ) := (x - (1 / p)) * (x - (1 / q)) * (x - (1 / r)) * (x - (1 / s))

theorem g_one_value (a b c d : ℝ) (h : 1 < a ∧ a < b ∧ b < c ∧ c < d) :
  ∃ (p q r s : ℝ), g p q r s 1 = (1 + a + b + c + d) / d :=
by
  sorry

end g_one_value_l114_114019


namespace problem_part_i_problem_part_ii_l114_114018

section IMO_1988_Shortlist_P29

def a_n (n : ℕ) : ℕ :=
  Nat.floor (Real.sqrt ((n - 1 : ℕ)^2 + n^2))

theorem problem_part_i :
  ∃ᶠ (m : ℕ) in at_top, a_n (m + 1) - a_n m > 1 :=
sorry

theorem problem_part_ii :
  ∃ᶠ (m : ℕ) in at_top, a_n (m + 1) - a_n m = 1 :=
sorry

end IMO_1988_Shortlist_P29

end problem_part_i_problem_part_ii_l114_114018


namespace greatest_product_of_two_integers_with_sum_300_l114_114225

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l114_114225


namespace greatest_product_l114_114292

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l114_114292


namespace problem1_result_problem2_result_l114_114586

/-
Problem 1: Calculate -{3^2}-{({-\frac{1}{2}})^{-2}}+{1^0}×{({-1})^{2021}}.
- Showing that:
  - 3^2 = 9
  - (-1/2)^(-2) = 4
  - 1^0 = 1
  - (-1)^2021 = -1
  - So, the overall expression equals -14
-/

theorem problem1_result : -(3^2) - ((-1/2)^(-2)) + (1^0) * ((-1)^2021) = -14 := 
by sorry

/-
Problem 2: Calculate a^9 / a^2 * a + (a^2)^4 - (-2a^4)^2
- Showing that:
  - a^9 / a^2 * a = a^8
  - (a^2)^4 = a^8
  - (-2a^4)^2 = 4a^8
  - So, the overall expression equals -2a^8
-/

theorem problem2_result (a : ℝ) : a^9 / a^2 * a + (a^2)^4 - (-2 * a^4)^2 = -2 * a^8 :=
by sorry

end problem1_result_problem2_result_l114_114586


namespace kendra_bought_3_hats_l114_114820

-- Define the price of a wooden toy
def price_of_toy : ℕ := 20

-- Define the price of a hat
def price_of_hat : ℕ := 10

-- Define the amount Kendra went to the shop with
def initial_amount : ℕ := 100

-- Define the number of wooden toys Kendra bought
def number_of_toys : ℕ := 2

-- Define the amount of change Kendra received
def change_received : ℕ := 30

-- Prove that Kendra bought 3 hats
theorem kendra_bought_3_hats : 
  initial_amount - change_received - (number_of_toys * price_of_toy) = 3 * price_of_hat := by
  sorry

end kendra_bought_3_hats_l114_114820


namespace chord_length_l114_114547

theorem chord_length (CD OA : ℝ) (O : point) (M : point) (r : ℝ)
  (h₁ : CD = 10) (h₂ : radius = 10) (h₃ : perp CD OA) (h₄ : midpoint M OA) : 
  (2 * sqrt(3) * 5 = 10 * sqrt(3)) :=
by
  sorry

end chord_length_l114_114547


namespace correct_order_l114_114594

noncomputable def f : ℝ → ℝ := sorry

axiom periodic : ∀ x : ℝ, f (x + 4) = f x
axiom increasing : ∀ (x₁ x₂ : ℝ), (0 ≤ x₁ ∧ x₁ < 2) → (0 ≤ x₂ ∧ x₂ ≤ 2) → x₁ < x₂ → f x₁ < f x₂
axiom symmetric : ∀ x : ℝ, f (x + 2) = f (2 - x)

theorem correct_order : f 4.5 < f 7 ∧ f 7 < f 6.5 :=
by
  sorry

end correct_order_l114_114594


namespace P_on_Apollonius_circle_of_O1O2A_l114_114573
-- Importing the entire Mathlib library for the required definitions and theorems

-- Defining the problem conditions
variables (O1 O2 A B P : ℝ) (r1 r2 : ℝ)

noncomputable def point_on_apollonius_circle 
    (O1 O2 A : ℝ) (P : ℝ) (r1 r2 : ℝ) : Prop := 
    ∃ k, (P ≠ O1) ∧ (P ≠ O2) ∧ 
          (r1 / r2 = k) ∧ 
          (measure_angle O1 P A = measure_angle O2 P B)

theorem P_on_Apollonius_circle_of_O1O2A (O1 O2 A B P : ℝ) (r1 r2 : ℝ) 
    (h1: r1 ≠ 0) (h2: r2 ≠ 0) (h3: P ≠ O1) (h4: P ≠ O2) :
    (point_on_apollonius_circle O1 O2 A P r1 r2) :=
begin
    sorry
end

end P_on_Apollonius_circle_of_O1O2A_l114_114573


namespace min_fraction_value_l114_114893

-- Define the conditions: geometric sequence, specific term relationship, product of terms

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q > 0, ∀ n, a (n + 1) = a n * q

def specific_term_relationship (a : ℕ → ℝ) : Prop :=
  a 3 = a 2 + 2 * a 1

def product_of_terms (a : ℕ → ℝ) (m n : ℕ) : Prop :=
  a m * a n = 64 * (a 1)^2

def min_value_fraction (m n : ℕ) : Prop :=
  1 / m + 9 / n = 2

theorem min_fraction_value (a : ℕ → ℝ) (m n : ℕ)
  (h1 : geometric_sequence a)
  (h2 : specific_term_relationship a)
  (h3 : product_of_terms a m n)
  : min_value_fraction m n := by
  sorry

end min_fraction_value_l114_114893


namespace circle_area_ratio_l114_114092

noncomputable def O := ℝ
noncomputable def P := ℝ
noncomputable def X := O + ((P - O) / 2)
noncomputable def OP := dist O P
noncomputable def OX := dist O X
noncomputable def OC := 2 * OP

theorem circle_area_ratio :
  let area_A := π * (OP ^ 2),
      area_B := π * ((OX) ^ 2),
      area_C := π * ((OC) ^ 2),
      total_area_AC := area_A + area_C in
    (area_B / total_area_AC) = (1 / 20) :=
by
  sorry

end circle_area_ratio_l114_114092


namespace cost_of_tie_l114_114026

theorem cost_of_tie 
  (cost_pants : ℕ) 
  (cost_shirt : ℕ) 
  (amount_paid : ℕ) 
  (amount_received : ℕ) 
  (total_spent : ℕ) 
  (cost_tie : ℕ) :
  cost_pants = 140 → 
  cost_shirt = 43 → 
  amount_paid = 200 → 
  amount_received = 2 → 
  total_spent = amount_paid - amount_received →
  cost_tie = total_spent - (cost_pants + cost_shirt) → 
  cost_tie = 15 :=
by
  intros h1 h2 h3 h4 h5 h6
  split
  { sorry }

end cost_of_tie_l114_114026


namespace find_smallest_c_plus_d_l114_114012

noncomputable def smallest_c_plus_d (c d : ℝ) :=
  c + d

theorem find_smallest_c_plus_d (c d : ℝ) (hc : 0 < c) (hd : 0 < d)
  (h1 : c ^ 2 ≥ 12 * d)
  (h2 : 9 * d ^ 2 ≥ 4 * c) :
  smallest_c_plus_d c d = 16 / 3 :=
by
  sorry

end find_smallest_c_plus_d_l114_114012


namespace mutual_exclusivity_complementary_problem_l114_114727

-- Define the events
def exactly_one_white (drawn : list ℕ) : Prop :=
  drawn.count 1 = 1 ∧ drawn.length = 3

def all_white (drawn : list ℕ) : Prop :=
  drawn.all (λ x, x = 1) ∧ drawn.length = 3

def at_least_one_white (drawn : list ℕ) : Prop :=
  (∃ x ∈ drawn, x = 1) ∧ drawn.length = 3

def all_black (drawn : list ℕ) : Prop :=
  drawn.all (λ x, x = 0) ∧ drawn.length = 3

-- Define the main problem statement
theorem mutual_exclusivity_complementary_problem :
  ∀ (drawn : list ℕ), 
    (at_least_one_white drawn) ∧ (all_black drawn) → 
    ¬((exactly_one_white drawn) ∧ (all_white drawn)) := 
sorry

end mutual_exclusivity_complementary_problem_l114_114727


namespace find_c_value_l114_114634

-- Given condition: x^2 + 300x + c = (x + a)^2
-- Problem statement: Prove that c = 22500 for the given conditions
theorem find_c_value (x a c : ℝ) : (x^2 + 300 * x + c = (x + 150)^2) → (c = 22500) :=
by
  intro h
  sorry

end find_c_value_l114_114634


namespace domain_of_sqrt_fraction_l114_114947

noncomputable def domain_of_function (x : ℝ) : bool :=
  (x ≠ 1) ∧ (x ≤ 2)

theorem domain_of_sqrt_fraction :
  { x : ℝ | domain_of_function x } = {x : ℝ | x ∈ (-∞, 1) ∪ (1, 2]} :=
by
  sorry

end domain_of_sqrt_fraction_l114_114947


namespace greatest_product_l114_114287

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l114_114287


namespace unique_four_digit_number_l114_114103

theorem unique_four_digit_number (a b c d : ℕ) 
  (h1 : b = 3 * a) 
  (h2 : c = a + b)
  (h3 : d = 3 * b) 
  (a_range : a ∈ {1, 2, 3}) 
  (b_range : b < 10) 
  (c_range : c < 10) 
  (d_range : d < 10): 
  1000 * a + 100 * b + 10 * c + d = 1349 := 
sorry

end unique_four_digit_number_l114_114103


namespace race_placement_l114_114731

def finished_places (nina zoey sam liam vince : ℕ) : Prop :=
  nina = 12 ∧
  sam = nina + 1 ∧
  zoey = nina - 2 ∧
  liam = zoey - 3 ∧
  vince = liam + 2 ∧
  vince = nina - 3

theorem race_placement (nina zoey sam liam vince : ℕ) :
  finished_places nina zoey sam liam vince →
  nina = 12 →
  sam = 13 →
  zoey = 10 →
  liam = 7 →
  vince = 5 →
  (8 ≠ sam ∧ 8 ≠ nina ∧ 8 ≠ zoey ∧ 8 ≠ liam ∧ 8 ≠ jodi ∧ 8 ≠ vince) := by
  sorry

end race_placement_l114_114731


namespace find_number_l114_114863

theorem find_number (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 := 
sorry

end find_number_l114_114863


namespace baseball_tickets_l114_114816

theorem baseball_tickets (B : ℕ) 
  (h1 : 25 = 2 * B + 6) : B = 9 :=
sorry

end baseball_tickets_l114_114816


namespace additional_carpet_needed_l114_114029

-- Definitions according to the given conditions
def length_feet := 18
def width_feet := 12
def covered_area := 4 -- in square yards
def feet_per_yard := 3

-- Prove that the additional square yards needed to cover the remaining part of the floor is 20
theorem additional_carpet_needed : 
  ((length_feet / feet_per_yard) * (width_feet / feet_per_yard) - covered_area) = 20 := 
by
  sorry

end additional_carpet_needed_l114_114029


namespace gauss_theorem_barycentric_l114_114537

-- Given a triangle PQR and let (p, q, r) be barycentric coordinates
variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables (P Q R D A C M N R : V)
variables (p q r : ℝ)

-- Definitions for barycentric coordinates
noncomputable def barycentric (p q r : ℝ) (P Q R : V) : V :=
(p / (p + q + r)) • P + (q / (p + q + r)) • Q + (r / (p + q + r)) • R

-- A and C defined by certain barycentric relationships
def point_A : V := barycentric p 0 r P (0 : V) R
def point_C : V := barycentric 0 q r (0 : V) Q R

-- Midpoint of a segment
def midpoint (A B : V) : V := (1/2) • A + (1/2) • B

-- M is the midpoint of AC
def point_M : V := midpoint point_A point_C

-- N is the midpoint of DR
def point_N : V := midpoint D R

-- Theorem statement
theorem gauss_theorem_barycentric : ∃ x : ℝ, 
  point_M = (1 / (1 + x)) • point_N + (x / (1 + x)) • R :=
sorry

end gauss_theorem_barycentric_l114_114537


namespace volume_of_rotated_cube_l114_114635

theorem volume_of_rotated_cube (a : ℝ) :
  let V := (2 * Real.pi * ∫ x in 0 .. (a / Real.sqrt(2)), x^2 + a^2)
  in
  V = (7 * Real.pi * a^3 * Real.sqrt 2) / 6 :=
by
  sorry

end volume_of_rotated_cube_l114_114635


namespace parabola_eqn_l114_114070

noncomputable def parabola_equation : Prop :=
  ∃ (a b c d e f : ℤ),
  c > 0 ∧ 
  Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.gcd |a| |b|) |c|) |d|) |e|) |f| = 1 ∧ 
  (∀ x y : ℝ, (x, y) = (2, 8) → a*x^2 + b*x*y + c*y^2 + d*x + e*y + f = 0) ∧ 
  (b = 0) ∧
  (a = 0) ∧
  (∀ v_y v_x : ℝ, v_y = 4 ∧ v_x = 0 → cy^2 + e*y + f = c(v_y^2 - 8*v_y + 16)) ∧
  (c = 1) ∧
  (d = -8) ∧
  (e = -8) ∧
  (f = 16) 

theorem parabola_eqn :
  parabola_equation :=
sorry

end parabola_eqn_l114_114070


namespace quotient_of_37_div_8_l114_114518

theorem quotient_of_37_div_8 : (37 / 8) = 4 :=
by
  sorry

end quotient_of_37_div_8_l114_114518


namespace greatest_product_of_two_integers_with_sum_300_l114_114230

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l114_114230


namespace find_divisor_l114_114532

theorem find_divisor 
    (x : ℕ) 
    (h : 83 = 9 * x + 2) : 
    x = 9 := 
  sorry

end find_divisor_l114_114532


namespace max_product_sum_300_l114_114313

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l114_114313


namespace greatest_product_of_sum_eq_300_l114_114115

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l114_114115


namespace greatest_product_obtainable_l114_114325

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l114_114325


namespace combined_population_after_two_years_l114_114592

def population_after_years (initial_population : ℕ) (yearly_changes : List (ℕ → ℕ)) : ℕ :=
  yearly_changes.foldl (fun pop change => change pop) initial_population

def townA_change_year1 (pop : ℕ) : ℕ :=
  pop + (pop * 8 / 100) + 200 - 100

def townA_change_year2 (pop : ℕ) : ℕ :=
  pop + (pop * 10 / 100) + 200 - 100

def townB_change_year1 (pop : ℕ) : ℕ :=
  pop - (pop * 2 / 100) + 50 - 200

def townB_change_year2 (pop : ℕ) : ℕ :=
  pop - (pop * 1 / 100) + 50 - 200

theorem combined_population_after_two_years :
  population_after_years 15000 [townA_change_year1, townA_change_year2] +
  population_after_years 10000 [townB_change_year1, townB_change_year2] = 27433 := 
  sorry

end combined_population_after_two_years_l114_114592


namespace probability_sqrt_lt_9_l114_114388

theorem probability_sqrt_lt_9 : 
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
in probability = 71 / 90 :=
by
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  sorry

end probability_sqrt_lt_9_l114_114388


namespace WangLi_final_score_l114_114554

def weightedFinalScore (writtenScore : ℕ) (demoScore : ℕ) (interviewScore : ℕ)
    (writtenWeight : ℕ) (demoWeight : ℕ) (interviewWeight : ℕ) : ℕ :=
  (writtenScore * writtenWeight + demoScore * demoWeight + interviewScore * interviewWeight) /
  (writtenWeight + demoWeight + interviewWeight)

theorem WangLi_final_score :
  weightedFinalScore 96 90 95 5 3 2 = 94 :=
  by
  -- proof goes here
  sorry

end WangLi_final_score_l114_114554


namespace reflection_in_x_axis_l114_114075

theorem reflection_in_x_axis (x y : ℝ) : 
  (∀ (x y : ℝ), reflect_x_axis (x, y) = (x, -y)) →
  reflect_x_axis (-2, -3) = (-2, 3) :=
by
  sorry

def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

end reflection_in_x_axis_l114_114075


namespace count_numbers_with_eight_operations_l114_114780

-- Define the operation rules as functions
def operation (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n + 1

-- Define the number of required operations to reach 1
def num_operations : ℕ → ℕ
| 1 := 0
| n := 1 + num_operations (operation n)

-- The theorem to prove
theorem count_numbers_with_eight_operations : 
  (finset.filter (λ n, num_operations n = 8) (finset.range 256)).card = 21 :=
begin
  sorry -- proof goes here
end

end count_numbers_with_eight_operations_l114_114780


namespace contractor_total_engaged_days_l114_114550

-- Definitions based on conditions
def earnings_per_work_day : ℝ := 25
def fine_per_absent_day : ℝ := 7.5
def total_earnings : ℝ := 425
def days_absent : ℝ := 10

-- The proof problem statement
theorem contractor_total_engaged_days :
  ∃ (x y : ℝ), y = days_absent ∧ total_earnings = earnings_per_work_day * x - fine_per_absent_day * y ∧ x + y = 30 :=
by
  -- let x be the number of working days
  -- let y be the number of absent days
  -- y is given as 10
  -- total_earnings = 25 * x - 7.5 * 10
  -- solve for x and sum x and y to get 30
  sorry

end contractor_total_engaged_days_l114_114550


namespace common_tangents_l114_114818

noncomputable def circle1 := { p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 4 }
noncomputable def circle2 := { p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 2)^2 = 9 }

theorem common_tangents (h : ∀ p : ℝ × ℝ, p ∈ circle1 → p ∈ circle2) : 
  ∃ tangents : ℕ, tangents = 2 :=
sorry

end common_tangents_l114_114818


namespace max_product_of_two_integers_whose_sum_is_300_l114_114165

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l114_114165


namespace greatest_product_l114_114290

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l114_114290


namespace coin_flip_probability_l114_114804

/--
Suppose we flip five coins simultaneously: a penny, a nickel, a dime, a quarter, and a half-dollar.
What is the probability that the penny and dime both come up heads, and the half-dollar comes up tails?
-/

theorem coin_flip_probability :
  let outcomes := 2^5
  let success := 1 * 1 * 1 * 2 * 2
  success / outcomes = (1 : ℚ) / 8 :=
by
  /- Proof goes here -/
  sorry

end coin_flip_probability_l114_114804


namespace geometric_series_sum_l114_114078

theorem geometric_series_sum (m : ℚ) :
  let a1 := -3 / 4
  in m = a1 / (1 - m) → m = -1 / 2 :=
by
  let a1 := (-3 / 4 : ℚ)
  intro h
  rw [←h, mul_comm (-1) (1 - m), ←neg_eq_neg_iff]
  sorry

end geometric_series_sum_l114_114078


namespace min_distance_l114_114022

-- Define the curve y = x^2
def curve (x : ℝ) : ℝ := x^2

-- Define the point P on the curve
def point_P (x0 : ℝ) : ℝ × ℝ := (x0, curve x0)

-- Define the tangent line slope at point P
def tangent_slope (x0 : ℝ) : ℝ := 2 * x0

-- Define the perpendicular line passing through P and its intersection with the curve
def perp_intersection (x0 x1 : ℝ) : ℝ := curve x1

-- Define the distance between points P and Q
def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

-- Minimum distance proof statement
theorem min_distance : ∃ x0 x1 : ℝ, x0 ≠ 0 → perp_intersection x0 x1 = curve x1 ∧ distance (point_P x0) (x1, curve x1) = 3 * Real.sqrt 3 / 2 :=
sorry

end min_distance_l114_114022


namespace arithmetic_series_sum_l114_114631

theorem arithmetic_series_sum : 
  let a1 := -41
  let d := 2
  let n := 22 in
  let an := a1 + (n - 1) * d in
  let S := n / 2 * (a1 + an) in
  an = 1 ∧ S = -440 :=
by
  sorry

end arithmetic_series_sum_l114_114631


namespace greatest_product_sum_300_l114_114248

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l114_114248


namespace cartesian_equation_of_curve_constant_value_of_intersections_l114_114985

noncomputable def parametric_line (a t θ : ℝ) : ℝ × ℝ :=
  (a + t * cos θ, t * sin θ)

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

def curve_polar_eq (ρ θ : ℝ) : Prop :=
  ρ - ρ * cos θ ^ 2 - 4 * cos θ = 0

def curve_cartesian_eq (x y : ℝ) : Prop :=
  y ^ 2 = 4 * x

theorem cartesian_equation_of_curve :
  ∀ (ρ θ : ℝ), curve_polar_eq ρ θ → ∃ (x y : ℝ), (x, y) = polar_to_cartesian ρ θ ∧ curve_cartesian_eq x y :=
by
  sorry

theorem constant_value_of_intersections (a θ : ℝ) (ha : a = 2) :
  ∀ (t1 t2 : ℝ), 
    (parametric_line a t1 θ).snd ^ 2 = 4 * (parametric_line a t1 θ).fst ∧
    (parametric_line a t2 θ).snd ^ 2 = 4 * (parametric_line a t2 θ).fst →
    (1 / ((parametric_line a t1 θ).1 - a) ^ 2 + 1 / ((parametric_line a t2 θ).1 - a) ^ 2 = 1 / 4) :=
by
  sorry

end cartesian_equation_of_curve_constant_value_of_intersections_l114_114985


namespace probability_sqrt_less_nine_l114_114509

theorem probability_sqrt_less_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  ∃ (p : ℚ), p = (finset.card (valid_numbers.to_finset) : ℚ) / (finset.card (two_digit_numbers.to_finset) : ℚ) ∧ p = 71 / 90 :=
by
  sorry

end probability_sqrt_less_nine_l114_114509


namespace greatest_product_of_sum_eq_300_l114_114108

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l114_114108


namespace find_number_l114_114865

theorem find_number (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 := 
sorry

end find_number_l114_114865


namespace max_product_two_integers_sum_300_l114_114152

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l114_114152


namespace binomial_alternating_sum_eq_zero_l114_114607

theorem binomial_alternating_sum_eq_zero :
  (∑ k in Finset.range 51, (-1)^k * Nat.choose 50 k) = 0 :=
by
  sorry

end binomial_alternating_sum_eq_zero_l114_114607


namespace vector_subtraction_example_l114_114611

theorem vector_subtraction_example :
  (⟨3, -2, 4⟩ : ℝ × ℝ × ℝ) - (3 • ⟨2, -1, 5⟩ : ℝ × ℝ × ℝ) = ⟨-3, 1, -11⟩ :=
by try_refl_tac

end vector_subtraction_example_l114_114611


namespace max_product_of_sum_300_l114_114216

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l114_114216


namespace probability_sqrt_lt_9_of_two_digit_l114_114366

-- Define the set of two-digit whole numbers
def two_digit_whole_numbers : set ℕ := {n | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate that checks if the square root of a number is less than 9
def sqrt_lt_9 (n : ℕ) : Prop := (n : ℝ)^2 < (9 : ℝ)^2

-- Calculate the probability
theorem probability_sqrt_lt_9_of_two_digit :
  let eligible_numbers := { n ∈ two_digit_whole_numbers | sqrt_lt_9 n } in
  (eligible_numbers.to_finset.card : ℚ) / (two_digit_whole_numbers.to_finset.card : ℚ) =
  71 / 90 :=
by
  sorry

end probability_sqrt_lt_9_of_two_digit_l114_114366


namespace ratio_of_areas_l114_114851

theorem ratio_of_areas (R : ℝ) (R_pos : 0 < R) :
  let A_hex := (3 * Real.sqrt 3 / 2) * R^2,
      s := 4 * R * Real.sqrt 3 / 3,
      A_tri := (Real.sqrt 3 / 4) * s^2
  in A_tri / A_hex = 2 :=
by {
  let A_hex := (3 * Real.sqrt 3 / 2) * R^2,
  let s := 4 * R * Real.sqrt 3 / 3,
  let A_tri := (Real.sqrt 3 / 4) * s^2,
  sorry
}

end ratio_of_areas_l114_114851


namespace cole_drive_time_correct_l114_114928

noncomputable def cole_drive_time : ℕ :=
  let distance_to_work := 45 -- derived from the given problem   
  let speed_to_work := 30
  let time_to_work := distance_to_work / speed_to_work -- in hours
  (time_to_work * 60 : ℕ) -- converting hours to minutes

theorem cole_drive_time_correct
  (speed_to_work speed_return: ℕ)
  (total_time: ℕ)
  (H1: speed_to_work = 30)
  (H2: speed_return = 90)
  (H3: total_time = 2):
  cole_drive_time = 90 := by
  -- Proof omitted
  sorry

end cole_drive_time_correct_l114_114928


namespace dot_product_is_neg31_l114_114588

def vector1 := (-3, 2)
def vector2 := (7, -5)
def dotProduct (v1 v2 : ℤ × ℤ) := v1.1 * v2.1 + v1.2 * v2.2

theorem dot_product_is_neg31 : dotProduct vector1 vector2 = -31 :=
by
  sorry

end dot_product_is_neg31_l114_114588


namespace ratio_of_points_on_segment_l114_114759

theorem ratio_of_points_on_segment 
  {V : Type*} [AddCommGroup V] [Module ℝ V] 
  (A B P : V) :
  ∃ (t u : ℝ), 
  AP : PB = 3:5 → 
  P = t • A + u • B ∧ 
  (t, u) = (5 / 8, 3 / 8) := 
by 
  sorry

end ratio_of_points_on_segment_l114_114759


namespace tangent_line_at_origin_even_derivative_l114_114006

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + (a - 3) * x
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + (a - 3)

theorem tangent_line_at_origin_even_derivative (a : ℝ) (h : ∀ x, f' a x = f' a (-x)) :
  let tangent_slope := f' a 0 in tangent_slope = -2 →
  2 = 2 * (0 : ℝ) + (0 : ℝ) :=
by
  sorry

end tangent_line_at_origin_even_derivative_l114_114006


namespace order_of_abc_l114_114008

noncomputable def a : ℝ := Real.log 0.9 / Real.log 0.6
noncomputable def b : ℝ := Real.log 0.9
noncomputable def c : ℝ := 2 ^ 0.9

theorem order_of_abc : b < a ∧ a < c :=
by
  sorry

end order_of_abc_l114_114008


namespace probability_sqrt_lt_9_l114_114429

theorem probability_sqrt_lt_9 : 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 →
  ∃ p : ℚ, p = 71 / 90 ∧ 
  ∑ k in (Finset.range 100).filter (λ x, 10 ≤ x ∧ sqrt x < 9), 1 / 90 = p := 
sorry

end probability_sqrt_lt_9_l114_114429


namespace quadruplets_satisfy_l114_114968

-- Define the condition in the problem
def equation (x y z w : ℝ) : Prop :=
  1 + (1 / x) + (2 * (x + 1) / (x * y)) + (3 * (x + 1) * (y + 2) / (x * y * z)) + (4 * (x + 1) * (y + 2) * (z + 3) / (x * y * z * w)) = 0

-- State the theorem
theorem quadruplets_satisfy (x y z w : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) :
  equation x y z w ↔ (x = -1 ∨ y = -2 ∨ z = -3 ∨ w = -4) :=
by
  sorry

end quadruplets_satisfy_l114_114968


namespace arithmetic_sequence_ratio_l114_114662

theorem arithmetic_sequence_ratio
  (a b : ℕ → ℚ)
  (Sa Tb : ℕ → ℚ)
  (hSa : ∀ (n : ℕ), Sa n = (∑ i in (finset.range (n+1)), a i))
  (hTb : ∀ (n : ℕ), Tb n = (∑ i in (finset.range (n+1)), b i))
  (hcond : ∀ (n : ℕ), Sa n / (Tb n) = (3*n + 4)/(n + 2)) :
  (a 3 + a 7 + a 8) / (b 2 + b 10) = 111 / 26 :=
begin
  sorry
end

end arithmetic_sequence_ratio_l114_114662


namespace greatest_product_sum_300_l114_114122

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l114_114122


namespace solution_set_of_abs_inequality_l114_114828

-- Define the condition as a predicate on real numbers
def abs_inequality (x : ℝ) : Prop := |2 * x - 1| ≤ 1

-- The statement to prove: The solution set of the inequality is [0, 1]
theorem solution_set_of_abs_inequality :
  {x : ℝ | abs_inequality x} = set.Icc 0 1 :=
by
  sorry

end solution_set_of_abs_inequality_l114_114828


namespace sin_A_proof_l114_114733

-- Let A, B, C be points forming a right triangle with ∠C = 90°.
variables (A B C : Type) [angles : angle]

-- The triangle is a right triangle.
def is_right_triangle (A B C : Type) [angles : angle] : Prop :=
  ∡ C = π / 2

-- Angle B is given such that cos B = 1/2
def cos_B_eq_one_half (A B C : Type) [angles : angle] : Prop :=
  cos (∡ B) = 1 / 2

-- The proposition to prove: ∠A = 30° implies sin A = 1/2
theorem sin_A_proof (A B C : Type) [angles : angle] 
  (H1 : is_right_triangle A B C) (H2 : cos_B_eq_one_half A B C) : sin (∡ A) = 1 / 2 :=
by sorry

end sin_A_proof_l114_114733


namespace probability_sqrt_lt_9_l114_114421

theorem probability_sqrt_lt_9 : 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 →
  ∃ p : ℚ, p = 71 / 90 ∧ 
  ∑ k in (Finset.range 100).filter (λ x, 10 ≤ x ∧ sqrt x < 9), 1 / 90 = p := 
sorry

end probability_sqrt_lt_9_l114_114421


namespace greatest_product_of_sum_eq_300_l114_114117

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l114_114117


namespace probability_sqrt_less_nine_l114_114502

theorem probability_sqrt_less_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  ∃ (p : ℚ), p = (finset.card (valid_numbers.to_finset) : ℚ) / (finset.card (two_digit_numbers.to_finset) : ℚ) ∧ p = 71 / 90 :=
by
  sorry

end probability_sqrt_less_nine_l114_114502


namespace most_accurate_reading_l114_114773

def temperature_reading (temp: ℝ) : Prop := 
  98.6 ≤ temp ∧ temp ≤ 99.1 ∧ temp ≠ 98.85 ∧ temp > 98.85

theorem most_accurate_reading (temp: ℝ) : temperature_reading temp → temp = 99.1 :=
by
  intros h
  sorry 

end most_accurate_reading_l114_114773


namespace arithmetic_series_sum_l114_114633

theorem arithmetic_series_sum : 
  let a := -41
  let d := 2
  let n := 22
  let l := 1
  let Sn := n * (a + l) / 2
  a = -41 ∧ d = 2 ∧ l = 1 ∧ n = 22 → Sn = -440 :=
by 
  intros a d n l Sn h
  sorry

end arithmetic_series_sum_l114_114633


namespace max_product_sum_300_l114_114300

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l114_114300


namespace probability_sqrt_less_than_nine_l114_114458

theorem probability_sqrt_less_than_nine :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let T := {n : ℕ | n ∈ S ∧ n < 81}
  (T.card : ℚ) / S.card = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114458


namespace greatest_product_sum_300_l114_114253

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l114_114253


namespace lcm_eq_210_l114_114785

noncomputable def lcm_four_coprime_numbers (a b c d : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm a b) (Nat.lcm c d)

theorem lcm_eq_210 (a b c d : ℕ)
  (coprime_ab : Nat.coprime a b)
  (coprime_cd : Nat.coprime c d)
  (coprime_ac : Nat.coprime a c)
  (coprime_ad : Nat.coprime a d)
  (coprime_bc : Nat.coprime b c)
  (coprime_bd : Nat.coprime b d)
  (h : a^2 * b^2 * c^3 * d^3 = 293930) :
  lcm_four_coprime_numbers a b c d = 210 := by
  sorry

end lcm_eq_210_l114_114785


namespace least_value_of_x_l114_114726

theorem least_value_of_x (x p : ℕ) (h1 : (x / (11 * p)) = 3) (h2 : x > 0) (h3 : Nat.Prime p) : x = 66 := by
  sorry

end least_value_of_x_l114_114726


namespace part1_subset_part2_range_l114_114080

-- Definitions of the solution sets A and B
def sol_set_A (a : ℝ) : set ℝ :=
  { x | x^2 + a = x }

def sol_set_B (a : ℝ) : set ℝ :=
  { x | (x^2 + a)^2 + a = x }

-- Condition: A is non-empty
axiom A_nonempty (a : ℝ) : (sol_set_A a).nonempty

-- Part (1): Prove A ⊆ B
theorem part1_subset (a : ℝ) (hA : A_nonempty a) : sol_set_A a ⊆ sol_set_B a :=
sorry

-- Part (2): If A = B, find the range of a
-- We need to prove that if A = B, then -3/4 ≤ a ≤ 1/4
theorem part2_range (a : ℝ) : sol_set_A a = sol_set_B a ↔ -3 / 4 ≤ a ∧ a ≤ 1 / 4 :=
sorry

end part1_subset_part2_range_l114_114080


namespace segment_sum_inequality_l114_114655

theorem segment_sum_inequality :
  let n := 2022
  let half_n := n / 2
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → (i <= half_n → is_red i) ∧ (i > half_n → is_blue i)) →
  (sum_of_lengths (red_left_endpoint segments) (blue_right_endpoint segments) ≠
   sum_of_lengths (blue_left_endpoint segments) (red_right_endpoint segments)) :=
by
  sorry

end segment_sum_inequality_l114_114655


namespace sin_summation_identity_l114_114714

theorem sin_summation_identity 
  (α : ℝ) 
  (h1 : Real.tan α - 1 / Real.tan α = 3 / 2) 
  (h2 : α ∈ Set.Ioo (π / 4) (π / 2)) :
  Real.sin (2 * α + π / 4) = sqrt 2 / 10 :=
sorry

end sin_summation_identity_l114_114714


namespace tile_15xn_with_pentominos_l114_114809

def pentomino_tiling_condition (n : Nat) : Prop := 
  n ≠ 2 ∧ n ≠ 4 ∧ n ≠ 7

theorem tile_15xn_with_pentominos (n : Nat) (h : n > 1) : pentomino_tiling_condition n → 
  ∃ tiles : List (List (Bool)), tiles_tilable (15 * n) tiles := sorry

end tile_15xn_with_pentominos_l114_114809


namespace probability_of_sqrt_lt_9_l114_114353

-- Define the set of two-digit whole numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the subset of numbers for which the square root is less than 9
def valid_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 80}

-- Define the probability calculation
noncomputable def probability_sqrt_lt_9 := (valid_numbers.to_finset.card : ℝ) / (two_digit_numbers.to_finset.card : ℝ)

-- The statement we aim to prove
theorem probability_of_sqrt_lt_9 : probability_sqrt_lt_9 = 71 / 90 := 
sorry

end probability_of_sqrt_lt_9_l114_114353


namespace cost_per_lb_of_mixture_l114_114912

def millet_weight : ℝ := 100
def millet_cost_per_lb : ℝ := 0.60
def sunflower_weight : ℝ := 25
def sunflower_cost_per_lb : ℝ := 1.10

theorem cost_per_lb_of_mixture :
  let millet_weight := 100
  let millet_cost_per_lb := 0.60
  let sunflower_weight := 25
  let sunflower_cost_per_lb := 1.10
  let millet_total_cost := millet_weight * millet_cost_per_lb
  let sunflower_total_cost := sunflower_weight * sunflower_cost_per_lb
  let total_cost := millet_total_cost + sunflower_total_cost
  let total_weight := millet_weight + sunflower_weight
  (total_cost / total_weight) = 0.70 :=
by
  sorry

end cost_per_lb_of_mixture_l114_114912


namespace net_profit_from_plant_sales_l114_114924

noncomputable def calculate_net_profit : ℝ :=
  let cost_basil := 2.00
  let cost_mint := 3.00
  let cost_zinnia := 7.00
  let cost_soil := 15.00
  let total_cost := cost_basil + cost_mint + cost_zinnia + cost_soil
  let basil_germinated := 20 * 0.80
  let mint_germinated := 15 * 0.75
  let zinnia_germinated := 10 * 0.70
  let revenue_healthy_basil := 12 * 5.00
  let revenue_small_basil := 8 * 3.00
  let revenue_healthy_mint := 10 * 6.00
  let revenue_small_mint := 4 * 4.00
  let revenue_healthy_zinnia := 5 * 10.00
  let revenue_small_zinnia := 2 * 7.00
  let total_revenue := revenue_healthy_basil + revenue_small_basil + revenue_healthy_mint + revenue_small_mint + revenue_healthy_zinnia + revenue_small_zinnia
  total_revenue - total_cost

theorem net_profit_from_plant_sales : calculate_net_profit = 197.00 := by
  sorry

end net_profit_from_plant_sales_l114_114924


namespace distance_from_center_to_plane_l114_114989

theorem distance_from_center_to_plane :
  ∀ (P A B C O : ℝ^3)
  (s : ℝ)
  (h1 : ∥O - P∥ = s)
  (h2 : ∥O - A∥ = s)
  (h3 : ∥O - B∥ = s)
  (h4 : ∥O - C∥ = s)
  (h5 : s = √3)
  (h6 : ∥P - A∥ = 2)
  (h7 : ∥P - B∥ = 2)
  (h8 : ∥P - C∥ = 2)
  (h9 : P - A = (2, 0, 0))
  (h10 : P - B = (0, 2, 0))
  (h11 : P - C = (0, 0, 2))
  , ∥plane_distance O A B C - (√3 / 3)∥ = 0 :=
by
  sorry

end distance_from_center_to_plane_l114_114989


namespace prob_sqrt_less_than_nine_l114_114473

/-- The probability that the square root of a randomly selected 
two-digit whole number is less than nine is 71/90. -/
theorem prob_sqrt_less_than_nine : (let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99};
                                     let A := {n : ℕ | 10 ≤ n ∧ n < 81};
                                     (A.card / S.card : ℚ) = 71 / 90) :=
by
  sorry

end prob_sqrt_less_than_nine_l114_114473


namespace least_common_multiple_condition_l114_114936

variable {n : ℕ}
variable {a : Fin n → ℕ}

-- LCM function, here for clarity (Lean has predefined LCM)
def lcm (x y : ℕ) : ℕ := x * (y / gcd x y)

theorem least_common_multiple_condition
  (h₁ : ∀ i j : Fin n, i < j → lcm (a i) (a j) > 2 * n)
  (h₂ : ∀ i : Fin n, a i ≤ 2 * n)
  (h₃ : ∀ i j : Fin n, i < j → a i ≤ a j) :
  a 0 > 2 * n / 3 :=
begin
  sorry
end

end least_common_multiple_condition_l114_114936


namespace equilateral_triangle_dodecagon_l114_114942

-- Definitions
def regular_dodecagon (S : Type) [metric_space S] (R : ℝ) : Prop :=
  sorry

-- Main theorem
theorem equilateral_triangle_dodecagon (S12 T12 : Type) [metric_space S12] [metric_space T12] (R : ℝ) :
  regular_dodecagon S12 R →
  (∀ side, ∃ triangle, equilateral_triangle_on_side side) →
  (∃ T12, regular_dodecagon T12 (2 * R) ∧ annular_area S12 T12 = area S12) :=
sorry

end equilateral_triangle_dodecagon_l114_114942


namespace solve_log_equation_l114_114043

noncomputable def problem_statement : Prop :=
  ∀ x : ℝ, log (2^x + 2 * x - 16) = x * (1 - log 5) → x = 8

theorem solve_log_equation :
  problem_statement :=
  by
    sorry

end solve_log_equation_l114_114043


namespace probability_sqrt_less_than_nine_is_correct_l114_114380

def probability_sqrt_less_than_nine : ℚ :=
  let total_two_digit_numbers := 99 - 10 + 1 in
  let satisfying_numbers := 80 - 10 + 1 in
  let probability := (satisfying_numbers : ℚ) / (total_two_digit_numbers : ℚ) in
  probability

theorem probability_sqrt_less_than_nine_is_correct :
  probability_sqrt_less_than_nine = 71 / 90 :=
by
  -- proof here
  sorry

end probability_sqrt_less_than_nine_is_correct_l114_114380


namespace probability_sqrt_less_than_nine_l114_114337

/-- Define the set of two-digit integers --/
def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Define the condition that the square root of the number is less than 9 --/
def sqrt_less_than_nine (n : Nat) : Prop := n < 81

/-- The number of integers from 10 to 80 --/
lemma count_satisfying_sqrt (n : Nat) : Prop :=
  is_two_digit n ∧ sqrt_less_than_nine n → n < 81

/-- Total number of two-digit integers --/
lemma count_two_digit_total (n : Nat) : Prop := is_two_digit n 

/-- The probability that a randomly selected two-digit integer's square root is less than 9. --/
theorem probability_sqrt_less_than_nine : 
  (∃ n, count_satisfying_sqrt n) / (∃ n, count_two_digit_total n) = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114337


namespace probability_sqrt_lt_nine_l114_114411

theorem probability_sqrt_lt_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  let probability := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  probability = 71 / 90 :=
by
  let two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n : ℕ | 10 ≤ n ∧ n < 81}
  have h1 : two_digit_numbers.card = 90 := sorry
  have h2 : valid_numbers.card = 71 := sorry
  let probability : ℚ := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  rw [h1, h2]
  simp
  norm_num

end probability_sqrt_lt_nine_l114_114411


namespace find_distance_l114_114544

variables (D S V : ℝ)

-- Conditions based on the problem statement
def initial_condition (S : ℝ) : Prop := D = S * 6
def new_condition : Prop := V = 30 ∧ D = V * 4

theorem find_distance (h1 : initial_condition S) (h2 : new_condition) : D = 120 := 
by
  sorry

end find_distance_l114_114544


namespace greatest_integer_bound_l114_114104

noncomputable def greatest_integer_less_than_or_equal (x : ℝ) : ℕ := ⌊x⌋

theorem greatest_integer_bound :
  greatest_integer_less_than_or_equal ( (5^105 + 4^105) / (5^99 + 4^99) ) = 15624 :=
sorry

end greatest_integer_bound_l114_114104


namespace sodium_hydroxide_formation_l114_114961

variable (NaH H₂O NaOH H₂ : Type)
variable [Molecule NaH] [Molecule H₂O] [Molecule NaOH] [Molecule H₂]

def reaction (n : Nat) :=
  (n : ℕ) * (1 : ℕ) + n * (1 : ℕ) = n * (1 : ℕ) + n * (1 : ℕ)

theorem sodium_hydroxide_formation :
  (reaction 2) -> 2 = 2 :=
by
  sorry

end sodium_hydroxide_formation_l114_114961


namespace sum_of_digits_base6_l114_114803

-- Definitions for the digits S, H, E
variables (S H E : ℕ)

-- Conditions: Each of S, H, E is a non-zero digit less than 6 and they are distinct
def valid_digits (a b c : ℕ) : Prop := a < 6 ∧ b < 6 ∧ c < 6 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Conditions: the base 6 arithmetic must hold
def base6_sum_condition (S H E : ℕ) : Prop :=
  (E + H = S ∨ E + H = S + 6) ∧
  (H + S = E ∨ H + S = E + 6) ∧
  (S + E = H ∨ S + E = H + 6)

-- The statement to prove
theorem sum_of_digits_base6 : valid_digits S H E ∧ base6_sum_condition S H E → S + H + E = 12 := 
by 
  intros,
  sorry

end sum_of_digits_base6_l114_114803


namespace probability_of_sqrt_lt_9_l114_114356

-- Define the set of two-digit whole numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the subset of numbers for which the square root is less than 9
def valid_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 80}

-- Define the probability calculation
noncomputable def probability_sqrt_lt_9 := (valid_numbers.to_finset.card : ℝ) / (two_digit_numbers.to_finset.card : ℝ)

-- The statement we aim to prove
theorem probability_of_sqrt_lt_9 : probability_sqrt_lt_9 = 71 / 90 := 
sorry

end probability_of_sqrt_lt_9_l114_114356


namespace number_division_l114_114860

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end number_division_l114_114860


namespace segment_sum_inequality_l114_114654

noncomputable def points := list ℤ (range 2022)

def is_red (p : ℤ) : Prop := (p % 2) = 1
def is_blue (p : ℤ) : Prop := (p % 2) = 0

def red_left_blue_right_sum (points : list ℤ) : ℤ := 
  points.foldr (λ p sum, if is_red p then sum + p else sum) 0

def blue_left_red_right_sum (points : list ℤ) : ℤ :=
  points.foldr (λ p sum, if is_blue p then sum + p else sum) 0

theorem segment_sum_inequality :
  red_left_blue_right_sum points ≠ blue_left_red_right_sum(points) :=
begin
  sorry,
end

end segment_sum_inequality_l114_114654


namespace probability_sqrt_less_than_nine_l114_114462

theorem probability_sqrt_less_than_nine :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let T := {n : ℕ | n ∈ S ∧ n < 81}
  (T.card : ℚ) / S.card = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114462


namespace find_last_year_rate_l114_114054

-- Define the problem setting with types and values (conditions)
def last_year_rate (r : ℝ) : Prop := 
  -- Let r be the annual interest rate last year
  1.1 * r = 0.09

-- Define the theorem to prove the interest rate last year given this year's rate
theorem find_last_year_rate :
  ∃ r : ℝ, last_year_rate r ∧ r = 0.09 / 1.1 := 
by
  sorry

end find_last_year_rate_l114_114054


namespace find_a1_a2_a3_l114_114990

def sequence (a : ℕ → ℕ) := ∀ n, a (n + 3) = 2 + a n

def sum_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) := ∀ n, S n = (Finset.range n).sum a

theorem find_a1_a2_a3 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h_sequence : sequence a)
  (h_sum : sum_n_terms a S)
  (h_sum_90 : S 90 = 2670) : a 1 + a 2 + a 3 = 2 :=
by
  sorry

end find_a1_a2_a3_l114_114990


namespace ratio_of_spheres_l114_114100

theorem ratio_of_spheres (a : ℝ) (h : a > 0) :
  let circumscribed_radius := (sqrt 30 / 8) * a,
      inscribed_radius := (3 * sqrt 3 / 8) * a in 
  circumscribed_radius / inscribed_radius = sqrt 10 / 3 :=
by
  sorry

end ratio_of_spheres_l114_114100


namespace probability_sqrt_lt_nine_l114_114413

theorem probability_sqrt_lt_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  let probability := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  probability = 71 / 90 :=
by
  let two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n : ℕ | 10 ≤ n ∧ n < 81}
  have h1 : two_digit_numbers.card = 90 := sorry
  have h2 : valid_numbers.card = 71 := sorry
  let probability : ℚ := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  rw [h1, h2]
  simp
  norm_num

end probability_sqrt_lt_nine_l114_114413


namespace max_intersections_quadrilateral_l114_114846

-- Define intersection properties
def max_intersections_side : ℕ := 2
def sides_of_quadrilateral : ℕ := 4

theorem max_intersections_quadrilateral : 
  (max_intersections_side * sides_of_quadrilateral) = 8 :=
by 
  -- The proof goes here
  sorry

end max_intersections_quadrilateral_l114_114846


namespace max_product_two_integers_l114_114192

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l114_114192


namespace angle_PQR_l114_114996

theorem angle_PQR (A B C P Q R : Type) [angle_universe : ∀ (a : A), angle a] 
    (h1: parallel AB PQ) (h2: parallel BC QR) (h3: angle_eq (angle ABC) 30) :
    (angle_eq (angle PQR) 30 ∨ angle_eq (angle PQR) 150) :=
by
  sorry

end angle_PQR_l114_114996


namespace find_p_11_neg_7_l114_114020

def p (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

variables (a b c d : ℝ)
variables (h1 : p(1) = 1993)
variables (h2 : p(2) = 3986)
variables (h3 : p(3) = 5979)

theorem find_p_11_neg_7 
  (a b c d : ℝ) 
  (h1 : p 1 = 1993) 
  (h2 : p 2 = 3986) 
  (h3 : p 3 = 5979) 
  : (1/4) * (p 11 + p (-7)) = 5233 :=
sorry

end find_p_11_neg_7_l114_114020


namespace true_compound_propositions_count_l114_114694

def p : Prop := 4 ∉ {2, 3}  -- This means that p is false
def q : Prop := 2 ∈ {2, 3}  -- This means that q is true

theorem true_compound_propositions_count : 
  let p_or_q := p ∨ q,
      p_and_q := p ∧ q,
      not_p := ¬p in
  (if p_or_q then 1 else 0) + (if p_and_q then 1 else 0) + (if not_p then 1 else 0) = 2 :=
  sorry

end true_compound_propositions_count_l114_114694


namespace probability_sqrt_two_digit_lt_nine_correct_l114_114430

noncomputable def probability_sqrt_two_digit_lt_nine : ℚ :=
  let two_digit_integers := finset.Icc 10 99
  let satisfying_integers := two_digit_integers.filter (λ n => n < 81)
  let probability := (satisfying_integers.card : ℚ) / (two_digit_integers.card : ℚ)
  probability

theorem probability_sqrt_two_digit_lt_nine_correct :
  probability_sqrt_two_digit_lt_nine = 71 / 90 := by
  sorry

end probability_sqrt_two_digit_lt_nine_correct_l114_114430


namespace repeated_digit_in_mod_sequence_l114_114604

theorem repeated_digit_in_mod_sequence : 
  ∃ (x y : ℕ), x ≠ y ∧ (2^1970 % 9 = 4) ∧ 
  (∀ n : ℕ, n < 10 → n = 2^1970 % 9 → n = x ∨ n = y) :=
sorry

end repeated_digit_in_mod_sequence_l114_114604


namespace probability_sqrt_lt_9_l114_114396

theorem probability_sqrt_lt_9 : 
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
in probability = 71 / 90 :=
by
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  sorry

end probability_sqrt_lt_9_l114_114396


namespace sum_of_reversed_square_digits_l114_114854

theorem sum_of_reversed_square_digits (n : ℕ) (h : n = 11111) : 
  let sq := n^2,
      rev := 123454321
  in (sq = rev) → (rev.digits.sum = 25) :=
by
  intros
  subst h
  sorry

end sum_of_reversed_square_digits_l114_114854


namespace cubic_polynomial_with_rational_roots_exists_l114_114613

theorem cubic_polynomial_with_rational_roots_exists (a b c : ℚ)
  (h1 : ∃ x : ℚ, x^3 + a * x^2 + b * x + c = 0)
  (h2 : ∃ y : ℚ, y^3 + a * y^2 + b * y + c = 0)
  (h3 : ∃ z : ℚ, z^3 + a * z^2 + b * z + c = 0) :
  ∃ P : Polynomial ℚ, P = Polynomial.Cubic a b c := by
  sorry

end cubic_polynomial_with_rational_roots_exists_l114_114613


namespace find_radius_l114_114548

open Real

def circumference (r : ℝ) : ℝ := 2 * π * r
def area (r : ℝ) : ℝ := π * r^2

theorem find_radius (r : ℝ) :
  circumference r = 69.11503837897544 ∧ area r = 380.132711084365 → r = 11 :=
by
  sorry

end find_radius_l114_114548


namespace probability_sqrt_less_than_nine_l114_114451

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sqrt_less_than_nine (n : ℕ) : Prop := n < 81

theorem probability_sqrt_less_than_nine :
  (∃ total_num favorable_num,
    (total_num = Finset.card (Finset.filter is_two_digit (Finset.range 100)) ∧
     favorable_num = Finset.card (Finset.filter (λ n, is_two_digit n ∧ sqrt_less_than_nine n) (Finset.range 100)) ∧
     (favorable_num : ℚ) / total_num = 71 / 90)) :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114451


namespace min_value_expression_l114_114762

theorem min_value_expression (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : a * b * c = 1/2) :
  a^2 + 4 * a * b + 12 * b^2 + 8 * b * c + 3 * c^2 ≥ 18 :=
sorry

end min_value_expression_l114_114762


namespace exists_unstable_ntuple_l114_114911

noncomputable def stable_tuple (a : Fin n → ℝ) : Prop :=
  ∀ k : Fin (n - 1), abs ((∑ i in Finset.range (k + 1), a i) / (k + 1 : ℝ) - a (k + 1)) < 1

theorem exists_unstable_ntuple (n : ℕ) : 
  ∃ a : Fin n → ℝ, stable_tuple a ∧ ∀ x : ℝ, ¬ stable_tuple (λ (i : Fin (n + 1)), if i = 0 then x else a (i - 1)) :=
sorry

end exists_unstable_ntuple_l114_114911


namespace max_product_two_integers_sum_300_l114_114163

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l114_114163


namespace probability_sqrt_lt_9_of_two_digit_l114_114368

-- Define the set of two-digit whole numbers
def two_digit_whole_numbers : set ℕ := {n | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate that checks if the square root of a number is less than 9
def sqrt_lt_9 (n : ℕ) : Prop := (n : ℝ)^2 < (9 : ℝ)^2

-- Calculate the probability
theorem probability_sqrt_lt_9_of_two_digit :
  let eligible_numbers := { n ∈ two_digit_whole_numbers | sqrt_lt_9 n } in
  (eligible_numbers.to_finset.card : ℚ) / (two_digit_whole_numbers.to_finset.card : ℚ) =
  71 / 90 :=
by
  sorry

end probability_sqrt_lt_9_of_two_digit_l114_114368


namespace part_a_l114_114526

theorem part_a (n : ℕ) (h1 : n ≥ 2) (a : Fin n → ℝ) (h2 : ∀ i, a i > 0) (h3 : (∏ i, a i) = 1) :
  (∑ i, 1 / (a i + 1)) ≥ 1 :=
sorry

end part_a_l114_114526


namespace probability_sqrt_less_than_nine_l114_114461

theorem probability_sqrt_less_than_nine :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let T := {n : ℕ | n ∈ S ∧ n < 81}
  (T.card : ℚ) / S.card = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114461


namespace gift_wrapping_combinations_l114_114552

theorem gift_wrapping_combinations :
  (10 * 4 * 5 * 2 = 400) := by
  sorry

end gift_wrapping_combinations_l114_114552


namespace angle_between_a_b_l114_114713

variables {T : Type*} [inner_product_space ℝ T]

-- Definitions from conditions:
def e1 : T := sorry -- Assume e1 is a given unit vector
def e2 : T := sorry -- Assume e2 is a given unit vector
def is_unit (v : T) : Prop := ∥v∥ = 1
def angle_between (v w : T) : ℝ := real.acos (inner_product_space.is_R_or_C.re (⟪v, w⟫) / (∥v∥ * ∥w∥))

-- Angle between unit vectors e1 and e2 is 60 degrees
axiom angle_e1_e2 : angle_between e1 e2 = real.to_radians 60

-- Given definitions for vectors a and b
def a := (2 : ℝ) • e1 + e2
def b := (-3 : ℝ) • e1 + (2 : ℝ) • e2

-- The goal to prove:
theorem angle_between_a_b : angle_between a b = real.to_radians 120 := sorry

end angle_between_a_b_l114_114713


namespace sin_of_angle_A_l114_114736

theorem sin_of_angle_A (A B C : ℝ) (hC : C = 90) (hB : cos B = 1 / 2) : sin A = 1 / 2 := 
  sorry

end sin_of_angle_A_l114_114736


namespace max_product_two_integers_sum_300_l114_114155

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l114_114155


namespace find_a_l114_114701

theorem find_a (a : ℝ) :
  let l1 := λ x y, a * x + 3 * y - 1 = 0
  let l2 := λ x y, 2 * x + (a - 1) * y + 1 = 0
  (∀ x y : ℝ, l1 x y = 0 → l2 x y = 0 → a ≠ 2 / 3) →
  (∀ x y : ℝ, l1 x y = 0 → l2 x y = 1 → False) →
  a = 3 :=
by
  sorry

end find_a_l114_114701


namespace number_in_sequence_l114_114915

theorem number_in_sequence : ∃ n : ℕ, n * (n + 2) = 99 :=
by
  sorry

end number_in_sequence_l114_114915


namespace probability_sqrt_lt_nine_l114_114403

theorem probability_sqrt_lt_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  let probability := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  probability = 71 / 90 :=
by
  let two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n : ℕ | 10 ≤ n ∧ n < 81}
  have h1 : two_digit_numbers.card = 90 := sorry
  have h2 : valid_numbers.card = 71 := sorry
  let probability : ℚ := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  rw [h1, h2]
  simp
  norm_num

end probability_sqrt_lt_nine_l114_114403


namespace greatest_product_of_sum_eq_300_l114_114106

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l114_114106


namespace rotated_clockwise_120_correct_l114_114871

-- Problem setup definitions
structure ShapePosition :=
  (triangle : Point)
  (smaller_circle : Point)
  (square : Point)

-- Conditions for the initial positions of the shapes
variable (initial : ShapePosition)

def rotated_positions (initial: ShapePosition) : ShapePosition :=
  { 
    triangle := initial.smaller_circle,
    smaller_circle := initial.square,
    square := initial.triangle 
  }

-- Problem statement: show that after a 120° clockwise rotation, 
-- the shapes move to the specified new positions.
theorem rotated_clockwise_120_correct (initial : ShapePosition) 
  (after_rotation : ShapePosition) :
  after_rotation = rotated_positions initial := 
sorry

end rotated_clockwise_120_correct_l114_114871


namespace minimun_receipts_to_buy_chocolates_l114_114040

def rms (x1 x2 x3 : ℝ) : ℝ :=
  Real.sqrt ((x1^2 + x2^2 + x3^2) / 3)

def canBuyWithoutVerification (S max_cost : ℝ) : Prop :=
  max_cost <= 3 * S

theorem minimun_receipts_to_buy_chocolates 
  (cost_per_chocolate : ℝ) (num_chocolates : ℕ)
  (prev1 prev2 prev3 : ℝ) (prev_purchases : prev1 = 300 ∧ prev2 = 300 ∧ prev3 = 300) :
  num_chocolates = 40 ∧ cost_per_chocolate = 50 →
  ∃ (num_receipts : ℕ), num_receipts = 2 :=
by
  sorry

end minimun_receipts_to_buy_chocolates_l114_114040


namespace point_C_not_in_plane_region_other_points_in_plane_region_l114_114571

-- Defining the inequality condition
def in_plane_region (x y : ℝ) : Prop := x + y - 1 ≤ 0

-- Points definitions
def point_A := (0, 0)
def point_B := (-1, 1)
def point_C := (-1, 3)
def point_D := (2, -3)

-- Theorem statement
theorem point_C_not_in_plane_region :
  ¬ in_plane_region (fst point_C) (snd point_C) :=
by
  sorry

theorem other_points_in_plane_region :
  in_plane_region (fst point_A) (snd point_A) ∧
  in_plane_region (fst point_B) (snd point_B) ∧
  in_plane_region (fst point_D) (snd point_D) :=
by
  sorry

end point_C_not_in_plane_region_other_points_in_plane_region_l114_114571


namespace jamal_books_remaining_l114_114746

variable (initial_books : ℕ := 51)
variable (history_books : ℕ := 12)
variable (fiction_books : ℕ := 19)
variable (children_books : ℕ := 8)
variable (misplaced_books : ℕ := 4)

theorem jamal_books_remaining : 
  initial_books - history_books - fiction_books - children_books + misplaced_books = 16 := by
  sorry

end jamal_books_remaining_l114_114746


namespace sum_log_expr_l114_114930

theorem sum_log_expr : 
  (∑ k in Finset.range 500, k * (if (Real.log k / Real.log (Real.sqrt 3)).frac = 0 then 0 else 1)) = 124430 :=
by
  sorry

end sum_log_expr_l114_114930


namespace greatest_product_of_two_integers_with_sum_300_l114_114235

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l114_114235


namespace reciprocal_of_one_third_l114_114826

theorem reciprocal_of_one_third : ∃ y : ℚ, (1 / 3) * y = 1 ∧ y = 3 :=
by
  use 3
  split
  · norm_num
  · refl

end reciprocal_of_one_third_l114_114826


namespace log_7_2400_is_4_l114_114844

theorem log_7_2400_is_4 :
  3 < Real.log 2400 / Real.log 7 ∧ Real.log 2400 / Real.log 7 < 4 
  → Round (Real.log 2400 / Real.log 7) = 4 := by
  intros h
  sorry

end log_7_2400_is_4_l114_114844


namespace probability_sqrt_less_than_nine_l114_114463

theorem probability_sqrt_less_than_nine :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let T := {n : ℕ | n ∈ S ∧ n < 81}
  (T.card : ℚ) / S.card = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114463


namespace probability_sqrt_less_than_nine_l114_114464

theorem probability_sqrt_less_than_nine :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let T := {n : ℕ | n ∈ S ∧ n < 81}
  (T.card : ℚ) / S.card = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114464


namespace max_product_of_two_integers_whose_sum_is_300_l114_114167

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l114_114167


namespace largest_possible_p_l114_114901

theorem largest_possible_p (m n p : ℕ) (h1 : m > 2) (h2 : n > 2) (h3 : p > 2) (h4 : gcd m n = 1) (h5 : gcd n p = 1) (h6 : gcd m p = 1)
  (h7 : (1/m : ℚ) + (1/n : ℚ) + (1/p : ℚ) = 1/2) : p ≤ 42 :=
by sorry

end largest_possible_p_l114_114901


namespace probability_of_sqrt_lt_9_l114_114355

-- Define the set of two-digit whole numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the subset of numbers for which the square root is less than 9
def valid_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 80}

-- Define the probability calculation
noncomputable def probability_sqrt_lt_9 := (valid_numbers.to_finset.card : ℝ) / (two_digit_numbers.to_finset.card : ℝ)

-- The statement we aim to prove
theorem probability_of_sqrt_lt_9 : probability_sqrt_lt_9 = 71 / 90 := 
sorry

end probability_of_sqrt_lt_9_l114_114355


namespace redistribution_not_always_possible_l114_114090

theorem redistribution_not_always_possible (a b : ℕ) (h : a ≠ b) :
  ¬(∃ k : ℕ, a - k = b + k ∧ 0 ≤ k ∧ k ≤ a ∧ k ≤ b) ↔ (a + b) % 2 = 1 := 
by 
  sorry

end redistribution_not_always_possible_l114_114090


namespace pentagon_ratio_l114_114597

theorem pentagon_ratio (s : ℝ) (h : s = 6) :
  let A := (5 * s^2 * real.tan (real.pi * 54 / 180)) / 4,
      P := 5 * s
  in A / P = (3 * real.tan (real.pi * 54 / 180)) / 2 :=
by
  -- skip proof
  sorry

end pentagon_ratio_l114_114597


namespace magnitude_v_l114_114046

open Complex

theorem magnitude_v (u v : ℂ) (h1 : u * v = 20 - 15 * Complex.I) (h2 : Complex.abs u = 5) :
  Complex.abs v = 5 := by
  sorry

end magnitude_v_l114_114046


namespace probability_sqrt_lt_nine_two_digit_l114_114498

theorem probability_sqrt_lt_nine_two_digit :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99} in
  let T := {n : ℕ | 10 ≤ n ∧ n < 81} in
  (Fintype.card T : ℚ) / (Fintype.card S) = 71 / 90 :=
by
  sorry

end probability_sqrt_lt_nine_two_digit_l114_114498


namespace prob_sqrt_less_than_nine_l114_114480

/-- The probability that the square root of a randomly selected 
two-digit whole number is less than nine is 71/90. -/
theorem prob_sqrt_less_than_nine : (let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99};
                                     let A := {n : ℕ | 10 ≤ n ∧ n < 81};
                                     (A.card / S.card : ℚ) = 71 / 90) :=
by
  sorry

end prob_sqrt_less_than_nine_l114_114480


namespace find_theta_l114_114619

theorem find_theta :
  ∃ θ : ℝ, θ = 80 ∧ cos (10 * Real.pi / 180) = sin (30 * Real.pi / 180) + sin (θ * Real.pi / 180) := 
by
  use 80
  have h1 : cos (10 * Real.pi / 180) = cos (10 * Real.pi / 180), by sorry
  have h2 : sin (30 * Real.pi / 180) = 1 / 2, by sorry
  have h3 : sin (80 * Real.pi / 180) = cos (10 * Real.pi / 180), by sorry
  rw [h2, h3]
  sorry

end find_theta_l114_114619


namespace arithmetic_sequence_26th_term_eq_neg48_l114_114055

def arithmetic_sequence_term (a₁ d n : ℤ) : ℤ := a₁ + (n - 1) * d

theorem arithmetic_sequence_26th_term_eq_neg48 : 
  arithmetic_sequence_term 2 (-2) 26 = -48 :=
by
  sorry

end arithmetic_sequence_26th_term_eq_neg48_l114_114055


namespace max_product_two_integers_sum_300_l114_114157

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l114_114157


namespace probability_sqrt_lt_nine_two_digit_l114_114496

theorem probability_sqrt_lt_nine_two_digit :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99} in
  let T := {n : ℕ | 10 ≤ n ∧ n < 81} in
  (Fintype.card T : ℚ) / (Fintype.card S) = 71 / 90 :=
by
  sorry

end probability_sqrt_lt_nine_two_digit_l114_114496


namespace greatest_product_l114_114285

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l114_114285


namespace exponential_values_l114_114696

theorem exponential_values (k m n : ℤ) 
  (h₁ : 3 ^ (k - 1) = 81)
  (h₂ : 4 ^ (m + 2) = 256)
  (h₃ : 5 ^ (n - 3) = 625) :
  2 ^ (4 * k - 3 * m + 5 * n) = 2 ^ 49 :=
by 
  sorry

end exponential_values_l114_114696


namespace find_v_l114_114971

noncomputable def v : ℝ × ℝ := (0.6, 4.1)
def u₁ : ℝ × ℝ := (3, 2)
def u₂ : ℝ × ℝ := (1, 4)
def proj (u v : ℝ × ℝ) : ℝ × ℝ := 
  let c := ((u.1 * v.1 + u.2 * v.2) / (u.1 * u.1 + u.2 * u.2))
  (c * u.1, c * u.2)
def w₁ : ℝ × ℝ := (2, 4 / 3)
def w₂ : ℝ × ℝ := (1, 4)

theorem find_v :
  proj u₁ v = w₁ ∧ proj u₂ v = w₂ :=
by
  sorry

end find_v_l114_114971


namespace greatest_product_sum_300_l114_114121

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l114_114121


namespace find_phi_l114_114760

theorem find_phi :
  (∃ s φ, (0 < s) ∧
          (0 ≤ φ) ∧
          (φ < 360) ∧
          (∏ root in {z | (P(z) = 0) ∧ (complex.imaginary z > 0)}, 
            root) = s * (cos (φ * (real.pi / 180)) + complex.I * sin (φ * (real.pi / 180)))
            ∧ φ = 150) :=
sorry

noncomputable def P (z : ℂ) : ℂ := z^8 + z^6 + z^5 + z^4 + 1

end find_phi_l114_114760


namespace parallel_cond_necessary_not_sufficient_l114_114807

theorem parallel_cond_necessary_not_sufficient :
  ∀ a : ℝ,
    (a = 3 ∨ a = -2) →
      (∀ x y : ℝ, (a * x + 2 * y + 3 * a = 0) ∧ (3 * x + (a - 1) * y = a - 7) →
                  (a = 3) :=
begin
  sorry
end

end parallel_cond_necessary_not_sufficient_l114_114807


namespace multiple_of_four_l114_114591

theorem multiple_of_four (n : ℕ) (x : ℕ → ℤ) (h₁ : ∀ i, 1 ≤ i ∧ i ≤ n → x i = 1 ∨ x i = -1)
  (h₂ : (finset.range n).sum (λ k, (x (k % n + 1) * x ((k+1) % n + 1) * x ((k+2) % n + 1) * x ((k+3) % n + 1))) = 0) :
  ∃ k, n = 4 * k :=
by 
  sorry

end multiple_of_four_l114_114591


namespace segment_sum_inequality_l114_114653

noncomputable def points := list ℤ (range 2022)

def is_red (p : ℤ) : Prop := (p % 2) = 1
def is_blue (p : ℤ) : Prop := (p % 2) = 0

def red_left_blue_right_sum (points : list ℤ) : ℤ := 
  points.foldr (λ p sum, if is_red p then sum + p else sum) 0

def blue_left_red_right_sum (points : list ℤ) : ℤ :=
  points.foldr (λ p sum, if is_blue p then sum + p else sum) 0

theorem segment_sum_inequality :
  red_left_blue_right_sum points ≠ blue_left_red_right_sum(points) :=
begin
  sorry,
end

end segment_sum_inequality_l114_114653


namespace probability_sqrt_two_digit_lt_nine_correct_l114_114439

noncomputable def probability_sqrt_two_digit_lt_nine : ℚ :=
  let two_digit_integers := finset.Icc 10 99
  let satisfying_integers := two_digit_integers.filter (λ n => n < 81)
  let probability := (satisfying_integers.card : ℚ) / (two_digit_integers.card : ℚ)
  probability

theorem probability_sqrt_two_digit_lt_nine_correct :
  probability_sqrt_two_digit_lt_nine = 71 / 90 := by
  sorry

end probability_sqrt_two_digit_lt_nine_correct_l114_114439


namespace probability_sqrt_lt_9_l114_114416

theorem probability_sqrt_lt_9 : 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 →
  ∃ p : ℚ, p = 71 / 90 ∧ 
  ∑ k in (Finset.range 100).filter (λ x, 10 ≤ x ∧ sqrt x < 9), 1 / 90 = p := 
sorry

end probability_sqrt_lt_9_l114_114416


namespace lcm_trees_problem_l114_114892

theorem lcm_trees_problem : ∃ n : ℕ, (n % 7 = 0 ∧ n % 6 = 0 ∧ n % 4 = 0) ∧ ∀ m : ℕ, (m % 7 = 0 ∧ m % 6 = 0 ∧ m % 4 = 0) → n ≤ m :=
begin
  use 84,
  split,
  { split,
    { norm_num, },  -- 84 % 7 = 0
    { split,
      { norm_num, },  -- 84 % 6 = 0
      { norm_num, }   -- 84 % 4 = 0
    }
  },
  { intros m hm,
    rcases hm with ⟨h7, h6, h4⟩,
    sorry -- Placeholder for proof portion verifying no number less than 84 meets the criteria
  }
end

end lcm_trees_problem_l114_114892


namespace part1_part2_part3_l114_114769

open Real

-- Part 1
theorem part1 (x : ℝ) (h₁ : f x = log (x + 2) > 1) : 0 < x < 1 / 8 :=
sorry

-- Part 2
theorem part2 (x : ℝ) (m : ℝ) (hx : 2 ≤ x ∧ x ≤ 3)
  (h₂ : f 0 = 1) 
  (h₃ : f x = (1 / sqrt 2) ^ x + λ) :
  λ ∈ [log 12 - 1 / 2, log 13 - sqrt 2 / 4] :=
sorry

-- Part 3
theorem part3 (x : ℝ) (n : ℕ) (k : ℕ)
  (hm : f 98 = 2)
  (h₄ : f (cos (2 ^ n * x)) < log 2) : 
  x ∈ (π / 2 + 2 * k * π) / (2 ^ n), (3 * π / 2 + 2 * k * π) / (2 ^ n) :=
sorry

end part1_part2_part3_l114_114769


namespace greatest_product_sum_300_l114_114131

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l114_114131


namespace max_product_of_two_integers_whose_sum_is_300_l114_114178

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l114_114178


namespace no_1234_or_3269_repeat_1975_appear_8197_l114_114740

-- Define the sequence based on the given conditions
def sequence : ℕ → ℕ
| 0 := 1
| 1 := 9
| 2 := 7
| 3 := 5
| n := (sequence (n - 1) + sequence (n - 2) + sequence (n - 3) + sequence (n - 4)) % 10

-- Prove that the sequences 1234 and 3269 will not appear
theorem no_1234_or_3269 : ¬(∃ n, sequence n = 1 ∧ sequence (n + 1) = 2 ∧ sequence (n + 2) = 3 ∧ sequence (n + 3) = 4) ∧
                          ¬(∃ n, sequence n = 3 ∧ sequence (n + 1) = 2 ∧ sequence (n + 2) = 6 ∧ sequence (n + 3) = 9) := sorry

-- Prove that the sequence 1975 will repeat
theorem repeat_1975 : ∃ n, sequence n = 1 ∧ sequence (n + 1) = 9 ∧ sequence (n + 2) = 7 ∧ sequence (n + 3) = 5 := sorry

-- Prove that the sequence 8197 will appear
theorem appear_8197 : ∃ n, sequence n = 8 ∧ sequence (n + 1) = 1 ∧ sequence (n + 2) = 9 ∧ sequence (n + 3) = 7 := sorry

end no_1234_or_3269_repeat_1975_appear_8197_l114_114740


namespace greatest_product_obtainable_l114_114327

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l114_114327


namespace other_root_of_quadratic_l114_114723

theorem other_root_of_quadratic (a b : ℝ) (h₀ : a ≠ 0) (h₁ : ∃ x : ℝ, (a * x ^ 2 = b) ∧ (x = 2)) : 
  ∃ m : ℝ, (a * m ^ 2 = b) ∧ (m = -2) := 
sorry

end other_root_of_quadratic_l114_114723


namespace probability_sqrt_less_than_nine_l114_114459

theorem probability_sqrt_less_than_nine :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let T := {n : ℕ | n ∈ S ∧ n < 81}
  (T.card : ℚ) / S.card = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114459


namespace marketing_survey_l114_114895

theorem marketing_survey
  (H_neither : Nat := 80)
  (H_only_A : Nat := 60)
  (H_ratio_Both_to_Only_B : Nat := 3)
  (H_both : Nat := 25) :
  H_neither + H_only_A + (H_ratio_Both_to_Only_B * H_both) + H_both = 240 := 
sorry

end marketing_survey_l114_114895


namespace double_and_halve_is_sixteen_l114_114516

-- Definition of the initial number
def initial_number : ℕ := 16

-- Doubling the number
def doubled (n : ℕ) : ℕ := n * 2

-- Halving the number
def halved (n : ℕ) : ℕ := n / 2

-- The theorem that needs to be proven
theorem double_and_halve_is_sixteen : halved (doubled initial_number) = 16 :=
by
  /-
  We need to prove that when the number 16 is doubled and then halved, 
  the result is 16.
  -/
  sorry

end double_and_halve_is_sixteen_l114_114516


namespace probability_sqrt_less_nine_l114_114508

theorem probability_sqrt_less_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  ∃ (p : ℚ), p = (finset.card (valid_numbers.to_finset) : ℚ) / (finset.card (two_digit_numbers.to_finset) : ℚ) ∧ p = 71 / 90 :=
by
  sorry

end probability_sqrt_less_nine_l114_114508


namespace seat_39_l114_114834

-- Defining the main structure of the problem
def circle_seating_arrangement (n k : ℕ) : ℕ :=
  if k = 1 then 1
  else sorry -- The pattern-based implementation goes here

-- The theorem to state the problem
theorem seat_39 (n k : ℕ) (h_n : n = 128) (h_k : k = 39) :
  circle_seating_arrangement n k = 51 :=
sorry

end seat_39_l114_114834


namespace wavelength_scientific_notation_l114_114074

theorem wavelength_scientific_notation :
  (0.000000193 : Float) = 1.93 * (10 : Float) ^ (-7) :=
sorry

end wavelength_scientific_notation_l114_114074


namespace pane_length_l114_114806

variable (r_inner : ℝ) (A_inner : ℝ) (A_total : ℝ) (R : ℝ) (x : ℝ)

-- Conditions
def inner_radius : r_inner = 20 := by sorry
def inner_area : A_inner = 400 * Real.pi := by sorry
def total_area : A_total = 3600 * Real.pi := by sorry
def outer_radius : R = r_inner + x := by sorry

-- To Prove
theorem pane_length : x = 40 := by
  have h_R : R ^ 2 = 3600 := by sorry
  have h_R_val : R = 60 := by sorry
  have h_x : x = R - r_inner := by sorry
  exact h_x.trans (by simp [h_R_val, r_inner]) 

end pane_length_l114_114806


namespace max_product_sum_300_l114_114301

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l114_114301


namespace atleast_666_composite_numbers_with_property_l114_114032

open Nat

noncomputable def N (k : ℕ) : ℕ := (10 ^ 2006 - 1) / 9 + 6 * 10 ^ k

theorem atleast_666_composite_numbers_with_property :
  (∃ N : ℕ, (N_digits N = 2006) ∧ 
    (∃ k : ℕ, k ∈ finset.range 2006 ∧ 
                   (N = (10 ^ 2006 - 1) / 9 + 6 * 10 ^ k ∧ 
                    N_is_composite N ∧ 
                    N_contains_digit N 7 ))) :=
sorry

end atleast_666_composite_numbers_with_property_l114_114032


namespace smallest_number_in_set_l114_114927

theorem smallest_number_in_set : ∃ n ∈ {5, 9, 10, 3, 6}, ∀ m ∈ {5, 9, 10, 3, 6}, n ≤ m :=
sorry

end smallest_number_in_set_l114_114927


namespace possible_values_x_l114_114909

theorem possible_values_x : 
  let x := Nat.gcd 112 168 
  ∃ d : Finset ℕ, d.card = 8 ∧ ∀ y ∈ d, y ∣ 112 ∧ y ∣ 168 := 
by
  let x := Nat.gcd 112 168
  have : x = 56 := by norm_num
  use Finset.filter (fun n => 56 % n = 0) (Finset.range 57)
  sorry

end possible_values_x_l114_114909


namespace max_product_of_sum_300_l114_114275

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l114_114275


namespace probability_sqrt_lt_9_l114_114399

theorem probability_sqrt_lt_9 : 
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
in probability = 71 / 90 :=
by
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  sorry

end probability_sqrt_lt_9_l114_114399


namespace greatest_product_l114_114291

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l114_114291


namespace greatest_product_of_sum_eq_300_l114_114119

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l114_114119


namespace two_algorithms_exist_for_quadratic_l114_114789

theorem two_algorithms_exist_for_quadratic (a b c : ℝ) (h : a ≠ 0) :
  a = 1 ∧ b = -5 ∧ c = 6 → 
  (∃ alg1 alg2, alg1 ≠ alg2 ∧ (∀ x, alg1 x = alg2 x)) :=
by
  intros
  use alg1, alg2
  split
  sorry, sorry

end two_algorithms_exist_for_quadratic_l114_114789


namespace greatest_product_of_two_integers_with_sum_300_l114_114237

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l114_114237


namespace largest_possible_number_l114_114540

-- Define the initial sequence and conditions
def initial_number := List.repeat [1, 2, 2, 5] 75

-- Defining the sum operation condition: Sum of two adjacent digits chosen from the list
def can_sum (a b : ℕ) : Prop := a + b ≤ 9

-- Prove the largest possible number that results
theorem largest_possible_number :
  ∃ n : ℕ, (∀ xs : List ℕ, 
    (xs = initial_number ∨ 
     ∃ ys : List ℕ, can_sum xs.head! ys.head! ∧ ys.tail! = xs.tail!.tail!.tail! ∧ xs.tail!.tail! <:+ xs) 
      → (xs.sum = 750)) → n = 375555555555555 :
  sorry

end largest_possible_number_l114_114540


namespace probability_sqrt_less_than_nine_l114_114469

theorem probability_sqrt_less_than_nine :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let T := {n : ℕ | n ∈ S ∧ n < 81}
  (T.card : ℚ) / S.card = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114469


namespace S2014_value_l114_114831

variable (S : ℕ → ℤ) -- S_n represents sum of the first n terms of the arithmetic sequence
variable (a1 : ℤ) -- First term of the arithmetic sequence
variable (d : ℤ) -- Common difference of the arithmetic sequence

-- Given conditions
variable (h1 : a1 = -2016)
variable (h2 : (S 2016) / 2016 - (S 2010) / 2010 = 6)

-- The proof problem
theorem S2014_value :
  S 2014 = -6042 :=
sorry -- Proof omitted

end S2014_value_l114_114831


namespace find_x_divisibility_l114_114519

theorem find_x_divisibility (n : ℕ) (m : ℕ) : 
  let R := n % m
  in if R = 0 then ∃ x, n + x = k * m for some k : ℕ
     else ∃ x, n + x = k * m for some k : ℕ ∧ x = m - R :=
by
  let R := 897326 % 456
  if R = 0 then
    use 0
    sorry
  else
    use (456 - R)
    sorry

end find_x_divisibility_l114_114519


namespace line_parallel_if_perpendicular_to_same_plane_l114_114038

noncomputable def line : Type := sorry
noncomputable def plane : Type := sorry

variables (a b : line) (alpha : plane)
#check (∥ : line → line → Prop)  -- parallel
#check (⊥ : line → plane → Prop) -- perpendicular 
#check (⊆ : line → plane → Prop) -- subset

theorem line_parallel_if_perpendicular_to_same_plane 
  (h1 : a ⊥ alpha) (h2 : b ⊥ alpha) : a ∥ b :=
  sorry

end line_parallel_if_perpendicular_to_same_plane_l114_114038


namespace probability_sqrt_less_than_nine_is_correct_l114_114379

def probability_sqrt_less_than_nine : ℚ :=
  let total_two_digit_numbers := 99 - 10 + 1 in
  let satisfying_numbers := 80 - 10 + 1 in
  let probability := (satisfying_numbers : ℚ) / (total_two_digit_numbers : ℚ) in
  probability

theorem probability_sqrt_less_than_nine_is_correct :
  probability_sqrt_less_than_nine = 71 / 90 :=
by
  -- proof here
  sorry

end probability_sqrt_less_than_nine_is_correct_l114_114379


namespace probability_sqrt_lt_nine_two_digit_l114_114492

theorem probability_sqrt_lt_nine_two_digit :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99} in
  let T := {n : ℕ | 10 ≤ n ∧ n < 81} in
  (Fintype.card T : ℚ) / (Fintype.card S) = 71 / 90 :=
by
  sorry

end probability_sqrt_lt_nine_two_digit_l114_114492


namespace meaningful_not_monotonic_interval_l114_114814

theorem meaningful_not_monotonic_interval (k : ℝ) :
  (∀ x, k - 1 < x ∧ x < k + 1 → True) ∧ (∃ x, k - 1 < 1 ∧ 1 < k + 1) ↔ 1 < k ∧ k < 2 :=
sorry

end meaningful_not_monotonic_interval_l114_114814


namespace initial_money_l114_114876

theorem initial_money :
  (∃ (initial_you initial_friend : ℝ) (weeks : ℕ), 
    initial_friend = 210 ∧ 
    weeks = 25 ∧ 
    initial_you + 7 * weeks = initial_friend + 5 * weeks) → 
  ∃ initial_you : ℝ, initial_you = 160 :=
begin
  sorry
end

end initial_money_l114_114876


namespace max_product_of_two_integers_whose_sum_is_300_l114_114170

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l114_114170


namespace prob_sqrt_less_than_nine_l114_114482

/-- The probability that the square root of a randomly selected 
two-digit whole number is less than nine is 71/90. -/
theorem prob_sqrt_less_than_nine : (let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99};
                                     let A := {n : ℕ | 10 ≤ n ∧ n < 81};
                                     (A.card / S.card : ℚ) = 71 / 90) :=
by
  sorry

end prob_sqrt_less_than_nine_l114_114482


namespace sign_of_x_and_y_l114_114720

theorem sign_of_x_and_y (x y : ℝ) (h1 : x * y > 1) (h2 : x + y ≥ 0) : x > 0 ∧ y > 0 :=
sorry

end sign_of_x_and_y_l114_114720


namespace min_am_hm_l114_114004

theorem min_am_hm (a b : ℝ) (ha : a > 0) (hb : b > 0) : (a + b) * (1/a + 1/b) ≥ 4 :=
by sorry

end min_am_hm_l114_114004


namespace probability_sqrt_lt_9_of_two_digit_l114_114371

-- Define the set of two-digit whole numbers
def two_digit_whole_numbers : set ℕ := {n | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate that checks if the square root of a number is less than 9
def sqrt_lt_9 (n : ℕ) : Prop := (n : ℝ)^2 < (9 : ℝ)^2

-- Calculate the probability
theorem probability_sqrt_lt_9_of_two_digit :
  let eligible_numbers := { n ∈ two_digit_whole_numbers | sqrt_lt_9 n } in
  (eligible_numbers.to_finset.card : ℚ) / (two_digit_whole_numbers.to_finset.card : ℚ) =
  71 / 90 :=
by
  sorry

end probability_sqrt_lt_9_of_two_digit_l114_114371


namespace greatest_product_sum_300_l114_114252

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l114_114252


namespace max_product_two_integers_sum_300_l114_114160

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l114_114160


namespace probability_sqrt_two_digit_lt_nine_correct_l114_114438

noncomputable def probability_sqrt_two_digit_lt_nine : ℚ :=
  let two_digit_integers := finset.Icc 10 99
  let satisfying_integers := two_digit_integers.filter (λ n => n < 81)
  let probability := (satisfying_integers.card : ℚ) / (two_digit_integers.card : ℚ)
  probability

theorem probability_sqrt_two_digit_lt_nine_correct :
  probability_sqrt_two_digit_lt_nine = 71 / 90 := by
  sorry

end probability_sqrt_two_digit_lt_nine_correct_l114_114438


namespace max_product_300_l114_114137

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l114_114137


namespace probability_sqrt_lt_9_l114_114401

theorem probability_sqrt_lt_9 : 
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
in probability = 71 / 90 :=
by
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  sorry

end probability_sqrt_lt_9_l114_114401


namespace am_gm_inequality_l114_114983

theorem am_gm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
begin
  sorry
end

end am_gm_inequality_l114_114983


namespace max_product_of_sum_300_l114_114217

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l114_114217


namespace exists_composite_l114_114665

theorem exists_composite (x y : ℕ) (hx : 2 ≤ x ∧ x ≤ 100) (hy : 2 ≤ y ∧ y ≤ 100) :
  ∃ n : ℕ, ∃ k : ℕ, x^(2^n) + y^(2^n) = k * (k + 1) :=
by {
  sorry -- proof goes here
}

end exists_composite_l114_114665


namespace max_product_of_sum_300_l114_114212

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l114_114212


namespace triangle_angle_contradiction_l114_114788

theorem triangle_angle_contradiction (A B C : ℝ) (hA : A > 60) (hB : B > 60) (hC : C > 60) (h_sum : A + B + C = 180) :
  false :=
by
  -- Here "A > 60, B > 60, C > 60 and A + B + C = 180" leads to a contradiction
  sorry

end triangle_angle_contradiction_l114_114788


namespace equilateral_triangles_similar_l114_114872

theorem equilateral_triangles_similar :
  ∀ (Δ1 Δ2 : Triangle), (equilateral Δ1) → (equilateral Δ2) → (similar Δ1 Δ2) :=
begin
  sorry
end

end equilateral_triangles_similar_l114_114872


namespace probability_sqrt_less_than_nine_l114_114466

theorem probability_sqrt_less_than_nine :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let T := {n : ℕ | n ∈ S ∧ n < 81}
  (T.card : ℚ) / S.card = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114466


namespace solution_set_l114_114081

variable (a b c x : ℝ)

def condition1 := ∀ x, -1 < x ∧ x < 2 → ax^2 + bx + c > 0
def condition2 := ∀ x, x = -1 ∨ x = 2 → ax^2 + bx + c = 0
def condition3 := a < 0

theorem solution_set (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a) :
  {x | 0 < x ∧ x < 3} = {x | a * (x^2 + 1) + b * (x - 1) + c > 2 * a * x} := sorry

end solution_set_l114_114081


namespace probability_sqrt_lt_nine_l114_114408

theorem probability_sqrt_lt_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  let probability := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  probability = 71 / 90 :=
by
  let two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n : ℕ | 10 ≤ n ∧ n < 81}
  have h1 : two_digit_numbers.card = 90 := sorry
  have h2 : valid_numbers.card = 71 := sorry
  let probability : ℚ := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  rw [h1, h2]
  simp
  norm_num

end probability_sqrt_lt_nine_l114_114408


namespace max_product_300_l114_114141

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l114_114141


namespace max_product_of_two_integers_whose_sum_is_300_l114_114176

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l114_114176


namespace probability_sqrt_lt_9_l114_114419

theorem probability_sqrt_lt_9 : 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 →
  ∃ p : ℚ, p = 71 / 90 ∧ 
  ∑ k in (Finset.range 100).filter (λ x, 10 ≤ x ∧ sqrt x < 9), 1 / 90 = p := 
sorry

end probability_sqrt_lt_9_l114_114419


namespace prob_sqrt_less_than_nine_l114_114483

/-- The probability that the square root of a randomly selected 
two-digit whole number is less than nine is 71/90. -/
theorem prob_sqrt_less_than_nine : (let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99};
                                     let A := {n : ℕ | 10 ≤ n ∧ n < 81};
                                     (A.card / S.card : ℚ) = 71 / 90) :=
by
  sorry

end prob_sqrt_less_than_nine_l114_114483


namespace w_bounded_1_w_bounded_2_w_bounded_general_w_approx_l114_114034

noncomputable def w (k : ℕ) : ℝ := 
  sorry -- placeholder for the actual definition of w

theorem w_bounded_1 (k : ℕ) (hk : 2 ≤ k) :
  sqrt (3 * (1/2)^2 * (3/4) * (1/(2*k))) ≤ w (2*k) ∧
  w (2*k) < sqrt (3 * (1/2)^2 * (1/(2*k))) := sorry

theorem w_bounded_2 (k : ℕ) (hk : 3 ≤ k) :
  sqrt (5 * (1/2 * 3/4)^2 * (5/6) * (1/(2*k))) ≤ w (2*k) ∧
  w (2*k) < sqrt (5 * (1/2 * 3/4)^2 * (1/(2*k))) := sorry 

theorem w_bounded_general (k : ℕ) (a : ℕ) (hk : a ≤ k) :
  sqrt ((2*a-1) * ((1/2) * (3/4) * ... * ((2*a-3)/(2*a-2)))^2 * ((2*a-1)/(2*a)) * (1/(2*k))) ≤ w (2*k) ∧
  w (2*k) < sqrt ((2*a-1) * ((1/2) * (3/4) * ... * ((2*a-3)/(2*a-2)))^2 * (1/(2*k))) := sorry

theorem w_approx (k : ℕ) (hk : 25 ≤ k) :
  abs (w (2*k) - 1 / sqrt (π * k)) < ε := sorry

end w_bounded_1_w_bounded_2_w_bounded_general_w_approx_l114_114034


namespace greatest_product_sum_300_l114_114123

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l114_114123


namespace max_product_300_l114_114140

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l114_114140


namespace flagpole_breakage_l114_114891

-- The flagpole problem translated to a Lean 4 statement:
theorem flagpole_breakage (a b : ℝ) (h₁ : a = 8) (h₂ : b = 3) (h₃ : x = real.sqrt (a^2 + b^2) / (2 * b)) :
  x = 4.6 :=
by {
  rw [h₁, h₂] at h₃,
  norm_num at h₃,
  sorry
}

end flagpole_breakage_l114_114891


namespace probability_sqrt_less_than_nine_l114_114454

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sqrt_less_than_nine (n : ℕ) : Prop := n < 81

theorem probability_sqrt_less_than_nine :
  (∃ total_num favorable_num,
    (total_num = Finset.card (Finset.filter is_two_digit (Finset.range 100)) ∧
     favorable_num = Finset.card (Finset.filter (λ n, is_two_digit n ∧ sqrt_less_than_nine n) (Finset.range 100)) ∧
     (favorable_num : ℚ) / total_num = 71 / 90)) :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114454


namespace average_age_of_team_l114_114056

theorem average_age_of_team :
  (∀ (A : ℕ),
   let total_age_team := 11 * A,
       age_captain := 25,
       age_wicket_keeper := 25 + 5,
       age_remaining_players := 9 * (A - 1),
       total_age_remaining := age_remaining_players + age_captain + age_wicket_keeper
   in total_age_team = total_age_remaining) →
  A = 32 :=
by
  sorry

end average_age_of_team_l114_114056


namespace greatest_product_of_sum_eq_300_l114_114110

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l114_114110


namespace parallel_lines_coplanar_l114_114673

axiom Plane : Type
axiom Point : Type
axiom Line : Type

axiom A : Point
axiom B : Point
axiom C : Point
axiom D : Point

axiom α : Plane
axiom β : Plane

axiom in_plane (p : Point) (π : Plane) : Prop
axiom parallel_plane (π1 π2 : Plane) : Prop
axiom parallel_line (l1 l2 : Line) : Prop
axiom line_through (P Q : Point) : Line
axiom coplanar (P Q R S : Point) : Prop

-- Conditions
axiom A_in_α : in_plane A α
axiom C_in_α : in_plane C α
axiom B_in_β : in_plane B β
axiom D_in_β : in_plane D β
axiom α_parallel_β : parallel_plane α β

-- Statement
theorem parallel_lines_coplanar :
  parallel_line (line_through A C) (line_through B D) ↔ coplanar A B C D :=
sorry

end parallel_lines_coplanar_l114_114673


namespace find_PQ_l114_114957

-- Definitions of the conditions
variable (P Q R : Point)
variable (PR PQ : ℝ)
variable (angleQPR angleQRP : ℝ)
variable (rightTriangle : is_right_triangle P Q R)
variable (angle30 : angleQRP = 30)
variable (PR_length : PR = 9 * √3)

-- Theorem statement
theorem find_PQ (hPQ : PQ = length (P - Q) (Q - P)) :
  PQ = 27 := sorry

end find_PQ_l114_114957


namespace product_of_any_five_greater_l114_114084

theorem product_of_any_five_greater (a : ℤ) (h : a > 1) :
  ∃ (nums : Fin 11 → ℤ), 
    (nums 0 = a) ∧ 
    (∀ i, 1 ≤ i → i < 11 → nums i = -1) ∧
    (∀ s : Finset (Fin 11), s.card = 5 → 
      (s.prod (λ i, nums i)) > ((Finset.univ \ s).prod (λ i, nums i))) :=
begin
  sorry
end

end product_of_any_five_greater_l114_114084


namespace dogs_grouping_l114_114052

-- Define the problem conditions
variables (dogs : Finset ℕ) (B W : ℕ)
hypothesis h1 : dogs.card = 12
hypothesis hB : B ∈ dogs 
hypothesis hW : W ∈ dogs
-- Define the groups
definition group_A : Finset ℕ := {B}
definition group_B : Finset ℕ := {W}
definition remaining_dogs : Finset ℕ := dogs \ {B, W}

-- Ensure dogs Buster and Whiskers are present in their respective groups
hypothesis group_A_cond : B ∈ group_A
hypothesis group_B_cond : W ∈ group_B

-- Define the tasks
theorem dogs_grouping : ∃ groups : Finset (Finset ℕ), 
  (∀ group ∈ groups, group.card = 4 ∨ group.card = 5 ∨ group.card = 3) ∧
  B ∈ group_A ∧
  W ∈ group_B ∧
  groups.card = 3 ∧
  group_A.card = 4 ∧
  group_B.card = 5 ∧
  (∃ group_C : Finset ℕ, group_C.card = 3 ∧ group_C ⊆ remaining_dogs) ∧
  ((binomial 10 3) * (binomial 7 4) = 4200) := sorry

end dogs_grouping_l114_114052


namespace probability_sqrt_less_than_nine_l114_114449

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sqrt_less_than_nine (n : ℕ) : Prop := n < 81

theorem probability_sqrt_less_than_nine :
  (∃ total_num favorable_num,
    (total_num = Finset.card (Finset.filter is_two_digit (Finset.range 100)) ∧
     favorable_num = Finset.card (Finset.filter (λ n, is_two_digit n ∧ sqrt_less_than_nine n) (Finset.range 100)) ∧
     (favorable_num : ℚ) / total_num = 71 / 90)) :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114449


namespace fourth_term_correct_l114_114589

def fourth_term_sequence : Nat :=
  4^0 + 4^1 + 4^2 + 4^3

theorem fourth_term_correct : fourth_term_sequence = 85 :=
by
  sorry

end fourth_term_correct_l114_114589


namespace quadratic_linear_term_coefficient_l114_114566

theorem quadratic_linear_term_coefficient:
  ∀ (a b c: ℝ) (x: ℝ), (3*x^2 = 8*x + 10) → (b = -8) :=
by
  assume a b c x h,
  sorry

end quadratic_linear_term_coefficient_l114_114566


namespace max_product_300_l114_114148

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l114_114148


namespace parabola_c_value_l114_114557

theorem parabola_c_value (b c : ℝ) 
  (h1 : 20 = 2*(-2)^2 + b*(-2) + c) 
  (h2 : 28 = 2*2^2 + b*2 + c) : 
  c = 16 :=
by
  sorry

end parabola_c_value_l114_114557


namespace probability_sqrt_lt_9_l114_114422

theorem probability_sqrt_lt_9 : 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 →
  ∃ p : ℚ, p = 71 / 90 ∧ 
  ∑ k in (Finset.range 100).filter (λ x, 10 ≤ x ∧ sqrt x < 9), 1 / 90 = p := 
sorry

end probability_sqrt_lt_9_l114_114422


namespace max_product_of_sum_300_l114_114277

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l114_114277


namespace part_one_part_two_l114_114795

-- Part (1): Simplify and evaluate the expression
theorem part_one : ( (25/9)^(1/2) + (1/10)^(-2) - real.pi^0 + 1/3 ) = 101 := 
by sorry

-- Part (2): Simplify the given algebraic expression under conditions x > 0 and y > 0
theorem part_two (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x * y^2 * x^(1/2) * y^(-1/2))^(1/3) * (x * y)^(1/2) = x * y :=
by sorry

end part_one_part_two_l114_114795


namespace parabola_focus_coordinates_l114_114658

noncomputable def parabola_focus (a b : ℝ) := (0, (1 / (4 * a)) + 2)

theorem parabola_focus_coordinates (a b : ℝ) (h₀ : a ≠ 0) (h₁ : ∀ x : ℝ, abs (a * x^2 + b * x + 2) ≥ 2) :
  parabola_focus a b = (0, 2 + (1 / (4 * a))) := sorry

end parabola_focus_coordinates_l114_114658


namespace second_solution_salt_percent_l114_114533

theorem second_solution_salt_percent (S : ℝ) (x : ℝ) 
  (h1 : 0.14 * S - 0.14 * (S / 4) + (x / 100) * (S / 4) = 0.16 * S) : 
  x = 22 :=
by 
  -- Proof omitted
  sorry

end second_solution_salt_percent_l114_114533


namespace triangle_median_sum_l114_114741

theorem triangle_median_sum (a b c : ℝ) (h : a ≥ b ∧ b ≥ c) :
  ∃ s, s = (2/3) * (sqrt (2*b^2 + 2*c^2 - a^2) +
                    sqrt (2*a^2 + 2*c^2 - b^2) +
                    sqrt (2*a^2 + 2*b^2 - c^2)) :=
sorry

end triangle_median_sum_l114_114741


namespace magnitude_of_v_l114_114050

theorem magnitude_of_v (u v : ℂ) (h1 : u * v = 20 - 15 * complex.i) (h2 : complex.abs u = 5) : complex.abs v = 5 :=
sorry

end magnitude_of_v_l114_114050


namespace max_product_of_sum_300_l114_114270

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l114_114270


namespace number_division_l114_114861

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end number_division_l114_114861


namespace no_equilateral_triangle_on_grid_regular_tetrahedron_on_grid_l114_114528

-- Define the context for part (a)
theorem no_equilateral_triangle_on_grid (x1 y1 x2 y2 x3 y3 : ℤ) :
  ¬ (x1 = x2 ∧ y1 = y2) ∧ (x2 = x3 ∧ y2 = y3) ∧ (x3 = x1 ∧ y3 = y1) ∧ -- vertices must not be the same
  ((x2 - x1)^2 + (y2 - y1)^2 = (x3 - x2)^2 + (y3 - y2)^2) ∧ -- sides must be equal
  ((x3 - x1)^2 + (y3 - y1)^2 = (x2 - x1)^2 + (y2 - y1)^2) ->
  false := 
sorry

-- Define the context for part (b)
theorem regular_tetrahedron_on_grid (x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 : ℤ) :
  ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2 = (x3 - x2)^2 + (y3 - y2)^2 + (z3 - z2)^2) ∧ -- first condition: edge lengths equal
  ((x3 - x1)^2 + (y3 - y1)^2 + (z3 - z1)^2 = (x4 - x3)^2 + (y4 - y3)^2 + (z4 - z3)^2) ∧ -- second condition: edge lengths equal
  ((x4 - x1)^2 + (y4 - y1)^2 + (z4 - z1)^2 = (x2 - x4)^2 + (y2 - y4)^2 + (z2 - z4)^2) -> -- third condition: edge lengths equal
  true := 
sorry

end no_equilateral_triangle_on_grid_regular_tetrahedron_on_grid_l114_114528


namespace probability_sqrt_less_nine_l114_114504

theorem probability_sqrt_less_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  ∃ (p : ℚ), p = (finset.card (valid_numbers.to_finset) : ℚ) / (finset.card (two_digit_numbers.to_finset) : ℚ) ∧ p = 71 / 90 :=
by
  sorry

end probability_sqrt_less_nine_l114_114504


namespace total_ways_to_split_is_12_l114_114835

-- Define the set of people
def people : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define the knows relationship based on given conditions
def knows (a b : ℕ) : Prop :=
  (a = b + 1) ∨ (a = b - 1) ∨ (a = b + 2) ∨ (a = b - 2) ∨ (a = b + 6) ∨ (a = b - 6)

-- Define a valid pairing
def valid_pairing (pairs : Finset (ℕ × ℕ)) : Prop :=
  pairs.card = 4 ∧
  (∀ {a b c}, (a, b) ∈ pairs → (c, a) ∈ pairs → c = b ∨ c = a) ∧
  (∀ (a b : ℕ), (a, b) ∈ pairs → knows a b)

-- Define the total valid pair configurations
def total_valid_pairs (people : Finset ℕ) : ℕ :=
  (Finset.powerset people).filter (λ pairs, valid_pairing pairs).card

theorem total_ways_to_split_is_12 :
  total_valid_pairs people = 12 := sorry

end total_ways_to_split_is_12_l114_114835


namespace probability_sqrt_less_than_nine_l114_114335

/-- Define the set of two-digit integers --/
def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Define the condition that the square root of the number is less than 9 --/
def sqrt_less_than_nine (n : Nat) : Prop := n < 81

/-- The number of integers from 10 to 80 --/
lemma count_satisfying_sqrt (n : Nat) : Prop :=
  is_two_digit n ∧ sqrt_less_than_nine n → n < 81

/-- Total number of two-digit integers --/
lemma count_two_digit_total (n : Nat) : Prop := is_two_digit n 

/-- The probability that a randomly selected two-digit integer's square root is less than 9. --/
theorem probability_sqrt_less_than_nine : 
  (∃ n, count_satisfying_sqrt n) / (∃ n, count_two_digit_total n) = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114335


namespace greatest_product_sum_300_l114_114134

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l114_114134


namespace greatest_product_of_two_integers_with_sum_300_l114_114234

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l114_114234


namespace regular_polygon_perimeter_l114_114902

theorem regular_polygon_perimeter
  (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ)
  (h1 : side_length = 8)
  (h2 : exterior_angle = 90)
  (h3 : n = 360 / exterior_angle) :
  n * side_length = 32 := by
  sorry

end regular_polygon_perimeter_l114_114902


namespace edward_made_in_summer_l114_114953

def edward_made_in_spring := 2
def cost_of_supplies := 5
def money_left_over := 24

theorem edward_made_in_summer : edward_made_in_spring + x - cost_of_supplies = money_left_over → x = 27 :=
by
  intros h
  sorry

end edward_made_in_summer_l114_114953


namespace max_product_sum_300_l114_114302

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l114_114302


namespace sum_possible_x_eq_16_5_l114_114079

open Real

noncomputable def sum_of_possible_x : Real :=
  let a := 2
  let b := -33
  let c := 87
  (-b) / (2 * a)

theorem sum_possible_x_eq_16_5 : sum_of_possible_x = 16.5 :=
  by
    -- The actual proof goes here
    sorry

end sum_possible_x_eq_16_5_l114_114079


namespace probability_sqrt_less_than_nine_is_correct_l114_114386

def probability_sqrt_less_than_nine : ℚ :=
  let total_two_digit_numbers := 99 - 10 + 1 in
  let satisfying_numbers := 80 - 10 + 1 in
  let probability := (satisfying_numbers : ℚ) / (total_two_digit_numbers : ℚ) in
  probability

theorem probability_sqrt_less_than_nine_is_correct :
  probability_sqrt_less_than_nine = 71 / 90 :=
by
  -- proof here
  sorry

end probability_sqrt_less_than_nine_is_correct_l114_114386


namespace probability_of_sqrt_lt_9_l114_114351

-- Define the set of two-digit whole numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the subset of numbers for which the square root is less than 9
def valid_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 80}

-- Define the probability calculation
noncomputable def probability_sqrt_lt_9 := (valid_numbers.to_finset.card : ℝ) / (two_digit_numbers.to_finset.card : ℝ)

-- The statement we aim to prove
theorem probability_of_sqrt_lt_9 : probability_sqrt_lt_9 = 71 / 90 := 
sorry

end probability_of_sqrt_lt_9_l114_114351


namespace matrix_power_2023_correct_l114_114931

noncomputable def matrix_power_2023 : Matrix (Fin 2) (Fin 2) ℤ :=
  let A := !![1, 0; 2, 1]  -- Define the matrix
  A^2023

theorem matrix_power_2023_correct :
  matrix_power_2023 = !![1, 0; 4046, 1] := by
  sorry

end matrix_power_2023_correct_l114_114931


namespace greatest_product_obtainable_l114_114323

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l114_114323


namespace max_product_sum_300_l114_114306

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l114_114306


namespace log_sum_identity_l114_114688

def f (x : ℝ) : ℝ := 1 / (2 ^ x + 1)

theorem log_sum_identity : f (Real.log 3 / Real.log 2) + f (-Real.log 3 / Real.log 2) = 1 :=
by
  sorry

end log_sum_identity_l114_114688


namespace max_product_two_integers_l114_114182

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l114_114182


namespace initial_ratio_of_milk_water_l114_114896

theorem initial_ratio_of_milk_water (M W : ℝ) (H1 : M + W = 85) (H2 : M / (W + 5) = 3) : M / W = 27 / 7 :=
by sorry

end initial_ratio_of_milk_water_l114_114896


namespace probability_sqrt_lt_9_l114_114420

theorem probability_sqrt_lt_9 : 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 →
  ∃ p : ℚ, p = 71 / 90 ∧ 
  ∑ k in (Finset.range 100).filter (λ x, 10 ≤ x ∧ sqrt x < 9), 1 / 90 = p := 
sorry

end probability_sqrt_lt_9_l114_114420


namespace smaug_hoard_value_l114_114800

theorem smaug_hoard_value : 
  let gold_coins := 100
  let silver_coins := 60
  let copper_coins := 33
  let copper_per_silver := 8
  let silver_per_gold := 3
  let value_in_copper :=
    (gold_coins * silver_per_gold * copper_per_silver) +
    (silver_coins * copper_per_silver) +
    copper_coins
  in value_in_copper = 2913 := 
by 
  sorry

end smaug_hoard_value_l114_114800


namespace probability_sqrt_less_than_nine_is_correct_l114_114375

def probability_sqrt_less_than_nine : ℚ :=
  let total_two_digit_numbers := 99 - 10 + 1 in
  let satisfying_numbers := 80 - 10 + 1 in
  let probability := (satisfying_numbers : ℚ) / (total_two_digit_numbers : ℚ) in
  probability

theorem probability_sqrt_less_than_nine_is_correct :
  probability_sqrt_less_than_nine = 71 / 90 :=
by
  -- proof here
  sorry

end probability_sqrt_less_than_nine_is_correct_l114_114375


namespace rectangle_circle_intersection_area_l114_114093

open Real

theorem rectangle_circle_intersection_area :
  let rect_vertices := [(3, 9), (20, 9), (20, -6), (3, -6)]
      circle_eq := ∀ x y, (x - 3)^2 + (y + 6)^2 = 25
  in ∃ area, area = 25 / 4 * π :=
by
  sorry

end rectangle_circle_intersection_area_l114_114093


namespace number_of_true_props_l114_114683

-- Define the propositions
def prop1 (p q : Prop) : Prop := ¬(p ∧ q) → ¬p ∧ ¬q
def prop2 (a b : ℝ) : Prop := (¬(a > b → 2^a > 2^b - 1)) = (a ≤ b → 2^a ≤ 2^b - 1)
def prop3 : Prop := (¬(∀ x : ℝ, x^2 + 1 ≥ 0)) = (∃ x : ℝ, x^2 + 1 < 0)
def prop4 (f : ℝ → ℝ) (x₀ : ℝ) [Differentiable ℝ f] : Prop := 
  let p := deriv f x₀ = 0 in
  let q := LocalExtremum f x₀ in
  p → q ∧ ¬(q → p)

-- Prove the number of true propositions is 3
theorem number_of_true_props : 3 = [prop2, prop3, prop4].count (λ p, p) sorry

end number_of_true_props_l114_114683


namespace part_a_part_b_l114_114539

def good_tuple {n : ℕ} (h : (Fin (n+1) → (Fin n → ℝ) → ℝ)) : Prop :=
  ∀ (f : Fin n → ℝ → ℝ), 
    (∀ i, (h i) (λ k, f k (i : ℝ)) = λ x, sorry) →
    (∀ k, sorry is_polynomial f k)

theorem part_a : ∀ (n : ℕ), ∃ h : (Fin (n+1) → (Fin n → ℝ) → ℝ),
  good_tuple h ∧ (∀ i, degree of (h i) > 1) :=
sorry

theorem part_b : ∀ (n : ℕ), n > 1 → 
  ¬(∃ h : (Fin (n+1) → (Fin n → ℝ) → ℝ), good_tuple h ∧ (∀ i, is_symmetric_polynomial (h i))) :=
sorry

end part_a_part_b_l114_114539


namespace greatest_product_of_two_integers_with_sum_300_l114_114231

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l114_114231


namespace part1_part2_l114_114689

-- Part 1
noncomputable def f (x a : ℝ) : ℝ := (x - 1) * Real.exp x - (1/3) * a * x ^ 3 - (1/2) * x ^ 2

noncomputable def f' (x a : ℝ) : ℝ := x * Real.exp x - a * x ^ 2 - x

noncomputable def g (x a : ℝ) : ℝ := f' x a / x

theorem part1 (a : ℝ) (h : a > 0) : g a a > 0 := by
  sorry

-- Part 2
theorem part2 (a : ℝ) (h : ∃ x, f' x a = 0) : a > 0 := by
  sorry

end part1_part2_l114_114689


namespace greatest_product_two_ints_sum_300_l114_114268

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l114_114268


namespace max_product_two_integers_sum_300_l114_114153

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l114_114153


namespace max_product_two_integers_sum_300_l114_114162

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l114_114162


namespace magnitude_of_angle_A_area_of_triangle_ABC_l114_114742

variables {S a b c A : ℝ}
variables {AB AC : ℝ}

-- Question (I)
theorem magnitude_of_angle_A
  (h1 : S = (sqrt 3 / 2) * AB * AC)
  (h2 : AB = b)
  (h3 : AC = c)
  (h4 : S = (1 / 2) * b * c * sin A) :
  A = π / 3 :=
sorry

-- Question (II)
theorem area_of_triangle_ABC
  (h1 : b + c = 5)
  (h2 : a = sqrt 7)
  (h3 : A = π / 3) :
  S = (3 * sqrt 3) / 2 :=
sorry

end magnitude_of_angle_A_area_of_triangle_ABC_l114_114742


namespace probability_sqrt_less_than_nine_is_correct_l114_114376

def probability_sqrt_less_than_nine : ℚ :=
  let total_two_digit_numbers := 99 - 10 + 1 in
  let satisfying_numbers := 80 - 10 + 1 in
  let probability := (satisfying_numbers : ℚ) / (total_two_digit_numbers : ℚ) in
  probability

theorem probability_sqrt_less_than_nine_is_correct :
  probability_sqrt_less_than_nine = 71 / 90 :=
by
  -- proof here
  sorry

end probability_sqrt_less_than_nine_is_correct_l114_114376


namespace count_valid_n_le_30_l114_114973

theorem count_valid_n_le_30 :
  ∀ n : ℕ, (0 < n ∧ n ≤ 30) → (n! * 2) % (n * (n + 1)) = 0 := by
  sorry

end count_valid_n_le_30_l114_114973


namespace trig_identity_l114_114669

variable {α : ℝ}

theorem trig_identity (h : 270 * Real.pi / 180 < α ∧ α < 360 * Real.pi / 180) :
  (sqrt (1 / 2 + 1 / 2 * sqrt (1 / 2 + 1 / 2 * cos (2 * α)))) = -cos (α / 2) := by
  sorry

end trig_identity_l114_114669


namespace max_product_300_l114_114146

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l114_114146


namespace min_length_BC_l114_114538

theorem min_length_BC (A B C D : Type) (AB AC DC BD BC : ℝ) :
  AB = 8 → AC = 15 → DC = 10 → BD = 25 → (BC > AC - AB) ∧ (BC > BD - DC) → BC ≥ 15 :=
by
  intros hAB hAC hDC hBD hIneq
  sorry

end min_length_BC_l114_114538


namespace probability_sqrt_lt_9_l114_114392

theorem probability_sqrt_lt_9 : 
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
in probability = 71 / 90 :=
by
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  sorry

end probability_sqrt_lt_9_l114_114392


namespace probability_sqrt_lt_nine_l114_114406

theorem probability_sqrt_lt_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  let probability := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  probability = 71 / 90 :=
by
  let two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n : ℕ | 10 ≤ n ∧ n < 81}
  have h1 : two_digit_numbers.card = 90 := sorry
  have h2 : valid_numbers.card = 71 := sorry
  let probability : ℚ := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  rw [h1, h2]
  simp
  norm_num

end probability_sqrt_lt_nine_l114_114406


namespace probability_sqrt_lt_9_l114_114391

theorem probability_sqrt_lt_9 : 
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
in probability = 71 / 90 :=
by
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  sorry

end probability_sqrt_lt_9_l114_114391


namespace greatest_product_sum_300_l114_114125

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l114_114125


namespace parabola_equation_l114_114065

theorem parabola_equation (a b c : ℝ) (h1 : a^2 = 3) (h2 : b^2 = 1) (h3 : c^2 = a^2 + b^2) : 
  (c = 2) → (vertex = 0) → (focus = 2) → ∀ x y, y^2 = 16 * x := 
by 
  sorry

end parabola_equation_l114_114065


namespace evaluate_expression_l114_114590

def f (x : ℝ) : ℝ := x^3 + x^2 + 2 * real.sqrt x

theorem evaluate_expression : 2 * f 3 - f 9 = -744 + 4 * real.sqrt 3 :=
by
  sorry

end evaluate_expression_l114_114590


namespace max_product_300_l114_114136

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l114_114136


namespace number_division_l114_114858

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end number_division_l114_114858


namespace max_product_two_integers_l114_114185

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l114_114185


namespace max_intersections_circle_quadrilateral_l114_114849

theorem max_intersections_circle_quadrilateral (circle : Type) (quadrilateral : Type) 
  (intersects : circle → quadrilateral → ℕ) (h : ∀ (c : circle) (line_segment : Type), intersects c line_segment ≤ 2) :
  ∃ (q : quadrilateral), intersects circle quadrilateral = 8 :=
by
  sorry

end max_intersections_circle_quadrilateral_l114_114849


namespace primes_between_30_and_70_greater_than_35_l114_114708

def is_prime (n : ℕ) : Prop := Nat.Prime n

def count_primes_in_range (a b : ℕ) : ℕ :=
  List.length (List.filter is_prime (List.range' a (b - a)))

theorem primes_between_30_and_70_greater_than_35 : count_primes_in_range 36 71 = 8 :=
by
  sorry

end primes_between_30_and_70_greater_than_35_l114_114708


namespace find_circle_equation_l114_114623

theorem find_circle_equation (a b r : ℝ)
  (h1 : 2 * a - b = 7)
  (h2 : a^2 + (-4 - b)^2 = r^2)
  (h3 : a^2 + (-2 - b)^2 = r^2) :
  (∃ a b r, ((x : ℝ) - a)^2 + ((y : ℝ) - b)^2 = r ^ 2) :=
by {
  use [2, -3, 5],
  sorry
}

end find_circle_equation_l114_114623


namespace wilsons_theorem_factorial_mod_square_l114_114527

theorem wilsons_theorem (p : ℕ) : 
  (∃ k : ℕ, (p - 1)! + 1 = k * p) ↔ (nat.prime p) := sorry

theorem factorial_mod_square (p : ℕ) : 
  (p.prime → ((p - 1)! ^ 2 ≡ 1 [MOD p])) ∧ 
  (¬ p.prime → ((p - 1)! ^ 2 ≡ 0 [MOD p])) := sorry

end wilsons_theorem_factorial_mod_square_l114_114527


namespace trigonometric_identity_l114_114580

theorem trigonometric_identity : 
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = Real.csc (20 * Real.pi / 180) := 
by
  sorry

end trigonometric_identity_l114_114580


namespace other_root_of_quadratic_l114_114725

theorem other_root_of_quadratic (a b : ℝ) (h : a ≠ 0) (h_eq : a * 2^2 = b) : 
  ∃ m : ℝ, a * m^2 = b ∧ 2 + m = 0 :=
begin
  use -2,
  split,
  { rw [mul_pow, h_eq, pow_two, mul_assoc, mul_comm 2, ←mul_assoc, mul_comm a, pow_two (-2)],
    sorry },
  { linarith }
end

end other_root_of_quadratic_l114_114725


namespace total_jellybeans_l114_114921

def nephews := 3
def nieces := 2
def jellybeans_per_child := 14
def children := nephews + nieces

theorem total_jellybeans : children * jellybeans_per_child = 70 := by
  sorry

end total_jellybeans_l114_114921


namespace cos_135_eq_neg_sqrt2_div_2_sin_135_eq_sqrt2_div_2_l114_114934

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

theorem sin_135_eq_sqrt2_div_2 : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by sorry

end cos_135_eq_neg_sqrt2_div_2_sin_135_eq_sqrt2_div_2_l114_114934


namespace find_other_parallel_side_length_l114_114625

variable (a b h A : ℝ)

-- Conditions
def length_one_parallel_side := a = 18
def distance_between_sides := h = 12
def area_trapezium := A = 228
def trapezium_area_formula := A = 1 / 2 * (a + b) * h

-- Target statement to prove
theorem find_other_parallel_side_length
    (h1 : length_one_parallel_side a)
    (h2 : distance_between_sides h)
    (h3 : area_trapezium A)
    (h4 : trapezium_area_formula a b h A) :
    b = 20 :=
sorry

end find_other_parallel_side_length_l114_114625


namespace general_term_formula_sum_Tn_number_of_sign_changes_l114_114660

variable (a : ℚ)

-- Declaring sequences and their conditions
def sequence (n : ℕ) : ℚ := if n = 1 then 1 else 2 * n - 5
def Sn (n : ℕ) : ℚ := n^2 - 4 * n + 4
def bn (n : ℕ) : ℚ := if n = 1 then 1 else (sequence n + 5) / 2
def Tn (n : ℕ) : ℚ := ∑ k in finset.range n, bn (k + 1) * 2^(k + 1)
def cn (n : ℕ) : ℚ := 1 - a / sequence n

-- Stating the problems
theorem general_term_formula (n : ℕ) :
  sequence n = if n = 1 then 1 else 2 * n - 5 := 
sorry

theorem sum_Tn (n : ℕ) :
  Tn n = (n-1) * 2^(n+1) + 4 :=
sorry

theorem number_of_sign_changes (a : ℚ) :
  let sign_changes := ∑ i in finset.range (n - 1), if (cn a i) * (cn a (i + 1)) < 0 then 1 else 0
  in sign_changes = 3 :=
sorry

end general_term_formula_sum_Tn_number_of_sign_changes_l114_114660


namespace probability_sqrt_lt_9_of_two_digit_l114_114370

-- Define the set of two-digit whole numbers
def two_digit_whole_numbers : set ℕ := {n | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate that checks if the square root of a number is less than 9
def sqrt_lt_9 (n : ℕ) : Prop := (n : ℝ)^2 < (9 : ℝ)^2

-- Calculate the probability
theorem probability_sqrt_lt_9_of_two_digit :
  let eligible_numbers := { n ∈ two_digit_whole_numbers | sqrt_lt_9 n } in
  (eligible_numbers.to_finset.card : ℚ) / (two_digit_whole_numbers.to_finset.card : ℚ) =
  71 / 90 :=
by
  sorry

end probability_sqrt_lt_9_of_two_digit_l114_114370


namespace probability_sqrt_lt_9_of_two_digit_l114_114365

-- Define the set of two-digit whole numbers
def two_digit_whole_numbers : set ℕ := {n | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate that checks if the square root of a number is less than 9
def sqrt_lt_9 (n : ℕ) : Prop := (n : ℝ)^2 < (9 : ℝ)^2

-- Calculate the probability
theorem probability_sqrt_lt_9_of_two_digit :
  let eligible_numbers := { n ∈ two_digit_whole_numbers | sqrt_lt_9 n } in
  (eligible_numbers.to_finset.card : ℚ) / (two_digit_whole_numbers.to_finset.card : ℚ) =
  71 / 90 :=
by
  sorry

end probability_sqrt_lt_9_of_two_digit_l114_114365


namespace team_arrangement_count_l114_114569

-- Definitions of the problem
def veteran_players := 2
def new_players := 3
def total_players := veteran_players + new_players
def team_size := 3

-- Conditions
def condition_veteran : Prop := 
  ∀ (team : Finset ℕ), team.card = team_size → Finset.card (team ∩ (Finset.range veteran_players)) ≥ 1

def condition_new_player : Prop := 
  ∀ (team : Finset ℕ), team.card = team_size → 
    ∃ (p1 p2 : ℕ), p1 ∈ team ∧ p2 ∈ team ∧ 
    p1 ≠ p2 ∧ p1 < team_size ∧ p2 < team_size ∧
    (p1 ∈ (Finset.Ico veteran_players total_players) ∨ p2 ∈ (Finset.Ico veteran_players total_players))

-- Goal
def number_of_arrangements := 48

-- The statement to prove
theorem team_arrangement_count : condition_veteran → condition_new_player → 
  (∃ (arrangements : ℕ), arrangements = number_of_arrangements) :=
by
  sorry

end team_arrangement_count_l114_114569


namespace solve_for_a_l114_114999

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x >= 0 then 4 ^ x else 2 ^ (a - x)

theorem solve_for_a (a : ℝ) (h : a ≠ 1) (h_eq : f a (1 - a) = f a (a - 1)) : a = 1 / 2 := 
by {
  sorry
}

end solve_for_a_l114_114999


namespace greatest_product_sum_300_l114_114249

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l114_114249


namespace probability_sqrt_less_than_nine_l114_114340

/-- Define the set of two-digit integers --/
def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Define the condition that the square root of the number is less than 9 --/
def sqrt_less_than_nine (n : Nat) : Prop := n < 81

/-- The number of integers from 10 to 80 --/
lemma count_satisfying_sqrt (n : Nat) : Prop :=
  is_two_digit n ∧ sqrt_less_than_nine n → n < 81

/-- Total number of two-digit integers --/
lemma count_two_digit_total (n : Nat) : Prop := is_two_digit n 

/-- The probability that a randomly selected two-digit integer's square root is less than 9. --/
theorem probability_sqrt_less_than_nine : 
  (∃ n, count_satisfying_sqrt n) / (∃ n, count_two_digit_total n) = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114340


namespace number_division_l114_114859

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end number_division_l114_114859


namespace find_multiple_l114_114612

theorem find_multiple (n m : ℕ) (h_n : n = 5) (h_eq : m * n - 15 = 2 * n + 10) : m = 7 :=
by
  sorry

end find_multiple_l114_114612


namespace greatest_product_two_ints_sum_300_l114_114258

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l114_114258


namespace problem1_problem2_l114_114583

-- Problem 1: Proove that the given expression equals 1
theorem problem1 : (2021 * 2023) / (2022^2 - 1) = 1 :=
  by
  sorry

-- Problem 2: Proove that the given expression equals 45000
theorem problem2 : 2 * 101^2 + 2 * 101 * 98 + 2 * 49^2 = 45000 :=
  by
  sorry

end problem1_problem2_l114_114583


namespace average_of_xyz_l114_114709

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 15) : 
  (x + y + z) / 3 = 4 :=
sorry

end average_of_xyz_l114_114709


namespace a_value_l114_114647

theorem a_value (a : ℝ) (h : (2 + Complex.i) * (1 + a * Complex.i) = (0 : ℂ).im) : a = 2 :=
sorry

end a_value_l114_114647


namespace weight_equivalence_l114_114749

-- Define the conditions
variable (weight_orange weight_apple weight_pear : ℚ)
variable (h1 : 7 * weight_orange = 5 * weight_apple)
variable (h2 : 10 * weight_pear = 6 * weight_apple)
variable (jimmy_oranges : ℕ := 49)

-- Define the main statements to be proved
theorem weight_equivalence :
  (49 * weight_orange = 35 * weight_apple) ∧
  (49 * weight_orange ≈ 58 * weight_pear) :=
by {
  sorry
}

end weight_equivalence_l114_114749


namespace type_B_ratio_l114_114772

theorem type_B_ratio
    (num_A : ℕ)
    (total_bricks : ℕ)
    (other_bricks : ℕ)
    (h1 : num_A = 40)
    (h2 : total_bricks = 150)
    (h3 : other_bricks = 90) :
    (total_bricks - num_A - other_bricks) / num_A = 1 / 2 :=
by
  sorry

end type_B_ratio_l114_114772


namespace max_product_sum_300_l114_114304

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l114_114304


namespace magnitude_of_v_l114_114049

theorem magnitude_of_v (u v : ℂ) (h1 : u * v = 20 - 15 * complex.i) (h2 : complex.abs u = 5) : complex.abs v = 5 :=
sorry

end magnitude_of_v_l114_114049


namespace stock_percentage_decrease_l114_114561

theorem stock_percentage_decrease (x : ℝ) :
  1.30 * x * (1 - 30.77 / 100) ≈ 0.90 * x :=
by
  sorry

end stock_percentage_decrease_l114_114561


namespace max_product_300_l114_114149

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l114_114149


namespace greatest_product_sum_300_l114_114126

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l114_114126


namespace trains_crossing_time_l114_114097

noncomputable def time_to_cross_each_other (L T1 T2 : ℝ) (H1 : L = 120) (H2 : T1 = 10) (H3 : T2 = 16) : ℝ :=
  let S1 := L / T1
  let S2 := L / T2
  let S := S1 + S2
  let D := L + L
  D / S

theorem trains_crossing_time : time_to_cross_each_other 120 10 16 (by rfl) (by rfl) (by rfl) = 240 / (12 + 7.5) :=
  sorry

end trains_crossing_time_l114_114097


namespace probability_sqrt_two_digit_lt_nine_correct_l114_114436

noncomputable def probability_sqrt_two_digit_lt_nine : ℚ :=
  let two_digit_integers := finset.Icc 10 99
  let satisfying_integers := two_digit_integers.filter (λ n => n < 81)
  let probability := (satisfying_integers.card : ℚ) / (two_digit_integers.card : ℚ)
  probability

theorem probability_sqrt_two_digit_lt_nine_correct :
  probability_sqrt_two_digit_lt_nine = 71 / 90 := by
  sorry

end probability_sqrt_two_digit_lt_nine_correct_l114_114436


namespace average_value_l114_114712

theorem average_value (x y z : ℝ) (h : (5/4) * (x + y + z) = 15) : (x + y + z) / 3 = 4 := 
by
  have h1 : x + y + z = 15 * (4 / 5) := sorry
  have h2 : x + y + z = 12 := sorry
  have h3 : (x + y + z) / 3 = 12 / 3 := by rw [h2]
  have h4 : 12 / 3 = 4 := sorry
  rw [h4] at h3
  exact h3

end average_value_l114_114712


namespace Mirella_read_purple_books_l114_114952

theorem Mirella_read_purple_books (P : ℕ) 
  (pages_per_purple_book : ℕ := 230)
  (pages_per_orange_book : ℕ := 510)
  (orange_books_read : ℕ := 4)
  (extra_orange_pages : ℕ := 890)
  (total_orange_pages : ℕ := orange_books_read * pages_per_orange_book)
  (total_purple_pages : ℕ := P * pages_per_purple_book)
  (condition : total_orange_pages - total_purple_pages = extra_orange_pages) :
  P = 5 := 
by 
  sorry

end Mirella_read_purple_books_l114_114952


namespace greatest_product_obtainable_l114_114320

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l114_114320


namespace MrBirdsIdealSpeed_l114_114027

theorem MrBirdsIdealSpeed :
  ∃ r : ℚ, r = 53.3̅ ∧ 
  (∀ d t : ℚ, d = 40 * (t + 5/60) ∧ d = 60 * (t - 2/60) ∧ d = 50 * (t - 1/60) → 
  r = d / t) :=
sorry

end MrBirdsIdealSpeed_l114_114027


namespace no_adjacent_black_balls_l114_114981

theorem no_adjacent_black_balls (m n : ℕ) (h : m > n) : 
  (m + 1).choose n = (m + 1).factorial / (n.factorial * (m + 1 - n).factorial) := by
  sorry

end no_adjacent_black_balls_l114_114981


namespace problem_1_problem_2_l114_114024

def is_in_solution_set (x : ℝ) : Prop := -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0

variables {a b : ℝ}

theorem problem_1 (ha : is_in_solution_set a) (hb : is_in_solution_set b) :
  |(1 / 3) * a + (1 / 6) * b| < 1 / 4 :=
sorry

theorem problem_2 (ha : is_in_solution_set a) (hb : is_in_solution_set b) :
  |1 - 4 * a * b| > 2 * |a - b| :=
sorry

end problem_1_problem_2_l114_114024


namespace number_of_cases_for_Ds_hearts_l114_114951

theorem number_of_cases_for_Ds_hearts (hA : 5 ≤ 13) (hB : 4 ≤ 13) (dist : 52 % 4 = 0) : 
  ∃ n, n = 5 ∧ 0 ≤ n ∧ n ≤ 13 := sorry

end number_of_cases_for_Ds_hearts_l114_114951


namespace find_number_dividedBy_50_addedTo_7500_to_get_7525_l114_114515

-- Define the problem conditions
def satisfies_equation (x : ℝ) : Prop :=
  7500 + (x / 50) = 7525

-- State the theorem to prove
theorem find_number_dividedBy_50_addedTo_7500_to_get_7525 :
  ∃ x : ℝ, satisfies_equation x ∧ x = 1250 :=
by
  existsi 1250
  split
  · rfl
  · rfl
  sorry

end find_number_dividedBy_50_addedTo_7500_to_get_7525_l114_114515


namespace odd_function_range_of_a_l114_114013

noncomputable theory

variable {f : ℝ → ℝ}

-- Question 1: Prove that f is an odd function given the conditions
theorem odd_function
  (h1 : ∀ x y : ℝ, f(x + y) = f(x) + f(y))
  (h2 : ∀ x y : ℝ, x ≤ y → f(x) ≤ f(y)) :
  ∀ x : ℝ, f(-x) = -f(x) :=
sorry

-- Question 2: Prove the range of a such that the inequality has exactly three positive integer solutions
theorem range_of_a (h1 : ∀ x y : ℝ, f(x + y) = f(x) + f(y))
  (h2 : ∀ x y : ℝ, x ≤ y → f(x) ≤ f(y)) :
  {a : ℝ | (5 < a ∧ a ≤ 6) ∧ ∃! (x : ℕ+), x ∈ {n : ℕ | f(n ^ 2) - 2 * f(n) < f(a * n) - 2 * f(a) ∧ n = x} } :=
sorry

end odd_function_range_of_a_l114_114013


namespace midpoint_line_l114_114782

theorem midpoint_line (a : ℝ) (P Q M : ℝ × ℝ) (hP : P = (a, 5 * a + 3)) (hQ : Q = (3, -2))
  (hM : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) : M.2 = 5 * M.1 - 7 := 
sorry

end midpoint_line_l114_114782


namespace probability_sqrt_less_than_nine_l114_114445

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sqrt_less_than_nine (n : ℕ) : Prop := n < 81

theorem probability_sqrt_less_than_nine :
  (∃ total_num favorable_num,
    (total_num = Finset.card (Finset.filter is_two_digit (Finset.range 100)) ∧
     favorable_num = Finset.card (Finset.filter (λ n, is_two_digit n ∧ sqrt_less_than_nine n) (Finset.range 100)) ∧
     (favorable_num : ℚ) / total_num = 71 / 90)) :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114445


namespace greatest_product_of_sum_eq_300_l114_114109

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l114_114109


namespace fuel_relationship_l114_114832

theorem fuel_relationship (y : ℕ → ℕ) (h₀ : y 0 = 80) (h₁ : y 1 = 70) (h₂ : y 2 = 60) (h₃ : y 3 = 50) :
  ∀ x : ℕ, y x = 80 - 10 * x :=
by
  sorry

end fuel_relationship_l114_114832


namespace integer_root_is_neg_six_l114_114819

noncomputable def polynomial_integer_root
  (p q : ℚ)
  (h_poly : ∀ x : ℂ, x^3 + p*x + q = 0)
  (h_root1 : (3 - complex.sqrt 5))
  (h_root2 : (3 + complex.sqrt 5)) : ℤ :=
  sorry

theorem integer_root_is_neg_six
  (p q : ℚ)
  (h_poly : ∀ x : ℂ, x^3 + p*x + q = 0)
  (h_root1 : (3 - complex.sqrt 5))
  (h_root2 : (3 + complex.sqrt 5)) :
  polynomial_integer_root p q h_poly h_root1 h_root2 = -6 :=
sorry

end integer_root_is_neg_six_l114_114819


namespace probability_sqrt_two_digit_lt_nine_correct_l114_114437

noncomputable def probability_sqrt_two_digit_lt_nine : ℚ :=
  let two_digit_integers := finset.Icc 10 99
  let satisfying_integers := two_digit_integers.filter (λ n => n < 81)
  let probability := (satisfying_integers.card : ℚ) / (two_digit_integers.card : ℚ)
  probability

theorem probability_sqrt_two_digit_lt_nine_correct :
  probability_sqrt_two_digit_lt_nine = 71 / 90 := by
  sorry

end probability_sqrt_two_digit_lt_nine_correct_l114_114437


namespace point_BI_dot_BA_l114_114993

noncomputable def vector_len {α : Type} [NormedAddCommGroup α] [NormedSpace ℝ α] (v : α) := ∥v∥

theorem point_BI_dot_BA (A B C P I : ℝ^3)
  (λ : ℝ)
  (h_cond1 : vector_len (P - A) - vector_len (P - B) = 2)
  (h_cond2 : vector_len (P - A - (P - B)) = 2 * Real.sqrt 5)
  (h_cond3 : (P - A) • (P - C) / vector_len (P - A) = (P - B) • (P - C) / vector_len (P - B))
  (h_cond4 : B + λ • ((C - A) / vector_len (C - A) + (P - A) / vector_len (P - A)) = I)
  (h_λ : λ > 0) :
  (I - B) • (B - A) / vector_len (B - A) = Real.sqrt 5 - 1 :=
sorry

end point_BI_dot_BA_l114_114993


namespace probability_of_purple_l114_114576

noncomputable theory

def bagX := {red := 5, green := 3}
def bagY := {orange := 8, purple := 2}
def bagZ := {orange := 3, purple := 7}

-- Probability of drawing a marble given a specified condition
def draw_prob (bag : Bag) (color : Color) : ℚ :=
  match color with
  | red => bag.red / (bag.red + bag.green)
  | green => bag.green / (bag.red + bag.green)
  | orange => bag.orange / (bag.orange + bag.purple)
  | purple => bag.purple / (bag.orange + bag.purple)

theorem probability_of_purple :
  let red_prob := draw_prob bagX red
  let purple_given_red := draw_prob bagY purple
  let green_prob := draw_prob bagX green
  let purple_given_green := draw_prob bagZ purple
  red_prob * purple_given_red + green_prob * purple_given_green = 31 / 80 :=
by
  sorry

end probability_of_purple_l114_114576


namespace start_time_of_l_l114_114531

noncomputable def solve_start_time (Tk T_meet distance : ℝ) (Vl factor : ℝ) : ℝ :=
  let Vk := Vl * factor
  let t := T_meet - Tk
  let equation := Vl * (t + (Tk - solve_start_time)) + Vk * t = distance
  solve_start_time

theorem start_time_of_l
  (Tk T_meet distance : ℝ)
  (Vl factor : ℝ)
  (h1 : Tk = 10) 
  (h2 : T_meet = 12)
  (h3 : distance = 300)
  (h4 : Vl = 50)
  (h5 : factor = 1.5) : 
  solve_start_time Tk T_meet distance Vl factor = 9 :=
by
  sorry

end start_time_of_l_l114_114531


namespace greatest_product_l114_114298

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l114_114298


namespace probability_sqrt_less_nine_l114_114506

theorem probability_sqrt_less_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  ∃ (p : ℚ), p = (finset.card (valid_numbers.to_finset) : ℚ) / (finset.card (two_digit_numbers.to_finset) : ℚ) ∧ p = 71 / 90 :=
by
  sorry

end probability_sqrt_less_nine_l114_114506


namespace probability_sqrt_less_nine_l114_114512

theorem probability_sqrt_less_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  ∃ (p : ℚ), p = (finset.card (valid_numbers.to_finset) : ℚ) / (finset.card (two_digit_numbers.to_finset) : ℚ) ∧ p = 71 / 90 :=
by
  sorry

end probability_sqrt_less_nine_l114_114512


namespace probability_sqrt_two_digit_lt_nine_correct_l114_114432

noncomputable def probability_sqrt_two_digit_lt_nine : ℚ :=
  let two_digit_integers := finset.Icc 10 99
  let satisfying_integers := two_digit_integers.filter (λ n => n < 81)
  let probability := (satisfying_integers.card : ℚ) / (two_digit_integers.card : ℚ)
  probability

theorem probability_sqrt_two_digit_lt_nine_correct :
  probability_sqrt_two_digit_lt_nine = 71 / 90 := by
  sorry

end probability_sqrt_two_digit_lt_nine_correct_l114_114432


namespace probability_sqrt_less_than_nine_l114_114447

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sqrt_less_than_nine (n : ℕ) : Prop := n < 81

theorem probability_sqrt_less_than_nine :
  (∃ total_num favorable_num,
    (total_num = Finset.card (Finset.filter is_two_digit (Finset.range 100)) ∧
     favorable_num = Finset.card (Finset.filter (λ n, is_two_digit n ∧ sqrt_less_than_nine n) (Finset.range 100)) ∧
     (favorable_num : ℚ) / total_num = 71 / 90)) :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114447


namespace number_division_l114_114868

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end number_division_l114_114868


namespace greatest_product_sum_300_l114_114243

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l114_114243


namespace quadratic_transformation_l114_114657

theorem quadratic_transformation (y m n : ℝ) 
  (h1 : 2 * y^2 - 2 = 4 * y) 
  (h2 : (y - m)^2 = n) : 
  (m - n)^2023 = -1 := 
  sorry

end quadratic_transformation_l114_114657


namespace sufficient_but_not_necessary_l114_114015

-- Define the complex number z based on a real number m
def z (m : ℝ) : ℂ := (m^2 - m - 2 : ℝ) + (m^2 - 3*m - 2 : ℝ) * complex.I

-- Define what it means for a complex number to be purely imaginary
def isPurelyImaginary (w : ℂ) : Prop := w.re = 0 ∧ w.im ≠ 0

-- The main theorem statement
theorem sufficient_but_not_necessary (m : ℝ) :
  (∀ m, m = -1 → isPurelyImaginary (z m)) ∧
  ¬( ∀ m, isPurelyImaginary (z m) → m = -1 ) :=
by
  sorry

end sufficient_but_not_necessary_l114_114015


namespace find_theta_l114_114620

theorem find_theta :
  ∃ θ : ℝ, θ = 80 ∧ cos (10 * Real.pi / 180) = sin (30 * Real.pi / 180) + sin (θ * Real.pi / 180) := 
by
  use 80
  have h1 : cos (10 * Real.pi / 180) = cos (10 * Real.pi / 180), by sorry
  have h2 : sin (30 * Real.pi / 180) = 1 / 2, by sorry
  have h3 : sin (80 * Real.pi / 180) = cos (10 * Real.pi / 180), by sorry
  rw [h2, h3]
  sorry

end find_theta_l114_114620


namespace greatest_product_two_ints_sum_300_l114_114260

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l114_114260


namespace factor_of_3_in_product_1_to_35_l114_114719

def product_1_to_n (n : ℕ) : ℕ := (List.range (n + 1)).map (λ x, x + 1).prod

theorem factor_of_3_in_product_1_to_35 : ∃ k : ℕ, (3 ^ k ∣ product_1_to_n 35) ∧ k = 15 :=
by
  use 15
  sorry

end factor_of_3_in_product_1_to_35_l114_114719


namespace length_of_AB_l114_114030

noncomputable def length_AB (x y u v : ℝ) (h1 : 5 * x = 3 * y) (h2 : 5 * (x + 3) = 4 * (y - 3)) : ℝ := x + y

theorem length_of_AB (x y u v : ℝ) (h1 : 5 * x = 3 * y) (h2 : 5 * (x + 3) = 4 * (y - 3)) (hx : x = 16.2) (hy : y = 27) : length_AB x y u v h1 h2 = 43.2 :=
by
  rw [length_AB, hx, hy]
  exact add_comm 16.2 27
  sorry

end length_of_AB_l114_114030


namespace range_of_independent_variable_l114_114824

theorem range_of_independent_variable (x : ℝ) : x ≠ -1 ↔ (∃ y : ℝ, y = x / (x + 1)) ↔ True := 
sorry

end range_of_independent_variable_l114_114824


namespace p_more_than_q_and_r_l114_114534

def P : ℝ := 47.99999999999999
def Q : ℝ := 1 / 6 * P
def R : ℝ := 1 / 6 * P

theorem p_more_than_q_and_r : P - (Q + R) = 32 := by
  calc
    P - (Q + R)
        = P - (1 / 6 * P + 1 / 6 * P) : by congr; exact (by reflexivity : Q = 1 / 6 * P); exact (by reflexivity : R = 1 / 6 * P)
    ... = P - (1 / 3 * P)             : by rw [←add_mul, one_div, one_div, nat.cast_add, mul_one]
    ... = 2 / 3 * P                   : by ring
    ... = 32                          : by norm_num

end p_more_than_q_and_r_l114_114534


namespace PA_squared_plus_PB_squared_plus_PC_squared_PA_PB_PC_area_l114_114913

noncomputable theory

def is_equilateral_triangle (A B C : ℝ × ℝ) (l : ℝ) : Prop :=
  dist A B = l ∧ dist B C = l ∧ dist C A = l

def is_on_incircle (P A B C : ℝ × ℝ) : Prop :=
  let incenter := (A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3 in
  let inradius := dist (incenter) (A) / (2 * sqrt 3) in
  dist P incenter = inradius

/-- Step 1: Show PA^2 + PB^2 + PC^2 = 5 -/
theorem PA_squared_plus_PB_squared_plus_PC_squared (A B C P : ℝ × ℝ) (l : ℝ) (h_eq : is_equilateral_triangle A B C l) (h_on_incircle : is_on_incircle P A B C) :
  (dist P A) ^ 2 + (dist P B) ^ 2 + (dist P C) ^ 2 = 5 :=
sorry

/-- Step 2: The triangle with sides PA, PB, and PC has an area of sqrt(3)/4 -/
theorem PA_PB_PC_area (A B C P : ℝ × ℝ) (l : ℝ) (h_eq : is_equilateral_triangle A B C l) (h_on_incircle : is_on_incircle P A B C) :
  let PA := dist P A in
  let PB := dist P B in
  let PC := dist P C in
  let s := (PA + PB + PC) / 2 in
  sqrt (s * (s - PA) * (s - PB) * (s - PC)) = sqrt 3 / 4 :=
sorry

end PA_squared_plus_PB_squared_plus_PC_squared_PA_PB_PC_area_l114_114913


namespace sum_of_valid_n_l114_114935

theorem sum_of_valid_n (n : ℕ) : 
  (n > 0 → n^n > 0 → nat.factors_count (n^n) = 325) → 
  list.sum (list.filter (λ n, nat.factors_count (n^n) = 325) (list.range 100)) = 93 := 
sorry

end sum_of_valid_n_l114_114935


namespace negation_equivalence_l114_114072

theorem negation_equivalence (x : ℝ) : ¬(∀ x, x^2 - x + 2 ≥ 0) ↔ ∃ x, x^2 - x + 2 < 0 :=
sorry

end negation_equivalence_l114_114072


namespace total_pieces_of_junk_mail_l114_114087

def houses : ℕ := 6
def pieces_per_house : ℕ := 4

theorem total_pieces_of_junk_mail : houses * pieces_per_house = 24 :=
by 
  sorry

end total_pieces_of_junk_mail_l114_114087


namespace probability_sqrt_less_than_nine_is_correct_l114_114381

def probability_sqrt_less_than_nine : ℚ :=
  let total_two_digit_numbers := 99 - 10 + 1 in
  let satisfying_numbers := 80 - 10 + 1 in
  let probability := (satisfying_numbers : ℚ) / (total_two_digit_numbers : ℚ) in
  probability

theorem probability_sqrt_less_than_nine_is_correct :
  probability_sqrt_less_than_nine = 71 / 90 :=
by
  -- proof here
  sorry

end probability_sqrt_less_than_nine_is_correct_l114_114381


namespace max_product_300_l114_114142

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l114_114142


namespace greatest_product_obtainable_l114_114319

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l114_114319


namespace percentage_increase_in_water_intake_l114_114703

theorem percentage_increase_in_water_intake :
  let current_intake := 15
  let recommended_intake := 21
  let increase := recommended_intake - current_intake
  let percentage_increase := (increase / current_intake.toFloat) * 100
  percentage_increase = 40 :=
by
  let current_intake := 15
  let recommended_intake := 21
  let increase := recommended_intake - current_intake
  let percentage_increase := (increase / current_intake.toFloat) * 100
  show percentage_increase = 40
  sorry

end percentage_increase_in_water_intake_l114_114703


namespace comparison_l114_114946

noncomputable def f (x : ℝ) : ℝ := sorry

theorem comparison (hf_odd : ∀ x : ℝ, f (-x) = -f x)
  (hf_condition : ∀ x : ℝ, x < 0 → f x + x * (derivative f) x < 0) :
  let a := 3 * f 3,
      b := f 1,
      c := -2 * f (-2)
  in a > c ∧ c > b :=
by
  let a := 3 * f 3
  let b := f 1
  let c := -2 * f (-2)
  sorry

end comparison_l114_114946


namespace line_CD_fixed_direction_l114_114675

variables {A B C D : affine_coordinate} {a b c e f g : ℝ}

-- Define the conic section equation
def conic_section := λ (x y : ℝ), a*x^2 + b*x*y + c*y^2 + e*x + f*y + g = 0

-- Assume A and B are fixed points on the conic section.
axiom A_on_conic : conic_section A.1 A.2 = 0
axiom B_on_conic : conic_section B.1 B.2 = 0
axiom a_nonzero : a ≠ 0

-- Assume any circle passing through A and B intersects the conic section at two additional points C and D.
axiom circle_passing_AB_intersects_CD : ∃ (h k m : ℝ),
  (∀ (x y : ℝ), x^2 + y^2 + h*x + k*y + m = 0) ∧
  conic_section C.1 C.2 = 0 ∧ conic_section D.1 D.2 = 0

-- To prove: the line CD has a fixed direction.
theorem line_CD_fixed_direction :
  ∃ (θ : ℝ), ∀ {x y : ℝ}, x = C.1 - D.1 ∧ y = C.2 - D.2 → (y ≠ 0 → x = θ * y) :=
sorry

end line_CD_fixed_direction_l114_114675


namespace probability_sqrt_lt_9_l114_114389

theorem probability_sqrt_lt_9 : 
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
in probability = 71 / 90 :=
by
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  sorry

end probability_sqrt_lt_9_l114_114389


namespace number_of_valid_orders_l114_114790

-- Define the set of flavors
inductive Flavor
| vanilla
| chocolate
| strawberry
| cherry
| mint

open Flavor

-- Define the stack with vanilla fixed at the bottom
def valid_arrangements : List (List Flavor) :=
  List.permutations [chocolate, strawberry, cherry, mint].map (λ l, vanilla :: l)

-- State the theorem
theorem number_of_valid_orders : valid_arrangements.length = 24 :=
by
  sorry

end number_of_valid_orders_l114_114790


namespace number_of_possible_values_of_m_l114_114815

open Real

theorem number_of_possible_values_of_m :
  let a := log 15 / log 10
      b := log 125 / log 10
      m := log m / log 10
  in (∃ (m : ℕ), 9 ≤ m ∧ m ≤ 1875) → (∃ (n : ℕ), n = 1867) :=
by
  let a := log 15 / log 10
  let b := log 125 / log 10
  let m_ := log (m : ℝ) / log 10
  have condition1 := a + m_ > b
  have condition2 := b + m_ > a
  have condition3 := m_ + a > b
  intro hex
  obtain ⟨m, h1, h2⟩ := hex
  -- Sorry for further proof details
  exact sorry

end number_of_possible_values_of_m_l114_114815


namespace spread_news_l114_114565

theorem spread_news (n m days : ℕ) (residents : Fin n → Type) (knows : (Fin n → Type) → (Fin n → Type) → Prop) :
  n = 1000 → m = 90 → days = 10 →
  (∀ r1 r2 : Fin n → Type, ∃ chain : List (Fin n → Type), chain.head = r1 ∧ chain.last = r2 ∧ ∀ (i : Fin chain.length), knows (chain.nth_le i sorry) (chain.nth_le (i+1) sorry)) →
  ∃ (s : Finset (Fin n → Type)), s.card ≤ m ∧ (∀ (r : Fin n → Type), ∃ (day : ℕ), day ≤ days ∧ r ∈ Finset.image (λ x, knows x).symm s) :=
begin
  sorry
end

end spread_news_l114_114565


namespace max_product_of_sum_300_l114_114283

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l114_114283


namespace rectangle_difference_l114_114063

theorem rectangle_difference (L B : ℝ) (h1 : 2 * (L + B) = 266) (h2 : L * B = 4290) :
  L - B = 23 :=
sorry

end rectangle_difference_l114_114063


namespace polynomial_expansion_problem_l114_114058

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def coefficient_x3 (f : Polynomial ℚ) : ℚ :=
  f.coeff 3

theorem polynomial_expansion_problem :
  let f := (1 - 3 * Polynomial.X) ^ 5 * (3 - Polynomial.X) in
  coefficient_x3 f = -900 :=
by
  -- The proof will involve calculating the coefficient
  sorry

end polynomial_expansion_problem_l114_114058


namespace greatest_product_l114_114296

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l114_114296


namespace shortest_trip_on_cube_surface_l114_114917

-- Define the conditions of the problem
def cube_edge_length : ℝ := 2
def is_midpoint (a b : ℝ × ℝ × ℝ) (m : ℝ × ℝ × ℝ) : Prop := 
  m = ((a.1 + b.1) / 2, (a.2 + b.2) / 2, (a.3 + b.3) / 2)
def opposite_edge_midpoints (a b d c : ℝ × ℝ × ℝ) : Prop :=
  ¬∃ (e : ℝ × ℝ × ℝ), (e = a ∧ (e = d ∨ e = c)) ∨ (e = b ∧ (e = d ∨ e = c))

-- Define the theorem to prove
theorem shortest_trip_on_cube_surface :
  ∀ (a b d c m n : ℝ × ℝ × ℝ),
  is_midpoint a b m ∧ is_midpoint d c n ∧ opposite_edge_midpoints a b d c →
  dist m n = 2 :=
by
  sorry

end shortest_trip_on_cube_surface_l114_114917


namespace crisp_stops_on_dime_at_p0_l114_114944

namespace Basketball

def dropDimeOn (n : ℕ) : Prop := n % 10 = 0
def dropNickelOn (n : ℕ) : Prop := n % 5 = 0 ∧ n % 10 ≠ 0

inductive Pos : Type
| zero : Pos
| pos : ℕ → Pos

structure State where
  pos : Pos
  prob : ℚ

def jump (s : State) : State :=
  { pos := match s.pos with
    | Pos.zero => Pos.pos 3
    | Pos.pos n => if (n + 3) % 10 = 0 ∨ (n + 3) % 5 = 0 then Pos.pos (n + 3)
                   else Pos.pos (n + 7), 
    prob := if (match s.pos with | Pos.zero => 0 | Pos.pos n => (n + 3) % 10 = 0 ∨ (n + 3) % 5 = 0)
            then 2 / 3
            else 1 / 3
  }

def p_0 : ℚ := sorry -- replace with the full equation solving steps if implementing the solution

theorem crisp_stops_on_dime_at_p0 : p_0 = 20 / 31 :=
by sorry

end Basketball

end crisp_stops_on_dime_at_p0_l114_114944


namespace greatest_product_sum_300_l114_114245

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l114_114245


namespace equation_of_line_l114_114682

theorem equation_of_line (x y : ℝ) (P : ℝ × ℝ) (C : ℝ × ℝ) (l : ℝ → ℝ) :
  P = (1, 1/2) ∧ ((x^2 + (y-1)^2) = 4) ∧ C = (0, 1) ∧ (l(1) = 1/2) ∧ 
  (∃ A B : ℝ × ℝ, (A ≠ B) ∧ (x^2 + (y-1)^2 = 4 → l A.fst = A.snd ∧ l B.fst = B.snd ∧
  ∠ A C B = min ∠ A C B)) →
  l = λ x, 2 * x - 1/2 := sorry

end equation_of_line_l114_114682


namespace max_product_of_two_integers_whose_sum_is_300_l114_114179

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l114_114179


namespace probability_sqrt_lt_9_l114_114400

theorem probability_sqrt_lt_9 : 
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
in probability = 71 / 90 :=
by
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  sorry

end probability_sqrt_lt_9_l114_114400


namespace tangents_concurrent_l114_114955

theorem tangents_concurrent
  (tetrahedron : Type)
  (sphere : Type)
  [tetrahedron_has_vertices : tetrahedron]
  [sphere_has_points_of_tangency : sphere]
  (edge_tangent : ∀ edge, edge ∈ tetrahedron → tangent_to_sphere edge sphere) :
  concurrent (line_segment (points_of_tangency edge_pair1))
             (line_segment (points_of_tangency edge_pair2))
             (line_segment (points_of_tangency edge_pair3)) :=
by
  sorry

end tangents_concurrent_l114_114955


namespace arithmetic_seq_and_general_formula_find_Tn_l114_114643

-- Given definitions
def S : ℕ → ℕ := sorry
def a : ℕ → ℕ := sorry

-- Conditions
axiom a1 : a 1 = 1
axiom a2 : ∀ n : ℕ, n > 0 → n * S n.succ = (n+1) * S n + n^2 + n

-- Problem 1: Prove and derive general formula for Sₙ
theorem arithmetic_seq_and_general_formula (n : ℕ) (h : n > 0) :
  ∃ S : ℕ → ℕ, (∀ n : ℕ, n > 0 → (S (n+1)) / (n+1) - (S n) / n = 1) ∧ (S n = n^2) := sorry

-- Problem 2: Given bₙ and Tₙ, find Tₙ
def b (n : ℕ) : ℕ := 1 / (a n * a (n+1))
def T : ℕ → ℕ := sorry

axiom b1 : ∀ n : ℕ, n > 0 → b 1 = 1
axiom b2 : ∀ n : ℕ, n > 0 → T n = 1 / (2 * n + 1)

theorem find_Tn (n : ℕ) (h : n > 0) : T n = n / (2 * n + 1) := sorry

end arithmetic_seq_and_general_formula_find_Tn_l114_114643


namespace fraction_of_EF_over_GH_l114_114783

theorem fraction_of_EF_over_GH (E F G H : Point) (x y : ℝ) 
    (h1 : G ≠ H)
    (h2 : E ≠ F)
    (h3 : F ≠ H)
    (h4 : (dist G E) = 5 * (dist E H))
    (h5 : (dist G F) = 10 * (dist F H)) :
    (dist E F) = (5 / 11) * (dist G H) := 
by
  sorry

end fraction_of_EF_over_GH_l114_114783


namespace greatest_product_of_two_integers_with_sum_300_l114_114239

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l114_114239


namespace probability_sqrt_lt_nine_two_digit_l114_114499

theorem probability_sqrt_lt_nine_two_digit :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99} in
  let T := {n : ℕ | 10 ≤ n ∧ n < 81} in
  (Fintype.card T : ℚ) / (Fintype.card S) = 71 / 90 :=
by
  sorry

end probability_sqrt_lt_nine_two_digit_l114_114499


namespace given_sequence_find_a_and_b_l114_114668

-- Define the general pattern of the sequence
def sequence_pattern (n a b : ℕ) : Prop :=
  n + (b / a : ℚ) = (n^2 : ℚ) * (b / a : ℚ)

-- State the specific case for n = 9
def sequence_case_for_9 (a b : ℕ) : Prop :=
  sequence_pattern 9 a b ∧ a + b = 89

-- Now, structure this as a theorem to be proven in Lean
theorem given_sequence_find_a_and_b :
  ∃ (a b : ℕ), sequence_case_for_9 a b :=
sorry

end given_sequence_find_a_and_b_l114_114668


namespace maximum_label_sum_l114_114829

noncomputable def chessboard_label (i j : ℕ) : ℝ :=
  j / (i + j)

noncomputable def maximum_sum_labels : ℝ :=
  (8 / 9) + 
  (7 / 9) + 
  (6 / 8) + 
  (5 / 7) + 
  (4 / 6) + 
  (3 / 5) + 
  (2 / 4) + 
  (1 / 9)

theorem maximum_label_sum : maximum_sum_labels ≈ 3.6389 :=
by
  sorry

end maximum_label_sum_l114_114829


namespace parabola_tangent_hyperbola_l114_114556

theorem parabola_tangent_hyperbola (m : ℝ) :
  (∀ x : ℝ, (x^2 + 5)^2 - m * x^2 = 4 → y = x^2 + 5)
  ∧ (∀ y : ℝ, y ≥ 5 → y^2 - m * x^2 = 4) →
  (m = 10 + 2 * Real.sqrt 21 ∨ m = 10 - 2 * Real.sqrt 21) :=
  sorry

end parabola_tangent_hyperbola_l114_114556


namespace problem1_problem2_problem3_l114_114581

-- Problem 1
theorem problem1 : -2.8 + (-3.6) + 3 - (-3.6) = 0.2 := 
by
  sorry

-- Problem 2
theorem problem2 : (-4) ^ 2010 * (-0.25) ^ 2009 + (-12) * (1 / 3 - 3 / 4 + 5 / 6) = -9 := 
by
  sorry

-- Problem 3
theorem problem3 : 13 * (16/60 : ℝ) * 5 - 19 * (12/60 : ℝ) / 6 = 13 * (8/60 : ℝ) + 50 := 
by
  sorry

end problem1_problem2_problem3_l114_114581


namespace max_product_two_integers_l114_114180

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l114_114180


namespace probability_sqrt_less_than_nine_l114_114453

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sqrt_less_than_nine (n : ℕ) : Prop := n < 81

theorem probability_sqrt_less_than_nine :
  (∃ total_num favorable_num,
    (total_num = Finset.card (Finset.filter is_two_digit (Finset.range 100)) ∧
     favorable_num = Finset.card (Finset.filter (λ n, is_two_digit n ∧ sqrt_less_than_nine n) (Finset.range 100)) ∧
     (favorable_num : ℚ) / total_num = 71 / 90)) :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114453


namespace total_pumped_volume_l114_114053

def powerJetA_rate : ℕ := 360
def powerJetB_rate : ℕ := 540
def powerJetA_time : ℕ := 30
def powerJetB_time : ℕ := 45

def pump_volume (rate : ℕ) (minutes : ℕ) : ℕ :=
  rate * (minutes / 60)

theorem total_pumped_volume : 
  pump_volume powerJetA_rate powerJetA_time + pump_volume powerJetB_rate powerJetB_time = 585 := 
by
  sorry

end total_pumped_volume_l114_114053


namespace correct_operation_among_given_ones_l114_114870

theorem correct_operation_among_given_ones
  (a : ℝ) :
  (a^2)^3 = a^6 :=
by {
  sorry
}

-- Auxiliary lemmas if needed (based on conditions):
lemma mul_powers_add_exponents (a : ℝ) (m n : ℕ) : a^m * a^n = a^(m + n) := by sorry

lemma power_of_a_power (a : ℝ) (m n : ℕ) : (a^m)^n = a^(m * n) := by sorry

lemma div_powers_subtract_exponents (a : ℝ) (m n : ℕ) : a^m / a^n = a^(m - n) := by sorry

lemma square_of_product (x y : ℝ) : (x * y)^2 = x^2 * y^2 := by sorry

end correct_operation_among_given_ones_l114_114870


namespace smallest_positive_period_f_f_geq_neg_one_half_l114_114690

noncomputable def f (x : ℝ) : ℝ :=
  2 * (sin x) ^ 2 - cos (2 * x + π / 3)

-- 1. The smallest positive period of f(x) is π.
theorem smallest_positive_period_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧ T = π := 
sorry

-- 2. For x ∈ [0, π / 2], f(x) ≥ -1 / 2.
theorem f_geq_neg_one_half (x : ℝ) (h : x ∈ set.Icc 0 (π / 2)) : f x ≥ -1 / 2 := 
sorry

end smallest_positive_period_f_f_geq_neg_one_half_l114_114690


namespace probability_sqrt_lt_9_of_two_digit_l114_114367

-- Define the set of two-digit whole numbers
def two_digit_whole_numbers : set ℕ := {n | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate that checks if the square root of a number is less than 9
def sqrt_lt_9 (n : ℕ) : Prop := (n : ℝ)^2 < (9 : ℝ)^2

-- Calculate the probability
theorem probability_sqrt_lt_9_of_two_digit :
  let eligible_numbers := { n ∈ two_digit_whole_numbers | sqrt_lt_9 n } in
  (eligible_numbers.to_finset.card : ℚ) / (two_digit_whole_numbers.to_finset.card : ℚ) =
  71 / 90 :=
by
  sorry

end probability_sqrt_lt_9_of_two_digit_l114_114367


namespace probability_sqrt_lt_nine_two_digit_l114_114489

theorem probability_sqrt_lt_nine_two_digit :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99} in
  let T := {n : ℕ | 10 ≤ n ∧ n < 81} in
  (Fintype.card T : ℚ) / (Fintype.card S) = 71 / 90 :=
by
  sorry

end probability_sqrt_lt_nine_two_digit_l114_114489


namespace integers_between_400_and_600_with_digit_sum_15_l114_114705

theorem integers_between_400_and_600_with_digit_sum_15 :
  (λ (count : ℕ), count = 17) (Finset.card (Finset.filter (λ n, (∃ a b c,
    n = 100 * a + 10 * b + c ∧ 400 ≤ n ∧ n ≤ 600 ∧ a + b + c = 15))
    (Finset.range' 400 201))) :=
by { sorry }

end integers_between_400_and_600_with_digit_sum_15_l114_114705


namespace greatest_product_two_ints_sum_300_l114_114256

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l114_114256


namespace greatest_product_obtainable_l114_114318

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l114_114318


namespace max_product_of_sum_300_l114_114219

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l114_114219


namespace general_formula_sum_first_n_terms_l114_114672

variable {a : ℕ → ℝ} (h_geom_seq : ∀ n m, a (n + m) = a n * a m)
variable (a_pos : ∀ n, 0 < a n)
variable (a1 : a 1 = 2)
variable (a2a4 : a 2 * a 4 = 64)

-- Define the first general formula for the sequence
theorem general_formula (n : ℕ) : a n = 2 ^ n := by
  sorry

-- Define b and logic for the second proof
noncomputable def b (n : ℕ) : ℝ := log 2 (a n)

theorem sum_first_n_terms (n : ℕ) :
    (∑ k in Finset.range n, 1 / (b k * b (k + 1))) = n / (n + 1) := by
  sorry

end general_formula_sum_first_n_terms_l114_114672


namespace angle_AMD_60_degrees_l114_114036

-- Define necessary geometric shapes and points for the conditions
structure Point :=
(x : ℝ)
(y : ℝ)

structure Rectangle :=
(A B C D : Point)
(h_proofs : (A.x = 0 ∧ A.y = 0) ∧ 
            (B.x = 8 ∧ B.y = 0) ∧ 
            (C.x = 8 ∧ C.y = 4) ∧ 
            (D.x = 0 ∧ D.y = 4))

-- Indicate that M is a point on AB such that ∆AMD is equilateral
structure EquilateralTriangle (A M D : Point) :=
(equilateral : (dist A M = dist M D) ∧ (dist M D = dist D A) ∧ (dist D A = dist A M))

-- Define the function to calculate distance between points
noncomputable def dist (P Q : Point) : ℝ :=
  real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2)

theorem angle_AMD_60_degrees (A B C D M : Point) (r : Rectangle A B C D)
  (h : EquilateralTriangle A M D) : 
  ∠AMD = 60 :=
sorry

end angle_AMD_60_degrees_l114_114036


namespace find_k_check_divisibility_l114_114949

-- Define the polynomial f(x) as 2x^3 - 8x^2 + kx - 10
def f (x k : ℝ) : ℝ := 2 * x^3 - 8 * x^2 + k * x - 10

-- Define the polynomial g(x) as 2x^3 - 8x^2 + 13x - 10 after finding k = 13
def g (x : ℝ) : ℝ := 2 * x^3 - 8 * x^2 + 13 * x - 10

-- The first proof problem: Finding k
theorem find_k : (f 2 k = 0) → k = 13 := 
sorry

-- The second proof problem: Checking divisibility by 2x^2 - 1
theorem check_divisibility : ¬ (∃ h : ℝ → ℝ, g x = (2 * x^2 - 1) * h x) := 
sorry

end find_k_check_divisibility_l114_114949


namespace Riverstone_Students_Walk_Home_l114_114836

variable (total_students buses carpools public_transit bikes walks : ℕ)

theorem Riverstone_Students_Walk_Home :
  total_students = 1500 →
  buses = total_students * 3 / 10 →
  carpools = total_students * 1 / 5 →
  public_transit = 80 →
  bikes = total_students * 20 / 100 →
  walks = total_students - (buses + carpools + public_transit + bikes) →
  walks = 370 :=
begin
  sorry
end

end Riverstone_Students_Walk_Home_l114_114836


namespace problem_solution_l114_114875

variable (a b c d : ℕ)

-- Conditions:
-- 1. Minuend and subtrahend extraction.
-- 2. Result in hundreds column is 7
-- 3. No borrowing from ten thousand's place.
def valid_subtraction_digits (a b c d X : ℕ) : Prop :=
  b = 2 ∧  
  -- Using the given conditions, enforce the formal subtractions:
  let minuend := a * 1000 + b * 100 + c * 10 + d in
  let subtrahend := d * 1000 + b * 100 + a * 10 + c in
  (minuend - subtrahend) % 10000 = X * 1000 + 7 * 100 + (minuend - subtrahend) % 100

theorem problem_solution (a b c d : ℕ) :
  valid_subtraction_digits a b c d 9 :=
  sorry

end problem_solution_l114_114875


namespace greatest_product_sum_300_l114_114250

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l114_114250


namespace greatest_product_sum_300_l114_114207

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l114_114207


namespace probability_sqrt_lt_nine_l114_114414

theorem probability_sqrt_lt_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  let probability := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  probability = 71 / 90 :=
by
  let two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n : ℕ | 10 ≤ n ∧ n < 81}
  have h1 : two_digit_numbers.card = 90 := sorry
  have h2 : valid_numbers.card = 71 := sorry
  let probability : ℚ := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  rw [h1, h2]
  simp
  norm_num

end probability_sqrt_lt_nine_l114_114414


namespace prob_sqrt_less_than_nine_l114_114478

/-- The probability that the square root of a randomly selected 
two-digit whole number is less than nine is 71/90. -/
theorem prob_sqrt_less_than_nine : (let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99};
                                     let A := {n : ℕ | 10 ≤ n ∧ n < 81};
                                     (A.card / S.card : ℚ) = 71 / 90) :=
by
  sorry

end prob_sqrt_less_than_nine_l114_114478


namespace find_y_coordinate_l114_114755

noncomputable theory
open_locale classical

structure Point :=
(x : ℝ)
(y : ℝ)

def distance (P Q : Point) : ℝ :=
real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

def equation_of_ellipse (P Q : Point) (f : ℝ) (x y : ℝ) :=
  (x - P.x)^2 / (f/2)^2 + (y - Q.y)^2 / ((real.sqrt (f^2 - (P.x - Q.x)^2))/2)^2 = 1

def A := Point.mk (-4) 0
def B := Point.mk (-3) 2
def C := Point.mk 3 2
def D := Point.mk 4 0

def P := Point.mk 0 (18 + 6 * real.sqrt 2) / 7

theorem find_y_coordinate :
  (equation_of_ellipse A D 10 P.x P.y) ∧ (equation_of_ellipse B C 10 P.x P.y) → 
  (P.y = (18 + 6 * real.sqrt 2) / 7) ∧ (18 + 6 + 2 + 7 = 33) :=
begin
  sorry
end

end find_y_coordinate_l114_114755


namespace incorrect_statement_sin_l114_114651

theorem incorrect_statement_sin (x : ℝ) (hx1 : 0 < x) (hx2 : x < 1/2) : ¬(sin (x + 1) > sin x) :=
sorry

end incorrect_statement_sin_l114_114651


namespace other_root_of_quadratic_l114_114724

theorem other_root_of_quadratic (a b : ℝ) (h : a ≠ 0) (h_eq : a * 2^2 = b) : 
  ∃ m : ℝ, a * m^2 = b ∧ 2 + m = 0 :=
begin
  use -2,
  split,
  { rw [mul_pow, h_eq, pow_two, mul_assoc, mul_comm 2, ←mul_assoc, mul_comm a, pow_two (-2)],
    sorry },
  { linarith }
end

end other_root_of_quadratic_l114_114724


namespace coach_path_length_ge_100_l114_114881

-- Define the athletes' speeds and the distance from A to B
variables {v1 v2 v3 : ℝ} (h_v1_gt_v2 : v1 > v2) (h_v2_gt_v3 : v2 > v3)
variable (d : ℝ) (h_d : d = 60)

-- Define the constant speeds on the return journey
variables {u1 u2 u3 : ℝ}
variable (h_finish_simultaneously : d / v1 + d / u1 = d / v2 + d / u2 = d / v3 + d / u3)

-- Define the coach's path length condition
def coach_path_length (l : ℝ) : Prop :=
  ∀ t : ℝ, l ≥ ∑ i in {v1, v2, v3}, ∥d - d∥ t

-- Prove that the coach cannot run less than 100 meters
theorem coach_path_length_ge_100 (h : coach_path_length 100) : false :=
sorry

end coach_path_length_ge_100_l114_114881


namespace probability_factor_lt_8_l114_114331

theorem probability_factor_lt_8 (n : ℕ) (h_n_eq : n = 90) :
  (number_factors_lt_8 : ℚ) = (5 / 12) :=
begin
  sorry
end

end probability_factor_lt_8_l114_114331


namespace greatest_product_sum_300_l114_114129

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l114_114129


namespace probability_of_sqrt_lt_9_l114_114357

-- Define the set of two-digit whole numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the subset of numbers for which the square root is less than 9
def valid_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 80}

-- Define the probability calculation
noncomputable def probability_sqrt_lt_9 := (valid_numbers.to_finset.card : ℝ) / (two_digit_numbers.to_finset.card : ℝ)

-- The statement we aim to prove
theorem probability_of_sqrt_lt_9 : probability_sqrt_lt_9 = 71 / 90 := 
sorry

end probability_of_sqrt_lt_9_l114_114357


namespace area_of_smaller_circle_l114_114841

-- Defining the problem conditions
def externally_tangent {C1 C2 : Type*} [metric_space C1] [metric_space C2]
  (o1 o2 : C1) (r1 : ℝ) (R2 : ℝ) : Prop :=
metric.ball o1 r1 ∩ metric.ball o2 R2 = ∅ ∧ 
metric.dist o1 o2 = r1 + R2

def tangent_line (P A B o : Type*) [metric_space P] [metric_space A] [metric_space B] [metric_space o] (PA : ℝ) (AB : ℝ): Prop :=
PA = 4 ∧ AB = 4

-- The main theorem statement
theorem area_of_smaller_circle {C1 C2 P A B : Type*} [metric_space C1] [metric_space C2] [metric_space P] [metric_space A] [metric_space B]
  (o1 o2 : C1) (r1 : ℝ) (R2 : ℝ) (PA AB : ℝ) (P_ : P) (A_ : A) (B_ : B) (o_1 : C1) (o_2 : C2)
  (tangent_condition : externally_tangent o1 o2 r1 R2)
  (line_condition : tangent_line P_ A_ B_ o_1 PA AB):
  π * r1^2 = 2 * π :=
sorry

end area_of_smaller_circle_l114_114841


namespace opposite_signs_and_positive_greater_abs_l114_114715

theorem opposite_signs_and_positive_greater_abs {a b : ℝ} (h1 : a * b < 0) (h2 : a + b > 0) :
  (a < 0 ∧ 0 < b ∧ ∥b∥ > ∥a∥) ∨ (b < 0 ∧ 0 < a ∧ ∥a∥ > ∥b∥) :=
by
  sorry

end opposite_signs_and_positive_greater_abs_l114_114715


namespace sum_of_factors_is_17_l114_114066

theorem sum_of_factors_is_17 :
  ∃ (a b c d e f g : ℤ), 
  (16 * x^4 - 81 * y^4) =
    (a * x + b * y) * 
    (c * x^2 + d * x * y + e * y^2) * 
    (f * x + g * y) ∧ 
    a + b + c + d + e + f + g = 17 :=
by
  sorry

end sum_of_factors_is_17_l114_114066


namespace sum_of_abs_coeffs_in_binomial_expansion_l114_114667

theorem sum_of_abs_coeffs_in_binomial_expansion :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ), 
  (3 * x - 1) ^ 7 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4 + a₅ * x ^ 5 + a₆ * x ^ 6 + a₇ * x ^ 7
  → |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 4 ^ 7 :=
by
  sorry

end sum_of_abs_coeffs_in_binomial_expansion_l114_114667


namespace probability_sqrt_lt_9_of_two_digit_l114_114372

-- Define the set of two-digit whole numbers
def two_digit_whole_numbers : set ℕ := {n | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate that checks if the square root of a number is less than 9
def sqrt_lt_9 (n : ℕ) : Prop := (n : ℝ)^2 < (9 : ℝ)^2

-- Calculate the probability
theorem probability_sqrt_lt_9_of_two_digit :
  let eligible_numbers := { n ∈ two_digit_whole_numbers | sqrt_lt_9 n } in
  (eligible_numbers.to_finset.card : ℚ) / (two_digit_whole_numbers.to_finset.card : ℚ) =
  71 / 90 :=
by
  sorry

end probability_sqrt_lt_9_of_two_digit_l114_114372


namespace trigonometric_identity_l114_114980

theorem trigonometric_identity (α : ℝ) (h1 : Real.sin α * Real.cos α = 1 / 8) (h2 : π < α ∧ α < 5 * π / 4) :
  Real.cos α - Real.sin α = -√3 / 2 :=
sorry

end trigonometric_identity_l114_114980


namespace Q_eq_sum_of_binom_l114_114974

open Nat

def Q (n k : ℕ) : ℕ :=
(coef k (expand (x + x^2 + x^3 + 1) ^ n))

theorem Q_eq_sum_of_binom 
  (n k : ℕ) :
  Q n k = ∑ j in range (n + 1), binom n j * binom n (k - 2 * j) :=
by
  sorry

end Q_eq_sum_of_binom_l114_114974


namespace max_product_two_integers_l114_114184

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l114_114184


namespace max_product_of_sum_300_l114_114222

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l114_114222


namespace total_paintable_wall_area_l114_114751

/-- 
  Conditions:
  - John's house has 4 bedrooms.
  - Each bedroom is 15 feet long, 12 feet wide, and 10 feet high.
  - Doorways, windows, and a fireplace occupy 85 square feet per bedroom.
  Question: Prove that the total paintable wall area is 1820 square feet.
--/
theorem total_paintable_wall_area 
  (num_bedrooms : ℕ)
  (length width height non_paintable_area : ℕ)
  (h_num_bedrooms : num_bedrooms = 4)
  (h_length : length = 15)
  (h_width : width = 12)
  (h_height : height = 10)
  (h_non_paintable_area : non_paintable_area = 85) :
  (num_bedrooms * ((2 * (length * height) + 2 * (width * height)) - non_paintable_area) = 1820) :=
by
  sorry

end total_paintable_wall_area_l114_114751


namespace largest_4_digit_integer_congruent_to_25_mod_26_l114_114330

theorem largest_4_digit_integer_congruent_to_25_mod_26 : ∃ x : ℕ, x < 10000 ∧ x ≥ 1000 ∧ x % 26 = 25 ∧ ∀ y : ℕ, y < 10000 ∧ y ≥ 1000 ∧ y % 26 = 25 → y ≤ x := by
  sorry

end largest_4_digit_integer_congruent_to_25_mod_26_l114_114330


namespace probability_sqrt_lt_nine_l114_114409

theorem probability_sqrt_lt_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  let probability := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  probability = 71 / 90 :=
by
  let two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n : ℕ | 10 ≤ n ∧ n < 81}
  have h1 : two_digit_numbers.card = 90 := sorry
  have h2 : valid_numbers.card = 71 := sorry
  let probability : ℚ := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  rw [h1, h2]
  simp
  norm_num

end probability_sqrt_lt_nine_l114_114409


namespace polygon_perimeter_l114_114905

theorem polygon_perimeter (side_length : ℝ) (ext_angle_deg : ℝ) (n : ℕ) (h1 : side_length = 8) 
  (h2 : ext_angle_deg = 90) (h3 : ext_angle_deg = 360 / n) : 
  4 * side_length = 32 := 
  by 
    sorry

end polygon_perimeter_l114_114905


namespace max_product_two_integers_sum_300_l114_114158

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l114_114158


namespace percentage_defective_meters_l114_114918

theorem percentage_defective_meters (total_meters : ℕ) (defective_meters : ℕ) (percentage : ℚ) :
  total_meters = 2500 →
  defective_meters = 2 →
  percentage = (defective_meters / total_meters) * 100 →
  percentage = 0.08 := 
sorry

end percentage_defective_meters_l114_114918


namespace fraction_female_participants_l114_114091

noncomputable def fraction_females (last_year_males : ℕ) (female_increase : ℝ) 
  (male_increase : ℝ) (total_increase : ℝ) : ℚ :=
  let this_year_males := last_year_males * (1 + male_increase) in
  let y := (total_increase * (last_year_males + (last_year_males * (1 - male_increase)))) / (male_increase + female_increase) in
  let this_year_females := y * female_increase in
  this_year_females / (this_year_females + this_year_males)

theorem fraction_female_participants :
  fraction_females 30 0.15 0.10 0.08 = 10 / 43 :=
sorry

end fraction_female_participants_l114_114091


namespace max_product_of_two_integers_whose_sum_is_300_l114_114175

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l114_114175


namespace greatest_product_sum_300_l114_114206

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l114_114206


namespace magnitude_v_l114_114045

open Complex

theorem magnitude_v (u v : ℂ) (h1 : u * v = 20 - 15 * Complex.I) (h2 : Complex.abs u = 5) :
  Complex.abs v = 5 := by
  sorry

end magnitude_v_l114_114045


namespace percentage_x_y_l114_114716

variable (x y P : ℝ)

theorem percentage_x_y 
  (h1 : 0.5 * (x - y) = (P / 100) * (x + y))
  (h2 : y = (1 / 9) * x) : 
  P = 40 :=
sorry

end percentage_x_y_l114_114716


namespace sum_of_combinations_l114_114582

theorem sum_of_combinations:
  (∀ n, (Nat.choose (n + 1) 3) - (Nat.choose n 3) = Nat.choose n 2) ∧
  (∀ n r, Nat.choose n r = n! / (r! * (n - r)!)) →
  (Nat.choose 2 2 + Nat.choose 3 2 + Nat.choose 4 2 + Nat.choose 5 2 + 
   Nat.choose 6 2 + Nat.choose 7 2 + Nat.choose 8 2 + Nat.choose 9 2 + 
   Nat.choose 10 2 = 165) :=
by
  intros
  sorry

end sum_of_combinations_l114_114582


namespace greatest_product_sum_300_l114_114204

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l114_114204


namespace probability_sqrt_less_than_nine_l114_114339

/-- Define the set of two-digit integers --/
def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Define the condition that the square root of the number is less than 9 --/
def sqrt_less_than_nine (n : Nat) : Prop := n < 81

/-- The number of integers from 10 to 80 --/
lemma count_satisfying_sqrt (n : Nat) : Prop :=
  is_two_digit n ∧ sqrt_less_than_nine n → n < 81

/-- Total number of two-digit integers --/
lemma count_two_digit_total (n : Nat) : Prop := is_two_digit n 

/-- The probability that a randomly selected two-digit integer's square root is less than 9. --/
theorem probability_sqrt_less_than_nine : 
  (∃ n, count_satisfying_sqrt n) / (∃ n, count_two_digit_total n) = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114339


namespace probability_sqrt_less_than_nine_is_correct_l114_114377

def probability_sqrt_less_than_nine : ℚ :=
  let total_two_digit_numbers := 99 - 10 + 1 in
  let satisfying_numbers := 80 - 10 + 1 in
  let probability := (satisfying_numbers : ℚ) / (total_two_digit_numbers : ℚ) in
  probability

theorem probability_sqrt_less_than_nine_is_correct :
  probability_sqrt_less_than_nine = 71 / 90 :=
by
  -- proof here
  sorry

end probability_sqrt_less_than_nine_is_correct_l114_114377


namespace probability_sqrt_less_than_nine_is_correct_l114_114383

def probability_sqrt_less_than_nine : ℚ :=
  let total_two_digit_numbers := 99 - 10 + 1 in
  let satisfying_numbers := 80 - 10 + 1 in
  let probability := (satisfying_numbers : ℚ) / (total_two_digit_numbers : ℚ) in
  probability

theorem probability_sqrt_less_than_nine_is_correct :
  probability_sqrt_less_than_nine = 71 / 90 :=
by
  -- proof here
  sorry

end probability_sqrt_less_than_nine_is_correct_l114_114383


namespace incorrect_statement_l114_114570

variable {p q : Prop}
variables {m n : ℝ} {x : ℝ}
noncomputable def prop1 := ¬(p ∧ q) → (p ∨ q)
def prop2 := ¬(m^2 + n^2 = 0) → (m ≠ 0 ∨ n ≠ 0)
def prop3 := (x = 0) → (x^2 - x = 0)
def prop4 := ∃ n : ℕ, n^2 > 2 * n

theorem incorrect_statement : ¬prop1 ∧ prop2 ∧ prop3 ∧ prop4 := by sorry

end incorrect_statement_l114_114570


namespace greatest_product_two_ints_sum_300_l114_114265

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l114_114265


namespace digit_A_divisibility_l114_114761

-- We define the problem as a theorem in Lean.

theorem digit_A_divisibility :
  ∃ A : ℕ, A < 10 ∧
    (∃ n : ℕ, 353808 * 10 + A = n * 2) ∧
    (∃ m : ℕ, 353808 * 10 + A = m * 3) ∧
    (∃ p : ℕ, 353808 * 10 + A = p * 4) ∧
    (∃ q : ℕ, 353808 * 10 + A = q * 5) ∧
    (∃ r : ℕ, 353808 * 10 + A = r * 6) ∧
    (∃ s : ℕ, 353808 * 10 + A = s * 8) ∧
    (∃ t : ℕ, 353808 * 10 + A = t * 9) ↔ A = 0 :=
begin
  sorry -- Proof goes here
end

end digit_A_divisibility_l114_114761


namespace smaug_hoard_value_l114_114797

def value_in_copper_coins (gold silver copper : ℕ) := gold * 24 + silver * 8 + copper

theorem smaug_hoard_value
  (gold_coins : ℕ) (silver_coins : ℕ) (copper_coins : ℕ)
  (value_silver_in_copper : ℕ) (value_gold_in_silver : ℕ) (hoarded_gold : gold_coins = 100)
  (hoarded_silver : silver_coins = 60) (hoarded_copper : copper_coins = 33)
  (value_silver : value_silver_in_copper = 8) (value_gold : value_gold_in_silver = 3) :
  value_in_copper_coins 100 60 33 = 2913 :=
by
  rw [hoarded_gold, hoarded_silver, hoarded_copper, value_silver_in_copper, value_gold]
  calc
    100 * (24:ℕ) + 60 * 8 + 33
    = 2400 + 480 + 33    : rfl
    ... = 2913           : rfl
  sorry

end smaug_hoard_value_l114_114797


namespace geometric_seq_sum_l114_114671

theorem geometric_seq_sum (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_a1_pos : a 1 > 0)
  (h_a4_7 : a 4 + a 7 = 2)
  (h_a5_6 : a 5 * a 6 = -8) :
  a 1 + a 4 + a 7 + a 10 = -5 := 
sorry

end geometric_seq_sum_l114_114671


namespace max_product_300_l114_114135

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l114_114135


namespace greatest_product_two_ints_sum_300_l114_114257

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l114_114257


namespace minimize_cost_l114_114884

-- Define the conditions
def p_event := 0.3
def loss_event := 4000000
def cost_A := 450000
def cost_B := 300000
def p_no_event_A := 0.9
def p_no_event_B := 0.85

-- Define the expected loss without any measures
def expected_loss := loss_event * p_event

-- Define the expected loss with only measure A
def expected_loss_A := loss_event * (1 - p_no_event_A)

-- Define the expected loss with only measure B
def expected_loss_B := loss_event * (1 - p_no_event_B)

-- Define the total cost with no measures
def total_cost_no_measures := expected_loss

-- Define the total cost with only measure A
def total_cost_A := cost_A + expected_loss_A

-- Define the total cost with only measure B
def total_cost_B := cost_B + expected_loss_B

-- Define the combined probability of the event with both measures
def combined_p_event := (1 - p_no_event_A) * (1 - p_no_event_B)

-- Define the expected loss with both measures
def expected_loss_AB := loss_event * combined_p_event

-- Define the total cost with both measures
def total_cost_AB := cost_A + cost_B + expected_loss_AB

-- Prove that using both measure A and measure B results in the lowest total cost
theorem minimize_cost : total_cost_AB < total_cost_no_measures ∧ total_cost_AB < total_cost_A ∧ total_cost_AB < total_cost_B := by
  sorry

end minimize_cost_l114_114884


namespace probability_sqrt_two_digit_lt_nine_correct_l114_114433

noncomputable def probability_sqrt_two_digit_lt_nine : ℚ :=
  let two_digit_integers := finset.Icc 10 99
  let satisfying_integers := two_digit_integers.filter (λ n => n < 81)
  let probability := (satisfying_integers.card : ℚ) / (two_digit_integers.card : ℚ)
  probability

theorem probability_sqrt_two_digit_lt_nine_correct :
  probability_sqrt_two_digit_lt_nine = 71 / 90 := by
  sorry

end probability_sqrt_two_digit_lt_nine_correct_l114_114433


namespace ratio_of_cookies_given_to_sister_l114_114791

-- Definitions from the conditions
def initial_cookies : ℕ := 20
def cookies_given_to_brother : ℕ := 10
def cookies_given_by_mother (cookies_given_to_brother : ℕ) : ℕ := cookies_given_to_brother / 2
def final_cookies : ℕ := 5

-- The proof goal
theorem ratio_of_cookies_given_to_sister (initial_cookies cookies_given_to_brother final_cookies : ℕ) :
  cookies_given_to_brother = 10 →
  initial_cookies = 20 →
  final_cookies = 5 →
  (let total_after_mother := initial_cookies - cookies_given_to_brother + cookies_given_by_mother cookies_given_to_brother
      in total_after_mother = 15 →
         let cookies_given_to_sister := total_after_mother - final_cookies
         in cookies_given_to_sister = 10 → 2 * total_after_mother = 3 * cookies_given_to_sister) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end ratio_of_cookies_given_to_sister_l114_114791


namespace probability_sqrt_lt_9_l114_114425

theorem probability_sqrt_lt_9 : 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 →
  ∃ p : ℚ, p = 71 / 90 ∧ 
  ∑ k in (Finset.range 100).filter (λ x, 10 ≤ x ∧ sqrt x < 9), 1 / 90 = p := 
sorry

end probability_sqrt_lt_9_l114_114425


namespace probability_sqrt_lt_9_l114_114393

theorem probability_sqrt_lt_9 : 
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
in probability = 71 / 90 :=
by
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  sorry

end probability_sqrt_lt_9_l114_114393


namespace max_product_two_integers_l114_114191

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l114_114191


namespace probability_sqrt_less_nine_l114_114505

theorem probability_sqrt_less_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  ∃ (p : ℚ), p = (finset.card (valid_numbers.to_finset) : ℚ) / (finset.card (two_digit_numbers.to_finset) : ℚ) ∧ p = 71 / 90 :=
by
  sorry

end probability_sqrt_less_nine_l114_114505


namespace find_positive_integer_pair_l114_114614

theorem find_positive_integer_pair (a b : ℕ) (h : ∀ n : ℕ, n > 0 → ∃ c_n : ℕ, a^n + b^n = c_n^(n + 1)) : a = 2 ∧ b = 2 := 
sorry

end find_positive_integer_pair_l114_114614


namespace prob_sqrt_less_than_nine_l114_114485

/-- The probability that the square root of a randomly selected 
two-digit whole number is less than nine is 71/90. -/
theorem prob_sqrt_less_than_nine : (let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99};
                                     let A := {n : ℕ | 10 ≤ n ∧ n < 81};
                                     (A.card / S.card : ℚ) = 71 / 90) :=
by
  sorry

end prob_sqrt_less_than_nine_l114_114485


namespace probability_sqrt_less_than_nine_l114_114342

/-- Define the set of two-digit integers --/
def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Define the condition that the square root of the number is less than 9 --/
def sqrt_less_than_nine (n : Nat) : Prop := n < 81

/-- The number of integers from 10 to 80 --/
lemma count_satisfying_sqrt (n : Nat) : Prop :=
  is_two_digit n ∧ sqrt_less_than_nine n → n < 81

/-- Total number of two-digit integers --/
lemma count_two_digit_total (n : Nat) : Prop := is_two_digit n 

/-- The probability that a randomly selected two-digit integer's square root is less than 9. --/
theorem probability_sqrt_less_than_nine : 
  (∃ n, count_satisfying_sqrt n) / (∃ n, count_two_digit_total n) = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114342


namespace max_product_of_sum_300_l114_114280

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l114_114280


namespace hyperbola_eqn_same_asymptotes_through_point_l114_114970

noncomputable def standard_equation_of_hyperbola_with_asymptotes (
  a b : ℝ) (p : ℝ × ℝ) :=
  ∃ m : ℝ, (a^2 * p.1^2 - b^2 * p.2^2) / (a^2 * b^2) = m

theorem hyperbola_eqn_same_asymptotes_through_point :
  standard_equation_of_hyperbola_with_asymptotes 3 4 (-(√3), 2 * √3) →
  ∃ a b : ℝ, a = 5 ∧ b = 15 / 4 ∧ (λ x y : ℝ, y^2 / a - x^2 / b = 1) :=
begin
  sorry
end

end hyperbola_eqn_same_asymptotes_through_point_l114_114970


namespace prob_sqrt_less_than_nine_l114_114474

/-- The probability that the square root of a randomly selected 
two-digit whole number is less than nine is 71/90. -/
theorem prob_sqrt_less_than_nine : (let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99};
                                     let A := {n : ℕ | 10 ≤ n ∧ n < 81};
                                     (A.card / S.card : ℚ) = 71 / 90) :=
by
  sorry

end prob_sqrt_less_than_nine_l114_114474


namespace point_in_second_quadrant_l114_114738

def point (x : ℤ) (y : ℤ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant : point (-1) 3 = true := by
  sorry

end point_in_second_quadrant_l114_114738


namespace greatest_product_of_two_integers_with_sum_300_l114_114228

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l114_114228


namespace box_height_correct_l114_114889

noncomputable def box_height : ℕ :=
  8

theorem box_height_correct (box_width box_length block_height block_width block_length : ℕ) (num_blocks : ℕ) :
  box_width = 10 ∧
  box_length = 12 ∧
  block_height = 3 ∧
  block_width = 2 ∧
  block_length = 4 ∧
  num_blocks = 40 →
  (num_blocks * block_height * block_width * block_length) /
  (box_width * box_length) = box_height :=
  by
  sorry

end box_height_correct_l114_114889


namespace probability_sqrt_less_than_nine_l114_114344

/-- Define the set of two-digit integers --/
def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Define the condition that the square root of the number is less than 9 --/
def sqrt_less_than_nine (n : Nat) : Prop := n < 81

/-- The number of integers from 10 to 80 --/
lemma count_satisfying_sqrt (n : Nat) : Prop :=
  is_two_digit n ∧ sqrt_less_than_nine n → n < 81

/-- Total number of two-digit integers --/
lemma count_two_digit_total (n : Nat) : Prop := is_two_digit n 

/-- The probability that a randomly selected two-digit integer's square root is less than 9. --/
theorem probability_sqrt_less_than_nine : 
  (∃ n, count_satisfying_sqrt n) / (∃ n, count_two_digit_total n) = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114344


namespace probability_sqrt_less_than_nine_l114_114334

/-- Define the set of two-digit integers --/
def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Define the condition that the square root of the number is less than 9 --/
def sqrt_less_than_nine (n : Nat) : Prop := n < 81

/-- The number of integers from 10 to 80 --/
lemma count_satisfying_sqrt (n : Nat) : Prop :=
  is_two_digit n ∧ sqrt_less_than_nine n → n < 81

/-- Total number of two-digit integers --/
lemma count_two_digit_total (n : Nat) : Prop := is_two_digit n 

/-- The probability that a randomly selected two-digit integer's square root is less than 9. --/
theorem probability_sqrt_less_than_nine : 
  (∃ n, count_satisfying_sqrt n) / (∃ n, count_two_digit_total n) = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114334


namespace elois_made_3_loaves_on_Monday_l114_114954

theorem elois_made_3_loaves_on_Monday
    (bananas_per_loaf : ℕ)
    (twice_as_many : ℕ)
    (total_bananas : ℕ) 
    (h1 : bananas_per_loaf = 4) 
    (h2 : twice_as_many = 2) 
    (h3 : total_bananas = 36)
  : ∃ L : ℕ, (4 * L + 8 * L = 36) ∧ L = 3 :=
sorry

end elois_made_3_loaves_on_Monday_l114_114954


namespace medicine_duration_l114_114748

theorem medicine_duration (days_per_third_pill : ℕ) (pills : ℕ) (days_per_month : ℕ)
  (h1 : days_per_third_pill = 3)
  (h2 : pills = 90)
  (h3 : days_per_month = 30) :
  ((pills * (days_per_third_pill * 3)) / days_per_month) = 27 :=
sorry

end medicine_duration_l114_114748


namespace circumcircle_through_circumcenters_l114_114017

variable {P : Type} [EuclideanGeometry P]

-- Given conditions
variables {A B C O X Y : P}
variables (hO : Circumcenter O A B C)
           (hX : AX = BX)
           (hY : AY = CY)

-- Define circumcenters of the sub-triangles
variables (OA OC : P)
variables (hOA : Circumcenter OA A O B)
           (hOC : Circumcenter OC A O C)

-- The theorem to be proved
theorem circumcircle_through_circumcenters
  (h : ∃ (k : Set P), IsCircumcircle k A X Y) :
  ∃ (k : Set P), IsCircumcircle k A X Y ∧
  OA ∈ k ∧ OC ∈ k :=
sorry

end circumcircle_through_circumcenters_l114_114017


namespace hyperbola_triangle_perimeter_l114_114678

theorem hyperbola_triangle_perimeter 
  (A B F₁ F₂ : Type)
  (h_hyperbola : ∀ x y : ℝ, x^2 - 4 * y^2 = 4)
  (h_foci : ∀ F₁ F₂ : ℝ, is_focus_of_hyperbola F₁ F₂ (λ x y : ℝ, x^2 - 4 * y^2 = 4))
  (h_line : ∀ A B : ℝ, is_on_line_segment A B F₁)
  (h_AB : dist A B = 5) : 
  ∃ AF₂ BF₂, triangle_perimeter A F₂ B = 18 :=
sorry

end hyperbola_triangle_perimeter_l114_114678


namespace max_product_two_integers_l114_114186

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l114_114186


namespace greatest_product_sum_300_l114_114128

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l114_114128


namespace minimal_distance_exists_l114_114000

theorem minimal_distance_exists (A B C : ℝ) (hA : A ≠ 0) (hB : B ≠ 0) :
  ∃ (x y : ℤ), dist (x,y) (Ax + By - C = 0) ≤ (1 / (2 * Real.sqrt 2)) :=
sorry

end minimal_distance_exists_l114_114000


namespace greatest_product_sum_300_l114_114199

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l114_114199


namespace min_value_c_and_d_l114_114010

theorem min_value_c_and_d (c d : ℝ) (h1 : c > 0) (h2 : d > 0)
  (h3 : c^2 - 12 * d ≥ 0)
  (h4 : 9 * d^2 - 4 * c ≥ 0) :
  c + d ≥ 5.74 :=
sorry

end min_value_c_and_d_l114_114010


namespace greatest_product_of_two_integers_with_sum_300_l114_114238

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l114_114238


namespace simplify_trig_expression_l114_114796

theorem simplify_trig_expression (x : ℝ) :
  (1 + real.sin x) / real.cos x * (real.sin (2 * x) / (2 * (real.cos (real.pi / 4 - x / 2))^2)) = 2 * real.sin x :=
by sorry

end simplify_trig_expression_l114_114796


namespace max_product_of_sum_300_l114_114279

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l114_114279


namespace probability_sqrt_lt_nine_l114_114412

theorem probability_sqrt_lt_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  let probability := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  probability = 71 / 90 :=
by
  let two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n : ℕ | 10 ≤ n ∧ n < 81}
  have h1 : two_digit_numbers.card = 90 := sorry
  have h2 : valid_numbers.card = 71 := sorry
  let probability : ℚ := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  rw [h1, h2]
  simp
  norm_num

end probability_sqrt_lt_nine_l114_114412


namespace problem_l114_114686

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - Real.log (x + 2) + Real.log a - 2

theorem problem (a x : ℝ) (h_extrem : ∀ x, (f a x = f a 0 → a = 1/2)) (h_nonneg : ∀ x, f a x ≥ 0) :
  (a ∈ Set.Icc Real.exp 1 0 (Set.Ioi (0 : ℝ)) := by
begin
  sorry
end

end problem_l114_114686


namespace edge_length_increase_l114_114856

theorem edge_length_increase (e e' : ℝ) (A : ℝ) (hA : ∀ e, A = 6 * e^2)
  (hA' : 2.25 * A = 6 * e'^2) :
  (e' - e) / e * 100 = 50 :=
by
  sorry

end edge_length_increase_l114_114856


namespace ratio_P_Q_l114_114801

def is_square_array (A : Matrix (Fin 50) (Fin 50) ℝ) : Prop :=
  true  -- A is a 50x50 matrix with unique measurements

def row_sum (A : Matrix (Fin 50) (Fin 50) ℝ) (i : Fin 50) : ℝ :=
  ∑ j, A i j

def col_sum (A : Matrix (Fin 50) (Fin 50) ℝ) (j : Fin 50) : ℝ :=
  ∑ i, A i j

def P (A : Matrix (Fin 50) (Fin 50) ℝ) : ℝ :=
  (∑ i, row_sum A i) / 50

def Q (A : Matrix (Fin 50) (Fin 50) ℝ) : ℝ :=
  (∑ j, col_sum A j) / 50

theorem ratio_P_Q (A : Matrix (Fin 50) (Fin 50) ℝ) (h : is_square_array A) :
  P A / Q A = 1 :=
by
  sorry

end ratio_P_Q_l114_114801


namespace sum_of_cubes_first_eight_pos_integers_l114_114925

theorem sum_of_cubes_first_eight_pos_integers : (∑ i in (Finset.range 8).map Nat.succ, i ^ 3) = 1296 := by
sorry

end sum_of_cubes_first_eight_pos_integers_l114_114925


namespace greatest_product_l114_114294

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l114_114294


namespace probability_of_sqrt_lt_9_l114_114347

-- Define the set of two-digit whole numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the subset of numbers for which the square root is less than 9
def valid_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 80}

-- Define the probability calculation
noncomputable def probability_sqrt_lt_9 := (valid_numbers.to_finset.card : ℝ) / (two_digit_numbers.to_finset.card : ℝ)

-- The statement we aim to prove
theorem probability_of_sqrt_lt_9 : probability_sqrt_lt_9 = 71 / 90 := 
sorry

end probability_of_sqrt_lt_9_l114_114347


namespace find_m_l114_114572

theorem find_m (m : ℝ) (a b : ℝ) (r s : ℝ) (S1 S2 : ℝ)
  (h1 : a = 10)
  (h2 : b = 10)
  (h3 : 10 * r = 5)
  (h4 : S1 = 20)
  (h5 : 10 * s = 5 + m)
  (h6 : S2 = 100 / (5 - m))
  (h7 : S2 = 3 * S1) :
  m = 10 / 3 := by
  sorry

end find_m_l114_114572


namespace apex_angle_of_quadrilateral_pyramid_l114_114962

theorem apex_angle_of_quadrilateral_pyramid :
  ∃ (α : ℝ), α = Real.arccos ((Real.sqrt 5 - 1) / 2) :=
sorry

end apex_angle_of_quadrilateral_pyramid_l114_114962


namespace greatest_product_sum_300_l114_114246

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l114_114246


namespace probability_sqrt_less_than_nine_l114_114456

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sqrt_less_than_nine (n : ℕ) : Prop := n < 81

theorem probability_sqrt_less_than_nine :
  (∃ total_num favorable_num,
    (total_num = Finset.card (Finset.filter is_two_digit (Finset.range 100)) ∧
     favorable_num = Finset.card (Finset.filter (λ n, is_two_digit n ∧ sqrt_less_than_nine n) (Finset.range 100)) ∧
     (favorable_num : ℚ) / total_num = 71 / 90)) :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114456


namespace solve_for_a_minus_b_l114_114982

theorem solve_for_a_minus_b (a b : ℝ) (h1 : |a| = 5) (h2 : |b| = 7) (h3 : |a + b| = a + b) : a - b = -2 := 
sorry

end solve_for_a_minus_b_l114_114982


namespace y_value_l114_114702

theorem y_value (x y : ℝ) (hx : 1 < x) (hy : 1 < y) (h_eq1 : (1 / x) + (1 / y) = 3 / 2) (h_eq2 : x * y = 9) : y = 6 :=
sorry

end y_value_l114_114702


namespace greatest_product_sum_300_l114_114244

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l114_114244


namespace greatest_product_sum_300_l114_114247

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l114_114247


namespace matrix_power_2023_correct_l114_114933

noncomputable def matrix_power_2023 : Matrix (Fin 2) (Fin 2) ℤ :=
  let A := !![1, 0; 2, 1]  -- Define the matrix
  A^2023

theorem matrix_power_2023_correct :
  matrix_power_2023 = !![1, 0; 4046, 1] := by
  sorry

end matrix_power_2023_correct_l114_114933


namespace range_of_x_value_of_a_l114_114770

-- Condition Definitions
def m (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
def n (x : ℝ) : ℝ × ℝ := (-Real.cos x, Real.sqrt 3 * Real.cos x)
def f (x : ℝ) : ℝ := (m x).1 * (0.5 * (m x).1 - (n x).1) + (m x).2 * (0.5 * (m x).2 - (n x).2)

-- Problem A: Range of x where f(x) ≥ 1/2
theorem range_of_x (x : ℝ) (k : ℤ) : 
  (f x ≥ 1/2) ↔ x ∈ Set.Icc (k * Real.pi - Real.pi / 2) (k * Real.pi + Real.pi / 6) :=
sorry

-- Problem B: Value of a given conditions in ΔABC
noncomputable def law_of_cosines (a b c B : ℝ) := b^2 = a^2 + c^2 - 2 * a * c * Real.cos B

theorem value_of_a (B : ℝ) (a b c : ℝ) (h : f (B / 2) = 1) (hb : b = 1) (hc : c = Real.sqrt 3) :
  law_of_cosines a b c B → a = 1 :=
sorry

end range_of_x_value_of_a_l114_114770


namespace find_years_until_ratio_is_3_2_l114_114094

-- Let j and m be John's and Mary's current ages respectively.
def john_current_age : ℕ
def mary_current_age : ℕ

-- Condition 1: Three years ago, John was twice as old as Mary.
axiom condition1 : john_current_age - 3 = 2 * (mary_current_age - 3)

-- Condition 2: Four years before that, John was three times as old as Mary.
axiom condition2 : john_current_age - 7 = 3 * (mary_current_age - 7)

-- We need to prove the number of years until the ratio of their ages is 3:2.
theorem find_years_until_ratio_is_3_2 (x : ℕ) : 
  (john_current_age + x) * 2 = (mary_current_age + x) * 3 → x = 5 :=
by
  sorry

end find_years_until_ratio_is_3_2_l114_114094


namespace line_properties_l114_114640

theorem line_properties (m : ℝ) :
  (∀ (y : ℝ), y = 0 → my + 1 = 1) ∧
  (m = 2 → (1 * 2^-1) / 2 = 1 / 4) :=
by
  split
  case left =>
    intro y hy
    rw [hy, mul_zero, add_one]
  case right =>
    intro hm
    rw [hm, mul_one, (div_eq_mul_inv 2), inv_of_one_eq_one, mul_one]
    norm_num

#check line_properties

end line_properties_l114_114640


namespace measure_angle_D_l114_114096

open Real

noncomputable def angle_A (A B : ℝ) : ℝ := A
noncomputable def angle_B (A B : ℝ) : ℝ := 5 * A
noncomputable def angle_D (A B : ℝ) : ℝ := A

theorem measure_angle_D (A B : ℝ) (h1 : A = (1/5) * B) (h2 : 6 * A = 180) : angle_D A B = 30 :=
by
  rw [angle_D]
  exact eq_of_mul_eq_mul_left (by norm_num) (by linarith)

#print axioms measure_angle_D

end measure_angle_D_l114_114096


namespace number_of_correct_props_l114_114997

-- Definitions to model planes, lines, and their relationships
variable (α β : Plane)
variable (m n : Line)

-- Proposition conditions in Lean terms
def prop1 := ∀ (h1 : m ⊥ α) (h2 : m ⊥ β), α ∥ β
def prop2 := ∀ (h1 : m ∥ α) (h2 : α ∩ β = n), m ∥ n
def prop3 := ∀ (h1 : m ∥ n) (h2 : m ⊥ α), n ⊥ α
def prop4 := ∀ (h1 : m ⊥ α) (h2 : m ∥ n) (h3 : n ⊂ β), α ⊥ β

-- The main theorem asserting that the total number of correct propositions is 3
theorem number_of_correct_props :
  (prop1 α β m ∧ prop3 α β m n ∧ prop4 α β m n) ∧ ¬ prop2 α β m n :=
  sorry

end number_of_correct_props_l114_114997


namespace find_x3_l114_114842

noncomputable def y1 (x1 : ℝ) : ℝ := Real.log x1
noncomputable def y2 (x2 : ℝ) : ℝ := Real.log x2
noncomputable def yC (x1 x2 : ℝ) : ℝ := (1/4) * Real.log x1 + (3/4) * Real.log x2
noncomputable def x3 (x1 x2 : ℝ) : ℝ := Real.exp yC x1 x2

theorem find_x3 (x1 x2 : ℝ) (h1 : x1 = 2) (h2 : x2 = 32) : x3 x1 x2 = 16 :=
by
  -- The proof goes here
  sorry

end find_x3_l114_114842


namespace parabola_vertex_l114_114060

theorem parabola_vertex :
  (∃ h k : ℝ, ∀ x : ℝ, (y : ℝ) = (x - 2)^2 + 5 ∧ h = 2 ∧ k = 5) :=
sorry

end parabola_vertex_l114_114060


namespace max_product_of_two_integers_whose_sum_is_300_l114_114177

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l114_114177


namespace cannot_place_four_in_E_l114_114603

/-- Define the set of numbers to be used, the circles, and the alignment condition --/
def set_of_numbers : set ℕ := {1, 2, 3, 4, 5, 6, 7}
def circles := {A, B, C, D, E, F, G} -- Represent circles with variables A-G
def alignment (f : circles → ℕ) : Prop :=
  ∀ (x y z : circles), x ≠ y ∧ y ≠ z ∧ x ≠ z → f x + f y + f z = s
  where s : ℕ := f A + f B + f C -- This s ensures sum of any aligned three numbers must be the same

/-- Define the proposition --/
def cannot_place_four_at_E (f : circles → ℕ) : Prop :=
  ∀ (n : ℕ), n ∈ set_of_numbers → 
  alignment f → 
  (f A = 4 → f E ≠ 4)

/-- Translate to a proof statement --/
theorem cannot_place_four_in_E (f : circles → ℕ) : cannot_place_four_at_E f :=
by {
  sorry
}

end cannot_place_four_in_E_l114_114603


namespace probability_sqrt_lt_9_l114_114428

theorem probability_sqrt_lt_9 : 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 →
  ∃ p : ℚ, p = 71 / 90 ∧ 
  ∑ k in (Finset.range 100).filter (λ x, 10 ≤ x ∧ sqrt x < 9), 1 / 90 = p := 
sorry

end probability_sqrt_lt_9_l114_114428


namespace greatest_product_sum_300_l114_114195

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l114_114195


namespace find_value_of_a_l114_114700

noncomputable def curves_intersect_and_perpendicular (a : ℝ) : Prop :=
  ∃ x : ℝ, 0 < x ∧ x < π / 2 ∧
           2 * Real.sin x = a * Real.cos x ∧
           (let f := 2 * Real.sin x, g := a * Real.cos x,
                f' := 2 * Real.cos x, g' := -a * Real.sin x in
             f' * g' = -1)

theorem find_value_of_a : ∃ a : ℝ, a = 2 * Real.sqrt 3 / 3 ∧ curves_intersect_and_perpendicular a :=
sorry

end find_value_of_a_l114_114700


namespace area_enclosed_by_curve_is_4_l114_114805

noncomputable def area_under_curve : ℝ :=
  ∫ x in -1..1, (3 - 3 * x^2)

theorem area_enclosed_by_curve_is_4 : area_under_curve = 4 :=
by
  sorry

end area_enclosed_by_curve_is_4_l114_114805


namespace problem_l114_114984

noncomputable def f : ℝ → ℝ
| x := if (-1 < x ∧ x <= 0) then 1 else if (0 < x ∧ x <= 1) then -1 else 0

axiom f_property : ∀ x : ℝ, f(x + 1) = f(-x)

theorem problem (x : ℝ) (hx : x = 3.5) : f(f(3.5)) = -1 :=
sorry

end problem_l114_114984


namespace greatest_product_obtainable_l114_114317

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l114_114317


namespace probability_sqrt_two_digit_lt_nine_correct_l114_114442

noncomputable def probability_sqrt_two_digit_lt_nine : ℚ :=
  let two_digit_integers := finset.Icc 10 99
  let satisfying_integers := two_digit_integers.filter (λ n => n < 81)
  let probability := (satisfying_integers.card : ℚ) / (two_digit_integers.card : ℚ)
  probability

theorem probability_sqrt_two_digit_lt_nine_correct :
  probability_sqrt_two_digit_lt_nine = 71 / 90 := by
  sorry

end probability_sqrt_two_digit_lt_nine_correct_l114_114442


namespace parallelogram_area_l114_114529

theorem parallelogram_area (base height : ℕ) (h_base : base = 32) (h_height : height = 14) :
  let area := base * height in
  area = 448 := by
  -- proof goes here
  sorry

end parallelogram_area_l114_114529


namespace min_x_plus_y_tan_l114_114969

theorem min_x_plus_y_tan (x y : ℝ) (hx : (Real.tan x - 2) * (Real.tan y - 2) = 5) :
  x + y ≥ π - Real.arctan (1 / 2) := 
sorry

-- Test for positive value
noncomputable def min_positive_x_plus_y_tan :=
  if h : ∃ x y : ℝ, (Real.tan x - 2) * (Real.tan y - 2) = 5 ∧ x + y > 0 then
     some ⟨y, x, h.2⟩ else 0

example : min_positive_x_plus_y_tan ≥ π - Real.arctan (1 / 2) := 
sorry

end min_x_plus_y_tan_l114_114969


namespace greatest_product_sum_300_l114_114254

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l114_114254


namespace root_certificate_l114_114967

theorem root_certificate :
  (∃ (x : ℝ), (x = -2 ∨ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2) ∧
  (x^4 - 4 * x^3 + 5 * x^2 - 2 * x - 8 = 0)) :=
begin
  sorry
end

end root_certificate_l114_114967


namespace segment_sum_inequality_l114_114656

theorem segment_sum_inequality :
  let n := 2022
  let half_n := n / 2
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → (i <= half_n → is_red i) ∧ (i > half_n → is_blue i)) →
  (sum_of_lengths (red_left_endpoint segments) (blue_right_endpoint segments) ≠
   sum_of_lengths (blue_left_endpoint segments) (red_right_endpoint segments)) :=
by
  sorry

end segment_sum_inequality_l114_114656


namespace tan_alpha_plus_beta_l114_114681

noncomputable theory
open Complex

variables {α β : ℝ}

def z : ℂ := cos α + sin α * Complex.I
def u : ℂ := cos β + sin β * Complex.I

theorem tan_alpha_plus_beta 
  (h : z + u = (4 / 5 : ℂ) + (3 / 5) * Complex.I) : 
  Real.tan (α + β) = 24 / 7 := 
sorry

end tan_alpha_plus_beta_l114_114681


namespace required_speed_to_cover_distance_in_new_time_l114_114545
-- Import the necessary libraries

-- Define the given conditions
def distance : ℝ := 270
def original_time : ℝ := 6
def new_time : ℝ := (3 / 2) * original_time

-- State the theorem
theorem required_speed_to_cover_distance_in_new_time :
  (distance / new_time) = 30 := by
  sorry

end required_speed_to_cover_distance_in_new_time_l114_114545


namespace solve_positive_solutions_system_l114_114615

def positive_solutions_system (n : ℕ) (n_gt_1 : n > 1) := 
  ∃ (x : Fin n → ℝ), 
    (∀ i, x i > 0) ∧ 
    (Finset.univ.sum (λ i, (i + 1) * x i) = 3) ∧ 
    (Finset.univ.sum (λ i, 1 / ((i + 1) * x i)) = 3)
    
theorem solve_positive_solutions_system :
  ∀ (n : ℕ) (n_gt_1 : n > 1), 
    positive_solutions_system n n_gt_1 ↔ 
      (n = 3 ∧ 
        ∃ x, x 0 = 1 ∧ x 1 = 1/2 ∧ x 2 = 1/3) ∨ 
      (n = 2 ∧ 
        (∃ x, x 0 = (3 + Real.sqrt 5) / 2 ∧ x 1 = (3 - Real.sqrt 5) / 4) ∨ 
        (∃ x, x 0 = (3 - Real.sqrt 5) / 2 ∧ x 1 = (3 + Real.sqrt 5) / 4))  :=
by
  intro n n_gt_1
  sorry

end solve_positive_solutions_system_l114_114615


namespace lower_limit_even_digits_l114_114085

theorem lower_limit_even_digits :
  ∃ L : ℕ, (∀ b, L ≤ b ∧ b ≤ 1000 → ∃ m, b = 10 * m ∧ ∃ c, c ∈ {0, 2, 4, 6, 8} ∧ c = b % 10) ∧ (L = 500) :=
by {
  let L := 500,
  split, 
  { intros b Hb,
    use b / 10,
    split, 
    { exact int.div_mul_cancel (nat.mod_eq_zero_of_dvd (by sorry)) },   -- This should demonstrate that b = 10 * m
    split_ifs with c0 c2 c4 c6 c8,
    { exact ⟨0, by simpa⟩ },
    { exact ⟨2, by simpa⟩ },
    { exact ⟨4, by simpa⟩ },
    { exact ⟨6, by simpa⟩ },
    { exact ⟨8, by simpa⟩ },
  },
  { use 500, }
}

end lower_limit_even_digits_l114_114085


namespace length_ab_is_constant_l114_114987

noncomputable def length_AB_constant (p : ℝ) (hp : p > 0) : Prop :=
  let parabola := { P : ℝ × ℝ | P.1 ^ 2 = 2 * p * P.2 }
  let line := { P : ℝ × ℝ | P.2 = P.1 + p / 2 }
  (∃ A B : ℝ × ℝ, A ∈ parabola ∧ B ∈ parabola ∧ A ∈ line ∧ B ∈ line ∧ 
    dist A B = 4 * p)

theorem length_ab_is_constant (p : ℝ) (hp : p > 0) : length_AB_constant p hp :=
by {
  sorry
}

end length_ab_is_constant_l114_114987


namespace greatest_product_sum_300_l114_114130

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l114_114130


namespace max_product_of_sum_300_l114_114221

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l114_114221


namespace cost_of_candy_bar_l114_114593

theorem cost_of_candy_bar (t c b : ℕ) (h1 : t = 13) (h2 : c = 6) (h3 : t = b + c) : b = 7 := 
by
  sorry

end cost_of_candy_bar_l114_114593


namespace vera_first_place_l114_114919

noncomputable def placement (anna vera katya natasha : ℕ) : Prop :=
  (anna ≠ 1 ∧ anna ≠ 4) ∧ (vera ≠ 4) ∧ (katya = 1) ∧ (natasha = 4)

theorem vera_first_place :
  ∃ (anna vera katya natasha : ℕ),
    (placement anna vera katya natasha) ∧ 
    (vera = 1) ∧ 
    (1 ≠ 4) → 
    ((anna ≠ 1 ∧ anna ≠ 4) ∧ (vera ≠ 4) ∧ (katya = 1) ∧ (natasha = 4)) ∧ 
    (1 = 1) ∧ 
    (∃ i j k l : ℕ, (i ≠ 1 ∧ i ≠ 4) ∧ (j = 1) ∧ (k ≠ 1) ∧ (l = 4)) ∧ 
    (vera = 1) :=
sorry

end vera_first_place_l114_114919


namespace greatest_product_sum_300_l114_114251

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l114_114251


namespace inverse_proportion_range_l114_114692

theorem inverse_proportion_range (k : ℝ) (x : ℝ) :
  (∀ x : ℝ, (x < 0 -> (k - 1) / x > 0) ∧ (x > 0 -> (k - 1) / x < 0)) -> k < 1 :=
by
  sorry

end inverse_proportion_range_l114_114692


namespace find_f_six_l114_114652

noncomputable def f : ℕ → ℤ := sorry

axiom f_one_eq_one : f 1 = 1
axiom f_add (x y : ℕ) : f (x + y) = f x + f y + 8 * x * y - 2
axiom f_seven_eq_163 : f 7 = 163

theorem find_f_six : f 6 = 116 := 
by {
  sorry
}

end find_f_six_l114_114652


namespace exists_nat_with_2001_zeros_divisors_product_l114_114599

theorem exists_nat_with_2001_zeros_divisors_product :
  ∃ N : ℕ, (∏ d in (Finset.filter (λ d, d ∣ N) (Finset.range (N+1))), d) % 10^2001 = 0 :=
sorry

end exists_nat_with_2001_zeros_divisors_product_l114_114599


namespace greatest_product_of_two_integers_with_sum_300_l114_114233

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l114_114233


namespace probability_sqrt_lt_9_l114_114426

theorem probability_sqrt_lt_9 : 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 →
  ∃ p : ℚ, p = 71 / 90 ∧ 
  ∑ k in (Finset.range 100).filter (λ x, 10 ≤ x ∧ sqrt x < 9), 1 / 90 = p := 
sorry

end probability_sqrt_lt_9_l114_114426


namespace inequality_proof_l114_114649

theorem inequality_proof
  (x y z : ℝ)
  (hx : x > y)
  (hy : y > 1)
  (hz : 1 > z)
  (hzpos : z > 0)
  (a : ℝ := (1 + x * z) / z)
  (b : ℝ := (1 + x * y) / x)
  (c : ℝ := (1 + y * z) / y) :
  a > b ∧ a > c :=
by
  sorry

end inequality_proof_l114_114649


namespace probability_sqrt_less_than_nine_l114_114343

/-- Define the set of two-digit integers --/
def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Define the condition that the square root of the number is less than 9 --/
def sqrt_less_than_nine (n : Nat) : Prop := n < 81

/-- The number of integers from 10 to 80 --/
lemma count_satisfying_sqrt (n : Nat) : Prop :=
  is_two_digit n ∧ sqrt_less_than_nine n → n < 81

/-- Total number of two-digit integers --/
lemma count_two_digit_total (n : Nat) : Prop := is_two_digit n 

/-- The probability that a randomly selected two-digit integer's square root is less than 9. --/
theorem probability_sqrt_less_than_nine : 
  (∃ n, count_satisfying_sqrt n) / (∃ n, count_two_digit_total n) = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114343


namespace days_B_can_finish_alone_l114_114543

theorem days_B_can_finish_alone (x : ℚ) : 
  (1 / 3 : ℚ) + (1 / x) = (1 / 2 : ℚ) → x = 6 := 
by
  sorry

end days_B_can_finish_alone_l114_114543


namespace probability_red_or_blue_l114_114838

theorem probability_red_or_blue 
  (total_marbles : ℕ)
  (p_white p_green p_orange p_violet : ℚ)
  (h_total : total_marbles = 120)
  (h_white_prob : p_white = 1/5)
  (h_green_prob: p_green = 1/10)
  (h_orange_prob: p_orange = 1/6)
  (h_violet_prob: p_violet = 1/8)
  : (49 / 120 : ℚ) = 1 - (p_white + p_green + p_orange + p_violet) :=
by
  sorry

end probability_red_or_blue_l114_114838


namespace max_product_of_sum_300_l114_114210

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l114_114210


namespace greatest_product_sum_300_l114_114242

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l114_114242


namespace derangements_fixed_point_relation_l114_114535

theorem derangements_fixed_point_relation (X : Finset ℕ) (ϕ : Equiv.Perm X) (f_n g_n : ℕ) :
  X = {1, 2, ..., n} →
  (∀ i ∈ X, ϕ i = i ↔ False) →
  f_n = ∑ i in X, if ϕ i = i then 0 else 1 →
  g_n = ∑ i in X, if ϕ i = i then 1 else 0 →
  |f_n - g_n| = 1 :=
by sorry

end derangements_fixed_point_relation_l114_114535


namespace min_value_expr_l114_114629

noncomputable def min_value (a b c : ℝ) := 4 * a^3 + 8 * b^3 + 18 * c^3 + 1 / (9 * a * b * c)

theorem min_value_expr (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  min_value a b c ≥ 8 / Real.sqrt 3 :=
by
  sorry

end min_value_expr_l114_114629


namespace yen_exchange_rate_l114_114563

theorem yen_exchange_rate (yen_per_dollar : ℕ) (dollars : ℕ) (y : ℕ) (h1 : yen_per_dollar = 120) (h2 : dollars = 10) : y = 1200 :=
by
  have h3 : y = yen_per_dollar * dollars := by sorry
  rw [h1, h2] at h3
  exact h3

end yen_exchange_rate_l114_114563


namespace angles_of_triangle_l114_114745

theorem angles_of_triangle 
  (α β γ : ℝ)
  (triangle_ABC : α + β + γ = 180)
  (median_bisector_height : (γ / 4) * 4 = 90) :
  α = 22.5 ∧ β = 67.5 ∧ γ = 90 :=
by
  sorry

end angles_of_triangle_l114_114745


namespace greatest_product_of_two_integers_with_sum_300_l114_114227

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l114_114227


namespace probability_sqrt_less_than_nine_l114_114450

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sqrt_less_than_nine (n : ℕ) : Prop := n < 81

theorem probability_sqrt_less_than_nine :
  (∃ total_num favorable_num,
    (total_num = Finset.card (Finset.filter is_two_digit (Finset.range 100)) ∧
     favorable_num = Finset.card (Finset.filter (λ n, is_two_digit n ∧ sqrt_less_than_nine n) (Finset.range 100)) ∧
     (favorable_num : ℚ) / total_num = 71 / 90)) :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114450


namespace number_of_points_on_parabola_l114_114626

theorem number_of_points_on_parabola :
  let f (x : ℕ) := - (x * x) / 3 + 13 * x + 42
  (finset.filter (λ x, f x ∈ finset.range 43) (finset.range 42)).card = 13 :=
by sorry

end number_of_points_on_parabola_l114_114626


namespace fraction_of_grid_covered_by_triangle_l114_114564

variables (A B C : ℝ × ℝ)
variable (grid_width : ℝ)
variable (grid_height : ℝ)
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
1 / 2 * (abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)))

def grid_area (grid_width grid_height : ℝ) : ℝ :=
grid_width * grid_height

theorem fraction_of_grid_covered_by_triangle :
  let A := (2, 2 : ℝ)
  let B := (6, 2 : ℝ)
  let C := (5, 5 : ℝ)
  let grid_width := 7 : ℝ
  let grid_height := 6 : ℝ
  (triangle_area A B C) / (grid_area grid_width grid_height) = 11 / 84 :=
by
  sorry

end fraction_of_grid_covered_by_triangle_l114_114564


namespace probability_sqrt_less_nine_l114_114501

theorem probability_sqrt_less_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  ∃ (p : ℚ), p = (finset.card (valid_numbers.to_finset) : ℚ) / (finset.card (two_digit_numbers.to_finset) : ℚ) ∧ p = 71 / 90 :=
by
  sorry

end probability_sqrt_less_nine_l114_114501


namespace range_of_a_l114_114005

theorem range_of_a (a : ℝ) :
  let A := {x : ℝ | (x - 1) * (x - a) ≥ 0 },
      B := {x : ℝ | x ≥ a - 1 } in
  (A ∪ B = set.univ) ↔ (a ∈ set.Iic 2) :=
sorry

end range_of_a_l114_114005


namespace geom_seq_general_formula_find_range_of_lambda_l114_114051

variable {λ : ℝ}

theorem geom_seq_general_formula (a : ℕ → ℝ) (q : ℝ) 
    (h1 : 0 < q ∧ q < 1) 
    (h2 : ∀ n, a (n + 1) = a n * q) 
    (h3 : a 0 + a 0 * q = 12) 
    (h4 : 2 * (a 0 * q + 1) = a 0 + a 0 * q ^ 2) 
    : ∀ n, a n = (1 / 2)^(n - 4) := 
sorry

theorem find_range_of_lambda (a : ℕ → ℝ) (λ : ℝ) 
    (h1 : ∀ n, a n = (1 / 2) ^ (n - 4)) 
    (h2 : ∀ n, a n * (n - λ) > a (n - 1) * (n - 1 - λ)) 
    : λ < 2 := 
sorry

end geom_seq_general_formula_find_range_of_lambda_l114_114051


namespace variance_of_scores_l114_114562

-- Define the student's scores
def scores : List ℕ := [130, 125, 126, 126, 128]

-- Define a function to calculate the mean
def mean (l : List ℕ) : ℕ :=
  l.sum / l.length

-- Define a function to calculate the variance
def variance (l : List ℕ) : ℕ :=
  let avg := mean l
  (l.map (λ x => (x - avg) * (x - avg))).sum / l.length

-- The proof statement (no proof provided, use sorry)
theorem variance_of_scores : variance scores = 3 := by sorry

end variance_of_scores_l114_114562


namespace area_of_quadrilateral_l114_114890

-- Given conditions
def curve1 (x y : ℝ) : Prop := x^4 + y^4 = 100
def curve2 (x y : ℝ) : Prop := x * y = 4

-- Quadrilateral vertex definition
def vertex (x y : ℝ) : Prop := curve1 x y ∧ curve2 x y

-- Set of vertices of the quadrilateral
def vertices : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | vertex p.1 p.2}

-- Area of convex quadrilateral formed by intersection points
noncomputable def area_qdrt (S : set (ℝ × ℝ)) : ℝ :=
sorry

theorem area_of_quadrilateral :
  area_qdrt vertices = 4 * sqrt 17 :=
sorry

end area_of_quadrilateral_l114_114890


namespace greatest_product_of_two_integers_with_sum_300_l114_114236

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l114_114236


namespace problem1_problem2_l114_114670

variable {α : ℝ}
-- Given condition
axiom condition (h : ℝ) (h ≠ 1): (Real.tan α) / (Real.tan α - 1) = -1

-- Prove the first statement
theorem problem1 (h : ℝ) (h ≠ 1): 
  (Real.sin α - 2 * Real.cos α) / (Real.sin α + Real.cos α) = -1 :=
by
  sorry

-- Prove the second statement
theorem problem2 (h : ℝ) (h ≠ 1): 
  (Real.sin α)^2 + (Real.sin α) * (Real.cos α) = 3 / 5 :=
by
  sorry

end problem1_problem2_l114_114670


namespace greatest_product_two_ints_sum_300_l114_114261

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l114_114261


namespace orchard_problem_l114_114525

variables (T F G : ℕ)
variable h_cross_pollinated : 0.10 * T = 0.1 * T 
variable h_fuji_plus_cross_pollinated : F + 0.10 * T = 238
variable h_three_quarters_fuji : F = 0.75 * T 

theorem orchard_problem : G = T - F - 0.10 * T → G = 42 := 
by
  intro h
  sorry

end orchard_problem_l114_114525


namespace percentage_gain_for_person_l114_114558

def cost_price_bowls (num_bought : ℕ) (cost_per_bowl : ℕ) : ℕ := num_bought * cost_per_bowl
def selling_price_bowls (num_sold : ℕ) (price_per_bowl : ℕ) : ℕ := num_sold * price_per_bowl
def gain (cp : ℕ) (sp : ℕ) : ℕ := sp - cp
def percentage_gain (gain : ℕ) (cp : ℕ) : ℚ := (gain : ℚ) / cp * 100

theorem percentage_gain_for_person :
  let num_bought := 250
  let cost_per_bowl := 18
  let total_cp := cost_price_bowls num_bought cost_per_bowl
  let num_sold := 200
  let price_per_bowl := 25
  let total_sp := selling_price_bowls num_sold price_per_bowl
  let total_gain := gain total_cp total_sp
  percentage_gain total_gain total_cp ≈ 11.11 := 
by
  have total_cp_calc : total_cp = 4500 := by native_decide
  have total_sp_calc : total_sp = 5000 := by native_decide
  have total_gain_calc : total_gain = 500 := by native_decide
  have percentage_gain_calc : percentage_gain total_gain total_cp = (500/4500) * 100 := by native_decide
  have percentage_gain_approx : (500/4500) * 100 ≈ 11.11 := by native_decide
  exact percentage_gain_approx

end percentage_gain_for_person_l114_114558


namespace max_product_of_sum_300_l114_114218

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l114_114218


namespace max_product_of_two_integers_whose_sum_is_300_l114_114174

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l114_114174


namespace find_a_value_l114_114718

theorem find_a_value (a x y : ℝ) (h1 : x = 4) (h2 : y = 5) (h3 : a * x - 2 * y = 2) : a = 3 :=
by
  sorry

end find_a_value_l114_114718


namespace min_value_expression_l114_114764

theorem min_value_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 4) :
  ∃ c : ℝ, (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → x * y * z = 4 → 
  (2 * (x / y) + 3 * (y / z) + 4 * (z / x)) ≥ c) ∧ c = 6 :=
by
  sorry

end min_value_expression_l114_114764


namespace part1_part2_part3_l114_114642

variables (x y : ℝ)

noncomputable def A := 3*x^2 - x + 2*y - 4*x*y
noncomputable def B := x^2 - 2*x - y + x*y - 5

-- Part (1)
theorem part1 : A x y - 3*B x y = 5*x + 5*y - 7*x*y + 15 :=
by
  sorry

-- Part (2)
theorem part2 (h : (x + y - 4/5)^2 + |x * y + 1| = 0) : A x y - 3*B x y = 26 :=
by
  have h1 : x + y = 4/5 := sorry
  have h2 : x * y = -1 := sorry
  rw [h1, h2]
  sorry

-- Part (3)
theorem part3 (h : ∀ y, A x y - 3*B x y = (A x 0 - 3*B x 0)) : x = 5/7 :=
by
  sorry

end part1_part2_part3_l114_114642


namespace zero_points_of_function_l114_114596

theorem zero_points_of_function : 
  (∃ x y : ℝ, y = x - 4 / x ∧ y = 0) → (∃! x : ℝ, x = -2 ∨ x = 2) :=
by
  sorry

end zero_points_of_function_l114_114596


namespace slope_of_line_MF_is_sqrt3_l114_114693

theorem slope_of_line_MF_is_sqrt3 {p : ℝ} (hp : p > 0) :
  let F := (p / 2, 0),
      Mx := 3 * p / 2,
      My_pos := real.sqrt (3 * p^2),
      My_neg := -real.sqrt (3 * p^2),
      dist_MF := 2 * p
  in (Mx, My_pos) ∈ { (x, y) | y^2 = 2 * p * x } → 
     (Mx, My_neg) ∈ { (x, y) | y^2 = 2 * p * x } →
     (real.sqrt (3 * p^2)) / (Mx - (p / 2)) = real.sqrt 3 ∧
     (-real.sqrt (3 * p^2)) / (Mx - (p / 2)) = -real.sqrt 3 :=
by {
  sorry
}

end slope_of_line_MF_is_sqrt3_l114_114693


namespace max_product_of_sum_300_l114_114211

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l114_114211


namespace range_of_t_l114_114680

theorem range_of_t (t : ℝ) (h : ∃ x : ℝ, x ∈ Set.Iic t ∧ (x^2 - 4*x + t ≤ 0)) : 0 ≤ t ∧ t ≤ 4 :=
sorry

end range_of_t_l114_114680


namespace other_root_of_quadratic_l114_114722

theorem other_root_of_quadratic (a b : ℝ) (h₀ : a ≠ 0) (h₁ : ∃ x : ℝ, (a * x ^ 2 = b) ∧ (x = 2)) : 
  ∃ m : ℝ, (a * m ^ 2 = b) ∧ (m = -2) := 
sorry

end other_root_of_quadratic_l114_114722


namespace max_product_two_integers_sum_300_l114_114161

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l114_114161


namespace prob_sqrt_less_than_nine_l114_114481

/-- The probability that the square root of a randomly selected 
two-digit whole number is less than nine is 71/90. -/
theorem prob_sqrt_less_than_nine : (let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99};
                                     let A := {n : ℕ | 10 ≤ n ∧ n < 81};
                                     (A.card / S.card : ℚ) = 71 / 90) :=
by
  sorry

end prob_sqrt_less_than_nine_l114_114481


namespace greatest_product_l114_114288

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l114_114288


namespace number_division_l114_114867

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end number_division_l114_114867


namespace sum_of_35_consecutive_squares_div_by_35_l114_114787

def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_35_consecutive_squares_div_by_35 (n : ℕ) :
  (sum_of_squares (n + 35) - sum_of_squares n) % 35 = 0 :=
by
  sorry

end sum_of_35_consecutive_squares_div_by_35_l114_114787


namespace probability_sqrt_less_than_nine_l114_114338

/-- Define the set of two-digit integers --/
def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Define the condition that the square root of the number is less than 9 --/
def sqrt_less_than_nine (n : Nat) : Prop := n < 81

/-- The number of integers from 10 to 80 --/
lemma count_satisfying_sqrt (n : Nat) : Prop :=
  is_two_digit n ∧ sqrt_less_than_nine n → n < 81

/-- Total number of two-digit integers --/
lemma count_two_digit_total (n : Nat) : Prop := is_two_digit n 

/-- The probability that a randomly selected two-digit integer's square root is less than 9. --/
theorem probability_sqrt_less_than_nine : 
  (∃ n, count_satisfying_sqrt n) / (∃ n, count_two_digit_total n) = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114338


namespace greatest_product_of_sum_eq_300_l114_114107

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l114_114107


namespace greatest_product_two_ints_sum_300_l114_114266

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l114_114266


namespace prob_sqrt_less_than_nine_l114_114484

/-- The probability that the square root of a randomly selected 
two-digit whole number is less than nine is 71/90. -/
theorem prob_sqrt_less_than_nine : (let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99};
                                     let A := {n : ℕ | 10 ≤ n ∧ n < 81};
                                     (A.card / S.card : ℚ) = 71 / 90) :=
by
  sorry

end prob_sqrt_less_than_nine_l114_114484


namespace exists_polynomial_sequence_no_polynomial_sequence_l114_114536

-- Definition of the function using the form f_i(x) = f_j(x) * f_k(x) + f_j(x) + f_k(x) + 1
def varphi (x y : ℝ) : ℝ := x * y + x + y + 1

-- Defining the sequence of polynomials
noncomputable def polynomial_sequence : ℕ → ℝ → ℝ
| 0       , x => x
| (n + 1) , x =>
  let f_j := polynomial_sequence n x
  let f_k := polynomial_sequence n x
  varphi f_j f_k

-- The goal to prove the polynomial
theorem exists_polynomial_sequence (n : ℕ) (x : ℝ) (H : n = 1982) :
  polynomial_sequence n x = (list.finRange 1983).sum (λ k, x ^ k) :=
  sorry

-- Definition of the function using the form f_i(x) = f_j(x) * f_k(x) + f_j(x) + f_k(x)
def psi (x y : ℝ) : ℝ := x * y + x + y

-- Proving it is impossible to compute the desired polynomial using psi
theorem no_polynomial_sequence (n : ℕ) (x : ℝ) (H : n = 1982) :
  ∀ f : ℕ → ℝ → ℝ, (f 0 x = x) → (∀ i < n + 1, ∃ j k < i, f (i+1) x = psi (f j x) (f k x)) →
  f (n + 1) x ≠ (list.finRange 1983).sum (λ k, x ^ k) :=
  sorry

end exists_polynomial_sequence_no_polynomial_sequence_l114_114536


namespace greatest_product_sum_300_l114_114132

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l114_114132


namespace solve_for_y_l114_114877

theorem solve_for_y (y : ℕ) : 9^y = 3^16 → y = 8 := by
  assume h : 9^y = 3^16
  have h1 : (3^2)^y = 3^16, by rw [pow_mul]
  rw [←h1] at h
  have h2 : 3^(2 * y) = 3^16, by rw [pow_mul]
  have h3 : 2 * y = 16, from by sorry
  exact (nat.mul_right_inj 2) h3 sorry

end solve_for_y_l114_114877


namespace average_speed_swim_run_l114_114878

/-- Tabby's average speed for swimming and running, given she swims at 1 mile per hour 
    and runs at 8 miles per hour, and assuming she covers the same distance in both events, 
    is 16/9 miles per hour. -/
theorem average_speed_swim_run (d : ℝ) (h1 : d > 0) :
  (2 * d) / (d + d / 8) = 16 / 9 :=
by
  have t_swim := d / 1
  have t_run := d / 8
  have dist_total := 2 * d
  have time_total := d + d / 8
  calc (2 * d) / (d + d / 8) = 2 * d / (9 * d / 8) : by rw [time_total]
                          ... = (2 * d) * (8 / (9 * d)) : by rw [div_mul_div]
                          ... = (2 * 8 * d) / (9 * d)   : by ring
                          ... = 16 / 9                : by rw [mul_div_cancel_left _ (ne_of_gt h1)]

end average_speed_swim_run_l114_114878


namespace probability_straight_flush_l114_114843

theorem probability_straight_flush (num_total_hands : ℕ) (num_straight_flushes : ℕ) : 
  num_total_hands = (nat.choose 52 5) → 
  num_straight_flushes = 40 → 
  (num_straight_flushes : ℚ) / (num_total_hands : ℚ) = 1 / 64974 :=
by
  intros h_total_hands h_straight_flushes
  rw [h_total_hands, h_straight_flushes]
  norm_num
  sorry

end probability_straight_flush_l114_114843


namespace lines_not_concurrent_l114_114753

noncomputable section

variable {A B C D E F P Q R : Type}

-- Definitions based on conditions
variable [ht : ∃D, ∃E, ∃F (D,E,F) ∈ feet altitudes on triangle ABC]
variable hP : is_foot_of_altitude P C F B
variable hQ : is_foot_of_altitude Q A D C
variable hR : is_foot_of_altitude R B E A

theorem lines_not_concurrent 
  (h₁ : foot_of_altitude D B C A)
  (h₂ : foot_of_altitude E C A B)
  (h₃ : foot_of_altitude F A B C)
  (h₄ : ∀ P, foot_of_altitude P C F B)
  (h₅ : ∀ Q, foot_of_altitude Q A D C)
  (h₆ : ∀ R, foot_of_altitude R B E A) :
  ¬ concurrent {A P, B Q, C R} := sorry

end lines_not_concurrent_l114_114753


namespace alexis_suit_coat_expense_l114_114567

theorem alexis_suit_coat_expense :
  let budget := 200
  let shirt_cost := 30
  let pants_cost := 46
  let socks_cost := 11
  let belt_cost := 18
  let shoes_cost := 41
  let leftover := 16
  let other_expenses := shirt_cost + pants_cost + socks_cost + belt_cost + shoes_cost
  budget - leftover - other_expenses = 38 := 
by
  let budget := 200
  let shirt_cost := 30
  let pants_cost := 46
  let socks_cost := 11
  let belt_cost := 18
  let shoes_cost := 41
  let leftover := 16
  let other_expenses := shirt_cost + pants_cost + socks_cost + belt_cost + shoes_cost
  sorry

end alexis_suit_coat_expense_l114_114567


namespace inscribed_circle_isosceles_right_triangle_radius_6_length_BC_l114_114077

theorem inscribed_circle_isosceles_right_triangle_radius_6_length_BC :
  ∀ (A B C O : Point) (r : ℝ),
    triangle_is_isosceles A B C ∧
    angle_at_vertex A B C = 45 ∧
    inscribed_circle_center O A B C ∧
    inscribed_circle_radius O A B C = 6 →
    length_BC A B C = 12 * sqrt 2 :=
begin
  -- This is where the proof steps would go
  sorry
end

end inscribed_circle_isosceles_right_triangle_radius_6_length_BC_l114_114077


namespace imaginary_part_z_l114_114676

theorem imaginary_part_z : ∀ z : ℂ, (1 + (1 + 2 * z) * complex.i = 0) → z.im = 1 / 2 :=
by
  intro z h
  sorry

end imaginary_part_z_l114_114676


namespace aleena_vs_bob_distance_l114_114062

theorem aleena_vs_bob_distance :
  let AleenaDistance := 75
  let BobDistance := 60
  AleenaDistance - BobDistance = 15 :=
by
  let AleenaDistance := 75
  let BobDistance := 60
  show AleenaDistance - BobDistance = 15
  sorry

end aleena_vs_bob_distance_l114_114062


namespace probability_sqrt_lt_nine_l114_114405

theorem probability_sqrt_lt_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  let probability := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  probability = 71 / 90 :=
by
  let two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n : ℕ | 10 ≤ n ∧ n < 81}
  have h1 : two_digit_numbers.card = 90 := sorry
  have h2 : valid_numbers.card = 71 := sorry
  let probability : ℚ := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  rw [h1, h2]
  simp
  norm_num

end probability_sqrt_lt_nine_l114_114405


namespace cost_of_flowers_l114_114522

theorem cost_of_flowers 
  (interval : ℕ) (perimeter : ℕ) (cost_per_flower : ℕ)
  (h_interval : interval = 30)
  (h_perimeter : perimeter = 1500)
  (h_cost : cost_per_flower = 5000) :
  (perimeter / interval) * cost_per_flower = 250000 :=
by
  sorry

end cost_of_flowers_l114_114522


namespace diagonal_possible_lengths_l114_114937

theorem diagonal_possible_lengths (a b c d : ℕ) (h1 : a = 7) (h2 : b = 9) 
  (h3 : c = 14) (h4 : d = 10) : 
  ∃ n : ℕ, (n = (finset.Ico 6 17).card ∧ n = 11) := 
by {
  sorry
}

end diagonal_possible_lengths_l114_114937


namespace polygon_perimeter_l114_114904

theorem polygon_perimeter (side_length : ℝ) (ext_angle_deg : ℝ) (n : ℕ) (h1 : side_length = 8) 
  (h2 : ext_angle_deg = 90) (h3 : ext_angle_deg = 360 / n) : 
  4 * side_length = 32 := 
  by 
    sorry

end polygon_perimeter_l114_114904


namespace eva_marks_l114_114606

theorem eva_marks
  (M : ℕ)
  (maths_first_sem := M + 10)
  (arts_first_sem := 90 - 15)
  (science_first_sem := 90 - (1/3) * 90)
  (total_marks := maths_first_sem + arts_first_sem + science_first_sem + M + 90 + 90):
  total_marks = 485 → M = 80 :=
by
  let M : ℕ := 80
  have maths_first_sem := M + 10
  have arts_first_sem := 90 - 15
  have science_first_sem := 90 - (1/3) * 90
  have total_marks := maths_first_sem + arts_first_sem + science_first_sem + M + 90 + 90
  show total_marks = 485 → M = 80
  sorry

end eva_marks_l114_114606


namespace max_product_of_two_integers_whose_sum_is_300_l114_114168

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l114_114168


namespace point_in_first_quadrant_l114_114674

-- Define the inverse proportion function
def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

-- Given conditions
variable (m : ℝ)
variable (hx : m = inverse_proportion 6 3)

-- Prove that the point (3, m) lies in the first quadrant
theorem point_in_first_quadrant (hx : m = 2) : (3 > 0) ∧ (m > 0) :=
by {
  simp [hx], -- Simplify hx
  split,     -- Split into two separate goals
  exact zero_lt_three, -- Prove 3 > 0
  linarith, -- Prove m > 0 from hx
}

end point_in_first_quadrant_l114_114674


namespace blue_tshirts_in_pack_l114_114025

theorem blue_tshirts_in_pack
  (packs_white : ℕ := 2) 
  (white_per_pack : ℕ := 5) 
  (packs_blue : ℕ := 4)
  (cost_per_tshirt : ℕ := 3)
  (total_cost : ℕ := 66)
  (B : ℕ := 3) :
  (packs_white * white_per_pack * cost_per_tshirt) + (packs_blue * B * cost_per_tshirt) = total_cost := 
by
  sorry

end blue_tshirts_in_pack_l114_114025


namespace exists_additive_function_close_to_f_l114_114802

variable (f : ℝ → ℝ)

theorem exists_additive_function_close_to_f (h : ∀ x y : ℝ, |f (x + y) - f x - f y| ≤ 1) :
  ∃ g : ℝ → ℝ, (∀ x : ℝ, |f x - g x| ≤ 1) ∧ (∀ x y : ℝ, g (x + y) = g x + g y) := by
  sorry

end exists_additive_function_close_to_f_l114_114802


namespace count_semiprimes_expressed_as_x_cubed_minus_1_l114_114906

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_semiprime (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p * q = n

theorem count_semiprimes_expressed_as_x_cubed_minus_1 :
  (∃ S : Finset ℕ, 
    S.card = 4 ∧ 
    ∀ n ∈ S, n < 2018 ∧ 
    ∃ x : ℕ, x > 0 ∧ x^3 - 1 = n ∧ is_semiprime n) :=
sorry

end count_semiprimes_expressed_as_x_cubed_minus_1_l114_114906


namespace regular_polygon_perimeter_l114_114903

theorem regular_polygon_perimeter
  (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ)
  (h1 : side_length = 8)
  (h2 : exterior_angle = 90)
  (h3 : n = 360 / exterior_angle) :
  n * side_length = 32 := by
  sorry

end regular_polygon_perimeter_l114_114903


namespace max_product_sum_300_l114_114311

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l114_114311


namespace ratio_folded_paper_l114_114560

noncomputable def fold_paper_ratio (A : ℝ) (B : ℝ) (sqrt_2 : ℝ) (sqrt_3 : ℝ) (sqrt_6 : ℝ) : ℝ :=
  let section_width := sqrt_2 / 4
  let base_length := Real.sqrt(1 + (sqrt_2 / 2) ^ 2)
  let height := 1 / 2
  let triangle_area := 1 / 2 * base_length * height
  let folded_area := sqrt_2 - triangle_area
  B / A

theorem ratio_folded_paper (l : ℝ) (sqrt_2 sqrt_3 sqrt_6 : ℝ) (A B : ℝ) :
  A = sqrt_2 * l * l →
  l = 1 →
  A = sqrt_2 →
  let section_width := sqrt_2 / 4 in
  let base_length := Real.sqrt(1 + (sqrt_2 / 2) ^ 2) in
  let height := 1 / 2 in
  let triangle_area := 1 / 2 * base_length * height in
  let folded_area := sqrt_2 - triangle_area in
  B = folded_area →
  B / A = (16 - sqrt_6) / 16 := 
begin
  intros h1 h2 h3 section_width base_length height triangle_area folded_area hB,
  sorry
end

end ratio_folded_paper_l114_114560


namespace exists_k_inequality_l114_114765

theorem exists_k_inequality 
    (n : ℕ) 
    (a : Fin n → ℝ) 
    (b : Fin n → Complex) : 
    ∃ (k : Fin n), 
    (∑ i, |a i - a k|) ≤ (∑ i, Complex.abs (b i - a k)) := by
  sorry

end exists_k_inequality_l114_114765


namespace jean_burglary_charges_l114_114747

theorem jean_burglary_charges:
  ∃ (B : ℕ), 
  (let L := 6 * B in 
   3 * 36 + 18 * B + 6 * L = 216) 
  ∧ B = 2 :=
by
  sorry

end jean_burglary_charges_l114_114747


namespace max_product_300_l114_114139

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l114_114139


namespace max_product_sum_300_l114_114303

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l114_114303


namespace inequality_solution_l114_114940

theorem inequality_solution (x : ℝ) (h : 4 ≤ |x + 2| ∧ |x + 2| ≤ 8) :
  (-10 : ℝ) ≤ x ∧ x ≤ -6 ∨ (2 : ℝ) ≤ x ∧ x ≤ 6 :=
sorry

end inequality_solution_l114_114940


namespace grade_assignment_ways_l114_114898

theorem grade_assignment_ways (n_students : ℕ) (n_grades : ℕ) (h_students : n_students = 12) (h_grades : n_grades = 4) :
  (n_grades ^ n_students) = 16777216 := by
  rw [h_students, h_grades]
  rfl

end grade_assignment_ways_l114_114898


namespace minimun_receipts_to_buy_chocolates_l114_114039

def rms (x1 x2 x3 : ℝ) : ℝ :=
  Real.sqrt ((x1^2 + x2^2 + x3^2) / 3)

def canBuyWithoutVerification (S max_cost : ℝ) : Prop :=
  max_cost <= 3 * S

theorem minimun_receipts_to_buy_chocolates 
  (cost_per_chocolate : ℝ) (num_chocolates : ℕ)
  (prev1 prev2 prev3 : ℝ) (prev_purchases : prev1 = 300 ∧ prev2 = 300 ∧ prev3 = 300) :
  num_chocolates = 40 ∧ cost_per_chocolate = 50 →
  ∃ (num_receipts : ℕ), num_receipts = 2 :=
by
  sorry

end minimun_receipts_to_buy_chocolates_l114_114039


namespace probability_sqrt_lt_nine_l114_114407

theorem probability_sqrt_lt_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  let probability := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  probability = 71 / 90 :=
by
  let two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n : ℕ | 10 ≤ n ∧ n < 81}
  have h1 : two_digit_numbers.card = 90 := sorry
  have h2 : valid_numbers.card = 71 := sorry
  let probability : ℚ := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  rw [h1, h2]
  simp
  norm_num

end probability_sqrt_lt_nine_l114_114407


namespace projection_bisector_intersection_l114_114021

variable {A B C D P Q R : Type*}
variable [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] 
variable [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]

-- Define Cyclic Quadrilateral and Projections
def CyclicQuadrilateral (A B C D : Point) : Prop := 
  co-circular A B C D

def OrthogonalProjection (D : Point) (l : Line) (P : Point) : Prop :=
  is_orthogonal_proy D l P

-- Main theorem statement
theorem projection_bisector_intersection
  (ABCD_cyclic : CyclicQuadrilateral A B C D)
  (proj_P: OrthogonalProjection D (line_through B C) P) 
  (proj_Q: OrthogonalProjection D (line_through C A) Q) 
  (proj_R: OrthogonalProjection D (line_through A B) R) :
  (dist P Q = dist Q R) ↔ (exists X : Point, is_on_segment X A C ∧ is_angle_bisector (∠ABC) (X) ∧ is_angle_bisector (∠ADC) (X)) :=
sorry

end projection_bisector_intersection_l114_114021


namespace max_intersections_circle_quadrilateral_l114_114848

theorem max_intersections_circle_quadrilateral (circle : Type) (quadrilateral : Type) 
  (intersects : circle → quadrilateral → ℕ) (h : ∀ (c : circle) (line_segment : Type), intersects c line_segment ≤ 2) :
  ∃ (q : quadrilateral), intersects circle quadrilateral = 8 :=
by
  sorry

end max_intersections_circle_quadrilateral_l114_114848


namespace sequence_general_formula_l114_114659

theorem sequence_general_formula (a : ℕ → ℕ) 
    (h₀ : a 1 = 3) 
    (h : ∀ n : ℕ, a (n + 1) = 2 * a n + 1) : 
    ∀ n : ℕ, a n = 2^(n+1) - 1 :=
by 
  sorry

end sequence_general_formula_l114_114659


namespace third_number_in_sequence_l114_114941

def arithmetic_segment_sequence : list ℕ :=
  [1, 1, 1, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3]

theorem third_number_in_sequence :
  arithmetic_segment_sequence.nth 2 = some 1 :=
by
  -- The proof part will be filled here
  sorry

end third_number_in_sequence_l114_114941


namespace solution_to_equation_l114_114960

noncomputable def set_of_solutions : set (ℝ × ℝ × ℝ × ℝ) :=
  { (1, 1, 1, 1), (-1, -1, -1, 3), (-1, -1, 3, -1), (-1, 3, -1, -1), (3, -1, -1, -1) }

theorem solution_to_equation (x1 x2 x3 x4 : ℝ) :
  (x₁ + x₂ * x₃ * x₄ = 2) ∧ (x₂ + x₃ * x₄ * x₁ = 2) ∧ (x₃ + x₄ * x₁ * x₂ = 2) ∧ (x₄ + x₁ * x₂ * x₃ = 2) →
  (x1, x2, x3, x4) ∈ set_of_solutions := by
  sorry

end solution_to_equation_l114_114960


namespace probability_sqrt_two_digit_lt_nine_correct_l114_114443

noncomputable def probability_sqrt_two_digit_lt_nine : ℚ :=
  let two_digit_integers := finset.Icc 10 99
  let satisfying_integers := two_digit_integers.filter (λ n => n < 81)
  let probability := (satisfying_integers.card : ℚ) / (two_digit_integers.card : ℚ)
  probability

theorem probability_sqrt_two_digit_lt_nine_correct :
  probability_sqrt_two_digit_lt_nine = 71 / 90 := by
  sorry

end probability_sqrt_two_digit_lt_nine_correct_l114_114443


namespace probability_sqrt_two_digit_lt_nine_correct_l114_114440

noncomputable def probability_sqrt_two_digit_lt_nine : ℚ :=
  let two_digit_integers := finset.Icc 10 99
  let satisfying_integers := two_digit_integers.filter (λ n => n < 81)
  let probability := (satisfying_integers.card : ℚ) / (two_digit_integers.card : ℚ)
  probability

theorem probability_sqrt_two_digit_lt_nine_correct :
  probability_sqrt_two_digit_lt_nine = 71 / 90 := by
  sorry

end probability_sqrt_two_digit_lt_nine_correct_l114_114440


namespace integral_bound_difference_l114_114766

theorem integral_bound_difference (f : ℝ → ℝ) (h_diff : Differentiable ℝ f)
  (h_f0 : f 0 = 0) (h_f1 : f 1 = 1)
  (h_f' : ∀ x, abs (deriv f x) ≤ 2) :
  ∃ a b : ℝ, (∀ y, ∫ x in 0..1, f x = y → y ∈ Ioo a b) ∧ (b - a = 3 / 4) :=
begin
  sorry
end

end integral_bound_difference_l114_114766


namespace probability_sqrt_lt_9_l114_114427

theorem probability_sqrt_lt_9 : 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 →
  ∃ p : ℚ, p = 71 / 90 ∧ 
  ∑ k in (Finset.range 100).filter (λ x, 10 ≤ x ∧ sqrt x < 9), 1 / 90 = p := 
sorry

end probability_sqrt_lt_9_l114_114427


namespace max_product_of_two_integers_whose_sum_is_300_l114_114169

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l114_114169


namespace stationery_store_loss_l114_114073

theorem stationery_store_loss :
  ∃ (x y : ℝ), (x * 1.2 = 60) ∧ (y * 0.8 = 60) ∧ ((x + y) - 120 = 5) :=
begin
  use [50, 75],
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
end

end stationery_store_loss_l114_114073


namespace probability_sqrt_lt_9_l114_114390

theorem probability_sqrt_lt_9 : 
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
in probability = 71 / 90 :=
by
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  sorry

end probability_sqrt_lt_9_l114_114390


namespace arrange_volunteers_for_tasks_l114_114920

-- Define the sets of volunteers and tasks
def volunteers : Fin 3 := Fin 3
def tasks : Fin 4 := Fin 4

-- Formalization of the problem
theorem arrange_volunteers_for_tasks :
  let ways_to_choose_tasks := Nat.choose 4 2 in
  let ways_to_arrange_volunteers := Nat.factorial 3 in
  ways_to_choose_tasks * ways_to_arrange_volunteers = 36 := by
  sorry

end arrange_volunteers_for_tasks_l114_114920


namespace probability_sqrt_less_than_nine_l114_114465

theorem probability_sqrt_less_than_nine :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let T := {n : ℕ | n ∈ S ∧ n < 81}
  (T.card : ℚ) / S.card = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l114_114465


namespace max_product_of_sum_300_l114_114215

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l114_114215


namespace strongest_signal_l114_114830

theorem strongest_signal (a b c d : ℤ) (ha : -50 = a) (hb : -60 = b) (hc : -70 = c) (hd : -80 = d) :
  (|a| < |b|) ∧ (|b| < |c|) ∧ (|c| < |d|) → a = -50 := 
begin
  intro h,
  cases h with hab hcd,
  cases hab with hab hbc,
  cases hcd with hbc hcd,
  exact ha,
end

end strongest_signal_l114_114830


namespace prob_sqrt_less_than_nine_l114_114472

/-- The probability that the square root of a randomly selected 
two-digit whole number is less than nine is 71/90. -/
theorem prob_sqrt_less_than_nine : (let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99};
                                     let A := {n : ℕ | 10 ≤ n ∧ n < 81};
                                     (A.card / S.card : ℚ) = 71 / 90) :=
by
  sorry

end prob_sqrt_less_than_nine_l114_114472


namespace max_product_of_sum_300_l114_114220

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l114_114220


namespace hyperbola_eccentricity_l114_114067

theorem hyperbola_eccentricity (p a b : ℝ) (h_p : p > 0) (h_a : a > 0) (h_b : b > 0)
(h_focus : real.sqrt (a^2 + b^2) = p / 2) (M : ℝ × ℝ)
(h_M_intersection : M.1 ^ 2 = 2 * p * M.1)
(h_M_focus : dist (M, (p / 2, 0)) = p) :
    real.sqrt (a^2 + b^2) / a = 1 + real.sqrt 2 := by
  sorry

end hyperbola_eccentricity_l114_114067


namespace coefficient_x2_expansion_l114_114057

-- Original terms and coefficients definitions
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- Conditions
def term_in_expansion (r : ℕ) : ℤ :=
  (-1 : ℤ)^r * (binomial_coefficient 7 r) * x^r

def combined_term := (1 + x⁻¹) * (1 - x)^7

-- Proof problem statement
theorem coefficient_x2_expansion : 
  (∑ r in range(8), term_in_expansion r).coeff x^2 = -14 :=
by
  sorry

end coefficient_x2_expansion_l114_114057


namespace system1_solution_system2_solution_l114_114044

-- For Question 1

theorem system1_solution (x y : ℝ) :
  (2 * x - y = 5) ∧ (7 * x - 3 * y = 20) ↔ (x = 5 ∧ y = 5) := 
sorry

-- For Question 2

theorem system2_solution (x y : ℝ) :
  (3 * (x + y) - 4 * (x - y) = 16) ∧ ((x + y)/2 + (x - y)/6 = 1) ↔ (x = 1/3 ∧ y = 7/3) := 
sorry

end system1_solution_system2_solution_l114_114044


namespace circumscribed_circle_perpendicular_to_circumcircle_l114_114822

-- Problem Statement: Given the following setup, prove the stated result
theorem circumscribed_circle_perpendicular_to_circumcircle
  (A B C D I P Q : Point)
  (Omega omega : Circle)
  (h1 : is_circumscribed_quadrilateral ABCD I)
  (h2 : is_in_circle ABCD Omega)
  (h3 : intersects (line_through A B) (line_through C D) P)
  (h4 : intersects (line_through B C) (line_through A D) Q)
  (h5 : circumcircle PIQ = omega) : perpendicular_circles omega Omega := 
sorry

end circumscribed_circle_perpendicular_to_circumcircle_l114_114822


namespace percentage_managers_decrease_l114_114837

theorem percentage_managers_decrease
  (employees : ℕ)
  (initial_percentage : ℝ)
  (managers_leave : ℝ)
  (new_percentage : ℝ)
  (h1 : employees = 200)
  (h2 : initial_percentage = 99)
  (h3 : managers_leave = 100)
  (h4 : new_percentage = 98) :
  ((initial_percentage / 100 * employees - managers_leave) / (employees - managers_leave) * 100 = new_percentage) :=
by
  -- To be proven
  sorry

end percentage_managers_decrease_l114_114837


namespace probability_sqrt_lt_9_of_two_digit_l114_114369

-- Define the set of two-digit whole numbers
def two_digit_whole_numbers : set ℕ := {n | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate that checks if the square root of a number is less than 9
def sqrt_lt_9 (n : ℕ) : Prop := (n : ℝ)^2 < (9 : ℝ)^2

-- Calculate the probability
theorem probability_sqrt_lt_9_of_two_digit :
  let eligible_numbers := { n ∈ two_digit_whole_numbers | sqrt_lt_9 n } in
  (eligible_numbers.to_finset.card : ℚ) / (two_digit_whole_numbers.to_finset.card : ℚ) =
  71 / 90 :=
by
  sorry

end probability_sqrt_lt_9_of_two_digit_l114_114369


namespace find_smallest_c_plus_d_l114_114011

noncomputable def smallest_c_plus_d (c d : ℝ) :=
  c + d

theorem find_smallest_c_plus_d (c d : ℝ) (hc : 0 < c) (hd : 0 < d)
  (h1 : c ^ 2 ≥ 12 * d)
  (h2 : 9 * d ^ 2 ≥ 4 * c) :
  smallest_c_plus_d c d = 16 / 3 :=
by
  sorry

end find_smallest_c_plus_d_l114_114011


namespace probability_of_sqrt_lt_9_l114_114358

-- Define the set of two-digit whole numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the subset of numbers for which the square root is less than 9
def valid_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 80}

-- Define the probability calculation
noncomputable def probability_sqrt_lt_9 := (valid_numbers.to_finset.card : ℝ) / (two_digit_numbers.to_finset.card : ℝ)

-- The statement we aim to prove
theorem probability_of_sqrt_lt_9 : probability_sqrt_lt_9 = 71 / 90 := 
sorry

end probability_of_sqrt_lt_9_l114_114358


namespace gcd_lcm_sum_l114_114514

theorem gcd_lcm_sum :
  Nat.gcd 44 64 + Nat.lcm 48 18 = 148 := 
by
  sorry

end gcd_lcm_sum_l114_114514


namespace greatest_product_sum_300_l114_114201

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l114_114201


namespace incorrect_parallel_statement_l114_114014

variables {m n : Line} {α β : Plane}

-- Definitions for the conditions
def perp_to (l : Line) (p : Plane) : Prop := l ⟂ p
def parallel_to (l : Line) (p : Plane) : Prop := l ∥ p
def planes_intersect (p q : Plane) (l : Line) : Prop := p ∩ q = l

-- The statement we need to prove (that option D is false)
theorem incorrect_parallel_statement (h₁ : parallel_to m α) (h₂ : planes_intersect α β n) : ¬parallel_to m n := 
sorry

end incorrect_parallel_statement_l114_114014


namespace total_cube_volume_l114_114587

theorem total_cube_volume 
  (carl_cubes : ℕ)
  (carl_cube_side : ℕ)
  (kate_cubes : ℕ)
  (kate_cube_side : ℕ)
  (hcarl : carl_cubes = 4)
  (hcarl_side : carl_cube_side = 3)
  (hkate : kate_cubes = 6)
  (hkate_side : kate_cube_side = 4) :
  (carl_cubes * carl_cube_side ^ 3) + (kate_cubes * kate_cube_side ^ 3) = 492 :=
by
  sorry

end total_cube_volume_l114_114587


namespace planted_fraction_correct_l114_114956
noncomputable def right_triangle_leg1 : ℕ := 5
noncomputable def right_triangle_leg2 : ℕ := 12
noncomputable def hypotenuse : ℕ := 13
noncomputable def square_to_hypotenuse_distance : ℕ := 3
noncomputable def planted_fraction : ℚ := 792 / 845

theorem planted_fraction_correct :
  -- Conditions
  hypotenuse = real.sqrt (right_triangle_leg1 ^ 2 + right_triangle_leg2 ^ 2) ∧
  square_to_hypotenuse_distance = 3 ∧
  -- Conclusion
  planted_fraction = 792 / 845 :=
sorry

end planted_fraction_correct_l114_114956


namespace probability_sqrt_lt_9_l114_114424

theorem probability_sqrt_lt_9 : 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 →
  ∃ p : ℚ, p = 71 / 90 ∧ 
  ∑ k in (Finset.range 100).filter (λ x, 10 ≤ x ∧ sqrt x < 9), 1 / 90 = p := 
sorry

end probability_sqrt_lt_9_l114_114424


namespace smallest_four_digit_divisible_by_33_l114_114853

theorem smallest_four_digit_divisible_by_33 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 33 = 0 ∧ n = 1023 := by 
  sorry

end smallest_four_digit_divisible_by_33_l114_114853


namespace quadratic_sum_eq_eight_l114_114076

theorem quadratic_sum_eq_eight : 
  ∀ x : ℝ, x^2 - 6 * x - 22 = 2 * x + 18 → (let a := 1 in let b := -8 in let c := -40 in (∃ x1 x2 : ℝ, x1 * a + x2 * a = b * (-1) - -8) ∧ x1 + x2 = 8) :=
by
  sorry

end quadratic_sum_eq_eight_l114_114076


namespace greatest_product_sum_300_l114_114133

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l114_114133


namespace count_of_sets_l114_114986

noncomputable def countSets (n : ℕ): ℕ :=
(2 * n)! * 2^(n * n)

theorem count_of_sets (n : ℕ):
  ∀ (S : Fin (n + 1) × Fin (n + 1) → Finset (Fin (2 * n))),
    (∀ i j, S (i, j).card = i + j) → 
    (∀ i j k t, 0 ≤ i → i ≤ k → k ≤ n → 0 ≤ j → j ≤ t → t ≤ n → S (i, j) ⊆ S (k, t)) →
    countSets n = (2 * n)! * 2^(n * n) := sorry

end count_of_sets_l114_114986


namespace max_product_of_sum_300_l114_114272

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l114_114272


namespace max_product_two_integers_sum_300_l114_114150

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l114_114150


namespace Petya_wrong_example_l114_114781

def a := 8
def b := 128

theorem Petya_wrong_example : (a^7 ∣ b^3) ∧ ¬ (a^2 ∣ b) :=
by {
  -- Prove the divisibility conditions and the counterexample
  sorry
}

end Petya_wrong_example_l114_114781


namespace greatest_product_l114_114297

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l114_114297


namespace probability_sqrt_lt_nine_l114_114410

theorem probability_sqrt_lt_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  let probability := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  probability = 71 / 90 :=
by
  let two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n : ℕ | 10 ≤ n ∧ n < 81}
  have h1 : two_digit_numbers.card = 90 := sorry
  have h2 : valid_numbers.card = 71 := sorry
  let probability : ℚ := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  rw [h1, h2]
  simp
  norm_num

end probability_sqrt_lt_nine_l114_114410


namespace sin_of_angle_A_l114_114735

theorem sin_of_angle_A (A B C : ℝ) (hC : C = 90) (hB : cos B = 1 / 2) : sin A = 1 / 2 := 
  sorry

end sin_of_angle_A_l114_114735


namespace find_number_l114_114864

theorem find_number (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 := 
sorry

end find_number_l114_114864


namespace arithmetic_series_sum_l114_114632

theorem arithmetic_series_sum : 
  let a := -41
  let d := 2
  let n := 22
  let l := 1
  let Sn := n * (a + l) / 2
  a = -41 ∧ d = 2 ∧ l = 1 ∧ n = 22 → Sn = -440 :=
by 
  intros a d n l Sn h
  sorry

end arithmetic_series_sum_l114_114632


namespace pastries_count_l114_114922

def C : ℕ := 19
def P : ℕ := C + 112

theorem pastries_count : P = 131 := by
  -- P = 19 + 112
  -- P = 131
  sorry

end pastries_count_l114_114922


namespace probability_all_same_color_l114_114542

theorem probability_all_same_color :
  let total_marbles := 30
  let red_marbles := 6
  let white_marbles := 7
  let blue_marbles := 8
  let green_marbles := 9
  let successful_red := Nat.choose red_marbles 4
  let successful_white := Nat.choose white_marbles 4
  let successful_blue := Nat.choose blue_marbles 4
  let successful_green := Nat.choose green_marbles 4
  let total_ways := Nat.choose total_marbles 4
  let probability := (successful_red + successful_white + successful_blue + successful_green) / total_ways
in probability = 82 / 9135 := sorry

end probability_all_same_color_l114_114542


namespace initial_salt_concentration_proof_l114_114908

noncomputable def initial_salt_concentration (x : ℝ) (S : ℝ) :=
  S / x

theorem initial_salt_concentration_proof (x : ℝ) (S : ℝ) (V_final : ℝ) (C_final : ℝ) :
  x = 104.99999999999997 →
  V_final = (3/4 * x) + 7 + 14 →
  C_final = 1/3 →
  S + 14 = C_final * V_final →
  initial_salt_concentration x S ≈ 0.1833 :=
by
  intro h₁ h₂ h₃ h₄
  -- The following lines are placeholders for intermediate calculations
  have : V_final = (3/4 * x) + 7 + 14 := by sorry
  have : S = 1/3 * V_final - 14 := by sorry
  have : initial_salt_concentration x S ≈ 19.25 / 105 := by sorry
  sorry    -- Final conclusion proof placeholder

end initial_salt_concentration_proof_l114_114908


namespace probability_sqrt_less_than_nine_is_correct_l114_114374

def probability_sqrt_less_than_nine : ℚ :=
  let total_two_digit_numbers := 99 - 10 + 1 in
  let satisfying_numbers := 80 - 10 + 1 in
  let probability := (satisfying_numbers : ℚ) / (total_two_digit_numbers : ℚ) in
  probability

theorem probability_sqrt_less_than_nine_is_correct :
  probability_sqrt_less_than_nine = 71 / 90 :=
by
  -- proof here
  sorry

end probability_sqrt_less_than_nine_is_correct_l114_114374


namespace max_intersections_quadrilateral_l114_114847

-- Define intersection properties
def max_intersections_side : ℕ := 2
def sides_of_quadrilateral : ℕ := 4

theorem max_intersections_quadrilateral : 
  (max_intersections_side * sides_of_quadrilateral) = 8 :=
by 
  -- The proof goes here
  sorry

end max_intersections_quadrilateral_l114_114847


namespace max_product_sum_300_l114_114305

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l114_114305


namespace greatest_product_sum_300_l114_114241

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l114_114241


namespace fox_catches_mole_l114_114071

-- Definition to represent the mounds and movements
def mound : Type := Fin 100

-- Defining the initial conditions
constant mole_initial_position : mound
constant fox_position : mound
constant mole_moves : mound → mound
constant fox_moves : mound → mound

-- Axiom to represent mole moves to a neighboring mound
axiom mole_neighboring_move : ∀ (mole : mound), (mole_moves mole = mole + 1) ∨ (mole_moves mole = mole - 1)

-- Axiom for fox can move to any mound each minute
axiom fox_any_move : ∀ (fox : mound), ∃ (next_fox : mound), fox_moves fox = next_fox

-- Goal: Prove the fox can catch the mole within 200 moves
theorem fox_catches_mole : ∃ (steps : ℕ) (fox_position mole_position : mound), 
                           (steps ≤ 200 ∧ fox_position = mole_position) :=
begin
  sorry
end

end fox_catches_mole_l114_114071


namespace max_product_sum_300_l114_114307

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l114_114307


namespace area_of_triangle_ABC_l114_114768

theorem area_of_triangle_ABC :
  let O := (0 : ℝ, 0, 0)
  let A := (real.qrt 48, 0, 0)
  let B := (0, real.sqrt 27, 0)
  let C := (0, 0, real.sqrt 12)
  angle_of_points (B, A, C) = π / 4 →
  area_of_triangle A B C = real.sqrt 117 :=
by
  sorry

end area_of_triangle_ABC_l114_114768
