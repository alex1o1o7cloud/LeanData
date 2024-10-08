import Mathlib

namespace time_after_1456_minutes_l226_226269

noncomputable def hours_in_minutes := 1456 / 60
noncomputable def minutes_remainder := 1456 % 60

def current_time : Nat := 6 * 60  -- 6:00 a.m. in minutes
def added_time : Nat := current_time + 1456

def six_sixteen_am : Nat := (6 * 60) + 16  -- 6:16 a.m. in minutes the next day

theorem time_after_1456_minutes : added_time % (24 * 60) = six_sixteen_am :=
by
  sorry

end time_after_1456_minutes_l226_226269


namespace line_contains_point_l226_226247

theorem line_contains_point (k : ℝ) (x : ℝ) (y : ℝ) (H : 2 - 2 * k * x = -4 * y) : k = -1 ↔ (x = 3 ∧ y = -2) :=
by
  sorry

end line_contains_point_l226_226247


namespace fixed_point_for_all_parabolas_l226_226183

theorem fixed_point_for_all_parabolas : ∃ (x y : ℝ), (∀ t : ℝ, y = 4 * x^2 + 2 * t * x - 3 * t) ∧ x = 1 ∧ y = 4 :=
by 
  sorry

end fixed_point_for_all_parabolas_l226_226183


namespace sphere_surface_area_ratio_l226_226293

axiom prism_has_circumscribed_sphere : Prop
axiom prism_has_inscribed_sphere : Prop

theorem sphere_surface_area_ratio 
  (h1 : prism_has_circumscribed_sphere)
  (h2 : prism_has_inscribed_sphere) : 
  ratio_surface_area_of_circumscribed_to_inscribed_sphere = 5 :=
sorry

end sphere_surface_area_ratio_l226_226293


namespace total_houses_in_neighborhood_l226_226653

-- Definition of the function f
def f (x : ℕ) : ℕ := x^2 + 3*x

-- Given conditions
def x := 40

-- The theorem states that the total number of houses in Mariam's neighborhood is 1760.
theorem total_houses_in_neighborhood : (x + f x) = 1760 :=
by
  sorry

end total_houses_in_neighborhood_l226_226653


namespace perfect_squares_of_diophantine_l226_226430

theorem perfect_squares_of_diophantine (a b : ℤ) (h : 2 * a^2 + a = 3 * b^2 + b) :
  ∃ k m : ℤ, (a - b) = k^2 ∧ (2 * a + 2 * b + 1) = m^2 := by
  sorry

end perfect_squares_of_diophantine_l226_226430


namespace value_of_a_value_of_sin_A_plus_pi_over_4_l226_226282

section TriangleABC

variables {a b c A B : ℝ}
variables (h_b : b = 3) (h_c : c = 1) (h_A_eq_2B : A = 2 * B)

theorem value_of_a : a = 2 * Real.sqrt 3 :=
sorry

theorem value_of_sin_A_plus_pi_over_4 : Real.sin (A + π / 4) = (4 - Real.sqrt 2) / 6 :=
sorry

end TriangleABC

end value_of_a_value_of_sin_A_plus_pi_over_4_l226_226282


namespace license_plate_count_is_correct_l226_226929

/-- Define the number of consonants in the English alphabet --/
def num_consonants : Nat := 20

/-- Define the number of possibilities for 'A' --/
def num_A : Nat := 1

/-- Define the number of even digits --/
def num_even_digits : Nat := 5

/-- Define the total number of valid four-character license plates --/
def total_license_plate_count : Nat :=
  num_consonants * num_A * num_consonants * num_even_digits

/-- Theorem stating that the total number of license plates is 2000 --/
theorem license_plate_count_is_correct : 
  total_license_plate_count = 2000 :=
  by
    -- The proof is omitted
    sorry

end license_plate_count_is_correct_l226_226929


namespace smallest_possible_sum_l226_226821

theorem smallest_possible_sum (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_diff : x ≠ y) (h_eq : 1/x + 1/y = 1/12) : x + y = 49 :=
by
  sorry

end smallest_possible_sum_l226_226821


namespace no_real_solution_arctan_eqn_l226_226725

theorem no_real_solution_arctan_eqn :
  ¬∃ x : ℝ, 0 < x ∧ (Real.arctan (1 / x ^ 2) + Real.arctan (1 / x ^ 4) = (Real.pi / 4)) :=
by
  sorry

end no_real_solution_arctan_eqn_l226_226725


namespace cost_of_soda_l226_226624

-- Define the system of equations
theorem cost_of_soda (b s f : ℕ): 
  3 * b + s = 390 ∧ 
  2 * b + 3 * s = 440 ∧ 
  b + 2 * f = 230 ∧ 
  s + 3 * f = 270 → 
  s = 234 := 
by 
  sorry

end cost_of_soda_l226_226624


namespace forty_percent_of_number_l226_226429

theorem forty_percent_of_number (N : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 15) :
  0.40 * N = 180 :=
by
  sorry

end forty_percent_of_number_l226_226429


namespace sawyer_joined_coaching_l226_226983

variable (daily_fees total_fees : ℕ)
variable (year_not_leap : Prop)
variable (discontinue_day : ℕ)

theorem sawyer_joined_coaching :
  daily_fees = 39 → 
  total_fees = 11895 → 
  year_not_leap → 
  discontinue_day = 307 → 
  ∃ start_day, start_day = 30 := 
by
  intros h_daily_fees h_total_fees h_year_not_leap h_discontinue_day
  sorry

end sawyer_joined_coaching_l226_226983


namespace min_xy_min_x_plus_y_l226_226190

theorem min_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y - x - y = 3) : x * y ≥ 9 :=
sorry

theorem min_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y - x - y = 3) : x + y ≥ 6 :=
sorry

end min_xy_min_x_plus_y_l226_226190


namespace sum_of_consecutive_integers_bound_sqrt_40_l226_226620

theorem sum_of_consecutive_integers_bound_sqrt_40 (a b : ℤ) (h₁ : a < Real.sqrt 40) (h₂ : Real.sqrt 40 < b) (h₃ : b = a + 1) : a + b = 13 :=
by
  sorry

end sum_of_consecutive_integers_bound_sqrt_40_l226_226620


namespace calculate_a_over_b_l226_226988

noncomputable def system_solution (x y a b : ℝ) : Prop :=
  (8 * x - 5 * y = a) ∧ (10 * y - 15 * x = b) ∧ (x ≠ 0) ∧ (y ≠ 0) ∧ (b ≠ 0)

theorem calculate_a_over_b (x y a b : ℝ) (h : system_solution x y a b) : a / b = 8 / 15 :=
by
  sorry

end calculate_a_over_b_l226_226988


namespace distance_from_point_to_focus_l226_226503

theorem distance_from_point_to_focus (P : ℝ × ℝ) (hP : P.2^2 = 8 * P.1) (hX : P.1 = 8) :
  dist P (2, 0) = 10 :=
sorry

end distance_from_point_to_focus_l226_226503


namespace roots_of_quadratic_equation_are_real_and_distinct_l226_226385

def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem roots_of_quadratic_equation_are_real_and_distinct :
  quadratic_discriminant 1 (-2) (-6) > 0 :=
by
  norm_num
  sorry

end roots_of_quadratic_equation_are_real_and_distinct_l226_226385


namespace minimize_sum_of_f_seq_l226_226645

def f (x : ℝ) : ℝ := x^2 - 8 * x + 10

def isArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem minimize_sum_of_f_seq
  (a : ℕ → ℝ)
  (h₀ : isArithmeticSequence a 1)
  (h₁ : a 1 = a₁)
  : f (a 1) + f (a 2) + f (a 3) = 3 * a₁^2 - 18 * a₁ + 30 →

  (∀ x, 3 * x^2 - 18 * x + 30 ≥ 3 * 3^2 - 18 * 3 + 30) →
  a₁ = 3 :=
by
  sorry

end minimize_sum_of_f_seq_l226_226645


namespace sum_of_squares_mul_l226_226944

theorem sum_of_squares_mul (a b c d : ℝ) :
(a^2 + b^2) * (c^2 + d^2) = (a * c + b * d)^2 + (a * d - b * c)^2 :=
by
  sorry

end sum_of_squares_mul_l226_226944


namespace February_March_Ratio_l226_226743

theorem February_March_Ratio (J F M : ℕ) (h1 : F = 2 * J) (h2 : M = 8800) (h3 : J + F + M = 12100) : F / M = 1 / 4 :=
by
  sorry

end February_March_Ratio_l226_226743


namespace smallest_solution_of_equation_l226_226543

theorem smallest_solution_of_equation : 
    ∃ x : ℝ, x*|x| = 3 * x - 2 ∧ 
            ∀ y : ℝ, y*|y| = 3 * y - 2 → x ≤ y :=
sorry

end smallest_solution_of_equation_l226_226543


namespace day_of_week_299th_day_2004_l226_226344

noncomputable def day_of_week (day: ℕ): ℕ := day % 7

theorem day_of_week_299th_day_2004 : 
  ∀ (d: ℕ), day_of_week d = 3 → d = 45 → day_of_week 299 = 5 :=
by
  sorry

end day_of_week_299th_day_2004_l226_226344


namespace hours_worked_each_day_l226_226212

-- Given conditions
def total_hours_worked : ℕ := 18
def number_of_days_worked : ℕ := 6

-- Statement to prove
theorem hours_worked_each_day : total_hours_worked / number_of_days_worked = 3 := by
  sorry

end hours_worked_each_day_l226_226212


namespace sum_series_eq_one_third_l226_226581

theorem sum_series_eq_one_third :
  ∑' n : ℕ, (if h : n > 0 then (2^n / (1 + 2^n + 2^(n + 1) + 2^(2 * n + 1))) else 0) = 1 / 3 :=
by
  sorry

end sum_series_eq_one_third_l226_226581


namespace leif_apples_l226_226071

-- Definitions based on conditions
def oranges : ℕ := 24
def apples (oranges apples_diff : ℕ) := oranges - apples_diff

-- Theorem stating the problem to prove
theorem leif_apples (oranges apples_diff : ℕ) (h1 : oranges = 24) (h2 : apples_diff = 10) : apples oranges apples_diff = 14 :=
by
  -- Using the definition of apples and given conditions, prove the number of apples
  rw [h1, h2]
  -- Calculating the number of apples
  show 24 - 10 = 14
  rfl

end leif_apples_l226_226071


namespace remainder_base12_2543_div_9_l226_226718

theorem remainder_base12_2543_div_9 : 
  let n := 2 * 12^3 + 5 * 12^2 + 4 * 12^1 + 3 * 12^0
  (n % 9) = 8 :=
by
  let n := 2 * 12^3 + 5 * 12^2 + 4 * 12^1 + 3 * 12^0
  sorry

end remainder_base12_2543_div_9_l226_226718


namespace find_coefficients_l226_226248

noncomputable def P (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^4 - 8 * a * x^3 + b * x^2 - 32 * c * x + 16 * c

theorem find_coefficients (a b c : ℝ) (h : a ≠ 0) :
  (∃ x1 x2 x3 x4 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ x4 > 0 ∧ P a b c x1 = 0 ∧ P a b c x2 = 0 ∧ P a b c x3 = 0 ∧ P a b c x4 = 0) →
  (b = 16 * a ∧ c = a) :=
by
  sorry

end find_coefficients_l226_226248


namespace max_squares_covered_by_card_l226_226220

theorem max_squares_covered_by_card : 
  let checkerboard_square_size := 1
  let card_side := 2
  let card_diagonal := Real.sqrt (card_side ^ 2 + card_side ^ 2)
  ∃ n, n = 9 :=
by
  let checkerboard_square_size := 1
  let card_side := 2
  let card_diagonal := Real.sqrt (card_side ^ 2 + card_side ^ 2)
  existsi 9
  sorry

end max_squares_covered_by_card_l226_226220


namespace village_population_l226_226205

theorem village_population (P : ℝ) (h : 0.8 * P = 64000) : P = 80000 := by
  sorry

end village_population_l226_226205


namespace solution_set_inequality_l226_226840

variable (f : ℝ → ℝ)
variable (h1 : ∀ x, f (x - 1/2) + f (x + 1) = 0)
variable (h2 : e ^ 3 * f 2018 = 1)
variable (h3 : ∀ x, f x > f'' (-x))
variable (h4 : ∀ x, f x = f (-x))

theorem solution_set_inequality :
  ∀ x, f (x - 1) > 1 / (e ^ x) ↔ x > 3 :=
sorry

end solution_set_inequality_l226_226840


namespace division_remainder_l226_226912

theorem division_remainder (dividend divisor quotient remainder : ℕ)
  (h₁ : dividend = 689)
  (h₂ : divisor = 36)
  (h₃ : quotient = 19)
  (h₄ : dividend = divisor * quotient + remainder) :
  remainder = 5 :=
by
  sorry

end division_remainder_l226_226912


namespace problem_l226_226869

open Complex

-- Given condition: smallest positive integer n greater than 3
def smallest_n_gt_3 (n : ℕ) : Prop :=
  n > 3 ∧ ∀ m : ℕ, m > 3 → m < n → False

-- Given condition: equation holds for complex numbers
def equation_holds (a b : ℝ) (n : ℕ) : Prop :=
  (a + b * I)^n + a = (a - b * I)^n + b

-- Proof problem: Given conditions, prove b / a = 1
theorem problem (n : ℕ) (a b : ℝ)
  (h1 : smallest_n_gt_3 n)
  (h2 : 0 < a) (h3 : 0 < b)
  (h4 : equation_holds a b n) :
  b / a = 1 :=
by
  sorry

end problem_l226_226869


namespace range_of_a_for_critical_point_l226_226467

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - 9 * Real.log x

theorem range_of_a_for_critical_point :
  ∀ a : ℝ, (∃ x ∈ Set.Icc (a - 1) (a + 1), deriv f x = 0) ↔ 2 < a ∧ a < 4 :=
by
  sorry

end range_of_a_for_critical_point_l226_226467


namespace measure_of_one_interior_angle_of_regular_octagon_l226_226374

theorem measure_of_one_interior_angle_of_regular_octagon 
  (n : Nat) (h_n : n = 8) : 
  (one_interior_angle : ℝ) = 135 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l226_226374


namespace decimal_to_vulgar_fraction_l226_226484

theorem decimal_to_vulgar_fraction (h : (34 / 100 : ℚ) = 0.34) : (0.34 : ℚ) = 17 / 50 := by
  sorry

end decimal_to_vulgar_fraction_l226_226484


namespace range_of_a_l226_226494

noncomputable def f (a : ℝ) (x : ℝ) := 2 * a * x^2 + 4 * (a - 3) * x + 5

theorem range_of_a (a : ℝ) :
  (∀ x < 3, f a x ≤ f a (3 : ℝ)) ↔ 0 ≤ a ∧ a ≤ 3 / 4 :=
by
  sorry

end range_of_a_l226_226494


namespace simplify_complex_fraction_l226_226310

-- Define the complex numbers involved
def numerator := 3 + 4 * Complex.I
def denominator := 5 - 2 * Complex.I

-- Define what we need to prove: the simplified form
theorem simplify_complex_fraction : 
    (numerator / denominator : Complex) = (7 / 29) + (26 / 29) * Complex.I := 
by
  -- Proof is omitted here
  sorry

end simplify_complex_fraction_l226_226310


namespace correct_option_is_D_l226_226011

noncomputable def data : List ℕ := [7, 5, 3, 5, 10]

theorem correct_option_is_D :
  let mean := (7 + 5 + 3 + 5 + 10) / 5
  let variance := (1 / 5 : ℚ) * ((7 - mean) ^ 2 + (5 - mean) ^ 2 + (5 - mean) ^ 2 + (3 - mean) ^ 2 + (10 - mean) ^ 2)
  let mode := 5
  let median := 5
  mean = 6 ∧ variance ≠ 3.6 ∧ mode ≠ 10 ∧ median ≠ 3 :=
by
  sorry

end correct_option_is_D_l226_226011


namespace num_complementary_sets_l226_226882

-- Definitions for shapes, colors, shades, and patterns
inductive Shape
| circle | square | triangle

inductive Color
| red | blue | green

inductive Shade
| light | medium | dark

inductive Pattern
| striped | dotted | plain

-- Definition of a card
structure Card where
  shape : Shape
  color : Color
  shade : Shade
  pattern : Pattern

-- Condition: Each possible combination is represented once in a deck of 81 cards.
def deck : List Card := sorry -- Construct the deck with 81 unique cards

-- Predicate for complementary sets of three cards
def is_complementary (c1 c2 c3 : Card) : Prop :=
  (c1.shape = c2.shape ∧ c2.shape = c3.shape ∧ c1.shape = c3.shape ∨
   c1.shape ≠ c2.shape ∧ c2.shape ≠ c3.shape ∧ c1.shape ≠ c3.shape) ∧
  (c1.color = c2.color ∧ c2.color = c3.color ∧ c1.color = c3.color ∨
   c1.color ≠ c2.color ∧ c2.color ≠ c3.color ∧ c1.color ≠ c3.color) ∧
  (c1.shade = c2.shade ∧ c2.shade = c3.shade ∧ c1.shade = c3.shade ∨
   c1.shade ≠ c2.shade ∧ c2.shade ≠ c3.shade ∧ c1.shade ≠ c3.shade) ∧
  (c1.pattern = c2.pattern ∧ c2.pattern = c3.pattern ∧ c1.pattern = c3.pattern ∨
   c1.pattern ≠ c2.pattern ∧ c2.pattern ≠ c3.pattern ∧ c1.pattern ≠ c3.pattern)

-- Statement of the theorem to prove
theorem num_complementary_sets : 
  ∃ (complementary_sets : List (Card × Card × Card)), 
  complementary_sets.length = 5400 ∧
  ∀ (c1 c2 c3 : Card), (c1, c2, c3) ∈ complementary_sets → is_complementary c1 c2 c3 :=
sorry

end num_complementary_sets_l226_226882


namespace inequality_proof_equality_case_l226_226819

-- Defining that a, b, c are positive real numbers
variables (a b c : ℝ)
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- The main theorem statement
theorem inequality_proof :
  a^2 + b^2 + c^2 + ( (1 / a) + (1 / b) + (1 / c) )^2 >= 6 * Real.sqrt 3 :=
sorry

-- Equality case
theorem equality_case :
  a = b ∧ b = c ∧ a = Real.sqrt 3^(1/4) →
  a^2 + b^2 + c^2 + ( (1 / a) + (1 / b) + (1 / c) )^2 = 6 * Real.sqrt 3 :=
sorry

end inequality_proof_equality_case_l226_226819


namespace sheepdog_catches_sheep_in_20_seconds_l226_226652

noncomputable def speed_sheep : ℝ := 12 -- feet per second
noncomputable def speed_sheepdog : ℝ := 20 -- feet per second
noncomputable def initial_distance : ℝ := 160 -- feet

theorem sheepdog_catches_sheep_in_20_seconds :
  (initial_distance / (speed_sheepdog - speed_sheep)) = 20 :=
by
  sorry

end sheepdog_catches_sheep_in_20_seconds_l226_226652


namespace equation_of_circle_O2_equation_of_tangent_line_l226_226822

-- Define circle O1
def circle_O1 (x y : ℝ) : Prop :=
  x^2 + (y + 1)^2 = 4

-- Define the center and radius of circle O2 given that they are externally tangent
def center_O2 : ℝ × ℝ := (3, 3)
def radius_O2 : ℝ := 3

-- Prove the equation of circle O2
theorem equation_of_circle_O2 :
  ∀ (x y : ℝ), (x - 3)^2 + (y - 3)^2 = 9 := by
  intro x y
  sorry

-- Prove the equation of the common internal tangent line to circles O1 and O2
theorem equation_of_tangent_line :
  ∀ (x y : ℝ), 3 * x + 4 * y - 21 = 0 := by
  intro x y
  sorry

end equation_of_circle_O2_equation_of_tangent_line_l226_226822


namespace jungsoo_number_is_correct_l226_226715

def J := (1 * 4) + (0.1 * 2) + (0.001 * 7)
def Y := 100 * J 
def S := Y + 0.05

theorem jungsoo_number_is_correct : S = 420.75 := by
  sorry

end jungsoo_number_is_correct_l226_226715


namespace order_of_f_values_l226_226648

noncomputable def f (x : ℝ) : ℝ := if x >= 1 then 3^x - 1 else 0 -- define f such that it handles the missing part

theorem order_of_f_values :
  (∀ x: ℝ, f (2 - x) = f (1 + x)) ∧ (∀ x: ℝ, x >= 1 → f x = 3^x - 1) →
  f 0 < f 3 ∧ f 3 < f (-2) :=
by
  sorry

end order_of_f_values_l226_226648


namespace solve_for_x_l226_226155

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
by
  sorry

end solve_for_x_l226_226155


namespace general_term_of_geometric_sequence_l226_226204

variable (a : ℕ → ℝ) (q : ℝ)

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

theorem general_term_of_geometric_sequence
  (h1 : a 1 + a 3 = 10)
  (h2 : a 4 + a 6 = 5 / 4)
  (hq : is_geometric_sequence a q)
  (q := 1/2) :
  ∃ a₀ : ℝ, ∀ n : ℕ, a n = a₀ * q^(n - 1) :=
sorry

end general_term_of_geometric_sequence_l226_226204


namespace triangle_dimensions_l226_226951

theorem triangle_dimensions (a b c : ℕ) (h1 : a > b) (h2 : b > c)
  (h3 : a = 2 * c) (h4 : b - 2 = c) (h5 : 2 * a / 3 = b) :
  a = 12 ∧ b = 8 ∧ c = 6 :=
by
  sorry

end triangle_dimensions_l226_226951


namespace anna_not_lose_l226_226307

theorem anna_not_lose :
  ∀ (cards : Fin 9 → ℕ),
    ∃ (A B C D : ℕ),
      (A + B ≥ C + D) :=
by
  sorry

end anna_not_lose_l226_226307


namespace deceased_member_income_l226_226739

theorem deceased_member_income (a b c d : ℝ)
    (h1 : a = 735) 
    (h2 : b = 650)
    (h3 : c = 4 * 735)
    (h4 : d = 3 * 650) :
    c - d = 990 := by
  sorry

end deceased_member_income_l226_226739


namespace nine_distinct_numbers_product_l226_226986

variable (a b c d e f g h i : ℕ)

theorem nine_distinct_numbers_product (ha : a = 12) (hb : b = 9) (hc : c = 2)
                                      (hd : d = 1) (he : e = 6) (hf : f = 36)
                                      (hg : g = 18) (hh : h = 4) (hi : i = 3) :
  (a * b * c = 216) ∧ (d * e * f = 216) ∧ (g * h * i = 216) ∧
  (a * d * g = 216) ∧ (b * e * h = 216) ∧ (c * f * i = 216) ∧
  (a * e * i = 216) ∧ (c * e * g = 216) :=
by
  sorry

end nine_distinct_numbers_product_l226_226986


namespace no_such_a_and_sequence_exists_l226_226839

theorem no_such_a_and_sequence_exists :
  ¬∃ (a : ℝ) (a_pos : 0 < a ∧ a < 1) (a_seq : ℕ → ℝ), (∀ n : ℕ, 0 < a_seq n) ∧ (∀ n : ℕ, 1 + a_seq (n + 1) ≤ a_seq n + (a / (n + 1)) * a_seq n) :=
by
  sorry

end no_such_a_and_sequence_exists_l226_226839


namespace factorize_polynomial_l226_226917

theorem factorize_polynomial (x : ℝ) : 12 * x ^ 2 + 8 * x = 4 * x * (3 * x + 2) := 
sorry

end factorize_polynomial_l226_226917


namespace evaluate_expression_l226_226060

theorem evaluate_expression : (8^5 / 8^2) * 2^12 = 2^21 := by
  sorry

end evaluate_expression_l226_226060


namespace arrange_in_circle_l226_226334

open Nat

noncomputable def smallest_n := 70

theorem arrange_in_circle (n : ℕ) (h : n = 70) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ n →
    (∀ j : ℕ, 1 ≤ j ∧ j ≤ 40 → k > ((k + j) % n)) ∨
    (∀ p : ℕ, 1 ≤ p ∧ p ≤ 30 → k < ((k + p) % n))) :=
by
  sorry

end arrange_in_circle_l226_226334


namespace games_left_is_correct_l226_226801

-- Define the initial number of DS games
def initial_games : ℕ := 98

-- Define the number of games given away
def games_given_away : ℕ := 7

-- Define the number of games left
def games_left : ℕ := initial_games - games_given_away

-- Theorem statement to prove that the number of games left is 91
theorem games_left_is_correct : games_left = 91 :=
by
  -- Currently, we use sorry to skip the actual proof part.
  sorry

end games_left_is_correct_l226_226801


namespace area_of_paper_l226_226735

theorem area_of_paper (L W : ℕ) (h1 : 2 * L + W = 34) (h2 : L + 2 * W = 38) : 
  L * W = 140 := 
by sorry

end area_of_paper_l226_226735


namespace range_of_a_l226_226778

noncomputable def f : ℝ → ℝ
| x => if x > 0 then Real.log x / Real.log 2 else Real.log (-x) / Real.log (1/2)

theorem range_of_a (a : ℝ) (h : f a > f (-a)) : a ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi (1 : ℝ) :=
by
  sorry

end range_of_a_l226_226778


namespace calc_miscellaneous_collective_expenses_l226_226325

def individual_needed_amount : ℕ := 450
def additional_needed_amount : ℕ := 475
def total_students : ℕ := 6
def first_day_amount : ℕ := 600
def second_day_amount : ℕ := 900
def third_day_amount : ℕ := 400
def days : ℕ := 4

def total_individual_goal : ℕ := individual_needed_amount + additional_needed_amount
def total_students_goal : ℕ := total_individual_goal * total_students
def total_first_3_days : ℕ := first_day_amount + second_day_amount + third_day_amount
def total_next_4_days : ℕ := (total_first_3_days / 2) * days
def total_raised : ℕ := total_first_3_days + total_next_4_days

def miscellaneous_collective_expenses : ℕ := total_raised - total_students_goal

theorem calc_miscellaneous_collective_expenses : miscellaneous_collective_expenses = 150 := by
  sorry

end calc_miscellaneous_collective_expenses_l226_226325


namespace fraction_of_managers_l226_226128

theorem fraction_of_managers (female_managers : ℕ) (total_female_employees : ℕ)
  (total_employees: ℕ) (male_employees: ℕ) (f: ℝ) :
  female_managers = 200 →
  total_female_employees = 500 →
  total_employees = total_female_employees + male_employees →
  (f * total_employees) = female_managers + (f * male_employees) →
  f = 0.4 :=
by
  intros h1 h2 h3 h4
  sorry

end fraction_of_managers_l226_226128


namespace correct_options_A_and_D_l226_226223

noncomputable def problem_statement :=
  ∃ A B C D : Prop,
  (A = (∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0)) ∧ 
  (B = ∀ (a b c d : ℝ), a > b → c > d → ¬(a * c > b * d)) ∧
  (C = ∀ m : ℝ, ¬((∀ x : ℝ, x > 0 → (m^2 - m - 1) * x^(m^2 - 2 * m - 3) < 0) → (-1 < m ∧ m < 2))) ∧
  (D = ∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ x₁ + x₂ = 3 - a ∧ x₁ * x₂ = a) → a < 0)

-- We need to prove that only A and D are true
theorem correct_options_A_and_D : problem_statement :=
  sorry

end correct_options_A_and_D_l226_226223


namespace robot_paths_from_A_to_B_l226_226693

/-- Define a function that computes the number of distinct paths a robot can take -/
def distinctPaths (A B : ℕ × ℕ) : ℕ := sorry

/-- Proof statement: There are 556 distinct paths from A to B, given the movement conditions -/
theorem robot_paths_from_A_to_B (A B : ℕ × ℕ) (h_move : (A, B) = ((0, 0), (10, 10))) :
  distinctPaths A B = 556 :=
sorry

end robot_paths_from_A_to_B_l226_226693


namespace total_trees_in_park_l226_226746

theorem total_trees_in_park (oak_planted_total maple_planted_total birch_planted_total : ℕ)
  (initial_oak initial_maple initial_birch : ℕ)
  (oak_removed_day2 maple_removed_day2 birch_removed_day2 : ℕ)
  (D1_oak_plant : ℕ) (D2_oak_plant : ℕ) (D1_maple_plant : ℕ) (D2_maple_plant : ℕ)
  (D1_birch_plant : ℕ) (D2_birch_plant : ℕ):
  initial_oak = 25 → initial_maple = 40 → initial_birch = 20 →
  oak_planted_total = 73 → maple_planted_total = 52 → birch_planted_total = 35 →
  D1_oak_plant = 29 → D2_oak_plant = 26 →
  D1_maple_plant = 26 → D2_maple_plant = 13 →
  D1_birch_plant = 10 → D2_birch_plant = 16 →
  oak_removed_day2 = 15 → maple_removed_day2 = 10 → birch_removed_day2 = 5 →
  (initial_oak + oak_planted_total - oak_removed_day2) +
  (initial_maple + maple_planted_total - maple_removed_day2) +
  (initial_birch + birch_planted_total - birch_removed_day2) = 215 :=
by
  intros h_initial_oak h_initial_maple h_initial_birch
         h_oak_planted_total h_maple_planted_total h_birch_planted_total
         h_D1_oak h_D2_oak h_D1_maple h_D2_maple h_D1_birch h_D2_birch
         h_oak_removed h_maple_removed h_birch_removed
  sorry

end total_trees_in_park_l226_226746


namespace apples_left_is_correct_l226_226846

-- Definitions for the conditions
def blue_apples : ℕ := 5
def yellow_apples : ℕ := 2 * blue_apples
def total_apples : ℕ := blue_apples + yellow_apples
def apples_given_to_son : ℚ := 1 / 5 * total_apples
def apples_left : ℚ := total_apples - apples_given_to_son

-- The main statement to be proven
theorem apples_left_is_correct : apples_left = 12 := by
  sorry

end apples_left_is_correct_l226_226846


namespace quadratic_inequality_solution_empty_l226_226191

theorem quadratic_inequality_solution_empty (m : ℝ) :
  (∀ x : ℝ, ((m + 1) * x^2 - m * x + m - 1 < 0) → false) →
  (m ≥ (2 * Real.sqrt 3) / 3 ∨ m ≤ -(2 * Real.sqrt 3) / 3) :=
by
  sorry

end quadratic_inequality_solution_empty_l226_226191


namespace eccentricity_of_hyperbola_is_e_l226_226395

-- Definitions and given conditions
variable (a b c : ℝ)
variable (h_a_pos : a > 0) (h_b_pos : b > 0)
variable (h_hyperbola : ∀ x y : ℝ, (x^2)/(a^2) - (y^2)/(b^2) = 1)
variable (h_left_focus : ∀ F : ℝ × ℝ, F = (-c, 0))
variable (h_circle : ∀ E : ℝ × ℝ, E.1^2 + E.2^2 = a^2)
variable (h_parabola : ∀ P : ℝ × ℝ, P.2^2 = 4*c*P.1)
variable (h_midpoint : ∀ E P F : ℝ × ℝ, E = (F.1 + P.1) / 2 ∧ E.2 = (F.2 + P.2) / 2)

-- The statement to be proved
theorem eccentricity_of_hyperbola_is_e :
    ∃ e : ℝ, e = (Real.sqrt 5 + 1) / 2 :=
sorry

end eccentricity_of_hyperbola_is_e_l226_226395


namespace tilly_total_profit_l226_226331

theorem tilly_total_profit :
  let bags_sold := 100
  let selling_price_per_bag := 10
  let buying_price_per_bag := 7
  let profit_per_bag := selling_price_per_bag - buying_price_per_bag
  let total_profit := bags_sold * profit_per_bag
  total_profit = 300 :=
by
  let bags_sold := 100
  let selling_price_per_bag := 10
  let buying_price_per_bag := 7
  let profit_per_bag := selling_price_per_bag - buying_price_per_bag
  let total_profit := bags_sold * profit_per_bag
  sorry

end tilly_total_profit_l226_226331


namespace arithmetic_sequence_a10_l226_226114

theorem arithmetic_sequence_a10 (a : ℕ → ℕ) (d : ℕ) 
  (h_seq : ∀ n, a (n + 1) = a n + d) 
  (h_positive : ∀ n, a n > 0) 
  (h_sum : a 1 + a 2 + a 3 = 15) 
  (h_geo : (a 1 + 2) * (a 3 + 13) = (a 2 + 5) * (a 2 + 5))  
  : a 10 = 21 := sorry

end arithmetic_sequence_a10_l226_226114


namespace simplify_expression_l226_226607

theorem simplify_expression (a b : ℤ) : 
  (17 * a + 45 * b) + (15 * a + 36 * b) - (12 * a + 42 * b) - 3 * (2 * a + 3 * b) = 14 * a + 30 * b :=
by
  sorry

end simplify_expression_l226_226607


namespace carol_invitations_l226_226179

-- Definitions: each package has 3 invitations, Carol bought 2 packs, and Carol needs 3 extra invitations.
def invitations_per_pack : ℕ := 3
def packs_bought : ℕ := 2
def extra_invitations : ℕ := 3

-- Total number of invitations Carol will have
def total_invitations : ℕ := (packs_bought * invitations_per_pack) + extra_invitations

-- Statement to prove: Carol wants to invite 9 friends.
theorem carol_invitations : total_invitations = 9 := by
  sorry  -- Proof omitted

end carol_invitations_l226_226179


namespace carson_gold_stars_l226_226221

theorem carson_gold_stars (yesterday_stars today_total_stars earned_today : ℕ) 
  (h1 : yesterday_stars = 6) 
  (h2 : today_total_stars = 15) 
  (h3 : earned_today = today_total_stars - yesterday_stars) 
  : earned_today = 9 :=
sorry

end carson_gold_stars_l226_226221


namespace number_of_sampled_medium_stores_is_five_l226_226426

-- Definitions based on the conditions
def total_stores : ℕ := 300
def large_stores : ℕ := 30
def medium_stores : ℕ := 75
def small_stores : ℕ := 195
def sample_size : ℕ := 20

-- Proportion calculation function
def medium_store_proportion := (medium_stores : ℚ) / (total_stores : ℚ)

-- Sampled medium stores calculation
def sampled_medium_stores := medium_store_proportion * (sample_size : ℚ)

-- Theorem stating the number of medium stores drawn using stratified sampling
theorem number_of_sampled_medium_stores_is_five :
  sampled_medium_stores = 5 := 
by 
  sorry

end number_of_sampled_medium_stores_is_five_l226_226426


namespace train_cars_count_l226_226510

theorem train_cars_count
  (cars_in_first_15_seconds : ℕ)
  (time_for_first_5_cars : ℕ)
  (total_time_to_pass : ℕ)
  (h_cars_in_first_15_seconds : cars_in_first_15_seconds = 5)
  (h_time_for_first_5_cars : time_for_first_5_cars = 15)
  (h_total_time_to_pass : total_time_to_pass = 210) :
  (total_time_to_pass / time_for_first_5_cars) * cars_in_first_15_seconds = 70 := 
by 
  sorry

end train_cars_count_l226_226510


namespace find_a_b_l226_226349

theorem find_a_b :
  ∃ a b : ℝ, 
    (a = -4) ∧ (b = -9) ∧
    (∀ x : ℝ, |8 * x + 9| < 7 ↔ a * x^2 + b * x - 2 > 0) := 
sorry

end find_a_b_l226_226349


namespace value_of_square_sum_l226_226824

theorem value_of_square_sum (x y : ℝ) (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
by 
  sorry

end value_of_square_sum_l226_226824


namespace sum_of_ages_of_alex_and_allison_is_47_l226_226141

theorem sum_of_ages_of_alex_and_allison_is_47 (diane_age_now : ℕ)
  (diane_age_at_30_alex_relation : diane_age_now + 14 = 30 ∧ diane_age_now + 14 = 60 / 2)
  (diane_age_at_30_allison_relation : diane_age_now + 14 = 30 ∧ 30 = 2 * (diane_age_now + 14 - (30 - 15)))
  : (60 - (30 - 16)) + (15 - (30 - 16)) = 47 :=
by
  sorry

end sum_of_ages_of_alex_and_allison_is_47_l226_226141


namespace original_price_sarees_l226_226660

theorem original_price_sarees
  (P : ℝ)
  (h : 0.90 * 0.85 * P = 378.675) :
  P = 495 :=
sorry

end original_price_sarees_l226_226660


namespace pure_imaginary_solution_l226_226438

theorem pure_imaginary_solution (a : ℝ) (i : ℂ) (h : i*i = -1) : (∀ z : ℂ, z = 1 + a * i → (z ^ 2).re = 0) → (a = 1 ∨ a = -1) := by
  sorry

end pure_imaginary_solution_l226_226438


namespace movie_duration_l226_226443

theorem movie_duration :
  let start_time := (13, 30)
  let end_time := (14, 50)
  let hours := end_time.1 - start_time.1
  let minutes := end_time.2 - start_time.2
  (if minutes < 0 then (hours - 1, minutes + 60) else (hours, minutes)) = (1, 20) := by
    sorry

end movie_duration_l226_226443


namespace probability_odd_multiple_of_5_l226_226779

theorem probability_odd_multiple_of_5 :
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100 ∧ 1 ≤ c ∧ c ≤ 100 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (a * b * c) % 2 = 1 ∧ (a * b * c) % 5 = 0) → 
  p = 3 / 125 := 
sorry

end probability_odd_multiple_of_5_l226_226779


namespace calculate_lego_set_cost_l226_226335

variable (total_revenue_after_tax : ℝ) (little_cars_base_price : ℝ)
  (discount_rate : ℝ) (tax_rate : ℝ) (num_little_cars : ℕ)
  (num_action_figures : ℕ) (num_board_games : ℕ)
  (lego_set_cost_before_tax : ℝ)

theorem calculate_lego_set_cost :
  total_revenue_after_tax = 136.50 →
  little_cars_base_price = 5 →
  discount_rate = 0.10 →
  tax_rate = 0.05 →
  num_little_cars = 3 →
  num_action_figures = 2 →
  num_board_games = 1 →
  lego_set_cost_before_tax = 85 :=
by
  sorry

end calculate_lego_set_cost_l226_226335


namespace quadratic_roots_relation_l226_226720

theorem quadratic_roots_relation (a b c d : ℝ) (h : ∀ x : ℝ, (c * x^2 + d * x + a = 0) → 
  (a * (2007 * x)^2 + b * (2007 * x) + c = 0)) : b^2 = d^2 := 
sorry

end quadratic_roots_relation_l226_226720


namespace max_sum_ge_zero_l226_226006

-- Definition for max and min functions for real numbers
noncomputable def max_real (x y : ℝ) := if x ≥ y then x else y
noncomputable def min_real (x y : ℝ) := if x ≤ y then x else y

-- Condition: a + b + c + d = 0
def sum_zero (a b c d : ℝ) := a + b + c + d = 0

-- Lean statement for Problem (a)
theorem max_sum_ge_zero (a b c d : ℝ) (h : sum_zero a b c d) : 
  max_real a b + max_real a c + max_real a d + max_real b c + max_real b d + max_real c d ≥ 0 :=
sorry

-- Lean statement for Problem (b)
def find_max_k : ℕ :=
2

end max_sum_ge_zero_l226_226006


namespace set_intersection_is_correct_l226_226001

def setA : Set ℝ := {x | x^2 - 4 * x > 0}
def setB : Set ℝ := {x | abs (x - 1) ≤ 2}
def setIntersection : Set ℝ := {x | -1 ≤ x ∧ x < 0}

theorem set_intersection_is_correct :
  setA ∩ setB = setIntersection := 
by
  sorry

end set_intersection_is_correct_l226_226001


namespace calc_f_y_eq_2f_x_l226_226872

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

theorem calc_f_y_eq_2f_x (x : ℝ) (h : -1 < x) (h' : x < 1) :
  f ( (2 * x + x^2) / (1 + 2 * x^2) ) = 2 * f x := by
  sorry

end calc_f_y_eq_2f_x_l226_226872


namespace percentage_chain_l226_226943

theorem percentage_chain (n : ℝ) (h : n = 6000) : 0.1 * (0.3 * (0.5 * n)) = 90 := by
  sorry

end percentage_chain_l226_226943


namespace complement_union_of_M_and_N_l226_226930

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end complement_union_of_M_and_N_l226_226930


namespace problem_solution_l226_226719

theorem problem_solution
  (P Q R S : ℕ)
  (h1 : 2 * Q = P + R)
  (h2 : R * R = Q * S)
  (h3 : R = 4 * Q / 3) :
  P + Q + R + S = 171 :=
by sorry

end problem_solution_l226_226719


namespace decimal_expansion_of_fraction_l226_226815

/-- 
Theorem: The decimal expansion of 13 / 375 is 0.034666...
-/
theorem decimal_expansion_of_fraction : 
  let numerator := 13
  let denominator := 375
  let resulting_fraction := (numerator * 2^3) / (denominator * 2^3)
  let decimal_expansion := 0.03466666666666667
  (resulting_fraction : ℝ) = decimal_expansion :=
sorry

end decimal_expansion_of_fraction_l226_226815


namespace negation_proposition_l226_226118

-- Define the proposition as a Lean function
def quadratic_non_negative (x : ℝ) : Prop := x^2 - 2*x + 1 ≥ 0

-- State the theorem that we need to prove
theorem negation_proposition : ∀ x : ℝ, quadratic_non_negative x :=
by 
  sorry

end negation_proposition_l226_226118


namespace avg_weight_section_b_l226_226342

/-- Definition of the average weight of section B based on given conditions --/
theorem avg_weight_section_b :
  let W_A := 50
  let W_class := 54.285714285714285
  let num_A := 40
  let num_B := 30
  let total_class_weight := (num_A + num_B) * W_class
  let total_A_weight := num_A * W_A
  let total_B_weight := total_class_weight - total_A_weight
  let W_B := total_B_weight / num_B
  W_B = 60 :=
by
  sorry

end avg_weight_section_b_l226_226342


namespace number_of_pages_500_l226_226103

-- Define the conditions as separate constants
def cost_per_page : ℕ := 3 -- cents
def total_cents : ℕ := 1500 

-- Define the number of pages calculation
noncomputable def number_of_pages := total_cents / cost_per_page

-- Statement we want to prove
theorem number_of_pages_500 : number_of_pages = 500 :=
sorry

end number_of_pages_500_l226_226103


namespace disjoint_subsets_with_same_sum_l226_226808

theorem disjoint_subsets_with_same_sum :
  ∀ (S : Finset ℕ), S.card = 10 ∧ (∀ x ∈ S, x ∈ Finset.range 101) →
  ∃ A B : Finset ℕ, A ⊆ S ∧ B ⊆ S ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by
  sorry

end disjoint_subsets_with_same_sum_l226_226808


namespace solve_investment_problem_l226_226786

def investment_problem
  (total_investment : ℝ) (etf_investment : ℝ) (mutual_funds_factor : ℝ) (mutual_funds_investment : ℝ) : Prop :=
  total_investment = etf_investment + mutual_funds_factor * etf_investment →
  mutual_funds_factor * etf_investment = mutual_funds_investment

theorem solve_investment_problem :
  investment_problem 210000 46666.67 3.5 163333.35 :=
by
  sorry

end solve_investment_problem_l226_226786


namespace Jim_paycheck_after_deductions_l226_226232

def calculateRemainingPay (grossPay : ℕ) (retirementPercentage : ℕ) 
                          (taxDeduction : ℕ) : ℕ :=
  let retirementAmount := (grossPay * retirementPercentage) / 100
  let afterRetirement := grossPay - retirementAmount
  let afterTax := afterRetirement - taxDeduction
  afterTax

theorem Jim_paycheck_after_deductions :
  calculateRemainingPay 1120 25 100 = 740 := 
by
  sorry

end Jim_paycheck_after_deductions_l226_226232


namespace f_of_integral_ratio_l226_226626

variable {f : ℝ → ℝ} (h_cont : ∀ x > 0, continuous_at f x)
variable (h_int : ∀ a b : ℝ, a > 0 → b > 0 → ∃ g : ℝ → ℝ, (∫ x in a..b, f x) = g (b / a))

theorem f_of_integral_ratio :
  (∃ c : ℝ, ∀ x > 0, f x = c / x) :=
sorry

end f_of_integral_ratio_l226_226626


namespace find_K_l226_226131

noncomputable def cylinder_paint (r h : ℝ) : ℝ := 2 * Real.pi * r * h
noncomputable def cube_surface_area (s : ℝ) : ℝ := 6 * s^2
noncomputable def cube_volume (s : ℝ) : ℝ := s^3

theorem find_K :
  (cylinder_paint 3 4 = 24 * Real.pi) →
  (∃ s, cube_surface_area s = 24 * Real.pi ∧ cube_volume s = 48 / Real.sqrt K) →
  K = 36 / Real.pi^3 :=
by
  sorry

end find_K_l226_226131


namespace john_has_25_roommates_l226_226016

def roommates_of_bob := 10
def roommates_of_john := 2 * roommates_of_bob + 5

theorem john_has_25_roommates : roommates_of_john = 25 := 
by
  sorry

end john_has_25_roommates_l226_226016


namespace karlson_wins_with_optimal_play_l226_226023

def game_win_optimal_play: Prop :=
  ∀ (total_moves: ℕ), 
  (total_moves % 2 = 1) 

theorem karlson_wins_with_optimal_play: game_win_optimal_play :=
by sorry

end karlson_wins_with_optimal_play_l226_226023


namespace necessary_condition_l226_226981

theorem necessary_condition (a b : ℝ) (h : b ≠ 0) (h2 : a > b) (h3 : b > 0) : (1 / a < 1 / b) :=
sorry

end necessary_condition_l226_226981


namespace cos_pi_minus_alpha_l226_226907

theorem cos_pi_minus_alpha (α : ℝ) (h : Real.sin (π / 2 + α) = 1 / 3) : Real.cos (π - α) = - (1 / 3) :=
by
  sorry

end cos_pi_minus_alpha_l226_226907


namespace set_contains_all_rationals_l226_226233

variable (S : Set ℚ)
variable (h1 : (0 : ℚ) ∈ S)
variable (h2 : ∀ x ∈ S, x + 1 ∈ S ∧ x - 1 ∈ S)
variable (h3 : ∀ x ∈ S, x ≠ 0 → x ≠ 1 → 1 / (x * (x - 1)) ∈ S)

theorem set_contains_all_rationals : ∀ q : ℚ, q ∈ S :=
by
  sorry

end set_contains_all_rationals_l226_226233


namespace fraction_B_A_plus_C_l226_226178

variable (A B C : ℝ)
variable (f : ℝ)
variable (hA : A = 1 / 3 * (B + C))
variable (hB : A = B + 30)
variable (hTotal : A + B + C = 1080)
variable (hf : B = f * (A + C))

theorem fraction_B_A_plus_C :
  f = 2 / 7 :=
sorry

end fraction_B_A_plus_C_l226_226178


namespace great_wall_scientific_notation_l226_226756

theorem great_wall_scientific_notation :
  6700000 = 6.7 * 10^6 :=
sorry

end great_wall_scientific_notation_l226_226756


namespace rhyme_around_3_7_l226_226366

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def rhymes_around (p q m : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p ≠ q ∧ ((p < m ∧ q > m ∧ q - m = m - p) ∨ (p > m ∧ q < m ∧ p - m = m - q))

theorem rhyme_around_3_7 : ∃ m : ℕ, rhymes_around 3 7 m ∧ m = 5 :=
by
  sorry

end rhyme_around_3_7_l226_226366


namespace functional_equation_solution_l226_226070

theorem functional_equation_solution (f : ℕ → ℕ) 
  (H : ∀ a b : ℕ, f (f a + f b) = a + b) : 
  ∀ n : ℕ, f n = n := 
by
  sorry

end functional_equation_solution_l226_226070


namespace log_product_solution_l226_226202

theorem log_product_solution (x : ℝ) (hx : 0 < x) : 
  (Real.log x / Real.log 2) * (Real.log x / Real.log 5) = Real.log 10 / Real.log 2 ↔ 
  x = 2 ^ Real.sqrt (6 * Real.log 2) :=
sorry

end log_product_solution_l226_226202


namespace ducks_arrival_quantity_l226_226421

variable {initial_ducks : ℕ} (arrival_ducks : ℕ)

def initial_geese (initial_ducks : ℕ) := 2 * initial_ducks - 10

def remaining_geese (initial_ducks : ℕ) := initial_geese initial_ducks - 10

def remaining_ducks (initial_ducks arrival_ducks : ℕ) := initial_ducks + arrival_ducks

theorem ducks_arrival_quantity :
  initial_ducks = 25 →
  remaining_geese initial_ducks = 30 →
  remaining_geese initial_ducks = remaining_ducks initial_ducks arrival_ducks + 1 →
  arrival_ducks = 4 :=
by
sorry

end ducks_arrival_quantity_l226_226421


namespace license_plate_count_l226_226058

theorem license_plate_count :
  let digits := 10
  let letters := 26
  let positions := 6
  positions * digits^5 * letters^3 = 105456000 := by
  sorry

end license_plate_count_l226_226058


namespace min_value_2_l226_226480

noncomputable def min_value (a b : ℝ) : ℝ :=
  1 / a + 1 / (b + 1)

theorem min_value_2 {a b : ℝ} (h1 : a > 0) (h2 : b > -1) (h3 : a + b = 1) : min_value a b = 2 :=
by
  sorry

end min_value_2_l226_226480


namespace solve_abs_eq_l226_226959

theorem solve_abs_eq : ∀ x : ℚ, (|2 * x + 6| = 3 * x + 9) ↔ (x = -3) := by
  intros x
  sorry

end solve_abs_eq_l226_226959


namespace tan_150_deg_l226_226762

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - (Real.sqrt 3) / 3 :=
by
  -- Conditions used for defining the theorem
  -- 1. 150^\circ = 180^\circ - 30^\circ
  -- 2. Coordinates of a point on the unit circle at angle θ are (cos θ, sin θ)
  -- 3. For 30^\circ, (cos 30^\circ, sin 30^\circ) = (√3/2, 1/2)
  -- 4. Reflect the point across the y-axis changes x-coordinate's sign
  -- 5. tan θ = y/x for a point (x, y) on the unit circle

  sorry

end tan_150_deg_l226_226762


namespace lineD_intersects_line1_l226_226100

-- Define the lines based on the conditions
def line1 (x y : ℝ) := x + y - 1 = 0
def lineA (x y : ℝ) := 2 * x + 2 * y = 6
def lineB (x y : ℝ) := x + y = 0
def lineC (x y : ℝ) := y = -x - 3
def lineD (x y : ℝ) := y = x - 1

-- Define the statement that line D intersects with line1
theorem lineD_intersects_line1 : ∃ (x y : ℝ), line1 x y ∧ lineD x y :=
by
  sorry

end lineD_intersects_line1_l226_226100


namespace parallelogram_area_l226_226896

theorem parallelogram_area (base height : ℝ) (h_base : base = 25) (h_height : height = 15) :
  base * height = 375 :=
by
  subst h_base
  subst h_height
  sorry

end parallelogram_area_l226_226896


namespace solution_set_f1_geq_4_min_value_pq_l226_226650

-- Define the function f(x) for the first question
def f1 (x : ℝ) : ℝ := |x - 1| + |x - 3|

-- Theorem for part (I)
theorem solution_set_f1_geq_4 (x : ℝ) : f1 x ≥ 4 ↔ x ≤ 0 ∨ x ≥ 4 :=
by
  sorry

-- Define the function f(x) for the second question
def f2 (m x : ℝ) : ℝ := |x - m| + |x - 3|

-- Theorem for part (II)
theorem min_value_pq (p q m : ℝ) (h_pos_p : p > 0) (h_pos_q : q > 0)
    (h_eq : 1 / p + 1 / (2 * q) = m)
    (h_min_f : ∀ x : ℝ, f2 m x ≥ 3) :
    pq = 1 / 18 :=
by
  sorry

end solution_set_f1_geq_4_min_value_pq_l226_226650


namespace certain_number_less_32_l226_226243

theorem certain_number_less_32 (x : ℤ) (h : x - 48 = 22) : x - 32 = 38 :=
by
  sorry

end certain_number_less_32_l226_226243


namespace max_value_a_plus_b_l226_226714

theorem max_value_a_plus_b
  (a b : ℝ)
  (h1 : 4 * a + 3 * b ≤ 10)
  (h2 : 3 * a + 5 * b ≤ 11) :
  a + b ≤ 156 / 55 :=
sorry

end max_value_a_plus_b_l226_226714


namespace brenda_num_cookies_per_box_l226_226185

def numCookiesPerBox (trays : ℕ) (cookiesPerTray : ℕ) (costPerBox : ℚ) (totalSpent : ℚ) : ℚ :=
  let totalCookies := trays * cookiesPerTray
  let numBoxes := totalSpent / costPerBox
  totalCookies / numBoxes

theorem brenda_num_cookies_per_box :
  numCookiesPerBox 3 80 3.5 14 = 60 := by
  sorry

end brenda_num_cookies_per_box_l226_226185


namespace least_sum_of_factors_l226_226499

theorem least_sum_of_factors (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 2400) : a + b = 98 :=
sorry

end least_sum_of_factors_l226_226499


namespace largest_divisor_of_polynomial_l226_226864

theorem largest_divisor_of_polynomial (n : ℕ) (h : n % 2 = 0) : 
  105 ∣ (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) :=
sorry

end largest_divisor_of_polynomial_l226_226864


namespace paul_homework_average_l226_226998

def hoursOnWeeknights : ℕ := 2 * 5
def hoursOnWeekend : ℕ := 5
def totalHomework : ℕ := hoursOnWeeknights + hoursOnWeekend
def practiceNights : ℕ := 2
def daysAvailable : ℕ := 7 - practiceNights
def averageHomeworkPerNight : ℕ := totalHomework / daysAvailable

theorem paul_homework_average :
  averageHomeworkPerNight = 3 := 
by
  -- sorry because we skip the proof
  sorry

end paul_homework_average_l226_226998


namespace general_term_of_arithmetic_seq_l226_226974

variable {a : ℕ → ℤ}

def arithmetic_seq (a : ℕ → ℤ) := ∃ d, ∀ n, a n = a 0 + n * d

theorem general_term_of_arithmetic_seq :
  arithmetic_seq a →
  a 2 = 9 →
  (∃ x y, (x ^ 2 - 16 * x + 60 = 0) ∧ (a 0 = x) ∧ (a 4 = y)) →
  ∀ n, a n = -n + 11 :=
by
  intros h_arith h_a2 h_root
  sorry

end general_term_of_arithmetic_seq_l226_226974


namespace people_in_room_l226_226351

theorem people_in_room (total_people total_chairs : ℕ) (h1 : (2/3 : ℚ) * total_chairs = 1/2 * total_people)
  (h2 : total_chairs - (2/3 : ℚ) * total_chairs = 8) : total_people = 32 := 
by
  sorry

end people_in_room_l226_226351


namespace statement_A_statement_C_statement_D_statement_B_l226_226144

variable (a b : ℝ)

theorem statement_A :
  4 * a^2 - a * b + b^2 = 1 → |a| ≤ 2 * Real.sqrt 15 / 15 :=
sorry

theorem statement_C :
  (4 * a^2 - a * b + b^2 = 1) → 4 / 5 ≤ 4 * a^2 + b^2 ∧ 4 * a^2 + b^2 ≤ 4 / 3 :=
sorry

theorem statement_D :
  4 * a^2 - a * b + b^2 = 1 → |2 * a - b| ≤ 2 * Real.sqrt 10 / 5 :=
sorry

theorem statement_B :
  4 * a^2 - a * b + b^2 = 1 → ¬(|a + b| < 1) :=
sorry

end statement_A_statement_C_statement_D_statement_B_l226_226144


namespace necessary_but_not_sufficient_condition_l226_226615

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (0 < a ∧ a < 1) → ((a + 1) * (a - 2) < 0) ∧ ((∃ b : ℝ, (b + 1) * (b - 2) < 0 ∧ ¬(0 < b ∧ b < 1))) :=
by
  sorry

end necessary_but_not_sufficient_condition_l226_226615


namespace george_run_speed_last_half_mile_l226_226031

theorem george_run_speed_last_half_mile :
  ∀ (distance_school normal_speed first_segment_distance first_segment_speed second_segment_distance second_segment_speed remaining_distance)
    (today_total_time normal_total_time remaining_time : ℝ),
    distance_school = 2 →
    normal_speed = 4 →
    first_segment_distance = 3 / 4 →
    first_segment_speed = 3 →
    second_segment_distance = 3 / 4 →
    second_segment_speed = 4 →
    remaining_distance = 1 / 2 →
    normal_total_time = distance_school / normal_speed →
    today_total_time = (first_segment_distance / first_segment_speed) + (second_segment_distance / second_segment_speed) →
    normal_total_time = today_total_time + remaining_time →
    (remaining_distance / remaining_time) = 8 :=
by
  intros distance_school normal_speed first_segment_distance first_segment_speed second_segment_distance second_segment_speed remaining_distance today_total_time normal_total_time remaining_time h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end george_run_speed_last_half_mile_l226_226031


namespace part1_eq_of_line_l_part2_eq_of_line_l1_l226_226308

def intersection_point (m n : ℝ × ℝ × ℝ) : ℝ × ℝ := sorry

def line_through_point_eq_dists (P A B : ℝ × ℝ) : ℝ × ℝ × ℝ := sorry
def line_area_triangle (P : ℝ × ℝ) (triangle_area : ℝ) : ℝ × ℝ × ℝ := sorry

-- Conditions defined:
def m : ℝ × ℝ × ℝ := (2, -1, -3)
def n : ℝ × ℝ × ℝ := (1, 1, -3)
def P : ℝ × ℝ := intersection_point m n
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (3, 2)
def triangle_area : ℝ := 4

-- Questions translated into Lean 4 statements:
theorem part1_eq_of_line_l : ∃ l : ℝ × ℝ × ℝ, 
  (l = line_through_point_eq_dists P A B) := sorry

theorem part2_eq_of_line_l1 : ∃ l1 : ℝ × ℝ × ℝ,
  (l1 = line_area_triangle P triangle_area) := sorry

end part1_eq_of_line_l_part2_eq_of_line_l1_l226_226308


namespace intersection_range_of_b_l226_226591

theorem intersection_range_of_b (b : ℝ) :
  (∀ (m : ℝ), ∃ (x y : ℝ), x^2 + 2 * y^2 = 3 ∧ y = m * x + b) ↔ 
  b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) := 
sorry

end intersection_range_of_b_l226_226591


namespace problem_l226_226517

theorem problem (x y z : ℕ) (h1 : xy + z = 56) (h2 : yz + x = 56) (h3 : zx + y = 56) : x + y + z = 21 :=
sorry

end problem_l226_226517


namespace tallest_player_height_correct_l226_226074

-- Define the height of the shortest player
def shortest_player_height : ℝ := 68.25

-- Define the height difference between the tallest and shortest player
def height_difference : ℝ := 9.5

-- Define the height of the tallest player based on the conditions
def tallest_player_height : ℝ :=
  shortest_player_height + height_difference

-- Theorem statement
theorem tallest_player_height_correct : tallest_player_height = 77.75 := by
  sorry

end tallest_player_height_correct_l226_226074


namespace arithmetic_sequence_sum_9_l226_226833

theorem arithmetic_sequence_sum_9 :
  ∀ (a : ℕ → ℝ) (d : ℝ),
  (∀ n, a n = 2 + n * d) ∧ d ≠ 0 ∧ (2 : ℝ) + 2 * d ≠ 0 ∧ (2 + 5 * d) ≠ 0 ∧ d = 0.5 →
  (2 + 2 * d)^2 = 2 * (2 + 5 * d) →
  (9 * 2 + (9 * 8 / 2) * 0.5) = 36 :=
by
  intros a d h1 h2
  sorry

end arithmetic_sequence_sum_9_l226_226833


namespace least_possible_sum_l226_226623

theorem least_possible_sum
  (a b x y z : ℕ)
  (hpos_a : 0 < a) (hpos_b : 0 < b)
  (hpos_x : 0 < x) (hpos_y : 0 < y)
  (hpos_z : 0 < z)
  (h : 3 * a = 7 * b ∧ 7 * b = 5 * x ∧ 5 * x = 4 * y ∧ 4 * y = 6 * z) :
  a + b + x + y + z = 459 :=
by
  sorry

end least_possible_sum_l226_226623


namespace percentage_decrease_in_area_l226_226285

variable (L B : ℝ)

def original_area (L B : ℝ) : ℝ := L * B
def new_length (L : ℝ) : ℝ := 0.70 * L
def new_breadth (B : ℝ) : ℝ := 0.85 * B
def new_area (L B : ℝ) : ℝ := new_length L * new_breadth B

theorem percentage_decrease_in_area (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  ((original_area L B - new_area L B) / original_area L B) * 100 = 40.5 :=
by
  sorry

end percentage_decrease_in_area_l226_226285


namespace quadratic_eq_has_distinct_real_roots_l226_226684

theorem quadratic_eq_has_distinct_real_roots (c : ℝ) (h : c = 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1 ^ 2 - 3 * x1 + c = 0) ∧ (x2 ^ 2 - 3 * x2 + c = 0)) :=
by {
  sorry
}

end quadratic_eq_has_distinct_real_roots_l226_226684


namespace system_equations_solution_exists_l226_226812

theorem system_equations_solution_exists (m : ℝ) :
  (∃ x y : ℝ, y = 3 * m * x + 6 ∧ y = (4 * m - 3) * x + 9) ↔ m ≠ 3 :=
by
  sorry

end system_equations_solution_exists_l226_226812


namespace exponentiation_and_division_l226_226529

theorem exponentiation_and_division (a b c : ℕ) (h : a = 6) (h₂ : b = 3) (h₃ : c = 15) :
  9^a * 3^b / 3^c = 1 := by
  sorry

end exponentiation_and_division_l226_226529


namespace problem_equivalence_l226_226809

-- Define the given circles and their properties
def E (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + y^2 = 25
def F (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 + y^2 = 1

-- Define the curve C as the trajectory of the center of the moving circle P
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l intersecting curve C at points A and B with midpoint M(1,1)
def M (A B : ℝ × ℝ) : Prop := (A.1 + B.1 = 2) ∧ (A.2 + B.2 = 2)
def l (x y : ℝ) : Prop := x + 4 * y - 5 = 0

theorem problem_equivalence :
  (∀ x y, E x y ∧ F x y → C x y) ∧
  (∃ A B : ℝ × ℝ, C A.1 A.2 ∧ C B.1 B.2 ∧ M A B → (∀ x y, l x y)) :=
sorry

end problem_equivalence_l226_226809


namespace smallest_next_divisor_l226_226830

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_divisor (a b : ℕ) : Prop := b % a = 0

theorem smallest_next_divisor 
  (m : ℕ) 
  (h1 : 1000 ≤ m ∧ m < 10000) 
  (h2 : is_even m) 
  (h3 : is_divisor 171 m)
  : ∃ k, k > 171 ∧ k = 190 ∧ is_divisor k m := 
by
  sorry

end smallest_next_divisor_l226_226830


namespace right_triangle_sides_l226_226644

/-- Given a right triangle with area 2 * r^2 / 3 where r is the radius of a circle touching one leg,
the extension of the other leg, and the hypotenuse, the sides of the triangle are given by r, 4/3 * r, and 5/3 * r. -/
theorem right_triangle_sides (r : ℝ) (x y : ℝ)
  (h_area : (x * y) / 2 = 2 * r^2 / 3)
  (h_hypotenuse : (x^2 + y^2) = (2 * r + x - y)^2) :
  x = r ∧ y = 4 * r / 3 :=
sorry

end right_triangle_sides_l226_226644


namespace max_equilateral_triangle_area_l226_226548

theorem max_equilateral_triangle_area (length width : ℝ) (h_len : length = 15) (h_width : width = 12) 
: ∃ (area : ℝ), area = 200.25 * Real.sqrt 3 - 450 := by
  sorry

end max_equilateral_triangle_area_l226_226548


namespace buicks_count_l226_226279

-- Definitions
def total_cars := 301
def ford_eqn (chevys : ℕ) := 3 + 2 * chevys
def buicks_eqn (chevys : ℕ) := 12 + 8 * chevys

-- Statement
theorem buicks_count (chevys : ℕ) (fords : ℕ) (buicks : ℕ) :
  total_cars = chevys + fords + buicks ∧
  fords = ford_eqn chevys ∧
  buicks = buicks_eqn chevys →
  buicks = 220 :=
by
  intros h
  sorry

end buicks_count_l226_226279


namespace kendra_sunday_shirts_l226_226061

def total_shirts := 22
def shirts_weekdays := 5 * 1
def shirts_after_school := 3
def shirts_saturday := 1

theorem kendra_sunday_shirts : 
  (total_shirts - 2 * (shirts_weekdays + shirts_after_school + shirts_saturday)) = 4 :=
by
  sorry

end kendra_sunday_shirts_l226_226061


namespace fedya_initial_deposit_l226_226898

theorem fedya_initial_deposit (n k : ℕ) (h₁ : k < 30) (h₂ : n * (100 - k) = 84700) : 
  n = 1100 :=
by
  sorry

end fedya_initial_deposit_l226_226898


namespace no_integers_six_digit_cyclic_permutation_l226_226302

theorem no_integers_six_digit_cyclic_permutation (n : ℕ) (a b c d e f : ℕ) (h : 10 ≤ a ∧ a < 10) :
  ¬(n = 5 ∨ n = 6 ∨ n = 8 ∧
    n * (a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f) =
    b * 10^5 + c * 10^4 + d * 10^3 + e * 10^2 + f * 10 + a) :=
by sorry

end no_integers_six_digit_cyclic_permutation_l226_226302


namespace sums_of_coordinates_of_A_l226_226053

theorem sums_of_coordinates_of_A (A B C : ℝ × ℝ) (a b : ℝ)
  (hB : B.2 = 0)
  (hC : C.1 = 0)
  (hAB : ∃ f : ℝ → ℝ, (∀ x, f x = a * x + 4 ∨ f x = 2 * x + b ∨ f x = (a / 2) * x + 8) ∧ (f A.1 = A.2) ∧ (f B.1 = B.2))
  (hBC : ∃ g : ℝ → ℝ, (∀ x, g x = a * x + 4 ∨ g x = 2 * x + b ∨ g x = (a / 2) * x + 8) ∧ (g B.1 = B.2) ∧ (g C.1 = C.2))
  (hAC : ∃ h : ℝ → ℝ, (∀ x, h x = a * x + 4 ∨ h x = 2 * x + b ∨ h x = (a / 2) * x + 8) ∧ (h A.1 = A.2) ∧ (h C.1 = C.2)) :
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sums_of_coordinates_of_A_l226_226053


namespace MaximMethod_CorrectNumber_l226_226022

theorem MaximMethod_CorrectNumber (x y : ℕ) (N : ℕ) (h_digit_x : 0 ≤ x ∧ x ≤ 9) (h_digit_y : 1 ≤ y ∧ y ≤ 9)
  (h_N : N = 10 * x + y)
  (h_condition : 1 / (10 * x + y : ℚ) = 1 / (x + y : ℚ) - 1 / (x * y : ℚ)) :
  N = 24 :=
sorry

end MaximMethod_CorrectNumber_l226_226022


namespace car_speed_ratio_l226_226880

theorem car_speed_ratio 
  (t D : ℝ) 
  (v_alpha v_beta : ℝ)
  (H1 : (v_alpha + v_beta) * t = D)
  (H2 : v_alpha * 4 = D - v_alpha * t)
  (H3 : v_beta * 1 = D - v_beta * t) : 
  v_alpha / v_beta = 2 :=
by
  sorry

end car_speed_ratio_l226_226880


namespace a7_value_l226_226673

theorem a7_value
  (a : ℕ → ℝ)
  (hx2 : ∀ n, n > 0 → a n ≠ 0)
  (slope_condition : ∀ n, n ≥ 2 → 2 * a n = 2 * a (n - 1) + 1)
  (point_condition : a 1 * 4 = 8) :
  a 7 = 5 :=
by
  sorry

end a7_value_l226_226673


namespace min_value_fraction_condition_l226_226934

noncomputable def minValue (a b : ℝ) := 1 / (2 * a) + a / (b + 1)

theorem min_value_fraction_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  minValue a b = 5 / 4 :=
by
  sorry

end min_value_fraction_condition_l226_226934


namespace points_per_other_player_l226_226474

-- Define the conditions as variables
variables (total_points : ℕ) (faye_points : ℕ) (total_players : ℕ)

-- Assume the given conditions
def conditions : Prop :=
  total_points = 68 ∧ faye_points = 28 ∧ total_players = 5

-- Define the proof problem: Prove that the points scored by each of the other players is 10
theorem points_per_other_player :
  conditions total_points faye_points total_players →
  (total_points - faye_points) / (total_players - 1) = 10 :=
by
  sorry

end points_per_other_player_l226_226474


namespace evaluate_expression_l226_226188

variables (a b c : ℝ)

theorem evaluate_expression (h1 : c = b - 20) (h2 : b = a + 4) (h3 : a = 2)
  (h4 : a^2 + a ≠ 0) (h5 : b^2 - 6 * b + 8 ≠ 0) (h6 : c^2 + 12 * c + 36 ≠ 0):
  (a^2 + 2 * a) / (a^2 + a) * (b^2 - 4) / (b^2 - 6 * b + 8) * (c^2 + 16 * c + 64) / (c^2 + 12 * c + 36) = 3 / 4 :=
by sorry

end evaluate_expression_l226_226188


namespace handshake_count_l226_226472

-- Definitions based on conditions
def groupA_size : ℕ := 25
def groupB_size : ℕ := 15

-- Total number of handshakes is calculated as product of their sizes
def total_handshakes : ℕ := groupA_size * groupB_size

-- The theorem we need to prove
theorem handshake_count : total_handshakes = 375 :=
by
  -- skipped proof
  sorry

end handshake_count_l226_226472


namespace cost_of_fencing_per_meter_in_cents_l226_226372

-- Definitions for the conditions
def ratio_length_width : ℕ := 3
def ratio_width_length : ℕ := 2
def total_area : ℕ := 3750
def total_fencing_cost : ℕ := 175

-- Main theorem statement with proof omitted
theorem cost_of_fencing_per_meter_in_cents :
  (ratio_length_width = 3) →
  (ratio_width_length = 2) →
  (total_area = 3750) →
  (total_fencing_cost = 175) →
  ∃ (cost_per_meter_in_cents : ℕ), cost_per_meter_in_cents = 70 :=
by
  intros h1 h2 h3 h4
  sorry

end cost_of_fencing_per_meter_in_cents_l226_226372


namespace election_votes_l226_226823

theorem election_votes (total_votes : ℕ) (h1 : (4 / 15) * total_votes = 48) : total_votes = 180 :=
sorry

end election_votes_l226_226823


namespace a_101_mod_49_l226_226040

def a (n : ℕ) : ℕ := 5 ^ n + 9 ^ n

theorem a_101_mod_49 : (a 101) % 49 = 0 :=
by
  -- proof to be filled here
  sorry

end a_101_mod_49_l226_226040


namespace tangent_line_at_P_l226_226759

/-- Define the center of the circle as the origin and point P --/
def center : ℝ × ℝ := (0, 0)

def P : ℝ × ℝ := (1, 2)

/-- Define the circle with radius squared r², where the radius passes through point P leading to r² = 5 --/
def circle_equation (x y : ℝ) : Prop := x * x + y * y = 5

/-- Define the condition that point P lies on the circle centered at the origin --/
def P_on_circle : Prop := circle_equation P.1 P.2

/-- Define what it means for a line to be the tangent at point P --/
def tangent_line (x y : ℝ) : Prop := x + 2 * y - 5 = 0

theorem tangent_line_at_P : P_on_circle → ∃ x y, tangent_line x y :=
by {
  sorry
}

end tangent_line_at_P_l226_226759


namespace probability_of_draw_l226_226175

-- Define probabilities
def P_A_wins : ℝ := 0.4
def P_A_not_loses : ℝ := 0.9

-- Theorem statement
theorem probability_of_draw : P_A_not_loses = P_A_wins + 0.5 :=
by
  -- Proof is skipped
  sorry

end probability_of_draw_l226_226175


namespace afternoon_to_morning_ratio_l226_226378

theorem afternoon_to_morning_ratio
  (A : ℕ) (M : ℕ)
  (h1 : A = 340)
  (h2 : A + M = 510) :
  A / M = 2 :=
by
  sorry

end afternoon_to_morning_ratio_l226_226378


namespace max_parts_three_planes_divide_space_l226_226968

-- Define the conditions given in the problem.
-- Condition 1: A plane divides the space into two parts.
def plane_divides_space (n : ℕ) : ℕ := 2

-- Condition 2: Two planes can divide the space into either three or four parts.
def two_planes_divide_space (n : ℕ) : ℕ := if n = 2 then 3 else 4

-- Condition 3: Three planes can divide the space into four, six, seven, or eight parts.
def three_planes_divide_space (n : ℕ) : ℕ := if n = 4 then 8 else sorry

-- The statement to be proved.
theorem max_parts_three_planes_divide_space : 
  ∃ n, three_planes_divide_space n = 8 := by
  use 4
  sorry

end max_parts_three_planes_divide_space_l226_226968


namespace solve_for_x_l226_226942

theorem solve_for_x (x : ℚ) : 
  5*x + 9*x = 450 - 10*(x - 5) -> x = 125/6 :=
by
  sorry

end solve_for_x_l226_226942


namespace question1_question2_l226_226252

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

end question1_question2_l226_226252


namespace last_four_digits_of_3_power_24000_l226_226656

theorem last_four_digits_of_3_power_24000 (h : 3^800 ≡ 1 [MOD 2000]) : 3^24000 ≡ 1 [MOD 2000] :=
  by sorry

end last_four_digits_of_3_power_24000_l226_226656


namespace students_only_in_math_l226_226723

-- Define the sets and their cardinalities according to the problem conditions
def total_students : ℕ := 120
def math_students : ℕ := 85
def foreign_language_students : ℕ := 65
def sport_students : ℕ := 50
def all_three_classes : ℕ := 10

-- Define the Lean theorem to prove the number of students taking only a math class
theorem students_only_in_math (total : ℕ) (M F S : ℕ) (MFS : ℕ)
  (H_total : total = 120)
  (H_M : M = 85)
  (H_F : F = 65)
  (H_S : S = 50)
  (H_MFS : MFS = 10) :
  (M - (MFS + MFS - MFS) = 35) :=
sorry

end students_only_in_math_l226_226723


namespace sin_neg_4_div_3_pi_l226_226189

theorem sin_neg_4_div_3_pi : Real.sin (- (4 / 3) * Real.pi) = Real.sqrt 3 / 2 :=
by sorry

end sin_neg_4_div_3_pi_l226_226189


namespace calc_expression_l226_226086

theorem calc_expression : 
  (abs (Real.sqrt 2 - Real.sqrt 3) + 2 * Real.cos (Real.pi / 4) - Real.sqrt 2 * Real.sqrt 6 = -Real.sqrt 3) :=
by
  -- Given that sqrt(3) > sqrt(2)
  have h1 : Real.sqrt 3 > Real.sqrt 2 := by sorry
  -- And cos(45°) = sqrt(2)/2
  have h2 : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 := by sorry
  -- Now prove the expression equivalency
  sorry

end calc_expression_l226_226086


namespace part1_part2_l226_226562

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (m n : ℝ) : f (m + n) = f m * f n
axiom positive_property (x : ℝ) (h : x > 0) : 0 < f x ∧ f x < 1

theorem part1 (x : ℝ) : f 0 = 1 ∧ (x < 0 → f x > 1) := by
  sorry

theorem part2 (x : ℝ) : 
  f (2 * x^2 - 4 * x - 1) < 1 ∧ f (x - 1) < 1 → x < -1/2 ∨ x > 2 := by
  sorry

end part1_part2_l226_226562


namespace expand_expression_l226_226077

variable (x : ℝ)

theorem expand_expression : 5 * (x + 3) * (2 * x - 4) = 10 * x^2 + 10 * x - 60 :=
by
  sorry

end expand_expression_l226_226077


namespace brownies_pieces_l226_226352

theorem brownies_pieces (pan_length pan_width piece_length piece_width : ℕ)
  (h_pan_dims : pan_length = 15) (h_pan_width : pan_width = 25)
  (h_piece_length : piece_length = 3) (h_piece_width : piece_width = 5) :
  (pan_length * pan_width) / (piece_length * piece_width) = 25 :=
by
  sorry

end brownies_pieces_l226_226352


namespace trace_ellipse_l226_226552

open Complex

theorem trace_ellipse (z : ℂ) (θ : ℝ) (h₁ : z = 3 * exp (θ * I))
  (h₂ : abs z = 3) : ∃ a b : ℝ, ∀ θ, z + 1/z = a * Real.cos θ + b * (I * Real.sin θ) :=
sorry

end trace_ellipse_l226_226552


namespace oranges_thrown_away_l226_226958

theorem oranges_thrown_away (initial_oranges new_oranges current_oranges : ℕ) (x : ℕ) 
  (h1 : initial_oranges = 50)
  (h2 : new_oranges = 24)
  (h3 : current_oranges = 34) : 
  initial_oranges - x + new_oranges = current_oranges → x = 40 :=
by
  intros h
  rw [h1, h2, h3] at h
  sorry

end oranges_thrown_away_l226_226958


namespace time_to_install_rest_of_windows_l226_226560

-- Definition of the given conditions:
def num_windows_needed : ℕ := 10
def num_windows_installed : ℕ := 6
def install_time_per_window : ℕ := 5

-- Statement that we aim to prove:
theorem time_to_install_rest_of_windows :
  install_time_per_window * (num_windows_needed - num_windows_installed) = 20 := by
  sorry

end time_to_install_rest_of_windows_l226_226560


namespace fuel_cost_per_liter_l226_226926

def service_cost_per_vehicle : ℝ := 2.20
def num_minivans : ℕ := 3
def num_trucks : ℕ := 2
def total_cost : ℝ := 347.7
def mini_van_tank_capacity : ℝ := 65
def truck_tank_increase : ℝ := 1.2
def truck_tank_capacity : ℝ := mini_van_tank_capacity * (1 + truck_tank_increase)

theorem fuel_cost_per_liter : 
  let total_service_cost := (num_minivans + num_trucks) * service_cost_per_vehicle
  let total_capacity_minivans := num_minivans * mini_van_tank_capacity
  let total_capacity_trucks := num_trucks * truck_tank_capacity
  let total_fuel_capacity := total_capacity_minivans + total_capacity_trucks
  let fuel_cost := total_cost - total_service_cost
  let cost_per_liter := fuel_cost / total_fuel_capacity
  cost_per_liter = 0.70 := 
  sorry

end fuel_cost_per_liter_l226_226926


namespace arithmetic_sequence_a1_l226_226730

/-- In an arithmetic sequence {a_n],
given a_3 = -2, a_n = 3 / 2, and S_n = -15 / 2,
prove that the value of a_1 is -3 or -19 / 6.
-/
theorem arithmetic_sequence_a1 (a_n S_n : ℕ → ℚ)
  (h1 : a_n 3 = -2)
  (h2 : ∃ n : ℕ, a_n n = 3 / 2)
  (h3 : ∃ n : ℕ, S_n n = -15 / 2) :
  ∃ x : ℚ, x = -3 ∨ x = -19 / 6 :=
by 
  sorry

end arithmetic_sequence_a1_l226_226730


namespace solve_a₃_l226_226399

noncomputable def geom_seq (a₁ a₅ a₃ : ℝ) : Prop :=
a₁ = 1 / 9 ∧ a₅ = 9 ∧ a₁ * a₅ = a₃^2

theorem solve_a₃ : ∃ a₃ : ℝ, geom_seq (1/9) 9 a₃ ∧ a₃ = 1 :=
by
  sorry

end solve_a₃_l226_226399


namespace present_age_of_son_l226_226641

theorem present_age_of_son (M S : ℕ) (h1 : M = S + 32) (h2 : M + 2 = 2 * (S + 2)) : S = 30 :=
by
  sorry

end present_age_of_son_l226_226641


namespace binomial_coeff_sum_l226_226330

theorem binomial_coeff_sum :
  ∀ a b : ℝ, 15 * a^4 * b^2 = 135 ∧ 6 * a^5 * b = -18 →
  (a + b) ^ 6 = 64 :=
by
  intros a b h
  sorry

end binomial_coeff_sum_l226_226330


namespace value_of_k_l226_226953

theorem value_of_k (a b k : ℝ) (h1 : 2 * a = k) (h2 : 3 * b = k) (h3 : 2 * a + b = a * b) (h4 : k ≠ 1) : k = 8 := 
sorry

end value_of_k_l226_226953


namespace ambulance_ride_cost_correct_l226_226700

noncomputable def total_bill : ℝ := 18000
noncomputable def medication_percentage : ℝ := 0.35
noncomputable def imaging_percentage : ℝ := 0.15
noncomputable def surgery_percentage : ℝ := 0.25
noncomputable def overnight_stays_percentage : ℝ := 0.10
noncomputable def doctors_fees_percentage : ℝ := 0.05

noncomputable def food_fee : ℝ := 300
noncomputable def consultation_fee : ℝ := 450
noncomputable def physical_therapy_fee : ℝ := 600

noncomputable def medication_cost : ℝ := medication_percentage * total_bill
noncomputable def imaging_cost : ℝ := imaging_percentage * total_bill
noncomputable def surgery_cost : ℝ := surgery_percentage * total_bill
noncomputable def overnight_stays_cost : ℝ := overnight_stays_percentage * total_bill
noncomputable def doctors_fees_cost : ℝ := doctors_fees_percentage * total_bill

noncomputable def percentage_based_costs : ℝ :=
  medication_cost + imaging_cost + surgery_cost + overnight_stays_cost + doctors_fees_cost

noncomputable def fixed_costs : ℝ :=
  food_fee + consultation_fee + physical_therapy_fee

noncomputable def total_known_costs : ℝ :=
  percentage_based_costs + fixed_costs

noncomputable def ambulance_ride_cost : ℝ :=
  total_bill - total_known_costs

theorem ambulance_ride_cost_correct :
  ambulance_ride_cost = 450 := by
  sorry

end ambulance_ride_cost_correct_l226_226700


namespace average_difference_l226_226820

theorem average_difference :
  let a1 := 20
  let a2 := 40
  let a3 := 60
  let b1 := 10
  let b2 := 70
  let b3 := 13
  (a1 + a2 + a3) / 3 - (b1 + b2 + b3) / 3 = 9 := by
sorry

end average_difference_l226_226820


namespace cylinder_surface_area_l226_226485

variable (height1 height2 radius1 radius2 : ℝ)
variable (π : ℝ)
variable (C1 : height1 = 6 * π)
variable (C2 : radius1 = 3)
variable (C3 : height2 = 4 * π)
variable (C4 : radius2 = 2)

theorem cylinder_surface_area : 
  (6 * π * 4 * π + 2 * π * radius1 ^ 2) = 24 * π ^ 2 + 18 * π ∨
  (4 * π * 6 * π + 2 * π * radius2 ^ 2) = 24 * π ^ 2 + 8 * π :=
by
  intros
  sorry

end cylinder_surface_area_l226_226485


namespace orchard_total_mass_l226_226642

def num_gala_trees := 20
def yield_gala_tree := 120
def num_fuji_trees := 10
def yield_fuji_tree := 180
def num_redhaven_trees := 30
def yield_redhaven_tree := 55
def num_elberta_trees := 15
def yield_elberta_tree := 75

def total_mass_gala := num_gala_trees * yield_gala_tree
def total_mass_fuji := num_fuji_trees * yield_fuji_tree
def total_mass_redhaven := num_redhaven_trees * yield_redhaven_tree
def total_mass_elberta := num_elberta_trees * yield_elberta_tree

def total_mass_fruit := total_mass_gala + total_mass_fuji + total_mass_redhaven + total_mass_elberta

theorem orchard_total_mass : total_mass_fruit = 6975 := by
  sorry

end orchard_total_mass_l226_226642


namespace find_original_selling_price_l226_226319

variable (SP : ℝ)
variable (CP : ℝ := 10000)
variable (discounted_SP : ℝ := 0.9 * SP)
variable (profit : ℝ := 0.08 * CP)

theorem find_original_selling_price :
  discounted_SP = CP + profit → SP = 12000 := by
sorry

end find_original_selling_price_l226_226319


namespace area_reflected_arcs_l226_226570

theorem area_reflected_arcs (s : ℝ) (h : s = 2) : 
  ∃ A, A = 2 * Real.pi * Real.sqrt 2 - 8 :=
by
  -- constants
  let r := Real.sqrt (2 * Real.sqrt 2)
  let sector_area := Real.pi * r^2 / 8
  let triangle_area := 1 -- Equilateral triangle properties
  let reflected_arc_area := sector_area - triangle_area
  let total_area := 8 * reflected_arc_area
  use total_area
  sorry

end area_reflected_arcs_l226_226570


namespace train_and_car_combined_time_l226_226856

theorem train_and_car_combined_time (car_time : ℝ) (train_time : ℝ) 
  (h1 : train_time = car_time + 2) (h2 : car_time = 4.5) : 
  car_time + train_time = 11 := 
by 
  -- Proof goes here
  sorry

end train_and_car_combined_time_l226_226856


namespace sqrt_11_bounds_l226_226744

theorem sqrt_11_bounds : ∃ a : ℤ, a < Real.sqrt 11 ∧ Real.sqrt 11 < a + 1 ∧ a = 3 := 
by
  sorry

end sqrt_11_bounds_l226_226744


namespace log_eq_l226_226246

theorem log_eq {a b : ℝ} (h₁ : a = Real.log 256 / Real.log 4) (h₂ : b = Real.log 27 / Real.log 3) : 
  a = (4 / 3) * b :=
by
  sorry

end log_eq_l226_226246


namespace man_l226_226143

theorem man's_age_ratio_father (M F : ℕ) (hF : F = 60)
  (h_age_relationship : M + 12 = (F + 12) / 2) :
  M / F = 2 / 5 :=
by
  sorry

end man_l226_226143


namespace tan_to_sin_cos_l226_226630

theorem tan_to_sin_cos (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.cos α = 2 / 5 := 
sorry

end tan_to_sin_cos_l226_226630


namespace min_combined_number_of_horses_and_ponies_l226_226328

theorem min_combined_number_of_horses_and_ponies :
  ∃ P H : ℕ, H = P + 4 ∧ (∃ k : ℕ, k = (3 * P) / 10 ∧ k = 16 * (3 * P) / (16 * 10) ∧ H + P = 36) :=
sorry

end min_combined_number_of_horses_and_ponies_l226_226328


namespace whole_numbers_between_sqrts_l226_226311

theorem whole_numbers_between_sqrts :
  let lower_bound := Real.sqrt 50
  let upper_bound := Real.sqrt 200
  let start := Nat.ceil lower_bound
  let end_ := Nat.floor upper_bound
  ∃ n, n = end_ - start + 1 ∧ n = 7 := by
  sorry

end whole_numbers_between_sqrts_l226_226311


namespace smallest_digit_never_in_units_place_of_odd_number_l226_226460

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l226_226460


namespace lagrange_intermediate_value_l226_226671

open Set

variable {a b : ℝ} (f : ℝ → ℝ)

-- Ensure that a < b for the interval [a, b]
axiom hab : a < b

-- Assume f is differentiable on [a, b]
axiom differentiable_on_I : DifferentiableOn ℝ f (Icc a b)

theorem lagrange_intermediate_value :
  ∃ (x0 : ℝ), x0 ∈ Ioo a b ∧ (deriv f x0) = (f a - f b) / (a - b) :=
sorry

end lagrange_intermediate_value_l226_226671


namespace maintain_order_time_l226_226542

theorem maintain_order_time :
  ∀ (x : ℕ), 
  (let ppl_per_min_norm := 9
   let ppl_per_min_cong := 3
   let total_people := 36 
   let teacher_time_saved := 6

   let time_without_order := total_people / ppl_per_min_cong
   let time_with_order := time_without_order - teacher_time_saved

   let ppl_passed_while_order := ppl_per_min_cong * x
   let ppl_passed_norm_order := ppl_per_min_norm * (time_with_order - x)

   ppl_passed_while_order + ppl_passed_norm_order = total_people) → 
  x = 3 :=
sorry

end maintain_order_time_l226_226542


namespace max_value_ratio_l226_226689

theorem max_value_ratio (a b c: ℝ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0) (h_eq: a * (a + b + c) = b * c) :
  (a / (b + c) ≤ (Real.sqrt 2 - 1) / 2) :=
sorry -- proof omitted

end max_value_ratio_l226_226689


namespace income_on_first_day_l226_226244

theorem income_on_first_day (income : ℕ → ℚ) (h1 : income 10 = 18)
  (h2 : ∀ n, income (n + 1) = 2 * income n) :
  income 1 = 0.03515625 :=
by
  sorry

end income_on_first_day_l226_226244


namespace proof_f_2008_l226_226922

theorem proof_f_2008 {f : ℝ → ℝ} 
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, f (3 * x + 1) = f (3 * (x + 1) + 1))
  (h3 : f (-1) = -1) : 
  f 2008 = 1 := 
by
  sorry

end proof_f_2008_l226_226922


namespace min_num_of_teams_l226_226056

theorem min_num_of_teams (num_athletes : ℕ) (max_team_size : ℕ) (h1 : num_athletes = 30) (h2 : max_team_size = 9) :
  ∃ (min_teams : ℕ), min_teams = 5 ∧ (∀ nal : ℕ, (nal > 0 ∧ num_athletes % nal = 0 ∧ nal ≤ max_team_size) → num_athletes / nal ≥ min_teams) :=
by
  sorry

end min_num_of_teams_l226_226056


namespace min_students_in_class_l226_226512

noncomputable def min_possible_students (b g : ℕ) : Prop :=
  (3 * b) / 4 = 2 * (2 * g) / 3 ∧ b = (16 * g) / 9

theorem min_students_in_class : ∃ (b g : ℕ), min_possible_students b g ∧ b + g = 25 :=
by
  sorry

end min_students_in_class_l226_226512


namespace ads_on_first_web_page_l226_226516

theorem ads_on_first_web_page 
  (A : ℕ)
  (second_page_ads : ℕ := 2 * A)
  (third_page_ads : ℕ := 2 * A + 24)
  (fourth_page_ads : ℕ := 3 * A / 2)
  (total_ads : ℕ := 68 * 3 / 2)
  (sum_of_ads : A + 2 * A + (2 * A + 24) + 3 * A / 2 = total_ads) :
  A = 12 := 
by
  sorry

end ads_on_first_web_page_l226_226516


namespace verify_magic_square_l226_226408

-- Define the grid as a 3x3 matrix
def magic_square := Matrix (Fin 3) (Fin 3) ℕ

-- Conditions for the magic square
def is_magic_square (m : magic_square) : Prop :=
  (∀ i : Fin 3, (m i 0) + (m i 1) + (m i 2) = 15) ∧
  (∀ j : Fin 3, (m 0 j) + (m 1 j) + (m 2 j) = 15) ∧
  ((m 0 0) + (m 1 1) + (m 2 2) = 15) ∧
  ((m 0 2) + (m 1 1) + (m 2 0) = 15)

-- Given specific filled numbers in the grid
def given_filled_values (m : magic_square) : Prop :=
  (m 0 1 = 5) ∧
  (m 1 0 = 2) ∧
  (m 2 2 = 8)

-- The complete grid based on the solution
def completed_magic_square : magic_square :=
  ![![4, 9, 2], ![3, 5, 7], ![8, 1, 6]]

-- The main theorem to prove
theorem verify_magic_square : 
  is_magic_square completed_magic_square ∧ 
  given_filled_values completed_magic_square := 
by 
  sorry

end verify_magic_square_l226_226408


namespace sum_congruent_mod_9_l226_226478

theorem sum_congruent_mod_9 : 
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := 
by 
  -- Proof steps here
  sorry

end sum_congruent_mod_9_l226_226478


namespace necessary_but_not_sufficient_condition_l226_226941

theorem necessary_but_not_sufficient_condition (x : ℝ) : x^2 - 4 = 0 → x + 2 = 0 :=
by
  sorry

end necessary_but_not_sufficient_condition_l226_226941


namespace minimum_route_length_l226_226717

/-- 
Given a city with the shape of a 5 × 5 square grid,
prove that the minimum length of a route that covers each street exactly once and 
returns to the starting point is 68, considering each street can be walked any number of times. 
-/
theorem minimum_route_length (n : ℕ) (h1 : n = 5) : 
  ∃ route_length : ℕ, route_length = 68 := 
sorry

end minimum_route_length_l226_226717


namespace find_digit_P_l226_226783

theorem find_digit_P (P Q R S T : ℕ) (digits : Finset ℕ) (h1 : digits = {1, 2, 3, 6, 8}) 
(h2 : P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ R ≠ S ∧ R ≠ T ∧ S ≠ T)
(h3 : P ∈ digits ∧ Q ∈ digits ∧ R ∈ digits ∧ S ∈ digits ∧ T ∈ digits)
(hPQR_div_6 : (100 * P + 10 * Q + R) % 6 = 0)
(hQRS_div_8 : (100 * Q + 10 * R + S) % 8 = 0)
(hRST_div_3 : (100 * R + 10 * S + T) % 3 = 0) : 
P = 2 := 
sorry

end find_digit_P_l226_226783


namespace value_of_m_minus_n_over_n_l226_226657

theorem value_of_m_minus_n_over_n (m n : ℚ) (h : (2/3 : ℚ) * m = (5/6 : ℚ) * n) :
  (m - n) / n = 1 / 4 := 
sorry

end value_of_m_minus_n_over_n_l226_226657


namespace remainder_when_divided_l226_226195

noncomputable def y : ℝ := 19.999999999999716
def quotient : ℝ := 76.4
def remainder : ℝ := 8

theorem remainder_when_divided (x : ℝ) (hx : x = y * 76 + y * 0.4) : x % y = 8 :=
by
  -- Proof is omitted
  sorry

end remainder_when_divided_l226_226195


namespace value_of_a_l226_226236

theorem value_of_a (a : ℤ) (h0 : 0 ≤ a) (h1 : a < 13) (h2 : 13 ∣ 12^20 + a) : a = 12 :=
by sorry

end value_of_a_l226_226236


namespace f_2019_value_l226_226646

noncomputable def f : ℕ → ℕ := sorry

theorem f_2019_value
  (h : ∀ m n : ℕ, f (m + n) ≥ f m + f (f n) - 1) :
  f 2019 = 2019 :=
sorry

end f_2019_value_l226_226646


namespace tailoring_cost_is_200_l226_226595

variables 
  (cost_first_suit : ℕ := 300)
  (total_paid : ℕ := 1400)

def cost_of_second_suit (tailoring_cost : ℕ) := 3 * cost_first_suit + tailoring_cost

theorem tailoring_cost_is_200 (T : ℕ) (h1 : cost_first_suit = 300) (h2 : total_paid = 1400) 
  (h3 : total_paid = cost_first_suit + cost_of_second_suit T) : 
  T = 200 := 
by 
  sorry

end tailoring_cost_is_200_l226_226595


namespace gcd_of_A_B_l226_226133

noncomputable def A (k : ℕ) := 2 * k
noncomputable def B (k : ℕ) := 5 * k

theorem gcd_of_A_B (k : ℕ) (h_lcm : Nat.lcm (A k) (B k) = 180) : Nat.gcd (A k) (B k) = 18 :=
by
  sorry

end gcd_of_A_B_l226_226133


namespace number_of_bought_bottle_caps_l226_226088

/-- Define the initial number of bottle caps and the final number of bottle caps --/
def initial_bottle_caps : ℕ := 40
def final_bottle_caps : ℕ := 47

/-- Proof that the number of bottle caps Joshua bought is equal to 7 --/
theorem number_of_bought_bottle_caps : final_bottle_caps - initial_bottle_caps = 7 :=
by
  sorry

end number_of_bought_bottle_caps_l226_226088


namespace forgotten_code_possibilities_l226_226987

theorem forgotten_code_possibilities:
  let digits_set := {d | ∀ n:ℕ, 0≤n ∧ n≤9 → n≠0 → 
                     (n + 4 + 4 + last_digit ≡ 0 [MOD 3]) ∨ 
                     (n + 7 + 7 + last_digit ≡ 0 [MOD 3]) ∨
                     (n + 4 + 7 + last_digit ≡ 0 [MOD 3]) ∨
                     (n + 7 + 4 + last_digit ≡ 0 [MOD 3])
                    }
  let valid_first_digits := {1, 2, 4, 5, 7, 8}
  let total_combinations := 4 * 3 + 4 * 3 -- middle combinations * valid first digit combinations
  total_combinations = 24 ∧ digits_set = valid_first_digits := by
  sorry

end forgotten_code_possibilities_l226_226987


namespace payment_for_150_books_equal_payment_number_of_books_l226_226166

/-- 
Xinhua Bookstore conditions:
- Both suppliers A and B price each book at 40 yuan. 
- Supplier A offers a 10% discount on all books.
- Supplier B offers a 20% discount on any books purchased exceeding 100 books.
-/

def price_per_book_supplier_A (n : ℕ) : ℝ := 40 * 0.9
def price_per_first_100_books_supplier_B : ℝ := 40
def price_per_excess_books_supplier_B (n : ℕ) : ℝ := 40 * 0.8

-- Prove that the payment amounts for 150 books from suppliers A and B are 5400 yuan and 5600 yuan respectively.
theorem payment_for_150_books :
  price_per_book_supplier_A 150 * 150 = 5400 ∧
  price_per_first_100_books_supplier_B * 100 + price_per_excess_books_supplier_B 50 * (150 - 100) = 5600 :=
  sorry

-- Prove the equal payment equivalence theorem for supplier A and B.
theorem equal_payment_number_of_books (x : ℕ) :
  price_per_book_supplier_A x * x = price_per_first_100_books_supplier_B * 100 + price_per_excess_books_supplier_B (x - 100) * (x - 100) → x = 200 :=
  sorry

end payment_for_150_books_equal_payment_number_of_books_l226_226166


namespace curve_touch_all_Ca_l226_226059

theorem curve_touch_all_Ca (a : ℝ) (a_pos : a > 0) (x y : ℝ) :
  ( (y - a^2)^2 = x^2 * (a^2 - x^2) ) → (y = (3 / 4) * x^2) :=
by
  sorry

end curve_touch_all_Ca_l226_226059


namespace x_cubed_lt_one_of_x_lt_one_abs_x_lt_one_of_x_lt_one_l226_226966

variable {x : ℝ}

theorem x_cubed_lt_one_of_x_lt_one (hx : x < 1) : x^3 < 1 :=
sorry

theorem abs_x_lt_one_of_x_lt_one (hx : x < 1) : |x| < 1 :=
sorry

end x_cubed_lt_one_of_x_lt_one_abs_x_lt_one_of_x_lt_one_l226_226966


namespace find_k_l226_226991

theorem find_k 
    (x y k : ℝ)
    (h1 : 1.5 * x + y = 20)
    (h2 : -4 * x + y = k)
    (hx : x = -6) :
    k = 53 :=
by
  sorry

end find_k_l226_226991


namespace problem_1_problem_2_l226_226249

-- Problem I
theorem problem_1 (x : ℝ) (h : |x - 2| + |x - 1| < 4) : (-1/2 : ℝ) < x ∧ x < 7/2 :=
sorry

-- Problem II
theorem problem_2 (a : ℝ) (h : ∀ x : ℝ, |x - a| + |x - 1| ≥ 2) : a ≤ -1 ∨ a ≥ 3 :=
sorry

end problem_1_problem_2_l226_226249


namespace container_volumes_l226_226105

theorem container_volumes (a r : ℝ) (h1 : (2 * a)^3 = (4 / 3) * Real.pi * r^3) :
  ((2 * a + 2)^3 > (4 / 3) * Real.pi * (r + 1)^3) :=
by sorry

end container_volumes_l226_226105


namespace total_gems_in_chest_l226_226108

theorem total_gems_in_chest (diamonds rubies : ℕ) 
  (h_diamonds : diamonds = 45)
  (h_rubies : rubies = 5110) : 
  diamonds + rubies = 5155 := 
by 
  sorry

end total_gems_in_chest_l226_226108


namespace added_number_after_doubling_l226_226826

theorem added_number_after_doubling (original_number : ℕ) (result : ℕ) (added_number : ℕ) 
  (h1 : original_number = 7)
  (h2 : 3 * (2 * original_number + added_number) = result)
  (h3 : result = 69) :
  added_number = 9 :=
by
  sorry

end added_number_after_doubling_l226_226826


namespace find_square_side_length_l226_226770

noncomputable def square_side_length (a : ℝ) : Prop :=
  let angle_deg := 30
  let a_sqr_minus_1 := Real.sqrt (a ^ 2 - 1)
  let a_sqr_minus_4 := Real.sqrt (a ^ 2 - 4)
  let dihedral_cos := Real.cos (Real.pi / 6)  -- 30 degrees in radians
  let dihedral_sin := Real.sin (Real.pi / 6)
  let area_1 := 0.5 * a_sqr_minus_1 * a_sqr_minus_4 * dihedral_sin
  let area_2 := 0.5 * Real.sqrt (a ^ 4 - 5 * a ^ 2)
  dihedral_cos = (Real.sqrt 3 / 2) -- Using the provided angle
  ∧ dihedral_sin = 0.5
  ∧ area_1 = area_2
  ∧ a = 2 * Real.sqrt 5

-- The theorem stating that the side length of the square is 2\sqrt{5}
theorem find_square_side_length (a : ℝ) (H : square_side_length a) : a = 2 * Real.sqrt 5 := by
  sorry

end find_square_side_length_l226_226770


namespace green_eyes_count_l226_226304

noncomputable def people_count := 100
noncomputable def blue_eyes := 19
noncomputable def brown_eyes := people_count / 2
noncomputable def black_eyes := people_count / 4
noncomputable def green_eyes := people_count - (blue_eyes + brown_eyes + black_eyes)

theorem green_eyes_count : green_eyes = 6 := by
  sorry

end green_eyes_count_l226_226304


namespace quadratic_roots_identity_l226_226505

theorem quadratic_roots_identity:
  ∀ {x₁ x₂ : ℝ}, (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → (x₁ + x₂) / (x₁ * x₂) = -1/4 :=
by
  intros x₁ x₂ h
  sorry

end quadratic_roots_identity_l226_226505


namespace solve_system_of_equations_l226_226976

theorem solve_system_of_equations :
  ∃ (x y : ℝ), x + 2 * y = 5 ∧ 3 * x - y = 1 ∧ x = 1 ∧ y = 2 := 
by
  sorry

end solve_system_of_equations_l226_226976


namespace additional_money_required_l226_226172

   theorem additional_money_required (patricia_money lisa_money charlotte_money total_card_cost : ℝ) 
       (h1 : patricia_money = 6)
       (h2 : lisa_money = 5 * patricia_money)
       (h3 : lisa_money = 2 * charlotte_money)
       (h4 : total_card_cost = 100) :
     (total_card_cost - (patricia_money + lisa_money + charlotte_money) = 49) := 
   by
     sorry
   
end additional_money_required_l226_226172


namespace completing_the_square_l226_226908

theorem completing_the_square :
  ∃ d, (∀ x: ℝ, (x^2 - 6 * x + 5 = 0) → ((x - 3)^2 = d)) ∧ d = 4 :=
by
  -- proof goes here
  sorry

end completing_the_square_l226_226908


namespace mrs_hilt_total_distance_l226_226096

def total_distance_walked (d n : ℕ) : ℕ := 2 * d * n

theorem mrs_hilt_total_distance :
  total_distance_walked 30 4 = 240 :=
by
  -- Proof goes here
  sorry

end mrs_hilt_total_distance_l226_226096


namespace isosceles_vertex_angle_l226_226858

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

theorem isosceles_vertex_angle (a b θ : ℝ)
  (h1 : a = golden_ratio * b) :
  ∃ θ, θ = 36 :=
by
  sorry

end isosceles_vertex_angle_l226_226858


namespace difference_of_squares_l226_226665

theorem difference_of_squares (a b : ℕ) (h₁ : a = 69842) (h₂ : b = 30158) :
  (a^2 - b^2) / (a - b) = 100000 :=
by
  rw [h₁, h₂]
  sorry

end difference_of_squares_l226_226665


namespace penny_paid_amount_l226_226669

-- Definitions based on conditions
def bulk_price : ℕ := 5
def minimum_spend : ℕ := 40
def tax_rate : ℕ := 1
def excess_pounds : ℕ := 32

-- Expression for total calculated cost
def total_pounds := (minimum_spend / bulk_price) + excess_pounds
def cost_before_tax := total_pounds * bulk_price
def total_tax := total_pounds * tax_rate
def total_cost := cost_before_tax + total_tax

-- Required proof statement
theorem penny_paid_amount : total_cost = 240 := 
by 
  sorry

end penny_paid_amount_l226_226669


namespace smallest_N_for_triangle_sides_l226_226847

theorem smallest_N_for_triangle_sides (a b c : ℝ) (h_triangle : a + b > c) (h_a_ne_b : a ≠ b) : (a^2 + b^2) / c^2 < 1 := 
sorry

end smallest_N_for_triangle_sides_l226_226847


namespace bill_pays_sales_tax_correct_l226_226932

def take_home_salary : ℝ := 40000
def property_tax : ℝ := 2000
def gross_salary : ℝ := 50000
def income_tax (gs : ℝ) : ℝ := 0.10 * gs
def total_taxes_paid (gs th : ℝ) : ℝ := gs - th
def sales_tax (ttp it pt : ℝ) : ℝ := ttp - it - pt

theorem bill_pays_sales_tax_correct :
  sales_tax
    (total_taxes_paid gross_salary take_home_salary)
    (income_tax gross_salary)
    property_tax = 3000 :=
by sorry

end bill_pays_sales_tax_correct_l226_226932


namespace kendra_and_tony_keep_two_each_l226_226975

-- Define the conditions
def kendra_packs : Nat := 4
def tony_packs : Nat := 2
def pens_per_pack : Nat := 3
def pens_given_to_friends : Nat := 14

-- Define the total pens each has
def kendra_pens : Nat := kendra_packs * pens_per_pack
def tony_pens : Nat := tony_packs * pens_per_pack

-- Define the total pens
def total_pens : Nat := kendra_pens + tony_pens

-- Define the pens left after distribution
def pens_left : Nat := total_pens - pens_given_to_friends

-- Define the number of pens each keeps
def pens_each_kept : Nat := pens_left / 2

-- Prove the final statement
theorem kendra_and_tony_keep_two_each :
  pens_each_kept = 2 :=
by
  sorry

end kendra_and_tony_keep_two_each_l226_226975


namespace sharpening_cost_l226_226940

theorem sharpening_cost
  (trees_chopped : ℕ)
  (trees_per_sharpening : ℕ)
  (total_cost : ℕ)
  (min_trees_chopped : trees_chopped ≥ 91)
  (trees_per_sharpening_eq : trees_per_sharpening = 13)
  (total_cost_eq : total_cost = 35) :
  total_cost / (trees_chopped / trees_per_sharpening) = 5 := by
  sorry

end sharpening_cost_l226_226940


namespace solve_system_of_equations_l226_226754

theorem solve_system_of_equations 
  (x y : ℝ) 
  (h1 : x / 3 - (y + 1) / 2 = 1) 
  (h2 : 4 * x - (2 * y - 5) = 11) : 
  x = 0 ∧ y = -3 :=
  sorry

end solve_system_of_equations_l226_226754


namespace ball_bounce_height_l226_226449

theorem ball_bounce_height :
  ∃ (k : ℕ), 10 * (1 / 2) ^ k < 1 ∧ (∀ m < k, 10 * (1 / 2) ^ m ≥ 1) :=
sorry

end ball_bounce_height_l226_226449


namespace positive_irrational_less_than_one_l226_226582

theorem positive_irrational_less_than_one : 
  ∃! (x : ℝ), 
    (x = (Real.sqrt 6) / 3 ∧ Irrational x ∧ 0 < x ∧ x < 1) ∨ 
    (x = -(Real.sqrt 3) / 3 ∧ Irrational x ∧ x < 0) ∨ 
    (x = 1 / 3 ∧ ¬Irrational x ∧ 0 < x ∧ x < 1) ∨ 
    (x = Real.pi / 3 ∧ Irrational x ∧ x > 1) :=
by
  sorry

end positive_irrational_less_than_one_l226_226582


namespace ratio_length_to_width_is_3_l226_226553

-- Define the conditions given in the problem
def area_of_garden : ℕ := 768
def width_of_garden : ℕ := 16

-- Define the length calculated from the area and width
def length_of_garden := area_of_garden / width_of_garden

-- Define the ratio to be proven
def ratio_of_length_to_width := length_of_garden / width_of_garden

-- Prove that the ratio is 3:1
theorem ratio_length_to_width_is_3 :
  ratio_of_length_to_width = 3 := by
  sorry

end ratio_length_to_width_is_3_l226_226553


namespace johns_height_in_feet_l226_226251

def initial_height := 66 -- John's initial height in inches
def growth_rate := 2      -- Growth rate in inches per month
def growth_duration := 3  -- Growth duration in months
def inches_per_foot := 12 -- Conversion factor from inches to feet

def total_growth : ℕ := growth_rate * growth_duration

def final_height_in_inches : ℕ := initial_height + total_growth

-- Now, proof that the final height in feet is 6
theorem johns_height_in_feet : (final_height_in_inches / inches_per_foot) = 6 :=
by {
  -- We would provide the detailed proof here
  sorry
}

end johns_height_in_feet_l226_226251


namespace no_blonde_girls_added_l226_226231

-- Initial number of girls
def total_girls : Nat := 80
def initial_blonde_girls : Nat := 30
def black_haired_girls : Nat := 50

-- Number of blonde girls added
def blonde_girls_added : Nat := total_girls - black_haired_girls - initial_blonde_girls

theorem no_blonde_girls_added : blonde_girls_added = 0 :=
by
  sorry

end no_blonde_girls_added_l226_226231


namespace solve_for_j_l226_226504

variable (j : ℝ)
variable (h1 : j > 0)
variable (v1 : ℝ × ℝ × ℝ := (3, 4, 5))
variable (v2 : ℝ × ℝ × ℝ := (2, j, 3))
variable (v3 : ℝ × ℝ × ℝ := (2, 3, j))

theorem solve_for_j :
  |(3 * (j * j - 3 * 3) - 2 * (4 * j - 5 * 3) + 2 * (4 * 3 - 5 * j))| = 36 →
  j = (9 + Real.sqrt 585) / 6 :=
by
  sorry

end solve_for_j_l226_226504


namespace mean_score_is_82_l226_226901

noncomputable def mean_score 
  (M A m a : ℝ) 
  (hM : M = 90) 
  (hA : A = 75) 
  (hm : m / a = 4 / 5) : ℝ := 
  (M * m + A * a) / (m + a)

theorem mean_score_is_82 
  (M A m a : ℝ) 
  (hM : M = 90) 
  (hA : A = 75) 
  (hm : m / a = 4 / 5) : 
  mean_score M A m a hM hA hm = 82 := 
    sorry

end mean_score_is_82_l226_226901


namespace average_age_increase_l226_226324

theorem average_age_increase 
    (num_students : ℕ) (avg_age_students : ℕ) (age_staff : ℕ)
    (H1: num_students = 32)
    (H2: avg_age_students = 16)
    (H3: age_staff = 49) : 
    ((num_students * avg_age_students + age_staff) / (num_students + 1) - avg_age_students = 1) :=
by
  sorry

end average_age_increase_l226_226324


namespace purchase_price_l226_226357

theorem purchase_price (P : ℝ)
  (down_payment : ℝ) (monthly_payment : ℝ) (number_of_payments : ℝ)
  (interest_rate : ℝ) (total_paid : ℝ)
  (h1 : down_payment = 12)
  (h2 : monthly_payment = 10)
  (h3 : number_of_payments = 12)
  (h4 : interest_rate = 0.10714285714285714)
  (h5 : total_paid = 132) :
  P = 132 / 1.1071428571428572 :=
by
  sorry

end purchase_price_l226_226357


namespace square_area_in_right_triangle_l226_226446

theorem square_area_in_right_triangle (XY ZC : ℝ) (hXY : XY = 40) (hZC : ZC = 70) : 
  ∃ s : ℝ, s^2 = 2800 ∧ s = (40 * 70) / (XY + ZC) := 
by
  sorry

end square_area_in_right_triangle_l226_226446


namespace soda_cost_90_cents_l226_226076

theorem soda_cost_90_cents
  (b s : ℕ)
  (h1 : 3 * b + 2 * s = 360)
  (h2 : 2 * b + 4 * s = 480) :
  s = 90 :=
by
  sorry

end soda_cost_90_cents_l226_226076


namespace moli_initial_payment_l226_226402

variable (R C S M : ℕ)

-- Conditions
def condition1 : Prop := 3 * R + 7 * C + 1 * S = M
def condition2 : Prop := 4 * R + 10 * C + 1 * S = 164
def condition3 : Prop := 1 * R + 1 * C + 1 * S = 32

theorem moli_initial_payment : condition1 R C S M ∧ condition2 R C S ∧ condition3 R C S → M = 120 := by
  sorry

end moli_initial_payment_l226_226402


namespace even_function_cos_sin_l226_226590

theorem even_function_cos_sin {f : ℝ → ℝ}
  (h_even : ∀ x, f x = f (-x))
  (h_neg : ∀ x, x < 0 → f x = Real.cos (3 * x) + Real.sin (2 * x)) :
  ∀ x, x > 0 → f x = Real.cos (3 * x) - Real.sin (2 * x) := by
  sorry

end even_function_cos_sin_l226_226590


namespace sector_area_proof_l226_226612

noncomputable def sector_area (r : ℝ) (theta : ℝ) : ℝ :=
  0.5 * theta * r^2

theorem sector_area_proof
  (r : ℝ) (l : ℝ) (perimeter : ℝ) (theta : ℝ) (h1 : perimeter = 2 * r + l)
  (h2 : l = r * theta) (h3 : perimeter = 16) (h4 : theta = 2) :
  sector_area r theta = 16 := by
  sorry

end sector_area_proof_l226_226612


namespace room_length_l226_226099

theorem room_length (length width rate cost : ℝ)
    (h_width : width = 3.75)
    (h_rate : rate = 1000)
    (h_cost : cost = 20625)
    (h_eq : cost = length * width * rate) :
    length = 5.5 :=
by
  -- the proof will go here
  sorry

end room_length_l226_226099


namespace largest_among_a_b_c_l226_226048

theorem largest_among_a_b_c (x : ℝ) (h0 : 0 < x) (h1 : x < 1)
  (a : ℝ := 2 * Real.sqrt x) 
  (b : ℝ := 1 + x) 
  (c : ℝ := 1 / (1 - x)) : c > b ∧ b > a := by
  sorry

end largest_among_a_b_c_l226_226048


namespace rectangle_ratio_l226_226519

theorem rectangle_ratio (s y x : ℝ) (hs : s > 0) (hy : y > 0) (hx : x > 0)
  (h1 : s + 2 * y = 3 * s)
  (h2 : x + y = 3 * s)
  (h3 : y = s)
  (h4 : x = 2 * s) :
  x / y = 2 := by
  sorry

end rectangle_ratio_l226_226519


namespace fixed_points_l226_226992

noncomputable def f (x : ℝ) : ℝ := x^2 - x - 3

theorem fixed_points : { x : ℝ | f x = x } = { -1, 3 } :=
by
  sorry

end fixed_points_l226_226992


namespace line_intersects_ellipse_if_and_only_if_l226_226370

theorem line_intersects_ellipse_if_and_only_if (k : ℝ) (m : ℝ) :
  (∀ x, ∃ y, y = k * x + 1 ∧ (x^2 / 5 + y^2 / m = 1)) ↔ (m ≥ 1 ∧ m ≠ 5) := 
sorry

end line_intersects_ellipse_if_and_only_if_l226_226370


namespace polynomial_expansion_correct_l226_226064

def polynomial1 (z : ℤ) : ℤ := 3 * z^3 + 4 * z^2 - 5
def polynomial2 (z : ℤ) : ℤ := 4 * z^4 - 3 * z^2 + 2
def expandedPolynomial (z : ℤ) : ℤ := 12 * z^7 + 16 * z^6 - 9 * z^5 - 32 * z^4 + 6 * z^3 + 23 * z^2 - 10

theorem polynomial_expansion_correct (z : ℤ) :
  (polynomial1 z) * (polynomial2 z) = expandedPolynomial z :=
by sorry

end polynomial_expansion_correct_l226_226064


namespace bags_weight_after_removal_l226_226605

theorem bags_weight_after_removal (sugar_weight salt_weight weight_removed : ℕ) (h1 : sugar_weight = 16) (h2 : salt_weight = 30) (h3 : weight_removed = 4) :
  sugar_weight + salt_weight - weight_removed = 42 := by
  sorry

end bags_weight_after_removal_l226_226605


namespace simplify_expression_l226_226676

theorem simplify_expression :
  (1 / (1 / ((1 / 3)^1) + 1 / ((1 / 3)^2) + 1 / ((1 / 3)^3))) = 1 / 39 :=
by
  sorry

end simplify_expression_l226_226676


namespace visitor_increase_l226_226093

variable (x : ℝ) -- The percentage increase each day

theorem visitor_increase (h1 : 1.2 * (1 + x)^2 = 2.5) : 1.2 * (1 + x)^2 = 2.5 :=
by exact h1

end visitor_increase_l226_226093


namespace like_terms_exponents_l226_226267

theorem like_terms_exponents {m n : ℕ} (h1 : 4 * a * b^n = 4 * (a^1) * (b^n)) (h2 : -2 * a^m * b^4 = -2 * (a^m) * (b^4)) :
  (m = 1 ∧ n = 4) :=
by sorry

end like_terms_exponents_l226_226267


namespace min_value_expression_l226_226636

open Real

/-- 
  Given that the function y = log_a(2x+3) - 4 passes through a fixed point P and the fixed point P lies on the line l: ax + by + 7 = 0,
  prove the minimum value of 1/(a+2) + 1/(4b) is 4/9, where a > 0, a ≠ 1, and b > 0.
-/
theorem min_value_expression (a b : ℝ) (h_a : 0 < a) (h_a_ne_1 : a ≠ 1) (h_b : 0 < b)
  (h_eqn : (a * -1 + b * -4 + 7 = 0) → (a + 2 + 4 * b = 9)):
  (1 / (a + 2) + 1 / (4 * b)) = 4 / 9 :=
by
  sorry

end min_value_expression_l226_226636


namespace turtle_ran_while_rabbit_sleeping_l226_226245

-- Define the constants and variables used in the problem
def total_distance : ℕ := 1000
def rabbit_speed_multiple : ℕ := 5
def rabbit_behind_distance : ℕ := 10

-- Define a function that represents the turtle's distance run while the rabbit is sleeping
def turtle_distance_while_rabbit_sleeping (total_distance : ℕ) (rabbit_speed_multiple : ℕ) (rabbit_behind_distance : ℕ) : ℕ :=
  total_distance - total_distance / (rabbit_speed_multiple + 1)

-- Prove that the turtle ran 802 meters while the rabbit was sleeping
theorem turtle_ran_while_rabbit_sleeping :
  turtle_distance_while_rabbit_sleeping total_distance rabbit_speed_multiple rabbit_behind_distance = 802 :=
by
  -- We reserve the proof and focus only on the statement
  sorry

end turtle_ran_while_rabbit_sleeping_l226_226245


namespace problem_l226_226522

noncomputable def a : ℝ := Real.log 8 / Real.log 3
noncomputable def b : ℝ := Real.log 25 / Real.log 4
noncomputable def c : ℝ := Real.log 24 / Real.log 4

theorem problem : a < c ∧ c < b :=
by
  sorry

end problem_l226_226522


namespace playdough_cost_l226_226030

-- Definitions of the costs and quantities
def lego_cost := 250
def sword_cost := 120
def playdough_quantity := 10
def total_paid := 1940

-- Variables representing the quantities bought
def lego_quantity := 3
def sword_quantity := 7

-- Function to calculate the total cost for lego and sword
def total_lego_cost := lego_quantity * lego_cost
def total_sword_cost := sword_quantity * sword_cost

-- Variable representing the cost of playdough
variable (P : ℝ)

-- The main statement to prove
theorem playdough_cost :
  total_lego_cost + total_sword_cost + playdough_quantity * P = total_paid → P = 35 :=
by
  sorry

end playdough_cost_l226_226030


namespace sin_315_degree_l226_226130

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l226_226130


namespace equal_roots_m_eq_minus_half_l226_226461

theorem equal_roots_m_eq_minus_half (x m : ℝ) 
  (h_eq: ∀ x, ( (x * (x - 1) - (m + 1)) / ((x - 1) * (m - 1)) = x / m )) :
  m = -1/2 := by 
  sorry

end equal_roots_m_eq_minus_half_l226_226461


namespace prob_club_then_diamond_then_heart_l226_226904

noncomputable def prob_first_card_club := 13 / 52
noncomputable def prob_second_card_diamond_given_first_club := 13 / 51
noncomputable def prob_third_card_heart_given_first_club_second_diamond := 13 / 50

noncomputable def overall_probability := 
  prob_first_card_club * 
  prob_second_card_diamond_given_first_club * 
  prob_third_card_heart_given_first_club_second_diamond

theorem prob_club_then_diamond_then_heart :
  overall_probability = 2197 / 132600 :=
by
  sorry

end prob_club_then_diamond_then_heart_l226_226904


namespace no_integers_satisfying_polynomials_l226_226752

theorem no_integers_satisfying_polynomials 
: ¬ ∃ (a b c d : ℤ), a * 19^3 + b * 19^2 + c * 19 + d = 1 ∧ a * 62^3 + b * 62^2 + c * 62 + d = 2 := 
by
  sorry

end no_integers_satisfying_polynomials_l226_226752


namespace count_p_values_l226_226997

theorem count_p_values (p : ℤ) (n : ℝ) :
  (n = 16 * 10^(-p)) →
  (-4 < p ∧ p < 4) →
  ∃ m, p ∈ m ∧ (m.count = 3 ∧ m = [-2, 0, 2]) :=
by 
  sorry

end count_p_values_l226_226997


namespace find_percentage_second_alloy_l226_226558

open Real

def percentage_copper_second_alloy (percentage_alloy1: ℝ) (ounces_alloy1: ℝ) (percentage_desired_alloy: ℝ) (total_ounces: ℝ) (percentage_second_alloy: ℝ) : Prop :=
  let copper_ounces_alloy1 := percentage_alloy1 * ounces_alloy1 / 100
  let desired_copper_ounces := percentage_desired_alloy * total_ounces / 100
  let needed_copper_ounces := desired_copper_ounces - copper_ounces_alloy1
  let ounces_alloy2 := total_ounces - ounces_alloy1
  (needed_copper_ounces / ounces_alloy2) * 100 = percentage_second_alloy

theorem find_percentage_second_alloy :
  percentage_copper_second_alloy 18 45 19.75 108 21 :=
by
  sorry

end find_percentage_second_alloy_l226_226558


namespace simplify_expression_l226_226978

theorem simplify_expression :
  (360 / 24) * (10 / 240) * (6 / 3) * (9 / 18) = 5 / 8 := by
  sorry

end simplify_expression_l226_226978


namespace calculate_first_worker_time_l226_226063

theorem calculate_first_worker_time
    (T : ℝ)
    (h : 1/T + 1/4 = 1/2.2222222222222223) :
    T = 5 := sorry

end calculate_first_worker_time_l226_226063


namespace scientific_notation_of_virus_diameter_l226_226029

theorem scientific_notation_of_virus_diameter :
  0.000000102 = 1.02 * 10 ^ (-7) :=
  sorry

end scientific_notation_of_virus_diameter_l226_226029


namespace equal_share_candy_l226_226575

theorem equal_share_candy :
  let hugh : ℕ := 8
  let tommy : ℕ := 6
  let melany : ℕ := 7
  let total_candy := hugh + tommy + melany
  let number_of_people := 3
  total_candy / number_of_people = 7 :=
by
  let hugh : ℕ := 8
  let tommy : ℕ := 6
  let melany : ℕ := 7
  let total_candy := hugh + tommy + melany
  let number_of_people := 3
  show total_candy / number_of_people = 7
  sorry

end equal_share_candy_l226_226575


namespace triangle_side_lengths_l226_226807

theorem triangle_side_lengths {x : ℤ} (h₁ : x + 4 > 10) (h₂ : x + 10 > 4) (h₃ : 10 + 4 > x) :
  ∃ (n : ℕ), n = 7 :=
by
  sorry

end triangle_side_lengths_l226_226807


namespace joe_first_lift_weight_l226_226913

theorem joe_first_lift_weight (x y : ℕ) 
  (h1 : x + y = 900)
  (h2 : 2 * x = y + 300) :
  x = 400 :=
by
  sorry

end joe_first_lift_weight_l226_226913


namespace problem_1_problem_3_problem_4_l226_226670

-- Definition of the function f(x)
def f (x : ℝ) (b c : ℝ) : ℝ := (|x| * x) + (b * x) + c

-- Prove that when b > 0, f(x) is monotonically increasing on ℝ
theorem problem_1 (b c : ℝ) (h : b > 0) : 
  ∀ x y : ℝ, x < y → f x b c < f y b c :=
sorry

-- Prove that the graph of f(x) is symmetric about the point (0, c) when b = 0
theorem problem_3 (b c : ℝ) (h : b = 0) :
  ∀ x : ℝ, f x b c = f (-x) b c :=
sorry

-- Prove that when b < 0, f(x) = 0 can have three real roots
theorem problem_4 (b c : ℝ) (h : b < 0) :
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ b c = 0 ∧ f x₂ b c = 0 ∧ f x₃ b c = 0 :=
sorry

end problem_1_problem_3_problem_4_l226_226670


namespace each_student_contribution_l226_226690

-- Definitions for conditions in the problem
def numberOfStudents : ℕ := 30
def totalAmount : ℕ := 480
def numberOfFridaysInTwoMonths : ℕ := 8

-- Statement to prove
theorem each_student_contribution (numberOfStudents : ℕ) (totalAmount : ℕ) (numberOfFridaysInTwoMonths : ℕ) : 
  totalAmount / (numberOfFridaysInTwoMonths * numberOfStudents) = 2 := 
by
  sorry

end each_student_contribution_l226_226690


namespace find_m_l226_226935

theorem find_m (x₁ x₂ y₁ y₂ : ℝ) (m : ℝ) 
  (h_parabola_A : y₁ = 2 * x₁^2) 
  (h_parabola_B : y₂ = 2 * x₂^2) 
  (h_symmetry : y₂ - y₁ = 2 * (x₂^2 - x₁^2)) 
  (h_product : x₁ * x₂ = -1/2) 
  (h_midpoint : (y₂ + y₁) / 2 = (x₂ + x₁) / 2 + m) :
  m = 3 / 2 :=
by
  sorry

end find_m_l226_226935


namespace simplify_expression_l226_226502

theorem simplify_expression : (4 + 3) + (8 - 3 - 1) = 11 := by
  sorry

end simplify_expression_l226_226502


namespace solve_s_l226_226518

theorem solve_s (s : ℝ) (h_pos : 0 < s) (h_eq : s^3 = 256) : s = 4 :=
sorry

end solve_s_l226_226518


namespace determine_number_of_students_l226_226486

theorem determine_number_of_students 
  (n : ℕ) 
  (h1 : n < 600) 
  (h2 : n % 25 = 24) 
  (h3 : n % 19 = 15) : 
  n = 399 :=
by
  -- The proof will be provided here.
  sorry

end determine_number_of_students_l226_226486


namespace knowledge_competition_score_l226_226347

theorem knowledge_competition_score (x : ℕ) (hx : x ≤ 20) : 5 * x - (20 - x) ≥ 88 :=
  sorry

end knowledge_competition_score_l226_226347


namespace system_of_equations_solution_l226_226081

theorem system_of_equations_solution (x y : ℝ) 
  (h1 : x + 3 * y = 7) 
  (h2 : y = 2 * x) : 
  x = 1 ∧ y = 2 :=
by
  sorry

end system_of_equations_solution_l226_226081


namespace number_of_girls_l226_226598

theorem number_of_girls (B G : ℕ) (h1 : B + G = 30) (h2 : 2 * B / 3 + G = 18) : G = 18 :=
by
  sorry

end number_of_girls_l226_226598


namespace airplane_children_l226_226876

theorem airplane_children (total_passengers men women children : ℕ) 
    (h1 : total_passengers = 80) 
    (h2 : men = women) 
    (h3 : men = 30) 
    (h4 : total_passengers = men + women + children) : 
    children = 20 := 
by
    -- We need to show that the number of children is 20.
    sorry

end airplane_children_l226_226876


namespace not_prime_expression_l226_226120

theorem not_prime_expression (x y : ℕ) : ¬ Prime (x^8 - x^7 * y + x^6 * y^2 - x^5 * y^3 + x^4 * y^4 
  - x^3 * y^5 + x^2 * y^6 - x * y^7 + y^8) :=
sorry

end not_prime_expression_l226_226120


namespace shells_picked_in_morning_l226_226210

-- Definitions based on conditions
def total_shells : ℕ := 616
def afternoon_shells : ℕ := 324

-- The goal is to prove that morning_shells = 292
theorem shells_picked_in_morning (morning_shells : ℕ) (h : total_shells = morning_shells + afternoon_shells) : morning_shells = 292 := 
by
  sorry

end shells_picked_in_morning_l226_226210


namespace wrapping_paper_cost_l226_226782

theorem wrapping_paper_cost :
  let cost_design1 := 4 * 4 -- 20 shirt boxes / 5 shirt boxes per roll * $4.00 per roll
  let cost_design2 := 3 * 8 -- 12 XL boxes / 4 XL boxes per roll * $8.00 per roll
  let cost_design3 := 3 * 12-- 6 XXL boxes / 2 XXL boxes per roll * $12.00 per roll
  cost_design1 + cost_design2 + cost_design3 = 76
:= by
  -- Definitions
  let cost_design1 := 4 * 4
  let cost_design2 := 3 * 8
  let cost_design3 := 3 * 12
  -- Proof (To be implemented)
  sorry

end wrapping_paper_cost_l226_226782


namespace solve_inequality_l226_226116

theorem solve_inequality (a : ℝ) : (6 * x^2 + a * x - a^2 < 0) ↔
  ((a > 0) ∧ (-a / 2 < x ∧ x < a / 3)) ∨
  ((a < 0) ∧ (a / 3 < x ∧ x < -a / 2)) ∨
  ((a = 0) ∧ false) :=
by 
  sorry

end solve_inequality_l226_226116


namespace division_remainder_l226_226102

theorem division_remainder : 1234567 % 112 = 0 := 
by 
  sorry

end division_remainder_l226_226102


namespace sum_of_xyz_l226_226979

theorem sum_of_xyz (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : (x + y + z)^3 - x^3 - y^3 - z^3 = 504) : x + y + z = 9 :=
by {
  sorry
}

end sum_of_xyz_l226_226979


namespace intersection_points_count_l226_226216

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem intersection_points_count :
  ∃ y1 y2 : ℝ, y1 ≠ y2 ∧ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = g x1 ∧ f x2 = g x2) := sorry

end intersection_points_count_l226_226216


namespace division_exponent_rule_l226_226773

theorem division_exponent_rule (a : ℝ) (h : a ≠ 0) : (a^8) / (a^2) = a^6 :=
sorry

end division_exponent_rule_l226_226773


namespace certain_number_is_213_l226_226900

theorem certain_number_is_213 (n : ℕ) (h : n * 16 = 3408) : n = 213 :=
sorry

end certain_number_is_213_l226_226900


namespace A_alone_days_l226_226729

theorem A_alone_days (A B C : ℝ) (hB: B = 9) (hC: C = 7.2) 
  (h: 1 / A + 1 / B + 1 / C = 1 / 2) : A = 2 :=
by
  rw [hB, hC] at h
  sorry

end A_alone_days_l226_226729


namespace find_finite_sets_l226_226265

open Set

theorem find_finite_sets (X : Set ℝ) (h1 : X.Nonempty) (h2 : X.Finite)
  (h3 : ∀ x ∈ X, (x + |x|) ∈ X) :
  ∃ (F : Set ℝ), F.Finite ∧ (∀ x ∈ F, x < 0) ∧ X = insert 0 F :=
sorry

end find_finite_sets_l226_226265


namespace size_of_coffee_cup_l226_226434

-- Define the conditions and the final proof statement
variable (C : ℝ) (h1 : (1/4) * C) (h2 : (1/2) * C) (remaining_after_cold : (1/4) * C - 1 = 2)

theorem size_of_coffee_cup : C = 6 := by
  -- Here the proof would go, but we omit it with sorry
  sorry

end size_of_coffee_cup_l226_226434


namespace ratio_w_y_l226_226126

theorem ratio_w_y (w x y z : ℚ) 
  (h1 : w / x = 5 / 4) 
  (h2 : y / z = 4 / 3)
  (h3 : z / x = 1 / 8) : 
  w / y = 15 / 2 := 
by
  sorry

end ratio_w_y_l226_226126


namespace dressing_q_vinegar_percentage_l226_226520

/-- 
Given:
1. P is 30% vinegar and 70% oil.
2. Q is V% vinegar and the rest is oil.
3. The new dressing is produced from 10% of P and 90% of Q and is 12% vinegar.
Prove:
The percentage of vinegar in dressing Q is 10%.
-/
theorem dressing_q_vinegar_percentage (V : ℝ) (h : 0.10 * 0.30 + 0.90 * V = 0.12) : V = 0.10 :=
by 
    sorry

end dressing_q_vinegar_percentage_l226_226520


namespace product_of_three_numbers_l226_226533

theorem product_of_three_numbers:
  ∃ (a b c : ℚ), 
    a + b + c = 30 ∧ 
    a = 2 * (b + c) ∧ 
    b = 5 * c ∧ 
    a * b * c = 2500 / 9 :=
by {
  sorry
}

end product_of_three_numbers_l226_226533


namespace probability_neither_cake_nor_muffin_l226_226534

noncomputable def probability_of_neither (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ) : ℚ :=
  (total - (cake + muffin - both)) / total

theorem probability_neither_cake_nor_muffin
  (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ) (h_total : total = 100)
  (h_cake : cake = 50) (h_muffin : muffin = 40) (h_both : both = 18) :
  probability_of_neither total cake muffin both = 0.28 :=
by
  rw [h_total, h_cake, h_muffin, h_both]
  norm_num
  sorry

end probability_neither_cake_nor_muffin_l226_226534


namespace mean_points_scored_is_48_l226_226954

def class_points : List ℤ := [50, 57, 49, 57, 32, 46, 65, 28]

theorem mean_points_scored_is_48 : (class_points.sum / class_points.length) = 48 := by
  sorry

end mean_points_scored_is_48_l226_226954


namespace friends_in_group_l226_226067

theorem friends_in_group (n : ℕ) 
  (avg_before_increase : ℝ := 800) 
  (avg_after_increase : ℝ := 850) 
  (individual_rent_increase : ℝ := 800 * 0.25) 
  (original_rent : ℝ := 800) 
  (new_rent : ℝ := 1000)
  (original_total : ℝ := avg_before_increase * n) 
  (new_total : ℝ := original_total + individual_rent_increase):
  new_total = avg_after_increase * n → 
  n = 4 :=
by
  sorry

end friends_in_group_l226_226067


namespace second_player_wins_l226_226852

def num_of_piles_initial := 3
def total_stones := 10 + 15 + 20
def num_of_piles_final := total_stones
def total_moves := num_of_piles_final - num_of_piles_initial

theorem second_player_wins : total_moves % 2 = 0 :=
sorry

end second_player_wins_l226_226852


namespace Martha_points_l226_226196

def beef_cost := 3 * 11
def fv_cost := 8 * 4
def spice_cost := 3 * 6
def other_cost := 37

def total_spent := beef_cost + fv_cost + spice_cost + other_cost
def points_per_10 := 50
def bonus := 250

def increments := total_spent / 10
def points := increments * points_per_10
def total_points := points + bonus

theorem Martha_points : total_points = 850 :=
by
  sorry

end Martha_points_l226_226196


namespace paying_students_pay_7_l226_226473

/-- At a school, 40% of the students receive a free lunch. 
These lunches are paid for by making sure the price paid by the 
paying students is enough to cover everyone's meal. 
It costs $210 to feed 50 students. 
Prove that each paying student pays $7. -/
theorem paying_students_pay_7 (total_students : ℕ) 
  (free_lunch_percentage : ℤ)
  (cost_per_50_students : ℕ) : 
  free_lunch_percentage = 40 ∧ cost_per_50_students = 210 →
  ∃ (paying_students_pay : ℕ), paying_students_pay = 7 :=
by
  -- Let the proof steps and conditions be set up as follows
  -- (this part is not required, hence using sorry)
  sorry

end paying_students_pay_7_l226_226473


namespace simple_interest_rate_l226_226163

theorem simple_interest_rate (P : ℝ) (T : ℝ) (R : ℝ) (SI : ℝ) (h1 : T = 15) (h2 : SI = 3 * P) (h3 : SI = P * R * T / 100) : R = 20 :=
by 
  sorry

end simple_interest_rate_l226_226163


namespace evaluate_expression_l226_226343

theorem evaluate_expression : 
  let expr := (15 / 8) ^ 2
  let ceil_expr := Nat.ceil expr
  let mult_expr := ceil_expr * (21 / 5)
  Nat.floor mult_expr = 16 := by
  sorry

end evaluate_expression_l226_226343


namespace exclude_domain_and_sum_l226_226637

noncomputable def g (x : ℝ) : ℝ :=
  1 / (2 + 1 / (2 + 1 / x))

theorem exclude_domain_and_sum :
  { x : ℝ | x = 0 ∨ x = -1/2 ∨ x = -1/4 } = { x : ℝ | ¬(x ≠ 0 ∧ (2 + 1 / x ≠ 0) ∧ (2 + 1 / (2 + 1 / x) ≠ 0)) } ∧
  (0 + (-1 / 2) + (-1 / 4) = -3 / 4) :=
by
  sorry

end exclude_domain_and_sum_l226_226637


namespace no_ingredient_pies_max_l226_226555

theorem no_ingredient_pies_max :
  ∃ (total apple blueberry cream chocolate no_ingredient : ℕ),
    total = 48 ∧
    apple = 24 ∧
    blueberry = 16 ∧
    cream = 18 ∧
    chocolate = 12 ∧
    no_ingredient = total - (apple + blueberry + chocolate - min apple blueberry - min apple chocolate - min blueberry chocolate) - cream ∧
    no_ingredient = 10 := sorry

end no_ingredient_pies_max_l226_226555


namespace simplify_expression_l226_226466

-- Define the algebraic expressions
def expr1 (x : ℝ) := (3 * x - 4) * (x + 9)
def expr2 (x : ℝ) := (x + 6) * (3 * x + 2)
def combined_expr (x : ℝ) := expr1 x + expr2 x
def result_expr (x : ℝ) := 6 * x^2 + 43 * x - 24

-- Theorem stating the equivalence
theorem simplify_expression (x : ℝ) : combined_expr x = result_expr x := 
by 
  sorry

end simplify_expression_l226_226466


namespace find_k_value_l226_226568

theorem find_k_value :
  (∃ p q : ℝ → ℝ,
    (∀ x, p x = 3 * x + 5) ∧
    (∃ k : ℝ, (∀ x, q x = k * x + 3) ∧
      (p (-4) = -7) ∧ (q (-4) = -7) ∧ k = 2.5)) :=
by
  sorry

end find_k_value_l226_226568


namespace total_selling_price_l226_226222

theorem total_selling_price (original_price : ℝ) (discount_rate : ℝ) (tax_rate : ℝ)
    (h1 : original_price = 80) (h2 : discount_rate = 0.25) (h3 : tax_rate = 0.10) :
  let discount_amt := original_price * discount_rate
  let sale_price := original_price - discount_amt
  let tax_amt := sale_price * tax_rate
  let total_price := sale_price + tax_amt
  total_price = 66 := by
  sorry

end total_selling_price_l226_226222


namespace interest_rate_C_l226_226995

theorem interest_rate_C (P A G : ℝ) (R : ℝ) (t : ℝ := 3) (rate_A : ℝ := 0.10) :
  P = 4000 ∧ rate_A = 0.10 ∧ G = 180 →
  (P * rate_A * t + G) = P * (R / 100) * t →
  R = 11.5 :=
by
  intros h_cond h_eq
  -- proof to be filled, use the given conditions and equations
  sorry

end interest_rate_C_l226_226995


namespace projectile_height_at_time_l226_226360

theorem projectile_height_at_time
  (y : ℝ)
  (t : ℝ)
  (h_eq : y = -16 * t ^ 2 + 64 * t) :
  ∃ t₀ : ℝ, t₀ = 3 ∧ y = 49 :=
by sorry

end projectile_height_at_time_l226_226360


namespace smallest_angle_of_triangle_l226_226799

theorem smallest_angle_of_triangle (a b c : ℕ) 
    (h1 : a = 60) (h2 : b = 70) (h3 : a + b + c = 180) : 
    c = 50 ∧ min a (min b c) = 50 :=
by {
    sorry
}

end smallest_angle_of_triangle_l226_226799


namespace apples_per_pie_l226_226261

/-- Let's define the parameters given in the problem -/
def initial_apples : ℕ := 62
def apples_given_to_students : ℕ := 8
def pies_made : ℕ := 6

/-- Define the remaining apples after handing out to students -/
def remaining_apples : ℕ := initial_apples - apples_given_to_students

/-- The statement we need to prove: each pie requires 9 apples -/
theorem apples_per_pie : remaining_apples / pies_made = 9 := by
  -- Add the proof here
  sorry

end apples_per_pie_l226_226261


namespace ln_of_x_sq_sub_2x_monotonic_l226_226145

noncomputable def ln_of_x_sq_sub_2x : ℝ → ℝ := fun x => Real.log (x^2 - 2*x)

theorem ln_of_x_sq_sub_2x_monotonic : ∀ x y : ℝ, (2 < x ∧ 2 < y ∧ x ≤ y) → ln_of_x_sq_sub_2x x ≤ ln_of_x_sq_sub_2x y :=
by
    intros x y h
    sorry

end ln_of_x_sq_sub_2x_monotonic_l226_226145


namespace prob_triangle_inequality_l226_226950

theorem prob_triangle_inequality (x y z : ℕ) (h1 : 1 ≤ x ∧ x ≤ 6) (h2 : 1 ≤ y ∧ y ≤ 6) (h3 : 1 ≤ z ∧ z ≤ 6) : 
  (∃ (p : ℚ), p = 37 / 72) := 
sorry

end prob_triangle_inequality_l226_226950


namespace q_sufficient_not_necessary_for_p_l226_226481

def p (x : ℝ) : Prop := abs x < 2
def q (x : ℝ) : Prop := x^2 - x - 2 < 0

theorem q_sufficient_not_necessary_for_p (x : ℝ) : (q x → p x) ∧ ¬(p x → q x) := 
by
  sorry

end q_sufficient_not_necessary_for_p_l226_226481


namespace regular_polygons_constructible_l226_226315

-- Define a right triangle where the smaller leg is half the length of the hypotenuse
structure RightTriangle30_60_90 :=
(smaller_leg hypotenuse : ℝ)
(ratio : smaller_leg = hypotenuse / 2)

-- Define the constructibility of polygons
def canConstructPolygon (n: ℕ) : Prop :=
n = 3 ∨ n = 4 ∨ n = 6 ∨ n = 12

theorem regular_polygons_constructible (T : RightTriangle30_60_90) :
  ∀ n : ℕ, canConstructPolygon n :=
by
  intro n
  sorry

end regular_polygons_constructible_l226_226315


namespace converse_false_l226_226728

variable {a b : ℝ}

theorem converse_false : (¬ (∀ a b : ℝ, (ab = 0 → a = 0))) :=
by
  sorry

end converse_false_l226_226728


namespace jackson_sandwiches_l226_226524

noncomputable def total_sandwiches (weeks : ℕ) (miss_wed : ℕ) (miss_fri : ℕ) : ℕ :=
  let total_wednesdays := weeks - miss_wed
  let total_fridays := weeks - miss_fri
  total_wednesdays + total_fridays

theorem jackson_sandwiches : total_sandwiches 36 1 2 = 69 := by
  sorry

end jackson_sandwiches_l226_226524


namespace pool_capacity_l226_226523

-- Conditions
variables (C : ℝ) -- total capacity of the pool in gallons
variables (h1 : 300 = 0.75 * C - 0.45 * C) -- the pool requires an additional 300 gallons to be filled to 75%
variables (h2 : 300 = 0.30 * C) -- pumping in these additional 300 gallons will increase the amount of water by 30%

-- Goal
theorem pool_capacity : C = 1000 :=
by sorry

end pool_capacity_l226_226523


namespace min_regions_l226_226857

namespace CircleDivision

def k := 12

-- Theorem statement: Given exactly 12 points where at least two circles intersect,
-- the minimum number of regions into which these circles divide the plane is 14.
theorem min_regions (k := 12) : ∃ R, R = 14 :=
by
  let R := 14
  existsi R
  exact rfl

end min_regions_l226_226857


namespace max_sum_of_first_n_terms_l226_226564

variable {a : ℕ → ℝ} -- Define sequence a with index ℕ and real values
variable {d : ℝ}      -- Common difference for the arithmetic sequence

-- Conditions and question are formulated into the theorem statement
theorem max_sum_of_first_n_terms (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_diff_neg : d < 0)
  (h_a4_eq_a12 : (a 4)^2 = (a 12)^2) :
  n = 7 ∨ n = 8 := 
sorry

end max_sum_of_first_n_terms_l226_226564


namespace total_seeds_l226_226584

-- Define the conditions given in the problem
def morningMikeTomato := 50
def morningMikePepper := 30

def morningTedTomato := 2 * morningMikeTomato
def morningTedPepper := morningMikePepper / 2

def morningSarahTomato := morningMikeTomato + 30
def morningSarahPepper := morningMikePepper + 30

def afternoonMikeTomato := 60
def afternoonMikePepper := 40

def afternoonTedTomato := afternoonMikeTomato - 20
def afternoonTedPepper := afternoonMikePepper

def afternoonSarahTomato := morningSarahTomato + 20
def afternoonSarahPepper := morningSarahPepper + 10

-- Prove that the total number of seeds planted is 685
theorem total_seeds (total: Nat) : 
    total = (
        (morningMikeTomato + afternoonMikeTomato) + 
        (morningTedTomato + afternoonTedTomato) + 
        (morningSarahTomato + afternoonSarahTomato) +
        (morningMikePepper + afternoonMikePepper) + 
        (morningTedPepper + afternoonTedPepper) + 
        (morningSarahPepper + afternoonSarahPepper)
    ) := 
    by 
        have tomato_seeds := (
            morningMikeTomato + afternoonMikeTomato +
            morningTedTomato + afternoonTedTomato + 
            morningSarahTomato + afternoonSarahTomato
        )
        have pepper_seeds := (
            morningMikePepper + afternoonMikePepper +
            morningTedPepper + afternoonTedPepper + 
            morningSarahPepper + afternoonSarahPepper
        )
        have total_seeds := tomato_seeds + pepper_seeds
        sorry

end total_seeds_l226_226584


namespace parallelogram_side_length_sum_l226_226891

theorem parallelogram_side_length_sum (x y z : ℚ) 
  (h1 : 3 * x - 1 = 12)
  (h2 : 4 * z + 2 = 7 * y + 3) :
  x + y + z = 121 / 21 :=
by
  sorry

end parallelogram_side_length_sum_l226_226891


namespace wheels_in_garage_l226_226800

-- Definitions of the entities within the problem
def cars : Nat := 2
def car_wheels : Nat := 4

def riding_lawnmower : Nat := 1
def lawnmower_wheels : Nat := 4

def bicycles : Nat := 3
def bicycle_wheels : Nat := 2

def tricycle : Nat := 1
def tricycle_wheels : Nat := 3

def unicycle : Nat := 1
def unicycle_wheels : Nat := 1

-- The total number of wheels in the garage
def total_wheels :=
  (cars * car_wheels) +
  (riding_lawnmower * lawnmower_wheels) +
  (bicycles * bicycle_wheels) +
  (tricycle * tricycle_wheels) +
  (unicycle * unicycle_wheels)

-- The theorem we wish to prove
theorem wheels_in_garage : total_wheels = 22 := by
  sorry

end wheels_in_garage_l226_226800


namespace width_of_field_l226_226439

theorem width_of_field (W L : ℝ) (h1 : L = (7 / 5) * W) (h2 : 2 * L + 2 * W = 360) : W = 75 :=
sorry

end width_of_field_l226_226439


namespace root_polynomial_sum_l226_226764

theorem root_polynomial_sum {b c : ℝ} (hb : b^2 - b - 1 = 0) (hc : c^2 - c - 1 = 0) : 
  (1 / (1 - b)) + (1 / (1 - c)) = -1 := 
sorry

end root_polynomial_sum_l226_226764


namespace abs_equality_holds_if_interval_l226_226358

noncomputable def quadratic_abs_equality (x : ℝ) : Prop :=
  |x^2 - 8 * x + 12| = x^2 - 8 * x + 12

theorem abs_equality_holds_if_interval (x : ℝ) :
  quadratic_abs_equality x ↔ (x ≤ 2 ∨ x ≥ 6) :=
by
  sorry

end abs_equality_holds_if_interval_l226_226358


namespace phone_numbers_divisible_by_13_l226_226423

theorem phone_numbers_divisible_by_13 :
  ∃ (x y z : ℕ), (x < 10) ∧ (y < 10) ∧ (z < 10) ∧ (100 * x + 10 * y + z) % 13 = 0 ∧ (2 * y = x + z) :=
  sorry

end phone_numbers_divisible_by_13_l226_226423


namespace chair_capacity_l226_226068

theorem chair_capacity
  (total_chairs : ℕ)
  (total_board_members : ℕ)
  (not_occupied_fraction : ℚ)
  (occupied_people_per_chair : ℕ)
  (attending_board_members : ℕ)
  (total_chairs_eq : total_chairs = 40)
  (not_occupied_fraction_eq : not_occupied_fraction = 2/5)
  (occupied_people_per_chair_eq : occupied_people_per_chair = 2)
  (attending_board_members_eq : attending_board_members = 48)
  : total_board_members = 48 := 
by
  sorry

end chair_capacity_l226_226068


namespace intersection_M_N_l226_226742

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l226_226742


namespace machines_solution_l226_226797

theorem machines_solution (x : ℝ) (h : x > 0) :
  (1 / (x + 10) + 1 / (x + 3) + 1 / (2 * x) = 1 / x) → x = 3 / 2 := 
by
  sorry

end machines_solution_l226_226797


namespace prob_draw_l226_226593

-- Define the probabilities as constants
def prob_A_winning : ℝ := 0.4
def prob_A_not_losing : ℝ := 0.9

-- Prove that the probability of a draw is 0.5
theorem prob_draw : prob_A_not_losing - prob_A_winning = 0.5 :=
by sorry

end prob_draw_l226_226593


namespace copy_pages_cost_l226_226298

theorem copy_pages_cost :
  (7 : ℕ) * (n : ℕ) = 3500 * 4 / 7 → n = 2000 :=
by
  sorry

end copy_pages_cost_l226_226298


namespace first_team_more_points_l226_226776

/-
Conditions:
  - Beth scored 12 points.
  - Jan scored 10 points.
  - Judy scored 8 points.
  - Angel scored 11 points.
Question:
  - How many more points did the first team get than the second team?
Prove that the first team scored 3 points more than the second team.
-/

theorem first_team_more_points
  (Beth_score : ℕ)
  (Jan_score : ℕ)
  (Judy_score : ℕ)
  (Angel_score : ℕ)
  (First_team_total : ℕ := Beth_score + Jan_score)
  (Second_team_total : ℕ := Judy_score + Angel_score)
  (Beth_score_val : Beth_score = 12)
  (Jan_score_val : Jan_score = 10)
  (Judy_score_val : Judy_score = 8)
  (Angel_score_val : Angel_score = 11)
  : First_team_total - Second_team_total = 3 := by
  sorry

end first_team_more_points_l226_226776


namespace choir_member_count_l226_226369

theorem choir_member_count (n : ℕ) : 
  (n ≡ 4 [MOD 7]) ∧ 
  (n ≡ 8 [MOD 6]) ∧ 
  (50 ≤ n ∧ n ≤ 200) 
  ↔ 
  (n = 60 ∨ n = 102 ∨ n = 144 ∨ n = 186) := 
by 
  sorry

end choir_member_count_l226_226369


namespace tetrahedron_inscribed_in_pyramid_edge_length_l226_226895

noncomputable def edge_length_of_tetrahedron := (Real.sqrt 2) / 2

theorem tetrahedron_inscribed_in_pyramid_edge_length :
  let A := (0,0,0)
  let B := (1,0,0)
  let C := (1,1,0)
  let D := (0,1,0)
  let E := (0.5, 0.5, 1)
  let v₁ := (0.5, 0, 0)
  let v₂ := (1, 0.5, 0)
  let v₃ := (0, 0.5, 0)
  dist (v₁ : ℝ × ℝ × ℝ) v₂ = edge_length_of_tetrahedron ∧
  dist v₂ v₃ = edge_length_of_tetrahedron ∧
  dist v₃ v₁ = edge_length_of_tetrahedron ∧
  dist E v₁ = dist E v₂ ∧
  dist E v₂ = dist E v₃ :=
by
  sorry

end tetrahedron_inscribed_in_pyramid_edge_length_l226_226895


namespace find_coordinates_of_P_l226_226796

structure Point where
  x : Int
  y : Int

def symmetric_origin (A B : Point) : Prop :=
  B.x = -A.x ∧ B.y = -A.y

def symmetric_y_axis (A B : Point) : Prop :=
  B.x = -A.x ∧ B.y = A.y

theorem find_coordinates_of_P :
  ∀ M N P : Point, 
  M = Point.mk (-4) 3 →
  symmetric_origin M N →
  symmetric_y_axis N P →
  P = Point.mk 4 3 := 
by 
  intros M N P hM hSymN hSymP
  sorry

end find_coordinates_of_P_l226_226796


namespace wendy_walked_l226_226879

theorem wendy_walked (x : ℝ) (h1 : 19.83 = x + 10.67) : x = 9.16 :=
sorry

end wendy_walked_l226_226879


namespace angle_Z_is_90_l226_226868

theorem angle_Z_is_90 (X Y Z : ℝ) (h_sum_XY : X + Y = 90) (h_Y_is_2X : Y = 2 * X) (h_sum_angles : X + Y + Z = 180) : Z = 90 :=
by
  sorry

end angle_Z_is_90_l226_226868


namespace seating_possible_l226_226014

theorem seating_possible (n : ℕ) (guests : Fin (2 * n) → Finset (Fin (2 * n))) 
  (h1 : ∀ i, n ≤ (guests i).card)
  (h2 : ∀ i j, (i ≠ j) → i ∈ guests j → j ∈ guests i) : 
  ∃ (a b c d : Fin (2 * n)), 
    (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ d) ∧ (d ≠ a) ∧
    (a ∈ guests b) ∧ (b ∈ guests c) ∧ (c ∈ guests d) ∧ (d ∈ guests a) := 
sorry

end seating_possible_l226_226014


namespace number_of_members_l226_226697

theorem number_of_members (n : ℕ) (h : n * n = 4624) : n = 68 :=
sorry

end number_of_members_l226_226697


namespace solve_system_of_equations_l226_226507

-- Definition of the system of equations as conditions
def eq1 (x y : ℤ) : Prop := 3 * x + y = 2
def eq2 (x y : ℤ) : Prop := 2 * x - 3 * y = 27

-- The theorem claiming the solution set is { (3, -7) }
theorem solve_system_of_equations :
  ∀ x y : ℤ, eq1 x y ∧ eq2 x y ↔ (x, y) = (3, -7) :=
by
  sorry

end solve_system_of_equations_l226_226507


namespace prob_green_ball_l226_226859

-- Definitions for the conditions
def red_balls_X := 3
def green_balls_X := 7
def total_balls_X := red_balls_X + green_balls_X

def red_balls_YZ := 7
def green_balls_YZ := 3
def total_balls_YZ := red_balls_YZ + green_balls_YZ

-- The probability of selecting any container
def prob_select_container := 1 / 3

-- The probabilities of drawing a green ball from each container
def prob_green_given_X := green_balls_X / total_balls_X
def prob_green_given_YZ := green_balls_YZ / total_balls_YZ

-- The combined probability of selecting a green ball
theorem prob_green_ball : 
  prob_select_container * prob_green_given_X + 
  prob_select_container * prob_green_given_YZ + 
  prob_select_container * prob_green_given_YZ = 13 / 30 := 
  by sorry

end prob_green_ball_l226_226859


namespace toothpick_removal_l226_226758

noncomputable def removalStrategy : ℕ :=
  let numToothpicks := 60
  let numUpward1Triangles := 22
  let numDownward1Triangles := 14
  let numUpward2Triangles := 4

  -- minimum toothpicks to remove to achieve the goal
  15

theorem toothpick_removal :
  let numToothpicks := 60
  let numUpward1Triangles := 22
  let numDownward1Triangles := 14
  let numUpward2Triangles := 4
  removalStrategy = 15 := by
  sorry

end toothpick_removal_l226_226758


namespace interest_rate_difference_correct_l226_226219

noncomputable def interest_rate_difference (P r R T : ℝ) :=
  let I := P * r * T
  let I' := P * R * T
  (I' - I) = 140

theorem interest_rate_difference_correct:
  ∀ (P r R T : ℝ),
  P = 1000 ∧ T = 7 ∧ interest_rate_difference P r R T →
  (R - r) = 0.02 :=
by
  intros P r R T h
  sorry

end interest_rate_difference_correct_l226_226219


namespace range_of_m_l226_226208

open Set

noncomputable def setA : Set ℝ := {y | ∃ x : ℝ, y = 2^x / (2^x + 1)}
noncomputable def setB (m : ℝ) : Set ℝ := {y | ∃ x : ℝ, x ∈ Icc (-1 : ℝ) (1 : ℝ) ∧ y = (1 / 3) * x + m}

theorem range_of_m {m : ℝ} (p q : Prop) :
  p ↔ ∃ x : ℝ, x ∈ setA →
  q ↔ ∃ x : ℝ, x ∈ setB m →
  ((p → q) ∧ ¬(q → p)) ↔ (1 / 3 < m ∧ m < 2 / 3) :=
by
  sorry

end range_of_m_l226_226208


namespace find_constants_l226_226348

noncomputable def f (x : ℕ) (a c : ℕ) : ℝ :=
  if x < a then c / Real.sqrt x else c / Real.sqrt a

theorem find_constants (a c : ℕ) (h₁ : f 4 a c = 30) (h₂ : f a a c = 5) : 
  c = 60 ∧ a = 144 := 
by
  sorry

end find_constants_l226_226348


namespace fish_total_after_transfer_l226_226556

-- Definitions of the initial conditions
def lilly_initial : ℕ := 10
def rosy_initial : ℕ := 9
def jack_initial : ℕ := 15
def fish_transferred : ℕ := 2

-- Total fish after Lilly transfers 2 fish to Jack
theorem fish_total_after_transfer : (lilly_initial - fish_transferred) + rosy_initial + (jack_initial + fish_transferred) = 34 := by
  sorry

end fish_total_after_transfer_l226_226556


namespace shopkeeper_profit_percent_l226_226696

theorem shopkeeper_profit_percent (cost_price profit : ℝ) (h1 : cost_price = 960) (h2 : profit = 40) : 
  (profit / cost_price) * 100 = 4.17 :=
by
  sorry

end shopkeeper_profit_percent_l226_226696


namespace complex_exp_identity_l226_226197

theorem complex_exp_identity (i : ℂ) (h : i^2 = -1) : (1 + i)^20 - (1 - i)^20 = 0 := by
  sorry

end complex_exp_identity_l226_226197


namespace sector_area_is_2pi_l226_226052

/-- Problem Statement: Prove that the area of a sector of a circle with radius 4 and central
    angle 45° (or π/4 radians) is 2π. -/
theorem sector_area_is_2pi (r : ℝ) (θ : ℝ) (h_r : r = 4) (h_θ : θ = π / 4) :
  (1 / 2) * θ * r^2 = 2 * π :=
by
  rw [h_r, h_θ]
  sorry

end sector_area_is_2pi_l226_226052


namespace necessary_but_not_sufficient_l226_226589

theorem necessary_but_not_sufficient (a : ℝ) : (a - 1 < 0 ↔ a < 1) ∧ (|a| < 1 → a < 1) ∧ ¬ (a < 1 → |a| < 1) := by
  sorry

end necessary_but_not_sufficient_l226_226589


namespace geom_seq_inverse_sum_l226_226036

theorem geom_seq_inverse_sum 
  (a_2 a_3 a_4 a_5 : ℚ) 
  (h1 : a_2 * a_5 = -3 / 4) 
  (h2 : a_2 + a_3 + a_4 + a_5 = 5 / 4) :
  1 / a_2 + 1 / a_3 + 1 / a_4 + 1 / a_5 = -4 / 3 :=
sorry

end geom_seq_inverse_sum_l226_226036


namespace therapy_sessions_l226_226437

theorem therapy_sessions (F A n : ℕ) 
  (h1 : F = A + 25)
  (h2 : F + A = 115)
  (h3 : F + (n - 1) * A = 250) : 
  n = 5 := 
by sorry

end therapy_sessions_l226_226437


namespace inequality_proof_l226_226321

variable {A B C a b c r : ℝ}

theorem inequality_proof (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hr : 0 < r) :
  (A + a + B + b) / (A + a + B + b + c + r) + (B + b + C + c) / (B + b + C + c + a + r) > (C + c + A + a) / (C + c + A + a + b + r) := 
    sorry

end inequality_proof_l226_226321


namespace calculation_101_squared_minus_99_squared_l226_226037

theorem calculation_101_squared_minus_99_squared : 101^2 - 99^2 = 400 :=
by
  sorry

end calculation_101_squared_minus_99_squared_l226_226037


namespace parabola_shifted_left_and_down_l226_226755

-- Define the initial parabola
def initial_parabola (x : ℝ) : ℝ :=
  3 * (x - 4) ^ 2 + 3

-- Define the transformation (shift 4 units to the left and 4 units down)
def transformed_parabola (x : ℝ) : ℝ :=
  initial_parabola (x + 4) - 4

-- Prove that after transformation the given parabola becomes y = 3x^2 - 1
theorem parabola_shifted_left_and_down :
  ∀ x : ℝ, transformed_parabola x = 3 * x ^ 2 - 1 := 
by 
  sorry

end parabola_shifted_left_and_down_l226_226755


namespace maximize_expression_l226_226164

noncomputable def max_value_expression (x y z : ℝ) : ℝ :=
(x^2 + x * y + y^2) * (x^2 + x * z + z^2) * (y^2 + y * z + z^2)

theorem maximize_expression (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 3) : 
    max_value_expression x y z ≤ 27 :=
sorry

end maximize_expression_l226_226164


namespace length_of_shorter_angle_trisector_l226_226939

theorem length_of_shorter_angle_trisector (BC AC : ℝ) (h1 : BC = 3) (h2 : AC = 4) :
  let AB := Real.sqrt (BC^2 + AC^2)
  let x := 2 * (12 / (4 * Real.sqrt 3 + 3))
  let PC := 2 * x
  AB = 5 ∧ PC = (32 * Real.sqrt 3 - 24) / 13 :=
by
  sorry

end length_of_shorter_angle_trisector_l226_226939


namespace city_roads_different_colors_l226_226599

-- Definitions and conditions
def Intersection (α : Type) := α × α × α

def City (α : Type) :=
  { intersections : α → Intersection α // 
    ∀ i : α, ∃ c₁ c₂ c₃ : α, intersections i = (c₁, c₂, c₃) 
    ∧ c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₃ ≠ c₁ 
  }

variables {α : Type}

-- Statement to prove that the three roads leading out of the city have different colors
theorem city_roads_different_colors (c : City α) 
  (roads_outside : α → Prop)
  (h : ∃ r₁ r₂ r₃, roads_outside r₁ ∧ roads_outside r₂ ∧ roads_outside r₃ ∧ 
  r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₃ ≠ r₁) : 
  true := 
sorry

end city_roads_different_colors_l226_226599


namespace distance_A_beats_B_l226_226409

theorem distance_A_beats_B
  (time_A time_B : ℝ)
  (dist : ℝ)
  (time_A_eq : time_A = 198)
  (time_B_eq : time_B = 220)
  (dist_eq : dist = 3) :
  (dist / time_A) * time_B - dist = 333 / 1000 :=
by
  sorry

end distance_A_beats_B_l226_226409


namespace arun_weight_upper_limit_l226_226154

theorem arun_weight_upper_limit (weight : ℝ) (avg_weight : ℝ) 
  (arun_opinion : 66 < weight ∧ weight < 72) 
  (brother_opinion : 60 < weight ∧ weight < 70) 
  (average_condition : avg_weight = 68) : weight ≤ 70 :=
by
  sorry

end arun_weight_upper_limit_l226_226154


namespace rachel_bought_3_tables_l226_226531

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

end rachel_bought_3_tables_l226_226531


namespace fraction_female_to_male_fraction_male_to_total_l226_226272

-- Define the number of male and female students
def num_male_students : ℕ := 30
def num_female_students : ℕ := 24
def total_students : ℕ := num_male_students + num_female_students

-- Prove the fraction of female students to male students
theorem fraction_female_to_male :
  (num_female_students : ℚ) / num_male_students = 4 / 5 :=
by sorry

-- Prove the fraction of male students to total students
theorem fraction_male_to_total :
  (num_male_students : ℚ) / total_students = 5 / 9 :=
by sorry

end fraction_female_to_male_fraction_male_to_total_l226_226272


namespace total_notebooks_l226_226242

theorem total_notebooks (num_boxes : ℕ) (parts_per_box : ℕ) (notebooks_per_part : ℕ) (h1 : num_boxes = 22)
  (h2 : parts_per_box = 6) (h3 : notebooks_per_part = 5) : 
  num_boxes * parts_per_box * notebooks_per_part = 660 := 
by
  sorry

end total_notebooks_l226_226242


namespace max_checkers_on_board_l226_226394

-- Define the size of the board.
def board_size : ℕ := 8

-- Define the max number of checkers per row/column.
def max_checkers_per_line : ℕ := 3

-- Define the conditions of the board.
structure BoardConfiguration :=
  (rows : Fin board_size → Fin (max_checkers_per_line + 1))
  (columns : Fin board_size → Fin (max_checkers_per_line + 1))
  (valid : ∀ (i : Fin board_size), rows i ≤ max_checkers_per_line ∧ columns i ≤ max_checkers_per_line)

-- Define the function to calculate the total number of checkers.
def total_checkers (config : BoardConfiguration) : ℕ :=
  Finset.univ.sum (λ i => config.rows i + config.columns i)

-- The theorem which states that the maximum number of checkers is 30.
theorem max_checkers_on_board : ∃ (config : BoardConfiguration), total_checkers config = 30 :=
  sorry

end max_checkers_on_board_l226_226394


namespace cubic_inequality_l226_226361

theorem cubic_inequality (p q x : ℝ) (h : x^3 + p * x + q = 0) : 4 * q * x ≤ p^2 :=
sorry

end cubic_inequality_l226_226361


namespace cryptarithm_no_solution_proof_l226_226176

def cryptarithm_no_solution : Prop :=
  ∀ (D O N K A L E V G R : ℕ),
    D ≠ O ∧ D ≠ N ∧ D ≠ K ∧ D ≠ A ∧ D ≠ L ∧ D ≠ E ∧ D ≠ V ∧ D ≠ G ∧ D ≠ R ∧
    O ≠ N ∧ O ≠ K ∧ O ≠ A ∧ O ≠ L ∧ O ≠ E ∧ O ≠ V ∧ O ≠ G ∧ O ≠ R ∧
    N ≠ K ∧ N ≠ A ∧ N ≠ L ∧ N ≠ E ∧ N ≠ V ∧ N ≠ G ∧ N ≠ R ∧
    K ≠ A ∧ K ≠ L ∧ K ≠ E ∧ K ≠ V ∧ K ≠ G ∧ K ≠ R ∧
    A ≠ L ∧ A ≠ E ∧ A ≠ V ∧ A ≠ G ∧ A ≠ R ∧
    L ≠ E ∧ L ≠ V ∧ L ≠ G ∧ L ≠ R ∧
    E ≠ V ∧ E ≠ G ∧ E ≠ R ∧
    V ≠ G ∧ V ≠ R ∧
    G ≠ R ∧
    (D * 100 + O * 10 + N) + (O * 100 + K * 10 + A) +
    (L * 1000 + E * 100 + N * 10 + A) + (V * 10000 + O * 1000 + L * 100 + G * 10 + A) =
    A * 100000 + N * 10000 + G * 1000 + A * 100 + R * 10 + A →
    false

theorem cryptarithm_no_solution_proof : cryptarithm_no_solution :=
by sorry

end cryptarithm_no_solution_proof_l226_226176


namespace mouse_cannot_eat_entire_cheese_l226_226419

-- Defining the conditions of the problem
structure Cheese :=
  (size : ℕ := 3)  -- The cube size is 3x3x3
  (central_cube_removed : Bool := true)  -- The central cube is removed

inductive CubeColor
| black
| white

structure Mouse :=
  (can_eat : CubeColor -> CubeColor -> Bool)
  (adjacency : Nat -> Nat -> Bool)

def cheese_problem (c : Cheese) (m : Mouse) : Bool := sorry

-- The main theorem: It is impossible for the mouse to eat the entire piece of cheese.
theorem mouse_cannot_eat_entire_cheese : ∀ (c : Cheese) (m : Mouse),
  cheese_problem c m = false := sorry

end mouse_cannot_eat_entire_cheese_l226_226419


namespace sum_consecutive_integers_product_1080_l226_226177

theorem sum_consecutive_integers_product_1080 :
  ∃ n : ℕ, n * (n + 1) = 1080 ∧ n + (n + 1) = 65 :=
by
  sorry

end sum_consecutive_integers_product_1080_l226_226177


namespace seating_arrangements_l226_226957

theorem seating_arrangements (n_seats : ℕ) (n_people : ℕ) (n_adj_empty : ℕ) (h1 : n_seats = 6) 
    (h2 : n_people = 3) (h3 : n_adj_empty = 2) : 
    ∃ arrangements : ℕ, arrangements = 48 := 
by
  sorry

end seating_arrangements_l226_226957


namespace m_range_l226_226147

theorem m_range (m : ℝ) :
  (∀ x : ℝ, 1 < x → 2 * x + m + 2 / (x - 1) > 0) ↔ m > -6 :=
by
  -- The proof will be provided later
  sorry

end m_range_l226_226147


namespace moss_flower_pollen_scientific_notation_l226_226483

theorem moss_flower_pollen_scientific_notation (d : ℝ) (h : d = 0.0000084) : ∃ n : ℤ, d = 8.4 * 10^n ∧ n = -6 :=
by
  use -6
  rw [h]
  simp
  sorry

end moss_flower_pollen_scientific_notation_l226_226483


namespace correct_statement_l226_226318

theorem correct_statement : ∀ (a b : ℝ), ((a ≠ b ∧ ¬(a < b ∧ ∀ x, (x = a ∨ x = b) → x = a ∨ x = b)) ∧
                                            ¬(∀ p q : ℝ, p = q → p = q) ∧
                                            ¬(∀ a : ℝ, |a| = -a → a < 0) ∧
                                            ¬(∀ a b : ℝ, (a ≠ 0 ∧ b ≠ 0 ∧ (a = -b)) → (a / b = -1))) :=
by sorry

-- Explanation of conditions:
-- a  ≠ b ensures two distinct points
-- ¬(a < b ∧ ∀ x, (x = a ∨ x = b) → x is between a and b) incorrectly rephrased as shortest distance as a line segment
-- ¬(∀ p q : ℝ, p = q → p = q) is not directly used, a minimum to refute the concept as required.
-- |a| = -a → a < 0 reinterpreted as a ≤ 0 but incorrectly stated as < 0 explicitly refuted
-- ¬(∀ a b : ℝ, a ≠ 0 and/or b ≠ 0 maintained where a / b not strictly required/misinterpreted)

end correct_statement_l226_226318


namespace total_water_carried_l226_226737

/-- Define the capacities of the four tanks in each truck -/
def tank1_capacity : ℝ := 200
def tank2_capacity : ℝ := 250
def tank3_capacity : ℝ := 300
def tank4_capacity : ℝ := 350

/-- The total capacity of one truck -/
def total_truck_capacity : ℝ := tank1_capacity + tank2_capacity + tank3_capacity + tank4_capacity

/-- Define the fill percentages for each truck -/
def fill_percentage (truck_number : ℕ) : ℝ :=
if truck_number = 1 then 1
else if truck_number = 2 then 0.75
else if truck_number = 3 then 0.5
else if truck_number = 4 then 0.25
else 0

/-- Define the amounts of water each truck carries -/
def water_carried_by_truck (truck_number : ℕ) : ℝ :=
(fill_percentage truck_number) * total_truck_capacity

/-- Prove that the total amount of water the farmer can carry in his trucks is 2750 liters -/
theorem total_water_carried : 
  water_carried_by_truck 1 + water_carried_by_truck 2 + water_carried_by_truck 3 +
  water_carried_by_truck 4 + water_carried_by_truck 5 = 2750 :=
by sorry

end total_water_carried_l226_226737


namespace problem1_problem2_l226_226920

-- Problem 1 Lean statement
theorem problem1 (x y : ℝ) (hx : x ≠ 1) (hx' : x ≠ -1) (hy : y ≠ 0) :
    (x^2 - 1) / y / ((x + 1) / y^2) = y * (x - 1) :=
sorry

-- Problem 2 Lean statement
theorem problem2 (m n : ℝ) (hm1 : m ≠ n) (hm2 : m ≠ -n) :
    m / (m + n) + n / (m - n) - 2 * m^2 / (m^2 - n^2) = -1 :=
sorry

end problem1_problem2_l226_226920


namespace balloons_lost_is_correct_l226_226602

def original_balloons : ℕ := 8
def current_balloons : ℕ := 6
def lost_balloons : ℕ := original_balloons - current_balloons

theorem balloons_lost_is_correct : lost_balloons = 2 := by
  sorry

end balloons_lost_is_correct_l226_226602


namespace compound_interest_semiannual_l226_226903

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) :=
  P * (1 + r / n) ^ (n * t)

theorem compound_interest_semiannual :
  compound_interest 150 0.20 2 1 = 181.50 :=
by
  sorry

end compound_interest_semiannual_l226_226903


namespace no_such_b_exists_l226_226028

theorem no_such_b_exists (k n : ℕ) (a : ℕ) 
  (hk : Odd k) (hn : Odd n)
  (hk_gt_one : k > 1) (hn_gt_one : n > 1) 
  (hka : k ∣ 2^a + 1) (hna : n ∣ 2^a - 1) : 
  ¬ ∃ b : ℕ, k ∣ 2^b - 1 ∧ n ∣ 2^b + 1 :=
sorry

end no_such_b_exists_l226_226028


namespace total_pages_allowed_l226_226276

noncomputable def words_total := 48000
noncomputable def words_per_page_large := 1800
noncomputable def words_per_page_small := 2400
noncomputable def pages_large := 4
noncomputable def total_pages : ℕ := 21

theorem total_pages_allowed :
  pages_large * words_per_page_large + (total_pages - pages_large) * words_per_page_small = words_total :=
  by sorry

end total_pages_allowed_l226_226276


namespace other_toys_cost_1000_l226_226187

-- Definitions of the conditions
def cost_of_other_toys : ℕ := sorry
def cost_of_lightsaber (cost_of_other_toys : ℕ) : ℕ := 2 * cost_of_other_toys
def total_spent (cost_of_lightsaber cost_of_other_toys : ℕ) : ℕ := cost_of_lightsaber + cost_of_other_toys

-- The proof goal
theorem other_toys_cost_1000 (T : ℕ) (H1 : cost_of_lightsaber T = 2 * T) 
                            (H2 : total_spent (cost_of_lightsaber T) T = 3000) : T = 1000 := by
  sorry

end other_toys_cost_1000_l226_226187


namespace dinner_time_correct_l226_226444

-- Definitions based on the conditions in the problem
def pounds_per_turkey : Nat := 16
def roasting_time_per_pound : Nat := 15  -- minutes
def num_turkeys : Nat := 2
def minutes_per_hour : Nat := 60
def latest_start_time_hours : Nat := 10

-- The total roasting time in hours
def total_roasting_time_hours : Nat := 
  (roasting_time_per_pound * pounds_per_turkey * num_turkeys) / minutes_per_hour

-- The expected dinner time
def expected_dinner_time_hours : Nat := latest_start_time_hours + total_roasting_time_hours

-- The proof problem
theorem dinner_time_correct : expected_dinner_time_hours = 18 := 
by
  -- Proof goes here
  sorry

end dinner_time_correct_l226_226444


namespace min_trips_is_157_l226_226915

theorem min_trips_is_157 :
  ∃ x y : ℕ, 31 * x + 32 * y = 5000 ∧ x + y = 157 :=
sorry

end min_trips_is_157_l226_226915


namespace final_price_is_correct_l226_226488

-- Define the conditions as constants
def price_smartphone : ℝ := 300
def price_pc : ℝ := price_smartphone + 500
def price_tablet : ℝ := price_smartphone + price_pc
def total_price : ℝ := price_smartphone + price_pc + price_tablet
def discount : ℝ := 0.10 * total_price
def price_after_discount : ℝ := total_price - discount
def sales_tax : ℝ := 0.05 * price_after_discount
def final_price : ℝ := price_after_discount + sales_tax

-- Theorem statement asserting the final price value
theorem final_price_is_correct : final_price = 2079 := by sorry

end final_price_is_correct_l226_226488


namespace determine_parallel_planes_l226_226299

-- Definition of planes and lines with parallelism
structure Plane :=
  (points : Set (ℝ × ℝ × ℝ))

structure Line :=
  (point1 point2 : ℝ × ℝ × ℝ)
  (in_plane : Plane)

def parallel_planes (α β : Plane) : Prop :=
  ∀ (l1 : Line) (l2 : Line), l1.in_plane = α → l2.in_plane = β → (l1 = l2)

def parallel_lines (l1 l2 : Line) : Prop :=
  ∀ p1 p2, l1.point1 = p1 → l1.point2 = p2 → l2.point1 = p1 → l2.point2 = p2


theorem determine_parallel_planes (α β γ : Plane)
  (h1 : parallel_planes γ α)
  (h2 : parallel_planes γ β)
  (l1 l2 : Line)
  (l1_in_alpha : l1.in_plane = α)
  (l2_in_alpha : l2.in_plane = α)
  (parallel_l1_l2 : ¬ (l1 = l2) → parallel_lines l1 l2)
  (l1_parallel_beta : ∀ l, l.in_plane = β → parallel_lines l l1)
  (l2_parallel_beta : ∀ l, l.in_plane = β → parallel_lines l l2) :
  parallel_planes α β := 
sorry

end determine_parallel_planes_l226_226299


namespace quantity_of_milk_in_original_mixture_l226_226563

variable (M W : ℕ)

-- Conditions
def ratio_original : Prop := M = 2 * W
def ratio_after_adding_water : Prop := M * 5 = 6 * (W + 10)

theorem quantity_of_milk_in_original_mixture
  (h1 : ratio_original M W)
  (h2 : ratio_after_adding_water M W) :
  M = 30 := by
  sorry

end quantity_of_milk_in_original_mixture_l226_226563


namespace john_took_11_more_l226_226094

/-- 
If Ray took 10 chickens, Ray took 6 chickens less than Mary, and 
John took 5 more chickens than Mary, then John took 11 more 
chickens than Ray. 
-/
theorem john_took_11_more (R M J : ℕ) (h1 : R = 10) 
  (h2 : R + 6 = M) (h3 : M + 5 = J) : J - R = 11 :=
by
  sorry

end john_took_11_more_l226_226094


namespace train_length_proof_l226_226608

def train_length_crosses_bridge (train_speed_kmh : ℕ) (bridge_length_m : ℕ) (crossing_time_s : ℕ) : ℕ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let distance := train_speed_ms * crossing_time_s
  distance - bridge_length_m

theorem train_length_proof : 
  train_length_crosses_bridge 72 150 20 = 250 :=
by
  let train_speed_kmh := 72
  let bridge_length_m := 150
  let crossing_time_s := 20
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let distance := train_speed_ms * crossing_time_s
  have h : distance = 400 := by sorry
  have h_eq : distance - bridge_length_m = 250 := by sorry
  exact h_eq

end train_length_proof_l226_226608


namespace A_union_B_l226_226159

noncomputable def A : Set ℝ := {x | ∃ y, y = 1 / Real.sqrt (1 - 2^x) ∧ x < 0}
noncomputable def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1) / Real.log 2 ∧ x > 0}
noncomputable def union_set : Set ℝ := {x | x < 0 ∨ x > 0}

theorem A_union_B :
  A ∪ B = union_set :=
by
  sorry

end A_union_B_l226_226159


namespace area_of_square_field_l226_226050

theorem area_of_square_field (s : ℕ) (area : ℕ) (cost_per_meter : ℕ) (total_cost : ℕ) (gate_width : ℕ) :
  (cost_per_meter = 3) →
  (total_cost = 1998) →
  (gate_width = 1) →
  (total_cost = cost_per_meter * (4 * s - 2 * gate_width)) →
  (area = s^2) →
  area = 27889 :=
by
  intros h_cost_per_meter h_total_cost h_gate_width h_cost_eq h_area_eq
  sorry

end area_of_square_field_l226_226050


namespace range_of_a_l226_226476

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then x + a / x + 7 else x + a / x - 7

theorem range_of_a (a : ℝ) (ha : 0 < a)
  (hodd : ∀ x : ℝ, f (-x) a = -f x a)
  (hcond : ∀ x : ℝ, 0 ≤ x → f x a ≥ 1 - a) :
  4 ≤ a := sorry

end range_of_a_l226_226476


namespace solution_set_fraction_inequality_l226_226065

theorem solution_set_fraction_inequality (x : ℝ) : 
  (x + 1) / (x - 1) ≤ 0 ↔ -1 ≤ x ∧ x < 1 :=
sorry

end solution_set_fraction_inequality_l226_226065


namespace factorial_mod_10_l226_226170

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the problem statement
theorem factorial_mod_10 : factorial 10 % 13 = 7 :=
by sorry

end factorial_mod_10_l226_226170


namespace daniel_pages_to_read_l226_226490

-- Definitions from conditions
def total_pages : ℕ := 980
def daniel_read_time_per_page : ℕ := 50
def emma_read_time_per_page : ℕ := 40

-- The theorem that states the solution
theorem daniel_pages_to_read (d : ℕ) :
  d = 436 ↔ daniel_read_time_per_page * d = emma_read_time_per_page * (total_pages - d) :=
by sorry

end daniel_pages_to_read_l226_226490


namespace find_candy_bars_per_week_l226_226643

-- Define the conditions
variables (x : ℕ)

-- Condition: Kim's dad buys Kim x candy bars each week
def candies_bought := 16 * x

-- Condition: Kim eats one candy bar every 4 weeks
def candies_eaten := 16 / 4

-- Condition: After 16 weeks, Kim has saved 28 candy bars
def saved_candies := 28

-- The theorem we want to prove
theorem find_candy_bars_per_week : (16 * x - (16 / 4) = 28) → x = 2 := by
  -- We will skip the actual proof for now.
  sorry

end find_candy_bars_per_week_l226_226643


namespace min_value_expression_l226_226186

theorem min_value_expression :
  ∃ x > 0, x^2 + 6 * x + 100 / x^3 = 3 * (50:ℝ)^(2/5) + 6 * (50:ℝ)^(1/5) :=
by
  sorry

end min_value_expression_l226_226186


namespace pigs_remaining_l226_226716

def initial_pigs : ℕ := 364
def pigs_joined : ℕ := 145
def pigs_moved : ℕ := 78

theorem pigs_remaining : initial_pigs + pigs_joined - pigs_moved = 431 := by
  sorry

end pigs_remaining_l226_226716


namespace robot_min_steps_l226_226457

theorem robot_min_steps {a b : ℕ} (ha : 0 < a) (hb : 0 < b) : ∃ n, n = a + b - Nat.gcd a b :=
by
  sorry

end robot_min_steps_l226_226457


namespace gcd_possible_values_l226_226571

theorem gcd_possible_values (a b : ℕ) (hab : a * b = 288) : 
  ∃ S : Finset ℕ, (∀ g : ℕ, g ∈ S ↔ ∃ p q r s : ℕ, p + r = 5 ∧ q + s = 2 ∧ g = 2^min p r * 3^min q s) 
  ∧ S.card = 14 := 
sorry

end gcd_possible_values_l226_226571


namespace calculate_sum_calculate_product_l226_226885

theorem calculate_sum : 13 + (-7) + (-6) = 0 :=
by sorry

theorem calculate_product : (-8) * (-4 / 3) * (-0.125) * (5 / 4) = -5 / 3 :=
by sorry

end calculate_sum_calculate_product_l226_226885


namespace hypotenuse_length_l226_226937

noncomputable def side_lengths_to_hypotenuse (a b : ℝ) : ℝ := 
  Real.sqrt (a^2 + b^2)

theorem hypotenuse_length 
  (AB BC : ℝ) 
  (h1 : Real.sqrt (AB * BC) = 8) 
  (h2 : (1 / 2) * AB * BC = 48) :
  side_lengths_to_hypotenuse AB BC = 4 * Real.sqrt 13 :=
by
  sorry

end hypotenuse_length_l226_226937


namespace inequality_proof_l226_226489

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c ≤ 3) : 
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l226_226489


namespace find_sum_of_cubes_l226_226140

theorem find_sum_of_cubes (a b c : ℝ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : (a^3 + 9) / a = (b^3 + 9) / b)
  (h₅ : (b^3 + 9) / b = (c^3 + 9) / c) :
  a^3 + b^3 + c^3 = -27 :=
by
  sorry

end find_sum_of_cubes_l226_226140


namespace Jessie_points_l226_226119

theorem Jessie_points (total_points team_points : ℕ) (players_points : ℕ) (P Q R : ℕ) (eq1 : total_points = 311) (eq2 : players_points = 188) (eq3 : team_points - players_points = 3 * P) (eq4 : P = Q) (eq5 : Q = R) : Q = 41 :=
by
  sorry

end Jessie_points_l226_226119


namespace suitable_survey_method_l226_226411

-- Definitions based on conditions
def large_population (n : ℕ) : Prop := n > 10000  -- Example threshold for large population
def impractical_comprehensive_survey : Prop := true  -- Given in condition

-- The statement of the problem
theorem suitable_survey_method (n : ℕ) (h1 : large_population n) (h2 : impractical_comprehensive_survey) : 
  ∃ method : String, method = "sampling survey" :=
sorry

end suitable_survey_method_l226_226411


namespace initial_population_l226_226772

theorem initial_population (P : ℝ) (h : P * (1.24 : ℝ)^2 = 18451.2) : P = 12000 :=
by
  sorry

end initial_population_l226_226772


namespace units_digit_of_square_ne_2_l226_226333

theorem units_digit_of_square_ne_2 (n : ℕ) : (n * n) % 10 ≠ 2 :=
sorry

end units_digit_of_square_ne_2_l226_226333


namespace integer_values_of_x_in_triangle_l226_226200

theorem integer_values_of_x_in_triangle (x : ℝ) :
  (x + 14 > 38 ∧ x + 38 > 14 ∧ 14 + 38 > x) → 
  ∃ (n : ℕ), n = 27 ∧ ∀ m : ℕ, (24 < m ∧ m < 52 ↔ (m : ℝ) > 24 ∧ (m : ℝ) < 52) :=
by {
  sorry
}

end integer_values_of_x_in_triangle_l226_226200


namespace great_grandson_age_l226_226012

theorem great_grandson_age (n : ℕ) : 
  ∃ n, (n * (n + 1)) / 2 = 666 :=
by
  -- Solution steps would go here
  sorry

end great_grandson_age_l226_226012


namespace find_ratio_l226_226712

-- Definitions
noncomputable def cost_per_gram_A : ℝ := 0.01
noncomputable def cost_per_gram_B : ℝ := 0.008
noncomputable def new_cost_per_gram_A : ℝ := 0.011
noncomputable def new_cost_per_gram_B : ℝ := 0.0072

def total_weight : ℝ := 1000

-- Theorem statement
theorem find_ratio (x y : ℝ) (h1 : x + y = total_weight)
    (h2 : cost_per_gram_A * x + cost_per_gram_B * y = new_cost_per_gram_A * x + new_cost_per_gram_B * y) :
    x / y = 4 / 5 :=
by
  sorry

end find_ratio_l226_226712


namespace velocity_at_1_eq_5_l226_226482

def S (t : ℝ) : ℝ := 2 * t^2 + t

theorem velocity_at_1_eq_5 : (deriv S 1) = 5 :=
by sorry

end velocity_at_1_eq_5_l226_226482


namespace denise_crayons_l226_226803

theorem denise_crayons (c : ℕ) :
  (∀ f p : ℕ, f = 30 ∧ p = 7 → c = f * p) → c = 210 :=
by
  intro h
  specialize h 30 7 ⟨rfl, rfl⟩
  exact h

end denise_crayons_l226_226803


namespace smallest_circle_equation_l226_226781

-- Definitions of the points A and B
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (3, 0)

-- The statement of the problem
theorem smallest_circle_equation : ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ 
  A.1 = -3 ∧ A.2 = 0 ∧ B.1 = 3 ∧ B.2 = 0 ∧ ((x - 0)^2 + (y - 0)^2 = 9) :=
by
  sorry

end smallest_circle_equation_l226_226781


namespace total_pieces_of_clothing_l226_226965

-- Define Kaleb's conditions
def pieces_in_one_load : ℕ := 19
def num_equal_loads : ℕ := 5
def pieces_per_load : ℕ := 4

-- The total pieces of clothing Kaleb has
theorem total_pieces_of_clothing : pieces_in_one_load + num_equal_loads * pieces_per_load = 39 :=
by
  sorry

end total_pieces_of_clothing_l226_226965


namespace probability_of_observing_color_change_l226_226073

def cycle_duration := 100
def observation_interval := 4
def change_times := [45, 50, 100]

def probability_of_change : ℚ :=
  (observation_interval * change_times.length : ℚ) / cycle_duration

theorem probability_of_observing_color_change :
  probability_of_change = 0.12 := by
  -- Proof goes here
  sorry

end probability_of_observing_color_change_l226_226073


namespace fraction_distance_walked_by_first_class_l226_226832

namespace CulturalCenterProblem

def walking_speed : ℝ := 4
def bus_speed_with_students : ℝ := 40
def bus_speed_empty : ℝ := 60

theorem fraction_distance_walked_by_first_class :
  ∃ (x : ℝ), 
    (x / walking_speed) = ((1 - x) / bus_speed_with_students) + ((1 - 2 * x) / bus_speed_empty)
    ∧ x = 5 / 37 :=
by
  sorry

end CulturalCenterProblem

end fraction_distance_walked_by_first_class_l226_226832


namespace units_digit_of_product_composites_l226_226633

def is_composite (n : ℕ) : Prop := 
  ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ n = m * k

theorem units_digit_of_product_composites (h1 : is_composite 9) (h2 : is_composite 10) (h3 : is_composite 12) :
  (9 * 10 * 12) % 10 = 0 :=
by
  sorry

end units_digit_of_product_composites_l226_226633


namespace problem_l226_226255

def f (x : ℝ) := 5 * x^3

theorem problem : f 2012 + f (-2012) = 0 := 
by
  sorry

end problem_l226_226255


namespace find_common_difference_l226_226596

variable (a₁ d : ℝ)

theorem find_common_difference
  (h1 : a₁ + (a₁ + 6 * d) = 22)
  (h2 : (a₁ + 3 * d) + (a₁ + 9 * d) = 40) :
  d = 3 := by
  sorry

end find_common_difference_l226_226596


namespace right_angled_isosceles_triangle_third_side_length_l226_226704

theorem right_angled_isosceles_triangle_third_side_length (a b c : ℝ) (h₀ : a = 50) (h₁ : b = 50) (h₂ : a + b + c = 160) : c = 60 :=
by
  -- TODO: Provide proof
  sorry

end right_angled_isosceles_triangle_third_side_length_l226_226704


namespace drink_total_amount_l226_226792

theorem drink_total_amount (total_amount: ℝ) (grape_juice: ℝ) (grape_proportion: ℝ) 
  (h1: grape_proportion = 0.20) (h2: grape_juice = 40) : total_amount = 200 :=
by
  -- Definitions and assumptions
  let calculation := grape_juice / grape_proportion
  -- Placeholder for the proof
  sorry

end drink_total_amount_l226_226792


namespace attendees_not_from_companies_l226_226162

theorem attendees_not_from_companies :
  let A := 30 
  let B := 2 * A
  let C := A + 10
  let D := C - 5
  let T := 185 
  T - (A + B + C + D) = 20 :=
by
  sorry

end attendees_not_from_companies_l226_226162


namespace speed_of_jogger_l226_226994

noncomputable def jogger_speed_problem (jogger_distance_ahead train_length train_speed_kmh time_to_pass : ℕ) :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := jogger_distance_ahead + train_length
  let relative_speed := total_distance / time_to_pass
  let jogger_speed_ms := train_speed_ms - relative_speed
  let jogger_speed_kmh := jogger_speed_ms * 3600 / 1000
  jogger_speed_kmh

theorem speed_of_jogger :
  jogger_speed_problem 240 210 45 45 = 9 :=
by
  sorry

end speed_of_jogger_l226_226994


namespace vertical_asymptote_singleton_l226_226018

theorem vertical_asymptote_singleton (c : ℝ) :
  (∃ x, (x^2 - 2 * x + c) = 0 ∧ ((x - 1) * (x + 3) = 0) ∧ (x ≠ 1 ∨ x ≠ -3)) 
  ↔ (c = 1 ∨ c = -15) :=
by
  sorry

end vertical_asymptote_singleton_l226_226018


namespace tan_of_alpha_l226_226007

theorem tan_of_alpha
  (α : ℝ)
  (h1 : Real.sin (α + Real.pi / 2) = 1 / 3)
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.tan α = 2 * Real.sqrt 2 := 
sorry

end tan_of_alpha_l226_226007


namespace solution_valid_l226_226158

noncomputable def verify_solution (x : ℝ) : Prop :=
  (Real.arcsin (3 * x) + Real.arccos (2 * x) = Real.pi / 4) ∧
  (|2 * x| ≤ 1) ∧
  (|3 * x| ≤ 1)

theorem solution_valid (x : ℝ) :
  verify_solution x ↔ (x = 1 / Real.sqrt (11 - 2 * Real.sqrt 2) ∨ x = -(1 / Real.sqrt (11 - 2 * Real.sqrt 2))) :=
by {
  sorry
}

end solution_valid_l226_226158


namespace coordinates_provided_l226_226359

-- Define the coordinates of point P in the Cartesian coordinate system
structure Point where
  x : ℝ
  y : ℝ

-- Define the point P with its given coordinates
def P : Point := {x := 3, y := -5}

-- Lean 4 statement for the proof problem
theorem coordinates_provided : (P.x, P.y) = (3, -5) := by
  -- Proof not provided
  sorry

end coordinates_provided_l226_226359


namespace olivia_hourly_rate_l226_226639

theorem olivia_hourly_rate (h_worked_monday : ℕ) (h_worked_wednesday : ℕ) (h_worked_friday : ℕ) (h_total_payment : ℕ) (h_total_hours : h_worked_monday + h_worked_wednesday + h_worked_friday = 13) (h_total_amount : h_total_payment = 117) :
  h_total_payment / (h_worked_monday + h_worked_wednesday + h_worked_friday) = 9 :=
by
  sorry

end olivia_hourly_rate_l226_226639


namespace line_intersects_x_axis_at_neg3_l226_226629

theorem line_intersects_x_axis_at_neg3 :
  ∃ (x y : ℝ), (5 * y - 7 * x = 21 ∧ y = 0) ↔ (x = -3 ∧ y = 0) :=
by
  sorry

end line_intersects_x_axis_at_neg3_l226_226629


namespace smaller_screen_diagonal_l226_226400

/-- The area of a 20-inch square screen is 38 square inches greater than the area
    of a smaller square screen. Prove that the length of the diagonal of the smaller screen is 18 inches. -/
theorem smaller_screen_diagonal (x : ℝ) (d : ℝ) (A₁ A₂ : ℝ)
  (h₀ : d = x * Real.sqrt 2)
  (h₁ : A₁ = 20 * Real.sqrt 2 * 20 * Real.sqrt 2)
  (h₂ : A₂ = x * x)
  (h₃ : A₁ = A₂ + 38) :
  d = 18 :=
by
  sorry

end smaller_screen_diagonal_l226_226400


namespace sum_coordinates_D_is_13_l226_226967

theorem sum_coordinates_D_is_13 
  (A B C D : ℝ × ℝ) 
  (hA : A = (4, 8))
  (hB : B = (2, 2))
  (hC : C = (6, 4))
  (hD : D = (8, 5))
  (h_mid1 : (A.1 + B.1) / 2 = 3 ∧ (A.2 + B.2) / 2 = 5)
  (h_mid2 : (B.1 + C.1) / 2 = 4 ∧ (B.2 + C.2) / 2 = 3)
  (h_mid3 : (C.1 + D.1) / 2 = 7 ∧ (C.2 + D.2) / 2 = 4.5)
  (h_mid4 : (D.1 + A.1) / 2 = 6 ∧ (D.2 + A.2) / 2 = 6.5)
  (h_square : ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (3, 5) ∧
               ((B.1 + C.1) / 2, (B.2 + C.2) / 2) = (4, 3) ∧
               ((C.1 + D.1) / 2, (C.2 + D.2) / 2) = (7, 4.5) ∧
               ((D.1 + A.1) / 2, (D.2 + A.2) / 2) = (6, 6.5))
  : (8 + 5) = 13 :=
by
  sorry

end sum_coordinates_D_is_13_l226_226967


namespace cylinder_volume_l226_226406

theorem cylinder_volume (r l : ℝ) (h1 : r = 1) (h2 : l = 2 * r) : 
  ∃ V : ℝ, V = 2 * Real.pi := 
by 
  sorry

end cylinder_volume_l226_226406


namespace necessary_condition_l226_226947

variable (P Q : Prop)

/-- If the presence of the dragon city's flying general implies that
    the horses of the Hu people will not cross the Yin Mountains,
    then "not letting the horses of the Hu people cross the Yin Mountains"
    is a necessary condition for the presence of the dragon city's flying general. -/
theorem necessary_condition (h : P → Q) : ¬Q → ¬P :=
by sorry

end necessary_condition_l226_226947


namespace number_of_solutions_l226_226787

noncomputable def f (x : ℝ) : ℝ := 3 * x^4 - 4 * x^3 - 12 * x^2 + 12

theorem number_of_solutions : ∃ x1 x2, f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 ∧
  ∀ x, f x = 0 → (x = x1 ∨ x = x2) :=
by
  sorry

end number_of_solutions_l226_226787


namespace anne_cleans_in_12_hours_l226_226933

theorem anne_cleans_in_12_hours (B A C : ℝ) (h1 : B + A + C = 1/4)
    (h2 : B + 2 * A + 3 * C = 1/3) (h3 : B + C = 1/6) : 1 / A = 12 :=
by
    sorry

end anne_cleans_in_12_hours_l226_226933


namespace plane_equation_l226_226418

theorem plane_equation 
  (P Q : ℝ×ℝ×ℝ) (A B : ℝ×ℝ×ℝ)
  (hp : P = (-1, 2, 5))
  (hq : Q = (3, -4, 1))
  (ha : A = (0, -2, -1))
  (hb : B = (3, 2, -1)) :
  ∃ (a b c d : ℝ), (a = 3 ∧ b = 4 ∧ c = 0 ∧ d = 1) ∧ (∀ x y z : ℝ, a * (x - 1) + b * (y + 1) + c * (z - 3) = d) :=
by
  sorry

end plane_equation_l226_226418


namespace trigonometric_expression_l226_226013

theorem trigonometric_expression (θ : ℝ) (h : Real.tan θ = -3) :
    2 / (3 * (Real.sin θ) ^ 2 - (Real.cos θ) ^ 2) = 10 / 13 :=
by
  -- sorry to skip the proof
  sorry

end trigonometric_expression_l226_226013


namespace sin_30_plus_cos_60_l226_226733

-- Define the trigonometric evaluations as conditions
def sin_30_degree := 1 / 2
def cos_60_degree := 1 / 2

-- Lean statement for proving the sum of these values
theorem sin_30_plus_cos_60 : sin_30_degree + cos_60_degree = 1 := by
  sorry

end sin_30_plus_cos_60_l226_226733


namespace Petya_time_comparison_l226_226878

-- Define the conditions
variables (a V : ℝ) (hV : V > 0)

noncomputable def T_planned := a / V

noncomputable def T1 := a / (2.5 * V)

noncomputable def T2 := a / (1.6 * V)

noncomputable def T_real := T1 + T2

-- State the main theorem
theorem Petya_time_comparison (ha : a > 0) : T_real a V > T_planned a V :=
by
  -- Proof to be filled in
  sorry

end Petya_time_comparison_l226_226878


namespace joe_total_toy_cars_l226_226616

def initial_toy_cars : ℕ := 50
def uncle_additional_factor : ℝ := 1.5

theorem joe_total_toy_cars :
  (initial_toy_cars : ℝ) + uncle_additional_factor * initial_toy_cars = 125 := 
by
  sorry

end joe_total_toy_cars_l226_226616


namespace fraction_to_decimal_l226_226080

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := 
by
  sorry

end fraction_to_decimal_l226_226080


namespace graphs_intersection_l226_226199

theorem graphs_intersection 
  (a b c d x y : ℝ) 
  (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) 
  (h1: y = ax^2 + bx + c) 
  (h2: y = ax^2 - bx + c + d) 
  : x = d / (2 * b) ∧ y = (a * d^2) / (4 * b^2) + d / 2 + c := 
sorry

end graphs_intersection_l226_226199


namespace binomial_coefficient_30_3_l226_226101

theorem binomial_coefficient_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_coefficient_30_3_l226_226101


namespace minimum_distance_l226_226661

section MinimumDistance
open Real

noncomputable def f (x : ℝ) : ℝ := exp x
noncomputable def g (x : ℝ) : ℝ := 2 * sqrt x
def t (x1 x2 : ℝ) := f x1 = g x2
def d (x1 x2 : ℝ) := abs (x2 - x1)

theorem minimum_distance : ∃ (x1 x2 : ℝ), t x1 x2 ∧ d x1 x2 = (1 - log 2) / 2 := 
sorry

end MinimumDistance

end minimum_distance_l226_226661


namespace no_such_n_exists_l226_226829

theorem no_such_n_exists :
  ¬ ∃ n : ℕ, 0 < n ∧
  (∃ a : ℕ, 2 * n^2 + 1 = a^2) ∧
  (∃ b : ℕ, 3 * n^2 + 1 = b^2) ∧
  (∃ c : ℕ, 6 * n^2 + 1 = c^2) :=
sorry

end no_such_n_exists_l226_226829


namespace correct_average_l226_226638

theorem correct_average (incorrect_avg : ℝ) (num_values : ℕ) (misread_value actual_value : ℝ) 
  (h1 : incorrect_avg = 16) 
  (h2 : num_values = 10)
  (h3 : misread_value = 26)
  (h4 : actual_value = 46) : 
  (incorrect_avg * num_values + (actual_value - misread_value)) / num_values = 18 := 
by
  sorry

end correct_average_l226_226638


namespace bah_rah_yah_equiv_l226_226585

-- We define the initial equivalences given in the problem statement.
theorem bah_rah_yah_equiv (bahs rahs yahs : ℕ) :
  (18 * bahs = 30 * rahs) ∧
  (12 * rahs = 20 * yahs) →
  (1200 * yahs = 432 * bahs) :=
by
  -- Placeholder for the actual proof
  sorry

end bah_rah_yah_equiv_l226_226585


namespace dealer_can_determine_values_l226_226910

def card_value_determined (a : Fin 100 → Fin 100) : Prop :=
  (∀ i j : Fin 100, i > j → a i > a j) ∧ (a 0 > a 99) ∧
  (∀ k : Fin 100, a k = k + 1)

theorem dealer_can_determine_values :
  ∃ (messages : Fin 100 → Fin 100), card_value_determined messages :=
sorry

end dealer_can_determine_values_l226_226910


namespace positive_iff_sum_and_product_positive_l226_226909

theorem positive_iff_sum_and_product_positive (a b : ℝ) :
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) :=
by
  sorry

end positive_iff_sum_and_product_positive_l226_226909


namespace length_of_rectangle_l226_226495

theorem length_of_rectangle (P L B : ℕ) (h₁ : P = 800) (h₂ : B = 300) (h₃ : P = 2 * (L + B)) : L = 100 := by
  sorry

end length_of_rectangle_l226_226495


namespace smallest_q_p_l226_226784

noncomputable def q_p_difference : ℕ := 3

theorem smallest_q_p (p q : ℕ) (hp : 0 < p) (hq : 0 < q) (h1 : 5 * q < 9 * p) (h2 : 9 * p < 5 * q) : q - p = q_p_difference → q = 7 :=
by
  sorry

end smallest_q_p_l226_226784


namespace work_time_A_and_C_together_l226_226453

theorem work_time_A_and_C_together
  (A_work B_work C_work : ℝ)
  (hA : A_work = 1/3)
  (hB : B_work = 1/6)
  (hBC : B_work + C_work = 1/3) :
  1 / (A_work + C_work) = 2 := by
  sorry

end work_time_A_and_C_together_l226_226453


namespace pizzaCostPerSlice_l226_226032

/-- Define the constants and parameters for the problem --/
def largePizzaCost : ℝ := 10.00
def numberOfSlices : ℕ := 8
def firstToppingCost : ℝ := 2.00
def secondThirdToppingCost : ℝ := 1.00
def otherToppingCost : ℝ := 0.50
def toppings : List String := ["pepperoni", "sausage", "ham", "olives", "mushrooms", "bell peppers", "pineapple"]

/-- Calculate the total number of toppings --/
def numberOfToppings : ℕ := toppings.length

/-- Calculate the total cost of the pizza including all toppings --/
noncomputable def totalPizzaCost : ℝ :=
  largePizzaCost + 
  firstToppingCost + 
  2 * secondThirdToppingCost + 
  (numberOfToppings - 3) * otherToppingCost

/-- Calculate the cost per slice --/
noncomputable def costPerSlice : ℝ := totalPizzaCost / numberOfSlices

/-- Proof statement: The cost per slice is $2.00 --/
theorem pizzaCostPerSlice : costPerSlice = 2 := by
  sorry

end pizzaCostPerSlice_l226_226032


namespace intersection_A_B_l226_226668

def A : Set ℝ := {x | x^2 - x = 0}
def B : Set ℝ := {x | x^2 + x = 0}

theorem intersection_A_B : A ∩ B = {0} :=
by
  sorry

end intersection_A_B_l226_226668


namespace find_n_divisibility_l226_226577

theorem find_n_divisibility :
  ∃ n : ℕ, n < 10 ∧ (6 * 10000 + n * 1000 + 2 * 100 + 7 * 10 + 2) % 11 = 0 ∧ (6 * 10000 + n * 1000 + 2 * 100 + 7 * 10 + 2) % 5 = 0 :=
by
  use 3
  sorry

end find_n_divisibility_l226_226577


namespace age_of_other_man_replaced_l226_226501

-- Define the conditions
variables (A : ℝ) (x : ℝ)
variable (average_age_women : ℝ := 50)
variable (num_men : ℕ := 10)
variable (increase_age : ℝ := 6)
variable (one_man_age : ℝ := 22)

-- State the theorem to be proved
theorem age_of_other_man_replaced :
  2 * average_age_women - (one_man_age + x) = 10 * (A + increase_age) - 10 * A →
  x = 18 :=
by
  sorry

end age_of_other_man_replaced_l226_226501


namespace area_of_parallelogram_l226_226768

def parallelogram_base : ℝ := 26
def parallelogram_height : ℝ := 14

theorem area_of_parallelogram : parallelogram_base * parallelogram_height = 364 := by
  sorry

end area_of_parallelogram_l226_226768


namespace contributions_before_john_l226_226884

theorem contributions_before_john
  (A : ℝ) (n : ℕ)
  (h1 : 1.5 * A = 75)
  (h2 : (n * A + 150) / (n + 1) = 75) :
  n = 3 :=
by
  sorry

end contributions_before_john_l226_226884


namespace sector_area_ratio_l226_226837

theorem sector_area_ratio (angle_AOE angle_FOB : ℝ) (h1 : angle_AOE = 40) (h2 : angle_FOB = 60) : 
  (180 - angle_AOE - angle_FOB) / 360 = 2 / 9 :=
by
  sorry

end sector_area_ratio_l226_226837


namespace polygon_sides_l226_226297

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 3 * 360)
  (h2 : n ≥ 3) : 
  n = 8 := 
by
  sorry

end polygon_sides_l226_226297


namespace shape_is_plane_l226_226135

noncomputable
def cylindrical_coordinates_shape (r θ z c : ℝ) := θ = 2 * c

theorem shape_is_plane (c : ℝ) : 
  ∀ (r : ℝ) (θ : ℝ) (z : ℝ), cylindrical_coordinates_shape r θ z c → (θ = 2 * c) :=
by
  sorry

end shape_is_plane_l226_226135


namespace marks_for_correct_answer_l226_226838

theorem marks_for_correct_answer (x : ℕ) 
  (total_marks : ℤ) (total_questions : ℕ) (correct_answers : ℕ) 
  (wrong_mark : ℤ) (result : ℤ) :
  total_marks = result →
  total_questions = 70 →
  correct_answers = 27 →
  (-1) * (total_questions - correct_answers) = wrong_mark →
  total_marks = (correct_answers : ℤ) * (x : ℤ) + wrong_mark →
  x = 3 := 
by
  intros h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end marks_for_correct_answer_l226_226838


namespace geometric_sequence_properties_l226_226613

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
∀ n, a (n + 1) = q * a n

theorem geometric_sequence_properties :
  ∀ (a : ℕ → ℝ),
    geometric_sequence a q →
    a 2 = 6 →
    a 5 - 2 * a 4 - a 3 + 12 = 0 →
    ∀ n, a n = 6 ∨ a n = 6 * (-1)^(n-2) ∨ a n = 6 * 2^(n-2) :=
by
  sorry

end geometric_sequence_properties_l226_226613


namespace least_whole_number_subtracted_l226_226136

theorem least_whole_number_subtracted (x : ℕ) :
  ((6 - x) / (7 - x) < (16 / 21)) → x = 3 :=
by
  sorry

end least_whole_number_subtracted_l226_226136


namespace number_of_true_propositions_l226_226627

-- Definitions based on the problem
def proposition1 (α β : ℝ) : Prop := (α + β = 180) → (α + β = 90)
def proposition2 (α β γ δ : ℝ) : Prop := (α = β) → (γ = δ)
def proposition3 (α β γ δ : ℝ) : Prop := (α = β) → (γ = δ)

-- Proof problem statement
theorem number_of_true_propositions : ∃ n : ℕ, n = 2 :=
by
  let p1 := false
  let p2 := false
  let p3 := true
  existsi (if p3 then 1 else 0 + if p2 then 1 else 0 + if p1 then 1 else 0)
  simp
  sorry

end number_of_true_propositions_l226_226627


namespace cubic_eq_solutions_l226_226227

theorem cubic_eq_solutions (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : ∀ x, x^3 + a * x^2 + b * x + c = 0 → (x = a ∨ x = -b ∨ x = c)) : (a, b, c) = (1, -1, -1) := 
by {
  -- Convert solution steps into a proof
  sorry
}

end cubic_eq_solutions_l226_226227


namespace garden_perimeter_l226_226112

theorem garden_perimeter
  (a b : ℝ)
  (h1 : a^2 + b^2 = 1156)
  (h2 : a * b = 240) :
  2 * (a + b) = 80 :=
sorry

end garden_perimeter_l226_226112


namespace linear_inequalities_solution_l226_226798

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end linear_inequalities_solution_l226_226798


namespace lines_not_intersecting_may_be_parallel_or_skew_l226_226567

theorem lines_not_intersecting_may_be_parallel_or_skew (a b : ℝ × ℝ → Prop) 
  (h : ∀ x, ¬ (a x ∧ b x)) : 
  (∃ c d : ℝ × ℝ → Prop, a = c ∧ b = d) := 
sorry

end lines_not_intersecting_may_be_parallel_or_skew_l226_226567


namespace solve_for_x_l226_226287

def star (a b : ℤ) := a * b + 3 * b - a

theorem solve_for_x : ∃ x : ℤ, star 4 x = 46 := by
  sorry

end solve_for_x_l226_226287


namespace total_time_correct_l226_226296

-- Definitions for the conditions
def dean_time : ℕ := 9
def micah_time : ℕ := (2 * dean_time) / 3
def jake_time : ℕ := micah_time + micah_time / 3

-- Proof statement for the total time
theorem total_time_correct : micah_time + dean_time + jake_time = 23 := by
  sorry

end total_time_correct_l226_226296


namespace point_in_fourth_quadrant_l226_226707

def Point : Type := ℤ × ℤ

def in_fourth_quadrant (p : Point) : Prop :=
  p.1 > 0 ∧ p.2 < 0

def A : Point := (-3, 7)
def B : Point := (3, -7)
def C : Point := (3, 7)
def D : Point := (-3, -7)

theorem point_in_fourth_quadrant : in_fourth_quadrant B :=
by {
  -- skipping the proof steps for the purpose of this example
  sorry
}

end point_in_fourth_quadrant_l226_226707


namespace polynomial_root_range_l226_226425

variable (a : ℝ)

theorem polynomial_root_range (h : ∀ x : ℂ, (2 * x^4 + a * x^3 + 9 * x^2 + a * x + 2 = 0) →
  ((x.re^2 + x.im^2 ≠ 1) ∧ x.im ≠ 0)) : (-2 * Real.sqrt 10 < a ∧ a < 2 * Real.sqrt 10) :=
sorry

end polynomial_root_range_l226_226425


namespace function_satisfies_conditions_l226_226985

-- Define the functional equation condition
def functional_eq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + x * y) = f x * f (y + 1)

-- Lean statement for the proof problem
theorem function_satisfies_conditions (f : ℝ → ℝ) (h : functional_eq f) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) ∨ (∀ x : ℝ, f x = x) :=
sorry

end function_satisfies_conditions_l226_226985


namespace find_C_l226_226184

theorem find_C (A B C : ℕ) (h1 : (19 + A + B) % 3 = 0) (h2 : (15 + A + B + C) % 3 = 0) : C = 1 := by
  sorry

end find_C_l226_226184


namespace income_ratio_l226_226436

theorem income_ratio (I1 I2 E1 E2 : ℝ) (h1 : I1 = 5500) (h2 : E1 = I1 - 2200) (h3 : E2 = I2 - 2200) (h4 : E1 / E2 = 3 / 2) : I1 / I2 = 5 / 4 := by
  -- This is where the proof would go, but it's omitted for brevity.
  sorry

end income_ratio_l226_226436


namespace inradius_of_triangle_area_twice_perimeter_l226_226137

theorem inradius_of_triangle_area_twice_perimeter (A p r s : ℝ) (hA : A = 2 * p) (hs : p = 2 * s)
  (hA_formula : A = r * s) : r = 4 :=
by
  sorry

end inradius_of_triangle_area_twice_perimeter_l226_226137


namespace solve_inequality_l226_226546

open Set

variable {f : ℝ → ℝ}
open Function

theorem solve_inequality (h_inc : ∀ x y, 0 < x → 0 < y → x < y → f x < f y)
  (h_func_eq : ∀ x y, 0 < x → 0 < y → f (x / y) = f x - f y)
  (h_f3 : f 3 = 1)
  (x : ℝ) (hx_pos : 0 < x)
  (hx_ge : x > 5)
  (h_ineq : f x - f (1 / (x - 5)) ≥ 2) :
  x ≥ (5 + Real.sqrt 61) / 2 := sorry

end solve_inequality_l226_226546


namespace minimum_n_l226_226215

noncomputable def a (n : ℕ) : ℕ := 2 ^ (n - 2)

noncomputable def b (n : ℕ) : ℕ := n - 6 + a n

noncomputable def S (n : ℕ) : ℕ := (n * (n - 11)) / 2 + (2 ^ n - 1) / 2

theorem minimum_n (n : ℕ) (hn : n ≥ 5) : S 5 > 0 := by
  sorry

end minimum_n_l226_226215


namespace find_c_l226_226854

structure ProblemData where
  (r : ℝ → ℝ)
  (s : ℝ → ℝ)
  (h : r (s 3) = 20)

def r (x : ℝ) : ℝ := 5 * x - 10
def s (x : ℝ) (c : ℝ) : ℝ := 4 * x - c

theorem find_c (c : ℝ) (h : (r (s 3 c)) = 20) : c = 6 :=
sorry

end find_c_l226_226854


namespace simplify_and_evaluate_l226_226647

theorem simplify_and_evaluate 
  (a b : ℚ) (h_a : a = -1/3) (h_b : b = -3) : 
  2 * (3 * a^2 * b - a * b^2) - (a * b^2 + 6 * a^2 * b) = 9 := 
  by 
    rw [h_a, h_b]
    sorry

end simplify_and_evaluate_l226_226647


namespace remainder_when_divided_by_84_l226_226229

/-- 
  Given conditions:
  x ≡ 11 [MOD 14]
  Find the remainder when x is divided by 84, which equivalently means proving: 
  x ≡ 81 [MOD 84]
-/

theorem remainder_when_divided_by_84 (x : ℤ) (h1 : x % 14 = 11) : x % 84 = 81 :=
by
  sorry

end remainder_when_divided_by_84_l226_226229


namespace problem_k_star_k_star_k_l226_226863

def star (x y : ℝ) : ℝ := 2 * x^2 - y

theorem problem_k_star_k_star_k (k : ℝ) : star k (star k k) = k :=
by
  sorry

end problem_k_star_k_star_k_l226_226863


namespace angle_measures_possible_l226_226181

theorem angle_measures_possible (A B : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : A + B = 180) (h4 : ∃ k, k > 0 ∧ A = k * B) : 
  ∃ n : ℕ, n = 18 := 
sorry

end angle_measures_possible_l226_226181


namespace find_x_l226_226726

theorem find_x (x : ℝ) (h : 0.009 / x = 0.05) : x = 0.18 :=
sorry

end find_x_l226_226726


namespace twelfth_term_arithmetic_sequence_l226_226317

theorem twelfth_term_arithmetic_sequence (a d : ℤ) (h1 : a + 2 * d = 13) (h2 : a + 6 * d = 25) : a + 11 * d = 40 := 
sorry

end twelfth_term_arithmetic_sequence_l226_226317


namespace coefficient_of_expression_l226_226085

theorem coefficient_of_expression :
  ∀ (a b : ℝ), (∃ (c : ℝ), - (2/3) * (a * b) = c * (a * b)) :=
by
  intros a b
  use (-2/3)
  sorry

end coefficient_of_expression_l226_226085


namespace joe_initial_paint_l226_226239

noncomputable def total_paint (P : ℕ) : Prop :=
  let used_first_week := (1 / 4 : ℚ) * P
  let remaining_after_first := (3 / 4 : ℚ) * P
  let used_second_week := (1 / 6 : ℚ) * remaining_after_first
  let total_used := used_first_week + used_second_week
  total_used = 135

theorem joe_initial_paint (P : ℕ) (h : total_paint P) : P = 463 :=
sorry

end joe_initial_paint_l226_226239


namespace power_function_passes_through_1_1_l226_226658

theorem power_function_passes_through_1_1 (a : ℝ) : (1 : ℝ) ^ a = 1 := 
by
  sorry

end power_function_passes_through_1_1_l226_226658


namespace product_of_roots_l226_226424

noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

noncomputable def f_prime (a b c : ℝ) (x : ℝ) : ℝ :=
  3 * a * x^2 + 2 * b * x + c

theorem product_of_roots (a b c d x₁ x₂ : ℝ) 
  (h1 : f a b c d 0 = 0)
  (h2 : f a b c d x₁ = 0)
  (h3 : f a b c d x₂ = 0)
  (h_ext1 : f_prime a b c 1 = 0)
  (h_ext2 : f_prime a b c 2 = 0) :
  x₁ * x₂ = 6 :=
sorry

end product_of_roots_l226_226424


namespace cos_double_angle_l226_226401

open Real

theorem cos_double_angle (α : Real) (h : tan α = 3) : cos (2 * α) = -4/5 :=
  sorry

end cos_double_angle_l226_226401


namespace minimum_value_of_2x_3y_l226_226468

noncomputable def minimum_value (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : (2/x) + (3/y) = 1) : ℝ :=
  2*x + 3*y

theorem minimum_value_of_2x_3y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : (2/x) + (3/y) = 1) : minimum_value x y hx hy hxy = 25 :=
sorry

end minimum_value_of_2x_3y_l226_226468


namespace number_is_multiple_of_15_l226_226631

theorem number_is_multiple_of_15
  (W X Y Z D : ℤ)
  (h1 : X - W = 1)
  (h2 : Y - W = 9)
  (h3 : Y - X = 8)
  (h4 : Z - W = 11)
  (h5 : Z - X = 10)
  (h6 : Z - Y = 2)
  (hD : D - X = 5) :
  15 ∣ D :=
by
  sorry -- Proof goes here

end number_is_multiple_of_15_l226_226631


namespace number_of_bricks_in_wall_l226_226618

noncomputable def rate_one_bricklayer (x : ℕ) : ℚ := x / 8
noncomputable def rate_other_bricklayer (x : ℕ) : ℚ := x / 12
noncomputable def combined_rate_with_efficiency (x : ℕ) : ℚ := (rate_one_bricklayer x + rate_other_bricklayer x - 15)
noncomputable def total_time (x : ℕ) : ℚ := 6 * combined_rate_with_efficiency x

theorem number_of_bricks_in_wall (x : ℕ) : total_time x = x → x = 360 :=
by sorry

end number_of_bricks_in_wall_l226_226618


namespace ratio_of_girls_to_boys_l226_226532

theorem ratio_of_girls_to_boys (x y : ℕ) (h1 : x + y = 28) (h2 : x - y = 4) : x = 16 ∧ y = 12 ∧ x / y = 4 / 3 :=
by
  sorry

end ratio_of_girls_to_boys_l226_226532


namespace prism_volume_l226_226760

theorem prism_volume 
    (x y z : ℝ) 
    (h_xy : x * y = 18) 
    (h_yz : y * z = 12) 
    (h_xz : x * z = 8) 
    (h_longest_shortest : max x (max y z) = 2 * min x (min y z)) : 
    x * y * z = 16 := 
  sorry

end prism_volume_l226_226760


namespace length_of_PS_l226_226235

theorem length_of_PS
  (PT TR QT TS PQ : ℝ)
  (h1 : PT = 5)
  (h2 : TR = 7)
  (h3 : QT = 9)
  (h4 : TS = 4)
  (h5 : PQ = 7) :
  PS = Real.sqrt 66.33 := 
  sorry

end length_of_PS_l226_226235


namespace exists_m_with_totient_ratio_l226_226888

variable (α β : ℝ)

theorem exists_m_with_totient_ratio (h0 : 0 ≤ α) (h1 : α < β) (h2 : β ≤ 1) :
  ∃ m : ℕ, α < (Nat.totient m : ℝ) / m ∧ (Nat.totient m : ℝ) / m < β := 
  sorry

end exists_m_with_totient_ratio_l226_226888


namespace cube_face_coloring_l226_226780

-- Define the type of a cube's face coloring
inductive FaceColor
| black
| white

open FaceColor

def countDistinctColorings : Nat :=
  -- Function to count the number of distinct colorings considering rotational symmetry
  10

theorem cube_face_coloring :
  countDistinctColorings = 10 :=
by
  -- Skip the proof, indicating it should be proved.
  sorry

end cube_face_coloring_l226_226780


namespace find_omitted_angle_l226_226415

-- Definitions and conditions
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

def omitted_angle (calculated_sum actual_sum : ℝ) : ℝ :=
  actual_sum - calculated_sum

-- The theorem to be proven
theorem find_omitted_angle (n : ℕ) (h₁ : 1958 + 22 = sum_of_interior_angles n) :
  omitted_angle 1958 (sum_of_interior_angles n) = 22 :=
by
  sorry

end find_omitted_angle_l226_226415


namespace raghu_investment_l226_226828

noncomputable def investment_problem (R T V : ℝ) : Prop :=
  V = 1.1 * T ∧
  T = 0.9 * R ∧
  R + T + V = 6358 ∧
  R = 2200

theorem raghu_investment
  (R T V : ℝ)
  (h1 : V = 1.1 * T)
  (h2 : T = 0.9 * R)
  (h3 : R + T + V = 6358) :
  R = 2200 :=
sorry

end raghu_investment_l226_226828


namespace lcm_is_only_function_l226_226554

noncomputable def f (x y : ℕ) : ℕ := Nat.lcm x y

theorem lcm_is_only_function 
    (f : ℕ → ℕ → ℕ)
    (h1 : ∀ x : ℕ, f x x = x) 
    (h2 : ∀ x y : ℕ, f x y = f y x) 
    (h3 : ∀ x y : ℕ, (x + y) * f x y = y * f x (x + y)) : 
  ∀ x y : ℕ, f x y = Nat.lcm x y := 
sorry

end lcm_is_only_function_l226_226554


namespace coal_consumption_rel_l226_226694

variables (Q a x y : ℝ)
variables (h₀ : 0 < x) (h₁ : x < a) (h₂ : Q ≠ 0) (h₃ : a ≠ 0) (h₄ : a - x ≠ 0)

theorem coal_consumption_rel :
  y = Q / (a - x) - Q / a :=
sorry

end coal_consumption_rel_l226_226694


namespace factors_of_expression_l226_226680

def total_distinct_factors : ℕ :=
  let a := 10
  let b := 3
  let c := 2
  (a + 1) * (b + 1) * (c + 1)

theorem factors_of_expression :
  total_distinct_factors = 132 :=
by 
  -- the proof goes here
  sorry

end factors_of_expression_l226_226680


namespace cone_radius_of_surface_area_and_lateral_surface_unfolds_to_semicircle_l226_226874

theorem cone_radius_of_surface_area_and_lateral_surface_unfolds_to_semicircle
  (surface_area : ℝ) (lateral_surface_unfolds_to_semicircle : Prop) :
  surface_area = 12 * Real.pi → lateral_surface_unfolds_to_semicircle → ∃ r : ℝ, r = 2 := by
  sorry

end cone_radius_of_surface_area_and_lateral_surface_unfolds_to_semicircle_l226_226874


namespace symmetric_point_of_M_neg2_3_l226_226264

-- Conditions
def symmetric_point (M : ℝ × ℝ) : ℝ × ℝ :=
  (-M.1, -M.2)

-- Main statement
theorem symmetric_point_of_M_neg2_3 :
  symmetric_point (-2, 3) = (2, -3) := 
by
  -- Proof goes here
  sorry

end symmetric_point_of_M_neg2_3_l226_226264


namespace alice_bob_numbers_sum_l226_226621

-- Fifty slips of paper numbered 1 to 50 are placed in a hat.
-- Alice and Bob each draw one number from the hat without replacement, keeping their numbers hidden from each other.
-- Alice cannot tell who has the larger number.
-- Bob knows who has the larger number.
-- Bob's number is composite.
-- If Bob's number is multiplied by 50 and Alice's number is added, the result is a perfect square.
-- Prove that the sum of Alice's and Bob's numbers is 29.

theorem alice_bob_numbers_sum (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 50) (hB : 1 ≤ B ∧ B ≤ 50) 
  (hAB_distinct : A ≠ B) (hA_unknown : ¬(A = 1 ∨ A = 50))
  (hB_composite : ∃ d > 1, d < B ∧ B % d = 0) (h_perfect_square : ∃ k, 50 * B + A = k ^ 2) :
  A + B = 29 := by
  sorry

end alice_bob_numbers_sum_l226_226621


namespace person_birth_date_l226_226692

theorem person_birth_date
  (x : ℕ)
  (h1 : 1937 - x = x^2 - x)
  (d m : ℕ)
  (h2 : 44 + m = d^2)
  (h3 : 0 < m ∧ m < 13)
  (h4 : d = 7 ∧ m = 5) :
  (x = 44 ∧ 1937 - (x + x^2) = 1892) ∧  d = 7 ∧ m = 5 :=
by
  sorry

end person_birth_date_l226_226692


namespace line_through_point_with_equal_intercepts_l226_226104

theorem line_through_point_with_equal_intercepts
  (P : ℝ × ℝ) (hP : P = (1, 3))
  (intercepts_equal : ∃ a : ℝ, a ≠ 0 ∧ (∀ x y : ℝ, (x/a) + (y/a) = 1 → x + y = 4 ∨ 3*x - y = 0)) :
  ∃ a b c : ℝ, (a, b, c) = (3, -1, 0) ∨ (a, b, c) = (1, 1, -4) ∧ (∀ x y : ℝ, a*x + b*y + c = 0 → (x + y = 4 ∨ 3*x - y = 0)) := 
by
  sorry

end line_through_point_with_equal_intercepts_l226_226104


namespace longest_side_AB_l226_226948

-- Definitions of angles in the quadrilateral
def angle_ABC := 65
def angle_BCD := 70
def angle_CDA := 60

/-- In a quadrilateral ABCD with angles as specified, prove that AB is the longest side. -/
theorem longest_side_AB (AB BC CD DA : ℝ) : 
  (angle_ABC = 65 ∧ angle_BCD = 70 ∧ angle_CDA = 60) → 
  AB > DA ∧ AB > BC ∧ AB > CD :=
by
  intros h
  sorry

end longest_side_AB_l226_226948


namespace unit_circle_solution_l226_226541

noncomputable def unit_circle_point_x (α : ℝ) (hα : α ∈ Set.Ioo (Real.pi / 6) (Real.pi / 2)) 
  (hcos : Real.cos (α + Real.pi / 3) = -11 / 13) : ℝ :=
  1 / 26

theorem unit_circle_solution (α : ℝ) (hα : α ∈ Set.Ioo (Real.pi / 6) (Real.pi / 2)) 
  (hcos : Real.cos (α + Real.pi / 3) = -11 / 13) :
  unit_circle_point_x α hα hcos = 1 / 26 :=
by
  sorry

end unit_circle_solution_l226_226541


namespace cubic_roots_c_over_d_l226_226687

theorem cubic_roots_c_over_d (a b c d : ℤ) (h : a ≠ 0)
  (h_roots : ∃ r1 r2 r3, r1 = -1 ∧ r2 = 3 ∧ r3 = 4 ∧ 
              a * r1 * r2 * r3 + b * (r1 * r2 + r2 * r3 + r3 * r1) + c * (r1 + r2 + r3) + d = 0)
  : (c : ℚ) / d = 5 / 12 := 
sorry

end cubic_roots_c_over_d_l226_226687


namespace power_sum_is_integer_l226_226831

theorem power_sum_is_integer (a : ℝ) (n : ℕ) (h_pos : 0 < n)
  (h_k : ∃ k : ℤ, k = a + 1/a) : 
  ∃ m : ℤ, m = a^n + 1/a^n := 
sorry

end power_sum_is_integer_l226_226831


namespace intersection_with_y_axis_l226_226241

theorem intersection_with_y_axis (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 + x - 2) : f 0 = -2 :=
by
  sorry

end intersection_with_y_axis_l226_226241


namespace garage_sale_items_l226_226127

theorem garage_sale_items (h : 34 = 13 + n + 1 + 14 - 14) : n = 22 := by
  sorry

end garage_sale_items_l226_226127


namespace smallest_whole_number_greater_than_sum_l226_226416

theorem smallest_whole_number_greater_than_sum : 
  (3 + (1 / 3) + 4 + (1 / 4) + 6 + (1 / 6) + 7 + (1 / 7)) < 21 :=
sorry

end smallest_whole_number_greater_than_sum_l226_226416


namespace average_velocity_mass_flow_rate_available_horsepower_l226_226938

/-- Average velocity of water flowing out of the sluice gate. -/
theorem average_velocity (g h₁ h₂ : ℝ) (h1_5m : h₁ = 5) (h2_5_4m : h₂ = 5.4) (g_9_81 : g = 9.81) :
    (1 / 2) * (Real.sqrt (2 * g * h₁) + Real.sqrt (2 * g * h₂)) = 10.1 :=
by
  sorry

/-- Mass flow rate of water per second when given average velocity and opening dimensions. -/
theorem mass_flow_rate (v A : ℝ) (v_10_1 : v = 10.1) (A_0_6 : A = 0.4 * 1.5) (rho : ℝ) (rho_1000 : rho = 1000) :
    ρ * A * v = 6060 :=
by
  sorry

/-- Available horsepower through turbines given mass flow rate and average velocity. -/
theorem available_horsepower (m v : ℝ) (m_6060 : m = 6060) (v_10_1 : v = 10.1 ) (hp : ℝ)
    (hp_735_5 : hp = 735.5 ) :
    (1 / 2) * m * v^2 / hp = 420 :=
by
  sorry

end average_velocity_mass_flow_rate_available_horsepower_l226_226938


namespace part1_part2_l226_226035

def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem part1 (a : ℝ) (h : a = 4) : 
  {x : ℝ | f x a ≥ 5} = {x : ℝ | x ≤ 0 ∨ x ≥ 5} :=
sorry

theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 4) : 
  a ≤ -3 ∨ a ≥ 5 :=
sorry

end part1_part2_l226_226035


namespace sector_area_is_2_l226_226547

-- Definition of the sector's properties
def sector_perimeter (r : ℝ) (θ : ℝ) : ℝ := r * θ + 2 * r

def sector_area (r : ℝ) (θ : ℝ) : ℝ := 0.5 * r^2 * θ

-- Theorem stating that the area of the sector is 2 cm² given the conditions
theorem sector_area_is_2 (r θ : ℝ) (h1 : sector_perimeter r θ = 6) (h2 : θ = 1) : sector_area r θ = 2 :=
by
  sorry

end sector_area_is_2_l226_226547


namespace problem1_problem2_problem3_l226_226843

variable {m n p x : ℝ}

-- Problem 1
theorem problem1 : m^2 * (n - 3) + 4 * (3 - n) = (n - 3) * (m + 2) * (m - 2) := 
sorry

-- Problem 2
theorem problem2 : (p - 3) * (p - 1) + 1 = (p - 2) ^ 2 := 
sorry

-- Problem 3
theorem problem3 (hx : x^2 + x + 1 / 4 = 0) : (2 * x + 1) / (x + 1) + (x - 1) / 1 / (x + 2) / (x^2 + 2 * x + 1) = -1 / 4 :=
sorry

end problem1_problem2_problem3_l226_226843


namespace required_run_rate_l226_226931

theorem required_run_rate (run_rate_first_10_overs : ℝ) (target_runs total_overs first_overs : ℕ) :
  run_rate_first_10_overs = 4.2 ∧ target_runs = 282 ∧ total_overs = 50 ∧ first_overs = 10 →
  (target_runs - run_rate_first_10_overs * first_overs) / (total_overs - first_overs) = 6 :=
by
  sorry

end required_run_rate_l226_226931


namespace ratio_perimeters_of_squares_l226_226918

theorem ratio_perimeters_of_squares (a b : ℝ) (h_diag : (a * Real.sqrt 2) / (b * Real.sqrt 2) = 2.5) : (4 * a) / (4 * b) = 10 :=
by
  sorry

end ratio_perimeters_of_squares_l226_226918


namespace george_speed_to_school_l226_226442

theorem george_speed_to_school :
  ∀ (D S_1 S_2 D_1 S_x : ℝ),
  D = 1.5 ∧ S_1 = 3 ∧ S_2 = 2 ∧ D_1 = 0.75 →
  S_x = (D - D_1) / ((D / S_1) - (D_1 / S_2)) →
  S_x = 6 :=
by
  intros D S_1 S_2 D_1 S_x h1 h2
  rw [h1.1, h1.2.1, h1.2.2.1, h1.2.2.2] at *
  sorry

end george_speed_to_school_l226_226442


namespace perpendicular_lines_intersection_l226_226875

theorem perpendicular_lines_intersection (a b c d : ℝ)
    (h_perpendicular : (a / 2) * (-2 / b) = -1)
    (h_intersection1 : a * 2 - 2 * (-3) = d)
    (h_intersection2 : 2 * 2 + b * (-3) = c) :
    d = 12 := 
sorry

end perpendicular_lines_intersection_l226_226875


namespace five_n_minus_twelve_mod_nine_l226_226817

theorem five_n_minus_twelve_mod_nine (n : ℤ) (h : n % 9 = 4) : (5 * n - 12) % 9 = 8 := by
  sorry

end five_n_minus_twelve_mod_nine_l226_226817


namespace find_height_l226_226989

namespace RightTriangleProblem

variables {x h : ℝ}

-- Given the conditions described in the problem
def right_triangle_proportional (a b c : ℝ) : Prop :=
  ∃ (x : ℝ), a = 3 * x ∧ b = 4 * x ∧ c = 5 * x

def hypotenuse (c : ℝ) : Prop := 
  c = 25

def leg (b : ℝ) : Prop :=
  b = 20

-- The theorem stating that the height h of the triangle is 12
theorem find_height (a b c : ℝ) (h : ℝ)
  (H1 : right_triangle_proportional a b c)
  (H2 : hypotenuse c)
  (H3 : leg b) :
  h = 12 :=
by
  sorry

end RightTriangleProblem

end find_height_l226_226989


namespace find_b_from_root_l226_226470

theorem find_b_from_root (b : ℝ) :
  (Polynomial.eval (-10) (Polynomial.C 1 * X^2 + Polynomial.C b * X + Polynomial.C (-30)) = 0) →
  b = 7 :=
by
  intro h
  sorry

end find_b_from_root_l226_226470


namespace boat_capacity_problem_l226_226095

variables (L S : ℕ)

theorem boat_capacity_problem
  (h1 : L + 4 * S = 46)
  (h2 : 2 * L + 3 * S = 57) :
  3 * L + 6 * S = 96 :=
sorry

end boat_capacity_problem_l226_226095


namespace time_for_train_to_pass_platform_is_190_seconds_l226_226289

def trainLength : ℕ := 1200
def timeToCrossTree : ℕ := 120
def platformLength : ℕ := 700
def speed (distance time : ℕ) := distance / time
def distanceToCrossPlatform (trainLength platformLength : ℕ) := trainLength + platformLength
def timeToCrossPlatform (distance speed : ℕ) := distance / speed

theorem time_for_train_to_pass_platform_is_190_seconds
  (trainLength timeToCrossTree platformLength : ℕ) (h1 : trainLength = 1200) (h2 : timeToCrossTree = 120) (h3 : platformLength = 700) :
  timeToCrossPlatform (distanceToCrossPlatform trainLength platformLength) (speed trainLength timeToCrossTree) = 190 := by
  sorry

end time_for_train_to_pass_platform_is_190_seconds_l226_226289


namespace price_of_case_l226_226695

variables (bottles_per_day : ℚ) (days : ℕ) (bottles_per_case : ℕ) (total_spent : ℚ)

def total_bottles_consumed (bottles_per_day : ℚ) (days : ℕ) : ℚ :=
  bottles_per_day * days

def cases_needed (total_bottles : ℚ) (bottles_per_case : ℕ) : ℚ :=
  total_bottles / bottles_per_case

def price_per_case (total_spent : ℚ) (cases : ℚ) : ℚ :=
  total_spent / cases

theorem price_of_case (h1 : bottles_per_day = 1/2)
                      (h2 : days = 240)
                      (h3 : bottles_per_case = 24)
                      (h4 : total_spent = 60) :
  price_per_case total_spent (cases_needed (total_bottles_consumed bottles_per_day days) bottles_per_case) = 12 := 
sorry

end price_of_case_l226_226695


namespace arithmetic_sequence_ninth_term_eq_l226_226380

theorem arithmetic_sequence_ninth_term_eq :
  let a_1 := (3 : ℚ) / 8
  let a_17 := (2 : ℚ) / 3
  let a_9 := (a_1 + a_17) / 2
  a_9 = (25 : ℚ) / 48 := by
  let a_1 := (3 : ℚ) / 8
  let a_17 := (2 : ℚ) / 3
  let a_9 := (a_1 + a_17) / 2
  sorry

end arithmetic_sequence_ninth_term_eq_l226_226380


namespace minimize_sum_of_cubes_l226_226044

theorem minimize_sum_of_cubes (x y : ℝ) (h : x + y = 8) : 
  (3 * x^2 - 3 * (8 - x)^2 = 0) → (x = 4) ∧ (y = 4) :=
by
  sorry

end minimize_sum_of_cubes_l226_226044


namespace complement_of_A_in_S_l226_226698

universe u

def S : Set ℕ := {x | 0 ≤ x ∧ x ≤ 5}
def A : Set ℕ := {x | 1 < x ∧ x < 5}

theorem complement_of_A_in_S : S \ A = {0, 1, 5} := 
by sorry

end complement_of_A_in_S_l226_226698


namespace percentage_design_black_is_57_l226_226921

noncomputable def circleRadius (n : ℕ) : ℝ :=
  3 * (n + 1)

noncomputable def circleArea (n : ℕ) : ℝ :=
  Real.pi * (circleRadius n) ^ 2

noncomputable def totalArea : ℝ :=
  circleArea 6

noncomputable def blackAreas : ℝ :=
  circleArea 0 + (circleArea 2 - circleArea 1) +
  (circleArea 4 - circleArea 3) +
  (circleArea 6 - circleArea 5)

noncomputable def percentageBlack : ℝ :=
  (blackAreas / totalArea) * 100

theorem percentage_design_black_is_57 :
  percentageBlack = 57 := 
by
  sorry

end percentage_design_black_is_57_l226_226921


namespace total_germs_calculation_l226_226574

def number_of_dishes : ℕ := 10800
def germs_per_dish : ℕ := 500
def total_germs : ℕ := 5400000

theorem total_germs_calculation : germs_per_ddish * number_of_idshessh = total_germs :=
by sorry

end total_germs_calculation_l226_226574


namespace arithmetic_sequence_sum_l226_226897

theorem arithmetic_sequence_sum (c d : ℕ) 
  (h1 : ∀ (a1 a2 a3 a4 a5 a6 : ℕ), a1 = 3 → a2 = 10 → a3 = 17 → a6 = 38 → (a2 - a1 = a3 - a2) → (a3 - a2 = c - a3) → (c - a3 = d - c) → (d - c = a6 - d)) : 
  c + d = 55 := 
by 
  sorry

end arithmetic_sequence_sum_l226_226897


namespace candy_remaining_l226_226713

def initial_candy : ℝ := 1012.5
def talitha_took : ℝ := 283.7
def solomon_took : ℝ := 398.2
def maya_took : ℝ := 197.6

theorem candy_remaining : initial_candy - (talitha_took + solomon_took + maya_took) = 133 := 
by
  sorry

end candy_remaining_l226_226713


namespace negate_exactly_one_even_l226_226280

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

theorem negate_exactly_one_even (a b c : ℕ) :
  ¬ ((is_even a ∧ is_odd b ∧ is_odd c) ∨ (is_odd a ∧ is_even b ∧ is_odd c) ∨ (is_odd a ∧ is_odd b ∧ is_even c)) ↔ 
  ((is_odd a ∧ is_odd b ∧ is_odd c) ∨ (is_even a ∧ is_even b) ∨ (is_even a ∧ is_even c) ∨ (is_even b ∧ is_even c)) :=
sorry

end negate_exactly_one_even_l226_226280


namespace trees_planted_l226_226699

def initial_trees : ℕ := 150
def total_trees_after_planting : ℕ := 225

theorem trees_planted (number_of_trees_planted : ℕ) : 
  number_of_trees_planted = total_trees_after_planting - initial_trees → number_of_trees_planted = 75 :=
by 
  sorry

end trees_planted_l226_226699


namespace common_ratio_of_sequence_l226_226353

variable {a : ℕ → ℝ}
variable {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) (n1 n2 n3 : ℕ) : Prop :=
  a n2 = a n1 * r ∧ a n3 = a n1 * r^2

theorem common_ratio_of_sequence {a : ℕ → ℝ} {d : ℝ}
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_sequence a ((a 2)/(a 1)) 2 3 6) :
  ((a 3) / (a 2)) = 3 ∨ ((a 3) / (a 2)) = 1 :=
sorry

end common_ratio_of_sequence_l226_226353


namespace vitamin_d_supplements_per_pack_l226_226153

theorem vitamin_d_supplements_per_pack :
  ∃ (x : ℕ), (∀ (n m : ℕ), 7 * n = x * m → 119 <= 7 * n) ∧ (7 * n = 17 * m) :=
by
  -- definition of conditions
  let min_sold := 119
  let vitaminA_per_pack := 7
  -- let x be the number of Vitamin D supplements per pack
  -- the proof is yet to be completed
  sorry

end vitamin_d_supplements_per_pack_l226_226153


namespace rhombus_area_l226_226087

theorem rhombus_area (d1 d2 : ℕ) (h1 : d1 = 30) (h2 : d2 = 16) : (d1 * d2) / 2 = 240 := by
  sorry

end rhombus_area_l226_226087


namespace smallest_percent_increase_l226_226537

-- Define the values of each question
def question_values : List ℕ :=
  [150, 250, 400, 600, 1100, 2300, 4700, 9500, 19000, 38000, 76000, 150000, 300000, 600000, 1200000]

-- Define a function to calculate the percent increase between two questions
def percent_increase (v1 v2 : ℕ) : Float :=
  ((v2 - v1).toFloat / v1.toFloat) * 100

-- Define the specific question transitions and their percent increases
def percent_increase_1_to_4 : Float := percent_increase question_values[0] question_values[3]  -- Question 1 to 4
def percent_increase_2_to_6 : Float := percent_increase question_values[1] question_values[5]  -- Question 2 to 6
def percent_increase_5_to_10 : Float := percent_increase question_values[4] question_values[9]  -- Question 5 to 10
def percent_increase_9_to_15 : Float := percent_increase question_values[8] question_values[14] -- Question 9 to 15

-- Prove that the smallest percent increase is from Question 1 to 4
theorem smallest_percent_increase :
  percent_increase_1_to_4 < percent_increase_2_to_6 ∧
  percent_increase_1_to_4 < percent_increase_5_to_10 ∧
  percent_increase_1_to_4 < percent_increase_9_to_15 :=
by
  sorry

end smallest_percent_increase_l226_226537


namespace day_of_week_150th_day_previous_year_l226_226683

theorem day_of_week_150th_day_previous_year (N : ℕ) 
  (h1 : (275 % 7 = 4))  -- Thursday is 4th day of the week if starting from Sunday as 0
  (h2 : (215 % 7 = 4))  -- Similarly, Thursday is 4th day of the week
  : (150 % 7 = 6) :=     -- Proving the 150th day of year N-1 is a Saturday (Saturday as 6th day of the week)
sorry

end day_of_week_150th_day_previous_year_l226_226683


namespace ring_groups_in_first_tree_l226_226890

variable (n : ℕ) (y1 y2 : ℕ) (t : ℕ) (groupsPerYear : ℕ := 6)

-- each tree's rings are in groups of 2 fat rings and 4 thin rings, representing 6 years
def group_represents_years : ℕ := groupsPerYear

-- second tree has 40 ring groups, so it is 40 * 6 = 240 years old
def second_tree_groups : ℕ := 40

-- first tree is 180 years older, so its age in years
def first_tree_age : ℕ := (second_tree_groups * groupsPerYear) + 180

-- number of ring groups in the first tree
def number_of_ring_groups_in_first_tree := first_tree_age / groupsPerYear

theorem ring_groups_in_first_tree :
  number_of_ring_groups_in_first_tree = 70 :=
by
  sorry

end ring_groups_in_first_tree_l226_226890


namespace age_ratio_l226_226925

theorem age_ratio (Tim_age : ℕ) (John_age : ℕ) (ratio : ℚ) 
  (h1 : Tim_age = 79) 
  (h2 : John_age = 35) 
  (h3 : Tim_age = ratio * John_age - 5) : 
  ratio = 2.4 := 
by sorry

end age_ratio_l226_226925


namespace find_a_of_exp_function_l226_226149

theorem find_a_of_exp_function (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : a ^ 2 = 9) : a = 3 :=
sorry

end find_a_of_exp_function_l226_226149


namespace nth_equation_l226_226218

theorem nth_equation (n : ℕ) (h : n > 0) : 9 * (n - 1) + n = (n - 1) * 10 + 1 :=
sorry

end nth_equation_l226_226218


namespace portrait_in_silver_box_l226_226150

-- Definitions for the first trial
def gold_box_1 : Prop := false
def gold_box_2 : Prop := true
def silver_box_1 : Prop := true
def silver_box_2 : Prop := false
def lead_box_1 : Prop := false
def lead_box_2 : Prop := true

-- Definitions for the second trial
def gold_box_3 : Prop := false
def gold_box_4 : Prop := true
def silver_box_3 : Prop := true
def silver_box_4 : Prop := false
def lead_box_3 : Prop := false
def lead_box_4 : Prop := true

-- The main theorem statement
theorem portrait_in_silver_box
  (gold_b1 : gold_box_1 = false)
  (gold_b2 : gold_box_2 = true)
  (silver_b1 : silver_box_1 = true)
  (silver_b2 : silver_box_2 = false)
  (lead_b1 : lead_box_1 = false)
  (lead_b2 : lead_box_2 = true)
  (gold_b3 : gold_box_3 = false)
  (gold_b4 : gold_box_4 = true)
  (silver_b3 : silver_box_3 = true)
  (silver_b4 : silver_box_4 = false)
  (lead_b3 : lead_box_3 = false)
  (lead_b4 : lead_box_4 = true) : 
  (silver_box_1 ∧ ¬lead_box_2) ∧ (silver_box_3 ∧ ¬lead_box_4) :=
sorry

end portrait_in_silver_box_l226_226150


namespace xiao_wang_scores_problem_l226_226198

-- Defining the problem conditions and solution as a proof problem
theorem xiao_wang_scores_problem (x y : ℕ) (h1 : (x * y + 98) / (x + 1) = y + 1) 
                                 (h2 : (x * y + 98 + 70) / (x + 2) = y - 1) :
  (x + 2 = 10) ∧ (y - 1 = 88) :=
by 
  sorry

end xiao_wang_scores_problem_l226_226198


namespace highway_length_proof_l226_226033

variable (L : ℝ) (v1 v2 : ℝ) (t : ℝ)

def highway_length : Prop :=
  v1 = 55 ∧ v2 = 35 ∧ t = 1 / 15 ∧ (L / v2 - L / v1 = t) ∧ L = 6.42

theorem highway_length_proof : highway_length L 55 35 (1 / 15) := by
  sorry

end highway_length_proof_l226_226033


namespace correct_option_l226_226206

variable (f : ℝ → ℝ)
variable (h_diff : ∀ x : ℝ, differentiable_at ℝ f x)
variable (h_cond : ∀ x : ℝ, f x > deriv f x)

theorem correct_option :
  e ^ 2016 * f (-2016) > f 0 ∧ f 2016 < e ^ 2016 * f 0 :=
sorry

end correct_option_l226_226206


namespace solution_set_of_f_lt_exp_l226_226845

noncomputable def f : ℝ → ℝ := sorry -- assume f is a differentiable function

-- Define the conditions
axiom h_deriv : ∀ x : ℝ, deriv f x < f x
axiom h_periodic : ∀ x : ℝ, f (x + 2) = f (x - 2)
axiom h_value_at_4 : f 4 = 1

-- The main statement to be proved
theorem solution_set_of_f_lt_exp :
  ∀ x : ℝ, (f x < Real.exp x ↔ x > 0) :=
by
  intro x
  sorry

end solution_set_of_f_lt_exp_l226_226845


namespace calc_f_7_2_l226_226445

variable {f : ℝ → ℝ}

axiom f_odd : ∀ x, f (-x) = -f x
axiom f_periodic : ∀ x, f (x + 2) = f x
axiom f_sqrt_on_interval : ∀ x, 0 < x ∧ x ≤ 1 → f x = Real.sqrt x

theorem calc_f_7_2 : f (7 / 2) = -Real.sqrt 2 / 2 := by
  sorry

end calc_f_7_2_l226_226445


namespace mila_social_media_time_week_l226_226498

theorem mila_social_media_time_week
  (hours_per_day_on_phone : ℕ)
  (half_on_social_media : ℕ)
  (days_in_week : ℕ)
  (h1 : hours_per_day_on_phone = 6)
  (h2 : half_on_social_media = hours_per_day_on_phone / 2)
  (h3 : days_in_week = 7) : 
  half_on_social_media * days_in_week = 21 := 
by
  rw [h2, h3]
  norm_num
  exact h1.symm ▸ rfl

end mila_social_media_time_week_l226_226498


namespace train_speed_km_per_hr_l226_226405

theorem train_speed_km_per_hr 
  (length : ℝ) 
  (time : ℝ) 
  (h_length : length = 150) 
  (h_time : time = 9.99920006399488) : 
  length / time * 3.6 = 54.00287976961843 :=
by
  sorry

end train_speed_km_per_hr_l226_226405


namespace remainder_of_product_mod_5_l226_226851

theorem remainder_of_product_mod_5 :
  (2685 * 4932 * 91406) % 5 = 0 :=
by
  sorry

end remainder_of_product_mod_5_l226_226851


namespace cube_properties_l226_226527

theorem cube_properties (x : ℝ) (h1 : 6 * (2 * (8 * x)^(1/3))^2 = x) : x = 13824 :=
sorry

end cube_properties_l226_226527


namespace solve_system_l226_226386

theorem solve_system (x y z : ℝ) (h1 : x + y + z = 6) (h2 : x * y + y * z + z * x = 11) (h3 : x * y * z = 6) :
  (x = 1 ∧ y = 2 ∧ z = 3) ∨ (x = 1 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 1 ∧ z = 3) ∨ 
  (x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 2 ∧ z = 1) :=
sorry

end solve_system_l226_226386


namespace smallest_c_minus_a_l226_226526

theorem smallest_c_minus_a (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_prod : a * b * c = 362880) (h_ineq : a < b ∧ b < c) : 
  c - a = 109 :=
sorry

end smallest_c_minus_a_l226_226526


namespace min_sum_of_2x2_grid_l226_226396

theorem min_sum_of_2x2_grid (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
(h_sum : a * b + c * d + a * c + b * d = 2015) : a + b + c + d = 88 :=
sorry

end min_sum_of_2x2_grid_l226_226396


namespace solve_eq1_solve_eq2_solve_eq3_solve_eq4_l226_226320

theorem solve_eq1 (x : ℝ) : (3 * x + 2) ^ 2 = 25 ↔ (x = 1 ∨ x = -7 / 3) := by
  sorry

theorem solve_eq2 (x : ℝ) : 3 * x ^ 2 - 1 = 4 * x ↔ (x = (2 + Real.sqrt 7) / 3 ∨ x = (2 - Real.sqrt 7) / 3) := by
  sorry

theorem solve_eq3 (x : ℝ) : (2 * x - 1) ^ 2 = 3 * (2 * x + 1) ↔ (x = -1 / 2 ∨ x = 1) := by
  sorry

theorem solve_eq4 (x : ℝ) : x ^ 2 - 7 * x + 10 = 0 ↔ (x = 5 ∨ x = 2) := by
  sorry

end solve_eq1_solve_eq2_solve_eq3_solve_eq4_l226_226320


namespace simplify_and_rationalize_l226_226300

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_and_rationalize_l226_226300


namespace discount_equivalence_l226_226774

theorem discount_equivalence :
  ∀ (p d1 d2 : ℝ) (d : ℝ),
    p = 800 →
    d1 = 0.15 →
    d2 = 0.10 →
    p * (1 - d1) * (1 - d2) = p * (1 - d) →
    d = 0.235 := by
  intros p d1 d2 d hp hd1 hd2 heq
  sorry

end discount_equivalence_l226_226774


namespace norm_2u_equals_10_l226_226603

-- Define u as a vector in ℝ² and the function for its norm.
variable (u : ℝ × ℝ)

-- Define the condition that the norm of u is 5.
def norm_eq_5 : Prop := Real.sqrt (u.1^2 + u.2^2) = 5

-- Statement of the proof problem
theorem norm_2u_equals_10 (h : norm_eq_5 u) : Real.sqrt ((2 * u.1)^2 + (2 * u.2)^2) = 10 :=
by
  sorry

end norm_2u_equals_10_l226_226603


namespace no_conf_of_7_points_and_7_lines_l226_226225

theorem no_conf_of_7_points_and_7_lines (points : Fin 7 → Prop) (lines : Fin 7 → (Fin 7 → Prop)) :
  (∀ p : Fin 7, ∃ l₁ l₂ l₃ : Fin 7, lines l₁ p ∧ lines l₂ p ∧ lines l₃ p ∧ l₁ ≠ l₂ ∧ l₂ ≠ l₃ ∧ l₁ ≠ l₃) ∧ 
  (∀ l : Fin 7, ∃ p₁ p₂ p₃ : Fin 7, lines l p₁ ∧ lines l p₂ ∧ lines l p₃ ∧ p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₁ ≠ p₃) 
  → false :=
by
  sorry

end no_conf_of_7_points_and_7_lines_l226_226225


namespace highest_total_zits_l226_226005

def zits_per_student_Swanson := 5
def students_Swanson := 25
def total_zits_Swanson := zits_per_student_Swanson * students_Swanson -- should be 125

def zits_per_student_Jones := 6
def students_Jones := 32
def total_zits_Jones := zits_per_student_Jones * students_Jones -- should be 192

def zits_per_student_Smith := 7
def students_Smith := 20
def total_zits_Smith := zits_per_student_Smith * students_Smith -- should be 140

def zits_per_student_Brown := 8
def students_Brown := 16
def total_zits_Brown := zits_per_student_Brown * students_Brown -- should be 128

def zits_per_student_Perez := 4
def students_Perez := 30
def total_zits_Perez := zits_per_student_Perez * students_Perez -- should be 120

theorem highest_total_zits : 
  total_zits_Jones = max total_zits_Swanson (max total_zits_Smith (max total_zits_Brown (max total_zits_Perez total_zits_Jones))) :=
by
  sorry

end highest_total_zits_l226_226005


namespace hexagon_can_be_divided_into_congruent_triangles_l226_226667

section hexagon_division

-- Definitions
variables {H : Type} -- H represents the type for hexagon

-- Conditions
variables (is_hexagon : H → Prop) -- A predicate stating that a shape is a hexagon
variables (lies_on_grid : H → Prop) -- A predicate stating that the hexagon lies on the grid
variables (can_cut_along_grid_lines : H → Prop) -- A predicate stating that cuts can only be made along the grid lines
variables (identical_figures : Type u → Prop) -- A predicate stating that the obtained figures must be identical
variables (congruent_triangles : Type u → Prop) -- A predicate stating that the obtained figures are congruent triangles
variables (area_division : H → Prop) -- A predicate stating that the area of the hexagon is divided equally

-- Theorem statement
theorem hexagon_can_be_divided_into_congruent_triangles (h : H)
  (H_is_hexagon : is_hexagon h)
  (H_on_grid : lies_on_grid h)
  (H_cut : can_cut_along_grid_lines h) :
  ∃ (F : Type u), identical_figures F ∧ congruent_triangles F ∧ area_division h :=
sorry

end hexagon_division

end hexagon_can_be_divided_into_congruent_triangles_l226_226667


namespace number_of_possible_triangles_with_side_5_not_shortest_l226_226250

-- Define and prove the number of possible triangles (a, b, c) with a, b, c positive integers,
-- such that one side is length 5 and it is not the shortest side is 10.
theorem number_of_possible_triangles_with_side_5_not_shortest (a b c : ℕ) (h1: a + b > c) (h2: a + c > b) (h3: b + c > a) 
(h4: 0 < a) (h5: 0 < b) (h6: 0 < c) (h7: a = 5 ∨ b = 5 ∨ c = 5) (h8: ¬ (a < 5 ∧ b < 5 ∧ c < 5)) :
∃ n, n = 10 := 
sorry

end number_of_possible_triangles_with_side_5_not_shortest_l226_226250


namespace greatest_of_consecutive_integers_l226_226271

theorem greatest_of_consecutive_integers (x y z : ℤ) (h1: y = x + 1) (h2: z = x + 2) (h3: x + y + z = 21) : z = 8 :=
by
  sorry

end greatest_of_consecutive_integers_l226_226271


namespace distance_y_axis_l226_226500

def point_M (m : ℝ) : ℝ × ℝ := (2 - m, 1 + 2 * m)

theorem distance_y_axis :
  ∀ m : ℝ, abs (2 - m) = 2 → (point_M m = (2, 1)) ∨ (point_M m = (-2, 9)) :=
by
  sorry

end distance_y_axis_l226_226500


namespace eccentricity_of_ellipse_l226_226741
-- Import the Mathlib library for mathematical tools and structures

-- Define the condition for the ellipse and the arithmetic sequence
variables {a b c : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : 2 * b = a + c) (h4 : b^2 = a^2 - c^2)

-- State the theorem to prove
theorem eccentricity_of_ellipse : ∃ e : ℝ, e = 3 / 5 :=
by
  -- Proof would go here
  sorry

end eccentricity_of_ellipse_l226_226741


namespace joe_paint_usage_l226_226905

theorem joe_paint_usage :
  ∀ (total_paint initial_remaining_paint final_remaining_paint paint_first_week paint_second_week total_used : ℕ),
  total_paint = 360 →
  initial_remaining_paint = total_paint - paint_first_week →
  final_remaining_paint = initial_remaining_paint - paint_second_week →
  paint_first_week = (2 * total_paint) / 3 →
  paint_second_week = (1 * initial_remaining_paint) / 5 →
  total_used = paint_first_week + paint_second_week →
  total_used = 264 :=
by
  sorry

end joe_paint_usage_l226_226905


namespace find_pairs_l226_226750

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_pairs (a b : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) : 
  (digit_sum (a^(b+1)) = a^b) ↔ 
  ((a = 1) ∨ (a = 3 ∧ b = 2) ∨ (a = 9 ∧ b = 1)) :=
by
  sorry

end find_pairs_l226_226750


namespace tom_pays_l226_226617

-- Definitions based on the conditions
def number_of_lessons : Nat := 10
def cost_per_lesson : Nat := 10
def free_lessons : Nat := 2

-- Desired proof statement
theorem tom_pays {number_of_lessons cost_per_lesson free_lessons : Nat} :
  (number_of_lessons - free_lessons) * cost_per_lesson = 80 :=
by
  sorry

end tom_pays_l226_226617


namespace conversion_7_dms_to_cms_conversion_5_hectares_to_sms_conversion_600_hectares_to_sqkms_conversion_200_sqsmeters_to_smeters_l226_226806

theorem conversion_7_dms_to_cms :
  7 * 100 = 700 :=
by
  sorry

theorem conversion_5_hectares_to_sms :
  5 * 10000 = 50000 :=
by
  sorry

theorem conversion_600_hectares_to_sqkms :
  600 / 100 = 6 :=
by
  sorry

theorem conversion_200_sqsmeters_to_smeters :
  200 / 100 = 2 :=
by
  sorry

end conversion_7_dms_to_cms_conversion_5_hectares_to_sms_conversion_600_hectares_to_sqkms_conversion_200_sqsmeters_to_smeters_l226_226806


namespace sqrt_meaningful_range_iff_l226_226338

noncomputable def sqrt_meaningful_range (x : ℝ) : Prop :=
  (∃ r : ℝ, r ≥ 0 ∧ r * r = x - 2023)

theorem sqrt_meaningful_range_iff {x : ℝ} : sqrt_meaningful_range x ↔ x ≥ 2023 :=
by
  sorry

end sqrt_meaningful_range_iff_l226_226338


namespace sum_first_odd_numbers_not_prime_l226_226180

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_first_odd_numbers_not_prime :
  ¬ (is_prime (1 + 3)) ∧
  ¬ (is_prime (1 + 3 + 5)) ∧
  ¬ (is_prime (1 + 3 + 5 + 7)) ∧
  ¬ (is_prime (1 + 3 + 5 + 7 + 9)) :=
by
  sorry

end sum_first_odd_numbers_not_prime_l226_226180


namespace triangle_angle_conditions_l226_226865

theorem triangle_angle_conditions
  (a b c : ℝ)
  (α β γ : ℝ)
  (h_triangle : c^2 = a^2 + 2 * b^2 * Real.cos β)
  (h_tri_angles : α + β + γ = 180):
  (γ = β / 2 + 90 ∧ α = 90 - 3 * β / 2 ∧ 0 < β ∧ β < 60) ∨ 
  (α = β / 2 ∧ γ = 180 - 3 * β / 2 ∧ 0 < β ∧ β < 120) :=
sorry

end triangle_angle_conditions_l226_226865


namespace cost_of_pumpkin_seeds_l226_226980

theorem cost_of_pumpkin_seeds (P : ℝ)
    (h1 : ∃(P_tomato P_chili : ℝ), P_tomato = 1.5 ∧ P_chili = 0.9) 
    (h2 : 3 * P + 4 * 1.5 + 5 * 0.9 = 18) 
    : P = 2.5 :=
by sorry

end cost_of_pumpkin_seeds_l226_226980


namespace satisfies_equation_l226_226795

theorem satisfies_equation (a b c : ℤ) (h₁ : a = b) (h₂ : b = c + 1) :
  a * (a - b) + b * (b - c) + c * (c - a) = 3 := 
by 
  sorry

end satisfies_equation_l226_226795


namespace annual_profit_function_correct_maximum_annual_profit_l226_226117

noncomputable def fixed_cost : ℝ := 60

noncomputable def variable_cost (x : ℝ) : ℝ :=
  if x < 12 then 
    0.5 * x^2 + 4 * x 
  else 
    11 * x + 100 / x - 39

noncomputable def selling_price_per_thousand : ℝ := 10

noncomputable def sales_revenue (x : ℝ) : ℝ := selling_price_per_thousand * x

noncomputable def annual_profit (x : ℝ) : ℝ := sales_revenue x - fixed_cost - variable_cost x

theorem annual_profit_function_correct : 
∀ x : ℝ, (0 < x ∧ x < 12 → annual_profit x = -0.5 * x^2 + 6 * x - fixed_cost) ∧ 
        (x ≥ 12 → annual_profit x = -x - 100 / x + 33) :=
sorry

theorem maximum_annual_profit : 
∃ x : ℝ, x = 12 ∧ annual_profit x = 38 / 3 :=
sorry

end annual_profit_function_correct_maximum_annual_profit_l226_226117


namespace problem_statement_l226_226203

theorem problem_statement (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 + 3 = 973 := 
by
  sorry

end problem_statement_l226_226203


namespace homework_time_decrease_l226_226234

theorem homework_time_decrease (x : ℝ) :
  let T_initial := 100
  let T_final := 70
  T_initial * (1 - x) * (1 - x) = T_final :=
by
  sorry

end homework_time_decrease_l226_226234


namespace sequence_bound_l226_226121

theorem sequence_bound (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) (h_seq : ∀ n, (a n) ^ 2 ≤ a (n + 1)) :
  ∀ n, a n < 1 / n :=
by
  intros
  sorry

end sequence_bound_l226_226121


namespace fractional_equation_solution_l226_226649

noncomputable def problem_statement (x : ℝ) : Prop :=
  (1 / (x - 1) + 1 = 2 / (x^2 - 1))

theorem fractional_equation_solution :
  ∀ x : ℝ, problem_statement x → x = -2 :=
by
  intro x hx
  sorry

end fractional_equation_solution_l226_226649


namespace exp_pi_gt_pi_exp_l226_226883

theorem exp_pi_gt_pi_exp (h : Real.pi > Real.exp 1) : Real.exp Real.pi > Real.pi ^ Real.exp 1 := by
  sorry

end exp_pi_gt_pi_exp_l226_226883


namespace john_total_beats_l226_226810

noncomputable def minutes_in_hour : ℕ := 60
noncomputable def hours_per_day : ℕ := 2
noncomputable def days_played : ℕ := 3
noncomputable def beats_per_minute : ℕ := 200

theorem john_total_beats :
  (beats_per_minute * hours_per_day * minutes_in_hour * days_played) = 72000 :=
by
  -- we will implement the proof here
  sorry

end john_total_beats_l226_226810


namespace walking_ring_width_l226_226336

theorem walking_ring_width (r₁ r₂ : ℝ) (h : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 20 * Real.pi) :
  r₁ - r₂ = 10 :=
by
  sorry

end walking_ring_width_l226_226336


namespace probability_scoring_80_or_above_probability_failing_exam_l226_226594

theorem probability_scoring_80_or_above (P : Set ℝ → ℝ) (B C D E : Set ℝ) :
  P B = 0.18 →
  P C = 0.51 →
  P D = 0.15 →
  P E = 0.09 →
  P (B ∪ C) = 0.69 :=
by
  intros hB hC hD hE
  sorry

theorem probability_failing_exam (P : Set ℝ → ℝ) (B C D E : Set ℝ) :
  P B = 0.18 →
  P C = 0.51 →
  P D = 0.15 →
  P E = 0.09 →
  P (B ∪ C ∪ D ∪ E) = 0.93 →
  1 - P (B ∪ C ∪ D ∪ E) = 0.07 :=
by
  intros hB hC hD hE hBCDE
  sorry

end probability_scoring_80_or_above_probability_failing_exam_l226_226594


namespace find_s_is_neg4_l226_226705

def g (x s : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

theorem find_s_is_neg4 : (∃ s : ℝ, g (-1) s = 0) ↔ (s = -4) :=
sorry

end find_s_is_neg4_l226_226705


namespace roots_calculation_l226_226963

theorem roots_calculation (c d : ℝ) (h : c^2 - 5*c + 6 = 0) (h' : d^2 - 5*d + 6 = 0) :
  c^3 + c^4 * d^2 + c^2 * d^4 + d^3 = 503 := by
  sorry

end roots_calculation_l226_226963


namespace LiFangOutfitChoices_l226_226129

variable (shirts skirts dresses : Nat) 

theorem LiFangOutfitChoices (h_shirts : shirts = 4) (h_skirts : skirts = 3) (h_dresses : dresses = 2) :
  shirts * skirts + dresses = 14 :=
by 
  -- Given the conditions and the calculations, the expected result follows.
  sorry

end LiFangOutfitChoices_l226_226129


namespace count_squares_and_cubes_l226_226767

theorem count_squares_and_cubes (bound : ℕ) (hk : bound = 1000) : 
  Nat.card {n : ℕ | n < bound ∧ (∃ k : ℕ, n = k ^ 6)} = 3 := 
by
  sorry

end count_squares_and_cubes_l226_226767


namespace math_problem_l226_226097

theorem math_problem
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a^3 + b^3 = 2) :
  (a + b) * (a^5 + b^5) ≥ 4 ∧ a + b ≤ 2 := 
by
  sorry

end math_problem_l226_226097


namespace find_b_l226_226990

open Real

noncomputable def triangle_b (a b c : ℝ) (A B C : ℝ) (sin_A sin_B : ℝ) (area : ℝ) : Prop :=
  B < π / 2 ∧
  sin_B = sqrt 7 / 4 ∧
  area = 5 * sqrt 7 / 4 ∧
  sin_A / sin_B = 5 * c / (2 * b) ∧
  a = 5 / 2 * c ∧
  area = 1 / 2 * a * c * sin_B

theorem find_b (a b c : ℝ) (A B C : ℝ) (sin_A sin_B : ℝ) (area : ℝ) :
  triangle_b a b c A B C sin_A sin_B area → b = sqrt 14 := by
  sorry

end find_b_l226_226990


namespace total_spending_in_4_years_is_680_l226_226916

-- Define the spending of each person
def trevor_spending_yearly : ℕ := 80
def reed_spending_yearly : ℕ := trevor_spending_yearly - 20
def quinn_spending_yearly : ℕ := reed_spending_yearly / 2

-- Define the total spending in one year
def total_spending_yearly : ℕ := trevor_spending_yearly + reed_spending_yearly + quinn_spending_yearly

-- Let's state what we need to prove
theorem total_spending_in_4_years_is_680 : 
  4 * total_spending_yearly = 680 := 
by 
  -- this is where the proof steps would go
  sorry

end total_spending_in_4_years_is_680_l226_226916


namespace nuts_eaten_condition_not_all_nuts_eaten_l226_226283

/-- proof problem with conditions and questions --/

-- Let's define the initial setup and the conditions:

def anya_has_all_nuts (nuts : Nat) := nuts > 3

def distribution (a b c : ℕ → ℕ) (n : ℕ) := 
  ((a (n + 1) = b n + c n + (a n % 2)) ∧ 
   (b (n + 1) = a n / 2) ∧ 
   (c (n + 1) = a n / 2))

def nuts_eaten (a b c : ℕ → ℕ) (n : ℕ) := 
  (a n % 2 > 0 ∨ b n % 2 > 0 ∨ c n % 2 > 0)

-- Prove at least one nut will be eaten
theorem nuts_eaten_condition (a b c : ℕ → ℕ) (n : ℕ) :
  anya_has_all_nuts (a 0) → distribution a b c n → nuts_eaten a b c n :=
sorry

-- Prove not all nuts will be eaten
theorem not_all_nuts_eaten (a b c : ℕ → ℕ):
  anya_has_all_nuts (a 0) → distribution a b c n → 
  ¬∀ (n: ℕ), (a n = 0 ∧ b n = 0 ∧ c n = 0) :=
sorry

end nuts_eaten_condition_not_all_nuts_eaten_l226_226283


namespace prob_mc_tf_correct_prob_at_least_one_mc_correct_l226_226899

-- Define the total number of questions and their types
def total_questions : ℕ := 5
def multiple_choice_questions : ℕ := 3
def true_false_questions : ℕ := 2
def total_outcomes : ℕ := total_questions * (total_questions - 1)

-- Probability calculation for one drawing a multiple-choice and the other drawing a true/false question
def prob_mc_tf : ℚ := (multiple_choice_questions * true_false_questions + true_false_questions * multiple_choice_questions) / total_outcomes

-- Probability calculation for at least one drawing a multiple-choice question
def prob_at_least_one_mc : ℚ := 1 - (true_false_questions * (true_false_questions - 1)) / total_outcomes

theorem prob_mc_tf_correct : prob_mc_tf = 3/5 := by
  sorry

theorem prob_at_least_one_mc_correct : prob_at_least_one_mc = 9/10 := by
  sorry

end prob_mc_tf_correct_prob_at_least_one_mc_correct_l226_226899


namespace units_digit_42_pow_4_add_24_pow_4_l226_226814

-- Define a function to get the units digit of a number.
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_42_pow_4_add_24_pow_4 : units_digit (42^4 + 24^4) = 2 := by
  sorry

end units_digit_42_pow_4_add_24_pow_4_l226_226814


namespace simplest_common_denominator_l226_226084

theorem simplest_common_denominator (x y : ℕ) (h1 : 2 * x ≠ 0) (h2 : 4 * y^2 ≠ 0) (h3 : 5 * x * y ≠ 0) :
  ∃ d : ℕ, d = 20 * x * y^2 :=
by {
  sorry
}

end simplest_common_denominator_l226_226084


namespace poly_diff_independent_of_x_l226_226228

theorem poly_diff_independent_of_x (x y: ℤ) (m n : ℤ) 
  (h1 : (1 - n = 0)) 
  (h2 : (m + 3 = 0)) :
  n - m = 4 := by
  sorry

end poly_diff_independent_of_x_l226_226228


namespace part_a_prob_part_b_expected_time_l226_226861

/--  
Let total_suitcases be 200.
Let business_suitcases be 10.
Let total_wait_time be 120.
Let arrival_interval be 2.
--/

def total_suitcases : ℕ := 200
def business_suitcases : ℕ := 10
def total_wait_time : ℕ := 120 
def arrival_interval : ℕ := 2

def two_minutes_suitcases : ℕ := total_wait_time / arrival_interval
def prob_last_suitcase_at_n_minutes (n : ℕ) : ℚ := 
  (Nat.choose (two_minutes_suitcases - 1) (business_suitcases - 1) : ℚ) / 
  (Nat.choose total_suitcases business_suitcases : ℚ)

theorem part_a_prob : 
  prob_last_suitcase_at_n_minutes 2 = 
  (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) := sorry

noncomputable def expected_position_last_suitcase (total_pos : ℕ) (suitcases_per_group : ℕ) : ℚ :=
  (total_pos * business_suitcases : ℚ) / (business_suitcases + 1)

theorem part_b_expected_time : 
  expected_position_last_suitcase 201 11 * arrival_interval = 
  (4020 : ℚ) / 11 := sorry

end part_a_prob_part_b_expected_time_l226_226861


namespace decimal_difference_l226_226122

theorem decimal_difference : (0.650 : ℝ) - (1 / 8 : ℝ) = 0.525 := by
  sorry

end decimal_difference_l226_226122


namespace circle_radius_l226_226355

theorem circle_radius : ∀ (x y : ℝ), x^2 + 10*x + y^2 - 8*y + 25 = 0 → False := sorry

end circle_radius_l226_226355


namespace total_cost_l226_226314

theorem total_cost (cost_sandwich cost_soda cost_cookie : ℕ)
    (num_sandwich num_soda num_cookie : ℕ) 
    (h1 : cost_sandwich = 4) 
    (h2 : cost_soda = 3) 
    (h3 : cost_cookie = 2) 
    (h4 : num_sandwich = 4) 
    (h5 : num_soda = 6) 
    (h6 : num_cookie = 7):
    cost_sandwich * num_sandwich + cost_soda * num_soda + cost_cookie * num_cookie = 48 :=
by
  sorry

end total_cost_l226_226314


namespace interest_calculation_l226_226465

variables (P R SI : ℝ) (T : ℕ)

-- Given conditions
def principal := (P = 8)
def rate := (R = 0.05)
def simple_interest := (SI = 4.8)

-- Goal
def time_calculated := (T = 12)

-- Lean statement combining the conditions
theorem interest_calculation : principal P → rate R → simple_interest SI → T = 12 :=
by
  intros hP hR hSI
  sorry

end interest_calculation_l226_226465


namespace units_digit_7_pow_5_pow_3_l226_226802

theorem units_digit_7_pow_5_pow_3 : (7 ^ (5 ^ 3)) % 10 = 7 := by
  sorry

end units_digit_7_pow_5_pow_3_l226_226802


namespace sum_of_squares_of_consecutive_integers_l226_226785

theorem sum_of_squares_of_consecutive_integers (a : ℕ) (h : (a - 1) * a * (a + 1) * (a + 2) = 12 * ((a - 1) + a + (a + 1) + (a + 2))) :
  (a - 1)^2 + a^2 + (a + 1)^2 + (a + 2)^2 = 86 :=
by
  sorry

end sum_of_squares_of_consecutive_integers_l226_226785


namespace transport_cost_l226_226139

theorem transport_cost (mass_g: ℕ) (cost_per_kg : ℕ) (mass_kg : ℝ) 
  (h1 : mass_g = 300) (h2 : mass_kg = (mass_g : ℝ) / 1000) 
  (h3: cost_per_kg = 18000)
  : mass_kg * cost_per_kg = 5400 := by
  sorry

end transport_cost_l226_226139


namespace find_a2_l226_226109

noncomputable def a_sequence (k : ℕ+) (n : ℕ) : ℚ :=
  -(1 / 2 : ℚ) * n^2 + k * n

theorem find_a2
  (k : ℕ+)
  (max_S : ∀ n : ℕ, a_sequence k n ≤ 8)
  (max_reached : ∃ n : ℕ, a_sequence k n = 8) :
  a_sequence 4 2 - a_sequence 4 1 = 5 / 2 :=
by
  -- To be proved, insert appropriate steps here
  sorry

end find_a2_l226_226109


namespace math_proof_problem_l226_226043

noncomputable def ellipse_standard_eq (a b : ℝ) : Prop :=
  a = 4 ∧ b = 2 * Real.sqrt 3

noncomputable def conditions (e : ℝ) (vertex : ℝ × ℝ) (p q : ℝ × ℝ) : Prop :=
  e = 1 / 2
  ∧ vertex = (0, 2 * Real.sqrt 3)  -- focus of the parabola
  ∧ p = (-2, -3)
  ∧ q = (-2, 3)

noncomputable def max_area_quadrilateral (area : ℝ) : Prop :=
  area = 12 * Real.sqrt 3

theorem math_proof_problem : 
  ∃ a b p q area, ellipse_standard_eq a b ∧ conditions (1/2) (0, 2 * Real.sqrt 3) p q 
  ∧ p = (-2, -3) ∧ q = (-2, 3) → max_area_quadrilateral area := 
  sorry

end math_proof_problem_l226_226043


namespace min_total_trees_l226_226761

theorem min_total_trees (L X : ℕ) (h1: 13 * L < 100 * X) (h2: 100 * X < 14 * L) : L ≥ 15 :=
  sorry

end min_total_trees_l226_226761


namespace angle_A_measure_in_triangle_l226_226160

theorem angle_A_measure_in_triangle (A B C : ℝ) 
  (h1 : B = 15)
  (h2 : C = 3 * B) 
  (angle_sum : A + B + C = 180) :
  A = 120 :=
by
  -- We'll fill in the proof steps later
  sorry

end angle_A_measure_in_triangle_l226_226160


namespace sparrows_initial_count_l226_226152

theorem sparrows_initial_count (a b c : ℕ) 
  (h1 : a + b + c = 24)
  (h2 : a - 4 = b + 1)
  (h3 : b + 1 = c + 3) : 
  a = 12 ∧ b = 7 ∧ c = 5 :=
by
  sorry

end sparrows_initial_count_l226_226152


namespace num_parallelograms_4x6_grid_l226_226497

noncomputable def numberOfParallelograms (m n : ℕ) : ℕ :=
  let numberOfRectangles := (Nat.choose (m + 1) 2) * (Nat.choose (n + 1) 2)
  let numberOfSquares := (m * n) + ((m - 1) * (n - 1)) + ((m - 2) * (n - 2)) + ((m - 3) * (n - 3))
  let numberOfRectanglesWithUnequalSides := numberOfRectangles - numberOfSquares
  2 * numberOfRectanglesWithUnequalSides

theorem num_parallelograms_4x6_grid : numberOfParallelograms 4 6 = 320 := by
  sorry

end num_parallelograms_4x6_grid_l226_226497


namespace emma_age_l226_226214

variables (O N L E : ℕ)

def oliver_eq : Prop := O = N - 5
def nancy_eq : Prop := N = L + 6
def emma_eq : Prop := E = L + 4
def oliver_age : Prop := O = 16

theorem emma_age :
  oliver_eq O N ∧ nancy_eq N L ∧ emma_eq E L ∧ oliver_age O → E = 19 :=
by
  sorry

end emma_age_l226_226214


namespace find_min_n_l226_226464

variable (a : Nat → Int)
variable (S : Nat → Int)
variable (d : Nat)
variable (n : Nat)

-- Definitions based on given conditions
def arithmetic_sequence (a : Nat → Int) (d : Nat) : Prop :=
  ∀ n, a (n + 1) = a n + d

def a1_eq_neg3 (a : Nat → Int) : Prop :=
  a 1 = -3

def condition (a : Nat → Int) (d : Nat) : Prop :=
  11 * a 5 = 5 * a 8

-- Correct answer condition
def minimized_sum_condition (a : Nat → Int) (S : Nat → Int) (d : Nat) (n : Nat) : Prop :=
  S n ≤ S (n + 1)

theorem find_min_n (a : Nat → Int) (S : Nat → Int) (d : Nat) :
  arithmetic_sequence a d ->
  a1_eq_neg3 a ->
  condition a 2 ->
  minimized_sum_condition a S 2 2 :=
by
  sorry

end find_min_n_l226_226464


namespace lines_are_skew_l226_226506

def line1 (a t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 3 * t, 3 + 4 * t, a + 5 * t)

def line2 (u : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 6 * u, 2 + 2 * u, 1 + 2 * u)

theorem lines_are_skew (a : ℝ) :
  ¬(∃ t u : ℝ, line1 a t = line2 u) ↔ a ≠ 5 / 3 :=
sorry

end lines_are_skew_l226_226506


namespace geometric_progression_solution_l226_226194

noncomputable def first_term_of_geometric_progression (b2 b6 : ℚ) (q : ℚ) : ℚ := 
  b2 / q
  
theorem geometric_progression_solution 
  (b2 b6 : ℚ)
  (h1 : b2 = 37 + 1/3)
  (h2 : b6 = 2 + 1/3) :
  ∃ a q : ℚ, a = 224 / 3 ∧ q = 1/2 ∧ b2 = a * q ∧ b6 = a * q^5 :=
by
  sorry

end geometric_progression_solution_l226_226194


namespace Zenobius_more_descendants_l226_226763

/-- Total number of descendants in King Pafnutius' lineage --/
def descendants_Pafnutius : Nat :=
  2 + 60 * 2 + 20 * 1

/-- Total number of descendants in King Zenobius' lineage --/
def descendants_Zenobius : Nat :=
  4 + 35 * 3 + 35 * 1

theorem Zenobius_more_descendants : descendants_Zenobius > descendants_Pafnutius := by
  sorry

end Zenobius_more_descendants_l226_226763


namespace sqrt_simplify_l226_226008

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l226_226008


namespace find_all_functions_l226_226083

theorem find_all_functions (f : ℕ → ℕ) : 
  (∀ a b : ℕ, 0 < a → 0 < b → f (a^2 + b^2) = f a * f b) →
  (∀ a : ℕ, 0 < a → f (a^2) = f a ^ 2) →
  (∀ n : ℕ, 0 < n → f n = 1) :=
by
  intros h1 h2 a ha
  sorry

end find_all_functions_l226_226083


namespace sufficient_but_not_necessary_l226_226138

theorem sufficient_but_not_necessary (a b : ℝ) : (a > b ∧ b > 0) → (a^2 > b^2) ∧ ¬((a^2 > b^2) → (a > b ∧ b > 0)) :=
by
  sorry

end sufficient_but_not_necessary_l226_226138


namespace volume_of_second_cube_is_twosqrt2_l226_226047

noncomputable def side_length (volume : ℝ) : ℝ :=
  volume^(1/3)

noncomputable def surface_area (side : ℝ) : ℝ :=
  6 * side^2

theorem volume_of_second_cube_is_twosqrt2
  (v1 : ℝ)
  (h1 : v1 = 1)
  (A1 := surface_area (side_length v1))
  (A2 := 2 * A1)
  (s2 := (A2 / 6)^(1/2)) :
  (s2^3 = 2 * Real.sqrt 2) :=
by
  sorry

end volume_of_second_cube_is_twosqrt2_l226_226047


namespace trig_identity_l226_226738

theorem trig_identity (α : ℝ) (h1 : (-Real.pi / 2) < α ∧ α < 0)
  (h2 : Real.sin α + Real.cos α = 1 / 5) :
  1 / (Real.cos α ^ 2 - Real.sin α ^ 2) = 25 / 7 := 
by 
  sorry

end trig_identity_l226_226738


namespace two_digits_same_in_three_digit_numbers_l226_226290

theorem two_digits_same_in_three_digit_numbers (h1 : (100 : ℕ) ≤ n) (h2 : n < 600) : 
  ∃ n, n = 140 := sorry

end two_digits_same_in_three_digit_numbers_l226_226290


namespace correct_calculation_l226_226540

variable (a b : ℝ)

theorem correct_calculation : (-a^3)^2 = a^6 := 
by 
  sorry

end correct_calculation_l226_226540


namespace cos_theta_value_l226_226664

open Real

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (2, 0)

noncomputable def cos_theta (u v : ℝ × ℝ) : ℝ :=
  (u.1 * v.1 + u.2 * v.2) / (Real.sqrt (u.1 ^ 2 + u.2 ^ 2) * Real.sqrt (v.1 ^ 2 + v.2 ^ 2))

theorem cos_theta_value :
  cos_theta a b = 3 * Real.sqrt 10 / 10 :=
by
  sorry

end cos_theta_value_l226_226664


namespace Q_current_age_l226_226082

-- Definitions for the current ages of P and Q
variable (P Q : ℕ)

-- Conditions
-- 1. P + Q = 100
-- 2. P = 3 * (Q - (P - Q))  (from P is thrice as old as Q was when P was as old as Q is now)

axiom age_sum : P + Q = 100
axiom age_relation : P = 3 * (Q - (P - Q))

theorem Q_current_age : Q = 40 :=
by
  sorry

end Q_current_age_l226_226082


namespace count_possible_P_l226_226327

-- Define the distinct digits with initial conditions
def digits : Type := {n // n ≥ 0 ∧ n ≤ 9}

-- Define the parameters P, Q, R, S as distinct digits
variables (P Q R S : digits)

-- Define the condition that P, Q, R, S are distinct.
def distinct (P Q R S : digits) : Prop := 
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S

-- Assertion conditions based on a valid subtraction layout
def valid_subtraction (P Q R S : digits) : Prop :=
  Q.val - P.val = S.val ∧ (P.val - R.val = P.val) ∧ (P.val - Q.val = S.val)

-- Prove that there are exactly 9 possible values for P.
theorem count_possible_P : ∃ n : ℕ, n = 9 ∧ ∀ P Q R S : digits, distinct P Q R S → valid_subtraction P Q R S → n = 9 :=
by sorry

end count_possible_P_l226_226327


namespace convex_polyhedron_theorems_l226_226151

-- Definitions for convex polyhedron and symmetric properties
structure ConvexSymmetricPolyhedron (α : Type*) :=
  (isConvex : Bool)
  (isCentrallySymmetric : Bool)
  (crossSection : α → α → α)
  (center : α)

-- Definitions for proofs required
def largest_cross_section_area
  (P : ConvexSymmetricPolyhedron ℝ) : Prop :=
  ∀ (p : ℝ), P.crossSection p P.center ≤ P.crossSection P.center P.center

def largest_radius_circle (P : ConvexSymmetricPolyhedron ℝ) : Prop :=
  ¬∀ (p : ℝ), P.crossSection p P.center = P.crossSection P.center P.center

-- The theorem combining both statements
theorem convex_polyhedron_theorems
  (P : ConvexSymmetricPolyhedron ℝ) :
  P.isConvex = true ∧ 
  P.isCentrallySymmetric = true →
  (largest_cross_section_area P) ∧ (largest_radius_circle P) :=
by 
  sorry

end convex_polyhedron_theorems_l226_226151


namespace wam_gm_gt_hm_l226_226788

noncomputable def wam (w v a b : ℝ) : ℝ := w * a + v * b
noncomputable def gm (a b : ℝ) : ℝ := Real.sqrt (a * b)
noncomputable def hm (a b : ℝ) : ℝ := (2 * a * b) / (a + b)

theorem wam_gm_gt_hm
  (a b w v : ℝ)
  (h1 : 0 < a ∧ 0 < b)
  (h2 : 0 < w ∧ 0 < v)
  (h3 : w + v = 1)
  (h4 : a ≠ b) :
  wam w v a b > gm a b ∧ gm a b > hm a b :=
by
  -- Proof omitted
  sorry

end wam_gm_gt_hm_l226_226788


namespace ones_digit_of_prime_sequence_l226_226432

theorem ones_digit_of_prime_sequence (p q r s : ℕ) (h1 : p > 5) 
    (h2 : p < q ∧ q < r ∧ r < s) (h3 : q - p = 8 ∧ r - q = 8 ∧ s - r = 8) 
    (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s) : 
    p % 10 = 3 :=
by
  sorry

end ones_digit_of_prime_sequence_l226_226432


namespace cone_tangent_min_lateral_area_l226_226431

/-- 
Given a cone with volume π / 6, prove that when the lateral area of the cone is minimized,
the tangent of the angle between the slant height and the base is sqrt(2).
-/
theorem cone_tangent_min_lateral_area :
  ∀ (r h l : ℝ), (π / 6 = (1 / 3) * π * r^2 * h) →
    (h = 1 / (2 * r^2)) →
    (l = Real.sqrt (r^2 + h^2)) →
    ((π * r * l) ≥ (3 / 4 * π)) →
    (r = Real.sqrt (2) / 2) →
    (h / r = Real.sqrt (2)) :=
by
  intro r h l V_cond h_cond l_def min_lateral_area r_val
  -- Proof steps go here (omitted as per the instruction)
  sorry

end cone_tangent_min_lateral_area_l226_226431


namespace vasya_faster_than_petya_l226_226550

theorem vasya_faster_than_petya 
  (L : ℝ) (v : ℝ) (x : ℝ) (t : ℝ) 
  (meeting_condition : (v + x * v) * t = L)
  (petya_lap : v * t = L)
  (vasya_meet_petya_after_lap : x * v * t = 2 * L) :
  x = (1 + Real.sqrt 5) / 2 := 
by 
  sorry

end vasya_faster_than_petya_l226_226550


namespace twenty_four_game_l226_226098

-- Definition of the cards' values
def card2 : ℕ := 2
def card5 : ℕ := 5
def cardJ : ℕ := 11
def cardQ : ℕ := 12

-- Theorem stating the proof
theorem twenty_four_game : card2 * (cardJ - card5) + cardQ = 24 :=
by
  sorry

end twenty_four_game_l226_226098


namespace factorize_m_square_minus_4m_l226_226407

theorem factorize_m_square_minus_4m (m : ℝ) : m^2 - 4 * m = m * (m - 4) :=
by
  sorry

end factorize_m_square_minus_4m_l226_226407


namespace max_area_of_rectangular_playground_l226_226230

theorem max_area_of_rectangular_playground (P : ℕ) (hP : P = 160) :
  (∃ (x y : ℕ), 2 * (x + y) = P ∧ x * y = 1600) :=
by
  sorry

end max_area_of_rectangular_playground_l226_226230


namespace inequality_solution_sum_of_m_and_2n_l226_226984

-- Define the function f(x) = |x - a|
def f (x a : ℝ) : ℝ := abs (x - a)

-- Part (1): The inequality problem for a = 2
theorem inequality_solution (x : ℝ) :
  f x 2 ≥ 4 - abs (x - 1) → x ≤ 2 / 3 := sorry

-- Part (2): Given conditions with solution set [0, 2] and condition on m and n
theorem sum_of_m_and_2n (m n : ℝ) (h₁ : m > 0) (h₂ : n > 0) (h₃ : ∀ x, f x 1 ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) (h₄ : 1 / m + 1 / (2 * n) = 1) :
  m + 2 * n ≥ 4 := sorry

end inequality_solution_sum_of_m_and_2n_l226_226984


namespace booth_earnings_after_5_days_l226_226110

def booth_daily_popcorn_earnings := 50
def booth_daily_cotton_candy_earnings := 3 * booth_daily_popcorn_earnings
def booth_total_daily_earnings := booth_daily_popcorn_earnings + booth_daily_cotton_candy_earnings
def booth_total_expenses := 30 + 75

theorem booth_earnings_after_5_days :
  5 * booth_total_daily_earnings - booth_total_expenses = 895 :=
by
  sorry

end booth_earnings_after_5_days_l226_226110


namespace net_percentage_change_l226_226440

theorem net_percentage_change (k m : ℝ) : 
  let scale_factor_1 := 1 - k / 100
  let scale_factor_2 := 1 + m / 100
  let overall_scale_factor := scale_factor_1 * scale_factor_2
  let percentage_change := (overall_scale_factor - 1) * 100
  percentage_change = m - k - k * m / 100 := 
by 
  sorry

end net_percentage_change_l226_226440


namespace fourth_bus_people_difference_l226_226889

def bus1_people : Nat := 12
def bus2_people : Nat := 2 * bus1_people
def bus3_people : Nat := bus2_people - 6
def total_people : Nat := 75
def bus4_people : Nat := total_people - (bus1_people + bus2_people + bus3_people)
def difference_people : Nat := bus4_people - bus1_people

theorem fourth_bus_people_difference : difference_people = 9 := by
  -- Proof logic here
  sorry

end fourth_bus_people_difference_l226_226889


namespace goods_train_speed_l226_226388

theorem goods_train_speed:
  let speed_mans_train := 100   -- in km/h
  let length_goods_train := 280 -- in meters
  let passing_time := 9         -- in seconds
  ∃ speed_goods_train: ℝ, 
  (speed_mans_train + speed_goods_train) * (5 / 18) * passing_time = length_goods_train ↔ speed_goods_train = 12 :=
by
  sorry

end goods_train_speed_l226_226388


namespace roots_reciprocal_sum_l226_226363

theorem roots_reciprocal_sum
  (a b c : ℂ)
  (h : Polynomial.roots (Polynomial.C 1 + Polynomial.X - Polynomial.C 1 * Polynomial.X^2 + Polynomial.C 1 * Polynomial.X^3) = {a, b, c}) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = -2 :=
by
  sorry

end roots_reciprocal_sum_l226_226363


namespace train_speed_in_km_hr_l226_226675

noncomputable def train_length : ℝ := 320
noncomputable def crossing_time : ℝ := 7.999360051195905
noncomputable def speed_in_meter_per_sec : ℝ := train_length / crossing_time
noncomputable def meter_per_sec_to_km_hr (speed_mps : ℝ) : ℝ := speed_mps * 3.6
noncomputable def expected_speed : ℝ := 144.018001125

theorem train_speed_in_km_hr :
  meter_per_sec_to_km_hr speed_in_meter_per_sec = expected_speed := by
  sorry

end train_speed_in_km_hr_l226_226675


namespace calculation_eq_990_l226_226804

theorem calculation_eq_990 : (0.0077 * 3.6) / (0.04 * 0.1 * 0.007) = 990 :=
by
  sorry

end calculation_eq_990_l226_226804


namespace square_perimeter_is_64_l226_226211

-- Given conditions
variables (s : ℕ)
def is_square_divided_into_four_congruent_rectangles : Prop :=
  ∀ (r : ℕ), r = 4 → (∀ (p : ℕ), p = (5 * s) / 2 → p = 40)

-- Lean 4 statement for the proof problem
theorem square_perimeter_is_64 
  (h : is_square_divided_into_four_congruent_rectangles s) 
  (hs : (5 * s) / 2 = 40) : 
  4 * s = 64 :=
by
  sorry

end square_perimeter_is_64_l226_226211


namespace inequality_solution_l226_226379

theorem inequality_solution
  (f : ℝ → ℝ)
  (h_deriv : ∀ x : ℝ, deriv f x > 2 * f x)
  (h_value : f (1/2) = Real.exp 1)
  (x : ℝ)
  (h_pos : 0 < x) :
  f (Real.log x) < x^2 ↔ x < Real.exp (1/2) :=
sorry

end inequality_solution_l226_226379


namespace beer_drawing_time_l226_226326

theorem beer_drawing_time :
  let rate_A := 1 / 5
  let rate_C := 1 / 4
  let combined_rate := 9 / 20
  let extra_beer := 12
  let total_drawn := 48
  let t := total_drawn / combined_rate
  t = 48 * 20 / 9 :=
by {
  sorry -- proof not required
}

end beer_drawing_time_l226_226326


namespace cannot_form_isosceles_triangle_l226_226420

theorem cannot_form_isosceles_triangle :
  ¬ ∃ (sticks : Finset ℕ) (a b c : ℕ), a ∈ sticks ∧ b ∈ sticks ∧ c ∈ sticks ∧
  a + b > c ∧ a + c > b ∧ b + c > a ∧ -- Triangle inequality
  (a = b ∨ b = c ∨ a = c) ∧ -- Isosceles condition
  sticks ⊆ {1, 2, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^9} := sorry

end cannot_form_isosceles_triangle_l226_226420


namespace solve_for_y_l226_226066

theorem solve_for_y (y : ℝ) (h : (5 - 2 / y)^(1/3) = -3) : y = 1 / 16 := 
sorry

end solve_for_y_l226_226066


namespace water_displaced_volume_square_l226_226579

-- Given conditions:
def radius : ℝ := 5
def height : ℝ := 10
def cube_side : ℝ := 6

-- Theorem statement for the problem
theorem water_displaced_volume_square (r h s : ℝ) (w : ℝ) 
  (hr : r = 5) 
  (hh : h = 10) 
  (hs : s = 6) : 
  (w * w) = 13141.855 :=
by 
  sorry

end water_displaced_volume_square_l226_226579


namespace positive_solution_l226_226134

theorem positive_solution (x : ℝ) (h : (1 / 2) * (3 * x^2 - 1) = (x^2 - 50 * x - 10) * (x^2 + 25 * x + 5)) : x = 25 + Real.sqrt 159 :=
sorry

end positive_solution_l226_226134


namespace arithmetic_sequence_second_term_l226_226549

theorem arithmetic_sequence_second_term (a d : ℝ) (h : a + (a + 2 * d) = 8) : a + d = 4 :=
sorry

end arithmetic_sequence_second_term_l226_226549


namespace range_of_fx_l226_226034

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^k

theorem range_of_fx (k : ℝ) (x : ℝ) (h1 : k < -1) (h2 : x ∈ Set.Ici (0.5)) :
  Set.Icc (0 : ℝ) 2 = {y | ∃ x, f x k = y ∧ x ∈ Set.Ici 0.5} :=
sorry

end range_of_fx_l226_226034


namespace weighted_average_correct_l226_226710

-- Define the marks and credits for each subject
def marks_english := 90
def marks_mathematics := 92
def marks_physics := 85
def marks_chemistry := 87
def marks_biology := 85

def credits_english := 3
def credits_mathematics := 4
def credits_physics := 4
def credits_chemistry := 3
def credits_biology := 2

-- Define the weighted sum and total credits
def weighted_sum := marks_english * credits_english + marks_mathematics * credits_mathematics + marks_physics * credits_physics + marks_chemistry * credits_chemistry + marks_biology * credits_biology
def total_credits := credits_english + credits_mathematics + credits_physics + credits_chemistry + credits_biology

-- Prove that the weighted average is 88.0625
theorem weighted_average_correct : (weighted_sum.toFloat / total_credits.toFloat) = 88.0625 :=
by 
  sorry

end weighted_average_correct_l226_226710


namespace number_of_clients_l226_226597

theorem number_of_clients (cars_clients_selects : ℕ)
                          (cars_selected_per_client : ℕ)
                          (each_car_selected_times : ℕ)
                          (total_cars : ℕ)
                          (h1 : total_cars = 18)
                          (h2 : cars_clients_selects = total_cars * each_car_selected_times)
                          (h3 : each_car_selected_times = 3)
                          (h4 : cars_selected_per_client = 3)
                          : total_cars * each_car_selected_times / cars_selected_per_client = 18 :=
by {
  sorry
}

end number_of_clients_l226_226597


namespace total_fuel_proof_l226_226259

def highway_consumption_60 : ℝ := 3 -- gallons per mile at 60 mph
def highway_consumption_70 : ℝ := 3.5 -- gallons per mile at 70 mph
def city_consumption_30 : ℝ := 5 -- gallons per mile at 30 mph
def city_consumption_15 : ℝ := 4.5 -- gallons per mile at 15 mph

def day1_highway_60_hours : ℝ := 2 -- hours driven at 60 mph on the highway
def day1_highway_70_hours : ℝ := 1 -- hours driven at 70 mph on the highway
def day1_city_30_hours : ℝ := 4 -- hours driven at 30 mph in the city

def day2_highway_70_hours : ℝ := 3 -- hours driven at 70 mph on the highway
def day2_city_15_hours : ℝ := 3 -- hours driven at 15 mph in the city
def day2_city_30_hours : ℝ := 1 -- hours driven at 30 mph in the city

def day3_highway_60_hours : ℝ := 1.5 -- hours driven at 60 mph on the highway
def day3_city_30_hours : ℝ := 3 -- hours driven at 30 mph in the city
def day3_city_15_hours : ℝ := 1 -- hours driven at 15 mph in the city

def total_fuel_consumption (c1 c2 c3 c4 : ℝ) (h1 h2 h3 h4 h5 h6 h7 h8 h9 : ℝ) :=
  (h1 * 60 * c1) + (h2 * 70 * c2) + (h3 * 30 * c3) + 
  (h4 * 70 * c2) + (h5 * 15 * c4) + (h6 * 30 * c3) +
  (h7 * 60 * c1) + (h8 * 30 * c3) + (h9 * 15 * c4)

theorem total_fuel_proof :
  total_fuel_consumption highway_consumption_60 highway_consumption_70 city_consumption_30 city_consumption_15
  day1_highway_60_hours day1_highway_70_hours day1_city_30_hours day2_highway_70_hours
  day2_city_15_hours day2_city_30_hours day3_highway_60_hours day3_city_30_hours day3_city_15_hours
  = 3080 := by
  sorry

end total_fuel_proof_l226_226259


namespace male_red_ants_percentage_l226_226557

noncomputable def percentage_of_total_ant_population_that_are_red_females (total_population_pct red_population_pct percent_red_are_females : ℝ) : ℝ :=
    (percent_red_are_females / 100) * red_population_pct

noncomputable def percentage_of_total_ant_population_that_are_red_males (total_population_pct red_population_pct percent_red_are_females : ℝ) : ℝ :=
    red_population_pct - percentage_of_total_ant_population_that_are_red_females total_population_pct red_population_pct percent_red_are_females

theorem male_red_ants_percentage (total_population_pct red_population_pct percent_red_are_females male_red_ants_pct : ℝ) :
    red_population_pct = 85 → percent_red_are_females = 45 → male_red_ants_pct = 46.75 →
    percentage_of_total_ant_population_that_are_red_males total_population_pct red_population_pct percent_red_are_females = male_red_ants_pct :=
by
sorry

end male_red_ants_percentage_l226_226557


namespace parametric_eq_C1_rectangular_eq_C2_max_dist_PQ_l226_226268

-- Curve C1 given by x^2 / 9 + y^2 = 1, prove its parametric form
theorem parametric_eq_C1 (α : ℝ) : 
  (∃ (x y : ℝ), x = 3 * Real.cos α ∧ y = Real.sin α ∧ (x ^ 2 / 9 + y ^ 2 = 1)) := 
sorry

-- Curve C2 given by ρ^2 - 8ρ sin θ + 15 = 0, prove its rectangular form
theorem rectangular_eq_C2 (ρ θ : ℝ) : 
  (∃ (x y : ℝ), x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ 
    (ρ ^ 2 - 8 * ρ * Real.sin θ + 15 = 0) ↔ (x ^ 2 + y ^ 2 - 8 * y + 15 = 0)) := 
sorry

-- Prove the maximum value of |PQ|
theorem max_dist_PQ : 
  (∃ (P Q : ℝ × ℝ), 
    (P = (3 * Real.cos α, Real.sin α)) ∧ 
    (Q = (0, 4)) ∧ 
    (∀ α : ℝ, Real.sqrt ((3 * Real.cos α) ^ 2 + (Real.sin α - 4) ^ 2) ≤ 8)) := 
sorry

end parametric_eq_C1_rectangular_eq_C2_max_dist_PQ_l226_226268


namespace sarah_reads_40_words_per_minute_l226_226588

-- Define the conditions as constants
def words_per_page := 100
def pages_per_book := 80
def reading_hours := 20
def number_of_books := 6

-- Convert hours to minutes
def total_reading_time := reading_hours * 60

-- Calculate the total number of words in one book
def words_per_book := words_per_page * pages_per_book

-- Calculate the total number of words in all books
def total_words := words_per_book * number_of_books

-- Define the words read per minute
def words_per_minute := total_words / total_reading_time

-- Theorem statement: Sarah reads 40 words per minute
theorem sarah_reads_40_words_per_minute : words_per_minute = 40 :=
by
  sorry

end sarah_reads_40_words_per_minute_l226_226588


namespace sticks_form_equilateral_triangle_l226_226628

theorem sticks_form_equilateral_triangle {n : ℕ} (h1 : n ≥ 5) (h2 : n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :
  (∃ a b c, a = b ∧ b = c ∧ a + b + c = (n * (n + 1)) / 2) :=
by
  sorry

end sticks_form_equilateral_triangle_l226_226628


namespace tax_diminished_by_16_percent_l226_226654

variables (T X : ℝ)

-- Condition: The new revenue is 96.6% of the original revenue
def new_revenue_effect : Prop :=
  (1.15 * (T - X) / 100) = (T / 100) * 0.966

-- Target: Prove that X is 16% of T
theorem tax_diminished_by_16_percent (h : new_revenue_effect T X) : X = 0.16 * T :=
sorry

end tax_diminished_by_16_percent_l226_226654


namespace Ashutosh_time_to_complete_job_l226_226708

noncomputable def SureshWorkRate : ℝ := 1 / 15
noncomputable def AshutoshWorkRate (A : ℝ) : ℝ := 1 / A
noncomputable def SureshWorkIn9Hours : ℝ := 9 * SureshWorkRate

theorem Ashutosh_time_to_complete_job (A : ℝ) :
  (1 - SureshWorkIn9Hours) * AshutoshWorkRate A = 14 / 35 →
  A = 35 :=
by
  sorry

end Ashutosh_time_to_complete_job_l226_226708


namespace mary_no_torn_cards_l226_226666

theorem mary_no_torn_cards
  (T : ℕ) -- number of Mary's initial torn baseball cards
  (initial_cards : ℕ := 18) -- initial baseball cards
  (fred_cards : ℕ := 26) -- baseball cards given by Fred
  (bought_cards : ℕ := 40) -- baseball cards bought
  (total_cards : ℕ := 84) -- total baseball cards Mary has now
  (h : initial_cards - T + fred_cards + bought_cards = total_cards)
  : T = 0 :=
by sorry

end mary_no_torn_cards_l226_226666


namespace point_P_coordinates_l226_226253

theorem point_P_coordinates :
  ∃ (P : ℝ × ℝ), P.1 > 0 ∧ P.2 < 0 ∧ abs P.2 = 3 ∧ abs P.1 = 8 ∧ P = (8, -3) :=
sorry

end point_P_coordinates_l226_226253


namespace quadratic_equal_roots_l226_226793

theorem quadratic_equal_roots (a : ℝ) : (∀ x : ℝ, x * (x + 1) + a * x = 0) → a = -1 :=
by sorry

end quadratic_equal_roots_l226_226793


namespace total_coins_is_16_l226_226124

theorem total_coins_is_16 (x y : ℕ) (h₁ : x ≠ y) (h₂ : x^2 - y^2 = 16 * (x - y)) : x + y = 16 := 
sorry

end total_coins_is_16_l226_226124


namespace probability_multiple_of_7_condition_l226_226306

theorem probability_multiple_of_7_condition :
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100 ∧ a ≠ b ∧ (ab + a + b + 1) % 7 = 0 → 
  (1295 / 4950 = 259 / 990) :=
sorry

end probability_multiple_of_7_condition_l226_226306


namespace rectangular_field_perimeter_l226_226766

theorem rectangular_field_perimeter
  (a b : ℝ)
  (diag_eq : a^2 + b^2 = 1156)
  (area_eq : a * b = 240)
  (side_relation : a = 2 * b) :
  2 * (a + b) = 91.2 :=
by
  sorry

end rectangular_field_perimeter_l226_226766


namespace anthony_pencils_l226_226025

def initial_pencils : ℝ := 56.0  -- Condition 1
def pencils_left : ℝ := 47.0     -- Condition 2
def pencils_given : ℝ := 9.0     -- Correct Answer

theorem anthony_pencils :
  initial_pencils - pencils_left = pencils_given :=
by
  sorry

end anthony_pencils_l226_226025


namespace sphere_surface_area_l226_226515

-- Let A, B, C, D be distinct points on the same sphere
variables (A B C D : ℝ)

-- Defining edges AB, AC, AD and their lengths
variables (AB AC AD : ℝ)
variable (is_perpendicular : AB * AC = 0 ∧ AB * AD = 0 ∧ AC * AD = 0)

-- Setting specific edge lengths
variables (AB_length : AB = 1) (AC_length : AC = 2) (AD_length : AD = 3)

-- The proof problem: Prove that the surface area of the sphere is 14π
theorem sphere_surface_area : 4 * Real.pi * ((1 + 4 + 9) / 4) = 14 * Real.pi :=
by
  sorry

end sphere_surface_area_l226_226515


namespace sequence_solution_l226_226260

theorem sequence_solution :
  ∃ (a : ℕ → ℕ), a 1 = 2 ∧ (∀ n : ℕ, n ∈ (Set.Icc 1 9) → 
    (n * a (n + 1) = (n + 1) * a n + 2)) ∧ a 10 = 38 :=
by
  sorry

end sequence_solution_l226_226260


namespace bags_of_oranges_l226_226827

-- Define the total number of oranges in terms of bags B
def totalOranges (B : ℕ) : ℕ := 30 * B

-- Define the number of usable oranges left after considering rotten oranges
def usableOranges (B : ℕ) : ℕ := totalOranges B - 50

-- Define the oranges to be sold after keeping some for juice
def orangesToBeSold (B : ℕ) : ℕ := usableOranges B - 30

-- The theorem to state that given 220 oranges will be sold,
-- we need to find B, the number of bags of oranges
theorem bags_of_oranges (B : ℕ) : orangesToBeSold B = 220 → B = 10 :=
by
  sorry

end bags_of_oranges_l226_226827


namespace positive_difference_of_two_numbers_l226_226545

theorem positive_difference_of_two_numbers (x y : ℝ)
  (h1 : x + y = 10) (h2 : x ^ 2 - y ^ 2 = 40) : abs (x - y) = 4 := 
by
  sorry

end positive_difference_of_two_numbers_l226_226545


namespace seven_searchlights_shadow_length_l226_226873

noncomputable def searchlight_positioning (n : ℕ) (angle : ℝ) (shadow_length : ℝ) : Prop :=
  ∃ (positions : Fin n → ℝ × ℝ), ∀ i : Fin n, ∃ shadow : ℝ, shadow = shadow_length ∧
  (∀ j : Fin n, i ≠ j → ∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  θ - angle / 2 < θ ∧ θ + angle / 2 > θ → shadow = shadow_length)

theorem seven_searchlights_shadow_length :
  searchlight_positioning 7 (Real.pi / 2) 7000 :=
sorry

end seven_searchlights_shadow_length_l226_226873


namespace monthly_compounding_greater_than_yearly_l226_226165

open Nat Real

theorem monthly_compounding_greater_than_yearly : 
  1 + 3 / 100 < (1 + 3 / (12 * 100)) ^ 12 :=
by
  -- This is the proof we need to write.
  sorry

end monthly_compounding_greater_than_yearly_l226_226165


namespace complement_A_correct_l226_226332

-- Define the universal set U
def U : Set ℝ := { x | x ≥ 1 ∨ x ≤ -1 }

-- Define the set A
def A : Set ℝ := { x | 1 < x ∧ x ≤ 2 }

-- Define the complement of A in U
def complement_A_in_U : Set ℝ := { x | x ≤ -1 ∨ x = 1 ∨ x > 2 }

-- Prove that the complement of A in U is as defined
theorem complement_A_correct : (U \ A) = complement_A_in_U := by
  sorry

end complement_A_correct_l226_226332


namespace area_of_square_with_given_diagonal_l226_226046

theorem area_of_square_with_given_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) : ∃ (A : ℝ), A = 64 :=
by
  use (8 * 8)
  sorry

end area_of_square_with_given_diagonal_l226_226046


namespace set_intersection_l226_226678

theorem set_intersection (A B : Set ℝ)
  (hA : A = { x : ℝ | 1 < x ∧ x < 4 })
  (hB : B = { x : ℝ | x^2 - 2 * x - 3 ≤ 0 }) :
  A ∩ (Set.univ \ B) = { x : ℝ | 3 < x ∧ x < 4 } :=
by
  sorry

end set_intersection_l226_226678


namespace smallest_nat_div_7_and_11_l226_226514

theorem smallest_nat_div_7_and_11 (n : ℕ) (h1 : n > 1) (h2 : n % 7 = 1) (h3 : n % 11 = 1) : n = 78 :=
by
  sorry

end smallest_nat_div_7_and_11_l226_226514


namespace h_at_neg_one_l226_226740

-- Definitions based on the conditions
def f (x : ℝ) : ℝ := 3 * x + 6
def g (x : ℝ) : ℝ := x ^ 3
def h (x : ℝ) : ℝ := f (g x)

-- The main statement to prove
theorem h_at_neg_one : h (-1) = 3 := by
  sorry

end h_at_neg_one_l226_226740


namespace Martha_cards_l226_226993

theorem Martha_cards :
  let initial_cards := 76.0
  let given_away_cards := 3.0
  initial_cards - given_away_cards = 73.0 :=
by 
  let initial_cards := 76.0
  let given_away_cards := 3.0
  have h : initial_cards - given_away_cards = 73.0 := by sorry
  exact h

end Martha_cards_l226_226993


namespace solve_triangle_l226_226258

noncomputable def angle_A := 45
noncomputable def angle_B := 60
noncomputable def side_a := Real.sqrt 2

theorem solve_triangle {A B : ℕ} {a b : Real}
    (hA : A = angle_A)
    (hB : B = angle_B)
    (ha : a = side_a) :
    b = Real.sqrt 3 := 
by sorry

end solve_triangle_l226_226258


namespace supplementary_angles_difference_l226_226576
-- Import necessary libraries

-- Define the conditions
def are_supplementary (θ₁ θ₂ : ℝ) : Prop := θ₁ + θ₂ = 180

def ratio_7_2 (θ₁ θ₂ : ℝ) : Prop := θ₁ / θ₂ = 7 / 2

-- State the theorem
theorem supplementary_angles_difference (θ₁ θ₂ : ℝ) 
  (h_supp : are_supplementary θ₁ θ₂) 
  (h_ratio : ratio_7_2 θ₁ θ₂) :
  |θ₁ - θ₂| = 100 :=
by
  sorry

end supplementary_angles_difference_l226_226576


namespace number_of_strawberries_stolen_l226_226539

-- Define the conditions
def daily_harvest := 5
def days_in_april := 30
def strawberries_given_away := 20
def strawberries_left_by_end := 100

-- Calculate total harvested strawberries
def total_harvest := daily_harvest * days_in_april
-- Calculate strawberries after giving away
def remaining_after_giveaway := total_harvest - strawberries_given_away

-- Prove the number of strawberries stolen
theorem number_of_strawberries_stolen : remaining_after_giveaway - strawberries_left_by_end = 30 := by
  sorry

end number_of_strawberries_stolen_l226_226539


namespace problem1_l226_226322

variable (x : ℝ)

theorem problem1 : 5 * x^2 * x^4 + x^8 / (-x)^2 = 6 * x^6 :=
  sorry

end problem1_l226_226322


namespace prime_factors_2310_l226_226305

-- Define the number in question
def number : ℕ := 2310

-- The main theorem statement
theorem prime_factors_2310 : ∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p) ∧ 
Finset.prod s id = number ∧ Finset.card s = 5 := by
  sorry

end prime_factors_2310_l226_226305


namespace intersection_eq_neg1_l226_226273

open Set

noncomputable def setA : Set Int := {x : Int | x^2 - 1 ≤ 0}
def setB : Set Int := {x : Int | x^2 - x - 2 = 0}

theorem intersection_eq_neg1 : setA ∩ setB = {-1} := by
  sorry

end intersection_eq_neg1_l226_226273


namespace old_clock_slow_by_12_minutes_l226_226789

theorem old_clock_slow_by_12_minutes (overlap_interval: ℕ) (standard_day_minutes: ℕ)
  (h1: overlap_interval = 66) (h2: standard_day_minutes = 24 * 60):
  standard_day_minutes - 24 * 60 / 66 * 66 = 12 :=
by
  sorry

end old_clock_slow_by_12_minutes_l226_226789


namespace km_to_m_is_750_l226_226365

-- Define 1 kilometer equals 5 hectometers
def km_to_hm := 5

-- Define 1 hectometer equals 10 dekameters
def hm_to_dam := 10

-- Define 1 dekameter equals 15 meters
def dam_to_m := 15

-- Theorem stating that the number of meters in one kilometer is 750
theorem km_to_m_is_750 : 1 * km_to_hm * hm_to_dam * dam_to_m = 750 :=
by 
  -- Proof goes here
  sorry

end km_to_m_is_750_l226_226365


namespace other_endpoint_product_l226_226448

theorem other_endpoint_product :
  ∀ (x y : ℤ), 
    (3 = (x + 7) / 2) → 
    (-5 = (y - 1) / 2) → 
    x * y = 9 :=
by
  intro x y h1 h2
  sorry

end other_endpoint_product_l226_226448


namespace number_of_triangles_l226_226732

theorem number_of_triangles (x y : ℕ) (P Q : ℕ × ℕ) (O : ℕ × ℕ := (0,0)) (area : ℕ) :
  (P ≠ Q) ∧ (P.1 * 31 + P.2 = 2023) ∧ (Q.1 * 31 + Q.2 = 2023) ∧ 
  (P.1 ≠ Q.1 → P.1 - Q.1 = n ∧ 2023 * n % 6 = 0) → area = 165 :=
sorry

end number_of_triangles_l226_226732


namespace find_number_l226_226969

variable (a n : ℝ)

theorem find_number (h1: 2 * a = 3 * n) (h2: a * n ≠ 0) (h3: (a / 3) / (n / 2) = 1) : 
  n = 2 * a / 3 :=
sorry

end find_number_l226_226969


namespace quadratic_has_two_roots_l226_226393

variables {R : Type*} [LinearOrderedField R]

theorem quadratic_has_two_roots (a1 a2 a3 b1 b2 b3 : R) 
  (h1 : a1 * a2 * a3 = b1 * b2 * b3) (h2 : a1 * a2 * a3 > 1) : 
  (4 * a1^2 - 4 * b1 > 0) ∨ (4 * a2^2 - 4 * b2 > 0) ∨ (4 * a3^2 - 4 * b3 > 0) :=
sorry

end quadratic_has_two_roots_l226_226393


namespace problem_a_b_c_ge_neg2_l226_226441

theorem problem_a_b_c_ge_neg2 {a b c : ℝ} (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  (a + 1 / b > -2) ∨ (b + 1 / c > -2) ∨ (c + 1 / a > -2) → False :=
by
  sorry

end problem_a_b_c_ge_neg2_l226_226441


namespace find_x_l226_226487

-- Definitions for the problem
def a (x : ℝ) : ℝ × ℝ := (1, x)
def b : ℝ × ℝ := (-2, 1)

-- Theorem statement
theorem find_x (x : ℝ) (h : ∃ k : ℝ, a x = k • b) : x = -1/2 := by
  sorry

end find_x_l226_226487


namespace total_arrangements_correct_adjacent_males_correct_descending_heights_correct_l226_226886

-- Total number of different arrangements of 3 male students and 2 female students.
def total_arrangements (males females : ℕ) : ℕ :=
  (males + females).factorial

-- Number of arrangements where exactly two male students are adjacent.
def adjacent_males (males females : ℕ) : ℕ :=
  if males = 3 ∧ females = 2 then 72 else 0

-- Number of arrangements where male students of different heights are arranged from tallest to shortest.
def descending_heights (heights : Nat → ℕ) (males females : ℕ) : ℕ :=
  if males = 3 ∧ females = 2 then 20 else 0

-- Theorem statements corresponding to the questions.
theorem total_arrangements_correct : total_arrangements 3 2 = 120 := sorry

theorem adjacent_males_correct : adjacent_males 3 2 = 72 := sorry

theorem descending_heights_correct (heights : Nat → ℕ) : descending_heights heights 3 2 = 20 := sorry

end total_arrangements_correct_adjacent_males_correct_descending_heights_correct_l226_226886


namespace max_value_x_sq_y_l226_226270

theorem max_value_x_sq_y (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 3 * y = 9) :
  x^2 * y ≤ 36 :=
sorry

end max_value_x_sq_y_l226_226270


namespace planks_needed_for_surface_l226_226492

theorem planks_needed_for_surface
  (total_tables : ℕ := 5)
  (total_planks : ℕ := 45)
  (planks_per_leg : ℕ := 4) :
  ∃ S : ℕ, total_tables * (planks_per_leg + S) = total_planks ∧ S = 5 :=
by
  use 5
  sorry

end planks_needed_for_surface_l226_226492


namespace max_min_values_on_circle_l226_226038

def on_circle (x y : ℝ) : Prop :=
  x ^ 2 + y ^ 2 - 4 * x - 4 * y + 7 = 0

theorem max_min_values_on_circle (x y : ℝ) (h : on_circle x y) :
  16 ≤ (x + 1) ^ 2 + (y + 2) ^ 2 ∧ (x + 1) ^ 2 + (y + 2) ^ 2 ≤ 36 :=
  sorry

end max_min_values_on_circle_l226_226038


namespace maximum_value_of_k_minus_b_l226_226039

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := Real.log x - a * x + b

theorem maximum_value_of_k_minus_b (b : ℝ) (k : ℝ) (x : ℝ) 
  (h₀ : 0 ≤ b ∧ b ≤ 2) 
  (h₁ : 1 ≤ x ∧ x ≤ Real.exp 1)
  (h₂ : ∀ x ∈ Set.Icc 1 (Real.exp 1), f x 1 b ≥ (k * x - x * Real.log x - 1)) :
  k - b ≤ 0 :=
sorry

end maximum_value_of_k_minus_b_l226_226039


namespace Tim_marble_count_l226_226855

theorem Tim_marble_count (Fred_marbles : ℕ) (Tim_marbles : ℕ) (h1 : Fred_marbles = 110) (h2 : Fred_marbles = 22 * Tim_marbles) : 
  Tim_marbles = 5 := 
sorry

end Tim_marble_count_l226_226855


namespace table_height_l226_226009

variable (l h w : ℝ)

-- Given conditions:
def conditionA := l + h - w = 36
def conditionB := w + h - l = 30

-- Proof that height of the table h is 33 inches
theorem table_height {l h w : ℝ} 
  (h1 : l + h - w = 36) 
  (h2 : w + h - l = 30) : 
  h = 33 := 
by
  sorry

end table_height_l226_226009


namespace total_clothing_donated_l226_226871

-- Definition of the initial donation by Adam
def adam_initial_donation : Nat := 4 + 4 + 4*2 + 20 -- 4 pairs of pants, 4 jumpers, 4 pajama sets (8 items), 20 t-shirts

-- Adam's friends' total donation
def friends_donation : Nat := 3 * adam_initial_donation

-- Adam's donation after keeping half
def adam_final_donation : Nat := adam_initial_donation / 2

-- Total donation being the sum of Adam's and friends' donations
def total_donation : Nat := adam_final_donation + friends_donation

-- The statement to prove
theorem total_clothing_donated : total_donation = 126 := by
  -- This is skipped as per instructions
  sorry

end total_clothing_donated_l226_226871


namespace decrease_percent_in_revenue_l226_226902

theorem decrease_percent_in_revenue
  (T C : ℝ)
  (h_pos_T : 0 < T)
  (h_pos_C : 0 < C)
  (h_new_tax : T_new = 0.80 * T)
  (h_new_consumption : C_new = 1.20 * C) :
  let original_revenue := T * C
  let new_revenue := 0.80 * T * 1.20 * C
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 4 := by
sorry

end decrease_percent_in_revenue_l226_226902


namespace number_of_arrangements_of_six_students_l226_226887

/-- A and B cannot stand together -/
noncomputable def arrangements_A_B_not_together (n: ℕ) (A B: ℕ) : ℕ :=
  if n = 6 then 480 else 0

theorem number_of_arrangements_of_six_students :
  arrangements_A_B_not_together 6 1 2 = 480 :=
sorry

end number_of_arrangements_of_six_students_l226_226887


namespace determine_k_if_even_function_l226_226751

noncomputable def f (x k : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

theorem determine_k_if_even_function (k : ℝ) (h_even: ∀ x : ℝ, f x k = f (-x) k ) : k = 1 :=
by
  sorry

end determine_k_if_even_function_l226_226751


namespace time_taken_by_A_l226_226462

-- Definitions for the problem conditions
def race_distance : ℕ := 1000  -- in meters
def A_beats_B_by_distance : ℕ := 48  -- in meters
def A_beats_B_by_time : ℕ := 12  -- in seconds

-- The formal statement to prove in Lean
theorem time_taken_by_A :
  ∃ T_a : ℕ, (1000 * (T_a + 12) = 952 * T_a) ∧ T_a = 250 :=
by
  sorry

end time_taken_by_A_l226_226462


namespace f_monotonic_intervals_g_not_below_f_inequality_holds_l226_226587

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 - 3 * x
noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem f_monotonic_intervals :
  ∀ x : ℝ, 0 < x → 
    (0 < x ∧ x < 1 / 2 → f x < f (x + 1)) ∧ 
    (1 / 2 < x ∧ x < 1 → f x > f (x + 1)) ∧ 
    (1 < x → f x < f (x + 1)) :=
sorry

theorem g_not_below_f :
  ∀ x : ℝ, 0 < x → f x < g x :=
sorry

theorem inequality_holds (n : ℕ) : (2 * n + 1)^2 > 4 * Real.log (Nat.factorial n) :=
sorry

end f_monotonic_intervals_g_not_below_f_inequality_holds_l226_226587


namespace minimum_slope_tangent_point_coordinates_l226_226508

theorem minimum_slope_tangent_point_coordinates :
  ∃ a : ℝ, a > 0 ∧ (∀ x : ℝ, (2 * x + a / x ≥ 4) ∧ (2 * x + a / x = 4 ↔ x = 1)) → 
  (1, 1) = (1, 1) := by
sorry

end minimum_slope_tangent_point_coordinates_l226_226508


namespace middle_person_distance_l226_226182

noncomputable def Al_position (t : ℝ) : ℝ := 6 * t
noncomputable def Bob_position (t : ℝ) : ℝ := 10 * t - 12
noncomputable def Cy_position (t : ℝ) : ℝ := 8 * t - 32

theorem middle_person_distance (t : ℝ) (h₁ : t ≥ 0) (h₂ : t ≥ 2) (h₃ : t ≥ 4) :
  (Al_position t = 52) ∨ (Bob_position t = 52) ∨ (Cy_position t = 52) :=
sorry

end middle_person_distance_l226_226182


namespace toadon_population_percentage_l226_226017

theorem toadon_population_percentage {pop_total G L T : ℕ}
    (h_total : pop_total = 80000)
    (h_gordonia : G = pop_total / 2)
    (h_lakebright : L = 16000)
    (h_total_population : pop_total = G + T + L) :
    (T * 100 / G) = 60 :=
by sorry

end toadon_population_percentage_l226_226017


namespace intersection_point_of_lines_l226_226226

noncomputable def line1 (x : ℝ) : ℝ := 3 * x - 4

noncomputable def line2 (x : ℝ) : ℝ := -1 / 3 * x + 10 / 3

def point : ℝ × ℝ := (4, 2)

theorem intersection_point_of_lines :
  ∃ (x y : ℝ), line1 x = y ∧ line2 x = y ∧ (x, y) = (2.2, 2.6) :=
by
  sorry

end intersection_point_of_lines_l226_226226


namespace parabola_directrix_l226_226403

theorem parabola_directrix (x y : ℝ) :
    x^2 = - (1 / 4) * y → y = - (1 / 16) :=
by
  sorry

end parabola_directrix_l226_226403


namespace distance_from_yz_plane_l226_226350

theorem distance_from_yz_plane (x z : ℝ) : 
  (abs (-6) = (abs x) / 2) → abs x = 12 :=
by
  sorry

end distance_from_yz_plane_l226_226350


namespace tangent_line_equation_l226_226459

theorem tangent_line_equation (x y : ℝ) :
  (y = Real.exp x + 2) →
  (x = 0) →
  (y = 3) →
  (Real.exp x = 1) →
  (x - y + 3 = 0) :=
by
  intros h_eq h_x h_y h_slope
  -- The following proof will use the conditions to show the tangent line equation.
  sorry

end tangent_line_equation_l226_226459


namespace remainder_div_7_l226_226722

theorem remainder_div_7 (k : ℕ) (h1 : k % 5 = 2) (h2 : k % 6 = 5) (h3 : k < 39) : k % 7 = 3 :=
sorry

end remainder_div_7_l226_226722


namespace quadratic_has_real_root_iff_b_in_interval_l226_226079

theorem quadratic_has_real_root_iff_b_in_interval (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ Set.Iic (-10) ∪ Set.Ici 10) := 
sorry

end quadratic_has_real_root_iff_b_in_interval_l226_226079


namespace exams_in_fourth_year_l226_226392

variable (a b c d e : ℕ)

theorem exams_in_fourth_year:
  a + b + c + d + e = 31 ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e = 3 * a → d = 8 := by
  sorry

end exams_in_fourth_year_l226_226392


namespace male_salmon_count_l226_226383

theorem male_salmon_count (total_salmon : ℕ) (female_salmon : ℕ) (male_salmon : ℕ) 
  (h1 : total_salmon = 971639) 
  (h2 : female_salmon = 259378) 
  (h3 : male_salmon = total_salmon - female_salmon) : 
  male_salmon = 712261 :=
by
  sorry

end male_salmon_count_l226_226383


namespace sheep_problem_system_l226_226777

theorem sheep_problem_system :
  (∃ (x y : ℝ), 5 * x - y = -90 ∧ 50 * x - y = 0) ↔ 
  (5 * x - y = -90 ∧ 50 * x - y = 0) := 
by
  sorry

end sheep_problem_system_l226_226777


namespace binomial_theorem_fifth_term_l226_226042
-- Import the necessary library

-- Define the theorem as per the given conditions and required proof
theorem binomial_theorem_fifth_term
  (a x : ℝ) 
  (hx : x ≠ 0) 
  (ha : a ≠ 0) : 
  (Nat.choose 8 4 * (a / x)^4 * (x / a^3)^4 = 70 / a^8) :=
by
  -- Applying the binomial theorem and simplifying the expression
  rw [Nat.choose]
  sorry

end binomial_theorem_fifth_term_l226_226042


namespace log_10_850_consecutive_integers_l226_226734

theorem log_10_850_consecutive_integers : 
  (2:ℝ) < Real.log 850 / Real.log 10 ∧ Real.log 850 / Real.log 10 < (3:ℝ) →
  ∃ (a b : ℕ), (a = 2) ∧ (b = 3) ∧ (2 < Real.log 850 / Real.log 10) ∧ (Real.log 850 / Real.log 10 < 3) ∧ (a + b = 5) :=
by
  sorry

end log_10_850_consecutive_integers_l226_226734


namespace darren_total_tshirts_l226_226956

def num_white_packs := 5
def num_white_tshirts_per_pack := 6
def num_blue_packs := 3
def num_blue_tshirts_per_pack := 9

def total_tshirts (wpacks : ℕ) (wtshirts_per_pack : ℕ) (bpacks : ℕ) (btshirts_per_pack : ℕ) : ℕ :=
  (wpacks * wtshirts_per_pack) + (bpacks * btshirts_per_pack)

theorem darren_total_tshirts : total_tshirts num_white_packs num_white_tshirts_per_pack num_blue_packs num_blue_tshirts_per_pack = 57 :=
by
  -- proof needed
  sorry

end darren_total_tshirts_l226_226956


namespace ellipse_major_axis_length_l226_226375

-- Given conditions
variable (radius : ℝ) (h_radius : radius = 2)
variable (minor_axis : ℝ) (h_minor_axis : minor_axis = 2 * radius)
variable (major_axis : ℝ) (h_major_axis : major_axis = 1.4 * minor_axis)

-- Proof problem statement
theorem ellipse_major_axis_length : major_axis = 5.6 :=
by
  sorry

end ellipse_major_axis_length_l226_226375


namespace bmw_length_l226_226346

theorem bmw_length : 
  let horiz1 : ℝ := 2 -- Length of each horizontal segment in 'B'
  let horiz2 : ℝ := 2 -- Length of each horizontal segment in 'B'
  let vert1  : ℝ := 2 -- Length of each vertical segment in 'B'
  let vert2  : ℝ := 2 -- Length of each vertical segment in 'B'
  let vert3  : ℝ := 2 -- Length of each vertical segment in 'M'
  let vert4  : ℝ := 2 -- Length of each vertical segment in 'M'
  let vert5  : ℝ := 2 -- Length of each vertical segment in 'W'
  let diag1  : ℝ := Real.sqrt 2 -- Length of each diagonal segment in 'W'
  let diag2  : ℝ := Real.sqrt 2 -- Length of each diagonal segment in 'W'
  (horiz1 + horiz2 + vert1 + vert2 + vert3 + vert4 + vert5 + diag1 + diag2) = 14 + 2 * Real.sqrt 2 :=
by
  sorry

end bmw_length_l226_226346


namespace cubic_inequality_l226_226952

theorem cubic_inequality (a : ℝ) (h : a ≠ -1) : 
  (1 + a^3) / (1 + a)^3 ≥ 1 / 4 :=
by sorry

end cubic_inequality_l226_226952


namespace area_of_circle_B_l226_226391

theorem area_of_circle_B (rA rB : ℝ) (h : π * rA^2 = 16 * π) (h1 : rB = 2 * rA) : π * rB^2 = 64 * π :=
by
  sorry

end area_of_circle_B_l226_226391


namespace find_ab_l226_226458

noncomputable def perpendicular_condition (a b : ℝ) :=
  a * (a - 1) - b = 0

noncomputable def point_on_l1_condition (a b : ℝ) :=
  -3 * a + b + 4 = 0

noncomputable def parallel_condition (a b : ℝ) :=
  a + b * (a - 1) = 0

noncomputable def distance_condition (a : ℝ) :=
  4 = abs ((-a) / (a - 1))

theorem find_ab (a b : ℝ) :
  (perpendicular_condition a b ∧ point_on_l1_condition a b ∧
   parallel_condition a b ∧ distance_condition a) →
  ((a = 2 ∧ b = 2) ∨ (a = 2 ∧ b = -2) ∨ (a = -2 ∧ b = 2)) :=
by
  sorry

end find_ab_l226_226458


namespace gcd_9009_14014_l226_226217

-- Given conditions
def decompose_9009 : 9009 = 9 * 1001 := by sorry
def decompose_14014 : 14014 = 14 * 1001 := by sorry
def coprime_9_14 : Nat.gcd 9 14 = 1 := by sorry

-- Proof problem statement
theorem gcd_9009_14014 : Nat.gcd 9009 14014 = 1001 := by
  have h1 : 9009 = 9 * 1001 := decompose_9009
  have h2 : 14014 = 14 * 1001 := decompose_14014
  have h3 : Nat.gcd 9 14 = 1 := coprime_9_14
  sorry

end gcd_9009_14014_l226_226217


namespace perp_condition_l226_226811

def a (x : ℝ) : ℝ × ℝ := (x-1, 2)
def b : ℝ × ℝ := (2, 1)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem perp_condition (x : ℝ) : dot_product (a x) b = 0 ↔ x = 0 :=
by 
  sorry

end perp_condition_l226_226811


namespace worker_net_salary_change_l226_226469

theorem worker_net_salary_change (S : ℝ) :
  let final_salary := S * 1.15 * 0.90 * 1.20 * 0.95
  let net_change := final_salary - S
  net_change = 0.0355 * S := by
  -- Proof goes here
  sorry

end worker_net_salary_change_l226_226469


namespace seokgi_walk_distance_correct_l226_226709

-- Definitions of distances as per conditions
def entrance_to_temple_km : ℕ := 4
def entrance_to_temple_m : ℕ := 436
def temple_to_summit_m : ℕ := 1999

-- Total distance Seokgi walked in kilometers
def total_walked_km : ℕ := 12870

-- Proof statement
theorem seokgi_walk_distance_correct :
  ((entrance_to_temple_km * 1000 + entrance_to_temple_m) + temple_to_summit_m) * 2 / 1000 = total_walked_km / 1000 :=
by
  -- We will fill this in with the proof steps
  sorry

end seokgi_walk_distance_correct_l226_226709


namespace andrea_average_distance_per_day_l226_226706

theorem andrea_average_distance_per_day
  (total_distance : ℕ := 168)
  (fraction_completed : ℚ := 3/7)
  (total_days : ℕ := 6)
  (days_completed : ℕ := 3) :
  (total_distance * (1 - fraction_completed) / (total_days - days_completed)) = 32 :=
by sorry

end andrea_average_distance_per_day_l226_226706


namespace pen_ratio_l226_226173

theorem pen_ratio (R J D : ℕ) (pen_cost : ℚ) (total_spent : ℚ) (total_pens : ℕ) 
  (hR : R = 4)
  (hJ : J = 3 * R)
  (h_total_spent : total_spent = 33)
  (h_pen_cost : pen_cost = 1.5)
  (h_total_pens : total_pens = total_spent / pen_cost)
  (h_pens_expr : D + J + R = total_pens) :
  D / J = 1 / 2 :=
by
  sorry

end pen_ratio_l226_226173


namespace rectangle_length_l226_226364

-- Define a structure for the rectangle.
structure Rectangle where
  breadth : ℝ
  length : ℝ
  area : ℝ

-- Define the given conditions.
def givenConditions (r : Rectangle) : Prop :=
  r.length = 3 * r.breadth ∧ r.area = 6075

-- State the theorem.
theorem rectangle_length (r : Rectangle) (h : givenConditions r) : r.length = 135 :=
by
  sorry

end rectangle_length_l226_226364


namespace shaded_area_of_three_circles_l226_226192

theorem shaded_area_of_three_circles :
  (∀ (r1 r2 : ℝ), (π * r1^2 = 100 * π) → (r2 = r1 / 2) → (shaded_area = (π * r1^2) / 2 + 2 * ((π * r2^2) / 2)) → (shaded_area = 75 * π)) :=
by
  sorry

end shaded_area_of_three_circles_l226_226192


namespace largest_part_of_proportional_division_l226_226373

theorem largest_part_of_proportional_division (sum : ℚ) (a b c largest : ℚ) 
  (prop1 prop2 prop3 : ℚ) 
  (h1 : sum = 156)
  (h2 : prop1 = 2)
  (h3 : prop2 = 1 / 2)
  (h4 : prop3 = 1 / 4)
  (h5 : sum = a + b + c)
  (h6 : a / prop1 = b / prop2 ∧ b / prop2 = c / prop3)
  (h7 : largest = max a (max b c)) :
  largest = 112 + 8 / 11 :=
by
  sorry

end largest_part_of_proportional_division_l226_226373


namespace problem1_problem2_l226_226291

-- Problem (1)
theorem problem1 (a : ℚ) (h : a = -1/2) : 
  a * (a - 4) - (a + 6) * (a - 2) = 16 := by
  sorry

-- Problem (2)
theorem problem2 (x y : ℚ) (hx : x = 8) (hy : y = -8) :
  (x + 2 * y) * (x - 2 * y) - (2 * x - y) * (-2 * x - y) = 0 := by
  sorry

end problem1_problem2_l226_226291


namespace M_inter_N_l226_226240

def M : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
noncomputable def N : Set ℝ := { x | ∃ y, y = Real.sqrt x + Real.log (1 - x) }

theorem M_inter_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by
  sorry

end M_inter_N_l226_226240


namespace root_eq_neg_l226_226681

theorem root_eq_neg {a : ℝ} (h : 3 * a - 9 < 0) : (a - 4) * (a - 5) > 0 :=
by
  sorry

end root_eq_neg_l226_226681


namespace union_of_sets_l226_226655

open Set

noncomputable def A (a : ℝ) : Set ℝ := {1, 2^a}
noncomputable def B (a b : ℝ) : Set ℝ := {a, b}

theorem union_of_sets (a b : ℝ) (h₁ : A a ∩ B a b = {1 / 2}) :
  A a ∪ B a b = {-1, 1 / 2, 1} :=
by
  sorry

end union_of_sets_l226_226655


namespace ab_cd_eq_neg_37_over_9_l226_226559

theorem ab_cd_eq_neg_37_over_9 (a b c d : ℝ)
  (h1 : a + b + c = 6)
  (h2 : a + b + d = 2)
  (h3 : a + c + d = 3)
  (h4 : b + c + d = -3) :
  a * b + c * d = -37 / 9 := by
  sorry

end ab_cd_eq_neg_37_over_9_l226_226559


namespace more_crayons_given_to_Lea_than_Mae_l226_226337

-- Define the initial number of crayons
def initial_crayons : ℕ := 4 * 8

-- Condition: Nori gave 5 crayons to Mae
def crayons_given_to_Mae : ℕ := 5

-- Condition: Nori has 15 crayons left after giving some to Lea
def crayons_left_after_giving_to_Lea : ℕ := 15

-- Define the number of crayons after giving to Mae
def crayons_after_giving_to_Mae : ℕ := initial_crayons - crayons_given_to_Mae

-- Define the number of crayons given to Lea
def crayons_given_to_Lea : ℕ := crayons_after_giving_to_Mae - crayons_left_after_giving_to_Lea

-- Prove the number of more crayons given to Lea than Mae
theorem more_crayons_given_to_Lea_than_Mae : (crayons_given_to_Lea - crayons_given_to_Mae) = 7 := by
  sorry

end more_crayons_given_to_Lea_than_Mae_l226_226337


namespace quadratic_function_min_value_l226_226835

theorem quadratic_function_min_value (a b c : ℝ) (h_a : a > 0) (h_b : b ≠ 0) 
(h_f0 : |c| = 1) (h_f1 : |a + b + c| = 1) (h_fn1 : |a - b + c| = 1) :
∃ f : ℝ → ℝ, (∀ x : ℝ, f x = a*x^2 + b*x + c) ∧
  (|f 0| = 1) ∧ (|f 1| = 1) ∧ (|f (-1)| = 1) ∧
  (f 0 = -(5/4) ∨ f 1 = -(5/4) ∨ f (-1) = -(5/4)) :=
by
  sorry

end quadratic_function_min_value_l226_226835


namespace card_giving_ratio_l226_226091

theorem card_giving_ratio (initial_cards cards_to_Bob cards_left : ℕ) 
  (h1 : initial_cards = 18) 
  (h2 : cards_to_Bob = 3)
  (h3 : cards_left = 9) : 
  (initial_cards - cards_left - cards_to_Bob) / gcd (initial_cards - cards_left - cards_to_Bob) cards_to_Bob = 2 / 1 :=
by sorry

end card_giving_ratio_l226_226091


namespace inverse_proportion_passes_first_and_third_quadrants_l226_226867

theorem inverse_proportion_passes_first_and_third_quadrants (m : ℝ) :
  ((∀ x : ℝ, x ≠ 0 → (x > 0 → (m - 3) / x > 0) ∧ (x < 0 → (m - 3) / x < 0)) → m = 5) := 
by 
  sorry

end inverse_proportion_passes_first_and_third_quadrants_l226_226867


namespace problem1_problem2_l226_226148

section ArithmeticSequence

variable {a : ℕ → ℤ} {a1 a5 a8 a6 a4 d : ℤ}

-- Problem 1: Prove that if a_5 = -1 and a_8 = 2, then a_1 = -5 and d = 1
theorem problem1 
  (h1 : a 5 = -1) 
  (h2 : a 8 = 2)
  (h3 : ∀ n, a n = a1 + n * d) : 
  a1 = -5 ∧ d = 1 := 
sorry 

-- Problem 2: Prove that if a_1 + a_6 = 12 and a_4 = 7, then a_9 = 17
theorem problem2 
  (h1 : a1 + a 6 = 12) 
  (h2 : a 4 = 7)
  (h3 : ∀ n, a n = a1 + n * d) 
  (h4 : ∀ m (hm : m ≠ 0), a1 = a 1): 
   a 9 = 17 := 
sorry

end ArithmeticSequence

end problem1_problem2_l226_226148


namespace sum_of_cubes_of_consecutive_integers_div_by_9_l226_226362

theorem sum_of_cubes_of_consecutive_integers_div_by_9 (x : ℤ) : 
  let a := (x - 1) ^ 3
  let b := x ^ 3
  let c := (x + 1) ^ 3
  (a + b + c) % 9 = 0 :=
by
  sorry

end sum_of_cubes_of_consecutive_integers_div_by_9_l226_226362


namespace line_intersects_circle_l226_226123

theorem line_intersects_circle (k : ℝ) : ∀ (x y : ℝ),
  (x + y) ^ 2 = x ^ 2 + y ^ 2 →
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ y = k * (x + 1 / 2)) ∧ 
  ((-1/2)^2 + (0)^2 < 1) →
  ∃ x y : ℝ, x^2 + y^2 = 1 ∧ y = k * (x + 1 / 2) := 
by
  intro x y h₁ h₂
  sorry

end line_intersects_circle_l226_226123


namespace transformed_roots_polynomial_l226_226275

-- Given conditions
variables {a b c : ℝ}
variables (h : ∀ x, (x - a) * (x - b) * (x - c) = x^3 - 4 * x + 6)

-- Prove the equivalent polynomial with the transformed roots
theorem transformed_roots_polynomial :
  (∀ x, (x - (a - 3)) * (x - (b - 3)) * (x - (c - 3)) = x^3 + 9 * x^2 + 23 * x + 21) :=
sorry

end transformed_roots_polynomial_l226_226275


namespace divisible_by_six_l226_226496

theorem divisible_by_six (m : ℕ) : 6 ∣ (m^3 + 11 * m) := 
sorry

end divisible_by_six_l226_226496


namespace polynomial_identity_l226_226027

theorem polynomial_identity (x : ℝ) : 
  (2 * x^2 + 5 * x + 8) * (x + 1) - (x + 1) * (x^2 - 2 * x + 50) 
  + (3 * x - 7) * (x + 1) * (x - 2) = 4 * x^3 - 2 * x^2 - 34 * x - 28 := 
by 
  sorry

end polynomial_identity_l226_226027


namespace complex_expression_ab_l226_226791

open Complex

theorem complex_expression_ab :
  ∀ (a b : ℝ), (2 + 3 * I) / I = a + b * I → a * b = 6 :=
by
  intros a b h
  sorry

end complex_expression_ab_l226_226791


namespace find_f_neg1_l226_226996

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

noncomputable def f : ℝ → ℝ
| x => if 0 < x then x^2 + 2 else if x = 0 then 2 else -(x^2 + 2)

axiom odd_f : is_odd_function f

theorem find_f_neg1 : f (-1) = -3 := by
  sorry

end find_f_neg1_l226_226996


namespace sequence_satisfies_n_squared_l226_226769

theorem sequence_satisfies_n_squared (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, n ≥ 2 → a n = a (n - 1) + 2 * n - 1) :
  ∀ n, a n = n^2 :=
by
  -- sorry
  sorry

end sequence_satisfies_n_squared_l226_226769


namespace rectangle_area_proof_l226_226125

def rectangle_width : ℕ := 5

def rectangle_length (width : ℕ) : ℕ := 3 * width

def rectangle_area (length width : ℕ) : ℕ := length * width

theorem rectangle_area_proof : rectangle_area (rectangle_length rectangle_width) rectangle_width = 75 := by
  sorry -- Proof can be added later

end rectangle_area_proof_l226_226125


namespace coefficients_equality_l226_226142

theorem coefficients_equality (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h : a_1 * (x-1)^4 + a_2 * (x-1)^3 + a_3 * (x-1)^2 + a_4 * (x-1) + a_5 = x^4)
  (h1 : a_1 = 1)
  (h2 : a_5 = 1)
  (h3 : 1 - a_2 + a_3 - a_4 + 1 = 0) :
  a_2 - a_3 + a_4 = 2 :=
sorry

end coefficients_equality_l226_226142


namespace students_in_class_l226_226340

theorem students_in_class
  (S : ℕ)
  (h1 : S / 3 * 4 / 3 = 12) :
  S = 36 := 
sorry

end students_in_class_l226_226340


namespace average_cookies_per_package_is_fifteen_l226_226928

def average_cookies_count (cookies : List ℕ) (n : ℕ) : ℕ :=
  (cookies.sum / n : ℕ)

theorem average_cookies_per_package_is_fifteen :
  average_cookies_count [5, 12, 18, 20, 21] 5 = 15 :=
by
  sorry

end average_cookies_per_package_is_fifteen_l226_226928


namespace milk_amount_at_beginning_l226_226493

theorem milk_amount_at_beginning (H: 0.69 = 0.6 * total_milk) : total_milk = 1.15 :=
sorry

end milk_amount_at_beginning_l226_226493


namespace max_a2_plus_b2_l226_226685

theorem max_a2_plus_b2 (a b : ℝ) 
  (h : abs (a - 1) + abs (a - 6) + abs (b + 3) + abs (b - 2) = 10) : 
  (a^2 + b^2) ≤ 45 :=
sorry

end max_a2_plus_b2_l226_226685


namespace school_team_profit_is_333_l226_226015

noncomputable def candy_profit (total_bars : ℕ) (price_800_bars : ℕ) (price_400_bars : ℕ) (sold_600_bars_price : ℕ) (remaining_600_bars_price : ℕ) : ℚ :=
  let cost_800_bars := 800 / 3
  let cost_400_bars := 400 / 4
  let total_cost := cost_800_bars + cost_400_bars
  let revenue_sold_600_bars := 600 / 2
  let revenue_remaining_600_bars := (600 * 2) / 3
  let total_revenue := revenue_sold_600_bars + revenue_remaining_600_bars
  total_revenue - total_cost

theorem school_team_profit_is_333 :
  candy_profit 1200 3 4 2 2 = 333 := by
  sorry

end school_team_profit_is_333_l226_226015


namespace math_problem_l226_226745

theorem math_problem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 1 / y) :
  (x - 1 / x) * (y + 1 / y) = x^2 - y^2 :=
by
  sorry

end math_problem_l226_226745


namespace new_person_weight_l226_226026

-- Define the given conditions as Lean definitions
def weight_increase_per_person : ℝ := 2.5
def num_people : ℕ := 8
def replaced_person_weight : ℝ := 65

-- State the theorem using the given conditions and the correct answer
theorem new_person_weight :
  (weight_increase_per_person * num_people) + replaced_person_weight = 85 :=
sorry

end new_person_weight_l226_226026


namespace max_bishops_correct_bishop_position_count_correct_l226_226054

-- Define the parameters and predicates
def chessboard_size : ℕ := 2015

def max_bishops (board_size : ℕ) : ℕ := 2 * board_size - 1 - 1

def bishop_position_count (board_size : ℕ) : ℕ := 2 ^ (board_size - 1) * 2 * 2

-- State the equalities to be proved
theorem max_bishops_correct : max_bishops chessboard_size = 4028 := by
  -- proof will be here
  sorry

theorem bishop_position_count_correct : bishop_position_count chessboard_size = 2 ^ 2016 := by
  -- proof will be here
  sorry

end max_bishops_correct_bishop_position_count_correct_l226_226054


namespace find_S12_l226_226580

theorem find_S12 (S : ℕ → ℕ) (h1 : S 3 = 6) (h2 : S 9 = 15) : S 12 = 18 :=
by
  sorry

end find_S12_l226_226580


namespace triangle_inequality_check_l226_226583

theorem triangle_inequality_check :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ↔
    ((a = 6 ∧ b = 9 ∧ c = 14) ∨ (a = 9 ∧ b = 6 ∧ c = 14) ∨ (a = 6 ∧ b = 14 ∧ c = 9) ∨
     (a = 14 ∧ b = 6 ∧ c = 9) ∨ (a = 9 ∧ b = 14 ∧ c = 6) ∨ (a = 14 ∧ b = 9 ∧ c = 6)) := sorry

end triangle_inequality_check_l226_226583


namespace juniors_in_club_l226_226625

theorem juniors_in_club
  (j s x y : ℝ)
  (h1 : x = 0.4 * j)
  (h2 : y = 0.25 * s)
  (h3 : j + s = 36)
  (h4 : x = 2 * y) :
  j = 20 :=
by
  sorry

end juniors_in_club_l226_226625


namespace find_quadruples_l226_226062

open Nat

theorem find_quadruples (a b p n : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n)
    (h : a^3 + b^3 = p^n) :
    (∃ k, a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3 * k + 1) ∨
    (∃ k, a = 3^k ∧ b = 2 * 3^k ∧ p = 3 ∧ n = 3 * k + 2) ∨
    (∃ k, a = 2 * 3^k ∧ b = 3^k ∧ p = 3 ∧ n = 3 * k + 2) :=
sorry

end find_quadruples_l226_226062


namespace percent_forgot_group_B_l226_226870

def num_students_group_A : ℕ := 20
def num_students_group_B : ℕ := 80
def percent_forgot_group_A : ℚ := 0.20
def total_percent_forgot : ℚ := 0.16

/--
There are two groups of students in the sixth grade. 
There are 20 students in group A, and 80 students in group B. 
On a particular day, 20% of the students in group A forget their homework, and a certain 
percentage of the students in group B forget their homework. 
Then, 16% of the sixth graders forgot their homework. 
Prove that 15% of the students in group B forgot their homework.
-/
theorem percent_forgot_group_B : 
  let num_forgot_group_A := percent_forgot_group_A * num_students_group_A
  let total_students := num_students_group_A + num_students_group_B
  let total_forgot := total_percent_forgot * total_students
  let num_forgot_group_B := total_forgot - num_forgot_group_A
  let percent_forgot_group_B := (num_forgot_group_B / num_students_group_B) * 100
  percent_forgot_group_B = 15 :=
by {
  sorry
}

end percent_forgot_group_B_l226_226870


namespace min_tosses_one_head_l226_226679

theorem min_tosses_one_head (n : ℕ) (P : ℝ) (h₁ : P = 1 - (1 / 2) ^ n) (h₂ : P ≥ 15 / 16) : n ≥ 4 :=
by
  sorry -- Proof to be filled in.

end min_tosses_one_head_l226_226679


namespace julia_paid_for_puppy_l226_226367

theorem julia_paid_for_puppy :
  let dog_food := 20
  let treat := 2.5
  let treats := 2 * treat
  let toys := 15
  let crate := 20
  let bed := 20
  let collar_leash := 15
  let discount_rate := 0.20
  let total_before_discount := dog_food + treats + toys + crate + bed + collar_leash
  let discount := discount_rate * total_before_discount
  let total_after_discount := total_before_discount - discount
  let total_spent := 96
  total_spent - total_after_discount = 20 := 
by 
  sorry

end julia_paid_for_puppy_l226_226367


namespace number_of_rows_with_7_eq_5_l226_226836

noncomputable def number_of_rows_with_7_people (x y : ℕ) : Prop :=
  7 * x + 6 * (y - x) = 59

theorem number_of_rows_with_7_eq_5 :
  ∃ x y : ℕ, number_of_rows_with_7_people x y ∧ x = 5 :=
by {
  sorry
}

end number_of_rows_with_7_eq_5_l226_226836


namespace find_expression_value_l226_226862

theorem find_expression_value (x : ℝ) (h : x^2 - 5*x = 14) : 
  (x-1)*(2*x-1) - (x+1)^2 + 1 = 15 := 
by 
  sorry

end find_expression_value_l226_226862


namespace most_entries_with_80_yuan_is_c_pass_pass_a_is_cost_effective_after_30_entries_l226_226924

noncomputable def most_entries_with_80_yuan : Nat :=
let cost_a := 120
let cost_b := 60
let cost_c := 40
let entry_b := 2
let entry_c := 3
let budget := 80
let entries_b := (budget - cost_b) / entry_b
let entries_c := (budget - cost_c) / entry_c
let entries_no_pass := budget / 10
if cost_a <= budget then 
  0
else
  max entries_b (max entries_c entries_no_pass)

theorem most_entries_with_80_yuan_is_c_pass : most_entries_with_80_yuan = 13 :=
by
  sorry

noncomputable def is_pass_a_cost_effective (x : Nat) : Prop :=
let cost_a := 120
let cost_b_entries := 60 + 2 * x
let cost_c_entries := 40 + 3 * x
let cost_no_pass := 10 * x
x > 30 → cost_a < cost_b_entries ∧ cost_a < cost_c_entries ∧ cost_a < cost_no_pass

theorem pass_a_is_cost_effective_after_30_entries : ∀ x : Nat, is_pass_a_cost_effective x :=
by
  sorry

end most_entries_with_80_yuan_is_c_pass_pass_a_is_cost_effective_after_30_entries_l226_226924


namespace problem1_l226_226632

variable {a b : ℝ}

theorem problem1 (ha : a > 0) (hb : b > 0) : 
  (1 / (a + b) ≤ 1 / 4 * (1 / a + 1 / b)) :=
sorry

end problem1_l226_226632


namespace find_q_l226_226045

noncomputable def solution_condition (p q : ℝ) : Prop :=
  (p > 1) ∧ (q > 1) ∧ (1 / p + 1 / q = 1) ∧ (p * q = 9)

theorem find_q (p q : ℝ) (h : solution_condition p q) : 
  q = (9 + 3 * Real.sqrt 5) / 2 :=
sorry

end find_q_l226_226045


namespace increasing_function_implies_a_nonpositive_l226_226010

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem increasing_function_implies_a_nonpositive (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) → a ≤ 0 :=
by
  sorry

end increasing_function_implies_a_nonpositive_l226_226010


namespace average_price_of_towels_l226_226021

theorem average_price_of_towels :
  let total_cost := 2350
  let total_towels := 10
  total_cost / total_towels = 235 :=
by
  sorry

end average_price_of_towels_l226_226021


namespace triangle_inequality_l226_226748

variable {a b c S n : ℝ}

theorem triangle_inequality (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
(habc : a + b > c) (habc' : a + c > b) (habc'' : b + c > a)
(hS : 2 * S = a + b + c) (hn : n ≥ 1) :
  (a^n / (b + c)) + (b^n / (c + a)) + (c^n / (a + b)) ≥ ((2 / 3)^(n - 2)) * S^(n - 1) :=
by
  sorry

end triangle_inequality_l226_226748


namespace Tameka_sold_40_boxes_on_Friday_l226_226999

noncomputable def TamekaSalesOnFriday (F : ℕ) : Prop :=
  let SaturdaySales := 2 * F - 10
  let SundaySales := (2 * F - 10) / 2
  F + SaturdaySales + SundaySales = 145

theorem Tameka_sold_40_boxes_on_Friday : ∃ F : ℕ, TamekaSalesOnFriday F ∧ F = 40 := 
by 
  sorry

end Tameka_sold_40_boxes_on_Friday_l226_226999


namespace greatest_a_l226_226790

theorem greatest_a (a : ℝ) : a^2 - 14*a + 45 ≤ 0 → a ≤ 9 :=
by
  -- placeholder for the actual proof
  sorry

end greatest_a_l226_226790


namespace line_through_point_with_equal_intercepts_l226_226161

theorem line_through_point_with_equal_intercepts :
  ∃ (m b : ℝ), ∀ (x y : ℝ), 
    ((y = m * x + b ∧ ((x = 0 ∨ y = 0) → (x = y))) ∧ 
    (1 = m * 1 + b ∧ 1 + 1 = b)) → 
    (m = 1 ∧ b = 0) ∨ (m = -1 ∧ b = 2) :=
by
  sorry

end line_through_point_with_equal_intercepts_l226_226161


namespace xy_equals_one_l226_226309

-- Define the mathematical theorem
theorem xy_equals_one (x y : ℝ) (h : x + y = 1 / x + 1 / y) (h₂ : x + y ≠ 0) : x * y = 1 := 
by
  sorry

end xy_equals_one_l226_226309


namespace largest_n_for_triangle_property_l226_226412

-- Define the triangle property for a set
def triangle_property (S : Set ℕ) : Prop :=
  ∀ {a b c : ℕ}, a ∈ S → b ∈ S → c ∈ S → a < b → b < c → a + b > c

-- Define the smallest subset that violates the triangle property
def violating_subset : Set ℕ := {5, 6, 11, 17, 28, 45, 73, 118, 191, 309}

-- Define the set of consecutive integers from 5 to n
def consecutive_integers (n : ℕ) : Set ℕ := {x : ℕ | 5 ≤ x ∧ x ≤ n}

-- The theorem we want to prove
theorem largest_n_for_triangle_property : ∀ (S : Set ℕ), S = consecutive_integers 308 → triangle_property S := sorry

end largest_n_for_triangle_property_l226_226412


namespace proof_problem_l226_226049

variable (x y : ℝ)

noncomputable def condition1 : Prop := x > y
noncomputable def condition2 : Prop := x * y = 1

theorem proof_problem (hx : condition1 x y) (hy : condition2 x y) : 
  (x^2 + y^2) / (x - y) ≥ 2 * Real.sqrt 2 := 
by
  sorry

end proof_problem_l226_226049


namespace question_statement_l226_226572

-- Definitions based on conditions
def all_cards : List ℕ := [8, 3, 6, 5, 0, 7]
def A : ℕ := 876  -- The largest number from the given cards.
def B : ℕ := 305  -- The smallest number from the given cards with non-zero hundreds place.

-- The proof problem statement
theorem question_statement :
  (A - B) * 6 = 3426 := by
  sorry

end question_statement_l226_226572


namespace find_d_l226_226209

theorem find_d : ∃ d : ℝ, (∀ x : ℝ, 2 * x^2 + 9 * x + d = 0 ↔ x = (-9 + Real.sqrt 17) / 4 ∨ x = (-9 - Real.sqrt 17) / 4) ∧ d = 8 :=
by
  sorry

end find_d_l226_226209


namespace xy_inequality_l226_226892

theorem xy_inequality (x y : ℝ) (n : ℕ) (hx : 0 < x) (hy : 0 < y) : 
  x * y ≤ (x^(n+2) + y^(n+2)) / (x^n + y^n) :=
sorry

end xy_inequality_l226_226892


namespace sum_fourth_power_l226_226619

  theorem sum_fourth_power (x y z : ℝ) 
    (h1 : x + y + z = 2) 
    (h2 : x^2 + y^2 + z^2 = 6) 
    (h3 : x^3 + y^3 + z^3 = 8) : 
    x^4 + y^4 + z^4 = 26 := 
  by 
    sorry
  
end sum_fourth_power_l226_226619


namespace max_m_n_l226_226002

theorem max_m_n (m n: ℕ) (h: m + 3*n - 5 = 2 * Nat.lcm m n - 11 * Nat.gcd m n) : 
  m + n ≤ 70 :=
sorry

end max_m_n_l226_226002


namespace additional_time_due_to_leak_is_six_l226_226020

open Real

noncomputable def filling_time_with_leak (R L : ℝ) : ℝ := 1 / (R - L)
noncomputable def filling_time_without_leak (R : ℝ) : ℝ := 1 / R
noncomputable def additional_filling_time (R L : ℝ) : ℝ :=
  filling_time_with_leak R L - filling_time_without_leak R

theorem additional_time_due_to_leak_is_six :
  additional_filling_time 0.25 (3 / 20) = 6 := by
  sorry

end additional_time_due_to_leak_is_six_l226_226020


namespace find_values_general_formula_l226_226146

variable (a_n S_n : ℕ → ℝ)

-- Conditions
axiom sum_sequence (n : ℕ) (hn : n > 0) :  S_n n = (1 / 3) * (a_n n - 1)

-- Questions
theorem find_values :
  (a_n 1 = 2) ∧ (a_n 2 = 5) ∧ (a_n 3 = 8) := sorry

theorem general_formula (n : ℕ) :
  n > 0 → a_n n = n + 1 := sorry

end find_values_general_formula_l226_226146


namespace relay_race_total_time_l226_226911

theorem relay_race_total_time :
  let t1 := 55
  let t2 := t1 + 10
  let t3 := t2 - 15
  let t4 := t1 - 25
  t1 + t2 + t3 + t4 = 200 := by
    sorry

end relay_race_total_time_l226_226911


namespace product_not_divisible_by_770_l226_226055

theorem product_not_divisible_by_770 (a b : ℕ) (h : a + b = 770) : ¬ (a * b) % 770 = 0 :=
sorry

end product_not_divisible_by_770_l226_226055


namespace grooming_time_correct_l226_226288

def time_to_groom_poodle : ℕ := 30
def time_to_groom_terrier : ℕ := time_to_groom_poodle / 2
def num_poodles : ℕ := 3
def num_terriers : ℕ := 8

def total_grooming_time : ℕ := 
  (num_poodles * time_to_groom_poodle) + (num_terriers * time_to_groom_terrier)

theorem grooming_time_correct : 
  total_grooming_time = 210 := by
  sorry

end grooming_time_correct_l226_226288


namespace avg_speed_additional_hours_l226_226384

/-- Definitions based on the problem conditions -/
def first_leg_speed : ℕ := 30 -- miles per hour
def first_leg_time : ℕ := 6 -- hours
def total_trip_time : ℕ := 8 -- hours
def total_avg_speed : ℕ := 34 -- miles per hour

/-- The theorem that ties everything together -/
theorem avg_speed_additional_hours : 
  ((total_avg_speed * total_trip_time) - (first_leg_speed * first_leg_time)) / (total_trip_time - first_leg_time) = 46 := 
sorry

end avg_speed_additional_hours_l226_226384


namespace num_values_x_satisfying_l226_226341

theorem num_values_x_satisfying (
  f : ℝ → ℝ → ℝ)
  (cos : ℝ → ℝ)
  (sin : ℝ → ℝ)
  (x : ℝ)
  (h_eq : ∀ x, f (cos x) (sin x) = 2 ↔ (cos x) ^ 2 + 3 * (sin x) ^ 2 = 2)
  (h_interval : ∀ x, -20 < x ∧ x < 90)
  (h_cos_sin : ∀ x, cos x = cos (x) ∧ sin x = sin (x)) :
  ∃ n, n = 70 := sorry

end num_values_x_satisfying_l226_226341


namespace evaluate_square_difference_l226_226371

theorem evaluate_square_difference:
  let a := 70
  let b := 30
  (a^2 - b^2) = 4000 :=
by
  sorry

end evaluate_square_difference_l226_226371


namespace all_children_receive_candy_iff_power_of_two_l226_226368

theorem all_children_receive_candy_iff_power_of_two (n : ℕ) : 
  (∀ (k : ℕ), k < n → ∃ (m : ℕ), (m * (m + 1) / 2) % n = k) ↔ ∃ (k : ℕ), n = 2^k :=
by sorry

end all_children_receive_candy_iff_power_of_two_l226_226368


namespace initial_money_eq_l226_226051

-- Definitions for the problem conditions
def spent_on_sweets : ℝ := 1.25
def spent_on_friends : ℝ := 2 * 1.20
def money_left : ℝ :=  4.85

-- Statement of the problem to prove
theorem initial_money_eq :
  spent_on_sweets + spent_on_friends + money_left = 8.50 := 
sorry

end initial_money_eq_l226_226051


namespace tom_age_ratio_l226_226538

theorem tom_age_ratio (T N : ℕ)
  (sum_children : T = T) 
  (age_condition : T - N = 3 * (T - 4 * N)) :
  T / N = 11 / 2 := 
sorry

end tom_age_ratio_l226_226538


namespace jane_uses_40_ribbons_l226_226404

theorem jane_uses_40_ribbons :
  (∀ dresses1 dresses2 ribbons_per_dress, 
  dresses1 = 2 * 7 ∧ 
  dresses2 = 3 * 2 → 
  ribbons_per_dress = 2 → 
  (dresses1 + dresses2) * ribbons_per_dress = 40)
:= 
by 
  sorry

end jane_uses_40_ribbons_l226_226404


namespace no_divisors_in_range_l226_226701

theorem no_divisors_in_range : ¬ ∃ n : ℕ, 80 < n ∧ n < 90 ∧ n ∣ (3^40 - 1) :=
by sorry

end no_divisors_in_range_l226_226701


namespace valid_outfits_count_l226_226569

-- Definitions based on problem conditions
def shirts : Nat := 5
def pants : Nat := 6
def invalid_combination : Nat := 1

-- Problem statement
theorem valid_outfits_count : shirts * pants - invalid_combination = 29 := by 
  sorry

end valid_outfits_count_l226_226569


namespace tan_product_ge_sqrt2_l226_226413

variable {α β γ : ℝ}

theorem tan_product_ge_sqrt2 (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) (hγ : 0 < γ ∧ γ < π / 2) 
  (h : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) : 
  Real.tan α * Real.tan β * Real.tan γ ≥ 2 * Real.sqrt 2 := 
by
  sorry

end tan_product_ge_sqrt2_l226_226413


namespace min_sticks_to_break_for_square_12_can_form_square_without_breaking_15_l226_226914

-- Part (a): For n = 12:
theorem min_sticks_to_break_for_square_12 : ∀ (n : ℕ), n = 12 → 
  (∃ (sticks : Finset ℕ), sticks.card = 12 ∧ sticks.sum id = 78 ∧ (¬ (78 % 4 = 0) → 
  ∃ (b : ℕ), b = 2)) := 
by sorry

-- Part (b): For n = 15:
theorem can_form_square_without_breaking_15 : ∀ (n : ℕ), n = 15 → 
  (∃ (sticks : Finset ℕ), sticks.card = 15 ∧ sticks.sum id = 120 ∧ (120 % 4 = 0)) :=
by sorry

end min_sticks_to_break_for_square_12_can_form_square_without_breaking_15_l226_226914


namespace total_wheels_in_neighborhood_l226_226702

def cars_in_Jordan_driveway := 2
def wheels_per_car := 4
def spare_wheel := 1
def bikes_with_2_wheels := 3
def wheels_per_bike := 2
def bike_missing_rear_wheel := 1
def bike_with_training_wheel := 2 + 1
def trash_can_wheels := 2
def tricycle_wheels := 3
def wheelchair_main_wheels := 2
def wheelchair_small_wheels := 2
def wagon_wheels := 4
def roller_skates_total_wheels := 4
def roller_skates_missing_wheel := 1

def pickup_truck_wheels := 4
def boat_trailer_wheels := 2
def motorcycle_wheels := 2
def atv_wheels := 4

theorem total_wheels_in_neighborhood :
  (cars_in_Jordan_driveway * wheels_per_car + spare_wheel + bikes_with_2_wheels * wheels_per_bike + bike_missing_rear_wheel + bike_with_training_wheel + trash_can_wheels + tricycle_wheels + wheelchair_main_wheels + wheelchair_small_wheels + wagon_wheels + (roller_skates_total_wheels - roller_skates_missing_wheel)) +
  (pickup_truck_wheels + boat_trailer_wheels + motorcycle_wheels + atv_wheels) = 47 := by
  sorry

end total_wheels_in_neighborhood_l226_226702


namespace product_of_divisors_18_l226_226703

-- Definitions
def num := 18
def divisors := [1, 2, 3, 6, 9, 18]

-- The theorem statement
theorem product_of_divisors_18 : 
  (divisors.foldl (·*·) 1) = 104976 := 
by sorry

end product_of_divisors_18_l226_226703


namespace perpendicular_vectors_parallel_vectors_l226_226601

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (2, x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x - 1, 1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors (x : ℝ) :
  dot_product (vector_a x) (vector_b x) = 0 ↔ x = 2 / 3 :=
by sorry

theorem parallel_vectors (x : ℝ) :
  (2 / (x - 1) = x) ∨ (x - 1 = 0) ∨ (2 = 0) ↔ (x = 2 ∨ x = -1) :=
by sorry

end perpendicular_vectors_parallel_vectors_l226_226601


namespace inequalities_l226_226475

variable {a b c : ℝ}

theorem inequalities (ha : a < 0) (hab : a < b) (hbc : b < c) :
  a^2 * b < b^2 * c ∧ a^2 * c < b^2 * c ∧ a^2 * b < a^2 * c :=
by
  sorry

end inequalities_l226_226475


namespace three_kids_savings_l226_226604

theorem three_kids_savings :
  (200 / 100) + (100 / 20) + (330 / 10) = 40 :=
by
  -- Proof goes here
  sorry

end three_kids_savings_l226_226604


namespace sacks_per_day_l226_226213

theorem sacks_per_day (total_sacks : ℕ) (total_days : ℕ) (harvest_per_day : ℕ) : 
  total_sacks = 56 → 
  total_days = 14 → 
  harvest_per_day = total_sacks / total_days → 
  harvest_per_day = 4 := 
by
  intros h_total_sacks h_total_days h_harvest_per_day
  rw [h_total_sacks, h_total_days] at h_harvest_per_day
  simp at h_harvest_per_day
  exact h_harvest_per_day

end sacks_per_day_l226_226213


namespace people_per_table_l226_226813

theorem people_per_table (initial_customers left_customers tables remaining_customers : ℕ) 
  (h1 : initial_customers = 21) 
  (h2 : left_customers = 12) 
  (h3 : tables = 3) 
  (h4 : remaining_customers = initial_customers - left_customers) 
  : remaining_customers / tables = 3 :=
by
  sorry

end people_per_table_l226_226813


namespace kendra_more_buttons_l226_226238

theorem kendra_more_buttons {K M S : ℕ} (hM : M = 8) (hS : S = 22) (hHalfK : S = K / 2) :
  K - 5 * M = 4 :=
by
  sorry

end kendra_more_buttons_l226_226238


namespace sqrt_meaningful_range_l226_226004

theorem sqrt_meaningful_range (x : ℝ) (h : 0 ≤ x - 2) : x ≥ 2 :=
sorry

end sqrt_meaningful_range_l226_226004


namespace sqrt_x_plus_inv_sqrt_x_eq_sqrt_152_l226_226919

-- Conditions
variable (x : ℝ) (h₀ : 0 < x) (h₁ : x + 1 / x = 150)

-- Statement to prove
theorem sqrt_x_plus_inv_sqrt_x_eq_sqrt_152 : (Real.sqrt x + Real.sqrt (1 / x) = Real.sqrt 152) := 
sorry -- Proof not needed, skip with sorry

end sqrt_x_plus_inv_sqrt_x_eq_sqrt_152_l226_226919


namespace range_of_a_l226_226850

theorem range_of_a (a : ℝ) : (∃ x y : ℝ, x - y + 1 = 0 ∧ (x - a)^2 + y^2 = 2) ↔ -3 ≤ a ∧ a ≤ 1 := by
  sorry

end range_of_a_l226_226850


namespace sumsquare_properties_l226_226816

theorem sumsquare_properties {a b c d e f g h i : ℕ} (hc1 : a + b + c = d + e + f) 
(hc2 : d + e + f = g + h + i) 
(hc3 : a + e + i = d + e + f) 
(hc4 : c + e + g = d + e + f) : 
∃ m : ℕ, m % 3 = 0 ∧ (a ≤ (2 * m / 3 - 1)) ∧ (b ≤ (2 * m / 3 - 1)) ∧ (c ≤ (2 * m / 3 - 1)) ∧ (d ≤ (2 * m / 3 - 1)) ∧ (e ≤ (2 * m / 3 - 1)) ∧ (f ≤ (2 * m / 3 - 1)) ∧ (g ≤ (2 * m / 3 - 1)) ∧ (h ≤ (2 * m / 3 - 1)) ∧ (i ≤ (2 * m / 3 - 1)) := 
by {
  sorry
}

end sumsquare_properties_l226_226816


namespace real_solution_count_l226_226672

noncomputable def f (x : ℝ) : ℝ :=
  (1/(x - 1)) + (2/(x - 2)) + (3/(x - 3)) + (4/(x - 4)) + 
  (5/(x - 5)) + (6/(x - 6)) + (7/(x - 7)) + (8/(x - 8)) + 
  (9/(x - 9)) + (10/(x - 10))

theorem real_solution_count : ∃ n : ℕ, n = 11 ∧ 
  ∃ x : ℝ, f x = x :=
sorry

end real_solution_count_l226_226672


namespace sum_of_surface_points_l226_226853

theorem sum_of_surface_points
  (n : ℕ) (h_n : n = 2012) 
  (total_sum : ℕ) (h_total : total_sum = n * 21)
  (matching_points_sum : ℕ) (h_matching : matching_points_sum = (n - 1) * 7)
  (x : ℕ) (h_x_range : 1 ≤ x ∧ x ≤ 6) :
  (total_sum - matching_points_sum + 2 * x = 28177 ∨
   total_sum - matching_points_sum + 2 * x = 28179 ∨
   total_sum - matching_points_sum + 2 * x = 28181 ∨
   total_sum - matching_points_sum + 2 * x = 28183 ∨
   total_sum - matching_points_sum + 2 * x = 28185 ∨
   total_sum - matching_points_sum + 2 * x = 28187) :=
by sorry

end sum_of_surface_points_l226_226853


namespace paul_eats_sandwiches_l226_226765

theorem paul_eats_sandwiches (S : ℕ) (h : (S + 2 * S + 4 * S) * 2 = 28) : S = 2 :=
by
  sorry

end paul_eats_sandwiches_l226_226765


namespace average_salary_of_employees_l226_226509

theorem average_salary_of_employees
  (A : ℝ)  -- Define the average monthly salary A of 18 employees
  (h1 : 18*A + 5800 = 19*(A + 200))  -- Condition given in the problem
  : A = 2000 :=  -- The conclusion we need to prove
by
  sorry

end average_salary_of_employees_l226_226509


namespace negation_of_all_have_trap_consumption_l226_226946

-- Definitions for the conditions
def domestic_mobile_phone : Type := sorry

def has_trap_consumption (phone : domestic_mobile_phone) : Prop := sorry

def all_have_trap_consumption : Prop := ∀ phone : domestic_mobile_phone, has_trap_consumption phone

-- Statement of the problem
theorem negation_of_all_have_trap_consumption :
  ¬ all_have_trap_consumption ↔ ∃ phone : domestic_mobile_phone, ¬ has_trap_consumption phone :=
sorry

end negation_of_all_have_trap_consumption_l226_226946


namespace swap_columns_produce_B_l226_226111

variables {n : ℕ} (A : Matrix (Fin n) (Fin n) (Fin n))

def K (B : Matrix (Fin n) (Fin n) (Fin n)) : ℕ :=
  Fintype.card {ij : (Fin n) × (Fin n) // B ij.1 ij.2 = ij.2}

theorem swap_columns_produce_B (A : Matrix (Fin n) (Fin n) (Fin n)) :
  ∃ (B : Matrix (Fin n) (Fin n) (Fin n)), (∀ i, ∃ j, B i j = A i j) ∧ K B ≤ n :=
sorry

end swap_columns_produce_B_l226_226111


namespace arithmetic_sum_l226_226092

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h1 : a 1 = 2)
  (h2 : a 2 + a 3 = 13) :
  a 4 + a 5 + a 6 = 42 :=
  sorry

end arithmetic_sum_l226_226092


namespace general_formula_an_bounds_Mn_l226_226848

variable {n : ℕ}

-- Define the sequence Sn
def S : ℕ → ℚ := λ n => n * (4 * n - 3) - 2 * n * (n - 1)

-- Define the sequence an based on Sn
def a : ℕ → ℚ := λ n =>
  if n = 0 then 0 else S n - S (n - 1)

-- Define the sequence Mn and the bounds to prove
def M : ℕ → ℚ := λ n => (1 / 4) * (1 - (1 / (4 * n + 1)))

-- Theorem: General formula for the sequence {a_n}
theorem general_formula_an (n : ℕ) (hn : 1 ≤ n) : a n = 4 * n - 3 :=
  sorry

-- Theorem: Bounds for the sequence {M_n}
theorem bounds_Mn (n : ℕ) (hn : 1 ≤ n) : (1 / 5 : ℚ) ≤ M n ∧ M n < (1 / 4) :=
  sorry

end general_formula_an_bounds_Mn_l226_226848


namespace right_triangle_hypotenuse_inequality_l226_226794

theorem right_triangle_hypotenuse_inequality
  (a b c m : ℝ)
  (h_right_triangle : c^2 = a^2 + b^2)
  (h_area_relation : a * b = c * m) :
  m + c > a + b :=
by
  sorry

end right_triangle_hypotenuse_inequality_l226_226794


namespace M_intersect_N_eq_l226_226955

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
def N : Set ℝ := {y | ∃ x : ℝ, y = x + 1}

-- Define what we need to prove
theorem M_intersect_N_eq : M ∩ N = {y | y ≥ 1} :=
by
  sorry

end M_intersect_N_eq_l226_226955


namespace probability_x_gt_3y_l226_226982

noncomputable def rect_region := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3020 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3010}

theorem probability_x_gt_3y : 
  (∫ p in rect_region, if p.1 > 3 * p.2 then 1 else (0:ℝ)) / 
  (∫ p in rect_region, (1:ℝ)) = 1007 / 6020 := sorry

end probability_x_gt_3y_l226_226982


namespace anne_wandering_time_l226_226894

theorem anne_wandering_time (distance speed : ℝ) (h_dist : distance = 3.0) (h_speed : speed = 2.0) : 
  distance / speed = 1.5 :=
by
  rw [h_dist, h_speed]
  norm_num

end anne_wandering_time_l226_226894


namespace polynomial_evaluation_l226_226313

theorem polynomial_evaluation (x : ℝ) (h1 : x^2 - 4 * x - 12 = 0) (h2 : 0 < x) : x^3 - 4 * x^2 - 12 * x + 16 = 16 := 
by
  sorry

end polynomial_evaluation_l226_226313


namespace first_year_with_digit_sum_seven_l226_226157

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem first_year_with_digit_sum_seven : ∃ y, y > 2023 ∧ sum_of_digits y = 7 ∧ ∀ z, z > 2023 ∧ z < y → sum_of_digits z ≠ 7 :=
by
  use 2032
  sorry

end first_year_with_digit_sum_seven_l226_226157


namespace length_PQ_l226_226329

theorem length_PQ (AB BC CA AH : ℝ) (P Q : ℝ) : 
  AB = 7 → BC = 8 → CA = 9 → 
  AH = 3 * Real.sqrt 5 → 
  PQ = AQ - AP → 
  AQ = 7 * (Real.sqrt 5) / 3 → 
  AP = 9 * (Real.sqrt 5) / 5 → 
  PQ = Real.sqrt 5 * 8 / 15 :=
by
  intros hAB hBC hCA hAH hPQ hAQ hAP
  sorry

end length_PQ_l226_226329


namespace mandy_quarters_l226_226397

theorem mandy_quarters (q : ℕ) : 
  40 < q ∧ q < 400 ∧ 
  q % 6 = 2 ∧ 
  q % 7 = 2 ∧ 
  q % 8 = 2 →
  (q = 170 ∨ q = 338) :=
by
  intro h
  sorry

end mandy_quarters_l226_226397


namespace least_product_of_distinct_primes_greater_than_50_l226_226724

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def distinct_primes_greater_than_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p ≠ q ∧ p > 50 ∧ q > 50

theorem least_product_of_distinct_primes_greater_than_50 :
  ∃ (p q : ℕ), distinct_primes_greater_than_50 p q ∧ p * q = 3127 :=
by
  sorry

end least_product_of_distinct_primes_greater_than_50_l226_226724


namespace arithmetic_sequence_sum_l226_226860

theorem arithmetic_sequence_sum (a_n : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : S 9 = a_n 4 + a_n 5 + a_n 6 + 72)
  (h2 : ∀ n, S n = n * (a_n 1 + a_n n) / 2)
  (h3 : ∀ n, a_n (n+1) - a_n n = d)
  (h4 : a_n 1 + a_n 9 = a_n 3 + a_n 7)
  (h5 : a_n 3 + a_n 7 = a_n 4 + a_n 6)
  (h6 : a_n 4 + a_n 6 = 2 * a_n 5) : 
  a_n 3 + a_n 7 = 24 := 
sorry

end arithmetic_sequence_sum_l226_226860


namespace sub_eight_l226_226398

theorem sub_eight (x y : ℝ) (h1 : 3 = 0.15 * x) (h2 : 3 = 0.25 * y) : x - y = 8 :=
by
  sorry

end sub_eight_l226_226398


namespace max_disjoint_regions_l226_226292

theorem max_disjoint_regions {p : ℕ} (hp : Nat.Prime p) (hp_ge3 : 3 ≤ p) : ∃ R, R = 3 * p^2 - 3 * p + 1 :=
by
  sorry

end max_disjoint_regions_l226_226292


namespace plastering_cost_l226_226513

variable (l w d : ℝ) (c : ℝ)

theorem plastering_cost :
  l = 60 → w = 25 → d = 10 → c = 0.90 →
    let A_bottom := l * w;
    let A_long_walls := 2 * (l * d);
    let A_short_walls := 2 * (w * d);
    let A_total := A_bottom + A_long_walls + A_short_walls;
    let C_total := A_total * c;
    C_total = 2880 :=
by sorry

end plastering_cost_l226_226513


namespace original_number_j_l226_226312

noncomputable def solution (n : ℚ) : ℚ := (3 * (n + 3) - 5) / 3

theorem original_number_j { n : ℚ } (h : solution n = 10) : n = 26 / 3 :=
by
  sorry

end original_number_j_l226_226312


namespace average_temperature_week_l226_226450

theorem average_temperature_week :
  let sunday := 99.1
  let monday := 98.2
  let tuesday := 98.7
  let wednesday := 99.3
  let thursday := 99.8
  let friday := 99.0
  let saturday := 98.9
  (sunday + monday + tuesday + wednesday + thursday + friday + saturday) / 7 = 99.0 :=
by
  sorry

end average_temperature_week_l226_226450


namespace parametrize_line_l226_226535

theorem parametrize_line (s h : ℝ) :
    s = -5/2 ∧ h = 20 → ∀ t : ℝ, ∃ x y : ℝ, 4 * x + 7 = y ∧ 
    (x = s + 5 * t ∧ y = -3 + h * t) :=
by
  sorry

end parametrize_line_l226_226535


namespace perimeter_of_irregular_pentagonal_picture_frame_l226_226168

theorem perimeter_of_irregular_pentagonal_picture_frame 
  (base : ℕ) (left_side : ℕ) (right_side : ℕ) (top_left_diagonal_side : ℕ) (top_right_diagonal_side : ℕ)
  (h_base : base = 10) (h_left_side : left_side = 12) (h_right_side : right_side = 11)
  (h_top_left_diagonal_side : top_left_diagonal_side = 6) (h_top_right_diagonal_side : top_right_diagonal_side = 7) :
  base + left_side + right_side + top_left_diagonal_side + top_right_diagonal_side = 46 :=
by {
  sorry
}

end perimeter_of_irregular_pentagonal_picture_frame_l226_226168


namespace arithmetic_sequence_a12_l226_226972

theorem arithmetic_sequence_a12 (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 7 + a 9 = 16) (h2 : a 4 = 1) 
  (h3 : ∀ n, a (n + 1) = a n + d) : a 12 = 15 := 
by {
  -- Proof steps would go here
  sorry
}

end arithmetic_sequence_a12_l226_226972


namespace ratio_of_areas_l226_226281

theorem ratio_of_areas 
  (t : ℝ) (q : ℝ)
  (h1 : t = 1 / 4)
  (h2 : q = 1 / 2) :
  q / t = 2 :=
by sorry

end ratio_of_areas_l226_226281


namespace determine_parabola_equation_l226_226207

-- Given conditions
variable (p : ℝ) (h_p : p > 0)
variable (x1 x2 : ℝ)
variable (AF BF : ℝ)
variable (h_AF : AF = x1 + p / 2)
variable (h_BF : BF = x2 + p / 2)
variable (h_AF_value : AF = 2)
variable (h_BF_value : BF = 3)

-- Prove the equation of the parabola
theorem determine_parabola_equation (h1 : x1 + x2 = 5 - p)
(h2 : x1 * x2 = p^2 / 4)
(h3 : AF * BF = 6) :
  y^2 = (24/5 : ℝ) * x := 
sorry

end determine_parabola_equation_l226_226207


namespace probability_white_balls_le_1_l226_226254

-- Definitions and conditions
def total_balls : ℕ := 6
def red_balls : ℕ := 4
def white_balls : ℕ := 2
def selected_balls : ℕ := 3

-- Combinatorial computations
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Calculations based on the conditions
def total_combinations : ℕ := C total_balls selected_balls
def red_combinations : ℕ := C red_balls selected_balls
def white_combinations : ℕ := C white_balls 1 * C red_balls 2

-- Probability calculations
def P_xi_le_1 : ℚ :=
  (red_combinations / total_combinations : ℚ) +
  (white_combinations / total_combinations : ℚ)

-- Problem statement: Prove that the calculated probability is 4/5
theorem probability_white_balls_le_1 : P_xi_le_1 = 4 / 5 := 
  sorry

end probability_white_balls_le_1_l226_226254


namespace general_formula_an_l226_226960

theorem general_formula_an {a : ℕ → ℝ} (S : ℕ → ℝ) (d : ℝ) (hS : ∀ n, S n = (n / 2) * (a 1 + a n)) (hd : d = a 2 - a 1) : 
  ∀ n, a n = a 1 + (n - 1) * d :=
sorry

end general_formula_an_l226_226960


namespace slower_speed_is_10_l226_226686

-- Define the problem conditions
def walked_distance (faster_speed slower_speed actual_distance extra_distance : ℕ) : Prop :=
  actual_distance / slower_speed = (actual_distance + extra_distance) / faster_speed

-- Define main statement to prove
theorem slower_speed_is_10 (actual_distance : ℕ) (extra_distance : ℕ) (faster_speed : ℕ) (slower_speed : ℕ) :
  walked_distance faster_speed slower_speed actual_distance extra_distance ∧ 
  faster_speed = 15 ∧ extra_distance = 15 ∧ actual_distance = 30 → slower_speed = 10 :=
by
  intro h
  sorry

end slower_speed_is_10_l226_226686


namespace coins_problem_l226_226727

theorem coins_problem : ∃ n : ℕ, (n % 8 = 6) ∧ (n % 9 = 7) ∧ (n % 11 = 8) :=
by {
  sorry
}

end coins_problem_l226_226727


namespace smallest_ratio_l226_226376

theorem smallest_ratio (r s : ℤ) (h1 : 3 * r ≥ 2 * s - 3) (h2 : 4 * s ≥ r + 12) : 
  (∃ r s, (r : ℚ) / s = 1 / 2) :=
by 
  sorry

end smallest_ratio_l226_226376


namespace hyperbola_foci_coords_l226_226936

theorem hyperbola_foci_coords :
  let a := 5
  let b := 2
  let c := Real.sqrt (a^2 + b^2)
  ∀ x y : ℝ, 4 * y^2 - 25 * x^2 = 100 →
  (x = 0 ∧ (y = c ∨ y = -c)) := by
  intros a b c x y h
  have h1 : 4 * y^2 = 100 + 25 * x^2 := by linarith
  have h2 : y^2 = 25 + 25/4 * x^2 := by linarith
  have h3 : x = 0 := by sorry
  have h4 : y = c ∨ y = -c := by sorry
  exact ⟨h3, h4⟩

end hyperbola_foci_coords_l226_226936


namespace symmetric_point_y_axis_l226_226663

-- Define the original point P
def P : ℝ × ℝ := (1, 6)

-- Define the reflection across the y-axis
def reflect_y_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.fst, point.snd)

-- Define the symmetric point with respect to the y-axis
def symmetric_point := reflect_y_axis P

-- Statement to prove
theorem symmetric_point_y_axis : symmetric_point = (-1, 6) :=
by
  -- Proof omitted
  sorry

end symmetric_point_y_axis_l226_226663


namespace distance_between_petya_and_misha_l226_226818

theorem distance_between_petya_and_misha 
  (v1 v2 v3 : ℝ) -- Speeds of Misha, Dima, and Petya
  (t1 : ℝ) -- Time taken by Misha to finish the race
  (d : ℝ := 1000) -- Distance of the race
  (h1 : d - (v1 * (d / v1)) = 0)
  (h2 : d - 0.9 * v1 * (d / v1) = 100)
  (h3 : d - 0.81 * v1 * (d / v1) = 100) :
  (d - 0.81 * v1 * (d / v1) = 190) := 
sorry

end distance_between_petya_and_misha_l226_226818


namespace single_elimination_games_l226_226528

theorem single_elimination_games (n : ℕ) (h : n = 512) : (n - 1) = 511 :=
by
  sorry

end single_elimination_games_l226_226528


namespace geometric_seq_reciprocal_sum_l226_226906

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop := ∀ n, a (n + 1) = a n * r

theorem geometric_seq_reciprocal_sum
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom : geometric_sequence a r)
  (h1 : a 2 * a 5 = -3/4)
  (h2 : a 2 + a 3 + a 4 + a 5 = 5/4) :
  (1 / a 2) + (1 / a 3) + (1 / a 4) + (1 / a 5) = -5/3 := sorry

end geometric_seq_reciprocal_sum_l226_226906


namespace problem_statement_l226_226964

-- Define the statement for positive integers m and n
def div_equiv (m n : ℕ) : Prop :=
  19 ∣ (11 * m + 2 * n) ↔ 19 ∣ (18 * m + 5 * n)

-- The final theorem statement
theorem problem_statement (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : div_equiv m n :=
by
  sorry

end problem_statement_l226_226964


namespace cubic_roots_c_div_d_l226_226753

theorem cubic_roots_c_div_d (a b c d : ℚ) :
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = -1 ∨ x = 1/2 ∨ x = 4) →
  (c / d = 9 / 4) :=
by
  intros h
  -- Proof would go here
  sorry

end cubic_roots_c_div_d_l226_226753


namespace problem1_part1_problem1_part2_problem2_l226_226410

open Set

-- Definitions for sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}
def U : Set ℝ := univ

-- Part (1) of the problem
theorem problem1_part1 : A ∩ B = {x | 2 < x ∧ x ≤ 3} :=
sorry

theorem problem1_part2 : A ∪ (U \ B) = {x | x ≤ 3} :=
sorry

-- Definitions for set C
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

-- Part (2) of the problem
theorem problem2 (a : ℝ) (h : C a ⊆ A) : 1 < a ∧ a ≤ 3 :=
sorry

end problem1_part1_problem1_part2_problem2_l226_226410


namespace time_to_fill_pool_l226_226805

variable (pool_volume : ℝ) (fill_rate : ℝ) (leak_rate : ℝ)

theorem time_to_fill_pool (h_pool_volume : pool_volume = 60)
    (h_fill_rate : fill_rate = 1.6)
    (h_leak_rate : leak_rate = 0.1) :
    (pool_volume / (fill_rate - leak_rate)) = 40 :=
by
  -- We skip the proof step, only the theorem statement is required
  sorry

end time_to_fill_pool_l226_226805


namespace power_congruence_l226_226525

theorem power_congruence (a b n : ℕ) (h : a ≡ b [MOD n]) : a^n ≡ b^n [MOD n^2] :=
sorry

end power_congruence_l226_226525


namespace additional_amount_per_10_cents_l226_226301

-- Definitions of the given conditions
def expected_earnings_per_share : ℝ := 0.80
def dividend_ratio : ℝ := 0.5
def actual_earnings_per_share : ℝ := 1.10
def shares_owned : ℕ := 600
def total_dividend_paid : ℝ := 312

-- Proof statement
theorem additional_amount_per_10_cents (additional_amount : ℝ) :
  (total_dividend_paid - (shares_owned * (expected_earnings_per_share * dividend_ratio))) / shares_owned / 
  ((actual_earnings_per_share - expected_earnings_per_share) / 0.10) = additional_amount :=
sorry

end additional_amount_per_10_cents_l226_226301


namespace intersection_A_B_l226_226090

def A (x : ℝ) : Prop := (2 * x - 1 > 0)
def B (x : ℝ) : Prop := (x * (x - 2) < 0)

theorem intersection_A_B :
  {x : ℝ | A x ∧ B x} = {x : ℝ | 1 / 2 < x ∧ x < 2} :=
by
  sorry

end intersection_A_B_l226_226090


namespace initial_volume_of_mixture_l226_226435

theorem initial_volume_of_mixture
  (x : ℕ)
  (h1 : 3 * x / (2 * x + 1) = 4 / 3)
  (h2 : x = 4) :
  5 * x = 20 :=
by
  sorry

end initial_volume_of_mixture_l226_226435


namespace sequence_solution_l226_226651

theorem sequence_solution :
  ∃ (a : ℕ → ℕ), a 1 = 5 ∧ a 8 = 8 ∧
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 6 → a i + a (i+1) + a (i+2) = 20) ∧
  (a 1 = 5 ∧ a 2 = 8 ∧ a 3 = 7 ∧ a 4 = 5 ∧ a 5 = 8 ∧ a 6 = 7 ∧ a 7 = 5 ∧ a 8 = 8) :=
by {
  sorry
}

end sequence_solution_l226_226651


namespace max_lambda_inequality_l226_226113

theorem max_lambda_inequality 
  (a b x y : ℝ) 
  (h1 : a ≥ 0) 
  (h2 : b ≥ 0)
  (h3 : x ≥ 0)
  (h4 : y ≥ 0)
  (h5 : a + b = 27) : 
  (a * x^2 + b * y^2 + 4 * x * y)^3 ≥ 4 * (a * x^2 * y + b * x * y^2)^2 :=
sorry

end max_lambda_inequality_l226_226113


namespace red_ball_probability_l226_226224

theorem red_ball_probability : 
  let red_A := 2
  let white_A := 3
  let red_B := 4
  let white_B := 1
  let total_A := red_A + white_A
  let total_B := red_B + white_B
  let prob_red_A := red_A / total_A
  let prob_white_A := white_A / total_A
  let prob_red_B_after_red_A := (red_B + 1) / (total_B + 1)
  let prob_red_B_after_white_A := red_B / (total_B + 1)
  (prob_red_A * prob_red_B_after_red_A + prob_white_A * prob_red_B_after_white_A) = 11 / 15 :=
by {
  sorry
}

end red_ball_probability_l226_226224


namespace quadratic_ineq_solution_range_of_b_for_any_a_l226_226417

variable {α : Type*} [LinearOrderedField α]

noncomputable def f (a b x : α) : α := -3 * x^2 + a * (5 - a) * x + b

theorem quadratic_ineq_solution (a b : α) : 
  (∀ x ∈ Set.Ioo (-1 : α) 3, f a b x > 0) →
  ((a = 2 ∧ b = 9) ∨ (a = 3 ∧ b = 9)) := 
  sorry

theorem range_of_b_for_any_a (a b : α) :
  (∀ a : α, f a b 2 < 0) → 
  b < -1 / 2 := 
  sorry

end quadratic_ineq_solution_range_of_b_for_any_a_l226_226417


namespace systematic_sampling_employee_l226_226825

theorem systematic_sampling_employee
    (n : ℕ)
    (employees : Finset ℕ)
    (sample : Finset ℕ)
    (h_n_52 : n = 52)
    (h_employees : employees = Finset.range 52)
    (h_sample_size : sample.card = 4)
    (h_systematic_sample : sample ⊆ employees)
    (h_in_sample : {6, 32, 45} ⊆ sample) :
    19 ∈ sample :=
by
  -- conditions 
  have h0 : 6 ∈ sample := Finset.mem_of_subset h_in_sample (by simp)
  have h1 : 32 ∈ sample := Finset.mem_of_subset h_in_sample (by simp)
  have h2 : 45 ∈ sample := Finset.mem_of_subset h_in_sample (by simp)
  have h_arith : 6 + 45 = 32 + 19 :=
    by linarith
  sorry

end systematic_sampling_employee_l226_226825


namespace sum_of_edge_lengths_of_truncated_octahedron_prism_l226_226747

-- Define the vertices, edge length, and the assumption of the prism being a truncated octahedron
def prism_vertices : ℕ := 24
def edge_length : ℕ := 5
def truncated_octahedron_edges : ℕ := 36

-- The Lean statement to prove the sum of edge lengths
theorem sum_of_edge_lengths_of_truncated_octahedron_prism :
  prism_vertices = 24 ∧ edge_length = 5 ∧ truncated_octahedron_edges = 36 →
  truncated_octahedron_edges * edge_length = 180 :=
by
  sorry

end sum_of_edge_lengths_of_truncated_octahedron_prism_l226_226747


namespace net_pay_rate_l226_226677

def travelTime := 3 -- hours
def speed := 50 -- miles per hour
def fuelEfficiency := 25 -- miles per gallon
def earningsRate := 0.6 -- dollars per mile
def gasolineCost := 3 -- dollars per gallon

theorem net_pay_rate
  (travelTime : ℕ)
  (speed : ℕ)
  (fuelEfficiency : ℕ)
  (earningsRate : ℚ)
  (gasolineCost : ℚ)
  (h_time : travelTime = 3)
  (h_speed : speed = 50)
  (h_fuelEfficiency : fuelEfficiency = 25)
  (h_earningsRate : earningsRate = 0.6)
  (h_gasolineCost : gasolineCost = 3) :
  (earningsRate * speed * travelTime - (speed * travelTime / fuelEfficiency) * gasolineCost) / travelTime = 24 :=
by
  sorry

end net_pay_rate_l226_226677


namespace correct_factorization_l226_226286

-- Define the expressions involved in the options
def option_A (x a b : ℝ) : Prop := x * (a - b) = a * x - b * x
def option_B (x y : ℝ) : Prop := x^2 - 1 + y^2 = (x - 1) * (x + 1) + y^2
def option_C (x : ℝ) : Prop := x^2 - 1 = (x + 1) * (x - 1)
def option_D (x a b c : ℝ) : Prop := a * x + b * x + c = x * (a + b) + c

-- Theorem stating that option C represents true factorization
theorem correct_factorization (x : ℝ) : option_C x := by
  sorry

end correct_factorization_l226_226286


namespace convex_polyhedron_space_diagonals_l226_226849

theorem convex_polyhedron_space_diagonals
  (vertices : ℕ)
  (edges : ℕ)
  (faces : ℕ)
  (triangular_faces : ℕ)
  (hexagonal_faces : ℕ)
  (total_faces : faces = triangular_faces + hexagonal_faces)
  (vertices_eq : vertices = 30)
  (edges_eq : edges = 72)
  (triangular_faces_eq : triangular_faces = 32)
  (hexagonal_faces_eq : hexagonal_faces = 12)
  (faces_eq : faces = 44) :
  ((vertices * (vertices - 1)) / 2) - edges - 
  (triangular_faces * 0 + hexagonal_faces * ((6 * (6 - 3)) / 2)) = 255 := by
sorry

end convex_polyhedron_space_diagonals_l226_226849


namespace slope_of_line_dividing_rectangle_l226_226471

theorem slope_of_line_dividing_rectangle (h_vertices : 
  ∃ (A B C D : ℝ × ℝ), A = (1, 0) ∧ B = (9, 0) ∧ C = (1, 2) ∧ D = (9, 2) ∧ 
  (∃ line : ℝ × ℝ, line = (0, 0) ∧ line = (5, 1))) : 
  ∃ m : ℝ, m = 1 / 5 :=
sorry

end slope_of_line_dividing_rectangle_l226_226471


namespace complex_pow_i_2019_l226_226479

theorem complex_pow_i_2019 : (Complex.I)^2019 = -Complex.I := 
by
  sorry

end complex_pow_i_2019_l226_226479


namespace sphere_surface_area_from_box_l226_226256

/--
Given a rectangular box with length = 2, width = 2, and height = 1,
prove that if all vertices of the rectangular box lie on the surface of a sphere,
then the surface area of the sphere is 9π.
--/
theorem sphere_surface_area_from_box :
  let length := 2
  let width := 2
  let height := 1
  ∃ (r : ℝ), ∀ (d := Real.sqrt (length^2 + width^2 + height^2)),
  r = d / 2 → 4 * Real.pi * r^2 = 9 * Real.pi :=
by
  sorry

end sphere_surface_area_from_box_l226_226256


namespace pipes_fill_tank_in_1_5_hours_l226_226659

theorem pipes_fill_tank_in_1_5_hours :
  (1 / 3 + 1 / 9 + 1 / 18 + 1 / 6) = (2 / 3) →
  (1 / (2 / 3)) = (3 / 2) :=
by sorry

end pipes_fill_tank_in_1_5_hours_l226_226659


namespace odometer_reading_at_lunch_l226_226757

axiom odometer_start : ℝ
axiom miles_traveled : ℝ
axiom odometer_at_lunch : ℝ
axiom starting_reading : odometer_start = 212.3
axiom travel_distance : miles_traveled = 159.7
axiom at_lunch_reading : odometer_at_lunch = odometer_start + miles_traveled

theorem odometer_reading_at_lunch :
  odometer_at_lunch = 372.0 :=
  by
  sorry

end odometer_reading_at_lunch_l226_226757


namespace correct_system_of_equations_l226_226973

theorem correct_system_of_equations
  (x y : ℝ)
  (h1 : x + (1 / 2) * y = 50)
  (h2 : y + (2 / 3) * x = 50) :
  (x + (1 / 2) * y = 50) ∧ (y + (2 / 3) * x = 50) :=
by
  exact ⟨h1, h2⟩

end correct_system_of_equations_l226_226973


namespace parametric_eqn_and_max_sum_l226_226971

noncomputable def polar_eq (ρ θ : ℝ) := ρ^2 = 4 * ρ * (Real.cos θ + Real.sin θ) - 6

theorem parametric_eqn_and_max_sum (θ : ℝ):
  (∃ (x y : ℝ), (2 + Real.sqrt 2 * Real.cos θ, 2 + Real.sqrt 2 * Real.sin θ) = (x, y)) ∧
  (∃ (θ : ℝ), θ = Real.pi / 4 → (3, 3) = (3, 3) ∧ 6 = 6) :=
by {
  sorry
}

end parametric_eqn_and_max_sum_l226_226971


namespace min_value_expr_l226_226003

theorem min_value_expr : ∀ (x : ℝ), 0 < x ∧ x < 4 → ∃ y : ℝ, y = (1 / (4 - x) + 2 / x) ∧ y = (3 + 2 * Real.sqrt 2) / 4 :=
by
  sorry

end min_value_expr_l226_226003


namespace simplify_expression_l226_226711

theorem simplify_expression (x : ℝ) : 
  6 * (x - 7) * (2 * x + 15) + (3 * x - 4) * (x + 5) = 15 * x^2 + 17 * x - 650 :=
by
  sorry

end simplify_expression_l226_226711


namespace net_rate_of_pay_l226_226262

theorem net_rate_of_pay
  (hours_travelled : ℕ)
  (speed : ℕ)
  (fuel_efficiency : ℕ)
  (pay_per_mile : ℝ)
  (price_per_gallon : ℝ)
  (net_rate_of_pay : ℝ) :
  hours_travelled = 3 →
  speed = 50 →
  fuel_efficiency = 25 →
  pay_per_mile = 0.60 →
  price_per_gallon = 2.50 →
  net_rate_of_pay = 25 := by
  sorry

end net_rate_of_pay_l226_226262


namespace length_of_bridge_l226_226771

theorem length_of_bridge (t : ℝ) (s : ℝ) (d : ℝ) : 
  (t = 24 / 60) ∧ (s = 10) ∧ (d = s * t) → d = 4 := by
  sorry

end length_of_bridge_l226_226771


namespace casey_stays_for_n_months_l226_226345

-- Definitions based on conditions.
def weekly_cost : ℕ := 280
def monthly_cost : ℕ := 1000
def weeks_per_month : ℕ := 4
def total_savings : ℕ := 360

-- Calculate monthly cost when paying weekly.
def monthly_cost_weekly := weekly_cost * weeks_per_month

-- Calculate savings per month when paying monthly instead of weekly.
def savings_per_month := monthly_cost_weekly - monthly_cost

-- Define the problem statement.
theorem casey_stays_for_n_months :
  (total_savings / savings_per_month) = 3 := by
  -- Proof is omitted.
  sorry

end casey_stays_for_n_months_l226_226345


namespace min_balls_in_circle_l226_226606

theorem min_balls_in_circle (b w n k : ℕ) 
  (h1 : b = 2 * w)
  (h2 : n = b + w) 
  (h3 : n - 2 * k = 6 * k) :
  n >= 24 :=
sorry

end min_balls_in_circle_l226_226606


namespace yearly_return_of_1500_investment_is_27_percent_l226_226277

-- Definitions based on conditions
def combined_yearly_return (x : ℝ) : Prop :=
  let investment1 := 500
  let investment2 := 1500
  let total_investment := investment1 + investment2
  let combined_return := 0.22 * total_investment
  let return_from_500 := 0.07 * investment1
  let return_from_1500 := combined_return - return_from_500
  x / 100 * investment2 = return_from_1500

-- Theorem statement to be proven
theorem yearly_return_of_1500_investment_is_27_percent : combined_yearly_return 27 :=
by sorry

end yearly_return_of_1500_investment_is_27_percent_l226_226277


namespace Ali_money_left_l226_226682

theorem Ali_money_left (initial_money : ℕ) 
  (spent_on_food_ratio : ℚ) 
  (spent_on_glasses_ratio : ℚ) 
  (spent_on_food : ℕ) 
  (left_after_food : ℕ) 
  (spent_on_glasses : ℕ) 
  (final_left : ℕ) :
    initial_money = 480 →
    spent_on_food_ratio = 1 / 2 →
    spent_on_food = initial_money * spent_on_food_ratio →
    left_after_food = initial_money - spent_on_food →
    spent_on_glasses_ratio = 1 / 3 →
    spent_on_glasses = left_after_food * spent_on_glasses_ratio →
    final_left = left_after_food - spent_on_glasses →
    final_left = 160 :=
by
  sorry

end Ali_money_left_l226_226682


namespace sum_of_two_numbers_l226_226387

theorem sum_of_two_numbers (x y : ℕ) (h1 : y = 2 * x - 3) (h2 : x = 14) : x + y = 39 :=
by
  sorry

end sum_of_two_numbers_l226_226387


namespace length_of_bridge_correct_l226_226573

noncomputable def L_train : ℝ := 180
noncomputable def v_km_per_hr : ℝ := 60  -- speed in km/hr
noncomputable def t : ℝ := 25

-- Convert speed from km/hr to m/s
noncomputable def km_per_hr_to_m_per_s (v: ℝ) : ℝ := v * (1000 / 3600)
noncomputable def v : ℝ := km_per_hr_to_m_per_s v_km_per_hr

-- Distance covered by the train while crossing the bridge
noncomputable def d : ℝ := v * t

-- Length of the bridge
noncomputable def L_bridge : ℝ := d - L_train

theorem length_of_bridge_correct :
  L_bridge = 236.75 :=
  by
    sorry

end length_of_bridge_correct_l226_226573


namespace max_liters_of_water_heated_l226_226106

theorem max_liters_of_water_heated
  (heat_initial : ℕ := 480) 
  (heat_drop : ℝ := 0.25)
  (temp_initial : ℝ := 20)
  (temp_boiling : ℝ := 100)
  (specific_heat_capacity : ℝ := 4.2)
  (kJ_to_liters_conversion : ℝ := 336) :
  (∀ m : ℕ, (m * kJ_to_liters_conversion > ((heat_initial : ℝ) / (1 - heat_drop)) → m ≤ 5)) :=
by
  sorry

end max_liters_of_water_heated_l226_226106


namespace system_of_equations_solution_l226_226447

theorem system_of_equations_solution (x1 x2 x3 x4 x5 : ℝ) (h1 : x1 + x2 = x3^2) (h2 : x2 + x3 = x4^2)
  (h3 : x3 + x4 = x5^2) (h4 : x4 + x5 = x1^2) (h5 : x5 + x1 = x2^2) :
  x1 = 2 ∧ x2 = 2 ∧ x3 = 2 ∧ x4 = 2 ∧ x5 = 2 := 
sorry

end system_of_equations_solution_l226_226447


namespace root_expression_value_l226_226284

theorem root_expression_value (m : ℝ) (h : 2 * m^2 + 3 * m - 1 = 0) : 4 * m^2 + 6 * m - 2019 = -2017 :=
by
  sorry

end root_expression_value_l226_226284


namespace max_S_n_value_arithmetic_sequence_l226_226927

-- Definitions and conditions
def S_n (n : ℕ) : ℤ := 3 * n - n^2

def a_n (n : ℕ) : ℤ := 
if n = 0 then 0 else S_n n - S_n (n - 1)

-- Statement of the first part of the proof problem
theorem max_S_n_value (n : ℕ) (h : n = 1 ∨ n = 2) : S_n n = 2 :=
sorry

-- Statement of the second part of the proof problem
theorem arithmetic_sequence :
  ∀ n : ℕ, n ≥ 1 → a_n (n + 1) - a_n n = -2 :=
sorry

end max_S_n_value_arithmetic_sequence_l226_226927


namespace inequality_sum_of_reciprocals_l226_226511

variable {a b c : ℝ}

theorem inequality_sum_of_reciprocals
  (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c)
  (hsum : a + b + c = 3) :
  (1 / (2 * a^2 + b^2 + c^2) + 1 / (2 * b^2 + c^2 + a^2) + 1 / (2 * c^2 + a^2 + b^2)) ≤ 3/4 :=
sorry

end inequality_sum_of_reciprocals_l226_226511


namespace fraction_calculation_l226_226294

theorem fraction_calculation : (4 / 9 + 1 / 9) / (5 / 8 - 1 / 8) = 10 / 9 := by
  sorry

end fraction_calculation_l226_226294


namespace black_cars_count_l226_226024

-- Conditions
def red_cars : ℕ := 28
def ratio_red_black : ℚ := 3 / 8

-- Theorem statement
theorem black_cars_count :
  ∃ (black_cars : ℕ), black_cars = 75 ∧ (red_cars : ℚ) / (black_cars) = ratio_red_black :=
sorry

end black_cars_count_l226_226024


namespace num_perfect_square_factors_of_180_l226_226634

theorem num_perfect_square_factors_of_180 (n : ℕ) (h : n = 180) :
  ∃ k : ℕ, k = 4 ∧ ∀ d : ℕ, d ∣ n → ∃ a b c : ℕ, d = 2^a * 3^b * 5^c ∧ a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 :=
by
  use 4
  sorry

end num_perfect_square_factors_of_180_l226_226634


namespace trapezoid_problem_l226_226339

theorem trapezoid_problem (b h x : ℝ) 
  (hb : b > 0)
  (hh : h > 0)
  (h_ratio : (b + 90) / (b + 30) = 3 / 4)
  (h_x_def : x = 150 * (h / (x - 90) - 90))
  (hx2 : x^2 = 26100) :
  ⌊x^2 / 120⌋ = 217 := sorry

end trapezoid_problem_l226_226339


namespace area_parallelogram_l226_226609

theorem area_parallelogram (AE EB : ℝ) (SAEF SCEF SAEC SBEC SABC SABCD : ℝ) (h1 : SAE = 2 * EB)
  (h2 : SCEF = 1) (h3 : SAE == 2 * SCEF / 3) (h4 : SAEC == SAE + SCEF) 
  (h5 : SBEC == 1/2 * SAEC) (h6 : SABC == SAEC + SBEC) (h7 : SABCD == 2 * SABC) :
  SABCD = 5 := sorry

end area_parallelogram_l226_226609


namespace geometric_common_ratio_l226_226586

noncomputable def geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop := ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_common_ratio (a : ℕ → ℝ) (q : ℝ) (h1 : q > 0) 
  (h2 : geometric_seq a q) (h3 : a 3 * a 7 = 4 * (a 4)^2) : q = 2 := 
by 
  sorry

end geometric_common_ratio_l226_226586


namespace equivalent_solution_l226_226561

theorem equivalent_solution (c x : ℤ) 
    (h1 : 3 * x + 9 = 6)
    (h2 : c * x - 15 = -5)
    (hx : x = -1) :
    c = -10 :=
sorry

end equivalent_solution_l226_226561


namespace no_n_satisfies_l226_226970

def sum_first_n_terms_arith_seq (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem no_n_satisfies (n : ℕ) (h_n : n ≠ 0) :
  let s1 := sum_first_n_terms_arith_seq 5 6 n
  let s2 := sum_first_n_terms_arith_seq 12 4 n
  (s1 * s2 = 24 * n^2) → False :=
by
  sorry

end no_n_satisfies_l226_226970


namespace inequality_proof_l226_226530

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 1) :
  ((1 / x^2 - x) * (1 / y^2 - y) * (1 / z^2 - z) ≥ (26 / 3)^3) :=
by sorry

end inequality_proof_l226_226530


namespace symmetric_points_origin_l226_226414

theorem symmetric_points_origin {a b : ℝ} (h₁ : a = -(-4)) (h₂ : b = -(3)) : a - b = 7 :=
by 
  -- since this is a statement template, the proof is omitted
  sorry

end symmetric_points_origin_l226_226414


namespace actual_time_of_storm_l226_226382

theorem actual_time_of_storm
  (malfunctioned_hours_tens_digit : ℕ)
  (malfunctioned_hours_units_digit : ℕ)
  (malfunctioned_minutes_tens_digit : ℕ)
  (malfunctioned_minutes_units_digit : ℕ)
  (original_time : ℕ × ℕ)
  (hours_tens_digit : ℕ := 2)
  (hours_units_digit : ℕ := 0)
  (minutes_tens_digit : ℕ := 0)
  (minutes_units_digit : ℕ := 9) :
  (malfunctioned_hours_tens_digit = hours_tens_digit + 1 ∨ malfunctioned_hours_tens_digit = hours_tens_digit - 1) →
  (malfunctioned_hours_units_digit = hours_units_digit + 1 ∨ malfunctioned_hours_units_digit = hours_units_digit - 1) →
  (malfunctioned_minutes_tens_digit = minutes_tens_digit + 1 ∨ malfunctioned_minutes_tens_digit = minutes_tens_digit - 1) →
  (malfunctioned_minutes_units_digit = minutes_units_digit + 1 ∨ malfunctioned_minutes_units_digit = minutes_units_digit - 1) →
  original_time = (11, 18) :=
by
  sorry

end actual_time_of_storm_l226_226382


namespace boy_lap_time_l226_226132

noncomputable def total_time_needed
  (side_lengths : List ℝ)
  (running_speeds : List ℝ)
  (obstacle_time : ℝ) : ℝ :=
(side_lengths.zip running_speeds).foldl (λ (acc : ℝ) ⟨len, speed⟩ => acc + (len / (speed / 60))) 0
+ obstacle_time

theorem boy_lap_time
  (side_lengths : List ℝ)
  (running_speeds : List ℝ)
  (obstacle_time : ℝ) :
  side_lengths = [80, 120, 140, 100, 60] →
  running_speeds = [250, 200, 300, 166.67, 266.67] →
  obstacle_time = 5 →
  total_time_needed side_lengths running_speeds obstacle_time = 7.212 := by
  intros h_lengths h_speeds h_obstacle_time
  rw [h_lengths, h_speeds, h_obstacle_time]
  sorry

end boy_lap_time_l226_226132


namespace ellipse_major_axis_min_length_l226_226069

theorem ellipse_major_axis_min_length (a b c : ℝ) 
  (h1 : b * c = 2)
  (h2 : a^2 = b^2 + c^2) 
  : 2 * a ≥ 4 :=
sorry

end ellipse_major_axis_min_length_l226_226069


namespace shortest_chord_length_l226_226691

theorem shortest_chord_length
  (x y : ℝ)
  (hx : x^2 + y^2 - 6 * x - 8 * y = 0)
  (point_on_circle : (3, 5) = (x, y)) :
  ∃ (length : ℝ), length = 4 * Real.sqrt 6 := 
by
  sorry

end shortest_chord_length_l226_226691


namespace train_crossing_time_l226_226893

def length_of_train : ℕ := 120
def speed_of_train_kmph : ℕ := 54
def length_of_bridge : ℕ := 660

def speed_of_train_mps : ℕ := speed_of_train_kmph * 1000 / 3600
def total_distance : ℕ := length_of_train + length_of_bridge
def time_to_cross_bridge : ℕ := total_distance / speed_of_train_mps

theorem train_crossing_time :
  time_to_cross_bridge = 52 :=
sorry

end train_crossing_time_l226_226893


namespace line_perpendicular_exists_k_line_intersects_circle_l226_226961

theorem line_perpendicular_exists_k (k : ℝ) :
  ∃ k, (k * (1 / 2)) = -1 :=
sorry

theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, (k * x - y + 2 * k = 0) ∧ (x^2 + y^2 = 8) :=
sorry

end line_perpendicular_exists_k_line_intersects_circle_l226_226961


namespace marissa_tied_boxes_l226_226354

def Total_ribbon : ℝ := 4.5
def Leftover_ribbon : ℝ := 1
def Ribbon_per_box : ℝ := 0.7

theorem marissa_tied_boxes : (Total_ribbon - Leftover_ribbon) / Ribbon_per_box = 5 := by
  sorry

end marissa_tied_boxes_l226_226354


namespace solve_exponent_problem_l226_226266

theorem solve_exponent_problem
  (h : (1 / 8) * (2 ^ 36) = 8 ^ x) : x = 11 :=
by
  sorry

end solve_exponent_problem_l226_226266


namespace find_star_l226_226174

-- Define the problem conditions and statement
theorem find_star (x : ℤ) (star : ℤ) (h1 : x = 5) (h2 : -3 * (star - 9) = 5 * x - 1) : star = 1 :=
by
  sorry -- Proof to be filled in

end find_star_l226_226174


namespace no_all_nine_odd_l226_226491

theorem no_all_nine_odd
  (a1 a2 a3 a4 a5 b1 b2 b3 b4 : ℤ)
  (h1 : a1 % 2 = 1) (h2 : a2 % 2 = 1) (h3 : a3 % 2 = 1)
  (h4 : a4 % 2 = 1) (h5 : a5 % 2 = 1) (h6 : b1 % 2 = 1)
  (h7 : b2 % 2 = 1) (h8 : b3 % 2 = 1) (h9 : b4 % 2 = 1)
  (sum_eq : a1 + a2 + a3 + a4 + a5 = b1 + b2 + b3 + b4) : 
  false :=
sorry

end no_all_nine_odd_l226_226491


namespace ab_equals_one_l226_226521

theorem ab_equals_one {a b : ℝ} (h : a ≠ b) (hf : |Real.log a| = |Real.log b|) : a * b = 1 :=
  sorry

end ab_equals_one_l226_226521


namespace gcd_360_504_l226_226721

theorem gcd_360_504 : Nat.gcd 360 504 = 72 := by
  sorry

end gcd_360_504_l226_226721


namespace solution_set_of_inequality_l226_226456

variable {R : Type*} [LinearOrderedField R]

def odd_function (f : R → R) : Prop :=
  ∀ x : R, f (-x) = -f x

theorem solution_set_of_inequality
  (f : R → R)
  (odd_f : odd_function f)
  (h1 : f (-2) = 0)
  (h2 : ∀ (x1 x2 : R), x1 ≠ x2 ∧ 0 < x1 ∧ 0 < x2 → (x2 * f x1 - x1 * f x2) / (x1 - x2) < 0) :
  { x : R | (f x) / x < 0 } = { x : R | x < -2 } ∪ { x : R | x > 2 } := 
sorry

end solution_set_of_inequality_l226_226456


namespace part_I_part_II_l226_226019

def f (x a : ℝ) := |2 * x - a| + 5 * x

theorem part_I (x : ℝ) : f x 3 ≥ 5 * x + 1 ↔ (x ≤ 1 ∨ x ≥ 2) := sorry

theorem part_II (a x : ℝ) (h : (∀ x, f x a ≤ 0 ↔ x ≤ -1)) : a = 3 := sorry

end part_I_part_II_l226_226019


namespace work_completion_days_l226_226842

theorem work_completion_days
    (A : ℝ) (B : ℝ) (h1 : 1 / A + 1 / B = 1 / 10)
    (h2 : B = 35) :
    A = 14 :=
by
  sorry

end work_completion_days_l226_226842


namespace roll_2_four_times_last_not_2_l226_226688

def probability_of_rolling_2_four_times_last_not_2 : ℚ :=
  (1/6)^4 * (5/6)

theorem roll_2_four_times_last_not_2 :
  probability_of_rolling_2_four_times_last_not_2 = 5 / 7776 := 
by
  sorry

end roll_2_four_times_last_not_2_l226_226688


namespace correct_operation_l226_226169

theorem correct_operation : ∃ (a : ℝ), (3 + Real.sqrt 2 ≠ 3 * Real.sqrt 2) ∧ 
  ((a ^ 2) ^ 3 ≠ a ^ 5) ∧
  (Real.sqrt ((-7 : ℝ) ^ 2) ≠ -7) ∧
  (4 * a ^ 2 * a = 4 * a ^ 3) :=
by
  sorry

end correct_operation_l226_226169


namespace number_of_dogs_on_boat_l226_226390

theorem number_of_dogs_on_boat 
  (initial_sheep : ℕ) (initial_cows : ℕ) (initial_dogs : ℕ)
  (drowned_sheep : ℕ) (drowned_cows : ℕ)
  (made_it_to_shore : ℕ)
  (H1 : initial_sheep = 20)
  (H2 : initial_cows = 10)
  (H3 : drowned_sheep = 3)
  (H4 : drowned_cows = 2 * drowned_sheep)
  (H5 : made_it_to_shore = 35)
  : initial_dogs = 14 := 
sorry

end number_of_dogs_on_boat_l226_226390


namespace three_y_squared_value_l226_226614

theorem three_y_squared_value : ∃ x y : ℤ, 3 * x + y = 40 ∧ 2 * x - y = 20 ∧ 3 * y ^ 2 = 48 :=
by
  sorry

end three_y_squared_value_l226_226614


namespace problem_a_l226_226841

theorem problem_a (x a : ℝ) (h : (x + a) * (x + 2 * a) * (x + 3 * a) * (x + 4 * a) = 3 * a^4) :
  x = (-5 * a + a * Real.sqrt 37) / 2 ∨ x = (-5 * a - a * Real.sqrt 37) / 2 :=
by
  sorry

end problem_a_l226_226841


namespace sufficient_but_not_necessary_condition_l226_226477

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  ((∀ x : ℝ, (1 < x) → (x^2 - m * x + 1 > 0)) ↔ (-2 < m ∧ m < 2)) :=
sorry

end sufficient_but_not_necessary_condition_l226_226477


namespace weight_of_person_replaced_l226_226278

theorem weight_of_person_replaced (W_new : ℝ) (h1 : W_new = 74) (h2 : (W_new - W_old) = 9) : W_old = 65 := 
by
  sorry

end weight_of_person_replaced_l226_226278


namespace percentage_calculation_l226_226237

theorem percentage_calculation (P : ℕ) (h1 : 0.25 * 16 = 4) 
    (h2 : P / 100 * 40 = 6) : P = 15 :=
by 
    sorry

end percentage_calculation_l226_226237


namespace tony_initial_amount_l226_226263

-- Define the initial amount P
variable (P : ℝ)

-- Define the conditions
def initial_amount := P
def after_first_year := 1.20 * P
def after_half_taken := 0.60 * P
def after_second_year := 0.69 * P
def final_amount : ℝ := 690

-- State the theorem to prove
theorem tony_initial_amount : 
  (after_second_year P = final_amount) → (initial_amount P = 1000) :=
by 
  intro h
  sorry

end tony_initial_amount_l226_226263


namespace robert_time_to_complete_l226_226323

noncomputable def time_to_complete_semicircle_path (length_mile : ℝ) (width_feet : ℝ) (speed_mph : ℝ) (mile_to_feet : ℝ) : ℝ :=
  let diameter_mile := width_feet / mile_to_feet
  let radius_mile := diameter_mile / 2
  let circumference_mile := 2 * Real.pi * radius_mile
  let semicircle_length_mile := circumference_mile / 2
  semicircle_length_mile / speed_mph

theorem robert_time_to_complete :
  time_to_complete_semicircle_path 1 40 5 5280 = Real.pi / 10 :=
by
  sorry

end robert_time_to_complete_l226_226323


namespace average_salary_correct_l226_226592

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 15000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def number_of_people : ℕ := 5

def average_salary : ℕ := total_salary / number_of_people

theorem average_salary_correct : average_salary = 9000 := by
  -- proof is skipped
  sorry

end average_salary_correct_l226_226592


namespace area_of_rectangle_l226_226171

-- Define the problem conditions in Lean
def circle_radius := 7
def circle_diameter := 2 * circle_radius
def width_of_rectangle := circle_diameter
def length_to_width_ratio := 3
def length_of_rectangle := length_to_width_ratio * width_of_rectangle

-- Define the statement to be proved (area of the rectangle)
theorem area_of_rectangle : 
  (length_of_rectangle * width_of_rectangle) = 588 := by
  sorry

end area_of_rectangle_l226_226171


namespace symmetric_point_y_axis_l226_226089

theorem symmetric_point_y_axis (B : ℝ × ℝ) (hB : B = (-3, 4)) : 
  ∃ A : ℝ × ℝ, A = (3, 4) ∧ A.2 = B.2 ∧ A.1 = -B.1 :=
by
  use (3, 4)
  sorry

end symmetric_point_y_axis_l226_226089


namespace percentage_of_sikh_boys_l226_226578

-- Define the conditions
def total_boys : ℕ := 850
def percentage_muslim_boys : ℝ := 0.46
def percentage_hindu_boys : ℝ := 0.28
def boys_other_communities : ℕ := 136

-- Theorem to prove the percentage of Sikh boys is 10%
theorem percentage_of_sikh_boys : 
  (((total_boys - 
      (percentage_muslim_boys * total_boys + 
       percentage_hindu_boys * total_boys + 
       boys_other_communities))
    / total_boys) * 100 = 10) :=
by
  -- sorry prevents the need to provide proof here
  sorry

end percentage_of_sikh_boys_l226_226578


namespace find_dividend_l226_226600

-- Conditions
def quotient : ℕ := 4
def divisor : ℕ := 4

-- Dividend computation
def dividend (q d : ℕ) : ℕ := q * d

-- Theorem to prove
theorem find_dividend : dividend quotient divisor = 16 := 
by
  -- Placeholder for the proof, not needed as per instructions
  sorry

end find_dividend_l226_226600


namespace sum_of_terms_in_geometric_sequence_eq_fourteen_l226_226866

theorem sum_of_terms_in_geometric_sequence_eq_fourteen
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = r * a n)
  (h_a1 : a 1 = 1)
  (h_arith : 4 * a 2 = 2 * a 3 ∧ 2 * a 3 - 4 * a 2 = a 4 - 2 * a 3) :
  a 2 + a 3 + a 4 = 14 :=
sorry

end sum_of_terms_in_geometric_sequence_eq_fourteen_l226_226866


namespace num_ways_first_to_fourth_floor_l226_226536

theorem num_ways_first_to_fourth_floor (floors : ℕ) (staircases_per_floor : ℕ) 
  (H_floors : floors = 4) (H_staircases : staircases_per_floor = 2) : 
  (staircases_per_floor) ^ (floors - 1) = 2^3 := 
by 
  sorry

end num_ways_first_to_fourth_floor_l226_226536


namespace choose_4_out_of_10_l226_226389

theorem choose_4_out_of_10 : Nat.choose 10 4 = 210 := by
  sorry

end choose_4_out_of_10_l226_226389


namespace symmetric_point_correct_line_passes_second_quadrant_l226_226640

theorem symmetric_point_correct (x y: ℝ) (h_line : y = x + 1) :
  (x, y) = (-1, 2) :=
sorry

theorem line_passes_second_quadrant (m x y: ℝ) (h_line: m * x + y + m - 1 = 0) :
  (x, y) = (-1, 1) :=
sorry

end symmetric_point_correct_line_passes_second_quadrant_l226_226640


namespace ratio_of_money_l226_226565

-- Conditions
def amount_given := 14
def cost_of_gift := 28

-- Theorem statement to prove
theorem ratio_of_money (h1 : amount_given = 14) (h2 : cost_of_gift = 28) :
  amount_given / cost_of_gift = 1 / 2 := by
  sorry

end ratio_of_money_l226_226565


namespace length_BC_fraction_of_AD_l226_226544

-- Define variables and conditions
variables (x y : ℝ)
variable (h1 : 4 * x = 8 * y) -- given: length of AD from both sides
variable (h2 : 3 * x) -- AB = 3 * BD
variable (h3 : 7 * y) -- AC = 7 * CD

-- State the goal to prove
theorem length_BC_fraction_of_AD (x y : ℝ) (h1 : 4 * x = 8 * y) :
  (y / (4 * x)) = 1 / 8 := by
  sorry

end length_BC_fraction_of_AD_l226_226544


namespace mean_proportional_l226_226674

variable (a b c d : ℕ)
variable (x : ℕ)

def is_geometric_mean (a b : ℕ) (x : ℕ) := x = Int.sqrt (a * b)

theorem mean_proportional (h49 : a = 49) (h64 : b = 64) (h81 : d = 81)
  (h_geometric1 : x = 56) (h_geometric2 : c = 72) :
  c = 64 := sorry

end mean_proportional_l226_226674


namespace amount_received_by_a_l226_226844

namespace ProofProblem

/-- Total amount of money divided -/
def total_amount : ℕ := 600

/-- Ratio part for 'a' -/
def part_a : ℕ := 1

/-- Ratio part for 'b' -/
def part_b : ℕ := 2

/-- Total parts in the ratio -/
def total_parts : ℕ := part_a + part_b

/-- Amount per part when total is divided evenly by the total number of parts -/
def amount_per_part : ℕ := total_amount / total_parts

/-- Amount received by 'a' when total amount is divided according to the given ratio -/
def amount_a : ℕ := part_a * amount_per_part

theorem amount_received_by_a : amount_a = 200 := by
  -- Proof will be filled in here
  sorry

end ProofProblem

end amount_received_by_a_l226_226844


namespace mary_should_drink_six_glasses_per_day_l226_226635

def daily_water_goal : ℕ := 1500
def glass_capacity : ℕ := 250
def required_glasses (daily_goal : ℕ) (capacity : ℕ) : ℕ := daily_goal / capacity

theorem mary_should_drink_six_glasses_per_day :
  required_glasses daily_water_goal glass_capacity = 6 :=
by
  sorry

end mary_should_drink_six_glasses_per_day_l226_226635


namespace algebraic_expression_value_l226_226427

theorem algebraic_expression_value (a : ℝ) (h : 2 * a^2 + 3 * a - 5 = 0) : 6 * a^2 + 9 * a - 5 = 10 :=
by
  sorry

end algebraic_expression_value_l226_226427


namespace diff_of_squares_635_615_l226_226834

theorem diff_of_squares_635_615 : 635^2 - 615^2 = 25000 :=
by
  sorry

end diff_of_squares_635_615_l226_226834


namespace problem_2_l226_226451

noncomputable def f (x a : ℝ) : ℝ := (1 / 2) * x ^ 2 + a * Real.log (1 - x)

theorem problem_2 (a : ℝ) (x₁ x₂ : ℝ) (h₀ : 0 < a) (h₁ : a < 1/4) (h₂ : f x₂ a = 0) 
  (h₃ : f x₁ a = 0) (hx₁ : 0 < x₁) (hx₂ : x₁ < 1/2) (h₄ : x₁ < x₂) :
  f x₂ a - x₁ > - (3 + Real.log 4) / 8 := sorry

end problem_2_l226_226451


namespace sequence_v_20_l226_226622

noncomputable def sequence_v : ℕ → ℝ → ℝ
| 0, b => b
| (n + 1), b => - (2 / (sequence_v n b + 2))

theorem sequence_v_20 (b : ℝ) (hb : 0 < b) : sequence_v 20 b = -(2 / (b + 2)) :=
by
  sorry

end sequence_v_20_l226_226622


namespace partial_fraction_decomposition_l226_226611

noncomputable def polynomial := λ x: ℝ => x^3 - 24 * x^2 + 88 * x - 75

theorem partial_fraction_decomposition
  (p q r A B C : ℝ)
  (hpq : p ≠ q)
  (hpr : p ≠ r)
  (hqr : q ≠ r)
  (hroots : polynomial p = 0 ∧ polynomial q = 0 ∧ polynomial r = 0)
  (hdecomposition: ∀ s: ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
                      1 / polynomial s = A / (s - p) + B / (s - q) + C / (s - r)) :
  (1 / A + 1 / B + 1 / C = 256) := sorry

end partial_fraction_decomposition_l226_226611


namespace intersection_empty_implies_range_l226_226454

-- Define the sets A and B
def setA := {x : ℝ | x ≤ 1} ∪ {x : ℝ | x ≥ 3}
def setB (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ a + 1}

-- Prove that if A ∩ B = ∅, then 1 < a < 2
theorem intersection_empty_implies_range (a : ℝ) (h : setA ∩ setB a = ∅) : 1 < a ∧ a < 2 :=
by
  sorry

end intersection_empty_implies_range_l226_226454


namespace average_of_remaining_numbers_l226_226945

theorem average_of_remaining_numbers 
  (numbers : List ℝ)
  (h_len : numbers.length = 15)
  (h_avg : (numbers.sum / 15) = 100)
  (h_remove : [80, 90, 95] ⊆ numbers) :
  ((numbers.sum - 80 - 90 - 95) / 12) = (1235 / 12) :=
sorry

end average_of_remaining_numbers_l226_226945


namespace minimize_expression_l226_226775

theorem minimize_expression (x : ℝ) : 
  ∃ (m : ℝ), m = 2023 ∧ ∀ x, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ m :=
sorry

end minimize_expression_l226_226775


namespace eq_has_infinite_solutions_l226_226949

theorem eq_has_infinite_solutions (b : ℤ) :
  (∀ x : ℤ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 :=
by 
  sorry

end eq_has_infinite_solutions_l226_226949


namespace halfway_between_l226_226356

-- Definitions based on given conditions
def a : ℚ := 1 / 7
def b : ℚ := 1 / 9

-- Theorem that needs to be proved
theorem halfway_between (h : True) : (a + b) / 2 = 8 / 63 := by
  sorry

end halfway_between_l226_226356


namespace trees_occupy_area_l226_226662

theorem trees_occupy_area
  (length : ℕ) (width : ℕ) (number_of_trees : ℕ)
  (h_length : length = 1000)
  (h_width : width = 2000)
  (h_trees : number_of_trees = 100000) :
  (length * width) / number_of_trees = 20 := 
by
  sorry

end trees_occupy_area_l226_226662


namespace ellipse_condition_l226_226274

variables (m n : ℝ)

-- Definition of the curve
def curve_eqn (x y : ℝ) := m * x^2 + n * y^2 = 1

-- Define the condition for being an ellipse
def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

def mn_positive (m n : ℝ) : Prop := m * n > 0

-- Prove that mn > 0 is a necessary but not sufficient condition
theorem ellipse_condition (m n : ℝ) : mn_positive m n → is_ellipse m n → False := sorry

end ellipse_condition_l226_226274


namespace probability_of_different_topics_l226_226610

theorem probability_of_different_topics (n : ℕ) (m : ℕ) (prob : ℚ)
  (h1 : n = 36)
  (h2 : m = 30)
  (h3 : prob = 5/6) :
  (m : ℚ) / (n : ℚ) = prob :=
sorry

end probability_of_different_topics_l226_226610


namespace find_principal_amount_l226_226977

noncomputable def principal_amount (difference : ℝ) (rate : ℝ) : ℝ :=
  let ci := rate / 2
  let si := rate
  difference / (ci ^ 2 - 1 - si)

theorem find_principal_amount :
  principal_amount 4.25 0.10 = 1700 :=
by 
  sorry

end find_principal_amount_l226_226977


namespace carla_correct_questions_l226_226041

theorem carla_correct_questions :
  ∀ (Drew_correct Drew_wrong Carla_wrong Total_questions Carla_correct : ℕ), 
    Drew_correct = 20 →
    Drew_wrong = 6 →
    Carla_wrong = 2 * Drew_wrong →
    Total_questions = 52 →
    Carla_correct = Total_questions - Carla_wrong →
    Carla_correct = 40 :=
by
  intros Drew_correct Drew_wrong Carla_wrong Total_questions Carla_correct
  intros h1 h2 h3 h4 h5
  subst_vars
  sorry

end carla_correct_questions_l226_226041


namespace divides_five_iff_l226_226731

theorem divides_five_iff (a : ℤ) : (5 ∣ a^2) ↔ (5 ∣ a) := sorry

end divides_five_iff_l226_226731


namespace worker_y_defective_rate_l226_226452

noncomputable def y_f : ℚ := 0.1666666666666668
noncomputable def d_x : ℚ := 0.005 -- converting percentage to decimal
noncomputable def d_total : ℚ := 0.0055 -- converting percentage to decimal

theorem worker_y_defective_rate :
  ∃ d_y : ℚ, d_y = 0.008 ∧ d_total = ((1 - y_f) * d_x + y_f * d_y) :=
by
  sorry

end worker_y_defective_rate_l226_226452


namespace unique_function_l226_226316

def functional_equation (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f (-f x - f y) = 1 - x - y

theorem unique_function :
  ∀ f : ℤ → ℤ, (functional_equation f) → (∀ x : ℤ, f x = x - 1) :=
by
  intros f h
  sorry

end unique_function_l226_226316


namespace line_through_point_l226_226075

theorem line_through_point (k : ℝ) :
  (∃ k : ℝ, ∀ x y : ℝ, (x = 3) ∧ (y = -2) → (2 - 3 * k * x = -4 * y)) → k = -2/3 :=
by
  sorry

end line_through_point_l226_226075


namespace find_triples_l226_226881

theorem find_triples (a b c : ℝ) :
  a^2 + b^2 + c^2 = 1 ∧ a * (2 * b - 2 * a - c) ≥ 1/2 ↔ 
  (a = 1 / Real.sqrt 6 ∧ b = 2 / Real.sqrt 6 ∧ c = -1 / Real.sqrt 6) ∨
  (a = -1 / Real.sqrt 6 ∧ b = -2 / Real.sqrt 6 ∧ c = 1 / Real.sqrt 6) := 
by 
  sorry

end find_triples_l226_226881


namespace total_lotus_flowers_l226_226877

theorem total_lotus_flowers (x : ℕ) (h1 : x > 0) 
  (c1 : 3 ∣ x)
  (c2 : 5 ∣ x)
  (c3 : 6 ∣ x)
  (c4 : 4 ∣ x)
  (h_total : x = x / 3 + x / 5 + x / 6 + x / 4 + 6) : 
  x = 120 :=
by
  sorry

end total_lotus_flowers_l226_226877


namespace part1_part2_l226_226072

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 4 * x + a + 3
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := b * x + 5 - 2 * b

theorem part1 (a : ℝ) : (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f x a = 0) ↔ -8 ≤ a ∧ a ≤ 0 :=
sorry

theorem part2 (b : ℝ) : (∀ x1 : ℝ, 1 ≤ x1 ∧ x1 ≤ 4 → ∃ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ 4 ∧ f x2 3 = g x1 b) ↔ -1 ≤ b ∧ b ≤ 1/2 :=
sorry

end part1_part2_l226_226072


namespace steve_paid_18_l226_226428

-- Define the conditions
def mike_price : ℝ := 5
def steve_multiplier : ℝ := 2
def shipping_rate : ℝ := 0.8

-- Define Steve's cost calculation
def steve_total_cost : ℝ :=
  let steve_dvd_price := steve_multiplier * mike_price
  let shipping_cost := shipping_rate * steve_dvd_price
  steve_dvd_price + shipping_cost

-- Prove that Steve's total payment is 18.
theorem steve_paid_18 : steve_total_cost = 18 := by
  -- Provide a placeholder for the proof
  sorry

end steve_paid_18_l226_226428


namespace speed_of_stream_l226_226057

-- Definitions of the problem's conditions
def downstream_distance := 72
def upstream_distance := 30
def downstream_time := 3
def upstream_time := 3

-- The unknowns
variables (b s : ℝ)

-- The effective speed equations based on the problem conditions
def effective_speed_downstream := b + s
def effective_speed_upstream := b - s

-- The core conditions of the problem
def condition1 : Prop := downstream_distance = effective_speed_downstream * downstream_time
def condition2 : Prop := upstream_distance = effective_speed_upstream * upstream_time

-- The problem statement transformed into a Lean theorem
theorem speed_of_stream (h1 : condition1) (h2 : condition2) : s = 7 := 
sorry

end speed_of_stream_l226_226057


namespace prove_correct_statement_l226_226455

-- Define the conditions; we use the negation of incorrect statements
def condition1 (a b : ℝ) : Prop := a ≠ b → ¬((a - b > 0) → (a > 0 ∧ b > 0))
def condition2 (x : ℝ) : Prop := ¬(|x| > 0)
def condition4 (x : ℝ) : Prop := x ≠ 0 → (¬(∃ y, y = 1 / x))

-- Define the statement we want to prove as the correct one
def correct_statement (q : ℚ) : Prop := 0 - q = -q

-- The main theorem that combines conditions and proves the correct statement
theorem prove_correct_statement (a b : ℝ) (q : ℚ) :
  condition1 a b →
  condition2 a →
  condition4 a →
  correct_statement q :=
  by
  intros h1 h2 h4
  unfold correct_statement
  -- Proof goes here
  sorry

end prove_correct_statement_l226_226455


namespace triangle_inequality_l226_226257

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
by
  sorry

end triangle_inequality_l226_226257


namespace find_constant_a_l226_226923

theorem find_constant_a (x y a : ℝ) (h1 : (ax + 4 * y) / (x - 2 * y) = 13) (h2 : x / (2 * y) = 5 / 2) : a = 7 :=
sorry

end find_constant_a_l226_226923


namespace min_value_geq_9div2_l226_226962

noncomputable def min_value (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 12) : ℝ := 
  (x + y + z : ℝ) * ((1 : ℝ) / (x + y) + (1 : ℝ) / (x + z) + (1 : ℝ) / (y + z))

theorem min_value_geq_9div2 (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 12) :
  min_value x y z hx hy hz h_sum ≥ 9 / 2 := 
sorry

end min_value_geq_9div2_l226_226962


namespace fraction_of_gasoline_used_l226_226156

-- Define the conditions
def gasoline_per_mile := 1 / 30  -- Gallons per mile
def full_tank := 12  -- Gallons
def speed := 60  -- Miles per hour
def travel_time := 5  -- Hours

-- Total distance traveled
def distance := speed * travel_time  -- Miles

-- Gasoline used
def gasoline_used := distance * gasoline_per_mile  -- Gallons

-- Fraction of the full tank used
def fraction_used := gasoline_used / full_tank

-- The theorem to be proved
theorem fraction_of_gasoline_used :
  fraction_used = 5 / 6 :=
by sorry

end fraction_of_gasoline_used_l226_226156


namespace no_three_even_segments_with_odd_intersections_l226_226078

open Set

def is_even_length (s : Set ℝ) : Prop :=
  ∃ a b : ℝ, s = Icc a b ∧ (b - a) % 2 = 0

def is_odd_length (s : Set ℝ) : Prop :=
  ∃ a b : ℝ, s = Icc a b ∧ (b - a) % 2 = 1

theorem no_three_even_segments_with_odd_intersections :
  ¬ ∃ (S1 S2 S3 : Set ℝ),
    (is_even_length S1) ∧
    (is_even_length S2) ∧
    (is_even_length S3) ∧
    (is_odd_length (S1 ∩ S2)) ∧
    (is_odd_length (S1 ∩ S3)) ∧
    (is_odd_length (S2 ∩ S3)) :=
by
  -- Proof here
  sorry

end no_three_even_segments_with_odd_intersections_l226_226078


namespace time_for_embankments_l226_226736

theorem time_for_embankments (rate : ℚ) (t1 t2 : ℕ) (w1 w2 : ℕ)
    (h1 : w1 = 75) (h2 : w2 = 60) (h3 : t1 = 4)
    (h4 : rate = 1 / (w1 * t1 : ℚ)) 
    (h5 : t2 = 1 / (w2 * rate)) : 
    t1 + t2 = 9 :=
sorry

end time_for_embankments_l226_226736


namespace largest_binomial_coeff_and_rational_terms_l226_226551

theorem largest_binomial_coeff_and_rational_terms 
  (n : ℕ) 
  (h_sum_coeffs : 4^n - 2^n = 992) 
  (T : ℕ → ℝ → ℝ)
  (x : ℝ) :
  (∃ (r1 r2 : ℕ), T r1 x = 270 * x^(22/3) ∧ T r2 x = 90 * x^6)
  ∧
  (∃ (r3 r4 : ℕ), T r3 x = 243 * x^10 ∧ T r4 x = 90 * x^6)
:= 
  
sorry

end largest_binomial_coeff_and_rational_terms_l226_226551


namespace nine_questions_insufficient_l226_226193

/--
We have 5 stones with distinct weights and we are allowed to ask nine questions of the form
"Is it true that A < B < C?". Prove that nine such questions are insufficient to always determine
the unique ordering of these stones.
-/
theorem nine_questions_insufficient (stones : Fin 5 → Nat) 
  (distinct_weights : ∀ i j : Fin 5, i ≠ j → stones i ≠ stones j) :
  ¬ (∃ f : { q : Fin 125 | q.1 ≤ 8 } → (Fin 5 → Fin 5 → Fin 5 → Bool),
    ∀ w1 w2 w3 w4 w5 : Fin 120,
      (f ⟨0, sorry⟩) = sorry  -- This line only represents the existence of 9 questions
      )
:=
sorry

end nine_questions_insufficient_l226_226193


namespace parabola_vertex_l226_226201

theorem parabola_vertex :
  ∀ (x : ℝ), (∃ v : ℝ × ℝ, (v.1 = -1 ∧ v.2 = 4) ∧ ∀ (x : ℝ), (x^2 + 2*x + 5 = ((x + 1)^2 + 4))) :=
by
  sorry

end parabola_vertex_l226_226201


namespace correct_operation_l226_226566

/-- Proving that among the given mathematical operations, only the second option is correct. -/
theorem correct_operation (m : ℝ) : ¬ (m^3 - m^2 = m) ∧ (3 * m^2 * 2 * m^3 = 6 * m^5) ∧ ¬ (3 * m^2 + 2 * m^3 = 5 * m^5) ∧ ¬ ((2 * m^2)^3 = 8 * m^5) :=
by
  -- These are the conditions, proof is omitted using sorry
  sorry

end correct_operation_l226_226566


namespace tape_recorder_cost_l226_226303

theorem tape_recorder_cost (x y : ℕ) (h1 : 170 ≤ x * y) (h2 : x * y ≤ 195)
  (h3 : (y - 2) * (x + 1) = x * y) : x * y = 180 :=
by
  sorry

end tape_recorder_cost_l226_226303


namespace find_max_term_of_sequence_l226_226000

theorem find_max_term_of_sequence :
  ∃ m : ℕ, (m = 8) ∧ ∀ n : ℕ, (0 < n → n ≠ m → a_n = (n - 7) / (n - 5 * Real.sqrt 2)) :=
by
  sorry

end find_max_term_of_sequence_l226_226000


namespace consecutive_composites_l226_226295

theorem consecutive_composites 
  (a t d r : ℕ) (h_a_comp : ∃ p q, p > 1 ∧ q > 1 ∧ a = p * q)
  (h_t_comp : ∃ p q, p > 1 ∧ q > 1 ∧ t = p * q)
  (h_d_comp : ∃ p q, p > 1 ∧ q > 1 ∧ d = p * q)
  (h_r_pos : r > 0) :
  ∃ n : ℕ, n > 0 ∧ ∀ k : ℕ, k < r → ∃ m : ℕ, m > 1 ∧ m ∣ (a * t^(n + k) + d) :=
  sorry

end consecutive_composites_l226_226295


namespace sought_circle_equation_l226_226749

def circle_passing_through_point (D E F : ℝ) : Prop :=
  ∀ (x y : ℝ), (x = 0) → (y = 2) → x^2 + y^2 + D * x + E * y + F = 0

def chord_lies_on_line (D E F : ℝ) : Prop :=
  (D + 1) / 5 = (E - 2) / 2 ∧ (D + 1) / 5 = (F + 3)

theorem sought_circle_equation :
  ∃ (D E F : ℝ), 
  circle_passing_through_point D E F ∧ 
  chord_lies_on_line D E F ∧
  (D = -6) ∧ (E = 0) ∧ (F = -4) ∧ 
  ∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 6 * x - 4 = 0 :=
by
  sorry

end sought_circle_equation_l226_226749


namespace inequality_correct_l226_226463

variable (m n c : ℝ)

theorem inequality_correct (h : m > n) : m + c > n + c := 
by sorry

end inequality_correct_l226_226463


namespace paul_spent_81_90_l226_226115

-- Define the original price of each racket
def originalPrice : ℝ := 60

-- Define the discount rates
def firstDiscount : ℝ := 0.20
def secondDiscount : ℝ := 0.50

-- Define the sales tax rate
def salesTax : ℝ := 0.05

-- Define the prices after discount
def firstRacketPrice : ℝ := originalPrice * (1 - firstDiscount)
def secondRacketPrice : ℝ := originalPrice * (1 - secondDiscount)

-- Define the total price before tax
def totalPriceBeforeTax : ℝ := firstRacketPrice + secondRacketPrice

-- Define the total sales tax
def totalSalesTax : ℝ := totalPriceBeforeTax * salesTax

-- Define the total amount spent
def totalAmountSpent : ℝ := totalPriceBeforeTax + totalSalesTax

-- The statement to prove
theorem paul_spent_81_90 : totalAmountSpent = 81.90 := 
by
  sorry

end paul_spent_81_90_l226_226115


namespace ned_initially_had_games_l226_226107

variable (G : ℕ)

theorem ned_initially_had_games (h1 : (3 / 4) * (2 / 3) * G = 6) : G = 12 := by
  sorry

end ned_initially_had_games_l226_226107


namespace stickers_remaining_l226_226377

theorem stickers_remaining (total_stickers : ℕ) (front_page_stickers : ℕ) (other_pages_stickers : ℕ) (num_other_pages : ℕ) (remaining_stickers : ℕ)
  (h0 : total_stickers = 89)
  (h1 : front_page_stickers = 3)
  (h2 : other_pages_stickers = 7)
  (h3 : num_other_pages = 6)
  (h4 : remaining_stickers = total_stickers - (front_page_stickers + other_pages_stickers * num_other_pages)) :
  remaining_stickers = 44 :=
by
  sorry

end stickers_remaining_l226_226377


namespace memorable_numbers_count_l226_226433

def is_memorable_number (d : Fin 10 → Fin 8 → ℕ) : Prop :=
  d 0 0 = d 1 0 ∧ d 0 1 = d 1 1 ∧ d 0 2 = d 1 2 ∧ d 0 3 = d 1 3

theorem memorable_numbers_count : 
  ∃ n : ℕ, n = 10000 ∧ ∀ (d : Fin 10 → Fin 8 → ℕ), is_memorable_number d → n = 10000 :=
sorry

end memorable_numbers_count_l226_226433


namespace find_ordered_pair_l226_226381

theorem find_ordered_pair (x y : ℝ) 
  (h1 : x + y = (7 - x) + (7 - y)) 
  (h2 : x - y = (x - 2) + (y - 2)) : 
  (x = 5 ∧ y = 2) :=
by
  sorry

end find_ordered_pair_l226_226381


namespace weekly_milk_production_l226_226422

-- Conditions
def number_of_cows : ℕ := 52
def milk_per_cow_per_day : ℕ := 1000
def days_in_week : ℕ := 7

-- Statement to prove
theorem weekly_milk_production : (number_of_cows * milk_per_cow_per_day * days_in_week) = 364000 := by
  sorry

end weekly_milk_production_l226_226422


namespace sum_of_digits_T_l226_226167

-- Conditions:
def horse_lap_times := [1, 2, 3, 4, 5, 6, 7, 8]
def S := 840
def total_horses := 8
def min_horses_at_start := 4

-- Question:
def T := 12 -- Least time such that at least 4 horses meet

/-- Prove that the sum of the digits of T is 3 -/
theorem sum_of_digits_T : (1 + 2) = 3 := by
  sorry

end sum_of_digits_T_l226_226167
