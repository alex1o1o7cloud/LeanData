import Mathlib

namespace correct_propositions_count_l214_214079

-- Define the propositions
def proposition1 (line : Type) (plane : Type) [determinate_space line plane] : Prop :=
  ∀ (a b c : line), (a ⊥ b ∧ b ∈ plane) ∧ (a ⊥ c ∧ c ∈ plane) ∧ (b ∥ c) → a ⊥ plane

def proposition2 (line : Type) (plane : Type) [determinate_space line plane] : Prop :=
  ∀ (a b : line), (a ∈ plane ∧ (∀ l ∈ plane, ∠ a l = ∠ b l)) → b ⊥ plane

def proposition3 (line : Type) (plane : Type) [determinate_space line plane] : Prop :=
  ∀ (a b c : line) (point : Type), (a ∈ plane ∧ (∀ x ∈ plane, project line plane x = point)) → a ⊥ plane

-- Define the total number of correct propositions
def count_correct_propositions (line : Type) (plane : Type) [determinate_space line plane] : ℕ :=
  [proposition1 line plane, proposition2 line plane, proposition3 line plane].filter (λ p, p).length

-- The main theorem statement
theorem correct_propositions_count (line : Type) (plane : Type) [determinate_space line plane] :
  count_correct_propositions line plane = 2 :=
by
  sorry

end correct_propositions_count_l214_214079


namespace initial_income_l214_214713

theorem initial_income (A : ℕ) (average_after_death : ℕ) (income_deceased : ℕ) 
                       (total_remaining_income : ℕ) (number_initial_earners : ℕ) 
                       (number_remaining_earners : ℕ) :
  number_initial_earners = 4 →
  average_after_death = 650 →
  income_deceased = 1178 →
  number_remaining_earners = 3 →
  total_remaining_income = number_remaining_earners * average_after_death →
  (number_initial_earners * A - income_deceased = total_remaining_income) →
  A = 782 :=
begin
  intros,
  sorry
end

end initial_income_l214_214713


namespace cube_vertices_on_sphere_surface_area_l214_214261

theorem cube_vertices_on_sphere_surface_area (a : ℝ) (h : a > 0) :
  let cube_surface_area := 6 * a^2 in
  let d := a * Real.sqrt 3 in
  let r := d / 2 in
  let sphere_surface_area := 4 * Real.pi * r^2 in
  sphere_surface_area = 3 * Real.pi * a^2 :=
by
  sorry

end cube_vertices_on_sphere_surface_area_l214_214261


namespace smallest_positive_period_triangle_area_l214_214836

-- Definitions
def f (x : ℝ) : ℝ := 2 * (sqrt 3) * sin x * cos x - 2 * (cos x) ^ 2 + 1
def g (x : ℝ) : ℝ := f (x + (π / 3))

-- Main theorems to prove
theorem smallest_positive_period : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π := sorry

theorem triangle_area (A B C : ℝ) (a b c : ℝ) (h1 : a = 2) (h2 : b + c = 4) 
    (h3 : g (A / 2) = 1) : ∃ S, S = sqrt 3 := sorry

end smallest_positive_period_triangle_area_l214_214836


namespace Susan_has_10_dollars_left_l214_214975

def initial_amount : ℝ := 80
def food_expense : ℝ := 15
def rides_expense : ℝ := 3 * food_expense
def games_expense : ℝ := 10
def total_expense : ℝ := food_expense + rides_expense + games_expense
def remaining_amount : ℝ := initial_amount - total_expense

theorem Susan_has_10_dollars_left : remaining_amount = 10 := by
  sorry

end Susan_has_10_dollars_left_l214_214975


namespace rainfall_mondays_l214_214593

theorem rainfall_mondays
  (M : ℕ)
  (rain_monday : ℝ)
  (rain_tuesday : ℝ)
  (num_tuesdays : ℕ)
  (extra_rain_tuesdays : ℝ)
  (h1 : rain_monday = 1.5)
  (h2 : rain_tuesday = 2.5)
  (h3 : num_tuesdays = 9)
  (h4 : num_tuesdays * rain_tuesday = rain_monday * M + extra_rain_tuesdays)
  (h5 : extra_rain_tuesdays = 12) :
  M = 7 := 
sorry

end rainfall_mondays_l214_214593


namespace option_D_correct_l214_214696

def A := { x : ℕ | 1 ≤ x ∧ x < 6 }

theorem option_D_correct : 3.5 ∉ A := 
by
  sorry

end option_D_correct_l214_214696


namespace base_conversion_l214_214513

theorem base_conversion (k : ℕ) (h : 26 = 3*k + 2) : k = 8 := 
by 
  sorry

end base_conversion_l214_214513


namespace bicycles_in_garage_l214_214664

theorem bicycles_in_garage :
  ∀ (wheels_in_garage cars lawnmower tricycle unicycle bicycles : ℕ),
    cars = 2 →
    lawnmower = 1 →
    tricycle = 1 →
    unicycle = 1 →
    wheels_in_garage = 22 →
    bicycles * 2 + cars * 4 + lawnmower * 4 + tricycle * 3 + unicycle * 1 = wheels_in_garage →
    bicycles = 3 :=
by
  intros wheels_in_garage cars lawnmower tricycle unicycle bicycles
  assume h1 : cars = 2
  assume h2 : lawnmower = 1
  assume h3 : tricycle = 1
  assume h4 : unicycle = 1
  assume h5 : wheels_in_garage = 22
  assume h6 : bicycles * 2 + cars * 4 + lawnmower * 4 + tricycle * 3 + unicycle * 1 = wheels_in_garage
  sorry

end bicycles_in_garage_l214_214664


namespace ellipse_properties_l214_214985

-- Define the structure and properties of an ellipse
structure Ellipse :=
  (a b : ℝ)
  (equation : ∀ (x y : ℝ), (x^2)/(a^2) + (y^2)/(b^2) = 1)

-- The given problem ellipse
def problem_ellipse : Ellipse :=
  { a := 2,
    b := 1,
    equation := λ x y, (x^2)/(2^2) + (y^2)/(1^2) = 1 }

-- The formal statement for the eccentricity and length of the focal axis
theorem ellipse_properties (e : Ellipse) (h : e = problem_ellipse) :
  let c := sqrt (e.a^2 - e.b^2) in
  let eccentricity := c / e.a in
  let focal_axis_length := 2 * c in
  eccentricity = (sqrt 3 / 2) ∧ focal_axis_length = 2 * sqrt 3 :=
by
  sorry

end ellipse_properties_l214_214985


namespace correct_choice_l214_214163

noncomputable def N_star := {x : ℕ // x > 0}

def A : set ℕ := {x | ∃ (m : N_star), x = 3 * m.val}
def B : set ℕ := {x | ∃ (m : N_star), x = 3 * m.val - 1}
def C : set ℕ := {x | ∃ (m : N_star), x = 3 * m.val - 2}

theorem correct_choice (a b c : ℕ) (ha : a ∈ A) (hb : b ∈ B) (hc : c ∈ C) : 
  2006 = a + b * c := 
sorry

end correct_choice_l214_214163


namespace solve_for_y_l214_214099

theorem solve_for_y 
  (x y : ℝ) 
  (h1 : 2 * x - 3 * y = 9) 
  (h2 : x + y = 8) : 
  y = 1.4 := 
sorry

end solve_for_y_l214_214099


namespace probability_median_three_is_half_l214_214145

noncomputable def probability_median_three (s : Finset ℕ) (a : ℕ) (n : ℕ) :=
  let combinations := s.powerset.filter (λ t, t.card = n) in
  let favorable := combinations.filter (λ t, (t.sort (· ≤ ·))[1] = a) in
  (favorable.card : ℚ) / (combinations.card : ℚ)

theorem probability_median_three_is_half :
  let s := {1, 2, 3, 6}.to_finset in
  probability_median_three s 3 3 = 1 / 2 :=
by
  sorry

end probability_median_three_is_half_l214_214145


namespace no_such_continuous_function_l214_214603

theorem no_such_continuous_function :
  ¬ ∃ (f : ℝ → ℝ), continuous f ∧ (∀ x, (∃ r, f x = r ∧ rat r) ↔ (∃ irr, f (x + 1) = irr ∧ irrational irr)) :=
by
  -- Insert the proof here, for now it's a placeholder
  sorry

end no_such_continuous_function_l214_214603


namespace difference_between_mean_and_median_l214_214532

def student_scores := [{percentage := 0.20, score := 60},
                       {percentage := 0.25, score := 75},
                       {percentage := 0.20, score := 85},
                       {percentage := 0.25, score := 95},
                       {percentage := 0.10, score := 100}]

def mean_score (scores : List {percentage : ℝ, score : ℕ}) : ℝ :=
  scores.foldl (λ acc s, acc + s.percentage * s.score) 0

def median_score (scores : List {percentage : ℝ, score : ℕ}) : ℕ :=
  let sorted_scores := scores.sortBy (λ s, s.score)
  let cumulative_percentages := sorted_scores.scanl (λ acc s, acc + s.percentage) 0
  let median_index := cumulative_percentages.indexWhere (λ p, p >= 0.5)
  sorted_scores.get! median_index score

theorem difference_between_mean_and_median 
  (difference : ℝ) : 
  let m := mean_score student_scores
        let med := median_score student_scores
        m - med = difference :=
sorry

end difference_between_mean_and_median_l214_214532


namespace smaller_square_area_percent_l214_214312

theorem smaller_square_area_percent (s L : ℝ) (r : ℝ)
  (hL : L = 4)
  (hR : r = 2 * Real.sqrt 2)
  (hSmaller : s = -Real.sqrt 2 / 2)
  : (4 * (s ^ 2) / (L ^ 2)) * 100 = 12.5 := by
  -- conditions
  have h1 : L = 4 := hL
  have h2 : L ^ 2 = 16 := by rw [hL]; norm_num
  have h3 : 4 * ((s ^ 2) / 16) = (4 * (s ^ 2)) / 16 := by rw [mul_div]
  have h4 : 4 * (s ^ 2) / 16 = (s ^ 2) / 4 := by rw [mul_div_cancel_left, mul_comm 4]; norm_num
  have h5 : s = -Real.sqrt 2 / 2 := hSmaller
  have h6 : (s ^ 2) = 2 / 4 := by rw [←h5, pow_two, div_pow, Real.sqrt_sq]; norm_num
  rw [h4, h6]
  norm_num
  sorry

end smaller_square_area_percent_l214_214312


namespace lcm_gcd_product_difference_l214_214782
open Nat

theorem lcm_gcd_product_difference :
  (Nat.lcm 12 9) * (Nat.gcd 12 9) - (Nat.gcd 15 9) = 105 :=
by
  sorry

end lcm_gcd_product_difference_l214_214782


namespace uma_income_is_20000_l214_214644

/-- Given that the ratio of the incomes of Uma and Bala is 4 : 3, 
the ratio of their expenditures is 3 : 2, and both save $5000 at the end of the year, 
prove that Uma's income is $20000. -/
def uma_bala_income : Prop :=
  ∃ (x y : ℕ), (4 * x - 3 * y = 5000) ∧ (3 * x - 2 * y = 5000) ∧ (4 * x = 20000)
  
theorem uma_income_is_20000 : uma_bala_income :=
  sorry

end uma_income_is_20000_l214_214644


namespace min_cubes_for_hollow_block_l214_214615

/-
Susan's block dimensions:
length = 3 cubes,
width = 9 cubes,
depth = 5 cubes.
Volume of each cube = 8 cubic cm.
We aim to prove that the minimum number of cubes needed to make a hollow block is 122 cubes.
-/

theorem min_cubes_for_hollow_block :
  ∃ (minCubes : ℕ), minCubes = 122 ∧
  ∀ (length width depth : ℕ), length = 3 → width = 9 → depth = 5 →
  let top_bottom_cubes := 2 * (length * width),
      side_cubes := 2 * ((width * depth - width * 2) + (length * depth - length * 2 - 2)) in
  minCubes = top_bottom_cubes + side_cubes :=
by
  sorry

end min_cubes_for_hollow_block_l214_214615


namespace distribution_plans_l214_214708

theorem distribution_plans (officials : Fin 6) (towns : Fin 3)
  (A B : Fin 6) 
  (hA : A ≠ 0)
  (hB : B ≠ 1)
  (h_diff : A ≠ B) :
  ∃ (plans : ℕ), plans = 78 :=
by
  let number_of_plans := 78
  use number_of_plans
  sorry

end distribution_plans_l214_214708


namespace triangle_non_existent_l214_214843

theorem triangle_non_existent (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
    (tangent_condition : (c^2) = 2 * (a^2) + 2 * (b^2)) : False := by
  sorry

end triangle_non_existent_l214_214843


namespace sum_of_integers_d_with_exactly_six_solutions_l214_214628

-- Define the function g(x)
def g (x : ℝ) : ℝ :=
  ((x - 6) * (x - 4) * (x - 2) * x * (x + 2) * (x + 4) * (x + 6)) / 945 - 2.5

-- State the theorem
theorem sum_of_integers_d_with_exactly_six_solutions :
  (∀ d : ℤ, (∃ s : ℕ, (∀ x : ℝ, g x = (d : ℝ) → (∃! x, g x = d)) → s = 6) →
  d ∈ {-2, -3}) → ( -2 + -3 = -5 ) :=
  by sorry

end sum_of_integers_d_with_exactly_six_solutions_l214_214628


namespace average_blinks_in_normal_conditions_l214_214207

theorem average_blinks_in_normal_conditions (blink_gaming : ℕ) (k : ℚ) (blink_normal : ℚ) 
  (h_blink_gaming : blink_gaming = 10)
  (h_k : k = (3 / 5))
  (h_condition : blink_gaming = blink_normal - k * blink_normal) : 
  blink_normal = 25 := 
by 
  sorry

end average_blinks_in_normal_conditions_l214_214207


namespace label_points_l214_214265

-- Assumptions: We are given n points in space.
-- Condition: Any three points form a triangle with at least one interior angle greater than 120 degrees.

theorem label_points (n : ℕ) (points : Fin n → ℝ × ℝ):
  (∀ i j k : Fin n, i ≠ j → j ≠ k → k ≠ i →
    ∃ (T : Triangle (points i) (points j) (points k)),
      (T.angle (points i) (points j) (points k)).degrees > 120) →
  ∃ (f : Fin n → Fin n), (∀ i j k : Fin n, 1 ≤ i ∧ i < j ∧ j < k ∧ k ≤ n →
    let A_i := points (f i)
    let A_j := points (f j)
    let A_k := points (f k)
    in ∃ (T' : Triangle A_i A_j A_k),
        (T'.angle A_i A_j A_k).degrees > 120) :=
sorry

end label_points_l214_214265


namespace evaluate_expression_l214_214000

theorem evaluate_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -3) :
  2 * a^2 - 3 * b^2 + 4 * a * b = -43 :=
by
  sorry

end evaluate_expression_l214_214000


namespace least_repeating_block_of_8_over_11_l214_214623

theorem least_repeating_block_of_8_over_11 : (∃ n : ℕ, (∀ m : ℕ, m < n → ¬(∃ a b : ℤ, (10^m - 1) * (8 * 10^n - b * 11 * 10^(n - t)) = a * 11 * 10^(m - t))) ∧ n ≤ 2) :=
by
  sorry

end least_repeating_block_of_8_over_11_l214_214623


namespace find_range_of_k_l214_214049

noncomputable def range_of_k (a : ℝ) : Set ℝ :=
  {k | ∃ x, log a (x - a * k) = log a (x^2 - a^2)}

theorem find_range_of_k (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) : 
  range_of_k a = {k | k ∈ set.Ioo 0 1 ∪ set.Iio (-1)} :=
sorry

end find_range_of_k_l214_214049


namespace emily_curtains_purchase_l214_214369

theorem emily_curtains_purchase 
    (c : ℕ) 
    (curtain_cost : ℕ := 30)
    (print_count : ℕ := 9)
    (print_cost_per_unit : ℕ := 15)
    (installation_cost : ℕ := 50)
    (total_cost : ℕ := 245) :
    (curtain_cost * c + print_count * print_cost_per_unit + installation_cost = total_cost) → c = 2 :=
by
  sorry

end emily_curtains_purchase_l214_214369


namespace arithmetic_progression_sum_at_least_66_l214_214872

-- Define the sum of the first n terms of an arithmetic progression
def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

-- Define the conditions for the arithmetic progression
def arithmetic_prog_conditions (a1 d : ℤ) (n : ℕ) :=
  sum_first_n_terms a1 d n ≥ 66

-- The main theorem to prove
theorem arithmetic_progression_sum_at_least_66 (n : ℕ) :
  (n >= 3 ∧ n <= 14) → arithmetic_prog_conditions 25 (-3) n :=
by
  sorry

end arithmetic_progression_sum_at_least_66_l214_214872


namespace exists_five_digit_number_l214_214780

theorem exists_five_digit_number :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧
           (n / 10000 % 10 ≠ 5) ∧
           (n % 10 ≠ 2) ∧
           (∀ i j : ℕ, i ≠ j → (n / 10^i % 10) ≠ (n / 10^j % 10)) :=
by
  use 21840
  repeat { try { simp }; apply sorry; sorry }


end exists_five_digit_number_l214_214780


namespace marching_band_members_l214_214633

open Int

def band_members (x : ℤ) :=
  x > 150 ∧ x < 250 ∧
  x % 3 = 1 ∧
  x % 6 = 2 ∧
  x % 8 = 3

theorem marching_band_members : ∃ x : ℤ, band_members x ∧ x = 203 :=
by {
  use 203,
  unfold band_members,
  split,
  { exact dec_trivial,
    split,
    { exact dec_trivial,
      split,
      { rw [Int.mod_eq_of_lt (by decide) dec_trivial], exact dec_trivial,
        split,
        { rw [Int.mod_eq_of_lt (by decide) dec_trivial], exact dec_trivial,
          rw [Int.mod_eq_of_lt (by decide) dec_trivial], exact dec_trivial,
          sorry

end marching_band_members_l214_214633


namespace anya_can_obtain_any_composite_number_l214_214635

theorem anya_can_obtain_any_composite_number (n : ℕ) (h : ∃ k, k > 1 ∧ k < n ∧ n % k = 0) : ∃ m ≥ 4, ∀ k, k > 1 → k < m → m % k = 0 → m = n :=
by
  sorry

end anya_can_obtain_any_composite_number_l214_214635


namespace coeff_x3_expansion_l214_214902

theorem coeff_x3_expansion : 
  let f := (3 * x - 2) * (x - 1) ^ 6 in
  polynomial.coeff f 3 = 85 :=
by sorry

end coeff_x3_expansion_l214_214902


namespace hexagon_diagonals_perpendicular_l214_214052

variables {A B C D E F : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
variables {AB BC CD DE EF FA : ℝ}
variables (angle_A angle_C : ℝ)
variables {F D : A} {B E : B}

theorem hexagon_diagonals_perpendicular
  (h1 : AB = BC)
  (h2 : CD = DE)
  (h3 : EF = FA)
  (h4 : angle_A = 90)
  (h5 : angle_C = 90) :
  (∠(F, D) ⊥ ∠(B, E)) :=
sorry

end hexagon_diagonals_perpendicular_l214_214052


namespace no_real_solutions_for_exponential_eq_l214_214784

theorem no_real_solutions_for_exponential_eq :
  ¬ ∃ x : ℝ, 2 ^ (x^2 - 3*x - 2) = 8 ^ (x - 4) :=
sorry

end no_real_solutions_for_exponential_eq_l214_214784


namespace max_distance_of_PC_l214_214540

noncomputable def circle_center : ℝ × ℝ := (0, 1)
noncomputable def radius : ℝ := 2
def is_chord_of_circle (A B : ℝ × ℝ) : Prop := (dist A circle_center ≤ radius) ∧ (dist B circle_center ≤ radius)
def perpendicular (p1 p2 : ℝ × ℝ) := p1.1 * p2.1 + p1.2 * p2.2 = 0

theorem max_distance_of_PC (A B P : ℝ × ℝ) :
    is_chord_of_circle A B →
    (∃ P, equilateral_triangle A B P) → 
    perpendicular (circle_center - P) (B - A) →
    dist P circle_center ≤ 4 :=
by sorry

end max_distance_of_PC_l214_214540


namespace evaluate_expression_at_2_l214_214372

theorem evaluate_expression_at_2: 
  let x := 2 in
  (2 * x ^ 2 - 3 * x + 4) = 6 :=
by
  let x := 2
  have : 2 * x ^ 2 - 3 * x + 4 = 6 := sorry
  exact this

end evaluate_expression_at_2_l214_214372


namespace percentages_equivalence_l214_214706

noncomputable def mass_moisture : ℝ := 1400
noncomputable def mass_difference : ℝ := 300

def divided_substance_percentage (m_divided : ℝ) : ℝ :=
  (mass_moisture / m_divided) * 100

def undivided_substance_percentage (m_divided : ℝ) : ℝ :=
  (mass_moisture / (m_divided + mass_difference)) * 100

theorem percentages_equivalence (m_divided : ℝ) : 
  divided_substance_percentage m_divided = undivided_substance_percentage m_divided + 105 :=
sorry

example : 
  divided_substance_percentage 500 = 280 ∧ undivided_substance_percentage 500 = 175 :=
by
  have h1 : divided_substance_percentage 500 = 280 := sorry
  have h2 : undivided_substance_percentage (500 : ℝ) = 175 := sorry
  exact ⟨h1, h2⟩

end percentages_equivalence_l214_214706


namespace sum_of_digits_of_nine_ab_l214_214905

noncomputable def a : ℕ := 10^1986 - 1
noncomputable def b : ℕ := (4 * (10^1986 - 1)) / 9
noncomputable def nine_ab : ℕ := 9 * a * b

-- Define a function to compute the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.toString.foldl (λ acc c => acc + (c.toNat - '0'.toNat)) 0

theorem sum_of_digits_of_nine_ab : sum_of_digits nine_ab = 15880 :=
  sorry

end sum_of_digits_of_nine_ab_l214_214905


namespace min_value_of_expression_l214_214512

variable {a b : ℝ}

theorem min_value_of_expression 
  (ha : a > 0)
  (hb : b > 0)
  (h_eq : 1 / a + 2 / b = 1) :
  ∃ (a b : ℝ), (a > 0 ∧ b > 0 ∧ 1 / a + 2 / b = 1) ∧ (∀ (x y : ℝ), x > 0 → y > 0 → 1 / x + 2 / y = 1 → (2 / (x - 1) + 1 / (y - 2)) ≥ 2) :=
begin
  sorry
end

end min_value_of_expression_l214_214512


namespace number_of_terms_in_arithmetic_sequence_l214_214360

theorem number_of_terms_in_arithmetic_sequence :
  ∃ n : ℕ, ∀ aₙ : ℤ, aₙ = 13 + (n - 1) * 6 → aₙ = 127 → n = 20 :=
begin
  sorry
end

end number_of_terms_in_arithmetic_sequence_l214_214360


namespace edge_lengths_angle_between_lines_CD1_and_AC1_radius_of_sphere_l214_214533

noncomputable def lengths_of_edges (d : ℝ) : (ℝ × ℝ × ℝ) :=
   (d * Real.sqrt 2, d * (Real.sqrt 2 + 1), d * (Real.sqrt 2 + 2))

theorem edge_lengths (d : ℝ) : let l := lengths_of_edges d in
    l.1 = d * Real.sqrt 2 ∧
    l.2 = d * (Real.sqrt 2 + 1) ∧
    l.3 = d * (Real.sqrt 2 + 2) := sorry

-- Angle between CD1 and AC1
noncomputable def angle_between_lines (d : ℝ) : ℝ :=
  Real.arccos ((-2 * (1 + Real.sqrt 2)) / (Real.sqrt 79 + 52 * Real.sqrt 2))

theorem angle_between_lines_CD1_and_AC1 (d : ℝ) : 
  angle_between_lines d = Real.arccos ((-2 * (1 + Real.sqrt 2)) / (Real.sqrt 79 + 52 * Real.sqrt 2)) := sorry

-- Radius of the spheres
noncomputable def radius_R (d : ℝ) : ℝ :=
  d * (3 + 3 * Real.sqrt 2 - Real.sqrt (5 + 6 * Real.sqrt 2)) / 4

theorem radius_of_sphere (d : ℝ) : 
  let R := radius_R d in
    2 * R = sqrt ((d * (Real.sqrt 2) + d - 2 * R)^2 
                  + (d * (Real.sqrt 2) + 2 * d - 2 * R)^2 
                  + (d * Real.sqrt 2 - 2 * R)^2) := sorry

end edge_lengths_angle_between_lines_CD1_and_AC1_radius_of_sphere_l214_214533


namespace small_boxes_in_large_box_l214_214311

def number_of_chocolate_bars_in_small_box := 25
def total_number_of_chocolate_bars := 375

theorem small_boxes_in_large_box : total_number_of_chocolate_bars / number_of_chocolate_bars_in_small_box = 15 := by
  sorry

end small_boxes_in_large_box_l214_214311


namespace calculate_magnitude_of_sum_l214_214853

variable (x : ℝ)

def vector_a := (1, 2)
def vector_b := (x, -2)

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem calculate_magnitude_of_sum :
  (vector_a (vector_b x) → (dot_product vector_a (vector_b x)) = -3) →
  magnitude (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2) = 2 :=
sorry

end calculate_magnitude_of_sum_l214_214853


namespace five_ab4_is_perfect_square_l214_214515

theorem five_ab4_is_perfect_square (a b : ℕ) (h : 5000 ≤ 5000 + 100 * a + 10 * b + 4 ∧ 5000 + 100 * a + 10 * b + 4 ≤ 5999) :
    ∃ n, n^2 = 5000 + 100 * a + 10 * b + 4 → a + b = 9 :=
by
  sorry

end five_ab4_is_perfect_square_l214_214515


namespace recurring_fraction_division_l214_214013

noncomputable def recurring_833 := 5 / 6
noncomputable def recurring_1666 := 5 / 3

theorem recurring_fraction_division : 
  (recurring_833 / recurring_1666) = 1 / 2 := 
by 
  sorry

end recurring_fraction_division_l214_214013


namespace select_3_males_2_females_select_at_least_1_captain_select_at_least_1_female_select_both_captain_and_female_l214_214133

variable (n m : ℕ) -- n for males, m for females
variable (mc fc : ℕ) -- mc for male captain, fc for female captain

def num_ways_3_males_2_females : ℕ :=
  (Nat.choose 6 3) * (Nat.choose 4 2)

def num_ways_at_least_1_captain : ℕ :=
  (2 * (Nat.choose 8 4)) + (Nat.choose 8 3)

def num_ways_at_least_1_female : ℕ :=
  (Nat.choose 10 5) - (Nat.choose 6 5)

def num_ways_both_captain_and_female : ℕ :=
  (Nat.choose 10 5) - (Nat.choose 8 5) - (Nat.choose 5 4)

theorem select_3_males_2_females : num_ways_3_males_2_females = 120 := by
  sorry
  
theorem select_at_least_1_captain : num_ways_at_least_1_captain = 196 := by
  sorry
  
theorem select_at_least_1_female : num_ways_at_least_1_female = 246 := by
  sorry
  
theorem select_both_captain_and_female : num_ways_both_captain_and_female = 191 := by
  sorry

end select_3_males_2_females_select_at_least_1_captain_select_at_least_1_female_select_both_captain_and_female_l214_214133


namespace minimum_square_area_l214_214271

-- Definitions of the given conditions
structure Rectangle where
  width : ℕ
  height : ℕ

def rect1 : Rectangle := { width := 2, height := 4 }
def rect2 : Rectangle := { width := 3, height := 5 }
def circle_diameter : ℕ := 3

-- Statement of the theorem
theorem minimum_square_area :
  ∃ sq_side : ℕ, 
    (sq_side ≥ 5 ∧ sq_side ≥ 7) ∧ 
    sq_side * sq_side = 49 := 
by
  use 7
  have h1 : 7 ≥ 5 := by norm_num
  have h2 : 7 ≥ 7 := by norm_num
  have h3 : 7 * 7 = 49 := by norm_num
  exact ⟨⟨h1, h2⟩, h3⟩

end minimum_square_area_l214_214271


namespace b_n_plus_1_inequality_l214_214943

noncomputable def f (x : ℝ) (a : Fin n → ℝ) : ℝ :=
  ∑ i, a i * x^i

theorem b_n_plus_1_inequality (a : Fin n → ℝ) (h : ∀ i, 0 ≤ a i ∧ a i ≤ a 0) :
  let f1 := ∑ i, a i in
  ∃ b : Fin (2 * n + 1) → ℝ, b (n + 1) ≤ 1 / 2 * f 1 f1 := sorry

end b_n_plus_1_inequality_l214_214943


namespace trips_Jean_l214_214343

theorem trips_Jean (x : ℕ) (h1 : x + (x + 6) = 40) : x + 6 = 23 := by
  sorry

end trips_Jean_l214_214343


namespace unique_not_in_range_of_g_l214_214354

-- Given constants a, b, c, d with a, c ≠ 0
variables {a b c d : ℝ}
axiom a_ne_zero : a ≠ 0
axiom c_ne_zero : c ≠ 0

-- Function definition
def g (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

-- Given conditions
axiom g_23 : g 23 = 23
axiom g_101 : g 101 = 101
axiom g_involution : ∀ (x : ℝ), x ≠ -d / c → g (g x) = x

-- The unique number not in the range of g
theorem unique_not_in_range_of_g : 62 ∉ (set.range g) :=
sorry

end unique_not_in_range_of_g_l214_214354


namespace sqrt_mul_simplify_l214_214222

theorem sqrt_mul_simplify : Real.sqrt 18 * Real.sqrt 32 = 24 := 
sorry

end sqrt_mul_simplify_l214_214222


namespace center_of_circle_constructible_l214_214276

noncomputable def find_center_of_circle (ruler_width : ℝ) (circle_radius : ℝ) (circle_center : Euclidean.Space ℝ 2) :=
  ∃ (P : Euclidean.Space ℝ 2) (Q : Euclidean.Space ℝ 2) 
    (P' : Euclidean.Space ℝ 2) (Q' : Euclidean.Space ℝ 2),
    circle_radius > ruler_width / 2 ∧
    let D := 2 * circle_radius in
    let A := circle_center + (D / 2) in
    let B := circle_center - (D / 2) in
    let C := circle_center + (D / 4) in
    let D := circle_center - (D / 4) in
    -- Parallel Chords
    let AB := line_through A B in
    let CD := line_through C D in
    parallel AB CD ∧
    -- Intersections
    let P := intersection (line_through A C) (line_through B D) in
    let Q := intersection (line_through A D) (line_through B C) in
    -- Line PQ
    line_through P Q ∈ circle_center ∧
    -- Repeat for P'Q'
    let P' := intersection (line_through A' C') (line_through B' D') in
    let Q' := intersection (line_through A' D') (line_through B' C') in
    line_through P' Q' ∈ circle_center ∧
    -- Center Finding
    intersection (line_through P Q) (line_through P' Q') = circle_center

theorem center_of_circle_constructible (ruler_width : ℝ) (circle_radius : ℝ) (circle_center : Euclidean.Space ℝ 2)
  (h : find_center_of_circle ruler_width circle_radius circle_center) :
  ∃ O : Euclidean.Space ℝ 2, O = circle_center := sorry

end center_of_circle_constructible_l214_214276


namespace PetesOriginalNumber_l214_214200

-- Define the context and problem
theorem PetesOriginalNumber (x : ℤ) (h : 3 * (2 * x + 12) = 90) : x = 9 :=
by
  -- proof goes here
  sorry

end PetesOriginalNumber_l214_214200


namespace f_2022_l214_214986

variable {R : Type*} [LinearOrder R]

def f (x : R) : R := sorry

axiom f_eq (a b : R) : f((a + 2 * b) / 3) = (f(a) + 2 * f(b)) / 3
axiom f_one : f(1) = 5
axiom f_four : f(4) = 2

theorem f_2022 : f(2022) = -2016 := sorry

end f_2022_l214_214986


namespace average_age_of_community_l214_214527

theorem average_age_of_community 
  (k : ℕ) 
  (hwomen : ℕ := 7 * k)
  (hmen : ℕ := 5 * k)
  (avg_age_women : ℝ := 30)
  (avg_age_men : ℝ := 35)
  (total_women_age : ℝ := avg_age_women * hwomen)
  (total_men_age : ℝ := avg_age_men * hmen)
  (total_population : ℕ := hwomen + hmen)
  (total_age : ℝ := total_women_age + total_men_age) : 
  total_age / total_population = 32 + 1 / 12 :=
by
  sorry

end average_age_of_community_l214_214527


namespace exists_ε_for_prob_l214_214251

noncomputable def ε : ℝ := 0.02

theorem exists_ε_for_prob : 
  ∃ ε > 0, 
  (∀ (X : ℕ) (n : ℕ) (p : ℝ), 
    n = 900 ∧ p = 0.5 ∧ 
    (1 / (real.sqrt (n * p * (1 - p))) * (X - n * p)) ∈ 
    (set.Icc (-ε) ε)) → 
  (prob (λ (X : ℕ) (n : ℕ) (p : ℝ), 
    (1 / (real.sqrt (n * p * (1 - p))) * (X - n * p)) ∈ 
    (set.Icc (-ε) ε) ∧ n = 900 ∧ p = 0.5) 
  = 0.77) :=
by
  use 0.02
  sorry

end exists_ε_for_prob_l214_214251


namespace largest_K_inequality_l214_214937

theorem largest_K_inequality :
  ∃ K : ℕ, (K < 12) ∧ (10 * K = 110) := by
  use 11
  sorry

end largest_K_inequality_l214_214937


namespace median_mode_l214_214128

def scores := [129, 130, 130, 130, 132, 132, 135, 135, 137, 137]

noncomputable def mode := 130
noncomputable def median := (scores.nth 4).getD 0 -- Lean indices start from 0, so (nth 4) and (nth 5) represent the 5th and 6th elements.

theorem median_mode :
  list.median scores = 132 ∧ list.mode scores = 130 :=
  sorry

end median_mode_l214_214128


namespace DE_eq_AE_l214_214170

theorem DE_eq_AE
  (A B C D E H O : Point)
  (triangle_ABC : Triangle A B C)
  (is_circumcenter : Circumcenter O triangle_ABC)
  (is_orthocenter : Orthocenter H triangle_ABC)
  (D_on_AB : OnLine D A B)
  (AD_eq_AH : AD = AH)
  (E_on_AC : OnLine E A C)
  (AE_eq_AO : AE = AO) :
  DE = AE := 
sorry

end DE_eq_AE_l214_214170


namespace pat_page_numbering_l214_214199

theorem pat_page_numbering (total_twos : ℕ) (initial_page : ℕ) (last_page : ℕ) :
  total_twos = 50 ∧ initial_page = 1 ∧ last_page = 299 → 
  (∀ n, (initial_page ≤ n ∧ n ≤ last_page) → 
   let count_digits := String.mk (n.repr.to_list.filter (λ c, c = '2')).length
   in ∑ i in Finset.range (last_page + 1), String.mk ((i + 1).repr.to_list.filter (λ c, c = '2')).length ≤ total_twos) :=
by
  sorry

end pat_page_numbering_l214_214199


namespace sqrt_mul_simplify_l214_214219

theorem sqrt_mul_simplify : Real.sqrt 18 * Real.sqrt 32 = 24 := 
sorry

end sqrt_mul_simplify_l214_214219


namespace probability_XOXOXOX_l214_214417

theorem probability_XOXOXOX (X O : ℕ) (h1 : X = 4) (h2 : O = 3) :
  let total_ways := Nat.choose (X + O) X,
      favorable_outcomes := 1 in
  (favorable_outcomes / total_ways : ℚ) = 1 / 35 := by
  sorry

end probability_XOXOXOX_l214_214417


namespace solution_set_f_pos_l214_214168

open Set Function

variables (f : ℝ → ℝ)
variables (h_even : ∀ x : ℝ, f (-x) = f x)
variables (h_diff : ∀ x ≠ 0, DifferentiableAt ℝ f x)
variables (h_pos : ∀ x : ℝ, x > 0 → f x + x * (f' x) > 0)
variables (h_at_2 : f 2 = 0)

theorem solution_set_f_pos :
  {x : ℝ | f x > 0} = (Iio (-2)) ∪ (Ioi 2) :=
by 
  sorry

end solution_set_f_pos_l214_214168


namespace probability_XOXOXOX_is_one_over_thirty_five_l214_214440

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_arrangement : ℤ :=
  let total_arrangements := binomial_coefficient 7 4
  let favorable_outcomes := 1
  favorable_outcomes / total_arrangements

theorem probability_XOXOXOX_is_one_over_thirty_five :
  probability_of_arrangement = (1 : ℤ) / 35 := 
  by
  sorry

end probability_XOXOXOX_is_one_over_thirty_five_l214_214440


namespace billing_error_l214_214700

theorem billing_error (x y : ℕ) (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) 
    (h : 100 * y + x - (100 * x + y) = 2970) : y - x = 30 ∧ 10 ≤ x ∧ x ≤ 69 ∧ 40 ≤ y ∧ y ≤ 99 := 
by
  sorry

end billing_error_l214_214700


namespace range_of_a_l214_214090

def A (a : ℝ) : set ℝ := {x | (x-1)*(a-x) > 0}
def B : set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

theorem range_of_a (a : ℝ) : (ℝ \ A a) ∪ B = set.univ ↔ (-1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l214_214090


namespace even_theta_range_of_y_l214_214841

-- Part I
theorem even_theta (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x) (θ : ℝ)
  (θ_in : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (even_f : ∀ x, f (x + θ) = f (- (x + θ))) :
  θ = Real.pi / 2 ∨ θ = 3 * Real.pi / 2 :=
sorry

-- Part II
theorem range_of_y (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x) :
  let y := (λ x, (f (x + Real.pi / 12))^2 + (f (x + Real.pi / 4))^2) in
  ∀ x, 1 - (Real.sqrt 3) / 2 ≤ y x ∧ y x ≤ 1 + (Real.sqrt 3) / 2 :=
sorry

end even_theta_range_of_y_l214_214841


namespace framing_feet_required_l214_214699

-- Definitions of the given conditions
def original_width := 4 -- inches
def original_height := 6 -- inches
def enlargement_factor := 4
def border_width := 3 -- inches
def inches_per_foot := 12

-- Enlargement dimensions
def enlarged_width := original_width * enlargement_factor
def enlarged_height := original_height * enlargement_factor

-- Dimensions with border
def final_width := enlarged_width + 2 * border_width
def final_height := enlarged_height + 2 * border_width

-- Perimeter of the final framed picture
def perimeter := 2 * (final_width + final_height)

-- Minimum number of feet of framing required
def minimum_linear_feet : Nat := (perimeter / inches_per_foot).ceil.toNat

theorem framing_feet_required : minimum_linear_feet = 9 := by
  -- To be proven
  sorry

end framing_feet_required_l214_214699


namespace find_range_of_k_l214_214050

noncomputable def range_of_k (a : ℝ) : Set ℝ :=
  {k | ∃ x, log a (x - a * k) = log a (x^2 - a^2)}

theorem find_range_of_k (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) : 
  range_of_k a = {k | k ∈ set.Ioo 0 1 ∪ set.Iio (-1)} :=
sorry

end find_range_of_k_l214_214050


namespace solve_equation_l214_214970

theorem solve_equation :
  ∀ x : ℝ, 18 / (x^2 - 9) - 3 / (x - 3) = 2 ↔ (x = 4.5 ∨ x = -3) :=
by
  sorry

end solve_equation_l214_214970


namespace problem_part1_problem_part2_problem_part3_problem_part3_gp_l214_214456

-- Definitions of the given geometric progression and related sequences
def a (n : ℕ) := 2011 * (-1/2)^n
def s (n : ℕ) := (2 / 3) * 2011 * (1 - (-1/2)^n)
def prod (n : ℕ) := List.prod (List.map a (List.range (n + 1)))
def d (n : ℕ) := a (n + 2) - a (n + 1)

-- Proof statements as hypotheses and goals
theorem problem_part1 (n : ℕ) : 2 ≤ n → (s 2 ≤ s n ∧ s n ≤ s 1) :=
by sorry

theorem problem_part2 (n : ℕ) (hn : ∀ m, |prod m| ≤ |prod 12|) : |prod n| = |prod 12| :=
by sorry

theorem problem_part3 : ∀ n, ∃ k, d n = 2011 * (3 / (2^(n+1))) :=
by sorry

theorem problem_part3_gp : ∀ n, d (n + 1) = (1/2) * d n :=
by sorry

end problem_part1_problem_part2_problem_part3_problem_part3_gp_l214_214456


namespace max_factors_x12_minus_1_l214_214995

theorem max_factors_x12_minus_1 :
  ∃ (m : ℕ), (∀ (f : ℕ → Polynomial ℝ), (x^12 - 1 = ∏ i in finset.range m, (f i)) → 
              (∀ (i : ℕ), i < m → ¬(f i).degree = (0 : ℕ)) → m = 6) :=
sorry

end max_factors_x12_minus_1_l214_214995


namespace consecutive_integers_satisfy_inequality_l214_214206

theorem consecutive_integers_satisfy_inequality :
  ∀ (n m : ℝ), n + 1 = m ∧ n < Real.sqrt 26 ∧ Real.sqrt 26 < m → m + n = 11 :=
by
  sorry

end consecutive_integers_satisfy_inequality_l214_214206


namespace sum_inverse_sequence_l214_214816

def sequence (a : ℕ → ℕ) : Prop := 
  a 1 = 1 ∧ ∀ m n, a (m + n) = a m + a n + m * n

theorem sum_inverse_sequence (a : ℕ → ℕ) (h : sequence a) :
  ∑ i in (finset.range 2017).map (finset.nat_cast_embedding ℕ), (1 / a i) = 2017 / 1009 := 
sorry

end sum_inverse_sequence_l214_214816


namespace find_t_l214_214464

variable (a b : ℝ × ℝ)
variable (c : ℝ × ℝ)
variable (t : ℝ)

axiom vector_a : a = (1, 2)
axiom vector_b : b = (3, 4)
axiom vector_c : c = (t, t + 2)
axiom perpendicular : (c.1 * (a.1 + b.1) + c.2 * (a.2 + b.2)) = 0

theorem find_t : t = -6 / 5 :=
by 
  rw [vector_a, vector_b, vector_c] at perpendicular
  sorry

end find_t_l214_214464


namespace distinct_partial_sums_inequality_l214_214634

theorem distinct_partial_sums_inequality (n : ℕ) (x : ℕ → ℕ) (h : ∀ s t, s ≠ t → (∑ i in s, x i) ≠ (∑ i in t, x i)) :
  (∑ i in finset.range n, x i * x i) ≥ (4^n - 1) / 3 := 
sorry

end distinct_partial_sums_inequality_l214_214634


namespace convex_pentagons_from_15_points_l214_214022

theorem convex_pentagons_from_15_points : (Nat.choose 15 5) = 3003 := 
by
  sorry

end convex_pentagons_from_15_points_l214_214022


namespace subset_A_implies_range_of_m_l214_214648

-- Definitions of the sets A and B
def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 5 }
def B (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2m - 1 }

-- Statement of the theorem
theorem subset_A_implies_range_of_m (m : ℝ) (h : B m ⊆ A) : m ∈ Iic 3 := by
  sorry

end subset_A_implies_range_of_m_l214_214648


namespace total_remaining_candle_life_l214_214560

theorem total_remaining_candle_life:
  let bedroom_candles := 20
      living_room_candles := bedroom_candles / 2
      hallway_candles := 20
      study_room_candles := bedroom_candles + living_room_candles + 5
      bedroom_remaining_life := bedroom_candles * 0.6
      living_room_remaining_life := living_room_candles * 0.8
      hallway_remaining_life := hallway_candles * 0.5
      study_room_remaining_life := study_room_candles * 0.7
      total_remaining_life := bedroom_remaining_life + living_room_remaining_life + hallway_remaining_life + study_room_remaining_life
  in total_remaining_life = 54.5 :=
by {
  -- Proof omitted
  sorry
}

end total_remaining_candle_life_l214_214560


namespace distance_between_parallel_lines_l214_214237

theorem distance_between_parallel_lines :
  ∀ (a b c₁ c₂ : ℝ), (a ≠ 0 ∨ b ≠ 0) → 
  abs (c₁ - c₂) / real.sqrt (a^2 + b^2) = 2 → 
  (3 * x - 4 * y - 5 = 0) ∧ (3 * x - 4 * y + 5 = 0) → 2 :=
begin
  intros a b c₁ c₂ hab dist_eq,
  sorry
end

end distance_between_parallel_lines_l214_214237


namespace nth_equation_l214_214194

theorem nth_equation (n : ℕ) (h : n > 0) : 9 * (n - 1) + n = 10 * (n - 1) + 1 :=
by
  sorry

end nth_equation_l214_214194


namespace factor_correct_l214_214377

noncomputable def factor_fraction (a b c : ℝ) : ℝ :=
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3)

theorem factor_correct (a b c : ℝ) : 
  factor_fraction a b c = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end factor_correct_l214_214377


namespace sin_add_eq_implies_y_zero_l214_214764

theorem sin_add_eq_implies_y_zero (y : ℝ) (h : y ∈ set.Icc 0 (real.pi / 2)) :
  (∀ x ∈ set.Icc 0 (real.pi / 2), real.sin (x + y) = real.sin x + real.sin y) → y = 0 :=
by
  sorry

end sin_add_eq_implies_y_zero_l214_214764


namespace find_f_2022_l214_214989

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_2022 :
  (∀ a b : ℝ, f((a + 2 * b) / 3) = (f(a) + 2 * f(b)) / 3) →
  f(1) = 5 →
  f(4) = 2 →
  f(2022) = -2016 := by
  sorry

end find_f_2022_l214_214989


namespace function_x_squared_belongs_to_set_M_l214_214491

-- Definitions based on the given conditions
def belongs_to_set_M (f : ℝ → ℝ) : Prop :=
  ∃ (a k : ℝ), k ≠ 0 ∧ ∀ x : ℝ, f(a + x) = k * f(a - x)

-- Specific function given in the problem
def f (x : ℝ) := x^2

-- The proof statement: we need to show that the function f belongs to set M
theorem function_x_squared_belongs_to_set_M : belongs_to_set_M f :=
sorry

end function_x_squared_belongs_to_set_M_l214_214491


namespace sum_of_irreducible_fractions_is_integer_iff_same_denominator_l214_214966

theorem sum_of_irreducible_fractions_is_integer_iff_same_denominator
  (a b c d A : ℤ) (h_irred1 : Int.gcd a b = 1) (h_irred2 : Int.gcd c d = 1) (h_sum : (a : ℚ) / b + (c : ℚ) / d = A) :
  b = d := 
by
  sorry

end sum_of_irreducible_fractions_is_integer_iff_same_denominator_l214_214966


namespace find_m_value_l214_214476

noncomputable def hyperbola_m_value (m : ℝ) : Prop :=
  let a := 1
  let b := 2 * a
  m = -(1/4)

theorem find_m_value :
  (∀ x y : ℝ, x^2 + m * y^2 = 1 → b = 2 * a) → hyperbola_m_value m :=
by
  intro h
  sorry

end find_m_value_l214_214476


namespace concurrence_T_concurrence_S_l214_214936

noncomputable theory

variables {n : ℕ} (A : fin n → ℝ × ℝ) (X : fin n → ℝ × ℝ)
variables (Γ : ℝ × ℝ × ℝ) -- Circle Γ with center and radius
variables (ω : fin n → ℝ × ℝ × ℝ) (Ω : fin n → ℝ × ℝ × ℝ)
variables (T : fin n → ℝ × ℝ) (S : fin n → ℝ × ℝ)
variables (common_point_T : ℝ × ℝ) (common_point_S : ℝ × ℝ)

-- Assuming the points A form a regular n-gon
def regular_ngon (A : fin n → ℝ × ℝ) : Prop := sorry

-- Xi is the intersection of the lines (A_{i-2} A_{i-1}) and (A_{i} A_{i+1})
def intersection_X (A : fin n → ℝ × ℝ) (X : fin n → ℝ × ℝ) : Prop := sorry

-- Points X_i are all inside circle Γ
def inside_circle_X (X : fin n → ℝ × ℝ) (Γ : ℝ × ℝ × ℝ) : Prop := sorry

-- Circle ω_i tangent to the rays [A_{i-1}X_{i}) and [A_{i}X_{i}) beyond X_{i} and internally tangent to Γ at T_{i}
def circle_tangent_ω (A : fin n → ℝ × ℝ) (X : fin n → ℝ × ℝ) (ω : fin n → ℝ × ℝ × ℝ) (T : fin n → ℝ × ℝ) : Prop := sorry

-- Circle Ω_i tangent to the rays [A_{i-1}X_{i}) and [A_{i}X_{i}) beyond X_{i} and externally tangent to Γ at S_{i}
def circle_tangent_Ω (A : fin n → ℝ × ℝ) (X : fin n → ℝ × ℝ) (Ω : fin n → ℝ × ℝ × ℝ) (S : fin n → ℝ × ℝ) : Prop := sorry

-- The first part: The n lines (X_{i} T_{i}) are concurrent
theorem concurrence_T (h1: regular_ngon A) (h2: intersection_X A X) (h3: inside_circle_X X Γ)
  (h4: circle_tangent_ω A X ω T): ∃ (P : ℝ × ℝ), ∀ (i : fin n), ∃ (λx: ℝ), P = X i + λx • T i := sorry

-- The second part: The n lines (X_{i} S_{i}) are concurrent
theorem concurrence_S (h1: regular_ngon A) (h2: intersection_X A X) (h3: inside_circle_X X Γ)
  (h4: circle_tangent_Ω A X Ω S): ∃ (P : ℝ × ℝ), ∀ (i : fin n), ∃ (λx: ℝ), P = X i + λx • S i := sorry

end concurrence_T_concurrence_S_l214_214936


namespace trigonometric_identity_l214_214103

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = -2) :
  Real.sin (π / 2 + α) * Real.cos (π + α) = -1 / 5 :=
by
  -- The proof will be skipped but the statement should be correct.
  sorry

end trigonometric_identity_l214_214103


namespace expected_value_of_8_sided_die_l214_214332

noncomputable def expected_value_of_winnings : ℝ := 
  let multiples_of_three := [3, 6]
  let probability_of_multiple_of_three := 2/8
  let expected_winnings := (probability_of_multiple_of_three * (3 + 6 : ℝ)) / 2
  expected_winnings

theorem expected_value_of_8_sided_die : expected_value_of_winnings = 2.25 := 
  by
    sorry

end expected_value_of_8_sided_die_l214_214332


namespace cos_identity_example_l214_214102

theorem cos_identity_example (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = 3 / 5) : Real.cos (Real.pi / 3 - α) = 3 / 5 := by
  sorry

end cos_identity_example_l214_214102


namespace number_of_even_integers_between_3000_and_6000_with_different_digits_l214_214857

def thousands_digit (n : ℕ) : ℕ := n / 1000
def units_digit (n : ℕ) : ℕ := n % 10

def valid_digits (n : ℕ) : Prop :=
  let d1 := thousands_digit n
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := units_digit n
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 ∧
  d1 ∈ {3, 4, 5} ∧ d4 % 2 = 0

noncomputable def even_integers_between_3000_and_6000_with_different_digits : ℕ :=
  (Finset.range 6000).filter (λ n => 3000 ≤ n ∧ n < 6000 ∧ valid_digits n).card

theorem number_of_even_integers_between_3000_and_6000_with_different_digits :
  even_integers_between_3000_and_6000_with_different_digits = 728 :=
by
  sorry

end number_of_even_integers_between_3000_and_6000_with_different_digits_l214_214857


namespace hyperbola_eccentricity_range_l214_214789

theorem hyperbola_eccentricity_range (a : ℝ) (h : a > 1) :
  ∃ e : ℝ, e = real.sqrt (1 + 1 / a^2) ∧ 1 < e ∧ e < real.sqrt 2 :=
by {
  -- Definitions and assumptions
  let e := real.sqrt (1 + 1 / a^2),
  -- Placeholder for the proof
  use e,
  sorry
}

end hyperbola_eccentricity_range_l214_214789


namespace parabola_focus_and_line_l214_214475

theorem parabola_focus_and_line 
  (F : ℝ × ℝ) (A B M P : ℝ × ℝ) (l : ℝ → ℝ) 
  (focus_parabola : ∀ y, (y, 2 * sqrt y))
  (line_through_F : l F.1 = F.2)
  (intersect_parabola_A : F.1 = (A.2)^2 / 4)
  (intersect_parabola_B : F.1 = (B.2)^2 / 4)
  (midpoint_M : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) 
  (vertical_through_M: ∃ x y, P = (M.1, y))
  (PF_distance : dist P F = 3 / 2) : 
  l(x) = sqrt 2 * x - sqrt 2 := 
sorry

end parabola_focus_and_line_l214_214475


namespace fred_dimes_l214_214805

-- Define the conditions as constants
constant bank_balance : ℕ := 90
constant dime_value : ℕ := 10

-- State the theorem to be proven
theorem fred_dimes : bank_balance / dime_value = 9 := by
  sorry

end fred_dimes_l214_214805


namespace quadratic_sum_of_b_and_c_l214_214253

theorem quadratic_sum_of_b_and_c :
  ∃ b c : ℝ, (∀ x : ℝ, x^2 - 20 * x + 36 = (x + b)^2 + c) ∧ b + c = -74 :=
by
  sorry

end quadratic_sum_of_b_and_c_l214_214253


namespace planes_perpendicular_l214_214067

noncomputable def u : ℝ × ℝ × ℝ := (-2, 2, 5)
noncomputable def v : ℝ × ℝ × ℝ := (6, -4, 4)

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

def planes_are_perpendicular (a b : ℝ × ℝ × ℝ) : Prop :=
  dot_product a b = 0

theorem planes_perpendicular : planes_are_perpendicular u v :=
  by
    -- Calculation for dot_product (-2, 2, 5) (6, -4, 4):
    -- (-2 * 6) + (2 * -4) + (5 * 4) = 0
    have h1 : dot_product u v = -2 * 6 + 2 * -4 + 5 * 4 := rfl
    have h2 : -2 * 6 + 2 * -4 + 5 * 4 = 0 := by norm_num
    rw [h1, h2]
    exact rfl

end planes_perpendicular_l214_214067


namespace items_left_in_cart_l214_214330

-- Define the initial items in the shopping cart
def initial_items : ℕ := 18

-- Define the items deleted from the shopping cart
def deleted_items : ℕ := 10

-- Theorem statement: Prove the remaining items are 8
theorem items_left_in_cart : initial_items - deleted_items = 8 :=
by
  -- Sorry marks the place where the proof would go.
  sorry

end items_left_in_cart_l214_214330


namespace Buffy_whiskers_l214_214798

def whiskers_Juniper : ℕ := 12
def whiskers_Puffy : ℕ := 3 * whiskers_Juniper
def whiskers_Scruffy : ℕ := 2 * whiskers_Puffy
def whiskers_Buffy : ℕ := (whiskers_Puffy + whiskers_Scruffy + whiskers_Juniper) / 3

theorem Buffy_whiskers : whiskers_Buffy = 40 := by
  sorry

end Buffy_whiskers_l214_214798


namespace correct_substitution_l214_214296

theorem correct_substitution (x : ℝ) : 
    (2 * x - 7)^2 + (5 * x - 17.5)^2 = 0 → 
    x = 7 / 2 :=
by
  sorry

end correct_substitution_l214_214296


namespace find_x_l214_214659

theorem find_x (x : ℝ) :
  (x * 13.26 + x * 9.43 + x * 77.31 = 470) → (x = 4.7) :=
by
  sorry

end find_x_l214_214659


namespace find_m_l214_214906

variable {V : Type} [AddGroup V]

theorem find_m 
  {A B C M : V}
  (h1 : M + B + C = (0 : V))
  (h2 : A + B + 0 * M = 0) :
  m = -3 := by
  sorry

end find_m_l214_214906


namespace even_integers_3000_6000_with_four_different_digits_l214_214859

def is_valid_number (n : ℕ) : Prop :=
  3000 ≤ n ∧ n < 6000 ∧ (n % 2 = 0) ∧ (∃ d1 d2 d3 d4 : ℕ, 
    n = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧
    d2 ≠ d3 ∧ d2 ≠ d4 ∧
    d3 ≠ d4)

theorem even_integers_3000_6000_with_four_different_digits : 
  ∃ count : ℕ, count = 784 ∧ 
  count = (∑ n in (finset.range 6000).filter is_valid_number, 1) :=
by
  sorry

end even_integers_3000_6000_with_four_different_digits_l214_214859


namespace measure_angle_A_l214_214123

open Real

def triangle_area (a b c S : ℝ) (A B C : ℝ) : Prop :=
  S = (1 / 2) * b * c * sin A

def sides_and_angles (a b c A B C : ℝ) : Prop :=
  A = 2 * B

theorem measure_angle_A (a b c S A B C : ℝ)
  (h1 : triangle_area a b c S A B C)
  (h2 : sides_and_angles a b c A B C)
  (h3 : S = (a ^ 2) / 4) :
  A = π / 2 ∨ A = π / 4 :=
  sorry

end measure_angle_A_l214_214123


namespace three_numbers_sum_at_least_five_l214_214173

theorem three_numbers_sum_at_least_five 
  (x : Fin 9 → ℝ) 
  (nonneg : ∀ i, 0 ≤ x i) 
  (sum_squares : ∑ i, (x i)^2 ≥ 25) : 
  ∃ i j k : Fin 9, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ (x i + x j + x k ≥ 5) :=
begin
  sorry
end

end three_numbers_sum_at_least_five_l214_214173


namespace farmer_rows_of_tomatoes_l214_214714

def num_rows (total_tomatoes yield_per_plant plants_per_row : ℕ) : ℕ :=
  (total_tomatoes / yield_per_plant) / plants_per_row

theorem farmer_rows_of_tomatoes (total_tomatoes yield_per_plant plants_per_row : ℕ)
    (ht : total_tomatoes = 6000)
    (hy : yield_per_plant = 20)
    (hp : plants_per_row = 10) :
    num_rows total_tomatoes yield_per_plant plants_per_row = 30 := 
by
  sorry

end farmer_rows_of_tomatoes_l214_214714


namespace initial_number_of_chairs_l214_214308

variable (x : ℕ)

-- Conditions
def condition1 := x - (x - 3) = 12

-- Required proof
theorem initial_number_of_chairs (h : condition1) : x = 15 := by
  sorry

end initial_number_of_chairs_l214_214308


namespace cyclist_avg_speed_l214_214329

theorem cyclist_avg_speed (d : ℝ) (h1 : d > 0) :
  let t_1 := d / 17
  let t_2 := d / 23
  let total_time := t_1 + t_2
  let total_distance := 2 * d
  (total_distance / total_time) = 19.55 :=
by
  -- Proof steps here
  sorry

end cyclist_avg_speed_l214_214329


namespace solve_equation_l214_214386

theorem solve_equation :
  ∀ x : ℝ,
  (1 / (x^2 + 12 * x - 9) + 
   1 / (x^2 + 3 * x - 9) + 
   1 / (x^2 - 12 * x - 9) = 0) ↔ 
  (x = 1 ∨ x = -9 ∨ x = 3 ∨ x = -3) := 
by
  sorry

end solve_equation_l214_214386


namespace percent_motorists_no_ticket_l214_214195

theorem percent_motorists_no_ticket (M : ℝ) :
  (0.14285714285714285 * M - 0.10 * M) / (0.14285714285714285 * M) * 100 = 30 :=
by
  sorry

end percent_motorists_no_ticket_l214_214195


namespace abs_eq_4_l214_214958

theorem abs_eq_4 (a : ℝ) : |a| = 4 ↔ a = 4 ∨ a = -4 :=
by
  sorry

end abs_eq_4_l214_214958


namespace solve_system_l214_214225

theorem solve_system :
  ∃ x y z : ℤ, (5732 * x + 2134 * y + 2134 * z = 7866) ∧
              (2134 * x + 5732 * y + 2134 * z = 670) ∧
              (2134 * x + 2134 * y + 5732 * z = 11464) ∧
              x = 1 ∧ y = -1 ∧ z = 2 :=
by
  use 1, -1, 2
  simp
  split ; { ring_nf } ; sorry

end solve_system_l214_214225


namespace find_a8_l214_214089

theorem find_a8 (a : ℕ → ℝ) (h1 : ∀ n : ℕ, n > 0 → a (n + 1) / (n + 1) = a n / n) (h2 : a 5 = 15) : a 8 = 24 :=
sorry

end find_a8_l214_214089


namespace sqrt_18_mul_sqrt_32_eq_24_l214_214214
  
theorem sqrt_18_mul_sqrt_32_eq_24 : (Real.sqrt 18 * Real.sqrt 32 = 24) :=
  sorry

end sqrt_18_mul_sqrt_32_eq_24_l214_214214


namespace problem_cos_function_l214_214362

def f (x : ℝ) : ℝ := Real.cos x

theorem problem_cos_function :
  (∀ x, f(π + x) = -f(x)) ∧ (∀ x, f(-x) = f(x)) :=
by
  -- placeholder for proof
  sorry

end problem_cos_function_l214_214362


namespace m_eq_3_n_eq_2_median_x_eq_75_mode_y_eq_70_estimate_excellent_students_B_l214_214528

/-
1. Given the scores for Class B, proving m = 3 and n = 2.
2. Given the scores for Class A, proving the median x = 75.
3. Given the scores for Class B, proving the mode y = 70.
4. Estimating that 20 students in Class B have excellent physical fitness.
-/
def ClassA_scores := [65, 75, 75, 80, 60, 50, 75, 90, 85, 65]
def ClassB_scores := [90, 55, 80, 70, 55, 70, 95, 80, 65, 70]

def counts_in_ranges_A : List ℕ := [1, 3, 3, 2, 1]

-- Proof problems
theorem m_eq_3 : (List.countInRange ClassB_scores (70, 80) = 3) := 
sorry

theorem n_eq_2 : (List.countInRange ClassB_scores (90, 100) = 2) := 
sorry

theorem median_x_eq_75 : (median ClassA_scores = 75) := 
sorry

theorem mode_y_eq_70 : (mode ClassB_scores = 70) := 
sorry

theorem estimate_excellent_students_B : (estimate_excellent_students ClassB_scores 50 = 20) := 
sorry

end m_eq_3_n_eq_2_median_x_eq_75_mode_y_eq_70_estimate_excellent_students_B_l214_214528


namespace initial_marbles_count_l214_214159

-- Leo's initial conditions and quantities
def initial_packs := 40
def marbles_per_pack := 10
def given_Manny (P: ℕ) := P / 4
def given_Neil (P: ℕ) := P / 8
def kept_by_Leo := 25

-- The equivalent proof problem stated in Lean
theorem initial_marbles_count (P: ℕ) (Manny_packs: ℕ) (Neil_packs: ℕ) (kept_packs: ℕ) :
  Manny_packs = given_Manny P → Neil_packs = given_Neil P → kept_packs = kept_by_Leo → P = initial_packs → P * marbles_per_pack = 400 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_marbles_count_l214_214159


namespace area_of_A1_is_correct_l214_214306

axiom width_of_field : ℝ
axiom total_length_of_fences : ℝ
axiom A1_width : ℝ
axiom A1_length : ℝ
axiom A1_area : ℝ
axiom A2_area : ℝ
axiom A3_area : ℝ

noncomputable def calculate_A1_area (A1_width A1_length : ℝ) : ℝ :=
  A1_width * A1_length

axiom field_width_condition : width_of_field = 45
axiom fence_length_condition : total_length_of_fences = 360
axiom A2_area_condition : A2_area = 4 * A1_area
axiom A3_area_condition : A3_area = 5 * A1_area

theorem area_of_A1_is_correct : calculate_A1_area A1_width A1_length = 150 := by
  have h1 : A1_width * 3 + (3 * A1_width) / 2 = 45, from sorry,
  have h2 : total_length_of_fences = 15 * A1_width + 14 * A1_length, from sorry,
  have h3 : A1_area = A1_width * A1_length, from sorry,
  have A1_width_eq_10 : A1_width = 10, from sorry,
  have A1_length_eq_15 : A1_length = 15, from sorry,
  have A1_area_eq_150 : A1_area = 150, from sorry,
  exact A1_area_eq_150

end area_of_A1_is_correct_l214_214306


namespace oil_spending_l214_214322

-- Define the original price per kg of oil
def original_price (P : ℝ) := P

-- Define the reduced price per kg of oil
def reduced_price (P : ℝ) := 0.75 * P

-- Define the reduced price as Rs. 60
def reduced_price_fixed := 60

-- State the condition that reduced price enables 5 kgs more oil
def extra_kg := 5

-- The amount of money spent by housewife at reduced price which is to be proven as Rs. 1200
def amount_spent (M : ℝ) := M

-- Define the problem to prove in Lean 4
theorem oil_spending (P X : ℝ) (h1 : reduced_price P = reduced_price_fixed) (h2 : X * original_price P = (X + extra_kg) * reduced_price_fixed) : amount_spent ((X + extra_kg) * reduced_price_fixed) = 1200 :=
  sorry

end oil_spending_l214_214322


namespace trips_Jean_l214_214342

theorem trips_Jean (x : ℕ) (h1 : x + (x + 6) = 40) : x + 6 = 23 := by
  sorry

end trips_Jean_l214_214342


namespace condition_b_leq_a_div_3_l214_214880

theorem condition_b_leq_a_div_3
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_f : ∀ x, f(x) = 3 * x + 2)
  (h1 : ∀ x, |f(x) + 4| < a)
  (h2 : ∀ x, |x + 2| < b) :
  b ≤ a / 3 :=
sorry

end condition_b_leq_a_div_3_l214_214880


namespace log_three_nine_cubed_l214_214005

theorem log_three_nine_cubed : log 3 (9 ^ 3) = 6 := by
  -- Importing, definitions, and theorem
  sorry

end log_three_nine_cubed_l214_214005


namespace directrix_parabola_l214_214488

variables {p m : ℝ}

theorem directrix_parabola (hp : 0 < p) 
  (hM_on_parabola : (1, m) ∈ (λ (x y : ℝ), y^2 = 2 * p * x)) 
  (h_dist_focus : dist (1, m) (p / 2, 0) = 5) :
  ∀ (x : ℝ), x = -4 :=
begin
  sorry
end

end directrix_parabola_l214_214488


namespace triangle_point_effects_l214_214355

variables (A B C A_1 B_1 C_1 A_2 B_2 C_2 : Type)
variables [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C]
variables [EuclideanGeometry A_1] [EuclideanGeometry B_1] [EuclideanGeometry C_1]
variables [EuclideanGeometry A_2] [EuclideanGeometry B_2] [EuclideanGeometry C_2]

-- Assume given conditions
variables (α β γ : ℝ)
variables (hαbγ : α > β) (hγbα : γ < α) (hαb180 : α < 180)

-- Definitions
def is_orthocenter (P Q R H : A) := ∃ O : Point, ¬ collinear P Q R ∧ collinear O P H ∧ collinear O Q H ∧ collinear O R H 
def is_midpoint (P Q M : A) := dist P M = dist Q M 

-- Hypotheses reflecting conditions
variables (hA_1 : is_orthocenter B C A A_1) (hB_1 : is_orthocenter A C B B_1) (hC_1 : is_orthocenter A B C C_1)
variables (hA_2 : is_midpoint B C A_2) (hB_2 : is_midpoint A C B_2) (hC_2 : is_midpoint A B C_2)

-- Theorem statement reflecting the solution
theorem triangle_point_effects : 
  ∃ (P Q R : Point), is_tri (P Q R ∧ 
  (P Q A B_A_1 B B_A_2 B C_B_1 C) ∧ 
  (360 - 3α) ∧ 
  (360 - 3β) ∧ 
  (180 - 3γ) :=
sorry

end triangle_point_effects_l214_214355


namespace cosine_between_a_b_l214_214845

-- Variables representing vectors in a real inner product space
variables {V : Type*} [inner_product_space ℝ V] (a b : V)

-- Conditions representing vector orthogonality
def cond1 : Prop := inner (a + b) (2 • a - b) = 0
def cond2 : Prop := inner (a - 2 • b) (2 • a + b) = 0

-- Goal: Cosine of the angle between vectors a and b
theorem cosine_between_a_b (h1 : cond1 a b) (h2 : cond2 a b) : 
  real.cos (inner_product_space.angle a b) = -real.sqrt 10 / 10 :=
by
  sorry

end cosine_between_a_b_l214_214845


namespace intersection_PD_QM_on_circumcircle_l214_214543

-- Define the conditions of the problem
variables {A B C D G H M O P Q : Type*}
variables [acute_angled_triangle : ∀ {A B C : Type*}, A ≠ C → Prop]
variables [centroid : ∀ {A B C G : Type*}, Prop]
variables [orthocenter : ∀ {A B C H : Type*}, Prop]
variables [perpendicular : ∀ {A B C D : Type*}, Prop]
variables [midpoint : ∀ {B C M : Type*}, Prop]
variables [circumcircle : ∀ {A B C O : Type*}, Prop]
variables [intersect_MH : ∀ {M H O P : Type*}, Circumcircle_intersect → Prop]
variables [intersect_DG : ∀ {D G O Q : Type*}, Circumcircle_intersect → Prop]
variables [intersection_PD_QM_circumcircle : ∀ {P D Q M O : Type*}, Point_lies_on_circumcircle → Prop]

-- State the proof problem
theorem intersection_PD_QM_on_circumcircle
    (h1 : acute_angled_triangle A B C)
    (h2 : A ≠ C)
    (h3 : centroid A B C G)
    (h4 : orthocenter A B C H)
    (h5 : perpendicular AD BC)
    (h6 : midpoint B C M)
    (h7 : circumcircle A B C O)
    (h8 : intersect_MH M H O P)
    (h9 : intersect_DG D G O Q) :
  intersection_PD_QM_circumcircle P D Q M O :=
sorry -- The proof will be here

end intersection_PD_QM_on_circumcircle_l214_214543


namespace flight_duration_l214_214762

theorem flight_duration :
  ∀ (h m : ℕ),
    (0 < m ∧ m < 60) →
    h = 5 →
    m = 1 →
    h + m = 6 :=
by {
  intros h m hm h_eq m_eq,
  rw [h_eq, m_eq],
  norm_num,
  sorry -- Proof steps skipped as per instructions
}

end flight_duration_l214_214762


namespace jungkook_collected_smallest_l214_214559

theorem jungkook_collected_smallest :
  let a := 6 / 3
  let b := 4
  let c := 5
  in min a (min b c) = a :=
by
  let a := 6 / 3
  let b := 4
  let c := 5
  sorry

end jungkook_collected_smallest_l214_214559


namespace binomial_coefficients_mod_congruence_l214_214940

theorem binomial_coefficients_mod_congruence {n : ℕ} (h : n > 0) :
  {k | ∃ i, i < 2^n ∧ k = Nat.choose (2^n - 1) i} ≡ {1, 3, 5, ..., 2^n - 1} [MOD 2^n] :=
sorry

end binomial_coefficients_mod_congruence_l214_214940


namespace length_difference_l214_214529

variable {A B C D M P K L : Point}
variable {AB CD : Real}

-- Definitions and given conditions
def isConvexQuadrilateral (A B C D : Point) : Prop := sorry
def isTangentCirclesWithDiameters (A B C D : Point) (AB CD : Real) : Prop := sorry
def midpoint (A B : Point) : Point := sorry
def concyclic (A B C D : Point) : Prop := sorry

-- Hypotheses based on conditions
hypothesis h1 : isConvexQuadrilateral A B C D
hypothesis h2 : isTangentCirclesWithDiameters A B C D AB CD
hypothesis h3 : M ≠ intersection (AC) (BD)
hypothesis h4 : P = midpoint A B
hypothesis h5 : extends MP K
hypothesis h6 : concyclic A M C K
hypothesis h7 : extends PM L
hypothesis h8 : concyclic B M D L

-- Main theorem to prove
theorem length_difference (A B C D M P K L : Point) (AB CD : Real) 
  [h1 : isConvexQuadrilateral A B C D] 
  [h2 : isTangentCirclesWithDiameters A B C D AB CD]
  [h3 : M ≠ intersection (AC) (BD)]
  [h4 : P = midpoint A B] 
  [h5 : extends MP K] 
  [h6 : concyclic A M C K]
  [h7 : extends PM L] 
  [h8 : concyclic B M D L] : 
  |distance M K - distance M L| = |AB - CD| := 
sorry

end length_difference_l214_214529


namespace eccentricity_range_l214_214833

theorem eccentricity_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := Real.sqrt (1 - (b / a)^2) in
  e ∈ Set.Ioo 0 ((Real.sqrt 5 - 1) / 2) :=
by
  let e := Real.sqrt (1 - (b / a)^2),
  have h1 : e > 0 := sorry,
  have h2 : e < (Real.sqrt 5 - 1) / 2 := sorry,
  exact ⟨h1, h2⟩

end eccentricity_range_l214_214833


namespace range_of_m_for_inequality_l214_214116

theorem range_of_m_for_inequality :
  {m : ℝ | ∀ x : ℝ, |x-1| + |x+m| > 3} = {m : ℝ | m < -4 ∨ m > 2} :=
sorry

end range_of_m_for_inequality_l214_214116


namespace z_is_negative_y_intercept_l214_214673

-- Define the objective function as an assumption or condition
def objective_function (x y z : ℝ) : Prop := z = 3 * x - y

-- Define what we need to prove: z is the negative of the y-intercept 
def negative_y_intercept (x y z : ℝ) : Prop := ∃ m b, (y = m * x + b) ∧ m = 3 ∧ b = -z

-- The theorem we need to prove
theorem z_is_negative_y_intercept (x y z : ℝ) (h : objective_function x y z) : negative_y_intercept x y z :=
  sorry

end z_is_negative_y_intercept_l214_214673


namespace no_snow_four_days_probability_l214_214252

theorem no_snow_four_days_probability :
  let p1 := 1 / 2
      p2 := 2 / 3
      p3 := 3 / 4
      p4 := 4 / 5
  in (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4) = 1 / 120 :=
by
  let p1 := 1 / 2
  let p2 := 2 / 3
  let p3 := 3 / 4
  let p4 := 4 / 5
  show (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4) = 1 / 120
  sorry

end no_snow_four_days_probability_l214_214252


namespace correct_answer_d_l214_214684

inductive Statement
| st1 : Statement  -- "Waiting for a rabbit by a tree stump"
| st2 : Statement  -- "Good things come through hardship"
| st3 : Statement  -- "Prevent minor issues from becoming major ones"
| st4 : Statement  -- "Insignificant"

def OptionA : List Statement := [Statement.st1, Statement.st2]
def OptionB : List Statement := [Statement.st1, Statement.st3]
def OptionC : List Statement := [Statement.st2, Statement.st4]
def OptionD : List Statement := [Statement.st2, Statement.st3]

def correctOption : List Statement := OptionD

theorem correct_answer_d 
  (A B C D : List Statement)
  (correct : D = [Statement.st2, Statement.st3]) : 
  D = correct := by
  sorry

end correct_answer_d_l214_214684


namespace lisa_investment_in_stocks_l214_214182

-- Definitions for the conditions
def total_investment (r : ℝ) : Prop := r + 7 * r = 200000
def stock_investment (r : ℝ) : ℝ := 7 * r

-- Given the conditions, we need to prove the amount invested in stocks
theorem lisa_investment_in_stocks (r : ℝ) (h : total_investment r) : stock_investment r = 175000 :=
by
  -- proof goes here
  sorry

end lisa_investment_in_stocks_l214_214182


namespace cos15_degree_value_l214_214401

theorem cos15_degree_value :
  cos (15 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  sorry

end cos15_degree_value_l214_214401


namespace solve_exponential_equation_l214_214033

theorem solve_exponential_equation :
  ∃ x : ℝ, 4^x - 2^(x-1) - 3 = 0 ∧ x = 1 :=
by
  sorry

end solve_exponential_equation_l214_214033


namespace constant_term_expansion_l214_214652

theorem constant_term_expansion (x : ℝ) (n : ℕ) (h : 2^n = 64) :
  let r := 2 in 
  let constant_term := (-1 : ℝ)^r * (Nat.choose n r) in
  constant_term = 15 := 
by
  -- Placeholder proof
  sorry

end constant_term_expansion_l214_214652


namespace alice_bob_meet_l214_214331

theorem alice_bob_meet (L : ℝ) (hL : L = 1)
 (i : is_facing_direction)
 (w : total_walking_distance) 
 (p : at_front_of_train) 
 (q : at_end_of_train): 
 ∃ (x : ℝ), x = 1.5 ∧ guaranteed_meet (alice bob) (train L) (total_distance x) :=
by
  sorry

end alice_bob_meet_l214_214331


namespace factor_correct_l214_214379

noncomputable def factor_fraction (a b c : ℝ) : ℝ :=
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3)

theorem factor_correct (a b c : ℝ) : 
  factor_fraction a b c = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end factor_correct_l214_214379


namespace Buffy_whiskers_l214_214800

def whiskers_Juniper : ℕ := 12
def whiskers_Puffy : ℕ := 3 * whiskers_Juniper
def whiskers_Scruffy : ℕ := 2 * whiskers_Puffy
def whiskers_Buffy : ℕ := (whiskers_Puffy + whiskers_Scruffy + whiskers_Juniper) / 3

theorem Buffy_whiskers : whiskers_Buffy = 40 := by
  sorry

end Buffy_whiskers_l214_214800


namespace log_base_3_l214_214010

theorem log_base_3 (a : ℝ) (h1 : a = 9) (h2 : ∀ (b : ℝ) (n : ℤ), log b (b ^ n) = n) : 
  log 3 (9 ^ 3) = 6 := 
by
  have h3 : 9 = 3^2, from by norm_num
  rw [h3] at h1
  rw [h1]
  have h4 : (3^2)^3 = 3^6, from pow_mul 3 2 3
  rw [h4]
  exact h2 3 6

end log_base_3_l214_214010


namespace number_of_pupils_l214_214233

theorem number_of_pupils (N T : ℕ) (H1 : T = 39 * N) (H2 : N > 7) (H3 : T - 147 = 45 * (N - 7)) : N = 28 :=
begin
  sorry
end

end number_of_pupils_l214_214233


namespace remainder_of_3456_div_97_l214_214676

theorem remainder_of_3456_div_97 :
  3456 % 97 = 61 :=
by
  sorry

end remainder_of_3456_div_97_l214_214676


namespace atomic_number_order_l214_214775

-- Define that elements A, B, C, D, and E are in the same period
variable (A B C D E : Type)

-- Define conditions based on the problem
def highest_valence_oxide_basic (x : Type) : Prop := sorry
def basicity_greater (x y : Type) : Prop := sorry
def gaseous_hydride_stability (x y : Type) : Prop := sorry
def smallest_ionic_radius (x : Type) : Prop := sorry

-- Assume conditions given in the problem
axiom basic_oxides : highest_valence_oxide_basic A ∧ highest_valence_oxide_basic B
axiom basicity_order : basicity_greater B A
axiom hydride_stabilities : gaseous_hydride_stability C D
axiom smallest_radius : smallest_ionic_radius E

-- Prove that the order of atomic numbers from smallest to largest is B, A, E, D, C
theorem atomic_number_order :
  ∃ (A B C D E : Type), highest_valence_oxide_basic A ∧ highest_valence_oxide_basic B
  ∧ basicity_greater B A ∧ gaseous_hydride_stability C D ∧ smallest_ionic_radius E
  ↔ B = B ∧ A = A ∧ E = E ∧ D = D ∧ C = C := sorry

end atomic_number_order_l214_214775


namespace even_integers_3000_6000_with_four_different_digits_l214_214860

def is_valid_number (n : ℕ) : Prop :=
  3000 ≤ n ∧ n < 6000 ∧ (n % 2 = 0) ∧ (∃ d1 d2 d3 d4 : ℕ, 
    n = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧
    d2 ≠ d3 ∧ d2 ≠ d4 ∧
    d3 ≠ d4)

theorem even_integers_3000_6000_with_four_different_digits : 
  ∃ count : ℕ, count = 784 ∧ 
  count = (∑ n in (finset.range 6000).filter is_valid_number, 1) :=
by
  sorry

end even_integers_3000_6000_with_four_different_digits_l214_214860


namespace jellybean_probability_l214_214701

theorem jellybean_probability :
  let total_jellybeans := 12,
      red_jellybeans := 5,
      blue_jellybeans := 3,
      white_jellybeans := 4,
      total_pick := 3,
      two_blue_one_white := (choose blue_jellybeans 2) * (choose white_jellybeans 1),
      total_ways := choose total_jellybeans total_pick in
  (two_blue_one_white / total_ways : ℚ) = 3 / 55 :=
by
  let total_jellybeans := 12
  let red_jellybeans := 5
  let blue_jellybeans := 3
  let white_jellybeans := 4
  let total_pick := 3
  let two_blue_one_white := choose blue_jellybeans 2 * choose white_jellybeans 1
  let total_ways := choose total_jellybeans total_pick
  have h1 : (two_blue_one_white : ℚ) = 12 := by sorry
  have h2 : (total_ways : ℚ) = 220 := by sorry
  calc
    (two_blue_one_white / total_ways : ℚ)
        = 12 / 220 : by rw [h1, h2]
    ... = 3 / 55 : by norm_num

end jellybean_probability_l214_214701


namespace prime_in_K_l214_214570

def K (n : ℕ) : Prop := ∃ k : ℕ, n = 1 + 100 * (10 ^ k - 1) / 9

theorem prime_in_K :
  ∀ n ∈ K, Prime n ↔ n = 101 :=
by sorry

end prime_in_K_l214_214570


namespace sqrt_mul_simplify_l214_214220

theorem sqrt_mul_simplify : Real.sqrt 18 * Real.sqrt 32 = 24 := 
sorry

end sqrt_mul_simplify_l214_214220


namespace convex_pentagons_from_15_points_l214_214021

theorem convex_pentagons_from_15_points : (Nat.choose 15 5) = 3003 := 
by
  sorry

end convex_pentagons_from_15_points_l214_214021


namespace max_value_y_l214_214246

-- The function definition
def y (x : ℝ) : ℝ := x + real.sqrt (1 - x) + 3

-- Condition on x
def x_condition (x : ℝ) : Prop := x ≤ 1

-- Main statement proving the maximum value
theorem max_value_y : ∀ (x : ℝ), x_condition x → y x ≤ 17 / 4 :=
sorry

end max_value_y_l214_214246


namespace motorboat_speed_l214_214316

theorem motorboat_speed 
  (c : ℝ) (h_c : c = 2.28571428571)
  (t_up : ℝ) (h_t_up : t_up = 20 / 60)
  (t_down : ℝ) (h_t_down : t_down = 15 / 60) :
  ∃ v : ℝ, v = 16 :=
by
  sorry

end motorboat_speed_l214_214316


namespace roots_diff_arith_prog_l214_214388

-- We define our polynomial
def poly : Polynomial ℝ := 81 * Polynomial.X^3 - 162 * Polynomial.X^2 + 81 * Polynomial.X - 8

-- The proof statement that the difference between the largest and smallest roots of the polynomial poly is 4 * sqrt(6) / 9, given that the roots are in arithmetic progression.
theorem roots_diff_arith_prog : ∃ (roots : List ℝ), 
  (∀ (x : ℝ), x ∈ roots → Polynomial.eval x poly = 0) ∧
  (roots.Pairwise (λ x y, x < y)) ∧
  (∃ (a d : ℝ), roots = [a - d, a, a + d]) ∧
  List.last' roots - List.head' roots = 4 * Real.sqrt 6 / 9 := by
  sorry

end roots_diff_arith_prog_l214_214388


namespace gcd_1821_2993_l214_214028

theorem gcd_1821_2993 : Nat.gcd 1821 2993 = 1 := 
by 
  sorry

end gcd_1821_2993_l214_214028


namespace cross_section_parallelogram_l214_214598

-- Define points and the pyramid
structure Point := (x : ℝ) (y : ℝ) (z : ℝ)
structure Pyramid := (A B C D : Point)

-- Given conditions
variables (A B C D K : Point)
variable (p : Pyramid)
axiom K_on_AB : K ≠ A ∧ K ≠ B ∧ ∃ t : ℝ, (0 < t) ∧ (t < 1) ∧ K = { x := A.x + t * (B.x - A.x), y := A.y + t * (B.y - A.y), z := A.z + t * (B.z - A.z) }

-- Definition of parallelism
def parallel (u v : Point → Point): Prop :=
  ∃ k : ℝ, ∀ x, u x = v x + k • (v x)

-- The plane is parallel to BC and AD
axiom plane_parallel_BC : parallel (λ x, x-C) (λ x, B-C)
axiom plane_parallel_AD : parallel (λ x, x-D) (λ x, A-D)

-- Theorem statement
theorem cross_section_parallelogram :
  ∃ L M N : Point, 
    K ≠ L ∧ K ≠ M ∧ K ≠ N ∧
    parallel (λ x, x-L) (λ x, K-L) ∧
    parallel (λ x, x-M) (λ x, K-M) ∧
    parallel (λ x, x-N) (λ x, K-N) ∧
    ((K-L) × (K-M) = 0) ∧ 
    ((L-M) × (L-N) = 0) ∧ 
    ((M-N) × (M-K) = 0) :=
sorry

end cross_section_parallelogram_l214_214598


namespace probability_XOXOXOX_l214_214407

theorem probability_XOXOXOX (arrangement : list char) 
  (h_len : arrangement.length = 7) 
  (h_X_count : arrangement.count 'X' = 4) 
  (h_O_count : arrangement.count 'O' = 3) :
  let total_arrangements := nat.choose 7 4 in 
  let favorable_outcomes := 1 in
  favorable_outcomes / total_arrangements = 1 / 35 :=
by
  -- proof
  sorry

end probability_XOXOXOX_l214_214407


namespace six_digit_numbers_even_odd_count_l214_214866

theorem six_digit_numbers_even_odd_count : 
  let digits := set.range(10) in
  let even_digits := {0, 2, 4, 6, 8} in
  let odd_digits := {1, 3, 5, 7, 9} in
  ∃ (first_digit ∈ digits \ {0}), 
  ∃ (positions_even : finset (fin 5)), positions_even.card = 2 ∧
  ∃ (positions_odd : finset (fin 5)), positions_odd.card = 3 ∧
  let total_choices := 
    9 * (nat.choose 5 2) * (5 * 5) * (2 * 2 * 2) in
  total_choices = 90000 :=
by
  sorry

end six_digit_numbers_even_odd_count_l214_214866


namespace factor_correct_l214_214378

noncomputable def factor_fraction (a b c : ℝ) : ℝ :=
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3)

theorem factor_correct (a b c : ℝ) : 
  factor_fraction a b c = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end factor_correct_l214_214378


namespace corrected_mean_l214_214247

theorem corrected_mean :
  ∀ (mean n wrong_obs correct_obs : ℝ), n = 100 → mean = 45 → wrong_obs = 20 → correct_obs = 65 →
  let incorrect_sum := mean * n in
  let difference := correct_obs - wrong_obs in
  let corrected_sum := incorrect_sum + difference in
  let corrected_mean := corrected_sum / n in
  corrected_mean = 45.45 :=
by
  intros mean n wrong_obs correct_obs h1 h2 h3 h4
  let incorrect_sum := mean * n
  let difference := correct_obs - wrong_obs
  let corrected_sum := incorrect_sum + difference
  let corrected_mean := corrected_sum / n
  sorry

end corrected_mean_l214_214247


namespace triangle_congruence_l214_214059

theorem triangle_congruence 
  {A B C D E P Q : Type*} 
  [is_triangle A B C]
  (BC_CA_AB : BC > CA ∧ CA > AB)
  (D_on_BC : is_point_on_line_segment D B C)
  (E_on_ext_BA : is_point_on_extension E B A)
  (BD_BE_CA : dist B D = dist B E ∧ dist B E = dist C A)
  (is_concyclic : is_concyclic E B D P)
  (BP_intersect : intersects_of_circumcircle BP (circumcircle_of_triangle ABC) Q)
  : BP = AQ + CQ :=
sorry

end triangle_congruence_l214_214059


namespace simplify_expr_l214_214212

theorem simplify_expr : (1 / (1 - Real.sqrt 3)) * (1 / (1 + Real.sqrt 3)) = -1 / 2 := by
  sorry

end simplify_expr_l214_214212


namespace centroids_form_similar_triangle_l214_214599

variables {A B C P : Point}
variables [plane ℝ Point]

-- Assuming we have points A, B, C forming a triangle ABC
-- and P is a point in the plane of the triangle

def midpoint (X Y : Point) : Point := (X + Y) / 2

def centroid (X Y Z : Point) : Point := (X + Y + Z) / 3

def triangles_are_similar (X₁ Y₁ Z₁ X₂ Y₂ Z₂ : Point) : Prop :=
∃ (k : ℝ), (k ≠ 0) ∧ ((dist X₁ Y₁ = k * dist X₂ Y₂) ∧ (dist Y₁ Z₁ = k * dist Y₂ Z₂) ∧ (dist Z₁ X₁ = k * dist Z₂ X₂))

theorem centroids_form_similar_triangle (A B C P : Point) :
  let PA := (P + A) / 2
      PB := (P + B) / 2
      PC := (P + C) / 2 in
      let G₁ := centroid P A B
          G₂ := centroid P B C
          G₃ := centroid P C A in
  triangles_are_similar G₁ G₂ G₃ A B C :=
sorry

end centroids_form_similar_triangle_l214_214599


namespace vector_subtraction_correct_l214_214146

variable (AB AC : ℝ × ℝ)

-- Given conditions
def AB_def : ℝ × ℝ := (2, 4)
def AC_def : ℝ × ℝ := (1, 3)
def CB := (prod.fst AB_def - prod.fst AC_def, prod.snd AB_def - prod.snd AC_def)

-- We need to prove that CB = (1, 1)
theorem vector_subtraction_correct : CB = (1, 1) := by
  unfold CB
  unfold AB_def
  unfold AC_def
  sorry

end vector_subtraction_correct_l214_214146


namespace number_of_even_integers_between_3000_and_6000_with_different_digits_l214_214858

def thousands_digit (n : ℕ) : ℕ := n / 1000
def units_digit (n : ℕ) : ℕ := n % 10

def valid_digits (n : ℕ) : Prop :=
  let d1 := thousands_digit n
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := units_digit n
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 ∧
  d1 ∈ {3, 4, 5} ∧ d4 % 2 = 0

noncomputable def even_integers_between_3000_and_6000_with_different_digits : ℕ :=
  (Finset.range 6000).filter (λ n => 3000 ≤ n ∧ n < 6000 ∧ valid_digits n).card

theorem number_of_even_integers_between_3000_and_6000_with_different_digits :
  even_integers_between_3000_and_6000_with_different_digits = 728 :=
by
  sorry

end number_of_even_integers_between_3000_and_6000_with_different_digits_l214_214858


namespace max_min_FA_FB_prod_l214_214458

open Real

-- Definitions for given line, ellipse, and focus
def line (t α : ℝ) : ℝ × ℝ := (2 + t * cos α, t * sin α)
def ellipse (φ : ℝ) : ℝ × ℝ := (3 * cos φ, sqrt 5 * sin φ)
def F : ℝ × ℝ := (sqrt 4, 0)  -- Focus of the ellipse C

-- Condition for α = π/4
def α := π / 4

-- Definition of distance between points (for FA and FB)
def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

-- Definition of |FA| * |FB|
def FA_FB_prod (A B : ℝ × ℝ) : ℝ :=
  distance F A * distance F B

-- Final theorem statement
theorem max_min_FA_FB_prod :
  ∃ A B, 
    line t α = A 
    ∧ ellipse φ = B  
    ∧ ((sin α = 0 → FA_FB_prod A B = 5) 
      ∧ (sin α = 1 ∨ sin α = -1 → FA_FB_prod A B = 25 / 9)) :=
sorry

end max_min_FA_FB_prod_l214_214458


namespace math_problem_l214_214455

noncomputable def f : ℝ → ℝ := sorry

theorem math_problem (h_decreasing : ∀ x y : ℝ, 2 < x → x < y → f y < f x)
  (h_even : ∀ x : ℝ, f (-x + 2) = f (x + 2)) :
  f 2 < f 3 ∧ f 3 < f 0 ∧ f 0 < f (-1) :=
by
  sorry

end math_problem_l214_214455


namespace men_and_women_arrangements_l214_214698

theorem men_and_women_arrangements (men women : ℕ) (uniq_gaps num_ways gaps : ℕ) :
  men = 4 → women = 3 → uniq_gaps = 5 → gaps = 3 → num_ways = 1440 → 
  (4.factorial * (uniq_gaps.choose gaps) * 3.factorial) = num_ways :=
by intros; simp [factorial, nat.choose]; sorry

end men_and_women_arrangements_l214_214698


namespace f_constant_1_l214_214575

theorem f_constant_1 (f : ℕ → ℕ) (h1 : ∀ n : ℕ, 0 < n → f (n + f n) = f n)
  (h2 : ∃ n0 : ℕ, 0 < n0 ∧ f n0 = 1) : ∀ n : ℕ, f n = 1 := 
by
  sorry

end f_constant_1_l214_214575


namespace sin_sum_inequality_l214_214601
-- Importing the entire math library

-- Lean 4 statement for the proof problem
theorem sin_sum_inequality (n : ℕ) : 
  (finset.range (3 * n + 1)).sum (λ i : ℕ, |Real.sin (i : ℝ)|) > (8 * n) / 5 :=
sorry

end sin_sum_inequality_l214_214601


namespace maisy_earnings_increase_l214_214947

-- Define the conditions from the problem
def current_job_hours_per_week : ℕ := 8
def current_job_wage_per_hour : ℕ := 10

def new_job_hours_per_week : ℕ := 4
def new_job_wage_per_hour : ℕ := 15
def new_job_bonus_per_week : ℕ := 35

-- Define the weekly earnings calculations
def current_job_earnings : ℕ := current_job_hours_per_week * current_job_wage_per_hour
def new_job_earnings_without_bonus : ℕ := new_job_hours_per_week * new_job_wage_per_hour
def new_job_earnings_with_bonus : ℕ := new_job_earnings_without_bonus + new_job_bonus_per_week

-- Define the difference in earnings
def earnings_difference : ℕ := new_job_earnings_with_bonus - current_job_earnings

-- The theorem to prove: Maisy will earn $15 more per week at her new job
theorem maisy_earnings_increase : earnings_difference = 15 := by
  sorry

end maisy_earnings_increase_l214_214947


namespace john_pays_12_dollars_l214_214917

/-- Define the conditions -/
def number_of_toys : ℕ := 5
def cost_per_toy : ℝ := 3
def discount_rate : ℝ := 0.2

/-- Define the total cost before discount -/
def total_cost_before_discount := number_of_toys * cost_per_toy

/-- Define the discount amount -/
def discount_amount := total_cost_before_discount * discount_rate

/-- Define the final amount John pays -/
def final_amount := total_cost_before_discount - discount_amount

/-- The theorem to be proven -/
theorem john_pays_12_dollars : final_amount = 12 := by
  -- Proof goes here
  sorry

end john_pays_12_dollars_l214_214917


namespace iterated_polynomial_identity_l214_214171

noncomputable def polynomial (f : ℝ → ℝ) : Prop :=
  ∃ p : polynomial ℚ, ∀ x : ℝ, f x = p.eval x

noncomputable def iterated (f : ℝ → ℝ) : ℕ → ℝ → ℝ
| 0 => id
| (n+1) => f ∘ (iterated f n)

theorem iterated_polynomial_identity
  (f : ℝ → ℝ)
  (hpoly : polynomial f)
  (α : ℝ)
  (hα : α^3 - α = 33^1992)
  (hfα : f(α)^3 - f(α) = 33^1992) :
  ∀ (n : ℕ), n ≥ 1 → (iterated f n α)^3 - iterated f n α = 33^1992 :=
by
  sorry

end iterated_polynomial_identity_l214_214171


namespace parallel_segment_bisector_l214_214044

variables {A B C P Q M T : Type}
variables [EuclideanGeometry A B C P Q M T]

/-- Given △ABC, points P, Q on AB, AC such that BP = CQ. 
  Let M, T be the midpoints of BC, PQ. Prove that MT is parallel to the angle bisector of ∠BAC. -/
theorem parallel_segment_bisector
    (h1: OnSegment P A B)
    (h2: OnSegment Q A C)
    (h3: Distance B P = Distance C Q)
    (M_mid: Midpoint M B C)
    (T_mid: Midpoint T P Q) :
    ParallelSegment (Segment M T) (AngleBisector (Angle A B C)) :=
sorry

end parallel_segment_bisector_l214_214044


namespace max_correct_answers_l214_214327

variable (c w b : ℕ) -- Define c, w, b as natural numbers

theorem max_correct_answers (h1 : c + w + b = 30) (h2 : 4 * c - w = 70) : c ≤ 20 := by
  sorry

end max_correct_answers_l214_214327


namespace inequality_condition_l214_214105

theorem inequality_condition (g : ℝ → ℝ) (c d : ℝ) (x : ℝ) : 
  (∀ x : ℝ, g x = 4 * x + 5) → 
  0 < c → 
  0 < d → 
  d ≤ c / 4 → 
  abs (g x + 7) < c ↔ abs (x + 3) < d :=
by simp [g]; sorry

end inequality_condition_l214_214105


namespace root_sum_of_reciprocals_l214_214039

theorem root_sum_of_reciprocals {m : ℝ} :
  (∃ (a b : ℝ), a ≠ b ∧ (a + b) = 2 * (m + 1) ∧ (a * b) = m^2 + 2 ∧ (1/a + 1/b) = 1) →
  m = 2 :=
by sorry

end root_sum_of_reciprocals_l214_214039


namespace parallel_lines_perpendicular_lines_y_intercept_1_l214_214495

-- Given conditions
def line1 (m n : ℝ) : ℝ × ℝ → Prop := λ (x y : ℝ), m * x + 8 * y + n = 0
def line2 (m : ℝ) : ℝ × ℝ → Prop := λ (x y : ℝ), 2 * x + m * y - 1 = 0

def is_parallel (m : ℝ) : Prop := m ≠ 0 ∧ (m = 4 ∧ ¬(n != -2)) ∨ (m = -4 ∧ ¬(n != 2))
def is_perpendicular_with_y_intercept (m n : ℝ) : Prop := m = 0 ∧ n = 8

-- Lean theorem statements
theorem parallel_lines (m n : ℝ) (x y : ℝ) : 
  is_parallel m → line1 m n (x, y) → ¬ line2 m (x, y) → 
  (m = 4 ∧ n ≠ -2 ∨ m = -4 ∧ n ≠ 2) := 
sorry

theorem perpendicular_lines_y_intercept_1 (m n : ℝ) (x y : ℝ) : 
  is_perpendicular_with_y_intercept m n → line1 m n (x, y) → line2 m (x, y) → 
  (m = 0 ∧ n = 8) := 
sorry

end parallel_lines_perpendicular_lines_y_intercept_1_l214_214495


namespace simplify_and_evaluate_l214_214609

theorem simplify_and_evaluate (a : ℝ) (h : a = -3 / 2) : 
  (a - 2) * (a + 2) - (a + 2)^2 = -2 := 
by 
  sorry

end simplify_and_evaluate_l214_214609


namespace value_of_expression_l214_214882

noncomputable def repeating_decimal := 2537 / 99000000

theorem value_of_expression :
  let x := 2 + 5 + 3 + 7
  in (((3 * x) ^ 4 - (2 * x) ^ 3) * repeating_decimal) = 171 :=
by
  sorry

end value_of_expression_l214_214882


namespace ellipse_standard_eq_and_triangle_area_l214_214821

theorem ellipse_standard_eq_and_triangle_area
  (a b : ℝ)
  (P : ℝ × ℝ)
  (F1 F2 : ℝ × ℝ)
  (eccentricity : ℝ)
  (angle : ℝ)
  (h1 : eccentricity = 4 / 5)
  (h2 : a > b)
  (h3 : b > 0)
  (h4 : P ∈ { p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1 })
  (h5 : dist P F1 + dist P F2 = 10)
  (h6 : angle = 60) :
  (a = 5 ∧ b = 3 ∧ \(\frac{x^2}{25} + \frac{y^2}{9} = 1\)) ∧
  (area_of_triangle F1 P F2 = 3 * sqrt 3) :=
by
  sorry

end ellipse_standard_eq_and_triangle_area_l214_214821


namespace rectangle_area_is_eight_l214_214335

theorem rectangle_area_is_eight (A B C D E F : Point)
  (ABCD_rect : rectangle A B C D)
  (AED_iso_rt : isosceles_right_triangle A E D)
  (BFC_iso_rt : isosceles_right_triangle B F C)
  (EF_eq_AD : dist E F = 2) 
  (AD_eq_2 : dist A D = 2) : 
  area_of_rect A B C D = 8 := 
sorry

end rectangle_area_is_eight_l214_214335


namespace first_mission_days_l214_214157

-- Definitions
variable (x : ℝ) (extended_first_mission : ℝ) (second_mission : ℝ) (total_mission_time : ℝ)

axiom h1 : extended_first_mission = 1.60 * x
axiom h2 : second_mission = 3
axiom h3 : total_mission_time = 11
axiom h4 : extended_first_mission + second_mission = total_mission_time

-- Theorem to prove
theorem first_mission_days : x = 5 :=
by
  sorry

end first_mission_days_l214_214157


namespace midpoint_one_sixth_one_twelfth_l214_214770

theorem midpoint_one_sixth_one_twelfth : (1 : ℚ) / 8 = (1 / 6 + 1 / 12) / 2 := by
  sorry

end midpoint_one_sixth_one_twelfth_l214_214770


namespace sum_adjacent_to_five_l214_214996

noncomputable def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ d, d > 1 ∧ n % d = 0)

def commonFactor (a b : ℕ) : Prop :=
  ∃ d : ℕ, Nat.gcd a b = d ∧ d > 1

def arrangesCircle (l : List ℕ) : Prop :=
  (∀ i j, l.get i = some 5 → l.get ((i + 1) % l.length) = some j → commonFactor 5 j)

theorem sum_adjacent_to_five : 
  let divs := divisors 245 in
  let filt_divs := divs.filter (λ x, x ≠ 1) in
  arrangesCircle filt_divs →
  ∃ a b : ℕ, (set.mem a filt_divs ∧ set.mem b filt_divs ∧ commonFactor 5 a ∧ commonFactor 5 b ∧ a ≠ b) ∧ a + b = 280 :=
by sorry

end sum_adjacent_to_five_l214_214996


namespace walnut_trees_planted_l214_214264

-- Given: Initial number of walnut trees in the park and the final number of walnut trees in the park
variables (initial final planted : ℕ)
hypothesis h1 : initial = 4
hypothesis h2 : final = 10

-- To Prove: The number of walnut trees planted is 6
theorem walnut_trees_planted : planted = final - initial → planted = 6 :=
by
-- Our hypothesis makes initial and final are constants, so we can calculate planted directly
calc
  planted = final - initial : sorry
  ... = 10 - 4 : by rw [h2, h1]
  ... = 6 : by norm_num

end walnut_trees_planted_l214_214264


namespace cricketer_runs_l214_214525

theorem cricketer_runs (avg1 avg2 matches1 matches2 boundary1 percent_boundary1 boundary2 percent_boundary2 :
  ℝ) 
  (havg1 : avg1 = 20) (havg2 : avg2 = 30)
  (hmatch1 : matches1 = 2) (hmatch2 : matches2 = 3)
  (hpercent_boundary1 : percent_boundary1 = 0.60) (hpercent_nonboundary1 : 1 - percent_boundary1 = 0.40)
  (hpercent_boundary2 : percent_boundary2 = 0.80) (hpercent_nonboundary2 : 1 - percent_boundary2 = 0.20) :
  let total_runs := avg1 * matches1 + avg2 * matches2
      boundary_runs := percent_boundary1 * avg1 * matches1 + percent_boundary2 * avg2 * matches2
      nonboundary_runs := (1 - percent_boundary1) * avg1 * matches1 + (1 - percent_boundary2) * avg2 * matches2
      avg_score := total_runs / (matches1 + matches2)
  in avg_score = 26 ∧ boundary_runs = 96 ∧ nonboundary_runs = 34 :=
by
  sorry

end cricketer_runs_l214_214525


namespace petersburg_1992_l214_214964

noncomputable def sqrt (x : ℝ) : ℝ := real.sqrt x

theorem petersburg_1992 (A : ℕ) (h : ¬∃ m : ℕ, m * m = A) :
  ∃ n : ℕ, A = ⌊ n + sqrt n + 1/2 ⌋ :=
by {
  sorry
}

end petersburg_1992_l214_214964


namespace tangent_line_equation_l214_214983

theorem tangent_line_equation (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x^3 - a * x) →
  (∀ x, deriv f x = 3 * x^2 - a) →
  (∃ ξ, ξ ∈ set.Icc (-1 : ℝ) 1 ∧ deriv f ξ = 0) →
  (tangent_line_eqn : ∀ x, deriv f x = 3 * x^2 - a) →
  f(2) = 2 →
  (deriv f 2 = 9) →
  (∀ x, 9 * x - (f(2) + 9 * (x - 2)) - 16 = 0) := sorry

end tangent_line_equation_l214_214983


namespace midpoint_M_BC_l214_214294

variables {A B C D E O O1 O2 P Q M : ℂ}

-- Conditions
-- A is at the origin
def A_origin : A = 0 := rfl

-- D is on AB, E is on AC
def D_on_AB (t : ℝ) (ht : 0 < t ∧ t < 1) : D = t * B :=
sorry

def E_on_AC (t : ℝ) (ht : 0 < t ∧ t < 1) : E = t * C :=
sorry

-- DE ∥ BC
def DE_parallel_BC : (E - D) * (B - C) = (E - D) * (B - C) := rfl

-- O1 and O2 are the circumcenters of ΔABE and ΔACD respectively
def circumcenter_O1 : O1 = (abs B^2 * E - abs E^2 * B) / (conj B * E - B * conj E) :=
sorry

def circumcenter_O2 : O2 = (abs C^2 * D - abs D^2 * C) / (conj C * D - C * conj D) :=
sorry

-- O1O2 intersects AB at P and AC at Q
def intersection_O1O2_AB : P = ? := sorry
def intersection_O1O2_AC : Q = ? := sorry

-- O is the circumcenter of ΔAPQ
def circumcenter_O : O = (abs P^2 * Q - abs Q^2 * P) / (conj P * Q - P * conj Q) :=
sorry

-- M is the intersection of line AO and BC
def intersection_AO_BC : M = a * O + (1-a) * (B + C) / 2 :=
sorry

-- Proof that M is the midpoint of BC
theorem midpoint_M_BC (hA: A = 0) (hDE_parallel: DE_parallel_BC) 
                      (hD_on_AB: ∃ t, 0 < t ∧ t < 1 ∧ D_on_AB t (and.intro h t))
                      (hE_on_AC: ∃ t, 0 < t ∧ t < 1 ∧ E_on_AC t (and.intro h t))
                      (hO1: O1 = circumcenter_O1) 
                      (hO2: O2 = circumcenter_O2) 
                      (hP: P = intersection_O1O2_AB) 
                      (hQ: Q = intersection_O1O2_AC)
                      (hO: O = circumcenter_O) 
                      (hM_inter: M = intersection_AO_BC) : 
  M = (B + C) / 2 :=
sorry

end midpoint_M_BC_l214_214294


namespace train_length_l214_214309

theorem train_length
  (speed_kmph : ℝ)
  (platform_length : ℝ)
  (crossing_time : ℝ)
  (conversion_factor : ℝ)
  (speed_mps : ℝ)
  (distance_covered : ℝ)
  (train_length : ℝ) :
  speed_kmph = 72 →
  platform_length = 240 →
  crossing_time = 26 →
  conversion_factor = 5 / 18 →
  speed_mps = speed_kmph * conversion_factor →
  distance_covered = speed_mps * crossing_time →
  train_length = distance_covered - platform_length →
  train_length = 280 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end train_length_l214_214309


namespace number_of_digits_in_Q_l214_214932

noncomputable def Q : ℕ :=
  12345678910111213 * 987654321098

theorem number_of_digits_in_Q : nat.log10 Q + 1 = 30 := 
begin
  sorry
end

end number_of_digits_in_Q_l214_214932


namespace problem_statement_l214_214544

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, ∃ d : ℤ, a (n + 1) = a n + d

noncomputable def given_conditions (a : ℕ → ℤ) : Prop :=
a 2 = 2 ∧ a 3 = 4

theorem problem_statement (a : ℕ → ℤ) (h1 : given_conditions a) (h2 : arithmetic_sequence a) :
  a 10 = 18 := by
  sorry

end problem_statement_l214_214544


namespace certain_event_of_not_drawing_four_same_color_l214_214132

theorem certain_event_of_not_drawing_four_same_color (red yellow white : ℕ) (total_balls : ℕ) :
  red = 3 → yellow = 1 → white = 1 → total_balls = red + yellow + white →
  ¬(∀ n, n = 4 → n ≤ total_balls → ∃ color, count color n ≥ 4) :=
by {
    intros,
    sorry
}

end certain_event_of_not_drawing_four_same_color_l214_214132


namespace quadrilateral_possible_l214_214683

theorem quadrilateral_possible {a b c d : ℕ} (h1 : a = 2) (h2 : b = 2) (h3 : c = 2) (h4 : d = 5) : a + b + c > d :=
by {
  -- from the conditions
  rw [h1, h2, h3, h4],
  -- forming our inequality
  exact Nat.lt_of_sub_pos (by norm_num),
}

end quadrilateral_possible_l214_214683


namespace masha_misha_length_equality_l214_214583

theorem masha_misha_length_equality (A B C D E F : Point)
  (h_square : square ABCD)
  (h_E_on_CD : E ∈ segment C D)
  (h_F_on_BC : F ∈ segment B C)
  (h_bisector : is_angle_bisector F ⟨B, A, E⟩) : 
  distance A E = distance B F + distance E D := 
sorry

end masha_misha_length_equality_l214_214583


namespace approximate_value_of_Γ_l214_214140

noncomputable def calculate_Γ (S : ℝ) (E_P : ℝ) : ℝ :=
  let E_r := (3 / S) * E_P * 10^(-7)
  10 * log10 (E_r / E_P)

theorem approximate_value_of_Γ :
  calculate_Γ 75 1 = -83.98 :=
by
  -- Assumptions and conditions
  sorry

end approximate_value_of_Γ_l214_214140


namespace sequence_conjecture_l214_214817

theorem sequence_conjecture (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) :
  (a 1 = 5) →
  (∀ n ≥ 2, a n = S (n-1)) →
  (∀ n, S n = ∑ i in Finset.range n, a (i + 1)) →
  (a n = if n = 1 then 5 else 5 * 2 ^ (n - 2)) :=
by
  sorry

end sequence_conjecture_l214_214817


namespace lattice_points_in_circles_l214_214502

theorem lattice_points_in_circles : 
  let count := λ (c : ℤ), c.bounded_by (5 : ℕ) in
  (count c : ℕ) = 13 := sorry

end lattice_points_in_circles_l214_214502


namespace coefficient_x5_in_expansion_l214_214779

theorem coefficient_x5_in_expansion :
  let binomial_expansion (n : ℕ) (a b : ℝ) :=
    ∑ r in Finset.range (n+1), (∑ x, a^(x) * b^(r) * x^(n-r))
  in binomial (8 : ℕ) (1 : ℝ) (-1/√(1 : ℝ)):
  ∀ (x : ℝ), coefficient_of x 5 (binomial_expansion 8 x (-1/√x)) = 28 :=
by
  sorry

end coefficient_x5_in_expansion_l214_214779


namespace min_increase_air_quality_days_l214_214923

theorem min_increase_air_quality_days {days_in_year : ℕ} (last_year_ratio next_year_ratio : ℝ) (good_air_days : ℕ) :
  days_in_year = 365 → last_year_ratio = 0.6 → next_year_ratio > 0.7 →
  (good_air_days / days_in_year < last_year_ratio → ∀ n: ℕ, good_air_days + n ≥ 37) :=
by
  intros hdays_in_year hlast_year_ratio hnext_year_ratio h_good_air_days
  sorry

end min_increase_air_quality_days_l214_214923


namespace proof_problem_l214_214973

variable (a b c : ℝ)

theorem proof_problem (h : a ∈ ℝ ∧ b ∈ ℝ ∧ c ∈ ℝ) : (a - b)^2 + (b - c)^2 + (a - c)^2 ≥ 0 :=
by
  sorry

end proof_problem_l214_214973


namespace log4_one_over_sqrt_16_eq_neg_one_l214_214371

theorem log4_one_over_sqrt_16_eq_neg_one (x : ℝ) : 
  (√(16) = 4) → (1 / √(16) = 1 / 4) → (1 / 4 = 4^(-1)) → (log 4 (1 / √(16)) = -1) :=
by
  intro h₁ h₂ h₃
  sorry

end log4_one_over_sqrt_16_eq_neg_one_l214_214371


namespace john_total_payment_l214_214916

def cost_per_toy := 3
def number_of_toys := 5
def discount_rate := 0.2

theorem john_total_payment :
  (number_of_toys * cost_per_toy) - ((number_of_toys * cost_per_toy) * discount_rate) = 12 :=
by
  sorry

end john_total_payment_l214_214916


namespace circle_tangent_to_circumcircle_l214_214928

/--
Let I be the incenter of triangle ABC. A perpendicular is drawn from I to AI intersecting sides AB and AC at points P and Q, respectively. 
Prove that the circle L, which is tangent to AB at P and to AC at Q, is also tangent to the circumcircle O of triangle ABC.
-/
theorem circle_tangent_to_circumcircle 
  {A B C I P Q O L : Type*} 
  [incenter A B C I]
  [perpendicular I AI AB P]
  [perpendicular I AI AC Q]
  [tangent_circle L AB P]
  [tangent_circle L AC Q]
  [circumcircle O A B C] :
  tangent L O :=
sorry

end circle_tangent_to_circumcircle_l214_214928


namespace f_increasing_l214_214242

noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.sin x

theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y :=
by
  sorry

end f_increasing_l214_214242


namespace polar_equation_of_C_max_area_OAB_l214_214542

noncomputable def Cartesian_to_Polar (x y : ℝ) : ℝ := 
  let ρ := Real.sqrt (x * x + y * y)
  let θ := Real.arctan2 y x
  ρ

theorem polar_equation_of_C (θ : ℝ) (hθ : 0 < θ ∧ θ < π) :
  ((Cartesian_to_Polar (ρ := |θ.cos| + θ.sin).1 (ρ := |θ.cos| + θ.sin).2) = ρ) :=
sorry

theorem max_area_OAB (α : ℝ) (hα : 0 < α ∧ α < π/2) :
  let ρ1 := |θ.cos| + θ.sin,
  let ρ2 := |θ.sin| + θ.cos
  (ρ1 * ρ2 / 2 = 1) :=
sorry

end polar_equation_of_C_max_area_OAB_l214_214542


namespace Mira_lap_time_improvement_l214_214189

theorem Mira_lap_time_improvement :
  (45 / 15) - (42 / 18) = 2 / 3 := 
by 
  -- Perform initial lap time calculation
  have h1 : 45 / 15 = 3 := by norm_num,
  -- Perform current lap time calculation
  have h2 : 42 / 18 = 7 / 3 := by norm_num,
  -- Prove the difference is 2/3
  calc
    3 - 7 / 3 = 2 / 3  := by norm_num

end Mira_lap_time_improvement_l214_214189


namespace law_of_cosines_l214_214447

theorem law_of_cosines (a b c C : ℝ) (habc : a^2 + b^2 - 2 * a * b * real.cos C = c^2) : 
  c^2 = a^2 + b^2 - 2 * a * b * real.cos C :=
by
  sorry

end law_of_cosines_l214_214447


namespace probability_XOXOXOX_l214_214415

theorem probability_XOXOXOX (X O : ℕ) (h1 : X = 4) (h2 : O = 3) :
  let total_ways := Nat.choose (X + O) X,
      favorable_outcomes := 1 in
  (favorable_outcomes / total_ways : ℚ) = 1 / 35 := by
  sorry

end probability_XOXOXOX_l214_214415


namespace lines_parallel_if_perpendicular_to_same_plane_l214_214829

variables {m n : Type}
variables {α β γ : Type}
variables [line m] [line n] [plane α] [plane β] [plane γ]

-- Given conditions:
def m_different_from_n (m n : Type) [line m] [line n] : Prop :=
  m ≠ n

def three_different_planes (α β γ : Type) [plane α] [plane β] [plane γ] : Prop :=
  α ≠ β ∧ β ≠ γ ∧ α ≠ γ

def m_perpendicular_to_α (m : Type) [line m] (α : Type) [plane α] : Prop :=
  perpendicular m α

def n_perpendicular_to_α (n : Type) [line n] (α : Type) [plane α] : Prop :=
  perpendicular n α

-- To prove:
theorem lines_parallel_if_perpendicular_to_same_plane
  (m n : Type) [line m] [line n]
  (α β γ : Type) [plane α] [plane β] [plane γ]
  (hne: m_different_from_n m n)
  (hp: three_different_planes α β γ)
  (hpa: m_perpendicular_to_α m α)
  (hpb: n_perpendicular_to_α n α) :
  parallel m n :=
sorry

end lines_parallel_if_perpendicular_to_same_plane_l214_214829


namespace solve_for_a_l214_214104

theorem solve_for_a (a b x : ℝ) (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * x^3) (h3 : a - b = x) :
  a = x * (1 + real.sqrt (107 / 3)) / 2 ∨ a = x * (1 - real.sqrt (107 / 3)) / 2 :=
by sorry

end solve_for_a_l214_214104


namespace option_b_results_in_2x_cubed_l214_214284

variable (x : ℝ)

theorem option_b_results_in_2x_cubed : |x^3| + x^3 = 2 * x^3 := 
sorry

end option_b_results_in_2x_cubed_l214_214284


namespace min_problems_needed_l214_214976

-- Define non-overlapping test papers
structure TestPaper (α : Type) where
  problems : Finset α
  h_card : problems.card = 4

theorem min_problems_needed {α : Type} (papers : Finset (TestPaper α)) 
  (h_papers_card : papers.card = 10) 
  (h_problems_intersection : ∀ (p1 p2 : TestPaper α), p1 ≠ p2 → (p1.problems ∩ p2.problems).card ≤ 1) : 
  ∃ (problems : Finset α), problems.card = 13 := 
sorry

end min_problems_needed_l214_214976


namespace angle_A_is_60_l214_214566

variable {α : Type*} [EuclideanGeometry α]

-- Defining the centroid G
def is_centroid (A B C G : α) : Prop :=
  G = point_average [A, B, C]

-- Defining the orthocenter H
def is_orthocenter (A B C H : α) : Prop :=
  -- Orthocenter definition here, typically involves perpendiculars and intersection
  perpendicular (A - B) (C - B) ∧
  perpendicular (B - C) (A - C) ∧
  perpendicular (C - A) (B - A) ∧
  H = intersection_point_perpendiculars A B C

-- Given conditions
variables {A B C G H : α}

theorem angle_A_is_60 (hG : is_centroid A B C G) (hH : is_orthocenter A B C H) (h_eq : distance A G = distance A H) :
  angle A B C = 60 := 
sorry

end angle_A_is_60_l214_214566


namespace minimize_I_l214_214109

-- Define the function H(p, q)
def H (p q : ℝ) : ℝ := -3*p*q + 4*p*(1-q) + 2*(1-p)*q - 5*(1-p)*(1-q)

-- Define the function I(p)
def I (p : ℝ) : ℝ :=
  max (4*p - 5) (1 + p - 5)

-- Define the statement that proves that the value of p that minimizes I(p) is 1/3
theorem minimize_I : ∀ p : ℝ, 0 ≤ p ∧ p ≤ 1 → I p = I (1/3) → p = 1/3 :=
by {
  intros p hp,
  have h : I p = I (1/3),
  sorry
}

end minimize_I_l214_214109


namespace vacation_probability_l214_214889

noncomputable def binomial_prob (p : ℚ) (n k : ℕ) : ℚ :=
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem vacation_probability :
  binomial_prob (40/100) 5 3 = 2304 / 10000 :=
by
  sorry

end vacation_probability_l214_214889


namespace sin_A_value_l214_214093

-- Given: a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively
variables (a b c : ℝ ) (A B C : ℝ)

-- Given conditions:
-- a = 2
-- b = 3
-- cos C = 1/3
-- Equation for the value of sin A
theorem sin_A_value 
  (h1 : a = 2) 
  (h2 : b = 3) 
  (h3 : real.cos C = 1 / 3) 
  (h4 : c^2 = a^2 + b^2 - 2 * a * b * real.cos C) :
  real.sin A = 4 * real.sqrt 2 / 9 :=
sorry

end sin_A_value_l214_214093


namespace angle_C_value_range_a2_b2_l214_214524

variables {A B C a b c : ℝ}

theorem angle_C_value
  (h1: tan C = (sin A + sin B) / (cos A + cos B)) :
  C = π / 3 :=
sorry

theorem range_a2_b2
  (h2 : C = π / 3)
  (h3 : ∀ (r : ℝ), a = 2 * r * sin A ∧ b = 2 * r * sin B)
  (h4 : 2 * r = 1) :
  (3 / 4) < a^2 + b^2 ∧ a^2 + b^2 ≤ 3 / 2 :=
sorry      

end angle_C_value_range_a2_b2_l214_214524


namespace salary_increase_more_than_45_l214_214190

theorem salary_increase_more_than_45 (S : ℝ) (h : S > 0) :
  let new_salary := S * (1.10)^4 in
  (new_salary - S) / S * 100 > 45 :=
by
  let new_salary := S * (1.10)^4
  have H : new_salary = S * 1.4641, by
    calc
      S * (1.10)^4 = S * 1.4641 : by norm_num
  have percentage_increase := (new_salary - S) / S * 100
  have percentage_increase_eq : percentage_increase = 46.41, by
    calc
      percentage_increase = (S * 1.4641 - S) / S * 100 : by rw [H]
                       ... = 0.4641 * 100 : by ring
                       ... = 46.41 : by norm_num
  show percentage_increase > 45, from by linarith

end salary_increase_more_than_45_l214_214190


namespace cos_alpha_minus_pi_over_4_l214_214045

noncomputable def alpha := sorry
noncomputable def tan_alpha := 2

theorem cos_alpha_minus_pi_over_4 : cos(alpha - π / 4) = 3 * sqrt(10) / 10 :=
  sorry

end cos_alpha_minus_pi_over_4_l214_214045


namespace equation_of_hyperbola_dot_product_zero_area_of_triangle_l214_214053

noncomputable def hyperbola_eccentricity : ℝ := Real.sqrt 2

noncomputable def hyperbola_equation (x y : ℝ) : ℝ := x^2 - y^2

def passes_through_P (x y : ℝ) : Prop := hyperbola_equation x y = 6

def is_point_on_hyperbola (m : ℝ) : Prop := hyperbola_equation 3 m = 6

theorem equation_of_hyperbola :
  hyperbola_equation 4 (-Real.sqrt 10) = 6 := by
  sorry

theorem dot_product_zero (m : ℝ) (h : is_point_on_hyperbola m) :
  let F1 := (-3 - 2 * Real.sqrt 3, -m)
      F2 := (2 * Real.sqrt 3 - 3, -m)
  in (F1.1 * F2.1 + F1.2 * F2.2 = 0) := by
  sorry

theorem area_of_triangle (m : ℝ) (h : is_point_on_hyperbola m)
  (hm : m = Real.sqrt 3 ∨ m = -Real.sqrt 3) :
  let F1 := (-3 - 2 * Real.sqrt 3, -m)
      F2 := (2 * Real.sqrt 3 - 3, -m)
      base := Real.dist (λ x, x.1) F1 F2
      height := abs m
  in (1/2 * base * height = 6) := by
  sorry

end equation_of_hyperbola_dot_product_zero_area_of_triangle_l214_214053


namespace Andre_final_price_l214_214196

theorem Andre_final_price :
  let treadmill_price := 1350
  let treadmill_discount_rate := 0.30
  let plate_price := 60
  let num_of_plates := 2
  let plate_discount_rate := 0.15
  let sales_tax_rate := 0.07
  let treadmill_discount := treadmill_price * treadmill_discount_rate
  let treadmill_discounted_price := treadmill_price - treadmill_discount
  let total_plate_price := plate_price * num_of_plates
  let plate_discount := total_plate_price * plate_discount_rate
  let plate_discounted_price := total_plate_price - plate_discount
  let total_price_before_tax := treadmill_discounted_price + plate_discounted_price
  let sales_tax := total_price_before_tax * sales_tax_rate
  let final_price := total_price_before_tax + sales_tax
  final_price = 1120.29 := 
by
  repeat { 
    sorry 
  }

end Andre_final_price_l214_214196


namespace triangle_side_length_l214_214823

noncomputable def cos_60 := Real.cos (Real.pi / 3)

theorem triangle_side_length (a b : ℝ) (C : ℝ) (c : ℝ) (h_conditions : a = 2 ∧ b = 1 ∧ C = 60 * (π / 180)) :
  c = Real.sqrt 3 :=
by 
  have h_cos : cos_60 = 1 / 2 := by
    simp [cos_60, Real.cos, Real.pi]
  -- Law of Cosines
  have h1 : c^2 = a^2 + b^2 - 2 * a * b * cos_60 := by rw [h_conditions.1, h_conditions.2.left, h_conditions.2.right, h_cos]
  -- Substituting a = 2 and b = 1 and simplifying
  have h2 : 2^2 + 1^2 - 2 * 2 * 1 * (1 / 2) = 3 :=
    by norm_num
  rw [h1, h2]
  -- Therefore, c = sqrt(3)
  have h3 : c = Real.sqrt 3 := by
    apply eq_of_sq_eq_sq (Real.sqrt_nonneg 3)
    rw sq_eq_sqrt
    exact h2
  exact h3

end triangle_side_length_l214_214823


namespace ellipse_fixed_point_l214_214820

theorem ellipse_fixed_point (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (c : ℝ) (h3 : c = 1) 
    (h4 : a = 2) (h5 : b = Real.sqrt 3) :
    (∀ P : ℝ × ℝ, (P.1^2 / a^2 + P.2^2 / b^2 = 1) → 
        ∃ M : ℝ × ℝ, (M.1 = 4) ∧ 
        ∃ Q : ℝ × ℝ, (Q.1= (P.1) ∧ Q.2 = - (P.2)) ∧ 
            ∃ fixed_point : ℝ × ℝ, (fixed_point.1 = 5 / 2) ∧ (fixed_point.2 = 0) ∧ 
            ∃ k, (Q.2 - M.2) = k * (Q.1 - M.1) ∧ 
            ∃ l, fixed_point.2 = l * (fixed_point.1 - M.1)) :=
sorry

end ellipse_fixed_point_l214_214820


namespace common_difference_d_l214_214653

theorem common_difference_d (a_1 d : ℝ) (h1 : a_1 + 2 * d = 4) (h2 : 9 * a_1 + 36 * d = 18) : d = -1 :=
by sorry

end common_difference_d_l214_214653


namespace compare_neg_two_powers_l214_214751

theorem compare_neg_two_powers : (-2)^3 = -2^3 := by sorry

end compare_neg_two_powers_l214_214751


namespace maisy_earns_more_l214_214952

theorem maisy_earns_more 
    (current_hours : ℕ) (current_wage : ℕ) 
    (new_hours : ℕ) (new_wage : ℕ) (bonus : ℕ)
    (h_current_job : current_hours = 8) 
    (h_current_wage : current_wage = 10)
    (h_new_job : new_hours = 4) 
    (h_new_wage : new_wage = 15)
    (h_bonus : bonus = 35) :
  (new_hours * new_wage + bonus) - (current_hours * current_wage) = 15 := 
by 
  sorry

end maisy_earns_more_l214_214952


namespace num_convex_pentagons_l214_214018

theorem num_convex_pentagons (n m : ℕ) (hn : n = 15) (hm : m = 5) : 
  Nat.choose n m = 3003 := by
  sorry

end num_convex_pentagons_l214_214018


namespace conjugate_of_z_l214_214077

theorem conjugate_of_z (z : ℂ) (h : z * complex.I = 1 + complex.I) : conj z = 1 + complex.I :=
sorry

end conjugate_of_z_l214_214077


namespace find_y_sq_l214_214663

theorem find_y_sq (y : ℝ) (h1 : 0 < y) (h2 : real.sin (real.arccos y) = y) : y^2 = 1/2 :=
sorry

end find_y_sq_l214_214663


namespace sum_b_4_l214_214647

def a (n : ℕ) : ℕ := 2 * n - 1

def b : ℕ → ℤ
| 1     := 2
| (n+2) := (2 : ℤ)^(n+2) * (1 - 2 * (n + 2))

def S (n : ℕ) : ℤ := ∑ i in finset.range n, b (i + 1)

theorem sum_b_4 : S 4 = -162 := sorry

end sum_b_4_l214_214647


namespace marys_score_l214_214188

def score (c w : ℕ) : ℕ := 30 + 4 * c - w
def valid_score_range (s : ℕ) : Prop := s > 90 ∧ s ≤ 170

theorem marys_score : ∃ c w : ℕ, c + w ≤ 35 ∧ score c w = 170 ∧ 
  ∀ (s : ℕ), (valid_score_range s ∧ ∃ c' w', score c' w' = s ∧ c' + w' ≤ 35) → 
  (s = 170) :=
by
  sorry

end marys_score_l214_214188


namespace number_wall_solution_l214_214551

theorem number_wall_solution :
  ∃ n : ℕ, n = 12 ∧ 
  let a := n + 6,
      b := a + 19,
      c := b + 33
  in c = 70 :=
by
  sorry

end number_wall_solution_l214_214551


namespace initial_food_supplies_l214_214724

theorem initial_food_supplies (x : ℝ) 
  (h1 : (3 / 5) * x - (3 / 5) * ((3 / 5) * x) = 96) : x = 400 :=
by
  sorry

end initial_food_supplies_l214_214724


namespace f_2006_plus_f_2007_l214_214047

variable (f : ℝ → ℝ)

-- Conditions
axiom even_function : ∀ x : ℝ, f(x) = f(-x)
axiom periodic_condition : ∀ x : ℝ, f(x + 4) = f(x) + f(2)
axiom initial_value : f(1) = 2

-- Proof statement
theorem f_2006_plus_f_2007 : f(2006) + f(2007) = 2 :=
by
  sorry

end f_2006_plus_f_2007_l214_214047


namespace quadrilateral_is_parallelogram_l214_214151

theorem quadrilateral_is_parallelogram
  (A B C D : Type)
  (angle_DAB angle_ABC angle_BAD angle_DCB : ℝ)
  (h1 : angle_DAB = 135)
  (h2 : angle_ABC = 45)
  (h3 : angle_BAD = 45)
  (h4 : angle_DCB = 45) :
  (A B C D : Type) → Prop :=
by
  -- Definitions and conditions are given.
  sorry

end quadrilateral_is_parallelogram_l214_214151


namespace B1C1_tangent_to_incircle_l214_214150

open EuclideanGeometry

variables (A B C : Point)
variables (C1 B1 : Point)
variables (α : Angle)
variables (circumcircle : Circle)

-- Conditions
axiom A_is_60_deg : ∡ B A C = 60° 
axiom C1_is_perp_bisector_of_AB_inter_AC : isPerpBisector (segment A B) (lineThrough A C) C1
axiom B1_is_perp_bisector_of_AC_inter_AB : isPerpBisector (segment A C) (lineThrough A B) B1

-- The proof goal
theorem B1C1_tangent_to_incircle (incircle : Circle) :
  isTangent (lineThrough B1 C1) incircle :=
by
  sorry

end B1C1_tangent_to_incircle_l214_214150


namespace angle_between_PQ_BC_is_right_angle_l214_214693

-- Definitions of the given problem
structure Point := (x y : ℝ)
structure Line := (start end_ : Point)

def cyclic_quadrilateral (A B C D : Point) : Prop :=
  -- Abstract definition implying A, B, C, D lie on some circle
  sorry

def midpoint (P Q : Point) : Point :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

def perpendicular (L1 L2 : Line) : Prop :=
  -- Abstract definition implying L1 is perpendicular to L2
  sorry

def acute_triangle (A B C : Point) : Prop :=
  -- Abstract definition implying triangle ABC is acute
  sorry

-- Main hypothesis and problem statement
theorem angle_between_PQ_BC_is_right_angle
  (A B C D P E F Q : Point)
  (h_cyclic : cyclic_quadrilateral A B C D)
  (h_diag_int : P = intersection (diagonal A C) (diagonal B D))
  (h_acute : acute_triangle A P D)
  (h_E : E = midpoint A B)
  (h_F : F = midpoint C D)
  (h_perp_E : perpendicular (line E Q) (diagonal A C))
  (h_perp_F : perpendicular (line F Q) (diagonal B D)) :
  angle (line P Q) (line B C) = 90 :=
begin
  sorry
end

end angle_between_PQ_BC_is_right_angle_l214_214693


namespace price_of_other_pieces_l214_214557

theorem price_of_other_pieces
    (total_spent : ℕ)
    (known_price1 : ℕ)
    (known_price2 : ℕ)
    (num_pieces : ℕ)
    (num_other_pieces : ℕ)
    (total_price_other_pieces : ℕ)
    (price_other_piece : ℕ)
    (multiple_of_5 : ℕ)
  : 
    let remaining_sum := total_spent - (known_price1 + known_price2)
    in price_other_piece = remaining_sum / num_other_pieces ∧ price_other_piece % multiple_of_5 = 0 
  :=
begin
  -- Define conditions
  let total_spent := 610,
  let known_price1 := 49,
  let known_price2 := 81,
  let num_pieces := 7,
  let num_other_pieces := 5,
  let total_price_other_pieces := total_spent - (known_price1 + known_price2),
  let price_other_piece := total_price_other_pieces / num_other_pieces,
  let multiple_of_5 := 5,
  -- Formulate the remaining sum condition
  let remaining_sum := total_spent - (known_price1 + known_price2),
  -- Verify the price is a multiple of 5
  sorry
end

end price_of_other_pieces_l214_214557


namespace count_uphill_integers_divisible_by_14_l214_214349

def is_uphill_integer (n : ℕ) : Prop :=
  ∀ (d : ℕ), (0 < d ∧ d < int.to_nat (real.log 10 (n.cast d))) → d % 10 < (d / 10) % 10

def is_divisible_by_14 (n : ℕ) : Prop :=
  n % 14 = 0

theorem count_uphill_integers_divisible_by_14 :
  (∑ n in finset.range 100000, if is_uphill_integer n ∧ is_divisible_by_14 n then 1 else 0) = 5 := 
by
  sorry

end count_uphill_integers_divisible_by_14_l214_214349


namespace sqrt_18_mul_sqrt_32_eq_24_l214_214216
  
theorem sqrt_18_mul_sqrt_32_eq_24 : (Real.sqrt 18 * Real.sqrt 32 = 24) :=
  sorry

end sqrt_18_mul_sqrt_32_eq_24_l214_214216


namespace intersection_complement_l214_214581

universe u

def U : Set (Set ℕ) := {a, b, c, d, e}
def A : Set (Set ℕ) := {a, b}
def B : Set (Set ℕ) := {b, c, d}
def C_U (s : Set (Set ℕ)) : Set (Set ℕ) := U \ s

theorem intersection_complement :
  A ∩ C_U B = {a} :=
by
  sorry

end intersection_complement_l214_214581


namespace probability_XOXOXOX_l214_214425

noncomputable def binomial (n k : ℕ) : ℕ := nat.choose n k

theorem probability_XOXOXOX :
  let num_X := 4,
      num_O := 3,
      total_arrangements := binomial 7 num_X in
  total_arrangements = 35 ∧
  1 / total_arrangements = (1 : ℚ) / 35 :=
by
  sorry

end probability_XOXOXOX_l214_214425


namespace skateboard_cost_l214_214155

-- Definitions from conditions
def toyCarsCost : ℝ := 14.88
def toyTrucksCost : ℝ := 5.86
def totalToysCost : ℝ := 25.62

-- Question to be proved as a theorem
theorem skateboard_cost :
  ∃ (skateboardCost : ℝ), skateboardCost = 4.88 :=
by
  let spend := totalToysCost - toyCarsCost - toyTrucksCost
  have h : spend = 4.88 := by sorry
  use spend
  apply h

end skateboard_cost_l214_214155


namespace molecular_weight_one_mole_l214_214282

theorem molecular_weight_one_mole
  (molecular_weight_7_moles : ℝ)
  (mole_count : ℝ)
  (h : molecular_weight_7_moles = 126)
  (k : mole_count = 7)
  : molecular_weight_7_moles / mole_count = 18 := 
sorry

end molecular_weight_one_mole_l214_214282


namespace shanghai2012_g_neg1_l214_214695
section Shanghai2012

variable {f : ℝ → ℝ} 
variable (H1 : ∀ x, f(x) + x^2 = -(f(-x) + (-x)^2)) -- y = f(x) + x^2 is odd function
variable (H2 : f 1 = 1)
def g (x : ℝ) : ℝ := f x + 2

theorem shanghai2012_g_neg1 : g H1 1 f H2 (-1) = -1 :=
by
  sorry

end Shanghai2012

end shanghai2012_g_neg1_l214_214695


namespace mutually_exclusive_not_opposite_l214_214773

theorem mutually_exclusive_not_opposite :
  ∀ (A B C : Type) (black red white : ℕ → Type) (dist : A → (ℕ → Type) → Prop),
    (∃ (a : A) (b : A) (c : A), dist a red ∧ dist b white ∧ dist c black) →
    (∀ (x y : A), (x ≠ y) → ¬(∃ (r: ℕ → Type), dist x r ∧ dist y r)) ∧
    (dist A red → ¬dist B red) :=
by
  intros A B C black red white dist h
  sorry

end mutually_exclusive_not_opposite_l214_214773


namespace triangle_similarity_l214_214548

variables {A B C M N O : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace N] [MetricSpace O]

-- Conditions:
-- 1. Triangle ABC is isosceles
def is_isosceles (A B C : Type) [MetricSpace α] (h : dist B C = dist A C) : Prop := sorry

-- 2. Points O, M, and N, and their properties
def semicircle_center_on_base (O : Type) [MetricSpace A] [MetricSpace B] (h : O ∈ line [A, B]) : Prop := sorry
def tangent_to_semicircle (M N : Type) [MetricSpace M] [MetricSpace N] [MetricSpace O] (O M N : MetricSpace α) : Prop := sorry

-- 3. Point M is on BC and point N is on AC
def point_on_line (P A B : Type) [MetricSpace α] [MetricSpace β] [MetricSpace γ] (h : P ∈ line [A, B]) : Prop := sorry

-- Triangles similarity definition
def similar (T1 T2 T3 : Type) [MetricSpace α] [MetricSpace β] [MetricSpace γ] (T1 T2 T3 : MetricSpace α) : Prop := sorry

-- The main goal to prove
theorem triangle_similarity (A B C M N O : Type)
  [MetricSpace α] [MetricSpace β] [MetricSpace γ]
  (h_iso : is_isosceles A B C)
  (h_center : semicircle_center_on_base O A B)
  (h_tangent : tangent_to_semicircle M N O)
  (h_M_BC : point_on_line M B C)
  (h_N_AC : point_on_line N A C) :
  similar (triangle A O N) (triangle B O M) (triangle M O N) :=
sorry

end triangle_similarity_l214_214548


namespace inequality_l214_214071

variable (a b m : ℝ)

theorem inequality (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < m) (h4 : a < b) :
  a / b < (a + m) / (b + m) :=
by
  sorry

end inequality_l214_214071


namespace solve_Cheolsu_weight_l214_214234

def Cheolsu_weight (C M F : ℝ) :=
  (C + M + F) / 3 = M ∧
  C = (2 / 3) * M ∧
  F = 72

theorem solve_Cheolsu_weight {C M F : ℝ} (h : Cheolsu_weight C M F) : C = 36 :=
by
  sorry

end solve_Cheolsu_weight_l214_214234


namespace calculate_total_cost_l214_214723

-- Define the cost per workbook
def cost_per_workbook (x : ℝ) : ℝ := x

-- Define the number of workbooks
def number_of_workbooks : ℝ := 400

-- Define the total cost calculation
def total_cost (x : ℝ) : ℝ := number_of_workbooks * cost_per_workbook x

-- State the theorem to prove
theorem calculate_total_cost (x : ℝ) : total_cost x = 400 * x :=
by sorry

end calculate_total_cost_l214_214723


namespace six_digit_numbers_even_odd_count_l214_214865

theorem six_digit_numbers_even_odd_count : 
  let digits := set.range(10) in
  let even_digits := {0, 2, 4, 6, 8} in
  let odd_digits := {1, 3, 5, 7, 9} in
  ∃ (first_digit ∈ digits \ {0}), 
  ∃ (positions_even : finset (fin 5)), positions_even.card = 2 ∧
  ∃ (positions_odd : finset (fin 5)), positions_odd.card = 3 ∧
  let total_choices := 
    9 * (nat.choose 5 2) * (5 * 5) * (2 * 2 * 2) in
  total_choices = 90000 :=
by
  sorry

end six_digit_numbers_even_odd_count_l214_214865


namespace circle_radius_in_square_l214_214894

-- Define the problem conditions
def square (A B C D : Point) (s : ℝ) : Prop :=
  distance A B = s ∧ distance B C = s ∧ distance C D = s ∧ distance D A = s ∧
  ∠ABC = π / 2 ∧ ∠BCD = π / 2 ∧ ∠CDA = π / 2 ∧ ∠DAB = π / 2

def circle (O : Point) (r : ℝ) (A D : Point) : Prop :=
  distance O A = r ∧ distance O D = r

def tangent (O : Point) (r : ℝ) (BC : Line) : Prop :=
  distance O BC = r

-- Theorem stating the problem
theorem circle_radius_in_square :
  ∀ (A B C D O : Point) (s r : ℝ),
    square A B C D s → circle O r A D → tangent O r (line B C) → s = 8 → r = 4 * sqrt 2 := 
begin
  intros A B C D O s r h_square h_circle h_tangent h_side_length,
  sorry
end

end circle_radius_in_square_l214_214894


namespace journey_time_proof_l214_214961

noncomputable def journey_time_on_wednesday (d s x : ℝ) : ℝ :=
  d / s

theorem journey_time_proof (d s x : ℝ) (usual_speed_nonzero : s ≠ 0) :
  (journey_time_on_wednesday d s x) = 11 * x :=
by
  have thursday_speed : ℝ := 1.1 * s
  have thursday_time : ℝ := d / thursday_speed
  have time_diff : ℝ := (d / s) - thursday_time
  have reduced_time_eq_x : time_diff = x := by sorry
  have journey_time_eq : (d / s) = 11 * x := by sorry
  exact journey_time_eq

end journey_time_proof_l214_214961


namespace train_length_eq_400_meters_l214_214510

-- Definitions for conditions
def train_speed_kmph : ℝ := 160
def travel_time_seconds : ℝ := 9

-- Conversion factor from kmph to m/s
def kmph_to_mps (speed_kmph : ℝ) : ℝ := (speed_kmph * 1000) / 3600

-- Speed of train in m/s
def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

-- Distance calculation
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Theorem to prove
theorem train_length_eq_400_meters :
  distance train_speed_mps travel_time_seconds ≈ 400 :=
by
  sorry

end train_length_eq_400_meters_l214_214510


namespace complex_solution_l214_214881

theorem complex_solution (z : ℂ) 
  (h : (3 - complex.I) * z = 1 - complex.I) : 
  z = (2 / 5) - (1 / 5) * complex.I :=
sorry

end complex_solution_l214_214881


namespace sum_of_x_values_l214_214034

-- Definitions for the conditions given in the problem
def sin_degrees (θ : ℝ) : ℝ := Real.sin (θ * Real.pi / 180)

def equation (x : ℝ) : Prop :=
  5 * (sin_degrees (2 * x))^3 - 5 * (sin_degrees (4 * x))^3 = 2 * (sin_degrees (3 * x))^3 * (sin_degrees x)^3

def in_range (x : ℝ) : Prop := 50 < x ∧ x < 150

-- The statement of the proof problem
theorem sum_of_x_values : 
  ∑ x in {x : ℝ | equation x ∧ in_range x}, x = 450 :=
sorry

end sum_of_x_values_l214_214034


namespace injective_iff_trivial_kernel_l214_214563

variables {G H : Type} [Group G] [Group H]

-- Define the group homomorphism
variable (ϕ : G → H) [Hom : IsGroupHom ϕ]

-- Statement of the problem
theorem injective_iff_trivial_kernel :
  Function.Injective ϕ ↔ (SetOf {x : G | ϕ x = 1} = {1}) :=
sorry

end injective_iff_trivial_kernel_l214_214563


namespace seokgi_initial_money_l214_214210

theorem seokgi_initial_money (X : ℝ) (h1 : X / 2 - X / 4 = 1250) : X = 5000 := by
  sorry

end seokgi_initial_money_l214_214210


namespace correct_option_l214_214283

theorem correct_option (x y a b : ℝ) :
  ((x + 2 * y) ^ 2 ≠ x ^ 2 + 4 * y ^ 2) ∧
  ((-2 * (a ^ 3)) ^ 2 = 4 * (a ^ 6)) ∧
  (-6 * (a ^ 2) * (b ^ 5) + a * b ^ 2 ≠ -6 * a * (b ^ 3)) ∧
  (2 * (a ^ 2) * 3 * (a ^ 3) ≠ 6 * (a ^ 6)) :=
by
  sorry

end correct_option_l214_214283


namespace find_a_if_local_min_at_2_max_min_on_interval_l214_214243

-- Define the function f
def f (x a : ℝ) : ℝ := (1 / 3) * x^3 - a * x^2 + 4

-- Part 1: Show that a = 1 given local minimum at x = 2
theorem find_a_if_local_min_at_2 : 
  (∀ x, (∂/∂ x, f x 1) = 0 → x = 2 → ∀ a, f'(2 - f') = 0 → a = 1) : sorry := sorry

-- Part 2: Prove max and min values on the interval [-1, 3]
theorem max_min_on_interval :
  let f_1 (x : ℝ) := f x 1;
  (∀ x, (¬ ∀ x, {_, _, _} ∂/∂ x, max {∃ n > 0 -> ∀ y < x < d }, y = n := 4 ∧ min {∃ n < 3 -> ∀ y < x < d, min {y, d = 3 }}, y = n := 8/3) : sorry


end find_a_if_local_min_at_2_max_min_on_interval_l214_214243


namespace sum_of_smallest_angles_in_ABCD_l214_214057

theorem sum_of_smallest_angles_in_ABCD (a d x y : ℝ) 
  (h1 : 4 * a + 6 * d = 360)
  (h2 : y + 3 * x = 180)
  (h3 : ∃ angle1 angle2 angle3 angle4 : ℝ, 
           angle1 = x ∧ angle2 = 3 * x ∧ 
           16 * x = 360 ∧ angle1 + angle2 = 90) :
  angle1 + angle2 = 90 :=
by
  obtain ⟨angle1, angle2, angle3, angle4, h_angle1, h_angle2, h_sumangles, h_sum⟩ := h3
  exact h_sum

end sum_of_smallest_angles_in_ABCD_l214_214057


namespace quadratic_inequality_range_of_k_l214_214106

theorem quadratic_inequality_range_of_k :
  ∀ k : ℝ , (∀ x : ℝ, k * x^2 + 2 * k * x - (k + 2) < 0) ↔ (-1 < k ∧ k ≤ 0) :=
sorry

end quadratic_inequality_range_of_k_l214_214106


namespace ratio_of_numbers_l214_214654

theorem ratio_of_numbers (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x > y) (hsum_diff : x + y = 7 * (x - y)) : x / y = 4 / 3 := 
by
  sorry

end ratio_of_numbers_l214_214654


namespace determine_one_div_x8_l214_214353

noncomputable def given_condition : Prop :=
  let x : ℝ := sorry -- x is some real number
  log (10 * x^3) 100 + log (1000 * x^4) 10 = -1

noncomputable def target_value : ℝ :=
  10^(8 - (4 * real.sqrt 3) / 3)

theorem determine_one_div_x8 (x : ℝ) (h : log (10 * x^3) 100 + log (1000 * x^4) 10 = -1) : 
  1 / (x^8) = target_value :=
sorry

end determine_one_div_x8_l214_214353


namespace factor_fraction_l214_214382

/- Definitions based on conditions -/
variables {a b c : ℝ}

theorem factor_fraction :
  ( (a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3 ) / 
  ((a - b)^3 + (b - c)^3 + (c - a)^3) = 
  (a^2 + ab + b^2) * (b^2 + bc + c^2) * (c^2 + ca + a^2) :=
begin
  sorry
end

end factor_fraction_l214_214382


namespace inequality_solution_l214_214026

theorem inequality_solution (x : ℝ) : (5 < x ∧ x ≤ 6) ↔ (x-3)/(x-5) ≥ 3 :=
by
  sorry

end inequality_solution_l214_214026


namespace final_single_digit_of_1999_factorial_l214_214288

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else sum_of_digits (n.digits 10).sum

def digital_root (n : ℕ) : ℕ :=
  1 + ((n - 1) % 9)

theorem final_single_digit_of_1999_factorial : 
  digital_root (factorial 1999) = 9 := by
  sorry

end final_single_digit_of_1999_factorial_l214_214288


namespace same_solution_for_equations_l214_214240

theorem same_solution_for_equations (b x : ℝ) :
  (2 * x + 7 = 3) → 
  (b * x - 10 = -2) → 
  b = -4 :=
by
  sorry

end same_solution_for_equations_l214_214240


namespace total_children_l214_214856

variable (S C B T : ℕ)

theorem total_children (h1 : T < 19) 
                       (h2 : S = 3 * C) 
                       (h3 : B = S / 2) 
                       (h4 : T = B + S + 1) : 
                       T = 10 := 
  sorry

end total_children_l214_214856


namespace buffy_whiskers_l214_214802

theorem buffy_whiskers :
  ∀ (Puffy Scruffy Buffy Juniper : ℕ),
    Juniper = 12 →
    Puffy = 3 * Juniper →
    Puffy = Scruffy / 2 →
    Buffy = (Juniper + Puffy + Scruffy) / 3 →
    Buffy = 40 :=
by
  intros Puffy Scruffy Buffy Juniper hJuniper hPuffy hScruffy hBuffy
  sorry

end buffy_whiskers_l214_214802


namespace finitely_many_n_real_roots_l214_214562

def polynomial_with_real_coeffs (p : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, polynomial.eval x p ∈ ℝ

noncomputable def q_n (p : Polynomial ℝ) (n : ℕ) : Polynomial ℝ :=
  Polynomial.c (n.to_real)
  (Polynomial.X + 1)^n * Polynomial.p + Polynomial.X^n * Polynomial.eval (Polynomial.X + 1) p

theorem finitely_many_n_real_roots (p : Polynomial ℝ) (p_nonconst : degree p > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n > N → ¬ ∀ (x : ℝ), x ∈ Polynomial.roots (q_n p n) :=
sorry

end finitely_many_n_real_roots_l214_214562


namespace correct_option_l214_214824

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem correct_option : M ∪ (U \ N) = U :=
by
  sorry

end correct_option_l214_214824


namespace f_sum_positive_l214_214300

-- Define the conditions as hypotheses
variable (f : ℝ → ℝ)
variable (a d : ℕ → ℝ)
variable (H_monotone : ∀ {x y}, x ≤ y → f(x) ≤ f(y))
variable (H_odd : ∀ {x}, f(-x) = -f(x))
variable (H_arith_seq : ∀ n, a(n + 1) - a(n) = d(n))
variable (H_a3_pos : a 3 > 0)

-- State the theorem to prove the question equals the answer given the conditions
theorem f_sum_positive :
  f(a 1) + f(a 3) + f(a 5) > 0 :=
  sorry

end f_sum_positive_l214_214300


namespace right_pan_at_least_left_pan_l214_214893

theorem right_pan_at_least_left_pan (weights : ℕ → ℕ) {left_pan right_pan : Finset ℕ}
  (h_left_distinct : ∀ x ∈ left_pan, ∃ n : ℕ, x = 2^n) 
  (h_eq : left_pan.sum weights = right_pan.sum weights) :
  left_pan.card ≤ right_pan.card :=
sorry

end right_pan_at_least_left_pan_l214_214893


namespace max_min_values_l214_214811

noncomputable def problem_statement (x y : ℝ) (h : x ^ 2 + 4 * y ^ 2 = 4) : Prop :=
  x ^ 2 + 2 * x * y + 4 * y ^ 2

theorem max_min_values (x y : ℝ) (h : x ^ 2 + 4 * y ^ 2 = 4) :
  2 ≤ problem_statement x y h ∧ problem_statement x y h ≤ 6 :=
by
  sorry

end max_min_values_l214_214811


namespace S₁₅_value_l214_214139

variables {a : ℕ → ℕ}
variables {S : ℕ → ℕ}
variables (n : ℕ)

-- Definitions using the conditions of the problem
def a₈ : ℕ := a 8
def S₁₅ : ℕ := 15 * (a 1 + a 15) / 2

-- Condition
axiom a₈_value : a₈ = 8

-- The problem to prove
theorem S₁₅_value : S₁₅ = 120 :=
by 
  have ha₁₅ : a 1 + a 15 = 2 * a₈ := sorry -- from arithmetic sequence property
  rw [a₈_value] at ha₁₅
  rw [S₁₅, ha₁₅]
  calc
    15 * (2 * 8) / 2 = 15 * 8 : by sorry
                   ... = 120 : by sorry

end S₁₅_value_l214_214139


namespace cost_of_one_shirt_l214_214108

theorem cost_of_one_shirt
  (cost_J : ℕ)  -- The cost of one pair of jeans
  (cost_S : ℕ)  -- The cost of one shirt
  (h1 : 3 * cost_J + 2 * cost_S = 69)
  (h2 : 2 * cost_J + 3 * cost_S = 81) :
  cost_S = 21 :=
by
  sorry

end cost_of_one_shirt_l214_214108


namespace fencing_required_l214_214321

theorem fencing_required 
  (L : ℝ) (W : ℝ) (A : ℝ) 
  (L_eq_20 : L = 20) 
  (area_eq_400 : A = 400) 
  (area_formula : A = L * W) : 
  (F : ℝ) (fencing_eq : F = L + 2 * W) : 
  F = 60 := 
  sorry

end fencing_required_l214_214321


namespace total_notebooks_l214_214125

theorem total_notebooks (n : ℕ)
  (h₁ : n = 60)
  (h₂ : (1 / 4 : ℚ) * n = 15)
  (h₃ : (1 / 5 : ℚ) * n = 12)
  (h₄ : (1 / 3 : ℚ) * n = 20)
  : let group1_notebooks := 15 * 5,
    let group2_notebooks := 12 * 3,
    let group3_notebooks := 20 * 7,
    let group4_notebooks := (n - 15 - 12 - 20) * 4,
    let total := group1_notebooks + group2_notebooks + group3_notebooks + group4_notebooks
in total = 303 :=
by intros; subst h₁; sorry

end total_notebooks_l214_214125


namespace cars_left_after_driving_out_l214_214499

theorem cars_left_after_driving_out (total_cars first_batch_out second_batch_out : ℕ) 
(h_total : total_cars = 24) 
(h_first : first_batch_out = 8) 
(h_second : second_batch_out = 6) : 
total_cars - first_batch_out - second_batch_out = 10 :=
by simp [h_total, h_first, h_second]; sorry

end cars_left_after_driving_out_l214_214499


namespace find_angle_between_a_c_l214_214660

variables (a b c d : ℝ^3)
variables (θ : ℝ)

-- Define the magnitudes and orthogonal condition
def norm_a : ℝ := ‖a‖ = 1
def norm_b : ℝ := ‖b‖ = 1
def norm_c : ℝ := ‖c‖ = 3
def norm_d : ℝ := ‖d‖ = 1
def orthogonal_bd : ℝ := b ⬝ d = 0

-- Main equation condition
def main_eq : ℝ := a × (a × c) + b + d = 0

-- Proof problem statement
theorem find_angle_between_a_c : 
  (norm_a a) → 
  (norm_b b) → 
  (norm_c c) → 
  (norm_d d) → 
  (orthogonal_bd b d) → 
  (main_eq a b c d) →
  θ = Real.arccos (√7 / 3) ∨ θ = Real.arccos (-(√7) / 3) := 
by
  sorry

end find_angle_between_a_c_l214_214660


namespace annual_interest_rate_l214_214191

noncomputable def compound_interest 
  (P : ℝ) (A : ℝ) (n : ℕ) (t : ℝ) (r : ℝ) : Prop :=
  A = P * (1 + r / n)^(n * t)

theorem annual_interest_rate 
  (P := 140) (A := 169.40) (n := 2) (t := 1) :
  ∃ r : ℝ, compound_interest P A n t r ∧ r = 0.2 :=
sorry

end annual_interest_rate_l214_214191


namespace min_value_fraction_l214_214460

theorem min_value_fraction 
  (a_n : ℕ → ℕ)
  (S_n : ℕ → ℕ)
  (a1 a3 a13 : ℕ)
  (d : ℕ) 
  (h1 : ∀ n, a_n n = a1 + (n - 1) * d)
  (h2 : d ≠ 0)
  (h3 : a1 = 1)
  (h4 : a3 ^ 2 = a1 * a13)
  (h5 : ∀ n, S_n n = n * (a1 + a_n n) / 2) :
  ∃ n, (2 * S_n n + 16) / (a_n n + 3) = 4 := 
sorry

end min_value_fraction_l214_214460


namespace cos_A_value_l214_214119

variable {α : Type} 
variable [RealField α]

noncomputable def triangle_cos_A (a b c A B C : α) := 
  (sqrt 3 * b - c) * cos A = a * cos C

theorem cos_A_value 
  (a b c A B C : α)
  (h_triangle_cos_A : triangle_cos_A a b c A B C) :
  cos A = sqrt 3 / 3 := sorry

end cos_A_value_l214_214119


namespace probability_XOXOXOX_l214_214416

theorem probability_XOXOXOX (X O : ℕ) (h1 : X = 4) (h2 : O = 3) :
  let total_ways := Nat.choose (X + O) X,
      favorable_outcomes := 1 in
  (favorable_outcomes / total_ways : ℚ) = 1 / 35 := by
  sorry

end probability_XOXOXOX_l214_214416


namespace total_cost_correct_l214_214337

def price_per_bag_of_popcorn : ℝ := 14.70 / 5
def price_per_can_of_soda : ℝ := 2

def popcorn_count : ℕ := 4
def soda_count : ℕ := 3

def discount_popcorn : ℝ := if popcorn_count > 3 then 0.10 else 0
def discount_soda : ℝ := if soda_count > 2 then 0.05 else 0

def sales_tax_rate_popcorn : ℝ := 0.06
def sales_tax_rate_soda : ℝ := 0.07

def total_cost : ℝ :=
  let initial_popcorn_cost := popcorn_count * price_per_bag_of_popcorn
  let initial_soda_cost := soda_count * price_per_can_of_soda
  let discounted_popcorn_cost := initial_popcorn_cost * (1 - discount_popcorn)
  let discounted_soda_cost := initial_soda_cost * (1 - discount_soda)
  let taxed_popcorn_cost := discounted_popcorn_cost * (1 + sales_tax_rate_popcorn) 
  let taxed_soda_cost := discounted_soda_cost * (1 + sales_tax_rate_soda)
  taxed_popcorn_cost + taxed_soda_cost

theorem total_cost_correct :
  (total_cost : ℝ) = 17.32 := by
  sorry

end total_cost_correct_l214_214337


namespace triangle_DEF_area_l214_214908

noncomputable def triangleArea (a b h : ℕ) : ℕ :=
  (a * b) / 2

theorem triangle_DEF_area : 
    ∀ (DE LF EF DL : ℕ) (h1 : DE = 15) (h2 : LF = 9) (h3 : EF = 18) (h4 : DL^2 + LF^2 = DE^2),
    triangleArea EF DL = 108 :=
begin
    intros DE LF EF DL h1 h2 h3 h4,
    sorry -- Proof goes here
end

end triangle_DEF_area_l214_214908


namespace find_divisor_l214_214705

theorem find_divisor (n : ℕ) (h_n : n = 36) : 
  ∃ D : ℕ, ((n + 10) * 2 / D) - 2 = 44 → D = 2 :=
by
  use 2
  intros h
  sorry

end find_divisor_l214_214705


namespace hyperbola_through_points_l214_214310

theorem hyperbola_through_points (s : ℝ) :
  let a := real.sqrt 7 
  let eq_hyperbola := λ x y : ℝ, y^2 / 4 - x^2 / a^2 = 1
  eq_hyperbola 0 (-2) ∧ eq_hyperbola 3 4 ∧ eq_hyperbola 2 s →
  s^2 = 44 / 7 :=
by {
  intros h,
  sorry
}

end hyperbola_through_points_l214_214310


namespace octahedron_side_length_l214_214710

theorem octahedron_side_length (A1 A2 A3 A4 A1' A2' A3' A4' : EuclideanSpace ℝ (Fin 3)) :
  (dist A1 A2 = 2) ∧ (dist A1 A3 = 2) ∧ (dist A1 A4 = 2) ∧ 
  (A1' = A1 + ⟨2, 2, 2⟩) ∧ (A2' = A2 + ⟨2, 2, 2⟩) ∧ (A3' = A3 + ⟨2, 2, 2⟩) ∧ (A4' = A4 + ⟨2, 2, 2⟩) ∧
  (∃ V1 V2 V3 V4 V1' V2' V3' V4', 
    V1 = A1 + ⟨2/3, 0, 0⟩ ∧ V2 = A1' + ⟨-2/3, 0, 0⟩ ∧ V3 = A1 + ⟨0, 2/3, 0⟩ ∧ V4 = A1 + ⟨0, 0, 2/3⟩ ∧ 
    V1' = A1' + ⟨0, 4/3, 0⟩ ∧ V2' = A2' + ⟨0, 4/3, 0⟩ ∧ V3' = A3' + ⟨0, 4/3, 0⟩ ∧ V4' = A4' + ⟨0, 4/3, 0⟩) →
  ∃ s, s = 2 * sqrt 17 / 3 := 
by {
  sorry
}

end octahedron_side_length_l214_214710


namespace triangle_c_is_3_l214_214122

noncomputable def triangle_side_c (a b : ℝ) (angle_B : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2 - 2 * a * b * real.cos angle_B)

theorem triangle_c_is_3 (a b : ℝ) (angle_B : ℝ) (h_a : a = 2) (h_b : b = real.sqrt 7) (h_B : angle_B = real.pi / 3) :
  triangle_side_c a b angle_B = 3 :=
by
  rw [h_a, h_b, h_B]
  -- Here, it would follow the math solution steps to conclude the proof.
  sorry

end triangle_c_is_3_l214_214122


namespace probability_not_black_correct_l214_214702

def total_balls : ℕ := 8 + 9 + 3

def non_black_balls : ℕ := 8 + 3

def probability_not_black : ℚ := non_black_balls / total_balls

theorem probability_not_black_correct : probability_not_black = 11 / 20 := 
by 
  -- The proof would go here
  sorry

end probability_not_black_correct_l214_214702


namespace probability_XOXOXOX_l214_214426

noncomputable def binomial (n k : ℕ) : ℕ := nat.choose n k

theorem probability_XOXOXOX :
  let num_X := 4,
      num_O := 3,
      total_arrangements := binomial 7 num_X in
  total_arrangements = 35 ∧
  1 / total_arrangements = (1 : ℚ) / 35 :=
by
  sorry

end probability_XOXOXOX_l214_214426


namespace five_star_two_l214_214250

def star (a b : ℝ) : ℝ := a^2 + a / b

theorem five_star_two : star 5 2 = 27.5 :=
by
  -- sorry is used to skip the proof
  sorry

end five_star_two_l214_214250


namespace total_books_l214_214208

variable (Sandy Benny Tim Rachel Alex Jordan : ℕ)

-- Conditions from the problem
def book_conditions : Prop :=
  Sandy = 10 ∧
  Benny = 24 ∧
  Tim = 33 ∧
  Rachel = 2 * Benny ∧
  Alex = (Tim / 2).toNat - 3 ∧
  Jordan = Sandy + Benny

-- The theorem that they together have 162 books
theorem total_books (h : book_conditions Sandy Benny Tim Rachel Alex Jordan) : 
  Sandy + Benny + Tim + Rachel + Alex + Jordan = 162 := 
sorry

end total_books_l214_214208


namespace lambda_value_l214_214854

-- Definitions based on the conditions
def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
def vec_b (y : ℝ) : ℝ × ℝ := (Real.cos y, -Real.sin y)
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2
def magnitude (a : ℝ × ℝ) : ℝ := Real.sqrt (a.1^2 + a.2^2)
def f (x λ : ℝ) : ℝ := dot_product (vec_a x) (vec_b x) - 2 * λ * magnitude (Real.cos x + Real.cos x, Real.sin x - Real.sin x)

-- Problem statement
theorem lambda_value (λ : ℝ) :
  (∃ x ∈ Set.Icc 0 Real.pi, f x λ = -3) ↔ λ = 1 / Real.sqrt 2 :=
sorry

end lambda_value_l214_214854


namespace math_problem_l214_214176

def f (x : ℝ) (a b c d : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

def condition_f_eq (a b c d : ℝ) : Prop :=
  a ≠ 0 ∧ (f 1 a b c d) + (f 3 a b c d) = 2 * (f 2 a b c d)

def statement1 (fx : ℝ → ℝ) : Prop :=
  ∀ x, (x ∈ set.Ioo 0 1 → fx x > 0) → (x ∈ set.Ioo 3 4 → fx x > 0)

def statement2 (fx : ℝ → ℝ) (a b : ℝ) : Prop :=
  (a * fx 1 ≥ a * fx 3) → (∃ x, fx x = 0)

def statement3 (fx : ℝ → ℝ) (a b c d : ℝ) (x0 : ℝ) : Prop :=
  ∃ x, (6 * a * x - 12 * a = 0) ∧ (c - 12 * a)(x - x0) + fx x0 = fx x

theorem math_problem (a b c d : ℝ) (fx : ℝ → ℝ) (x0 : ℝ) :
  condition_f_eq a b c d →
  statement1 (f a (-6 * a) c d) →
  statement2 (f a (-6 * a) c d) a →
  statement3 (f a (-6 * a) c d) a (-6 * a) c d x0 →
  (∃ N : ℕ, N = 3) :=
by
  sorry

end math_problem_l214_214176


namespace proof_simplify_expression_l214_214573

noncomputable def simplify_expression (a b : ℝ) : ℝ :=
  (a / b + b / a)^2 - 1 / (a^2 * b^2)

theorem proof_simplify_expression 
  (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^3 + b^3 = a + b) :
  simplify_expression a b = 2 / (a * b) := by
  sorry

end proof_simplify_expression_l214_214573


namespace investment_amount_approx_l214_214867

constant FV : ℝ := 600000
constant r : ℝ := 0.06
constant n : ℕ := 15
noncomputable def PV := FV / (1 + r)^n

theorem investment_amount_approx :
  PV ≈ 250310.77 := by
  sorry

end investment_amount_approx_l214_214867


namespace find_k_l214_214582

noncomputable def vector_space := ℝ

structure unit_vector (v : vector_space) :=
(norm_eq_one : ∥v∥ = 1)

variables (e1 e2 : vector_space) (k : ℝ)

def angle_cosine (θ : ℝ) := real.cos θ

def dot_product (v1 v2 : vector_space) := v1 * v2 -- Placeholder for the actual dot product calculation

axiom unit_vectors {e1 e2 : vector_space} (h1 : unit_vector e1) (h2 : unit_vector e2) : 
  dot_product e1 e2 = angle_cosine (2 * real.pi / 3)

axiom perpendicularity_condition (e1 e2 : vector_space) (k : ℝ) :
  dot_product (e1 - 2 * e2) (k * e1 + e2) = 0 → k = 5 / 4

theorem find_k (e1 e2 : vector_space) (k : ℝ) (h1 : unit_vector e1) (h2 : unit_vector e2) :
  angle_cosine (2 * real.pi / 3) = dot_product e1 e2 →
  dot_product (e1 - 2 * e2) (k * e1 + e2) = 0 → k = 5 / 4 := 
sorry

end find_k_l214_214582


namespace range_of_exponential_function_l214_214642

theorem range_of_exponential_function :
  (∃ y : ℚ, (y = (1 / 2)^x ∧ x ≥ 8) ↔ y ∈ (set.Ioc 0 (1 / 256))) :=
sorry

end range_of_exponential_function_l214_214642


namespace debt_limit_l214_214561

theorem debt_limit (daily_cost number_of_days payment initial_balance : ℝ)
  (h_daily_cost : daily_cost = 0.5)
  (h_number_of_days : number_of_days = 25)
  (h_payment : payment = 7)
  (h_initial_balance : initial_balance = 0) :
  initial_balance + payment < daily_cost * number_of_days →
  (daily_cost * number_of_days - payment) = 5.5 :=
by
  intros h
  rw [h_daily_cost, h_number_of_days, h_payment, h_initial_balance]
  sorry

end debt_limit_l214_214561


namespace geometric_seq_xyz_eq_neg_two_l214_214076

open Real

noncomputable def geometric_seq (a b c d e : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r ∧ e = d * r

theorem geometric_seq_xyz_eq_neg_two (x y z : ℝ) :
  geometric_seq (-1) x y z (-2) → x * y * z = -2 :=
by
  intro h
  obtain ⟨r, hx, hy, hz, he⟩ := h
  rw [hx, hy, hz, he] at *
  sorry

end geometric_seq_xyz_eq_neg_two_l214_214076


namespace john_pays_after_discount_l214_214912

theorem john_pays_after_discount :
  ∀ (num_toys : ℕ) (cost_per_toy : ℕ) (discount_rate : ℚ),
  num_toys = 5 → cost_per_toy = 3 → discount_rate = 0.20 →
  let total_cost := num_toys * cost_per_toy in
  let discount := discount_rate * ↑total_cost in
  let amount_paid := total_cost - discount in
  amount_paid = 12 :=
by
  intros num_toys cost_per_toy discount_rate hnum_toys hcost_per_toy hdiscount_rate
  rw [hnum_toys, hcost_per_toy, hdiscount_rate]
  let total_cost := num_toys * cost_per_toy
  let discount := discount_rate * ↑total_cost
  let amount_paid := total_cost - discount
  have htotal_cost : total_cost = 15 := by norm_num
  have hdiscount : discount = 3 := by norm_num
  have hamount_paid : amount_paid = 12 := by norm_num
  exact hamount_paid

end john_pays_after_discount_l214_214912


namespace parakeet_eats_2_grams_per_day_l214_214201

-- Define the conditions
def parrot_daily : ℕ := 14
def finch_daily (parakeet_daily : ℕ) : ℕ := parakeet_daily / 2
def num_parakeets : ℕ := 3
def num_parrots : ℕ := 2
def num_finches : ℕ := 4
def total_weekly_consumption : ℕ := 266

-- Define the daily consumption equation for all birds
def daily_consumption (parakeet_daily : ℕ) : ℕ :=
  num_parakeets * parakeet_daily + num_parrots * parrot_daily + num_finches * finch_daily parakeet_daily

-- Define the weekly consumption equation
def weekly_consumption (parakeet_daily : ℕ) : ℕ :=
  7 * daily_consumption parakeet_daily

-- State the theorem to prove that each parakeet eats 2 grams per day
theorem parakeet_eats_2_grams_per_day :
  (weekly_consumption 2) = total_weekly_consumption ↔ 2 = 2 :=
by
  sorry

end parakeet_eats_2_grams_per_day_l214_214201


namespace sin_three_pi_four_l214_214024

theorem sin_three_pi_four :
  sin (3 * π / 4) = (sqrt 2) / 2 :=
by
  have h1 : (3:ℝ) * π / 4 = π / 2 + π / 4 := by ring
  rw [h1, Real.sin_add]
  have h2 : sin (π / 2) = 1 := Real.sin_pi_div_two
  have h3 : cos (π / 2) = 0 := Real.cos_pi_div_two
  have h4 : sin (π / 4) = sqrt 2 / 2 := Real.sin_pi_div_four
  have h5 : cos (π / 4) = sqrt 2 / 2 := Real.cos_pi_div_four
  rw [h2, h3, h4, h5]
  norm_num

end sin_three_pi_four_l214_214024


namespace probability_XOXOXOX_is_1_div_35_l214_214422

def count_combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def num_ways_to_choose_positions_for_X (total_positions : ℕ) (num_X : ℕ) : ℕ := 
  count_combinations total_positions num_X

def num_ways_for_specific_arrangement_XOXOXOX : ℕ := 1

def probability_of_XOXOXOX (num_ways_total : ℕ) (num_ways_specific : ℕ) : ℚ := 
  num_ways_specific / num_ways_total

theorem probability_XOXOXOX_is_1_div_35 :
  probability_of_XOXOXOX (num_ways_to_choose_positions_for_X 7 4) num_ways_for_specific_arrangement_XOXOXOX = 1 / 35 := by
  sorry

end probability_XOXOXOX_is_1_div_35_l214_214422


namespace proof_problem_l214_214885

noncomputable def triangle_proof_problem (A B C O I D E M : Type) 
  [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq O] [decidable_eq I] [decidable_eq D] [decidable_eq E] [decidable_eq M]
  (triangle_ABC : triangle A B C)
  (circumcenter_O : circumcenter O A B C)
  (incenter_I : incenter I A B C)
  (D_on_AC : point_on_line D A C)
  (E_on_BC : point_on_line E B C)
  (AD_eq_BE_eq_AB : length A D = length B E ∧ length A D = length A B ∧ length B E = length A B)
  (angle_C_30 : angle_measure A B C = 30) 
  (point_M : point_on_circumcircle_extension M A I circumcenter_O)
: Prop :=
(OI_perp_DE : perpendicular O I D E) ∧ (length O I = length D E)

-- The theorem statement that needs to be proven
theorem proof_problem : ∀ (A B C O I D E M : Type) 
  [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq O] [decidable_eq I] [decidable_eq D] [decidable_eq E] [decidable_eq M]
  (triangle_ABC : triangle A B C)
  (circumcenter_O : circumcenter O A B C)
  (incenter_I : incenter I A B C)
  (D_on_AC : point_on_line D A C)
  (E_on_BC : point_on_line E B C)
  (AD_eq_BE_eq_AB : length A D = length B E ∧ length A D = length A B ∧ length B E = length A B)
  (angle_C_30 : angle_measure A B C = 30) 
  (point_M : point_on_circumcircle_extension M A I circumcenter_O),
  triangle_proof_problem A B C O I D E M triangle_ABC circumcenter_O incenter_I D_on_AC E_on_BC AD_eq_BE_eq_AB angle_C_30 point_M :=
sorry

end proof_problem_l214_214885


namespace probability_XOXOXOX_l214_214430

noncomputable def binomial (n k : ℕ) : ℕ := nat.choose n k

theorem probability_XOXOXOX :
  let num_X := 4,
      num_O := 3,
      total_arrangements := binomial 7 num_X in
  total_arrangements = 35 ∧
  1 / total_arrangements = (1 : ℚ) / 35 :=
by
  sorry

end probability_XOXOXOX_l214_214430


namespace minimum_value_of_n_l214_214115

theorem minimum_value_of_n : ∃ n ∈ ℕ, (∀ r ∈ ℕ, 2 * n = (7 * r) / 3) ∧ n = 7 :=
by 
  sorry

end minimum_value_of_n_l214_214115


namespace cans_purchased_l214_214229

variable (N P T : ℕ)

theorem cans_purchased (N P T : ℕ) : N * (5 * (T - 1)) / P = 5 * N * (T - 1) / P :=
by
  sorry

end cans_purchased_l214_214229


namespace monotonically_increasing_sine_function_range_l214_214468

theorem monotonically_increasing_sine_function_range (ω : ℝ) (hω : ω > 0) :
  ∀ x y : ℝ, (x ∈ set.Icc (-π/3) (π/4)) → (y ∈ set.Icc (-π/3) (π/4)) 
  → x ≤ y → (2 * sin(ω * x)) ≤ (2 * sin(ω * y)) ↔ (0 < ω ∧ ω ≤ 12 / 7) := 
sorry

end monotonically_increasing_sine_function_range_l214_214468


namespace isosceles_right_triangle_rotation_volumes_l214_214069

noncomputable def volume_of_solid_rotated_around_leg 
  (leg_length : ℝ) 
  (h : leg_length = 1) : ℝ :=
(1 / 3) * π * leg_length^2 * leg_length

noncomputable def volume_of_solid_rotated_around_hypotenuse 
  (leg_length : ℝ) 
  (h : leg_length = 1) : ℝ :=
2 * (1 / 3) * π * (leg_length / √2)^2 * (leg_length / √2)

theorem isosceles_right_triangle_rotation_volumes
  (leg_length : ℝ)
  (h : leg_length = 1) :
  (volume_of_solid_rotated_around_leg leg_length h = π / 3) ∨ 
  (volume_of_solid_rotated_around_hypotenuse leg_length h = (√2 * π) / 6) :=
by
  sorry

end isosceles_right_triangle_rotation_volumes_l214_214069


namespace other_discount_l214_214291

theorem other_discount (list_price : ℝ) (final_price : ℝ) (first_discount : ℝ) (other_discount : ℝ) :
  list_price = 70 → final_price = 61.74 → first_discount = 10 → (list_price * (1 - first_discount / 100) * (1 - other_discount / 100) = final_price) → other_discount = 2 := 
by
  intros h1 h2 h3 h4
  sorry

end other_discount_l214_214291


namespace golf_balls_dozen_count_l214_214209

theorem golf_balls_dozen_count (n d : Nat) (h1 : n = 108) (h2 : d = 12) : n / d = 9 :=
by
  sorry

end golf_balls_dozen_count_l214_214209


namespace obtuse_angles_at_intersection_l214_214273

theorem obtuse_angles_at_intersection (lines_intersect_x_at_diff_points : Prop) (lines_not_perpendicular : Prop) 
(lines_form_obtuse_angle_at_intersection : Prop) : 
(lines_intersect_x_at_diff_points ∧ lines_not_perpendicular ∧ lines_form_obtuse_angle_at_intersection) → 
  ∃ (n : ℕ), n = 2 :=
by 
  sorry

end obtuse_angles_at_intersection_l214_214273


namespace solve_system_of_equations_solve_linear_inequality_l214_214297

-- Part 1: System of equations
theorem solve_system_of_equations (x y : ℝ) (h1 : 5 * x + 2 * y = 25) (h2 : 3 * x + 4 * y = 15) : 
  x = 5 ∧ y = 0 := sorry

-- Part 2: Linear inequality
theorem solve_linear_inequality (x : ℝ) (h : 2 * x - 6 < 3 * x) : 
  x > -6 := sorry

end solve_system_of_equations_solve_linear_inequality_l214_214297


namespace sum_of_cube_faces_l214_214711

-- Define the cube numbers as consecutive integers starting from 15.
def cube_faces (faces : List ℕ) : Prop :=
  faces = [15, 16, 17, 18, 19, 20]

-- Define the condition that the sum of numbers on opposite faces is the same.
def opposite_faces_condition (pairs : List (ℕ × ℕ)) : Prop :=
  ∀ (p : ℕ × ℕ) (hp : p ∈ pairs), (p.1 + p.2) = 35

theorem sum_of_cube_faces : ∃ faces : List ℕ, cube_faces faces ∧ (∃ pairs : List (ℕ × ℕ), opposite_faces_condition pairs ∧ faces.sum = 105) :=
by
  sorry

end sum_of_cube_faces_l214_214711


namespace total_miles_driven_l214_214530

theorem total_miles_driven (miles_per_gallon_car1 : ℕ) (gallons_consumed_car1 : ℕ) 
                           (miles_per_gallon_car2 : ℕ) (total_gas_consumed : ℕ)
                           (gallons_consumed_car1' : ℕ) :
  (miles_per_gallon_car1 = 25) ∧ (gallons_consumed_car1 = 30) ∧ 
  (miles_per_gallon_car2 = 40) ∧ (total_gas_consumed = 55) ∧ 
  (gallons_consumed_car1' = 30) →
  let m1 := miles_per_gallon_car1 * gallons_consumed_car1 in
  let m2 := miles_per_gallon_car2 * (total_gas_consumed - gallons_consumed_car1') in
  m1 + m2 = 1750 :=
by
  sorry

end total_miles_driven_l214_214530


namespace sum_of_derivatives_of_zeros_gt_zero_l214_214712

noncomputable def cubic_polynomial := 
  { f : Polynomial ℝ // Polynomial.degree f = 3 ∧ 
                      leading_coeff f > 0 }

variable {f : cubic_polynomial}

def has_three_positive_zeros (f : Polynomial ℝ) (a b c : ℝ) :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧
  f.aeval a = 0 ∧ f.aeval b = 0 ∧ f.aeval c = 0

theorem sum_of_derivatives_of_zeros_gt_zero (a b c : ℝ) 
  (h_zeros : has_three_positive_zeros f.val a b c) : 
  f.val.derivative.eval a + f.val.derivative.eval b + f.val.derivative.eval c > 0 := 
sorry

end sum_of_derivatives_of_zeros_gt_zero_l214_214712


namespace expected_swaps_floor_l214_214717

noncomputable def expected_swaps (n : ℕ) := ∑ k in Finset.range n, (k / 2 : ℚ)

theorem expected_swaps_floor : (⌊expected_swaps 99⌋ : ℕ) = 2425 := by
  sorry

end expected_swaps_floor_l214_214717


namespace hyperbola_eccentricity_l214_214457

theorem hyperbola_eccentricity (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : (a/b) * (2/1) = 2 * sqrt 3) : 
  (sqrt ((a^2 + b^2)/a^2)) = sqrt 13 := by 
sorry

end hyperbola_eccentricity_l214_214457


namespace maximum_value_of_f_for_n_le_1991_l214_214172

def f : ℕ → ℕ 
| 0       := 0
| 1       := 1
| (n + 2) := f (n / 2) + n % 2 -- this is translated from f(n) = f(floor(n/2)) + n - 2*floor(n/2)

theorem maximum_value_of_f_for_n_le_1991 : ∀ (_ : 0 ≤ n ∧ n ≤ 1991), f n ≤ 10 ∧ (∃ n, 0 ≤ n ∧ n ≤ 1991 ∧ f n = 10) :=
by
  sorry

end maximum_value_of_f_for_n_le_1991_l214_214172


namespace is_largest_interesting_l214_214280

def is_interesting (n : ℕ) : Prop :=
  let digits := (n.digits 10).reverse
  ∀ i, 1 ≤ i ∧ i < digits.length - 1 → 2 * (digits.get ⟨i, by simp [digits]⟩) < (digits.get ⟨i-1, by simp [digits]⟩ + digits.get ⟨i+1, by simp [digits]⟩)

def largest_interesting_number : ℕ :=
  96433469

theorem is_largest_interesting : is_interesting (largest_interesting_number) ∧ ∀ n, is_interesting n → n ≤ largest_interesting_number := 
  by
    sorry

end is_largest_interesting_l214_214280


namespace sqrt_total_stripes_approx_l214_214587

def stripes_per_shoe_olga : ℕ := 3
def stripes_per_shoe_rick : ℕ := stripes_per_shoe_olga - 1
def stripes_per_shoe_hortense : ℕ := stripes_per_shoe_olga * 2
def stripes_per_shoe_ethan : ℕ := stripes_per_shoe_hortense + 2

def avg_stripes_per_shoe : ℝ :=
  (stripes_per_shoe_olga + stripes_per_shoe_rick + stripes_per_shshoe h ortense + stripes_per_shoe_ethan) / 4

def stripes_per_shoe_sierra : ℕ :=
  Int.floor (avg_stripes_per_shoe - 3)

def total_stripes : ℕ :=
  2 * (stripes_per_shoe_olga +
       stripes_per_shoe_rick +
       stripes_per_shoe_hortense +
       stripes_per_shoe_ethan +
       stripes_per_shoe_sierra)

def sqrt_total_stripes : ℝ := Real.sqrt total_stripes

theorem sqrt_total_stripes_approx : sqrt_total_stripes ≈ 6.32 := 
by 
  sorry

end sqrt_total_stripes_approx_l214_214587


namespace mixture_weight_l214_214622

theorem mixture_weight 
  (C : ℝ) -- Cost per pound of green tea and coffee in June
  (coffee_price_increase : 100)       -- Coffee price increased by 100%
  (green_tea_price_drop : 70)         -- Green tea price dropped by 70%
  (mixture_cost : ℝ)                  -- Mixture cost
  (green_tea_price_july : ℝ)          -- Price of green tea in July
  (equal_mix_cost : ℝ)                -- Cost of equal mixture
  (mixture_total_cost : ℝ)            -- Total cost of the mixture
  
  (h1 : coffee_price_increase = 100)
  (h2 : green_tea_price_drop = 70)
  (h3 : equal_mix_cost = 3.45)
  (h4 : green_tea_price_july = 0.3)
  (h5 : mixture_total_cost = 3.45) :
  mixture_total_cost / ((green_tea_price_july + (C * 2)) / 2) = 3 :=
by 
  sorry

end mixture_weight_l214_214622


namespace pipe_B_to_A_ratio_l214_214725

-- Definitions of the rates of the pipes
def rate_A : ℝ := 1 / 28
def rate_B : ℝ
def rate_C : ℝ := 2 * rate_B
def combined_rate : ℝ := rate_A + rate_B + rate_C

-- Theorem to prove the ratio of the speed of pipe B to the speed of pipe A
theorem pipe_B_to_A_ratio :
  (combined_rate = 1 / 4) →
  (rate_C = 2 * rate_B) →
  (rate_A = 1 / 28) →
  (rate_B / rate_A = 2) :=
by
  intro h1 h2 h3
  -- placeholder for proof
  sorry

end pipe_B_to_A_ratio_l214_214725


namespace cos_angle_between_vectors_l214_214826

-- Definitions for the vectors and angle
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (4, 2)
def θ : ℝ := sorry -- θ is the angle between a and b

-- The proof statement
theorem cos_angle_between_vectors :
  let dot_product := (a.1 * b.1 + a.2 * b.2)
  let mag_a := real.sqrt (a.1 ^ 2 + a.2 ^ 2)
  let mag_b := real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  cos θ = dot_product / (mag_a * mag_b) := by
  let dot_product := (1 * 4 + 2 * 2)
  let mag_a := real.sqrt (1 ^ 2 + 2 ^ 2)
  let mag_b := real.sqrt (4 ^ 2 + 2 ^ 2)
  have h_dot_product : dot_product = 8 := by simp [dot_product]
  have h_mag_a : mag_a = real.sqrt 5 := by simp [mag_a]
  have h_mag_b : mag_b = 2 * real.sqrt 5 := by simp [mag_b]
  have h_cos_theta : cos θ = dot_product / (mag_a * mag_b) := by
    rw [h_dot_product, h_mag_a, h_mag_b]
    have h : cos θ = 8 / (real.sqrt 5 * (2 * real.sqrt 5)) := by simp 
    rw [real.sqrt_mul real.sqrt_nonneg real.sqrt_nonneg]
    exact h 
  exact h_cos_theta

end cos_angle_between_vectors_l214_214826


namespace find_y_l214_214715

noncomputable def x : Real := 1.6666666666666667
def y : Real := 5

theorem find_y (h : x ≠ 0) (h1 : (x * y) / 3 = x^2) : y = 5 := 
by sorry

end find_y_l214_214715


namespace expression_equals_20_over_9_l214_214745

noncomputable def complex_fraction_expression := 
  let a := 11 + 1 / 9
  let b := 3 + 2 / 5
  let c := 1 + 2 / 17
  let d := 8 + 2 / 5
  let e := 3.6
  let f := 2 + 6 / 25
  ((a - b * c) - d / e) / f

theorem expression_equals_20_over_9 : complex_fraction_expression = 20 / 9 :=
by
  sorry

end expression_equals_20_over_9_l214_214745


namespace probability_XOXOXOX_is_one_over_thirty_five_l214_214438

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_arrangement : ℤ :=
  let total_arrangements := binomial_coefficient 7 4
  let favorable_outcomes := 1
  favorable_outcomes / total_arrangements

theorem probability_XOXOXOX_is_one_over_thirty_five :
  probability_of_arrangement = (1 : ℤ) / 35 := 
  by
  sorry

end probability_XOXOXOX_is_one_over_thirty_five_l214_214438


namespace probability_of_perfect_square_l214_214974

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem probability_of_perfect_square :
  let total_outcomes := 8 in
  let perfect_squares := { n | n ∈ {1, 4} } in
  let successful_outcomes := finset.card perfect_squares in
  (successful_outcomes / total_outcomes : ℝ) = (1 / 4 : ℝ) :=
by
  sorry

end probability_of_perfect_square_l214_214974


namespace how_many_years_for_Audrey_to_be_twice_as_old_l214_214338

variables (H_age A_age x : ℕ)
variables (h1 : A_age = H_age + 7)
variables (h2 : H_age = 10)
variables (h3 : A_age + x = 20)

theorem how_many_years_for_Audrey_to_be_twice_as_old :
  x = 3 :=
by {
  -- Definitions directly from conditions
  have hA : A_age = 17, from by { rw [h2, h1], norm_num },
  rw hA at h3,
  simp at h3,
  exact h3.symm
}

end how_many_years_for_Audrey_to_be_twice_as_old_l214_214338


namespace repeating_decimal_subtraction_l214_214346

noncomputable def repeating_decimal_to_fraction (n : ℕ) : ℚ := 
  let k := 10 ^ (nat.log10 n + 1)
  (n : ℚ) / (k -1)

-- Definitions of repeating decimals
def a : ℚ := repeating_decimal_to_fraction 234
def b : ℚ := repeating_decimal_to_fraction 567
def c : ℚ := repeating_decimal_to_fraction 891

-- Theorem statement
theorem repeating_decimal_subtraction :
  a - b - c = -1224 / 999 :=
by {
  sorry
}

end repeating_decimal_subtraction_l214_214346


namespace andy_coats_l214_214333

theorem andy_coats 
  (initial_minks : ℕ)
  (offspring_4_minks count_4_offspring : ℕ)
  (offspring_6_minks count_6_offspring : ℕ)
  (offspring_8_minks count_8_offspring : ℕ)
  (freed_percentage coat_requirement total_minks offspring_minks freed_minks remaining_minks coats : ℕ) :
  initial_minks = 30 ∧
  offspring_4_minks = 10 ∧ count_4_offspring = 4 ∧
  offspring_6_minks = 15 ∧ count_6_offspring = 6 ∧
  offspring_8_minks = 5 ∧ count_8_offspring = 8 ∧
  freed_percentage = 60 ∧ coat_requirement = 15 ∧
  total_minks = initial_minks + offspring_minks ∧
  offspring_minks = offspring_4_minks * count_4_offspring + offspring_6_minks * count_6_offspring + offspring_8_minks * count_8_offspring ∧
  freed_minks = total_minks * freed_percentage / 100 ∧
  remaining_minks = total_minks - freed_minks ∧
  coats = remaining_minks / coat_requirement →
  coats = 5 :=
sorry

end andy_coats_l214_214333


namespace value_of_4m_plus_2n_l214_214474

-- Given that the equation 2kx + 2m = 6 - 2x + nk 
-- has a solution independent of k
theorem value_of_4m_plus_2n (m n : ℝ) 
  (h : ∃ x : ℝ, ∀ k : ℝ, 2 * k * x + 2 * m = 6 - 2 * x + n * k) : 
  4 * m + 2 * n = 12 :=
by
  sorry

end value_of_4m_plus_2n_l214_214474


namespace third_circle_radius_l214_214594

-- Definitions
variables {R x : ℝ}
variables {O1 O2 O3 M A : ℝ → ℝ}

-- Conditions
def Circle1 (O1 : ℝ → ℝ) (R : ℝ) : Prop := ∀ P : ℝ → ℝ, dist O1 P = R
def Circle2 (O2 : ℝ → ℝ) (R : ℝ) : Prop := ∀ P : ℝ → ℝ, dist O2 P = R / 2
def Circle3 (O3 : ℝ → ℝ) (x : ℝ) : Prop := ∀ P : ℝ → ℝ, dist O3 P = x
def Tangent (C1 C2 : ℝ → ℝ → Prop) (P : ℝ → ℝ) : Prop := dist C1 P = dist C2 P

noncomputable def radius_of_third_circle (R : ℝ) := 4 * R / 9

-- Theorem
theorem third_circle_radius (R : ℝ) (O1 O2 O3 : ℝ → ℝ) (x : ℝ) :
  ∃ A M : ℝ → ℝ, 
  Circle1 O1 R ∧ Circle2 O2 R ∧ Circle3 O3 x ∧ 
  Tangent O1 A ∧ Tangent O3 M ∧
  dist O1 O3 = √(R^2 - 2*R*x + x^2) ∧
  dist O2 O3 = O2 (x + R / 2)  ∧
  radius_of_third_circle R = x :=
sorry

end third_circle_radius_l214_214594


namespace inequality_proof_l214_214810

theorem inequality_proof (n : ℕ)
  (a : ℕ → ℝ) 
  (h1 : ∀ i j : ℕ, i < j → a i < a j) -- condition: a1 < a2 < ... < a2n

  (h2 : 0 < n) : 
  let S := ∑ i in range n, a (2 * i)
  let T := ∑ i in range n, a (2 * i + 1)
  in 
  2 * S * T > sqrt ((2 * n / (n - 1)) * (S + T) * 
  (S * (∑ i in range n, ∑ j in range n, if 1 < i ∧ i < j ∧ j ≤ n then a (2 * i) * a (2 * j) else 0) + 
   T * (∑ i in range n, ∑ j in range n, if 1 ≤ i ∧ i < j ∧ j ≤ n then a (2 * i - 1) * a (2 * j - 1) else 0))) := 
sorry

end inequality_proof_l214_214810


namespace count_six_digit_numbers_with_three_even_three_odd_l214_214864

-- Definitions of the problem conditions
def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

def has_three_even_and_three_odd_digits (n : ℕ) : Prop :=
  (nat.num_even_digits n = 3) ∧ (nat.num_odd_digits n = 3)

-- Lean 4 statement
theorem count_six_digit_numbers_with_three_even_three_odd :
  ∃ (count : ℕ), count = 281250 ∧
  (∀ n : ℕ, is_six_digit n ∧ n / 100000 > 0 ∧ has_three_even_and_three_odd_digits n → True) :=
sorry

end count_six_digit_numbers_with_three_even_three_odd_l214_214864


namespace susan_average_speed_correct_l214_214232

-- Let's define the distances in miles and speeds in mph
def first_segment_distance : ℝ := 50
def first_segment_speed : ℝ := 30

def second_segment_distance : ℝ := 100
def second_segment_speed : ℝ := 60

def break_time_1 : ℝ := 0.5 -- in hours, 30 minutes

def third_segment_distance : ℝ := 100
def third_segment_speed : ℝ := 50

def fourth_segment_distance : ℝ := 50
def fourth_segment_speed : ℝ := 40

def break_time_2 : ℝ := 0.25 -- in hours, 15 minutes

-- Total distance covered
def total_distance : ℝ := first_segment_distance + second_segment_distance + third_segment_distance + fourth_segment_distance

-- Total time taken
def total_time : ℝ :=
  (first_segment_distance / first_segment_speed) +
  (second_segment_distance / second_segment_speed) +
  break_time_1 +
  (third_segment_distance / third_segment_speed) +
  (fourth_segment_distance / fourth_segment_speed) +
  break_time_2

-- Susan's average speed
def susan_average_speed : ℝ := total_distance / total_time

theorem susan_average_speed_correct :
  susan_average_speed ≈ 40.92 :=
sorry

end susan_average_speed_correct_l214_214232


namespace max_OP_OQ_l214_214899

noncomputable def C1_polar_eq (theta : ℝ) : ℝ := 4 * Real.cos theta
noncomputable def C2_polar_eq (theta : ℝ) : ℝ := 2 * Real.sin theta

theorem max_OP_OQ (alpha : ℝ) :
  let OP := Real.sqrt (8 + 8 * Real.cos alpha),
      OQ := Real.sqrt (2 + 2 * Real.sin alpha) in
  OP * OQ ≤ 4 + 2 * Real.sqrt 2 :=
sorry

end max_OP_OQ_l214_214899


namespace max_f_value_l214_214982

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x / (x^2 + m)

theorem max_f_value (m : ℝ) : 
  (m > 1) ↔ (∀ x : ℝ, f x m < 1) ∧ ¬((∀ x : ℝ, f x m < 1) → (m > 1)) :=
by
  sorry

end max_f_value_l214_214982


namespace set_M_listing_l214_214014

def is_divisor (a b : ℕ) : Prop := b % a = 0

theorem set_M_listing : 
  (M = { m : ℕ | is_divisor (m + 1) 10 ∧ m > 0 ∧ m + 1 ≥ 2 }) = {1, 4, 9} :=
by
  sorry

end set_M_listing_l214_214014


namespace westbound_vehicles_in_150_mile_section_l214_214591

theorem westbound_vehicles_in_150_mile_section
    (eastbound_speed : ℝ) (westbound_speed : ℝ) (observed_vehicles : ℕ)
    (observation_time : ℕ) (section_length : ℝ)
    (h1 : eastbound_speed = 80)
    (h2 : westbound_speed = 60)
    (h3 : observed_vehicles = 30)
    (h4 : observation_time = 10)
    (h5 : section_length = 150) :
    let relative_speed := eastbound_speed + westbound_speed in
    let observation_time_hours := (observation_time : ℝ) / 60 in
    let relative_distance := relative_speed * observation_time_hours in
    let vehicle_density := observed_vehicles / relative_distance in
    let estimated_vehicles := section_length * vehicle_density in
    estimated_vehicles ≈ 193 := sorry

end westbound_vehicles_in_150_mile_section_l214_214591


namespace empty_is_subset_of_singleton_l214_214646

theorem empty_is_subset_of_singleton : ∅ ⊆ ({0} : set ℕ) :=
sorry

end empty_is_subset_of_singleton_l214_214646


namespace polar_equation_of_C_max_area_OAB_l214_214541

noncomputable def Cartesian_to_Polar (x y : ℝ) : ℝ := 
  let ρ := Real.sqrt (x * x + y * y)
  let θ := Real.arctan2 y x
  ρ

theorem polar_equation_of_C (θ : ℝ) (hθ : 0 < θ ∧ θ < π) :
  ((Cartesian_to_Polar (ρ := |θ.cos| + θ.sin).1 (ρ := |θ.cos| + θ.sin).2) = ρ) :=
sorry

theorem max_area_OAB (α : ℝ) (hα : 0 < α ∧ α < π/2) :
  let ρ1 := |θ.cos| + θ.sin,
  let ρ2 := |θ.sin| + θ.cos
  (ρ1 * ρ2 / 2 = 1) :=
sorry

end polar_equation_of_C_max_area_OAB_l214_214541


namespace num_good_subsets_correct_l214_214938

-- Definition of the set S and subset A.
def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 1990}
def is_good_subset (A : Set ℕ) : Prop := A ⊆ S ∧ A.card = 26 ∧ (∑ x in A, x) % 5 = 0

-- Definition of the number of good subsets.
noncomputable def number_of_good_subsets : ℕ := Nat.choose 1990 26 / 5

-- The theorem statement that asserts the number of good subsets.
theorem num_good_subsets_correct : 
  (∑ A in (Finset.powersetLen 26 (Finset.range 1991)), if is_good_subset A.to_set then 1 else 0) 
  = number_of_good_subsets := 
by
  sorry

end num_good_subsets_correct_l214_214938


namespace find_p_l214_214025

theorem find_p (p : ℚ) : 12 ^ 3 = (9 ^ 2 / 3) * 2 ^ (12 * p) ↔ p = 1 / 2 := by
  sorry

end find_p_l214_214025


namespace number_of_companion_relation_subsets_l214_214111

def isCompanionRelationSet (A : Set ℚ) : Prop :=
  ∀ x ∈ A, (x ≠ 0 → (1 / x) ∈ A)

def M : Set ℚ := {-1, 0, 1 / 3, 1 / 2, 1, 2, 3, 4}

theorem number_of_companion_relation_subsets :
  ∃ n, n = 15 ∧
  (∀ A ⊆ M, isCompanionRelationSet A) :=
sorry

end number_of_companion_relation_subsets_l214_214111


namespace right_triangle_conditions_l214_214904

theorem right_triangle_conditions (A B C : ℝ)
  (h1 : A + B = C)
  (h2 : A / B = 1 / 2)
  (h3 : A = 90 - B)
  (h4 : A = B - C) :
  (∃ (i : Fin 4), ∃ (cond : i = 0 ∧ A + B = C ∨ i = 1 ∧ A / B = 1 / 2 ∨ i = 2 ∧ A = 90 - B ∨ i = 3 ∧ A = B - C), A + B + C = 180) → 
  ∃ i, (A + B + C = 180 ∧ (C = 90 ∨ B = 90)) :=
by
  sorry

end right_triangle_conditions_l214_214904


namespace triangle_area_l214_214786

-- Defining the problem in Lean 4:
theorem triangle_area (A B C M : ℝ) (a b c : ℝ)
  (hB : B = π / 6)
  (hM : M = sqrt 7)
  : (1 / 2) * b^2 * sin (2 * π / 3) = sqrt 3 :=
by
  sorry

end triangle_area_l214_214786


namespace max_value_sqrt_three_eights_l214_214926

noncomputable def max_value_expression (a b c d e : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d) (h₅ : 0 < e) : ℝ :=
  (ab+bc+cd+de)/(2a^2+b^2+2c^2+d^2+2e^2)

theorem max_value_sqrt_three_eights (a b c d e : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d) (h₅ : 0 < e) : 
  (max_value_expression a b c d e h₁ h₂ h₃ h₄ h₅) ≤ sqrt (3 / 8) :=
sorry

end max_value_sqrt_three_eights_l214_214926


namespace max_difference_1010_l214_214550

-- Definition and conditions
def convex_gon_2020 : Type := Fin 2020 → ℕ

def placement_valid (V : convex_gon_2020) : Prop :=
  ∀ (i : Fin 2020), ∃ j < 3, V (i + j) = 6 ∧ ∃ k < 3, V (i + k) = 7

def sum_on_sides (V : convex_gon_2020) : ℕ :=
  ∑ i in Finset.range 2020, V i * V (i + 1)

def sum_on_diagonals (V : convex_gon_2020) : ℕ :=
  ∑ i in Finset.range 2020, V i * V (i + 2)

def max_val_diff : convex_gon_2020 → ℕ :=
  λ V, sum_on_diagonals V - sum_on_sides V

-- The main theorem statement
theorem max_difference_1010 {V : convex_gon_2020} :
  placement_valid V → max_val_diff V = 1010 :=
by
  sorry

end max_difference_1010_l214_214550


namespace probability_XOXOXOX_l214_214408

theorem probability_XOXOXOX (arrangement : list char) 
  (h_len : arrangement.length = 7) 
  (h_X_count : arrangement.count 'X' = 4) 
  (h_O_count : arrangement.count 'O' = 3) :
  let total_arrangements := nat.choose 7 4 in 
  let favorable_outcomes := 1 in
  favorable_outcomes / total_arrangements = 1 / 35 :=
by
  -- proof
  sorry

end probability_XOXOXOX_l214_214408


namespace number_of_children_l214_214365

theorem number_of_children (total_pencils : ℕ) (pencils_per_child : ℕ) (total_pencils_eq : total_pencils = 22) (pencils_per_child_eq : pencils_per_child = 2) : 
  total_pencils / pencils_per_child = 11 :=
by
  rw [total_pencils_eq, pencils_per_child_eq]
  exact (nat.div_eq_of_eq_mul_left (by norm_num) rfl).symm

end number_of_children_l214_214365


namespace planting_mowing_ratio_l214_214350

theorem planting_mowing_ratio:
    ∀ (m_lawn: ℕ) (t_mow_perLine: ℕ) (b_rows: ℕ) (f_perRow: ℕ) (t_gardening: ℕ) (p_fl:int),
    m_lawn = 40 →
    t_mow_perLine = 2 →
    b_rows = 8 →
    f_perRow = 7 →
    t_gardening = 108 →
    (t_gardening - m_lawn * t_mow_perLine) / (b_rows * f_perRow):ℚ / t_mow_perLine:ℚ = 0.25 :=
by
  intros m_lawn t_mow_perLine b_rows f_perRow t_gardening p_fl
  sorry

end planting_mowing_ratio_l214_214350


namespace problem1_problem2_l214_214611

open Classical

theorem problem1 (x : ℝ) : -x^2 + 4 * x - 4 < 0 ↔ x ≠ 2 :=
sorry

theorem problem2 (x : ℝ) : (1 - x) / (x - 5) > 0 ↔ 1 < x ∧ x < 5 :=
sorry

end problem1_problem2_l214_214611


namespace parallel_lines_slope_eq_l214_214518

theorem parallel_lines_slope_eq (m : ℝ) :
  (x : ℝ) (y : ℝ), (x + m * y + 6 = 0 ∧ (m - 2) * x + 3 * y + 2 * m = 0) → m = -1 :=
by
  sorry

end parallel_lines_slope_eq_l214_214518


namespace train_speed_is_correct_l214_214691

-- Define the conditions
def train_length : ℝ := 120
def crossing_time : ℝ := 12

-- Define the speed of each train
noncomputable def train_speed_m_per_s : ℝ := 
  let relative_speed := (2 * train_length) / crossing_time in
  relative_speed / 2

noncomputable def train_speed_km_per_hr : ℝ :=
  train_speed_m_per_s * 3.6

-- Statement to prove
theorem train_speed_is_correct :
  train_speed_km_per_hr = 36 := 
by sorry

end train_speed_is_correct_l214_214691


namespace alternating_sum_squares_geq_sum_l214_214056

theorem alternating_sum_squares_geq_sum:
  ∀ (k : ℕ) (a : ℕ → ℝ), 
  (∀ i ∈ {1, 2, ..., 2*k+1}, a i ≥ 0) ∧ (∀ i ∈ {1, 2, ..., 2*k}, a i ≥ a (i+1)) → 
  ∑ i in finset.range (2*k+1), if (i + 1) % 2 = 1 then (a (i + 1))^2 else - (a (i + 1))^2 ≥
  (∑ i in finset.range (2*k+1), if (i + 1) % 2 = 1 then a (i + 1) else - a (i + 1))^2 :=
  sorry

end alternating_sum_squares_geq_sum_l214_214056


namespace gcd_of_g_y_and_y_l214_214473

theorem gcd_of_g_y_and_y (y : ℤ) (h : 9240 ∣ y) : Int.gcd ((5 * y + 3) * (11 * y + 2) * (17 * y + 8) * (4 * y + 7)) y = 168 := by
  sorry

end gcd_of_g_y_and_y_l214_214473


namespace solve_quadratic_inequality_l214_214258

variable (a b : ℝ)
variable (h_sol_set : ∀ x : ℝ, -1 < x ∧ x < 2 ↔ ax^2 + bx + 2 > 0)

theorem solve_quadratic_inequality : ∀ x : ℝ, -2 < x ∧ x < 1 ↔ bx^2 - ax - 2 > 0 := 
begin
  -- We state and prove the given conditions
  have ha_neg : a < 0, sorry,
  have solutions_eq : ∀ x : ℝ, ax^2 + bx + 2 = 0 ↔ x = -1 ∨ x = 2, sorry,
  -- Now we should prove the transformation of the second inequality
  -- But skipping proof steps, we'll just complete the statement with sorry
  sorry
end

end solve_quadratic_inequality_l214_214258


namespace probability_XOXOXOX_is_1_div_35_l214_214419

def count_combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def num_ways_to_choose_positions_for_X (total_positions : ℕ) (num_X : ℕ) : ℕ := 
  count_combinations total_positions num_X

def num_ways_for_specific_arrangement_XOXOXOX : ℕ := 1

def probability_of_XOXOXOX (num_ways_total : ℕ) (num_ways_specific : ℕ) : ℚ := 
  num_ways_specific / num_ways_total

theorem probability_XOXOXOX_is_1_div_35 :
  probability_of_XOXOXOX (num_ways_to_choose_positions_for_X 7 4) num_ways_for_specific_arrangement_XOXOXOX = 1 / 35 := by
  sorry

end probability_XOXOXOX_is_1_div_35_l214_214419


namespace jane_coffees_l214_214910

open Nat

theorem jane_coffees (b m c n : Nat) 
  (h1 : b + m + c = 6)
  (h2 : 75 * b + 60 * m + 100 * c = 100 * n) :
  c = 1 :=
by sorry

end jane_coffees_l214_214910


namespace distinct_non_zero_real_numbers_l214_214471

theorem distinct_non_zero_real_numbers (
  a b c : ℝ
) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ax^2 + 2 * b * x1 + c = 0 ∧ ax^2 + 2 * b * x2 + c = 0) 
  ∨ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ bx^2 + 2 * c * x1 + a = 0 ∧ bx^2 + 2 * c * x2 + a = 0)
  ∨ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ cx^2 + 2 * a * x1 + b = 0 ∧ cx^2 + 2 * a * x2 + b = 0) :=
sorry

end distinct_non_zero_real_numbers_l214_214471


namespace number_of_pairs_satisfying_x_sq_minus_y_sq_eq_100_l214_214098

theorem number_of_pairs_satisfying_x_sq_minus_y_sq_eq_100 :
  ∃! (n : ℕ), n = 3 ∧ ∀ (x y : ℕ), x > 0 → y > 0 → x^2 - y^2 = 100 ↔ (x, y) = (26, 24) ∨ (x, y) = (15, 10) ∨ (x, y) = (15, 5) :=
by
  sorry

end number_of_pairs_satisfying_x_sq_minus_y_sq_eq_100_l214_214098


namespace tangent_line_range_of_a_l214_214838

-- Definitions
noncomputable def f (x a : ℝ) : ℝ := (x + 1) * real.log x - a * (x - 1)

-- Problem (Ⅰ)
theorem tangent_line (a : ℝ) (ha : a = 4) : 
  let y := f x 4 in
  y.diff x | x=1 = -2 
  ∧ y = -2 * (x - 1) + f 1 4 
  := by
  unfold f
  have hy := calc 
    f(x,a) * y.diff 
  sorry

-- Problem (Ⅱ)
theorem range_of_a (a : ℝ) :
  (∀ x > 1, (x + 1) * real.log x - a * (x - 1) > 0) → 
  a ≤ 2 
  := by
  unfold f
  intro h
  have ha:= ... sorry 

end tangent_line_range_of_a_l214_214838


namespace number_added_to_x_l214_214400

theorem number_added_to_x (x : ℕ) (some_number : ℕ) (h1 : x = 3) (h2 : x + some_number = 4) : some_number = 1 := 
by
  -- Given hypotheses can be used here
  sorry

end number_added_to_x_l214_214400


namespace exists_line_with_49_points_on_one_side_no_line_with_47_points_on_one_side_l214_214592

-- Definitions for the conditions
def points := fin 100 → ℝ × ℝ
def O := (0, 0) : ℝ × ℝ

def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.1 - p1.1) * (p3.2 - p1.2) = (p3.1 - p1.1) * (p2.2 - p1.2)

axiom no_three_points_collinear (P : points) :
  ∀ i j k : fin 100, i ≠ j → j ≠ k → i ≠ k → ¬ collinear (P i) (P j) (P k)

axiom exists_line_with_48_points_on_one_side (P : points) :
  ∃ l : ℝ × ℝ, ∃ u : ℝ × ℝ,
    (l.1 * O.1 + l.2 * O.2) = 0 ∧
    ∑ i in finset.univ.filter (λ i, l.1 * (P i).1 + l.2 * (P i).2 > 0), 1 = 48

-- Statement for (1)
theorem exists_line_with_49_points_on_one_side (P : points) :
  ∃ l : ℝ × ℝ, ∃ u : ℝ × ℝ,
    (l.1 * O.1 + l.2 * O.2) = 0 ∧
    ∑ i in finset.univ.filter (λ i, l.1 * (P i).1 + l.2 * (P i).2 > 0), 1 = 49 :=
sorry

-- Statement for (2)
theorem no_line_with_47_points_on_one_side (P : points) :
  ¬ ∃ l : ℝ × ℝ, ∃ u : ℝ × ℝ,
    (l.1 * O.1 + l.2 * O.2) = 0 ∧
    ∑ i in finset.univ.filter (λ i, l.1 * (P i).1 + l.2 * (P i).2 > 0), 1 = 47 :=
sorry

end exists_line_with_49_points_on_one_side_no_line_with_47_points_on_one_side_l214_214592


namespace find_clubs_l214_214289

theorem find_clubs (S D H C : ℕ) (h1 : S + D + H + C = 13)
  (h2 : S + C = 7) 
  (h3 : D + H = 6) 
  (h4 : D = 2 * S) 
  (h5 : H = 2 * D) 
  : C = 6 :=
by
  sorry

end find_clubs_l214_214289


namespace sequences_correct_l214_214469

noncomputable def a_n (n : ℕ) : ℕ := 3 * n - 2
noncomputable def b_n (n : ℕ) : ℕ := 2 ^ n

noncomputable def S_n (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a n + a 1)) / 2

noncomputable def sum_T_n (n : ℕ) : ℕ :=
  let a := a_n
  let b := b_n
  let sum_seq := λ i : ℕ, (3 * 2 * i - 2) * 2 ^ i
  (3 * n - 4) * 2 ^ (n + 2) + 16
  
theorem sequences_correct :
  (a_n n = 3 * n - 2) ∧ (b_n n = 2 ^ n) ∧ 
  (S_n a_n 11 = 11 * b_n 4) ∧ 
  (∀ n : ℕ, sum_T_n n = (3 * n - 4) * 2 ^ (n + 2) + 16) :=
by
  sorry

end sequences_correct_l214_214469


namespace triangle_longest_side_l214_214991

variable (x : ℝ) 

def side1 : ℝ := 9
def side2 : ℝ := x + 5
def side3 : ℝ := 2 * x + 2
def perimeter : ℝ := 42

theorem triangle_longest_side : 
  (side1 + side2 + side3 = perimeter) → 
  (x = 26 / 3) →
  max side1 (max side2 side3) = 58 / 3 :=
by
  intros
  sorry

end triangle_longest_side_l214_214991


namespace ravi_mobile_phone_purchase_price_l214_214205

theorem ravi_mobile_phone_purchase_price 
  (P : ℝ)
  (cond1 : 15000 - 0.04 * 15000 = 14400)
  (cond2 : 1.10 * P)
  (cond3 : 14400 + 1.10 * P = 15000 + P + 200) :
  P = 6000 := 
sorry

end ravi_mobile_phone_purchase_price_l214_214205


namespace trapezoid_base_length_sets_l214_214035

open Nat

theorem trapezoid_base_length_sets :
  ∃ (sets : Finset (ℕ × ℕ)), sets.card = 5 ∧ 
    (∀ p ∈ sets, ∃ (b1 b2 : ℕ), b1 = 10 * p.1 ∧ b2 = 10 * p.2 ∧ b1 + b2 = 90) :=
by
  sorry

end trapezoid_base_length_sets_l214_214035


namespace find_num_white_balls_l214_214887

theorem find_num_white_balls
  (W : ℕ)
  (total_balls : ℕ := 15 + W)
  (prob_black : ℚ := 7 / total_balls)
  (given_prob : ℚ := 0.38095238095238093) :
  prob_black = given_prob → W = 3 :=
by
  intro h
  sorry

end find_num_white_balls_l214_214887


namespace ellipse_chord_slope_relation_l214_214567

theorem ellipse_chord_slope_relation
    (a b : ℝ) (h : a > b) (h1 : b > 0)
    (A B M : ℝ × ℝ)
    (hM_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
    (hAB_slope : A.1 ≠ B.1)
    (K_AB K_OM : ℝ)
    (hK_AB : K_AB = (B.2 - A.2) / (B.1 - A.1))
    (hK_OM : K_OM = (M.2 - 0) / (M.1 - 0)) :
  K_AB * K_OM = - (b ^ 2) / (a ^ 2) := 
  sorry

end ellipse_chord_slope_relation_l214_214567


namespace x_minus_q_in_terms_of_q_l214_214509

theorem x_minus_q_in_terms_of_q (x q : ℝ) (h1 : |x - 3| = q) (h2 : x < 3) : x - q = 3 - 2q :=
sorry

end x_minus_q_in_terms_of_q_l214_214509


namespace area_triangle_l214_214078

noncomputable def ellipse : Set (Real × Real) :=
  { p : ℝ × ℝ | (p.1^2 / 4) + (p.2^2 / 3) = 1 }

def focus1 : ℝ × ℝ := (1, 0)
def focus2 : ℝ × ℝ := (-1, 0)

def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1, P.2) ∈ ellipse

def is_right_triangle (P F1 F2 : ℝ × ℝ) : Prop :=
  90 = 90 -- Replace this with actual condition for right angle at F1 or F2.

def ell_area : ℝ := 3 / 2

theorem area_triangle (P : ℝ × ℝ) 
  (P_on_ellipse : is_on_ellipse P)
  (right_triangle : is_right_triangle P focus1 focus2) :
  (1 / 2) * 2 * (sqrt (4 - 3)) * (3 / 2) = ell_area :=
sorry

end area_triangle_l214_214078


namespace non_intersecting_chords_20_points_non_intersecting_chords_general_l214_214688

-- Part (a)
-- Given 20 points on a circle, show that the number of ways to pair these points with 10 non-intersecting chords is 16796
theorem non_intersecting_chords_20_points : 
  let Φ := fun (n : ℕ) =>
    if n = 0 then 1
    else ∑ k in Finset.range n, (Φ k) * (Φ (n - 1 - k))
  in Φ 10 = 16796 := by
  let Φ n := if n = 0 then 1 else ∑ k in Finset.range n, Φ k * Φ (n - 1 - k)
  have base_case : Φ 0 = 1 := rfl
  sorry

-- Part (b)
-- Given 2n points on a circle, show that the number of ways to pair these points with n non-intersecting chords
-- is given by the nth Catalan number: Φn = (2n)! / ((n+1)! * n!)
theorem non_intersecting_chords_general (n : ℕ) : 
  let Φ := fun (n : ℕ) =>
    if n = 0 then 1
    else ∑ k in Finset.range n, (Φ k) * (Φ (n - 1 - k))
  in Φ n = Nat.factorial (2 * n) / (Nat.factorial (n + 1) * Nat.factorial n) := by
  let Φ n := if n = 0 then 1 else ∑ k in Finset.range n, Φ k * Φ (n - 1 - k)
  have base_case : Φ 0 = 1 := rfl
  sorry

end non_intersecting_chords_20_points_non_intersecting_chords_general_l214_214688


namespace jane_last_day_vases_l214_214556

def vasesPerDay : Nat := 16
def totalVases : Nat := 248

theorem jane_last_day_vases : totalVases % vasesPerDay = 8 := by
  sorry

end jane_last_day_vases_l214_214556


namespace last_digit_sum_3_powers_l214_214193

theorem last_digit_sum_3_powers : 
  ∃ d : ℕ, last_digit (∑ i in finset.range (2020 + 1), 3 ^ (i + 1)) = d ∧ d = 0 :=
by 
  -- Proof goes here
  sorry

end last_digit_sum_3_powers_l214_214193


namespace floor_length_l214_214719

-- Define the conditions
def breadth : ℝ
def costPerSqM : ℝ := 5
def totalCost : ℝ := 1360
def length := 5.5 * breadth
def area := (totalCost / costPerSqM : ℝ)

-- The length of the floor should equal approximately 38.665 meters 
theorem floor_length (b : ℝ) (h_area: length * b = area) : length = 38.665 := 
sorry

end floor_length_l214_214719


namespace probability_XOXOXOX_is_one_over_thirty_five_l214_214439

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_arrangement : ℤ :=
  let total_arrangements := binomial_coefficient 7 4
  let favorable_outcomes := 1
  favorable_outcomes / total_arrangements

theorem probability_XOXOXOX_is_one_over_thirty_five :
  probability_of_arrangement = (1 : ℤ) / 35 := 
  by
  sorry

end probability_XOXOXOX_is_one_over_thirty_five_l214_214439


namespace find_amount_with_r_l214_214290

def total_money : ℝ := 9000
def amount_with_r (R : ℝ) : Prop := 3 * R = 2 * (total_money - R)

theorem find_amount_with_r : ∃ R : ℝ, amount_with_r R ∧ R = 3600 :=
by
  existsi 3600
  unfold amount_with_r
  split
  calc
    3 * 3600 = 10800 : by norm_num
    2 * (9000 - 3600) = 2 * 5400 : by norm_num
               ... = 10800 : by norm_num
  rfl

end find_amount_with_r_l214_214290


namespace find_AX_l214_214142

variable (A B X C : Type)
variable (dist : A → A → ℝ)

-- Conditions as given in the problem
variables (h1 : dist A B = 80)
variables (h2 : ∀ (A C X : A), angle A C X = angle X C B)
variables (h3 : dist A C = 45)
variables (h4 : dist B C = 90)

theorem find_AX : dist A X = 80 / 3 :=
by sorry

end find_AX_l214_214142


namespace inradius_circumradius_l214_214204

variables {T : Type} [MetricSpace T]

theorem inradius_circumradius (K k : ℝ) (d r rho : ℝ) (triangle : T)
  (h1 : (k / K) = (rho / r))
  (h2 : k ≤ K / 2)
  (h3 : 2 * r * rho = r^2 - d^2)
  (h4 : d ≥ 0) :
  r ≥ 2 * rho :=
sorry

end inradius_circumradius_l214_214204


namespace next_meeting_day_l214_214558

-- Definitions of visit periods
def jia_visit_period : ℕ := 6
def yi_visit_period : ℕ := 8
def bing_visit_period : ℕ := 9

-- August 17th
def initial_meeting_day : ℕ := 17

-- Prove that they will meet again 72 days after August 17th
theorem next_meeting_day : ∃ n : ℕ, n = 72 ∧ (∀ m : ℕ, (m = jia_visit_period ∨
m = yi_visit_period ∨ m = bing_visit_period) → ∃ k : ℕ, initial_meeting_day + 72 = m * k) :=
by
  sorry

end next_meeting_day_l214_214558


namespace angle_BAC_correct_l214_214981

-- Define the conditions in a)
def smaller_arc_angle : ℝ := 130
def larger_arc_angle : ℝ := 360 - smaller_arc_angle
def ratio_AC : ℕ := 31
def ratio_BC : ℕ := 15
def total_ratio : ℕ := ratio_AC + ratio_BC
def arc_BC : ℝ := (ratio_BC / total_ratio.toReal) * larger_arc_angle
def angle_BAC : ℝ := arc_BC / 2
def angle_BAC_degrees : ℝ := 37 + 30 / 60

-- Prove the equivalence between the computed angle and the expected answer
theorem angle_BAC_correct : angle_BAC = angle_BAC_degrees := by
  sorry

end angle_BAC_correct_l214_214981


namespace evaluate_expression_l214_214012

noncomputable def lg (x : ℝ) : ℝ := Real.log x

theorem evaluate_expression :
  lg 5 * lg 50 - lg 2 * lg 20 - lg 625 = -2 :=
by
  sorry

end evaluate_expression_l214_214012


namespace f_zero_l214_214939

noncomputable def f (x : ℚ) : ℚ := (x*x - 15*x + 85) / (x*x - 15*x + 85)

theorem f_zero : 
  (∀ n : ℚ, n ∈ {1, 2, 3, 4, 5} → f n = n^3) ∧ 
  ∃ (q p : ℚ → ℚ), 
    (∀ n : ℚ, n ∈ {1, 2, 3, 4, 5} → p n = n^3 * q n) ∧ 
    (q 0 = 85 ∧ p 0 = 120) → 
  f 0 = 24 / 17 := 
sorry

end f_zero_l214_214939


namespace M_gt_N_l214_214449

-- Given conditions
variables {x y : ℝ}
def M (x y : ℝ) : ℝ := x^2 + y^2 - 4x + 2y
def N : ℝ := -5

-- Theorem statement
theorem M_gt_N (h : x ≠ 2 ∨ y ≠ -1) : M x y > N := by
  sorry

end M_gt_N_l214_214449


namespace chord_intersects_inner_circle_probability_l214_214269

noncomputable def probability_of_chord_intersecting_inner_circle
  (radius_inner : ℝ) (radius_outer : ℝ)
  (chord_probability : ℝ) : Prop :=
  radius_inner = 3 ∧ radius_outer = 5 ∧ chord_probability = 0.205

theorem chord_intersects_inner_circle_probability :
  probability_of_chord_intersecting_inner_circle 3 5 0.205 :=
by {
  sorry
}

end chord_intersects_inner_circle_probability_l214_214269


namespace compare_neg_two_powers_l214_214750

theorem compare_neg_two_powers : (-2)^3 = -2^3 := by sorry

end compare_neg_two_powers_l214_214750


namespace definite_integral_l214_214741

namespace IntegralDefinite

def integrand (x : ℝ) : ℝ :=
  (2 * Real.cos x + 3 * Real.sin x) / (2 * Real.sin x - 3 * Real.cos x)^3

theorem definite_integral : 
  ∫ x in 0..(Real.pi / 4), integrand x = -17 / 18 :=
by
  -- proof goes here
  sorry

end IntegralDefinite

end definite_integral_l214_214741


namespace value_of_f_2010_and_f_2011_l214_214061

noncomputable def f : ℝ → ℝ
| x => if 2 ≤ x ∧ x < 4 then log (x - 1) / log 2 else sorry

lemma f_even (x : ℝ) : f x = f (-x) :=
sorry

lemma f_minus_4 (x : ℝ) : f x = -f (4 - x) :=
sorry

lemma f_periodicity (x : ℝ) : f x = f (x - 8) :=
sorry

theorem value_of_f_2010_and_f_2011 : f 2010 + f 2011 = 1 :=
sorry

end value_of_f_2010_and_f_2011_l214_214061


namespace probability_XOXOXOX_l214_214410

theorem probability_XOXOXOX (arrangement : list char) 
  (h_len : arrangement.length = 7) 
  (h_X_count : arrangement.count 'X' = 4) 
  (h_O_count : arrangement.count 'O' = 3) :
  let total_arrangements := nat.choose 7 4 in 
  let favorable_outcomes := 1 in
  favorable_outcomes / total_arrangements = 1 / 35 :=
by
  -- proof
  sorry

end probability_XOXOXOX_l214_214410


namespace solve_linear_combination_l214_214448

theorem solve_linear_combination (x y z : ℤ) 
    (h1 : x + 2 * y - z = 8) 
    (h2 : 2 * x - y + z = 18) : 
    8 * x + y + z = 70 := 
by 
    sorry

end solve_linear_combination_l214_214448


namespace omar_rolls_l214_214589

-- Define the conditions
def karen_rolls : ℕ := 229
def total_rolls : ℕ := 448

-- Define the main theorem to prove the number of rolls by Omar
theorem omar_rolls : (total_rolls - karen_rolls) = 219 := by
  sorry

end omar_rolls_l214_214589


namespace log_three_nine_cubed_l214_214006

theorem log_three_nine_cubed : log 3 (9 ^ 3) = 6 := by
  -- Importing, definitions, and theorem
  sorry

end log_three_nine_cubed_l214_214006


namespace expected_value_xi_l214_214459

noncomputable def E_xi : ℚ := 
  let possible_values := {0, 1, 2, 3, 4, 5, 6}
  let frequency := λ x, match x with
    | 0 => 3
    | 1 => 6
    | 2 => 9
    | 3 => 9
    | 4 => 6
    | 5 => 3
    | 6 => 1
    | _ => 0
  let total_occurrences := 37
  let total_outcomes := 21
  Σ x in possible_values, (x * frequency x) / total_outcomes

theorem expected_value_xi :
  E_xi = 32 / 7 :=
sorry

end expected_value_xi_l214_214459


namespace probability_XOXOXOX_l214_214413

theorem probability_XOXOXOX (X O : ℕ) (h1 : X = 4) (h2 : O = 3) :
  let total_ways := Nat.choose (X + O) X,
      favorable_outcomes := 1 in
  (favorable_outcomes / total_ways : ℚ) = 1 / 35 := by
  sorry

end probability_XOXOXOX_l214_214413


namespace tigers_total_games_l214_214736

theorem tigers_total_games :
  ∃ y : ℕ, ∃ x : ℕ, (x = 0.60 * y) ∧ ((x + 8) = 0.65 * (y + 11)) ∧ (y + 11 = 28) :=
sorry

end tigers_total_games_l214_214736


namespace probability_XOXOXOX_l214_214432

theorem probability_XOXOXOX :
  let X := 4;
      O := 3;
      total_positions := 7;
      specific_arrangement := 1;
      total_arrangements := Nat.choose total_positions X in
  1 / total_arrangements = 1 / 35 := by
  sorry

end probability_XOXOXOX_l214_214432


namespace find_a_l214_214564

open Set

theorem find_a (a : ℝ) : let A := {-4, 2 * a - 1, a^2}
                        let B := {a - 1, 1 - a, 9}
                        (A ∩ B = {9}) → a = 3 :=
by
  intro A B h
  sorry

end find_a_l214_214564


namespace fraction_zero_implies_x_is_neg_2_l214_214118

theorem fraction_zero_implies_x_is_neg_2 {x : ℝ} 
  (h₁ : x^2 - 4 = 0)
  (h₂ : x^2 - 4 * x + 4 ≠ 0) 
  : x = -2 := 
by
  sorry

end fraction_zero_implies_x_is_neg_2_l214_214118


namespace tetrahedron_dot_product_l214_214818

-- Define vector space over ℝ (real numbers)
variables {V : Type*} [inner_product_space ℝ V]
variables (A B C D E F : V)
-- Define constant length a
variable (a : ℝ)

-- Define the conditions from the problem
-- Length of each edge is a
axiom AB_length : ∥A - B∥ = a
axiom AC_length : ∥A - C∥ = a
axiom AD_length : ∥A - D∥ = a
axiom BC_length : ∥B - C∥ = a
axiom BD_length : ∥B - D∥ = a
axiom CD_length : ∥C - D∥ = a

-- Midpoints E and F
axiom E_mid : E = (B + C) / 2
axiom F_mid : F = (A + D) / 2

-- Goal
theorem tetrahedron_dot_product :
  (A - E) ⬝ (A - F) = a^2 / 4 :=
sorry

end tetrahedron_dot_product_l214_214818


namespace ratio_of_metals_l214_214855

theorem ratio_of_metals (G C S : ℝ) (h1 : 11 * G + 5 * C + 7 * S = 9 * (G + C + S)) : 
  G / C = 1 / 2 ∧ G / S = 1 :=
by
  sorry

end ratio_of_metals_l214_214855


namespace base_r_is_2_l214_214895

theorem base_r_is_2 (r : ℕ) (h1 : 521_r = 5 * r^2 + 2 * r + 1) 
  (h2 : 110_r = r^2 + r) (h3 : 371_r = 3 * r^2 + 7 * r + 1) 
  (h4 : 1002_r = r^3 + 2) 
  (transaction_eq : (5 * r^2 + 2 * r + 1) + (r^2 + r) - (3 * r^2 + 7 * r + 1) = r^3 + 2) : 
  r = 2 := 
by 
  sorry

end base_r_is_2_l214_214895


namespace min_value_fraction_l214_214814

-- Definitions for the sequence and conditions
def is_arithmetic_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2 * a (n - 1)

-- Main theorem statement
theorem min_value_fraction (a : ℕ → ℝ) (m n : ℕ) (a1 : ℝ)
  (h_seq : is_arithmetic_geometric_sequence a)
  (h_cond1 : a 7 = a 6 + 2 * a 5)
  (h_cond2 : sqrt (a m * a n) = 4 * a 1)
  (h_a1_pos : a 1 > 0) :
  ∃ (m n : ℕ), m + n = 6 ∧ (1 / m + 4 / n) = 3 / 2 :=
sorry

end min_value_fraction_l214_214814


namespace sequence_monotonically_increasing_iff_lambda_gt_neg3_l214_214489

theorem sequence_monotonically_increasing_iff_lambda_gt_neg3 (λ : ℝ) :
  (∀ n : ℕ, n > 0 → (n^2 + λ * n - 2018) < ((n + 1)^2 + λ * (n + 1) - 2018)) ↔ λ > -3 :=
by 
  sorry

end sequence_monotonically_increasing_iff_lambda_gt_neg3_l214_214489


namespace expenditure_of_neg_50_l214_214545

/-- In the book "Nine Chapters on the Mathematical Art," it is noted that
"when two calculations have opposite meanings, they should be named positive
and negative." This means: if an income of $80 is denoted as $+80, then $-50
represents an expenditure of $50. -/
theorem expenditure_of_neg_50 :
  (∀ (income : ℤ), income = 80 → -income = -50 → ∃ (expenditure : ℤ), expenditure = 50) := sorry

end expenditure_of_neg_50_l214_214545


namespace least_m_of_covering_triangles_l214_214030

open Real

theorem least_m_of_covering_triangles :
  ∃ (m : ℝ), (∀ (A B C D E : ℝ), equilaterial_triangle A → equilaterial_triangle B → equilaterial_triangle C → equilaterial_triangle D → equilaterial_triangle E → (A + B + C + D + E = m) → covers_triangle (A + B + C + D + E)) ↔ m = 2 :=
by
  sorry

end least_m_of_covering_triangles_l214_214030


namespace relation_among_a_b_c_l214_214808

noncomputable def a : ℝ := (2 / 5) ^ 2
noncomputable def b : ℝ := (5 / 2) ^ 2
noncomputable def c : ℝ := Real.logBase 3 (2 / 5)

theorem relation_among_a_b_c : c < a ∧ a < b :=
by {
  sorry
}

end relation_among_a_b_c_l214_214808


namespace part1_part2_l214_214851

theorem part1 (θ : ℝ) (a b : ℝ × ℝ) (ha : a = (1, 2 * sin θ)) (hb : b = (5 * cos θ, 3)) :
  (∀ x y : ℝ × ℝ, x = a → y = b → (1 * y.2 - 2 * sin θ * 5 * cos θ = 0) → (sin 2 * θ = 3 / 5)) :=
by
  intros x y hxa hyb hxy
  rw [hxa, ha] at hxy
  rw [hyb, hb] at hxy
  sorry

theorem part2 (θ : ℝ) (a b : ℝ × ℝ) (ha : a = (1, 2 * sin θ)) (hb : b = (5 * cos θ, 3)) :
  (∀ x y : ℝ × ℝ, x = a → y = b → (1 * 5 * cos θ + 2 * sin θ * 3 = 0) → (tan (θ + π / 4) = -1)) :=
by
  intros x y hxa hyb hxy
  rw [hxa, ha] at hxy
  rw [hyb, hb] at hxy
  sorry

end part1_part2_l214_214851


namespace obtuse_dihedral_angles_l214_214965

theorem obtuse_dihedral_angles (AOB BOC COA : ℝ) (h1 : AOB > 90) (h2 : BOC > 90) (h3 : COA > 90) :
  ∃ α β γ : ℝ, α > 90 ∧ β > 90 ∧ γ > 90 :=
sorry

end obtuse_dihedral_angles_l214_214965


namespace inequality_one_inequality_two_inequality_three_l214_214971

-- (1) Prove that the solution set for 5^x < 0.2 is (-∞, -1)
theorem inequality_one (x : ℝ) : (5 ^ x < 0.2) ↔ x < -1 := by
  sorry

-- (2) Prove that the solution set for log_{0.2}(x - 2) > 1 is (2, 2.2)
theorem inequality_two (x : ℝ) : (log 0.2 (x - 2) > 1) ↔ 2 < x ∧ x < 2.2 := by
  sorry

-- (3) Prove that the solution set for 5^{x + 2} > 2 is (log_{5}2 - 2, +∞)
theorem inequality_three (x : ℝ) : (5 ^ (x + 2) > 2) ↔ x > log 5 2 - 2 := by
  sorry

end inequality_one_inequality_two_inequality_three_l214_214971


namespace problem_equivalence_l214_214334

noncomputable def sequence := {a : Fin 38 → ℕ // (a 0 = 37) ∧ (a 1 = 1) ∧ 
  ∀ k : Fin 36, (∑ i in Finset.range (k + 1), a ⟨i, Fin.is_lt _ _⟩) % a ⟨k.succ, Fin.is_lt _ _⟩ = 0}

def a_37 (s : sequence) : ℕ := s.val 36

def a_3 (s : sequence) : ℕ := s.val 2

theorem problem_equivalence (s : sequence) : a_37 s = 19 ∧ a_3 s = 2 := 
  sorry

end problem_equivalence_l214_214334


namespace sequence_general_term_l214_214613

noncomputable def sequence (a : ℕ → ℕ) : ℕ → ℕ
| 0       := 0
| (n + 1) := a n^2 - n*a n + 1

theorem sequence_general_term (a : ℕ → ℕ) 
    (h : ∀ n, a (n + 1) = a n^2 - n * a n + 1)
    (h0 : a 1 = 2) :
    ∀ n, a n = n + 1 :=
by
  sorry

end sequence_general_term_l214_214613


namespace john_pays_12_dollars_l214_214919

/-- Define the conditions -/
def number_of_toys : ℕ := 5
def cost_per_toy : ℝ := 3
def discount_rate : ℝ := 0.2

/-- Define the total cost before discount -/
def total_cost_before_discount := number_of_toys * cost_per_toy

/-- Define the discount amount -/
def discount_amount := total_cost_before_discount * discount_rate

/-- Define the final amount John pays -/
def final_amount := total_cost_before_discount - discount_amount

/-- The theorem to be proven -/
theorem john_pays_12_dollars : final_amount = 12 := by
  -- Proof goes here
  sorry

end john_pays_12_dollars_l214_214919


namespace cos_difference_min_translation_l214_214496

-- Definitions for vectors and magnitude condition
def vector_a (x : ℝ) : (ℝ × ℝ) := (Real.cos x, Real.sin x)
def vector_b (y : ℝ) : (ℝ × ℝ) := (Real.cos y, Real.sin y)
def vector_c (x : ℝ) : (ℝ × ℝ) := (Real.sin x, Real.cos x)

-- Prove cos(x - y) = 3/5 given the magnitude condition
theorem cos_difference (x y : ℝ) : 
  (Real.sqrt ((vector_a x).fst - (vector_b y).fst)^2 + ((vector_a x).snd - (vector_b y).snd)^2) = (2 * Real.sqrt 5) / 5 
  → Real.cos (x - y) = 3 / 5 := 
by 
  sorry

-- Prove the minimum value of m is π/4 given the translation and symmetry condition
theorem min_translation (x m : ℝ) : 
  ((λ m : ℝ, ∃ k : ℤ,  m = k * Real.pi / 2 + Real.pi / 4) ∧ m > 0) 
  → m ≥ Real.pi / 4 := 
by 
  sorry

end cos_difference_min_translation_l214_214496


namespace buffy_whiskers_l214_214803

theorem buffy_whiskers :
  ∀ (Puffy Scruffy Buffy Juniper : ℕ),
    Juniper = 12 →
    Puffy = 3 * Juniper →
    Puffy = Scruffy / 2 →
    Buffy = (Juniper + Puffy + Scruffy) / 3 →
    Buffy = 40 :=
by
  intros Puffy Scruffy Buffy Juniper hJuniper hPuffy hScruffy hBuffy
  sorry

end buffy_whiskers_l214_214803


namespace sqrt_18_mul_sqrt_32_eq_24_l214_214218
  
theorem sqrt_18_mul_sqrt_32_eq_24 : (Real.sqrt 18 * Real.sqrt 32 = 24) :=
  sorry

end sqrt_18_mul_sqrt_32_eq_24_l214_214218


namespace max_value_of_M_l214_214165

theorem max_value_of_M : ∀ (x y : ℝ), (x > 0) ∧ (y > 0) → max (min {x, -1/y, y + 1/x}) = sqrt 2 :=
by
  -- Proof goes here
  sorry

end max_value_of_M_l214_214165


namespace area_of_ABCN_is_zero_l214_214755

theorem area_of_ABCN_is_zero
  (polygon : List (ℝ × ℝ))
  (h1 : polygon.length = 16)
  (h2 : ∀ i, (polygon.nth_le i h1).2 = 3)
  (h3 : ∀ i, ∠ (polygon.nth_le i h1) (polygon.nth_le (i + 1) h1) (polygon.nth_le (i + 2) h1) = 90)
  (A I J : ℝ × ℝ)
  (AI_parallel_CJ : ∃ N : ℝ × ℝ, line_through A I ∩ line_through C J = {N}) :
  area_of_quadrilateral A B C N = 0 :=
sorry

end area_of_ABCN_is_zero_l214_214755


namespace base_conversion_least_sum_l214_214636

theorem base_conversion_least_sum :
  ∃ (c d : ℕ), (5 * c + 8 = 8 * d + 5) ∧ c > 0 ∧ d > 0 ∧ (c + d = 15) := by
sorry

end base_conversion_least_sum_l214_214636


namespace triangle_larger_segment_l214_214257

theorem triangle_larger_segment
  (a b c : ℝ)
  (h : ℝ)
  (hac : c = 90)
  (hab : a = 40)
  (hbc : b = 50)
  (h_ge_0 : h ≥ 0) :
  let x := sqrt (a ^ 2 - h ^ 2) in
  let larger_segment := c - x in
  larger_segment = 50 :=
by
  sorry

end triangle_larger_segment_l214_214257


namespace number_of_merchants_l214_214776

-- Definitions of roles
inductive Role
| engineer
| merchant

-- Function to indicate the truth-telling properties
def tellsTruth : Role → Prop
| Role.engineer => true
| Role.merchant => false

-- Assignments based on problem conditions
def F := Role.engineer
def G := Role.engineer
def engine : ∀ (r : Role), tellsTruth r = (r = Role.engineer) := by
  intro r
  cases r <;> simp

-- Definitions of the chain of declarations
def declaration_chain : Prop :=
  -- A announces that B claims that C assures that D says that E insists that F denies that G is an engineer.
  ∃ (A B C D E : Role),
    A = Role.merchant ∧
    tellsTruth E = false ∧ -- Because F denies that G is an engineer contradicts the truth (since F and G are engineers).
    tellsTruth D = tellsTruth E ∧
    tellsTruth C = (tellsTruth D = false) ∧
    tellsTruth B = (tellsTruth C = (tellsTruth D = false)) ∧
    tellsTruth A = (tellsTruth B = (tellsTruth C = (tellsTruth D = false)))

-- Proving the number of merchants given the condition A is a merchant
theorem number_of_merchants : ∀ (A : Role), A = Role.merchant → ∃ n, n = 3 :=
by
  intros A hA
  use 3
  sorry

end number_of_merchants_l214_214776


namespace trips_Jean_l214_214344

theorem trips_Jean (x : ℕ) (h1 : x + (x + 6) = 40) : x + 6 = 23 := by
  sorry

end trips_Jean_l214_214344


namespace smallest_positive_period_axis_of_symmetry_minimum_value_on_interval_l214_214485

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 3 * Real.pi / 5)

theorem smallest_positive_period :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧
  ∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T := by
  sorry

theorem axis_of_symmetry :
  ∃ k : ℤ, (∀ x, f x = f (11 * Real.pi / 20 + k * Real.pi / 2)) := by
  sorry

theorem minimum_value_on_interval :
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = -1 := by
  sorry

end smallest_positive_period_axis_of_symmetry_minimum_value_on_interval_l214_214485


namespace distance_amanda_to_kimberly_l214_214554

-- Define the given conditions
def amanda_speed : ℝ := 2 -- miles per hour
def amanda_time : ℝ := 3 -- hours

-- Prove that the distance is 6 miles
theorem distance_amanda_to_kimberly : amanda_speed * amanda_time = 6 := by
  sorry

end distance_amanda_to_kimberly_l214_214554


namespace probability_XOXOXOX_l214_214428

noncomputable def binomial (n k : ℕ) : ℕ := nat.choose n k

theorem probability_XOXOXOX :
  let num_X := 4,
      num_O := 3,
      total_arrangements := binomial 7 num_X in
  total_arrangements = 35 ∧
  1 / total_arrangements = (1 : ℚ) / 35 :=
by
  sorry

end probability_XOXOXOX_l214_214428


namespace real_roots_of_cubic_equation_l214_214766

theorem real_roots_of_cubic_equation : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, (x^3 - 2 * x + 1)^2 = 9) ∧ S.card = 2 := 
by
  sorry

end real_roots_of_cubic_equation_l214_214766


namespace part1_part2_l214_214846

open Set

variable {U A B : Set ℝ}

def U : Set ℝ := { x | -2 < x ∧ x < 12 }
def A : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 ≤ x ∧ x < 5 }

theorem part1 : (A ∪ B) = { x | 2 ≤ x ∧ x < 7 } := sorry

theorem part2 : (U \ A ∩ B) = { x | 2 ≤ x ∧ x < 3 } := sorry

end part1_part2_l214_214846


namespace buffy_whiskers_l214_214795

/-- Definition of whisker counts for the cats --/
def whiskers_of_juniper : ℕ := 12
def whiskers_of_puffy : ℕ := 3 * whiskers_of_juniper
def whiskers_of_scruffy : ℕ := 2 * whiskers_of_puffy
def whiskers_of_buffy : ℕ := (whiskers_of_juniper + whiskers_of_puffy + whiskers_of_scruffy) / 3

/-- Proof statement for the number of whiskers of Buffy --/
theorem buffy_whiskers : whiskers_of_buffy = 40 := 
by
  -- Proof is omitted
  sorry

end buffy_whiskers_l214_214795


namespace pentagons_from_15_points_l214_214017

theorem pentagons_from_15_points (n : ℕ) (h : n = 15) : (nat.choose 15 5) = 3003 := by
  rw h
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num)

end pentagons_from_15_points_l214_214017


namespace PQ_parallel_CD_l214_214135

open EuclideanGeometry

variables {A B C D P Q: Point}

-- Define the conditions in the problem
def cond1 (ABCD_is_quadrilateral : Quadrilateral A B C D): Prop :=
  ∠₁ A = ∠₃ C

def cond2 (bisector_B_intersects_AD_at_P : Line B (bisector B A D) ∩ Line A D = P): Prop :=
  True

def cond3 (perpendicular_to_BP_through_A_meets_BC_at_Q : Line A (perpendicular_through A (Line B P)) ∩ Line B C = Q): Prop :=
  True

-- Define the theorem to be proven
theorem PQ_parallel_CD (ABCD_is_quadrilateral : Quadrilateral A B C D)
                       (h1 : cond1 ABCD_is_quadrilateral)
                       (h2 : bisector_B_intersects_AD_at_P)
                       (h3 : perpendicular_to_BP_through_A_meets_BC_at_Q):
  Parallel (Line P Q) (Line C D) :=
sorry

end PQ_parallel_CD_l214_214135


namespace james_parking_tickets_l214_214153

-- Define the conditions
def ticket_cost_1 := 150
def ticket_cost_2 := 150
def ticket_cost_3 := 1 / 3 * ticket_cost_1
def total_cost := ticket_cost_1 + ticket_cost_2 + ticket_cost_3
def roommate_pays := total_cost / 2
def james_remaining_money := 325
def james_original_money := james_remaining_money + roommate_pays

-- Define the theorem we want to prove
theorem james_parking_tickets (h1: ticket_cost_1 = 150)
                              (h2: ticket_cost_1 = ticket_cost_2)
                              (h3: ticket_cost_3 = 1 / 3 * ticket_cost_1)
                              (h4: total_cost = ticket_cost_1 + ticket_cost_2 + ticket_cost_3)
                              (h5: roommate_pays = total_cost / 2)
                              (h6: james_remaining_money = 325)
                              (h7: james_original_money = james_remaining_money + roommate_pays):
                              total_cost = 350 :=
by
  sorry

end james_parking_tickets_l214_214153


namespace number_of_divisors_720_l214_214606

-- The problem states to prove that the number of divisors of 720 in the range from 1 to 720 is 30.
open Nat

theorem number_of_divisors_720 : 
  {x : ℕ | x > 0 ∧ x ≤ 720 ∧ 720 % x = 0}.to_finset.card = 30 :=
by
  sorry

end number_of_divisors_720_l214_214606


namespace exist_odd_distinct_integers_l214_214063

theorem exist_odd_distinct_integers (n : ℕ) (h1 : n % 2 = 1) (h2 : n > 3) (h3 : n % 3 ≠ 0) : 
  ∃ a b c : ℕ, a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  3 / (n : ℚ) = 1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) :=
sorry

end exist_odd_distinct_integers_l214_214063


namespace tangent_line_property_l214_214657

variables (a b c : ℝ) (AC1 AB1 : ℝ)
noncomputable def p := (a + b + c) / 2

theorem tangent_line_property (x y : ℝ) (hAC1 : AC1 = x) (hAB1 : AB1 = y) :
  p a b c * x * y - b * c * (x + y) + b * c * (p a b c - a) = 0 :=
by sorry

end tangent_line_property_l214_214657


namespace smallest_n_for_partition_l214_214771

theorem smallest_n_for_partition (n : ℕ) : 
  (∀ p : ℕ, p ≥ n → ∃ f : ℕ → ℕ, ∀ i < p, f i ≥ 1 ∧ ∑ i in (finset.range p), (f i)^2 = 1) 
  ↔ n = 6 := 
by 
  sorry
  
end smallest_n_for_partition_l214_214771


namespace frog_escape_pad_3_l214_214892

noncomputable def probability_of_escape (N : ℕ) : ℚ :=
  match N with
  | 0    => 0
  | 20   => 1
  | N    => if N < 20 then (2 * N / 20) * probability_of_escape (N - 1) + (1 - 2 * N / 20) * probability_of_escape (N + 1) else 0

theorem frog_escape_pad_3 : probability_of_escape 3 = 4 / 11 := by
  sorry

end frog_escape_pad_3_l214_214892


namespace john_pays_after_discount_l214_214913

theorem john_pays_after_discount :
  ∀ (num_toys : ℕ) (cost_per_toy : ℕ) (discount_rate : ℚ),
  num_toys = 5 → cost_per_toy = 3 → discount_rate = 0.20 →
  let total_cost := num_toys * cost_per_toy in
  let discount := discount_rate * ↑total_cost in
  let amount_paid := total_cost - discount in
  amount_paid = 12 :=
by
  intros num_toys cost_per_toy discount_rate hnum_toys hcost_per_toy hdiscount_rate
  rw [hnum_toys, hcost_per_toy, hdiscount_rate]
  let total_cost := num_toys * cost_per_toy
  let discount := discount_rate * ↑total_cost
  let amount_paid := total_cost - discount
  have htotal_cost : total_cost = 15 := by norm_num
  have hdiscount : discount = 3 := by norm_num
  have hamount_paid : amount_paid = 12 := by norm_num
  exact hamount_paid

end john_pays_after_discount_l214_214913


namespace john_total_payment_l214_214915

def cost_per_toy := 3
def number_of_toys := 5
def discount_rate := 0.2

theorem john_total_payment :
  (number_of_toys * cost_per_toy) - ((number_of_toys * cost_per_toy) * discount_rate) = 12 :=
by
  sorry

end john_total_payment_l214_214915


namespace garrett_bought_peanut_granola_bars_l214_214806

def garrett_granola_bars (t o : ℕ) (h_t : t = 14) (h_o : o = 6) : ℕ :=
  t - o

theorem garrett_bought_peanut_granola_bars : garrett_granola_bars 14 6 rfl rfl = 8 :=
  by
    unfold garrett_granola_bars
    rw [Nat.sub_eq_of_eq_add]
    sorry

end garrett_bought_peanut_granola_bars_l214_214806


namespace unique_divisor_of_2_pow_n_minus_1_l214_214029

theorem unique_divisor_of_2_pow_n_minus_1 : ∀ (n : ℕ), n ≥ 1 → n ∣ (2^n - 1) → n = 1 := 
by
  intro n h1 h2
  sorry

end unique_divisor_of_2_pow_n_minus_1_l214_214029


namespace find_values_of_a2_b2_l214_214465

-- Define the conditions
variables {a b : ℝ}
variable (h1 : a > b)
variable (h2 : b > 0)
variable (hP : (-2, (Real.sqrt 14) / 2) ∈ { p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1 })
variable (hCircle : ∀ Q : ℝ × ℝ, (Q ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 2 }) → (∃ tA tB : ℝ × ℝ, (tA ∈ { p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1 }) ∧ (tB ∈ { p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1 }) ∧ (tA = - tB ∨ tB = - tA) ∧ ((tA.1 + tB.1)/2 = (-2 + tA.1)/2) ))

-- The theorem to be proven
theorem find_values_of_a2_b2 : a^2 + b^2 = 15 :=
sorry

end find_values_of_a2_b2_l214_214465


namespace geometric_progression_first_term_l214_214655

theorem geometric_progression_first_term (a r : ℝ) 
    (h_sum_inf : a / (1 - r) = 8)
    (h_sum_two : a * (1 + r) = 5) :
    a = 2 * (4 - Real.sqrt 6) ∨ a = 2 * (4 + Real.sqrt 6) :=
sorry

end geometric_progression_first_term_l214_214655


namespace max_x_inequality_k_l214_214522

theorem max_x_inequality_k (k : ℝ) (h : ∀ x : ℝ, |x^2 - 4 * x + k| + |x - 3| ≤ 5 → x ≤ 3) : k = 8 :=
sorry

end max_x_inequality_k_l214_214522


namespace Buffy_whiskers_l214_214799

def whiskers_Juniper : ℕ := 12
def whiskers_Puffy : ℕ := 3 * whiskers_Juniper
def whiskers_Scruffy : ℕ := 2 * whiskers_Puffy
def whiskers_Buffy : ℕ := (whiskers_Puffy + whiskers_Scruffy + whiskers_Juniper) / 3

theorem Buffy_whiskers : whiskers_Buffy = 40 := by
  sorry

end Buffy_whiskers_l214_214799


namespace magazine_page_height_l214_214953

theorem magazine_page_height
  (charge_per_sq_inch : ℝ := 8)
  (half_page_cost : ℝ := 432)
  (page_width : ℝ := 12) : 
  ∃ h : ℝ, (1/2) * h * page_width * charge_per_sq_inch = half_page_cost :=
by sorry

end magazine_page_height_l214_214953


namespace minor_axis_length_l214_214065

open Real

noncomputable def length_minor_axis (P1 P2 P3 P4 P5: ℝ × ℝ) (conic_axes_parallel: Prop) : ℝ :=
  sorry

theorem minor_axis_length :
  length_minor_axis (-2, 1) (0, 0) (0, 3) (4, 0) (4, 3) 
    (¬ collinear (-2, 1) (0, 0) (0, 3) ∧ ¬ collinear (-2, 1) (0, 0) (4, 0)
     ∧ ¬ collinear (-2, 1) (0, 0) (4, 3) ∧ ¬ collinear (0, 0) (0, 3) (4, 0)
     ∧ ¬ collinear (0, 0) (0, 3) (4, 3) ∧ ¬ collinear (0, 3) (4, 0) (4, 3))
    = 2 * sqrt 3 :=
sorry


end minor_axis_length_l214_214065


namespace mask_digit_correctness_l214_214624

noncomputable def elephant_mask_digit : ℕ :=
  6
  
noncomputable def mouse_mask_digit : ℕ :=
  4

noncomputable def guinea_pig_mask_digit : ℕ :=
  8

noncomputable def panda_mask_digit : ℕ :=
  1

theorem mask_digit_correctness :
  (∃ (d1 d2 d3 d4 : ℕ), d1 * d1 = 16 ∧ d2 * d2 = 64 ∧ d3 * d3 = 49 ∧ d4 * d4 = 81) →
  elephant_mask_digit = 6 ∧ mouse_mask_digit = 4 ∧ guinea_pig_mask_digit = 8 ∧ panda_mask_digit = 1 :=
by
  -- skip the proof
  sorry

end mask_digit_correctness_l214_214624


namespace probability_correct_l214_214317

open Classical
noncomputable theory

def a_set : Set ℚ := {1/3, 1/2, 2, 3}
def b_set : Set ℚ := {-1, 1, -2, 2}

def passes_through_third_quadrant (a b : ℚ) : Prop :=
∃ x : ℚ, a ^ x + b < 0 ∧ x < 0 

def valid_cases : Finset (ℚ × ℚ) :=
{((1/3), -2), ((1/2), -2), (2, -1), (2, -2), (3, -1), (3, -2)}

def total_cases : ℕ := 4 * 4

def favorable_cases : ℕ := (valid_cases.card : ℕ)

def probability_third_quadrant : ℚ := favorable_cases / total_cases

theorem probability_correct :
  probability_third_quadrant = 3/8 :=
sorry

end probability_correct_l214_214317


namespace fixed_point_of_exponential_function_l214_214629

theorem fixed_point_of_exponential_function (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) :
  f 1 = 3 :=
by
  let f : ℝ → ℝ := λ x, a^(x-1) + 2
  sorry

end fixed_point_of_exponential_function_l214_214629


namespace equation_one_real_root_l214_214514

-- Definitions of the conditions
def domain_pos (k : ℝ) : Set ℝ := { x | x > 0 }
def domain_neg (k : ℝ) : Set ℝ := { x | -1 < x ∧ x < 0 }

-- Main theorem statement
theorem equation_one_real_root (k : ℝ) :
  (∀ x ∈ (if k > 0 then domain_pos k else domain_neg k), log (k * x) = 2 * log (x + 1))
  → (k = 4 ∨ k < 0) :=
sorry

end equation_one_real_root_l214_214514


namespace probability_XOXOXOX_is_one_over_thirty_five_l214_214437

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_arrangement : ℤ :=
  let total_arrangements := binomial_coefficient 7 4
  let favorable_outcomes := 1
  favorable_outcomes / total_arrangements

theorem probability_XOXOXOX_is_one_over_thirty_five :
  probability_of_arrangement = (1 : ℤ) / 35 := 
  by
  sorry

end probability_XOXOXOX_is_one_over_thirty_five_l214_214437


namespace prove_if_alpha_parallel_beta_and_a_perpendicular_beta_then_a_perpendicular_alpha_l214_214054

-- Definitions of the entities involved
variables {L : Type} -- All lines
variables {P : Type} -- All planes

-- Relations
variables (perpendicular : L → P → Prop)
variables (parallel : P → P → Prop)

-- Conditions
variables (a b : L)
variables (α β : P)

-- Statements we want to prove
theorem prove_if_alpha_parallel_beta_and_a_perpendicular_beta_then_a_perpendicular_alpha
  (H1 : parallel α β) 
  (H2 : perpendicular a β) : 
  perpendicular a α :=
  sorry

end prove_if_alpha_parallel_beta_and_a_perpendicular_beta_then_a_perpendicular_alpha_l214_214054


namespace factor_expression_l214_214375

theorem factor_expression (a b c : ℝ) :
  let num := (a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3
  let denom := (a - b)^3 + (b - c)^3 + (c - a)^3
  (denom ≠ 0) →
  num / denom = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end factor_expression_l214_214375


namespace sum_of_real_solutions_eq_neg_47_over_10_l214_214398

theorem sum_of_real_solutions_eq_neg_47_over_10 :
  ∑ (x : ℝ) in {x : ℝ | (x - 3) / (x^2 + 3 * x - 4) = (x - 4) / (x^2 - 8 * x + 7)}, x = -47 / 10 :=
by sorry

end sum_of_real_solutions_eq_neg_47_over_10_l214_214398


namespace first_division_percentage_l214_214536

-- Defining the problem conditions
def total_students : ℕ := 300
def percentage_second_division : ℕ := 54
def students_just_passed : ℕ := 63
def students_failed : ℕ := 0

-- Proving the percentage of students who got first division is 25%
theorem first_division_percentage : 
  (total_students - ((percentage_second_division * total_students / 100).to_nat + students_just_passed)).to_rat / total_students * 100 = 25 := 
by 
  -- sorry to skip the proof
  sorry

end first_division_percentage_l214_214536


namespace length_within_cube_l214_214042

noncomputable def point_X : ℝ × ℝ × ℝ := (0, 0, 0)
noncomputable def point_Y : ℝ × ℝ × ℝ := (5, 5, 5)
noncomputable def cube_with_edge_4_min : ℝ × ℝ × ℝ := (-2, -2, -2)
noncomputable def cube_with_edge_4_max : ℝ × ℝ × ℝ := (2, 2, 2)

theorem length_within_cube :
  let segment_length := dist point_X point_Y in
  let portion_within_cube := dist cube_with_edge_4_min cube_with_edge_4_max in
  portion_within_cube = 4 * real.sqrt 3 :=
by sorry

end length_within_cube_l214_214042


namespace sum_not_divisible_by_5_l214_214962

theorem sum_not_divisible_by_5 (n : ℕ) : ¬ (5 ∣ ∑ k in finset.range (n + 1), 2^(3 * k) * nat.choose (2 * n + 1) (2 * k + 1)) :=
sorry

end sum_not_divisible_by_5_l214_214962


namespace thirty_seven_times_twenty_three_eq_eight_hundred_fifty_one_l214_214752

theorem thirty_seven_times_twenty_three_eq_eight_hundred_fifty_one : 37 * 23 = 851 := by
  sorry

end thirty_seven_times_twenty_three_eq_eight_hundred_fifty_one_l214_214752


namespace ruby_total_classes_l214_214968

noncomputable def average_price_per_class (pack_cost : ℝ) (pack_classes : ℕ) : ℝ :=
  pack_cost / pack_classes

noncomputable def additional_class_price (average_price : ℝ) : ℝ :=
  average_price + (1/3 * average_price)

noncomputable def total_classes_taken (total_payment : ℝ) (pack_cost : ℝ) (pack_classes : ℕ) : ℕ :=
  let avg_price := average_price_per_class pack_cost pack_classes
  let additional_price := additional_class_price avg_price
  let additional_classes := (total_payment - pack_cost) / additional_price
  pack_classes + Nat.floor additional_classes -- We use Nat.floor to convert from real to natural number of classes

theorem ruby_total_classes 
  (pack_cost : ℝ) 
  (pack_classes : ℕ) 
  (total_payment : ℝ) 
  (h_pack_cost : pack_cost = 75) 
  (h_pack_classes : pack_classes = 10) 
  (h_total_payment : total_payment = 105) :
  total_classes_taken total_payment pack_cost pack_classes = 13 :=
by
  -- The proof would go here
  sorry

end ruby_total_classes_l214_214968


namespace mf_length_l214_214487

theorem mf_length (F M N D : Point) (x1 x2 k : ℝ)
    (h1 : y^2 = 4 * x)
    (h2 : y = k * (x - 1))
    (h3 : MD = 2 * FN)
    (h4 : D ∈ line MF)
    (h5 : QD ⊥ MF)
    (h6 : MD = 2 * FN)
    (h7 : PF ⊥ MN) :
  |MF| = sqrt 3 + 2 :=
  sorry

end mf_length_l214_214487


namespace triangle_A_B_C_with_P_conditions_l214_214148

theorem triangle_A_B_C_with_P_conditions :
  ∃ (p q r : ℕ), 
    let ABC_is_right := ∀ (A B C : Point), C = A.right_angle,
        BAC_lt_45 := ∠BAC < 45,
        AB_eq_5 := 5,
        P_on_AB_and_angle_relation := ∃ (P : Point), P ∈ line_segment (A, B) ∧ ∠APC = 3 * ∠ACP,
        CP_eq_2 := CP = 2 in
    let AP_BP_ratio := p + q * √r = 3 + 2 * √130 in
    p + q + r = 135 ∧ p > 0 ∧ q > 0 ∧ r > 0 := sorry

end triangle_A_B_C_with_P_conditions_l214_214148


namespace range_of_g_l214_214783

open Real 

noncomputable def g (x : ℝ) : ℝ :=
  (cos x * (2 * (sin x)^2 + (sin x)^4 + 2 * (cos x)^2 + (cos x)^2 * (sin x)^2)) /
  (cot x * ((csc x) - cos x * cot x))

theorem range_of_g :
  ∀ (x : ℝ), (∃ n : ℤ, x = n * π) → (g x ∈ Set.Icc 2 3) := 
by
  intro x h
  sorry

end range_of_g_l214_214783


namespace gcd_98_63_l214_214669

-- Definition of gcd
def gcd_euclidean := ∀ (a b : ℕ), ∃ (g : ℕ), gcd a b = g

-- Statement of the problem using Lean
theorem gcd_98_63 : gcd 98 63 = 7 := by
  sorry

end gcd_98_63_l214_214669


namespace cube_painting_distinct_ways_l214_214761

theorem cube_painting_distinct_ways : ∃ n : ℕ, n = 7 := sorry

end cube_painting_distinct_ways_l214_214761


namespace correct_operation_l214_214682

theorem correct_operation (a b : ℝ) : 
  (2 * a) * (3 * a) = 6 * a^2 :=
by
  -- The proof would be here; using "sorry" to skip the actual proof steps.
  sorry

end correct_operation_l214_214682


namespace probability_XOXOXOX_is_1_div_35_l214_214421

def count_combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def num_ways_to_choose_positions_for_X (total_positions : ℕ) (num_X : ℕ) : ℕ := 
  count_combinations total_positions num_X

def num_ways_for_specific_arrangement_XOXOXOX : ℕ := 1

def probability_of_XOXOXOX (num_ways_total : ℕ) (num_ways_specific : ℕ) : ℚ := 
  num_ways_specific / num_ways_total

theorem probability_XOXOXOX_is_1_div_35 :
  probability_of_XOXOXOX (num_ways_to_choose_positions_for_X 7 4) num_ways_for_specific_arrangement_XOXOXOX = 1 / 35 := by
  sorry

end probability_XOXOXOX_is_1_div_35_l214_214421


namespace Marcy_sips_interval_l214_214183

theorem Marcy_sips_interval:
  ∀ (total_volume_ml sip_volume_ml total_time min_per_sip: ℕ),
  total_volume_ml = 2000 →
  sip_volume_ml = 40 →
  total_time = 250 →
  min_per_sip = total_time / (total_volume_ml / sip_volume_ml) →
  min_per_sip = 5 :=
by
  intros total_volume_ml sip_volume_ml total_time min_per_sip hv hs ht hm
  rw [hv, hs, ht] at hm
  simp at hm
  exact hm

end Marcy_sips_interval_l214_214183


namespace edward_games_start_l214_214367

theorem edward_games_start (sold_games : ℕ) (boxes : ℕ) (games_per_box : ℕ) 
  (h_sold : sold_games = 19) (h_boxes : boxes = 2) (h_game_box : games_per_box = 8) : 
  sold_games + boxes * games_per_box = 35 := 
  by 
    sorry

end edward_games_start_l214_214367


namespace expected_flips_non_leap_year_l214_214729

noncomputable def expected_flips_per_day : ℚ := 32 / 9

theorem expected_flips_non_leap_year (days : ℕ) (h_days : days = 365) : 
    ∑ (i : ℕ) in Finset.range days, expected_flips_per_day = 1306.67 := 
    by
    sorry -- Proof goes here

end expected_flips_non_leap_year_l214_214729


namespace michael_can_escape_l214_214585

-- Define the basic conditions and entities
structure Circle where
  radius : ℝ

structure Position where
  x : ℝ
  y : ℝ

def distance (p1 p2 : Position) : ℝ :=
  (p2.x - p1.x)^2 + (p2.y - p1.y)^2

def initial_position : Position := { x := 0, y := 0 }
def move_distance : ℝ := 1
def max_radius : ℝ := 100
def circle : Circle := { radius := max_radius }

-- Definition of the main problem
theorem michael_can_escape (steps : ℕ) (pos : Position) (r : ℝ) : steps = 10000 ∧ distance initial_position pos ≥ r^2 ∧ r = max_radius ->
  ∃ (new_pos : Position), distance initial_position new_pos > max_radius^2 :=
  sorry -- Proof to be provided

end michael_can_escape_l214_214585


namespace find_f1_l214_214480

def f (x : ℝ) (a b : ℝ) : ℝ := (1/3) * x^3 + a^2 * x^2 + a * x + b

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := x^2 + 2 * a^2 * x + a

theorem find_f1 (a b : ℝ) 
  (h1 : f (-1) a b = -7 / 12) 
  (h2 : f_prime (-1) a = 0) : 
  f 1 a b = 25 / 12 ∨ f 1 a b = 1 / 12 :=
sorry

end find_f1_l214_214480


namespace product_evaluation_l214_214742

theorem product_evaluation :
  (∏ k in Finset.range 10, ((k+1) * (k+4)) / ((k+6) * (k+6))) = 3628800 / 44100 := by
  sorry

end product_evaluation_l214_214742


namespace steve_initial_amount_l214_214228

theorem steve_initial_amount
  (P : ℝ) 
  (h : (1.1^2) * P = 121) : 
  P = 100 := 
by 
  sorry

end steve_initial_amount_l214_214228


namespace unique_flavors_count_l214_214610

def numOrangeCandies : ℕ := 6
def numPurpleCandies : ℕ := 4

-- Define a set of all possible candy combinations, excluding (0, 0)
def candy_combinations : Set (ℕ × ℕ) :=
  { p | p.1 ≤ numOrangeCandies ∧ p.2 ≤ numPurpleCandies ∧ ¬(p.1 = 0 ∧ p.2 = 0) }

-- Define a function to reduce a ratio to its simplest form
def simplify_ratio (x y : ℕ) : Option (ℕ × ℕ) :=
  if y = 0 then none
  else if x = 0 then some (0, 1)
  else some ((x / (Nat.gcd x y)), (y / (Nat.gcd x y)))

-- Create a set of simplified ratios from the candy combinations
def simplified_ratios : Set (Option (ℕ × ℕ)) :=
  { simplify_ratio p.1 p.2 | p ∈ candy_combinations }

theorem unique_flavors_count :
  simplified_ratios.to_finset.card = 14 := sorry

end unique_flavors_count_l214_214610


namespace max_value_of_x_plus_y_l214_214692

theorem max_value_of_x_plus_y :
  ∃ (a b c d x y : ℕ), 
  {189, 320, 287, 264, x, y} = 
  {a + b, a + c, a + d, b + c, b + d, c + d} ∧ 
  x + y = 530 :=
sorry

end max_value_of_x_plus_y_l214_214692


namespace find_n_l214_214834

theorem find_n (m n : ℕ) (h1: m = 34)
               (h2: (1^(m+1) / 5^(m+1)) * (1^n / 4^n) = 1 / (2 * 10^35)) : 
               n = 18 :=
by
  sorry

end find_n_l214_214834


namespace final_number_left_l214_214957

theorem final_number_left (n : ℕ) (h_n : n = 2019) : 
  (1 + 2 + 3 + ... + 2019) % 17 = 1 :=
by
  have h_sum : (1 + 2 + 3 + ... + 2019) = (2019 * (2019 + 1)) / 2,
    from sorry, -- This follows from the formula for the sum of the first n natural numbers
  rw h_n at h_sum,
  have h_mod : (2019 * (2019 + 1) / 2) % 17 = 1,
    from sorry, -- This follows from the modulo calculation steps
  exact h_mod

end final_number_left_l214_214957


namespace base_equivalence_l214_214107

theorem base_equivalence : 
  ∀ (b : ℕ), (b^3 + 3*b^2 + 4)^2 = 9*b^4 + 9*b^3 + 2*b^2 + 2*b + 5 ↔ b = 10 := 
by
  sorry

end base_equivalence_l214_214107


namespace books_finished_correct_l214_214954

def miles_traveled : ℕ := 6760
def miles_per_book : ℕ := 450
def books_finished (miles_traveled miles_per_book : ℕ) : ℕ :=
  miles_traveled / miles_per_book

theorem books_finished_correct :
  books_finished miles_traveled miles_per_book = 15 :=
by
  -- The steps of the proof would go here
  sorry

end books_finished_correct_l214_214954


namespace length_of_second_train_is_correct_l214_214686

-- Defining the conditions
def first_train_length : ℝ := 270
def first_train_speed_kmph : ℝ := 120
def second_train_speed_kmph : ℝ := 80
def crossing_time_seconds : ℝ := 9

-- Defining conversion factors
def kmph_to_mps (kmph : ℝ) : ℝ := kmph * (1000 / 3600)

-- Converting speeds to m/s
def first_train_speed : ℝ := kmph_to_mps first_train_speed_kmph
def second_train_speed : ℝ := kmph_to_mps second_train_speed_kmph

-- Calculating relative speed
def relative_speed : ℝ := first_train_speed + second_train_speed

-- Using the distance = speed * time formula
def total_distance_covered : ℝ := relative_speed * crossing_time_seconds

-- Equation for the total distance covered
def length_of_other_train : ℝ := total_distance_covered - first_train_length

-- The proof statement
theorem length_of_second_train_is_correct : length_of_other_train = 229.95 :=
by
  -- Proof is omitted
  sorry

end length_of_second_train_is_correct_l214_214686


namespace shaoxing_2014_height_of_cloud_l214_214898

noncomputable def height_of_cloud (h : ℝ) (elevation_angle : ℝ) (depression_angle : ℝ) : ℝ :=
  let x := (h * elevation_angle.tan) in
  (h + x * depression_angle.tan) / (1 - depression_angle.tan * elevation_angle.tan)

theorem shaoxing_2014_height_of_cloud :
  height_of_cloud 10 (Real.toRadians 30) (Real.toRadians 45) = 37.3 :=
by
  sorry

end shaoxing_2014_height_of_cloud_l214_214898


namespace number_of_valid_permutations_l214_214569

-- Definition of the problem in terms of Lean
def is_valid_permutation (p : List ℕ) : Prop :=
p.perm  (List.range 5).map (· + 1) ∧
∀ i j k : ℕ, i < j → j < k → k < 5 → p.get? i < p.get? j → p.get? j < p.get? k → False

-- Problem statement in Lean
theorem number_of_valid_permutations : 
   (Finset.univ.filter is_valid_permutation).card = 42 :=
sorry

end number_of_valid_permutations_l214_214569


namespace hyperbola_equation_is_correct_l214_214830

-- Given Conditions
def hyperbola_eq (x y : ℝ) (a : ℝ) : Prop := (x^2) / (a^2) - (y^2) / 4 = 1
def asymptote_eq (x y : ℝ) : Prop := y = (1 / 2) * x

-- Correct answer to be proven
def hyperbola_correct (x y : ℝ) : Prop := (x^2) / 16 - (y^2) / 4 = 1

theorem hyperbola_equation_is_correct (x y : ℝ) (a : ℝ) :
  (hyperbola_eq x y a) → (asymptote_eq x y) → (a = 4) → hyperbola_correct x y :=
by 
  intros h_hyperbola h_asymptote h_a
  sorry

end hyperbola_equation_is_correct_l214_214830


namespace planes_through_point_parallel_to_line_l214_214874

theorem planes_through_point_parallel_to_line {Point Line Plane : Type} 
  (P : Point) (l : Line) 
  (plane_passes_through_point : Plane → Point → Prop)
  (plane_parallel_to_line : Plane → Line → Prop)
  (point_outside_line : Point → Line → Prop) : 
  point_outside_line P l → 
  (∃ plane_set : set Plane, (∀ π ∈ plane_set, plane_passes_through_point π P ∧ plane_parallel_to_line π l) ∧ set.infinite plane_set) :=
sorry

end planes_through_point_parallel_to_line_l214_214874


namespace parallelogram_from_two_equidistant_lines_l214_214672

open set

variables {Point : Type} [topological_space Point]

/-- A point type in a topological space -/
def isEquidistantLine (A B C D e : set Point) :=
  ∀ (p ∈ {A, B, C, D}), dist p e = dist p e

theorem parallelogram_from_two_equidistant_lines 
  {A B C D e f : set Point} 
  (h₁ : isEquidistantLine A B C D e)
  (h₂ : isEquidistantLine A B C D f) : 
  form_parallelogram A B C D :=
sorry

end parallelogram_from_two_equidistant_lines_l214_214672


namespace find_minimum_and_maximum_of_function_l214_214392

theorem find_minimum_and_maximum_of_function :
  (∀ x : ℝ, -4/3 ≤ x ∧ x ≤ 3/4 → 
    (let y := sqrt (3*x + 4) + sqrt (3 - 4*x) in
    (∀ ε > 0, y ≥ 5/2 - ε) ∧ (∀ ε > 0, y ≤ 5 * sqrt 21 / 6 + ε))) :=
begin
  sorry,
end

end find_minimum_and_maximum_of_function_l214_214392


namespace divisors_of_64n5_l214_214791

theorem divisors_of_64n5 (n : ℕ) (h_positive : 0 < n) (h_divisors : ∃ k, 210 * k^4 = 210 ∧ number_of_divisors (210 * k^4) = 210) : 
  number_of_divisors (64 * n^5) = 3960 := 
sorry

end divisors_of_64n5_l214_214791


namespace triangle_side_length_l214_214972

theorem triangle_side_length (R Q S : ℝ) (h₁: ∃ (cos_R : ℝ), cos_R = 5/13)
  (h₂: ∃ (RS : ℝ), RS = 13)
  (h₃: ∃ (QR : ℝ), QR = 5)
  (h₄: ∃ (right_triangle : Prop), right_triangle)
  (right_triangle_def : right_triangle ↔ (QS^2 + QR^2 = RS^2)) :
  ∃ (QS : ℝ), QS = 12 :=
by {
  sorry
}

end triangle_side_length_l214_214972


namespace circles_concur_at_incenter_l214_214759

noncomputable def incenter (A B C : Point) : Point := sorry -- Define the incenter

theorem circles_concur_at_incenter
  (A B C : Point)
  (D E F : Point)
  (circle1 : Circle)
  (circle2 : Circle)
  (circle3 : Circle)
  (hD : D = midpoint (arc BC))
  (hE : E = midpoint (arc CA))
  (hF : F = midpoint (arc AB))
  (hcircle1 : circle1.center = D ∧ (B ∈ circle1 ∧ C ∈ circle1))
  (hcircle2 : circle2.center = E ∧ (C ∈ circle2 ∧ A ∈ circle2))
  (hcircle3 : circle3.center = F ∧ (A ∈ circle3 ∧ B ∈ circle3)) :
  ∃ I : Point, I = incenter A B C ∧ I ∈ circle1 ∧ I ∈ circle2 ∧ I ∈ circle3 :=
sorry

end circles_concur_at_incenter_l214_214759


namespace decreasing_intervals_of_cosine_shift_l214_214650

noncomputable def strictly_decreasing_intervals (k : ℤ) : set ℝ := Ioo (2 * k * Real.pi - Real.pi / 4) (2 * k * Real.pi + 3 * Real.pi / 4)

theorem decreasing_intervals_of_cosine_shift :
  ∀ (x k : ℤ) (a ∈ strictly_decreasing_intervals k), 
  ∀ b ∈ strictly_decreasing_intervals k, 
  a < b → 3 * Real.cos (Real.pi / 4 + a) > 3 * Real.cos (Real.pi / 4 + b) :=
sorry

end decreasing_intervals_of_cosine_shift_l214_214650


namespace prime_factorization_344_sum_first_last_torn_out_pages_quantity_torn_out_pages_l214_214921

theorem prime_factorization_344 : prime_factors 344 = [2, 2, 2, 43] :=
sorry

theorem sum_first_last_torn_out_pages (n : ℕ) (x y : ℕ) : 
  (n = 344) → 
  (∃ y, 2x + 2y + 1 = 43) →
  (x = 21 - y) →
  (sum_of_torn_out_pages x y = n) →
  ((x + 1) + (x + 2y) = 43) :=
sorry

theorem quantity_torn_out_pages (n : ℕ) (y : ℕ) :
  (n = 344) →
  (2x + 2y + 1 = 43) →
  quantity_of_pages TornOut y = 16 :=
sorry

end prime_factorization_344_sum_first_last_torn_out_pages_quantity_torn_out_pages_l214_214921


namespace center_of_mass_of_equilateral_triangle_l214_214992

open Real

structure MassPoint :=
  (mass : ℝ)
  (point : ℝ × ℝ)

def centerOfMass (points : List MassPoint) : ℝ × ℝ :=
  let total_mass := points.foldl (λ acc p => acc + p.mass) 0
  let (cx, cy) := points.foldl (λ (accx, accy) p => 
    (accx + p.mass * p.point.fst, accy + p.mass * p.point.snd)) (0, 0)
  (cx / total_mass, cy / total_mass)

theorem center_of_mass_of_equilateral_triangle :
  ∀ (a : ℝ) (m1 m2 m3 : ℝ), a = 1 → m1 = 8 → m2 = 9 → m3 = 10 →
  let A := (0, 0)
  let B := (a, 0)
  let C := (a / 2, (a * sqrt 3) / 2)
  centerOfMass [
    MassPoint.mk m1 A,
    MassPoint.mk m2 B,
    MassPoint.mk m3 C
  ] = (0.5121, 0.2588) := sorry

print center_of_mass_of_equilateral_triangle

end center_of_mass_of_equilateral_triangle_l214_214992


namespace marge_funds_l214_214186

theorem marge_funds (initial_winnings : ℕ)
    (tax_fraction : ℕ)
    (loan_fraction : ℕ)
    (savings_amount : ℕ)
    (investment_fraction : ℕ)
    (tax_paid leftover_for_loans savings_after_loans final_leftover final_leftover_after_investment : ℕ) :
    initial_winnings = 12006 →
    tax_fraction = 2 →
    leftover_for_loans = initial_winnings / tax_fraction →
    loan_fraction = 3 →
    savings_after_loans = leftover_for_loans / loan_fraction →
    savings_amount = 1000 →
    final_leftover = leftover_for_loans - savings_after_loans - savings_amount →
    investment_fraction = 5 →
    final_leftover_after_investment = final_leftover - (savings_amount / investment_fraction) →
    final_leftover_after_investment = 2802 :=
by
  intros
  sorry

end marge_funds_l214_214186


namespace valid_sequences_count_100_l214_214537

def is_valid_sequence (seq : List ℕ) : Prop :=
  ∀ i, 0 < i ∧ i < seq.length → abs (seq.nthLe i (by sorry) - seq.nthLe (i-1) (by sorry)) ≤ 1

def count_valid_sequences (n : ℕ) : ℕ :=
  let numbers := List.range (n + 1)
  let permutations := List.permutations numbers
  permutations.count is_valid_sequence

theorem valid_sequences_count_100 : count_valid_sequences 100 = 2 :=
by
  sorry

end valid_sequences_count_100_l214_214537


namespace convex_pentagons_from_15_points_l214_214023

theorem convex_pentagons_from_15_points : (Nat.choose 15 5) = 3003 := 
by
  sorry

end convex_pentagons_from_15_points_l214_214023


namespace count_six_digit_numbers_with_three_even_three_odd_l214_214863

-- Definitions of the problem conditions
def is_six_digit (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

def has_three_even_and_three_odd_digits (n : ℕ) : Prop :=
  (nat.num_even_digits n = 3) ∧ (nat.num_odd_digits n = 3)

-- Lean 4 statement
theorem count_six_digit_numbers_with_three_even_three_odd :
  ∃ (count : ℕ), count = 281250 ∧
  (∀ n : ℕ, is_six_digit n ∧ n / 100000 > 0 ∧ has_three_even_and_three_odd_digits n → True) :=
sorry

end count_six_digit_numbers_with_three_even_three_odd_l214_214863


namespace find_b_l214_214639

theorem find_b (a b c : ℝ) (h₁ : c = 3)
  (h₂ : -a / 3 = c)
  (h₃ : -a / 3 = 1 + a + b + c) :
  b = -16 :=
by
  -- The solution steps are not necessary to include here.
  sorry

end find_b_l214_214639


namespace first_number_in_list_is_202_l214_214979

variable (numbers : List ℤ) (x : ℤ)

def average_of_list (lst : List ℤ) : ℚ :=
  (lst.sum : ℚ) / lst.length

theorem first_number_in_list_is_202 (h : average_of_list [202, 204, 205, 206, 209, 209, 210, 212, x] = 207) :
  numbers.head = 202 :=
by
  have l : List ℤ := [202, 204, 205, 206, 209, 209, 210, 212, x]
  have average : average_of_list l = 207 := h
  sorry

end first_number_in_list_is_202_l214_214979


namespace number_of_zesty_two_digit_numbers_l214_214878

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

def is_zesty (n : ℕ) : Prop :=
  ∃ x y : ℕ, x > 1 ∧ y > 1 ∧ x * y = n ∧ sum_of_digits x * sum_of_digits y = sum_of_digits n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem number_of_zesty_two_digit_numbers : 
  finset.card {n : ℕ | is_two_digit n ∧ is_zesty n}.to_finset = 34 :=
sorry

end number_of_zesty_two_digit_numbers_l214_214878


namespace fraction_of_sum_of_arithmetic_sequences_l214_214990

-- Define the sequences, their sums, and the given ratio condition
def isArithmeticSequence (a : ℕ → ℚ) : Prop :=
∀ n : ℕ, ∃ d : ℚ, a (n + 1) = a n + d

def sumOfFirstNTerms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
(0 to n).sum a

theorem fraction_of_sum_of_arithmetic_sequences 
  (a b : ℕ → ℚ)
  (Sa : ℕ → ℚ := sumOfFirstNTerms a)
  (Sb : ℕ → ℚ := sumOfFirstNTerms b) :
  isArithmeticSequence a →
  isArithmeticSequence b →
  (∀ n : ℕ, Sa n / Sb n = (3 * n + 1) / (n + 3)) →
  (a2 a := a 2) (a20 a := a 20) (b7 b := b 7) (b15 b := b 15) :
  (a2 + a20) / (b7 + b15) = 8 / 3 := by
sorry

end fraction_of_sum_of_arithmetic_sequences_l214_214990


namespace money_earned_l214_214967

variable (earn_per_lawn total_lawns forgotten_lawns : ℕ)

-- Given conditions
def earn_per_lawn := 9
def total_lawns := 14
def forgotten_lawns := 8

-- Define and prove the money actually earned
theorem money_earned : total_lawns - forgotten_lawns * earn_per_lawn = 54 :=
by
  sorry

end money_earned_l214_214967


namespace solution_l214_214092

noncomputable def problem_statement : Prop :=
  ∀ (l m n : Line) (α β : Plane), 
    (¬ collinear l m n) ∧ (α ≠ β) →
    (m ⊥ α) ∧ (m ⊥ β) →
    α ∥ β

-- statement in Lean 4 without the proof
theorem solution : problem_statement :=
  sorry

end solution_l214_214092


namespace special_fractions_sums_count_correct_l214_214756

def special_fractions_sums_count : Nat :=
  let special_pairs := [(1, 11), (2, 10), (3, 9), (4, 8), (5, 7), (6, 6), (7, 5), (8, 4), (9, 3), (10, 2), (11, 1)]
  let special_fractions := special_pairs.map (λ (a, b) => (a : ℚ) / b)
  let sums := special_fractions.bind (λ x => special_fractions.map (λ y => x + y))
  let distinct_sums := sums.filter (λ sum => (sum.ceil - sum.floor) = 0).eraseDups
  distinct_sums.length

theorem special_fractions_sums_count_correct :
  special_fractions_sums_count = 8 :=
by
  sorry

end special_fractions_sums_count_correct_l214_214756


namespace intersect_M_N_l214_214066

def M := {x : ℝ | x^2 + 2 * x - 3 = 0}
def N := {x : ℝ | ∃ y : ℝ, y = real.sqrt (2^x - 1 / 2)}

theorem intersect_M_N : M ∩ N = {1} :=
by sorry

end intersect_M_N_l214_214066


namespace planes_parallel_if_perpendicular_line_l214_214462

/-- Proposition about the relationship between planes and a perpendicular line -/
theorem planes_parallel_if_perpendicular_line (α β : Type)
  [plane α] [plane β]
  (h_diff : α ≠ β)
  (m : Type)
  [line m]
  (h_perp_α : m ⟂ α)
  (h_perp_β : m ⟂ β) : 
  α ∥ β :=
sorry

end planes_parallel_if_perpendicular_line_l214_214462


namespace obtuse_triangle_count_205320_l214_214847

-- Define the vertices of the regular 120-gon
def vertices := Fin 120

-- Define the condition to form an obtuse triangle
def is_obtuse_triangle (k l m : vertices) : Prop :=
  let m_k := (m.val - k.val) % 120
  0 < m_k ∧ m_k < 60

-- Define the number of ways to choose three vertices to form an obtuse triangle
def obtuse_triangle_count : ℕ :=
  ∑ k in vertices, ∑ l in vertices, ∑ m in vertices, if k.val < l.val ∧ l.val < m.val ∧ is_obtuse_triangle k l m then 1 else 0

theorem obtuse_triangle_count_205320 : obtuse_triangle_count = 205320 := sorry

end obtuse_triangle_count_205320_l214_214847


namespace remainder_of_division_l214_214677

noncomputable def P (x : ℝ) := 8 * x^4 - 20 * x^3 + 30 * x^2 - 40 * x + 17
noncomputable def D (x : ℝ) := 2 * x - 4
theorem remainder_of_division :
  let x := 2 in
  ∃ r, r = P x ∧ r = 25 :=
by
  sorry

end remainder_of_division_l214_214677


namespace typing_speed_in_6_minutes_l214_214275

theorem typing_speed_in_6_minutes (total_chars : ℕ) (chars_first_minute : ℕ) (chars_last_minute : ℕ) (chars_other_minutes : ℕ) :
  total_chars = 2098 →
  chars_first_minute = 112 →
  chars_last_minute = 97 →
  chars_other_minutes = 1889 →
  (1889 / 6 : ℝ) < 315 → 
  ¬(∀ n, 1 ≤ n ∧ n ≤ 14 - 6 + 1 → chars_other_minutes / 6 ≥ 946) :=
by
  -- Given that analyzing the content, 
  -- proof is skipped here, replace this line with the actual proof.
  sorry

end typing_speed_in_6_minutes_l214_214275


namespace rectangle_perimeter_bounds_l214_214508

/-- Given 12 rectangular cardboard pieces, each measuring 4 cm in length and 3 cm in width,
  if these pieces are assembled to form a larger rectangle (possibly including squares),
  without overlapping or leaving gaps, then the minimum possible perimeter of the resulting 
  rectangle is 48 cm and the maximum possible perimeter is 102 cm. -/
theorem rectangle_perimeter_bounds (n : ℕ) (l w : ℝ) (total_area : ℝ) :
  n = 12 ∧ l = 4 ∧ w = 3 ∧ total_area = n * l * w →
  ∃ (min_perimeter max_perimeter : ℝ),
    min_perimeter = 48 ∧ max_perimeter = 102 :=
by
  intros
  sorry

end rectangle_perimeter_bounds_l214_214508


namespace log_base_3_l214_214009

theorem log_base_3 (a : ℝ) (h1 : a = 9) (h2 : ∀ (b : ℝ) (n : ℤ), log b (b ^ n) = n) : 
  log 3 (9 ^ 3) = 6 := 
by
  have h3 : 9 = 3^2, from by norm_num
  rw [h3] at h1
  rw [h1]
  have h4 : (3^2)^3 = 3^6, from pow_mul 3 2 3
  rw [h4]
  exact h2 3 6

end log_base_3_l214_214009


namespace solution_set_inequality_l214_214166

theorem solution_set_inequality (a b : ℝ) (h : |a - b| > 2) : 
  {x : ℝ | |x - a| + |x - b| > 2} = set.univ :=
by 
  sorry

end solution_set_inequality_l214_214166


namespace pencils_given_out_l214_214661

-- Defining the conditions
def num_children : ℕ := 4
def pencils_per_child : ℕ := 2

-- Formulating the problem statement, with the goal to prove the total number of pencils
theorem pencils_given_out : num_children * pencils_per_child = 8 := 
by 
  sorry

end pencils_given_out_l214_214661


namespace largest_n_unique_k_l214_214674

theorem largest_n_unique_k : 
  ∃ n : ℕ, (∀ k : ℤ, (5 / 12 : ℚ) < n / (n + k) ∧ n / (n + k) < (4 / 9 : ℚ) → k = 9) ∧ n = 7 :=
by
  sorry

end largest_n_unique_k_l214_214674


namespace multiple_choice_test_l214_214287

-- Define the sequence P.
def P : ℕ → ℤ 
| 0     := 1
| 1     := 3
| (n+2) := P n.succ * P n - P n

-- Define the problem conditions.
def question_1_condition (q2: char) (q3: char) : bool :=
  if q2 = 'D' then if q3 = 'C' then true else false else false

def question_2_condition : bool :=
  true

def question_3_condition : bool :=
  (P 2002 % 9 = 0)

-- Prove the correct answers for all questions.
theorem multiple_choice_test : 
  question_1_condition 'D' 'C' = true ∧ 
  (question_2_condition = true) ∧ 
  (question_3_condition = true) := 
  by 
  repeat {sorry}

-- The lean code defines the conditions and the theorem, which ensures correct answers 
-- according to the given problem structure. Proof is marked with 'sorry' 
-- as actual steps are not needed.

end multiple_choice_test_l214_214287


namespace determine_compound_A_l214_214679

noncomputable def compound_A_is_HIO3 : Prop :=
  ∃ (x : ℕ), x = 3 ∧ 
  let y : ℕ := 128 + 16 * x in 
  (0.015 * y = 0.88)

theorem determine_compound_A :
  compound_A_is_HIO3 :=
by 
  -- Definitions based only on the conditions
  let mass_of_iodine_precipitate := 0.48
  let moles_of_iodine := mass_of_iodine_precipitate / 32.0  -- molar mass of I2
  let mass_of_compound_A := 0.88 
  let x := (mass_of_compound_A - 0.015 * 128) / (0.015 * 16)
  existsi x
  split; assumption
  sorry

end determine_compound_A_l214_214679


namespace proof_inequality_1_proof_inequality_2_l214_214579

variables {a b c x x1 x2 xn : ℝ}

-- Conditions
def quadratic_function (f : ℝ → ℝ) := ∃ a b c, (a > 0) ∧ ∀ x, f x = a * x^2 + b * x + c

def has_two_roots_in_interval (f : ℝ → ℝ) (x1 x2 : ℝ) := 
  (0 < x1) ∧ (x1 < x2) ∧ (x2 < 1 / a) ∧ ∀ x, f(x) - x = 0 ↔ (x = x1) ∨ (x = x2)

def axis_of_symmetry (f : ℝ → ℝ) (xn : ℝ) := ∀ x, xn = (x1 + x2) / 2 - 1 / (2 * a)

-- Proof Problem 1
theorem proof_inequality_1 {f : ℝ → ℝ} :
  quadratic_function f →
  has_two_roots_in_interval f x1 x2 →
  ∀ x, (0 < x) ∧ (x < x1) → (x < f(x)) ∧ (f(x) < x1) :=
by
  sorry

-- Proof Problem 2
theorem proof_inequality_2 {f : ℝ → ℝ} :
  quadratic_function f →
  has_two_roots_in_interval f x1 x2 →
  axis_of_symmetry f xn →
  xn < x1 / 2 :=
by
  sorry

end proof_inequality_1_proof_inequality_2_l214_214579


namespace correct_options_l214_214241

-- Define the ellipse C and its properties
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

-- Define the foci F1 and F2 (not needed directly in the Lean statement, but define as context if necessary)
def F1 : ℝ × ℝ := (-sqrt 2, 0)
def F2 : ℝ × ℝ := (sqrt 2, 0)

-- Define the line l
def line_l (x y : ℝ) : Prop := (sqrt 2) * x + y = 4

-- Define the Monge circle
def Monge_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define distance from a point to a line
def distance_to_line (x y : ℝ) (a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / sqrt (a^2 + b^2)

-- Triangle area
def triangle_area (A B O : ℝ × ℝ) : ℝ :=
  1 / 2 * abs (A.fst * B.snd + B.fst * O.snd + O.fst * A.snd
               - (A.snd * B.fst + B.snd * O.fst + O.snd * A.fst))

theorem correct_options (A B : Prop) (d : ℝ) (AF2 : ℝ) :
  (∀ (x y : ℝ), Monge_circle x y ↔ ellipse x y) ∧
  (∃ (d AF2 : ℝ), (d - AF2 = 0) ∧
   ∃ (A B : ℝ × ℝ), triangle_area A B (0,0) = sqrt 3 / 2) :=
by
  sorry

end correct_options_l214_214241


namespace mn_eq_one_l214_214478

-- Definitions of the vectors and their properties
variable (a b : Vector3)
variable (m n : ℝ)
variable (A B C : Point3)
variable (h₁ : ¬ Collinear a b) -- Vectors a and b are not collinear
variable (h₂ : AB = a + m • b) -- AB vector definition
variable (h₃ : AC = n • a + b) -- AC vector definition
variable (h₄ : Collinear A B C) -- Points A, B, and C are collinear

theorem mn_eq_one (h₁ : ¬ Collinear a b) (h₂ : AB = a + m • b) (h₃ : AC = n • a + b) (h₄ : Collinear A B C) : m * n = 1 := 
sorry

end mn_eq_one_l214_214478


namespace rowing_trip_time_l214_214319

theorem rowing_trip_time
  (v_0 : ℝ) -- Rowing speed in still water
  (v_c : ℝ) -- Velocity of current
  (d : ℝ) -- Distance to the place
  (h_v0 : v_0 = 10) -- Given condition that rowing speed is 10 kmph
  (h_vc : v_c = 2) -- Given condition that current speed is 2 kmph
  (h_d : d = 144) -- Given condition that distance is 144 km :
  : (d / (v_0 - v_c) + d / (v_0 + v_c)) = 30 := -- Proving the total round trip time is 30 hours
by
  sorry

end rowing_trip_time_l214_214319


namespace digit_sum_pow_expr_l214_214772

theorem digit_sum_pow_expr (n m l : ℕ) : 
  (2^n * 5^m * 7) = l → 
  let num := 28 * 10^n in
  (num.digits.sum = 10) :=
by
  intro h
  rw [←h]
  sorry

end digit_sum_pow_expr_l214_214772


namespace part_one_part_two_l214_214088

variable (a b x k : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + 1

theorem part_one:
  f a b (-1) = 0 → f a b x = x^2 + 2 * x + 1 :=
by
  sorry

theorem part_two:
  (∀ x, -3 ≤ x ∧ x ≤ -1 → f 1 2 x > x + k) ↔ k < 1 :=
by
  sorry

end part_one_part_two_l214_214088


namespace solution_inequality_l214_214027

theorem solution_inequality (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ 2) :
  (frac(x + 2, x - 2) + frac(x + 4, 3 * x) ≥ 4) ↔ 
  (x ∈ Ioo((7 - Real.sqrt 33) / 4, (7 + Real.sqrt 33) / 4) ∨ x ∈ Ioi 2) :=
sorry

end solution_inequality_l214_214027


namespace find_four_digit_number_l214_214307

theorem find_four_digit_number : ∃ x : ℕ, (1000 ≤ x ∧ x ≤ 9999) ∧ (x % 7 = 0) ∧ (x % 29 = 0) ∧ (19 * x % 37 = 3) ∧ x = 5075 :=
by
  sorry

end find_four_digit_number_l214_214307


namespace multiples_count_1000_3000_l214_214097

def lcm (a b : ℕ) : ℕ := sorry

theorem multiples_count_1000_3000 (a b c : ℕ) (h1 : a = 18) (h2 : b = 24) (h3 : c = 36) : 
  ∃ k : ℕ, (1000 ≤ k ∧ k ≤ 3000 ∧ (k % lcm (lcm a b) c = 0)) ∧ (set.iota 1000 3000).filter (λ n, n % lcm (lcm a b) c = 0).card = 28 := 
begin
  sorry
end

end multiples_count_1000_3000_l214_214097


namespace algae_plants_in_milford_lake_l214_214586

theorem algae_plants_in_milford_lake (original : ℕ) (increase : ℕ) : (original = 809) → (increase = 2454) → (original + increase = 3263) :=
by
  sorry

end algae_plants_in_milford_lake_l214_214586


namespace inverse_B_squared_l214_214101

variable (B : Matrix (Fin 2) (Fin 2) ℝ)

def B_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![ -3, 2],
    ![  1, -1 ]]

theorem inverse_B_squared :
  B⁻¹ = B_inv →
  (B^2)⁻¹ = B_inv * B_inv :=
by sorry

end inverse_B_squared_l214_214101


namespace problem_M_solution_problem_a_solution_l214_214484

noncomputable def f (x : ℝ) : ℝ := abs (x + 2) - 2 * abs (x - 1)

theorem problem_M_solution :
  {x : ℝ | -2/3 ≤ x ∧ x ≤ 6} = {x : ℝ | f x ≥ -2} :=
sorry

theorem problem_a_solution :
  ∀ (x a : ℝ), 
    (x ∈ set.Ici a → f x ≤ x - a) ↔ (a ∈ set.Iic (-2) ∨ a ∈ set.Ici 4) :=
sorry

end problem_M_solution_problem_a_solution_l214_214484


namespace op_example_l214_214301

def op (a b : ℚ) : ℚ := a * b / (a + b)

theorem op_example : op (op 3 5) (op 5 4) = 60 / 59 := by
  sorry

end op_example_l214_214301


namespace midpoint_after_translation_l214_214638

structure Point where
  x : ℝ
  y : ℝ

def translate (p : Point) (dx dy : ℝ) : Point :=
  {x := p.x + dx, y := p.y - dy}

def midpoint (p1 p2 : Point) : Point :=
  { x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2 }

theorem midpoint_after_translation :
  let B := Point.mk 1 1
  let G := Point.mk 5 1
  let B' := translate B 3 4
  let G' := translate G 3 4
  midpoint B' G' = Point.mk 6 (-3) := 
  by
    sorry

end midpoint_after_translation_l214_214638


namespace derivative_x_exp_x_l214_214984

theorem derivative_x_exp_x : ∀ (x : ℝ), deriv (λ x : ℝ, x * exp x) x = (1 + x) * exp x :=
by
  sorry

end derivative_x_exp_x_l214_214984


namespace count_representation_of_2023_l214_214929

theorem count_representation_of_2023 :
  let M := {⟨b₃, b₂, b₁, b₀⟩ : ℕ × ℕ × ℕ × ℕ | 2023 = b₃ * 10^3 + b₂ * 10^2 + b₁ * 10 + b₀ ∧ 
                                            0 ≤ b₃ ∧ b₃ ≤ 999 ∧ 
                                            0 ≤ b₂ ∧ b₂ ≤ 999 ∧ 
                                            0 ≤ b₁ ∧ b₁ ≤ 999 ∧ 
                                            0 ≤ b₀ ∧ b₀ ≤ 999}. 
  M.card = 306 :=
by
  sorry

end count_representation_of_2023_l214_214929


namespace common_roots_of_cubic_polynomials_l214_214394

/-- The polynomials \( x^3 + 6x^2 + 11x + 6 \) and \( x^3 + 7x^2 + 14x + 8 \) have two distinct roots in common. -/
theorem common_roots_of_cubic_polynomials :
  ∃ r s : ℝ, r ≠ s ∧ (r^3 + 6 * r^2 + 11 * r + 6 = 0) ∧ (s^3 + 6 * s^2 + 11 * s + 6 = 0)
  ∧ (r^3 + 7 * r^2 + 14 * r + 8 = 0) ∧ (s^3 + 7 * s^2 + 14 * s + 8 = 0) :=
sorry

end common_roots_of_cubic_polynomials_l214_214394


namespace ratio_of_hexagon_areas_l214_214924

open Real

-- Define the given conditions about the hexagon and the midpoints
structure Hexagon :=
  (s : ℝ)
  (regular : True)
  (midpoints : True)

theorem ratio_of_hexagon_areas (h : Hexagon) : 
  let s := 2
  ∃ (area_ratio : ℝ), area_ratio = 4 / 7 :=
by
  sorry

end ratio_of_hexagon_areas_l214_214924


namespace number_of_polynomials_l214_214781

theorem number_of_polynomials :
  let satisfies_conditions :=
    λ (n : ℕ) (a : ℕ → ℤ),
      n + ((Finset.range (n + 1)).sum (λ i, |a i|)) = 5 ∧
      ∃ i, i ≤ n ∧ a i < 0
  in
  (Finset.sum (Finset.range 6) 
    (λ n, (Finset.piFinset' (Finset.range (n + 1)) (λ _, Finset.univ.filter (λ x, satisfies_conditions n (λ i, x i)))).card)) = 6 := 
by 
  sorry

end number_of_polynomials_l214_214781


namespace john_pays_12_dollars_l214_214918

/-- Define the conditions -/
def number_of_toys : ℕ := 5
def cost_per_toy : ℝ := 3
def discount_rate : ℝ := 0.2

/-- Define the total cost before discount -/
def total_cost_before_discount := number_of_toys * cost_per_toy

/-- Define the discount amount -/
def discount_amount := total_cost_before_discount * discount_rate

/-- Define the final amount John pays -/
def final_amount := total_cost_before_discount - discount_amount

/-- The theorem to be proven -/
theorem john_pays_12_dollars : final_amount = 12 := by
  -- Proof goes here
  sorry

end john_pays_12_dollars_l214_214918


namespace solutions_correct_l214_214359

noncomputable def solutions : Set ℝ :=
{ x | (x ≠ 3 ∧ x ≠ 5) ∧ 
      ( ( (x - 2)^2 * (x - 6) * (x - 5) = 1 ) ∧
        ( (x = (-3 + sqrt 5) / 2) ∨ (x = (-3 - sqrt 5) / 2) ))}
      
theorem solutions_correct : solutions = {x | x = (-3 + sqrt 5) / 2 ∨ x = (-3 - sqrt 5) / 2} :=
by
  sorry

end solutions_correct_l214_214359


namespace probability_XOXOXOX_is_1_div_35_l214_214423

def count_combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def num_ways_to_choose_positions_for_X (total_positions : ℕ) (num_X : ℕ) : ℕ := 
  count_combinations total_positions num_X

def num_ways_for_specific_arrangement_XOXOXOX : ℕ := 1

def probability_of_XOXOXOX (num_ways_total : ℕ) (num_ways_specific : ℕ) : ℚ := 
  num_ways_specific / num_ways_total

theorem probability_XOXOXOX_is_1_div_35 :
  probability_of_XOXOXOX (num_ways_to_choose_positions_for_X 7 4) num_ways_for_specific_arrangement_XOXOXOX = 1 / 35 := by
  sorry

end probability_XOXOXOX_is_1_div_35_l214_214423


namespace john_pays_after_discount_l214_214911

theorem john_pays_after_discount :
  ∀ (num_toys : ℕ) (cost_per_toy : ℕ) (discount_rate : ℚ),
  num_toys = 5 → cost_per_toy = 3 → discount_rate = 0.20 →
  let total_cost := num_toys * cost_per_toy in
  let discount := discount_rate * ↑total_cost in
  let amount_paid := total_cost - discount in
  amount_paid = 12 :=
by
  intros num_toys cost_per_toy discount_rate hnum_toys hcost_per_toy hdiscount_rate
  rw [hnum_toys, hcost_per_toy, hdiscount_rate]
  let total_cost := num_toys * cost_per_toy
  let discount := discount_rate * ↑total_cost
  let amount_paid := total_cost - discount
  have htotal_cost : total_cost = 15 := by norm_num
  have hdiscount : discount = 3 := by norm_num
  have hamount_paid : amount_paid = 12 := by norm_num
  exact hamount_paid

end john_pays_after_discount_l214_214911


namespace speed_of_slower_train_l214_214668

noncomputable theory

-- Definitions of given conditions
def speed_of_faster_train : ℝ := 100 -- km/h
def length_of_slower_train : ℝ := 500 / 1000 -- meters to kilometers
def length_of_faster_train : ℝ := 700 / 1000 -- meters to kilometers
def time_to_cross : ℝ := 19.6347928529354 / 3600 -- seconds to hours
def total_distance : ℝ := length_of_slower_train + length_of_faster_train

-- Proof statement to find the speed of the slower train
theorem speed_of_slower_train :
  let V_r := total_distance / time_to_cross in
  let speed_of_slower_train := V_r - speed_of_faster_train in
  abs (speed_of_slower_train - 120.127) < 0.001 :=
by
  sorry

end speed_of_slower_train_l214_214668


namespace algebraic_expression_value_l214_214451

theorem algebraic_expression_value (x y : ℝ) (h1 : x * y = -2) (h2 : x + y = 4) : x^2 * y + x * y^2 = -8 := 
by
  sorry

end algebraic_expression_value_l214_214451


namespace probability_XOXOXOX_l214_214429

noncomputable def binomial (n k : ℕ) : ℕ := nat.choose n k

theorem probability_XOXOXOX :
  let num_X := 4,
      num_O := 3,
      total_arrangements := binomial 7 num_X in
  total_arrangements = 35 ∧
  1 / total_arrangements = (1 : ℚ) / 35 :=
by
  sorry

end probability_XOXOXOX_l214_214429


namespace solve_g_g_3_l214_214086

def g (x : ℝ) : ℝ := (1 / x) - (1 / (x - 1))

theorem solve_g_g_3 :
  g(g(3)) = -(36 / 7) := by
  sorry

end solve_g_g_3_l214_214086


namespace matrix_B_power_103_l214_214160

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

theorem matrix_B_power_103 :
  B ^ 103 = B :=
by
  sorry

end matrix_B_power_103_l214_214160


namespace sum_of_max_values_l214_214578

/-- Let the function f(x) = exp x * (sin x - cos x) defined on the interval [0, 4 * π].
    We aim to prove that the sum of all the maximum values of f(x) over this interval
    is equal to exp π + exp (3 * π). -/
theorem sum_of_max_values (f : ℝ → ℝ) (h : ∀ x, f x = Real.exp x * (Real.sin x - Real.cos x)) :
  let S := (exp (π : ℝ) + exp (3 * π : ℝ))
  ∑ k : ℕ in finset.range 2, f ((2 * k + 1) * π) = S :=
by sorry

end sum_of_max_values_l214_214578


namespace unique_mag_b_if_theta_determined_l214_214933

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V) (θ : ℝ)
variables (h1 : a ≠ 0) (h2 : b ≠ 0)
variables (h_min : ∀ t : ℝ, ∥b + t • a∥ ≥ 1)

theorem unique_mag_b_if_theta_determined
  (h_theta : ∃θ : ℝ, ∀ t : ℝ, ∥b + t • a∥ = 1) :
  ∃ M : ℝ, M = ∥b∥ :=
sorry

end unique_mag_b_if_theta_determined_l214_214933


namespace largest_K_is_1_l214_214391

noncomputable def largest_K_vip (K : ℝ) : Prop :=
  ∀ (k : ℝ) (a b c : ℝ), 
  0 ≤ k ∧ k ≤ K → 
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c → 
  a^2 + b^2 + c^2 + k * a * b * c = k + 3 → 
  a + b + c ≤ 3

theorem largest_K_is_1 : largest_K_vip 1 :=
sorry

end largest_K_is_1_l214_214391


namespace john_total_payment_l214_214914

def cost_per_toy := 3
def number_of_toys := 5
def discount_rate := 0.2

theorem john_total_payment :
  (number_of_toys * cost_per_toy) - ((number_of_toys * cost_per_toy) * discount_rate) = 12 :=
by
  sorry

end john_total_payment_l214_214914


namespace replaced_person_weight_l214_214890

def weight_of_replaced_person 
  (n : ℕ) (avg_weight_increase : ℝ) (new_person_weight : ℝ) (w : ℝ) : Prop :=
  n = 8 ∧ avg_weight_increase = 2.5 ∧ new_person_weight = 55 ∧ 
  (8 * (avg_weight_increase) = new_person_weight - w)

theorem replaced_person_weight : ∃ (w : ℝ),
  weight_of_replaced_person 8 2.5 55 w ∧ w = 35 :=
by
  use 35
  rw weight_of_replaced_person
  split
  { -- Prove the conditions
    repeat {split};
    exact rfl}
  { -- Prove w = 35
    linarith }

end replaced_person_weight_l214_214890


namespace find_f_2022_l214_214988

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_2022 :
  (∀ a b : ℝ, f((a + 2 * b) / 3) = (f(a) + 2 * f(b)) / 3) →
  f(1) = 5 →
  f(4) = 2 →
  f(2022) = -2016 := by
  sorry

end find_f_2022_l214_214988


namespace g_cross_horizontal_asymptote_at_l214_214794

noncomputable def g (x : ℝ) : ℝ :=
  (3 * x^2 - 7 * x - 8) / (x^2 - 5 * x + 6)

theorem g_cross_horizontal_asymptote_at (x : ℝ) : g x = 3 ↔ x = 13 / 4 :=
by
  sorry

end g_cross_horizontal_asymptote_at_l214_214794


namespace point_p_coordinates_l214_214546

theorem point_p_coordinates :
  ∃ (P : ℝ × ℝ), (P.1 < 0 ∧ P.2 > 0) ∧ (P.2 = 4 ∧ |P.1| = 5) ∧ P = (-5, 4) :=
by
  let P := (-5, 4)
  use P
  split
  exact ⟨by norm_num, by norm_num⟩
  split
  exact ⟨by norm_num, by norm_num⟩
  refl

end point_p_coordinates_l214_214546


namespace solve_for_z_l214_214385

theorem solve_for_z (z : ℝ) (h : sqrt (5 - 4 * z + 1) = 7) : z = -43 / 4 := 
by
  sorry

end solve_for_z_l214_214385


namespace price_decrease_percentage_l214_214640

variables (P Q : ℝ)
variables (Q' R R' : ℝ)

-- Condition: the number sold increased by 60%
def quantity_increase_condition : Prop :=
  Q' = Q * (1 + 0.60)

-- Condition: the total revenue increased by 28.000000000000025%
def revenue_increase_condition : Prop :=
  R' = R * (1 + 0.28000000000000025)

-- Definition: the original revenue R
def original_revenue : Prop :=
  R = P * Q

-- The new price P' after decreasing by x%
variables (P' : ℝ) (x : ℝ)
def new_price_condition : Prop :=
  P' = P * (1 - x / 100)

-- The new revenue R'
def new_revenue : Prop :=
  R' = P' * Q'

-- The proof problem
theorem price_decrease_percentage (P Q Q' R R' P' x : ℝ)
  (h1 : quantity_increase_condition Q Q')
  (h2 : revenue_increase_condition R R')
  (h3 : original_revenue P Q R)
  (h4 : new_price_condition P P' x)
  (h5 : new_revenue P' Q' R') :
  x = 20 :=
sorry

end price_decrease_percentage_l214_214640


namespace find_g_add_h_l214_214641

-- Define the problem conditions
variables {d g h a b : ℝ}

-- The product of the polynomials condition
def poly_product_condition : Prop :=
  (8 * d ^ 2 - 4 * d + g) * (2 * d ^ 2 + h * d - 7) = 16 * d ^ 4 - 28 * d ^ 3 + a * h ^ 2 * d ^ 2 - b * d + 49

-- The goal statement
theorem find_g_add_h (h_condition : poly_product_condition) : g + h = -3 :=
  sorry

end find_g_add_h_l214_214641


namespace marge_funds_l214_214187

theorem marge_funds (initial_winnings : ℕ)
    (tax_fraction : ℕ)
    (loan_fraction : ℕ)
    (savings_amount : ℕ)
    (investment_fraction : ℕ)
    (tax_paid leftover_for_loans savings_after_loans final_leftover final_leftover_after_investment : ℕ) :
    initial_winnings = 12006 →
    tax_fraction = 2 →
    leftover_for_loans = initial_winnings / tax_fraction →
    loan_fraction = 3 →
    savings_after_loans = leftover_for_loans / loan_fraction →
    savings_amount = 1000 →
    final_leftover = leftover_for_loans - savings_after_loans - savings_amount →
    investment_fraction = 5 →
    final_leftover_after_investment = final_leftover - (savings_amount / investment_fraction) →
    final_leftover_after_investment = 2802 :=
by
  intros
  sorry

end marge_funds_l214_214187


namespace maisy_earns_more_l214_214950

theorem maisy_earns_more 
    (current_hours : ℕ) (current_wage : ℕ) 
    (new_hours : ℕ) (new_wage : ℕ) (bonus : ℕ)
    (h_current_job : current_hours = 8) 
    (h_current_wage : current_wage = 10)
    (h_new_job : new_hours = 4) 
    (h_new_wage : new_wage = 15)
    (h_bonus : bonus = 35) :
  (new_hours * new_wage + bonus) - (current_hours * current_wage) = 15 := 
by 
  sorry

end maisy_earns_more_l214_214950


namespace smallest_n_satisfying_property_l214_214822

open Nat

theorem smallest_n_satisfying_property : ∃ n : ℕ, n > 3 ∧ (∀ S_n : Finset ℕ, (∀ a b c (hₐ : a ∈ S_n) (hᵇ : b ∈ S_n) (h𝚌 : c ∈ S_n), ab = c) → n = 243) :=
sorry

end smallest_n_satisfying_property_l214_214822


namespace infinite_series_sum_l214_214351

theorem infinite_series_sum : 
  (∑' n : ℕ, (4 * n + 1 : ℝ) / ((4 * n - 1)^3 * (4 * n + 3)^3)) = 1 / 972 := 
by 
  sorry

end infinite_series_sum_l214_214351


namespace max_sum_of_ab_l214_214632

-- Given: LCM and GCD conditions
variables {a b : ℕ}
axiom lcm_eq_140 : Nat.lcm a b = 140
axiom gcd_eq_5 : Nat.gcd a b = 5

-- Question: Prove that the maximum of a + b is 145
theorem max_sum_of_ab : a + b ≤ 145 :=
begin
  sorry
end

end max_sum_of_ab_l214_214632


namespace range_of_derivative_l214_214577

theorem range_of_derivative (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ 5 * Real.pi / 12) :
  let f := λ x : ℝ, (sin θ / 3) * x^3 + (sqrt 3 * cos θ / 2) * x^2 + tan θ in
  let f' := λ x : ℝ, deriv f x in
  f' 1 ∈ Set.Icc (sqrt 2) 2 :=
by
  sorry

end range_of_derivative_l214_214577


namespace min_visible_sum_of_prism_faces_l214_214721

theorem min_visible_sum_of_prism_faces :
  let corners := 8
  let edges := 8
  let face_centers := 12
  let min_corner_sum := 6 -- Each corner dice can show 1, 2, and 3
  let min_edge_sum := 3    -- Each edge dice can show 1 and 2
  let min_face_center_sum := 1 -- Each face center dice can show 1
  let total_sum := corners * min_corner_sum + edges * min_edge_sum + face_centers * min_face_center_sum
  total_sum = 84 := 
by
  -- The proof is omitted
  sorry

end min_visible_sum_of_prism_faces_l214_214721


namespace compare_abc_l214_214070

noncomputable def a : ℝ := (1 / 4) * Real.logb 2 3
noncomputable def b : ℝ := 1 / 2
noncomputable def c : ℝ := (1 / 2) * Real.logb 5 3

theorem compare_abc : c < a ∧ a < b := sorry

end compare_abc_l214_214070


namespace average_last_9_numbers_l214_214618

theorem average_last_9_numbers (S1 S2 : ℕ) (avg_first_9 avg_all_17 : ℕ)
    (avg_first_9_eq : avg_first_9 = S1 / 9)
    (avg_all_17_eq : avg_all_17 = (S1 + S2 - 68) / 17)
    (S1_value : S1 = 56 * 9)
    (avg_all_17_value : avg_all_17 = 59)
    (N9_value : 68)
    (result : S2 / 9 = 63) : S2 / 9 = 63 :=
by 
  sorry

end average_last_9_numbers_l214_214618


namespace log_three_nine_cubed_l214_214007

theorem log_three_nine_cubed : log 3 (9 ^ 3) = 6 := by
  -- Importing, definitions, and theorem
  sorry

end log_three_nine_cubed_l214_214007


namespace fraction_not_sum_of_unit_fractions_l214_214608

open Nat

theorem fraction_not_sum_of_unit_fractions (p : ℕ) (hp : Prime p) (h : 5 ≤ p) :
    ¬ ∃ a b : ℕ, a > b ∧ (1 / a + 1 / b = (p - 1) / p) := 
by
  sorry

end fraction_not_sum_of_unit_fractions_l214_214608


namespace MargaretsMeanScore_l214_214804

theorem MargaretsMeanScore :
  ∀ (scores : List ℕ)
    (cyprian_mean : ℝ)
    (highest_lowest_different : Prop),
    scores = [82, 85, 88, 90, 92, 95, 97, 99] →
    cyprian_mean = 88.5 →
    highest_lowest_different →
    ∃ (margaret_mean : ℝ), margaret_mean = 93.5 := by
  sorry

end MargaretsMeanScore_l214_214804


namespace multiples_count_1000_3000_l214_214096

def lcm (a b : ℕ) : ℕ := sorry

theorem multiples_count_1000_3000 (a b c : ℕ) (h1 : a = 18) (h2 : b = 24) (h3 : c = 36) : 
  ∃ k : ℕ, (1000 ≤ k ∧ k ≤ 3000 ∧ (k % lcm (lcm a b) c = 0)) ∧ (set.iota 1000 3000).filter (λ n, n % lcm (lcm a b) c = 0).card = 28 := 
begin
  sorry
end

end multiples_count_1000_3000_l214_214096


namespace linda_total_miles_l214_214180

def calculate_total_miles (x : ℕ) : ℕ :=
  (60 / x) + (60 / (x + 4)) + (60 / (x + 8)) + (60 / (x + 12)) + (60 / (x + 16))

theorem linda_total_miles (x : ℕ) (hx1 : x > 0)
(hdx2 : 60 % x = 0)
(hdx3 : 60 % (x + 4) = 0) 
(hdx4 : 60 % (x + 8) = 0) 
(hdx5 : 60 % (x + 12) = 0) 
(hdx6 : 60 % (x + 16) = 0) :
  calculate_total_miles x = 33 := by
  sorry

end linda_total_miles_l214_214180


namespace exist_2005_palindrome_pairs_l214_214318

def is_palindrome (n : ℕ) : Prop :=
  n.toString = n.toString.reverse

theorem exist_2005_palindrome_pairs :
  ∃ (pairs : List (ℕ × ℕ)), pairs.length = 2005 ∧ ∀ (p : ℕ × ℕ) ∈ pairs, is_palindrome p.fst ∧ is_palindrome (p.fst + 110) :=
by
  sorry

end exist_2005_palindrome_pairs_l214_214318


namespace graph_is_empty_l214_214769

/-- The given equation 3x² + 4y² - 12x - 16y + 36 = 0 has no real solutions. -/
theorem graph_is_empty : ∀ (x y : ℝ), 3 * x^2 + 4 * y^2 - 12 * x - 16 * y + 36 ≠ 0 :=
by
  intro x y
  sorry

end graph_is_empty_l214_214769


namespace triangle_XYZ_perimeter_l214_214535

theorem triangle_XYZ_perimeter {X Y Z : Type} [MetricSpace X]
  (h1 : IsIsosceles X Y Z) (XZ : Real) (hXZ : XZ = 8)
  (YZ : Real) (hYZ : YZ = 11) :
  XZ + YZ + YZ = 30 := by
  sorry

end triangle_XYZ_perimeter_l214_214535


namespace probability_XOXOXOX_l214_214427

noncomputable def binomial (n k : ℕ) : ℕ := nat.choose n k

theorem probability_XOXOXOX :
  let num_X := 4,
      num_O := 3,
      total_arrangements := binomial 7 num_X in
  total_arrangements = 35 ∧
  1 / total_arrangements = (1 : ℚ) / 35 :=
by
  sorry

end probability_XOXOXOX_l214_214427


namespace find_a3_in_arith_geo_seq_l214_214819

theorem find_a3_in_arith_geo_seq
  (a : ℕ → ℚ)
  (S : ℕ → ℚ)
  (h1 : S 6 / S 3 = -19 / 8)
  (h2 : a 4 - a 2 = -15 / 8) :
  a 3 = 9 / 4 :=
sorry

end find_a3_in_arith_geo_seq_l214_214819


namespace coeff_x3_y3_expansion_l214_214298

theorem coeff_x3_y3_expansion 
  : ( ( (x + 2 * y) * (x + y) ^ 5 ).coeff ⟨ 3, 3 ⟩ ) = 30 :=
sorry

end coeff_x3_y3_expansion_l214_214298


namespace probability_sum_not_exceed_4_probability_n_lt_m_plus_2_l214_214124

-- Define the setup
def balls : List ℕ := [1, 2, 3, 4]

-- Problem 1
theorem probability_sum_not_exceed_4 : 
  (∃ (outcomes : List (ℕ × ℕ)), 
    outcomes = [(1, 2), (1, 3)] ∧ 
    (↑(outcomes.length) / ↑((balls.length * (balls.length - 1) / 2)) = (1 / 3))) :=
by
  sorry

-- Problem 2
theorem probability_n_lt_m_plus_2 :
  (∃ (total_outcomes : List (ℕ × ℕ)), 
    total_outcomes = [(x, y) | x <- balls, y <- balls ∧ y < x + 2] ∧
    (↑(total_outcomes.length) / ↑(balls.length * balls.length) = (13 / 16))) :=
by
  sorry

end probability_sum_not_exceed_4_probability_n_lt_m_plus_2_l214_214124


namespace polynomial_properties_l214_214604

noncomputable def polynomial : Polynomial ℚ :=
  -3/8 * (Polynomial.X ^ 5) + 5/4 * (Polynomial.X ^ 3) - 15/8 * (Polynomial.X)

theorem polynomial_properties (f : Polynomial ℚ) :
  (Polynomial.degree f = 5) ∧
  (∃ q : Polynomial ℚ, f + 1 = Polynomial.X - 1 ^ 3 * q) ∧
  (∃ p : Polynomial ℚ, f - 1 = Polynomial.X + 1 ^ 3 * p) ↔
  f = polynomial :=
by sorry

end polynomial_properties_l214_214604


namespace train_length_l214_214152

theorem train_length (L : ℝ) (h1 : (L + 120) / 60 = L / 20) : L = 60 := 
sorry

end train_length_l214_214152


namespace probability_Margo_paired_with_Irma_l214_214888

-- Definitions of the problem setup
def num_students := 32
def num_glasses_wearers := num_students / 2
def num_no_glasses := num_students - num_glasses_wearers
def margo_is_glasses_wearer : Prop := true
def irma_is_no_glasses : Prop := true
def special_group_size := 4
def special_group_requirement : Prop := special_group_size = 4 ∧ 
                                         ∀ (student : ℕ), student ∈ ({a | a ∈ Finset.range num_no_glasses}) → 
                                         {student ∉ ({b | b ∈ Finset.range num_no_glasses})}

-- Assuming conditions
axiom H1 : num_glasses_wearers = 16
axiom H2 : num_no_glasses = 16
axiom H3 : margo_is_glasses_wearer
axiom H4 : irma_is_no_glasses
axiom H5 : special_group_size = 4 → ∀ (i : ℕ), i ∈ Finset.range special_group_size → 
                        ∃ j, j ∈ Finset.range num_glasses_wearers ∧ irma_is_no_glasses

-- Probabilistic proof statement
theorem probability_Margo_paired_with_Irma : 
  num_glasses_wearers = 16 → num_no_glasses = 16 → margo_is_glasses_wearer → irma_is_no_glasses →
  (special_group_size = 4 → ∀ (i : ℕ), i ∈ Finset.range special_group_size → 
  ∃ j, j ∈ Finset.range num_glasses_wearers) → (1 / 16 : ℝ) = 1 / 16 := 
by
  intros _ _ _ _ _
  -- statement only, proof omitted
  sorry

end probability_Margo_paired_with_Irma_l214_214888


namespace find_a_l214_214466

open Set

theorem find_a (a : ℝ) (A B : Set ℝ) (hA : A = {0, 2}) (hB : B = {1, a^2}) (hU : A ∪ B = {0, 1, 2, 4}) : a = 2 ∨ a = -2 := by
  sorry

end find_a_l214_214466


namespace exist_pairwise_tangent_circles_l214_214848

noncomputable def circle (P : ℝ × ℝ) (r : ℝ) : Type := 
  { Q : ℝ × ℝ // dist P Q = r }

def tangents (C : circle) (P : ℝ × ℝ) : Prop :=
  dist P.1 P.2 = r

theorem exist_pairwise_tangent_circles 
  (A B C : ℝ × ℝ) :
  ∃ (P Q R : ℝ × ℝ),
    ∃ (c1 c2 c3 : circle),
      dist P B = dist P C ∧
      dist Q A = dist Q C ∧
      dist R A = dist R B ∧
      tangents c1 B ∧
      tangents c2 C ∧
      tangents c3 A ∧
      c1 ≠ c2 ∧ c2 ≠ c3 ∧c1 ≠ c3 :=
sorry

end exist_pairwise_tangent_circles_l214_214848


namespace primes_unique_l214_214387

-- Let's define that p, q, r are prime numbers, and define the main conditions.
def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem primes_unique (p q r : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q)
  (hr : is_prime r)
  (div1 : (p^4 - 1) % (q * r) = 0)
  (div2 : (q^4 - 1) % (p * r) = 0)
  (div3 : (r^4 - 1) % (p * q) = 0) :
  (p = 2 ∧ q = 3 ∧ r = 5) ∨ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 3 ∧ q = 2 ∧ r = 5) ∨ 
  (p = 3 ∧ q = 5 ∧ r = 2) ∨ (p = 5 ∧ q = 2 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) :=
by sorry

end primes_unique_l214_214387


namespace max_diff_of_distances_l214_214064

noncomputable def distance (p1 p2 : ℝ × ℝ) :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def C1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1
def C2 (x y : ℝ) : Prop := (x + 3)^2 + (y - 4)^2 = 9
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

theorem max_diff_of_distances :
  ∃ (P : ℝ × ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ),
    on_x_axis P ∧ C1 M.1 M.2 ∧ C2 N.1 N.2 ∧
    (abs (distance P N - distance P M) ≤ 4 + real.sqrt 26) ∧
    (abs (distance P N - distance P M) = 4 + real.sqrt 26) := 
sorry

end max_diff_of_distances_l214_214064


namespace total_cost_of_soup_l214_214665

theorem total_cost_of_soup 
  (pounds_beef : ℕ) (pounds_veg : ℕ) (cost_veg_per_pound : ℕ) (beef_price_multiplier : ℕ)
  (h1 : pounds_beef = 4)
  (h2 : pounds_veg = 6)
  (h3 : cost_veg_per_pound = 2)
  (h4 : beef_price_multiplier = 3):
  (pounds_veg * cost_veg_per_pound + pounds_beef * (cost_veg_per_pound * beef_price_multiplier)) = 36 :=
by
  sorry

end total_cost_of_soup_l214_214665


namespace find_star_l214_214625

-- Define the problem conditions and statement
theorem find_star (x : ℤ) (star : ℤ) (h1 : x = 5) (h2 : -3 * (star - 9) = 5 * x - 1) : star = 1 :=
by
  sorry -- Proof to be filled in

end find_star_l214_214625


namespace find_b2_l214_214256

noncomputable def sequence (n : ℕ) : ℕ := sorry

axiom b1_eq_15 : sequence 1 = 15
axiom b10_eq_105 : sequence 10 = 105
axiom recursive_arithmetic_mean : ∀ n, n ≥ 3 → sequence n = (sequence 1 + sequence 2 + ∑ i in (finset.range (n - 1)), sequence (i + 1)) / (n - 1)

theorem find_b2 : sequence 2 = 195 :=
sorry

end find_b2_l214_214256


namespace grid_arrangements_1296_l214_214406

theorem grid_arrangements_1296 : 
  let four_As := mk_set ("A", 4);
      four_Bs := mk_set ("B", 4);
      four_Cs := mk_set ("C", 4);
      four_Ds := mk_set ("D", 4);
      grid := mk_grid (4,4);
      conditions := (∀ row ∈ grid.rows, ∀ col ∈ grid.cols, count_in(row, "A") = 1 ∧ count_in(row, "B") = 1 ∧ count_in(row, "C") = 1 ∧ count_in(row, "D") = 1 ∧
                                             count_in(col, "A") = 1 ∧ count_in(col, "B") = 1 ∧ count_in(col, "C") = 1 ∧ count_in(col, "D") = 1)
  in conditions ∧ grid[0][3] = "A" ->
  grid.count_possible_arrangements = 1296 :=
begin
  sorry
end

end grid_arrangements_1296_l214_214406


namespace find_f_value_l214_214505

theorem find_f_value (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, x ≠ 0 → f (1 - 2 * x) = (1 - x^2) / x^2) : 
  f (1 / 2) = 15 :=
sorry

end find_f_value_l214_214505


namespace hyperbola_focus_l214_214112

-- Definition of the hyperbola equation and foci
def is_hyperbola (x y : ℝ) (k : ℝ) : Prop :=
  x^2 - k * y^2 = 1

-- Definition of the hyperbola having a focus at (3, 0) and the value of k
def has_focus_at (k : ℝ) : Prop :=
  ∃ x y : ℝ, is_hyperbola x y k ∧ (x, y) = (3, 0)

theorem hyperbola_focus (k : ℝ) (h : has_focus_at k) : k = 1 / 8 :=
  sorry

end hyperbola_focus_l214_214112


namespace complex_number_z_l214_214046

theorem complex_number_z (z : ℂ) (h : (conj z) / (1 + I) = 2 + I) : z = 1 - 3 * I :=
by sorry

end complex_number_z_l214_214046


namespace inequality_a_i_positive_l214_214051

open Real

theorem inequality_a_i_positive 
  (n : ℕ) 
  (a : Fin n → ℝ) 
  (h_pos : ∀ i, 0 < a i) 
  (k : ℝ) 
  (h_k : 1 ≤ k) 
  : (∑ i : Fin n, (a i / (∑ j, if j ≠ i then a j else 0)) ^ k) 
    ≥ n / (n - 1) ^ k := 
sorry

end inequality_a_i_positive_l214_214051


namespace upper_limit_of_interval_l214_214399

theorem upper_limit_of_interval :
  ∀ f : ℝ → ℝ, (∀ x : ℝ, f x = x - 5) → (∃ x : ℝ, f x = 8 ∧ x = 13) :=
by
  intros f hf
  use 13
  split
  sorry
  sorry

end upper_limit_of_interval_l214_214399


namespace function_increasing_interval_l214_214626

noncomputable def f (x : ℝ) : ℝ := x * sin x + cos x

theorem function_increasing_interval :
  ∀ x ∈ Set.Ioo (3 * π / 2) (5 * π / 2), (deriv f x) > 0 :=
by
  intro x hx
  have h_deriv : deriv f x = x * cos x := by
    calc
      deriv f x = (x * sin x + cos x)' : by sorry
              ... = (x * sin x)' + (cos x)' : by sorry
              ... = sin x + x * cos x + (-sin x) : by sorry
              ... = x * cos x : by sorry
  rw h_deriv
  sorry

end function_increasing_interval_l214_214626


namespace pd_squared_eq_pe_times_pf_l214_214925

theorem pd_squared_eq_pe_times_pf 
  (Γ : Type*) [metric_space Γ] [normed_add_group Γ] [inner_product_space ℝ Γ]
  (A B C P D E F : Γ)
  (hA : ∀ (x : Γ), x ∈ metric.sphere A (dist A B))
  (hB₁ : dist A B = dist A C)
  (hB₂ : ∀ (x : Γ), x ∈ metric.sphere A (dist A B) → dist x B = dist x C)
  (hC : ∀ (x : Γ), x ∈ metric.sphere P (dist P D))
  (hgamma : metric.sphere P (dist P D) = metric.sphere P (dist P E))
  (hD : ∀ (x : Γ), x ∈ metric.sphere D (dist D B))
  (hE : ∀ (x : Γ), x ∈ metric.sphere E (dist E C))
  (hF : ∀ (x : Γ), x ∈ metric.sphere F (dist F A))
  (h1 : ∀ (x : Γ), dist A x = dist A B)
  (h2 : ∀ (x : Γ), dist P x = dist P D)
  (h3 : ∀ (x : Γ), dist C x = dist C E)
  (h4 : ∀ (x : Γ), dist B x = dist B F)
  (h5 : ∀ (x : Γ), abs (dist P x - dist P F) = abs (dist P x - dist P E)) : 
  ∥P - D∥ ^ 2 = ∥P - E∥ * ∥P - F∥ :=
by {
  sorry
}

end pd_squared_eq_pe_times_pf_l214_214925


namespace volume_of_truncated_reflected_cone_l214_214998

noncomputable def volume_of_resulting_solid (R : ℝ) (h : ℝ) : ℝ :=
  2 * π * R^2 * h^2 * (1 - h)

theorem volume_of_truncated_reflected_cone (R : ℝ) (h : ℝ) (h_pos : 0 ≤ h) (h_le : h ≤ 1) :
  let V := volume_of_resulting_solid R h in
  V = 2 * π * R^2 * h^2 * (1 - h) :=
sorry

end volume_of_truncated_reflected_cone_l214_214998


namespace general_term_formula_l214_214490

variable {a_n : ℕ → ℕ} -- Sequence {a_n}
variable {S_n : ℕ → ℕ} -- Sum of the first n terms

-- Condition given in the problem
def S_n_condition (n : ℕ) : ℕ :=
  2 * n^2 + n

theorem general_term_formula (n : ℕ) (h₀ : ∀ (n : ℕ), S_n n = 2 * n^2 + n) :
  a_n n = 4 * n - 1 :=
sorry

end general_term_formula_l214_214490


namespace num_convex_pentagons_l214_214019

theorem num_convex_pentagons (n m : ℕ) (hn : n = 15) (hm : m = 5) : 
  Nat.choose n m = 3003 := by
  sorry

end num_convex_pentagons_l214_214019


namespace corrected_mean_is_correct_l214_214292

-- Define the initial conditions
def initial_mean : ℝ := 36
def n_obs : ℝ := 50
def incorrect_obs : ℝ := 23
def correct_obs : ℝ := 45

-- Calculate the incorrect total sum
def incorrect_total_sum : ℝ := initial_mean * n_obs

-- Define the corrected total sum
def corrected_total_sum : ℝ := incorrect_total_sum - incorrect_obs + correct_obs

-- State the main theorem to be proved
theorem corrected_mean_is_correct : corrected_total_sum / n_obs = 36.44 := by
  sorry

end corrected_mean_is_correct_l214_214292


namespace solution_set_f_inequality_l214_214837

theorem solution_set_f_inequality {f : ℝ → ℝ} (h1 : f 1 = 1) (h2 : ∀ x, f' x < 1) :
  {x : ℝ | f (2 ^ x) > 2 ^ x} = set.Iio 0 :=
by
  sorry

end solution_set_f_inequality_l214_214837


namespace polar_to_cartesian_eq_l214_214075

theorem polar_to_cartesian_eq (x y : ℝ) : (x^2 + y^2 = 1) ↔ (∀ (θ : ℝ), ∃ (ρ : ℝ), ρ = 1 ∧ (x, y) = (ρ * cos θ, ρ * sin θ)) :=
by
  sorry

end polar_to_cartesian_eq_l214_214075


namespace inequality_proof_l214_214697

theorem inequality_proof (a : ℝ) (h : a > 0) : sqrt (a + 1/a) - sqrt 2 ≥ sqrt a + 1/sqrt a - 2 :=
sorry

end inequality_proof_l214_214697


namespace compare_neg_two_cubed_l214_214749

-- Define the expressions
def neg_two_cubed : ℤ := (-2) ^ 3
def neg_two_cubed_alt : ℤ := -(2 ^ 3)

-- Statement of the problem
theorem compare_neg_two_cubed : neg_two_cubed = neg_two_cubed_alt :=
by
  sorry

end compare_neg_two_cubed_l214_214749


namespace cyclic_quadrilateral_diagonals_l214_214129

theorem cyclic_quadrilateral_diagonals
  (ABCD_cyclic : Cyclic ABCD)
  (CD_perp_AB : Perpendicular CD AB)
  (CD_eq_a : CD = a)
  (angle_CDB_eq_alpha : ∠CDB = α)
  (angle_BDA_eq_beta : ∠BDA = β) :
  (diagonal_AC = a * sin (α + β) / cos (2 * α + β)) ∧ (diagonal_BD = a * cos (α + β) / cos (2 * α + β)) :=
sorry

end cyclic_quadrilateral_diagonals_l214_214129


namespace range_of_m_l214_214082

open Real

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x - 1

theorem range_of_m :
  {m : ℝ | ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧
   f(x1) = m ∧ f(x2) = m ∧ f(x3) = m} = Ioo (-3) 1 :=
by
  sorry

end range_of_m_l214_214082


namespace janet_dresses_total_pockets_l214_214154

theorem janet_dresses_total_pockets :
  ∃ dresses pockets pocket_2 pocket_3,
  dresses = 24 ∧ 
  pockets = dresses / 2 ∧ 
  pocket_2 = pockets / 3 ∧ 
  pocket_3 = pockets - pocket_2 ∧ 
  (pocket_2 * 2 + pocket_3 * 3) = 32 := by
    sorry

end janet_dresses_total_pockets_l214_214154


namespace alpha_plus_beta_eq_l214_214445

variable {α β : ℝ}
variable (h1 : 0 < α ∧ α < π)
variable (h2 : 0 < β ∧ β < π)
variable (h3 : Real.sin (α - β) = 5 / 6)
variable (h4 : Real.tan α / Real.tan β = -1 / 4)

theorem alpha_plus_beta_eq : α + β = 7 * Real.pi / 6 := by
  sorry

end alpha_plus_beta_eq_l214_214445


namespace multiplication_factor_l214_214619

theorem multiplication_factor (n : ℕ) (a b x : ℕ) 
  (h1 : n = 7)
  (h2 : a = 25)
  (h3 : b = 125)
  (h4 : ∑ i in finset.range n, (λ i, 25) i = n * a)
  (h5 : ∑ i in finset.range n, (λ i, 125) i = n * b) :
  x = 5 :=
by
  sorry

end multiplication_factor_l214_214619


namespace sqrt_mul_simplify_l214_214221

theorem sqrt_mul_simplify : Real.sqrt 18 * Real.sqrt 32 = 24 := 
sorry

end sqrt_mul_simplify_l214_214221


namespace handshaking_remainder_mod_1000_l214_214891

theorem handshaking_remainder_mod_1000:
  let M := -- Define the formula for M based on the combinatorial ways of handshaking arrangements
  M % 1000 = 150 := 
begin
  sorry
end

end handshaking_remainder_mod_1000_l214_214891


namespace find_original_cost_of_ring_l214_214336

noncomputable def original_cost_of_ring : ℝ :=
let R := sorry in 
let B := 2 * R in 
let necklace_cost := 20 in
let earrings_cost := 15 in
4 * necklace_cost + 8 * R + 5 * earrings_cost + 6 * B = 320 ∧
320 - 30 + 0.1 * (320 - 30) = 320 in 
R

theorem find_original_cost_of_ring : original_cost_of_ring = 8.25 :=
sorry

end find_original_cost_of_ring_l214_214336


namespace original_volume_of_ice_l214_214728

theorem original_volume_of_ice (V : ℝ) 
  (h1 : V * (1/4) * (1/4) = 0.4) : 
  V = 6.4 :=
sorry

end original_volume_of_ice_l214_214728


namespace investment_difference_l214_214156

noncomputable def john_investment (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

noncomputable def alice_investment (P : ℝ) (r : ℝ) (n_m : ℕ) : ℝ :=
  P * (1 + r / 12)^(12 * n_m)

theorem investment_difference :
  let P := 70000 in
  let r := 0.03 in
  let n := 5 in
  let n_m := 5 in
  let john := john_investment P r n in
  let alice := alice_investment P r n_m in
  alice - john = 168.21 :=
by
  sorry

end investment_difference_l214_214156


namespace repeated_digit_sum_modulo_9_l214_214744

def repeated_digit_sum (d : ℕ) : ℕ :=
match d with
| 2 => 2
| 3 => 33
| 4 => 444
| 5 => 5555
| 6 => 66666
| 7 => 777777
| 8 => 8888888
| 9 => 99999999
| _ => 0
end

theorem repeated_digit_sum_modulo_9 :
  ((repeated_digit_sum 2) + 
   (repeated_digit_sum 3) + 
   (repeated_digit_sum 4) + 
   (repeated_digit_sum 5) + 
   (repeated_digit_sum 6) + 
   (repeated_digit_sum 7) + 
   (repeated_digit_sum 8) + 
   (repeated_digit_sum 9)) % 9 = 6 :=
by sorry

end repeated_digit_sum_modulo_9_l214_214744


namespace calculate_manuscript_cost_l214_214352

/-- Definition of the service rates and manuscript details -/
structure ManuscriptCost where
  pages : Nat
  firstRevPages : Nat
  secondRevPages : Nat
  thirdRevPages : Nat
  noRevPages : Nat
  images : Nat
  tables : Nat
  graphs : Nat
  specialElements : Nat
  longPages : Nat

def serviceRates (m : ManuscriptCost) : Nat :=
  (m.pages * 12) +
  (m.firstRevPages * 6) +
  (m.secondRevPages * 8) +
  (m.thirdRevPages * 8) +
  ((m.images + m.tables + m.graphs) * 1) +
  (m.specialElements * 0.5) +
  (m.longPages * 2)

theorem calculate_manuscript_cost : serviceRates { 
  pages := 110, 
  firstRevPages := 30, 
  secondRevPages := 20, 
  thirdRevPages := 10, 
  noRevPages := 50, 
  images := 100, 
  tables := 25, 
  graphs := 10, 
  specialElements := 50, 
  longPages := 60 
} = 2020 := by
  sorry

end calculate_manuscript_cost_l214_214352


namespace probability_XOXOXOX_l214_214418

theorem probability_XOXOXOX (X O : ℕ) (h1 : X = 4) (h2 : O = 3) :
  let total_ways := Nat.choose (X + O) X,
      favorable_outcomes := 1 in
  (favorable_outcomes / total_ways : ℚ) = 1 / 35 := by
  sorry

end probability_XOXOXOX_l214_214418


namespace dihedral_angle_cosine_l214_214272

noncomputable def cosine_dihedral_angle : ℝ :=
let k := 2 in
let beta := 45 in
1 - 2 * ((k - 1) / (k + 1) * (1 / Math.sin (Real.pi / 4)))^2

theorem dihedral_angle_cosine : cosine_dihedral_angle = 5 / 9 :=
by sorry

end dihedral_angle_cosine_l214_214272


namespace compare_neg_two_cubed_l214_214748

-- Define the expressions
def neg_two_cubed : ℤ := (-2) ^ 3
def neg_two_cubed_alt : ℤ := -(2 ^ 3)

-- Statement of the problem
theorem compare_neg_two_cubed : neg_two_cubed = neg_two_cubed_alt :=
by
  sorry

end compare_neg_two_cubed_l214_214748


namespace probability_XOXOXOX_l214_214434

theorem probability_XOXOXOX :
  let X := 4;
      O := 3;
      total_positions := 7;
      specific_arrangement := 1;
      total_arrangements := Nat.choose total_positions X in
  1 / total_arrangements = 1 / 35 := by
  sorry

end probability_XOXOXOX_l214_214434


namespace original_combined_price_l214_214402

theorem original_combined_price (C S : ℝ) 
  (candy_box_increased : C * 1.25 = 18.75)
  (soda_can_increased : S * 1.50 = 9) : 
  C + S = 21 :=
by
  sorry

end original_combined_price_l214_214402


namespace marge_funds_for_fun_l214_214184

-- Definitions based on given conditions
def lottery_amount : ℕ := 12006
def taxes_paid : ℕ := lottery_amount / 2
def remaining_after_taxes : ℕ := lottery_amount - taxes_paid
def student_loans_paid : ℕ := remaining_after_taxes / 3
def remaining_after_loans : ℕ := remaining_after_taxes - student_loans_paid
def savings : ℕ := 1000
def remaining_after_savings : ℕ := remaining_after_loans - savings
def stock_market_investment : ℕ := savings / 5
def remaining_after_investment : ℕ := remaining_after_savings - stock_market_investment

-- The proof goal
theorem marge_funds_for_fun : remaining_after_investment = 2802 :=
sorry

end marge_funds_for_fun_l214_214184


namespace new_sequence_69th_term_l214_214144

-- Definitions and conditions
def original_sequence (a : ℕ → ℕ) (n : ℕ) : ℕ := a n

def new_sequence (a : ℕ → ℕ) (k : ℕ) : ℕ :=
if k % 4 = 1 then a (k / 4 + 1) else 0  -- simplified modeling, the inserted numbers are denoted arbitrarily as 0

-- The statement to be proven
theorem new_sequence_69th_term (a : ℕ → ℕ) : new_sequence a 69 = a 18 :=
by
  sorry

end new_sequence_69th_term_l214_214144


namespace number_of_paths_l214_214547

noncomputable def paths_in_grid (n m : ℕ) : ℕ :=
  let P : ℕ × ℕ → ℕ := λ ⟨i, j⟩, 
    if i = 0 ∧ j = 0 then 1
    else if i = 0 then P (i, j - 1)
    else if j = 0 then P (i - 1, j)
    else P (i, j - 1) + P (i - 1, j) + P (i - 1, j - 1)
  in P (n, m)

theorem number_of_paths (n m : ℕ) (h : n = 365 ∧ m = 365) : paths_in_grid n m = 372 := 
by sorry

end number_of_paths_l214_214547


namespace function_even_and_period_l214_214244

noncomputable def f (x : ℝ) : ℝ := abs (sin x + cos x) + abs (sin x - cos x)

theorem function_even_and_period : 
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, f (x + π / 2) = f x) :=
by 
  split;
  sorry

end function_even_and_period_l214_214244


namespace fill_cistern_time_l214_214707

-- Definitions based on conditions
def rate_A : ℚ := 1 / 8
def rate_B : ℚ := 1 / 16
def rate_C : ℚ := -1 / 12

-- Combined rate
def combined_rate : ℚ := rate_A + rate_B + rate_C

-- Time to fill the cistern
def time_to_fill := 1 / combined_rate

-- Lean statement of the proof
theorem fill_cistern_time : time_to_fill = 9.6 := by
  sorry

end fill_cistern_time_l214_214707


namespace find_a_plus_d_l214_214506

noncomputable def f (a b c d x : ℚ) : ℚ := (a * x + b) / (c * x + d)

theorem find_a_plus_d (a b c d : ℚ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : d ≠ 0)
  (h₄ : ∀ x : ℚ, f a b c d (f a b c d x) = x) :
  a + d = 0 := by
  sorry

end find_a_plus_d_l214_214506


namespace distance_M1_M12_l214_214084

noncomputable def M1 := (Real.pi / 12, 1 / 2)
noncomputable def M12 := (5 * Real.pi + 5 * Real.pi / 12, 1 / 2)

theorem distance_M1_M12 : Real.dist (prod.fst M1) (prod.fst M12) = 16 * Real.pi / 3 := by
  sorry

end distance_M1_M12_l214_214084


namespace z_in_first_quadrant_l214_214694

def i : ℂ := complex.I

def z : ℂ := i * (1 - i)

theorem z_in_first_quadrant : ∃ (z : ℂ), z = i * (1 - i) ∧ (real_part z > 0 ∧ imaginary_part z > 0) := by
  sorry

end z_in_first_quadrant_l214_214694


namespace find_point_P_coordinates_l214_214470

noncomputable def point_coordinates (a : ℝ) : ℝ × ℝ :=
  if h : a = 0 then (0, 0) else
    let P := ((72 / 25 : ℝ), (96 / 25 : ℝ)) in
    P

theorem find_point_P_coordinates (a : ℝ) : 
  ∀ x y : ℝ, 
    (x + a * y - 3 * a = 0) 
    → (a * x - y - 4 * a = 0) 
    → (x > 0 ∧ y > 0) 
    → (∀ A B : ℝ × ℝ, 
      A = (4, 0) → 
      B = (0, 3) → 
      |B.snd - y| - |A.fst - x| = |A.fst - 0| - |B.snd - 0| = 1) 
    → (x, y) = point_coordinates a := 
by
  sorry

end find_point_P_coordinates_l214_214470


namespace identify_irrational_number_l214_214730

theorem identify_irrational_number :
  ∀ (x : ℝ), (x = 5 + Real.pi) ∨ (x = 3.14) ∨ (x = 22 / 7) ∨ (x = 0.3030030003) →
  (x = 5 + Real.pi) ↔ irrational x :=
begin
  sorry
end

end identify_irrational_number_l214_214730


namespace sum_min_max_x_y_l214_214960

theorem sum_min_max_x_y (x y : ℕ) (h : 6 * x + 7 * y = 2012): 288 + 335 = 623 :=
by
  sorry

end sum_min_max_x_y_l214_214960


namespace Riccati_solution_l214_214716

theorem Riccati_solution (C : ℝ) : 
  ∀ (x : ℝ), 
  let y := λ x, e^x + 1 / (C - x) in 
  (deriv y x) + 2 * e^x * y x - (y x)^2 = e^x + e^(2 * x) := 
by 
  sorry 

end Riccati_solution_l214_214716


namespace no_such_function_exists_l214_214572

open Set

theorem no_such_function_exists
  (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, 0 < x → 0 < y → y > x → f y > (y - x) * f x ^ 2) :
  False :=
sorry

end no_such_function_exists_l214_214572


namespace find_d_in_triangle_ABC_l214_214523

theorem find_d_in_triangle_ABC (AB BC AC : ℝ) (P : Type) (d : ℝ) 
  (h_AB : AB = 480) (h_BC : BC = 500) (h_AC : AC = 550)
  (h_segments_equal : ∀ (D D' E E' F F' : Type), true) : 
  d = 132000 / 654 :=
sorry

end find_d_in_triangle_ABC_l214_214523


namespace centroids_collinear_and_bisect_l214_214732

noncomputable def centroid (A B C : Point) : Point := 
    (1 / 3) • (A + B + C : Vect3)

-- Definition of isogonal conjugates
def isogonal_conjugate (P : Point) (Q : Point) (A B C : Point) : Prop :=
    sorry  -- Define the formal properties of isogonal conjugates

-- Given conditions in Lean
variables (A B C D E F G H I J : Point)
(h_isogonal1 : isogonal_conjugate F I A B C)
(h_isogonal2 : isogonal_conjugate D J A B C)
(h_isogonal3 : isogonal_conjugate E H A B C)

-- Centroids definitions
def G_center : Point := centroid A B C
def G1_center : Point := centroid D E F
def G2_center : Point := centroid J H I

-- Statement of the problem in Lean
theorem centroids_collinear_and_bisect :
  collinear (G_center A B C) (G1_center D E F) (G2_center J H I) ∧
  midpoint G_center (G1_center D E F) (G2_center J H I) :=
  sorry

end centroids_collinear_and_bisect_l214_214732


namespace arithmetic_mean_of_fractions_l214_214768
-- Import the Mathlib library to use fractional arithmetic

-- Define the problem in Lean
theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 :=
by
  let a : ℚ := 3 / 8
  let b : ℚ := 5 / 9
  have := (a + b) / 2 = 67 / 144
  sorry

end arithmetic_mean_of_fractions_l214_214768


namespace tshirts_per_package_l214_214286

noncomputable def T : ℝ := 71.0
noncomputable def P : ℝ := 11.83333333

theorem tshirts_per_package : T / P ≈ 6 :=
by
  sorry

end tshirts_per_package_l214_214286


namespace product_real_parts_quadratic_complex_l214_214942

theorem product_real_parts_quadratic_complex :
  let i := Complex.I in
  let z := Complex in
  ∀ (a b c : z),
  (a = 1) ∧ (b = 2) ∧ (c = 3 - 7 * i) → 
  let Δ := b * b - 4 * a * c in
  let root1 := (-b + Complex.sqrt Δ) / (2 * a) in
  let root2 := (-b - Complex.sqrt Δ) / (2 * a) in
  let re_root1 := root1.re in
  let re_root2 := root2.re in
  re_root1 * re_root2 = -45 / 4 :=
by
  let i := Complex.I
  let a := 1 : ℂ
  let b := 2 : ℂ
  let c := (3 : ℂ) - 7 * i
  let Δ := b * b - 4 * a * c
  let root1 := (-b + Complex.sqrt Δ) / (2 * a)
  let root2 := (-b - Complex.sqrt Δ) / (2 * a)
  let re_root1 := root1.re
  let re_root2 := root2.re
  have re_root1_Product :=
    calc 
      re_root1 * re_root2
      -- Perform verification/calculation steps to reach the conclusion
    sorry

end product_real_parts_quadratic_complex_l214_214942


namespace probability_XOXOXOX_l214_214435

theorem probability_XOXOXOX :
  let X := 4;
      O := 3;
      total_positions := 7;
      specific_arrangement := 1;
      total_arrangements := Nat.choose total_positions X in
  1 / total_arrangements = 1 / 35 := by
  sorry

end probability_XOXOXOX_l214_214435


namespace projection_a_in_direction_of_b_l214_214827

-- Definitions of the given vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (3, 4)

-- Definition of dot product of 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Definition of magnitude of a 2D vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Definition of vector projection
def projection (v w : ℝ × ℝ) : ℝ :=
  (dot_product v w) / (magnitude w)

-- The theorem to prove the projection of a in the direction of b is 2
theorem projection_a_in_direction_of_b : projection a b = 2 :=
by
  sorry

end projection_a_in_direction_of_b_l214_214827


namespace incorrect_conclusion_in_triangle_l214_214120

theorem incorrect_conclusion_in_triangle
  (A B C : ℝ)
  (h_triangle : A + B + C = π) -- Sum of angles in a triangle
  (h_A_gt_B : A > B)
  (h_A_pos : 0 < A)
  (h_B_pos : 0 < B)
  (h_C_pos : 0 < C) :
  ¬ (A ≤ π / 2 ∧ B ≤ π / 2 ∧ sin 2A > sin 2B) := 
by
  sorry

end incorrect_conclusion_in_triangle_l214_214120


namespace sixth_number_tenth_row_l214_214656

theorem sixth_number_tenth_row : 
  let first_term := (1 : ℝ) / 4
  let common_difference := (1 : ℝ) / 4
  let common_ratio := (1 : ℝ) / 2
  let tenth_row_first_number := first_term + 9 * common_difference
  let sixth_number_tenth_row := tenth_row_first_number * common_ratio^5
  sixth_number_tenth_row = (5 : ℝ) / 64 := by
  let first_term := (1 / 4 : ℝ)
  let common_difference := (1 / 4 : ℝ)
  let common_ratio := (1 / 2 : ℝ)
  let tenth_row_first_number := first_term + 9 * common_difference
  let sixth_number_tenth_row := tenth_row_first_number * common_ratio^5
  show sixth_number_tenth_row = (5 / 64 : ℝ) from sorry

end sixth_number_tenth_row_l214_214656


namespace abs_value_expression_l214_214507

theorem abs_value_expression (x : ℝ) (h : x > 2) : |x - real.sqrt ((x - 3)^2)| = 3 := by
  sorry

end abs_value_expression_l214_214507


namespace find_ratio_l214_214231

variable {x y k x1 x2 y1 y2 : ℝ}

-- Inverse proportionality
def inverse_proportional (x y k : ℝ) : Prop := x * y = k

-- Given conditions
axiom h1 : inverse_proportional x1 y1 k
axiom h2 : inverse_proportional x2 y2 k
axiom h3 : x1 ≠ 0
axiom h4 : x2 ≠ 0
axiom h5 : y1 ≠ 0
axiom h6 : y2 ≠ 0
axiom h7 : x1 / x2 = 3 / 4

theorem find_ratio : y1 / y2 = 4 / 3 :=
by
  sorry

end find_ratio_l214_214231


namespace range_of_x_in_right_triangle_l214_214538

variable (a b c x : ℝ)
variable (h1 : ∠C = 90°)
variable (ha : a + b = c * x)
variable (pythagorean : a^2 + b^2 = c^2)

theorem range_of_x_in_right_triangle (h1 : ∠C = 90°)
  (ha : a + b = c * x)
  (pythagorean : a^2 + b^2 = c^2) :
  1 < x ∧ x ≤ Real.sqrt 2 :=
sorry

end range_of_x_in_right_triangle_l214_214538


namespace probability_XOXOXOX_l214_214414

theorem probability_XOXOXOX (X O : ℕ) (h1 : X = 4) (h2 : O = 3) :
  let total_ways := Nat.choose (X + O) X,
      favorable_outcomes := 1 in
  (favorable_outcomes / total_ways : ℚ) = 1 / 35 := by
  sorry

end probability_XOXOXOX_l214_214414


namespace parallel_lines_distance_l214_214849

theorem parallel_lines_distance (m n : ℝ) (l1 l2 : ℝ → ℝ → Prop) :
  (∀ x y, l1 x y = mx + 8 * y + n = 0 ∧ l2 x y = 2 * x + m * y - 1 = 0) → 
  (∀ x y, ¬ l1 x y = l2 x y) →
  (dist_lines l1 l2 = sqrt 5) →
  (l1 = (λ x y, 2 * x + 4 * y - 11 = 0) ∨ 
   l1 = (λ x y, 2 * x + 4 * y + 9 = 0) ∨ 
   l1 = (λ x y, 2 * x - 4 * y + 9 = 0) ∨ 
   l1 = (λ x y, 2 * x - 4 * y - 11 = 0)) :=
sorry

end parallel_lines_distance_l214_214849


namespace perpendicular_lines_l214_214517

theorem perpendicular_lines (a : ℝ) :
  (if a ≠ 0 then a^2 ≠ 0 else true) ∧ (a^2 * a + (-1/a) * 2 = -1) → (a = 2 ∨ a = 0) :=
by
  sorry

end perpendicular_lines_l214_214517


namespace correct_propositions_l214_214249

-- Define the conditions for each proposition
structure Line (α : Type) := (parallel : α → Prop) (subset : α → Prop)
structure Plane (α : Type)

constants m n : Line ℝ
constants α β : Plane ℝ

def p1 := 
  m ≠ n ∧ α ≠ β ∧ m.parallel α ∧ n.parallel α ∧ m.subset β ∧ n.subset β → α ∥ β

def p2 := ¬(∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 ≤ 0) = ∀ x : ℝ, x^3 - x^2 + 1 > 0

def p3 := ∃ ω : ℝ, (∀ t : ℤ, ω = t * real.pi + real.pi / 6) ∧ 
            (∃ x : ℝ, y = real.sin (ω * x + real.pi / 6) → y ≤ 1)

def p4 := 
  let µ := 6
  let σ := 2 in
  let X := measure_theory.probability_theory.normal ℝ µ σ in
  measure_theory.measure.in_measure X {x | 2 < x ∧ x ≤ 8} = 0.8185

-- Proving the number of correct propositions is 2
theorem correct_propositions : 
  ((p3 = true) ∧ (p4 = true) ∧ (¬p1) ∧ (¬p2)) → 2 := 
sorry

end correct_propositions_l214_214249


namespace magnitude_sum_sq_l214_214941

open Real

noncomputable section

def a : ℝ × ℝ := (x, 3 * x)
def m : ℝ × ℝ := (4, 6)
def b : ℝ × ℝ := (8 - x, 12 - 3 * x)

theorem magnitude_sum_sq (x : ℝ) 
  (h_dot : a.1 * b.1 + a.2 * b.2 = 12)
  (h_mid : (a.1 + b.1) / 2 = m.1 ∧ (a.2 + b.2) / 2 = m.2) :
  (a.1 ^ 2 + a.2 ^ 2) + (b.1 ^ 2 + b.2 ^ 2) = 184 := 
sorry

end magnitude_sum_sq_l214_214941


namespace total_amount_spent_l214_214870
-- Since we need broader imports, we include the whole Mathlib library

-- Definition of the prices of each CD and the quantity purchased
def price_the_life_journey : ℕ := 100
def price_a_day_a_life : ℕ := 50
def price_when_you_rescind : ℕ := 85
def quantity_purchased : ℕ := 3

-- Tactic to calculate the total amount spent
theorem total_amount_spent : (price_the_life_journey * quantity_purchased) + 
                             (price_a_day_a_life * quantity_purchased) + 
                             (price_when_you_rescind * quantity_purchased) 
                             = 705 := by
  sorry

end total_amount_spent_l214_214870


namespace least_diagonal_eq_sqrt_162_l214_214614

theorem least_diagonal_eq_sqrt_162 (l w : ℝ) (h_perimeter : 2 * l + 2 * w = 36) : 
  ∃ l w, (l + w = 18) ∧ (sqrt (2 * ((l - 9) ^ 2) + 162) = sqrt 162) :=
by 
  -- setup
  use [9, 9]
  split
  { -- proof that l + w = 18
    exact calc
      9 + 9 = 18 : by norm_num }
  { -- proof that diagonal is sqrt 162
    exact calc
      sqrt (2 * (9 - 9)^2 + 162) = sqrt 162 : by simp }

end least_diagonal_eq_sqrt_162_l214_214614


namespace prob_same_suit_is_correct_prob_same_rank_is_correct_prob_same_suit_or_rank_is_correct_prob_second_higher_rank_is_correct_l214_214671

-- Definition for Hungarian card
structure HungarianCard where
  suit : Fin 4  -- one of the 4 suits
  rank : Fin 8  -- one of the 8 ranks

-- Conditions
def deck : Finset HungarianCard := 
  Finset.univ

-- Definitions of probabilities based on conditions
noncomputable def prob_same_suit : ℝ := 8 / 32
noncomputable def prob_same_rank : ℝ := 4 / 32
noncomputable def prob_same_suit_or_rank : ℝ := 11 / 32
noncomputable def prob_second_higher_rank : ℝ := 28 / 64

-- Proof statements
theorem prob_same_suit_is_correct :
  prob_same_suit = 0.25 := sorry

theorem prob_same_rank_is_correct :
  prob_same_rank = 0.125 := sorry

theorem prob_same_suit_or_rank_is_correct :
  prob_same_suit_or_rank = 0.344 := sorry

theorem prob_second_higher_rank_is_correct :
  prob_second_higher_rank = 0.438 := sorry

end prob_same_suit_is_correct_prob_same_rank_is_correct_prob_same_suit_or_rank_is_correct_prob_second_higher_rank_is_correct_l214_214671


namespace is_largest_interesting_l214_214279

def is_interesting (n : ℕ) : Prop :=
  let digits := (n.digits 10).reverse
  ∀ i, 1 ≤ i ∧ i < digits.length - 1 → 2 * (digits.get ⟨i, by simp [digits]⟩) < (digits.get ⟨i-1, by simp [digits]⟩ + digits.get ⟨i+1, by simp [digits]⟩)

def largest_interesting_number : ℕ :=
  96433469

theorem is_largest_interesting : is_interesting (largest_interesting_number) ∧ ∀ n, is_interesting n → n ≤ largest_interesting_number := 
  by
    sorry

end is_largest_interesting_l214_214279


namespace problem1_problem2_problem3_l214_214850

-- Problem 1
theorem problem1 (x1 y1 x2 y2 : ℝ) (hA : y1 = 1/2 + log (x1 / (1 - x1))) (hB : y2 = 1/2 + log (x2 / (1 - x2))) (hx : x1 + x2 = 1) :
  (1/2) * (y1 + y2) = 1/2 := sorry

-- Problem 2
theorem problem2 (n : ℕ) (h : 1 < n) :
  ∑ i in range (n - 1), (1/2 + log (↑i / (↑n - ↑i))) = (n - 1) / 2 := sorry

-- Problem 3
noncomputable def sequence_a (n : ℕ) : ℝ :=
  if n = 1 then 2 / 3 else 1 / (((n - 1) / 2 + 1) * (n / 2 + 1))

noncomputable def sequence_T (n : ℕ) : ℝ :=
  finset.sum (finset.range (n + 1)) (λ i, sequence_a (i + 1))

theorem problem3 (λ : ℝ) (h : ∀ n : ℕ, 0 < n → sequence_T n < λ * ((n / 2 + 1))) :
  1 / 2 < λ := sorry

end problem1_problem2_problem3_l214_214850


namespace roots_irrational_of_product_eq_10_l214_214403

theorem roots_irrational_of_product_eq_10 (k : ℝ) 
  (h : (3 * k ^ 2 + 1 = 10)) : 
  ∃ a b : ℝ, (a ≠ b) ∧ (a ^ 2 - 4 * k * a + 3 * k ^ 2 + 1 = 0) ∧ (b ^ 2 - 4 * k * b + 3 * k ^ 2 + 1 = 0) ∧ 
  irrational a ∧ irrational b :=
begin
  sorry
end

end roots_irrational_of_product_eq_10_l214_214403


namespace min_period_sin_l214_214993

def f (x : ℝ) : ℝ := Real.sin x

theorem min_period_sin : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 2 * Real.pi := by
  sorry

end min_period_sin_l214_214993


namespace collinear_A_l214_214162

noncomputable theory
open EuclideanGeometry

variables {A B C H_a H_b H_c T_a T_b T_c A' B' C': Point}
variables (ABC: Triangle)
variables (H: ABC.Orthocenter)
variables (I: ABC.Incenter)
variables [IncircleTouches ABC T_a T_b T_c]
variables [AltitudesFeet ABC H_a H_b H_c]
variables [CircumcirclesIntersectAt ABC A' B' C']

-- Final theorem statement
theorem collinear_A'_B'_C' :
  collinear (H :: I :: [A', B', C']) :=
sorry

end collinear_A_l214_214162


namespace ones_digit_of_14_pow_14_pow_7_pow_7_l214_214393

theorem ones_digit_of_14_pow_14_pow_7_pow_7 :
  (14 ^ (14 * (7 ^ 7))) % 10 = 4 :=
by
  -- Conditions derived as definitions
  have h1 : ∀ n, (14^n) % 10 = (4^n) % 10,
  intro n,
  exact Nat.mod_eq_of_lt (Nat.pow_mod 14 n 10) (Nat.pow_le_pow_of_le_left 14 (Nat.le_refl n)),

  have h2 : ∀ n, (4^n % 10 = 4 ∨ 4^n % 10 = 6),
  intro n,
  induction n with d hd,
  left, exact rfl,
  cases hd,
  right, rw hd, exact rfl,
  left, rw hd, exact rfl,

  have h3 : (14 * (7^7)) % 2 = 0,
  exact Nat.zero_mod (14 * (7 ^ 7) % 2),
  
  -- Result depends on the periodicity of 4^n % 10    
  rw h1,
  suffices h4: (14 * (7^7)) % 2 = 0, by
    cases Nat.mod_eq_of_lt (14 * (7^7) % 2) 2 with H H,
    exact h2.1,
    exact h2.2 hl,
  sorry

end ones_digit_of_14_pow_14_pow_7_pow_7_l214_214393


namespace denise_crayons_l214_214763

theorem denise_crayons (c : ℕ) :
  (∀ f p : ℕ, f = 30 ∧ p = 7 → c = f * p) → c = 210 :=
by
  intro h
  specialize h 30 7 ⟨rfl, rfl⟩
  exact h

end denise_crayons_l214_214763


namespace decreasing_intervals_l214_214839

noncomputable def f (x : ℝ) : ℝ := -x^2 + |x|

theorem decreasing_intervals :
  ∀ x, (x ∈ Icc (- (1 : ℝ) / 2) 0 ∨ x ∈ Ici (1 / 2)) → (∀ ε > 0, f (x + ε) ≤ f x) :=
by
  sorry

end decreasing_intervals_l214_214839


namespace sad_girls_count_l214_214956

variables (total_children happy_children sad_children neither_children : ℕ)
variables (total_boys total_girls happy_boys sad_children total_sad_boys : ℕ)

theorem sad_girls_count :
  total_children = 60 ∧ 
  happy_children = 30 ∧ 
  sad_children = 10 ∧ 
  neither_children = 20 ∧ 
  total_boys = 17 ∧ 
  total_girls = 43 ∧ 
  happy_boys = 6 ∧ 
  neither_boys = 5 ∧ 
  sad_children = total_sad_boys + (sad_children - total_sad_boys) ∧ 
  total_sad_boys = total_boys - happy_boys - neither_boys → 
  (sad_children - total_sad_boys = 4) := 
by
  intros h
  sorry

end sad_girls_count_l214_214956


namespace ratio_of_areas_is_one_l214_214597

theorem ratio_of_areas_is_one {A B C D : Type}
  (hBAC : ∠A B C = 90) 
  (hABC : ∠A B C = 45)
  (hDBC : ∠D B C = 45) :
  (area (triangle A D B)) / (area (triangle C D B)) = 1 := 
sorry

end ratio_of_areas_is_one_l214_214597


namespace integral_equality_l214_214390

-- Definitions for the problem using given conditions
def m := 1 / 3
def n := 1 / 2
def p := 3

noncomputable def target_integral (x : ℝ) := ∫ (x^(m) * (2 + 3 * x^(n))^p) dx

-- Correct answer transformed into a Lean expression
noncomputable def correct_answer (x : ℝ) :=
  6 * real.rpow x (4 / 3) +
  216 / 11 * real.rpow x (11 / 6) +
  162 / 7 * real.rpow x (7 / 3) +
  162 / 17 * real.rpow x (17 / 6) +
  arbitrary -- represents the constant of integration

-- Theorem statement without the proof 
theorem integral_equality (x : ℝ) : target_integral x = correct_answer x :=
sorry -- proof is not required

end integral_equality_l214_214390


namespace pentagons_from_15_points_l214_214016

theorem pentagons_from_15_points (n : ℕ) (h : n = 15) : (nat.choose 15 5) = 3003 := by
  rw h
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num)

end pentagons_from_15_points_l214_214016


namespace ratio_of_radii_is_sqrt_1_84_l214_214254

noncomputable def ratio_of_radii (r1 r2 : ℝ) : Prop :=
  let A1 := π * r1^2
  let A2 := π * r2^2
  A2 = 1.84 * A1

theorem ratio_of_radii_is_sqrt_1_84 (r1 r2 : ℝ) 
  (h : ratio_of_radii r1 r2) : 
  r2 / r1 = Real.sqrt 1.84 :=
by 
  sorry

end ratio_of_radii_is_sqrt_1_84_l214_214254


namespace sequences_properties_l214_214036

-- Define sequences and properties
def has_P_property (a : ℕ → ℕ) : Prop := ∀ n, n > 0 → ∃ k, a n + n = k^2
def is_permutation (a b : ℕ → ℕ) : Prop := ∀ n, n > 0 → ∃ m, b m = a n
def has_transformed_P_property (a : ℕ → ℕ) : Prop := 
  ∃ b, is_permutation a b ∧ has_P_property b

def seq1 (n : ℕ) : ℕ := [1, 2, 3, 4, 5].nth (n-1).get_or_else 0
def seq2 (n : ℕ) : ℕ := if n ≤ 12 then n else 0
def seq3 (n : ℕ) : ℕ := if n = 1 then 0 else n^2 - n

-- Formalizing the proof problem in Lean
theorem sequences_properties :
  has_transformed_P_property seq1 ∧ has_transformed_P_property seq2 ∧ has_P_property seq3 :=
by
  sorry

end sequences_properties_l214_214036


namespace baker_sold_cakes_l214_214339

def initialCakes : Nat := 110
def additionalCakes : Nat := 76
def remainingCakes : Nat := 111
def cakesSold : Nat := 75

theorem baker_sold_cakes :
  initialCakes + additionalCakes - remainingCakes = cakesSold := by
  sorry

end baker_sold_cakes_l214_214339


namespace boat_speed_ratio_l214_214314

axioms (B S D : ℝ)
-- Defining the times taken to row against and with the stream
def T_against : ℝ := D / (B - S)
def T_with : ℝ := D / (B + S)

-- Given condition that the time taken against the stream is twice that with the stream
axiom time_condition : T_against = 2 * T_with

-- Lean theorem statement to prove the ratio of the speed of the boat in still water to the speed of the stream
theorem boat_speed_ratio : B / S = 3 :=
by
  sorry

end boat_speed_ratio_l214_214314


namespace john_purchased_large_bottles_l214_214920

noncomputable def large_bottle_cost : ℝ := 1.75
noncomputable def small_bottle_cost : ℝ := 1.35
noncomputable def num_small_bottles : ℝ := 690
noncomputable def avg_price_paid : ℝ := 1.6163438256658595
noncomputable def total_small_cost : ℝ := num_small_bottles * small_bottle_cost
noncomputable def total_cost (L : ℝ) : ℝ := large_bottle_cost * L + total_small_cost
noncomputable def total_bottles (L : ℝ) : ℝ := L + num_small_bottles

theorem john_purchased_large_bottles : ∃ L : ℝ, 
  (total_cost L / total_bottles L = avg_price_paid) ∧ 
  (L = 1380) := 
sorry

end john_purchased_large_bottles_l214_214920


namespace number_of_sums_l214_214161

theorem number_of_sums {n : ℕ} : 
  let S_n := (n * (n + 1)) / 2 in
  (S_n + 1) = ((n * (n + 1)) / 2 + 1) :=
by
  sorry

end number_of_sums_l214_214161


namespace probability_XOXOXOX_l214_214433

theorem probability_XOXOXOX :
  let X := 4;
      O := 3;
      total_positions := 7;
      specific_arrangement := 1;
      total_arrangements := Nat.choose total_positions X in
  1 / total_arrangements = 1 / 35 := by
  sorry

end probability_XOXOXOX_l214_214433


namespace find_vector_v_l214_214788

theorem find_vector_v :
  ∃ (v : ℝ × ℝ), 
    let proj1_v := ((v.1 * 3 + v.2 * 1) / (3^2 + 1^2)) * (⟨3, 1⟩ : ℝ × ℝ),
    let proj2_v := ((v.1 * 1 + v.2 * 4) / (1^2 + 4^2)) * (⟨1, 4⟩ : ℝ × ℝ) in
    proj1_v = (⟨4.5, 1.5⟩ : ℝ × ℝ) ∧
    proj2_v = (⟨65/17, 260/17⟩ : ℝ × ℝ) ∧
    v = (1.208556, 15.374331 : ℝ × ℝ) :=
sorry

end find_vector_v_l214_214788


namespace number_of_real_solutions_l214_214500

theorem number_of_real_solutions :
  (∃ (x : ℝ), (x ^ 2 + x - 12) ^ 2 = 81) →
  (∃! (x : ℝ), (x ^ 2 + x - 12) ^ 2 = 81) :=
sorry

end number_of_real_solutions_l214_214500


namespace younger_brother_initial_bricks_l214_214263

theorem younger_brother_initial_bricks (total_bricks : ℕ) (x : ℕ)
  (h1 : total_bricks = 26)
  (h2 : x = younger_brother_initial_bricks)
  (h3 : elder_brother_bricks := total_bricks - x + x / 2)
  (h4 : final_younger_brother_bricks := x / 2 + (elder_brother_bricks / 2) - 5)
  (h5 : final_elder_brother_bricks := elder_brother_bricks / 2 + 5)
  (h6 : final_elder_brother_bricks = final_younger_brother_bricks + 2) :
  x = 16 := 
by 
  sorry

end younger_brother_initial_bricks_l214_214263


namespace range_of_sum_l214_214807

variable {x y t : ℝ}

theorem range_of_sum :
  (1 = x^2 + 4*y^2 - 2*x*y) ∧ (x < 0) ∧ (y < 0) →
  -2 <= x + 2*y ∧ x + 2*y < 0 :=
by {
  sorry
}

end range_of_sum_l214_214807


namespace max_value_x_y_l214_214239

theorem max_value_x_y : ∃ x y : ℤ, 31 * 38 * Real.sqrt (x) + Real.sqrt (y) = Real.sqrt 2009 ∧ 
  (∃ m : ℤ, x + y = m ∧ ∀ z w : ℤ, 31 * 38 * Real.sqrt (z) + Real.sqrt (w) = Real.sqrt 2009 → 
  x + y ≥ z + w) :=
begin
  sorry,
end

end max_value_x_y_l214_214239


namespace solution_set_of_inequality_l214_214649

theorem solution_set_of_inequality (x : ℝ) :
  x^2 - |x| - 2 < 0 ↔ -2 < x ∧ x < 2 := by
  sorry

end solution_set_of_inequality_l214_214649


namespace identify_counterfeit_bag_l214_214266

theorem identify_counterfeit_bag (n : ℕ) (w W : ℕ) (H : ∃ k : ℕ, k ≤ n ∧ W = w * (n * (n + 1) / 2) - k) : 
  ∃ bag_num, bag_num = w * (n * (n + 1) / 2) - W := by
  sorry

end identify_counterfeit_bag_l214_214266


namespace range_of_a_l214_214178

open Set Real

def set_M (a : ℝ) : Set ℝ := { x | x * (x - a - 1) < 0 }
def set_N : Set ℝ := { x | x^2 - 2*x - 3 ≤ 0 }

theorem range_of_a (a : ℝ) : set_M a ⊆ set_N ↔ -2 ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_a_l214_214178


namespace largest_interesting_number_is_96433469_l214_214278

def interesting_number (n : Nat) : Prop :=
  let digits := Nat.digits 10 n
  ∀ i, 1 ≤ i ∧ i < Array.size digits - 1 → digits.get ⟨i, Nat.lt_of_le_of_lt (Nat.succ_le_of_lt (Array.getLt' _ _ _)) (Nat.pred_lt')⟩ * 2 <
    digits.get ⟨i - 1, Nat.sub_lt (Nat.le_refl 1) zero_lt_one⟩ + 
    digits.get ⟨i + 1, Nat.add_lt_add (Nat.lt_of_le_of_lt (Nat.succ_le_of_lt (Array.getLt' _ _ _)) (Nat.pred_lt')) Nat.one_lt_succ_succ_tsub⟩

theorem largest_interesting_number_is_96433469 : 
  ∀ n, interesting_number n → n ≤ 96433469 :=
sorry

end largest_interesting_number_is_96433469_l214_214278


namespace sum_of_squares_mod_6_l214_214396

theorem sum_of_squares_mod_6 : 
  (∑ i in Finset.range 201, i^2) % 6 = 2 :=
begin
  sorry
end

end sum_of_squares_mod_6_l214_214396


namespace ceil_sqrt_sum_l214_214004

theorem ceil_sqrt_sum :
  ∀ (x y z : ℝ),
  (1 < sqrt x ∧ sqrt x < 2) →
  (5 < sqrt y ∧ sqrt y < 6) →
  (18 < sqrt z ∧ sqrt z < 19) →
  (⌈sqrt x⌉ + ⌈sqrt y⌉ + ⌈sqrt z⌉ = 27) :=
by
  intros x y z h1 h2 h3
  have h4 : ⌈sqrt 3⌉ = 2 := sorry  -- Details of proof omitted.
  have h5 : ⌈sqrt 35⌉ = 6 := sorry  -- Details of proof omitted.
  have h6 : ⌈sqrt 350⌉ = 19 := sorry  -- Details of proof omitted.
  exact eq.trans (by simp [h4, h5, h6]) sorry -- Simplify and sum values.

end ceil_sqrt_sum_l214_214004


namespace constant_term_in_expansion_rational_terms_in_expansion_l214_214831

noncomputable def term_formula (n r : ℕ) : ℚ := (Nat.choose n r) * ((-1/2)^r) * x^(2*n - 3*r) / 4

theorem constant_term_in_expansion (n : ℕ) (x : ℚ) :
  (term_formula n 4).is_constant ↔ n = 6 := by sorry

theorem rational_terms_in_expansion (n : ℕ) (x : ℚ) :
  n = 6 → ∃ r : ℕ, term_formula n r ∈ [term_formula 6 0, term_formula 6 4] := by sorry

end constant_term_in_expansion_rational_terms_in_expansion_l214_214831


namespace circumcircle_diameter_of_triangle_l214_214446

open Real

def triangle_circumcircle_diameter (AB AC ∠BAC BC R Diameter : ℝ) : Prop :=
  AB = sqrt 2 ∧
  AC = 4 ∧
  ∠BAC = 45 * (π / 180) ∧
  BC = sqrt (AC^2 + AB^2 - 2 * AB * AC * cos ∠BAC) ∧
  R = BC / (2 * sin ∠BAC) ∧
  Diameter = 2 * R

theorem circumcircle_diameter_of_triangle : triangle_circumcircle_diameter (sqrt 2) 4 (45 * (π / 180)) (sqrt 10) (sqrt 5) (2 * sqrt 5) :=
by
  sorry

end circumcircle_diameter_of_triangle_l214_214446


namespace sqrt_18_mul_sqrt_32_eq_24_l214_214217
  
theorem sqrt_18_mul_sqrt_32_eq_24 : (Real.sqrt 18 * Real.sqrt 32 = 24) :=
  sorry

end sqrt_18_mul_sqrt_32_eq_24_l214_214217


namespace congruent_polygons_exist_l214_214404

theorem congruent_polygons_exist
  (n p : ℕ)
  (hn : 6 ≤ n)
  (hp1 : 4 ≤ p)
  (hp2 : p ≤ n / 2) :
  ∃ k ≥ (p / 2).floor + 1, 
    (∃ (red_polygon blue_polygon : finset ℕ), 
      (red_polygon.card = k) ∧ (blue_polygon.card = k) ∧
      (∀ v ∈ red_polygon, v < n) ∧ 
      (∀ v ∈ blue_polygon, v < n) ∧
      (∀ v ∈ red_polygon, is_red v) ∧ 
      (∀ v ∈ blue_polygon, is_blue v)) :=
sorry

end congruent_polygons_exist_l214_214404


namespace area_expression_l214_214526

noncomputable def area_of_visible_shaded_region : ℝ :=
  let total_area := 36 * (2 * 2)
  let small_circle_area := 4 * (Real.pi * (1.5 * 1.5))
  let large_circle_area := 2 * (Real.pi * (2 * 2))
  total_area - (small_circle_area + large_circle_area)

theorem area_expression:
  (A B : ℝ) (H : area_of_visible_shaded_region = A - B * Real.pi) :
  A + B = 161 :=
by {
  have H1: area_of_visible_shaded_region = 144 - 17 * Real.pi,
  from sorry,
  have A_eq_144: A = 144,
  from sorry,
  have B_eq_17 : B = 17,
  from sorry,
  rw [A_eq_144, B_eq_17],
  norm_num,
}

end area_expression_l214_214526


namespace abcf_cyclic_l214_214134

noncomputable def is_cyclic_quadrilateral 
  (A B C F D E : Point) 
  (BF AF FC CD AE AC : ℝ) : Prop :=
(BF = AF + FC) ∧ 
(BD = AC) ∧ 
(BE = AC) ∧ 
(AF * CD = FC * AE) → 
(cyclic_quadrilateral A B C F)

-- Assuming the existence of these points and lengths, 
-- let's define the Lean 4 statement for the proof problem.
theorem abcf_cyclic {A B C F D E : Point} 
  (BF AF FC CD AE AC : ℝ) 
  (h1 : BF = AF + FC)
  (h2 : BD = AC)
  (h3 : BE = AC)
  (h4 : AF * CD = FC * AE) : 
  cyclic_quadrilateral A B C F := 
sorry

end abcf_cyclic_l214_214134


namespace yard_area_l214_214043

theorem yard_area (L W : ℕ) (hL : L = 40) (hFence : 2 * W + L = 50) : L * W = 200 := by
  have hW : W = 5 := by
    linarith
  rw [hL, hW]
  norm_num

end yard_area_l214_214043


namespace don_left_2_hours_after_cara_l214_214746

theorem don_left_2_hours_after_cara
  (distance : ℕ)
  (cara_speed : ℕ)
  (don_speed : ℕ)
  (cara_distance_when_meet : ℕ)
  (meet_distance : ℕ) :
  distance = 45 →
  cara_speed = 6 →
  don_speed = 5 →
  cara_distance_when_meet = 30 →
  meet_distance = (distance - cara_distance_when_meet) →
  let t := cara_distance_when_meet / cara_speed in
  let t_D := meet_distance / don_speed in
  t - t_D = 2 := by
  intros _ _ _ _ _
  rw meet_distance
  -- placeholders for completing the rigorous proof, not necessary according to the problem's requirements.
  sorry

end don_left_2_hours_after_cara_l214_214746


namespace least_element_in_T_least_possible_value_in_T_l214_214568

def is_set_T (T : Set ℕ) : Prop :=
  T ⊆ {n | 1 ≤ n ∧ n ≤ 18} ∧ T.card = 8 ∧
  ∀ x y ∈ T, x < y → ¬ (y % x = 0 ∨ ∃ k : ℕ, y = x^k)

theorem least_element_in_T (T : Set ℕ) (hT : is_set_T T) : ∃ x ∈ T, ∀ y ∈ T, x ≤ y :=
begin
  sorry
end

theorem least_possible_value_in_T : ∃ T : Set ℕ, is_set_T T ∧ ∃ x ∈ T, x = 3 :=
begin
  sorry
end

end least_element_in_T_least_possible_value_in_T_l214_214568


namespace pencil_price_decrease_l214_214600

theorem pencil_price_decrease :
  let original_price := (4 : ℝ) / 3
  let new_price := (3 : ℝ) / 4
  (original_price - new_price) / original_price * 100 ≈ 43.6 :=
by 
  let original_price := (4 : ℝ) / 3
  let new_price := (3 : ℝ) / 4
  let percent_decrease := (original_price - new_price) / original_price * 100
  sorry

end pencil_price_decrease_l214_214600


namespace buffy_whiskers_l214_214801

theorem buffy_whiskers :
  ∀ (Puffy Scruffy Buffy Juniper : ℕ),
    Juniper = 12 →
    Puffy = 3 * Juniper →
    Puffy = Scruffy / 2 →
    Buffy = (Juniper + Puffy + Scruffy) / 3 →
    Buffy = 40 :=
by
  intros Puffy Scruffy Buffy Juniper hJuniper hPuffy hScruffy hBuffy
  sorry

end buffy_whiskers_l214_214801


namespace system_has_infinitely_many_solutions_l214_214504

theorem system_has_infinitely_many_solutions :
  ∃ (S : Set (ℝ × ℝ × ℝ)), (∀ x y z : ℝ, (x + y = 2 ∧ xy - z^2 = 1) ↔ (x, y, z) ∈ S) ∧ S.Infinite :=
by
  sorry

end system_has_infinitely_many_solutions_l214_214504


namespace solve_ab_c_eq_l214_214778

theorem solve_ab_c_eq (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_eq : 11^a + 3^b = c^2) :
  a = 4 ∧ b = 5 ∧ c = 122 :=
by
  sorry

end solve_ab_c_eq_l214_214778


namespace third_trial_point_is_336_l214_214553

noncomputable def third_trial_point (a b : ℕ) (ϕ : ℝ) (x1 x2 : ℝ) : ℝ :=
  a + ϕ * (x2 - a)

theorem third_trial_point_is_336
  (a b : ℕ) (ϕ : ℝ) (hϕ : ϕ = 0.618) (ha : a = 100) (hb : b = 1100)
  (hx1 : x1 = (a : ℝ) + hϕ * (b - a)) (hx2 : x2 = (a : ℝ) + ((b : ℝ) - x1))
  (h_condition : x1 > x2) :
  third_trial_point a x2 ϕ = 336 :=
begin
  sorry
end

end third_trial_point_is_336_l214_214553


namespace tangent_line_slope_eq_k_l214_214520

noncomputable def tangent_slope_at_ln_x (x : ℝ) (hx : x > 0) : ℝ := by
  exact 1 / x

theorem tangent_line_slope_eq_k (k : ℝ) (x : ℝ) (hx : x > 0) (h_tangent : k = tangent_slope_at_ln_x x hx) : k = 1 / Real.exp(1) := by
  have : x = Real.exp(1) := by
    sorry -- proof that x = e
  rw [this] at h_tangent
  exact h_tangent
  sorry -- fill in the rest of the proof


end tangent_line_slope_eq_k_l214_214520


namespace caskets_condition_satisfied_l214_214498

noncomputable def golden_box_inscription : Prop :=
  (P ↔ Q)

noncomputable def silver_box_inscription : Prop :=
  ¬Q

theorem caskets_condition_satisfied
  (P: Prop)
  (Q: Prop)
  (golden_box: P ↔ Q)
  (silver_box: ¬Q) :
  (¬(Q ∧ ¬P) ∧ (P ∨ ¬P) ∧ (Q ∨ ¬Q)) :=
by
  sorry

end caskets_condition_satisfied_l214_214498


namespace calc_sqrt_identity_l214_214048

theorem calc_sqrt_identity (x : ℝ) (hx : x + x⁻¹ = 3) 
  : x^(1/2) + x^(-1/2) = Real.sqrt 5 := 
by
  sorry

end calc_sqrt_identity_l214_214048


namespace ap_dot_ab_ac_l214_214074

variables {V : Type*} [inner_product_space ℝ V]
variables (A B C P : V)
variables (a b : V)
variables (t : ℝ)

-- Conditions
def equilateral_triangle (A B C : V) (side_length : ℝ) : Prop :=
  dist A B = side_length ∧ dist B C = side_length ∧ dist C A = side_length

theorem ap_dot_ab_ac (h1 : equilateral_triangle A B C 4)
                     (h2 : P ∈ line_segment ℝ B C)
                     (ha : a = B - A)
                     (hb : b = C - A)
                     (hab : inner_product_space.angle a b = real.pi / 3)
                     (h_norm_a : ∥a∥ = 4)
                     (h_norm_b : ∥b∥ = 4) :
  let ap := (1 - t) • a + t • b in
  inner ap (a + b) = 24 :=
by
  sorry

end ap_dot_ab_ac_l214_214074


namespace number_of_two_digit_primes_with_digit_sum_12_l214_214793

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_digit_sum_12 : 
  ∃! n, is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 12 :=
by
  sorry

end number_of_two_digit_primes_with_digit_sum_12_l214_214793


namespace find_x_l214_214395

noncomputable def x : ℝ := 3 * Real.sqrt 15

theorem find_x
  (c d : ℂ)
  (h_c : complex.abs c = 3)
  (h_d : complex.abs d = 4)
  (h_cd : c * d = x - 3 * complex.I) :
  x = 3 * Real.sqrt 15 :=
by
  sorry

end find_x_l214_214395


namespace count_non_powers_l214_214861

theorem count_non_powers (n : ℕ) : 
  n = 1000000 →
  let total := n in
  let sqrt_count := Nat.floor (Real.sqrt total) in
  let cbrt_count := Nat.floor (Real.cbrt total) in
  let sixth_power_count := Nat.floor (Real.root 6 total) in
  let included_excluded := sqrt_count + cbrt_count - sixth_power_count in
  let result := total - included_excluded in
  result = 998910 :=
by
  intros h_n total sqrt_count cbrt_count sixth_power_count included_excluded result
  dsimp only [total, sqrt_count, cbrt_count, sixth_power_count, included_excluded, result]
  have : total = 1000000 := h_n
  rw [this]
  have sqrt_val : sqrt_count = Nat.floor (Real.sqrt 1000000) := rfl
  rw [sqrt_val]
  have sqrt_floor : sqrt_count = Nat.floor 1000 := rfl
  rw [sqrt_floor]
  have sqrt_count_val : sqrt_count = 1000 := Nat.floor_eq id
    {norm_cast, apply Real.floor_of_nonneg, linarith}
  rw [sqrt_count_val]
  have cbrt_val : cbrt_count = Nat.floor (Real.cbrt 1000000) := rfl
  rw [cbrt_val]
  have cbrt_floor : cbrt_count = Nat.floor 100 := rfl
  rw [cbrt_floor]
  have cbrt_count_val : cbrt_count = 100 := Nat.floor_eq id
    {norm_cast, apply Real.floor_of_nonneg, linarith}
  rw [cbrt_count_val]
  have sixth_power_val : sixth_power_count = Nat.floor (Real.root 6 1000000) := rfl
  rw [sixth_power_val]
  have sixth_power_floor : sixth_power_count = Nat.floor 10 := rfl
  rw [sixth_power_floor]
  have sixth_power_count_val : sixth_power_count = 10 := Nat.floor_eq id
    {norm_cast, apply Real.floor_of_nonneg, linarith}
  rw [sixth_power_count_val]
  have inclusion_exclusion_val : included_excluded = 1000 + 100 - 10 := rfl
  rw [inclusion_exclusion_val]
  have inclusion_exclusion_result : included_excluded = 1090 := by
    norm_num
  rw [inclusion_exclusion_result]
  have final_count_val : result = 1000000 - 1090 := rfl
  rw [final_count_val]
  have final_count_result : result = 998910 := by
    norm_num
  exact final_count_result

end count_non_powers_l214_214861


namespace probability_XOXOXOX_is_one_over_thirty_five_l214_214442

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_arrangement : ℤ :=
  let total_arrangements := binomial_coefficient 7 4
  let favorable_outcomes := 1
  favorable_outcomes / total_arrangements

theorem probability_XOXOXOX_is_one_over_thirty_five :
  probability_of_arrangement = (1 : ℤ) / 35 := 
  by
  sorry

end probability_XOXOXOX_is_one_over_thirty_five_l214_214442


namespace find_a_l214_214840

noncomputable def f (a x : ℝ) : ℝ := a * x * Real.log x

theorem find_a (a : ℝ) (h : (deriv (f a)) e = 3) : a = 3 / 2 :=
by
-- placeholder for the proof
sorry

end find_a_l214_214840


namespace number_of_solutions_l214_214825

theorem number_of_solutions (a b : ℕ) (m : ℕ) (h_m : m = 2 * 10^4 + a * 10^3 + 1 * 10^2 + b * 10 + 9) :
  m ≡ 1 [MOD 13] ∨ m ≡ 3 [MOD 13] ∨ m ≡ 9 [MOD 13] →
  (m^2019 ≡ 1 [MOD 13]) → 
  (0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) →
  ∃ n : ℕ, n = 23 ∧ n = (number of ordered pairs (a, b) satisfying the conditions) :=
by
  sorry

end number_of_solutions_l214_214825


namespace probability_XOXOXOX_is_1_div_35_l214_214424

def count_combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def num_ways_to_choose_positions_for_X (total_positions : ℕ) (num_X : ℕ) : ℕ := 
  count_combinations total_positions num_X

def num_ways_for_specific_arrangement_XOXOXOX : ℕ := 1

def probability_of_XOXOXOX (num_ways_total : ℕ) (num_ways_specific : ℕ) : ℚ := 
  num_ways_specific / num_ways_total

theorem probability_XOXOXOX_is_1_div_35 :
  probability_of_XOXOXOX (num_ways_to_choose_positions_for_X 7 4) num_ways_for_specific_arrangement_XOXOXOX = 1 / 35 := by
  sorry

end probability_XOXOXOX_is_1_div_35_l214_214424


namespace divisibility_by_17_l214_214211

theorem divisibility_by_17 (n : ℤ) : 17 ∣ (3 * 5^(2 * n + 1) + 2^(3 * n + 1)) :=
by sorry

end divisibility_by_17_l214_214211


namespace height_of_parallelogram_l214_214980

open Classical

noncomputable def base : ℝ := 1 -- We can assume the value of the base since it is not provided

theorem height_of_parallelogram (height_triangle height_parallelogram : ℝ) (A_triangle A_parallelogram : ℝ) : 
  (height_triangle = 10) → 
  (A_triangle = A_parallelogram) → 
  (height_parallelogram = height_triangle / 2) → 
  (∃ base : ℝ, (base * height_triangle / 2 = base * height_parallelogram)) → 
  height_parallelogram = 5 :=
by
  intros h1 h2 h3 h4
  rw [h1, h3]
  exact eq_of_mul_eq_mul_left (by norm_num : base > 0) h4.2.symm
  sorry -- skipping the proof


end height_of_parallelogram_l214_214980


namespace f_2022_l214_214987

variable {R : Type*} [LinearOrder R]

def f (x : R) : R := sorry

axiom f_eq (a b : R) : f((a + 2 * b) / 3) = (f(a) + 2 * f(b)) / 3
axiom f_one : f(1) = 5
axiom f_four : f(4) = 2

theorem f_2022 : f(2022) = -2016 := sorry

end f_2022_l214_214987


namespace greatest_whole_number_lt_100_with_odd_factors_l214_214955

theorem greatest_whole_number_lt_100_with_odd_factors :
  ∃ n, n < 100 ∧ (∃ p : ℕ, n = p * p) ∧ 
    ∀ m, (m < 100 ∧ (∃ q : ℕ, m = q * q)) → m ≤ n :=
sorry

end greatest_whole_number_lt_100_with_odd_factors_l214_214955


namespace maisy_earns_more_l214_214951

theorem maisy_earns_more 
    (current_hours : ℕ) (current_wage : ℕ) 
    (new_hours : ℕ) (new_wage : ℕ) (bonus : ℕ)
    (h_current_job : current_hours = 8) 
    (h_current_wage : current_wage = 10)
    (h_new_job : new_hours = 4) 
    (h_new_wage : new_wage = 15)
    (h_bonus : bonus = 35) :
  (new_hours * new_wage + bonus) - (current_hours * current_wage) = 15 := 
by 
  sorry

end maisy_earns_more_l214_214951


namespace trajectory_equation_of_M_equation_of_line_l_l214_214055

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def C1 : Circle := ⟨(-4, 0), real.sqrt 2⟩
def C2 : Circle := ⟨(4, 0), real.sqrt 2⟩

def P := (2 : ℝ, 1 : ℝ)

theorem trajectory_equation_of_M :
  ∀ (M : Circle),
  (dist M.center C1.center = M.radius + real.sqrt 2) →
  (dist M.center C2.center = M.radius - real.sqrt 2) →
  (∃ x y : ℝ, (x ≥ real.sqrt 2) → (x, y) ∈ set_of (λ p, p.1 ^ 2 / 2 - p.2 ^ 2 / 14 = 1)) :=
by sorry

theorem equation_of_line_l :
  ∀ (A B : ℝ × ℝ),
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  ((A.1^2) / 2 - (A.2^2) / 14 = 1) →
  ((B.1^2) / 2 - (B.2^2) / 14 = 1) →
  (∃ l : ℝ → ℝ, l = λ x, 14 * x - 27) :=
by sorry

end trajectory_equation_of_M_equation_of_line_l_l214_214055


namespace count_ordered_quadruples_l214_214862

open Real

theorem count_ordered_quadruples : 
  { (a, b, c, d : ℤ) // (2 ≤ a) ∧ (1 ≤ b) ∧ (0 ≤ c) ∧ (1 ≤ d) ∧ (log a b = c ^ 2005) ∧ (a + b + c + d = 2006) }.card = 2 :=
by sorry

end count_ordered_quadruples_l214_214862


namespace intersection_of_A_and_B_l214_214262

def U : Set ℝ := { x | true }

def A : Set ℝ := { x | x^2 - 4 ≤ 0 }

def B : Set ℝ := { x | 2^(x-1) > 1 }

theorem intersection_of_A_and_B : A ∩ B = { x | 1 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_of_A_and_B_l214_214262


namespace degree_combined_poly_l214_214361

-- Let b through f be nonzero constants
variables {b c d e f : ℝ} (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0)

-- Define the polynomial factors
def poly1 := (x : ℝ) ↦ x^5 + b * x^8 + c
def poly2 := (x : ℝ) ↦ 2 * x^4 + d * x^3 + e
def poly3 := (x : ℝ) ↦ 3 * x^2 + f

-- Define the combined polynomial
def combined_poly := (x : ℝ) ↦ poly1 x * poly2 x * poly3 x

-- The degree of the polynomial
theorem degree_combined_poly : ∀ x : ℝ, (combined_poly x).degree = 14 :=
by sorry

end degree_combined_poly_l214_214361


namespace sum_first_three_terms_of_arithmetic_sequence_l214_214238

theorem sum_first_three_terms_of_arithmetic_sequence
    (a₈ : ℤ) (d : ℤ) (h₁ : a₈ = 20) (h₂ : d = 2) :
    let a₁ := a₈ - 7 * d in
    let a₂ := a₁ + d in
    let a₃ := a₁ + 2 * d in
    a₁ + a₂ + a₃ = 24 := by
    sorry

end sum_first_three_terms_of_arithmetic_sequence_l214_214238


namespace transformed_farthest_vertex_l214_214227

-- Definitions based on conditions
def center : (ℝ × ℝ) := (5, -3)
def area : ℝ := 16
def scale_factor : ℝ := 3
def dilation_center : (ℝ × ℝ) := (0, 0)

-- The theorem statement
theorem transformed_farthest_vertex :
  let side_length := real.sqrt area
  let initial_vertices := [
    (center.1 - side_length / 2, center.2 - side_length / 2),
    (center.1 - side_length / 2, center.2 + side_length / 2),
    (center.1 + side_length / 2, center.2 + side_length / 2),
    (center.1 + side_length / 2, center.2 - side_length / 2)
  ]
  let dilated_vertices := initial_vertices.map (λ v, (v.1 * scale_factor, v.2 * scale_factor))
  let reflected_vertices := dilated_vertices.map (λ v, (v.1, -v.2))
  let distances_from_origin := reflected_vertices.map (λ v, real.sqrt (v.1^2 + v.2^2))
  let max_distance_vertex := reflected_vertices.argmax distances_from_origin
  max_distance_vertex = (21, 15) :=
by
  sorry

end transformed_farthest_vertex_l214_214227


namespace total_number_of_bricks_required_l214_214687

noncomputable def courtyard_length_m := 25 -- in meters
noncomputable def courtyard_width_m := 16 -- in meters
noncomputable def brick_length_cm := 20 -- in centimeters
noncomputable def brick_width_cm := 10 -- in centimeters
noncomputable def meters_to_centimeters(m : ℕ) := m * 100 -- conversion factor

theorem total_number_of_bricks_required : 
  let courtyard_length_cm := meters_to_centimeters courtyard_length_m
  let courtyard_width_cm := meters_to_centimeters courtyard_width_m
  let courtyard_area_cm2 := courtyard_length_cm * courtyard_width_cm
  let brick_area_cm2 := brick_length_cm * brick_width_cm
  courtyard_area_cm2 / brick_area_cm2 = 20000 :=
by {
  -- Let L be the length and W be the width of the courtyard in cm
  let L := meters_to_centimeters courtyard_length_m,
  let W := meters_to_centimeters courtyard_width_m,
  
  -- Let A_courtyard be the area of the courtyard in square centimeters
  let A_courtyard := L * W,
  -- Let A_brick be the area of one brick in square centimeters
  let A_brick := brick_length_cm * brick_width_cm,
  
  -- We need to show that A_courtyard / A_brick equals 20000
  have : A_courtyard / A_brick = 20000 := by {
    calc A_courtyard / A_brick = (2500 * 1600) / (20 * 10) : sorry
                          ...  = 4000000 / 200 : sorry
                          ...  = 20000 : sorry,
  },
  exact this,
}

end total_number_of_bricks_required_l214_214687


namespace continuous_at_5_l214_214405

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x > 5 then x - 2 else 3 * x + b

theorem continuous_at_5 {b : ℝ} : ContinuousAt (fun x => f x b) 5 ↔ b = -12 := by
  sorry

end continuous_at_5_l214_214405


namespace range_of_f_omega_pi_l214_214835

noncomputable def f (ω x : ℝ) : ℝ :=
  sin (ω * x + (Real.pi / 2)) * sin (ω * x + (Real.pi / 4))

theorem range_of_f_omega_pi
  (ω : ℝ) (hω : ω > 0)
  (h_period : ∀ x, f ω (x + Real.pi) = f ω x) :
  (Set.range (fun x => f ω x) ∩ Set.Icc 0 (Real.pi / 2)) = Set.Icc 0 ((2 + Real.sqrt 2) / 4) :=
sorry

end range_of_f_omega_pi_l214_214835


namespace john_uses_16_planks_of_wood_l214_214041

theorem john_uses_16_planks_of_wood (nails_per_plank : ℕ) (total_nails : ℕ) (h1 : nails_per_plank = 2) (h2 : total_nails = 32) : (total_nails / nails_per_plank) = 16 :=
by
  -- hypotheses
  have h : total_nails = 32, from h2,
  have k : nails_per_plank = 2, from h1,
  -- required result
  show total_nails / nails_per_plank = 16, from sorry

end john_uses_16_planks_of_wood_l214_214041


namespace num_convex_pentagons_l214_214020

theorem num_convex_pentagons (n m : ℕ) (hn : n = 15) (hm : m = 5) : 
  Nat.choose n m = 3003 := by
  sorry

end num_convex_pentagons_l214_214020


namespace wire_length_rect_prism_l214_214670

theorem wire_length_rect_prism (w l h : ℕ) (hw : w = 10) (hl : l = 8) (hh : h = 5) : 
  4 * l + 4 * w + 4 * h = 92 := 
by 
  rw [hw, hl, hh] 
  norm_num

end wire_length_rect_prism_l214_214670


namespace train_speed_correct_l214_214328

noncomputable def train_speed : ℝ :=
  let distance := 120 -- meters
  let time := 5.999520038396929 -- seconds
  let speed_m_s := distance / time -- meters per second
  speed_m_s * 3.6 -- converting to km/hr

theorem train_speed_correct : train_speed = 72.004800384 := by
  simp [train_speed]
  sorry

end train_speed_correct_l214_214328


namespace sunzi_classic_equation_l214_214900

theorem sunzi_classic_equation (x : ℕ) : 3 * (x - 2) = 2 * x + 9 :=
  sorry

end sunzi_classic_equation_l214_214900


namespace total_money_spent_l214_214869

def cost_life_journey_cd : ℕ := 100
def cost_day_life_cd : ℕ := 50
def cost_when_rescind_cd : ℕ := 85
def number_of_cds_each : ℕ := 3

theorem total_money_spent :
  number_of_cds_each * cost_life_journey_cd +
  number_of_cds_each * cost_day_life_cd +
  number_of_cds_each * cost_when_rescind_cd = 705 :=
sorry

end total_money_spent_l214_214869


namespace storks_initially_l214_214612

-- Definitions for conditions
variable (S : ℕ) -- initial number of storks
variable (B : ℕ) -- initial number of birds

theorem storks_initially (h1 : B = 2) (h2 : S = B + 3 + 1) : S = 6 := by
  -- proof goes here
  sorry

end storks_initially_l214_214612


namespace avg_income_difference_l214_214605

theorem avg_income_difference :
  let avg_income := 1025.68 in
  let greatest_even_income := 1024 in
  avg_income - greatest_even_income = 1.68 :=
by
  let avg_income := 1025.68
  let greatest_even_income := 1024
  show avg_income - greatest_even_income = 1.68
  sorry

end avg_income_difference_l214_214605


namespace diana_clothes_cost_l214_214001

noncomputable def cost_winter_clothes : ℕ :=
  let toddlers := 6 in
  let school_age := 2 * toddlers in
  let pre_teens := school_age / 2 in
  let teenagers := 4 * toddlers + toddlers in
  let total_children := toddlers + school_age + pre_teens + teenagers in
  let toddler_cost := 35 in
  let school_age_cost := 45 in
  let pre_teens_cost := 55 in
  let teenagers_cost := 65 in
  let discount := 0.30 in
  let pre_teens_before_discount := pre_teens * pre_teens_cost in
  let pre_teens_discount := pre_teens_before_discount * discount in
  let pre_teens_after_discount := pre_teens_before_discount - pre_teens_discount in
  let total_cost_before_discount := 
    (toddlers * toddler_cost) + 
    (school_age * school_age_cost) + 
    (pre_teens_before_discount) + 
    (teenagers * teenagers_cost) in
  let total_cost_after_discount := 
    (toddlers * toddler_cost) + 
    (school_age * school_age_cost) + 
    (pre_teens_after_discount) + 
    (teenagers * teenagers_cost) in
  total_cost_after_discount

theorem diana_clothes_cost : cost_winter_clothes = 2931 :=
by
  sorry

end diana_clothes_cost_l214_214001


namespace problem1_problem2_l214_214348

-- Problem 1: Prove that \(\sqrt{27}+3\sqrt{\frac{1}{3}}-\sqrt{24} \times \sqrt{2} = 0\)
theorem problem1 : Real.sqrt 27 + 3 * Real.sqrt (1 / 3) - Real.sqrt 24 * Real.sqrt 2 = 0 := 
by sorry

-- Problem 2: Prove that \((\sqrt{5}-2)(2+\sqrt{5})-{(\sqrt{3}-1)}^{2} = -3 + 2\sqrt{3}\)
theorem problem2 : (Real.sqrt 5 - 2) * (2 + Real.sqrt 5) - (Real.sqrt 3 - 1) ^ 2 = -3 + 2 * Real.sqrt 3 := 
by sorry

end problem1_problem2_l214_214348


namespace find_line_equation_l214_214832

variables {R : Type*} [LinearOrderedField R]

def line_equation (a b c : R) : Prop := a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

theorem find_line_equation (a b c : R) (l : R → R → R) :
  (∃ m x : R, y = m * x - 3 ∧ sqrt(x^2 + 9) = 5 ∧ l = 3*x - 4*y - 12 ∨ l = 3*x + 4*y + 12) → line_equation 3 (-4) (-12) ∨ line_equation 3 4 12 :=
by sorry

end find_line_equation_l214_214832


namespace conjugate_of_fraction_l214_214169

noncomputable def conjugate (z : ℂ) : ℂ := complex.conj z

theorem conjugate_of_fraction : conjugate (2 / (1 + complex.i)) = 1 + complex.i :=
by
  sorry

end conjugate_of_fraction_l214_214169


namespace area_within_C_outside_A_B_l214_214747

-- Define the conditions
structure Circle :=
  (radius : ℝ)
  (center : ℝ × ℝ)

def areTangent (c1 c2 : Circle) : Prop :=
  let dist := (c1.center.1 - c2.center.1) ^ 2 + (c1.center.2 - c2.center.2) ^ 2
  dist = (c1.radius + c2.radius) ^ 2

def tangentPoint (c1 c2 : Circle) : Prop :=
  areTangent c1 c2 ∧ (2 * c1.radius) = (2 * c2.radius)

noncomputable def areaOutsideCircles (cA cB cC : Circle) : ℝ :=
  (π * cC.radius ^ 2) - (π * cA.radius ^ 2) - (π * cB.radius ^ 2)

-- Define the problem to prove
theorem area_within_C_outside_A_B :
  ∀ (cA cB cC : Circle),
  cA.radius = 1 →
  cB.radius = 1 →
  cC.radius = 2 →
  tangentPoint cA cB →
  distances cA cB = 2 →
  isTangentToMidpoint cC (midpoint cA cB) →
  areaOutsideCircles cA cB cC = 3 * π + 4 :=
by
  intros cA cB cC hA hB hC tAB dAB midC
  sorry

end area_within_C_outside_A_B_l214_214747


namespace convex_heptagon_divide_2011_equal_triangles_l214_214002

theorem convex_heptagon_divide_2011_equal_triangles :
  ∃ (H : convex_heptagon), can_divide_into_equal_triangles H 2011 := 
sorry

end convex_heptagon_divide_2011_equal_triangles_l214_214002


namespace quadratic_root_3_m_value_l214_214113

theorem quadratic_root_3_m_value (m : ℝ) : (∃ x : ℝ, 2*x*x - m*x + 3 = 0 ∧ x = 3) → m = 7 :=
by
  sorry

end quadratic_root_3_m_value_l214_214113


namespace h_2030_eq_2030_l214_214175

noncomputable def h : ℕ → ℝ
| 1     := 2
| 2     := 2
| (n+3) := h(n+2) - h(n+1) + (n+3) + Real.sin((n+3) * Real.pi / 6)

theorem h_2030_eq_2030 : h 2030 = 2030 :=
by {
  sorry
}

end h_2030_eq_2030_l214_214175


namespace sin_C_correct_l214_214907

noncomputable def sinC_in_triangle (a c cosA : ℝ) (ha : a = 4) (hc : c = 6) (hcosA : cosA = 12/13) : ℝ :=
  let sinA := Real.sqrt (1 - cosA^2)
  let sinC := (c * sinA) / a
  sinC

theorem sin_C_correct : 
  ∀ (a c cosA : ℝ), a = 4 → c = 6 → cosA = 12 / 13 → sinC_in_triangle a c cosA a (rfl) c (rfl) cosA (rfl) = 15 / 26 :=
by
  intros a c cosA ha hc hcosA
  rw [sinC_in_triangle]
  have hs := Real.sqrt_eq_rfl (1 - (cosA ^ 2))
  have := hs (cosA) (by {
    rw hcosA,
    field_simp,
    norm_num }),
  sorry

end sin_C_correct_l214_214907


namespace triangle_sides_l214_214270

theorem triangle_sides (x y z : ℕ) (h : x^2 + y^2 = 5 * z^2) : 
  (x = 22 → y = 19 → z = 13) :=
begin
  sorry
end

end triangle_sides_l214_214270


namespace diameter_of_the_gate_l214_214680

noncomputable def circle_equation : ℝ × ℝ → Prop :=
λ p, (p.1^2 + p.2^2 + 2 * p.1 - 4 * p.2 = -3)

theorem diameter_of_the_gate : ∃ d : ℝ, d = 2 * Real.sqrt 2 ∧ ∀ p : ℝ × ℝ, circle_equation p → (p.1 + 1)^2 + (p.2 - 2)^2 = 2 :=
by
  sorry

end diameter_of_the_gate_l214_214680


namespace Petya_final_stickers_l214_214959

variable (initial_stickers trades increase_per_trade : ℕ)

theorem Petya_final_stickers : 
  (initial_stickers = 1) → 
  (trades = 50) → 
  (increase_per_trade = 4) → 
  (initial_stickers + trades * increase_per_trade = 201) :=
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end Petya_final_stickers_l214_214959


namespace start_given_l214_214531

noncomputable def Va := 1000
noncomputable def Vb := 900
noncomputable def Vc := 944.44

theorem start_given (Va Vb Vc : ℝ) (T : ℝ) :
  (Va = 1000) → (Vb = 900) →
  (Vb = 1000) → (Vc = 944.44) →
  Va / Vc = 1.17647 → ∃ d : ℝ, d = 150 :=
by
  intro h1 h2 h3 h4 h5
  have h6 : Va / Vc = 1000 / 850 := by sorry
  have h7 : 1000 - 850 = 150 := by sorry
  existsi 150
  exact h7

end start_given_l214_214531


namespace count_n_sequences_l214_214062

def is_n_sequence (n : ℕ) (seq : List ℕ) : Prop :=
  (3 ≤ seq.length) ∧ (seq.all (λ x, x ∈ Finset.range (n + 1))) ∧
  (∀ k : ℕ, (k < seq.length - 2) → (seq.nth_le (k + 1) ((lt_of_lt_of_le (lt_add_one k) ((seq.length).pred_lt_succ (lt_of_le_of_ne (zero_le (seq.length)) (ne_of_gt (lt_trans zero_lt_three (lt_trans (lt_add_one 1) (le_of_lt (lt_of_le_of_ne (zero_le (seq.length)) (ne_of_gt (lt_trans zero_lt_three (le_of_lt (lt_of_le_of_ne (zero_le (seq.length)) (ne_of_gt (lt_trans zero_lt_three (lt_pred (seq.length))))) eq_bot_lt))) eq_bot_lt) eq_bot_lt))) eq_bot_lt)))) (lt_pred (seq.length))).pred)).val -
    seq.nth_le k (lt_of_lt_of_le (lt_pred (seq.length)) (ge_of_eq (succ_eq_add_one 2)))) *
    (seq.nth_le (k + 2) ((lt_trans (lt_add_one k) (lt_pred (seq.length))).pred)).val - 
    seq.nth_le k (lt_of_lt_of_le (lt_pred (seq.length)) (ge_of_eq (succ_eq_add_one 2))))).val < 0)

theorem count_n_sequences (n : ℕ) (h : 3 ≤ n) :
  (∃ (count : ℕ), count = 2^(n + 1) - n^2 - n - 2 ∧ 
    (count = (List.fin_range n).subsets.filter is_n_sequence).length) := 
sorry

end count_n_sequences_l214_214062


namespace probability_XOXOXOX_l214_214411

theorem probability_XOXOXOX (arrangement : list char) 
  (h_len : arrangement.length = 7) 
  (h_X_count : arrangement.count 'X' = 4) 
  (h_O_count : arrangement.count 'O' = 3) :
  let total_arrangements := nat.choose 7 4 in 
  let favorable_outcomes := 1 in
  favorable_outcomes / total_arrangements = 1 / 35 :=
by
  -- proof
  sorry

end probability_XOXOXOX_l214_214411


namespace range_of_x_max_a_l214_214315

-- Definitions based on conditions
def original_profit (A : ℕ) : ℕ := 1.5 * (500 - A) * 1.05

def profit_A_improvement (A : ℕ) : ℕ := 1.5 * 500

def profit_B (x : ℕ) (a : ℕ) : ℕ := 1.5 * (a - 13 / 1000 * x) * x

def condition_1 (x : ℕ) : Prop :=
  1.5 * (500 - x) * (1 + 0.005 * x) ≥ profit_A_improvement 500

def condition_2 (x : ℕ) (a : ℕ) : Prop :=
  profit_B x a ≤ 1.5 * (500 - x) * (1 + 0.005 * x)

theorem range_of_x (x : ℕ) (h : condition_1 x) : 0 < x ∧ x ≤ 300 := sorry

theorem max_a (a : ℕ) (h : ∀ (x : ℕ), 0 < x → x ≤ 250 → condition_2 x a) : a ≤ 5.5 := sorry

end range_of_x_max_a_l214_214315


namespace cyclic_projection_l214_214203

-- Definitions: Let quadrilateral ABCD with perpendicular diagonals AC and BD
variable (A B C D I A₁ B₁ C₁ D₁ : Type) [EuclideanGeometry A B C D I A₁ B₁ C₁ D₁]

-- Conditions: AC ⊥ BD
axiom perpendicular_diagonals : ∠AIC + ∠BID = 90

-- Prove: Quadrilateral A₁B₁C₁D₁ formed by the projections of I onto AB, BC, CD, DA is cyclic
theorem cyclic_projection (h : perpendicular_diagonals) :
  cyclic_quadrilateral A₁ B₁ C₁ D₁ :=
sorry

end cyclic_projection_l214_214203


namespace rate_of_change_of_circle_area_at_2_sec_l214_214774

/-- Given the radius of a circle expanding at 6 m/s, prove that the rate of change 
    of the circle's area at t = 2 seconds is 144π m²/s. -/
theorem rate_of_change_of_circle_area_at_2_sec :
  let radius := (λ t : ℝ, 6 * t),
      area := (λ t : ℝ, (radius t) ^ 2 * Real.pi) in
  (deriv area 2) = 144 * Real.pi :=
by
  let radius := (λ t : ℝ, 6 * t)
  let area := (λ t : ℝ, (radius t) ^ 2 * Real.pi)
  have radius_definition : ∀ t, radius t = 6 * t := by simp [radius]
  have area_definition : ∀ t, area t = 36 * t ^ 2 * Real.pi := by
    intros t; simp [area, (radius t) ^ 2, Real.pi, radius_definition]
  calc
    (deriv area 2) = (72 * 2 * Real.pi) : by
      apply deriv_of_times; simp [area_definition];
      sorry -- skip detailed calculation
    ... = 144 * Real.pi : by simp

end rate_of_change_of_circle_area_at_2_sec_l214_214774


namespace hyperbola_real_axis_length_l214_214630

-- Definitions from the conditions
def hyperbola_eq (λ : ℝ) : ℝ × ℝ → Prop :=
  λ (x : ℝ), x.1 ^ 2 - x.2 ^ 2 = λ

def directrix_eq : ℝ → Prop :=
  λ x, x = -1

-- Defining the conditions
variables (λ y : ℝ)
axiom center_origin : hyperbola_eq λ (0, 0)
axiom foci_x_axis : exists (c : ℝ), hyperbola_eq λ (c, 0) ∧ hyperbola_eq λ (-c, 0)
axiom intersection_points : hyperbola_eq λ (-1, y) ∧ hyperbola_eq λ (-1, -y)

-- Given the condition |AB| = √3
axiom ab_distance : |y - (-y)| = sqrt 3

theorem hyperbola_real_axis_length : λ = 1/4 → 1 :=
by
  -- Assuming λ = 1/4, we need to prove the length of the real axis is 1
  sorry

end hyperbola_real_axis_length_l214_214630


namespace smallest_positive_multiple_of_77_with_2020_digits_l214_214397

theorem smallest_positive_multiple_of_77_with_2020_digits :
  ∃ k : ℕ, let N := 10000 * k + 2020 in N % 77 = 0 ∧ N = 722020 :=
sorry

end smallest_positive_multiple_of_77_with_2020_digits_l214_214397


namespace range_of_a_l214_214809

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x ∈ Icc (-2:ℝ) (2:ℝ), f x ≥ 0) :
  -7 ≤ a ∧ a ≤ 2 :=
by
  have h0 : f = (λ x, x^2 + a * x + 3 - a) := by rfl
  sorry

end range_of_a_l214_214809


namespace triangle_area_l214_214765

/-- The area of triangle QCB where Q(0, 15), A(3, 15), C(0, p) and B(15, 0) in terms of p --/
theorem triangle_area (p : ℝ) :
  let Q := (0 : ℝ, 15 : ℝ),
    A := (3 : ℝ, 15 : ℝ),
    C := (0 : ℝ, p : ℝ),
    B := (15 : ℝ, 0 : ℝ) in
  ∃ area : ℝ, area = (225 - 15 * p) / 2 := 
by
  sorry

end triangle_area_l214_214765


namespace sum_reciprocal_inequality_l214_214467

def f (x t : ℝ) := (2 * x - t) / (x^2 + 1)

noncomputable def g (t : ℝ) : ℝ :=
  (8 * Real.sqrt (t^2 + 1) * (2 * t^2 + 5)) / (16 * t^2 + 25)

theorem sum_reciprocal_inequality (u1 u2 u3 : ℝ) 
  (h1 : 0 < u1) (h2 : u1 < π / 2) (h3 : 0 < u2) (h4 : u2 < π / 2) 
  (h5 : 0 < u3) (h6 : u3 < π / 2) 
  (h7 : Real.sin u1 + Real.sin u2 + Real.sin u3 = 1) :
  (1 / g (Real.tan u1)) + (1 / g (Real.tan u2)) + (1 / g (Real.tan u3)) < (3 / 4) * Real.sqrt 6 :=
sorry

end sum_reciprocal_inequality_l214_214467


namespace gcd_16_12_eq_4_l214_214389

theorem gcd_16_12_eq_4 : Int.gcd 16 12 = 4 := by
  sorry

end gcd_16_12_eq_4_l214_214389


namespace find_range_of_k_l214_214637

variables (x y k: ℝ)

-- Condition: Circle equation
def circle_eq := x^2 + y^2 + 2*x - 4*y + k - 2 = 0

-- Condition: Point (1, 2) can be the intersection of two tangents
def point_intersect_tangents (k: ℝ) : Prop :=
  (3 < k) ∧ (k < 7)

theorem find_range_of_k (k: ℝ) : (circle_eq = 0) → point_intersect_tangents k :=
by
  sorry

end find_range_of_k_l214_214637


namespace pop_spending_original_l214_214969

-- Given conditions
def total_spent := 150
def crackle_spending (P : ℝ) := 3 * P
def snap_spending (P : ℝ) := 2 * crackle_spending P

-- Main statement to prove
theorem pop_spending_original : ∃ P : ℝ, snap_spending P + crackle_spending P + P = total_spent ∧ P = 15 :=
by
  sorry

end pop_spending_original_l214_214969


namespace curve_equation_slope_constant_l214_214136

/-- 
  In the Cartesian coordinate plane, define a curve C such that any point P(x, y) on the curve C
  satisfies the condition of locus being perpendicular to the line x = -8. Further, for P ∈ C,
  OP is perpendicular to OQ, where Q is the intersection point of the perpendicular line passing
  through P and x = -8, prove that the equation of curve C is y^2 = 8x.
-/
theorem curve_equation (P : ℝ × ℝ) (O Q : ℝ × ℝ) (x y : ℝ) (hx : Q.1 = -8) (h_perp : 
  (P.1 * Q.1 + P.2 * y^2 = 0) ) (O_eq : O = (0, 0)) : P.2^2 = 8 * P.1 :=
by 
  sorry

/-- 
  Given a line l not parallel to the y-axis and passing through point (2, 0), 
  which intersects curve C at points M and N, where the equation of curve C 
  is y^2 = 8x, define point A as (-3, 0). Let k, k1, and k2 be the slopes of 
  lines l, AM, and AN respectively. Prove that the expression (k / k1) + (k / k2) 
  is constant and equals -1/2.
-/
theorem slope_constant (l A M N : ℝ × ℝ) (k k1 k2 : ℝ) (hx_l : l.1 = 2 ∧ l.2 = 0) 
  (hx_A : A = (-3, 0)) (h_C : ∀ P : ℝ × ℝ, P ∈ curve C ↔ P.2^2 = 8 * P.1) : 
  (k / k1) + (k / k2) = -1/2 :=
by 
  sorry

end curve_equation_slope_constant_l214_214136


namespace sin_alpha_of_point_on_terminal_side_l214_214595

theorem sin_alpha_of_point_on_terminal_side (a : ℝ) (h : a ≠ 0) (α : ℝ) (P : ℝ × ℝ) (hP : P = (a, a)) :
  sin α = sqrt 2 / 2 ∨ sin α = - (sqrt 2 / 2) :=
  sorry

end sin_alpha_of_point_on_terminal_side_l214_214595


namespace z_squared_l214_214812

noncomputable def z (z : ℂ) : Prop :=
  z * complex.I = 2 + complex.I

theorem z_squared {z : ℂ} (hz : z * complex.I = 2 + complex.I) : z^2 = -3 - 4 * complex.I :=
sorry

end z_squared_l214_214812


namespace increasing_sequences_mod_1000_l214_214345

theorem increasing_sequences_mod_1000 :
  (nat.choose 680 12) % 1000 = 680 :=
by sorry

end increasing_sequences_mod_1000_l214_214345


namespace problem_solution_l214_214678

theorem problem_solution 
  (x y : ℝ) (hxy_pos : 0 < x ∧ 0 < y) :
  4 * x^4 + 17 * x^2 * y + 4 * y^2 < (m / 4) * (x^4 + 2 * x^2 * y + y^2) ↔ 25 < m :=
sorry

end problem_solution_l214_214678


namespace find_initial_amount_l214_214267

-- Define the initial amount of money in the cash register
def initial_amount (x : ℝ) : Prop :=
  let after_first := 2 * x - 10 in
  let after_second := 2 * after_first - 10 in
  let after_third := 2 * after_second - 10 in
  after_third = 0

-- State the theorem
theorem find_initial_amount : ∃ x : ℝ, initial_amount x ∧ x = 8.75 :=
by {
  use 8.75,
  unfold initial_amount,
  repeat {linarith},
  sorry
}

end find_initial_amount_l214_214267


namespace sum_coordinates_A_l214_214927

noncomputable def A_sum_coordinates (A B C : ℝ × ℝ) : Prop :=
  let AC := (A.1 - C.1, A.2 - C.2)
  let AB := (A.1 - B.1, A.2 - B.2)
  let CB := (C.1 - B.1, C.2 - B.2)
  (AC.1 / AB.1 = 1 / 3) ∧ (AC.2 / AB.2 = 1 / 3) ∧ 
  (CB.1 / AB.1 = 1 / 3) ∧ (CB.2 / AB.2 = 1 / 3)

theorem sum_coordinates_A (A B C : ℝ × ℝ) :
  (A_sum_coordinates A B C) → sum.fst A + sum.snd A = 8.5 :=
by
  sorry

end sum_coordinates_A_l214_214927


namespace cos_8_minus_sin_8_l214_214497

theorem cos_8_minus_sin_8 (α m : ℝ) (h : Real.cos (2 * α) = m) :
  Real.cos α ^ 8 - Real.sin α ^ 8 = m * (1 + m^2) / 2 :=
by
  sorry

end cos_8_minus_sin_8_l214_214497


namespace transform_and_check_properties_l214_214511

noncomputable def f (x : ℝ) : ℝ := cos (4 * x + π / 3)

-- The period of a general cosine function cos(kx) is 2π/k.
def period (k : ℝ) : ℝ := 2 * π / k

theorem transform_and_check_properties :
  (f(0) = cos (π / 3)) ∧                             -- Just an example for verification
  (period 4 = π / 2) ∧                               -- Period B: is π/2
  (∀ x : ℝ, f(x) = f(-π / 3 - x)) :=                 -- Symmetric about the line x = -π / 12 (for x = -π / 12, we move π / 4)
by
  split
  sorry -- Proof of f(0) = cos (π / 3)
  split
  -- Period calc for 4
  sorry -- Proof
  -- Symmetry
  sorry -- Proof

end transform_and_check_properties_l214_214511


namespace necessary_sufficient_condition_l214_214248

theorem necessary_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ a ≥ 4 :=
by
  sorry

end necessary_sufficient_condition_l214_214248


namespace find_x_in_interval_l214_214787

noncomputable def a : ℝ := Real.sqrt 2014 - Real.sqrt 2013

theorem find_x_in_interval :
  ∀ x : ℝ, (0 < x) → (x < Real.pi) →
  (a^(Real.tan x ^ 2) + (Real.sqrt 2014 + Real.sqrt 2013)^(-Real.tan x ^ 2) = 2 * a^3) →
  (x = Real.pi / 3 ∨ x = 2 * Real.pi / 3) := by
  -- add proof here
  sorry

end find_x_in_interval_l214_214787


namespace difference_between_sevens_l214_214293

-- Define the numeral
def numeral : ℕ := 54179759

-- Define a function to find the place value of a digit at a specific position in a number
def place_value (n : ℕ) (pos : ℕ) : ℕ :=
  let digit := (n / 10^pos) % 10
  digit * 10^pos

-- Define specific place values for the two sevens
def first_seven_place : ℕ := place_value numeral 4  -- Ten-thousands place
def second_seven_place : ℕ := place_value numeral 1 -- Tens place

-- Define their values
def first_seven_value : ℕ := 7 * 10^4  -- 70,000
def second_seven_value : ℕ := 7 * 10^1  -- 70

-- Prove the difference between these place values
theorem difference_between_sevens : first_seven_value - second_seven_value = 69930 := by
  sorry

end difference_between_sevens_l214_214293


namespace evaluate_expression_l214_214454

theorem evaluate_expression (x y : ℝ) (h1 : x * y = -2) (h2 : x + y = 4) : x^2 * y + x * y^2 = -8 :=
by
  sorry

end evaluate_expression_l214_214454


namespace total_shaded_area_l214_214903

theorem total_shaded_area (r₁ r₂ : ℝ) (A₁ A₂ : ℝ)
  (h₁ : r₁ = 3)
  (h₂ : r₂ = 6)
  (hA₁ : A₁ = 18 - (9 * Real.pi / 2))
  (hA₂ : A₂ = 72 - (18 * Real.pi)) :
  A₁ + A₂ = 90 - 22.5 * Real.pi :=
by
  rw [h₁, h₂, hA₁, hA₂]
  simp
  sorry

end total_shaded_area_l214_214903


namespace total_floors_combined_l214_214616

-- Let C be the number of floors in the Chrysler Building
-- Let L be the number of floors in the Leeward Center
-- Given that C = 23 and C = L + 11
-- Prove that the total floors in both buildings combined equals 35

theorem total_floors_combined (C L : ℕ) (h1 : C = 23) (h2 : C = L + 11) : C + L = 35 :=
by
  sorry

end total_floors_combined_l214_214616


namespace find_t_l214_214463

variable (a b : ℝ × ℝ)
variable (c : ℝ × ℝ)
variable (t : ℝ)

axiom vector_a : a = (1, 2)
axiom vector_b : b = (3, 4)
axiom vector_c : c = (t, t + 2)
axiom perpendicular : (c.1 * (a.1 + b.1) + c.2 * (a.2 + b.2)) = 0

theorem find_t : t = -6 / 5 :=
by 
  rw [vector_a, vector_b, vector_c] at perpendicular
  sorry

end find_t_l214_214463


namespace find_ab_l214_214737

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

def f (x : ℝ) : ℝ := a * tan (b * x)

theorem find_ab 
  (ha_pos : a > 0) 
  (hb_pos : b > 0) 
  (hx : f (π / 4) = 3) 
  (hT : ∀ x, f (x + 3 * π / 4) = f x) : 
  a * b = 4 * sqrt 3 / 3 := 
sorry

end find_ab_l214_214737


namespace find_f_neg1_l214_214828

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_neg1 :
  (∀ x, f (-x) = -f x) →
  (∀ x, (0 < x) → f x = 2 * x * (x + 1)) →
  f (-1) = -4 := by
  intros h1 h2
  sorry

end find_f_neg1_l214_214828


namespace value_of_f_neg_a_l214_214758

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

-- Given condition: f(a) = 2
variable {a : ℝ}
hypothesis : f a = 2

-- Prove that f(-a) = 0
theorem value_of_f_neg_a : f (-a) = 0 :=
by
  sorry

end value_of_f_neg_a_l214_214758


namespace least_positive_integer_solution_l214_214675

theorem least_positive_integer_solution :
  ∃ x : ℤ, x > 0 ∧ ∃ n : ℤ, (3 * x + 29)^2 = 43 * n ∧ x = 19 :=
by
  sorry

end least_positive_integer_solution_l214_214675


namespace length_of_DB_l214_214901

theorem length_of_DB (A B C D : Type) (AC AD CD DB : ℝ) 
  (h1 : ∠(A, B, C) = 90)  -- ∠ABC is a right angle
  (h2 : ∠(A, D, B) = 90)  -- ∠ADB is a right angle
  (h3 : AC = 20)          -- AC = 20 units
  (h4 : AD = 6)           -- AD = 6 units
  (h5 : AC - AD = 14)     -- DC = AC - AD = 14 units
  (h6 : ∠(A, D, B) and ∠(B, D, C) similar) --the triangles are similar
  : DB = 2 * sqrt 21 := sorry

end length_of_DB_l214_214901


namespace work_completed_after_20_days_l214_214268

-- Given conditions
def personA_rate := (1 : ℝ) / 30
def personB_rate := (1 : ℝ) / 40
def personC_rate := (1 : ℝ) / 60

-- Define the total work completed after 20 days
def work_done_10_days_together := 10 * (personA_rate + personB_rate + personC_rate)
def work_done_10_days_withoutA := 10 * (personB_rate + personC_rate)
def total_work_done := work_done_10_days_together + work_done_10_days_withoutA

-- Lean 4 statement to prove the work completed after 20 days
theorem work_completed_after_20_days : total_work_done = 1 := by
  calc
    total_work_done = 10 * (personA_rate + personB_rate + personC_rate) + 10 * (personB_rate + personC_rate) : by rfl
    ... = 10 * (1/30 + 1/40 + 1/60) + 10 * (1/40 + 1/60) : by rfl
    ... = 10 * ((4 + 3 + 2) / 120) + 10 * ((3 + 2) / 120) : by norm_num
    ... = 10 * (9 / 120) + 10 * (5 / 120) : by rfl
    ... = 10 * (3 / 40) + 10 * (1 / 24) : by norm_num
    ... = 3/4 + 5/12 : by norm_num
    ... = (3 * 3 / 12) + 5 / 12 : by norm_num
    ... = 9/12 + 5/12 : by norm_num
    ... = 14/12 : by norm_num
    ... = 1 : by norm_num

end work_completed_after_20_days_l214_214268


namespace union_of_A_and_B_l214_214493

def A : Set ℝ := {x | x^2 - x ≤ 0}
def B : Set ℝ := {x | ∃ f, f(x) = Real.log (1 - |x|)}

theorem union_of_A_and_B :
  A ∪ B = Set.Ioc (-1 : ℝ) 1 := sorry

end union_of_A_and_B_l214_214493


namespace find_a_of_complex_on_line_l214_214114

theorem find_a_of_complex_on_line (a : ℝ) :
  let z := (1 - a * complex.I) / complex.I in
  let p := (-(a : ℝ), -1) in
  (p.1 + 2 * p.2 + 5 = 0) ↔ (a = 3) := 
by
  let z := (1 - a * complex.I) / complex.I
  let p := (-(a : ℝ), -1)
  sorry

end find_a_of_complex_on_line_l214_214114


namespace plantingMethodsCalculation_l214_214978

noncomputable def numPlantingMethods : Nat :=
  let totalSeeds := 5
  let endChoices := 3 * 2 -- Choosing 2 seeds for the ends from 3 remaining types
  let middleChoices := 6 -- Permutations of (A, B, another type) = 3! = 6
  endChoices * middleChoices

theorem plantingMethodsCalculation : numPlantingMethods = 24 := by
  sorry

end plantingMethodsCalculation_l214_214978


namespace c_investment_l214_214689

theorem c_investment (x : ℝ) (h1 : 5000 / (5000 + 8000 + x) * 88000 = 36000) : 
  x = 20454.5 :=
by
  sorry

end c_investment_l214_214689


namespace buffy_whiskers_l214_214796

/-- Definition of whisker counts for the cats --/
def whiskers_of_juniper : ℕ := 12
def whiskers_of_puffy : ℕ := 3 * whiskers_of_juniper
def whiskers_of_scruffy : ℕ := 2 * whiskers_of_puffy
def whiskers_of_buffy : ℕ := (whiskers_of_juniper + whiskers_of_puffy + whiskers_of_scruffy) / 3

/-- Proof statement for the number of whiskers of Buffy --/
theorem buffy_whiskers : whiskers_of_buffy = 40 := 
by
  -- Proof is omitted
  sorry

end buffy_whiskers_l214_214796


namespace problem_complex_sol_86_l214_214934

theorem problem_complex_sol_86 :
  ∃ (z : ℂ), (∃ (n : ℕ), n = 86) ∧ (∀ z, ((z : ℝ).im > 0) ∧ ((z^2 - 2 * (z * Complex.I) - 3).re ∈ (Set.Icc (-5 : ℝ) 5)) ∧ ((z^2 - 2 * (z * Complex.I) - 3).im ∈ (Set.Icc (-5 : ℝ) 5))) :=
sorry

end problem_complex_sol_86_l214_214934


namespace math_problem_l214_214094

open BigOperators

/-- Given vectors in the plane OA = (-1, -3), OB = (5, 3), and OM = (2, 2),
point P is on line OM, and PA • PB = -16.
1. Find the coordinates of OP.
2. Find the cosine of ∠APB.
3. Let t ∈ ℝ, find the minimum value of |OA + t OP|. -/
theorem math_problem
  (OA OB OM : ℝ × ℝ) (P : ℝ × ℝ)
  (hOA : OA = (-1, -3)) (hOB : OB = (5, 3)) (hOM : OM = (2, 2))
  (hP_on_OM : ∃ λ : ℝ, P = (λ * 2, λ * 2))
  (hPA_PB : (P.1 - OA.1) * (P.1 - OB.1) + (P.2 - OA.2) * (P.2 - OB.2) = -16) :
  (P = (1, 1)) ∧
  (let PA := (P.1 - OA.1, P.2 - OA.2),
       PB := (P.1 - OB.1, P.2 - OB.2)
   in (PA.1 * PB.1 + PA.2 * PB.2) /
     (real.sqrt (PA.1^2 + PA.2^2) * real.sqrt (PB.1^2 + PB.2^2)) = -4 / 5) ∧
  (∀ t : ℝ, real.sqrt ((OA.1 + t * P.1)^2 + (OA.2 + t * P.2)^2) =
    if t = 2 then real.sqrt 2 else sorry) :=
begin
  sorry
end

end math_problem_l214_214094


namespace positive_difference_perimeters_l214_214757

def perimeter_rectangle (length : ℕ) (width : ℕ) : ℕ :=
  2 * (length + width)

def perimeter_cross_shape : ℕ := 
  let top_and_bottom := 3 + 3 -- top and bottom edges
  let left_and_right := 3 + 3 -- left and right edges
  let internal_subtraction := 4
  top_and_bottom + left_and_right - internal_subtraction

theorem positive_difference_perimeters :
  let length := 4
  let width := 3
  perimeter_rectangle length width - perimeter_cross_shape = 6 :=
by
  let length := 4
  let width := 3
  sorry

end positive_difference_perimeters_l214_214757


namespace octahedron_cube_volume_ratio_l214_214323

theorem octahedron_cube_volume_ratio (x : ℝ) (hx : 0 < x) :
  let volume_cube := (2 * x) ^ 3,
      edge_octahedron := x * Real.sqrt 2,
      volume_octahedron := (edge_octahedron ^ 3 * Real.sqrt 2) / 3,
      ratio := volume_octahedron / volume_cube
  in ratio = 1 / 6 :=
sorry

end octahedron_cube_volume_ratio_l214_214323


namespace perimeter_of_framed_area_l214_214720

def height_of_painting : ℕ := 12
def width_of_painting : ℕ := 15
def frame_border_width : ℕ := 3

theorem perimeter_of_framed_area : 
  ∃ (perimeter : ℕ), perimeter = 
  2 * ((height_of_painting + 2 * frame_border_width) + (width_of_painting + 2 * frame_border_width)) := 
begin
  sorry
end

end perimeter_of_framed_area_l214_214720


namespace vector_dot_product_sum_l214_214494

noncomputable def points_in_plane (A B C : Type) (dist_AB dist_BC dist_CA : ℝ) : Prop :=
  dist_AB = 3 ∧ dist_BC = 5 ∧ dist_CA = 6

theorem vector_dot_product_sum (A B C : Type) (dist_AB dist_BC dist_CA : ℝ) (HA : points_in_plane A B C dist_AB dist_BC dist_CA) :
    ∃ (AB BC CA : ℝ), AB * BC + BC * CA + CA * AB = -35 :=
by
  sorry

end vector_dot_product_sum_l214_214494


namespace probability_blue_ball_higher_l214_214666

theorem probability_blue_ball_higher :
  let bins : ℕ → ℚ := λ k, 3^(-k) in
  let total_prob := ∑' k, bins k * bins k in
  let same_prob := total_prob in
  let diff_prob := (1 - same_prob) / 2 in
  diff_prob = 7 / 16 :=
by
  sorry

end probability_blue_ball_higher_l214_214666


namespace electric_water_bill_ratio_l214_214364

-- Definitions of the given conditions
def earnings : ℝ := 6000
def house_rental : ℝ := 640
def food_expense : ℝ := 380
def insurance_cost : ℝ := earnings / 5
def remaining_amount : ℝ := 2280

-- The proof problem statement to show that the electric and water bill to earnings is 1/4
theorem electric_water_bill_ratio:
  let total_expenses := house_rental + food_expense + insurance_cost in
  let total_spent_on_bills := earnings - remaining_amount in
  let electric_water_bill := total_spent_on_bills - total_expenses in
  (electric_water_bill / earnings) = 1 / 4 :=
sorry

end electric_water_bill_ratio_l214_214364


namespace largest_whole_number_x_l214_214281

theorem largest_whole_number_x :
  ∃ x : ℕ, (∀ y : ℕ, (y > x -> 1/4 + y/9 ≥ 1)) ∧ 1/4 + x/9 < 1 ∧ ∀ y : ℕ, y ≤ x -> y = 6 :=
begin
  sorry
end

end largest_whole_number_x_l214_214281


namespace percentage_supports_policy_l214_214718

theorem percentage_supports_policy
    (men_support_percentage : ℝ)
    (women_support_percentage : ℝ)
    (num_men : ℕ)
    (num_women : ℕ)
    (total_surveyed : ℕ)
    (total_supporters : ℕ)
    (overall_percentage : ℝ) :
    (men_support_percentage = 0.70) →
    (women_support_percentage = 0.75) →
    (num_men = 200) →
    (num_women = 800) →
    (total_surveyed = num_men + num_women) →
    (total_supporters = (men_support_percentage * num_men) + (women_support_percentage * num_women)) →
    (overall_percentage = (total_supporters / total_surveyed) * 100) →
    overall_percentage = 74 :=
by
  intros
  sorry

end percentage_supports_policy_l214_214718


namespace arc_length_correct_l214_214740

noncomputable def arc_length_ch_x_plus_3 : ℝ :=
  ∫ x in 0..1, Real.cosh x

theorem arc_length_correct : arc_length_ch_x_plus_3 = Real.sinh 1 :=
by
  sorry

end arc_length_correct_l214_214740


namespace def_value_l214_214876

theorem def_value (a b c : ℕ) (abc def : ℕ) 
  (h1 : a = b + 1)
  (h2 : b = c + 2)
  (h3 : abc * 3 + 4 = def)
  (h4 : abc = 100 * a + 10 * b + c) : def = 964 :=
by {
  sorry
}

end def_value_l214_214876


namespace probability_even_sum_of_spins_l214_214274

theorem probability_even_sum_of_spins :
  let prob_even_first := 3 / 6
  let prob_odd_first := 3 / 6
  let prob_even_second := 2 / 5
  let prob_odd_second := 3 / 5
  let prob_both_even := prob_even_first * prob_even_second
  let prob_both_odd := prob_odd_first * prob_odd_second
  prob_both_even + prob_both_odd = 1 / 2 := 
by 
  sorry

end probability_even_sum_of_spins_l214_214274


namespace jacob_total_distance_l214_214555

/- Jacob jogs at a constant rate of 4 miles per hour.
   He jogs for 2 hours, then stops to take a rest for 30 minutes.
   After the break, he continues jogging for another 1 hour.
   Prove that the total distance jogged by Jacob is 12.0 miles.
-/
theorem jacob_total_distance :
  let joggingSpeed := 4 -- in miles per hour
  let jogBeforeBreak := 2 -- in hours
  let restDuration := 0.5 -- in hours (though it does not affect the distance)
  let jogAfterBreak := 1 -- in hours
  let totalDistance := joggingSpeed * jogBeforeBreak + joggingSpeed * jogAfterBreak
  totalDistance = 12.0 := 
by
  sorry

end jacob_total_distance_l214_214555


namespace max_sum_of_radii_in_prism_l214_214130

noncomputable def sum_of_radii (AB AD AA1 : ℝ) : ℝ :=
  let r (t : ℝ) := 2 - 2 * t
  let R (t : ℝ) := 3 * t / (1 + t)
  let f (t : ℝ) := R t + r t
  let t_max := 1 / 2
  f t_max

theorem max_sum_of_radii_in_prism :
  let AB := 5
  let AD := 3
  let AA1 := 4
  sum_of_radii AB AD AA1 = 21 / 10 := by
sorry

end max_sum_of_radii_in_prism_l214_214130


namespace six_cards_arrangement_count_l214_214224

def six_cards := {1, 2, 3, 4, 5, 6}

theorem six_cards_arrangement_count :
  (∃ (count : ℕ), count = 40) :=
sorry

end six_cards_arrangement_count_l214_214224


namespace tangent_line_slope_eq_k_l214_214521

noncomputable def tangent_slope_at_ln_x (x : ℝ) (hx : x > 0) : ℝ := by
  exact 1 / x

theorem tangent_line_slope_eq_k (k : ℝ) (x : ℝ) (hx : x > 0) (h_tangent : k = tangent_slope_at_ln_x x hx) : k = 1 / Real.exp(1) := by
  have : x = Real.exp(1) := by
    sorry -- proof that x = e
  rw [this] at h_tangent
  exact h_tangent
  sorry -- fill in the rest of the proof


end tangent_line_slope_eq_k_l214_214521


namespace n_to_the_4_plus_4_to_the_n_composite_l214_214607

theorem n_to_the_4_plus_4_to_the_n_composite (n : ℕ) (h : n ≥ 2) : ¬Prime (n^4 + 4^n) := 
sorry

end n_to_the_4_plus_4_to_the_n_composite_l214_214607


namespace find_p_from_binomial_distribution_l214_214580

theorem find_p_from_binomial_distribution (p : ℝ) (h₁ : 0 ≤ p ∧ p ≤ 1) 
    (h₂ : ∀ n k : ℕ, k ≤ n → 0 ≤ p^(k:ℝ) * (1-p)^((n-k):ℝ)) 
    (h₃ : (1 - (1 - p)^2 = 5 / 9)) : p = 1 / 3 :=
by sorry

end find_p_from_binomial_distribution_l214_214580


namespace midpoint_probability_l214_214164

def is_valid_midpoint (x1 y1 z1 x2 y2 z2 : ℕ) (x_mid y_mid z_mid : ℕ) : Prop :=
  (x_mid = (x1 + x2) / 2) ∧ (y_mid = (y1 + y2) / 2) ∧ (z_mid = (z1 + z2) / 2)

def midpoint_in_S (x1 y1 z1 x2 y2 z2 : ℕ) : Prop :=
  0 <= (x1 + x2) / 2 ∧ (x1 + x2) / 2 <= 3 ∧
  0 <= (y1 + y2) / 2 ∧ (y1 + y2) / 2 <= 4 ∧
  0 <= (z1 + z2) / 2 ∧ ( (z1 + z2) / 2 <= 5

theorem midpoint_probability (m n : ℕ) 
  (h_rel_prime: Nat.gcd m n = 1) :
  let S := {p : ℕ × ℕ × ℕ | 0 <= p.1 ∧ p.1 <= 3 ∧ 0 <= p.2 ∧ p.2 <= 4 ∧ 0 <= (p.2 + p.3) ∧ (p.2 + p.3) <= 5} in
  ∃ (p1 p2 : ℕ × ℕ × ℕ) (ne_p1_p2 : p1 ≠ p2), midpoint_in_S p1.1 p1.2 p1.3 p2.1 p2.2 p2.3
  <-> (m = 24 ∧ n = 119 ∧ m + n = 143) :=
by
  sorry

end midpoint_probability_l214_214164


namespace barney_average_speed_l214_214735

-- Definitions
def initial_odometer := 2332
def final_odometer := 2772
def time_first_day := 5
def time_second_day := 7

-- Average speed calculation
def average_speed (initial_odometer final_odometer : ℕ) (time_first_day time_second_day : ℕ) : ℝ :=
  (final_odometer - initial_odometer) / (time_first_day + time_second_day : ℝ)

-- Theorem statement
theorem barney_average_speed :
  average_speed initial_odometer final_odometer time_first_day time_second_day = 36 := by
  sorry

end barney_average_speed_l214_214735


namespace max_area_circle_inscribed_l214_214235

noncomputable def max_area_inscribed_circle
  (θ : ℝ) (r : ℝ) (h1 : 0 < θ ∧ θ < π / 2) : ℝ :=
  let r1 := r * sin θ / (1 + sin θ) in
  let r2 := r1 * (1 - sin θ) / (1 + sin θ) in
  r2^2 * π / r^2

theorem max_area_circle_inscribed
  (θ : ℝ) (r : ℝ) (h1 : 0 < θ ∧ θ < π / 2)
  (h2 : sin θ = 1 / 3) : 
  max_area_inscribed_circle θ r h1 = r^2 * π / 64 := by
  sorry

end max_area_circle_inscribed_l214_214235


namespace event_D_is_random_l214_214285

-- Definitions of the events
def event_A : Prop := "It rains heavily without clouds in the sky."
def event_B : Prop := "Like charges repel each other."
def event_C : Prop := "Seeds germinate without water."
def event_D : Prop := "Drawing one card from ten cards numbered 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, and getting the card numbered 1."

-- Definition of a random event
def random_event (e : Prop) : Prop :=
  ¬(∀ (cond : Prop), cond → e) ∧ ¬(∀ (cond : Prop), cond → ¬e)

-- Prove that event D is a random event
theorem event_D_is_random : random_event event_D :=
by
  sorry

end event_D_is_random_l214_214285


namespace convex_polygon_number_of_sides_l214_214709

theorem convex_polygon_number_of_sides :
  ∀ (n : ℕ) (α : ℝ),
  (convex_polygon n) →
  (α = (2 / 3) * Real.pi) →
  (∀ k, (k ≥ 1 ∧ k ≤ n) → 
    (interior_angle k = α + (k - 1) * (Real.pi / 36))) →
  n = 9 := by
  sorry

end convex_polygon_number_of_sides_l214_214709


namespace log_base_3_l214_214008

theorem log_base_3 (a : ℝ) (h1 : a = 9) (h2 : ∀ (b : ℝ) (n : ℤ), log b (b ^ n) = n) : 
  log 3 (9 ^ 3) = 6 := 
by
  have h3 : 9 = 3^2, from by norm_num
  rw [h3] at h1
  rw [h1]
  have h4 : (3^2)^3 = 3^6, from pow_mul 3 2 3
  rw [h4]
  exact h2 3 6

end log_base_3_l214_214008


namespace num_divisors_of_64m4_l214_214037

-- A positive integer m such that 120 * m^3 has 120 divisors
def has_120_divisors (m : ℕ) : Prop := (m > 0) ∧ ((List.range (120 * m^3 + 1)).filter (λ d, (120 * m^3) % d = 0)).length = 120

-- Prove that if such an m exists, then 64 * m^4 has 675 divisors
theorem num_divisors_of_64m4 (m : ℕ) (h : has_120_divisors m) : ((List.range (64 * m^4 + 1)).filter (λ d, (64 * m^4) % d = 0)).length = 675 :=
by
  sorry

end num_divisors_of_64m4_l214_214037


namespace solutions_g_g_x_equals_5_l214_214174

def g (x : ℝ) : ℝ :=
if x ≤ 1 then -x + 4 else 3 * x - 6

theorem solutions_g_g_x_equals_5 : set.count { x : ℝ | g (g x) = 5 } = 2 :=
sorry

end solutions_g_g_x_equals_5_l214_214174


namespace distance_from_mo_l214_214549

-- Definitions based on conditions
-- 1. Grid squares have side length 1 cm.
-- 2. Shape shaded gray on the grid.
-- 3. The total shaded area needs to be divided into two equal parts.
-- 4. The line to be drawn is parallel to line MO.

noncomputable def grid_side_length : ℝ := 1.0
noncomputable def shaded_area : ℝ := 10.0
noncomputable def line_mo_distance (d : ℝ) : Prop := 
  ∃ parallel_line_distance, parallel_line_distance = d ∧ 
    ∃ equal_area, 2 * equal_area = shaded_area ∧ equal_area = 5.0

-- Theorem: The parallel line should be drawn at 2.6 cm 
theorem distance_from_mo (d : ℝ) : 
  d = 2.6 ↔ line_mo_distance d := 
by
  sorry

end distance_from_mo_l214_214549


namespace petyas_number_l214_214202

theorem petyas_number :
  ∃ (N : ℕ), 
  (N % 2 = 1 ∧ ∃ (M : ℕ), N = 149 * M ∧ (M = Nat.mod (N : ℕ) (100))) →
  (N = 745 ∨ N = 3725) :=
by
  sorry

end petyas_number_l214_214202


namespace probability_of_prime_or_square_l214_214325

def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 7 ∨ n = 11

def is_square (n : ℕ) : Prop := n = 4 ∨ n = 9

def spinner_numbers : List ℕ := [2, 3, 4, 7, 8, 9, 10, 11]

def favorable_numbers (numbers : List ℕ) : List ℕ :=
  numbers.filter (fun n => is_prime n ∨ is_square n)

def number_of_sections := 8

def probability_favorable (favorable : List ℕ) (total_sections : ℕ) : ℚ :=
  favorable.length / total_sections

theorem probability_of_prime_or_square :
  probability_favorable (favorable_numbers spinner_numbers) number_of_sections = 3 / 4 := 
sorry

end probability_of_prime_or_square_l214_214325


namespace new_price_after_reduction_l214_214126

-- Define the conditions
def initial_price_per_kg : ℝ := 60
def increase_factor_apple : ℝ := 1.5
def increase_factor_revenue : ℝ := 1.125

-- Define the amounts and revenues
variables (a : ℝ) -- the amount of apples sold on the first day
def first_day_revenue := initial_price_per_kg * a
def second_day_revenue := increase_factor_revenue * first_day_revenue
def apples_sold_second_day := increase_factor_apple * a

-- Define the new price per kg we need to prove
def new_price_per_kg := 
  (second_day_revenue = new_price_per_kg * apples_sold_second_day)

theorem new_price_after_reduction : 
  new_price_per_kg = 45 :=
by
  -- According to conditions and correct answer
  sorry

end new_price_after_reduction_l214_214126


namespace factor_expression_l214_214374

theorem factor_expression (a b c : ℝ) :
  let num := (a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3
  let denom := (a - b)^3 + (b - c)^3 + (c - a)^3
  (denom ≠ 0) →
  num / denom = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end factor_expression_l214_214374


namespace find_a_inequality_proof_l214_214482

-- Question 1: Prove that a = 1
theorem find_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = (x+1) * log x - a * (x - 1)) :
  (∃ x0 y0, f'(x0) = 1 ∧ f x0 = y0 ∧ y0 = x0 - 1) → a = 1 :=
by
  sorry

-- Question 2: Prove the inequality given 1 < x < 2
theorem inequality_proof (x : ℝ) :
  1 < x ∧ x < 2 → (1 / log x - 1 / log (x - 1) < 1 / ((x - 1) * (2 - x))) :=
by
  sorry

end find_a_inequality_proof_l214_214482


namespace find_isosceles_triangle_l214_214479

variables {M O N A B X Y Z : Type}
variables [metric_space X] [metric_space Y] [metric_space Z]
variables (OM : set X) (ON : set X) (angleMON : ℝ)
variables (XA : set X) (XB : set X)
variables (intersection_XA_ON : ∃ Y, Y ∈ XA ∩ ON)
variables (intersection_XB_ON : ∃ Z, Z ∈ XB ∩ ON)
variables (X_on_OM : X ∈ OM)

theorem find_isosceles_triangle 
  (h1 : Y = classical.some intersection_XA_ON)
  (h2 : Z = classical.some intersection_XB_ON)
  (isosceles_triangle : dist X Y = dist X Z) :
  ∃ X ∈ OM, ∃ Y Z, Y ∈ XA ∩ ON ∧ Z ∈ XB ∩ ON ∧ dist X Y = dist X Z := 
sorry

end find_isosceles_triangle_l214_214479


namespace probability_angle_AMB_acute_l214_214131

theorem probability_angle_AMB_acute :
  let side_length := 4
  let square_area := side_length * side_length
  let semicircle_area := (1 / 2) * Real.pi * (side_length / 2) ^ 2
  let probability := 1 - semicircle_area / square_area
  probability = 1 - (Real.pi / 8) :=
sorry

end probability_angle_AMB_acute_l214_214131


namespace polynomial_irreducible_l214_214177

theorem polynomial_irreducible 
  (n : ℕ) 
  (a : fin n → ℤ)
  (p : ℤ) 
  [fact (prime p)] 
  (h : ∀ i, 0 ≤ i ∧ i < n → ∑ k, |a k| < p) 
  (an_ne_zero : a n ≠ 0)
  (f : polynomial ℤ) :
  f = polynomial.sum (λ (k : ℕ), monomial k (a k)) + p * monomial 0 1 →
  irreducible f :=
sorry

end polynomial_irreducible_l214_214177


namespace sum_sqrtk_eq_sum_logk_l214_214790

theorem sum_sqrtk_eq_sum_logk (n : ℕ) (hn: n ≥ 2) : 
  (∑ k in Finset.range (n + 1), Nat.floor (real.rpow (n : ℝ) (1 / k))) = 
  (∑ k in Finset.range (n + 1), Nat.floor ((real.log (n : ℝ)) / (real.log (k : ℝ)))) := 
sorry

end sum_sqrtk_eq_sum_logk_l214_214790


namespace compute_y_value_l214_214754

theorem compute_y_value : 
  (∑' n : ℕ, (1 / 3)^n) * (∑' n : ℕ, (-1 / 3)^n) = ∑' n : ℕ, (1 / (9 : ℝ))^n := 
by 
  sorry

end compute_y_value_l214_214754


namespace cost_of_fencing_l214_214690

noncomputable theory

-- Definitions of given conditions
def ratio : ℚ := 3 / 2
def area : ℚ := 4704
def cost_per_meter_paise : ℚ := 50
def cost_per_meter_rupees : ℚ := cost_per_meter_paise / 100

-- Proof goal: Cost of fencing the park is 140 rupees
theorem cost_of_fencing : 
  (3 * (sqrt (4704 / 6)) * 2 * (sqrt (4704 / 6)) * 2 * 0.5) = 140 :=
by
  sorry

end cost_of_fencing_l214_214690


namespace journey_total_distance_l214_214727

theorem journey_total_distance (D : ℝ) 
  (h1 : (D / 3) / 21 + (D / 3) / 14 + (D / 3) / 6 = 12) : 
  D = 126 :=
sorry

end journey_total_distance_l214_214727


namespace reimbursement_correct_l214_214305

def total_medical_expenses (a b c : ℝ) : ℝ := a + b + c
def couple_payment (total_cost : ℝ) : ℝ := 0.60 * total_cost
def deductions_from_salary (cost_of_medical: ℝ) (salary: ℝ) (tax_rate: ℝ) (living_costs : ℝ) : ℝ :=
  (0.40 * cost_of_medical) + (tax_rate * salary) + living_costs
def excess_payment (total_deductions: ℝ) (salary: ℝ) : ℝ := total_deductions - salary
def total_couple_payment (payment: ℝ) (excess: ℝ) : ℝ := payment + excess
def each_person_share (total_payment : ℝ) : ℝ := total_payment / 2
def reimbursement (husband_paid: ℝ) (each_share : ℝ) : ℝ := husband_paid - each_share

theorem reimbursement_correct : 
  let a := 128
  let b := 256
  let c := 64
  let salary := 160
  let tax_rate := 0.05
  let living_costs := 25
  let husband_paid := couple_payment (total_medical_expenses a b c)
in
  reimbursement husband_paid (each_person_share (total_couple_payment (couple_payment (total_medical_expenses a b c)) (excess_payment (deductions_from_salary (total_medical_expenses a b c) salary tax_rate living_costs) salary))) = 108.3 :=
by
  sorry

end reimbursement_correct_l214_214305


namespace digit_one_appears_more_frequently_than_digit_five_l214_214192

theorem digit_one_appears_more_frequently_than_digit_five :
  ∀ (n : ℕ) (hn : n = 1000000000) 
  (f : ℕ → ℕ),
  (∀ k, f k = if k < 10 then k else f (Nat.digits 10 k).sum) →
  (let final_digits := List.map f (List.range n) in 
   final_digits.count 1 > final_digits.count 5) :=
  
  sorry

end digit_one_appears_more_frequently_than_digit_five_l214_214192


namespace sum_of_possible_jk_values_l214_214141

theorem sum_of_possible_jk_values :
  ∀ (j k : ℕ), (0 < j ∧ 0 < k) ∧ ((1 / j : ℝ) + (1 / k) = 1 / 4) →
  ∑ (S : finset ℕ), S = {25, 18, 16} := 59 :=
by
  sorry

end sum_of_possible_jk_values_l214_214141


namespace a_2013_value_l214_214358

-- Define the sequence {a_n} according to the problem statement
def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  a 2 = 3 ∧
  (∀ n > 2, a (n) = ((λ b, nat.base_repr b (n-1)) (a (n-1)) |> (λ c, string.to_nat c n)) + 2)

-- Define the specific term a_2013
def a_2013 : ℕ := 23097

-- Prove that a_2013 is the 2013th term in the sequence
theorem a_2013_value (a : ℕ → ℕ) (h : sequence a) : a 2013 = 23097 := 
by
  sorry -- proof skipped

end a_2013_value_l214_214358


namespace heesu_has_greatest_sum_l214_214226

def sum_cards (cards : List Int) : Int :=
  cards.foldl (· + ·) 0

theorem heesu_has_greatest_sum :
  let sora_cards := [4, 6]
  let heesu_cards := [7, 5]
  let jiyeon_cards := [3, 8]
  sum_cards heesu_cards > sum_cards sora_cards ∧ sum_cards heesu_cards > sum_cards jiyeon_cards :=
by
  let sora_cards := [4, 6]
  let heesu_cards := [7, 5]
  let jiyeon_cards := [3, 8]
  sorry

end heesu_has_greatest_sum_l214_214226


namespace graph_passes_through_point_l214_214040

theorem graph_passes_through_point : ∀ (a : ℝ), a > 0 ∧ a ≠ 1 → (∃ x y, (x, y) = (0, 2) ∧ y = a^x + 1) :=
by
  intros a ha
  use 0
  use 2
  obtain ⟨ha1, ha2⟩ := ha
  have h : a^0 = 1 := by simp
  simp [h]
  sorry

end graph_passes_through_point_l214_214040


namespace savings_after_20_days_l214_214095

-- Definitions based on conditions
def daily_earnings : ℕ := 80
def days_worked : ℕ := 20
def total_spent : ℕ := 1360

-- Prove the savings after 20 days
theorem savings_after_20_days : daily_earnings * days_worked - total_spent = 240 :=
by
  sorry

end savings_after_20_days_l214_214095


namespace complement_B_l214_214091

def A : Set ℤ := {-1, 0, 1, 2, 3}

def B : Set ℕ := {x | -2 < (x : ℤ) ∧ x ≤ 2}

def complement (A B : Set ℤ) : Set ℤ := {x | x ∈ A ∧ x ∉ B}

theorem complement_B :
  complement A B = {-1, 3} :=
sorry

end complement_B_l214_214091


namespace satellite_orbits_in_one_week_l214_214722

theorem satellite_orbits_in_one_week :
  let hours_per_day := 24
  let days_per_week := 7
  let hours_per_week := days_per_week * hours_per_day
  let hours_per_orbit := 7 in
  hours_per_week / hours_per_orbit = 24 := by
  sorry

end satellite_orbits_in_one_week_l214_214722


namespace prop_one_correct_l214_214472

variables {m n : Type} {α β : Type}
variables [line m] [line n] [plane α] [plane β]

-- Hypotheses
variables (hmn_diff : m ≠ n)
variables (halphabeta_diff : α ≠ β)

-- The proposition to be proved
theorem prop_one_correct (h1 : parallel m n) (h2 : perpendicular m β) : perpendicular n β :=
sorry

end prop_one_correct_l214_214472


namespace factor_fraction_l214_214381

/- Definitions based on conditions -/
variables {a b c : ℝ}

theorem factor_fraction :
  ( (a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3 ) / 
  ((a - b)^3 + (b - c)^3 + (c - a)^3) = 
  (a^2 + ab + b^2) * (b^2 + bc + c^2) * (c^2 + ca + a^2) :=
begin
  sorry
end

end factor_fraction_l214_214381


namespace area_of_square_l214_214767

namespace SquareArea

-- Define the vertices of the square
def P := (1 : ℤ, 2 : ℤ)
def Q := (-4 : ℤ, 3 : ℤ)
def R := (-3 : ℤ, -2 : ℤ)
def S := (2 : ℤ, -3 : ℤ)

-- Define the function to compute the squared distance between two points
def squared_dist (A B : ℤ × ℤ) : ℤ :=
  (B.1 - A.1)^2 + (B.2 - A.2)^2

-- Define the area of a square using the squared length of its side
def square_area (A B : ℤ × ℤ) : ℤ :=
  squared_dist A B

-- The theorem that needs to be proved
theorem area_of_square : 
  square_area P Q = 26 := by
  sorry
end SquareArea

end area_of_square_l214_214767


namespace youngest_child_age_is_5_l214_214127

theorem youngest_child_age_is_5 :
  ∃ age, 
  (5 = age) ∧ 
  (∀ (k ∈ {2, 6, 8, 12, 14}), Prime (age + k)) ∧ 
  Prime age :=
by
  sorry

end youngest_child_age_is_5_l214_214127


namespace cost_of_five_dozen_apples_l214_214588

-- Define the initial conditions
def totalCostThreeDozen : ℝ := 23.40
def numberOfDozensPurchased : ℕ := 3
def numberOfDozensToPurchase : ℕ := 5

-- Define the rate per dozen calculation
def ratePerDozen (totalCost : ℝ) (dozens : ℕ) : ℝ := totalCost / dozens

-- Define the total cost calculation
def totalCost (rate : ℝ) (dozens : ℕ) : ℝ := rate * dozens

-- Prop statement to be proven
theorem cost_of_five_dozen_apples :
  totalCost (ratePerDozen totalCostThreeDozen numberOfDozensPurchased) numberOfDozensToPurchase = 39.00 := sorry

end cost_of_five_dozen_apples_l214_214588


namespace toms_total_profit_l214_214922

def total_earnings_mowing : ℕ := 4 * 12 + 3 * 15 + 1 * 20
def total_earnings_side_jobs : ℕ := 2 * 10 + 3 * 8 + 1 * 12
def total_earnings : ℕ := total_earnings_mowing + total_earnings_side_jobs
def total_expenses : ℕ := 17 + 5
def total_profit : ℕ := total_earnings - total_expenses

theorem toms_total_profit : total_profit = 147 := by
  -- Proof omitted
  sorry

end toms_total_profit_l214_214922


namespace sqrt_18_mul_sqrt_32_eq_24_l214_214215
  
theorem sqrt_18_mul_sqrt_32_eq_24 : (Real.sqrt 18 * Real.sqrt 32 = 24) :=
  sorry

end sqrt_18_mul_sqrt_32_eq_24_l214_214215


namespace phi_eq_c_implies_cone_l214_214539

-- Definitions based on the given conditions:
def φ_polar_angle {r θ φ : ℝ} : Prop := φ = Real.atan2 (sqrt (r^2 * (cos θ)^2 + r^2 * (sin θ)^2)) r

def shape_described_by_phi_eq_c {c : ℝ} : Prop :=
  ∀ r θ φ, φ = c → φ_polar_angle → geometry.shape (r, θ, φ) = geometry.cone

theorem phi_eq_c_implies_cone {c : ℝ} (h : ∀ r θ φ, φ = c → φ_polar_angle): 
  shape_described_by_phi_eq_c :=
by
  intros,
  sorry

end phi_eq_c_implies_cone_l214_214539


namespace sum_log_base_5_divisors_l214_214651

theorem sum_log_base_5_divisors (n : ℕ) (h : n * (n + 1) / 2 = 264) : n = 23 :=
by
  sorry

end sum_log_base_5_divisors_l214_214651


namespace maisy_earnings_increase_l214_214949

-- Define the conditions from the problem
def current_job_hours_per_week : ℕ := 8
def current_job_wage_per_hour : ℕ := 10

def new_job_hours_per_week : ℕ := 4
def new_job_wage_per_hour : ℕ := 15
def new_job_bonus_per_week : ℕ := 35

-- Define the weekly earnings calculations
def current_job_earnings : ℕ := current_job_hours_per_week * current_job_wage_per_hour
def new_job_earnings_without_bonus : ℕ := new_job_hours_per_week * new_job_wage_per_hour
def new_job_earnings_with_bonus : ℕ := new_job_earnings_without_bonus + new_job_bonus_per_week

-- Define the difference in earnings
def earnings_difference : ℕ := new_job_earnings_with_bonus - current_job_earnings

-- The theorem to prove: Maisy will earn $15 more per week at her new job
theorem maisy_earnings_increase : earnings_difference = 15 := by
  sorry

end maisy_earnings_increase_l214_214949


namespace general_term_formula_l214_214143

noncomputable def a_sequence : ℕ → ℤ
| 0       := 1
| (n + 1) := 2 * a_sequence n + 3

theorem general_term_formula (n : ℕ) : a_sequence n = 2^(n+1) - 3
:= sorry

end general_term_formula_l214_214143


namespace find_lambda_l214_214852

-- conditions
variables (e1 e2 : ℝ → Prop) (a b : ℝ → Prop)
variable λ : ℝ
variable angle1 : ℝ

-- definitions
def unit_vector (v : ℝ → Prop) : Prop := v = sorry  -- Defining the unit vectors
def is_perpendicular (v w : ℝ → Prop) : Prop := ∀ x, (v x) * (w x) = 0

-- actual question and conditionals as Lean statement
theorem find_lambda (h1 : unit_vector e1) (h2 : unit_vector e2) 
  (h3 : a = (λ x, 2 * e1 x + e2 x)) 
  (h4 : b = (λ x, e1 x - λ * e2 x)) 
  (h5 : angle1 = π/3)
  (h6 : is_perpendicular a b) : 
  λ = 5/4 :=
sorry

end find_lambda_l214_214852


namespace translated_parabola_eq_new_equation_l214_214245

-- Definitions following directly from the condition
def original_parabola (x : ℝ) : ℝ := 2 * x^2
def new_vertex : (ℝ × ℝ) := (-2, -2)
def new_parabola (x : ℝ) : ℝ := 2 * (x + 2)^2 - 2

-- Statement to prove the equivalency of the translated parabola equation
theorem translated_parabola_eq_new_equation :
  (∀ (x : ℝ), (original_parabola x = new_parabola (x - 2))) :=
by
  sorry

end translated_parabola_eq_new_equation_l214_214245


namespace minimum_total_cost_l214_214356

def total_cost (x : ℝ) : ℝ := 360 * x + 5760 / x + 1120

theorem minimum_total_cost : ∃ x > 0, total_cost x = 4000 :=
by
  sorry

end minimum_total_cost_l214_214356


namespace max_product_value_l214_214230

variables {R : Type*} [linear_ordered_field R]

-- Define functions f and g with given ranges
def f (x : R) : R := sorry
def g (x : R) : R := sorry

-- Conditions in the problem defined in Lean
axiom f_range : ∀ x, -7 ≤ f(x) ∧ f(x) ≤ 4
axiom g_range : ∀ x, 0 ≤ g(x) ∧ g(x) ≤ 2

-- Statement to prove: The maximum value of f(x) * g(x) is 14
theorem max_product_value : ∃ b, b = 14 ∧ ∀ x, f(x) * g(x) ≤ b :=
sorry

end max_product_value_l214_214230


namespace ear_muffs_total_l214_214341

theorem ear_muffs_total (a b : ℕ) (h1 : a = 1346) (h2 : b = 6444) : a + b = 7790 :=
by
  sorry

end ear_muffs_total_l214_214341


namespace hexagon_area_with_apothem_eq_l214_214058

noncomputable def area_of_hexagon_formed_by_midpoints (a : ℝ) (s : ℝ) : ℝ :=
  let si := s / (2 * (1 + Real.sqrt 2))
  let hex_side := si / 2
  (3 * Real.sqrt 3 / 2) * (hex_side ^ 2)

theorem hexagon_area_with_apothem_eq :
  ∀ (s : ℝ), s = 6 * (1 - Real.sqrt 2) → 
  area_of_hexagon_formed_by_midpoints 3 s = 27 * Real.sqrt 3 * (3 - 2 * Real.sqrt 2) :=
begin
  intros s hs,
  rw [area_of_hexagon_formed_by_midpoints, hs],
  sorry
end

end hexagon_area_with_apothem_eq_l214_214058


namespace at_least_one_gt_one_l214_214450

theorem at_least_one_gt_one (x y : ℝ) (h : x + y > 2) : ¬(x > 1 ∨ y > 1) → (x ≤ 1 ∧ y ≤ 1) := 
by
  sorry

end at_least_one_gt_one_l214_214450


namespace prime_divides_a_and_b_l214_214815

theorem prime_divides_a_and_b
  (p : ℕ) [hp : Fact (Nat.prime p)] (h1 : p % 4 = 3) (a b : ℕ)
  (h2 : (a^2 + b^2) % p = 0) :
  p ∣ a ∧ p ∣ b := by
  sorry

end prime_divides_a_and_b_l214_214815


namespace intersection_M_N_eq_2_4_l214_214930

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℕ := {x | ∃ y, y = Real.log (6 - x) ∧ x < 6}

theorem intersection_M_N_eq_2_4 : M ∩ N = {2, 4} :=
by sorry

end intersection_M_N_eq_2_4_l214_214930


namespace sum_of_inradii_constant_l214_214731

variables {n : ℕ} (P : Type)
variables (O : P) (R : ℝ) (triangles : Fin (n-2) → Type)
variables (r : Fin (n-2) → ℝ)
variables (C : ℝ)

-- Hypothesis stating the conditions
hypothesis (inscribed_ngon : ∃ (O : P) (R : ℝ), True)  -- P is an inscribed n-gon in circle with center O and radius R
hypothesis (triangulation : ∀ i : Fin (n-2), ∃ (T_i : Type), True)  -- Triangulation results in n-2 non-overlapping triangles
hypothesis (inradius : ∀ i : Fin (n-2), ∃ (r_i : ℝ), True)  -- Each triangle has an inscribed circle with radius r_i

theorem sum_of_inradii_constant :
  ∃ C : ℝ, (∑ i : Fin (n-2), r i) = C :=
sorry

end sum_of_inradii_constant_l214_214731


namespace probability_XOXOXOX_is_one_over_thirty_five_l214_214441

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_arrangement : ℤ :=
  let total_arrangements := binomial_coefficient 7 4
  let favorable_outcomes := 1
  favorable_outcomes / total_arrangements

theorem probability_XOXOXOX_is_one_over_thirty_five :
  probability_of_arrangement = (1 : ℤ) / 35 := 
  by
  sorry

end probability_XOXOXOX_is_one_over_thirty_five_l214_214441


namespace cos_beta_half_l214_214072

variables (α β : ℝ)
variables (hα1 : 0 < α ∧ α < π / 2)
variables (hβ1 : 0 < β ∧ β < π / 2)
variables (hcosα : real.cos α = 1 / 7)
variables (hsinαβ : real.sin (α + β) = 5 * real.sqrt 3 / 14)

theorem cos_beta_half : real.cos β = 1 / 2 :=
sorry

end cos_beta_half_l214_214072


namespace max_radius_of_sphere_in_cones_l214_214667

open Real

theorem max_radius_of_sphere_in_cones :
  ∃ (r : ℝ), (0 < r) ∧ (r = 40 / 13) ∧ 
             ∀ (cone1 cone2 : Cone), is_congruent_cone cone1 cone2 ∧
                                      cone1.base_radius = 5 ∧ cone1.height = 12 ∧
                                      cone2.base_radius = 5 ∧ cone2.height = 12 ∧
                                      axes_intersect_at_right_angles cone1 cone2 4 → 
                                      sphere_within_both_cones cone1 cone2 r :=
by
  sorry

end max_radius_of_sphere_in_cones_l214_214667


namespace find_coordinates_of_B_l214_214138

theorem find_coordinates_of_B (A : ℝ × ℝ) (hx : A.1 = -4) (hy : A.2 = 3)
  (AB_parallel_y_axis : ∀ B : ℝ × ℝ, B.1 = A.1)
  (AB_length : ∀ B : ℝ × ℝ, abs (B.2 - A.2) = 5) :
  ∃ B : ℝ × ℝ, (B = (-4, 8) ∨ B = (-4, -2)) :=
by
  have Bx := AB_parallel_y_axis (A.1, A.2 + 5)
  have By := AB_length (A.1, A.2 + 5)
  constructor
  sorry

end find_coordinates_of_B_l214_214138


namespace combined_total_time_l214_214158

theorem combined_total_time
  (Katherine_time : Real := 20)
  (Naomi_time : Real := Katherine_time * (1 + 1 / 4))
  (Lucas_time : Real := Katherine_time * (1 + 1 / 3))
  (Isabella_time : Real := Katherine_time * (1 + 1 / 2))
  (Naomi_total : Real := Naomi_time * 10)
  (Lucas_total : Real := Lucas_time * 10)
  (Isabella_total : Real := Isabella_time * 10) :
  Naomi_total + Lucas_total + Isabella_total = 816.7 := sorry

end combined_total_time_l214_214158


namespace cannot_be_external_diagonals_of_prism_l214_214681

theorem cannot_be_external_diagonals_of_prism :
  ∀ d₁ d₂ d₃ ∈ ({5, 6, 8}, {5, 6, 9}, {6, 7, 9}, {6, 8, 10}, {7, 8, 11} : Finset (Finset ℕ)), 
  (d₁^2 + d₂^2 < d₃^2) ∨ (d₁^2 + d₃^2 < d₂^2) ∨ (d₂^2 + d₃^2 < d₁^2) :=
begin
  intros d₁ d₂ d₃ h,
  fin_cases h, 
  { cases h; try { cases h; try {norm_num at h} ; try {linarith} } },
  sorry
end

end cannot_be_external_diagonals_of_prism_l214_214681


namespace cos_of_angle_l214_214873

variable (A : ℝ) 

-- Condition: Given tan A + sec A = 3
axiom tan_add_sec_eq_3 : tan A + sec A = 3

-- Task: Prove that the only valid cos A = 3/5.
theorem cos_of_angle (h : tan A + sec A = 3) : cos A = 3/5 :=
sorry

end cos_of_angle_l214_214873


namespace closest_point_on_parabola_to_line_l214_214621

theorem closest_point_on_parabola_to_line :
  ∃ (x : ℝ), (∀ y, y = x^2) ∧ (d * ((2 * x) - y - 4) / sqrt 5) :=
sorry

end closest_point_on_parabola_to_line_l214_214621


namespace person_cannot_pass_through_gap_l214_214324

-- Define the constants used in the problem
def initial_rope_length (R : ℝ) : ℝ := 2 * Real.pi * R
def new_rope_length (R : ℝ) : ℝ := initial_rope_length R + 1 / 100 -- 1 cm

-- Define the height gap function
def gap_height (R : ℝ) : ℝ := (new_rope_length R - initial_rope_length R) / (2 * Real.pi)

-- The minimal height for a human to pass through a gap, let's assume it's 150 cm (an arbitrary assumption for the height of a human)
def min_height_for_human : ℝ := 150

-- The statement to be proved:
theorem person_cannot_pass_through_gap (R : ℝ) : gap_height R < min_height_for_human :=
by 
    sorry

end person_cannot_pass_through_gap_l214_214324


namespace choices_of_N_l214_214738

def base7_representation (N : ℕ) : ℕ := 
  (N / 49) * 100 + ((N % 49) / 7) * 10 + (N % 7)

def base8_representation (N : ℕ) : ℕ := 
  (N / 64) * 100 + ((N % 64) / 8) * 10 + (N % 8)

theorem choices_of_N : 
  ∃ (N_set : Finset ℕ), 
    (∀ N ∈ N_set, 100 ≤ N ∧ N < 1000 ∧ 
      ((base7_representation N * base8_representation N) % 100 = (3 * N) % 100)) 
    ∧ N_set.card = 15 :=
by
  sorry

end choices_of_N_l214_214738


namespace tangent_length_external_tangent_length_internal_l214_214997

theorem tangent_length_external
  (R r a : ℝ) (h : R > r) (AB : ℝ = a) :
  ∃ BM : ℝ, BM = a * real.sqrt(1 + r / R) := 
sorry

theorem tangent_length_internal
  (R r a : ℝ) (h : R > r) (AB : ℝ = a) :
  ∃ BM : ℝ, BM = a * real.sqrt(1 - r / R) := 
sorry

end tangent_length_external_tangent_length_internal_l214_214997


namespace range_of_a_l214_214085

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2^x - a ≥ 0) ↔ (a ≤ 0) :=
by
  sorry

end range_of_a_l214_214085


namespace probability_rounded_sum_eq_4_l214_214617

-- Declare the non-negative real number x and the split condition
variable {x : ℝ} (hx : 0 ≤ x ∧ x ≤ 3.5)

-- Define the rounding intervals for x
def round_to_nearest (y : ℝ) : ℤ :=
  if y < 0.5 then 0
  else if y < 1.5 then 1
  else if y < 2.5 then 2
  else 3

-- The proof statement
theorem probability_rounded_sum_eq_4 :
  (measure {x : ℝ | 0.5 ≤ x ∧ x < 1.5} + measure {x : ℝ | 1.5 ≤ x ∧ x < 2.5} + measure {x : ℝ | 2.5 ≤ x ∧ x ≤ 3.5}) / measure {x : ℝ | 0 ≤ x ∧ x ≤ 3.5} = 6 / 7 :=
sorry

end probability_rounded_sum_eq_4_l214_214617


namespace max_stamps_9_cities_l214_214662

-- Define a function to represent the stamp counting problem
def max_stamps (cities : ℕ) (connections : cities → cities → ℕ) : ℕ :=
  let num_letters := cities * (cities - 1)
  let worst_case_scenario := 2  -- Considering worst case in minimal spanning route scenario multiplied
  num_letters * worst_case_scenario

-- Lean 4 statement to prove the solution
theorem max_stamps_9_cities : max_stamps 9 (λ _ _, 1) = 240 := 
by 
  -- Given there are 9 cities and the worst-case number of stamps is found to be 240
  let expected := 240
  have calculation_num_letters : 9 * (9 - 1) = 72 := by simp
  have ultimate_bound : 72 * 2 = expected := by norm_num
  exact ultimate_bound

end max_stamps_9_cities_l214_214662


namespace num_positive_multiples_of_6_ending_in_4_lt_500_l214_214503

theorem num_positive_multiples_of_6_ending_in_4_lt_500 : 
  {n : ℕ // 6 * n < 500 ∧ (6 * n) % 10 = 4} .card = 8 := 
sorry

end num_positive_multiples_of_6_ending_in_4_lt_500_l214_214503


namespace liters_to_pints_l214_214073

theorem liters_to_pints (liters pints : ℝ) (conversion_factor : ℝ) (h1 : 0.75 = liters) (h2 : 1.575 = pints) :
  let conversion_factor := pints / liters in
  2 * conversion_factor = 4.2 :=
  sorry

end liters_to_pints_l214_214073


namespace maisy_earnings_increase_l214_214948

-- Define the conditions from the problem
def current_job_hours_per_week : ℕ := 8
def current_job_wage_per_hour : ℕ := 10

def new_job_hours_per_week : ℕ := 4
def new_job_wage_per_hour : ℕ := 15
def new_job_bonus_per_week : ℕ := 35

-- Define the weekly earnings calculations
def current_job_earnings : ℕ := current_job_hours_per_week * current_job_wage_per_hour
def new_job_earnings_without_bonus : ℕ := new_job_hours_per_week * new_job_wage_per_hour
def new_job_earnings_with_bonus : ℕ := new_job_earnings_without_bonus + new_job_bonus_per_week

-- Define the difference in earnings
def earnings_difference : ℕ := new_job_earnings_with_bonus - current_job_earnings

-- The theorem to prove: Maisy will earn $15 more per week at her new job
theorem maisy_earnings_increase : earnings_difference = 15 := by
  sorry

end maisy_earnings_increase_l214_214948


namespace value_of_a_l214_214844

variable (a : ℤ)

def M : Set ℤ := {a, 0}
def N : Set ℤ := {x | 2 * x ^ 2 - 5 * x < 0 ∧ x ∈ ℤ}

theorem value_of_a (h : (M a ∩ N) ≠ ∅) : a = 1 ∨ a = 2 :=
by
  sorry

end value_of_a_l214_214844


namespace machine_A_sprockets_per_hour_l214_214946

theorem machine_A_sprockets_per_hour 
  (A T_Q : ℝ)
  (h1 : 550 = 1.1 * A * T_Q)
  (h2 : 550 = A * (T_Q + 10)) 
  : A = 5 :=
by
  sorry

end machine_A_sprockets_per_hour_l214_214946


namespace find_square_tiles_l214_214703

variable {s p : ℕ}

theorem find_square_tiles (h1 : s + p = 30) (h2 : 4 * s + 5 * p = 110) : s = 20 :=
by
  sorry

end find_square_tiles_l214_214703


namespace volleyball_match_win_probability_l214_214977

/-- Probability of a team winning a best-of-five match 3:0 given different probabilities for each set -/
theorem volleyball_match_win_probability :
  let p₁ := 2/3 -- Probability of winning any of the first 4 sets
  let p₅ := 1/2 -- Probability of winning the fifth set
  p₁ * p₁ * p₁ = 8/27 :=
by
  sorry

end volleyball_match_win_probability_l214_214977


namespace ordered_triple_P_l214_214149

-- Given conditions as defined in the problem
variables {A B C E F P : Type}
variables (vecA vecB vecC vecE vecF vecP : A)
variables (r1 r2 : ℚ)

-- Define the conditions
def AE_EC_ratio (hE : E) : Prop :=
  vecE = (1/3) * vecA + (2/3) * vecC

def AF_FB_ratio (hF : F) : Prop :=
  vecF = (4/5) * vecA + (1/5) * vecB

def intersect_BE_CF (hP : P) : Prop :=
  ∃ hE : E, ∃ hF : F, AE_EC_ratio vecA vecC vecE hE ∧ AF_FB_ratio vecA vecB vecF hF ∧
    vecP = r1 * vecB + r2 * vecF

-- Prove the final ordered triple
theorem ordered_triple_P (hP : P) (hE : E) (hF : F) (h1 : AE_EC_ratio vecA vecC vecE hE)
  (h2 : AF_FB_ratio vecA vecB vecF hF) : 
  vecP = (4/13) * vecA + (1/13) * vecB + (8/13) * vecC :=
sorry

end ordered_triple_P_l214_214149


namespace not_possible_to_make_all_divisible_l214_214909

/-- Predicate representing the state of the cube where each vertex is labeled with an integer -/
def CubeState := Fin 8 → ℤ

/-- Initial state of the cube -/
def initial_state : CubeState :=
  ![1, 0, 0, 0, 0, 0, 0, 0]

/-- Define the allowed operation: incrementing the numbers at the ends of a given edge -/
def allowed_operation (s : CubeState) (v1 v2 : Fin 8) : CubeState :=
  fun v => if v = v1 ∨ v = v2 then s v + 1 else s v

/-- Define the checkerboard coloring of the vertices -/
def is_black (v : Fin 8) : Prop :=
  match v with
  | 0 | 2 | 4 | 6 => True
  | _            => False

/-- Sum of the numbers at black vertices -/
def sum_black (s : CubeState) : ℤ :=
  s 0 + s 2 + s 4 + s 6

/-- Sum of the numbers at white vertices -/
def sum_white (s : CubeState) : ℤ :=
  s 1 + s 3 + s 5 + s 7

/-- Difference between the sums at black and white vertices -/
def difference (s : CubeState) : ℤ :=
  sum_black s - sum_white s

/-- Predicate to check if all vertex numbers are divisible by 3 -/
def all_divisible_by_three (s : CubeState) : Prop :=
  ∀ v, s v % 3 = 0

/-- The proof problem: Prove that it's not possible to make all vertex numbers divisible by 3 -/
theorem not_possible_to_make_all_divisible :
  ¬(∃ s, (∃ f : CubeState → CubeState, (initial_state = s ∨ ∃ v1 v2 : Fin 8, s = allowed_operation f v1 v2) ∧ all_divisible_by_three s)) :=
sorry

end not_possible_to_make_all_divisible_l214_214909


namespace g_x_minus_3_l214_214083

def g (x : ℝ) : ℝ := x^2

theorem g_x_minus_3 (x : ℝ) : g (x - 3) = x^2 - 6 * x + 9 :=
by
  -- This is where the proof would go
  sorry

end g_x_minus_3_l214_214083


namespace Genevieve_bought_277_kg_l214_214443

variable kg_price : ℝ := 8
variable discount : ℝ := 0.10
variable short_amount : ℝ := 400
variable cash_on_hand : ℝ := 1600
variable discounted_price : ℝ := kg_price * (1 - discount)
variable total_cost : ℝ := cash_on_hand + short_amount
variable kg := total_cost / discounted_price

theorem Genevieve_bought_277_kg :
  kg = 277 :=
by
  rw [discounted_price, total_cost]
  sorry

end Genevieve_bought_277_kg_l214_214443


namespace sum_binomial_coefficient_l214_214785

theorem sum_binomial_coefficient : 
  (∑ k in Finset.range 11, (k + 1) * Nat.choose 10 (k + 1)) = 5120 := 
by
  sorry

end sum_binomial_coefficient_l214_214785


namespace cannot_be_value_of_omega_l214_214081

theorem cannot_be_value_of_omega (ω : ℤ) (φ : ℝ) (k n : ℤ) 
  (h1 : 0 < ω) 
  (h2 : |φ| < π / 2)
  (h3 : ω * (π / 12) + φ = k * π + π / 2)
  (h4 : -ω * (π / 6) + φ = n * π) : 
  ∀ m : ℤ, ω ≠ 4 * m := 
sorry

end cannot_be_value_of_omega_l214_214081


namespace find_k_l214_214945

def line_p (x y : ℝ) : Prop := y = -2 * x + 3
def line_q (x y k : ℝ) : Prop := y = k * x + 4
def intersection (x y k : ℝ) : Prop := line_p x y ∧ line_q x y k

theorem find_k (k : ℝ) (h_inter : intersection 1 1 k) : k = -3 :=
sorry

end find_k_l214_214945


namespace expected_value_correct_l214_214584

noncomputable def coin_problem_expected_value : ℚ :=
  let p := (1:ℚ) / 25 in
  24 / 7

theorem expected_value_correct :
  let p := (1:ℚ) / 25 in
  ∃ H T : ℕ, 
  ∃ f : ℕ → ℕ, 
  (∀ n, f n ∈ {0, 1}) ∧
  H = ∑ n in range (f.size), if f n = 1 then 1 else 0 ∧
  T = ∑ n in range (f.size), if f n = 0 then 1 else 0 ∧
  (∃ m : ℕ, f m = 1 → p = 1/25) → -- this models the coin being blown away
  E (λ H T : ℕ, |H - T|) = 24 / 7 :=
sorry

end expected_value_correct_l214_214584


namespace BX_squared_l214_214565

-- Definitions of the problem
variables (A B C M N X : Type) 
variables
  [Inhabited A] [Inhabited B] [Inhabited C] 
  [Inhabited M] [Inhabited N] [Inhabited X]

-- Conditions
axiom midpoint_AC : midpoint A C M
axiom angle_bisector_C : angle_bisector C N (line A B)
axiom intersection_BM_CN : ∃! X, intersection (line B M) (line C N) X
axiom right_triangle_BNX : right_triangle B N X
axiom angle_BXN_90 : angle B X N = 90
axiom length_AC_4 : length (segment A C) = 4

-- Assertion to be proved
theorem BX_squared : length (segment B X) ^ 2 = 2 :=
sorry

end BX_squared_l214_214565


namespace factor_fraction_l214_214380

/- Definitions based on conditions -/
variables {a b c : ℝ}

theorem factor_fraction :
  ( (a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3 ) / 
  ((a - b)^3 + (b - c)^3 + (c - a)^3) = 
  (a^2 + ab + b^2) * (b^2 + bc + c^2) * (c^2 + ca + a^2) :=
begin
  sorry
end

end factor_fraction_l214_214380


namespace problem_l214_214060

variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f (-x) = f x)  -- f is an even function
variable (h_mono : ∀ x y : ℝ, 0 < x → x < y → f y < f x)  -- f is monotonically decreasing on (0, +∞)

theorem problem : f 3 < f (-2) ∧ f (-2) < f 1 :=
by
  sorry

end problem_l214_214060


namespace find_possible_C_l214_214366

def is_palindromic_integer (n : ℕ) : Prop :=
  n.to_digits.reverse = n.to_digits

def four_digit_palindromic_integer (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ is_palindromic_integer n

def three_digit_palindromic_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ is_palindromic_integer n

theorem find_possible_C (A B C : ℤ) 
  (hA : four_digit_palindromic_integer A)
  (hB : four_digit_palindromic_integer B)
  (hC : three_digit_palindromic_integer C)
  (h_eq : A - B = C) :
  C = 121 :=
begin
  sorry
end

end find_possible_C_l214_214366


namespace rank_emma_highest_cassie_below_l214_214370

variable (score : Type*) [LinearOrder score]

variables (emma cassie bridget hannah : score)

def ranking :=
  (emma > cassie) ∧ (emma > bridget) ∧ (emma > hannah) ∧
  (cassie > bridget) ∧ (cassie > hannah) ∧
  (bridget > hannah)

theorem rank_emma_highest_cassie_below :
  emma > cassie → ¬bridget = min bridget hannah → ranking emma cassie bridget hannah :=
by
  sorry

end rank_emma_highest_cassie_below_l214_214370


namespace problem_statement_l214_214347

-- Define each term separately based on the conditions in a)
def cube_root_64 := 64 ^ (1 / 3 : ℝ)
def negative_fraction_zero_exp := (-2 / 3 : ℝ) ^ 0
def log_base2_of_4 := Real.logb 2 4

-- State the theorem
theorem problem_statement : cube_root_64 - negative_fraction_zero_exp + log_base2_of_4 = 5 := by
  sorry

end problem_statement_l214_214347


namespace problem_not_unique_triangle_determination_l214_214363

def vertex_angle_and_side_does_not_determine_equilateral_triangle : Prop :=
  ¬ ∀ (angle : ℝ) (side : ℝ), ∃ (triangle : Triangle),
    is_equilateral triangle ∧ vertex_angle triangle = angle ∧ side_of triangle = side

theorem problem_not_unique_triangle_determination :
  vertex_angle_and_side_does_not_determine_equilateral_triangle :=
sorry

end problem_not_unique_triangle_determination_l214_214363


namespace total_money_spent_l214_214868

def cost_life_journey_cd : ℕ := 100
def cost_day_life_cd : ℕ := 50
def cost_when_rescind_cd : ℕ := 85
def number_of_cds_each : ℕ := 3

theorem total_money_spent :
  number_of_cds_each * cost_life_journey_cd +
  number_of_cds_each * cost_day_life_cd +
  number_of_cds_each * cost_when_rescind_cd = 705 :=
sorry

end total_money_spent_l214_214868


namespace num_valid_arrangements_l214_214896

-- Define the set of numbers we are working with
def numbers := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the odd and even subsets
def odds := {1, 3, 5, 7, 9}
def evens := {0, 2, 4, 6, 8}

-- Statement to prove the number of valid arrangements
theorem num_valid_arrangements : 
  ∃ n : ℕ, n = (nat.choose 10 5) ∧ n = 252 :=
by {
  apply Exists.intro (nat.choose 10 5),
  split,
  {refl},
  {norm_num}
}

end num_valid_arrangements_l214_214896


namespace count_integers_in_interval_l214_214501

theorem count_integers_in_interval : 
  let pi := Real.pi in
  - Real.ofInt 5 * pi ≤ m ∧ m ≤ Real.ofInt 15 * pi →
  (∃ lb ub : ℤ, (lb : ℝ) > -5 * pi ∧ (ub : ℝ) < 15 * pi ∧ lb ≤ m ∧ m ≤ ub) →
  (∃ count : ℤ, count = 63 ∧ (∀ m : ℤ, lb ≤ m ∧ m ≤ ub → count = ub - lb + 1)) :=
by
  sorry

end count_integers_in_interval_l214_214501


namespace g_inv_undefined_at_one_l214_214877

noncomputable def g (x : ℝ) : ℝ := (x - 5) / (x - 6)

theorem g_inv_undefined_at_one :
  ∀ (x : ℝ), (∃ (y : ℝ), g y = x ∧ ¬ ∃ (z : ℝ), g z = y ∧ g z = 1) ↔ x = 1 :=
by
  sorry

end g_inv_undefined_at_one_l214_214877


namespace probability_XOXOXOX_l214_214431

theorem probability_XOXOXOX :
  let X := 4;
      O := 3;
      total_positions := 7;
      specific_arrangement := 1;
      total_arrangements := Nat.choose total_positions X in
  1 / total_arrangements = 1 / 35 := by
  sorry

end probability_XOXOXOX_l214_214431


namespace find_x_l214_214303

theorem find_x (x : ℝ) (h : 0.65 * x = 0.2 * 617.50) : x = 190 :=
by
  sorry

end find_x_l214_214303


namespace pentagons_from_15_points_l214_214015

theorem pentagons_from_15_points (n : ℕ) (h : n = 15) : (nat.choose 15 5) = 3003 := by
  rw h
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num)

end pentagons_from_15_points_l214_214015


namespace base4_arith_proof_l214_214383

def base4_to_base10 (n : list ℕ) : ℕ :=
n.reverse.zipWithIndex.foldl (λ acc ⟨d, i⟩ => acc + d * 4^i) 0

theorem base4_arith_proof :
  let x := base4_to_base10 [1, 2, 0] -- 120_4 in base 10
  let y := base4_to_base10 [2, 1] -- 21_4 in base 10
  let z := base4_to_base10 [3, 3] -- 33_4 in base 10
  let a := base4_to_base10 [1, 3] -- 13_4 in base 10
  let b := base4_to_base10 [3] -- 3_4 in base 10
  let result := (z * a) / b
  let base10_to_base4 (n : ℕ) : list ℕ :=
    let quotRem := Nat.divMod
    let rec loop (n : ℕ) (acc : list ℕ) :=
        if n == 0 then acc else
          let (q, r) := quotRem n 4
          loop q (r :: acc)
    loop n []
  base10_to_base4 result = [2, 0, 3] -- 203_4 in base 10
  := sorry

end base4_arith_proof_l214_214383


namespace cos_C_equals_neg_one_fourth_l214_214121

-- Definition of the problem's conditions
variables {A B C : ℝ} [triangle : triangle ABC] -- Assume triangles are represented in some way

-- Define the given conditions
variables {sinA sinB sinC : ℝ}
variable (h_ratio : sinA / sinB = 2 / 3 ∧ sinB / sinC = 3 / 4)

-- The theorem we want to prove, using the Law of Cosines
theorem cos_C_equals_neg_one_fourth (a b c k : ℝ) (h_a : a = 2 * k) (h_b : b = 3 * k) (h_c : c = 4 * k) (h_law_of_sines : a / sinA = b / sinB ∧ b / sinB = c / sinC) : 
  cos C = -1 / 4 :=
by sorry

end cos_C_equals_neg_one_fourth_l214_214121


namespace prob_calculation_l214_214999

noncomputable def normal_distribution (mu sigma : ℝ) : Measure ℝ :=
  MeasureTheory.Measure.dirac mu

open MeasureTheory

def prob_le (x : ℝ) (dist : Measure ℝ) (a : ℝ) : ℝ := dist {y | y ≤ a}
def prob_interval (dist : Measure ℝ) (a b : ℝ) : ℝ := dist {y | a < y ∧ y ≤ b}

axiom normal_property {x : ℝ} {σ : ℝ} :
  ∀ (mu : ℝ), (∀ (a : ℝ), x ∼ (fun y => Measure.dirac y) mu σ)  →  prob_le 2 (normal_distribution 3 σ) = 0.3

theorem prob_calculation (σ : ℝ) (h : prob_le 2 (normal_distribution 3 σ) = 0.3) :
  prob_interval (normal_distribution 3 σ) 3 4 = 0.2 :=
sorry

end prob_calculation_l214_214999


namespace cos_alpha_solution_l214_214117

theorem cos_alpha_solution :
  let x := 2 * Real.cos (120 * Real.pi / 180),
      y := Real.sqrt 2 * Real.sin (225 * Real.pi / 180),
      p := (x, y),
      alpha := Real.arccos (x / Real.sqrt (x^2 + y^2))
  in x / Real.sqrt (x^2 + y^2) = -Real.sqrt 2 / 2 := 
by
  sorry

end cos_alpha_solution_l214_214117


namespace correct_option_is_D_l214_214087

def p : Prop := 3 ≥ 3
def q : Prop := 3 > 4

theorem correct_option_is_D (hp : p) (hq : ¬ q) :
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬ ¬ p :=
by
  sorry

end correct_option_is_D_l214_214087


namespace factor_fraction_eq_l214_214373

theorem factor_fraction_eq (a b c : ℝ) :
  ((a^2 + b^2)^3 + (b^2 + c^2)^3 + (c^2 + a^2)^3) 
  / ((a + b)^3 + (b + c)^3 + (c + a)^3) = 
  ((a^2 + b^2) * (b^2 + c^2) * (c^2 + a^2)) 
  / ((a + b) * (b + c) * (c + a)) :=
by
  sorry

end factor_fraction_eq_l214_214373


namespace comparison_problems_l214_214777

theorem comparison_problems :
  (80000 > 8000) ∧ 
  (9 < 90) ∧ 
  (45 * 3 * 4 = 45 * 12) ∧ 
  (100 > 1) ∧ 
  (3.8 = 3.80) ∧ 
  (9.08 < 9.8) :=
by
  have h1 : 80000 > 8000 := sorry,
  have h2 : 9 < 90 := sorry,
  have h3 : 45 * 3 * 4 = 45 * 12 := sorry,
  have h4 : 100 > 1 := sorry,
  have h5 : 3.8 = 3.80 := sorry,
  have h6 : 9.08 < 9.8 := sorry,
  exact ⟨h1, h2, h3, h4, h5, h6⟩

end comparison_problems_l214_214777


namespace painting_time_l214_214368

-- Definitions based on the conditions
def num_people1 := 8
def num_houses1 := 3
def time1 := 12
def num_people2 := 9
def num_houses2 := 4
def k := (num_people1 * time1) / num_houses1

-- The statement we want to prove
theorem painting_time : (num_people2 * t = k * num_houses2) → (t = 128 / 9) :=
by sorry

end painting_time_l214_214368


namespace intersection_M_N_eq_2_4_l214_214931

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℕ := {x | ∃ y, y = Real.log (6 - x) ∧ x < 6}

theorem intersection_M_N_eq_2_4 : M ∩ N = {2, 4} :=
by sorry

end intersection_M_N_eq_2_4_l214_214931


namespace exists_n_eq_1992_l214_214792

-- Definition of greatest odd divisor
def g (x : ℕ) : ℕ := 
  if h : x = 0 then 0 else (List.range (x + 1)).filter (λ d, d > 0 ∧ d % 2 = 1 ∧ x % d = 0).maximum' sorry

-- Definition of f
def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then 
    x / 2 + x / g x 
  else 
    2 ^ ((x + 1) / 2)

-- Definition of the sequence
def x_seq : ℕ → ℕ 
| 0     := 1
| (n+1) := f (x_seq n)

-- The theorem statement
theorem exists_n_eq_1992 : ∃ n : ℕ, x_seq n = 1992 ∧ n = 8253 :=
sorry

end exists_n_eq_1992_l214_214792


namespace no_solutions_for_quadratic_inequality_l214_214259

theorem no_solutions_for_quadratic_inequality :
  ∀ x : ℝ, ¬ (-x^2 + 2 * x - 3 > 0) :=
by
  intro x
  suffices h : x^2 - 2 * x + 3 ≥ 0 by
    exact not_lt_of_ge h
  calc
    x^2 - 2 * x + 3 = (x - 1)^2 + 2 : by ring
    _ ≥ 2 : by apply add_nonneg; 
              { exact sq_nonneg _ }

end no_solutions_for_quadratic_inequality_l214_214259


namespace hyperbola_equation_correct_l214_214842

noncomputable def is_hyperbola_eq (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (focal_length : ℝ) (asymptote_eq : ∀ (x y : ℝ), 2*x + y = 0 ∨ 2*x - y = 0) : Prop :=
a^2 + b^2 = 20 ∧ a = 2 ∧ b = 4

theorem hyperbola_equation_correct : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ is_hyperbola_eq a b 4 (2 * Real.sqrt 5) ∧
  (∀ (x y : ℝ), 2*x + y = 0 ∨ 2*x - y = 0) ∧ 
  ∀ (x y : ℝ), (x^2) / 4 - (y^2) / 16 = 1 :=
sorry

end hyperbola_equation_correct_l214_214842


namespace derivative_at_neg2_l214_214884

def f (x : ℝ) : ℝ := a * x^3 + b * x + c

theorem derivative_at_neg2 (a b c : ℝ) (h : 12 * a + b = 2) : (3 * a * (-2)^2 + b) = 2 :=
by
  sorry

end derivative_at_neg2_l214_214884


namespace buffy_whiskers_l214_214797

/-- Definition of whisker counts for the cats --/
def whiskers_of_juniper : ℕ := 12
def whiskers_of_puffy : ℕ := 3 * whiskers_of_juniper
def whiskers_of_scruffy : ℕ := 2 * whiskers_of_puffy
def whiskers_of_buffy : ℕ := (whiskers_of_juniper + whiskers_of_puffy + whiskers_of_scruffy) / 3

/-- Proof statement for the number of whiskers of Buffy --/
theorem buffy_whiskers : whiskers_of_buffy = 40 := 
by
  -- Proof is omitted
  sorry

end buffy_whiskers_l214_214797


namespace part_a_part_b_l214_214236

variables {A B C D O O1 O2 O3 O4 : Type} [InnerProductSpace ℝ A]
variables {AC BD : ℝ} {r1 r2 r3 r4 : ℝ}
variables (AOB BOC COD DOA : Triangle) (S1 S2 S3 S4 : Circle)
variables (O1 O2 O3 O4 : Point)

-- Conditions of the problem
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry
def diagonals_perpendicular (A B C D O : Point) : Prop := sorry 
def circles_inscribed_in_triangles (A B C D O1 O2 O3 O4 : Point) 
  (r1 r2 r3 r4 : ℝ) : Prop := sorry

-- Questions to be proved
theorem part_a : 
  (is_convex_quadrilateral A B C D) ∧ 
  (diagonals_perpendicular A B C D O) ∧ 
  (circles_inscribed_in_triangles A B C D O1 O2 O3 O4 r1 r2 r3 r4) → 
  2 * (r1 + r2 + r3 + r4) ≤ (2 - real.sqrt 2) * (AC + BD) := 
sorry

theorem part_b : 
  (is_convex_quadrilateral A B C D) ∧ 
  (diagonals_perpendicular A B C D O) ∧ 
  (circles_inscribed_in_triangles A B C D O1 O2 O3 O4 r1 r2 r3 r4) → 
  dist O1 O2 + dist O2 O3 + dist O3 O4 + dist O4 O1 ≤ 
  2 * (real.sqrt 2 - 1) * (AC + BD) := 
sorry

end part_a_part_b_l214_214236


namespace exists_multiple_of_power_of_2_with_non_zero_digits_l214_214963

theorem exists_multiple_of_power_of_2_with_non_zero_digits (n : ℕ) (hn : n ≥ 1) :
  ∃ a : ℕ, (∀ d ∈ a.digits 10, d = 1 ∨ d = 2) ∧ 2^n ∣ a :=
by
  sorry

end exists_multiple_of_power_of_2_with_non_zero_digits_l214_214963


namespace sum_of_digits_of_63_l214_214384

theorem sum_of_digits_of_63 (x y : ℕ) (h : 10 * x + y = 63) (h1 : x + y = 9) (h2 : x - y = 3) : x + y = 9 :=
by
  sorry

end sum_of_digits_of_63_l214_214384


namespace sin_pi_plus_alpha_l214_214875

theorem sin_pi_plus_alpha {α : ℝ} (h1 : cos (2 * Real.pi - α) = -Real.sqrt 5 / 3) (h2 : α ∈ Set.Ioo Real.pi (3 * Real.pi / 2)) : sin (Real.pi + α) = 2 / 3 := by
  sorry

end sin_pi_plus_alpha_l214_214875


namespace fox_maximum_berries_eaten_l214_214596

theorem fox_maximum_berries_eaten : 
  let total_berries := (2^100 - 1)
  ∧ let remainder := (total_berries % 100)
  ∧ remainder = 75 
  ∧ max_fox_berries := total_berries - remainder - 1
  in max_fox_berries = 2^100 - 101 :=
by
  intros total_berries remainder max_fox_berries
  sorry

end fox_maximum_berries_eaten_l214_214596


namespace wave_pulse_travel_time_l214_214302

-- Defining the conditions as constants
constant unstretched_length : ℝ := 1.0
constant initial_stretched_length : ℝ := 10.0
constant initial_travel_time : ℝ := 1.0
constant new_stretched_length : ℝ := 20.0

-- Defining the question and proof statement
theorem wave_pulse_travel_time :
  (initial_travel_time / initial_stretched_length) * new_stretched_length = 1.0 :=
sorry

end wave_pulse_travel_time_l214_214302


namespace determine_a_l214_214574

theorem determine_a (a b c : ℕ) (h₁ : a ≥ b) (h₂ : b ≥ c) 
  (h₃ : a^2 - b^2 - c^2 + a * b = 2015)
  (h₄ : a^2 + 3 * b^2 + 3 * c^2 - 3 * a * b - 2 * a * c - 2 * b * c = -1993) :
  a = 255 :=
sorry

end determine_a_l214_214574


namespace evaluate_expression_l214_214453

theorem evaluate_expression (x y : ℝ) (h1 : x * y = -2) (h2 : x + y = 4) : x^2 * y + x * y^2 = -8 :=
by
  sorry

end evaluate_expression_l214_214453


namespace fifth_number_in_tenth_row_l214_214313

def nth_number_in_row (n k : ℕ) : ℕ :=
  7 * n - (7 - k)

theorem fifth_number_in_tenth_row : nth_number_in_row 10 5 = 68 :=
by
  sorry

end fifth_number_in_tenth_row_l214_214313


namespace minimum_dot_product_l214_214137

open Real

noncomputable def f (x : ℝ) : ℝ := -x + exp x

theorem minimum_dot_product : ∃ x : ℝ, f x = 1 := by
  use 0
  unfold f
  simp
  norm_num

end minimum_dot_product_l214_214137


namespace solution_l214_214068

noncomputable def problem (α : ℝ) : Prop :=
  sin(5 * real.pi - α) = 1 / 2

theorem solution (α : ℝ) (h : sin(real.pi + α) = -1 / 2) : problem α :=
by
  sorry

end solution_l214_214068


namespace inverse_of_true_implies_negation_true_l214_214516

variable (P : Prop)
theorem inverse_of_true_implies_negation_true (h : ¬ P) : ¬ P :=
by 
  exact h

end inverse_of_true_implies_negation_true_l214_214516


namespace range_of_a_l214_214486

noncomputable def f (x a : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

theorem range_of_a {a : ℝ} : 
  (∃ x : ℝ, ∀ y : ℝ, f y x ≤ f x x) ∧ (∃ x : ℝ, ∀ y : ℝ, f x x ≤ f y x) → 
  a ∈ set.Ioo (-∞ : ℝ) -3 ∪ set.Ioo 6 (∞ : ℝ) :=
sorry

end range_of_a_l214_214486


namespace points_on_opposite_sides_of_line_range_m_l214_214477

theorem points_on_opposite_sides_of_line_range_m :
  (∀ (m : ℝ), (3 * 3 - 2 * 1 + m) * (3 * -4 - 2 * 6 + m) < 0 → -7 < m ∧ m < 24) := 
by sorry

end points_on_opposite_sides_of_line_range_m_l214_214477


namespace perpendicular_point_sets_l214_214492

-- Define what it means for a set to be a perpendicular point set
def perpendicular_point_set (M : Set (ℝ × ℝ)) : Prop :=
  ∀ (x₁ y₁ : ℝ), (x₁, y₁) ∈ M → ∃ (x₂ y₂ : ℝ), (x₂, y₂) ∈ M ∧ x₁ * x₂ + y₁ * y₂ = 0

-- Define sets M1, M2, M3, and M4
def M1 : Set (ℝ × ℝ) := { p | ∃ (x : ℝ), p = (x, 1 / x^2) }
def M2 : Set (ℝ × ℝ) := { p | ∃ (x : ℝ), x > 0 ∧ p = (x, log x / log 2) }
def M3 : Set (ℝ × ℝ) := { p | ∃ (x : ℝ), p = (x, 2^x - 2) }
def M4 : Set (ℝ × ℝ) := { p | ∃ (x : ℝ), p = (x, sin x + 1) }

theorem perpendicular_point_sets :
  perpendicular_point_set M1 ∧
  ¬ perpendicular_point_set M2 ∧
  perpendicular_point_set M3 ∧
  perpendicular_point_set M4 :=
by {
  sorry -- Proof goes here
}

end perpendicular_point_sets_l214_214492


namespace max_f₁_value_min_f₂_value_l214_214299

def f₁ (x : ℝ) : ℝ := (Real.cos x) ^ 2 - Real.sin x
def f₂ (x : ℝ) (a : ℝ) : ℝ := (Real.cos x) ^ 2 - a * Real.sin x

theorem max_f₁_value : ∃ x : ℝ, ∀ y : ℝ, -1 ≤ Real.sin y ∧ Real.sin y ≤ 1 → f₁ y ≤ f₁ x ∧ f₁ x = 5 / 4 := by
  sorry

theorem min_f₂_value :
  ∀ (a : ℝ), 
  (∀ (x : ℝ), -1 ≤ Real.sin x ∧ Real.sin x ≤ 1 → 
    ((a ≤ 0 → ∃ y : ℝ, f₂ y a = a ∧ (∀ z : ℝ, -1 ≤ Real.sin z ∧ Real.sin z ≤ 1 → a ≤ f₂ z a)) ∧ 
    (a > 0 → ∃ y : ℝ, f₂ y a = -a ∧ (∀ z : ℝ, -1 ≤ Real.sin z ∧ Real.sin z ≤ 1 → -a ≤ f₂ z a)))) := by
  sorry

end max_f₁_value_min_f₂_value_l214_214299


namespace range_of_t_min_value_of_f_max_value_of_f_l214_214080

def f (x : ℝ) : ℝ := (Real.log x / Real.log 2)^2 + 3 * (Real.log x / Real.log 2) + 2 

def log2 (x : ℝ) := Real.log x / Real.log 2

theorem range_of_t (x : ℝ) (hx : 1/4 ≤ x ∧ x ≤ 4) : -2 ≤ log2 x ∧ log2 x ≤ 2 := sorry

theorem min_value_of_f (x : ℝ) (hx : 1/2/Real.sqrt 2 = x) : f x = -1/4 := sorry

theorem max_value_of_f (x : ℝ) (hx : 4 = x) : f x = 12 := sorry

end range_of_t_min_value_of_f_max_value_of_f_l214_214080


namespace remainder_of_division_l214_214198

theorem remainder_of_division (d : ℝ) (q : ℝ) (r : ℝ) : 
  d = 187.46067415730337 → q = 89 → 16698 = (d * q) + r → r = 14 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  simp at h3
  sorry

end remainder_of_division_l214_214198


namespace smallest_interior_angle_l214_214147

open Real

theorem smallest_interior_angle (A B C : ℝ) (hA : 0 < A ∧ A < π)
    (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π)
    (h_sum_angles : A + B + C = π)
    (h_ratio : sin A / sin B = 2 / sqrt 6 ∧ sin A / sin C = 2 / (sqrt 3 + 1)) :
    min A (min B C) = π / 4 := 
  by sorry

end smallest_interior_angle_l214_214147


namespace middle_integer_is_five_l214_214260

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def are_consecutive_odd_integers (a b c : ℤ) : Prop :=
  a < b ∧ b < c ∧ (∃ n : ℤ, a = b - 2 ∧ c = b + 2 ∧ is_odd a ∧ is_odd b ∧ is_odd c)

def sum_is_one_eighth_product (a b c : ℤ) : Prop :=
  a + b + c = (a * b * c) / 8

theorem middle_integer_is_five :
  ∃ (a b c : ℤ), are_consecutive_odd_integers a b c ∧ sum_is_one_eighth_product a b c ∧ b = 5 :=
by
  sorry

end middle_integer_is_five_l214_214260


namespace tan_identity_proof_l214_214658

theorem tan_identity_proof :
  (1 - Real.tan (100 * Real.pi / 180)) * (1 - Real.tan (35 * Real.pi / 180)) = 2 :=
by
  have tan_135 : Real.tan (135 * Real.pi / 180) = -1 := by sorry -- This needs a separate proof.
  have tan_sum_formula : ∀ A B : ℝ, Real.tan (A + B) = (Real.tan A + Real.tan B) / (1 - Real.tan A * Real.tan B) := by sorry -- This needs a deeper exploration
  sorry -- Main proof to be filled

end tan_identity_proof_l214_214658


namespace range_of_f_l214_214255

-- Definitions
def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 3 then 2 * x - x^2
  else if -2 ≤ x ∧ x < 0 then x^2 + 6 * x
  else 0

-- Theorem statement
theorem range_of_f : Set.range f = Set.Icc (-9 : ℝ) (1 : ℝ) :=
by
  sorry

end range_of_f_l214_214255


namespace simplify_expression_l214_214213

variable {m : ℝ}
variable (m_ne_zero : m ≠ 0)

theorem simplify_expression : 
  \left( -\frac{2}{5m} \right)^{-3} * (5m)^2 = -\frac{3125m^5}{8} :=
by
  sorry

end simplify_expression_l214_214213


namespace smallest_possible_value_expression_l214_214031

open Real

noncomputable def min_expression_value (a b c : ℝ) : ℝ :=
  (a + b)^2 + (b - c)^2 + (c - a)^2 / a^2

theorem smallest_possible_value_expression :
  ∀ (a b c : ℝ), a > b → b > c → a + c = 2 * b → a ≠ 0 → min_expression_value a b c = 7 / 2 := by
  sorry

end smallest_possible_value_expression_l214_214031


namespace student_score_l214_214326

variable {Points : Type} [AddCommGroup Points] [VectorSpace ℚ Points]

def directly_proportional (score : ℚ) (hours : ℚ) : Prop :=
  ∃ k : ℚ, score = k * hours

theorem student_score
  (h_study_2_hours : directly_proportional 90 2)
  (h_study_5_hours : ∃ k : ℚ, ∀ x, directly_proportional x 5 ↔ x = k * 5) :
  (∃ x : ℚ, directly_proportional x 5) ∧ x = 225 :=
by
  sorry

end student_score_l214_214326


namespace problem_statement_l214_214481

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then real.log (1 - x) / real.log 2 - 1 else 2 ^ x

theorem problem_statement : f (f (-2)) = 3 / 2 :=
by
  sorry

end problem_statement_l214_214481


namespace base9_add_subtract_l214_214753

theorem base9_add_subtract :
  let n1 := 3 * 9^2 + 5 * 9 + 1
  let n2 := 4 * 9^2 + 6 * 9 + 5
  let n3 := 1 * 9^2 + 3 * 9 + 2
  let n4 := 1 * 9^2 + 4 * 9 + 7
  (n1 + n2 + n3 - n4 = 8 * 9^2 + 4 * 9 + 7) :=
by
  sorry

end base9_add_subtract_l214_214753


namespace real_part_of_z_l214_214645

noncomputable def imaginary_unit : ℂ := complex.i
def z : ℂ := (1 + imaginary_unit) * (1 + 2 * imaginary_unit)

theorem real_part_of_z :
  z.re = -1 :=
sorry

end real_part_of_z_l214_214645


namespace find_smallest_m_l214_214181

def theta := Real.arctan (20 / 91)
def alpha := Real.pi / 60
def beta := Real.pi / 48

def R (θ : ℝ) : ℝ := θ + Real.pi / 120
def R_n (l : ℝ) (n : ℕ) : ℝ :=
  match n with
  | 0     => l
  | n + 1 => R (R_n l n)

theorem find_smallest_m (m : ℕ) (h : m > 0) :
  (R_n theta m) = theta ↔ m = 120 :=
sorry

end find_smallest_m_l214_214181


namespace probability_XOXOXOX_is_1_div_35_l214_214420

def count_combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def num_ways_to_choose_positions_for_X (total_positions : ℕ) (num_X : ℕ) : ℕ := 
  count_combinations total_positions num_X

def num_ways_for_specific_arrangement_XOXOXOX : ℕ := 1

def probability_of_XOXOXOX (num_ways_total : ℕ) (num_ways_specific : ℕ) : ℚ := 
  num_ways_specific / num_ways_total

theorem probability_XOXOXOX_is_1_div_35 :
  probability_of_XOXOXOX (num_ways_to_choose_positions_for_X 7 4) num_ways_for_specific_arrangement_XOXOXOX = 1 / 35 := by
  sorry

end probability_XOXOXOX_is_1_div_35_l214_214420


namespace solve_inequality_l214_214032

noncomputable def solutionSet := { x : ℝ | 0 < x ∧ x < 1 }

theorem solve_inequality (x : ℝ) : x^2 < x ↔ x ∈ solutionSet := 
sorry

end solve_inequality_l214_214032


namespace find_x_y_sum_l214_214444

theorem find_x_y_sum (x y : ℝ) 
  (h1 : 3^x = 27^(y + 2)) 
  (h2 : 16^y = 4^(x - 6)) : 
  x + y = 18 :=
sorry

end find_x_y_sum_l214_214444


namespace area_triangle_bce_l214_214897

theorem area_triangle_bce (A B C D E : Type) [geometry A B] (h_parallel : parallel AB CD) (h_perpendicular : perpendicular BC AB) (h_intersect : intersect AC BD = E) 
  (h_AB : length AB = 20) (h_BC : length BC = 2016) (h_CD : length CD = 16) : 
  area_triangle BCE = 8960 := 
by
  sorry

end area_triangle_bce_l214_214897


namespace sqrt_mul_simplify_l214_214223

theorem sqrt_mul_simplify : Real.sqrt 18 * Real.sqrt 32 = 24 := 
sorry

end sqrt_mul_simplify_l214_214223


namespace min_top_block_sum_l214_214320

theorem min_top_block_sum : 
  ∀ (assign_numbers : ℕ → ℕ) 
  (layer_1 : Fin 16 → ℕ) (layer_2 : Fin 9 → ℕ) (layer_3 : Fin 4 → ℕ) (top_block : ℕ),
  (∀ i, layer_3 i = layer_2 (i / 2) + layer_2 ((i / 2) + 1) + layer_2 ((i / 2) + 3) + layer_2 ((i / 2) + 4)) →
  (∀ i, layer_2 i = layer_1 (i / 2) + layer_1 ((i / 2) + 1) + layer_1 ((i / 2) + 3) + layer_1 ((i / 2) + 4)) →
  (top_block = layer_3 0 + layer_3 1 + layer_3 2 + layer_3 3) →
  top_block = 40 :=
sorry

end min_top_block_sum_l214_214320


namespace coin_order_l214_214197

-- Define the coins as an inductive data type.
inductive Coin
| A | B | C | D | E | F
deriving DecidableEq, Repr

-- Define the conditions as predicates.
def is_not_covered (x : Coin) (xs : List Coin) : Prop := ∀ (c : Coin), c ∉ xs

def is_covered (x : Coin) (covered_by : List Coin) (xs : List Coin) : Prop := 
  x ∈ covered_by ∧ ∀ c, (c ∈ xs → c = x ∨ c ∈ covered_by)

def covers (x : Coin) (covered_coins : List Coin) : Prop :=
  ∀ c, c ∈ covered_coins → c = x ∨ Coin.F ∈ covered_coins

-- Define the specific conditions.
def conditions : Prop :=
  is_not_covered Coin.F [Coin.A, Coin.B, Coin.C, Coin.D, Coin.E] ∧
  is_covered Coin.B [Coin.F] [Coin.A, Coin.B, Coin.C, Coin.D, Coin.E] ∧
  is_covered Coin.C [Coin.F, Coin.D] [Coin.A, Coin.B, Coin.C, Coin.D, Coin.E] ∧
  covers Coin.D [Coin.A, Coin.C] ∧
  is_covered Coin.E [Coin.F] [Coin.A] ∧
  ∃ C, Coin.A ∈ C ∧ ∀ c, c ∈ [Coin.B, Coin.C, Coin.D, Coin.E, Coin.F] → Coin.A <> c

-- The theorem statement with the correct order.
theorem coin_order : conditions → [Coin.F, Coin.D, Coin.E, Coin.C, Coin.B, Coin.A] = [Coin.F, Coin.D, Coin.E, Coin.C, Coin.B, Coin.A] :=
by
  sorry

end coin_order_l214_214197


namespace tangent_line_eqn_minimum_value_in_interval_l214_214483

noncomputable def e : ℝ := Real.exp 1

def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_eqn (x : ℝ) (hx : x = e) :
  let y := 2 * (x - e) + e in y = 2 * x - e := by
  sorry

theorem minimum_value_in_interval (a : ℝ) (h_a : a > 1 / (2 * e)) :
  (if a >= 1 / e then f a = a * Real.log a
   else f (1 / e) = -1 / e) := by
  sorry

end tangent_line_eqn_minimum_value_in_interval_l214_214483


namespace interchanged_digit_multiple_of_sum_l214_214110

theorem interchanged_digit_multiple_of_sum (n a b : ℕ) 
  (h1 : n = 10 * a + b) 
  (h2 : n = 3 * (a + b)) 
  (h3 : 1 ≤ a) (h4 : a ≤ 9) 
  (h5 : 0 ≤ b) (h6 : b ≤ 9) : 
  10 * b + a = 8 * (a + b) := 
by 
  sorry

end interchanged_digit_multiple_of_sum_l214_214110


namespace DM_perp_AC_l214_214620

variable (A B C D K L M : Type)
variable [parallelogram ABCD] (circumcircle_triangle_ABC: circle ABC) (intersect_AD: K ∈ circumcircle_triangle_ABC ∩ AD) (intersect_CD: L ∈ circumcircle_triangle_ABC ∩ CD)
variable (midpoint_arc_KL: M is_midpoint_arс KL that_does_not_contain_B)

theorem DM_perp_AC 
  (ABCD : parallelogram A B C D) 
  (circumcircle_ABC_intersect_AD: intersection (circumcircle_triangle_ABC A B C) (segment A D)) 
  (circumcircle_ABC_intersect_CD: intersection (circumcircle_triangle_ABC A B C) (segment C D)) 
  (M_midpoint_not_contain_B : midpoint_arc_not_contain_B M K L B) :
  perpendicular (line D M) (line A C) := 
sorry

end DM_perp_AC_l214_214620


namespace flight_to_Phoenix_time_l214_214304

noncomputable def V := 425  -- Speed of the plane in still air in mph
def wind_speed := 25  -- Speed of the wind in mph
def distance := 900  -- Distance from Dallas to Phoenix in miles
def time_back := 2  -- Time taken for the return flight in hours

theorem flight_to_Phoenix_time : 
  ∃ T, distance = (V - wind_speed) * T ∧ T = 2.25 :=
by
  let effective_speed_to := V - wind_speed
  let eq1 := distance = effective_speed_to * 2.25
  exact ⟨2.25, eq1, rfl⟩

end flight_to_Phoenix_time_l214_214304


namespace geometry_problem_l214_214295

-- Given a quarter circle AOB with semicircles ACO and OCB intersecting at C,
-- Prove the following properties:
theorem geometry_problem 
    (O A B C : Point)
    (quarter_circle : Circle O (dist O A))
    (semicircle_ACO : Semicircle O A C)
    (semicircle_OCB : Semicircle O C B)
    (H1 : dist O A = dist O B)
    (H2 : dist A C = dist C B) : 
    (∠ AOC = ∠ COB) ∧ collinear A C B ∧ arc_eq quarter_circle A C = arc_eq quarter_circle C B := 
sorry

end geometry_problem_l214_214295


namespace barbara_total_candies_l214_214340

-- Condition: Barbara originally has 9 candies.
def C1 := 9

-- Condition: Barbara buys 18 more candies.
def C2 := 18

-- Question (proof problem): Prove that the total number of candies Barbara has is 27.
theorem barbara_total_candies : C1 + C2 = 27 := by
  -- Proof steps are not required, hence using sorry.
  sorry

end barbara_total_candies_l214_214340


namespace find_n_x_y_l214_214179

theorem find_n_x_y (n x y : ℝ) :
  let seq_avg := (3 + 16 + 33 + (n + 1) + x + y) / 6
  in seq_avg = 25 → n + x + y = 97 :=
by
  sorry

end find_n_x_y_l214_214179


namespace shaded_area_of_four_circles_l214_214534

open Real

noncomputable def area_shaded_region (r : ℝ) (num_circles : ℕ) : ℝ :=
  let area_quarter_circle := (π * r^2) / 4
  let area_triangle := (r * r) / 2
  let area_one_checkered_region := area_quarter_circle - area_triangle
  let num_checkered_regions := num_circles * 2
  num_checkered_regions * area_one_checkered_region

theorem shaded_area_of_four_circles : area_shaded_region 5 4 = 50 * (π - 2) :=
by
  sorry

end shaded_area_of_four_circles_l214_214534


namespace tickets_difference_l214_214733

-- Definitions of conditions
def tickets_won : Nat := 19
def tickets_for_toys : Nat := 12
def tickets_for_clothes : Nat := 7

-- Theorem statement: Prove that the difference between tickets used for toys and tickets used for clothes is 5
theorem tickets_difference : (tickets_for_toys - tickets_for_clothes = 5) := by
  sorry

end tickets_difference_l214_214733


namespace ratio_a_d_l214_214944

variables (a b c d : ℕ)

-- Given conditions
def ratio_ab := 8 / 3
def ratio_bc := 1 / 5
def ratio_cd := 3 / 2
def b_value := 27

theorem ratio_a_d (h₁ : a / b = ratio_ab)
                  (h₂ : b / c = ratio_bc)
                  (h₃ : c / d = ratio_cd)
                  (h₄ : b = b_value) :
  a / d = 4 / 5 :=
sorry

end ratio_a_d_l214_214944


namespace probability_XOXOXOX_l214_214436

theorem probability_XOXOXOX :
  let X := 4;
      O := 3;
      total_positions := 7;
      specific_arrangement := 1;
      total_arrangements := Nat.choose total_positions X in
  1 / total_arrangements = 1 / 35 := by
  sorry

end probability_XOXOXOX_l214_214436


namespace radius_of_outer_circle_l214_214631

theorem radius_of_outer_circle (C_inner : ℝ) (width : ℝ) (h : C_inner = 880) (w : width = 25) :
  ∃ r_outer : ℝ, r_outer = 165 :=
by
  have r_inner := C_inner / (2 * Real.pi)
  have r_outer := r_inner + width
  use r_outer
  sorry

end radius_of_outer_circle_l214_214631


namespace faster_with_bicycle_l214_214739

theorem faster_with_bicycle (D v V : ℝ) (h_speed: V > v) (h_approx: (D / (2 * V) + D / (2 * v)) ≈ (1 / 2) * (D / v)) : V ≫ v :=
by
  sorry

end faster_with_bicycle_l214_214739


namespace part1_part2_l214_214552

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Given conditions:
-- a_3 = 7 when S_5 = 35 (from the given properties of arithmetic sequences)
axiom h1 : 5 * a 3 = 35  
def a_3 := (5 * a 3 = 35 : Prop)

-- a_1, a_4, a_13 form a geometric sequence
axiom h2 : (a 4) ^ 2 = (a 1) * (a 13)
def a_4_geometric := ((a 4) ^ 2 = (a 1) * (a 13) : Prop)

noncomputable def a_n_formula : ℕ → ℤ := λ n, 2 * n + 1

-- We declare the sum of the first n terms of the sequence S
noncomputable def S (n : ℕ) : ℕ := n * (n + 2) 

-- Given the definition of T_n
noncomputable def T (n : ℕ) : ℚ := (3 / 4) - ((2 * n + 3) / (2 * (n + 1) * (n + 2)))

-- To prove: the sum of the first n terms of the sequence {1 / S_n} is 
-- T_n = 3 / 4 - (2n + 3) / (2(n + 1)(n + 2))

theorem part1 : ∀ n, a n = a_n_formula n := sorry

theorem part2 : ∀ n, ∑ i in finset.range n, (1 : ℚ) / S i = T n := sorry

end part1_part2_l214_214552


namespace probability_XOXOXOX_l214_214412

theorem probability_XOXOXOX (arrangement : list char) 
  (h_len : arrangement.length = 7) 
  (h_X_count : arrangement.count 'X' = 4) 
  (h_O_count : arrangement.count 'O' = 3) :
  let total_arrangements := nat.choose 7 4 in 
  let favorable_outcomes := 1 in
  favorable_outcomes / total_arrangements = 1 / 35 :=
by
  -- proof
  sorry

end probability_XOXOXOX_l214_214412


namespace circle_condition_m_l214_214883

theorem circle_condition_m (m : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2 * x + m = 0) → m < 1 := 
by
  sorry

end circle_condition_m_l214_214883


namespace marge_funds_for_fun_l214_214185

-- Definitions based on given conditions
def lottery_amount : ℕ := 12006
def taxes_paid : ℕ := lottery_amount / 2
def remaining_after_taxes : ℕ := lottery_amount - taxes_paid
def student_loans_paid : ℕ := remaining_after_taxes / 3
def remaining_after_loans : ℕ := remaining_after_taxes - student_loans_paid
def savings : ℕ := 1000
def remaining_after_savings : ℕ := remaining_after_loans - savings
def stock_market_investment : ℕ := savings / 5
def remaining_after_investment : ℕ := remaining_after_savings - stock_market_investment

-- The proof goal
theorem marge_funds_for_fun : remaining_after_investment = 2802 :=
sorry

end marge_funds_for_fun_l214_214185


namespace angle_in_third_quadrant_l214_214100

theorem angle_in_third_quadrant (α : ℝ) (k : ℤ) (h : π + 2 * k * π < α ∧ α < 3 * π / 2 + 2 * k * π) :
  ∃ m : ℤ, -π - 2 * m * π < π / 2 - α ∧ (π / 2 - α) < -π / 2 - 2 * m * π :=
by
  -- Lean users note: The proof isn't required here, just setting up the statement as instructed.
  sorry

end angle_in_third_quadrant_l214_214100


namespace special_op_2_3_eq_1_some_expression_for_3_2_l214_214879

-- Definitions for the custom operation and conditions
def special_op (a b : ℕ) : ℕ := 2 * a - 3 * b + a * b

theorem special_op_2_3_eq_1 :
  special_op 2 3 = 1 :=
by
  sorry

theorem some_expression_for_3_2 :
  special_op 3 2 + 1 = 7 :=
by
  have h1 : special_op 3 2 = 6 := by
    sorry
  have h2 : 6 + 1 = 7 := by
    sorry
  h2

end special_op_2_3_eq_1_some_expression_for_3_2_l214_214879


namespace probability_XOXOXOX_l214_214409

theorem probability_XOXOXOX (arrangement : list char) 
  (h_len : arrangement.length = 7) 
  (h_X_count : arrangement.count 'X' = 4) 
  (h_O_count : arrangement.count 'O' = 3) :
  let total_arrangements := nat.choose 7 4 in 
  let favorable_outcomes := 1 in
  favorable_outcomes / total_arrangements = 1 / 35 :=
by
  -- proof
  sorry

end probability_XOXOXOX_l214_214409


namespace factor_expression_l214_214376

theorem factor_expression (a b c : ℝ) :
  let num := (a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3
  let denom := (a - b)^3 + (b - c)^3 + (c - a)^3
  (denom ≠ 0) →
  num / denom = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end factor_expression_l214_214376


namespace cricket_runs_l214_214685

theorem cricket_runs (A B C : ℕ) (h1 : A / B = 1 / 3) (h2 : B / C = 1 / 5) (h3 : A + B + C = 95) : C = 75 :=
by
  -- Skipping proof details
  sorry

end cricket_runs_l214_214685


namespace math_problem_l214_214167

noncomputable def compute_value (c d : ℝ) : ℝ := 100 * c + d

-- Problem statement as a theorem
theorem math_problem
  (c d : ℝ)
  (H1 : ∀ x : ℝ, (x + c) * (x + d) * (x + 10) = 0 → x = -c ∨ x = -d ∨ x = -10)
  (H2 : ∀ x : ℝ, (x + 3 * c) * (x + 5) * (x + 8) = 0 → (x = -4 ∧ ∀ y : ℝ, y ≠ -4 → (y + d) * (y + 10) ≠ 0))
  (H3 : c ≠ 4 / 3 → 3 * c = d ∨ 3 * c = 10) :
  compute_value c d = 141.33 :=
by sorry

end math_problem_l214_214167


namespace rectangle_length_l214_214643

-- Definitions: width, perimeter, area
variables (w l : ℝ)
variables (P : ℝ := 2 * l + 2 * w)

-- Conditions
def condition1 : Prop := P = 5 * w
def condition2 : Prop := l * w = 150

-- Theorem statement: Proving the length is 15
theorem rectangle_length :
  condition1 → condition2 → l = 15 :=
by
  sorry

end rectangle_length_l214_214643


namespace num_divisors_of_64m4_l214_214038

-- A positive integer m such that 120 * m^3 has 120 divisors
def has_120_divisors (m : ℕ) : Prop := (m > 0) ∧ ((List.range (120 * m^3 + 1)).filter (λ d, (120 * m^3) % d = 0)).length = 120

-- Prove that if such an m exists, then 64 * m^4 has 675 divisors
theorem num_divisors_of_64m4 (m : ℕ) (h : has_120_divisors m) : ((List.range (64 * m^4 + 1)).filter (λ d, (64 * m^4) % d = 0)).length = 675 :=
by
  sorry

end num_divisors_of_64m4_l214_214038


namespace hypergeom_distribution_expected_value_X_independence_test_l214_214704

-- Defining the hypergeometric distribution and its expectation
noncomputable def hypergeom_dist := 
  let n := 2 in
  let K := 6 in
  let N := 25 in
  let P_X_0 := (19.choose 2) / (25.choose 2 : ℚ) in
  let P_X_1 := (19.choose 1 * 6.choose 1) / (25.choose 2 : ℚ) in
  let P_X_2 := (6.choose 2) / (25.choose 2 : ℚ) in
  (P_X_0, P_X_1, P_X_2)

noncomputable def expectation_X : ℚ := 
  let P_X_0 := (19.choose 2) / (25.choose 2 : ℚ) in
  let P_X_1 := (19.choose 1 * 6.choose 1) / (25.choose 2 : ℚ) in
  let P_X_2 := (6.choose 2) / (25.choose 2 : ℚ) in
  0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2

-- Defining Chi-Square Testing for independence
noncomputable def chi_square_test :=
  let a := 18 in
  let b := 7 in
  let c := 6 in
  let d := 19 in
  let n := 50 in
  let χ² := n * (a * d - b * c)^2 / (24 * 26 * 25 * 25 : ℚ) in
  χ²

-- Tests for independence 
def test_independence :=
  let critical_value := 10.828 in
  let χ² := χ²_test in
  χ² > critical_value

-- Lean theorem proving the results
theorem hypergeom_distribution :
  let (P_X_0, P_X_1, P_X_2) := hypergeom_dist in
  P_X_0 = 57 / 100 ∧ P_X_1 = 19 / 50 ∧ P_X_2 = 1 / 20 := sorry

theorem expected_value_X : 
  expectation_X = 12 / 25 := sorry

theorem independence_test :
  test_independence := sorry

end hypergeom_distribution_expected_value_X_independence_test_l214_214704


namespace range_scaling_transformation_l214_214003

theorem range_scaling_transformation (x y : ℝ) (θ : ℝ) 
  (h1 : x = 2 * cos θ) 
  (h2 : y = 4 * sin θ) : 
  -4 ≤ (√3 * x + (1 / 2) * y) ∧ (√3 * x + (1 / 2) * y) ≤ 4 :=
by sorry

end range_scaling_transformation_l214_214003


namespace limit_at_1_f_at_1_1_l214_214011

noncomputable def f (x : ℝ) : ℝ :=
  if x = 1 then 0 else (x^2 - 4*x + 3) / (x - 1)

theorem limit_at_1 :
  tendsto (λ x : ℝ, if x = 1 then 0 else (x^2 - 4*x + 3) / (x - 1)) (𝓝 1) (𝓝 (-2)) := by
  sorry

theorem f_at_1_1 :
  f 1.1 = -1.9 := by
  sorry

end limit_at_1_f_at_1_1_l214_214011


namespace fraction_sum_l214_214743

theorem fraction_sum : (1 / 4 : ℚ) + (3 / 8) = 5 / 8 :=
by
  sorry

end fraction_sum_l214_214743


namespace algebraic_expression_value_l214_214452

theorem algebraic_expression_value (x y : ℝ) (h1 : x * y = -2) (h2 : x + y = 4) : x^2 * y + x * y^2 = -8 := 
by
  sorry

end algebraic_expression_value_l214_214452


namespace water_flow_restrictor_l214_214726

theorem water_flow_restrictor :
  let original_flow_rate := 5
  let reduced_flow_rate := 2
  let six_tenths_original := 0.6 * original_flow_rate
  six_tenths_original - reduced_flow_rate = 1 := by
  let original_flow_rate := 5
  let reduced_flow_rate := 2
  let six_tenths_original := 0.6 * original_flow_rate
  show six_tenths_original - reduced_flow_rate = 1
from sorry

end water_flow_restrictor_l214_214726


namespace collinear_X_Y_Z_l214_214571

-- Definitions of points and segments
variables (A1 B1 A2 B2 A3 B3 X Y Z : Type) 
variables [Segment A1 B1] [Segment A2 B2] [Segment A3 B3]

-- Conditions
variable (parallel_segments : parallel (Segment A1 B1) (Segment A2 B2) ∧ 
                                 parallel (Segment A2 B2) (Segment A3 B3) ∧ 
                                 parallel (Segment A3 B3) (Segment A1 B1))
variable (X_intersec : intersection (Line A1 A2) (Line B1 B2) = X)
variable (Y_intersec : intersection (Line A1 A3) (Line B1 B3) = Y)
variable (Z_intersec : intersection (Line A2 A3) (Line B2 B3) = Z)

-- Goal/Question
theorem collinear_X_Y_Z : collinear X Y Z :=
sorry

end collinear_X_Y_Z_l214_214571


namespace deborah_international_letters_l214_214357

theorem deborah_international_letters (standard_postage : ℝ) 
                                      (additional_charge : ℝ) 
                                      (total_letters : ℕ) 
                                      (total_cost : ℝ) 
                                      (h_standard_postage: standard_postage = 1.08)
                                      (h_additional_charge: additional_charge = 0.14)
                                      (h_total_letters: total_letters = 4)
                                      (h_total_cost: total_cost = 4.60) :
                                      ∃ (x : ℕ), x = 2 :=
by
  sorry

end deborah_international_letters_l214_214357


namespace inequality1_inequality2_l214_214886

theorem inequality1 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / a^3) + (1 / b^3) + (1 / c^3) + a * b * c ≥ 2 * Real.sqrt 3 :=
by
  sorry

theorem inequality2 (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) :
  (1 / A) + (1 / B) + (1 / C) ≥ 9 / Real.pi :=
by
  sorry

end inequality1_inequality2_l214_214886


namespace log_condition_l214_214627

noncomputable def log : ℝ → ℝ → ℝ := λ a x, Real.log x / Real.log a

theorem log_condition (a x : ℝ) (hx : x ∈ set.Ici 2) (ha : a ≠ 1) : 
  (∀ x, x ∈ set.Ici 2 → (|log a x| > 1)) → a < 2 := 
begin
  sorry
end

end log_condition_l214_214627


namespace center_on_line_y_eq_sqrt3x_common_point_range_common_chord_length_external_common_tangent_len_l214_214461

def circle_O (x y : ℝ) := x^2 + y^2 = 1
def circle_Ck (x y k : ℝ) := (x - k)^2 + (y - sqrt(3) * k)^2 = 4

theorem center_on_line_y_eq_sqrt3x (k : ℝ) :
  (circle_Ck k (sqrt(3) * k) k) :=
by sorry

theorem common_point_range (k : ℝ) :
  (∃ x y : ℝ, circle_O x y ∧ circle_Ck x y k) →
  (1/2 ≤ |k| ∧ |k| ≤ 3/2) :=
by sorry

theorem common_chord_length (k : ℝ) :
  (∃ x y : ℝ, circle_O x y ∧ circle_Ck x y k ∧ -- Add common chord length condition here, need clarification
    (x, y) coincidence point) →
  (k = 1 ∨ k = -1 ∨ k = 3/4 ∨ k = -3/4) :=
by sorry

theorem external_common_tangent_len (k : ℝ) :
  (k = 3/2 ∨ k = -3/2) →
  external_common_tangent_length 2 (k, y) = 2*sqrt(2) :=
by sorry

end center_on_line_y_eq_sqrt3x_common_point_range_common_chord_length_external_common_tangent_len_l214_214461


namespace find_sum_u_l214_214935

noncomputable def g (x : ℚ) : ℚ := (3 * x)^2 + 3 * x + 3

theorem find_sum_u :
  (∑ u in finset.filter (λ u, g (3 * u) = 11) (finset.range 10), u) = -1 / 9 :=
by
  sorry

end find_sum_u_l214_214935


namespace Marika_father_age_l214_214590

theorem Marika_father_age :
  ∃ y, y = 2006 + 10 ∧ (50 + (y - 2006) = 3 * (10 + (y - 2006))) :=
by
  use 2016
  simp
  sorry

end Marika_father_age_l214_214590


namespace odd_function_condition_l214_214994

theorem odd_function_condition (f : ℝ → ℝ) (a b : ℝ) (h : ∀ x, f x = x * abs(x + a) + b) :
  (∀ x, f (-x) = -f x) ↔ a^2 + b^2 = 0 := by
sorry

end odd_function_condition_l214_214994


namespace usb_drives_total_available_space_l214_214760

def available_space (total_space : ℝ) (percent_used : ℝ) : ℝ :=
  total_space * (1 - percent_used)

theorem usb_drives_total_available_space :
  let usb64 := 64.0
  let usb128 := 128.0
  let usb256 := 256.0
  let used64 := 0.60
  let used128 := 0.75
  let used256 := 0.80
  available_space usb64 used64 +
  available_space usb128 used128 +
  available_space usb256 used256 =
  108.8 :=
by
  let usb64 := 64.0
  let usb128 := 128.0
  let usb256 := 256.0
  let used64 := 0.60
  let used128 := 0.75
  let used256 := 0.80
  show available_space usb64 used64 +
       available_space usb128 used128 +
       available_space usb256 used256 = 108.8
  sorry

end usb_drives_total_available_space_l214_214760


namespace math_problem_l214_214576

noncomputable def f (x a b : ℝ) : ℝ := x + a * x^2 + b * Real.log x

theorem math_problem
  (a b : ℝ)
  (h1 : f 1 a b = 0)
  (h2 : (f 1 a b).deriv 1 = 2) :
  a = -1 ∧ b = 3 ∧ ∀ x : ℝ, f x a b ≤ 2 * x - 2 :=
by
  sorry

end math_problem_l214_214576


namespace largest_interesting_number_is_96433469_l214_214277

def interesting_number (n : Nat) : Prop :=
  let digits := Nat.digits 10 n
  ∀ i, 1 ≤ i ∧ i < Array.size digits - 1 → digits.get ⟨i, Nat.lt_of_le_of_lt (Nat.succ_le_of_lt (Array.getLt' _ _ _)) (Nat.pred_lt')⟩ * 2 <
    digits.get ⟨i - 1, Nat.sub_lt (Nat.le_refl 1) zero_lt_one⟩ + 
    digits.get ⟨i + 1, Nat.add_lt_add (Nat.lt_of_le_of_lt (Nat.succ_le_of_lt (Array.getLt' _ _ _)) (Nat.pred_lt')) Nat.one_lt_succ_succ_tsub⟩

theorem largest_interesting_number_is_96433469 : 
  ∀ n, interesting_number n → n ≤ 96433469 :=
sorry

end largest_interesting_number_is_96433469_l214_214277


namespace sierpinski_gasket_area_sum_l214_214813

noncomputable def totalRemovedArea (n : ℕ) : ℝ := 
  ∑ k in Finset.range n, (3 : ℝ)^(k-1) * (Real.sqrt 3) / (4^(k+1))

theorem sierpinski_gasket_area_sum (n : ℕ) : 
  totalRemovedArea n = ∑ k in Finset.range n, (3 : ℝ)^(k-1) * (Real.sqrt 3) / (4 ^ (k+1)) :=
by
  sorry

end sierpinski_gasket_area_sum_l214_214813


namespace total_amount_spent_l214_214871
-- Since we need broader imports, we include the whole Mathlib library

-- Definition of the prices of each CD and the quantity purchased
def price_the_life_journey : ℕ := 100
def price_a_day_a_life : ℕ := 50
def price_when_you_rescind : ℕ := 85
def quantity_purchased : ℕ := 3

-- Tactic to calculate the total amount spent
theorem total_amount_spent : (price_the_life_journey * quantity_purchased) + 
                             (price_a_day_a_life * quantity_purchased) + 
                             (price_when_you_rescind * quantity_purchased) 
                             = 705 := by
  sorry

end total_amount_spent_l214_214871


namespace total_consumer_installment_credit_l214_214734

-- Conditions
def auto_instalment_credit (C : ℝ) : ℝ := 0.2 * C
def auto_finance_extends_1_third (auto_installment : ℝ) : ℝ := 57
def student_loans (C : ℝ) : ℝ := 0.15 * C
def credit_card_debt (C : ℝ) (auto_installment : ℝ) : ℝ := 0.25 * C
def other_loans (C : ℝ) : ℝ := 0.4 * C

-- Correct Answer
theorem total_consumer_installment_credit (C : ℝ) :
  auto_instalment_credit C / 3 = auto_finance_extends_1_third (auto_instalment_credit C) ∧
  student_loans C = 80 ∧
  credit_card_debt C (auto_instalment_credit C) = auto_instalment_credit C + 100 ∧
  credit_card_debt C (auto_instalment_credit C) = 271 →
  C = 1084 := 
by
  sorry

end total_consumer_installment_credit_l214_214734


namespace product_of_sum_and_reciprocal_nonneg_l214_214602

theorem product_of_sum_and_reciprocal_nonneg (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 :=
by
  sorry

end product_of_sum_and_reciprocal_nonneg_l214_214602


namespace tangent_line_circle_l214_214519

noncomputable def isTangentToCircle (a : ℝ) : Prop :=
  let line := λ x y : ℝ, x + y + a
  let circle := λ x y : ℝ, (x - a) ^ 2 + y ^ 2 - 2
  ∀ x y : ℝ, circle x y = 0 → (∃ x y : ℝ, line x y = 0 ∧ (x, y) = (a, 0))

theorem tangent_line_circle (a : ℝ) (h : isTangentToCircle a): a = 1 ∨ a = -1 :=
  sorry

end tangent_line_circle_l214_214519
