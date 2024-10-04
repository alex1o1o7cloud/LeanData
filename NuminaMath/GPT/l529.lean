import Mathlib

namespace max_value_of_f_l529_529843

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 5) * Real.sin (x + Real.pi / 3) + Real.cos (x - Real.pi / 6)

theorem max_value_of_f : ∃ x, f x ≤ 6 / 5 :=
sorry

end max_value_of_f_l529_529843


namespace simplify_fraction_l529_529783

theorem simplify_fraction (a b : ℝ) (h : a ≠ b): 
  (a - b) / a / (a - (2 * a * b - b^2) / a) = 1 / (a - b) :=
by
  sorry

end simplify_fraction_l529_529783


namespace polynomial_sum_of_squares_l529_529205

open Polynomial

noncomputable def positive_semidefinite (f : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ eval x f

theorem polynomial_sum_of_squares (f : Polynomial ℝ)
  (h : positive_semidefinite f) :
  ∃ n : ℕ, ∃ f_i : Fin n → Polynomial ℝ, f = ∑ i, (f_i i)^2 := 
sorry

end polynomial_sum_of_squares_l529_529205


namespace open_lock_possible_l529_529771

-- Define the key orientation as an inductive type
inductive Orientation
| H  -- Horizontal
| V  -- Vertical

-- Define the grid type
def Grid : Type := list (list Orientation)

-- Define a function to toggle a key and the corresponding row and column in the grid
def toggle (g : Grid) (i j : ℕ) : Grid :=
  sorry  -- Implementation of toggle

-- Define a predicate that checks if all keys in the grid are vertically oriented
def all_vertical (g : Grid) : Prop :=
  ∀ row, row ∈ g → ∀ key, key ∈ row → key = Orientation.V

-- The proof problem statement
theorem open_lock_possible (g : Grid) : ∃ seq : list (ℕ × ℕ), all_vertical (seq.foldl (λ g ij, toggle g ij.1 ij.2) g) :=
  sorry

end open_lock_possible_l529_529771


namespace largest_x_eq_48_div_7_l529_529110

theorem largest_x_eq_48_div_7 :
  ∃ x : ℝ, (⟨floor x / x⟩ = 7 / 8) ∧ (x = 48 / 7) := 
begin
  sorry
end

end largest_x_eq_48_div_7_l529_529110


namespace bus_no_251_probability_l529_529800

theorem bus_no_251_probability :
  let x := Uniform 0 5, y := Uniform 0 7 in
  Probability (y < x) = 5 / 14 :=
begin
  sorry
end

end bus_no_251_probability_l529_529800


namespace maximum_lambda_for_inequality_l529_529586

noncomputable def max_lambda : ℝ := sqrt 2 - 1 / 2

def inequality_holds (λ : ℝ) (a b c : ℝ) : Prop := ab + b^2 + c^2 ≥ λ * (a + b) * c

def condition (a b c : ℝ) : Prop := b + c ≥ a

theorem maximum_lambda_for_inequality :
  (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → condition a b c → inequality_holds max_lambda a b c) ∧
  (∀ λ, (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → condition a b c → inequality_holds λ a b c) → λ ≤ max_lambda) :=
by
  sorry

end maximum_lambda_for_inequality_l529_529586


namespace units_digit_seven_l529_529637

noncomputable def units_digit (n : ℕ) (x : ℝ) : ℕ :=
  ((x^(2^n) + x^(-(2^n))).toInt % 10).natAbs

theorem units_digit_seven (x : ℝ) (h1 : x ≠ 0) (h2 : x + x⁻¹ = 3) (n : ℕ) (hn : 0 < n) :
  units_digit n x = 7 :=
by
  sorry

end units_digit_seven_l529_529637


namespace find_z_coordinate_l529_529411

noncomputable def line_through_points (p1 p2 : ℝ × ℝ × ℝ) (t : ℝ) : ℝ × ℝ × ℝ :=
  (p1.1 + t * (p2.1 - p1.1), p1.2 + t * (p2.2 - p1.2), p1.3 + t * (p2.3 - p1.3))

theorem find_z_coordinate :
  let p1 := (1, 3, 2)
  let p2 := (4, 2, -1)
  let t := (7 - p1.1) / (p2.1 - p1.1)
  let p := line_through_points p1 p2 t
  p.1 = 7 → p.3 = -4 :=
by
  intros p1 p2 t p p_x_eq_7
  let z := p.3
  sorry

end find_z_coordinate_l529_529411


namespace digit_7_appears_602_times_l529_529224

theorem digit_7_appears_602_times :
  ∑ n in Finset.range 2018, has_digit 7 n = 602 := 
sorry

end digit_7_appears_602_times_l529_529224


namespace find_orange_flowers_killed_l529_529406

theorem find_orange_flowers_killed 
  (red_seeds : ℕ) (yellow_seeds : ℕ) (orange_seeds : ℕ) (purple_seeds : ℕ)
  (red_killed : ℕ) (yellow_killed : ℕ) (purple_killed : ℕ)
  (bouquet_count : ℕ) (flowers_per_bouquet : ℕ) 
  (total_orange_seeds : ℕ) (X : ℕ) :
  red_seeds = 125 ∧ yellow_seeds = 125 ∧ orange_seeds = 125 ∧ purple_seeds = 125 ∧
  red_killed = 45 ∧ yellow_killed = 61 ∧ purple_killed = 40 ∧
  bouquet_count = 36 ∧ flowers_per_bouquet = 9 ∧
  total_orange_seeds = 125 ∧
  (let survived_red := red_seeds - red_killed in
   let survived_yellow := yellow_seeds - yellow_killed in
   let survived_purple := purple_seeds - purple_killed in
   let total_flowers_needed := bouquet_count * flowers_per_bouquet in
   let survived_total := survived_red + survived_yellow + survived_purple in
   let needed_orange := total_flowers_needed - survived_total in
   total_orange_seeds - X = needed_orange) →
  X = 30 :=
by
  intros; sorry

end find_orange_flowers_killed_l529_529406


namespace zeta_inequality_l529_529579

noncomputable def riemann_zeta_function : ℝ → ℝ := sorry
-- Definition of the Riemann Zeta Function

theorem zeta_inequality (s : ℝ) (hs : s > 1) :
  (∑' k:ℕ, 1 / (1 + k^s)) ≥ (riemann_zeta_function s) / (1 + riemann_zeta_function s) :=
sorry

end zeta_inequality_l529_529579


namespace sum_q_p_at_points_l529_529567

def p (x : ℤ) : ℤ := abs x - 3

def q (x : ℤ) : ℤ := -abs x + 1

theorem sum_q_p_at_points :
  (Finset.range (2*5 + 1)).sum (λ i, q (p (i - 5))) = 3 := by
  sorry

end sum_q_p_at_points_l529_529567


namespace largest_x_eq_48_div_7_l529_529111

theorem largest_x_eq_48_div_7 :
  ∃ x : ℝ, (⟨floor x / x⟩ = 7 / 8) ∧ (x = 48 / 7) := 
begin
  sorry
end

end largest_x_eq_48_div_7_l529_529111


namespace max_diff_y_intersections_l529_529052

noncomputable def f (x : ℝ) : ℝ := 2 - x^3 + x^4
noncomputable def g (x : ℝ) : ℝ := 1 + 2x^3 + x^4

theorem max_diff_y_intersections : ∀ x : ℝ, f(x) = g(x) → 0 = (f(x) - g(x)) := by
  intro x
  intro h
  sorry

end max_diff_y_intersections_l529_529052


namespace correct_operation_l529_529736

theorem correct_operation : (3 * a^2 * b^3 - 2 * a^2 * b^3 = a^2 * b^3) ∧ 
                            ¬(a^2 * a^3 = a^6) ∧ 
                            ¬(a^6 / a^2 = a^3) ∧ 
                            ¬((a^2)^3 = a^5) :=
by
  sorry

end correct_operation_l529_529736


namespace jelly_bean_problem_l529_529349

theorem jelly_bean_problem :
  ∀ (total_beans : ℕ) (num_people : ℕ) (last_four_people : ℕ → ℕ) (remaining_beans : ℕ),
    total_beans = 8000 →
    num_people = 10 →
    (∀ i, 7 ≤ i → 10 > i → last_four_people i = 400) →
    remaining_beans = 1600 →
    let total_last_four_beans := last_four_people 7 + last_four_people 8 + last_four_people 9 + last_four_people 10 in
    let total_first_few_beans := total_beans - remaining_beans - total_last_four_beans in
    let num_first_few_people := total_first_few_beans / (2 * 400) in
    num_first_few_people = 6 :=
by
  intros total_beans num_people last_four_people remaining_beans
  assume h1 h2 h3 h4
  let total_last_four_beans := last_four_people 7 + last_four_people 8 + last_four_people 9 + last_four_people 10
  let total_first_few_beans := total_beans - remaining_beans - total_last_four_beans
  let num_first_few_people := total_first_few_beans / (2 * 400)
  have : num_first_few_people = 6 := sorry
  exact this

end jelly_bean_problem_l529_529349


namespace expected_value_full_circles_l529_529025

-- Definition of the conditions
def num_small_triangles (n : ℕ) : ℕ :=
  n^2

def potential_full_circle_vertices (n : ℕ) : ℕ :=
  if n < 3 then 0 else (n - 2) * (n - 1) / 2

def prob_full_circle : ℚ :=
  1 / 729

-- The expected number of full circles formed
def expected_full_circles (n : ℕ) : ℚ :=
  potential_full_circle_vertices n * prob_full_circle

-- The mathematical equivalence to be proved
theorem expected_value_full_circles (n : ℕ) : expected_full_circles n = (n - 2) * (n - 1) / 1458 := 
  sorry

end expected_value_full_circles_l529_529025


namespace number_of_valid_four_digit_integers_l529_529572

theorem number_of_valid_four_digit_integers : 
  let is_valid_sequence (a b c d : ℕ) := 
    a ≠ 0 ∧ 
    (10 * b + c) - (10 * a + b) = 4 ∧
    (10 * c + d) - (10 * b + c) = 4 ∧ 
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    0 ≤ d ∧ d ≤ 9
  in 
    finset.card (finset.filter (λ abcd, let ⟨a, b, c, d⟩ := abcd in is_valid_sequence a b c d) 
                            (finset.range 10 ×ˢ finset.range 10 ×ˢ finset.range 10 ×ˢ finset.range 10)) = 12 :=
by
  sorry

end number_of_valid_four_digit_integers_l529_529572


namespace tangent_line_perpendicular_to_given_line_l529_529499

theorem tangent_line_perpendicular_to_given_line :
  let f (x : ℝ) := x^3 + 3 * x^2 - 1 in
  let df (x : ℝ) := 3 * x^2 + 6 * x in
  ∃ (tangent_line : ℝ → ℝ) (x0 : ℝ),
  (df x0 = -3) ∧
  (tangent_line = λ x, -3 * (x - x0) + f x0) ∧
  (tangent_line = λ x, -(3 * x + 2)) :=
by
  let f : ℝ → ℝ := λ x, x^3 + 3 * x^2 - 1
  let df : ℝ → ℝ := λ x, 3 * x^2 + 6 * x
  existsi λ x : ℝ, -3 * (x + 1) + 1
  existsi -1
  sorry

end tangent_line_perpendicular_to_given_line_l529_529499


namespace conic_section_is_parabola_l529_529824

theorem conic_section_is_parabola:
  ∀ x y : ℝ, (abs (y - 3) = sqrt ((x + 4)^2 + (y - 1)^2)) → (∃ a b c : ℝ, y = a*x^2 + b*x + c) :=
by
  intros x y h
  sorry

end conic_section_is_parabola_l529_529824


namespace lowest_positive_integer_divisible_by_primes_between_10_and_50_l529_529503

def primes_10_to_50 : List ℕ := [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

def lcm_list (lst : List ℕ) : ℕ :=
lst.foldr Nat.lcm 1

theorem lowest_positive_integer_divisible_by_primes_between_10_and_50 :
  lcm_list primes_10_to_50 = 614889782588491410 :=
by
  sorry

end lowest_positive_integer_divisible_by_primes_between_10_and_50_l529_529503


namespace range_of_a_l529_529874

variable (x a : ℝ)

def p : Prop := x^2 - 2 * x - 3 ≥ 0

def q : Prop := x^2 - (2 * a - 1) * x + a * (a - 1) ≥ 0

def sufficient_but_not_necessary (p q : Prop) : Prop := 
  (p → q) ∧ ¬(q → p)

theorem range_of_a (a : ℝ) : (∃ x, sufficient_but_not_necessary (p x) (q a x)) → (0 ≤ a ∧ a ≤ 3) := 
sorry

end range_of_a_l529_529874


namespace find_coefficients_l529_529149

-- Definitions for points and conditions
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (P Q R S P' Q' R' S' : V)

-- Assumptions of midpoints for construction
def midpoint (A B : V) := (1/2 • A + 1/2 • B)

-- Given Conditions: Each point is midpoint of certain points.
axiom Q_midpoint : Q = midpoint P P'
axiom R_midpoint : R = midpoint Q Q'
axiom S_midpoint : S = midpoint R R'
axiom P_midpoint : P = midpoint S S'

-- Prove the required ordered quadruple
theorem find_coefficients :
  P = (1 / 15 : ℝ) • P' + (2 / 15 : ℝ) • Q' + (4 / 15 : ℝ) • R' + (8 / 15 : ℝ) • S' := 
sorry

end find_coefficients_l529_529149


namespace fraction_habitable_earth_l529_529929

theorem fraction_habitable_earth (one_fifth_land: ℝ) (one_third_inhabitable: ℝ)
  (h_land_fraction : one_fifth_land = 1 / 5)
  (h_inhabitable_fraction : one_third_inhabitable = 1 / 3) :
  (one_fifth_land * one_third_inhabitable) = 1 / 15 :=
by
  sorry

end fraction_habitable_earth_l529_529929


namespace apothem_ratio_l529_529421

-- Define the conditions for the rectangle
def rectangle_width : ℝ := 8 / 3
def rectangle_length : ℝ := 8
def rectangle_apothem : ℝ := rectangle_width / 2

-- Define the conditions for the hexagon
def hexagon_side : ℝ := 4 / real.sqrt 3
def hexagon_apothem : ℝ := (hexagon_side * real.sqrt 3) / 2

-- Define the theorem to prove the relationship between the apothems
theorem apothem_ratio :
  (rectangle_apothem / hexagon_apothem) = (2/3) :=
by
  sorry

end apothem_ratio_l529_529421


namespace payment_amount_l529_529200

-- Define exponential growth factor
def e : ℝ := 1.04

-- Define future value of annual payments at the end of 10 years
def S₁₀ (x : ℝ) : ℝ := x * (e^10 - 1) / (e - 1)

-- Define the present value adjusted back to the start of the first year
def PV₁ (x : ℝ) : ℝ := S₁₀ x / e^9

-- Define future value at the beginning of the 29th year for a series of 1500 payments
def Q₂₉ : ℝ := 1500 * (e^15 - 1) / (e - 1)

-- The present value of Q₂₉ adjusted to the 10th year
def PV₁_Q₂₉ : ℝ := Q₂₉ / e^28

theorem payment_amount (x : ℝ) : PV₁ x = PV₁_Q₂₉ → x = 1188 := 
sorry

end payment_amount_l529_529200


namespace harry_drank_last_mile_l529_529912

theorem harry_drank_last_mile :
  ∀ (T D start_water end_water leak_rate drink_rate leak_time first_miles : ℕ),
    start_water = 10 →
    end_water = 2 →
    leak_rate = 1 →
    leak_time = 2 →
    drink_rate = 1 →
    first_miles = 3 →
    T = leak_rate * leak_time →
    D = drink_rate * first_miles →
    start_water - end_water = T + D + (start_water - end_water - T - D) →
    start_water - end_water - T - D = 3 :=
by
  sorry

end harry_drank_last_mile_l529_529912


namespace shortest_side_of_triangle_l529_529985

theorem shortest_side_of_triangle 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_triangle : a + b > c ∧ b + c > a ∧ a + c > b) 
  (h_inequal : a^2 + b^2 > 5 * c^2) :
  c < a ∧ c < b := 
by 
  sorry

end shortest_side_of_triangle_l529_529985


namespace length_of_second_train_l529_529398

noncomputable def speed_in_mps (speed_kmph : ℝ) : ℝ := (speed_kmph * 1000) / 3600

theorem length_of_second_train : 
  ∀ (length_first_train speed_first_train speed_second_train time_crossing : ℝ),
  length_first_train = 350 →
  speed_first_train = 150 →
  speed_second_train = 100 →
  time_crossing = 6 →
  let speed_first_train_mps := speed_in_mps speed_first_train in
  let speed_second_train_mps := speed_in_mps speed_second_train in
  let relative_speed := speed_first_train_mps + speed_second_train_mps in
  let total_distance := relative_speed * time_crossing in
  let length_second_train := total_distance - length_first_train in
  length_second_train = 66.7 := 
by
  intros length_first_train speed_first_train speed_second_train time_crossing 
  intro h1 h2 h3 h4
  let speed_first_train_mps := speed_in_mps speed_first_train
  let speed_second_train_mps := speed_in_mps speed_second_train
  let relative_speed := speed_first_train_mps + speed_second_train_mps
  let total_distance := relative_speed * time_crossing
  let length_second_train := total_distance - length_first_train
  show length_second_train = 66.7 from sorry

end length_of_second_train_l529_529398


namespace estimated_red_light_runners_l529_529722

theorem estimated_red_light_runners (n y : ℕ) (h1 : n = 600) (h2 : y = 180) :
  let p := ((y : ℚ) / n - 1 / 4) * 2 in
  n * p = 60 := 
by
  sorry

end estimated_red_light_runners_l529_529722


namespace min_workers_to_profit_l529_529766

/-- Given a company's daily maintenance fee, worker pay per hour, workers' productivity,
widget selling price, and workday duration, determine the minimum number of workers needed to make a profit --/
theorem min_workers_to_profit 
  (maintenance_fee : ℝ)
  (worker_pay_per_hour : ℝ)
  (worker_productivity : ℝ)
  (widget_selling_price : ℝ)
  (workday_duration : ℝ)
  (n : ℤ) :
  maintenance_fee = 520 →
  worker_pay_per_hour = 18 →
  worker_productivity = 3 →
  widget_selling_price = 4.5 →
  workday_duration = 8 →
  (∃ n : ℤ, n ≥ 15 ∧ 108 * n > 520 + 144 * n) :=
begin
  sorry
end

end min_workers_to_profit_l529_529766


namespace initial_cupcakes_baked_l529_529611

variable (toddAte := 21)       -- Todd ate 21 cupcakes.
variable (packages := 6)       -- She could make 6 packages.
variable (cupcakesPerPackage := 3) -- Each package contains 3 cupcakes.
variable (cupcakesLeft := packages * cupcakesPerPackage) -- Cupcakes left after Todd ate some.

theorem initial_cupcakes_baked : cupcakesLeft + toddAte = 39 :=
by
  -- Proof placeholder
  sorry

end initial_cupcakes_baked_l529_529611


namespace quadratic_always_has_real_roots_find_m_given_difference_between_roots_is_three_l529_529868

-- Part 1: Prove that the quadratic equation always has two real roots for all real m.
theorem quadratic_always_has_real_roots (m : ℝ) :
  let Δ := (m-1)^2 - 4 * (m-2) in 
  Δ ≥ 0 := 
by 
  have Δ := (m-1)^2 - 4 * (m-2);
  suffices Δ_nonneg : (m-3)^2 ≥ 0, from Δ_nonneg;
  sorry

-- Part 2: Given the difference between the roots is 3, find the value of m.
theorem find_m_given_difference_between_roots_is_three (m : ℝ) :
  let x1 := 1,
      x2 := m - 2,
      diff := |x1 - x2| in
  diff = 3 → m = 0 ∨ m = 6 := 
by 
  let x1 := 1,
      x2 := m - 2,
      diff := |x1 - x2|;
  assume h : diff = 3;
  have abs_eq_three : |3 - m| = 3 := by {
      calc |1 - (m - 2)| = ... := sorry
  };
  cases abs_eq_three with
  | inl h₁ => have m_eq_0 : m = 0 := sorry;
              exact Or.inl m_eq_0
  | inr h₂ => have m_eq_6 : m = 6 := sorry;
              exact Or.inr m_eq_6

end quadratic_always_has_real_roots_find_m_given_difference_between_roots_is_three_l529_529868


namespace largest_x_exists_largest_x_largest_real_number_l529_529083

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : x ≤ 48 / 7 :=
sorry

theorem exists_largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  ∃ x, (⌊x⌋ : ℝ) / x = 7 / 8 ∧ x = 48 / 7 :=
sorry

theorem largest_real_number (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  x = 48 / 7 :=
sorry

end largest_x_exists_largest_x_largest_real_number_l529_529083


namespace least_changes_to_unique_sums_l529_529817

def initial_matrix : Matrix (Fin 4) (Fin 4) ℕ :=
  ![![1, 2, 3, 4],
    ![5, 6, 7, 8],
    ![9, 10, 11, 12],
    ![13, 14, 15, 16]]

def altered_matrix : Matrix (Fin 4) (Fin 4) ℕ :=
  ![![1, 2, 3, 4],
    ![5, 6, 7, 8],
    ![9, 10, 11, 12],
    ![13, 14, 15, 21]]

def row_sums (m : Matrix (Fin 4) (Fin 4) ℕ) : Vector ℕ 4 :=
  ⟨(finset.univ.map fin.coe).map (λ i, finset.univ.sum (λ j, m i j)), sorry⟩

def col_sums (m : Matrix (Fin 4) (Fin 4) ℕ) : Vector ℕ 4 :=
  ⟨(finset.univ.map fin.coe).map (λ j, finset.univ.sum (λ i, m i j)), sorry⟩

theorem least_changes_to_unique_sums :
  1 = 1 := sorry

end least_changes_to_unique_sums_l529_529817


namespace ratio_S5_a5_l529_529168

noncomputable def sequence (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, 3 * S n - 6 = 2 * a n

noncomputable def specific_seq (a : ℕ → ℤ) : Prop :=
  ∀ n, a n = if n = 0 then 6 else -2 * a (n - 1)

theorem ratio_S5_a5 (a : ℕ → ℤ) (S : ℕ → ℤ)
  (hseq : sequence a S)
  (ha_initial : specific_seq a) :
  S 5 / a 5 = 11 / 16 := sorry

end ratio_S5_a5_l529_529168


namespace inclination_angle_x_plus_3_eq_0_l529_529711

theorem inclination_angle_x_plus_3_eq_0 :
  let line_eq := (x + 3 = 0)
  in ∀ x, line_eq → ∃ θ : ℝ, θ = 90 :=
by
  sorry

end inclination_angle_x_plus_3_eq_0_l529_529711


namespace arithmetic_sequence_sum_thirty_l529_529948

-- Definitions according to the conditions
def arithmetic_seq_sums (S : ℕ → ℤ) : Prop :=
  ∃ a d : ℤ, ∀ n : ℕ, S n = a + n * d

-- Main statement we need to prove
theorem arithmetic_sequence_sum_thirty (S : ℕ → ℤ)
  (h1 : S 10 = 10)
  (h2 : S 20 = 30)
  (h3 : arithmetic_seq_sums S) : 
  S 30 = 50 := 
sorry

end arithmetic_sequence_sum_thirty_l529_529948


namespace inscribed_rectangle_area_l529_529429

theorem inscribed_rectangle_area (h a b x : ℝ) (ha_gt_b : a > b) :
  ∃ A : ℝ, A = (b * x / h) * (h - x) :=
by
  sorry

end inscribed_rectangle_area_l529_529429


namespace find_f_neg_2_l529_529164

def f (x : ℝ) : ℝ := sorry -- The actual function f is undefined here.

theorem find_f_neg_2 (h : ∀ x ≠ 0, f (1 / x) + (1 / x) * f (-x) = 2 * x) :
  f (-2) = 7 / 2 :=
sorry

end find_f_neg_2_l529_529164


namespace part_one_part_two_combined_problems_l529_529558

noncomputable def f (x a : ℝ) : ℝ := x^2 - a * x - Real.log x

noncomputable def f_derivative (x a : ℝ) : ℝ :=
(2 * x^2 - a * x - 1) / x

theorem part_one (a : ℝ) (h : f_derivative 1 a = 1) : a = 0 := sorry

noncomputable def f_min (a : ℝ) : ℝ :=
let x₀ := (a + Real.sqrt (a^2 + 8)) / 4 in
f x₀ a

theorem part_two (a : ℝ) (h : a ≥ -1) : f_min a ≤ 3 / 4 + Real.log 2 := sorry

-- Combine both problems into one comprehensive theorem:
theorem combined_problems (a : ℝ) (h1 : f_derivative 1 a = 1) (h2 : a ≥ -1) :
  a = 0 ∧ f_min a ≤ 3 / 4 + Real.log 2 :=
begin
  split,
  { apply part_one,
    exact h1 },
  { apply part_two,
    exact h2 }
end

end part_one_part_two_combined_problems_l529_529558


namespace min_value_3x_4y_l529_529930

theorem min_value_3x_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : 3 / x + 1 / y = 1) : 
  3 * x + 4 * y ≥ 25 :=
sorry

end min_value_3x_4y_l529_529930


namespace complement_intersection_sets_l529_529273

open set

theorem complement_intersection_sets :
  let M := {x : ℝ | |x| < 1},
      N := {y : ℝ | ∃ (x : ℝ), x ∈ M ∧ y = 2^x} in
  compl (M ∩ N) = {z : ℝ | z ≤ 0.5} ∪ {z : ℝ | z ≥ 1} :=
by
  sorry

end complement_intersection_sets_l529_529273


namespace ADC_perimeter_range_l529_529593

noncomputable def perimeter_range (A B C : ℝ) (D : ℝ) : ℝ :=
if A > π / 3 ∧ A < 2 * π / 3 then
  2 * sin A + sqrt 3
else 
  sorry

theorem ADC_perimeter_range
  (A B C : ℝ) (D : ℝ)
  (h1 : B = π / 3)
  (h2 : AC = sqrt 3)
  (h3 : AB = AD)
  (h4 : D ∈ segment ℝ B C) :
  (2 * sqrt 3 < perimeter_range A B C D ∧ 
   perimeter_range A B C D ≤ 2 + sqrt 3) :=
by sorry

end ADC_perimeter_range_l529_529593


namespace three_digit_numbers_satisfying_condition_l529_529738

theorem three_digit_numbers_satisfying_condition :
  ∀ (N : ℕ), (100 ≤ N ∧ N < 1000) →
    ∃ (a b c : ℕ),
      (N = 100 * a + 10 * b + c) ∧ (N = 11 * (a^2 + b^2 + c^2)) 
    ↔ (N = 550 ∨ N = 803) :=
by
  sorry

end three_digit_numbers_satisfying_condition_l529_529738


namespace prob_goal_l529_529563

open MeasureTheory

variables {ξ : ℝ → Measure ℝ}
variables {μ σ : ℝ}

-- Assuming ξ follows a normal distribution N(μ, σ^2)
axiom normal_distribution : ∀ x, ξ x = Measure.normal_pdf μ σ x

-- Given conditions
axiom prob_1 : ξ {x | x < 1} = 0.5
axiom prob_2 : ξ {x | x > 2} = 0.4

-- Goal: Prove P(0 < ξ < 1) = 0.1
theorem prob_goal : ξ {x | 0 < x ∧ x < 1} = 0.1 :=
by
  sorry

end prob_goal_l529_529563


namespace tangent_parabola_line_l529_529123
open Real

theorem tangent_parabola_line (a : ℝ) :
  (∀ x : ℝ, ax^2 + 6 = 2x + 3 → (x ≠ x ∨ ax^2 + 6 = 2x + 3)) → a = 1 / 3 :=
begin
  sorry
end

end tangent_parabola_line_l529_529123


namespace decode_division_problem_l529_529058

theorem decode_division_problem :
  let dividend := 1089708
  let divisor := 12
  let quotient := 90809
  dividend / divisor = quotient :=
by {
  -- Definitions of given and derived values
  let dividend := 1089708
  let divisor := 12
  let quotient := 90809
  -- The statement to prove
  sorry
}

end decode_division_problem_l529_529058


namespace angle_between_a_b_l529_529963

variable {V : Type*} [inner_product_space ℝ V]

def is_unit_vector {V : Type*} [inner_product_space ℝ V] (v : V) : Prop :=
  ⟪v, v⟫ = 1

noncomputable def angle_between_vectors (a b : V) [inner_product_space ℝ V] : ℝ :=
  real.arccos (⟪a, b⟫ / (∥a∥ * ∥b∥))

theorem angle_between_a_b
  (a b : V)
  (h_a : is_unit_vector a)
  (h_b : is_unit_vector b)
  (orth : ⟪a + 3 • b, 4 • a - 3 • b⟫ = 0) :
  angle_between_vectors a b = real.arccos (5 / 9) :=
sorry

end angle_between_a_b_l529_529963


namespace integral_sine_product_eq_zero_l529_529958

open Real

noncomputable def integral_sine_product (α β : ℝ) (hα : 2 * α = tan α) (hβ : 2 * β = tan β) (h_distinct : α ≠ β) : ℝ :=
∫ x in 0..1, sin (α * x) * sin (β * x)

theorem integral_sine_product_eq_zero {α β : ℝ} (hα : 2 * α = tan α) (hβ : 2 * β = tan β) (h_distinct : α ≠ β):
  integral_sine_product α β hα hβ h_distinct = 0 :=
sorry

end integral_sine_product_eq_zero_l529_529958


namespace arrange_logs_in_order_l529_529857

noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 0.8 / Real.log 1.2
noncomputable def c : ℝ := Real.sqrt 1.5

theorem arrange_logs_in_order : b < a ∧ a < c := by
  sorry

end arrange_logs_in_order_l529_529857


namespace basketball_club_boys_l529_529399

theorem basketball_club_boys (B G : ℕ)
  (h1 : B + G = 30)
  (h2 : B + (1 / 3) * G = 18) : B = 12 := 
by
  sorry

end basketball_club_boys_l529_529399


namespace correct_expressions_count_l529_529523

variable {V : Type*} [AddCommGroup V] [VectorSpace ℝ V]
variable (A B C D E F : V)
variable (H : regular_hexagon A B C D E F)

theorem correct_expressions_count : 
  (∃ BC CD EC DC FE ED BA : V, 
    (BC + CD + EC = A - C) ∧ 
    (2*BC + DC = A - C) ∧ 
    (FE + ED = A - C) ∧ 
    (BC - BA = A - C)) → 
  4 := 
  by
    sorry

end correct_expressions_count_l529_529523


namespace incorrect_statement_B_l529_529603

variable {α β γ : Type} [plane α] [plane β] [plane γ]
variables (L : set (line_in α)) -- Lines within the plane α 
variable (h : α ⊥ β) -- Condition given for planes α and β being perpendicular

theorem incorrect_statement_B (AllLinesPerp : ∀ l ∈ L, ⊥ β) : False :=
by
  -- Proof that ∀ l ∈ L, ⊥ β is incorrect 
  sorry

end incorrect_statement_B_l529_529603


namespace true_proposition_l529_529648

def proposition_p := ∀ (x : ℤ), x^2 > x
def proposition_q := ∃ (x : ℝ) (hx : x > 0), x + (2 / x) > 4

theorem true_proposition :
  (¬ proposition_p) ∨ proposition_q :=
by
  sorry

end true_proposition_l529_529648


namespace binomial_expansion_coeff_x_l529_529316

theorem binomial_expansion_coeff_x :
  let term := (1 - 3 * x) ^ 8 in
  polynomial.coeff term 1 = -56 :=
sorry

end binomial_expansion_coeff_x_l529_529316


namespace prob_interval_positive_halfspace_l529_529937
noncomputable theory

open MeasureTheory ProbabilityTheory

-- Let X be a random variable following a normal distribution N(1, σ^2)
variable {σ : ℝ} (hσ : 0 < σ)

-- Definition of the random variable X
def X : MeasureTheory.MeasureSpace ℝ :=
probability_distribution (Normal 1 (σ^2))

-- Given condition: P(0 < X < 1) = 0.3
axiom prob_interval : P({ x | 0 < x ∧ x < 1 }) = 0.3

-- Proof statement: P(0 < X < +∞) = 0.8
theorem prob_interval_positive_halfspace : P({ x | 0 < x }) = 0.8 :=
sorry

end prob_interval_positive_halfspace_l529_529937


namespace remainder_23_pow_2003_mod_7_l529_529367

theorem remainder_23_pow_2003_mod_7 : 23 ^ 2003 % 7 = 4 :=
by sorry

end remainder_23_pow_2003_mod_7_l529_529367


namespace range_of_eccentricity_l529_529562

-- Given conditions
variables (a : ℝ) (hyp_a : a > 0)
variables (P F A : ℝ × ℝ)

-- Definitions corresponding to points and hyperbola properties
def hyperbola (x y : ℝ) : Prop := (x^2) / (a^2) - y^2 = 1
def right_focus : ℝ × ℝ := (0, real.sqrt(a^2 + 1))
def point_A : ℝ × ℝ := (0, -a)
def left_branch (P : ℝ × ℝ) : Prop := P.1 < 0 ∧ hyperbola P.1 P.2
def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Main theorem to prove the range of eccentricity
theorem range_of_eccentricity
  (P : ℝ × ℝ)
  (P_on_left_branch : left_branch P)
  (distance_condition : distance P point_A + distance P right_focus = 7) :
  mul (real.sqrt (a^2 + 1)) a >= real.sqrt 5 / 2 :=
begin
  sorry
end

end range_of_eccentricity_l529_529562


namespace intersection_points_polar_coordinates_max_distance_from_point_P_to_line_l_l529_529946

-- Define the curves and line
def C1_polar (ρ θ : ℝ) := ρ * sin θ = -1
def C2_param (θ : ℝ) := (2 * cos θ, -2 + 2 * sin θ)
def line_l (x y : ℝ) := x - y + 2 = 0

-- First proof for the intersection points in polar coordinates
theorem intersection_points_polar_coordinates:
  ∀ (θ1 θ2 : ℝ),
    C1_polar 2 (-π/6) ∧ C1_polar 2 (7*π/6) ∧
    (C2_param θ1 = (sqrt 3, -1) ∨ C2_param θ1 = (-sqrt 3, -1)) ∧ 
    (C2_param θ2 = (sqrt 3, -1) ∨ C2_param θ2 = (-sqrt 3, -1)) :=
sorry

-- Second proof for the maximum distance from point P to line l
theorem max_distance_from_point_P_to_line_l:
  ∀ (x y : ℝ),
    ∃ (P : ℝ × ℝ) (max_dist : ℝ),
    ∃ (θ : ℝ),
      P = C2_param θ ∧ 
      max_dist = 2 * sqrt 2 + 2 ∧ 
      line_l x y :=
sorry

end intersection_points_polar_coordinates_max_distance_from_point_P_to_line_l_l529_529946


namespace angle_BAC_is_67_5_degrees_l529_529728

-- Define the geometric context
variables (circle : Type) [MetricSpace circle]

-- Assumptions
variable {A B C : circle} -- Points on the circle
variable (O : circle) -- Center of the circle
variable [is_tangent : tangent_point B A] -- B is tangent from A
variable [is_tangent' : tangent_point C A] -- C is tangent from A
variable (arcBC arcCB' : ℝ) -- Lengths of arcs BC and CB'
variable (h_ratio : arcBC / arcCB' = 3 / 5) -- Given ratio of arc lengths

-- Conclusion to prove
theorem angle_BAC_is_67_5_degrees :
  ∠ BAC = 67.5 :=
sorry

end angle_BAC_is_67_5_degrees_l529_529728


namespace florist_rose_count_l529_529769

theorem florist_rose_count (initial_sold_picked : Nat × Nat × Nat) : 
    initial_sold_picked = (37, 16, 19) → 
    let (initial, sold, picked) := initial_sold_picked in
    initial - sold + picked = 40 :=
by
  intro h
  cases initial_sold_picked with x y
  cases y with z w
  simp [*, show x - z + w = 37 - 16 + 19, from rfl]
  sorry

end florist_rose_count_l529_529769


namespace most_likely_units_digit_is_5_l529_529827

-- Define the problem conditions
def in_range (n : ℕ) := 1 ≤ n ∧ n ≤ 8
def Jack_pick (J : ℕ) := in_range J
def Jill_pick (J K : ℕ) := in_range K ∧ J ≠ K

-- Define the function to get the units digit of the sum
def units_digit (J K : ℕ) := (J + K) % 10

-- Define the proposition stating the most likely units digit is 5
theorem most_likely_units_digit_is_5 :
  ∃ (d : ℕ), d = 5 ∧
    (∃ (J K : ℕ), Jack_pick J → Jill_pick J K → units_digit J K = d) :=
sorry

end most_likely_units_digit_is_5_l529_529827


namespace triangle_half_angle_identity_l529_529234

variable {ABC : Triangle}
variable {b c : ℝ} -- sides of the triangle
variable {B C : Angle ℝ} -- angles of the triangle
variable {s : ℝ} -- semi-perimeter of the triangle
variable {r : ℝ} -- inradius of the triangle

-- cotangent of the half-angles in terms of sides and semi-perimeter
def cot_half_angle_B : ℝ := (s - b) / r
def cot_half_angle_C : ℝ := (s - c) / r

theorem triangle_half_angle_identity :
  \(\frac{b - c}{cot_half_angle_B - cot_half_angle_C} = -r \)
:= sorry

end triangle_half_angle_identity_l529_529234


namespace set_intersection_complement_l529_529977

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}

theorem set_intersection_complement :
  P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end set_intersection_complement_l529_529977


namespace kishore_savings_l529_529791

-- Define expenses
def rent := 5000
def milk := 1500
def groceries := 4500
def education := 2500
def petrol := 2000
def miscellaneous := 2500
def electricity := 1200
def water := 800
def dining_out := 3500
def medical := 1000

-- Define total expenses
def total_expenses := rent + milk + groceries + education + petrol + miscellaneous + electricity + water + dining_out + medical

-- Define the fraction of savings, 15%
def savings_rate := 0.15

-- Define Mr. Kishore's monthly salary
noncomputable def monthly_salary := 2350000 / 85

-- Define Savings
noncomputable def savings := savings_rate * monthly_salary

-- Define the theorem to prove
theorem kishore_savings : savings ≈ 4147.06 :=
by
  sorry

end kishore_savings_l529_529791


namespace length_of_paper_is_8_l529_529940

-- Define the conditions

def width_of_paper : ℝ := 72
def volume_of_cube_ft : ℝ := 8
def ft_to_inch : ℝ := 12
def area_of_face_of_cube_in_inch :=
  let side_length_of_cube_ft := volume_of_cube_ft^(1/3)
  let side_length_of_cube_in := side_length_of_cube_ft * ft_to_inch
  side_length_of_cube_in^2

def paper_length : ℝ :=
  area_of_face_of_cube_in_inch / width_of_paper

-- The theorem to be proved
theorem length_of_paper_is_8 :
  paper_length = 8 := 
sorry

end length_of_paper_is_8_l529_529940


namespace salt_mixture_problem_l529_529574

theorem salt_mixture_problem :
  ∃ (m : ℝ), 0.20 = (150 + 0.05 * m) / (600 + m) :=
by
  sorry

end salt_mixture_problem_l529_529574


namespace steve_pencils_left_l529_529679

-- Conditions
def initial_pencils : Nat := 24
def pencils_given_to_Lauren : Nat := 6
def extra_pencils_given_to_Matt : Nat := 3

-- Question: How many pencils does Steve have left?
theorem steve_pencils_left :
  initial_pencils - (pencils_given_to_Lauren + (pencils_given_to_Lauren + extra_pencils_given_to_Matt)) = 9 := by
  -- You need to provide a proof here
  sorry

end steve_pencils_left_l529_529679


namespace length_OS_l529_529765

-- Definitions based on the problem's conditions
def circle (radius : ℝ) := {C : ℝ × ℝ | dist C.1 C.2 = radius}

-- Given conditions from the problem
def O := (0, 0)
def P := (14, 0)
def Q := (10, 0)
def radius_O := 10
def radius_P := 4

-- Use these given conditions to declare that two circles are tangent outside each other
def externally_tangent (O P Q : ℝ × ℝ) (rO rP : ℝ) : Prop :=
  dist O P = rO + rP

-- Define the tangent segment TS based on the problem's description
def common_external_tangent_length (O P : ℝ × ℝ) (rO rP : ℝ) : ℝ :=
  4 * real.sqrt 10

-- Define OS segment based on right triangle properties
def segment_OS_length (O P : ℝ × ℝ) (rO rP : ℝ) : ℝ :=
  real.sqrt (rO ^ 2 + common_external_tangent_length O P rO rP ^ 2)

-- The theorem we need to prove
theorem length_OS : externally_tangent O P Q radius_O radius_P →
  segment_OS_length O P radius_O radius_P = 2 * real.sqrt 65 := 
sorry  -- proof to be filled in

end length_OS_l529_529765


namespace mode_representation_l529_529215

noncomputable def mode_in_histogram (hist : ℕ → ℕ) : ℕ :=
  (argmax hist).fst

theorem mode_representation {hist : ℕ → ℕ} :
  (mode_in_histogram hist) = (argmax hist).fst :=
by
  sorry

end mode_representation_l529_529215


namespace tom_won_whack_a_mole_l529_529033

variable (W : ℕ)  -- let W be the number of tickets Tom won playing 'whack a mole'
variable (won_skee_ball : ℕ := 25)  -- Tom won 25 tickets playing 'skee ball'
variable (spent_on_hat : ℕ := 7)  -- Tom spent 7 tickets on a hat
variable (tickets_left : ℕ := 50)  -- Tom has 50 tickets left

theorem tom_won_whack_a_mole :
  W + 25 + 50 = 57 →
  W = 7 :=
by
  sorry  -- proof goes here

end tom_won_whack_a_mole_l529_529033


namespace quadratic_ratio_l529_529335

theorem quadratic_ratio (b c : ℤ) (h : ∀ x : ℤ, x^2 + 1400 * x + 1400 = (x + b) ^ 2 + c) : c / b = -698 :=
sorry

end quadratic_ratio_l529_529335


namespace fraction_area_enclosed_l529_529004

-- Define a regular hexagon ABCDEF with vertices numbered consecutively.
def regular_hexagon (A B C D E F : Type) (s : ℝ) := 
  -- Conditions for the vertices to form a regular hexagon.
  sorry

-- Define the midpoints G, H, I of sides AB, CD, EF respectively.
def midpoint (A B : Type) : Type := sorry

-- Let G, H, I be the midpoints of sides AB, CD, and EF respectively.
def G := midpoint A B
def H := midpoint C D
def I := midpoint E F

-- Define the function to compute the area of the regular hexagon given the side length s.
def area_hexagon (s : ℝ) : ℝ := (3 * real.sqrt 3 / 2) * (s ^ 2)

-- Define the function to compute the area of the square formed by midpoints G, H, and I given the side length s.
def area_square (s : ℝ) : ℝ := (s / 2) ^ 2

-- Prove that the fraction of the area of hexagon ABCDEF enclosed by the square GHIJ is 1 / (6 * sqrt 3).
theorem fraction_area_enclosed (s : ℝ) : (area_square s) / (area_hexagon s) = 1 / (6 * real.sqrt 3) := 
  sorry

end fraction_area_enclosed_l529_529004


namespace angle_between_vectors_l529_529884

variables (a b c : EuclideanSpace ℝ (Fin 3)) -- considering 3-dimensional Euclidean space

-- Given conditions
axiom h1 : ‖a‖ = 1
axiom h2 : ‖b‖ = Real.sqrt 2
axiom h3 : c = a + b
axiom h4 : c ⬝ a = 0 -- dot product perpendicularity condition

-- Prove statement
theorem angle_between_vectors : 
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi ∧ Real.cos θ = -Real.sqrt 2 / 2 ∧ θ = 3 * Real.pi / 4 :=
sorry

end angle_between_vectors_l529_529884


namespace vanessa_birthday_money_l529_529730

theorem vanessa_birthday_money {M : ℕ} (h1 : ∃ k : ℕ, M = 9 * k + 1)
: ∃ m, M = m ∧ (m % 9 = 1) :=
by {
  cases' h1 with k hk,
  use M,
  split,
  exact hk,
  rw hk,
  rw Nat.add_mod,
  rw Nat.mul_mod,
  rw Nat.mod_self,
  exact Nat.zero_add 1,
}

end vanessa_birthday_money_l529_529730


namespace operation_is_addition_l529_529169

theorem operation_is_addition : (5 + (-5) = 0) :=
by
  sorry

end operation_is_addition_l529_529169


namespace complement_intersection_l529_529152

variable {U : Type} (A B I : Set U)

variables [decidable_pred (λ x, x ∈ A)]
variables [decidable_pred (λ x, x ∈ B)]
variables [decidable_pred (λ x, x ∈ I)]

-- Define the complement of a set S with respect to I as C_I(S)
def C (S I : Set U) := {x ∈ I | x ∉ S}

theorem complement_intersection (
  hA : A ⊆ I -- A is a subset of I
) (hB : B ⊆ I -- B is a subset of I
): C (A ∩ B) I = {x ∈ I | x ∉ A ∨ x ∉ B} :=
by sorry

end complement_intersection_l529_529152


namespace prod_ineq_geometric_mean_l529_529645

open Real

theorem prod_ineq_geometric_mean (n : ℕ) (x : Fin n → ℝ) (m : ℝ) (hx : ∀ i, 0 < x i) (hm : 0 < m) :
  (∏ i, (m + x i)) ≥ (m + (∏ i, x i) ^ (1 / n)) ^ n :=
sorry

end prod_ineq_geometric_mean_l529_529645


namespace geometric_series_sum_l529_529044

theorem geometric_series_sum : 
  let a : ℕ := 2
  let r : ℕ := 3
  let n : ℕ := 6
  let S_n := (a * (r^n - 1)) / (r - 1)
  S_n = 728 :=
by
  sorry

end geometric_series_sum_l529_529044


namespace steve_pencils_left_l529_529680

-- Conditions
def initial_pencils : Nat := 24
def pencils_given_to_Lauren : Nat := 6
def extra_pencils_given_to_Matt : Nat := 3

-- Question: How many pencils does Steve have left?
theorem steve_pencils_left :
  initial_pencils - (pencils_given_to_Lauren + (pencils_given_to_Lauren + extra_pencils_given_to_Matt)) = 9 := by
  -- You need to provide a proof here
  sorry

end steve_pencils_left_l529_529680


namespace prod_ge_power_of_sqrt_prod_l529_529642

theorem prod_ge_power_of_sqrt_prod (n : Nat) (m : ℝ) (x : Fin n → ℝ) 
    (h_pos : ∀ i, 0 < x i) (h_m : 0 < m) : 
    ∏ i, (m + x i) ≥ (m + real.sqrt (∏ i, x i)) ^ n := 
begin
  -- Proof goes here
  sorry
end

end prod_ge_power_of_sqrt_prod_l529_529642


namespace functionG_has_inverse_l529_529571

noncomputable def functionG : ℝ → ℝ := -- function G described in the problem.
sorry

-- Define the horizontal line test
def horizontal_line_test (f : ℝ → ℝ) : Prop :=
∀ y : ℝ, ∃! x : ℝ, f x = y

theorem functionG_has_inverse : horizontal_line_test functionG :=
sorry

end functionG_has_inverse_l529_529571


namespace normal_probability_l529_529547

noncomputable def xi : MeasureTheory.Measure ℝ := MeasureTheory.Measure.gaussian 1 2

noncomputable def P (a b : ℝ) : ℝ := MeasureTheory.Measure.probMeasure xi (Set.Ioc a b)

theorem normal_probability :
  P (-3) 5 = 0.9544 := by
  sorry

end normal_probability_l529_529547


namespace significant_digits_length_l529_529846

noncomputable def A : ℝ := 2.0561
noncomputable def w : ℝ := 1.8
def num_significant_digits : ℝ → ℕ
-- Definition for counting significant digits
| n := sorry

theorem significant_digits_length :
  let l := A / w in num_significant_digits l = 3 :=
begin
  -- placeholder for the proof
  sorry
end

end significant_digits_length_l529_529846


namespace largest_x_exists_largest_x_largest_real_number_l529_529084

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : x ≤ 48 / 7 :=
sorry

theorem exists_largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  ∃ x, (⌊x⌋ : ℝ) / x = 7 / 8 ∧ x = 48 / 7 :=
sorry

theorem largest_real_number (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  x = 48 / 7 :=
sorry

end largest_x_exists_largest_x_largest_real_number_l529_529084


namespace largest_real_number_condition_l529_529073

theorem largest_real_number_condition (x : ℝ) (hx : ⌊x⌋ / x = 7 / 8) : x ≤ 48 / 7 :=
by
  sorry

end largest_real_number_condition_l529_529073


namespace inequality_not_hold_l529_529178

variable {f : ℝ → ℝ}
variable {x : ℝ}

theorem inequality_not_hold (h : ∀ x : ℝ, -π/2 < x ∧ x < π/2 → (deriv f x) * cos x + f x * sin x > 0) : 
  ¬(sqrt 2 * f (π / 3) < f (π / 4)) :=
sorry

end inequality_not_hold_l529_529178


namespace transformed_data_average_and_variance_l529_529526

variables (x : ℕ → ℝ)
variables (n : ℕ)

noncomputable def average (xs : List ℝ) : ℝ := xs.sum / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let mean := average xs
  (xs.map (λ x, (x - mean)^2)).sum / (xs.length)

theorem transformed_data_average_and_variance
  (h_avg : average ([x 1, x 2, x 3, x 4, x 5] : List ℝ) = 2)
  (h_var : variance ([x 1, x 2, x 3, x 4, x 5] : List ℝ) = 3) :
  average ([2 * x 1 + 1, 2 * x 2 + 1, 2 * x 3 + 1, 2 * x 4 + 1, 2 * x 5 + 1, 1, 2, 3, 4, 5] : List ℝ) = 4 ∧
  variance ([2 * x 1 + 1, 2 * x 2 + 1, 2 * x 3 + 1, 2 * x 4 + 1, 2 * x 5 + 1, 1, 2, 3, 4, 5] : List ℝ) = 8 :=
  by sorry

end transformed_data_average_and_variance_l529_529526


namespace part_I_part_II_l529_529184

def f (x a : ℝ) : ℝ := |x - 4 * a| + |x|

theorem part_I (a : ℝ) (h : -4 ≤ a ∧ a ≤ 4) :
  ∀ x : ℝ, f x a ≥ a^2 := 
sorry

theorem part_II (x y z : ℝ) (h : 4 * x + 2 * y + z = 4) :
  (x + y)^2 + y^2 + z^2 ≥ 16 / 21 :=
sorry

end part_I_part_II_l529_529184


namespace range_of_M_l529_529628

-- Defining the problem conditions and the main statement to prove
theorem range_of_M (a b c : ℝ) (h_sum : a + b + c = 1) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
    let M := (1 / a - 1) * (1 / b - 1) * (1 / c - 1)
    in M ≥ 8 := 
sorry

end range_of_M_l529_529628


namespace sudoku_solution_l529_529328

def sudoku_problem (f : ℕ × ℕ → ℕ) (A B : ℕ) : Prop :=
  -- each row and each column contains exactly the numbers 1, 2, 3
  (∀ i, ∀ x ∈ {0, 1, 2}, ∃ j, f (i, j) = x + 1) ∧ 
  (∀ j, ∀ y ∈ {0, 1, 2}, ∃ i, f (i, j) = y + 1) ∧ 
  -- Table initially has 2 in the top-left position
  (f (0, 0) = 2) ∧ 
  -- Sum of the numbers in the bottom row is 7
  ((f (2, 0) + f (2, 1) + f (2, 2)) = 7) ∧ 
  -- Definitions of A and B
  (A = f (1, 2)) ∧ 
  (B = f (2, 2))

theorem sudoku_solution : ∃ f A B, sudoku_problem f A B ∧ A + B = 4 := 
  sorry

end sudoku_solution_l529_529328


namespace range_of_a_for_root_l529_529903

noncomputable def has_root_in_interval (a : ℝ) : Prop :=
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ (a * x^2 + 2 * x - 1) = 0

theorem range_of_a_for_root :
  { a : ℝ | has_root_in_interval a } = { a : ℝ | -1 ≤ a } :=
by 
  sorry

end range_of_a_for_root_l529_529903


namespace arrange_photos_l529_529514

def tallest_to_shortest ({tallest, shortest, neither_tallest_nor_shortest, taller_between} : Prop) : Prop :=
(¬ tallest) && shortest && neither_tallest_nor_shortest && taller_between

theorem arrange_photos (h : exactly_one_true tallest_to_shortest):
  tallest_to_shortest = [Celine, Amy, David, Bob] :=
begin
  sorry
end

end arrange_photos_l529_529514


namespace exists_non_transformable_number_l529_529979

-- Definitions: Complication of a number as described
def complication (n d : ℕ) : ℕ :=
  -- Function definition for adding a single digit d to a natural number n in various positions.
  sorry

-- Proof statement: Existence of a natural number that cannot be transformed into a perfect square with up to 100 complications.
theorem exists_non_transformable_number : 
  ∃ n : ℕ, ∀ k : ℕ, (∃ f : ℕ → ℕ, (∀ i : ℕ, complication (f i) k) → k ≠ n * n) :=
begin
  sorry
end

end exists_non_transformable_number_l529_529979


namespace right_angled_triangles_count_l529_529630

/-- 
Let T be the set of points (x, y, z) where x, y, z ∈ {0, 1, 2, 3}.
Prove that the number of right-angled triangles that can be formed 
where all vertices are in T and at least one angle is 90 degrees is 216.
-/
theorem right_angled_triangles_count (T : set (ℕ × ℕ × ℕ)) (hT : ∀ p ∈ T, ∃ x y z, p = (x, y, z) ∧ x ∈ {0, 1, 2, 3} ∧ y ∈ {0, 1, 2, 3} ∧ z ∈ {0, 1, 2, 3}): 
  ∃ n : ℕ, n = 216 ∧ (∀ (a b c : \tuple_three_points), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ is_right_angled a b c → True ) sorry

end right_angled_triangles_count_l529_529630


namespace sugar_content_increases_after_adding_sugar_l529_529713

-- Given conditions and definitions
variable (mass_sugar : ℝ) (mass_water : ℝ)
variable (initial_sugar_content : ℝ) (added_sugar : ℝ)

-- Initial condition
def initial_condition := initial_sugar_content = 10 / 100 * (mass_sugar + mass_water)

-- Additional sugar added
def new_sugar_content := (mass_sugar + added_sugar) / (mass_sugar + added_sugar + mass_water) * 100

-- Theorem to prove that the sugar content increases after adding sugar
theorem sugar_content_increases_after_adding_sugar
  (h : initial_sugar_content = 10 / 100 * (mass_sugar + mass_water))
  (added_sugar : ℝ) (added_sugar_pos : 0 < added_sugar) :
  new_sugar_content mass_sugar mass_water + added_sugar > 10 :=
sorry

end sugar_content_increases_after_adding_sugar_l529_529713


namespace largest_x_satisfies_condition_l529_529094

theorem largest_x_satisfies_condition :
  ∃ x : ℝ, (⌊x⌋ / x = 7 / 8) ∧ (∀ y : ℝ, (⌊y⌋ / y = 7 / 8) → y ≤ 48 / 7) :=
sorry

end largest_x_satisfies_condition_l529_529094


namespace nine_segments_three_intersections_impossible_l529_529239

theorem nine_segments_three_intersections_impossible : ¬ ∃ (G : SimpleGraph (Fin 9)), (∀ v, G.degree v = 3) :=
by
  sorry

end nine_segments_three_intersections_impossible_l529_529239


namespace weight_of_one_pencil_l529_529773

theorem weight_of_one_pencil (total_weight : ℝ) (num_pencils : ℕ) (h : total_weight = 141.5) (k : num_pencils = 5) : (total_weight / num_pencils) = 28.3 :=
by
  rw [h, k]
  norm_num
  sorry

end weight_of_one_pencil_l529_529773


namespace area_of_triangle_is_correct_l529_529493

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
    let v := (A.1 - C.1, A.2 - C.2)
    let w := (B.1 - C.1, B.2 - C.2)
    let parallelogram_area := (v.1 * w.2) - (v.2 * w.1)
    (1/2) * (Real.abs parallelogram_area)

theorem area_of_triangle_is_correct :
    area_of_triangle (4, -3) (-3, 2) (2, -7) = 19 :=
by
    sorry

end area_of_triangle_is_correct_l529_529493


namespace cost_contribution_l529_529244

/-- Definitions of areas and costs --/
def cost_per_gallon (cost : ℝ) : ℝ := 45
def coverage_per_gallon (coverage : ℝ) : ℝ := 400
def jason_wall_area (area : ℝ) : ℝ := 1025
def jason_coats (coats : ℝ) : ℝ := 3
def jeremy_wall_area (area : ℝ) : ℝ := 1575
def jeremy_coats (coats : ℝ) : ℝ := 2

/-- Proof statement --/
theorem cost_contribution :
  let total_jason_area := jason_wall_area 1025 * jason_coats 3,
      total_jeremy_area := jeremy_wall_area 1575 * jeremy_coats 2,
      total_area := total_jason_area + total_jeremy_area,
      total_gallons := (total_area / (coverage_per_gallon 400)).ceil,
      total_cost := total_gallons * (cost_per_gallon 45)
  in total_cost / 2 = 360 :=
by
  sorry

end cost_contribution_l529_529244


namespace product_of_three_consecutive_cubes_divisible_by_504_l529_529334

theorem product_of_three_consecutive_cubes_divisible_by_504 (a : ℤ) : 
  ∃ k : ℤ, (a^3 - 1) * a^3 * (a^3 + 1) = 504 * k :=
by
  -- Proof omitted
  sorry

end product_of_three_consecutive_cubes_divisible_by_504_l529_529334


namespace base8_to_base10_conversion_l529_529009

theorem base8_to_base10_conversion : 
  ∃ n : ℕ, n = 2 * 8^2 + 3 * 8^1 + 7 * 8^0 ∧ n = 159 :=
by
  use 159
  split
  · reflexivity
  · reflexivity

end base8_to_base10_conversion_l529_529009


namespace triangle_ABC_area_is_9_l529_529354

-- Define the vertices of the triangle
def A : Prod ℝ ℝ := (1, 2)
def B : Prod ℝ ℝ := (4, 0)
def C : Prod ℝ ℝ := (1, -4)

-- Define the function to calculate the area of the triangle given three vertices
def triangle_area (A B C : Prod ℝ ℝ) : ℝ :=
  1 / 2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Proving the area of triangle A B C is 9
theorem triangle_ABC_area_is_9 : triangle_area A B C = 9 := by
  sorry

end triangle_ABC_area_is_9_l529_529354


namespace base_conversion_subtraction_l529_529831

def base6_to_base10 (n : Nat) : Nat :=
  n / 100000 * 6^5 +
  (n / 10000 % 10) * 6^4 +
  (n / 1000 % 10) * 6^3 +
  (n / 100 % 10) * 6^2 +
  (n / 10 % 10) * 6^1 +
  (n % 10) * 6^0

def base7_to_base10 (n : Nat) : Nat :=
  n / 10000 * 7^4 +
  (n / 1000 % 10) * 7^3 +
  (n / 100 % 10) * 7^2 +
  (n / 10 % 10) * 7^1 +
  (n % 10) * 7^0

theorem base_conversion_subtraction :
  base6_to_base10 543210 - base7_to_base10 43210 = 34052 := by
  sorry

end base_conversion_subtraction_l529_529831


namespace largest_x_l529_529114

def largest_x_with_condition_eq_7_over_8 (x : ℝ) : Prop :=
  ⌊x⌋ / x = 7 / 8

theorem largest_x (x : ℝ) (h : largest_x_with_condition_eq_7_over_8 x) :
  x = 48 / 7 :=
sorry

end largest_x_l529_529114


namespace bright_spot_area_l529_529754

-- Define the given values
def n : ℝ := 1.5
def R : ℝ := 1
def h : ℝ := 1.73
def a : ℝ := 1

-- Define the proof problem
theorem bright_spot_area : 
  let α := Real.atan (h / R),
      R1 := a * Real.tan α,
      R2 := (h + a) * Real.tan α - R,
      S := Real.pi * (R2^2 - R1^2)
  in S = 34 :=
by {
  -- Substituting in the values for α, R1, and R2 based on the problem conditions
  let α := Real.atan (1.73 / 1),
  let R1 := 1 * Real.tan α,
  let R2 := (1.73 + 1) * Real.tan α - 1,
  let S := Real.pi * (R2^2 - R1^2),
  have h1 : Real.tan α = Math.sqrt 3 := sorry,
  have h2 : R1 = 1.73 := sorry,
  have h3 : R2 = 3.7289 := sorry,
  have h4 : S = 34 := sorry,
  exact h4
}

end bright_spot_area_l529_529754


namespace problem1_problem2_l529_529039

-- First problem
theorem problem1 :
  2 * Real.sin (Real.pi / 3) - 3 * Real.tan (Real.pi / 6) - (-1 / 3) ^ 0 + (-1) ^ 2023 = -2 :=
by
  sorry

-- Second problem
theorem problem2 :
  abs (1 - Real.sqrt 2) - Real.sqrt 12 + (1 / 3) ^ (-1 : ℤ) - 2 * Real.cos (Real.pi / 4) = 2 - 2 * Real.sqrt 3 :=
by
  sorry

end problem1_problem2_l529_529039


namespace average_speed_is_correct_l529_529764

def distance_1 : ℝ := 300
def speed_1 : ℝ := 60
def distance_2 : ℝ := 195
def speed_2 : ℝ := 75

def time_1 : ℝ := distance_1 / speed_1
def time_2 : ℝ := distance_2 / speed_2

def total_distance : ℝ := distance_1 + distance_2
def total_time : ℝ := time_1 + time_2

def average_speed : ℝ := total_distance / total_time

theorem average_speed_is_correct : average_speed ≈ 65.13 := 
by
  sorry

end average_speed_is_correct_l529_529764


namespace projection_matrix_correct_l529_529256

noncomputable def projection_matrix_Q : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![(5 : ℚ) / 7, 1 / 7, -3 / 7],
    ![1 / 7, 13 / 14, 3 / 14],
    ![-3 / 7, 3 / 14, 5 / 14]
  ]

def normal_vector_n : Fin 3 → ℚ :=
  ![2, -1, 3]

def plane_Q (u q : Fin 3 → ℚ) : Prop :=
  u - q = ((u ⬝ᵥ normal_vector_n) / (normal_vector_n ⬝ᵥ normal_vector_n)) • normal_vector_n

def is_projection_matrix (Q : Matrix (Fin 3) (Fin 3) ℚ) : Prop :=
  ∀ u : Fin 3 → ℚ, plane_Q u (Q.mulVec u)

theorem projection_matrix_correct :
  is_projection_matrix projection_matrix_Q :=
sorry

end projection_matrix_correct_l529_529256


namespace max_k_value_l529_529925

open Real

theorem max_k_value (x y k : ℝ)
  (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h_pos_k : 0 < k)
  (h_eq : 6 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ 3 / 2 :=
by
  sorry

end max_k_value_l529_529925


namespace part_I_part_II_l529_529158

variable (α : ℝ)

-- The given conditions.
variable (h1 : π < α)
variable (h2 : α < (3 * π) / 2)
variable (h3 : Real.sin α = -4/5)

-- Part (I): Prove cos α = -3/5
theorem part_I : Real.cos α = -3/5 :=
sorry

-- Part (II): Prove sin 2α + 3 tan α = 24/25 + 4
theorem part_II : Real.sin (2 * α) + 3 * Real.tan α = 24/25 + 4 :=
sorry

end part_I_part_II_l529_529158


namespace smallest_n_l529_529510

theorem smallest_n (x : ℕ → ℝ) (n : ℕ) (h1 : ∀ i : ℕ, i ∈ Finset.range n → |x i| < 1)
  (h2 : (Finset.range n).sum (λ i, |x i|) = 2005 + |(Finset.range n).sum (λ i, x i)|) : 
  n = 2006 :=
sorry

end smallest_n_l529_529510


namespace problem_statement_l529_529520

noncomputable def a : ℝ := 0.99^3
noncomputable def b : ℝ := Real.log 0.6 / Real.log 2
noncomputable def c : ℝ := Real.log π / Real.log 3

theorem problem_statement : b < a ∧ a < c := by
  sorry

end problem_statement_l529_529520


namespace arithmetic_seq_formula_l529_529947

variable (a : ℕ → ℤ)

-- Given conditions
axiom h1 : a 1 + a 2 + a 3 = 0
axiom h2 : a 4 + a 5 + a 6 = 18

-- Goal: general formula for the arithmetic sequence
theorem arithmetic_seq_formula (n : ℕ) : a n = 2 * n - 4 := by
  sorry

end arithmetic_seq_formula_l529_529947


namespace radical_axes_intersect_at_one_point_l529_529231

-- Define the circles with their centers and radii
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define that the centers are not collinear
def not_collinear (c₁ c₂ c₃ : Circle) : Prop :=
  let (x1, y1) := c₁.center;
  let (x2, y2) := c₂.center;
  let (x3, y3) := c₃.center;
  (x2 - x1) * (y3 - y1) ≠ (x3 - x1) * (y2 - y1)

-- Given three circles
variables (C1 C2 C3 : Circle)

-- The theorem to prove that the radical axes of three non-collinear circles intersect at one point
theorem radical_axes_intersect_at_one_point
  (h1 : not_collinear C1 C2 C3) :
  ∃ (P : ℝ × ℝ), 
    (power_eq_with_respect_to_two_circles C1 C2 P) ∧ 
    (power_eq_with_respect_to_two_circles C2 C3 P) ∧ 
    (power_eq_with_respect_to_two_circles C1 C3 P) := 
sorry

-- Definition of power of a point with respect to a circle
def power (C : Circle) (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P;
  let (xc, yc) := C.center;
  (x - xc)^2 + (y - yc)^2 - C.radius^2

-- Definition of a point having equal power with respect to two circles
def power_eq_with_respect_to_two_circles (C1 C2 : Circle) (P : ℝ × ℝ) : Prop :=
  power C1 P = power C2 P

end radical_axes_intersect_at_one_point_l529_529231


namespace simplify_expression_l529_529672

variable (x y : ℝ)

theorem simplify_expression : 3 * y + 5 * y + 6 * y + 2 * x + 4 * x = 14 * y + 6 * x :=
by
  sorry

end simplify_expression_l529_529672


namespace people_going_to_zoo_l529_529677

theorem people_going_to_zoo (buses people_per_bus total_people : ℕ) 
  (h1 : buses = 3) 
  (h2 : people_per_bus = 73) 
  (h3 : total_people = buses * people_per_bus) : 
  total_people = 219 := by
  rw [h1, h2] at h3
  exact h3

end people_going_to_zoo_l529_529677


namespace nth_equation_l529_529897

theorem nth_equation (n : ℕ) : 
  (nat.rec_on n (sqrt 2) (λ n ih, sqrt (2 + ih))) = 2 * cos (π / 2^(n + 1)) := 
sorry

end nth_equation_l529_529897


namespace coefficient_of_x3_term_l529_529693

noncomputable def coefficient_x3 (f : ℝ → ℝ) : ℝ :=
-- Function to extract the coefficient of x^3 term from a polynomial
sorry

theorem coefficient_of_x3_term :
  coefficient_x3 (λ x : ℝ, (1 - 2 * x)^5 * (2 + x)) = -120 :=
sorry

end coefficient_of_x3_term_l529_529693


namespace bond_value_after_8_years_l529_529614

theorem bond_value_after_8_years (r t1 t2 : ℕ) (A1 A2 P : ℚ) :
  r = 4 / 100 ∧ t1 = 3 ∧ t2 = 8 ∧ A1 = 560 ∧ A1 = P * (1 + r * t1) 
  → A2 = P * (1 + r * t2) ∧ A2 = 660 :=
by
  intro h
  obtain ⟨hr, ht1, ht2, hA1, hA1eq⟩ := h
  -- Proof needs to be filled in here
  sorry

end bond_value_after_8_years_l529_529614


namespace min_value_sum_of_squares_l529_529142

noncomputable def minSumSquares (a : ℝ) (n : ℕ) (x : ℕ → ℝ) : ℝ :=
  if h : 0 < a ∧ a < 1 ∧ (∑ i in Finset.range (n + 1), x i = ↑n + a) ∧ 
          (∑ i in Finset.range (n + 1), (x i)⁻¹ = ↑n + a⁻¹) then 
    ∑ i in Finset.range (n + 1), (x i)^2 
  else 
    0

theorem min_value_sum_of_squares 
  (a : ℝ) (n : ℕ)
  (x : ℕ → ℝ)
  (h_a : 0 < a ∧ a < 1)
  (h_sum : ∑ i in Finset.range (n + 1), x i = ↑n + a)
  (h_reciprocal_sum : ∑ i in Finset.range (n + 1), (x i)⁻¹ = ↑n + a⁻¹) :
  minSumSquares a n x = n + a^2 := 
  sorry

end min_value_sum_of_squares_l529_529142


namespace main_theorem_l529_529638

def y : ℂ := Complex.exp (Complex.I * 4 * Real.pi / 9)

theorem main_theorem : 
  (3 * y^2 + y^4) * (3 * y^4 + y^8) * (3 * y^6 + y^12) * (3 * y^8 + y^16) * (3 * y^10 + y^20) * (3 * y^12 + y^24) = -8 := 
by 
  sorry

end main_theorem_l529_529638


namespace coeff_a_neg_half_l529_529228

-- Define the binomial expression
def binomial_expr (a : ℝ) := (a - 1 / (Real.sqrt a))^7

-- The proof goal is to show that the coefficient of a^(-1/2) is -21
theorem coeff_a_neg_half (a : ℝ) :
  let term_coeff := Nat.choose 7 5 * (-1)^5
  term_coeff = -21 := by
  sorry

end coeff_a_neg_half_l529_529228


namespace BANANA_perm_count_l529_529918

/-- The number of distinct permutations of the letters in the word "BANANA". -/
def distinctArrangementsBANANA : ℕ :=
  let total := 6
  let freqB := 1
  let freqA := 3
  let freqN := 2
  total.factorial / (freqB.factorial * freqA.factorial * freqN.factorial)

theorem BANANA_perm_count : distinctArrangementsBANANA = 60 := by
  unfold distinctArrangementsBANANA
  simp [Nat.factorial_succ]
  exact le_of_eq (decide_eq_true (Nat.factorial_dvd_factorial (Nat.le_succ 6)))
  sorry

end BANANA_perm_count_l529_529918


namespace probability_fifth_term_integer_l529_529612

theorem probability_fifth_term_integer :
  (∀ n, ∃ a_n : ℕ, (a_1 = 8) ∧ (∀ k, a_{k+1} = 3 * a_k - 2)) →
  (∃ q : ℚ, q = (1/2)^5 ∧ q = 1/32) :=
by {
  sorry
}

end probability_fifth_term_integer_l529_529612


namespace slope_of_AC_l529_529232

theorem slope_of_AC (A B C : ℝ → ℝ → Prop)
  (right_triangle_ABC : ∠ B = 90°)
  (length_AC : AC = 100)
  (length_AB : AB = 80) :
  slope_of (line_segment AC) = 4 / 3 :=
  sorry

end slope_of_AC_l529_529232


namespace each_plate_weight_correct_l529_529723

def weight_felt_lowered (total_weight : ℝ) : ℝ := 1.2 * total_weight
def total_weight (each_plate_weight : ℝ) (num_plates : ℕ) : ℝ := each_plate_weight * num_plates

theorem each_plate_weight_correct :
  (∀ (each_plate_weight : ℝ) (total_weight_lb : ℝ), total_weight_lb = total_weight each_plate_weight 10 →
    360 = weight_felt_lowered total_weight_lb → each_plate_weight = 30) :=
begin
  intros each_plate_weight total_weight_lb h_total_weight h_weight_felt,
  sorry
end

end each_plate_weight_correct_l529_529723


namespace train_speed_l529_529001

noncomputable def jogger_speed : ℝ := 9 -- speed in km/hr
noncomputable def jogger_distance : ℝ := 150 / 1000 -- distance in km
noncomputable def train_length : ℝ := 100 / 1000 -- length in km
noncomputable def time_to_pass : ℝ := 25 -- time in seconds

theorem train_speed 
  (v_j : ℝ := jogger_speed)
  (d_j : ℝ := jogger_distance)
  (L : ℝ := train_length)
  (t : ℝ := time_to_pass) :
  (train_speed_in_kmh : ℝ) = 36 :=
by 
  sorry

end train_speed_l529_529001


namespace probability_is_correct_l529_529439

noncomputable def probability_cashier_opens_early : ℝ :=
  let x1 : ℝ := sorry
  let x2 : ℝ := sorry
  let x3 : ℝ := sorry
  let x4 : ℝ := sorry
  let x5 : ℝ := sorry
  let x6 : ℝ := sorry
  if (0 <= x1) ∧ (x1 <= 15) ∧
     (0 <= x2) ∧ (x2 <= 15) ∧
     (0 <= x3) ∧ (x3 <= 15) ∧
     (0 <= x4) ∧ (x4 <= 15) ∧
     (0 <= x5) ∧ (x5 <= 15) ∧
     (0 <= x6) ∧ (x6 <= 15) ∧
     (x1 < x6) ∧ (x2 < x6) ∧ (x3 < x6) ∧ (x4 < x6) ∧ (x5 < x6) then 
    let p_not_A : ℝ := (12 / 15) ^ 6
    1 - p_not_A
  else
    0

theorem probability_is_correct : probability_cashier_opens_early = 0.738 :=
by sorry

end probability_is_correct_l529_529439


namespace largest_real_number_condition_l529_529075

theorem largest_real_number_condition (x : ℝ) (hx : ⌊x⌋ / x = 7 / 8) : x ≤ 48 / 7 :=
by
  sorry

end largest_real_number_condition_l529_529075


namespace min_quotient_value_l529_529844

noncomputable def minimum_quotient : ℕ :=
  let four_digit_numbers := {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ (∀ i j : ℕ, i ≠ j → ((n / 10^i % 10) ≠ (n / 10^j % 10))) ∧ (∑ i in finset.range 4, if (n / 10^i % 10) % 2 = 0 then 1 else 0) ≥ 2} in
  four_digit_numbers.foldl (λ acc n, min acc ((n / finset.univ.sum (λ i, n / 10^i % 10)) : ℕ)) 9999

theorem min_quotient_value : minimum_quotient = 87 :=
  sorry

end min_quotient_value_l529_529844


namespace largest_real_number_condition_l529_529074

theorem largest_real_number_condition (x : ℝ) (hx : ⌊x⌋ / x = 7 / 8) : x ≤ 48 / 7 :=
by
  sorry

end largest_real_number_condition_l529_529074


namespace maximum_triangle_area_l529_529343

noncomputable def cubic_parabola : ℝ → ℝ := λ x, - (1 / 3) * x^3 + 3 * x

def inflection_point : ℝ × ℝ := (0, 0)

def line_through_inflection (m : ℝ) : ℝ → ℝ :=
  λ x, m * x

def intersection_points (m : ℝ) : list (ℝ × ℝ) :=
  let x := real.sqrt (9 - 3 * m) in
  [(sqrt(9-3*m), m * sqrt(9-3*m)), (-sqrt(9-3*m), -m * sqrt(9-3*m))]

def triangle_area (m : ℝ) : ℝ :=
  let x := real.sqrt (9 - 3 * m) in
  m * (9 - 3 * m)

theorem maximum_triangle_area :
  ∃ m : ℝ, 0 < m ∧ m < 3 ∧ triangle_area m = 6.75 :=
by
  let m := 1.5
  use m
  split
  · norm_num
  split
  · norm_num
  · norm_num
  sorry

end maximum_triangle_area_l529_529343


namespace pirate_attack_49_vessels_l529_529855

def paths_49_vessels := set (list (ℕ × ℕ))

def is_valid_path (path : list (ℕ × ℕ)) : Prop :=
  ∃ x y, path = (x, y) :: _ ∧ (list.last path []).fst = x ∧ (list.last path []).snd = y ∧
  ∀ i, i < path.length - 1 → (path.nth i).is_some ∧ 
  (((path.nth i).iget.fst = (path.nth (i+1)).iget.fst ∧ (path.nth i).iget.snd ≠ (path.nth (i+1)).iget.snd) ∨ 
   ((path.nth i).iget.snd = (path.nth (i+1)).iget.snd ∧ (path.nth i).iget.fst ≠ (path.nth (i+1)).iget.fst))

theorem pirate_attack_49_vessels :
  ∃ p ∈ paths_49_vessels, p.length = 13 ∧
  is_valid_path p ∧ 
  set.to_finset (set.of_list p) = finset.univ {v | ∃ x y, v = (x, y) ∧ x < 7 ∧ y < 7} :=
begin
  sorry
end

end pirate_attack_49_vessels_l529_529855


namespace log_decreasing_interval_l529_529318

noncomputable def p (x : ℝ) : ℝ := 2*x^2 - 3*x + 4

theorem log_decreasing_interval :
  ∀ x, x ∈ Set.Ici (3/4) → ∀ y, y ∈ Set.Ici (3/4) → x ≤ y → p x ≤ p y →
  (log (1/2) (p y) ≤ log (1/2) (p x)) := sorry

end log_decreasing_interval_l529_529318


namespace no_point_C_l529_529598

theorem no_point_C (A B : (ℝ × ℝ)) (AB_distance : dist A B = 12) :
  ¬(∃ C : (ℝ × ℝ),
    (dist A C + dist B C + dist A B = 60) ∧
    (∃ h : ℝ, 1/2 * 12 * h = 180 ∧ (C.2 = h ∨ C.2 = -h))) :=
begin
  sorry
end

end no_point_C_l529_529598


namespace solution_proof_l529_529434

noncomputable def problem_statement : Prop :=
  let x := [x1, x2, x3, x4, x5, x6] in
  let B := (∀ i < 5, x[i] < x[5]) in
  let A := (x[5] ≥ 12) in
  ∃ x1 x2 x3 x4 x5 x6 : ℝ,
    (0 ≤ x1 ∧ x1 ≤ 15) ∧ (0 ≤ x2 ∧ x2 ≤ 15) ∧ (0 ≤ x3 ∧ x3 ≤ 15) ∧
    (0 ≤ x4 ∧ x4 ≤ 15) ∧ (0 ≤ x5 ∧ x5 ≤ 15) ∧ (0 ≤ x6 ∧ x6 ≤ 15) ∧
    B ∧ (classical.some ((measure_theory.measure_space.measure (λ x, x < 12 <= x) B).to_real) = 0.738)

theorem solution_proof : problem_statement := sorry

end solution_proof_l529_529434


namespace no_9_segments_with_3_intersections_l529_529237

theorem no_9_segments_with_3_intersections :
  ¬ ∃ (G : SimpleGraph (Fin 9)), (∀ v : Fin 9, degree G v = 3) :=
by
  sorry

end no_9_segments_with_3_intersections_l529_529237


namespace inequality_holds_l529_529267

theorem inequality_holds (a b : ℝ) (h1 : a > 1) (h2 : 1 > b) (h3 : b > -1) : a > b^2 :=
by
  sorry

end inequality_holds_l529_529267


namespace largest_real_solution_l529_529102

theorem largest_real_solution (x : ℝ) (h : (⌊x⌋ / x = 7 / 8)) : x ≤ 48 / 7 := by
  sorry

end largest_real_solution_l529_529102


namespace complex_modulus_extreme_values_l529_529543

open Complex Real

noncomputable def maxValueMinValue {z : ℂ} (h : abs (z - (1 + 2 * I)) = 1) : ℝ × ℝ :=
  let maxval : ℝ := sqrt 5 + 1
  let minval : ℝ := sqrt 5 - 1
  (maxval, minval)

theorem complex_modulus_extreme_values: ∀ z : ℂ, abs (z - (1 + 2 * I)) = 1 →
  maxValueMinValue z = (sqrt 5 + 1, sqrt 5 - 1) :=
by
  intro z h
  simp [maxValueMinValue]
  sorry

end complex_modulus_extreme_values_l529_529543


namespace remainder_of_S_div_1000_l529_529961

/-- 
S is the sum of all three-digit positive integers with the property that:
1. The number has three distinct digits.
2. The units digit is always even.
-/
def S : ℕ := ∑ n in {n | 100 ≤ n ∧ n < 1000 ∧ ∃ a b c, n = 100 * a + 10 * b + c ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ c % 2 = 0}, n

/-- 
Compute the remainder when S is divided by 1000.
-/
theorem remainder_of_S_div_1000 : S % 1000 = 520 := sorry

end remainder_of_S_div_1000_l529_529961


namespace min_value_sq_sum_l529_529641

theorem min_value_sq_sum (x1 x2 : ℝ) (h : x1 * x2 = 2013) : (x1 + x2)^2 ≥ 8052 :=
by
  sorry

end min_value_sq_sum_l529_529641


namespace number_of_zeros_when_a_lt_neg2_l529_529553

open Real

-- Define the function f
def f (x a : ℝ) : ℝ := (x^2 - 2 * x) * log x + (a - 1/2) * x^2 + 2 * (1 - a) * x + a

-- Define the derivative of f
def f' (x a : ℝ) : ℝ := 2 * (x - 1) * (log x + a)

theorem number_of_zeros_when_a_lt_neg2 (a : ℝ) (h : a < -2) : 
  ∃ z1 z2 z3 : ℝ, (0 < z1 ∧ z1 < 1 ∧ f z1 a = 0) ∧ (1 < z2 ∧ z2 < exp (-a) ∧ f z2 a = 0) ∧ (exp (-a) < z3 ∧ f z3 a = 0) :=
sorry

end number_of_zeros_when_a_lt_neg2_l529_529553


namespace probability_X_eq_2_l529_529141

namespace Hypergeometric

def combin (n k : ℕ) : ℕ := n.choose k

noncomputable def hypergeometric (N M n k : ℕ) : ℚ :=
  (combin M k * combin (N - M) (n - k)) / combin N n

theorem probability_X_eq_2 :
  hypergeometric 8 5 3 2 = 15 / 28 := by
  sorry

end Hypergeometric

end probability_X_eq_2_l529_529141


namespace angle_ENF_eq_25_l529_529355

/-- Given an isosceles triangle DEF with DF = EF and ∠DFE = 120°, 
    and a point N within the triangle such that ∠DNF = 15° and ∠DFN = 25°, 
    prove that ∠ENF is 25°. -/
theorem angle_ENF_eq_25 (D E F N : Type)
  (h1 : isosceles_triangle D E F)
  (h2 : angle D F E = 120)
  (h3 : point_interior_triangle N D E F)
  (h4 : angle D N F = 15)
  (h5 : angle D F N = 25) :
  angle E N F = 25 := 
sorry

end angle_ENF_eq_25_l529_529355


namespace leo_class_girls_l529_529212

theorem leo_class_girls (g b : ℕ) 
  (h_ratio : 3 * b = 4 * g) 
  (h_total : g + b = 35) : g = 15 := 
by
  sorry

end leo_class_girls_l529_529212


namespace total_days_to_finish_tea_and_coffee_l529_529208

-- Define the given conditions formally before expressing the theorem
def drinks_coffee_together (days : ℕ) : Prop := days = 10
def drinks_coffee_alone_A (days : ℕ) : Prop := days = 12
def drinks_tea_together (days : ℕ) : Prop := days = 12
def drinks_tea_alone_B (days : ℕ) : Prop := days = 20

-- The goal is to prove that A and B together finish a pound of tea and a can of coffee in 35 days
theorem total_days_to_finish_tea_and_coffee : 
  ∃ days : ℕ, 
    drinks_coffee_together 10 ∧ 
    drinks_coffee_alone_A 12 ∧ 
    drinks_tea_together 12 ∧ 
    drinks_tea_alone_B 20 ∧ 
    days = 35 :=
by
  sorry

end total_days_to_finish_tea_and_coffee_l529_529208


namespace find_x_l529_529518

def vector_a (x : ℝ) : ℝ × ℝ := (2, x)
def vector_b : ℝ × ℝ := (-3, 2)

def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem find_x (x : ℝ) (h : is_perpendicular (vector_a x + vector_b) vector_b) : 
  x = -7 / 2 :=
  sorry

end find_x_l529_529518


namespace primes_squared_sum_prime_l529_529835

theorem primes_squared_sum_prime (p q : ℕ) (hp : prime p) (hq : prime q) : 
  prime (2^2 + p^2 + q^2) ↔ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) :=
by 
  sorry

end primes_squared_sum_prime_l529_529835


namespace trebled_resultant_is_correct_l529_529415

-- Let's define the initial number and the transformations
def initial_number := 17
def doubled (n : ℕ) := n * 2
def added_five (n : ℕ) := n + 5
def trebled (n : ℕ) := n * 3

-- Finally, we state the problem to prove
theorem trebled_resultant_is_correct : 
  trebled (added_five (doubled initial_number)) = 117 :=
by
  -- Here we just print sorry which means the proof is expected but not provided yet.
  sorry

end trebled_resultant_is_correct_l529_529415


namespace girl_scouts_earnings_l529_529323

theorem girl_scouts_earnings :
  ∃ E : ℝ, 
    (let cost_per_person := 2.50 in
     let num_people := 10 in
     let total_cost := cost_per_person * num_people in
     let amount_left := 5 in
     E = total_cost + amount_left) ∧
     E = 30.00 :=
by
   use 30.00
   sorry

end girl_scouts_earnings_l529_529323


namespace three_digit_numbers_eq_11_sum_squares_l529_529741

theorem three_digit_numbers_eq_11_sum_squares :
  ∃ (N : ℕ), 
    (N = 550 ∨ N = 803) ∧
    (∃ (a b c : ℕ), 
      N = 100 * a + 10 * b + c ∧ 
      100 * a + 10 * b + c = 11 * (a ^ 2 + b ^ 2 + c ^ 2) ∧
      1 ≤ a ∧ a ≤ 9 ∧
      0 ≤ b ∧ b ≤ 9 ∧
      0 ≤ c ∧ c ≤ 9) :=
sorry

end three_digit_numbers_eq_11_sum_squares_l529_529741


namespace count_reflectional_symmetry_l529_529577

def tetrominoes : List String := ["I", "O", "T", "S", "Z", "L", "J"]

def has_reflectional_symmetry (tetromino : String) : Bool :=
  match tetromino with
  | "I" => true
  | "O" => true
  | "T" => true
  | "S" => false
  | "Z" => false
  | "L" => false
  | "J" => false
  | _   => false

theorem count_reflectional_symmetry : 
  (tetrominoes.filter has_reflectional_symmetry).length = 3 := by
  sorry

end count_reflectional_symmetry_l529_529577


namespace largest_x_satisfies_condition_l529_529097

theorem largest_x_satisfies_condition :
  ∃ x : ℝ, (⌊x⌋ / x = 7 / 8) ∧ (∀ y : ℝ, (⌊y⌋ / y = 7 / 8) → y ≤ 48 / 7) :=
sorry

end largest_x_satisfies_condition_l529_529097


namespace original_amount_of_milk_is_720_l529_529588

variable (M : ℝ) -- The original amount of milk in milliliters

theorem original_amount_of_milk_is_720 :
  ((5 / 6) * M) - ((2 / 5) * ((5 / 6) * M)) - ((2 / 3) * (((5 / 6) * M) - ((2 / 5) * ((5 / 6) * M)))) = 120 → 
  M = 720 := by
  sorry

end original_amount_of_milk_is_720_l529_529588


namespace difference_red_white_l529_529457

/-
Allie picked 100 wildflowers. The categories of flowers are given as below:
- 13 of the flowers were yellow and white
- 17 of the flowers were red and yellow
- 14 of the flowers were red and white
- 16 of the flowers were blue and yellow
- 9 of the flowers were blue and white
- 8 of the flowers were red, blue, and yellow
- 6 of the flowers were red, white, and blue

The goal is to define the number of flowers containing red and white, and
prove that the difference between the number of flowers containing red and 
those containing white is 3.
-/

def total_flowers : ℕ := 100
def yellow_and_white : ℕ := 13
def red_and_yellow : ℕ := 17
def red_and_white : ℕ := 14
def blue_and_yellow : ℕ := 16
def blue_and_white : ℕ := 9
def red_blue_and_yellow : ℕ := 8
def red_white_and_blue : ℕ := 6

def flowers_with_red : ℕ := red_and_yellow + red_and_white + red_blue_and_yellow + red_white_and_blue
def flowers_with_white : ℕ := yellow_and_white + red_and_white + blue_and_white + red_white_and_blue

theorem difference_red_white : flowers_with_red - flowers_with_white = 3 := by
  rw [flowers_with_red, flowers_with_white]
  sorry

end difference_red_white_l529_529457


namespace ratio_largest_smallest_root_geometric_progression_l529_529339

theorem ratio_largest_smallest_root_geometric_progression (a b c d : ℤ)
  (h_poly : a * x^3 + b * x^2 + c * x + d = 0) 
  (h_in_geo_prog : ∃ r1 r2 r3 q : ℝ, r1 < r2 ∧ r2 < r3 ∧ r1 * q = r2 ∧ r2 * q = r3 ∧ q ≠ 0) : 
  ∃ R : ℝ, R = 1 := 
by
  sorry

end ratio_largest_smallest_root_geometric_progression_l529_529339


namespace correct_simplification_l529_529785

-- Step 1: Define the initial expression
def initial_expr (a b : ℝ) : ℝ :=
  (a - b) / a / (a - (2 * a * b - b^2) / a)

-- Step 2: Define the correct simplified form
def simplified_expr (a b : ℝ) : ℝ :=
  1 / (a - b)

-- Step 3: State the theorem that proves the simplification is correct
theorem correct_simplification (a b : ℝ) (h : a ≠ b): 
  initial_expr a b = simplified_expr a b :=
by {
  sorry,
}

end correct_simplification_l529_529785


namespace initial_students_l529_529031

theorem initial_students {f : ℕ → ℕ} {g : ℕ → ℕ} (h_f : ∀ t, t ≥ 15 * 60 + 3 → (f t = 4 * ((t - (15 * 60 + 3)) / 3 + 1))) 
    (h_g : ∀ t, t ≥ 15 * 60 + 10 → (g t = 8 * ((t - (15 * 60 + 10)) / 10 + 1))) 
    (students_at_1544 : f 15 * 60 + 44 - g 15 * 60 + 44 + initial = 27) : 
    initial = 3 := 
sorry

end initial_students_l529_529031


namespace max_area_of_square_l529_529297

noncomputable def maxInscribedQuadrilateralArea (R : ℝ) (hR : R > 0)
  (ABCD : Type) [quad : Quadrilateral ABCD] [circInscribed : CircleInscribed ABCD]
  (x : ℝ) (hx : x < 2 * R) : Prop :=
  ∃ (S : ℝ), (Square ABCD → S) ∧ (∀ (quad' : Quadrilateral ABCD), area quad' ≤ S)

theorem max_area_of_square (R : ℝ) (hR : R > 0)
  (ABCD : Type) [quad : Quadrilateral ABCD] [circInscribed : CircleInscribed ABCD]
  (hx : ∀ (x : ℝ), x < 2 * R → AD ABCD = x) :
  maxInscribedQuadrilateralArea R hR ABCD hx :=
sorry

end max_area_of_square_l529_529297


namespace smallest_sector_angle_l529_529249

theorem smallest_sector_angle :
  ∃ a1 d, 
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 15 → ∃ ai, ai = a1 + (i - 1) * d) ∧  -- angles form an arithmetic sequence
  (∑ i in (finset.range 15).map (function.embedding.subtype _), a1 + i * d = 360) ∧  -- sum of angles is 360 degrees
  a1 < 6 ∧  -- the smallest angle is smaller than 6 degrees
  a1 = 3 :=  -- degree measure of the smallest possible sector angle is 3 degrees
sorry

end smallest_sector_angle_l529_529249


namespace min_g_on_interval_range_of_a_l529_529176

-- Define the function f
def f (e a b x : ℝ) : ℝ := Real.exp x - a * x ^ 2 - b * x - 1

-- Define the first derivative g
def g (e a b x : ℝ) : ℝ := Real.exp x - 2 * a * x - b

-- Problem part (1)
theorem min_g_on_interval (e a b : ℝ) (h_exp_pos : 1 ≤ Real.exp x ∧ Real.exp x ≤ e) : 
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, g e a b y ≥ g e a b x) := sorry

-- Problem part (2)
theorem range_of_a (e a b : ℝ) (hf1 : f e a b 1 = 0) : 
  (∃ x ∈ Set.Ioo 0 1, f e a b x = 0) ↔ (e - 2 < a ∧ a < 1) := sorry

end min_g_on_interval_range_of_a_l529_529176


namespace proper_subset_count_l529_529862

open Set

-- Define the sets A and B according to the given conditions
def A : Set ℤ := {x | x^2 - 3 * x - 4 ≤ 0}
def B : Set ℤ := {x | 2 * x^2 - x - 6 > 0}

-- Define the intersection of A and B
def A_inter_B : Set ℤ := A ∩ B

-- State the theorem that the number of proper subsets of A_inter_B is 3
theorem proper_subset_count : (A_inter_B : Finset ℤ).card = 2 → (2 ^ 2 - 1 = 3) := by
  sorry

end proper_subset_count_l529_529862


namespace number_of_red_cars_l529_529708

theorem number_of_red_cars (B R : ℕ) (h1 : R / B = 3 / 8) (h2 : B = 70) : R = 26 :=
by
  sorry

end number_of_red_cars_l529_529708


namespace complement_unions_subset_condition_l529_529877

open Set

-- Condition Definitions
def A : Set ℝ := {x | 1 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 3 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | x < a + 1}

-- Questions Translated to Lean Statements
theorem complement_unions (U : Set ℝ)
  (hU : U = univ) : (compl A ∪ compl B) = compl (A ∩ B) := by sorry

theorem subset_condition (a : ℝ)
  (h : B ⊆ C a) : a ≥ 8 := by sorry

end complement_unions_subset_condition_l529_529877


namespace largest_x_satisfies_condition_l529_529089

theorem largest_x_satisfies_condition (x : ℝ) (h : (⌊x⌋ / x) = 7 / 8) : x ≤ 48 / 7 :=
sorry

end largest_x_satisfies_condition_l529_529089


namespace intersection_pairs_pass_through_B_l529_529486

universe u
variables (α : Type u) [metric_space α] [normed_group α] [normed_space ℝ α]

def Thales_circle (A B : α) : set α := { x | dist x ((A + B) / 2) = dist A ((A + B) / 2) }

theorem intersection_pairs_pass_through_B
  (A B C : α)
  (F : α := (A + C) / 2)
  (k : set α := { x | dist x F = arbitrary_radius})
  (k_A : set α := Thales_circle B C)
  (k_C : set α := Thales_circle B A)
  (P Q P' Q' : α)
  (hP : P ∈ k ∩ k_A) (hQ : Q ∈ k ∩ k_A)
  (hP' : P' ∈ k ∩ k_C) (hQ' : Q' ∈ k ∩ k_C) :
  ∃ (f : fin 4 → α), function.injective f ∧
  (f 0 = P ∧ f 1 = P' ∧ f 2 = Q ∧ f 3 = Q') ∧
  (∃ i j k l, f i ≠ f j ∧ f k ≠ f l ∧
    line_through f i f j = line_through f k f l ∧ vertex B ∈ line_through f i f j) :=
sorry

end intersection_pairs_pass_through_B_l529_529486


namespace beads_used_total_l529_529624

theorem beads_used_total :
  let necklaces_monday := 10
  let necklaces_tuesday := 2
  let bracelets_wednesday := 5
  let earrings_wednesday := 7
  let beads_per_necklace := 20
  let beads_per_bracelet := 10
  let beads_per_earring := 5

  (necklaces_monday + necklaces_tuesday) * beads_per_necklace + 
  bracelets_wednesday * beads_per_bracelet + 
  earrings_wednesday * beads_per_earring = 325 := by
  let necklaces_monday := 10
  let necklaces_tuesday := 2
  let bracelets_wednesday := 5
  let earrings_wednesday := 7
  let beads_per_necklace := 20
  let beads_per_bracelet := 10
  let beads_per_earring := 5

  calc
    (necklaces_monday + necklaces_tuesday) * beads_per_necklace + 
    bracelets_wednesday * beads_per_bracelet + 
    earrings_wednesday * beads_per_earring
    = (10 + 2) * 20 + 5 * 10 + 7 * 5 : by rfl
    ... = 12 * 20 + 5 * 10 + 7 * 5 : by rfl
    ... = 240 + 50 + 35 : by rfl
    ... = 325 : by rfl

end beads_used_total_l529_529624


namespace eccentricity_of_ellipse_l529_529904

theorem eccentricity_of_ellipse 
  (m n : ℝ)
  (h_m_pos : 0 < m)
  (h_n_pos : 0 < n)
  (h_hyperbola : mx^2 - ny^2 = 1)
  (h_eccentricity_hyperbola : ecc = 2) :
  eccentricity (mx^2 + ny^2 = 1) = sqrt(6) / 3 :=
sorry

end eccentricity_of_ellipse_l529_529904


namespace proof_problem_l529_529978

-- Definitions based on the conditions
def x := 70 + 0.11 * 70
def y := x + 0.15 * x
def z := y - 0.2 * y

-- The statement to prove
theorem proof_problem : 3 * z - 2 * x + y = 148.407 :=
by
  sorry

end proof_problem_l529_529978


namespace fibonacci_geometric_sequence_l529_529310

-- The Fibonacci sequence is defined
def fibonacci : ℕ → ℕ 
| 0     := 0
| 1     := 1
| (n+2) := fibonacci n + fibonacci (n+1)

-- The statement of the problem in Lean
theorem fibonacci_geometric_sequence (a b c : ℕ) (h_sum : a + b + c = 3000) (h_geom : ∃ r, fibonacci b = r * fibonacci a ∧ fibonacci c = r * fibonacci b) :
  a = 999 :=
by
  sorry

end fibonacci_geometric_sequence_l529_529310


namespace train_length_l529_529428

theorem train_length
  (cross_time : ℝ)
  (speed_kmh : ℝ)
  (cross_time_eq : cross_time = 10)
  (speed_kmh_eq : speed_kmh = 288) :
  let speed_ms := (speed_kmh * 1000) / 3600 in
  let length_train := speed_ms * cross_time in
  length_train = 800 := 
by
  sorry

end train_length_l529_529428


namespace susan_juice_bottles_l529_529191

theorem susan_juice_bottles 
  (paul_bottles : ℕ): 
  paul_bottles = 2 ∧
  ∀ donald_bottles, donald_bottles = 2 * paul_bottles + 3 ∧
  ∀ susan_bottles, susan_bottles = 1.5 * donald_bottles - 2.5 →
  susan_bottles = 8 :=
sorry

end susan_juice_bottles_l529_529191


namespace find_pq_sum_l529_529422

theorem find_pq_sum (p q : ℕ) (hpq : Nat.coprime p q) :
    let height := (p : ℝ) / (q : ℝ)
    let width := 15
    let length := 20
    let area := 50
    (10 + 2 * height) * (width / 2) * (length / 2) = 2 * area ->
    p + q = 11 :=
by
    sorry

end find_pq_sum_l529_529422


namespace circumscribed_inscribed_circle_ratio_l529_529341

theorem circumscribed_inscribed_circle_ratio (a b c : ℝ)
  (h1 : a = 13) (h2 : b = 14) (h3 : c = 15) :
  let p := (a + b + c) / 2 in
  let S := Real.sqrt (p * (p - a) * (p - b) * (p - c)) in
  let r := S / p in
  let R := (a * b * c) / (4 * S) in
  (R / r)^2 = (65 / 32)^2 :=
by
  sorry

end circumscribed_inscribed_circle_ratio_l529_529341


namespace max_length_interval_l529_529180

theorem max_length_interval
  (a : ℝ)
  (h_neg : a < 0)
  (h_bound : ∀ x ∈ [0, (1 + sqrt 5) / 2], abs (a * x^2 + 8 * x + 3) ≤ 5) :
  a = -8 ∧ (1 + sqrt 5) / 2 = (∃ l, (l = (1 + sqrt 5) / 2) ∧ ∀ x ∈ [0, l], abs (a * x^2 + 8 * x + 3) ≤ 5) :=
sorry

end max_length_interval_l529_529180


namespace quadratic_coefficients_l529_529706

theorem quadratic_coefficients (x : ℝ) : 
  let a := 3
  let b := -5
  let c := 1
  3 * x^2 + 1 = 5 * x → a * x^2 + b * x + c = 0 := by
sorry

end quadratic_coefficients_l529_529706


namespace geometric_series_sum_l529_529046

theorem geometric_series_sum :
  let a := 2
  let r := 3
  let n := 6
  S = a * (r ^ n - 1) / (r - 1) → S = 728 :=
by
  intros a r n h
  sorry

end geometric_series_sum_l529_529046


namespace mark_gig_schedule_l529_529280

theorem mark_gig_schedule 
  (every_other_day : ∀ weeks, ∃ gigs, gigs = weeks * 7 / 2) 
  (songs_per_gig : 2 * 5 + 10 = 20) 
  (total_minutes : ∃ gigs, 280 = gigs * 20) : 
  ∃ weeks, weeks = 4 := 
by 
  sorry

end mark_gig_schedule_l529_529280


namespace age_of_b_l529_529746

/-- Define the ages as natural numbers. -/
variables (A B C D : ℕ)

/-- Conditions of the problem translated into Lean statements. -/
def condition1 : Prop := A = B + 2
def condition2 : Prop := B = 2 * C
def condition3 : Prop := D = B - 3
def condition4 : Prop := A + B + C + D = 60

/-- The theorem to be proven: under the given conditions, B is approximately 17. -/
theorem age_of_b (h1 : condition1 A B) (h2 : condition2 B C) (h3 : condition3 B D) (h4 : condition4 A B C D) : B = 17 := 
by {
  sorry
}

end age_of_b_l529_529746


namespace number_of_integers_satisfying_condition_l529_529505

theorem number_of_integers_satisfying_condition
  (f : ℤ → ℤ)
  (h_f_def : ∀ n : ℤ, f n = (⌈(99 * n : ℚ) / 100⌉ - ⌊(100 * n : ℚ) / 101⌋)) :
  {n : ℤ | 1 + (⌊(100 * n : ℚ) / 101⌋) = ⌈(99 * n : ℚ) / 100⌉}.to_finset.card = 10100 :=
by sorry

end number_of_integers_satisfying_condition_l529_529505


namespace g_neg_eleven_eq_neg_two_l529_529968

def f (x : ℝ) : ℝ := 2 * x - 7
def g (y : ℝ) : ℝ := 3 * y^2 + 4 * y - 6

theorem g_neg_eleven_eq_neg_two : g (-11) = -2 := by
  sorry

end g_neg_eleven_eq_neg_two_l529_529968


namespace count_multiples_6_or_10_but_not_both_l529_529198

def multiples_6_less_151 := { x : ℕ | x < 151 ∧ x % 6 = 0 }
def multiples_10_less_151 := { x : ℕ | x < 151 ∧ x % 10 = 0 }
def multiples_30_less_151 := { x : ℕ | x < 151 ∧ x % 30 = 0 }

theorem count_multiples_6_or_10_but_not_both :
  (multiples_6_less_151.card + multiples_10_less_151.card - multiples_30_less_151.card) = 35 :=
by sorry

end count_multiples_6_or_10_but_not_both_l529_529198


namespace neg_sin_leq_one_l529_529906

theorem neg_sin_leq_one (p : Prop) :
  (∀ x : ℝ, Real.sin x ≤ 1) → (¬(∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x : ℝ, Real.sin x > 1) :=
by
  sorry

end neg_sin_leq_one_l529_529906


namespace solution_proof_l529_529431

noncomputable def problem_statement : Prop :=
  let x := [x1, x2, x3, x4, x5, x6] in
  let B := (∀ i < 5, x[i] < x[5]) in
  let A := (x[5] ≥ 12) in
  ∃ x1 x2 x3 x4 x5 x6 : ℝ,
    (0 ≤ x1 ∧ x1 ≤ 15) ∧ (0 ≤ x2 ∧ x2 ≤ 15) ∧ (0 ≤ x3 ∧ x3 ≤ 15) ∧
    (0 ≤ x4 ∧ x4 ≤ 15) ∧ (0 ≤ x5 ∧ x5 ≤ 15) ∧ (0 ≤ x6 ∧ x6 ≤ 15) ∧
    B ∧ (classical.some ((measure_theory.measure_space.measure (λ x, x < 12 <= x) B).to_real) = 0.738)

theorem solution_proof : problem_statement := sorry

end solution_proof_l529_529431


namespace largest_real_solution_l529_529105

theorem largest_real_solution (x : ℝ) (h : (⌊x⌋ / x = 7 / 8)) : x ≤ 48 / 7 := by
  sorry

end largest_real_solution_l529_529105


namespace arithmetic_sequence_a3_is_8_l529_529604

-- Define the arithmetic sequence
def arithmetic_sequence (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

-- Theorem to prove a3 = 8 given a1 = 4 and d = 2
theorem arithmetic_sequence_a3_is_8 (a1 d : ℕ) (h1 : a1 = 4) (h2 : d = 2) : arithmetic_sequence a1 d 3 = 8 :=
by
  sorry -- Proof not required as per instruction

end arithmetic_sequence_a3_is_8_l529_529604


namespace quadratic_inequality_solution_l529_529886

def range_of_k (k : ℝ) : Prop := (k ≥ 4) ∨ (k ≤ 2)

theorem quadratic_inequality_solution (k : ℝ) (x : ℝ) (h : x = 1) :
  k^2*x^2 - 6*k*x + 8 ≥ 0 → range_of_k k := 
sorry

end quadratic_inequality_solution_l529_529886


namespace coefficient_of_x_neg3_l529_529933

noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem coefficient_of_x_neg3 (n : ℕ) 
  (h1 : (3 * (1:ℝ) - (1 / real.cbrt((1:ℝ) ^ 2))) ^ n = 128) :
  ∃ coeff : ℕ, coeff = 21 ∧ 
    (∃ r : ℕ, r = 6 ∧ (3 * (1 : ℝ) - (1 / real.cbrt((1 : ℝ) ^ 2)))^(n - r) * 
    binomial_coeff n r * ((1:ℝ) ^ (n - (5 * r) / 3) = (coeff * ((1/x^3) : ℝ))) := 
sorry

end coefficient_of_x_neg3_l529_529933


namespace parabola_focus_distance_l529_529170

theorem parabola_focus_distance (p x₀ : ℝ) (hp : 0 < p) (h₁ : (2 : ℝ) ^ 2 = 2 * p * x₀)
  (h₂ : abs (x₀ + p / 2) = 3 * abs (p / 2)) :
  p = real.sqrt 2 :=
by
  sorry

end parabola_focus_distance_l529_529170


namespace largest_x_satisfies_condition_l529_529086

theorem largest_x_satisfies_condition (x : ℝ) (h : (⌊x⌋ / x) = 7 / 8) : x ≤ 48 / 7 :=
sorry

end largest_x_satisfies_condition_l529_529086


namespace distance_between_x_intercepts_l529_529412

-- Define the slopes and the intersecting point
def slope1 : ℝ := 4
def slope2 : ℝ := 6
def intersection_point : ℝ × ℝ := (8, 12)

-- Define the equations of the lines using point-slope form
def line1 (x : ℝ) : ℝ := slope1 * (x - intersection_point.1) + intersection_point.2
def line2 (x : ℝ) : ℝ := slope2 * (x - intersection_point.1) + intersection_point.2

-- Calculate the x-intercepts
def x_intercept1 : ℝ := intersection_point.2 / slope1 + intersection_point.1
def x_intercept2 : ℝ := intersection_point.2 / slope2 + intersection_point.1

-- State the theorem and implicitly prove it by using the correct answer provided in the problem
theorem distance_between_x_intercepts : abs (x_intercept2 - x_intercept1) = 1 :=
by
  have h1 : x_intercept1 = 5 := calc
    x_intercept1 = intersection_point.2 / slope1 + intersection_point.1 : rfl
    ... = 12 / 4 + 8 : by rfl
    ... = 3 + 8 : by rfl
    ... = 11 : by norm_num
    ... = 5 : by norm_num

  have h2 : x_intercept2 = 6 := calc
    x_intercept2 = intersection_point.2 / slope2 + intersection_point.1 : rfl
    ... = 12 / 6 + 8 : by rfl
    ... = 2 + 8 : by rfl
    ... = 10 : by norm_num
    ... = 6 : by norm_num

  show abs (x_intercept2 - x_intercept1) = 1,
  calc
    abs (x_intercept2 - x_intercept1) = abs (6 - 5) : by rw [h1, h2]
    ... = 1 : by norm_num


end distance_between_x_intercepts_l529_529412


namespace smallest_number_after_removal_largest_number_after_removal_l529_529353

noncomputable def concatenated_first_10_primes : Nat :=
  2357111317192329

theorem smallest_number_after_removal (n : Nat) (h : n = concatenated_first_10_primes) :
  min_number_after_removal n 8 = 11111229 := sorry

theorem largest_number_after_removal (n : Nat) (h : n = concatenated_first_10_primes) :
  max_number_after_removal n 8 = 77192329 := sorry

end smallest_number_after_removal_largest_number_after_removal_l529_529353


namespace find_phi_l529_529559

def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem find_phi (φ : ℝ) (h : ∀ x : ℝ, f x φ ≤ |f (π / 6) φ|) : φ = π / 6 :=
sorry

end find_phi_l529_529559


namespace cashier_window_open_probability_l529_529443

noncomputable def probability_window_opens_in_3_minutes_of_scientist_arrival : ℝ := 
  0.738

theorem cashier_window_open_probability :
  let x : ℝ → ℝ := λ x, if x ≥ 12 then 0.738 else 0.262144 in
  ∀ (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ),
    (∀ i ∈ [x₁, x₂, x₃, x₄, x₅], i < x₆) ∧ 0 ≤ x₆ ≤ 15 →
    x x₆ = 0.738 :=
by
  sorry

end cashier_window_open_probability_l529_529443


namespace not_perfect_square_l529_529227

theorem not_perfect_square
  (A : ℕ)
  (h1 : ∀ n, decimal_representation A n → (n ≠ 0 → A.digit n = 0))
  (h2 : ∃ n, A.digit n ≠ 0)
  (h3 : ∃ n, 3 ≤ A.digit_count n): 
  ¬ ∃ x : ℕ, x * x = A :=
by
  sorry

end not_perfect_square_l529_529227


namespace largest_x_l529_529118

def largest_x_with_condition_eq_7_over_8 (x : ℝ) : Prop :=
  ⌊x⌋ / x = 7 / 8

theorem largest_x (x : ℝ) (h : largest_x_with_condition_eq_7_over_8 x) :
  x = 48 / 7 :=
sorry

end largest_x_l529_529118


namespace rearrangements_count_l529_529199

-- Define the problem set and constraints

def is_consecutively_alphabetized (c₁ c₂ : Char) : Prop :=
  (c₁.val < c₂.val) ∧ (c₁.val + 1 = c₂.val)

def is_reverse_neighbor (c₁ c₂ : Char) : Prop :=
  (c₁ = 'a' ∧ c₂ = 'f') ∨ (c₁ = 'f' ∧ c₂ = 'a') ∨
  (c₁ = 'b' ∧ c₂ = 'e') ∨ (c₁ = 'e' ∧ c₂ = 'b')

def is_valid_rearrangement (s : List Char) : Prop :=
  s = ['a', 'b', 'c', 'e', 'f'].erase_dups ∧
  (∀ (l₁ l₂ : Char), l₁ :: l₂ :: _ = s →
    ¬ is_consecutively_alphabetized l₁ l₂ ∧
    ¬ is_reverse_neighbor l₁ l₂)

-- The main statement to prove
theorem rearrangements_count : 
  Nat.card {s : List Char // is_valid_rearrangement s} = 4 :=
sorry

end rearrangements_count_l529_529199


namespace f_neg_two_l529_529172

def f (a b : ℝ) (x : ℝ) :=
  -a * x^5 - x^3 + b * x - 7

theorem f_neg_two (a b : ℝ) (h : f a b 2 = -9) : f a b (-2) = -5 :=
by sorry

end f_neg_two_l529_529172


namespace average_price_l529_529616

noncomputable def average_price_per_bottle (largeCount : ℕ) (largePrice : ℚ) (smallCount : ℕ) (smallPrice : ℚ) : ℚ :=
  let totalCost := (largeCount * largePrice) + (smallCount * smallPrice)
  let totalCount := largeCount + smallCount
  totalCost / totalCount

theorem average_price (h_largeCount : 1375) (h_largePrice : 1.75) (h_smallCount : 690) (h_smallPrice : 1.35) :
  average_price_per_bottle 1375 1.75 690 1.35 = 1.616 :=
by
  sorry

end average_price_l529_529616


namespace poodles_on_tuesday_l529_529808

theorem poodles_on_tuesday (hours_per_poodle hours_per_chihuahua hours_per_labrador : ℕ) 
  (poodles_monday chihuahuas_monday_chihuahuas_tuesday labradors_wednesday : ℕ)
  (total_hours available_hours: ℕ) 
  (hours_spent : ℕ) :

  hours_per_poodle = 2 ->
  hours_per_chihuahua = 1 ->
  hours_per_labrador = 3 ->
  poodles_monday = 4 ->
  chihuahuas_monday_chihuahuas_tuesday = 2 ->
  labradors_wednesday = 4 ->
  total_hours = 32 ->
  hours_spent = (poodles_monday * hours_per_poodle) + 
                (2 * hours_per_chihuahua) + 
                (4 * hours_per_labrador) + 
                (2 * hours_per_chihuahua) -> -- Adding hours for chihuahuas on Tuesday
  available_hours = total_hours - hours_spent ->
  available_hours / hours_per_poodle = 4 :=
by {
  intros,
  sorry
}

end poodles_on_tuesday_l529_529808


namespace total_cost_price_all_items_l529_529987

noncomputable def bicycle_cost_price (sp : ℝ) (profit_percent : ℝ) : ℝ :=
    sp / (1 + profit_percent / 100)

noncomputable def bicycle_cost_price_loss (sp : ℝ) (loss_percent : ℝ) : ℝ :=
    sp / (1 - loss_percent / 100)

noncomputable def scooter_cost_price (sp : ℝ) (profit_percent : ℝ) : ℝ :=
    sp / (1 + profit_percent / 100)

noncomputable def scooter_cost_price_loss (sp : ℝ) (loss_percent : ℝ) : ℝ :=
    sp / (1 - loss_percent / 100)

theorem total_cost_price_all_items :
    let cp_bicycle1 := bicycle_cost_price 990 10 in
    let cp_bicycle2 := bicycle_cost_price 1100 5 in
    let cp_bicycle3 := bicycle_cost_price_loss 1210 12 in
    let cp_scooter1 := scooter_cost_price 4150 7 in
    let cp_scooter2 := scooter_cost_price_loss 3450 15 in
    cp_bicycle1 + cp_bicycle2 + cp_bicycle3 + cp_scooter1 + cp_scooter2 = 11260.94 :=
by
    sorry

end total_cost_price_all_items_l529_529987


namespace estimate_students_height_at_least_165_l529_529132

theorem estimate_students_height_at_least_165 
  (sample_size : ℕ)
  (total_school_size : ℕ)
  (students_165_170 : ℕ)
  (students_170_175 : ℕ)
  (h_sample : sample_size = 100)
  (h_total_school : total_school_size = 1000)
  (h_students_165_170 : students_165_170 = 20)
  (h_students_170_175 : students_170_175 = 30)
  : (students_165_170 + students_170_175) * (total_school_size / sample_size) = 500 := 
by
  sorry

end estimate_students_height_at_least_165_l529_529132


namespace probA_probB1_probB2_l529_529013

-- Problem A in Lean 4:
def C2_in_cartesian (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 4

def C1_intersection_conditions (k : ℝ) : Prop :=
  ∀ x y, (C2_in_cartesian x y → (y = -4/3 * |x| + 2))

theorem probA (k : ℝ) :
  C1_intersection_conditions k →
  k = -4/3 := sorry

-- Problem B in Lean 4:
def f (x a : ℝ) :=
  abs (x + 1) - abs (a*x - 1)

def f_greater_one (a : ℝ) : set ℝ :=
  { x : ℝ | f x a > 1 }

theorem probB1 :
  (f_greater_one 1) = { x : ℝ | x > 1/2 } := sorry

def f_greater_x (a : ℝ) : Prop :=
  ∀ x, 0 < x ∧ x < 1 → f x a > x

theorem probB2 :
  (∀ a, f_greater_x a ↔ 0 < a ∧ a ≤ 2) := sorry

end probA_probB1_probB2_l529_529013


namespace original_people_in_room_l529_529289

theorem original_people_in_room (x : ℕ) (h1 : 18 = (2 * x / 3) - (x / 6)) : x = 36 :=
by sorry

end original_people_in_room_l529_529289


namespace midpoint_of_diagonal_l529_529294

-- Definition of the points
def point1 : ℝ × ℝ := (2, -3)
def point2 : ℝ × ℝ := (14, 9)

-- Statement about the midpoint of a diagonal in a rectangle
theorem midpoint_of_diagonal : 
  ∀ (x1 y1 x2 y2 : ℝ), (x1, y1) = point1 → (x2, y2) = point2 →
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  (midpoint_x, midpoint_y) = (8, 3) :=
by
  intros
  sorry

end midpoint_of_diagonal_l529_529294


namespace rotary_club_extra_omelets_l529_529688

theorem rotary_club_extra_omelets
  (small_children_tickets : ℕ)
  (older_children_tickets : ℕ)
  (adult_tickets : ℕ)
  (senior_tickets : ℕ)
  (eggs_total : ℕ)
  (omelet_for_small_child : ℝ)
  (omelet_for_older_child : ℝ)
  (omelet_for_adult : ℝ)
  (omelet_for_senior : ℝ)
  (eggs_per_omelet : ℕ)
  (extra_omelets : ℕ) :
  small_children_tickets = 53 →
  older_children_tickets = 35 →
  adult_tickets = 75 →
  senior_tickets = 37 →
  eggs_total = 584 →
  omelet_for_small_child = 0.5 →
  omelet_for_older_child = 1 →
  omelet_for_adult = 2 →
  omelet_for_senior = 1.5 →
  eggs_per_omelet = 2 →
  extra_omelets = (eggs_total - (2 * (small_children_tickets * omelet_for_small_child +
                                      older_children_tickets * omelet_for_older_child +
                                      adult_tickets * omelet_for_adult +
                                      senior_tickets * omelet_for_senior))) / eggs_per_omelet →
  extra_omelets = 25 :=
by
  intros hsmo_hold hsoc_hold hat_hold hsnt_hold htot_hold
        hosm_hold hocc_hold hact_hold hsen_hold hepom_hold hres_hold
  sorry

end rotary_club_extra_omelets_l529_529688


namespace rotation_result_l529_529872

def initial_vector : ℝ × ℝ × ℝ := (3, -1, 1)

def rotate_180_z (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match v with
  | (x, y, z) => (-x, -y, z)

theorem rotation_result :
  rotate_180_z initial_vector = (-3, 1, 1) :=
by
  sorry

end rotation_result_l529_529872


namespace books_a_count_l529_529313

-- Variables representing the number of books (a) and (b)
variables (A B : ℕ)

-- Conditions given in the problem
def condition1 : Prop := A + B = 20
def condition2 : Prop := A = B + 4

-- The theorem to prove
theorem books_a_count (h1 : condition1 A B) (h2 : condition2 A B) : A = 12 :=
sorry

end books_a_count_l529_529313


namespace find_a_l529_529271

def f (x : ℝ) : ℝ :=
if x < 1 then -x else (x - 1)^2

theorem find_a (a : ℝ) (h : f(a) = 1) : a = -1 ∨ a = 2 :=
by 
  sorry

end find_a_l529_529271


namespace plankton_consumption_difference_l529_529789

theorem plankton_consumption_difference 
  (x : ℕ) 
  (d : ℕ) 
  (total_hours : ℕ := 9) 
  (total_consumption : ℕ := 360)
  (sixth_hour_consumption : ℕ := 43)
  (total_series_sum : x + (x + d) + (x + 2 * d) + (x + 3 * d) + (x + 4 * d) + (x + 5 * d) + (x + 6 * d) + (x + 7 * d) + (x + 8 * d) = total_consumption)
  (sixth_hour_eq : x + 5 * d = sixth_hour_consumption)
  : d = 3 :=
by
  sorry

end plankton_consumption_difference_l529_529789


namespace base_10_representation_of_BCD_l529_529767

-- Definitions of the digits and their transitions in base-4
def C : ℕ := 3
def D : ℕ := 0
def B : ℕ := 1
def A : ℕ := 1

-- Definition of the coded integer BCD in base-4
def BCD_base4 : ℕ := 310

-- Conversion from base-4 to base-10
theorem base_10_representation_of_BCD : 
  (3 * 4^2 + 1 * 4^1 + 0 * 4^0) = 52 := by
  calc
    3 * 4^2 + 1 * 4^1 + 0 * 4^0
    = 3 * 16 + 1 * 4 + 0 : by rfl
    = 48 + 4 + 0 : by rfl
    = 52 : by rfl

-- Proof skipping
sorry

end base_10_representation_of_BCD_l529_529767


namespace exists_non_integral_point_p_eq_zero_l529_529625

theorem exists_non_integral_point_p_eq_zero
  (b_0 b_1 b_2 b_3 b_4 b_5 b_6 b_7 b_8 b_9 b_{10} b_{11} : ℝ)
  (p : ℝ → ℝ → ℝ := λ x y,
    b_0 + b_1 * x + b_2 * y + b_3 * x^2 + b_4 * x * y + b_5 * y^2 +
    b_6 * x^3 + b_7 * x^2 * y + b_8 * x * y^2 + b_9 * y^3 +
    b_{10} * x^4 + b_{11} * y^4) :
  (p 0 0 = 0) →
  (p 1 0 = 0) →
  (p (-1) 0 = 0) →
  (p 0 1 = 0) →
  (p 0 (-1) = 0) →
  (p 1 1 = 0) →
  (p 1 (-1) = 0) →
  (p 2 2 = 0) →
  (p (-1) (-1) = 0) →
  ∃ r s : ℝ, p r s = 0 ∧ (r ∉ ℤ ∨ s ∉ ℤ) := 
by
  sorry

end exists_non_integral_point_p_eq_zero_l529_529625


namespace drawing_probability_l529_529735

noncomputable def probability_A_not_losing : ℝ := 0.8
noncomputable def probability_B_not_losing : ℝ := 0.7
noncomputable def probability_A_winning : ℝ := 0.3
noncomputable def probability_drawing_game : ℝ := 0.5

theorem drawing_probability (P : ℝ) : (probability_A_not_losing = probability_A_winning + P) → (P = probability_drawing_game) :=
by
  intro h,
  sorry

end drawing_probability_l529_529735


namespace largest_x_eq_48_div_7_l529_529106

theorem largest_x_eq_48_div_7 :
  ∃ x : ℝ, (⟨floor x / x⟩ = 7 / 8) ∧ (x = 48 / 7) := 
begin
  sorry
end

end largest_x_eq_48_div_7_l529_529106


namespace maximum_third_altitude_l529_529219

noncomputable def altitude_relation (PQ PR QR : ℕ) (h1 h2 h3 : ℕ) :=
  h1 * QR = h2 * PQ ∧ h2 * QR = h3 * PR

theorem maximum_third_altitude (PQ PR QR : ℕ) (h1 h2 : ℕ) :
  h1 = 18 ∧ h2 = 6 →
  (∃ h3: ℕ, altitude_relation PQ PR QR h1 h2 h3) →
  ∃ h3 : ℕ, h3 = 6 :=
begin
  intros h_altitudes h_third_altitude,
  sorry
end

end maximum_third_altitude_l529_529219


namespace range_of_m_l529_529859

variable {x1 x2 : ℝ}

theorem range_of_m 
  (f : ℝ → ℝ) (g : ℝ → ℝ → ℝ) (m : ℝ)
  (hf : ∀ x, f x = x^2)
  (hg : ∀ x m, g x m = (1 / 2)^x - m)
  (h_cond : ∀ x1 ∈ Icc (-1 : ℝ) 3, ∃ x2 ∈ Icc (0 : ℝ) 1, f x1 ≥ g x2 m) :
  m ≥ (1 / 2) :=
by
  sorry

end range_of_m_l529_529859


namespace symmetric_points_are_concyclic_l529_529626

-- Definitions and given conditions
variables (A B C D E : Type) [EuclideanGeometry A] [EuclideanGeometry B] 
  [EuclideanGeometry C] [EuclideanGeometry D] [EuclideanGeometry E]

-- Perpendicularity of diagonals at E
variable (h1 : intersects_perpendicular_at A C B D E)

-- Definition: symmetric points relative to sides
def symmetric_point (P Q : A) : A := sorry

-- Points E1, E2, E3, E4 as symmetric to E relative to sides AB, BC, CD, DA
def E1 := symmetric_point E (line_through A B)
def E2 := symmetric_point E (line_through B C)
def E3 := symmetric_point E (line_through C D)
def E4 := symmetric_point E (line_through D A)

-- Statement to prove: E1, E2, E3, E4 are concyclic
theorem symmetric_points_are_concyclic : concyclic E1 E2 E3 E4 :=
sorry

end symmetric_points_are_concyclic_l529_529626


namespace f_inv_undefined_at_one_l529_529202

def f (x : ℝ) : ℝ := (x - 5) / (x - 6)
noncomputable def f_inv (y : ℝ) : ℝ := (6 * y - 5) / (1 - y)

theorem f_inv_undefined_at_one :
  ∃ x : ℝ, (f_inv x = (6 * x - 5) / (1 - x)) ∧ (f_inv x = 0) ↔ x = 1 :=
by
  sorry

end f_inv_undefined_at_one_l529_529202


namespace mowgli_received_nuts_is_20_l529_529982

-- Definitions from the conditions
def monkeys : ℕ := 5
def N : ℕ -- Number of nuts each monkey gathered initially

-- Total number of nuts each monkey threw to others
def nuts_thrown_by_each_monkey (n : ℕ) : ℕ := monkeys - 1

-- Total number of nuts thrown by all monkeys
def total_nuts_thrown (n : ℕ) : ℕ := n * nuts_thrown_by_each_monkey monkeys

-- Nuts Delivery condition
def nuts_delivered (n : ℕ) : ℕ := (monkeys * n) / 2

-- Total nuts delivered is the same as total nuts thrown
theorem mowgli_received_nuts_is_20 : ∃ N : ℕ, nuts_delivered N = total_nuts_thrown N ∧ nuts_delivered N = 20 :=
by
  sorry

end mowgli_received_nuts_is_20_l529_529982


namespace simplify_fraction_l529_529781

theorem simplify_fraction (a b : ℝ) (h : a ≠ b): 
  (a - b) / a / (a - (2 * a * b - b^2) / a) = 1 / (a - b) :=
by
  sorry

end simplify_fraction_l529_529781


namespace find_row_with_sum_2013_squared_l529_529653

-- Define the sum of the numbers in the nth row
def sum_of_row (n : ℕ) : ℕ := (2 * n - 1)^2

theorem find_row_with_sum_2013_squared : (∃ n : ℕ, sum_of_row n = 2013^2) ∧ (sum_of_row 1007 = 2013^2) :=
by
  sorry

end find_row_with_sum_2013_squared_l529_529653


namespace perpendicular_circles_l529_529910

theorem perpendicular_circles 
  {circle1 circle2 : set Point}
  (O1 : Point)
  (A C B D : Point)
  (line : Line)
  (H1 : line ∈ (line_through O1 A ∩ line_intersects_circle A C circle1))
  (H2 : line ∈ (line_intersects_circle B D circle2))
  (H_ratio : |AB| / |BC| = |AD| / |DC|) :
  ∠(tangent_line circle1 A) (tangent_line circle2 A) = 90 :=
sorry

end perpendicular_circles_l529_529910


namespace small_to_large_circle_ratio_l529_529939

theorem small_to_large_circle_ratio (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : π * b^2 - π * a^2 = 5 * π * a^2) :
  a / b = 1 / Real.sqrt 6 :=
by
  sorry

end small_to_large_circle_ratio_l529_529939


namespace min_perimeter_triangle_AED_l529_529524

noncomputable def perim_triangle (A E D : Point) : ℝ :=
  dist A E + dist E D + dist D A

theorem min_perimeter_triangle_AED :
  ∀ (V A B C E D : Point), 
  is_regular_pyramid V A B C 4 8 → 
  on_plane_passing_through_A A E D V B C →
  perimeter_works_out_to_eleven →
  perim_triangle A E D = 11 :=
begin
  sorry
end

end min_perimeter_triangle_AED_l529_529524


namespace projection_point_P_is_correct_min_value_of_MN_is_correct_l529_529944

-- Define the parametric equations of line l
def line_l (t : ℝ) : ℝ × ℝ :=
  (-3 + (√3 / 2) * t, (1 / 2) * t)

-- Define the polar coordinates of projection point P
def polar_coordinates_of_P : ℝ × ℝ :=
  (3 / 2, 2 / 3 * Real.pi)

-- Define the polar equation of curve C
def curve_C (θ : ℝ) : ℝ :=
  4 * Real.cos θ

-- Define the minimum value of |MN|
def min_distance_MN : ℝ :=
  1 / 2

-- Lean statement for the given proof problems

theorem projection_point_P_is_correct :
  (let (x, y) := line_l ((3 : ℝ) / 2 * √3);
  (Real.sqrt (x^2 + y^2), Real.atan2 y x)) = polar_coordinates_of_P :=
by
  sorry

theorem min_value_of_MN_is_correct :
  (let ρ := curve_C;
  let E := (2 : ℝ, 0 : ℝ);
  let d := 5 / 2;
  let r := 2;
  d - r = min_distance_MN) :=
by
  sorry

end projection_point_P_is_correct_min_value_of_MN_is_correct_l529_529944


namespace inequality_proof_l529_529128

theorem inequality_proof 
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c) + b / Real.sqrt (b^2 + 8 * a * c) + c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
  sorry

end inequality_proof_l529_529128


namespace yolanda_avg_three_point_baskets_l529_529382

noncomputable theory

def yolanda_points_season : ℕ := 345
def total_games : ℕ := 15
def free_throws_per_game : ℕ := 4
def two_point_baskets_per_game : ℕ := 5

theorem yolanda_avg_three_point_baskets :
  (345 - (15 * (4 * 1 + 5 * 2))) / 3 / 15 = 3 :=
by sorry

end yolanda_avg_three_point_baskets_l529_529382


namespace AcmeExtendedVowelSoup_correct_l529_529016

noncomputable def AcmeExtendedVowelSoup : Prop :=
  let vowels := 5
  let semi_vowel_y := 3
  let total_words := (vowels^5) + (5 * (vowels^4)) + (Nat.choose 5 2 * (vowels^3)) + (Nat.choose 5 3 * (vowels^2))
  total_words = 7750

theorem AcmeExtendedVowelSoup_correct : AcmeExtendedVowelSoup := by
  sorry -- Proof is omitted as requested.

end AcmeExtendedVowelSoup_correct_l529_529016


namespace range_of_m_max_area_l529_529474

noncomputable def curve_intersection_range (a : ℝ) : Prop :=
a > 0 → (
(0 < a ∧ a < 1 → (∀ m : ℝ, m = (a^2 + 1) / 2 ∨ (-a < m ∧ m ≤ a))) ∧ 
(a ≥ 1 → (∀ m : ℝ, -a < m ∧ m < a))
)

noncomputable def max_area_triangle (a : ℝ) : ℝ :=
if (a > 0 ∧ a < 1 / 2) then (1/2 * a * real.sqrt(1 - a^2)) else 0

-- theorem for Part 1
theorem range_of_m (a : ℝ) : curve_intersection_range a :=
sorry

-- theorem for Part 2
theorem max_area (a : ℝ) (h : 0 < a ∧ a < 1/2) : 
  max_area_triangle a = 1/2 * a * real.sqrt(1 - a^2) :=
sorry

end range_of_m_max_area_l529_529474


namespace geometric_series_sum_l529_529047

theorem geometric_series_sum :
  let a := 2
  let r := 3
  let n := 6
  S = a * (r ^ n - 1) / (r - 1) → S = 728 :=
by
  intros a r n h
  sorry

end geometric_series_sum_l529_529047


namespace candy_difference_l529_529515

theorem candy_difference (frankie_candies : ℕ) (max_candies : ℕ) (h1 : frankie_candies = 74) (h2 : max_candies = 92) : max_candies - frankie_candies = 18 := by
  sorry

end candy_difference_l529_529515


namespace relationship_among_abc_l529_529891

variables {f : ℝ → ℝ}

def a := 3^0.2 * f (3^0.2)
def b := (Real.log 2 / Real.log π) * f (Real.log 2 / Real.log π)
def c := (-2) * f (-2)

theorem relationship_among_abc
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_ineq : ∀ x : ℝ, 0 < x → f x + x * deriv f x < 0) :
  c > b ∧ b > a :=
by
  sorry

end relationship_among_abc_l529_529891


namespace additional_airplanes_needed_l529_529278

theorem additional_airplanes_needed (total_current_airplanes : ℕ) (airplanes_per_row : ℕ) 
  (h_current_airplanes : total_current_airplanes = 37) 
  (h_airplanes_per_row : airplanes_per_row = 8) : 
  ∃ additional_airplanes : ℕ, additional_airplanes = 3 ∧ 
  ((total_current_airplanes + additional_airplanes) % airplanes_per_row = 0) :=
by
  sorry

end additional_airplanes_needed_l529_529278


namespace largest_x_exists_largest_x_largest_real_number_l529_529082

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : x ≤ 48 / 7 :=
sorry

theorem exists_largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  ∃ x, (⌊x⌋ : ℝ) / x = 7 / 8 ∧ x = 48 / 7 :=
sorry

theorem largest_real_number (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  x = 48 / 7 :=
sorry

end largest_x_exists_largest_x_largest_real_number_l529_529082


namespace banana_distinct_arrangements_l529_529916

theorem banana_distinct_arrangements :
  let n := 6
  let f_B := 1
  let f_N := 2
  let f_A := 3
  (n.factorial) / (f_B.factorial * f_N.factorial * f_A.factorial) = 60 := by
sorry

end banana_distinct_arrangements_l529_529916


namespace problem_l529_529153

noncomputable def incorrect_statement_d (α β : Plane) (m n : Line) : Prop :=
m ∥ α → α ∩ β = n → ¬ (m ∥ n)

theorem problem (α β : Plane) (m n : Line) (h1 : α ≠ β) (h2 : m ≠ n) :
  incorrect_statement_d α β m n :=
by
  sorry

end problem_l529_529153


namespace smallest_k_l529_529607

-- Define the 100-gon and the initial arrangement of chips
def is_initial_arrangement (arrangement : list ℕ) : Prop :=
  arrangement = list.range' 1 100

-- Allowed move: swap two adjacent chips if their numbers differ by no more than k
def allowed_move (k : ℕ) (arrangement : list ℕ) (i : ℕ) : Prop :=
  i < 99 ∧ (arrangement.nth i).bind (λ a, (arrangement.nth (i + 1)).map (λ b, |a - b| ≤ k)) = some tt

-- Function to check if the arrangement is valid after a series of allowed moves
def valid_arrangement_after_moves (k : ℕ) (initial_arrangement final_arrangement : list ℕ) : Prop :=
  ∃ moves : list ℕ, moves.length > 0 ∧
  (∀ i, i < moves.length → allowed_move k initial_arrangement (moves.nth i).iget) ∧
  list.rotate initial_arrangement 1 = final_arrangement

-- The main theorem statement
theorem smallest_k (initial_arrangement : list ℕ) : ∃ (k : ℕ), k = 50 ∧ 
  ∀ final_arrangement, valid_arrangement_after_moves k initial_arrangement final_arrangement ↔
  list.rotate initial_arrangement 1 = final_arrangement := 
begin
  -- Placeholder for the proof
  sorry
end

end smallest_k_l529_529607


namespace product_series_value_l529_529489

theorem product_series_value :
  (∏ k in (finset.range 200).filter (λ k, k > 0), (1 - (1 / (k + 1)))) * (1 + (1 / 100)) = 101 / 20000 :=
by sorry

end product_series_value_l529_529489


namespace wage_recovery_l529_529928

theorem wage_recovery (W : ℝ) (h : W > 0) : (1 - 0.3) * W * (1 + 42.86 / 100) = W :=
by
  sorry

end wage_recovery_l529_529928


namespace KarleeRemainingFruits_l529_529620

def KarleeInitialGrapes : ℕ := 100
def StrawberryRatio : ℚ := 3 / 5
def PortionGivenToFriends : ℚ := 1 / 5

theorem KarleeRemainingFruits :
  let initialGrapes := KarleeInitialGrapes
  let initialStrawberries := (StrawberryRatio * initialGrapes).to_nat
  let grapesGivenPerFriend := (PortionGivenToFriends * initialGrapes).to_nat
  let totalGrapesGiven := 2 * grapesGivenPerFriend
  let remainingGrapes := initialGrapes - totalGrapesGiven
  let strawberriesGivenPerFriend := (PortionGivenToFriends * initialStrawberries).to_nat
  let totalStrawberriesGiven := 2 * strawberriesGivenPerFriend
  let remainingStrawberries := initialStrawberries - totalStrawberriesGiven
  let totalRemainingFruits := remainingGrapes + remainingStrawberries
  totalRemainingFruits = 96 :=
by
  sorry

end KarleeRemainingFruits_l529_529620


namespace large_number_appears_l529_529146

theorem large_number_appears (a : ℕ → ℕ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_initial : ∀ i, i ≤ 2020 → a i > 0)
  (h_recursive : ∀ n, n ≥ 2021 → a n = Nat.find (λ x, x > 0 ∧ (∀ m < n, a m ≠ x) ∧ ¬ (x ∣ List.prod (List.map a (List.range' (n - 2020) 2020))))) :
  ∀ k, ∃ n, n ≥ k ∧ a n = k :=
sorry

end large_number_appears_l529_529146


namespace probability_window_opens_correct_l529_529452

noncomputable def probability_window_opens_no_later_than_3_minutes_after_scientist_arrives 
  (arrival_times : Fin 6 → ℝ) : ℝ :=
  if (∀ i, arrival_times i ∈ Set.Icc 0 15) ∧ 
     (∀ i j, i ≠ j → arrival_times i < arrival_times j) ∧ 
     ((∃ i, arrival_times i ≥ 12)) then
    1 - (0.8 ^ 6)
  else
    0

theorem probability_window_opens_correct : 
  ∀ (arrival_times : Fin 6 → ℝ),
    (∀ i, arrival_times i ∈ Set.Icc 0 15) →
    (∀ i j, i ≠ j → arrival_times i < arrival_times j) →
    (∃ i, arrival_times i = arrival_times 5) →
    abs (probability_window_opens_no_later_than_3_minutes_after_scientist_arrives arrival_times - 0.738) < 0.001 :=
by
  sorry

end probability_window_opens_correct_l529_529452


namespace values_of_a_l529_529996

axiom exists_rat : (x y a : ℚ) → Prop

theorem values_of_a (a : ℚ) (h1 : ∀ x y : ℚ, (x/2 - (2*x - 3*y)/5 = a - 1)) (h2 : ∀ x y : ℚ, (x + 3 = y/3)) :
  0.7 < a ∧ a < 6.4 ↔ (∃ x y : ℚ, x < 0 ∧ y > 0) :=
by
  sorry

end values_of_a_l529_529996


namespace number_in_10th_row_5th_column_l529_529798

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem number_in_10th_row_5th_column :
  let diagonal_14 := triangular_number 14 in
  let diagonal_13 := triangular_number 13 in
  let start_10th_row := diagonal_13 + 1 in
  (start_10th_row + 4) = 101 := 
by
  let diagonal_14 := triangular_number 14
  let diagonal_13 := triangular_number 13
  let start_10th_row := diagonal_13 + 1
  show (start_10th_row + 4) = 101
  sorry

end number_in_10th_row_5th_column_l529_529798


namespace yolanda_three_point_avg_l529_529379

-- Definitions based on conditions
def total_points_season := 345
def total_games := 15
def free_throws_per_game := 4
def two_point_baskets_per_game := 5

-- Definitions based on the derived quantities
def average_points_per_game := total_points_season / total_games
def points_from_two_point_baskets := two_point_baskets_per_game * 2
def points_from_free_throws := free_throws_per_game * 1
def points_from_non_three_point_baskets := points_from_two_point_baskets + points_from_free_throws
def points_from_three_point_baskets := average_points_per_game - points_from_non_three_point_baskets
def three_point_baskets_per_game := points_from_three_point_baskets / 3

-- The theorem to prove that Yolanda averaged 3 three-point baskets per game
theorem yolanda_three_point_avg:
  three_point_baskets_per_game = 3 := sorry

end yolanda_three_point_avg_l529_529379


namespace correct_graph_l529_529211
-- Define the conditions
def pct_1990 : ℝ := 30
def pct_2000 : ℝ := 45
def pct_2010 : ℝ := 60
def pct_2020 : ℝ := 82

-- The key proof statement: Given the conditions about the percentages from 1990 to 2020,
-- prove that Graph D represents the data correctly.

theorem correct_graph :
  (if pct_1990 = 30 ∧
      pct_2000 = 45 ∧ 
      pct_2010 = 60 ∧ 
      pct_2020 = 82 
   then Graph D)
  := by
    sorry

end correct_graph_l529_529211


namespace rates_sum_of_squares_l529_529829

theorem rates_sum_of_squares :
  ∃ (b j s : ℕ), 3 * b + 2 * j + 4 * s = 84 ∧ 4 * b + 3 * j + 2 * s = 106 ∧ b^2 + j^2 + s^2 = 1125 :=
begin
  sorry
end

end rates_sum_of_squares_l529_529829


namespace heartsuit_sum_l529_529127

-- Define the function heartsuit
def heartsuit (x : ℝ) : ℝ := (x + x^2 + x^3) / 3

-- State the theorem
theorem heartsuit_sum :
  heartsuit 1 + heartsuit 2 + heartsuit 4 = 101 / 3 :=
by
  sorry

end heartsuit_sum_l529_529127


namespace bruised_more_than_wormy_l529_529275

noncomputable def total_apples : ℕ := 85
noncomputable def fifth_of_apples (n : ℕ) : ℕ := n / 5
noncomputable def apples_left_to_eat_raw : ℕ := 42

noncomputable def wormy_apples : ℕ := fifth_of_apples total_apples
noncomputable def total_non_raw_eatable_apples : ℕ := total_apples - apples_left_to_eat_raw
noncomputable def bruised_apples : ℕ := total_non_raw_eatable_apples - wormy_apples

theorem bruised_more_than_wormy :
  bruised_apples - wormy_apples = 43 - 17 :=
by sorry

end bruised_more_than_wormy_l529_529275


namespace part_I_part_II_l529_529807

-- Problem (I)
theorem part_I :
  (Nat.choose 5 2 + Nat.choose 5 3) / Nat.fact (5 - 3) = (1 / 3) := sorry

-- Problem (II)
def f (x : ℝ) : ℝ := Real.exp (-x) * Real.sin (2 * x)

theorem part_II :
  ∀ x : ℝ, deriv f x = Real.exp (-x) * (- Real.sin (2 * x) + 2 * Real.cos (2 * x)) := sorry

end part_I_part_II_l529_529807


namespace quadratic_root_a_value_l529_529582

theorem quadratic_root_a_value (a : ℝ) :
  (∃ x : ℝ, x = -2 ∧ x^2 + (3 / 2) * a * x - a^2 = 0) → (a = 1 ∨ a = -4) := 
by
  intro h
  sorry

end quadratic_root_a_value_l529_529582


namespace log_relationship_l529_529545

open Real

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then log x / log 2 else f (-x)

theorem log_relationship :
  let a := f (-3)
  let b := f (1 / 4)
  let c := f 2
  a > c ∧ c > b :=
by
  let a := f (-3)
  let b := f (1 / 4)
  let c := f 2
  sorry

end log_relationship_l529_529545


namespace minimum_edges_for_two_triangles_l529_529143

theorem minimum_edges_for_two_triangles (n : ℕ) (h : n ≥ 5) :
  ∃ (m : ℕ), (∀ (G : SimpleGraph (Fin n)), (G.edgeCount = m) → 
  (∃ (u : Fin n) (T₁ T₂ : triangle G), (T₁ ≠ T₂) ∧ (u ∈ T₁) ∧ (u ∈ T₂))) 
  ∧ m = (n^2 \div 4) + 2 := 
sorry

end minimum_edges_for_two_triangles_l529_529143


namespace inequality_proof_l529_529522

variable {n : ℕ}
variable {x : Fin n → ℝ}

theorem inequality_proof
  (h₁ : ∀ i, 0 < x i)
  (h₂ : 2 ≤ n)
  (h₃ : (∑ i, x i) = 1) :
  (∑ i, x i / real.sqrt (1 - x i)) ≥ (∑ i, real.sqrt (x i)) / real.sqrt (n - 1) :=
by
  sorry

end inequality_proof_l529_529522


namespace correct_simplification_l529_529784

-- Step 1: Define the initial expression
def initial_expr (a b : ℝ) : ℝ :=
  (a - b) / a / (a - (2 * a * b - b^2) / a)

-- Step 2: Define the correct simplified form
def simplified_expr (a b : ℝ) : ℝ :=
  1 / (a - b)

-- Step 3: State the theorem that proves the simplification is correct
theorem correct_simplification (a b : ℝ) (h : a ≠ b): 
  initial_expr a b = simplified_expr a b :=
by {
  sorry,
}

end correct_simplification_l529_529784


namespace constant_term_in_first_equation_l529_529513

/-- Given the system of equations:
  1. 5x + y = C
  2. x + 3y = 1
  3. 3x + 2y = 10
  Prove that the constant term C is 19.
-/
theorem constant_term_in_first_equation
  (x y C : ℝ)
  (h1 : 5 * x + y = C)
  (h2 : x + 3 * y = 1)
  (h3 : 3 * x + 2 * y = 10) :
  C = 19 :=
by
  sorry

end constant_term_in_first_equation_l529_529513


namespace quotient_of_sum_of_squares_mod_13_l529_529678

theorem quotient_of_sum_of_squares_mod_13 :
  let n_list := [1, 2, 3, 4, 5, 6, 7]
  let remainders := n_list.map (λ n => (n * n) % 13)
  let distinct_remainders := remainders.erasedup
  let m := distinct_remainders.sum
  m / 13 = 3 :=
by
  let n_list := [1, 2, 3, 4, 5, 6, 7]
  let remainders := n_list.map (λ n => (n * n) % 13)
  let distinct_remainders := remainders.erasedup
  let m := distinct_remainders.sum
  have h : m = 39 := sorry --proof of m = 39 goes here
  show 39 / 13 = 3
  calc
    39 / 13 = 3 : by norm_num

end quotient_of_sum_of_squares_mod_13_l529_529678


namespace perimeter_of_region_l529_529775

-- Define a square with side length 4/π
def side_length : ℝ := 4 / Real.pi

-- Define the diameter of semicircles formed on each side of the square
def diameter : ℝ := side_length

-- Define the radius of each semicircle
def radius : ℝ := diameter / 2

-- Define the circumference of one semicircle
def semicircumference : ℝ := Real.pi * (diameter / 2) / 2

-- Define the number of semicircular arcs
def number_of_semicircles : ℕ := 4

-- Define the total perimeter of the region bounded by the semicircular arcs
def total_perimeter : ℝ := number_of_semicircles * semicircumference

-- The theorem to be proved: the total perimeter is 8
theorem perimeter_of_region : total_perimeter = 8 := by
  sorry

end perimeter_of_region_l529_529775


namespace min_sum_reciprocal_l529_529159

theorem min_sum_reciprocal (a b c : ℝ) (hp0 : 0 < a) (hp1 : 0 < b) (hp2 : 0 < c) (h : a + b + c = 1) : 
  (1 / a) + (1 / b) + (1 / c) ≥ 9 :=
by
  sorry

end min_sum_reciprocal_l529_529159


namespace find_value_l529_529973

variable (a₁ a₂ a₃ a₄ a₅ : ℝ)
variable (h_conditions : ∀ k, k ∈ {1, 2, 3, 4, 5} → 
  (a₁ / (k^2 + 1) + a₂ / (k^2 + 2) + a₃ / (k^2 + 3) + a₄ / (k^2 + 4) + a₅ / (k^2 + 5) = 1 / k^2))

theorem find_value : 
  (a₁ / 37 + a₂ / 38 + a₃ / 39 + a₄ / 40 + a₅ / 41) = 187465 / 6744582 := by
  sorry

end find_value_l529_529973


namespace solutions_P_square_prime_l529_529819

def P (n : ℤ) : ℤ := n^3 - n^2 - 5n + 2

def is_prime (p : ℤ) : Prop := p > 1 ∧ (∀ d : ℤ, d ∣ p → d = 1 ∨ d = p)

theorem solutions_P_square_prime :
  ∀ n : ℤ, (∃ p : ℕ, P(n)^2 = p^2 ∧ is_prime p) ↔ 
    n = -1 ∨ n = -3 ∨ n = 0 ∨ n = 3 ∨ n = 1 := sorry

end solutions_P_square_prime_l529_529819


namespace simplify_fraction_l529_529674

theorem simplify_fraction (a b : ℕ) (h : a = 150) (hb : b = 450) : a / b = 1 / 3 := by
  sorry

end simplify_fraction_l529_529674


namespace logans_model_height_l529_529277

noncomputable def scaling_factor : ℝ := (200000 / 0.05)^(1/3)

def actual_height : ℝ := 60

def model_height : ℝ := actual_height / scaling_factor

theorem logans_model_height : model_height = 0.3 := by
  -- proof omitted
  sorry

end logans_model_height_l529_529277


namespace inequality_proof_l529_529640

theorem inequality_proof (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 1) : 
  a^4 + b^4 + c^4 ≥ a * b * c := 
by {
  sorry
}

end inequality_proof_l529_529640


namespace total_crosswalk_lines_l529_529375

theorem total_crosswalk_lines (n m l : ℕ) (h1 : n = 5) (h2 : m = 4) (h3 : l = 20) :
  n * (m * l) = 400 := by
  sorry

end total_crosswalk_lines_l529_529375


namespace probability_dice_l529_529792

noncomputable def probability_of_three_dice_less_than_six (n_dice : ℕ) (faces_per_die : ℕ) : ℚ :=
let p_success := 5 / 12 in
let p_failure := 7 / 12 in
let combinations := nat.choose 7 3 in
combinations * (p_success ^ 3) * (p_failure ^ 4)

theorem probability_dice :
  probability_of_three_dice_less_than_six 7 12 = 10504375 / 373248 :=
by sorry

end probability_dice_l529_529792


namespace rain_and_humidity_probability_l529_529790

theorem rain_and_humidity_probability 
  (P_Rain : ℝ) 
  (P_Humidity_given_Rain : ℝ) 
  (h1 : P_Rain = 0.4) 
  (h2 : P_Humidity_given_Rain = 0.6) : 
  P_Rain * P_Humidity_given_Rain = 0.24 :=
by
  rw [h1, h2]
  norm_num
  sorry

end rain_and_humidity_probability_l529_529790


namespace exists_non_tetrahedron_polyhedron_with_property_l529_529776

def Point : Type := ℝ × ℝ × ℝ

structure Polyhedron :=
(vertices : set Point)
(faces : set (set Point))
(property : ∀ (A B : Point), A ≠ B → ∃ C : Point, {A, B, C} ∈ faces ∧ dist A C = dist B C ∧ dist A B = dist A C)

theorem exists_non_tetrahedron_polyhedron_with_property :
  ∃ P : Polyhedron, (∃ {A B C D : Point}, {A, B, C, D} ⊆ P.vertices ∧ A ≠ D) ∧ P.property := 
sorry

end exists_non_tetrahedron_polyhedron_with_property_l529_529776


namespace rectangle_center_sum_l529_529666

theorem rectangle_center_sum (A B C D : ℝ × ℝ) 
  (hA : A = (6, 4)) 
  (hB : B = (8, 2)) 
  (hDA : ∃ m : ℝ, ∀ x : ℝ, y = m * (x - 2)) 
  (hCB : ∃ m : ℝ, ∀ x : ℝ, y = m * (x - 6)) 
  (hAB : ∃ m : ℝ, ∀ x : ℝ, y = - 1 / m * (x - 10)) 
  (hCD : ∃ m : ℝ, ∀ x : ℝ, y = - 1 / m * (x - 18)) 
  (ratio : ∃ k : ℝ, k = 2) :
  let center := ((A.1 + C.1) / 2, (A.2 + C.2) / 2) in
  center.1 + center.2 = 10 := sorry

end rectangle_center_sum_l529_529666


namespace parabola_directrix_l529_529321

theorem parabola_directrix (y : ℝ) : (∃ p : ℝ, x = (1 / (4 * p)) * y^2 ∧ p = 2) → x = -2 :=
by
  sorry

end parabola_directrix_l529_529321


namespace remainder_when_divided_by_5_l529_529967

variable (b: Fin 20 → ℕ)

def strictly_increasing (b: Fin 20 → ℕ) : Prop :=
  ∀ (i j: Fin 20), i < j → b i < b j

def positive_integers (b: Fin 20 → ℕ) : Prop :=
  ∀ i, 0 < b i

def sum_eq_420 (b: Fin 20 → ℕ) : Prop :=
  (∑ i, b i) = 420

theorem remainder_when_divided_by_5 
  (b: Fin 20 → ℕ) 
  (h_inc: strictly_increasing b)
  (h_pos: positive_integers b)
  (h_sum: sum_eq_420 b) : (∑ i, (b i)^2) % 5 = 0 := 
by 
  sorry

end remainder_when_divided_by_5_l529_529967


namespace max_distance_AB_proof_l529_529875

noncomputable def max_distance_AB (a b c : ℝ) (ha : a ≥ b) (hb : b ≥ c) (h_sum : a + b + c = 0) (h_neq : a ≠ 0) :
    real :=
  let x1 := (-b + sqrt (b^2 - 4 * a * c)) / (2 * a)
  let x2 := (-b - sqrt (b^2 - 4 * a * c)) / (2 * a)
  sqrt(2) * (3 / 2)

theorem max_distance_AB_proof (a b c : ℝ) (ha : a ≥ b) (hb : b ≥ c) (h_sum : a + b + c = 0) (h_neq : a ≠ 0) :
    max_distance_AB a b c ha hb h_sum h_neq = sqrt(2) * (3 / 2) :=
by
  -- Proof omitted
  sorry

end max_distance_AB_proof_l529_529875


namespace kylie_beads_total_l529_529622

def number_necklaces_monday : Nat := 10
def number_necklaces_tuesday : Nat := 2
def number_bracelets_wednesday : Nat := 5
def number_earrings_wednesday : Nat := 7

def beads_per_necklace : Nat := 20
def beads_per_bracelet : Nat := 10
def beads_per_earring : Nat := 5

theorem kylie_beads_total :
  (number_necklaces_monday + number_necklaces_tuesday) * beads_per_necklace + 
  number_bracelets_wednesday * beads_per_bracelet + 
  number_earrings_wednesday * beads_per_earring = 325 := 
by
  sorry

end kylie_beads_total_l529_529622


namespace find_alpha_l529_529155

-- Declare the conditions
variables (α : ℝ) (h₀ : 0 < α) (h₁ : α < 90) (h₂ : Real.sin (α - 10 * Real.pi / 180) = Real.sqrt 3 / 2)

theorem find_alpha : α = 70 * Real.pi / 180 :=
sorry

end find_alpha_l529_529155


namespace nine_segments_three_intersections_impossible_l529_529240

theorem nine_segments_three_intersections_impossible : ¬ ∃ (G : SimpleGraph (Fin 9)), (∀ v, G.degree v = 3) :=
by
  sorry

end nine_segments_three_intersections_impossible_l529_529240


namespace ordered_triples_lcm_l529_529631

theorem ordered_triples_lcm:
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
    (∀ trip ∈ S, 
      let (a, b, c) := trip in 
      ([a, b] = 1200 ∧ [b, c] = 2400 ∧ [c, a] = 2400)) ∧ 
    S.card = 3 :=
by sorry

end ordered_triples_lcm_l529_529631


namespace Lindsay_dolls_ratio_l529_529649

theorem Lindsay_dolls_ratio :
  ∃ (B K : ℕ), 
    4 + 26 = B + K ∧
    K = B - 2 ∧
    B / 4 = 4 :=
begin
  sorry
end

end Lindsay_dolls_ratio_l529_529649


namespace circle_eq_proof_PA_dot_PB_range_proof_l529_529138

-- Definitions of variables and conditions
variables (a b r x y : ℝ)
noncomputable def circle_eq : (x - a) ^ 2 + (y - b) ^ 2 = r ^ 2 := sorry

axiom chord_lengths_on_axes : (b ^ 2 + 5 = r ^ 2) ∧ (a ^ 2 + 8 = r ^ 2)
axiom center_condition : 3 * a + b = 1 ∧ a > 0 ∧ b < 0
axiom geom_sequence_condition : (x - 1) ^ 2 - y ^ 2 = 5 / 2

def circle_c_eq := (x - 1) ^ 2 + (y + 2) ^ 2 = 9

theorem circle_eq_proof : circle_eq = circle_c_eq := by sorry

def PA_dot_PB (P : ℝ × ℝ) : ℝ := 
  let (x₀, y₀) := P in
  (x₀ - 1 - sqrt 5) * (x₀ - 1 + sqrt 5) + y₀ ^ 2 - 5

noncomputable def PA_dot_PB_range : ℝ × ℝ :=
  (let (y₀, _) := P in (2 * (y₀ ^ 2) - 5 / 2, 10))

theorem PA_dot_PB_range_proof (P : ℝ × ℝ) : 
  PA_dot_PB P ∈ PA_dot_PB_range P := by sorry

end circle_eq_proof_PA_dot_PB_range_proof_l529_529138


namespace panda_age_probability_l529_529704

theorem panda_age_probability :
  (P_living_10 : ℚ) (P_living_15 : ℚ)
  (h1 : P_living_10 = 0.8)
  (h2 : P_living_15 = 0.6) :
  P_living_15 / P_living_10 = 0.75 :=
by
  -- Definitions and assumptions imported directly from the conditions
  -- sorry added to skip the proof
  sorry

end panda_age_probability_l529_529704


namespace segments_equal_with_angle_l529_529989

-- Variables and conditions
variables (ABCD : Type*) [add_comm_group ABCD] [module ℝ ABCD]
variables (A B C D O1 O2 O3 O4 : ABCD)
variables (alpha : ℝ)

-- Definitions related to rhombuses and their properties
def is_center_of_rhombus (O1 A B : ABCD) (alpha : ℝ) : Prop := sorry
def is_convex_quadrilateral (A B C D : ABCD) : Prop := sorry
def angles_of_rhombuses_adjacent_equal (O1 O2 A : ABCD) (alpha : ℝ) : Prop := sorry

-- Main theorem to be proven
theorem segments_equal_with_angle
  (convex : is_convex_quadrilateral A B C D)
  (rhombus1 : is_center_of_rhombus O1 A B alpha)
  (rhombus2 : is_center_of_rhombus O2 B C alpha)
  (rhombus3 : is_center_of_rhombus O3 C D alpha)
  (rhombus4 : is_center_of_rhombus O4 D A alpha)
  (angles_equal : angles_of_rhombuses_adjacent_equal O1 O2 A alpha) :
  ∃ (O1 O2 O3 O4 : ABCD), (O1 - O3) = (O2 - O4) ∧ ∠(O1 - O3, O2 - O4) = alpha := 
sorry

end segments_equal_with_angle_l529_529989


namespace cashier_window_open_probability_l529_529441

noncomputable def probability_window_opens_in_3_minutes_of_scientist_arrival : ℝ := 
  0.738

theorem cashier_window_open_probability :
  let x : ℝ → ℝ := λ x, if x ≥ 12 then 0.738 else 0.262144 in
  ∀ (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ),
    (∀ i ∈ [x₁, x₂, x₃, x₄, x₅], i < x₆) ∧ 0 ≤ x₆ ≤ 15 →
    x x₆ = 0.738 :=
by
  sorry

end cashier_window_open_probability_l529_529441


namespace exists_100_distinct_naturals_cubed_sum_l529_529482

theorem exists_100_distinct_naturals_cubed_sum : 
  ∃ (S : Finset ℕ), S.card = 100 ∧ 
  ∃ (x : ℕ), x ∈ S ∧ 
  x ^ 3 = (S.erase x).sum (λ y, y ^ 3) :=
begin
  sorry
end

end exists_100_distinct_naturals_cubed_sum_l529_529482


namespace red_cells_in_9x11_l529_529983

-- Define the type to represent color of cells
inductive Color
| red
| blue

-- Define the condition for a 2x3 rectangle containing exactly 2 red cells
def valid_2x3 (rect : List (List Color)) : Prop :=
  rect.length = 2 ∧ (rect.all (fun row => row.length = 3)) ∧
  (rect.toVector.flatten.count Col.red = 2)

-- Define the function to validate the entire grid
def valid_grid (grid : List (List Color)) : Prop :=
  ∀ (i : ℕ) (j : ℕ), (i + 1 < grid.length) ∧ (j + 2 < grid.head.length ∧ 
  (valid_2x3 (grid.drop i |>.take 2).map (List.drop j |> List.take 3)))

-- Formalize the problem statement to prove that in a 9x11 grid meeting the condition, there are exactly 33 red cells
theorem red_cells_in_9x11 (grid : List (List Color)) 
  (h : grid.length = 9 ∧ grid.head.length = 11 ∧ valid_grid grid) : 
  grid.flatten.count Color.red = 33 :=
sorry

end red_cells_in_9x11_l529_529983


namespace same_grade_percentage_is_correct_l529_529006

-- Definition of the grade distribution table
def grade_distribution : (Nat × Nat × Nat × Nat) :=
  (3, 5, 6, 3)  -- Diagonal values (AA, BB, CC, DD)

-- Total number of students
def total_students : Nat := 40

-- Calculate the number of students with the same grade on both quizzes
def same_grade_students (grades: Nat × Nat × Nat × Nat) : Nat :=
  grades.1 + grades.2 + grades.3 + grades.4

-- Calculate the percentage of students with the same grade
def same_grade_percentage (same_grade: Nat) (total: Nat) : Float :=
  (same_grade.toFloat / total.toFloat) * 100

-- The main theorem
theorem same_grade_percentage_is_correct :
  same_grade_percentage (same_grade_students grade_distribution) total_students = 42.5 := by
  sorry

end same_grade_percentage_is_correct_l529_529006


namespace find_alpha_l529_529154

-- Declare the conditions
variables (α : ℝ) (h₀ : 0 < α) (h₁ : α < 90) (h₂ : Real.sin (α - 10 * Real.pi / 180) = Real.sqrt 3 / 2)

theorem find_alpha : α = 70 * Real.pi / 180 :=
sorry

end find_alpha_l529_529154


namespace chocolates_difference_l529_529990

theorem chocolates_difference :
  ∀ (robert_chocolates nickel_chocolates : ℕ), robert_chocolates = 13 → nickel_chocolates = 4 → robert_chocolates - nickel_chocolates = 9 :=
by
  intros robert_chocolates nickel_chocolates h_robert h_nickel
  rw [h_robert, h_nickel]
  exact rfl

end chocolates_difference_l529_529990


namespace min_value_2a_plus_b_l529_529148

noncomputable def min_value (a b : ℝ) := (2 * a + b)

theorem min_value_2a_plus_b (a b : ℝ) (h : a * b * (a + b) = 4) (ha : 0 < a) (hb : 0 < b) : 
  ∃ x, x = 2 * real.sqrt(3) ∧ ∀ y, (2 * a + b) y ≥ 2 * real.sqrt(3) := sorry

end min_value_2a_plus_b_l529_529148


namespace anusha_solution_l529_529991

variable (A B E : ℝ) -- Defining the variables for amounts received by Anusha, Babu, and Esha
variable (total_amount : ℝ) (h_division : 12 * A = 8 * B) (h_division2 : 8 * B = 6 * E) (h_total : A + B + E = 378)

theorem anusha_solution : A = 84 :=
by
  -- Using the given conditions and deriving the amount Anusha receives
  sorry

end anusha_solution_l529_529991


namespace correct_proposition_C_l529_529881

-- Declaring the entities
variable (m n : Line)
variable (alpha beta : Plane)

-- Declaring the conditions
variable (par_m_n : IsParallel m n)
variable (perp_m_alpha : IsPerpendicular m alpha)
variable (par_n_beta : IsParallel n beta)

-- The theorem that we aim to prove
theorem correct_proposition_C (h1 : IsPerpendicular m alpha) (h2 : IsParallel m n) (h3 : IsParallel n beta) : IsPerpendicular alpha beta := 
sorry

end correct_proposition_C_l529_529881


namespace correct_option_l529_529372

def condition_A : Prop := abs ((-5 : ℤ)^2) = -5
def condition_B : Prop := abs (9 : ℤ) = 3 ∨ abs (9 : ℤ) = -3
def condition_C : Prop := abs (3 : ℤ) / abs (((-2)^3 : ℤ)) = -2
def condition_D : Prop := (2 * abs (3 : ℤ))^2 = 6 

theorem correct_option : ¬condition_A ∧ ¬condition_B ∧ condition_C ∧ ¬condition_D :=
by
  sorry

end correct_option_l529_529372


namespace probability_window_opens_correct_l529_529453

noncomputable def probability_window_opens_no_later_than_3_minutes_after_scientist_arrives 
  (arrival_times : Fin 6 → ℝ) : ℝ :=
  if (∀ i, arrival_times i ∈ Set.Icc 0 15) ∧ 
     (∀ i j, i ≠ j → arrival_times i < arrival_times j) ∧ 
     ((∃ i, arrival_times i ≥ 12)) then
    1 - (0.8 ^ 6)
  else
    0

theorem probability_window_opens_correct : 
  ∀ (arrival_times : Fin 6 → ℝ),
    (∀ i, arrival_times i ∈ Set.Icc 0 15) →
    (∀ i j, i ≠ j → arrival_times i < arrival_times j) →
    (∃ i, arrival_times i = arrival_times 5) →
    abs (probability_window_opens_no_later_than_3_minutes_after_scientist_arrives arrival_times - 0.738) < 0.001 :=
by
  sorry

end probability_window_opens_correct_l529_529453


namespace smallest_a10_l529_529633

noncomputable theory
open_locale big_operators

def SumSet (S : set ℕ) : ℕ := ∑ x in S, x

theorem smallest_a10 (A : finset ℕ) (hA : A.card = 11)
  (h_sorted : ∀ a b ∈ A, a < b → ∃ i j : ℕ, a = A.min' finset.nonempty_coe_sort ∨ b = A.max' finset.nonempty_coe_sort)
  (h_sum : ∀ n : ℕ, n ≤ 1500 → ∃ S ⊆ A, n = SumSet S) :
  ∃ a ∈ A, a = 248 :=
begin
  sorry
end

end smallest_a10_l529_529633


namespace four_digit_numbers_subtract_sum_digits_l529_529121

theorem four_digit_numbers_subtract_sum_digits : 
  {n : ℕ | let s := n.toString.toList.map (λ c, c.toNat - '0'.toNat) in 
             s.sum ≤ 36 ∧ 
             n ≥ 1000 ∧ 
             n < 10000 ∧ 
             n - s.sum = 2007} = 
  {2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019} :=
by
  sorry

end four_digit_numbers_subtract_sum_digits_l529_529121


namespace find_rhombus_acute_angle_l529_529837

-- Definitions and conditions
def rhombus_angle (V1 V2 : ℝ) (α : ℝ) : Prop :=
  V1 / V2 = 1 / (2 * Real.sqrt 5)
  
-- Theorem statement
theorem find_rhombus_acute_angle (V1 V2 a : ℝ) (α : ℝ) (h : rhombus_angle V1 V2 α) :
  α = Real.arccos (1 / 9) :=
sorry

end find_rhombus_acute_angle_l529_529837


namespace represent_nat_as_combinations_l529_529994

theorem represent_nat_as_combinations (n : ℕ) :
  ∃ x y z : ℕ,
  (0 ≤ x ∧ x < y ∧ y < z ∨ 0 = x ∧ x = y ∧ y < z) ∧
  (n = Nat.choose x 1 + Nat.choose y 2 + Nat.choose z 3) :=
sorry

end represent_nat_as_combinations_l529_529994


namespace limit_of_sequence_l529_529709

noncomputable def sequence := ℕ → ℝ
variables {a : sequence}

def a1 : ℝ := π / 4

def a_n (n : ℕ) (prev : ℝ) : ℝ :=
  ∫ x in 0..(1/2), (cos (π * x) + prev) * cos (π * x)

theorem limit_of_sequence :
  (∃ L, Tendsto (λ n, a n) atTop (𝓝 L) ∧ L = π / (4 * (π - 1))) :=
by
  sorry

end limit_of_sequence_l529_529709


namespace senior_students_in_sample_l529_529007

theorem senior_students_in_sample 
  (total_students : ℕ) (total_seniors : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 2000)
  (h2 : total_seniors = 500)
  (h3 : sample_size = 200) : 
  (total_seniors * sample_size / total_students = 50) :=
by {
  sorry
}

end senior_students_in_sample_l529_529007


namespace common_root_and_param_l529_529516

theorem common_root_and_param :
  ∀ (x : ℤ) (P p : ℚ),
    (P = -((x^2 - x - 2) / (x - 1)) ∧ x ≠ 1) →
    (p = -((x^2 + 2*x - 1) / (x + 2)) ∧ x ≠ -2) →
    (-x + (2 / (x - 1)) = -x + (1 / (x + 2))) →
    x = -5 ∧ p = 14 / 3 :=
by
  intros x P p hP hp hroot
  sorry

end common_root_and_param_l529_529516


namespace largest_real_solution_l529_529103

theorem largest_real_solution (x : ℝ) (h : (⌊x⌋ / x = 7 / 8)) : x ≤ 48 / 7 := by
  sorry

end largest_real_solution_l529_529103


namespace problem_statement_l529_529999

namespace ProofProblem

variable (t : ℚ) (y : ℚ)

/-- Given equations and condition, we want to prove y = 21 / 2 -/
theorem problem_statement (h1 : x = 3 - 2 * t) (h2 : y = 3 * t + 6) (h3 : x = 0) : y = 21 / 2 :=
by sorry

end ProofProblem

end problem_statement_l529_529999


namespace largest_x_satisfies_condition_l529_529098

theorem largest_x_satisfies_condition :
  ∃ x : ℝ, (⌊x⌋ / x = 7 / 8) ∧ (∀ y : ℝ, (⌊y⌋ / y = 7 / 8) → y ≤ 48 / 7) :=
sorry

end largest_x_satisfies_condition_l529_529098


namespace minimum_area_sum_l529_529151

-- Define the coordinates and the conditions
variable {x1 y1 x2 y2 : ℝ}
variable (on_parabola_A : y1^2 = x1)
variable (on_parabola_B : y2^2 = x2)
variable (y1_pos : y1 > 0)
variable (y2_neg : y2 < 0)
variable (dot_product : x1 * x2 + y1 * y2 = 2)

-- Define the function to calculate areas
noncomputable def area_sum (y1 y2 x1 x2 : ℝ) : ℝ :=
  1/2 * 2 * (y1 - y2) + 1/2 * 1/4 * y1

theorem minimum_area_sum :
  ∃ y1 y2 x1 x2, y1^2 = x1 ∧ y2^2 = x2 ∧ y1 > 0 ∧ y2 < 0 ∧ x1 * x2 + y1 * y2 = 2 ∧
  (area_sum y1 y2 x1 x2 = 3) := sorry

end minimum_area_sum_l529_529151


namespace probability_is_correct_l529_529437

noncomputable def probability_cashier_opens_early : ℝ :=
  let x1 : ℝ := sorry
  let x2 : ℝ := sorry
  let x3 : ℝ := sorry
  let x4 : ℝ := sorry
  let x5 : ℝ := sorry
  let x6 : ℝ := sorry
  if (0 <= x1) ∧ (x1 <= 15) ∧
     (0 <= x2) ∧ (x2 <= 15) ∧
     (0 <= x3) ∧ (x3 <= 15) ∧
     (0 <= x4) ∧ (x4 <= 15) ∧
     (0 <= x5) ∧ (x5 <= 15) ∧
     (0 <= x6) ∧ (x6 <= 15) ∧
     (x1 < x6) ∧ (x2 < x6) ∧ (x3 < x6) ∧ (x4 < x6) ∧ (x5 < x6) then 
    let p_not_A : ℝ := (12 / 15) ^ 6
    1 - p_not_A
  else
    0

theorem probability_is_correct : probability_cashier_opens_early = 0.738 :=
by sorry

end probability_is_correct_l529_529437


namespace parallelepiped_volume_half_l529_529924

noncomputable def volume_of_parallelepiped (a b : ℝ^3) : ℝ :=
  |a • ((b + 2 * (b × a)) × b)|

theorem parallelepiped_volume_half (a b : ℝ^3)
  (ha : ‖a‖ = 1)
  (hb : ‖b‖ = 1)
  (angle_ab : real.arccos (a • b / (‖a‖ * ‖b‖)) = π/4) : 
  volume_of_parallelepiped a b = 1/2 :=
sorry

end parallelepiped_volume_half_l529_529924


namespace find_x2_y2_l529_529833

theorem find_x2_y2 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : x * y + 2 * x + 2 * y = 152)
  (h2 : x^2 * y + x * y^2 = 1512) :
  x^2 + y^2 = 1136 ∨ x^2 + y^2 = 221 := by
  sorry

end find_x2_y2_l529_529833


namespace largest_x_satisfies_condition_l529_529090

theorem largest_x_satisfies_condition (x : ℝ) (h : (⌊x⌋ / x) = 7 / 8) : x ≤ 48 / 7 :=
sorry

end largest_x_satisfies_condition_l529_529090


namespace not_compact_sequence_1_compact_sequence_2_compact_geometric_sequence_l529_529274

-- Problem (1)
theorem not_compact_sequence_1 (a : ℕ → ℝ) (h_a : ∀ n, a n = (n^2 + 2*n)/(4^n)) :
  ¬ (∀ n, (1/2) ≤ a (n + 1) / a n ∧ a (n + 1) / a n ≤ 2) :=
sorry

-- Problem (2)
theorem compact_sequence_2 (S : ℕ → ℝ) (h_S : ∀ n, S n = (1/4) * (n^2 + 3*n)) :
  ∀ n, (1/2) ≤ (S (n+1) - S n) / (S n - S (n-1)) ∧ (S (n+1) - S n) / (S n - S (n-1)) ≤ 2 :=
sorry

-- Problem (3)
theorem compact_geometric_sequence (a S : ℕ → ℝ) (q : ℝ)
  (h_a : ∀ n, a n = a 0 * q^n)
  (h_S : ∀ n, S n = if q = 1 then (n + 1) * a 0 else a 0 * (1 - q^(n+1)) / (1 - q))
  (h_compact_a : ∀ n, (1/2) ≤ a (n + 1) / a n ∧ a (n + 1) / a n ≤ 2)
  (h_compact_S : ∀ n, (1/2) ≤ S (n + 1) / S n ∧ S (n + 1) / S n ≤ 2) :
  q ∈ Icc (1/2 : ℝ) 1 :=
sorry

end not_compact_sequence_1_compact_sequence_2_compact_geometric_sequence_l529_529274


namespace largest_x_satisfies_condition_l529_529091

theorem largest_x_satisfies_condition (x : ℝ) (h : (⌊x⌋ / x) = 7 / 8) : x ≤ 48 / 7 :=
sorry

end largest_x_satisfies_condition_l529_529091


namespace steve_pencils_left_l529_529681

def initial_pencils : ℕ := 2 * 12
def pencils_given_to_lauren : ℕ := 6
def pencils_given_to_matt : ℕ := pencils_given_to_lauren + 3

def pencils_left (initial_pencils given_lauren given_matt : ℕ) : ℕ :=
  initial_pencils - given_lauren - given_matt

theorem steve_pencils_left : pencils_left initial_pencils pencils_given_to_lauren pencils_given_to_matt = 9 := by
  sorry

end steve_pencils_left_l529_529681


namespace alicia_tax_deduction_l529_529456

theorem alicia_tax_deduction (earnings_per_hour_in_cents : ℕ) (tax_rate : ℚ) 
  (h1 : earnings_per_hour_in_cents = 2500) (h2 : tax_rate = 0.02) : 
  earnings_per_hour_in_cents * tax_rate = 50 := 
  sorry

end alicia_tax_deduction_l529_529456


namespace largest_possible_value_l529_529639

variable (a b : ℝ)

theorem largest_possible_value (h1 : 4 * a + 3 * b ≤ 10) (h2 : 3 * a + 6 * b ≤ 12) :
  2 * a + b ≤ 5 :=
sorry

end largest_possible_value_l529_529639


namespace infinite_nested_radical_l529_529369

theorem infinite_nested_radical : ∀ (x : ℝ), (x > 0) → (x = Real.sqrt (12 + x)) → x = 4 :=
by
  intro x
  intro hx_pos
  intro hx_eq
  sorry

end infinite_nested_radical_l529_529369


namespace length_of_adjacent_side_l529_529541

variable (a b : ℝ)

theorem length_of_adjacent_side (area : ℝ) (side : ℝ) :
  area = 6 * a^3 + 9 * a^2 - 3 * a * b →
  side = 3 * a →
  (area / side = 2 * a^2 + 3 * a - b) :=
by
  intro h_area
  intro h_side
  sorry

end length_of_adjacent_side_l529_529541


namespace distinct_pos_xyz_l529_529675

noncomputable theory
open Real

theorem distinct_pos_xyz (x y z a b : ℝ) :
  (x + y + z = a) ∧ (x^2 + y^2 + z^2 = b^2) ∧ (xy = z^2) →
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x ≠ y) ∧ (y ≠ z) ∧ (x ≠ z) →
  (a > 0) ∧ (|b| < a) ∧ (a < sqrt 2 * |b|) :=
by
  intro hconditions hdist_pos
  sorry

end distinct_pos_xyz_l529_529675


namespace sets_B_equals_D_l529_529020

-- Definitions based on conditions
def A := { x : ℝ | ∃ y : ℝ, y = x^2 + 1 }
def B := { y : ℝ | ∃ x : ℝ, y = x^2 + 1 }
def C := { p : ℝ × ℝ | ∃ x : ℝ, p = (x, x^2 + 1) }
def D := { y : ℝ | 1 ≤ y }

-- Prove equivalence of the sets B and D
theorem sets_B_equals_D : B = D := 
sorry

end sets_B_equals_D_l529_529020


namespace correct_operation_l529_529374

theorem correct_operation : ∀ (a b : ℤ), 3 * a^2 * b - 2 * b * a^2 = a^2 * b :=
by
  sorry

end correct_operation_l529_529374


namespace leadership_structure_count_l529_529024

theorem leadership_structure_count : 
  (let n := 12 in
   let main_chief := 1 in
   let supporting_chiefs := 2 in
   let inferior_officers_per_chief := 3 in
   n.choose main_chief * (n - main_chief).choose supporting_chiefs * 
   (n - main_chief - supporting_chiefs).choose (supporting_chiefs * inferior_officers_per_chief)) = 2217600 :=
by
  let n := 12
  let main_chief := 1
  let supporting_chiefs := 2
  let inferior_officers_per_chief := 3
  let result := n.choose main_chief * (n - main_chief).choose supporting_chiefs *
    (n - main_chief - supporting_chiefs).choose (supporting_chiefs * inferior_officers_per_chief)
  have h_result_eq : result = 2217600 := by calc
    result = 12 * 11 * 10 * Nat.choose 9 3 * Nat.choose 6 3 : sorry
         ... = 2217600 : sorry
  exact h_result_eq
  };
end

end leadership_structure_count_l529_529024


namespace num_integers_satisfying_inequality_l529_529576

theorem num_integers_satisfying_inequality :
  let s := { x : ℤ | -6 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9 }
  (s.card = 5) :=
by
sorry

end num_integers_satisfying_inequality_l529_529576


namespace angle_CED_120_degrees_l529_529725

variable (r1 r2 : ℝ) (A B C D E : Type)
-- Assume circle definitions, distances, and intersection conditions 
-- where r1 = 2 * r2

noncomputable def r1_eq_2r2 : Prop := r1 = 2 * r2

noncomputable def A_center_B_radius_r2 : Prop := EuclideanGeometry.Circle A r2

noncomputable def B_center_A_radius_r1 : Prop := EuclideanGeometry.Circle B r1

noncomputable def A_passes_through_B : Prop := EuclideanGeometry.Circle A r2.contains B

noncomputable def B_passes_through_A : Prop := EuclideanGeometry.Circle B r1.contains A

noncomputable def AB_line_intersects_C_D : Prop :=
  EuclideanGeometry.Line A B ∈ EuclideanGeometry.Circle.intersection_points (EuclideanGeometry.Circle A r2) (EuclideanGeometry.Circle B r1) = {C, D}

noncomputable def circles_intersect_E : Prop :=
  E ∈ EuclideanGeometry.Circle.intersection_points (EuclideanGeometry.Circle A r2) (EuclideanGeometry.Circle B r1)

theorem angle_CED_120_degrees
  (h1 : r1_eq_2r2 r1 r2)
  (h2 : A_passes_through_B A B r2)
  (h3 : B_passes_through_A B A r1)
  (h4 : AB_line_intersects_C_D A B C D)
  (h5 : circles_intersect_E A B r2 r1 E) :
  EuclideanGeometry.angle A E D = 120 :=
begin
  sorry,
end

end angle_CED_120_degrees_l529_529725


namespace largest_x_exists_largest_x_largest_real_number_l529_529079

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : x ≤ 48 / 7 :=
sorry

theorem exists_largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  ∃ x, (⌊x⌋ : ℝ) / x = 7 / 8 ∧ x = 48 / 7 :=
sorry

theorem largest_real_number (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  x = 48 / 7 :=
sorry

end largest_x_exists_largest_x_largest_real_number_l529_529079


namespace total_crosswalk_lines_l529_529376

theorem total_crosswalk_lines (n m l : ℕ) (h1 : n = 5) (h2 : m = 4) (h3 : l = 20) :
  n * (m * l) = 400 := by
  sorry

end total_crosswalk_lines_l529_529376


namespace students_taking_german_l529_529214

theorem students_taking_german
  (total_students : ℕ)
  (french_students : ℕ)
  (both_courses_students : ℕ)
  (no_course_students : ℕ)
  (h1 : total_students = 87)
  (h2 : french_students = 41)
  (h3 : both_courses_students = 9)
  (h4 : no_course_students = 33)
  : ∃ german_students : ℕ, german_students = 22 := 
by
  -- proof can be filled in here
  sorry

end students_taking_german_l529_529214


namespace exists_100_nat_nums_with_cube_sum_property_l529_529485

theorem exists_100_nat_nums_with_cube_sum_property :
  ∃ (a : Fin 100 → ℕ), Function.Injective a ∧
    ∃ (k : Fin 100), (a k)^3 = ∑ i in Finset.univ \ {k}, (a i)^3 :=
by
  sorry

end exists_100_nat_nums_with_cube_sum_property_l529_529485


namespace geometric_sequence_product_l529_529140

noncomputable def geometric_sequence_condition (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∑ k in Finset.range n, a (2 * k + 1) = 1 - 2^n

theorem geometric_sequence_product (a : ℕ → ℝ) (q : ℝ) (n : ℕ) (hq : q^2 = 2) 
  (ha1 : a 1 = -1) 
  (h_cond : geometric_sequence_condition a n) : 
  a 2 * a 3 * a 4 = -8 :=
sorry

end geometric_sequence_product_l529_529140


namespace oblique_projection_intuitive_diagrams_correct_l529_529358

-- Definitions based on conditions
structure ObliqueProjection :=
  (lines_parallel_x_axis_same_length : Prop)
  (lines_parallel_y_axis_halved_length : Prop)
  (perpendicular_relationship_becomes_45_angle : Prop)

-- Definitions based on statements
def intuitive_triangle_projection (P : ObliqueProjection) : Prop :=
  P.lines_parallel_x_axis_same_length ∧ 
  P.lines_parallel_y_axis_halved_length ∧ 
  P.perpendicular_relationship_becomes_45_angle

def intuitive_parallelogram_projection (P : ObliqueProjection) : Prop := 
  P.lines_parallel_x_axis_same_length ∧ 
  P.lines_parallel_y_axis_halved_length ∧ 
  P.perpendicular_relationship_becomes_45_angle

def intuitive_square_projection (P : ObliqueProjection) : Prop := 
  P.lines_parallel_x_axis_same_length ∧ 
  P.lines_parallel_y_axis_halved_length ∧ 
  P.perpendicular_relationship_becomes_45_angle

def intuitive_rhombus_projection (P : ObliqueProjection) : Prop := 
  P.lines_parallel_x_axis_same_length ∧ 
  P.lines_parallel_y_axis_halved_length ∧ 
  P.perpendicular_relationship_becomes_45_angle

-- Theorem stating which intuitive diagrams are correctly represented under the oblique projection method.
theorem oblique_projection_intuitive_diagrams_correct : 
  ∀ (P : ObliqueProjection), 
    intuitive_triangle_projection P ∧ 
    intuitive_parallelogram_projection P ∧
    ¬intuitive_square_projection P ∧
    ¬intuitive_rhombus_projection P :=
by 
  sorry

end oblique_projection_intuitive_diagrams_correct_l529_529358


namespace polyhedron_volume_formula_l529_529774

noncomputable def polyhedron_volume (H S1 S2 S3 : ℝ) : ℝ :=
  (1 / 6) * H * (S1 + S2 + 4 * S3)

theorem polyhedron_volume_formula 
  (H S1 S2 S3 : ℝ)
  (bases_parallel_planes : Prop)
  (lateral_faces_trapezoids_parallelograms_or_triangles : Prop)
  (H_distance : Prop) 
  (S1_area_base : Prop) 
  (S2_area_base : Prop) 
  (S3_area_cross_section : Prop) : 
  polyhedron_volume H S1 S2 S3 = (1 / 6) * H * (S1 + S2 + 4 * S3) :=
sorry

end polyhedron_volume_formula_l529_529774


namespace find_common_difference_l529_529715

variable {α : Type*} [LinearOrderedField α]

-- Define the properties of the arithmetic sequence
def arithmetic_sum (a1 d : α) (n : ℕ) : α := n * a1 + (n * (n - 1) * d) / 2

variables (a1 d : α) -- First term and common difference of the arithmetic sequence (to be found)
variable (S : ℕ → α) -- Sum of the first n terms of the arithmetic sequence

-- Conditions given in the problem
axiom sum_3_eq_6 : S 3 = 6
axiom term_3_eq_4 : a1 + 2 * d = 4

-- The question translated into a theorem statement that the common difference is 2
theorem find_common_difference : d = 2 :=
by
  sorry

end find_common_difference_l529_529715


namespace problem_l529_529322

variable {f : ℝ → ℝ}

-- Assume f is an increasing function on (-∞, +∞)
def increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f(x) ≤ f(y)

-- Assume for any x1, x2 ∈ ℝ, the inequality f(x1) + f(x2) ≥ f(-x1) + f(-x2) holds.
def condition (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, f(x1) + f(x2) ≥ f(-x1) + f(-x2)

-- Define the proof problem
theorem problem (f : ℝ → ℝ) [increasing f] [condition f] : 
  ∀ x1 x2 : ℝ, x1 + x2 ≥ 0 :=
by
  sorry

end problem_l529_529322


namespace plane_speed_with_tailwind_l529_529419

theorem plane_speed_with_tailwind (V : ℝ) (tailwind_speed : ℝ) (ground_speed_against_tailwind : ℝ) 
  (H1 : tailwind_speed = 75) (H2 : ground_speed_against_tailwind = 310) (H3 : V - tailwind_speed = ground_speed_against_tailwind) :
  V + tailwind_speed = 460 :=
by
  sorry

end plane_speed_with_tailwind_l529_529419


namespace binom_20_18_eq_190_l529_529804

theorem binom_20_18_eq_190 : binomial 20 18 = 190 := 
by sorry

end binom_20_18_eq_190_l529_529804


namespace melanie_plums_left_l529_529282

/-
  Melanie picked 12 plums and 6 oranges from the orchard. If she ate 2 plums and gave away 1/3 of the remaining plums to Sam,
  prove that she has 7 plums left.
-/

def initial_plums : ℕ := 12
def eaten_plums : ℕ := 2
def fraction_given_away : ℚ := 1 / 3

theorem melanie_plums_left : 
  let remaining_plums_after_eating := initial_plums - eaten_plums,
      given_away_plums := (remaining_plums_after_eating : ℚ) * fraction_given_away,
      final_plums := remaining_plums_after_eating - given_away_plums.toNat
  in final_plums = 7 := 
by
  -- Proof goes here
  sorry

end melanie_plums_left_l529_529282


namespace largest_x_l529_529115

def largest_x_with_condition_eq_7_over_8 (x : ℝ) : Prop :=
  ⌊x⌋ / x = 7 / 8

theorem largest_x (x : ℝ) (h : largest_x_with_condition_eq_7_over_8 x) :
  x = 48 / 7 :=
sorry

end largest_x_l529_529115


namespace functional_eq_solutions_l529_529834

theorem functional_eq_solutions (f : ℝ → ℝ) :
  (∀ x y z : ℝ, f (x * y + z) = f x * f y + f z) → (f = (λ x, 0) ∨ f = id) :=
by
  sorry

end functional_eq_solutions_l529_529834


namespace team_tournament_probability_l529_529350

theorem team_tournament_probability : 
  let total_teams := 30
  let total_games := (total_teams * (total_teams - 1)) / 2
  let wins_distribution_factorial := Nat.factorial total_teams
  let possible_outcomes := 2 ^ total_games
  ∃ (p q : ℕ), Nat.coprime p q ∧ q = 2 ^ (total_games - (total_teams / 2 + total_teams / 4 + total_teams / 8 + total_teams / 16)) ∧ log2 q = 409 := 
by
  let total_teams := 30
  let total_games := (total_teams * (total_teams - 1)) / 2
  let wins_distribution_factorial := Nat.factorial total_teams
  let possible_outcomes := 2 ^ total_games
  have h_term : Nat.coprime (Nat.factorial total_teams) (2 ^ (total_games - (total_teams / 2 + total_teams / 4 + total_teams / 8 + total_teams / 16))) := sorry
  exists wins_distribution_factorial, (2 ^ (total_games - (total_teams / 2 + total_teams / 4 + total_teams / 8 + total_teams / 16))), by
    split
    · exact h_term
    · sorry


end team_tournament_probability_l529_529350


namespace joe_used_paint_total_l529_529247

theorem joe_used_paint_total :
  let first_airport_paint := 360
  let second_airport_paint := 600
  let first_week_first_airport := (1/4 : ℝ) * first_airport_paint
  let remaining_first_airport := first_airport_paint - first_week_first_airport
  let second_week_first_airport := (1/6 : ℝ) * remaining_first_airport
  let total_first_airport := first_week_first_airport + second_week_first_airport
  let first_week_second_airport := (1/3 : ℝ) * second_airport_paint
  let remaining_second_airport := second_airport_paint - first_week_second_airport
  let second_week_second_airport := (1/5 : ℝ) * remaining_second_airport
  let total_second_airport := first_week_second_airport + second_week_second_airport
  total_first_airport + total_second_airport = 415 :=
by
  let first_airport_paint := 360
  let second_airport_paint := 600
  let first_week_first_airport := (1/4 : ℝ) * first_airport_paint
  let remaining_first_airport := first_airport_paint - first_week_first_airport
  let second_week_first_airport := (1/6 : ℝ) * remaining_first_airport
  let total_first_airport := first_week_first_airport + second_week_first_airport
  let first_week_second_airport := (1/3 : ℝ) * second_airport_paint
  let remaining_second_airport := second_airport_paint - first_week_second_airport
  let second_week_second_airport := (1/5 : ℝ) * remaining_second_airport
  let total_second_airport := first_week_second_airport + second_week_second_airport
  show total_first_airport + total_second_airport = 415
  sorry

end joe_used_paint_total_l529_529247


namespace pentagon_area_correct_l529_529694

noncomputable def pentagon_FGHIJ_area (F G H I J : ℝ) (cos : ℝ → ℝ) : ℝ :=
  -- Conditions:
  -- Pentagon FGHIJ is convex with angles and side lengths given
  let angle_F : ℝ := 120
  let angle_G : ℝ := 120
  let angle_H : ℝ := 100
  let length_IJ : ℝ := 2
  let length_JF : ℝ := 2
  let length_FG : ℝ := 2
  let length_GH : ℝ := 3
  let length_HI : ℝ := 3 in
  
  -- Correct Answer:
  -- Area of pentagon FGHIJ
  9.526

theorem pentagon_area_correct (F G H I J : ℝ) (cos : ℝ → ℝ) :
    pentagon_FGHIJ_area F G H I J cos = 9.526 := 
  sorry

end pentagon_area_correct_l529_529694


namespace find_gain_percent_l529_529745

-- Definitions based on the conditions
def CP : ℕ := 900
def SP : ℕ := 1170

-- Calculation of gain
def Gain := SP - CP

-- Calculation of gain percent
def GainPercent := (Gain * 100) / CP

-- The theorem to prove the gain percent is 30%
theorem find_gain_percent : GainPercent = 30 := 
by
  sorry -- Proof to be filled in.

end find_gain_percent_l529_529745


namespace marked_price_of_each_article_l529_529417

open Real

theorem marked_price_of_each_article
  (pair_cost : ℝ)
  (discount : ℝ)
  (discounted_cost : ℝ)
  (marked_price : ℝ)
  (marked_price_each : ℝ) :
  pair_cost = 50 →
  discount = 0.60 →
  discounted_cost = pair_cost →
  0.40 * marked_price = discounted_cost →
  marked_price_each = marked_price / 2 →
  marked_price_each = 62.50 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  sorry

end marked_price_of_each_article_l529_529417


namespace simplify_fraction_l529_529293

theorem simplify_fraction (e : ℤ) : 
  (∃ k : ℤ, e = 13 * k + 12) ↔ (∃ d : ℤ, d ≠ 1 ∧ (16 * e - 10) % d = 0 ∧ (10 * e - 3) % d = 0) :=
by
  sorry

end simplify_fraction_l529_529293


namespace general_term_formula_l529_529907

noncomputable theory

open Nat

-- Define the sequence {a_n} where the sum of the first n terms S_n = 3n^2 + 8n
def S : ℕ → ℕ
| 0       := 0      -- We define S(0) for convenience; S(1) should correspond to S₁
| (n + 1) := 3 * (n + 1) * (n + 1) + 8 * (n + 1)

def a : ℕ → ℕ
| 0       := S 1    -- a₁ defined as S₁
| (n + 1) := S (n + 2) - S (n + 1) -- aₙ = Sₙ - Sₙ₋₁ for n ≥ 1

theorem general_term_formula (n : ℕ) : 
  a n.succ = 6 * n.succ + 5 :=
sorry

end general_term_formula_l529_529907


namespace range_of_g_l529_529122

-- Define the function g
def g (x : ℝ) := (Real.arcsin (x / 3))^2 - π * Real.arccos (x / 3) +
                  (Real.arccos (x / 3))^2 + (π^2 / 8) * (x^2 - 4 * x + 3)

-- State the theorem about the range of g over the interval [-3, 3]
theorem range_of_g : 
  (∀ x, -3 ≤ x ∧ x ≤ 3 → 
  (g x) ≥ (π^2 / 4) ∧ (g x) ≤ (33 * π^2 / 8)) := 
sorry

end range_of_g_l529_529122


namespace max_principals_in_10_years_l529_529828

theorem max_principals_in_10_years (p : ℕ) (is_principal_term : p = 4) : 
  ∃ n : ℕ, n = 4 ∧ ∀ k : ℕ, (k = 10 → n ≤ 4) :=
by
  sorry

end max_principals_in_10_years_l529_529828


namespace extreme_points_f_l529_529309

theorem extreme_points_f (a b : ℝ)
  (h1 : 3 * (-2)^2 + 2 * a * (-2) + b = 0)
  (h2 : 3 * 4^2 + 2 * a * 4 + b = 0) :
  a - b = 21 :=
sorry

end extreme_points_f_l529_529309


namespace common_root_l529_529854

variable (m x : ℝ)
variable (h₁ : m * x - 1000 = 1021)
variable (h₂ : 1021 * x = m - 1000 * x)

theorem common_root (hx : m * x - 1000 = 1021 ∧ 1021 * x = m - 1000 * x) : m = 2021 ∨ m = -2021 := sorry

end common_root_l529_529854


namespace smallest_fraction_l529_529373

theorem smallest_fraction {a b c d e : ℚ}
  (ha : a = 7/15)
  (hb : b = 5/11)
  (hc : c = 16/33)
  (hd : d = 49/101)
  (he : e = 89/183) :
  (b < a) ∧ (b < c) ∧ (b < d) ∧ (b < e) := 
sorry

end smallest_fraction_l529_529373


namespace hyperbola_eccentricity_l529_529889

theorem hyperbola_eccentricity (a b c : ℚ) (h1 : (c : ℚ) = 5)
  (h2 : (b / a) = 3 / 4) (h3 : c^2 = a^2 + b^2) :
  (c / a : ℚ) = 5 / 4 :=
by
  sorry

end hyperbola_eccentricity_l529_529889


namespace necessary_and_sufficient_l529_529517

variable (α β : ℝ)
variable (p : Prop := α > β)
variable (q : Prop := α + Real.sin α * Real.cos β > β + Real.sin β * Real.cos α)

theorem necessary_and_sufficient : (p ↔ q) :=
by
  sorry

end necessary_and_sufficient_l529_529517


namespace regular_triangular_pyramid_surface_area_l529_529542

theorem regular_triangular_pyramid_surface_area (base_edge_length height : ℝ)
  (h_base_edge : base_edge_length = 6)
  (h_height : height = 3) :
  let surface_area := 27 * Real.sqrt 3 in
  surface_area = 27 * Real.sqrt 3 :=
by
  sorry

end regular_triangular_pyramid_surface_area_l529_529542


namespace trigonometric_identity_l529_529669

variable (α β : ℝ)

theorem trigonometric_identity :
  1 - cos (β - α) + cos α - cos β = 4 * cos (α / 2) * sin (β / 2) * sin ((β - α) / 2) :=
by
  sorry

end trigonometric_identity_l529_529669


namespace sum_digits_3031_3032_3033_l529_529246

-- Define the initial sequence as a repeating pattern of [1, 2, 3, 4, 5, 6].
def initialSequence : List ℕ :=
  List.replicate (12000 / 6) [1, 2, 3, 4, 5, 6].flatten

-- Function to perform erasure of every nth element in a given list.
def eraseEveryNth (seq : List ℕ) (n : ℕ) : List ℕ :=
  seq.enum.filter (λ p, (p.fst + 1) % n ≠ 0).map Prod.snd

-- Erase every 4th digit.
def afterFirstErasure : List ℕ := eraseEveryNth initialSequence 4

-- Erase every 5th digit in the remaining list.
def afterSecondErasure : List ℕ := eraseEveryNth afterFirstErasure 5

-- Erase every 6th digit in the remaining list.
def finalList : List ℕ := eraseEveryNth afterSecondErasure 6

-- Get the digits at specific positions.
def digitAt (seq : List ℕ) (pos : ℕ) : ℕ := seq.getD (pos - 1) 0

-- Sum of digits at positions 3031, 3032, and 3033.
def sumDigitsAtPositions : ℕ :=
  digitAt finalList 3031 + digitAt finalList 3032 + digitAt finalList 3033

theorem sum_digits_3031_3032_3033 :
  sumDigitsAtPositions = 6 :=
sorry

end sum_digits_3031_3032_3033_l529_529246


namespace defendant_innocence_l529_529418

-- Definitions representing the problem
inductive Person
| knight    -- A person who always tells the truth
| liar      -- A person who always lies

open Person

-- Function to determine if a person tells the truth
def tells_truth (p : Person) : Prop :=
match p with
| knight => True
| liar   => False
end

-- Statement made by the defendant
def statement (criminal : Person) : Prop :=
criminal = liar

-- Proving the defendant's statement helps their situation
theorem defendant_innocence (d : Person) (committed : Person) (h : tells_truth d → statement committed) (nh : ¬tells_truth d → ¬statement committed) : 
d ≠ committed :=
by
  cases d <;> cases committed <;> simp [tells_truth, statement, h, nh]
  -- knight - knight case
  -- knight - liar case
  -- liar - knight case
  -- liar - liar case
  sorry

end defendant_innocence_l529_529418


namespace an_plus_cn_eq_bn_l529_529027

-- Given conditions
variables {A B C I D M Q N : Type}

def triangle (A B C : Type) : Prop := True -- representation of a triangle
def incenter (I : Type) (A B C : Type) : Prop := True -- representation of the incenter
def midpoint (M : Type) (B C : Type) : Prop := True -- representation of the midpoint
def extension_eq {I M Q : Type} : Prop := True -- representation that IM = MQ
def extension_ai_intersects_circum (A I : Type) (c : Type)  (D : Type) : Prop := True -- extension of AI intersects the circumcircle at D
def dq_intersects_circum (D Q : Type) (c : Type) (N : Type) : Prop := True -- DQ intersects the circumcircle at N

-- Prove the statement
theorem an_plus_cn_eq_bn :
  ∀ (A B C I D M Q N : Type),
  triangle A B C →
  incenter I A B C →
  midpoint M B C →
  extension_eq →
  extension_ai_intersects_circum A I (circumcircle A B C) D →
  dq_intersects_circum D Q (circumcircle A B C) N →
  AN + CN = BN := 
begin
  sorry, -- the proof goes here
end

end an_plus_cn_eq_bn_l529_529027


namespace amount_saved_by_Dalton_l529_529821

-- Defining the costs of each item and the given conditions
def jump_rope_cost : ℕ := 7
def board_game_cost : ℕ := 12
def playground_ball_cost : ℕ := 4
def uncle_gift : ℕ := 13
def additional_needed : ℕ := 4

-- Calculated values based on the conditions
def total_cost : ℕ := jump_rope_cost + board_game_cost + playground_ball_cost
def total_money_needed : ℕ := uncle_gift + additional_needed

-- The theorem that needs to be proved
theorem amount_saved_by_Dalton : total_cost - total_money_needed = 6 :=
by
  sorry -- Proof to be filled in

end amount_saved_by_Dalton_l529_529821


namespace at_least_one_triangle_possible_without_triangle_l529_529816

variable (n : ℕ) (points : finset (ℕ × ℕ)) (num_segments : ℕ)

-- Conditions
def valid_points_and_segments (points : finset (ℕ × ℕ)) (n : ℕ) :=
  points.card = 2 * n ∧ n > 1

def has_enough_segments (num_segments : ℕ) (n : ℕ) :=
  num_segments >= n^2 + 1

def has_exact_segments (num_segments : ℕ) (n : ℕ) :=
  num_segments = n^2

-- Theorem statements
theorem at_least_one_triangle {points : finset (ℕ × ℕ)} {num_segments : ℕ} :
  valid_points_and_segments points n →
  has_enough_segments num_segments n →
  ∃ triangle : finset (ℕ × ℕ), triangle ⊆ points ∧ triangle.card = 3 :=
sorry

theorem possible_without_triangle {points : finset (ℕ × ℕ)} {num_segments : ℕ} :
  valid_points_and_segments points n →
  has_exact_segments num_segments n →
  ∃ configuration, (∀ triangle : finset (ℕ × ℕ), triangle ⊆ points → triangle.card = 3 → false) :=
sorry

end at_least_one_triangle_possible_without_triangle_l529_529816


namespace no_9_segments_with_3_intersections_l529_529238

theorem no_9_segments_with_3_intersections :
  ¬ ∃ (G : SimpleGraph (Fin 9)), (∀ v : Fin 9, degree G v = 3) :=
by
  sorry

end no_9_segments_with_3_intersections_l529_529238


namespace collinear_points_b_value_l529_529061

theorem collinear_points_b_value (b : ℚ) :
  (∃ b : ℚ, (4, -3) ∈ ℚ.prod ℚ ∧ (2 * b + 1, 5) ∈ ℚ.prod ℚ ∧ (-b + 3, 1) ∈ ℚ.prod ℚ ∧
  ((5 - (-3)) / (2 * b + 1 - 4)) = ((1 - (-3)) / (-b + 3 - 4)) ) → b = 1/4 := 
by
  sorry

end collinear_points_b_value_l529_529061


namespace total_distance_traveled_is_7_75_l529_529210

open Real

def walking_time_minutes : ℝ := 30
def walking_rate : ℝ := 3.5

def running_time_minutes : ℝ := 45
def running_rate : ℝ := 8

theorem total_distance_traveled_is_7_75 :
  let walking_hours := walking_time_minutes / 60
  let distance_walked := walking_rate * walking_hours
  let running_hours := running_time_minutes / 60
  let distance_run := running_rate * running_hours
  let total_distance := distance_walked + distance_run
  total_distance = 7.75 :=
by
  sorry

end total_distance_traveled_is_7_75_l529_529210


namespace mean_of_other_two_numbers_l529_529995

theorem mean_of_other_two_numbers (a b c d e f g h : ℕ)
  (h_tuple : a = 1871 ∧ b = 2011 ∧ c = 2059 ∧ d = 2084 ∧ e = 2113 ∧ f = 2167 ∧ g = 2198 ∧ h = 2210)
  (h_mean : (a + b + c + d + e + f) / 6 = 2100) :
  ((g + h) / 2 : ℚ) = 2056.5 :=
by
  sorry

end mean_of_other_two_numbers_l529_529995


namespace ordered_triples_lcm_count_l529_529196

theorem ordered_triples_lcm_count :
  let valid_triples := λ a b c : ℕ, (nat.lcm a b = 120) ∧ (nat.lcm a c = 450) ∧ (nat.lcm b c = 720)
  in (setOf (λ (abc : ℕ × ℕ × ℕ), valid_triples abc.1 abc.2.1 abc.2.2)).finite.card = 8 :=
by
  sorry

end ordered_triples_lcm_count_l529_529196


namespace largest_x_eq_48_div_7_l529_529109

theorem largest_x_eq_48_div_7 :
  ∃ x : ℝ, (⟨floor x / x⟩ = 7 / 8) ∧ (x = 48 / 7) := 
begin
  sorry
end

end largest_x_eq_48_div_7_l529_529109


namespace triangle_area_ratio_l529_529957

theorem triangle_area_ratio (ABC : Triangle) (A B C H D : Point) 
  (h1 : H ∈ Segment A C) (h2 : D ∈ Segment B C) 
  (h3 : ∠ B H ⊥ Line A C) (h4 : ∠ H D ⊥ Line B C) 
  (O1 O2 O3 : Point)
  (hO1 : O1 = circumcenter (Triangle.mk A B H))
  (hO2 : O2 = circumcenter (Triangle.mk B H D))
  (hO3 : O3 = circumcenter (Triangle.mk H D C)) :
  area (Triangle.mk O1 O2 O3) / area (Triangle.mk A B H) = 1 / 4 :=
  sorry

end triangle_area_ratio_l529_529957


namespace area_triangle_ABC_l529_529595

theorem area_triangle_ABC (A B C D E F : Type) [has_area A B C D E F]
  (hD : midpoint D B C)
  (hE : ratio E A C 2 3)
  (hF : ratio F A D 2 1)
  (area_DEF : area (Δ D E F) = 18) :
  area (Δ A B C) = 270 := 
sorry

end area_triangle_ABC_l529_529595


namespace infinite_series_sum_l529_529813

theorem infinite_series_sum :
  (∑' j : ℕ, ∑' k : ℕ, 2^(-(2 * k + j + (k + j) ^ 2) : ℤ)) = 4 / 3 :=
by
  sorry

end infinite_series_sum_l529_529813


namespace T_8_equals_546_l529_529565

-- Define the sum of the first n natural numbers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the sum of the squares of the first n natural numbers
def sum_squares_first_n (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

-- Define T_n based on the given formula
def T (n : ℕ) : ℕ := (sum_first_n n ^ 2 - sum_squares_first_n n) / 2

-- The proof statement we need to prove
theorem T_8_equals_546 : T 8 = 546 := sorry

end T_8_equals_546_l529_529565


namespace max_consecutive_odd_terms_l529_529220

noncomputable def largest_digit (n : ℕ) : ℕ :=
  n.digits 10 |>.map (fun d => d.toNat) |>.maximum'.getD 0

def seq (a : ℕ) : ℕ → ℕ
| 0       => a
| (n + 1) => let prev := seq n in prev + largest_digit prev

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem max_consecutive_odd_terms (a1 : ℕ) :
  (∀ i, is_odd (seq a1 i)) → i < 5 → i < 5 :=
sorry

end max_consecutive_odd_terms_l529_529220


namespace sum_of_given_geom_series_l529_529049

-- Define the necessary conditions
def first_term (a : ℕ) := a = 2
def common_ratio (r : ℕ) := r = 3
def number_of_terms (n : ℕ) := n = 6

-- Define the sum of the geometric series
def sum_geom_series (a r n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

-- State the theorem
theorem sum_of_given_geom_series :
  first_term 2 → common_ratio 3 → number_of_terms 6 → sum_geom_series 2 3 6 = 728 :=
by
  intros h1 h2 h3
  rw [first_term] at h1
  rw [common_ratio] at h2
  rw [number_of_terms] at h3
  have h1 : 2 = 2 := by exact h1
  have h2 : 3 = 3 := by exact h2
  have h3 : 6 = 6 := by exact h3
  exact sorry

end sum_of_given_geom_series_l529_529049


namespace fairy_tale_book_weight_l529_529768

-- Define the total weight on one side of the scale
def total_other_side_weight : ℝ := 0.5 + 0.3 + 0.3

-- Given that the scale is level
def scale_is_level (F : ℝ) : Prop := F = total_other_side_weight

-- The theorem statement to prove
theorem fairy_tale_book_weight (F : ℝ) (h : scale_is_level F) : F = 1.1 :=
by
  rw [scale_is_level] at h
  exact h

sorry

end fairy_tale_book_weight_l529_529768


namespace prime_solution_l529_529511

-- Function to calculate the sum of the digits of a number
def sumOfDigits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum -- Lean's way to get digits of a number and sum them up

-- Main theorem to prove the given problem
theorem prime_solution (n : ℕ) (h1 : n.prime) (h2 : n = 2999)
  (h3 : n + sumOfDigits n + sumOfDigits (sumOfDigits n) = 3005) : True := 
by
  sorry

end prime_solution_l529_529511


namespace solve_for_a_l529_529521

theorem solve_for_a (a : ℝ) (h : |2022 - a| + real.sqrt (a - 2023) = a) : a - 2022^2 = 2023 :=
sorry

end solve_for_a_l529_529521


namespace find_height_of_light_source_l529_529012

theorem find_height_of_light_source (x : ℝ) :
  let edge := 3
  let base_area := edge ^ 2
  let shadow_area := 75
  let total_shadow_area := base_area + shadow_area
  let shadow_side_length := real.sqrt total_shadow_area
  let increase_in_height := shadow_side_length - edge
  let similar_triangles_ratio := increase_in_height / edge
  (total_shadow_area = 84) →
  (shadow_side_length = 2 * real.sqrt 21) →
  (increase_in_height = 2 * real.sqrt 21 - edge) →
  (x = edge * increase_in_height / (increase_in_height - edge)) →
  x = 7 := 
by
  sorry

end find_height_of_light_source_l529_529012


namespace sum_equals_four_thirds_l529_529812

-- Define the given sum as a function
def problem_sum : ℝ :=
  ∑' j, ∑' k, 2^(-(2 * k + j + (k + j)^2))

-- The theorem statement to prove the correctness of the sum
theorem sum_equals_four_thirds : problem_sum = 4 / 3 :=
by
  sorry -- proof to demonstrate the result

end sum_equals_four_thirds_l529_529812


namespace part_I_part_II_l529_529533

variables {x a : ℝ} (p : Prop) (q : Prop)

-- Proposition p
def prop_p (x a : ℝ) : Prop := x^2 - 5*a*x + 4*a^2 < 0 ∧ a > 0

-- Proposition q
def prop_q (x : ℝ) : Prop := (x^2 - 2*x - 8 ≤ 0) ∧ (x^2 + 3*x - 10 > 0)

-- Part (I)
theorem part_I (a : ℝ) (h : a = 1) : (prop_p x a) → (prop_q x) → (2 < x ∧ x < 4) :=
by
  sorry

-- Part (II)
theorem part_II (a : ℝ) : ¬(∃ x, prop_p x a) → ¬(∃ x, prop_q x) → (1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end part_I_part_II_l529_529533


namespace events_related_with_99_confidence_l529_529368

theorem events_related_with_99_confidence (K_squared : ℝ) (h : K_squared > 6.635) : 
  events_A_B_related_with_99_confidence :=
sorry

end events_related_with_99_confidence_l529_529368


namespace factorization_of_difference_of_squares_l529_529832

theorem factorization_of_difference_of_squares (m n : ℝ) : m^2 - n^2 = (m + n) * (m - n) := 
by sorry

end factorization_of_difference_of_squares_l529_529832


namespace percentage_difference_l529_529018

-- Define the quantities involved
def milk_in_A : ℕ := 1264
def transferred_milk : ℕ := 158

-- Define the quantities of milk in container B and C after transfer
noncomputable def quantity_in_B : ℕ := milk_in_A / 2
noncomputable def quantity_in_C : ℕ := quantity_in_B

-- Prove that the percentage difference between the quantity of milk in container B
-- and the capacity of container A is 50%
theorem percentage_difference :
  ((milk_in_A - quantity_in_B) * 100 / milk_in_A) = 50 := sorry

end percentage_difference_l529_529018


namespace positive_difference_l529_529691

theorem positive_difference (y : ℤ) (h : (46 + y) / 2 = 52) : |y - 46| = 12 := by
  sorry

end positive_difference_l529_529691


namespace probability_slope_gte_one_eq_nine_l529_529629

theorem probability_slope_gte_one_eq_nine :
  let Q := (x, y) in 
  \{Q : set point \ | (0 < Q.x ∧ Q.x < 1) ∧ (0 < Q.y ∧ Q.y < 1)\} -> 
  let P : point := (3 / 4, 1 / 4),
  prob(Qslope_gt_one) == (1 / 8)
  → p = 1 ∧ q = 8 → p + q = 9 :=
by sorry

end probability_slope_gte_one_eq_nine_l529_529629


namespace boys_sitting_10_boys_sitting_11_l529_529761

def exists_two_boys_with_4_between (n : ℕ) : Prop :=
  ∃ (b : Finset ℕ), b.card = n ∧ ∀ (i j : ℕ) (h₁ : i ≠ j) (h₂ : i < 25) (h₃ : j < 25),
    (i + 5) % 25 = j

theorem boys_sitting_10 :
  ¬exists_two_boys_with_4_between 10 :=
sorry

theorem boys_sitting_11 :
  exists_two_boys_with_4_between 11 :=
sorry

end boys_sitting_10_boys_sitting_11_l529_529761


namespace digit_7_appears_602_times_l529_529223

theorem digit_7_appears_602_times :
  ∑ n in Finset.range 2018, has_digit 7 n = 602 := 
sorry

end digit_7_appears_602_times_l529_529223


namespace problem_solution_l529_529894

-- Define the ellipse equation and foci positions.
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + (y^2 / 2) = 1
def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

-- Define the line equation
def line (x y k : ℝ) : Prop := y = k * x + 1

-- Define the intersection points A and B
variable (A B : ℝ × ℝ)
variable (k : ℝ)

-- Define the points lie on the line and ellipse
def A_on_line := ∃ x y, A = (x, y) ∧ line x y k
def B_on_line := ∃ x y, B = (x, y) ∧ line x y k

-- Define the parallel and perpendicular conditions
def parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k, v1.1 = k * v2.1 ∧ v1.2 = k * v2.2
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Lean theorem for the conclusions of the problem
theorem problem_solution (A_cond : A_on_line A k ∧ ellipse A.1 A.2) 
                          (B_cond : B_on_line B k ∧ ellipse B.1 B.2) :

  -- Prove these two statements
  ¬ parallel (A.1 + 1, A.2) (B.1 - 1, B.2) ∧
  ¬ perpendicular (A.1 + 1, A.2) (A.1 - 1, A.2) :=
sorry

end problem_solution_l529_529894


namespace horizontal_asymptote_of_rational_function_l529_529324

open_locale topological_space

theorem horizontal_asymptote_of_rational_function :
  (∀ x: ℝ, x ≠ 0 → (6 * x^2 - 11) / (4 * x^2 + 6 * x + 3) → 3 / 2) :=
by 
  intro x hx,
  have h: ∀ x, x ≠ 0 → (6 * x^2 - 11) / (4 * x^2 + 6 * x + 3) = 6 / (4 + 6 / x + 3 / x^2) - 11 / (4 * x^2 + 6 * x + 3) := sorry,
  sorry

end horizontal_asymptote_of_rational_function_l529_529324


namespace find_a_range_l529_529878

open Real

-- Define the points A, B, and C
def A : (ℝ × ℝ) := (4, 1)
def B : (ℝ × ℝ) := (-1, -6)
def C : (ℝ × ℝ) := (-3, 2)

-- Define the system of inequalities representing the region D
def region_D (x y : ℝ) : Prop :=
  7 * x - 5 * y - 23 ≤ 0 ∧
  x + 7 * y - 11 ≤ 0 ∧
  4 * x + y + 10 ≥ 0

-- Define the inequality condition for points B and C on opposite sides of the line 4x - 3y - a = 0
def opposite_sides (a : ℝ) : Prop :=
  (14 - a) * (-18 - a) < 0

-- Lean statement to prove the given problem
theorem find_a_range : 
  ∃ a : ℝ, region_D 0 0 ∧ opposite_sides a → -18 < a ∧ a < 14 :=
by 
  sorry

end find_a_range_l529_529878


namespace percent_increase_after_reduction_l529_529797

theorem percent_increase_after_reduction:
  ∀ (P : ℝ), P > 0 → ((1.2 * P) / (0.85 * P) - 1) * 100 ≈ 41.18 := 
by
  intros P hP
  have h := (1.2 / 0.85 - 1) * 100
  have h_correct : h ≈ 41.18 := sorry
  exact h_correct

end percent_increase_after_reduction_l529_529797


namespace solve_inequality_solution_set_l529_529712

def solution_set (x : ℝ) : Prop := -x^2 + 5 * x > 6

theorem solve_inequality_solution_set :
  { x : ℝ | solution_set x } = { x : ℝ | 2 < x ∧ x < 3 } :=
sorry

end solve_inequality_solution_set_l529_529712


namespace maximum_value_of_f_l529_529556

noncomputable def f (x : ℝ) : ℝ :=
  (2 - real.sqrt 2 * real.sin (real.pi / 4 * x)) / (x^2 + 4 * x + 5)

theorem maximum_value_of_f :
  ∃ x ∈ set.Icc (-4 : ℝ) (0 : ℝ), ∀ y ∈ set.Icc (-4 : ℝ) (0 : ℝ), f y ≤ f x ∧ f x = 2 + real.sqrt 2 :=
begin
  sorry
end

end maximum_value_of_f_l529_529556


namespace women_at_conference_l529_529218

theorem women_at_conference (W : ℕ) :
  let men := 700 in
  let children := 800 in
  let indian_men := 0.20 * men in
  let indian_women := 0.40 * W in 
  let indian_children := 0.10 * children in
  let total_people := men + W + children in
  let total_indians := indian_men + indian_women + indian_children in
  0.21 * total_people = total_indians → W = 500 :=
by
  sorry

end women_at_conference_l529_529218


namespace tangent_line_perpendicular_to_given_line_l529_529498

theorem tangent_line_perpendicular_to_given_line :
  let f (x : ℝ) := x^3 + 3 * x^2 - 1 in
  let df (x : ℝ) := 3 * x^2 + 6 * x in
  ∃ (tangent_line : ℝ → ℝ) (x0 : ℝ),
  (df x0 = -3) ∧
  (tangent_line = λ x, -3 * (x - x0) + f x0) ∧
  (tangent_line = λ x, -(3 * x + 2)) :=
by
  let f : ℝ → ℝ := λ x, x^3 + 3 * x^2 - 1
  let df : ℝ → ℝ := λ x, 3 * x^2 + 6 * x
  existsi λ x : ℝ, -3 * (x + 1) + 1
  existsi -1
  sorry

end tangent_line_perpendicular_to_given_line_l529_529498


namespace combined_area_eq_l529_529936

/-- In triangle ABC, E is the midpoint of side BC, and D is on side AC.
If side AC has length 2 units and the angles are ∠BAC = 45°, ∠ABC = 85°, and 
∠ACB = 50°, and angle ∠DEC = 70°, then the combined area of 
ΔABC plus twice the area of ΔCDE is equal to 2sin(85°) + 2sin(70°). -/
theorem combined_area_eq:
  let A B C D E : Type*
  let AC : ℝ := 2
  let ∠BAC : ℝ := 45
  let ∠ABC : ℝ := 85
  let ∠ACB : ℝ := 50
  let ∠DEC : ℝ := 70
  let area (ΔABC) : ℝ := 2 * Real.sin 85
  let area (ΔCDE) : ℝ := Real.sin 70
  in (area (ΔABC) + 2 * area (ΔCDE)) = 
     (2 * Real.sin 85) + (2 * Real.sin 70) 
:= sorry

end combined_area_eq_l529_529936


namespace probability_window_opens_correct_l529_529451

noncomputable def probability_window_opens_no_later_than_3_minutes_after_scientist_arrives 
  (arrival_times : Fin 6 → ℝ) : ℝ :=
  if (∀ i, arrival_times i ∈ Set.Icc 0 15) ∧ 
     (∀ i j, i ≠ j → arrival_times i < arrival_times j) ∧ 
     ((∃ i, arrival_times i ≥ 12)) then
    1 - (0.8 ^ 6)
  else
    0

theorem probability_window_opens_correct : 
  ∀ (arrival_times : Fin 6 → ℝ),
    (∀ i, arrival_times i ∈ Set.Icc 0 15) →
    (∀ i j, i ≠ j → arrival_times i < arrival_times j) →
    (∃ i, arrival_times i = arrival_times 5) →
    abs (probability_window_opens_no_later_than_3_minutes_after_scientist_arrives arrival_times - 0.738) < 0.001 :=
by
  sorry

end probability_window_opens_correct_l529_529451


namespace binom_9_6_l529_529471

theorem binom_9_6 : nat.choose 9 6 = 84 := by
  sorry

end binom_9_6_l529_529471


namespace angle_z_value_l529_529605

theorem angle_z_value
    (angle_ABC : ℕ)
    (angle_ACB : ℕ)
    (angle_CDE : ℕ)
    (straight_angle : ∀ (θ : ℕ), θ = 180) :
    ∃ z : ℕ, z = 166 := 
by
  assume angle_ABC = 50
  assume angle_ACB = 90
  assume angle_CDE = 54
  let angle_BAC := 180 - angle_ABC - angle_ACB
  let angle_ADE := 180 - angle_CDE
  let angle_EAD := angle_BAC
  let angle_AED := 180 - angle_ADE - angle_EAD
  have angle_DEB := 180 - angle_AED
  exists 166
  sorry

end angle_z_value_l529_529605


namespace arithmetic_sequence_sum_general_term_formula_sequence_T_n_sum_l529_529528

noncomputable def general_term_formula (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n, a n = 2 * n + 1

noncomputable def geometric_sequence (a₁ a₄ a₁₃ : ℕ) :=
  a₄^2 = a₁ * a₁₃

theorem arithmetic_sequence_sum_general_term_formula :
  (∃ (a : ℕ → ℕ), (∃ d, S 3 a + S 5 a = 50 ∧ geometric_sequence a 1 a 4 a 13 ∧ d ≠ 0) →
  general_term_formula a d) :=
sorry

theorem sequence_T_n_sum :
  (∀ (a b : ℕ → ℕ) (n : ℕ), 
  (∀ n, a n = 2 * n + 1) →
  (∀ n,  \sum_{k=1}^n (b k) / (3 ^ n) = a n - 1) →
  T n (λ i, i * b i) = ((2 * n + 1) * (3 ^ (n + 1)) + 3) / 2 ) :=
sorry

end arithmetic_sequence_sum_general_term_formula_sequence_T_n_sum_l529_529528


namespace proof_S5_a5_l529_529166

variables (S a : ℕ → ℕ)

-- Conditions
axiom h1 : ∀ n, 3 * S n - 6 = 2 * a n
axiom S_def : S 1 = a 1 

-- Given the sum of the sequence and the initial condition
def Sn (n : ℕ) : ℕ :=
  2 * (1 - (-2)^n)

def an (n : ℕ) : ℕ :=
  6 * (-2)^(n - 1)

-- Proof problem
theorem proof_S5_a5 :
  S 5 / a 5 = 11 / 16 :=
sorry

end proof_S5_a5_l529_529166


namespace length_of_each_train_l529_529752

variable (L : ℝ) -- The length of each train in meters
variable (v1 v2 : ℝ) -- The speed of the faster and slower trains in km/hr
variable (t : ℝ) -- Time in seconds

-- Condition 1: Two trains running in the same direction at given speeds
axiom (h1 : v1 = 46)
axiom (h2 : v2 = 36)

-- Condition 2: The faster train passes the slower train in 54 seconds
axiom (h3 : t = 54)

-- Define the relative speed in m/s
def relative_speed (v1 v2 : ℝ) : ℝ := (v1 - v2) * (1000 / 3600)

-- Define the distance covered by the faster train while passing the slower train
def distance_covered (v1 v2 t : ℝ) : ℝ := relative_speed v1 v2 * t

-- Theorem to prove: The length of each train is 75 meters
theorem length_of_each_train : h1 ∧ h2 ∧ h3 → 2 * L = distance_covered v1 v2 t → L = 75 :=
by
  intro h
  have h_rel_speed : relative_speed 46 36 = 2.7777777777777777 := sorry
  have h_distance : distance_covered 46 36 54 = 150 := sorry
  sorry

end length_of_each_train_l529_529752


namespace basketball_league_total_games_l529_529763

theorem basketball_league_total_games
  (teams : ℕ)
  (games_per_opponent : ℕ)
  (teams_eq : teams = 12)
  (games_per_opponent_eq : games_per_opponent = 4) :
  ∃ total_games : ℕ, total_games = 264 :=
by
  have total_games := (teams * (teams - 1) / 2) * games_per_opponent
  rw [teams_eq, games_per_opponent_eq] at total_games
  use total_games
  show total_games = 264
  sorry

end basketball_league_total_games_l529_529763


namespace least_positive_integer_with_divisors_l529_529502

theorem least_positive_integer_with_divisors {m k : ℕ} :
  (∃ n : ℕ, n = m * 12^k ∧ 
            (∀ d : ℕ, (d | 12) → ¬(d | m)) ∧ 
            ∃ a b c : ℕ, n = 2^a * 3^b * c ∧ (a + 1) * (b + 1) * (c + 1) = 2023) →
  m + k = 6569 :=
sorry

end least_positive_integer_with_divisors_l529_529502


namespace houston_to_dallas_bus_encounters_l529_529803

theorem houston_to_dallas_bus_encounters :
  (∀ (depart  : Nat), 0 ≤ depart ∧ depart < 24 → ∃ (count : Nat), count = 11) →
  (∀ (depart : Nat), 0 ≤ depart ∧ depart < 24 → (depart + 1) = 45 ∧ (depart + 2) = 6) →
  (∀ (depart : Nat), 0 ≤ depart  ∧ depart < 24 ∧ (depart_mod 1 = 0) → 
  (∃ (timing: Nat), timing = 0 ∨ hbound_buses = 0)) →
  sorry

end houston_to_dallas_bus_encounters_l529_529803


namespace magnitude_of_vec_sub_is_sqrt3_l529_529911

def a : ℝ × ℝ × ℝ := (2, 3, 1)
def b : ℝ × ℝ × ℝ := (1, 2, 0)

def vec_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

theorem magnitude_of_vec_sub_is_sqrt3 : magnitude (vec_sub a b) = Real.sqrt 3 :=
by
  sorry

end magnitude_of_vec_sub_is_sqrt3_l529_529911


namespace calculate_surface_area_l529_529762

-- Define the initial cube dimensions and properties
def big_cube_length : ℕ := 12
def small_cube_length : ℕ := 2

-- Define the removal operations
def center_cubes_removed : ℕ := 7
def face_cubes_removed : ℕ := 6 -- Each face's central 2x2x2 removed

-- Define the expected surface area
def expected_surface_area : ℕ := 3304

-- Prove the surface area calculation correctness
theorem calculate_surface_area (n : ℕ) 
  (initial_cube_length : ℕ = big_cube_length) 
  (small_cube_length' : ℕ = small_cube_length)
  (center_removed : ℕ = center_cubes_removed) 
  (face_removed : ℕ = face_cubes_removed) : 
  -- Correct Surface Area Calculation of the Final Structure
  n = expected_surface_area := 
by
  -- Expected proof here
  sorry

end calculate_surface_area_l529_529762


namespace largest_real_solution_l529_529101

theorem largest_real_solution (x : ℝ) (h : (⌊x⌋ / x = 7 / 8)) : x ≤ 48 / 7 := by
  sorry

end largest_real_solution_l529_529101


namespace melanie_last_number_l529_529283

def marking_skipping_sequence (n : ℕ) : ℕ → List ℕ
| 0     := List.range (n + 1)
| (k+1) := 
  let previous_round := marking_skipping_sequence k in
  let remaining := previous_round.enum.filter (λ ⟨idx, _⟩, (idx % (k+2 + (2*((k+1) % 2))) ≠ 1)).map Prod.snd in
  remaining

theorem melanie_last_number (n : ℕ) : n = 50 → 
  ∃ x, marking_skipping_sequence n x ∧ x == 47 := 
begin
  intros h,
  sorry

end melanie_last_number_l529_529283


namespace construct_triangle_given_angles_radius_l529_529056

-- Define the given angles and radius
variables (α β : ℝ) (r : ℝ)

-- Define the conditions for the construction of the triangle ABC
theorem construct_triangle_given_angles_radius (h1 : 0 < α ∧ α < 180) (h2 : 0 < β ∧ β < 180) (h3 : 0 < r) :
  ∃ (A B C : Type), 
    is_triangle A B C ∧ 
    has_incircle A B C r ∧ 
    angle A B C = α ∧
    angle B A C = β :=
begin
  sorry -- the proof of the construction will be here
end

end construct_triangle_given_angles_radius_l529_529056


namespace tangent_line_to_parabola_l529_529002

variable {c : ℝ}
def line (x : ℝ) := 3 * x + c
def parabola (y : ℝ) := y^2 = 12 * (y / 3 - c / 3)

theorem tangent_line_to_parabola : ∃ c : ℝ, (c = 3 ∧ ∀ y, parabola y = (line (y / 3 - c / 3))^2) :=
by
  exists 3
  sorry

end tangent_line_to_parabola_l529_529002


namespace f_inequalities_l529_529557

def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem f_inequalities :
  f 1 < f (real.sqrt 3) ∧ f (real.sqrt 3) < f (-1) :=
by
  sorry

end f_inequalities_l529_529557


namespace correct_statement_is_D_l529_529459

-- Definitions based on conditions
def isFrequencyDistributionEqual (area : ℝ) (frequency : ℝ) : Prop := area = frequency
def isStandardDeviationSquareRootOfVariance (stddev : ℝ) (variance : ℝ) : Prop := stddev = real.sqrt variance
def variance_of (data : list ℝ) : ℝ := -- definition of variance
  let mean := (list.sum data) / (data.length : ℝ) in
  (list.sum (data.map (λ x, (x - mean) ^ 2))) / (data.length : ℝ)
def isVarianceRelated (data1 data2 : list ℝ) (ratio : ℝ) : Prop :=
  variance_of data1 = ratio * variance_of data2

-- Conditions specific to the problem
def data1 := [2, 3, 4, 5]
def data2 := [4, 6, 8, 10]

-- Proof problem statement
theorem correct_statement_is_D :
  ¬ isFrequencyDistributionEqual (area := some_area) (frequency := some_frequency) ∧
  ¬ isStandardDeviationSquareRootOfVariance (stddev := some_stddev) (variance := some_variance) ∧
  ¬ isVarianceRelated data1 data2 (ratio := 1/2) ∧
  (∀ (data : list ℝ), ∃ (var1 var2 : ℝ), var1 > var2 → var1 > variance_of data) → -- condition for D
  true := -- proving statement D
sorry

end correct_statement_is_D_l529_529459


namespace find_k_l529_529818

def f (x : ℝ) : ℝ := 7 * x^2 - 1 / x + 5
def g (x : ℝ) (k : ℝ) : ℝ := x^2 - k

theorem find_k : ∃ k : ℝ, f 3 - g 3 k = 3 ∧ k = 176 / 3 := by
  sorry

end find_k_l529_529818


namespace parabola_intercepts_sum_l529_529326

theorem parabola_intercepts_sum (a b c d : ℝ) (h1 : a = 0) (h2 : b = 0) (h3 : c = 0) (h4 : d = 0) :
  a + b + c + d = 0 :=
by
  rw [h1, h2, h3, h4]
  exact add_zero (add_zero (add_zero zero))

end parabola_intercepts_sum_l529_529326


namespace fixed_point_of_exponential_function_l529_529695

theorem fixed_point_of_exponential_function (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : 
  ∃ p : ℝ × ℝ, p = (-2, 1) ∧ ∀ x : ℝ, (x, a^(x + 2)) = p → x = -2 ∧ a^(x + 2) = 1 :=
by
  sorry

end fixed_point_of_exponential_function_l529_529695


namespace math_proof_equation_and_area_and_PD_PE_l529_529531

noncomputable def ellipse_equation : Prop :=
  ∃ a b : ℝ, a^2 = 4 ∧ b^2 = 1 ∧
  ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ (x^2 / 4) + y^2 = 1

noncomputable def minimum_area_triangle : Prop :=
  ∀ (l : ℝ → ℝ) (h : ∃ k : ℝ, l = λ x, k * x + sqrt (4 * k^2 + 1) ∧ k < 0),
  let area := 1 / 2 * (sqrt (4 * (h.some_spec.some) ^ 2 + 1) ^ 2 / (-h.some_spec.some)) in
  area = 2

noncomputable def PD_equals_PE (A B D E C P : ℝ × ℝ) : Prop :=
  let perpend_orth := ∀ (u v : ℝ × ℝ), (u.1 - A.1) * (v.1 - B.1) + (u.2 - A.2) * (v.2 - B.2) = 0 in
  let par_parallel := ∀ (u v : ℝ × ℝ), (u.1 - A.1) / (v.1 - D.1) = (u.2 - A.2) / (v.2 - D.2) in
  let midpoint := ∀ u v : ℝ × ℝ, (u.1 + v.1) / 2 = P.1 ∧ (u.2 + v.2) / 2 = P.2 in
  perpend_orth (B - A) (C - B) ∧ par_parallel (D - A) (C - 0) ∧ midpoint D E →
  (D - P) = (E - P)

theorem math_proof_equation_and_area_and_PD_PE :
  ellipse_equation ∧ minimum_area_triangle ∧
  ∀ (A B D E C P : ℝ × ℝ), PD_equals_PE A B D E C P :=
by sorry

end math_proof_equation_and_area_and_PD_PE_l529_529531


namespace odd_and_shifted_even_implies_f4_zero_l529_529529

variable (f : ℝ → ℝ)

-- Define the conditions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Define the given problem as a Lean theorem
theorem odd_and_shifted_even_implies_f4_zero
  (h_odd : is_odd f)
  (h_even : is_even (λ x, f(x+1))) :
  f 4 = 0 :=
by sorry

end odd_and_shifted_even_implies_f4_zero_l529_529529


namespace min_value_max_abs_quad_expression_l529_529120

theorem min_value_max_abs_quad_expression : 
  let f (x y : ℝ) : ℝ := |x^2 - 2 * x * y|
  let x_range (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1
  ∃ y ∈ Set.univ, let m := (⨅ x ∈ Set.Icc 0 1, f x y), m = (3 - 2 * Real.sqrt 2) := 
sorry

end min_value_max_abs_quad_expression_l529_529120


namespace part_1_part_2_l529_529909

-- Definitions for sets M and N
def M : Set ℝ := {x | -2 < x ∧ x < 3}
def N (m : ℝ) : Set ℝ := {x | x ≥ m}

-- Proof problem 1: Prove that if M ∪ N = N, then m ≤ -2
theorem part_1 (m : ℝ) : (M ∪ N m = N m) → m ≤ -2 :=
by sorry

-- Proof problem 2: Prove that if M ∩ N = ∅, then m ≥ 3
theorem part_2 (m : ℝ) : (M ∩ N m = ∅) → m ≥ 3 :=
by sorry

end part_1_part_2_l529_529909


namespace length_of_diagonal_of_cube_l529_529365

theorem length_of_diagonal_of_cube (S : ℝ) (h : S = 864) : 
  let s := real.sqrt (S / 6) in
  let d := s * real.sqrt 3 in
  d = 12 * real.sqrt 3 :=
by
  sorry

end length_of_diagonal_of_cube_l529_529365


namespace parallelepiped_volume_l529_529922

-- Definitions
variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1)
variable (angle_pi_div4 : real.angle_between a b = real.pi / 4)

-- Theorem statement
theorem parallelepiped_volume : 
  real.abs (inner_product_space.Inner ℝ a ((b + 2 • (b ×ₗ a)) ×ₗ b)) = 1 :=
sorry

end parallelepiped_volume_l529_529922


namespace perpendicular_tangent_line_l529_529497

def f (x : ℝ) : ℝ := x^3 + 3 * x^2 - 1

def g : AffineLine ℝ :=
{ a := 2,
  b := -6,
  c := 1 }

def m : Slope ℝ :=
{ num := 1,
  denom := 3 }

theorem perpendicular_tangent_line :
  ∃ a b c x0 y0 : ℝ,
    f x0 = y0 ∧
    y0 = -1 + 3 - 1 ∧
    g.slope = m ∧
    (a * x0 + b * y0 + c = 0) ∧
    (a = 3 ∧ b = 1 ∧ c = 2) :=
sorry

end perpendicular_tangent_line_l529_529497


namespace probability_is_correct_l529_529438

noncomputable def probability_cashier_opens_early : ℝ :=
  let x1 : ℝ := sorry
  let x2 : ℝ := sorry
  let x3 : ℝ := sorry
  let x4 : ℝ := sorry
  let x5 : ℝ := sorry
  let x6 : ℝ := sorry
  if (0 <= x1) ∧ (x1 <= 15) ∧
     (0 <= x2) ∧ (x2 <= 15) ∧
     (0 <= x3) ∧ (x3 <= 15) ∧
     (0 <= x4) ∧ (x4 <= 15) ∧
     (0 <= x5) ∧ (x5 <= 15) ∧
     (0 <= x6) ∧ (x6 <= 15) ∧
     (x1 < x6) ∧ (x2 < x6) ∧ (x3 < x6) ∧ (x4 < x6) ∧ (x5 < x6) then 
    let p_not_A : ℝ := (12 / 15) ^ 6
    1 - p_not_A
  else
    0

theorem probability_is_correct : probability_cashier_opens_early = 0.738 :=
by sorry

end probability_is_correct_l529_529438


namespace largest_n_satisfying_condition_l529_529501

theorem largest_n_satisfying_condition
  (exists_n_elements: ∀ (n : ℕ) (S : Finset ℕ), S.card = n → 
    (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → ¬ (x ∣ y)) → 
    (∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S → (∃ (w : ℕ), w ∈ S ∧ w ∣ (x + y))) → 
    n ≤ 6) : 
  ∃ n (S : Finset ℕ), n = 6 ∧ S.card = n ∧
    (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → ¬ (x ∣ y)) ∧
    (∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S → 
      ∃ (w : ℕ), w ∈ S ∧ w ∣ (x + y)) :=
sorry

end largest_n_satisfying_condition_l529_529501


namespace expected_value_consecutive_red_draws_l529_529684

/-- The expected value of the total number of draws until two consecutive red balls are drawn,
given a bag of four differently colored balls drawn independently with replacement. --/
theorem expected_value_consecutive_red_draws :
  let ζ := -- define the total number of draws until two consecutive red draws here
  ζ = 2 -- because the variables E, ζ, and condition are not provided explicitly as definitions
 )
  -- some expectation mechanism and probability involved directly similar in the proof definition
  sorry
  :=
  sorry
where E is the expected value defined in steps cases, we need justification on letter precise defined.
  )
  20 :=
sorry

end expected_value_consecutive_red_draws_l529_529684


namespace cashier_window_open_probability_l529_529440

noncomputable def probability_window_opens_in_3_minutes_of_scientist_arrival : ℝ := 
  0.738

theorem cashier_window_open_probability :
  let x : ℝ → ℝ := λ x, if x ≥ 12 then 0.738 else 0.262144 in
  ∀ (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ),
    (∀ i ∈ [x₁, x₂, x₃, x₄, x₅], i < x₆) ∧ 0 ≤ x₆ ≤ 15 →
    x x₆ = 0.738 :=
by
  sorry

end cashier_window_open_probability_l529_529440


namespace largest_real_solution_l529_529100

theorem largest_real_solution (x : ℝ) (h : (⌊x⌋ / x = 7 / 8)) : x ≤ 48 / 7 := by
  sorry

end largest_real_solution_l529_529100


namespace three_digit_numbers_satisfying_condition_l529_529739

theorem three_digit_numbers_satisfying_condition :
  ∀ (N : ℕ), (100 ≤ N ∧ N < 1000) →
    ∃ (a b c : ℕ),
      (N = 100 * a + 10 * b + c) ∧ (N = 11 * (a^2 + b^2 + c^2)) 
    ↔ (N = 550 ∨ N = 803) :=
by
  sorry

end three_digit_numbers_satisfying_condition_l529_529739


namespace bones_weight_in_meat_l529_529408

theorem bones_weight_in_meat (cost_with_bones : ℝ) (cost_without_bones : ℝ) (cost_bones : ℝ) :
  cost_with_bones = 165 → cost_without_bones = 240 → cost_bones = 40 → 
  ∃ x : ℝ, (40 * x + 240 * (1 - x) = 165) ∧ (x * 1000 = 375) :=
by
  intros h1 h2 h3
  use 0.375
  split
  · calc
      40 * 0.375 + 240 * (1 - 0.375)
        = 15 + 240 * 0.625 : by rw [show 0.375 = 3 / 8, by norm_num]
        = 15 + 150 : by rw [show 240 * 0.625 = 150, by norm_num]
        = 165 : by norm_num
  · calc
      0.375 * 1000 = 375 : by norm_num

-- The complete proof is included to demonstrate correctness and ensure the validity of the statement.

end bones_weight_in_meat_l529_529408


namespace inequality_solution_inequality_proof_l529_529560

def f (x: ℝ) := |x - 5|

theorem inequality_solution : {x : ℝ | f x + f (x + 2) ≤ 3} = {x | 5 / 2 ≤ x ∧ x ≤ 11 / 2} :=
sorry

theorem inequality_proof (a x : ℝ) (h : a < 0) : f (a * x) - f (5 * a) ≥ a * f x :=
sorry

end inequality_solution_inequality_proof_l529_529560


namespace smallest_positive_period_of_f_minimum_value_of_f_value_of_f_2alpha_l529_529175

noncomputable def f (x : ℝ) : ℝ := sin (x + 7 * Real.pi / 4) + cos (x - 3 * Real.pi / 4)

theorem smallest_positive_period_of_f : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x :=
  sorry

theorem minimum_value_of_f : ∃ m, ∀ x : ℝ, f x ≥ m ∧ (∃ x0, f x0 = m) := 
  sorry

theorem value_of_f_2alpha (α : ℝ) (h1 : 0 < α) (h2 : α < 3 * Real.pi / 4)
    (h : f α = 6 / 5) : f (2 * α) = 31 * Real.sqrt 2 / 25 :=
  sorry

end smallest_positive_period_of_f_minimum_value_of_f_value_of_f_2alpha_l529_529175


namespace largest_x_satisfies_condition_l529_529095

theorem largest_x_satisfies_condition :
  ∃ x : ℝ, (⌊x⌋ / x = 7 / 8) ∧ (∀ y : ℝ, (⌊y⌋ / y = 7 / 8) → y ≤ 48 / 7) :=
sorry

end largest_x_satisfies_condition_l529_529095


namespace line_count_l529_529410

def point (α : Type) := α × α

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def line_through_point (k x y : ℝ) : Prop := y = k * (x + 4) + 1

def intersects_only_one_point (ℓ: ℝ → ℝ → Prop) (P: point ℝ) (C: ℝ → ℝ → Prop) : Prop :=
∀ x y, ℓ x y ∧ C x y → ∃! x0 y0, ℓ x0 y0 ∧ C x0 y0

theorem line_count (k : ℝ) :
  ∃ (k1 k2 k3 : ℝ), k ≠ 0 ∧
  (line_through_point k1 (-4) 1 ∧ parabola k1 (k1 * (-4 + 4) + 1) ∧ intersects_only_one_point (line_through_point k1) (-4, 1) (parabola k1)) ∧
  (line_through_point k2 (-4) 1 ∧ parabola k2 (k2 * (-4 + 4) + 1) ∧ intersects_only_one_point (line_through_point k2) (-4, 1) (parabola k2)) ∧
  (line_through_point k3 (-4) 1 ∧ parabola k3 (k3 * (-4 + 4) + 1) ∧ intersects_only_one_point (line_through_point k3) (-4, 1) (parabola k3)) ∧
  ¬∃ k4, k4 ≠ k1 ∧ k4 ≠ k2 ∧ k4 ≠ k3 ∧
  line_through_point k4 (-4) 1 ∧ parabola k4 (k4 * (-4 + 4) + 1) ∧ intersects_only_one_point (line_through_point k4) (-4, 1) (parabola k4) :=
sorry

end line_count_l529_529410


namespace system_solution_l529_529997

theorem system_solution 
  (x₁ x₂ x₃ : ℝ)
  (h1 : 3*x₁ - x₂ + 3*x₃ = 5)
  (h2 : 2*x₁ - x₂ + 4*x₃ = 5)
  (h3 : x₁ + 2*x₂ - 3*x₃ = 0) :
  x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1 :=
by
  sorry

end system_solution_l529_529997


namespace solution_l529_529883

noncomputable def problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1/x + 4/y = 1) : Prop :=
  x + y ≥ 9

theorem solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1/x + 4/y = 1) : problem x y h1 h2 h3 :=
  sorry

end solution_l529_529883


namespace probability_of_green_ball_l529_529057

-- Definitions according to the conditions.
def containerA : ℕ × ℕ := (4, 6) -- 4 red balls, 6 green balls
def containerB : ℕ × ℕ := (6, 4) -- 6 red balls, 4 green balls
def containerC : ℕ × ℕ := (6, 4) -- 6 red balls, 4 green balls

-- Proving the probability of selecting a green ball.
theorem probability_of_green_ball :
  let pA := 1 / 3
  let pB := 1 / 3
  let pC := 1 / 3
  let pGreenA := (containerA.2 : ℚ) / (containerA.1 + containerA.2)
  let pGreenB := (containerB.2 : ℚ) / (containerB.1 + containerB.2)
  let pGreenC := (containerC.2 : ℚ) / (containerC.1 + containerC.2)
  pA * pGreenA + pB * pGreenB + pC * pGreenC = 7 / 15
  :=
by
  -- Formal proof will be filled in here.
  sorry

end probability_of_green_ball_l529_529057


namespace max_total_area_of_rectangles_l529_529014

theorem max_total_area_of_rectangles (ABC : Triangle) (hABC : ABC.area = 1) 
    (R : Rectangle) (S : Rectangle) 
    (vertices_R_on_sides : R.vertices_on_sides ABC) 
    (vertices_S_on_sides_and_R34 : S.vertices_on_sides_and_R34 ABC R) : 
    (∀ k ∈ Icc (0 : ℝ) 1, 2 * k * (1 - k) + 2 * (1 / 2) * (1 - 1 / 2) * k ^ 2 ≤ 2 / 3) :=
sorry

end max_total_area_of_rectangles_l529_529014


namespace area_triangle_ASD_ratio_smallest_triangle_section_l529_529393

-- Definitions and Conditions
def is_regular_hexagon (AB BC CD DE EF FA : ℝ) : Prop :=
  AB = BC ∧ BC = CD ∧ CD = DE ∧ DE = EF ∧ EF = FA ∧ FA = AB

def is_perpendicular (SA : ℝ) (base_plane : ℝ) : Prop :=
  SA ⊥ base_plane

def distance_to_line (B C : ℝ) (SD : ℝ) : Prop :=
  dist B SD = real.sqrt (23 / 14) ∧ dist C SD = real.sqrt (15 / 14)

-- Theorem for Part (a)
theorem area_triangle_ASD 
  (a h : ℝ) (AB BC CD DE EF FA : ℝ)
  (Hhex : is_regular_hexagon AB BC CD DE EF FA)
  (Hperp : is_perpendicular SA (base_plane))
  (Hdist : distance_to_line B C SD) :
  area_triangle ASD = (8 * real.sqrt 11) / (3 * real.sqrt 33) :=
sorry

-- Theorem for Part (b)
theorem ratio_smallest_triangle_section
  (a h : ℝ) (AB BC CD DE EF FA : ℝ)
  (Hhex : is_regular_hexagon AB BC CD DE EF FA)
  (Hperp : is_perpendicular SA (base_plane))
  (Hdist : distance_to_line B C SD) :
  ratio_smallest_area_to_triangle_ASD = real.sqrt (14 / 15) :=
sorry

end area_triangle_ASD_ratio_smallest_triangle_section_l529_529393


namespace find_lambda_l529_529257

def Q := {q : ℚ // true}
def Z := {z : ℤ // true}

def A (m : ℤ) : set (ℚ × ℚ) :=
  { p : ℚ × ℚ | p.1 ≠ 0 ∧ p.2 ≠ 0 ∧ (p.1 * p.2 / m : ℚ).den = 1 }

def f (m : ℤ) (MN : set (ℚ × ℚ)) : ℕ :=
  MN.to_finset.filter (A m).has_mem.mem).card

theorem find_lambda :
  ∃ λ : ℝ, (λ = 2) ∧ 
            ∀ l : set (ℚ × ℚ),
            ∃ β : ℝ,
            ∀ M N ∈ l,
            f 2016 {t : ℚ × ℚ | t.1 = M ∨ t.2 = N} ≤ λ * f 2015 {t : ℚ × ℚ | t.1 = M ∨ t.2 = N} + β :=
begin
  sorry
end

end find_lambda_l529_529257


namespace prove_real_roots_and_find_m_l529_529865

-- Condition: The quadratic equation
def quadratic_eq (m x : ℝ) : Prop := x^2 - (m-1)*x + m-2 = 0

-- Condition: Discriminant
def discriminant (m : ℝ) : ℝ := (m-3)^2

-- Define the problem as a proposition
theorem prove_real_roots_and_find_m (m : ℝ) :
  (discriminant m ≥ 0) ∧ 
  (|3 - m| = 3 → (m = 0 ∨ m = 6)) :=
by
  sorry

end prove_real_roots_and_find_m_l529_529865


namespace beads_used_total_l529_529623

theorem beads_used_total :
  let necklaces_monday := 10
  let necklaces_tuesday := 2
  let bracelets_wednesday := 5
  let earrings_wednesday := 7
  let beads_per_necklace := 20
  let beads_per_bracelet := 10
  let beads_per_earring := 5

  (necklaces_monday + necklaces_tuesday) * beads_per_necklace + 
  bracelets_wednesday * beads_per_bracelet + 
  earrings_wednesday * beads_per_earring = 325 := by
  let necklaces_monday := 10
  let necklaces_tuesday := 2
  let bracelets_wednesday := 5
  let earrings_wednesday := 7
  let beads_per_necklace := 20
  let beads_per_bracelet := 10
  let beads_per_earring := 5

  calc
    (necklaces_monday + necklaces_tuesday) * beads_per_necklace + 
    bracelets_wednesday * beads_per_bracelet + 
    earrings_wednesday * beads_per_earring
    = (10 + 2) * 20 + 5 * 10 + 7 * 5 : by rfl
    ... = 12 * 20 + 5 * 10 + 7 * 5 : by rfl
    ... = 240 + 50 + 35 : by rfl
    ... = 325 : by rfl

end beads_used_total_l529_529623


namespace kylie_beads_total_l529_529621

def number_necklaces_monday : Nat := 10
def number_necklaces_tuesday : Nat := 2
def number_bracelets_wednesday : Nat := 5
def number_earrings_wednesday : Nat := 7

def beads_per_necklace : Nat := 20
def beads_per_bracelet : Nat := 10
def beads_per_earring : Nat := 5

theorem kylie_beads_total :
  (number_necklaces_monday + number_necklaces_tuesday) * beads_per_necklace + 
  number_bracelets_wednesday * beads_per_bracelet + 
  number_earrings_wednesday * beads_per_earring = 325 := 
by
  sorry

end kylie_beads_total_l529_529621


namespace repeated_sequence_appear_l529_529327

theorem repeated_sequence_appear (n : ℕ) : 
  let AB = ['A', 'B', 'C', 'D'] in
  let digits = [2, 0, 2, 3] in
  (∀ k, k % 4 = 0 → (List.rotate AB k = AB ∧ List.rotate digits k = digits)) :=
begin
  intros k hk,
  split;
  { rw List.rotate_eq_drop_append_take,
    rw At_top.ne_zero_iff,
    exact hk },
end

end repeated_sequence_appear_l529_529327


namespace matrix_A_pow4_eq_l529_529470

open Matrix

variable {α : Type*} [Fintype α] [DecidableEq α]
variables (A : Matrix (Fin 2) (Fin 2) ℤ)

def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![[1, -1], [1, 0]]

def matrix_A3 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![[-1, 0], [0, -1]]

theorem matrix_A_pow4_eq :
  matrix_A ^ 4 = ![[-1, 1], [-1, 0]] :=
by
  /- proof would go here -/
  sorry

end matrix_A_pow4_eq_l529_529470


namespace line_equation_intersections_l529_529062

theorem line_equation_intersections (m b k : ℝ) (h1 : b ≠ 0) 
  (h2 : m * 2 + b = 7) (h3 : abs (k^2 + 8*k + 7 - (m*k + b)) = 4) :
  m = 6 ∧ b = -5 :=
by {
  sorry
}

end line_equation_intersections_l529_529062


namespace maximize_l_a_l529_529182

noncomputable def max_l (a : ℝ) (h : a < 0) : ℝ := 
if h : a = -2 then 2 + real.sqrt 3 
else if h : a = -8 then 1 
else 0

theorem maximize_l_a (a l : ℝ) (h1 : a < 0) 
  (h2 : ∀ x, 0 ≤ x ∧ x ≤ l → abs (a * x^2 + 8 * x + 3) ≤ 5) :
  a = -2 ∧ l = 2 + real.sqrt 3 :=
by
  -- Proof goes here
  sorry

end maximize_l_a_l529_529182


namespace cashier_opens_probability_l529_529445

-- Definition of the timeline and arrival times
variables {x₁ x₂ x₃ x₄ x₅ x₆ : ℝ}
-- Condition that all arrival times are between 0 and 15 minutes
def arrival_times_within_bounds : Prop := 
  0 ≤ x₁ ∧ x₁ ≤ 15 ∧ 
  0 ≤ x₂ ∧ x₂ ≤ 15 ∧
  0 ≤ x₃ ∧ x₃ ≤ 15 ∧
  0 ≤ x₄ ∧ x₄ ≤ 15 ∧
  0 ≤ x₅ ∧ x₅ ≤ 15 ∧
  0 ≤ x₆ ∧ x₆ ≤ 15

-- Condition that the Scientist arrives last
def scientist_arrives_last : Prop := 
  x₁ < x₆ ∧ x₂ < x₆ ∧ x₃ < x₆ ∧ x₄ < x₆ ∧ x₅ < x₆

-- Event A: Cashier opens no later than 3 minutes after the Scientist arrives, i.e., x₆ ≥ 12
def event_A : Prop := x₆ ≥ 12

-- The correct answer
theorem cashier_opens_probability :
  arrival_times_within_bounds ∧ scientist_arrives_last → 
  Pr[x₆ ≥ 12 | x₁, x₂, x₃, x₄, x₅ < x₆] = 0.738 :=
sorry

end cashier_opens_probability_l529_529445


namespace solve_cubic_equation_l529_529492

theorem solve_cubic_equation (x : ℚ) : (∃ x : ℚ, ∛(5 - x) = -5/3 ∧ x = 260/27) :=
by
  sorry

end solve_cubic_equation_l529_529492


namespace total_respondents_l529_529654

theorem total_respondents (X Y : ℕ) 
  (hX : X = 60) 
  (hRatio : 3 * Y = X) : 
  X + Y = 80 := 
by
  sorry

end total_respondents_l529_529654


namespace minutes_away_l529_529772

/--
Shortly after 6:00 PM, the hands of a watch form an angle of 130 degrees. The man returns before 
7:00 PM and sees the same 130 degrees angle again. Prove that the man was away for 47 minutes.
-/
theorem minutes_away (n : ℝ) (h1 : |180 + n / 2 - 6 * n| = 130) 
  (h2 : 6 < n ∧ n < 60) :
  n = 47 :=
begin
  sorry
end

end minutes_away_l529_529772


namespace length_of_TrainA_correct_l529_529351

variables {TrainA_speed_kmph TrainB_speed_kmph TrainC_speed_kmph : ℚ}
variables {Time_to_pass_TrainB Time_to_pass_TrainC : ℚ}
variables (length_TrainA : ℚ)

noncomputable def kmph_to_mps (v : ℚ) : ℚ := v * (5 / 18)

-- Given conditions
axiom h1 : TrainA_speed_kmph = 72
axiom h2 : TrainB_speed_kmph = 36
axiom h3 : TrainC_speed_kmph = 54
axiom h4 : Time_to_pass_TrainB = 18
axiom h5 : Time_to_pass_TrainC = 36

-- Prove the length of Train A is 180 meters
theorem length_of_TrainA_correct :
  length_TrainA = 180 :=
sorry

end length_of_TrainA_correct_l529_529351


namespace no_integer_solution_l529_529241

theorem no_integer_solution (N : ℤ) (digits : ℕ) (pattern : ℤ) (h_pattern : pattern = Nat.iterate digits (λ n, 10 * n + 2) 0) :
  ¬(2008 * N = pattern) :=
by
  sorry

end no_integer_solution_l529_529241


namespace sparrows_initial_count_l529_529346

theorem sparrows_initial_count (a b c : ℕ) 
  (h1 : a + b + c = 24)
  (h2 : a - 4 = b + 1)
  (h3 : b + 1 = c + 3) : 
  a = 12 ∧ b = 7 ∧ c = 5 :=
by
  sorry

end sparrows_initial_count_l529_529346


namespace find_k_value_l529_529656

theorem find_k_value 
  (A B C k : ℤ)
  (hA : A = -3)
  (hB : B = -5)
  (hC : C = 6)
  (hSum : A + B + C + k = -A - B - C - k) : 
  k = 2 :=
sorry

end find_k_value_l529_529656


namespace simplify_fraction_l529_529782

theorem simplify_fraction (a b : ℝ) (h : a ≠ b): 
  (a - b) / a / (a - (2 * a * b - b^2) / a) = 1 / (a - b) :=
by
  sorry

end simplify_fraction_l529_529782


namespace distinct_reciprocal_sum_l529_529203

theorem distinct_reciprocal_sum :
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ (6 ∣ A) ∧ (6 ∣ B) ∧ (6 ∣ C) ∧ 
  (1 / (A : ℚ) + 1 / (B : ℚ) + 1 / (C : ℚ) = 1 / 6) ∧ 
  (A = 12) ∧ (B = 18) ∧ (C = 36) :=
begin
  use [12, 18, 36],
  split, { exact nat.ne_of_lt (nat.lt_of_lt_of_le dec_trivial (le_refl 18)) },
  split, { exact nat.ne_of_lt (nat.lt_of_lt_of_le dec_trivial (le_of_lt_succ (nat.succ_pos 35))) },
  split, { exact nat.ne_of_lt (nat.lt_of_lt_of_le dec_trivial (le_of_lt_succ (nat.succ_pos 11))) },
  split,
  { use 2, refl },
  split,
  { use 3, refl },
  split,
  { use 6, refl },
  split,
  { norm_num },
  split,
  { refl },
  split,
  { refl },
  { refl }
end

end distinct_reciprocal_sum_l529_529203


namespace divide_100_representatives_l529_529718

noncomputable def divide_representatives (n : ℕ) (k : ℕ) (m : ℕ) : Prop :=
  ∃ (grouping : Fin n → Fin m), (∀ i : Fin n, grouping i < Fin k) ∧
    (∀ i : Fin n, ∀ j : Fin n, i ≠ j → 
      seating_adjacent i j → grouping i ≠ grouping j) ∧
    (∀ i : Fin n, ∃! (j : Fin n), rep_country i = rep_country j ∧ grouping j = grouping (j+1)%n)

-- The main theorem statement
theorem divide_100_representatives :
  divide_representatives 100 4 25 :=
sorry

end divide_100_representatives_l529_529718


namespace count_young_diagrams_4_count_young_diagrams_5_count_young_diagrams_6_count_young_diagrams_7_l529_529845

-- Define the weight 's'.
variable (s : ℕ)

-- Define the function that counts the number of Young diagrams for a given weight.
def countYoungDiagrams (s : ℕ) : ℕ :=
  -- Placeholder for actual implementation of counting Young diagrams.
  sorry

-- Prove that the count of Young diagrams for s = 4 is 5
theorem count_young_diagrams_4 : countYoungDiagrams 4 = 5 :=
by sorry

-- Prove that the count of Young diagrams for s = 5 is 7
theorem count_young_diagrams_5 : countYoungDiagrams 5 = 7 :=
by sorry

-- Prove that the count of Young diagrams for s = 6 is 11
theorem count_young_diagrams_6 : countYoungDiagrams 6 = 11 :=
by sorry

-- Prove that the count of Young diagrams for s = 7 is 15
theorem count_young_diagrams_7 : countYoungDiagrams 7 = 15 :=
by sorry

end count_young_diagrams_4_count_young_diagrams_5_count_young_diagrams_6_count_young_diagrams_7_l529_529845


namespace exists_100_distinct_naturals_cubed_sum_l529_529483

theorem exists_100_distinct_naturals_cubed_sum : 
  ∃ (S : Finset ℕ), S.card = 100 ∧ 
  ∃ (x : ℕ), x ∈ S ∧ 
  x ^ 3 = (S.erase x).sum (λ y, y ^ 3) :=
begin
  sorry
end

end exists_100_distinct_naturals_cubed_sum_l529_529483


namespace find_initial_nickels_l529_529668

variable (initial_nickels current_nickels borrowed_nickels : ℕ)

def initial_nickels_equation (initial_nickels current_nickels borrowed_nickels : ℕ) : Prop :=
  initial_nickels - borrowed_nickels = current_nickels

theorem find_initial_nickels (h : initial_nickels_equation initial_nickels current_nickels borrowed_nickels) 
                             (h_current : current_nickels = 11) 
                             (h_borrowed : borrowed_nickels = 20) : 
                             initial_nickels = 31 :=
by
  sorry

end find_initial_nickels_l529_529668


namespace find_point_P_l529_529207

theorem find_point_P :
  ∃ (P : ℝ × ℝ), P.1 = 1 ∧ P.2 = 0 ∧ 
  (P.2 = P.1^4 - P.1) ∧
  (∃ m, m = 4 * P.1^3 - 1 ∧ m = 3) :=
by
  sorry

end find_point_P_l529_529207


namespace largest_real_number_condition_l529_529077

theorem largest_real_number_condition (x : ℝ) (hx : ⌊x⌋ / x = 7 / 8) : x ≤ 48 / 7 :=
by
  sorry

end largest_real_number_condition_l529_529077


namespace power_multiplication_same_base_l529_529759

theorem power_multiplication_same_base :
  (10 ^ 655 * 10 ^ 650 = 10 ^ 1305) :=
by {
  sorry
}

end power_multiplication_same_base_l529_529759


namespace partA_l529_529748

theorem partA (a b c : ℤ) (h : ∀ x : ℤ, ∃ k : ℤ, a * x ^ 2 + b * x + c = k ^ 4) : a = 0 ∧ b = 0 := 
sorry

end partA_l529_529748


namespace number_of_tacos_you_ordered_l529_529383

variable {E : ℝ} -- E represents the cost of one enchilada in dollars

-- Conditions
axiom h1 : ∃ t : ℕ, 0.9 * (t : ℝ) + 3 * E = 7.80
axiom h2 : 0.9 * 3 + 5 * E = 12.70

theorem number_of_tacos_you_ordered (E : ℝ) : ∃ t : ℕ, t = 2 := by
  sorry

end number_of_tacos_you_ordered_l529_529383


namespace reciprocal_eq_self_is_one_or_neg_one_l529_529337

/-- If a rational number equals its own reciprocal, then the number is either 1 or -1. -/
theorem reciprocal_eq_self_is_one_or_neg_one (x : ℚ) (h : x = 1 / x) : x = 1 ∨ x = -1 := 
by
  sorry

end reciprocal_eq_self_is_one_or_neg_one_l529_529337


namespace monotonic_decrease_range_of_a_l529_529544

theorem monotonic_decrease_range_of_a :
  ∀ (a : ℝ),
    (∀ x ∈ Ioo (2 : ℝ) 3, f' x ≤ 0) → a ∈ Icc (3 : ℝ) 4 :=
begin
  intro a,
  intro h,
  have f := λ x : ℝ, log (2 : ℝ) ((a * x) - (x ^ 2)),
  have f' := λ x : ℝ, (a - 2 * x) / ((a * x) - (x ^ 2)),
  sorry,
end

end monotonic_decrease_range_of_a_l529_529544


namespace brad_red_balloons_l529_529464

theorem brad_red_balloons (total balloons green : ℕ) (h1 : total = 17) (h2 : green = 9) : total - green = 8 := 
by {
  sorry
}

end brad_red_balloons_l529_529464


namespace exists_fraction_equal_to_d_minus_1_l529_529292

theorem exists_fraction_equal_to_d_minus_1 (n d : ℕ) (hdiv : d > 0 ∧ n % d = 0) :
  ∃ k : ℕ, k < n ∧ (n - k) / (n - (n - k)) = d - 1 :=
by
  sorry

end exists_fraction_equal_to_d_minus_1_l529_529292


namespace triangulation_is_catalan_l529_529665

noncomputable def catalan (n : ℕ) : ℕ :=
  if n = 0 then 1 else ∑ i in Finset.range n, catalan i * catalan (n - 1 - i)

def triangulation_count (n : ℕ) : ℕ :=
  if n = 2 then 1 else 
  if n = 3 then 1 else 
  if n = 4 then 2 else ∑ i in (Finset.range (n - 1)).filter (λ i, i ≥ 2), 
      triangulation_count i * triangulation_count (n - i)

theorem triangulation_is_catalan (n : ℕ) : 
  triangulation_count n = catalan (n - 2) :=
sorry

end triangulation_is_catalan_l529_529665


namespace steve_pencils_left_l529_529682

def initial_pencils : ℕ := 2 * 12
def pencils_given_to_lauren : ℕ := 6
def pencils_given_to_matt : ℕ := pencils_given_to_lauren + 3

def pencils_left (initial_pencils given_lauren given_matt : ℕ) : ℕ :=
  initial_pencils - given_lauren - given_matt

theorem steve_pencils_left : pencils_left initial_pencils pencils_given_to_lauren pencils_given_to_matt = 9 := by
  sorry

end steve_pencils_left_l529_529682


namespace find_f_5_l529_529858

-- Definitions from conditions
def f (x : ℝ) (a b : ℝ) : ℝ := a * x ^ 3 - b * x + 2

-- Stating the theorem
theorem find_f_5 (a b : ℝ) (h : f (-5) a b = 17) : f 5 a b = -13 :=
by
  sorry

end find_f_5_l529_529858


namespace circles_tangent_at_E_l529_529252

-- Define the triangle ABC, its circumcircle Gamma, the incenter I, and the other circle omega
variables (A B C I E : Point)
variables (Gamma omega : Circle)

-- Conditions:
-- 1. ABC is a triangle
axiom triangle_ABC : Triangle A B C
-- 2. Gamma is the circumcircle of triangle ABC
axiom circumcircle_ABC : ∀ (P : Point), P ∈ Gamma ↔ P ∈ {A, B, C} ∨ Angle A P C = 180
-- 3. I is the incenter of the triangle ABC
axiom incenter_ABC : Incenter I A B C
-- 4. omega is tangent to the line AI at I
axiom tangent_A_I : Tangent omega (Line A I) I
-- 5. omega is tangent to the side BC at E
axiom tangent_BC_E : Tangent omega (Line B C) E

-- Given the above conditions, prove that the circles Gamma and omega are tangent at point E
theorem circles_tangent_at_E : Tangent Gamma omega E := sorry

end circles_tangent_at_E_l529_529252


namespace angle_between_a_b_l529_529962

variable {V : Type*} [inner_product_space ℝ V]

def is_unit_vector {V : Type*} [inner_product_space ℝ V] (v : V) : Prop :=
  ⟪v, v⟫ = 1

noncomputable def angle_between_vectors (a b : V) [inner_product_space ℝ V] : ℝ :=
  real.arccos (⟪a, b⟫ / (∥a∥ * ∥b∥))

theorem angle_between_a_b
  (a b : V)
  (h_a : is_unit_vector a)
  (h_b : is_unit_vector b)
  (orth : ⟪a + 3 • b, 4 • a - 3 • b⟫ = 0) :
  angle_between_vectors a b = real.arccos (5 / 9) :=
sorry

end angle_between_a_b_l529_529962


namespace original_number_in_magician_game_l529_529599

theorem original_number_in_magician_game (a b c : ℕ) (habc : 100 * a + 10 * b + c = 332) (N : ℕ) (hN : N = 4332) :
    222 * (a + b + c) = 4332 → 100 * a + 10 * b + c = 332 :=
by 
  sorry

end original_number_in_magician_game_l529_529599


namespace joe_pockets_balance_l529_529613

theorem joe_pockets_balance :
  ∀ (total lr one_fourth lr_after_first lr_after_second rr),
  total = 200 →
  lr = 160 →
  one_fourth = lr / 4 →
  lr_after_first = lr - one_fourth →
  rr = total - lr →
  lr_after_second = lr_after_first - 20 →
  rr + one_fourth + 20 = total - lr + one_fourth + 20 →
  lr_after_second = rr + one_fourth + 20 - total + lr :=
begin
  intros total lr one_fourth lr_after_first lr_after_second rr,
  assume ht : total = 200,
  assume hl : lr = 160,
  assume ho : one_fourth = lr / 4,
  assume hlf : lr_after_first = lr - one_fourth,
  assume hr : rr = total - lr,
  assume hlfs : lr_after_second = lr_after_first - 20,
  assume hr_plus : rr + one_fourth + 20 = total - lr + one_fourth + 20,
  rw [ht, hl, ho] at *,
  sorry
end

end joe_pockets_balance_l529_529613


namespace union_of_subsets_l529_529569

open Set

variable (A B : Set ℕ)

theorem union_of_subsets (m : ℕ) (hA : A = {1, 3}) (hB : B = {1, 2, m}) (hSubset : A ⊆ B) :
    A ∪ B = {1, 2, 3} :=
  sorry

end union_of_subsets_l529_529569


namespace water_flow_l529_529005

theorem water_flow
  (depth : ℝ := 5) 
  (width : ℝ := 35) 
  (flow_rate_kmph : ℝ := 2) :
  let flow_rate_m_per_min := flow_rate_kmph * 1000 / 60 in
  width * depth * flow_rate_m_per_min = 5832.75 :=
by
  unfold flow_rate_m_per_min
  sorry

end water_flow_l529_529005


namespace subdivide_rectangle_l529_529853

theorem subdivide_rectangle (n : ℤ) (h₁ : n > 1) :
  (∃ r : ℝ, (r > 1) ∧ (∀ k : ℕ, k = n → exists_rectangle k r)) → n ≥ 3 :=
by
  sorry

end subdivide_rectangle_l529_529853


namespace count_positive_integers_l529_529332

def star (a b : ℤ) : ℤ := a^2 / b

theorem count_positive_integers (count_x : ℤ) :
  (∃ count_x, (∀ x, 15 ^ 2 = n * x → (15 ^ 2) / x > 0)) → 
  count_x = 9 :=
sorry

end count_positive_integers_l529_529332


namespace value_of_y_l529_529201

variable (x y : ℤ)

-- Define the conditions
def condition1 : Prop := 3 * (x^2 + x + 1) = y - 6
def condition2 : Prop := x = -3

-- Theorem to prove
theorem value_of_y (h1 : condition1 x y) (h2 : condition2 x) : y = 27 := by
  sorry

end value_of_y_l529_529201


namespace average_income_l529_529744

theorem average_income (income1 income2 income3 income4 income5 : ℝ)
    (h1 : income1 = 600) (h2 : income2 = 250) (h3 : income3 = 450) (h4 : income4 = 400) (h5 : income5 = 800) :
    (income1 + income2 + income3 + income4 + income5) / 5 = 500 := by
    sorry

end average_income_l529_529744


namespace find_n_l529_529969

theorem find_n (n : ℕ) (b : ℕ → ℝ)
  (h0 : b 0 = 40)
  (h1 : b 1 = 70)
  (h2 : b n = 0)
  (h3 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n - 1 → b (k + 1) = b (k - 1) - 2 / b k) :
  n = 1401 :=
sorry

end find_n_l529_529969


namespace heaviest_vs_lightest_diff_total_excess_weight_total_earnings_l529_529591

section BuddhaHand

def standard_weight : ℝ := 0.5
def weight_diffs : List ℝ := [0.1, 0, -0.05, -0.25, 0.15, 0.2, 0.05, -0.1]
def price_per_kg : ℝ := 42
def num_fruits : ℕ := 8

theorem heaviest_vs_lightest_diff :
  let max_diff := list.max weight_diffs
  let min_diff := list.min weight_diffs
  max_diff - min_diff = 0.45 :=
by sorry

theorem total_excess_weight :
  list.sum weight_diffs = 0.1 :=
by sorry

theorem total_earnings :
  let total_weight := (standard_weight * num_fruits) + (list.sum weight_diffs)
  total_weight * price_per_kg = 172.2 :=
by sorry

end BuddhaHand

end heaviest_vs_lightest_diff_total_excess_weight_total_earnings_l529_529591


namespace problem_statement_l529_529636

-- Definitions of types for lines and planes
universe u
constant Line : Type u
constant Plane : Type u

-- Predicates for parallelism and perpendicularity between lines and planes
constant parallel : Line → Line → Prop
constant perpendicular : Line → Line → Prop
constant lies_in : Line → Plane → Prop
constant parallel_plane : Plane → Plane → Prop

-- Given conditions
variables {l m n : Line} {α β : Plane}

-- Define the necessary conditions
axiom parallel_plane_cond : parallel_plane α β
axiom lies_in_cond : lies_in l α

-- Conjecture: if α is parallel to β and l lies in α, then l is parallel to β
theorem problem_statement : parallel l β :=
by {
  -- We can use "sorry" here to skip the proof.
  sorry,
}

end problem_statement_l529_529636


namespace smallest_n_with_divisors_l529_529481

noncomputable def τ (n : ℕ) : ℕ :=
  if n = 0 then 0 else
    n.divisors.card

theorem smallest_n_with_divisors (k : ℕ) :
  (k ∈ {2, 3, 4, 5, 6, 8, 10, 16, 36, 64, 100, 216, 1000, 1000000}) →
  ∃ N : ℕ, ∀ M : ℕ, τ M = k → N ≤ M :=
begin
  sorry
end

end smallest_n_with_divisors_l529_529481


namespace no_correct_calculation_l529_529458

theorem no_correct_calculation :
  ¬ ( (-|-3| = 3) ∨ ((a + b)^2 = a^2 + b^2) ∨ (a^3 * a^4 = a^12) ∨ (| -3^2 | = 3) ) :=
by
  sorry

end no_correct_calculation_l529_529458


namespace S_on_PQ_l529_529253

theorem S_on_PQ {A B C D S P Q : Type*} [Square A B C D]
  (intersect_AC_BD : S = midpoint A C ∧ S = midpoint B D)
  (circle_k : Circle k A C)
  (circle_k' : Circle k' B D)
  (intersect_k_k' : P ≠ Q ∧ P ∈ k ∧ P ∈ k' ∧ Q ∈ k ∧ Q ∈ k') :

  S ∈ line_through P Q :=
sorry

end S_on_PQ_l529_529253


namespace distinct_rational_numbers_l529_529480

theorem distinct_rational_numbers (m : ℚ) :
  abs m < 100 ∧ (∃ x : ℤ, 4 * x^2 + m * x + 15 = 0) → 
  ∃ n : ℕ, n = 48 :=
sorry

end distinct_rational_numbers_l529_529480


namespace walter_bus_time_l529_529359

/--
Walter wakes up at 6:30 a.m., leaves for the bus at 7:30 a.m., attends 7 classes that each last 45 minutes,
enjoys a 40-minute lunch, and spends 2.5 hours of additional time at school for activities.
He takes the bus home and arrives at 4:30 p.m.
Prove that Walter spends 35 minutes on the bus.
-/
theorem walter_bus_time : 
  let total_time_away := 9 * 60 -- in minutes
  let class_time := 7 * 45 -- in minutes
  let lunch_time := 40 -- in minutes
  let additional_school_time := 2.5 * 60 -- in minutes
  total_time_away - (class_time + lunch_time + additional_school_time) = 35 := 
by
  sorry

end walter_bus_time_l529_529359


namespace increasing_on_1_to_infinity_max_and_min_on_1_to_4_l529_529177

noncomputable def f (x : ℝ) : ℝ := x + (1 / x)

theorem increasing_on_1_to_infinity : ∀ (x1 x2 : ℝ), 1 ≤ x1 → x1 < x2 → (1 ≤ x2) → f x1 < f x2 := by
  sorry

theorem max_and_min_on_1_to_4 : 
  (∀ (x : ℝ), 1 ≤ x → x ≤ 4 → f x ≤ f 4) ∧ 
  (∀ (x : ℝ), 1 ≤ x → x ≤ 4 → f 1 ≤ f x) := by
  sorry

end increasing_on_1_to_infinity_max_and_min_on_1_to_4_l529_529177


namespace normal_symmetric_about_zero_l529_529272

noncomputable def X : ℝ → ℝ := sorry -- definition of standard normal distribution

def f (x : ℝ) : ℝ := P(X ≥ x) -- given condition

theorem normal_symmetric_about_zero (x : ℝ) (hx : 0 < x) : 
  f(-x) = 1 - f(x) :=
by
  sorry

end normal_symmetric_about_zero_l529_529272


namespace inequality_sum_abs_diff_l529_529755

theorem inequality_sum_abs_diff (p : ℕ) (a : fin p → ℝ) (M m : ℝ)
  (hM : M = finset.univ.sup a)
  (hm : m = finset.univ.inf a) :
  (p-1) * (M - m) ≤ ∑ i j, |a i - a j| ∧ ∑ i j, |a i - a j| ≤ p^2 * (M - m) / 4 :=
by
  sorry

end inequality_sum_abs_diff_l529_529755


namespace num_two_digit_integers_l529_529171

theorem num_two_digit_integers : 
  let digits := [1, 3, 5, 7, 9] in
  (∀ d1 d2, d1 ∈ digits → d2 ∈ digits → d1 ≠ d2 →
   (10 * d1 + d2) ∈ {n : ℕ | 10 ≤ n ∧ n < 100}) ∧
  (∀ n ∈ {n : ℕ | 10 ≤ n ∧ n < 100}, ∃ d1 d2, d1 ∈ digits ∧ d2 ∈ digits ∧ 
   d1 ≠ d2 ∧ n = 10 * d1 + d2) ∧
  ∥{n : ℕ | ∃ d1 d2, d1 ∈ digits ∧ d2 ∈ digits ∧ d1 ≠ d2 ∧ n = 10 * d1 + d2}∥ = 20 :=
begin
  let digits := [1, 3, 5, 7, 9],
  have h1 : ∀ d1 d2, d1 ∈ digits → d2 ∈ digits → d1 ≠ d2 → (10 * d1 + d2) ∈ {n : ℕ | 10 ≤ n ∧ n < 100}, 
  { sorry },
  have h2 : ∀ n ∈ {n : ℕ | 10 ≤ n ∧ n < 100}, ∃ d1 d2, d1 ∈ digits ∧ d2 ∈ digits ∧ d1 ≠ d2 ∧ n = 10 * d1 + d2, 
  { sorry },
  have h3 : ∥{n : ℕ | ∃ d1 d2, d1 ∈ digits ∧ d2 ∈ digits ∧ d1 ≠ d2 ∧ n = 10 * d1 + d2}∥ = 20,
  { sorry },
  exact ⟨h1, h2, h3⟩,
end

end num_two_digit_integers_l529_529171


namespace sin_cos_values_l529_529537

theorem sin_cos_values (α : ℝ) (h : sin α + 3 * cos α = 0) :
  (sin α = 3 * sqrt 10 / 10 ∧ cos α = - sqrt 10 / 10) ∨ 
  (sin α = -3 * sqrt 10 / 10 ∧ cos α = sqrt 10 / 10) :=
sorry

end sin_cos_values_l529_529537


namespace Jill_tax_on_clothing_l529_529658

theorem Jill_tax_on_clothing 
  (spent_clothing : ℝ) (spent_food : ℝ) (spent_other : ℝ) (total_spent : ℝ) (tax_clothing : ℝ) 
  (tax_other_rate : ℝ) (total_tax_rate : ℝ) 
  (h_clothing : spent_clothing = 0.5 * total_spent) 
  (h_food : spent_food = 0.2 * total_spent) 
  (h_other : spent_other = 0.3 * total_spent) 
  (h_other_tax : tax_other_rate = 0.1) 
  (h_total_tax : total_tax_rate = 0.055) 
  (h_total_spent : total_spent = 100):
  (tax_clothing * spent_clothing + tax_other_rate * spent_other) = total_tax_rate * total_spent → 
  tax_clothing = 0.05 :=
by
  sorry

end Jill_tax_on_clothing_l529_529658


namespace largest_x_satisfies_condition_l529_529088

theorem largest_x_satisfies_condition (x : ℝ) (h : (⌊x⌋ / x) = 7 / 8) : x ≤ 48 / 7 :=
sorry

end largest_x_satisfies_condition_l529_529088


namespace largest_x_l529_529117

def largest_x_with_condition_eq_7_over_8 (x : ℝ) : Prop :=
  ⌊x⌋ / x = 7 / 8

theorem largest_x (x : ℝ) (h : largest_x_with_condition_eq_7_over_8 x) :
  x = 48 / 7 :=
sorry

end largest_x_l529_529117


namespace find_angle_between_vectors_l529_529964

open Real EuclideanSpace

noncomputable def angle_between_unit_vectors {n : ℕ} (a b : ℝⁿ) := 
  real.cos⁻¹ ((inner a b) / (∥a∥ * ∥b∥))

theorem find_angle_between_vectors {n : ℕ} (a b : ℝⁿ) 
  (ha : ∥a∥ = 1) (hb : ∥b∥ = 1)
  (orthogonal : inner (a + 3 • b) (4 • a - 3 • b) = 0) : 
  angle_between_unit_vectors a b = real.cos⁻¹ (5 / 9) :=
  sorry

end find_angle_between_vectors_l529_529964


namespace find_value_l529_529532

def pos_num (x : ℝ) : Prop := 0 < x

def log_cond (a b : ℝ): Prop := 
  (log 2 a = log 5 b) ∧ (log 5 b = log 10 (a + b))

theorem find_value (a b : ℝ) (ha : pos_num a) (hb : pos_num b) (condition : log_cond a b) : 
  (1 / a) + (1 / b) = 1 :=
by
  -- this is where the proof would go
  sorry

end find_value_l529_529532


namespace compare_abc_l529_529757

variable (a b c : ℝ)

noncomputable def a_def : ℝ := Real.logBase 2 0.7
noncomputable def b_def : ℝ := (1 / 5) ^ (2 / 3)
noncomputable def c_def : ℝ := (1 / 2) ^ (-3)

theorem compare_abc (ha : a = a_def) (hb : b = b_def) (hc : c = c_def) : c > b > a := by
  sorry

end compare_abc_l529_529757


namespace centroid_iff_deltas_zero_l529_529333

variables {R : Type} [Real R]

structure Point (R : Type) :=
  (x: R)
  (y: R)

def line_passing_through (P: Point R) (w: R) (P_val: R) : Prop :=
  P.x * cos w + P.y * sin w - P_val = 0

def delta (p: Point R) (w: R) (P_val: R) : R :=
  p.x * cos w + p.y * sin w - P_val

theorem centroid_iff_deltas_zero {P0 P1 P2 P3 : Point R} {w : R} (P_val: R)
  (hP0: line_passing_through P0 w P_val) :
  (P0.x = (P1.x + P2.x + P3.x) / 3 ∧ P0.y = (P1.y + P2.y + P3.y) / 3) ↔
  (delta P1 w P_val + delta P2 w P_val + delta P3 w P_val = 0) :=
sorry

end centroid_iff_deltas_zero_l529_529333


namespace percent_non_politics_equal_25_l529_529938

def total_reporters : ℕ := 100

def percent_A : ℝ := 0.20
def percent_B : ℝ := 0.25
def percent_C : ℝ := 0.15

def cover_A := total_reporters * percent_A
def cover_B := total_reporters * percent_B
def cover_C := total_reporters * percent_C

def total_cover_local := cover_A + cover_B + cover_C

def percent_non_local_coverers : ℝ := 0.20
def percent_cover_politics := total_cover_local / (1 - percent_non_local_coverers)
def percent_non_politics := 1 - percent_cover_politics

theorem percent_non_politics_equal_25 :
  percent_non_politics * 100 = 25 :=
sorry

end percent_non_politics_equal_25_l529_529938


namespace composition_is_rotation_or_translation_l529_529298

open Real

/-- Define the rotation function. -/
noncomputable def rotation (center : Point ℝ) (angle : ℝ) (p : Point ℝ) : Point ℝ :=
sorry -- Implementation of rotation function is skipped

/-- Define the composition of two rotations. -/
noncomputable def composition_of_rotations (A B : Point ℝ) (α β : ℝ) : Point ℝ → Point ℝ :=
rotation B β ∘ rotation A α

/-- The main theorem stating the required properties. -/
theorem composition_is_rotation_or_translation (A B : Point ℝ) (α β : ℝ) :
  A ≠ B →
  (∃ O : Point ℝ, ∃ θ : ℝ, θ = α + β ∧ (θ % 360 ≠ 0 → composition_of_rotations A B α β = rotation O θ) ∧
  (θ % 360 = 0 → ∃ T : Point ℝ → Point ℝ, composition_of_rotations A B α β = T)) :=
sorry -- Proof omitted

end composition_is_rotation_or_translation_l529_529298


namespace area_of_triangle_ACE_l529_529926

theorem area_of_triangle_ACE (h : hexagon)
  (length_side : ∀ (p q : points_of_hexagon), distance p q = 6)
  (equilateral : ∀ (t : triangle), is_equilateral t) 
  : area (triangle_ACE (as_hexagon h)) = 27 * sqrt 3 :=
sorry

end area_of_triangle_ACE_l529_529926


namespace angles_with_same_terminal_side_l529_529022

def same_terminal_side (θ₁ θ₂ : ℝ) : Prop :=
  ∃ (k : ℤ), θ₁ = θ₂ + 2 * π * k

theorem angles_with_same_terminal_side (k : ℤ) :
  same_terminal_side ((2 * k + 1) * π) ((4 * k + 1) * π) ∧
  same_terminal_side ((2 * k + 1) * π) ((4 * k - 1) * π) := 
by
  sorry

end angles_with_same_terminal_side_l529_529022


namespace ratio_january_february_l529_529802

variable (F : ℕ)

def total_savings := 19 + F + 8 

theorem ratio_january_february (h : total_savings F = 46) : 19 / F = 1 := by
  sorry

end ratio_january_february_l529_529802


namespace center_of_mass_quarter_circle_correct_l529_529838

noncomputable def center_of_mass_quarter_circle (a k : ℝ) : ℝ × ℝ :=
  let sqr x := x * x in
  let δ x y := k * x * y in
  let moments_x := ∫ x in 0..a, δ x (sqrt (sqr a - sqr x)) * x
  let moments_y := ∫ x in 0..a, δ x (sqrt (sqr a - sqr x)) * (sqrt (sqr a - sqr x))
  let total_mass := ∫ x in 0..a, δ x (sqrt (sqr a - sqr x))
  let x_c := moments_x / total_mass
  let y_c := moments_y / total_mass
  (x_c, y_c)

theorem center_of_mass_quarter_circle_correct (a k : ℝ) :
  center_of_mass_quarter_circle a k = (2 / 3 * a, 2 / 3 * a) :=
sorry

end center_of_mass_quarter_circle_correct_l529_529838


namespace integral_sin_x_plus_x_l529_529036

open Real

theorem integral_sin_x_plus_x :
  ∫ x in -π/2..π/2, (sin x + x) = 0 := 
by
  sorry

end integral_sin_x_plus_x_l529_529036


namespace least_number_to_divisible_by_9_l529_529841

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem least_number_to_divisible_by_9 :
  let n := 228712 in
  sum_of_digits n = 22 →
  (∃ k, (n + k) % 9 = 0 ∧ ∀ m, (n + m) % 9 = 0 → k ≤ m) := by
  sorry

end least_number_to_divisible_by_9_l529_529841


namespace expected_value_of_bernoulli_l529_529270

noncomputable def P (p : ℝ) (k : ℕ) : ℝ :=
if k = 0 then (1 - p) else if k = 1 then p else 0

def bernoulli_distribution_condition (X : ℕ → Prop) (p : ℝ) : Prop :=
(∀ k, X k ↔ k = 0 ∨ k = 1) ∧ ∀ (k : ℕ), X k → P p k ∈ set.Ioo 0 1

theorem expected_value_of_bernoulli (X : ℕ → Prop) (p : ℝ) (h1 : 0 < p) (h2 : p < 1) (h3 : bernoulli_distribution_condition X p) :
  ∑ k in {0, 1}, P p k * k = p :=
sorry

end expected_value_of_bernoulli_l529_529270


namespace part_I_equality_condition_part_II_l529_529554

-- Lean statement for Part (I)
theorem part_I (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 5) : 2 * Real.sqrt x + Real.sqrt (5 - x) ≤ 5 :=
sorry

theorem equality_condition (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 5) :
  (2 * Real.sqrt x + Real.sqrt (5 - x) = 5) ↔ (x = 4) :=
sorry

-- Lean statement for Part (II)
theorem part_II (m : ℝ) :
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 5) → 2 * Real.sqrt x + Real.sqrt (5 - x) ≤ |m - 2|) →
  (m ≥ 7 ∨ m ≤ -3) :=
sorry

end part_I_equality_condition_part_II_l529_529554


namespace collinear_X_Y_I_M_l529_529402

/-- 
Given: 
1. A circle ω with center I is inscribed in a segment of a disk with chord AB.
2. M is the midpoint of the arc AB.
3. N is the midpoint of the complementary arc.
4. Tangents from N touch ω at points C and D.
5. The opposite sidelines AC and BD of quadrilateral ABCD meet at point X.
6. The diagonals of ABCD meet at point Y.
Prove: X, Y, I, and M are collinear.
-/
theorem collinear_X_Y_I_M 
  (ω : Circle) (I : Point) (A B M N C D X Y : Point)
  (h1 : center ω = I)
  (h2 : is_midpoint M (arc A B))
  (h3 : is_midpoint N (complementary_arc A B))
  (h4 : tangent_at N ω C)
  (h5 : tangent_at N ω D)
  (h6 : meet (line_through A C) (line_through B D) = X)
  (h7 : meet (line_through A B) (line_through C D) = Y) :
  collinear X Y I M := sorry

end collinear_X_Y_I_M_l529_529402


namespace slopes_product_constant_l529_529145

-- Definitions of constants and ellipse's parameters
def a := 2 * sqrt 2
def b := sqrt 6
def c := sqrt 2
def e := sqrt 3 / 2

-- Definition of ellipse C
def ellipse_C (x y : ℝ) : Prop := (x^2) / (a^2) + (y^2) / (b^2) = 1

-- Definitions of points M, N, and T
def point_T : ℝ × ℝ := (2 * sqrt 2, 0)

-- Slope definition
def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Main theorem
theorem slopes_product_constant (M N : ℝ × ℝ) (hM : M ∈ ellipse_C) (hN : N ∈ ellipse_C) (hT : True) :
  let k_MT := slope M point_T
  let k_NT := slope N point_T
  k_MT * k_NT = (3 + 2 * sqrt 2) / 4 :=
sorry

end slopes_product_constant_l529_529145


namespace work_days_together_l529_529743

-- Define the conditions
def work_rate_a : ℝ := 1 / 20
def work_rate_b : ℝ := 1 / 80.00000000000001

-- Prove that a and b together can finish the work in 16 days
theorem work_days_together : (work_rate_a + work_rate_b) * 16 = 1 :=
by
  sorry

end work_days_together_l529_529743


namespace find_container_height_l529_529799

noncomputable def container_height :=
  let r_A := 2    -- radius of container A
  let r_B := 3    -- radius of container B
  let h_B := (2 / 3 * x - 6) -- height of water in container B
  
  -- Volume of water in container A
  let V_A := Real.pi * r_A^2 * x

  -- Volume of water in container B after it is transferred
  let V_B := Real.pi * r_B^2 * h_B

  -- Equate the volumes and express the height of container A
  ∀ x : ℝ, V_A = V_B → x = 27

theorem find_container_height (x : ℝ) :
  let r_A := 2
  let r_B := 3
  let h_B := (2 / 3 * x - 6)
  let V_A := Real.pi * r_A^2 * x
  let V_B := Real.pi * r_B^2 * h_B
  (V_A = V_B) → x = 27 := by
  intros
  sorry

end find_container_height_l529_529799


namespace infinite_sets_of_eight_l529_529192

theorem infinite_sets_of_eight :
  ∃ (S : Set (Set ℝ)), 
    (∀ s ∈ S, s.card = 8 ∧ (∀ x ∈ s, ∃ y z ∈ s, x = y * z)) ∧ S.infinite :=
by
  sorry

end infinite_sets_of_eight_l529_529192


namespace max_n_sum_leq_2046_l529_529869

section
  -- Define the sequence {a_n}
  def a : ℕ → ℕ
  | 0     => 1  -- There's no a_0, but indexing from 1 is more convenient in Lean
  | 1     => 2
  | (n+2) => if n % 2 = 0 then 2 * a n + 1
             else (-1) ^ (n / 2) * n
  
  -- Define the sum of the first n terms of the sequence
  def S (n : ℕ) : ℕ :=
  (List.range n).map a |>.sum

  -- Problem statement
  theorem max_n_sum_leq_2046 :
    ∃ n : ℕ, S n ≤ 2046 ∧ ∀ m : ℕ, S m ≤ 2046 → m ≤ n :=
  sorry
end

end max_n_sum_leq_2046_l529_529869


namespace min_fraction_sum_l529_529160

theorem min_fraction_sum {a b : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 6) : 
  (∃ (min_val : ℝ), (min_val = (1/a + 2/b)) ∧ min_val = 4/3) :=
begin
  sorry
end

end min_fraction_sum_l529_529160


namespace correct_statements_count_l529_529850

-- Define the double factorial
def double_factorial : ℕ → ℕ
| 0 => 1
| 1 => 1
| n + 2 => (n + 2) * double_factorial n

-- Statements based on double factorial definition
def statement1 : Prop := (double_factorial 2010) * (double_factorial 2009) = Nat.factorial 2010
def statement2 : Prop := double_factorial 2010 = 2 * Nat.factorial 1005
def statement3 : Prop := (double_factorial 2010 % 10 = 0)
def statement4 : Prop := (double_factorial 2009 % 10 = 5)

-- Main theorem to prove
theorem correct_statements_count : (statement1 ∧ ¬statement2 ∧ statement3 ∧ statement4) ↔ 3 = 3 := by sorry

end correct_statements_count_l529_529850


namespace sum_of_factors_72_l529_529038

theorem sum_of_factors_72 : 
  let n := 72
  in let factors := 
    ([1, 2, 4, 8].map (λ x, x * 1) ++ [1, 2, 4, 8].map (λ x, x * 3) ++ [1, 2, 4, 8].map (λ x, x * 9))
    in n = 2^3 * 3^2 →
       (factors.sum) = 195 := 
by
  intros n factors h1
  sorry

end sum_of_factors_72_l529_529038


namespace PR_PS_eq_AE_l529_529627

-- Define the parallelogram ABCD with properties AB parallel to CD and BC parallel to AD
variables {A B C D P S R Q E : Type*}
variables [AffEuc_geometry A B C D P S R Q E]
variables (AB_parallel_CD : ∥ A - B ∥ = ∥ C - D ∥)
variables (BC_parallel_AD : ∥ B - C ∥ = ∥ A - D ∥)
variables (P_on_AB : A ∈ line_segment A B)
variables (PS_perp_CD : ∥ P - S ∥ = ∥ C - D ∥)
variables (PR_perp_BD : ∥ P - R ∥ = ∥ B - D ∥)
variables (E_foot_per_A : foot_of_perpendicular A D = E)
variables (PQ_perp_AE : ∥ P - Q ∥ = ∥ A - E ∥)

-- Prove that PR + PS = AE
theorem PR_PS_eq_AE : ∥ P - R ∥ + ∥ P - S ∥ = ∥ A - E ∥ :=
by
  sorry

end PR_PS_eq_AE_l529_529627


namespace length_of_wall_l529_529465

variables (boys1 boys2 : ℕ) (length1 length2 : ℝ) (days1 days2 : ℝ)

-- Define the conditions
def work_rate (boys : ℕ) (length : ℝ) (days : ℝ) : ℝ := length / days / boys.to_real

def condition1 := work_rate 6 60 5 = 12 / 6
def condition2 := 8 * (3.125) = days2

-- Main theorem to be proved
theorem length_of_wall : 
  condition1 ∧ condition2 →
  length2 = 50 :=
by
  sorry

end length_of_wall_l529_529465


namespace find_x_l529_529734

def x : ℕ := 70

theorem find_x :
  x + (5 * 12) / (180 / 3) = 71 :=
by
  sorry

end find_x_l529_529734


namespace vertical_asymptote_l529_529851

theorem vertical_asymptote (x : ℝ) : (y = (2*x - 3) / (4*x + 5)) → (4*x + 5 = 0) → x = -5/4 := 
by 
  intros h1 h2
  sorry

end vertical_asymptote_l529_529851


namespace area_triangle_MNP_l529_529494

-- Define the given conditions
def triangle_MNP_is_right (M N P : ℝ → ℝ → Prop) : Prop :=
  ∃ (x1 y1 x2 y2 x3 y3 : ℝ), M x1 y1 ∧ N x2 y2 ∧ P x3 y3 ∧
  (x1 = 0 ∧ y1 = 0) ∧
  (x2 = sqrt 3 ∧ y2 = 0) ∧
  (x3 = 0 ∧ y3 = 1) ∧
  ∠ MNP = 90 ∧
  angle_xm (0, 0) (sqrt 3, 0) (0, 1) = 60

-- Prove the area of triangle MNP
theorem area_triangle_MNP : ∀ (M N P : ℝ → ℝ → Prop), 
  triangle_MNP_is_right M N P → 
  area_of_triangle M N P = 28.125 * sqrt 3 :=
by
  sorry

-- Definitions of types and properties used
def area_of_triangle (A B C : ℝ → ℝ → Prop) : ℝ :=
  1 / 2 * (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

def angle_xm (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  Real.arccos ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) / 
    (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)))

end area_triangle_MNP_l529_529494


namespace base_2_representation_of_236_l529_529362

-- Define the base 2 representation function
def base2_repr (n : Nat) : List Nat :=
  if n = 0 then [] else base2_repr (n / 2) ++ [n % 2]

def asBase2String (digits: List Nat) : String :=
  String.concat (digits.map toString)

-- Theorem statement
theorem base_2_representation_of_236 : asBase2String (base2_repr 236) = "111010100" := 
by
  sorry

end base_2_representation_of_236_l529_529362


namespace odd_number_difference_of_squares_not_unique_l529_529953

theorem odd_number_difference_of_squares_not_unique :
  ∀ n : ℤ, Odd n → ∃ X Y X' Y' : ℤ, (n = X^2 - Y^2) ∧ (n = X'^2 - Y'^2) ∧ (X ≠ X' ∨ Y ≠ Y') :=
sorry

end odd_number_difference_of_squares_not_unique_l529_529953


namespace number_of_valid_x_l529_529330

-- Define the operation ⋆ as described in the problem
def star (a b : ℤ) : ℤ := (a ^ 2) / b

-- Define a function that checks if star(a, b) is a positive integer
def isPositiveInteger (a b : ℤ) : Prop := star a b > 0 ∧ ∃ k : ℤ, star a b = k

-- Define the main theorem stating that the number of integer values of x making 15 ⋆ x a positive integer is 9
theorem number_of_valid_x : ∃ n : ℕ, n = 9 ∧ ∀ x : ℤ, isPositiveInteger 15 x ↔ x ∣ 225 :=
by
  exists 9
  intros x
  constructor
  sorry  -- Proof required here

end number_of_valid_x_l529_529330


namespace probability_window_opens_correct_l529_529450

noncomputable def probability_window_opens_no_later_than_3_minutes_after_scientist_arrives 
  (arrival_times : Fin 6 → ℝ) : ℝ :=
  if (∀ i, arrival_times i ∈ Set.Icc 0 15) ∧ 
     (∀ i j, i ≠ j → arrival_times i < arrival_times j) ∧ 
     ((∃ i, arrival_times i ≥ 12)) then
    1 - (0.8 ^ 6)
  else
    0

theorem probability_window_opens_correct : 
  ∀ (arrival_times : Fin 6 → ℝ),
    (∀ i, arrival_times i ∈ Set.Icc 0 15) →
    (∀ i j, i ≠ j → arrival_times i < arrival_times j) →
    (∃ i, arrival_times i = arrival_times 5) →
    abs (probability_window_opens_no_later_than_3_minutes_after_scientist_arrives arrival_times - 0.738) < 0.001 :=
by
  sorry

end probability_window_opens_correct_l529_529450


namespace find_f_one_l529_529179

theorem find_f_one (f : ℝ → ℝ) (h : ∀ x : ℝ, f(x+1) - 1 = - (f(-x+1) - 1)) :
  f(1) = 1 :=
sorry

end find_f_one_l529_529179


namespace f_alpha_l529_529535

variables (α : Real) (x : Real)

noncomputable def f (x : Real) : Real := 
  (Real.cos (Real.pi + x) * Real.sin (2 * Real.pi - x)) / Real.cos (Real.pi - x)

lemma sin_alpha {α : Real} (hcos : Real.cos α = 1 / 3) (hα : 0 < α ∧ α < Real.pi) : 
  Real.sin α = 2 * Real.sqrt 2 / 3 :=
sorry

lemma tan_alpha {α : Real} (hsin : Real.sin α = 2 * Real.sqrt 2 / 3) (hcos : Real.cos α = 1 / 3) :
  Real.tan α = 2 * Real.sqrt 2 :=
sorry

theorem f_alpha {α : Real} (hcos : Real.cos α = 1 / 3) (hα : 0 < α ∧ α < Real.pi) :
  f α = -2 * Real.sqrt 2 / 3 :=
sorry

end f_alpha_l529_529535


namespace four_digit_numbers_div_by_5_l529_529573

theorem four_digit_numbers_div_by_5 : 
  let digits := {0, 1, 3, 5, 7}
  in let is_four_digit (n : ℤ) := n / 1000 ≥ 1 ∧ n / 1000 < 10
  in let divisible_by_5 (n : ℤ) := n % 5 = 0
  in let no_repeated_digits (n : ℤ) := 
         let ds := Int.digits 10 n in list.nodup ds
  in ∑ n in (finset.filter (λ n, is_four_digit n ∧ divisible_by_5 n ∧ no_repeated_digits n) (finset.range 10000)), 1 = 42
:= 
sorry

end four_digit_numbers_div_by_5_l529_529573


namespace calculate_expression_l529_529467

theorem calculate_expression :
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5) + (1 / 6)) = 57 :=
by
  sorry

end calculate_expression_l529_529467


namespace find_b_l529_529733

-- Define the lines and the condition of parallelism
def line1 := ∀ (x y b : ℝ), 4 * y + 8 * b = 16 * x
def line2 := ∀ (x y b : ℝ), y - 2 = (b - 3) * x
def are_parallel (m1 m2 : ℝ) := m1 = m2

-- Translate the problem to a Lean statement
theorem find_b (b : ℝ) : (∀ x y, 4 * y + 8 * b = 16 * x) → (∀ x y, y - 2 = (b - 3) * x) → b = 7 :=
by
  sorry

end find_b_l529_529733


namespace lowest_price_per_component_l529_529386

theorem lowest_price_per_component (cost_per_component shipping_per_component fixed_costs num_components : ℕ) 
  (h_cost_per_component : cost_per_component = 80)
  (h_shipping_per_component : shipping_per_component = 5)
  (h_fixed_costs : fixed_costs = 16500)
  (h_num_components : num_components = 150) :
  (cost_per_component + shipping_per_component) * num_components + fixed_costs = 29250 ∧
  29250 / 150 = 195 :=
by
  sorry

end lowest_price_per_component_l529_529386


namespace fixed_point_l529_529555

-- Define the function and conditions.
def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1) - 1

theorem fixed_point (a : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) : f a (-1) = 0 :=
by
  rw [f, pow_zero, sub_self]
  sorry

end fixed_point_l529_529555


namespace smaller_root_of_quadratic_l529_529820

theorem smaller_root_of_quadratic (p q r : ℝ) (h_seq : p ≥ q ∧ q ≥ r ∧ r ≥ 0)
  (h_arith_seq : q = p - d ∧ r = p - 2 * d)
  (h_roots : ∃ α : ℝ, (px : ℝ) quadratic_eq (p * x^2 + q * x + r = 0) ∧ quadratic_eq_roots α 2α) : 
  ∃ root : ℝ, root = -1/6 := 
sorry

end smaller_root_of_quadratic_l529_529820


namespace sin_inequality_l529_529254

theorem sin_inequality (d n : ℤ) (hd : d ≥ 1) (hnsq : ∀ k : ℤ, k * k ≠ d) (hn : n ≥ 1) :
  (n * Real.sqrt d + 1) * |Real.sin (n * Real.pi * Real.sqrt d)| ≥ 1 := by
  sorry

end sin_inequality_l529_529254


namespace parallelepiped_volume_half_l529_529923

noncomputable def volume_of_parallelepiped (a b : ℝ^3) : ℝ :=
  |a • ((b + 2 * (b × a)) × b)|

theorem parallelepiped_volume_half (a b : ℝ^3)
  (ha : ‖a‖ = 1)
  (hb : ‖b‖ = 1)
  (angle_ab : real.arccos (a • b / (‖a‖ * ‖b‖)) = π/4) : 
  volume_of_parallelepiped a b = 1/2 :=
sorry

end parallelepiped_volume_half_l529_529923


namespace hyperbola_property_l529_529000

/-- A hyperbola centered at (1,1) with a vertex at (3, 1)
    and passing through points (4, 2) and (t, 4). -/
theorem hyperbola_property (t : ℝ) (a square of : 4) (b square of : 4/5) : 
  ((t - 1)^2 = 49) -> (t = 8 or t = -6) :=
  sorry

end hyperbola_property_l529_529000


namespace triangles_and_areas_l529_529028

-- Definitions:
def triangle := ℕ -- Simplistic representation for the sake of the problem

variables (ABC : triangle)
variables (A B C D E F O : triangle)

-- Conditions:
axiom equilateral_ABC : (AB = BC) ∧ (BC = CA)
axiom midpoints : (D is_midpoint_of AB) ∧ (E is_midpoint_of BC) ∧ (F is_midpoint_of CA)
axiom medians_intersect : (AD intersects BE at O) ∧ (BE intersects CF at O) ∧ (CF intersects AD at O)

-- Theorem statement: There are 16 triangles in the figure and 4 distinct area values
theorem triangles_and_areas (equilateral_ABC : ABC.is_equilateral) 
                              (midpoints : (D is_midpoint_of AB) ∧ (E is_midpoint_of BC) ∧ (F is_midpoint_of CA)) 
                              (medians_intersect : (AD intersects BE at O) ∧ (BE intersects CF at O) ∧ (CF intersects AD at O)) : 
  ∃ count_areas : ℕ, count_areas = 16 ∧ ∃ distinct_areas : ℕ, distinct_areas = 4 :=
sorry

end triangles_and_areas_l529_529028


namespace size_of_each_serving_l529_529015

noncomputable def mix_ratio (concentrate water: ℕ) : ℕ :=
  concentrate + water

def cans_required : ℕ := 34
def servings : ℕ := 272
def can_volume : ℕ := 12
def total_volume : ℕ := cans_required * mix_ratio 1 3 * can_volume

theorem size_of_each_serving : (total_volume / servings) = 6 := by
  sorry

end size_of_each_serving_l529_529015


namespace heaviest_vs_lightest_diff_total_excess_weight_total_earnings_l529_529592

section BuddhaHand

def standard_weight : ℝ := 0.5
def weight_diffs : List ℝ := [0.1, 0, -0.05, -0.25, 0.15, 0.2, 0.05, -0.1]
def price_per_kg : ℝ := 42
def num_fruits : ℕ := 8

theorem heaviest_vs_lightest_diff :
  let max_diff := list.max weight_diffs
  let min_diff := list.min weight_diffs
  max_diff - min_diff = 0.45 :=
by sorry

theorem total_excess_weight :
  list.sum weight_diffs = 0.1 :=
by sorry

theorem total_earnings :
  let total_weight := (standard_weight * num_fruits) + (list.sum weight_diffs)
  total_weight * price_per_kg = 172.2 :=
by sorry

end BuddhaHand

end heaviest_vs_lightest_diff_total_excess_weight_total_earnings_l529_529592


namespace modular_arithmetic_l529_529264

theorem modular_arithmetic (b : ℕ) (h₁ : b ≡ (2⁻¹ + 4⁻¹ + 8⁻¹)⁻¹ * 3 [MOD 13]) : b % 13 = 9 :=
by {
  sorry
}

end modular_arithmetic_l529_529264


namespace greatest_whole_number_solution_l529_529070

theorem greatest_whole_number_solution :
  ∃ (x : ℕ), (5 * x - 4 < 3 - 2 * x) ∧ ∀ (y : ℕ), (5 * y - 4 < 3 - 2 * y) → y ≤ x ∧ x = 0 :=
by
  sorry

end greatest_whole_number_solution_l529_529070


namespace trigonometric_identity_l529_529652

theorem trigonometric_identity (α : ℝ) : 
  (sin 30)^2 + (cos 60)^2 + (sin 30) * (cos 60) = 3 / 4 ∧ 
  (sin 10)^2 + (cos 40)^2 + (sin 10) * (cos 40) = 3 / 4 ∧
  (sin 6)^2 + (cos 36)^2 + (sin 6) * (cos 36) = 3 / 4 → 
  (sin α)^2 + (cos (30 + α))^2 + (sin α) * (cos (30 + α)) = 3 / 4 := by
  sorry

end trigonometric_identity_l529_529652


namespace unit_vector_opposite_AB_is_l529_529147

open Real

noncomputable def unit_vector_opposite_dir (A B : ℝ × ℝ) : ℝ × ℝ :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BA := (-AB.1, -AB.2)
  let mag_BA := sqrt (BA.1^2 + BA.2^2)
  (BA.1 / mag_BA, BA.2 / mag_BA)

theorem unit_vector_opposite_AB_is (A B : ℝ × ℝ) (hA : A = (1, 2)) (hB : B = (-2, 6)) :
  unit_vector_opposite_dir A B = (3/5, -4/5) :=
by
  sorry

end unit_vector_opposite_AB_is_l529_529147


namespace area_addition_of_subtriangles_l529_529756

variables {A B C D O K : Point}
variables {S : Triangle → ℝ}

def trapezoid (A B C D : Point) : Prop :=
  side_parallel A B D C ∧ side_parallel A D B C

def symmetric_relative_to (B O : Point) : Point :=
  O + (O - B)

def intersects (ℓ₁ ℓ₂ : Line) (P : Point) : Prop :=
  (P ∈ ℓ₁) ∧ (P ∈ ℓ₂)

def line_through (P Q : Point) : Line :=
  {x // ∃ α, x = P + α • (Q - P)}

def line_passes_through_symmetric_and_intersects_base (C B O A D : Point) (K : Point) : Prop :=
  ∃ E : Point, E = symmetric_relative_to B O ∧ intersects (line_through C E) (line_through A D) K

theorem area_addition_of_subtriangles
  (h_trap : trapezoid A B C D)
  (h_diags : intersects (line_through A C) (line_through B D) O)
  (h_line : line_passes_through_symmetric_and_intersects_base C B O A D K) :
  S (triangle_of A O K) = S (triangle_of A O B) + S (triangle_of D O K) :=
sorry

end area_addition_of_subtriangles_l529_529756


namespace gcd_min_value_l529_529581

theorem gcd_min_value (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : Nat.gcd a b = 18) :
  Nat.gcd (12 * a) (20 * b) = 72 :=
by
  sorry

end gcd_min_value_l529_529581


namespace tan_pi_plus_alpha_l529_529134

noncomputable def given_condition (α : ℝ) : Prop :=
  sin α = -2 / 3 ∧ (π < α ∧ α < 3 * π / 2)

theorem tan_pi_plus_alpha (α : ℝ) (h : given_condition α) : 
  tan (π + α) = 2 * real.sqrt 5 / 5 :=
by
  sorry

end tan_pi_plus_alpha_l529_529134


namespace calculate_F_l529_529135

def f(a : ℝ) : ℝ := a^2 - 5 * a + 6
def F(a b c : ℝ) : ℝ := b^2 + a * c + 1

theorem calculate_F : F 3 (f 3) (f 5) = 19 :=
by
  sorry

end calculate_F_l529_529135


namespace martha_jackets_l529_529281

theorem martha_jackets (j : ℕ) (t_shirts_bought := 9) (total_clothes := 18)
    (free_jacket_cond : ∀ j, j / 2 = jnat_div j 2)
    (free_tshirt_cond : ∀ t, t / 3 = tnat_div t 3) : j = 4 :=
by
  have total_jackets := j + j / 2
  have total_tshirts := t_shirts_bought + t_shirts_bought / 3
  have total_clothes_eq := total_jackets + total_tshirts
  have total_eq := total_clothes_eq = total_clothes
  sorry

end martha_jackets_l529_529281


namespace circumscribed_circle_circle_with_center_on_line_tangent_and_chord_circle_l529_529840

-- Problem 1: Circle circumscribing around the triangle
theorem circumscribed_circle (x y : ℝ) :
  (x - 1)^2 + (y + 3)^2 = 25 ↔
  (x, y) = (4, 1) ∨ (x, y) = (6, -3) ∨ (x, y) = (-3, 0) := sorry

-- Problem 2: Circle passing through two points with center on given line
theorem circle_with_center_on_line (x y : ℝ) :
  (x - 2)^2 + (y - 1)^2 = 100 ↔
  (x = 5 ∧ y = 2) ∨ (x = 3 ∧ y = -2) ∧
  ∃ a: ℝ, (2* a - y = 3) := sorry

-- Problem 3: Circle tangent to x-axis with specified chord length and center on a line
theorem tangent_and_chord_circle (x y : ℝ) :
  ((x - 1)^2 + (y - 3)^2 = 9 ∨ (x + 1)^2 + (y + 3)^2 = 9) ↔
  (∃ a: ℝ, (x = a ∧ y = 3 * a) ∧ r = abs (3 * a) ∧ 2 * sqrt(7 * a^2) = 2 * sqrt(7)) := sorry

end circumscribed_circle_circle_with_center_on_line_tangent_and_chord_circle_l529_529840


namespace horse_running_time_l529_529311

def area_of_square_field : Real := 625
def speed_of_horse_around_field : Real := 25

theorem horse_running_time : (4 : Real) = 
  let side_length := Real.sqrt area_of_square_field
  let perimeter := 4 * side_length
  perimeter / speed_of_horse_around_field :=
by
  sorry

end horse_running_time_l529_529311


namespace speed_of_train_is_correct_l529_529795

noncomputable def speedOfTrain := 
  let lengthOfTrain : ℝ := 800 -- length of the train in meters
  let timeToCrossMan : ℝ := 47.99616030717543 -- time in seconds to cross the man
  let speedOfMan : ℝ := 5 * (1000 / 3600) -- speed of the man in m/s (conversion from km/hr to m/s)
  let relativeSpeed : ℝ := lengthOfTrain / timeToCrossMan -- relative speed of the train
  let speedOfTrainInMS : ℝ := relativeSpeed + speedOfMan -- speed of the train in m/s
  let speedOfTrainInKMHR : ℝ := speedOfTrainInMS * (3600 / 1000) -- speed in km/hr
  64.9848 -- result is approximately 64.9848 km/hr

theorem speed_of_train_is_correct :
  speedOfTrain = 64.9848 :=
by
  sorry

end speed_of_train_is_correct_l529_529795


namespace negation_of_universal_proposition_l529_529699

theorem negation_of_universal_proposition (f : ℕ+ → ℕ+) :
  (¬ ∀ n : ℕ+, f n ∈ ℕ+ ∧ f n ≤ n) ↔ (∃ n : ℕ+, f n ∉ ℕ+ ∨ f n > n) :=
by
  sorry

end negation_of_universal_proposition_l529_529699


namespace length_of_AC_is_4_l529_529594

-- Define the conditions

variable (A B C D F : Type) -- Points in the plane
variable (d_AC d_BD d_DC d_FC : ℝ) -- Distances between points

-- Given conditions
axiom D_on_AC : D ∈ open_segment A C
axiom F_on_BC : F ∈ open_segment B C
axiom BD_eq_DC : d_BD = 2 ∧ d_DC = 2
axiom FC_eq_2 : d_FC = 2
axiom AB_perp_AC : is_perpendicular (line_through A B) (line_through A C)
axiom AF_perp_BC : is_perpendicular (line_through A F) (line_through B C)

-- The theorem to prove that the length of AC is 4
theorem length_of_AC_is_4 : d_AC = 4 := 
sorry

#check length_of_AC_is_4

end length_of_AC_is_4_l529_529594


namespace quadratic_always_has_real_roots_find_m_given_difference_between_roots_is_three_l529_529867

-- Part 1: Prove that the quadratic equation always has two real roots for all real m.
theorem quadratic_always_has_real_roots (m : ℝ) :
  let Δ := (m-1)^2 - 4 * (m-2) in 
  Δ ≥ 0 := 
by 
  have Δ := (m-1)^2 - 4 * (m-2);
  suffices Δ_nonneg : (m-3)^2 ≥ 0, from Δ_nonneg;
  sorry

-- Part 2: Given the difference between the roots is 3, find the value of m.
theorem find_m_given_difference_between_roots_is_three (m : ℝ) :
  let x1 := 1,
      x2 := m - 2,
      diff := |x1 - x2| in
  diff = 3 → m = 0 ∨ m = 6 := 
by 
  let x1 := 1,
      x2 := m - 2,
      diff := |x1 - x2|;
  assume h : diff = 3;
  have abs_eq_three : |3 - m| = 3 := by {
      calc |1 - (m - 2)| = ... := sorry
  };
  cases abs_eq_three with
  | inl h₁ => have m_eq_0 : m = 0 := sorry;
              exact Or.inl m_eq_0
  | inr h₂ => have m_eq_6 : m = 6 := sorry;
              exact Or.inr m_eq_6

end quadratic_always_has_real_roots_find_m_given_difference_between_roots_is_three_l529_529867


namespace exists_point_with_sum_distances_ge_1983_l529_529287

open Real

noncomputable theory

def sum_of_distances (points : list (ℝ × ℝ)) (N : ℝ × ℝ) := 
  points.foldr (λ P acc, acc + dist P N) 0

theorem exists_point_with_sum_distances_ge_1983
    (M : list (ℝ × ℝ)) 
    (hM : M.length = 1983) 
    (C : ℝ × ℝ) 
    (r : ℝ)
    (hr : r = 1) :
    ∃ N ∈ (circle C r), sum_of_distances M N ≥ 1983 :=
sorry

end exists_point_with_sum_distances_ge_1983_l529_529287


namespace xiao_ding_choices_l529_529602

theorem xiao_ding_choices : 
  (∑ i in ((finset.range 4).filter (λ n, 2 <= n)), (nat.choose 3 i) * (nat.choose 3 (3 - i))) = 10 :=
by
  sorry

end xiao_ding_choices_l529_529602


namespace solve_g_eq_5_l529_529646

noncomputable def g (x : ℝ) : ℝ :=
if x < 0 then 4 * x + 8 else 3 * x - 15

theorem solve_g_eq_5 : {x : ℝ | g x = 5} = {-3/4, 20/3} :=
by
  sorry

end solve_g_eq_5_l529_529646


namespace train_length_108_3_meters_l529_529011

noncomputable def length_of_train 
  (speed_train : ℝ) (speed_car : ℝ) (crossing_time : ℝ) : ℝ :=
  let relative_speed := (speed_train - speed_car) * (1000 / 3600) in
  relative_speed * crossing_time

theorem train_length_108_3_meters : 
  length_of_train 56 30 15 = 108.3 := 
by
  unfold length_of_train
  sorry

end train_length_108_3_meters_l529_529011


namespace probability_product_zero_l529_529726

def finite_set : set ℤ := {-3, -2, -1, 0, 2, 4, 5, 7}

def total_combinations (s : set ℤ) : ℕ := 
  nat.choose (set.finite_to_finset (set.finite_univ s).card) 2

def favorable_combinations_product_zero (s : set ℤ) : ℕ := 
  (set.size (s \ {0})) -- size of the set without 0

theorem probability_product_zero : 
  let s := finite_set in
  total_combinations s = 28 → 
  favorable_combinations_product_zero s = 7 → 
  (7 : ℚ) / 28 = 1 / 4 :=
by {
  intros s h_total_combinations h_favorable_combinations,
  -- Total number of combinations is 28
  have h1 : s.finite := finite_univ s,
  have h2 : s.card = 8 := h1.card,
  have h3 : total_combinations s = nat.choose 8 2 := rfl,
  rw h_total_combinations at h3,
  -- Number of favorable combinations that result in a product of zero is 7
  have h4 : favorable_combinations_product_zero s = 7 := 
    by 
    { 
      have : (s \ {0}).finite := finite_of (s \ {0}),
      have : (s \ {0}).card = 7 := h_favorable_combinations,
      rw this,
    },
  -- Calculating the probability
  have h5 : (7 : ℚ) / 28 = 1 / 4 := by norm_num,
  exact h5,
}


end probability_product_zero_l529_529726


namespace find_number_l529_529189

theorem find_number (x : ℝ) (h : (1/2) * x + 7 = 17) : x = 20 :=
sorry

end find_number_l529_529189


namespace directrix_of_parabola_l529_529905

theorem directrix_of_parabola (p : ℝ) (hp : 0 < p) (h_point : ∃ (x y : ℝ), y^2 = 2 * p * x ∧ (x = 2 ∧ y = 2)) :
  x = -1/2 :=
sorry

end directrix_of_parabola_l529_529905


namespace P_eq_Q_at_neg_one_l529_529568

def P (x : ℝ) : ℝ := 3 * x^3 - 2 * x + 1

def Q (x : ℝ) : ℝ := 0.5 * x^3 + 0.5 * x^2 + 0.5 * x + 0.5

theorem P_eq_Q_at_neg_one : P (-1) = Q (-1) :=
by
  calc
    P (-1) = 3 * (-1)^3 - 2 * (-1) + 1 : rfl
         ... = -3 + 2 + 1 : by norm_num
         ... = 0 : by norm_num
    Q (-1) = 0.5 * (-1)^3 + 0.5 * (-1)^2 + 0.5 * (-1) + 0.5 : by norm_num
         ... = -0.5 + 0.5 - 0.5 + 0.5 : by norm_num
         ... = 0 : by norm_num
  sorry

end P_eq_Q_at_neg_one_l529_529568


namespace more_no_real_roots_l529_529054

theorem more_no_real_roots :
  let S1 := {⟨p, q⟩ | (1 ≤ p ∧ p ≤ 1997) ∧ (1 ≤ q ∧ q ≤ 1997) ∧ (∃ r s : ℤ, r + s = -p ∧ r * s = q)}
  let S2 := {⟨p, q⟩ | (1 ≤ p ∧ p ≤ 1997) ∧ (1 ≤ q ∧ q ≤ 1997) ∧ (p^2 < 4 * q)}
  S2.card > S1.card :=
sorry

end more_no_real_roots_l529_529054


namespace mode_of_data_set_is_9_l529_529870

open Nat

def data_set (x : ℕ) : List ℕ := [0, 3, 5, x, 9, 13]

def median_condition (x : ℕ) : Prop := (5 + x) / 2 = 7

theorem mode_of_data_set_is_9 (x : ℕ) (h : median_condition x) : list.mode (data_set x) = some 9 :=
by
  sorry

end mode_of_data_set_is_9_l529_529870


namespace common_difference_is_1_l529_529549

def arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ d a1, ∀ n, a n = a1 + n * d

def nth_term (a : ℕ → ℤ) (n : ℕ) :=
  a n

def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) :=
  ∑ i in finset.range n, a (i + 1)

theorem common_difference_is_1 (a : ℕ → ℤ) :
  arithmetic_sequence a →
  nth_term a 10 = 10 →
  sum_first_n_terms a 10 = 55 →
  ∃ d, d = 1 :=
  begin
    sorry
  end

end common_difference_is_1_l529_529549


namespace geometric_series_sum_l529_529045

theorem geometric_series_sum :
  let a := 2
  let r := 3
  let n := 6
  S = a * (r ^ n - 1) / (r - 1) → S = 728 :=
by
  intros a r n h
  sorry

end geometric_series_sum_l529_529045


namespace average_age_of_remaining_people_l529_529312

theorem average_age_of_remaining_people 
  (initial_average_age : ℕ) 
  (num_initial_people : ℕ) 
  (leaving_age : ℕ) 
  (num_remaining_people : ℕ) 
  (h1 : initial_average_age = 35) 
  (h2 : num_initial_people = 8) 
  (h3 : leaving_age = 25) 
  (h4 : num_remaining_people = 7) : 
  initial_average_age * num_initial_people - leaving_age / num_remaining_people = 36.42857 :=
by
  sorry

end average_age_of_remaining_people_l529_529312


namespace simplest_square_root_is_sqrt_5_l529_529737

def is_simplest_square_root (x : ℝ) : Prop :=
  ∀ y : ℝ, y = \(\sqrt{4}\) ∨ y = \(\sqrt{8}\) ∨ y = \(\sqrt{\frac{5}{2}}\) → x ≤ y

theorem simplest_square_root_is_sqrt_5 :
  is_simplest_square_root (\(\sqrt{5}\)) :=
sorry

end simplest_square_root_is_sqrt_5_l529_529737


namespace KarleeRemainingFruits_l529_529619

def KarleeInitialGrapes : ℕ := 100
def StrawberryRatio : ℚ := 3 / 5
def PortionGivenToFriends : ℚ := 1 / 5

theorem KarleeRemainingFruits :
  let initialGrapes := KarleeInitialGrapes
  let initialStrawberries := (StrawberryRatio * initialGrapes).to_nat
  let grapesGivenPerFriend := (PortionGivenToFriends * initialGrapes).to_nat
  let totalGrapesGiven := 2 * grapesGivenPerFriend
  let remainingGrapes := initialGrapes - totalGrapesGiven
  let strawberriesGivenPerFriend := (PortionGivenToFriends * initialStrawberries).to_nat
  let totalStrawberriesGiven := 2 * strawberriesGivenPerFriend
  let remainingStrawberries := initialStrawberries - totalStrawberriesGiven
  let totalRemainingFruits := remainingGrapes + remainingStrawberries
  totalRemainingFruits = 96 :=
by
  sorry

end KarleeRemainingFruits_l529_529619


namespace yolanda_avg_three_point_baskets_l529_529381

noncomputable theory

def yolanda_points_season : ℕ := 345
def total_games : ℕ := 15
def free_throws_per_game : ℕ := 4
def two_point_baskets_per_game : ℕ := 5

theorem yolanda_avg_three_point_baskets :
  (345 - (15 * (4 * 1 + 5 * 2))) / 3 / 15 = 3 :=
by sorry

end yolanda_avg_three_point_baskets_l529_529381


namespace problem_statement_l529_529361

theorem problem_statement : (29.7 + 83.45) - 0.3 = 112.85 := sorry

end problem_statement_l529_529361


namespace solution_proof_l529_529430

noncomputable def problem_statement : Prop :=
  let x := [x1, x2, x3, x4, x5, x6] in
  let B := (∀ i < 5, x[i] < x[5]) in
  let A := (x[5] ≥ 12) in
  ∃ x1 x2 x3 x4 x5 x6 : ℝ,
    (0 ≤ x1 ∧ x1 ≤ 15) ∧ (0 ≤ x2 ∧ x2 ≤ 15) ∧ (0 ≤ x3 ∧ x3 ≤ 15) ∧
    (0 ≤ x4 ∧ x4 ≤ 15) ∧ (0 ≤ x5 ∧ x5 ≤ 15) ∧ (0 ≤ x6 ∧ x6 ≤ 15) ∧
    B ∧ (classical.some ((measure_theory.measure_space.measure (λ x, x < 12 <= x) B).to_real) = 0.738)

theorem solution_proof : problem_statement := sorry

end solution_proof_l529_529430


namespace max_abs_sum_l529_529255

theorem max_abs_sum (n : ℕ) (h : 0 < n) :
  ∃ S : ℝ, S = (⌊n / 2⌋ : ℝ) * (⌈n / 2⌉ : ℝ) :=
by
  sorry

end max_abs_sum_l529_529255


namespace carrots_thrown_out_l529_529463

variable (x : ℕ)

theorem carrots_thrown_out :
  let initial_carrots := 23
  let picked_later := 47
  let total_carrots := 60
  initial_carrots - x + picked_later = total_carrots → x = 10 :=
by
  intros
  sorry

end carrots_thrown_out_l529_529463


namespace equation_of_line_l529_529495

-- Define the points P and Q
def P : (ℝ × ℝ) := (3, 2)
def Q : (ℝ × ℝ) := (4, 7)

-- Prove that the equation of the line passing through points P and Q is 5x - y - 13 = 0
theorem equation_of_line : ∃ (A B C : ℝ), A = 5 ∧ B = -1 ∧ C = -13 ∧
  ∀ x y : ℝ, (y - 2) / (7 - 2) = (x - 3) / (4 - 3) → 5 * x - y - 13 = 0 :=
by
  sorry

end equation_of_line_l529_529495


namespace max_value_equals_one_l529_529258

noncomputable def max_value_complex_expression (α β : ℂ) : ℝ :=
  |(β - α) / (1 - conj(α) * β)|

theorem max_value_equals_one (α β : ℂ)
  (h1 : |α| = 2)
  (h2 : |β| = 1)
  (h3 : conj(α) * β ≠ 1) :
  max_value_complex_expression α β = 1 :=
sorry

end max_value_equals_one_l529_529258


namespace Buddha_hand_fruits_problems_l529_529589

/-- Given 8 Buddha's hand fruits with their weight deviations from the standard weight of 0.5 kg
    recorded as: 0.1, 0, -0.05, -0.25, 0.15, 0.2, 0.05, -0.1 (kg). Prove the following:

    1. The heaviest fruit is 0.45 kg heavier than the lightest fruit.
    2. The total deviation of these fruits' weight from the standard sum is 0.1 kg.
    3. If the selling price is ¥42 per kg, the farmer's earnings from selling these fruits are ¥172.2.
-/
theorem Buddha_hand_fruits_problems :
  let deviations := [0.1, 0, -0.05, -0.25, 0.15, 0.2, 0.05, -0.1] in
  let standard_weight := 0.5 in
  let price_per_kg := 42 in
  let heaviest := list.maximum deviations in
  let lightest := list.minimum deviations in
  heaviest - lightest = 0.45 ∧
  list.sum deviations = 0.1 ∧
  ((8 * standard_weight) + (list.sum deviations)) * price_per_kg = 172.2 :=
by
  sorry

end Buddha_hand_fruits_problems_l529_529589


namespace vertical_asymptote_product_l529_529697

theorem vertical_asymptote_product :
  (let p q : Rat := 
    let ⟨p, hp⟩ := exists_root_of_splits ⟨by apply_instance, splitting_field.exists_eq_root_of_splits (polynomial.X ^ 2 + C 11 * X + C 4) (polynomial.splits_of_is_scalar_tower_nz (1 : ℝ))⟩ in
    let ⟨q, hq⟩ := exists_root_of_splits ⟨by apply_instance, splitting_field.exists_eq_root_of_splits (polynomial.X ^ 2 + C 11 * X + C 4) (polynomial.splits_of_is_scalar_tower_nz (1 : ℝ))⟩ in
    p * q
  ) = 2/3 :=
by
  sorry

end vertical_asymptote_product_l529_529697


namespace ratio_S5_a5_l529_529167

noncomputable def sequence (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, 3 * S n - 6 = 2 * a n

noncomputable def specific_seq (a : ℕ → ℤ) : Prop :=
  ∀ n, a n = if n = 0 then 6 else -2 * a (n - 1)

theorem ratio_S5_a5 (a : ℕ → ℤ) (S : ℕ → ℤ)
  (hseq : sequence a S)
  (ha_initial : specific_seq a) :
  S 5 / a 5 = 11 / 16 := sorry

end ratio_S5_a5_l529_529167


namespace inclination_line_eq_l529_529163

theorem inclination_line_eq (l : ℝ → ℝ) (h1 : ∃ x, l x = 2 ∧ ∃ y, l y = 2) (h2 : ∃ θ, θ = 135) :
  ∃ a b c, a = 1 ∧ b = 1 ∧ c = -4 ∧ ∀ x y, y = l x → a * x + b * y + c = 0 :=
by 
  sorry

end inclination_line_eq_l529_529163


namespace subset_0_X_l529_529133

variables {X : set ℝ}
def X_def : X = {x | x > -4} := sorry

theorem subset_0_X : {0} ⊆ X :=
by sorry

end subset_0_X_l529_529133


namespace sequence_relation_l529_529778

theorem sequence_relation (b : ℕ → ℚ) : 
  b 1 = 2 ∧ b 2 = 5 / 11 ∧ (∀ n ≥ 3, b n = b (n-2) * b (n-1) / (3 * b (n-2) - b (n-1)))
  ↔ b 2023 = 5 / 12137 :=
by sorry

end sequence_relation_l529_529778


namespace probability_is_correct_l529_529436

noncomputable def probability_cashier_opens_early : ℝ :=
  let x1 : ℝ := sorry
  let x2 : ℝ := sorry
  let x3 : ℝ := sorry
  let x4 : ℝ := sorry
  let x5 : ℝ := sorry
  let x6 : ℝ := sorry
  if (0 <= x1) ∧ (x1 <= 15) ∧
     (0 <= x2) ∧ (x2 <= 15) ∧
     (0 <= x3) ∧ (x3 <= 15) ∧
     (0 <= x4) ∧ (x4 <= 15) ∧
     (0 <= x5) ∧ (x5 <= 15) ∧
     (0 <= x6) ∧ (x6 <= 15) ∧
     (x1 < x6) ∧ (x2 < x6) ∧ (x3 < x6) ∧ (x4 < x6) ∧ (x5 < x6) then 
    let p_not_A : ℝ := (12 / 15) ^ 6
    1 - p_not_A
  else
    0

theorem probability_is_correct : probability_cashier_opens_early = 0.738 :=
by sorry

end probability_is_correct_l529_529436


namespace linda_loan_interest_difference_l529_529276

theorem linda_loan_interest_difference :
  let P : ℝ := 8000
  let r : ℝ := 0.10
  let t : ℕ := 3
  let n_monthly : ℕ := 12
  let n_annual : ℕ := 1
  let A_monthly : ℝ := P * (1 + r / (n_monthly : ℝ))^(n_monthly * t)
  let A_annual : ℝ := P * (1 + r)^t
  A_monthly - A_annual = 151.07 :=
by
  sorry

end linda_loan_interest_difference_l529_529276


namespace range_of_a_l529_529664

def discriminant (a : ℝ) : ℝ := (a - 1) ^ 2 - 4 * a ^ 2
def increasing_function (a x : ℝ) : ℝ := (2 * a ^ 2 - a) ^ x

def proposition_p (a : ℝ) : Prop := discriminant a < 0
def proposition_q (a : ℝ) : Prop := 2 * a ^ 2 - a > 1

theorem range_of_a (a : ℝ) :
  (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a) →
  (1 / 3 < a ∧ a ≤ 1) ∨ (-1 ≤ a ∧ a < -1 / 2) :=
by
  sorry

end range_of_a_l529_529664


namespace min_value_expression_l529_529268

theorem min_value_expression (a b : ℝ) (h : a > b) (h0 : b > 0) :
  ∃ m : ℝ, m = (a^2 + 1 / (a * b) + 1 / (a * (a - b))) ∧ m = 4 :=
sorry

end min_value_expression_l529_529268


namespace feed_corn_cost_l529_529420

theorem feed_corn_cost (
    (num_sheep num_cows : ℕ) 
    (sheep_grass_month cow_grass_month : ℕ) 
    (corn_cost : ℕ) 
    (corn_for_cow_per_month : ℕ) 
    (corn_for_sheep_per_month : ℕ) 
    (pasture_acres : ℕ) 
    (year_months : ℕ)
    (total_grass_consumed_monthly : ℕ :=
      num_sheep * sheep_grass_month + num_cows * cow_grass_month)
    (months_pasture_lasts : ℕ :=
      pasture_acres / total_grass_consumed_monthly)
    (months_need_feed : ℕ :=
      year_months - months_pasture_lasts)
    (corn_bags_monthly : ℕ :=
      num_cows * corn_for_cow_per_month + num_sheep / corn_for_sheep_per_month)
    (total_bags_needed : ℕ :=
      corn_bags_monthly * months_need_feed)
    (total_cost : ℕ :=
      total_bags_needed * corn_cost)
    (valid_conditions: num_sheep = 8 ∧ num_cows = 5 ∧ sheep_grass_month = 1 ∧ cow_grass_month = 2
                       ∧ corn_cost = 10 ∧ corn_for_cow_per_month = 1 
                       ∧ corn_for_sheep_per_month = 2 
                       ∧ pasture_acres = 144 ∧ year_months = 12)
  ) : total_cost = 360 :=
by
  sorry

end feed_corn_cost_l529_529420


namespace castor_chess_players_l529_529660

theorem castor_chess_players (total_players : ℕ) (never_lost_to_ai : ℕ)
  (h1 : total_players = 40) (h2 : never_lost_to_ai = total_players / 4) :
  (total_players - never_lost_to_ai) = 30 :=
by
  sorry

end castor_chess_players_l529_529660


namespace find_circumcircle_l529_529187

-- Definition of the points
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, Real.sqrt 3)
def C : ℝ × ℝ := (2, Real.sqrt 3)

-- Circle equation definition
noncomputable def circle_eq (x y D E F : ℝ) : ℝ := x^2 + y^2 + D * x + E * y + F

theorem find_circumcircle : 
  ∃ D E F : ℝ, circle_eq 1 0 D E F = 0 ∧
             circle_eq 0 (Real.sqrt 3) D E F = 0 ∧
             circle_eq 2 (Real.sqrt 3) D E F = 0 ∧ 
             (D = -2) ∧ 
             (E = -4 * Real.sqrt 3 / 3) ∧ 
             (F = 1) :=
by {
  let D := -2,
  let E := -4 * Real.sqrt 3 / 3,
  let F := 1,
  use [D, E, F],
  split,
  { show circle_eq 1 0 D E F = 0, sorry },
  split,
  { show circle_eq 0 (Real.sqrt 3) D E F = 0, sorry },
  split,
  { show circle_eq 2 (Real.sqrt 3) D E F = 0, sorry },
  repeat {split},
  { exact rfl },
  { exact rfl },
  { exact rfl }
}

end find_circumcircle_l529_529187


namespace acute_angle_sine_solution_l529_529157

theorem acute_angle_sine_solution (α : ℝ) (h1 : 0 < α) (h2 : α < 90) (h3 : sin (α - 10 * real.pi / 180) = real.sqrt 3 / 2) : α = 70 * real.pi / 180 := 
by
  sorry

end acute_angle_sine_solution_l529_529157


namespace simplify_expr_l529_529673

theorem simplify_expr : 2 - 2 / (1 + Real.sqrt 2) - 2 / (1 - Real.sqrt 2) = -2 := by
  sorry

end simplify_expr_l529_529673


namespace zero_points_sum_l529_529893

noncomputable theory

-- Definitions of the functions f and g
def f (x : ℝ) : ℝ := 3 ^ x + x - 3
def g (x : ℝ) : ℝ := log 3 x + x - 3

-- Statement of the problem
theorem zero_points_sum : ∃ (x1 x2 : ℝ), f x1 = 0 ∧ g x2 = 0 ∧ x1 + x2 = 3 :=
by
  sorry

end zero_points_sum_l529_529893


namespace largest_gcd_sum780_l529_529345

theorem largest_gcd_sum780 (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 780) : 
  ∃ d, d = Nat.gcd a b ∧ d ≤ 390 ∧ (∀ (d' : ℕ), d' = Nat.gcd a b → d' ≤ 390) :=
sorry

end largest_gcd_sum780_l529_529345


namespace remaining_days_l529_529760

theorem remaining_days (x : ℕ) (initial_workers : ℕ) (additional_workers : ℕ) (initial_days : ℕ) (work_done_days : ℕ) :
  initial_workers = 15 →
  initial_days = 12 →
  additional_workers = 7 →
  work_done_days = 5 →
  x = nat.ceil (105 / 22) →
  x = 5 :=
  by
    intros hw hi ha hd hx,
    sorry

end remaining_days_l529_529760


namespace tan_identity_l529_529538

theorem tan_identity
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 3 / 7)
  (h2 : Real.tan (β - Real.pi / 4) = -1 / 3)
  : Real.tan (α + Real.pi / 4) = 8 / 9 := by
  sorry

end tan_identity_l529_529538


namespace castor_chess_players_l529_529662

theorem castor_chess_players : 
  let total_players := 40 in
  let never_lost_to_ai := total_players / 4 in
  let lost_to_ai := total_players - never_lost_to_ai in
  lost_to_ai = 30 :=
by
  let total_players := 40
  let never_lost_to_ai := total_players / 4
  let lost_to_ai := total_players - never_lost_to_ai
  show lost_to_ai = 30
  exact sorry

end castor_chess_players_l529_529662


namespace angle_between_CE_and_BF_l529_529663

theorem angle_between_CE_and_BF 
  {A B C D E F : Point}
  (h_midpoint: midpoint B C A)
  (h_square: square A B D E)
  (h_equilateral: equilateral_triangle C F A)
  (h_halfplane: same_half_plane BC ABDE CFA) : 
  angle CE BF = 105 :=
sorry

end angle_between_CE_and_BF_l529_529663


namespace limit_problem_l529_529585

variable {ℝ : Type*}

variable (f : ℝ → ℝ)
variable (x_0 : ℝ)

#eval "Given that the derivative of f at x_0 is equal to -3, prove that the limit as h approaches 0 of (f(x_0 + h) - f(x_0 - 3h)) / h is -12."

theorem limit_problem
  (h : ℝ)
  (H_diff : deriv f x_0 = -3) :
  ∃ l : ℝ, l = -12 ∧ Tendsto (fun h => (f (x_0 + h) - f (x_0 - 3 * h)) / h) (nhds 0) (nhds l) :=
sorry

end limit_problem_l529_529585


namespace perpendicular_tangent_line_l529_529496

def f (x : ℝ) : ℝ := x^3 + 3 * x^2 - 1

def g : AffineLine ℝ :=
{ a := 2,
  b := -6,
  c := 1 }

def m : Slope ℝ :=
{ num := 1,
  denom := 3 }

theorem perpendicular_tangent_line :
  ∃ a b c x0 y0 : ℝ,
    f x0 = y0 ∧
    y0 = -1 + 3 - 1 ∧
    g.slope = m ∧
    (a * x0 + b * y0 + c = 0) ∧
    (a = 3 ∧ b = 1 ∧ c = 2) :=
sorry

end perpendicular_tangent_line_l529_529496


namespace max_distinct_values_is_two_l529_529512

-- Definitions of non-negative numbers and conditions
variable (a b c d : ℝ)
variable (ha : 0 ≤ a)
variable (hb : 0 ≤ b)
variable (hc : 0 ≤ c)
variable (hd : 0 ≤ d)
variable (h1 : Real.sqrt (a + b) + Real.sqrt (c + d) = Real.sqrt (a + c) + Real.sqrt (b + d))
variable (h2 : Real.sqrt (a + c) + Real.sqrt (b + d) = Real.sqrt (a + d) + Real.sqrt (b + c))

-- Theorem stating that the maximum number of distinct values among a, b, c, d is 2.
theorem max_distinct_values_is_two : 
  ∃ (u v : ℝ), 0 ≤ u ∧ 0 ≤ v ∧ (u = a ∨ u = b ∨ u = c ∨ u = d) ∧ (v = a ∨ v = b ∨ v = c ∨ v = d) ∧ 
  ∀ (x y : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d) → (y = a ∨ y = b ∨ y = c ∨ y = d) → x = y ∨ x = u ∨ x = v :=
sorry

end max_distinct_values_is_two_l529_529512


namespace problem_1_problem_2_l529_529698

theorem problem_1 
  (h1 : 1 < Real.sqrt 2 ∧ Real.sqrt 2 < 2) : 
  Int.floor (5 - Real.sqrt 2) = 3 :=
sorry

theorem problem_2 
  (h2 : Real.sqrt 3 > 1) : 
  abs (1 - 2 * Real.sqrt 3) = 2 * Real.sqrt 3 - 1 :=
sorry

end problem_1_problem_2_l529_529698


namespace digits_in_product_of_exponentiated_numbers_l529_529193

theorem digits_in_product_of_exponentiated_numbers : 
  (natAbs (log10 (6^3 * 3^9))) + 1 = 7 := by
  sorry

end digits_in_product_of_exponentiated_numbers_l529_529193


namespace recipe_total_cups_l529_529336

noncomputable def total_cups (butter_ratio flour_ratio sugar_ratio sugar_cups : ℕ) : ℕ :=
  let part := sugar_cups / sugar_ratio
  let butter_cups := butter_ratio * part
  let flour_cups := flour_ratio * part
  butter_cups + flour_cups + sugar_cups

theorem recipe_total_cups : 
  total_cups 2 7 5 10 = 28 :=
by
  sorry

end recipe_total_cups_l529_529336


namespace inequality_areas_l529_529352

theorem inequality_areas (a b c α β γ : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hα : α > 0) (hβ : β > 0) (hγ : γ > 0) :
  a / α + b / β + c / γ ≥ 3 / 2 :=
by
  -- Insert the AM-GM inequality application and simplifications
  sorry

end inequality_areas_l529_529352


namespace cashier_opens_probability_l529_529448

-- Definition of the timeline and arrival times
variables {x₁ x₂ x₃ x₄ x₅ x₆ : ℝ}
-- Condition that all arrival times are between 0 and 15 minutes
def arrival_times_within_bounds : Prop := 
  0 ≤ x₁ ∧ x₁ ≤ 15 ∧ 
  0 ≤ x₂ ∧ x₂ ≤ 15 ∧
  0 ≤ x₃ ∧ x₃ ≤ 15 ∧
  0 ≤ x₄ ∧ x₄ ≤ 15 ∧
  0 ≤ x₅ ∧ x₅ ≤ 15 ∧
  0 ≤ x₆ ∧ x₆ ≤ 15

-- Condition that the Scientist arrives last
def scientist_arrives_last : Prop := 
  x₁ < x₆ ∧ x₂ < x₆ ∧ x₃ < x₆ ∧ x₄ < x₆ ∧ x₅ < x₆

-- Event A: Cashier opens no later than 3 minutes after the Scientist arrives, i.e., x₆ ≥ 12
def event_A : Prop := x₆ ≥ 12

-- The correct answer
theorem cashier_opens_probability :
  arrival_times_within_bounds ∧ scientist_arrives_last → 
  Pr[x₆ ≥ 12 | x₁, x₂, x₃, x₄, x₅ < x₆] = 0.738 :=
sorry

end cashier_opens_probability_l529_529448


namespace problem1_general_integral_l529_529500

theorem problem1_general_integral (x y C : ℝ) :
  (x + 1)^3 * (derivative y x) = (y - 2)^2 → -1/(y - 2) + 1/(2 * (x + 1)^2) = C :=
sorry

end problem1_general_integral_l529_529500


namespace sum_equals_four_thirds_l529_529811

-- Define the given sum as a function
def problem_sum : ℝ :=
  ∑' j, ∑' k, 2^(-(2 * k + j + (k + j)^2))

-- The theorem statement to prove the correctness of the sum
theorem sum_equals_four_thirds : problem_sum = 4 / 3 :=
by
  sorry -- proof to demonstrate the result

end sum_equals_four_thirds_l529_529811


namespace sin_1440_eq_zero_l529_529468

-- Define the periodicity condition of sine function
def sin_periodic (θ : ℝ) : Prop := ∀ n : ℤ, sin (θ + n * 360) = sin θ

-- Given the periodicity of sine function, show that sin 1440° = 0
theorem sin_1440_eq_zero : sin 1440 = 0 :=
by
  sorry

end sin_1440_eq_zero_l529_529468


namespace triangle_labeling_l529_529788

theorem triangle_labeling (M : Type) [polygon_division M]
  (even_triangles : ∀ v ∈ vertices M, even (num_triangles v)) :
  ∃ (label : vertices M → {1, 2, 3}),
  ∀ (t ∈ triangles M), let ⟨v₁, v₂, v₃⟩ := vertices t in
    label v₁ ≠ label v₂ ∧ label v₂ ≠ label v₃ ∧ label v₁ ≠ label v₃ :=
begin
  sorry
end

end triangle_labeling_l529_529788


namespace inequality_to_prove_l529_529890

def is_even (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

theorem inequality_to_prove
  (f : ℝ → ℝ)
  (hf_even : is_even f)
  (hf_derivative : ∀ x ∈ Ico 0 (π / 2), deriv f x * real.cos x + f x * real.sin x > 0) :
  f (π / 4) < sqrt 3 * f (π / 3) := 
begin
  sorry
end

end inequality_to_prove_l529_529890


namespace largest_x_exists_largest_x_largest_real_number_l529_529080

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : x ≤ 48 / 7 :=
sorry

theorem exists_largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  ∃ x, (⌊x⌋ : ℝ) / x = 7 / 8 ∧ x = 48 / 7 :=
sorry

theorem largest_real_number (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  x = 48 / 7 :=
sorry

end largest_x_exists_largest_x_largest_real_number_l529_529080


namespace ice_cream_ordering_ways_l529_529935

-- Define the possible choices for each category.
def cone_choices : Nat := 2
def scoop_choices : Nat := 1 + 10 + 20  -- Total choices for 1, 2, and 3 scoops.
def topping_choices : Nat := 1 + 4 + 6  -- Total choices for no topping, 1 topping, and 2 toppings.

-- Theorem to state the number of ways ice cream can be ordered.
theorem ice_cream_ordering_ways : cone_choices * scoop_choices * topping_choices = 748 := by
  let calc_cone := cone_choices  -- Number of cone choices.
  let calc_scoop := scoop_choices  -- Number of scoop combinations.
  let calc_topping := topping_choices  -- Number of topping combinations.
  have h1 : calc_cone * calc_scoop * calc_topping = 748 := sorry  -- Calculation hint.
  exact h1

end ice_cream_ordering_ways_l529_529935


namespace intervals_of_increase_decrease_a_neg1_max_min_values_a_neg2_no_a_for_monotonic_function_l529_529395

noncomputable def quadratic_function (a : ℝ) (x : ℝ) : ℝ := x^2 + 2 * a * x + 3

theorem intervals_of_increase_decrease_a_neg1 : 
  ∀ x : ℝ, quadratic_function (-1) x = x^2 - 2 * x + 3 → 
  (∀ x ≥ 1, quadratic_function (-1) x ≥ quadratic_function (-1) 1) ∧ 
  (∀ x ≤ 1, quadratic_function (-1) x ≤ quadratic_function (-1) 1) :=
  sorry

theorem max_min_values_a_neg2 :
  ∃ min : ℝ, min = -1 ∧ (∀ x : ℝ, quadratic_function (-2) x ≥ min) ∧ 
  (∀ x : ℝ, ∃ y : ℝ, y > x → quadratic_function (-2) y > quadratic_function (-2) x) :=
  sorry

theorem no_a_for_monotonic_function : 
  ∀ a : ℝ, ¬ (∀ x y : ℝ, x ≤ y → quadratic_function a x ≤ quadratic_function a y) ∧ ¬ (∀ x y : ℝ, x ≤ y → quadratic_function a x ≥ quadratic_function a y) :=
  sorry

end intervals_of_increase_decrease_a_neg1_max_min_values_a_neg2_no_a_for_monotonic_function_l529_529395


namespace gwen_spent_zero_l529_529509

theorem gwen_spent_zero 
  (m : ℕ) 
  (d : ℕ) 
  (S : ℕ) 
  (h1 : m = 8) 
  (h2 : d = 5)
  (h3 : (m - S) = (d - S) + 3) : 
  S = 0 :=
by
  sorry

end gwen_spent_zero_l529_529509


namespace intersection_height_correct_l529_529356

noncomputable def height_of_intersection (height1 height2 distance : ℝ) : ℝ :=
  let line1 (x : ℝ) := - (height1 / distance) * x + height1
  let line2 (x : ℝ) := - (height2 / distance) * x
  let x_intersect := - (height2 * distance) / (height1 - height2)
  line1 x_intersect

theorem intersection_height_correct :
  height_of_intersection 40 60 120 = 120 :=
by
  sorry

end intersection_height_correct_l529_529356


namespace disjoint_polynomial_sets_l529_529972

theorem disjoint_polynomial_sets (A B : ℤ) : 
  ∃ C : ℤ, ∀ x1 x2 : ℤ, x1^2 + A * x1 + B ≠ 2 * x2^2 + 2 * x2 + C :=
by
  sorry

end disjoint_polynomial_sets_l529_529972


namespace f_difference_l529_529125

def sum_of_divisors (n : ℕ) : ℕ :=
  Finset.sum (Finset.filter (λ d, n % d = 0) (Finset.range (n+1))) id

def f (n : ℕ) : ℚ := sum_of_divisors n / n

theorem f_difference (h960 : 960 > 0) (h640 : 640 > 0) : f 960 - f 640 = 5 / 8 :=
by
  sorry

end f_difference_l529_529125


namespace symmetric_center_coordinates_l529_529900

theorem symmetric_center_coordinates (ω φ : ℝ) (hω : ω > 0) (hφ : |φ| < (π / 2)) 
  (h_period : (2 * π) / ω = 4 * π) (h_inequality : ∀ x, sin (ω * x + φ) ≤ sin (ω * (π / 3) + φ)) :
  (-2 * π / 3, 0) = (-2 * π / 3, 0) :=
by
  sorry

end symmetric_center_coordinates_l529_529900


namespace equilateral_triangle_ab_l529_529701

theorem equilateral_triangle_ab (a b : ℝ) :
  -- Conditions
  (∃ (a b : ℝ),
    -- Points form an equilateral triangle
    let P0 : ℂ := 0,
        P1 : ℝ := a + 8 * I,
        P2 : ℝ := b + 20 * I in
    P1 * exp (I * π / 3) = P2 ∨ P1 * exp (-I * π / 3) = P2)
  -- Conclusion
  → a * b = 320 / 3 :=
by
  sorry

end equilateral_triangle_ab_l529_529701


namespace part_1_part_2_part_3_l529_529261

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then - (2^x) / (1 + 4^x) else if x = 0 then 0 else (2^x) / (1 + 4^x)

theorem part_1 {x : ℝ} (h : x ∈ Ioo (-1 : ℝ) 1) : 
  f x = if x < 0 then - (2^x) / (1 + 4^x) else if x = 0 then 0 else (2^x) / (1 + 4^x) :=
sorry

theorem part_2 : ∀ x1 x2 ∈ Ioo 0 1, x1 < x2 → f x1 > f x2 :=
sorry

theorem part_3 : ∀ λ : ℝ, (λ ∈ Ioo (-1/2) (-2/5) ∨ λ ∈ Ioo (2/5) (1/2) ∨ λ = 0) ↔ 
  ∃ x : ℝ, x ∈ Ioo (-1) 1 ∧ f x = λ :=
sorry

end part_1_part_2_part_3_l529_529261


namespace lowest_test_score_dropped_is_35_l529_529750

theorem lowest_test_score_dropped_is_35 
  (A B C D : ℕ) 
  (h1 : (A + B + C + D) / 4 = 50)
  (h2 : min A (min B (min C D)) = D)
  (h3 : (A + B + C) / 3 = 55) : 
  D = 35 := by
  sorry

end lowest_test_score_dropped_is_35_l529_529750


namespace H_triple_nested_l529_529976

def H (x : ℝ) : ℝ := x^2 - 2 * x - 3

theorem H_triple_nested {
  -- definition of H
  dom : ∀ x, x ∈ set.Icc (-6 : ℝ) 6 →
  H(H(H 2)) = 0
}:=
by
  sorry

end H_triple_nested_l529_529976


namespace sine_cosine_relation_l529_529984

theorem sine_cosine_relation
  {a b c : ℝ} {α β γ : ℝ}
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a * sin α + b * sin β + c * sin γ = 0)
  (h5 : a * cos α + b * cos β + c * cos γ = 0) :
  sin (β - γ) / a = sin (γ - α) / b ∧ sin (γ - α) / b = sin (α - β) / c :=
by
  sorry

end sine_cosine_relation_l529_529984


namespace BANANA_perm_count_l529_529917

/-- The number of distinct permutations of the letters in the word "BANANA". -/
def distinctArrangementsBANANA : ℕ :=
  let total := 6
  let freqB := 1
  let freqA := 3
  let freqN := 2
  total.factorial / (freqB.factorial * freqA.factorial * freqN.factorial)

theorem BANANA_perm_count : distinctArrangementsBANANA = 60 := by
  unfold distinctArrangementsBANANA
  simp [Nat.factorial_succ]
  exact le_of_eq (decide_eq_true (Nat.factorial_dvd_factorial (Nat.le_succ 6)))
  sorry

end BANANA_perm_count_l529_529917


namespace isosceles_triangle_altitude_is_median_and_angle_bisector_l529_529295

-- Define the isosceles triangle and its properties.
variables {A B C D : Type} [OrderedRing A] 
variables (triangle : IsoscelesTriangle A B C) (h : A = B)

-- Claim to prove the given problem
theorem isosceles_triangle_altitude_is_median_and_angle_bisector 
  (triangle : IsoscelesTriangle A B C) 
  (h : AB = AC) 
  (AD_is_altitude : Altitude AD BC)
  : 
  (AD_is_median : Median AD BC) 
  ∧ 
  (AD_is_angle_bisector : AngleBisector AD (∠ BAC)) :=
sorry

end isosceles_triangle_altitude_is_median_and_angle_bisector_l529_529295


namespace largest_x_l529_529116

def largest_x_with_condition_eq_7_over_8 (x : ℝ) : Prop :=
  ⌊x⌋ / x = 7 / 8

theorem largest_x (x : ℝ) (h : largest_x_with_condition_eq_7_over_8 x) :
  x = 48 / 7 :=
sorry

end largest_x_l529_529116


namespace coeff_x2_y3_z2_l529_529479

theorem coeff_x2_y3_z2 (x y z : ℕ → ℕ) :
  coeff (expand (x - 2 * y + 3 * z)^7) (monomial 2 3 2) = -15120 := 
sorry

end coeff_x2_y3_z2_l529_529479


namespace yellow_chip_value_l529_529229

theorem yellow_chip_value
  (y b g : ℕ)
  (hb : b = g)
  (hchips : y^4 * (4 * b)^b * (5 * g)^g = 16000)
  (h4yellow : y = 2) :
  y = 2 :=
by {
  sorry
}

end yellow_chip_value_l529_529229


namespace staff_member_pays_l529_529405

noncomputable def calculate_final_price (d : ℝ) : ℝ :=
  let discounted_price := 0.55 * d
  let staff_discounted_price := 0.33 * d
  let final_price := staff_discounted_price + 0.08 * staff_discounted_price
  final_price

theorem staff_member_pays (d : ℝ) : calculate_final_price d = 0.3564 * d :=
by
  unfold calculate_final_price
  sorry

end staff_member_pays_l529_529405


namespace amphibian_frog_count_l529_529950

-- Definitions
def is_toal (A : Type) : Prop := ∀ t : A, tells_truth t
def is_frog (A : Type) : Prop := ∀ t : A, tells_lie t

structure Amphibian :=
  (name : String)
  (tells_truth : Prop)
  (tells_lie : Prop)

def Brian : Amphibian := ⟨"Brian", sorry, sorry⟩
def Chris : Amphibian := ⟨"Chris", sorry, sorry⟩
def LeRoy : Amphibian := ⟨"LeRoy", sorry, sorry⟩
def Mike : Amphibian := ⟨"Mike", sorry, sorry⟩
def David : Amphibian := ⟨"David", sorry, sorry⟩

-- Conditions
def condition1 : Prop := (Brian.tells_truth ↔ Mike.tells_lie) ∨ (Brian.tells_lie ↔ Mike.tells_truth)
def condition2 : Prop := (Chris.tells_truth ↔ LeRoy.tells_lie)
def condition3 : Prop := (LeRoy.tells_truth ↔ Chris.tells_lie)
def condition4 : Prop := (Mike.tells_truth ↔ ∃ t1 t2 : Amphibian, is_toal t1 ∧ is_toal t2)
def condition5 : Prop := (David.tells_truth → Brian.tells_truth ∧ ∃ t1 t2 : Amphibian, is_frog t1 ∧ is_frog t2)

-- Proof goal: exactly 3 are frogs
theorem amphibian_frog_count : condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ condition5 → ∃ f1 f2 f3 : Amphibian, is_frog f1 ∧ is_frog f2 ∧ is_frog f3 :=
begin
  sorry
end

end amphibian_frog_count_l529_529950


namespace solution_set_of_f_l529_529461

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def is_monotonically_increasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → x2 < 0 → f (x1) ≤ f (x2)

def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  is_odd_function f ∧ is_monotonically_increasing_on_neg f ∧ f (-1) = 0

theorem solution_set_of_f (f : ℝ → ℝ) (h : satisfies_conditions f) :
  {x : ℝ | f x < 0} = set.Iio (-1) :=
sorry

end solution_set_of_f_l529_529461


namespace find_a_l529_529908

theorem find_a (a : ℝ) :
  (∀ x : ℝ, (|x - 1| > 2 ↔ (x > 3 ∨ x < -1))) →
  (∀ x : ℝ, (x^2 - (a + 1) * x + a < 0 ↔ ((x - 1) * (x - a) < 0))) →
  (set.inter (set_of (λ x, |x - 1| > 2)) (set_of (λ x, x^2 - (a + 1) * x + a < 0)) = set.Ioo 3 5) →
  a = 5 :=
by
  sorry

end find_a_l529_529908


namespace time_needed_to_fill_two_thirds_l529_529721

-- Definitions based on problem conditions
def pool_fill_time_B (x : ℝ) := x
def pool_fill_time_A (x : ℝ) := x + 5
def pool_fill_time_C (x : ℝ) := x - 4

def flow_rate (t : ℝ) := 1 / t

-- Flow rates for the pools
def flow_rate_A (x : ℝ) := flow_rate (pool_fill_time_A x)
def flow_rate_B (x : ℝ) := flow_rate (pool_fill_time_B x)
def flow_rate_C (x : ℝ) := flow_rate (pool_fill_time_C x)

-- Combined flow rates of fountains
def combined_flow_rate (x : ℝ) := flow_rate_A x + flow_rate_B x

-- Mathematically set up the key equation
def key_equation (x : ℝ) := combined_flow_rate x = flow_rate_C x

-- Prove (question == answer) given the conditions
theorem time_needed_to_fill_two_thirds (x : ℝ) (h : key_equation x) : (2/3) * (pool_fill_time_C x) = 4 := 
by sorry

end time_needed_to_fill_two_thirds_l529_529721


namespace A_and_B_time_l529_529400

-- Definitions of work rates
def work_rate (time : ℝ) : ℝ := 1 / time

-- Conditions from the problem
def A_rate := work_rate 3
def BC_rate := work_rate 2
def C_rate := work_rate 3
def B_rate := BC_rate - C_rate

-- Theorem statement: A and B together can complete the work in 2 hours
theorem A_and_B_time : A_rate + B_rate = work_rate 2 := 
by
  -- Here you'd write the proof
  sorry

end A_and_B_time_l529_529400


namespace complex_problem_l529_529885

open Complex

theorem complex_problem (a b : ℝ) :
  (3 + 4 * I) * (a + b * I) = 10 * I → 3 * a - 4 * b = 0 :=
by
  intro h
  have h1 : (3 + 4 * I) * (a + b * I) = ((3 * a - 4 * b) + (4 * a + 3 * b) * I) := by ring
  rw h1 at h
  rw ext_iff at h
  cases h with h_real h_imag
  rw [add_zero] at h_real
  exact h_real

end complex_problem_l529_529885


namespace solve_inequality_l529_529371

theorem solve_inequality (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2)
  (h3 : (x^2 + 3*x - 1) / (4 - x^2) < 1)
  (h4 : (x^2 + 3*x - 1) / (4 - x^2) ≥ -1) :
  x < -5 / 2 ∨ (-1 ≤ x ∧ x < 1) :=
by sorry

end solve_inequality_l529_529371


namespace correct_answer_D_l529_529460

-- Definitions based on problem conditions
def is_right_angled_triangle (A B C : ℕ) : Prop :=
  A + B + C = 180 ∧ (A = 15) ∧ (B = 75) ∧ (C = 90)

-- Lean proof statement
theorem correct_answer_D (A B C : ℕ) (h : A = 15 ∧ B = 75 ∧ C = 90) : is_right_angled_triangle A B C :=
by 
  have h1 : A + B + C = 180, from nat.add_assoc 15 75 90 
  exact ⟨h1, h⟩

end correct_answer_D_l529_529460


namespace evaluate_cos_double_angle_identity_l529_529716

theorem evaluate_cos_double_angle_identity : 2 * cos (π / 12)^2 - 1 = sqrt 3 / 2 :=
by sorry

end evaluate_cos_double_angle_identity_l529_529716


namespace min_abs_expression_l529_529731

theorem min_abs_expression (E : ℝ) (x : ℝ) (h : | x - 4 | + | E | + | x - 5 | = 11) :
  ∃ E_min : ℝ, E_min = 10 :=
by {
  -- Details of the proof would go here, but we are skipping it with sorry.
  sorry
}

end min_abs_expression_l529_529731


namespace slope_angle_l529_529162

-- Define the parametric equations of line l
def line (t α : ℝ) : ℝ × ℝ :=
  (1 + t * Real.cos α, t * Real.sin α)

-- Define the polar to rectangular conversion and the circle equation
def polarToRect (ρ θ : ℝ) : (ℝ × ℝ) :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

def circleEq (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + y ^ 2 = 4

-- Define the condition of intersection distance |-|-|-|= ...-|-|
def intersect (t1 t2 α : ℝ) : Prop :=
  (t1 + t2 = 2 * Real.cos α) ∧
  (t1 * t2 = -3) ∧
  (Real.sqrt ((t1 - t2)^2) = Real.sqrt 14)

-- Define the main theorem
theorem slope_angle {α : ℝ} (hα : 0 ≤ α ∧ α ≤ 2 * Real.pi) :
  (∃ t1 t2 : ℝ, intersect t1 t2 α) →
  (α = Real.pi / 4 ∨ α = 3 * Real.pi / 4) :=
by
  sorry

end slope_angle_l529_529162


namespace inequality_range_l529_529206

theorem inequality_range (a : ℝ) (h : ∀ x : ℝ, |x - 3| + |x + 1| > a) : a < 4 := by
  sorry

end inequality_range_l529_529206


namespace poly_has_prime_divisors_l529_529300

theorem poly_has_prime_divisors (a b : ℤ) : ∃ n : ℕ, (n > 0) ∧ (∃ primes : finset ℕ, primes.card ≥ 2018 ∧ ∀ p ∈ primes, p.prime ∧ p ∣ (n^2 + a * n + b)) :=
by
  sorry

end poly_has_prime_divisors_l529_529300


namespace trig_solutions_l529_529129

noncomputable def solve_trig_eq (n : ℕ) (x : ℝ) : Prop :=
  (n > 0) ∧ (sin x * sin (2 * x) * ... * sin (n * x) + cos x * cos (2 * x) * ... * cos (n * x) = 1)

theorem trig_solutions (n : ℕ) (x : ℝ) (h : solve_trig_eq n x) (hn : n > 0) :
  ∃ m : ℤ, x = m * real.pi :=
sorry

end trig_solutions_l529_529129


namespace three_digit_numbers_eq_11_sum_squares_l529_529740

theorem three_digit_numbers_eq_11_sum_squares :
  ∃ (N : ℕ), 
    (N = 550 ∨ N = 803) ∧
    (∃ (a b c : ℕ), 
      N = 100 * a + 10 * b + c ∧ 
      100 * a + 10 * b + c = 11 * (a ^ 2 + b ^ 2 + c ^ 2) ∧
      1 ≤ a ∧ a ≤ 9 ∧
      0 ≤ b ∧ b ≤ 9 ∧
      0 ≤ c ∧ c ≤ 9) :=
sorry

end three_digit_numbers_eq_11_sum_squares_l529_529740


namespace solution_set_abs_squared_minus_linear_minus_fifteen_l529_529342

theorem solution_set_abs_squared_minus_linear_minus_fifteen (x : ℝ) :
  (|x| ^ 2 - 2 * |x| - 15 > 0) ↔ x ∈ Set.Ioo (-∞) (-5) ∪ Set.Ioo (5) (∞) :=
sorry

end solution_set_abs_squared_minus_linear_minus_fifteen_l529_529342


namespace correct_simplification_l529_529786

-- Step 1: Define the initial expression
def initial_expr (a b : ℝ) : ℝ :=
  (a - b) / a / (a - (2 * a * b - b^2) / a)

-- Step 2: Define the correct simplified form
def simplified_expr (a b : ℝ) : ℝ :=
  1 / (a - b)

-- Step 3: State the theorem that proves the simplification is correct
theorem correct_simplification (a b : ℝ) (h : a ≠ b): 
  initial_expr a b = simplified_expr a b :=
by {
  sorry,
}

end correct_simplification_l529_529786


namespace equivalent_annual_rate_correct_l529_529683

noncomputable def quarterly_rate (annual_rate : ℝ) : ℝ :=
  annual_rate / 4

noncomputable def effective_annual_rate (quarterly_rate : ℝ) : ℝ :=
  (1 + quarterly_rate / 100)^4

noncomputable def equivalent_annual_rate (annual_rate : ℝ) : ℝ :=
  (effective_annual_rate (quarterly_rate annual_rate) - 1) * 100

theorem equivalent_annual_rate_correct :
  equivalent_annual_rate 8 = 8.24 := 
by
  sorry

end equivalent_annual_rate_correct_l529_529683


namespace regular_polygon_sides_l529_529887

theorem regular_polygon_sides (interior_angle : ℝ) (h : interior_angle = 144) :
  ∃ n : ℕ, n * (180 - interior_angle) = 360 ∧ n = 10 :=
by
  use 10
  split
  sorry
  refl

end regular_polygon_sides_l529_529887


namespace geometric_series_sum_l529_529043

theorem geometric_series_sum : 
  let a : ℕ := 2
  let r : ℕ := 3
  let n : ℕ := 6
  let S_n := (a * (r^n - 1)) / (r - 1)
  S_n = 728 :=
by
  sorry

end geometric_series_sum_l529_529043


namespace no_odd_power_terms_in_P_l529_529296

noncomputable def A (x : ℂ) : ℂ := ∑ i in (Finset.range 101), (-1)^i * x^i
noncomputable def B (x : ℂ) : ℂ := ∑ i in (Finset.range 101), x^i
noncomputable def P (x : ℂ) : ℂ := A x * B x

theorem no_odd_power_terms_in_P :
  ∀ x : ℂ, ∀ n : ℕ, P x = (1 - x^202) / (1 - x^2) ∧ (n % 2 = 1 → coeff (Polynomial.ofLaurent (LaurentSeries.ofComplex P x)) n = 0) :=
by
  sorry

end no_odd_power_terms_in_P_l529_529296


namespace a3_value_l529_529974

noncomputable def a0 (n : ℕ) : ℤ := 3^n

noncomputable def a1_to_an_sum (n : ℕ) : ℤ := 1 - 3^n

theorem a3_value :
  ∀ (n : ℕ), n = 5 →
  (a1_to_an_sum n = -242 → 
   ∑ i in (finset.range (n + 1)).filter (λ x, 1 ≤ x), (binom n i) * 3^(n - i) * (-2)^i = -720) := 
by
  intros n hn hs
  rw hs at *
  have hn_eq : n = 5 := hn
  rw hn_eq at *
  sorry

end a3_value_l529_529974


namespace midpoint_of_PQ_coincides_with_excircle_center_l529_529566

noncomputable theory
open_locale classical

variables {A B C P Q : Point} (circ : Circle) (k : Circle)
variables (h_tangent_ab : k.Tangent (Ray A B))
variables (h_tangent_ac : k.Tangent (Ray A C))
variables (h_tangent_circumcircle : k.Tangent circ)
variables (h_center_excircle_opposite_BC : CenterExcircleOppositeBC A B C)

theorem midpoint_of_PQ_coincides_with_excircle_center :
  Midpoint (Segment PQ) = h_center_excircle_opposite_BC :=
sorry

end midpoint_of_PQ_coincides_with_excircle_center_l529_529566


namespace optionA_does_not_specify_unique_triangle_l529_529825

inductive TriangleType
| General
| Right
| Scalene
| Isosceles

inductive CombinationSpecifiesUniqueTriangle
| OptionB : CombinationSpecifiesUniqueTriangle -- Height and base length; right triangle
| OptionD : CombinationSpecifiesUniqueTriangle -- Two angles and one non-included side; general triangle
| OptionE : CombinationSpecifiesUniqueTriangle -- One angle and the sum of the lengths of the two sides forming it; isosceles triangle

theorem optionA_does_not_specify_unique_triangle :
  ¬ (TriangleType.General ∧ (exists s₁ s₂ s₃ : ℝ, s₁ = s₂ + s₃)) → 
  ∀ t, ¬ (t ∈ CombinationSpecifiesUniqueTriangle) :=
by
  intro h
  intro t
  cases t
  sorry

end optionA_does_not_specify_unique_triangle_l529_529825


namespace find_matrix_M_l529_529842

def matrix_M : Type := Matrix (Fin 4) (Fin 4) ℤ

def given_matrix : matrix_M :=
  ![![ -4,  3, 0, 0],
    ![  6, -8, 0, 0],
    ![  0,  0, 2, 1],
    ![  0,  0, 1, 2]]

def identity_matrix : matrix_M :=
  ![![1, 0, 0, 0],
    ![0, 1, 0, 0],
    ![0, 0, 1, 0],
    ![0, 0, 0, 1]]

def is_correct_matrix (M : matrix_M) : Prop :=
  M * given_matrix = identity_matrix

theorem find_matrix_M : ∃ (M : matrix_M), is_correct_matrix M := sorry

end find_matrix_M_l529_529842


namespace largest_x_l529_529119

def largest_x_with_condition_eq_7_over_8 (x : ℝ) : Prop :=
  ⌊x⌋ / x = 7 / 8

theorem largest_x (x : ℝ) (h : largest_x_with_condition_eq_7_over_8 x) :
  x = 48 / 7 :=
sorry

end largest_x_l529_529119


namespace arithmetic_sequence_y_value_l529_529475

theorem arithmetic_sequence_y_value : 
    (∃ y : ℚ, ((y - 2) - (3 / 4) = (4 * y - (y - 2))) ∧ y = -19 / 8) :=
by
  exist y : ℚ,
  sorry

end arithmetic_sequence_y_value_l529_529475


namespace gcd_computation_l529_529041

theorem gcd_computation : 
  let a := 107^7 + 1,
  let b := 107^7 + 107^3 + 1 in
  Int.gcd a b = 1 := by 
  sorry

end gcd_computation_l529_529041


namespace distance_walked_east_l529_529302

-- Definitions for distances
def s1 : ℕ := 25   -- distance walked south
def s2 : ℕ := 20   -- distance walked east
def s3 : ℕ := 25   -- distance walked north
def final_distance : ℕ := 35   -- final distance from the starting point

-- Proof problem: Prove that the distance walked east in the final step is as expected
theorem distance_walked_east (d : Real) :
  d = Real.sqrt (final_distance ^ 2 - s2 ^ 2) :=
sorry

end distance_walked_east_l529_529302


namespace proof_S5_a5_l529_529165

variables (S a : ℕ → ℕ)

-- Conditions
axiom h1 : ∀ n, 3 * S n - 6 = 2 * a n
axiom S_def : S 1 = a 1 

-- Given the sum of the sequence and the initial condition
def Sn (n : ℕ) : ℕ :=
  2 * (1 - (-2)^n)

def an (n : ℕ) : ℕ :=
  6 * (-2)^(n - 1)

-- Proof problem
theorem proof_S5_a5 :
  S 5 / a 5 = 11 / 16 :=
sorry

end proof_S5_a5_l529_529165


namespace arithmetic_square_root_of_sqrt_16_l529_529689

theorem arithmetic_square_root_of_sqrt_16 : ∃ (n : ℝ), n = 4 ∧ n^2 = sqrt 16 := 
by 
  have h : sqrt 16 = 4 := sorry -- Calculation that sqrt 16 is 4
  existsi (4:ℝ)
  split
  . refl
  . exact h

end arithmetic_square_root_of_sqrt_16_l529_529689


namespace average_of_seventeen_numbers_is_59_l529_529690

-- Definitions based on conditions
def nine_numbers_avg_1 := 56 : ℕ
def nine_numbers_avg_2 := 63 : ℕ
def ninth_number := 68 : ℕ

-- Proof statement
theorem average_of_seventeen_numbers_is_59 : ((9 * nine_numbers_avg_1 + 9 * nine_numbers_avg_2 - ninth_number) / 17 = 59) :=
by
  sorry

end average_of_seventeen_numbers_is_59_l529_529690


namespace sum_of_T_l529_529966

-- Definition of digits and distinct function.
def is_nonzero_distinct_digit (d : ℕ) : Prop := d > 0 ∧ d < 10
def is_nonzero_distinct (a b c : ℕ) : Prop :=
  is_nonzero_distinct_digit a ∧ is_nonzero_distinct_digit b ∧ is_nonzero_distinct_digit c ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Definition of the set T
def T : set ℝ := {x | ∃ (a b c : ℕ), is_nonzero_distinct a b c ∧ x = (a * 100 + b * 10 + c) / 999}

-- Statement of the theorem
theorem sum_of_T : ∑ x in T, x = 278 :=
by
-- Proof goes here
sorry

end sum_of_T_l529_529966


namespace angle_ACP_l529_529975

-- Definition and conditions
def AB_len : ℝ := 12
def midpoint (x y : ℝ) : ℝ := (x + y) / 2
def C := midpoint 0 AB_len
def mid_length (x : ℝ) : ℝ := x / 2
def R1 := mid_length AB_len
def BC_len := mid_length AB_len
def R2 := mid_length BC_len
def area_of_semicircle (r : ℝ) : ℝ := (1 / 2) * Real.pi * r ^ 2
def A1 := area_of_semicircle R1
def A2 := area_of_semicircle R2
def total_area := A1 + A2
def larger_area := (3 / 4) * total_area
def fractional_area := larger_area / A1
def total_sector_angle := 360 * fractional_area
def theta_ACP := total_sector_angle - 180

-- Theorem statement to prove angle measure
theorem angle_ACP : theta_ACP = 157.5 := by
  sorry

end angle_ACP_l529_529975


namespace proportional_function_is_A_l529_529021

def direct_proportional (f : ℝ → ℝ) : Prop := ∃ k : ℝ, ∀ x : ℝ, f(x) = k * x

def option_A : ℝ → ℝ := λ x, (1/2) * x
def option_B : ℝ → ℝ := λ x, 2 * x + 1
def option_C : ℝ → ℝ := λ x, 2 / x
def option_D : ℝ → ℝ := λ x, x^2

theorem proportional_function_is_A :
  direct_proportional option_A ∧ 
  ¬ direct_proportional option_B ∧ 
  ¬ direct_proportional option_C ∧
  ¬ direct_proportional option_D := 
by 
  sorry

end proportional_function_is_A_l529_529021


namespace exists_100_nat_nums_with_cube_sum_property_l529_529484

theorem exists_100_nat_nums_with_cube_sum_property :
  ∃ (a : Fin 100 → ℕ), Function.Injective a ∧
    ∃ (k : Fin 100), (a k)^3 = ∑ i in Finset.univ \ {k}, (a i)^3 :=
by
  sorry

end exists_100_nat_nums_with_cube_sum_property_l529_529484


namespace six_nats_false_statement_l529_529063

theorem six_nats_false_statement :
  ∃ (a1 a2 a3 a4 a5 a6 : ℕ), ¬(∃ (b c d: ℕ),
    ({b, c, d} ⊆ {a1, a2, a3, a4, a5, a6} ∧ 
     (∀ (p q: ℕ), {p, q} ⊆ {b, c, d} → gcd p q = 1) ∨ 
     gcd b c > 1 ∧ gcd b d > 1 ∧ gcd c d > 1)) :=
by
  -- To be solved
  sorry

end six_nats_false_statement_l529_529063


namespace denis_fourth_board_score_l529_529059

theorem denis_fourth_board_score :
  ∀ (darts_per_board points_first_board points_second_board points_third_board points_total_boards : ℕ),
    darts_per_board = 3 →
    points_first_board = 30 →
    points_second_board = 38 →
    points_third_board = 41 →
    points_total_boards = (points_first_board + points_second_board + points_third_board) / 2 →
    points_total_boards = 34 :=
by
  intros darts_per_board points_first_board points_second_board points_third_board points_total_boards h1 h2 h3 h4 h5
  sorry

end denis_fourth_board_score_l529_529059


namespace four_letter_words_with_vowels_l529_529913

theorem four_letter_words_with_vowels (s : Finset Char) (len : ℕ) (vowels : Finset Char) :
  (s = {'A', 'B', 'C', 'D', 'E'}) ∧ (len = 4) ∧ (vowels = {'A', 'E'}) →
  (number_of_words_with_at_least_one_vowel s len vowels = 544) := by
  sorry

end four_letter_words_with_vowels_l529_529913


namespace option_D_not_right_angled_l529_529794

def is_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def option_A (a b c : ℝ) : Prop :=
  b^2 = a^2 - c^2

def option_B (a b c : ℝ) : Prop :=
  a = 3 * c / 5 ∧ b = 4 * c / 5

def option_C (A B C : ℝ) : Prop :=
  C = A - B ∧ A + B + C = 180

def option_D (A B C : ℝ) : Prop :=
  A / 3 = B / 4 ∧ B / 4 = C / 5

theorem option_D_not_right_angled (a b c A B C : ℝ) :
  ¬ is_right_angled_triangle a b c ↔ option_D A B C :=
  sorry

end option_D_not_right_angled_l529_529794


namespace umbrellaNumberCount_l529_529427

-- Definition: Umbrella number
def isUmbrellaNumber (n : ℕ) : Prop :=
  let d₁ := n / 100 in                        -- hundreds place
  let d₂ := (n % 100) / 10 in                 -- tens place
  let d₃ := n % 10 in                         -- units place
  d₂ > d₁ ∧ d₂ > d₃ ∧ d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃ ∧ 
  d₁ ∈ {1, 2, 3, 4, 5, 6} ∧ 
  d₂ ∈ {1, 2, 3, 4, 5, 6} ∧
  d₃ ∈ {1, 2, 3, 4, 5, 6}

-- Proof statement: Number of Umbrella numbers formed
theorem umbrellaNumberCount : 
  { n : ℕ | isUmbrellaNumber n }.card = 20 := 
sorry

end umbrellaNumberCount_l529_529427


namespace sum_ages_of_brothers_l529_529606

theorem sum_ages_of_brothers (x : ℝ) (ages : List ℝ) 
  (h1 : ages = [x, x + 1.5, x + 3, x + 4.5, x + 6, x + 7.5, x + 9])
  (h2 : x + 9 = 4 * x) : 
    List.sum ages = 52.5 := 
  sorry

end sum_ages_of_brothers_l529_529606


namespace number_of_solutions_is_3_l529_529197

noncomputable def ab_plus_c_eq_17 (a b c : ℤ) : Prop := a * b + c = 17
noncomputable def a_plus_bc_eq_19 (a b c : ℤ) : Prop := a + b * c = 19

theorem number_of_solutions_is_3 :
  (∃ abc : Finset (ℤ × ℤ × ℤ), ∀ (a b c : ℤ), (a, b, c) ∈ abc ↔ ab_plus_c_eq_17 a b c ∧ a_plus_bc_eq_19 a b c) ∧ 
  abc.card = 3 :=
sorry

end number_of_solutions_is_3_l529_529197


namespace remaining_payment_l529_529749
noncomputable def total_cost (deposit : ℝ) (percentage : ℝ) : ℝ :=
  deposit / percentage

noncomputable def remaining_amount (deposit : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost - deposit

theorem remaining_payment (deposit : ℝ) (percentage : ℝ) (total_cost : ℝ) (remaining_amount : ℝ) :
  deposit = 140 → percentage = 0.1 → total_cost = deposit / percentage → remaining_amount = total_cost - deposit → remaining_amount = 1260 :=
by
  intros
  sorry

end remaining_payment_l529_529749


namespace find_genuine_stacks_l529_529288

theorem find_genuine_stacks
  (a b c d : ℕ) (different : ℕ)
  (h : a + b + c + d = 37)
  (stack_sizes : a = 5 ∧ b = 6 ∧ c = 7 ∧ d = 19)
  (stack_contains_different : (different = a ∨ different = b ∨ different = c ∨ different = d) ∧ different ∈ {a, b, c, d}) :
  (∃ h1 h2 : ℕ, (h1 = 5 ∧ h2 = 7) ∨ (h1 = 6 ∧ h2 = 19)) :=
  sorry

end find_genuine_stacks_l529_529288


namespace largest_real_number_condition_l529_529072

theorem largest_real_number_condition (x : ℝ) (hx : ⌊x⌋ / x = 7 / 8) : x ≤ 48 / 7 :=
by
  sorry

end largest_real_number_condition_l529_529072


namespace dave_walking_probability_l529_529032

theorem dave_walking_probability :
  ∃ m n : ℕ, m + n = 74 ∧ ∀ (g₁ g₂ : ℕ), 
  (6 ≤ g₁ ∧ g₁ ≤ 15) ∧ (6 ≤ g₂ ∧ g₂ ≤ 15) ∧ g₁ ≠ g₂ →
  let distance := abs (g₂ - g₁) * 100 in
  distance ≤ 300 →
  (m / n : ℚ) = 29 / 45 :=
begin
  sorry
end

end dave_walking_probability_l529_529032


namespace parallelepiped_volume_l529_529921

-- Definitions
variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1)
variable (angle_pi_div4 : real.angle_between a b = real.pi / 4)

-- Theorem statement
theorem parallelepiped_volume : 
  real.abs (inner_product_space.Inner ℝ a ((b + 2 • (b ×ₗ a)) ×ₗ b)) = 1 :=
sorry

end parallelepiped_volume_l529_529921


namespace jerry_bought_3_pounds_l529_529954

-- Definitions based on conditions:
def cost_mustard_oil := 2 * 13
def cost_pasta_sauce := 5
def total_money := 50
def money_left := 7
def cost_gluten_free_pasta_per_pound := 4

-- The proof goal based on the correct answer:
def pounds_gluten_free_pasta : Nat :=
  let total_spent := total_money - money_left
  let spent_on_mustard_and_sauce := cost_mustard_oil + cost_pasta_sauce
  let spent_on_pasta := total_spent - spent_on_mustard_and_sauce
  spent_on_pasta / cost_gluten_free_pasta_per_pound

theorem jerry_bought_3_pounds :
  pounds_gluten_free_pasta = 3 := by
  -- the proof should follow here
  sorry

end jerry_bought_3_pounds_l529_529954


namespace sufficient_but_not_necessary_condition_l529_529390

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := abs (x * (m * x + 2))

theorem sufficient_but_not_necessary_condition (m : ℝ) : 
  (∃ m0 : ℝ, m0 > 0 ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f m0 x1 ≤ f m0 x2)) ∧ 
  ¬ (∀ m : ℝ, (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f m x1 ≤ f m x2) → m > 0) :=
by sorry

end sufficient_but_not_necessary_condition_l529_529390


namespace pencils_per_child_l529_529826

theorem pencils_per_child (total_pencils : ℕ) (num_children : ℕ) (H1 : total_pencils = 32) (H2 : num_children = 8) : (total_pencils / num_children) = 4 :=
by {
  rw [H1, H2],
  norm_num,
  sorry
}

end pencils_per_child_l529_529826


namespace area_of_rectangular_field_l529_529242

theorem area_of_rectangular_field (length width : ℝ) (h_length: length = 5.9) (h_width: width = 3) : 
  length * width = 17.7 := 
by
  sorry

end area_of_rectangular_field_l529_529242


namespace possible_values_l529_529188

-- Define vectors OA, OB, and OC
def vec_OA : ℝ × ℝ := (3, -4)
def vec_OB : ℝ × ℝ := (6, -3)
def vec_OC (m : ℝ) : ℝ × ℝ := (5 - m, -3 - m)

-- Define the vectors BA and BC
def vec_BA : ℝ × ℝ := (vec_OA.1 - vec_OB.1, vec_OA.2 - vec_OB.2)
def vec_BC (m : ℝ) : ℝ × ℝ := (vec_OC m).1 - vec_OB.1, (vec_OC m).2 - vec_OB.2

-- Dot product of two 2D vectors
def dot_prod (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define the condition for the angle ABC to be acute
def angle_ABC_acute (m : ℝ) : Prop := dot_prod vec_BA (vec_BC m) > 0

-- The result we wish to prove
def result (m : ℝ) : Prop := m = 0 ∨ m = 1

-- The theorem stating the equivalence
theorem possible_values (m : ℝ) (h : angle_ABC_acute m) : result m :=
sorry

end possible_values_l529_529188


namespace angle_A_eq_pi_div_3_area_of_triangle_l529_529260

variables (A B C : ℝ) (a b c : ℝ)

-- Given conditions
variable (h1 : a * sin B = sqrt 3 * b * cos A)
variable (cos_C : cos C = 1/3)
variable (c_val : c = 4 * sqrt 2)

-- Proof Statements
theorem angle_A_eq_pi_div_3 : A = Real.pi / 3 :=
  sorry

theorem area_of_triangle :
  a = 3 * sqrt 3 → 
  (1/2) * a * c * sin (A + C) = 4 * sqrt 3 + 3 * sqrt 2 :=
  sorry

end angle_A_eq_pi_div_3_area_of_triangle_l529_529260


namespace max_value_of_y_l529_529269

noncomputable def arg (z : ℂ) : ℝ :=
  complex.arg z

def y_function (θ : ℝ) : ℝ :=
  let z := complex.of_real (3 * real.cos θ) + complex.I * (2 * real.sin θ)
  θ - arg z

theorem max_value_of_y :
  ∃ θ ∈ set.Ioo 0 (π / 2), y_function θ = y_function (π / 3)  :=
sorry

end max_value_of_y_l529_529269


namespace yolanda_three_point_avg_l529_529380

-- Definitions based on conditions
def total_points_season := 345
def total_games := 15
def free_throws_per_game := 4
def two_point_baskets_per_game := 5

-- Definitions based on the derived quantities
def average_points_per_game := total_points_season / total_games
def points_from_two_point_baskets := two_point_baskets_per_game * 2
def points_from_free_throws := free_throws_per_game * 1
def points_from_non_three_point_baskets := points_from_two_point_baskets + points_from_free_throws
def points_from_three_point_baskets := average_points_per_game - points_from_non_three_point_baskets
def three_point_baskets_per_game := points_from_three_point_baskets / 3

-- The theorem to prove that Yolanda averaged 3 three-point baskets per game
theorem yolanda_three_point_avg:
  three_point_baskets_per_game = 3 := sorry

end yolanda_three_point_avg_l529_529380


namespace castor_chess_players_l529_529661

theorem castor_chess_players : 
  let total_players := 40 in
  let never_lost_to_ai := total_players / 4 in
  let lost_to_ai := total_players - never_lost_to_ai in
  lost_to_ai = 30 :=
by
  let total_players := 40
  let never_lost_to_ai := total_players / 4
  let lost_to_ai := total_players - never_lost_to_ai
  show lost_to_ai = 30
  exact sorry

end castor_chess_players_l529_529661


namespace sin4_minus_cos4_l529_529919

theorem sin4_minus_cos4 (α : ℝ) (h : cos (2 * α) = 3 / 5) : sin α ^ 4 - cos α ^ 4 = -3 / 5 :=
by
  sorry

end sin4_minus_cos4_l529_529919


namespace possible_values_of_b2_l529_529777

def sequence (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = |b (n + 1) - b n|

variable (b : ℕ → ℕ)
variable (initial_condition : b 1 = 500)
variable (condition : b 2 < 500)
variable (final_condition : b 2006 = 0)

theorem possible_values_of_b2 :
  (∃ b2_values : list ℕ, ∀ b2 ∈ b2_values, 
    sequence b → b 1 = 500 → b 2 < 500 → b 2006 = 0 → b2 ∈ {2, 4, 10, 20, 50, 100, 250} ∧ b2_values.length = 8) :=
sorry

end possible_values_of_b2_l529_529777


namespace probability_window_opens_correct_l529_529454

noncomputable def probability_window_opens_no_later_than_3_minutes_after_scientist_arrives 
  (arrival_times : Fin 6 → ℝ) : ℝ :=
  if (∀ i, arrival_times i ∈ Set.Icc 0 15) ∧ 
     (∀ i j, i ≠ j → arrival_times i < arrival_times j) ∧ 
     ((∃ i, arrival_times i ≥ 12)) then
    1 - (0.8 ^ 6)
  else
    0

theorem probability_window_opens_correct : 
  ∀ (arrival_times : Fin 6 → ℝ),
    (∀ i, arrival_times i ∈ Set.Icc 0 15) →
    (∀ i j, i ≠ j → arrival_times i < arrival_times j) →
    (∃ i, arrival_times i = arrival_times 5) →
    abs (probability_window_opens_no_later_than_3_minutes_after_scientist_arrives arrival_times - 0.738) < 0.001 :=
by
  sorry

end probability_window_opens_correct_l529_529454


namespace geometric_series_sum_l529_529042

theorem geometric_series_sum : 
  let a : ℕ := 2
  let r : ℕ := 3
  let n : ℕ := 6
  let S_n := (a * (r^n - 1)) / (r - 1)
  S_n = 728 :=
by
  sorry

end geometric_series_sum_l529_529042


namespace range_of_composite_function_l529_529707

theorem range_of_composite_function :
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ (2^(x+1) = y)) ↔ (1 ≤ y ∧ y ≤ 4) := 
by {
  sorry
}

end range_of_composite_function_l529_529707


namespace percent_decrease_is_12_l529_529955

-- Define the initial conditions
def last_month_price_per_packet : ℝ := 7.50 / 6
def this_month_price_per_packet : ℝ := 11 / 10

-- Define the percent decrease calculation
def percent_decrease (old_price new_price : ℝ) : ℝ :=
  ((old_price - new_price) / old_price) * 100

-- The problem statement in Lean
theorem percent_decrease_is_12 :
  percent_decrease last_month_price_per_packet this_month_price_per_packet = 12 :=
by
  unfold last_month_price_per_packet this_month_price_per_packet percent_decrease
  -- proof goes here
  sorry

end percent_decrease_is_12_l529_529955


namespace parallelogram_area_l529_529466

noncomputable def area_parallelogram (a b p q : ℝ → ℝ) : ℝ :=
  |a × b|

variable {p q : ℝ → ℝ}
variable (θ : ℝ)
variable (h1 : p = λ i, 3) (h2 : q = λ i, 4) (h3 : θ = (Real.pi / 3))

noncomputable def cross_product (u v : ℝ → ℝ) : ℝ → ℝ :=
  λ i, (u i) * (v i) -- simplified version, actual cross product involves determinate calculation

theorem parallelogram_area :
  let a := λ i, 3 * p i - q i in
  let b := λ i, p i + 2 * q i in
  area_parallelogram a b p q = 42 * Real.sqrt 3 :=
by
  intro a b p q
  have h_p : |p| = 3 := h1
  have h_q : |q| = 4 := h2
  have hθ : θ = (Real.pi / 3) := h3
  sorry

end parallelogram_area_l529_529466


namespace inequality_solution_l529_529998

noncomputable def f (x : ℝ) : ℝ := real.arcsin x^2 + real.arcsin x + x^6 + x^3

theorem inequality_solution {x : ℝ} (hx : x ∈ Icc (-1 : ℝ) 1) : f x > 0 ↔ 0 < x ∧ x ≤ 1 := by
  sorry

end inequality_solution_l529_529998


namespace a_and_b_are_kth_powers_l529_529634

theorem a_and_b_are_kth_powers (k : ℕ) (h_k : 1 < k) (a b : ℤ) (h_rel_prime : Int.gcd a b = 1)
  (c : ℤ) (h_ab_power : a * b = c^k) : ∃ (m n : ℤ), a = m^k ∧ b = n^k :=
by
  sorry

end a_and_b_are_kth_powers_l529_529634


namespace find_t2_l529_529010

def P1 : ℝ := 1035
def P2 : ℝ := 1656
def r1 : ℝ := 3 / 100
def r2 : ℝ := 5 / 100
def t1 : ℝ := 8

theorem find_t2 : ∃ t2 : ℝ, P1 * r1 * t1 = P2 * r2 * t2 ∧ t2 = 3 := 
by
  use 3
  split
  calc
    P1 * r1 * t1 = 1035 * 0.03 * 8 : by rfl
    ... = 248.4 : by norm_num
    P2 * r2 * 3 = 1656 * 0.05 * 3 : by rfl
    ... = 248.4 : by norm_num
  done

end find_t2_l529_529010


namespace red_2003rd_is_3943_l529_529942

section RedSequence

def is_red (n : ℕ) : Prop :=
  ∃ k m : ℕ, k > 0 ∧ m > 0 ∧ (∑ i in finset.range k, i + 1) ≤ n ∧ n ≤ (∑ i in finset.range k, i + 1) + m

theorem red_2003rd_is_3943 : ∃ n : ℕ, is_red n ∧ n = 3943 :=
by
  sorry

end RedSequence

end red_2003rd_is_3943_l529_529942


namespace problem_l529_529927

variable (x y : ℝ)

theorem problem (h1 : x + 3 * y = 6) (h2 : x * y = -12) : x^2 + 9 * y^2 = 108 :=
sorry

end problem_l529_529927


namespace largest_real_number_condition_l529_529076

theorem largest_real_number_condition (x : ℝ) (hx : ⌊x⌋ / x = 7 / 8) : x ≤ 48 / 7 :=
by
  sorry

end largest_real_number_condition_l529_529076


namespace exists_distinct_hamiltonian_cycle_l529_529216
-- Import the entire math library for necessary definitions and theorems

-- Define the cubic graph and the existence of a Hamiltonian cycle
variable {V : Type*} [Fintype V] (G : SimpleGraph V)

-- Assume the graph is cubic (each vertex has degree exactly 3)
def is_cubic (G : SimpleGraph V) : Prop :=
  ∀ v : V, G.degree v = 3

-- Assume a Hamiltonian cycle exists in the graph
variable (C : Cycle V)
def is_hamiltonian_cycle (C : Cycle V) (G : SimpleGraph V) : Prop :=
  (∀ v : V, v ∈ C.vertices) ∧ (C.edges ⊆ G.edge_set)

-- The proposition to prove
theorem exists_distinct_hamiltonian_cycle (G : SimpleGraph V) [Fintype V] (hG : is_cubic G) 
  (hC : is_hamiltonian_cycle C G) :
  ∃ C' : Cycle V, is_hamiltonian_cycle C' G ∧ C' ≠ C ∧ C'.reverse ≠ C :=
by sorry

end exists_distinct_hamiltonian_cycle_l529_529216


namespace length_PQ_calc_l529_529409

noncomputable def length_PQ 
  (F : ℝ × ℝ) 
  (P Q : ℝ × ℝ) 
  (hF : F = (1, 0)) 
  (hP_on_parabola : P.2 ^ 2 = 4 * P.1) 
  (hQ_on_parabola : Q.2 ^ 2 = 4 * Q.1) 
  (hLine_through_focus : F.1 = ((P.2 - Q.2) / (P.1 - Q.1)) * 1 + P.1) 
  (hx1x2 : P.1 + Q.1 = 9) : ℝ :=
|P.1 - Q.1|

theorem length_PQ_calc : ∀ F P Q
  (hF : F = (1, 0))
  (hP_on_parabola : P.2 ^ 2 = 4 * P.1)
  (hQ_on_parabola : Q.2 ^ 2 = 4 * Q.1)
  (hLine_through_focus : F.1 = ((P.2 - Q.2) / (P.1 - Q.1)) * 1 + P.1)
  (hx1x2 : P.1 + Q.1 = 9),
  length_PQ F P Q hF hP_on_parabola hQ_on_parabola hLine_through_focus hx1x2 = 11 := 
by
  sorry

end length_PQ_calc_l529_529409


namespace find_finleys_age_l529_529667

-- Definitions for given problem
def rogers_age (J A : ℕ) := (J + A) / 2
def alex_age (F : ℕ) := 3 * (F + 10) - 5

-- Given conditions
def jills_age : ℕ := 20
def in_15_years_age_difference (R J F : ℕ) := R + 15 - (J + 15) = F - 30
def rogers_age_twice_jill_plus_five (J : ℕ) := 2 * J + 5

-- Theorem stating the problem assertion
theorem find_finleys_age (F : ℕ) :
  rogers_age jills_age (alex_age F) = rogers_age_twice_jill_plus_five jills_age ∧ 
  in_15_years_age_difference (rogers_age jills_age (alex_age F)) jills_age F →
  F = 15 :=
by
  sorry

end find_finleys_age_l529_529667


namespace only_sqrt_three_is_irrational_l529_529023

-- Definitions based on conditions
def zero_rational : Prop := ∃ p q : ℤ, q ≠ 0 ∧ (0 : ℝ) = p / q
def neg_three_rational : Prop := ∃ p q : ℤ, q ≠ 0 ∧ (-3 : ℝ) = p / q
def one_third_rational : Prop := ∃ p q : ℤ, q ≠ 0 ∧ (1/3 : ℝ) = p / q
def sqrt_three_irrational : Prop := ¬ ∃ p q : ℤ, q ≠ 0 ∧ (Real.sqrt 3) = p / q

-- The proof problem statement
theorem only_sqrt_three_is_irrational :
  zero_rational ∧
  neg_three_rational ∧
  one_third_rational ∧
  sqrt_three_irrational :=
by sorry

end only_sqrt_three_is_irrational_l529_529023


namespace prove_real_roots_and_find_m_l529_529866

-- Condition: The quadratic equation
def quadratic_eq (m x : ℝ) : Prop := x^2 - (m-1)*x + m-2 = 0

-- Condition: Discriminant
def discriminant (m : ℝ) : ℝ := (m-3)^2

-- Define the problem as a proposition
theorem prove_real_roots_and_find_m (m : ℝ) :
  (discriminant m ≥ 0) ∧ 
  (|3 - m| = 3 → (m = 0 ∨ m = 6)) :=
by
  sorry

end prove_real_roots_and_find_m_l529_529866


namespace alex_pays_less_l529_529017

noncomputable def alex_cost_per_trip := 2.25
noncomputable def sam_cost_per_trip := 3.00
def number_of_trips := 20

theorem alex_pays_less : (number_of_trips * sam_cost_per_trip) - (number_of_trips * alex_cost_per_trip) = 15 := by
  sorry

end alex_pays_less_l529_529017


namespace find_m_value_find_m_range_l529_529901

def f (m : ℝ) (x : ℝ) : ℝ := m * x ^ 3 - 3 * (m + 1) * x ^ 2 + (3 * m + 6) * x + 1

def f_prime (m : ℝ) (x : ℝ) : ℝ := 3 * m * x ^ 2 - 6 * (m + 1) * x + 3 * m + 6

theorem find_m_value : 
  ∃ m : ℝ, m < 0 ∧ (∀ x ∈ set.Ioo 0 1, f_prime m x > 0) ∧ f_prime m 0 = 0 ∧ f_prime m 1 = 0 → m = -2 :=
sorry

theorem find_m_range :
  ∃ m : ℝ, m ∈ Ioo (-4 / 3) 0 ∧ (∀ x ∈ Icc (-1 : ℝ) 1, 3 * m * x ^ 2 - 6 * (m + 1) * x + 6 > 3 * m) :=
sorry

end find_m_value_find_m_range_l529_529901


namespace line_through_points_l529_529986

-- Define the conditions and the required proof statement
theorem line_through_points (x1 y1 z1 x2 y2 z2 x y z m n p : ℝ) :
  (∃ m n p, (x-x1) / m = (y-y1) / n ∧ (y-y1) / n = (z-z1) / p) → 
  (x-x1) / (x2 - x1) = (y-y1) / (y2 - y1) ∧ 
  (y-y1) / (y2 - y1) = (z-z1) / (z2 - z1) :=
sorry

end line_through_points_l529_529986


namespace MN_perp_OH_l529_529527

theorem MN_perp_OH 
  (A B C O H E F D I M N : Point)
  (h_acute_triangle : AcuteTriangle A B C)
  (h_sides_diff_length : DiffLengthSides A B C)
  (h_circumcenter : is_circumcenter O A B C)
  (h_orthocenter : is_orthocenter H A B C)
  (h_B_altitude : is_altitude B E A C)
  (h_C_altitude : is_altitude C F A B)
  (h_AH_intersects_circumcircle_at_D : intersects_second_time (line_through A H) (circumcircle A B C) D)
  (h_midpoint_I : is_midpoint I A H)
  (h_EI_intersects_BD_at_M : intersects (line_through E I) (line_through B D) M)
  (h_FI_intersects_CD_at_N : intersects (line_through F I) (line_through C D) N) :
  perpendicular (line_through M N) (line_through O H) :=
sorry

end MN_perp_OH_l529_529527


namespace range_f6_range_f2n_l529_529548

open Real

-- Define the functions
def f4 (x : ℝ) : ℝ := (sin x) ^ 4 + (cos x) ^ 4
def f6 (x : ℝ) : ℝ := (sin x) ^ 6 + (cos x) ^ 6
def f2n (x : ℝ) (n : ℕ) : ℝ := (sin x) ^ (2 * n) + (cos x) ^ (2 * n)

-- Conditions provided in the problem
axiom range_f4 : ∀ x, 1 / 2 ≤ f4 x ∧ f4 x ≤ 1
axiom trig_identity : ∀ x, (sin x) ^ 2 + (cos x) ^ 2 = 1

-- Proof problem 1
theorem range_f6 : ∃ a b, (∀ x, a ≤ f6 x ∧ f6 x ≤ b) ∧ a = 1 / 4 ∧ b = 1 :=
by
  sorry

-- Proof problem 2
theorem range_f2n : ∀ n : ℕ, n > 0 → ∃ a b, (∀ x, a ≤ f2n x n ∧ f2n x n ≤ b) ∧ a = 1 / 2^(n-1) ∧ b = 1 :=
by
  sorry

end range_f6_range_f2n_l529_529548


namespace unknown_number_average_l529_529388

theorem unknown_number_average :
  let x := 19 in
  (20 + 40 + 60) / 3 = ((10 + 70 + x) / 3 + 7) ↔ x = 19 :=
by
  sorry

end unknown_number_average_l529_529388


namespace infinite_series_sum_l529_529814

theorem infinite_series_sum :
  (∑' j : ℕ, ∑' k : ℕ, 2^(-(2 * k + j + (k + j) ^ 2) : ℤ)) = 4 / 3 :=
by
  sorry

end infinite_series_sum_l529_529814


namespace maximize_l_a_l529_529183

noncomputable def max_l (a : ℝ) (h : a < 0) : ℝ := 
if h : a = -2 then 2 + real.sqrt 3 
else if h : a = -8 then 1 
else 0

theorem maximize_l_a (a l : ℝ) (h1 : a < 0) 
  (h2 : ∀ x, 0 ≤ x ∧ x ≤ l → abs (a * x^2 + 8 * x + 3) ≤ 5) :
  a = -2 ∧ l = 2 + real.sqrt 3 :=
by
  -- Proof goes here
  sorry

end maximize_l_a_l529_529183


namespace ratio_fraction_l529_529069

theorem ratio_fraction (x y : ℝ) (h : 5 / 34 * y = 34 / 7 * x) : 
  x / y ≈ 0.9478672985781991 :=
by
  sorry

end ratio_fraction_l529_529069


namespace digit_7_occurrences_in_range_1_to_2017_l529_529221

-- Define the predicate that checks if a digit appears in a number
def digit_occurrences (d n : Nat) : Nat :=
  Nat.digits 10 n |>.count d

-- Define the range of numbers we are interested in
def range := (List.range' 1 2017)

-- Sum up the occurrences of digit 7 in the defined range
def total_occurrences (d : Nat) (range : List Nat) : Nat :=
  range.foldr (λ n acc => digit_occurrences d n + acc) 0

-- The main theorem to prove
theorem digit_7_occurrences_in_range_1_to_2017 : total_occurrences 7 range = 602 := by
  -- The proof should go here, but we only need to define the statement.
  sorry

end digit_7_occurrences_in_range_1_to_2017_l529_529221


namespace range_of_f_l529_529137

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 2 * x - 1 else -2 * x + 6

theorem range_of_f (t : ℝ) : f t > 2 ↔ t < 0 ∨ t > 3 :=
by 
  sorry

end range_of_f_l529_529137


namespace ellipse_equation_fixed_point_passes_fixed_point_l529_529647

-- The conditions defining the ellipse and the focus constraint.
theorem ellipse_equation (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) 
  (h₂ : ∃ (c : ℝ), |c + 2 * sqrt 2| / sqrt 2 = 3) 
  (h₃ : (-1, -sqrt(6)/2) ∈ { p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1 })
  : (a^2 = 4) ∧ (b^2 = 2) :=
begin
  sorry
end

-- The condition and conclusion about the fixed point for the line intersecting the ellipse.
theorem fixed_point (m : ℝ) (h₀ : ∀ (t : ℝ), t ≠ -2 → 
  ∃ (M N : ℝ × ℝ), M ≠ (-2, 0) ∧ N ≠ (-2, 0) ∧ 
  (M ≠ N) ∧ 
  (x - m * y + t = 0) ∧
  (x^2 / 4 + y^2 / 2 = 1) ∧
  ((M - (-2, 0)) • (N - (-2, 0)) = 0)) 
  : ∃ (t : ℝ), t = -2/3 :=
begin
  sorry
end

-- A statement confirming the fixed point at (-2/3, 0) on line l.
theorem passes_fixed_point (m : ℝ) (h₀ : ∀ (t : ℝ), 
  (x - m * y + t = 0) ∧ (x^2 / 4 + y^2 / 2 = 1) 
    → t = -2 / 3 ∧ (-2 / 3, 0) ∈ {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 2 = 1})
  : ∃ (p : ℝ × ℝ), p = (-2 / 3, 0) :=
begin
  sorry
end

end ellipse_equation_fixed_point_passes_fixed_point_l529_529647


namespace parabola_cubic_intersection_points_l529_529067

def parabola (x : ℝ) : ℝ := 3 * x^2 - 12 * x - 15

def cubic (x : ℝ) : ℝ := x^3 - 6 * x^2 + 11 * x - 6

theorem parabola_cubic_intersection_points :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    p1 = (-1, 0) ∧ p2 = (1, -24) ∧ p3 = (9, 162) ∧
    parabola p1.1 = p1.2 ∧ cubic p1.1 = p1.2 ∧
    parabola p2.1 = p2.2 ∧ cubic p2.1 = p2.2 ∧
    parabola p3.1 = p3.2 ∧ cubic p3.1 = p3.2 :=
by {
  -- This is the statement
  sorry
}

end parabola_cubic_intersection_points_l529_529067


namespace no_extrema_1_1_l529_529476

noncomputable def f (x : ℝ) : ℝ :=
  x^3 - 3 * x

theorem no_extrema_1_1 : ∀ x : ℝ, (x > -1) ∧ (x < 1) → ¬ (∃ c : ℝ, c ∈ Set.Ioo (-1) (1) ∧ (∀ y ∈ Set.Ioo (-1) (1), f y ≤ f c ∨ f c ≤ f y)) :=
by
  sorry

end no_extrema_1_1_l529_529476


namespace car_speeds_comparison_l529_529040

variable (u v : ℝ)

def avg_speed_car_a (u v : ℝ) : ℝ := 3 / ((1 / u) + (2 / v))

def avg_speed_car_b (u v : ℝ) : ℝ := (u + 2 * v) / 3

theorem car_speeds_comparison (u v : ℝ) : avg_speed_car_a u v ≤ avg_speed_car_b u v :=
by
  sorry

end car_speeds_comparison_l529_529040


namespace original_investment_amount_l529_529424

-- Definitions
def annual_interest_rate : ℝ := 0.04
def investment_period_years : ℝ := 0.25
def final_amount : ℝ := 10204

-- Statement to prove
theorem original_investment_amount :
  let P := final_amount / (1 + annual_interest_rate * investment_period_years)
  P = 10104 :=
by
  -- Placeholder for the proof
  sorry

end original_investment_amount_l529_529424


namespace kite_diagonals_sum_l529_529670

theorem kite_diagonals_sum (a b e f : ℝ) (h₁ : a ≥ b) 
    (h₂ : e < 2 * a) (h₃ : f < a + b) : 
    e + f < 2 * a + b := by 
    sorry

end kite_diagonals_sum_l529_529670


namespace largest_x_l529_529113

def largest_x_with_condition_eq_7_over_8 (x : ℝ) : Prop :=
  ⌊x⌋ / x = 7 / 8

theorem largest_x (x : ℝ) (h : largest_x_with_condition_eq_7_over_8 x) :
  x = 48 / 7 :=
sorry

end largest_x_l529_529113


namespace convex_polygon_diagonals_l529_529504

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) :
  (∃ m : ℕ, m = 375 ∧
    ∀ (d : ℕ), d = (nat.choose 30 2 - 30 - 30) → m = d) := 
by
  sorry

end convex_polygon_diagonals_l529_529504


namespace distinct_integer_products_of_special_fractions_l529_529980

theorem distinct_integer_products_of_special_fractions : 
  (finset.card ((finset.pi ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}.filter (λ pair, pair.1 + pair.2 = 12)
  ).image (λ (pair : ℕ × ℕ), (pair.1: ℤ) * (pair.2: ℤ)))) = 6 :=
sorry

end distinct_integer_products_of_special_fractions_l529_529980


namespace largest_real_number_condition_l529_529071

theorem largest_real_number_condition (x : ℝ) (hx : ⌊x⌋ / x = 7 / 8) : x ≤ 48 / 7 :=
by
  sorry

end largest_real_number_condition_l529_529071


namespace necessary_but_not_sufficient_for_gt_one_l529_529860

variable (x : ℝ)

theorem necessary_but_not_sufficient_for_gt_one (h : x^2 > 1) : ¬(x^2 > 1 ↔ x > 1) ∧ (x > 1 → x^2 > 1) :=
by
  sorry

end necessary_but_not_sufficient_for_gt_one_l529_529860


namespace polar_symmetric_coords_l529_529864

-- Define the structure for polar coordinates
structure PolarCoordinates where
  rho : ℝ
  theta : ℝ

-- Define the function to find symmetric point's polar coordinates
def symmetric_point (M : PolarCoordinates) : PolarCoordinates :=
  ⟨M.rho, M.theta + π⟩

-- State the theorem
theorem polar_symmetric_coords (M : PolarCoordinates) :
  symmetric_point M = ⟨M.rho, M.theta + π⟩ :=
by
  sorry

end polar_symmetric_coords_l529_529864


namespace remainder_when_divided_by_s_minus_2_l529_529847

noncomputable def f (s : ℤ) : ℤ := s^15 + s^2 + 3

theorem remainder_when_divided_by_s_minus_2 : f 2 = 32775 := 
by
  sorry

end remainder_when_divided_by_s_minus_2_l529_529847


namespace carpenter_sinks_l529_529064

theorem carpenter_sinks (houses: ℕ) (sinks_per_house: ℕ) (total_houses: ℕ) (h1: sinks_per_house = 6) (h2: total_houses = 44):
  houses * sinks_per_house = 264 :=
by
  rw [h1, h2]
  exact calc
    44 * 6 = 264 : by norm_num

end carpenter_sinks_l529_529064


namespace number_of_correct_conclusions_l529_529019

theorem number_of_correct_conclusions :
  let stmt1 : Prop := (8:ℝ)^(2/3) > (16 / 81 : ℝ)^(-3/4)
  let stmt2 : Prop := Real.log 10 > Real.log Real.exp 1
  let stmt3 : Prop := (0.8:ℝ)^(-0.1) > (0.8:ℝ)^(-0.2)
  let stmt4 : Prop := (8:ℝ)^(0.1) > (9:ℝ)^(0.1)
  (stmt1 ∧ stmt2 ∧ ¬stmt3 ∧ ¬stmt4) → 2 = 2 :=
by
  sorry

end number_of_correct_conclusions_l529_529019


namespace count_sweet_numbers_from_1_to_50_l529_529384

def sequence_next (n : ℕ) : ℕ :=
  if n <= 25 then 2 * n else n - 12

def generates_20 (n : ℕ) : Prop :=
  ∃ k, nat.iterate sequence_next k n = 20

def is_sweet (n : ℕ) : Prop :=
  ¬ generates_20 n

theorem count_sweet_numbers_from_1_to_50 : 
  finset.filter is_sweet (finset.range 51).erase 0 = 
  finset.range 51 \ {k | ∃ k ≤ 50, sequence_next k = 20} → sorry. :=
by sorry

end count_sweet_numbers_from_1_to_50_l529_529384


namespace repeating_decimal_to_fraction_in_lowest_terms_l529_529066

theorem repeating_decimal_to_fraction_in_lowest_terms : 
  (let x := 8 + 397 / 999 in x = (932 / 111)) :=
by 
  let x := 8 + 397 / 999
  have h : 8 + 397 / 999 = 8 + 397 / 999 := by rfl
  exact sorry

end repeating_decimal_to_fraction_in_lowest_terms_l529_529066


namespace trajectory_equation_perpendicularly_intersecting_points_l529_529601

noncomputable def ellipse_C : set (ℝ × ℝ) := {P | let x := P.1, y := P.2 in
  (y^2 / 4 + x^2 = 1)}

def perpendicular (A B : ℝ × ℝ) : Prop := 
  let ⟨x1, y1⟩ := A 
  let ⟨x2, y2⟩ := B in 
  x1 * x2 + y1 * y2 = 0

theorem trajectory_equation :
  ∀ P : ℝ × ℝ, (dist P (0, -sqrt 3) + dist P (0, sqrt 3) = 4) ↔ P ∈ ellipse_C :=
sorry

theorem perpendicularly_intersecting_points :
  ∀ k : ℝ, let A := (x1, k * x1 + 1)
            let B := (x2, k * x2 + 1)
            (y := k * x1 + 1) → 
            (y = k * x2 + 1) →
            (x1, k * x1 + 1) ∈ ellipse_C →
            (x2, k * x2 + 1) ∈ ellipse_C →
            perpendicular (0, 0) (x1, k * x1 + 1) (0, 0) (x2, k * x2 + 1)
            ↔ k = 1/2 ∨ k = -1/2 :=
sorry

end trajectory_equation_perpendicularly_intersecting_points_l529_529601


namespace line_equation_under_transformation_l529_529546

noncomputable def T1_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, -1],
  ![1, 0]
]

noncomputable def T2_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![2, 0],
  ![0, 3]
]

noncomputable def NM_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, -2],
  ![3, 0]
]

theorem line_equation_under_transformation :
  ∀ x y : ℝ, (∃ x' y' : ℝ, NM_matrix.mulVec ![x, y] = ![x', y'] ∧ x' = y') → 3 * x + 2 * y = 0 :=
by sorry

end line_equation_under_transformation_l529_529546


namespace part1_part2_l529_529898

noncomputable def f (x : ℝ) (θ : ℝ) : ℝ := x^2 + 2 * x * Real.tan θ - 1

def theta_domain : Set ℝ := {θ | -Real.pi / 2 < θ ∧ θ < Real.pi / 2}

def is_monotonic_on_interval (θ : ℝ) : Prop :=
  (∀ x₁ x₂ ∈ Set.Icc (-1 : ℝ) (Real.sqrt 3), x₁ ≤ x₂ → f x₁ θ ≤ f x₂ θ) ∨
  (∀ x₁ x₂ ∈ Set.Icc (-1 : ℝ) (Real.sqrt 3), x₁ ≤ x₂ → f x₁ θ ≥ f x₂ θ)

theorem part1 (θ : ℝ) :
  θ ∈ theta_domain →
  is_monotonic_on_interval θ ↔ 
  θ ∈ {θ | (Real.pi / 4 ≤ θ ∧ θ < Real.pi / 2) ∨ (-Real.pi / 2 < θ ∧ θ ≤ -Real.pi / 3)} :=
sorry

def g (θ : ℝ) : ℝ :=
  if -Real.pi / 3 ≤ θ ∧ θ < Real.pi / 4 then
    - (Real.tan θ) ^ 2 - 1
  else if Real.pi / 4 ≤ θ ∧ θ ≤ Real.pi / 3 then
    - 2 * Real.tan θ
  else
    0

theorem part2 (θ : ℝ) :
  θ ∈ Set.Icc (-Real.pi / 3) (Real.pi / 3) →
  g θ = if -Real.pi / 3 ≤ θ ∧ θ < Real.pi / 4 then
          - (Real.tan θ) ^ 2 - 1
        else if Real.pi / 4 ≤ θ ∧ θ ≤ Real.pi / 3 then
          - 2 * Real.tan θ
        else
          0 :=
sorry

end part1_part2_l529_529898


namespace number_of_valid_subsets_l529_529822

def A (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x : ℕ, x ≤ n) (Finset.range (n + 1))

def A_0 : Finset ℕ := Finset.filter (λ x : ℕ, x % 3 = 0) (A 120)
def A_1 : Finset ℕ := Finset.filter (λ x : ℕ, x % 3 = 1) (A 120)
def A_2 : Finset ℕ := Finset.filter (λ x : ℕ, x % 3 = 2) (A 120)

def three_subsets_divisible_by_3 (s : Finset (Finset ℕ)) : ℕ :=
  s.filter (λ t, t.card = 3 ∧ t.sum % 3 = 0).card

theorem number_of_valid_subsets : three_subsets_divisible_by_3 (A 120).powerset = 93640 := 
sorry

end number_of_valid_subsets_l529_529822


namespace intersection_of_M_and_N_l529_529186

def M := {0, 1, 2}
def N := {x : Int | -1 ≤ x ∧ x ≤ 1 ∧ x ∈ Int}

theorem intersection_of_M_and_N :
  M ∩ N = {0, 1} :=
sorry

end intersection_of_M_and_N_l529_529186


namespace even_factors_count_l529_529914

def n : ℕ := (2 ^ 3) * (5 ^ 1) * (11 ^ 2)

theorem even_factors_count : (∑ a in finset.range 4, ∑ b in finset.range 2, ∑ c in finset.range 3, 2 ^ a * 5 ^ b * 11 ^ c % 2 = 0) = 18 :=
sorry

end even_factors_count_l529_529914


namespace additional_hours_to_travel_l529_529787

theorem additional_hours_to_travel (distance1 time1 distance2 : ℝ) (rate : ℝ) 
  (h1 : distance1 = 270) 
  (h2 : time1 = 3)
  (h3 : distance2 = 180)
  (h4 : rate = distance1 / time1) :
  distance2 / rate = 2 := by
  sorry

end additional_hours_to_travel_l529_529787


namespace seed_selection_l529_529685

noncomputable def seed_table := [
  [84, 42, 17, 53, 31, 57, 24, 55, 06, 88, 77, 04, 74, 47, 67, 21, 76, 33, 50, 25, 83, 92, 12, 06, 76],
  [63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38, 79],
  [33, 21, 12, 34, 29, 78, 64, 56, 07, 82, 52, 42, 07, 44, 38, 15, 51, 00, 13, 42, 99, 66, 02, 79, 54]
]

def extract_seeds (table : list (list nat)) (row col : nat) : list nat :=
  let numbers := table.nth row |>.getD [] |>.drop col
  in numbers.scanl (λ acc n, if n < 100 then acc * 10 + n else n) 0

theorem seed_selection :
  extract_seeds seed_table 1 6 = [785, 567, 199] :=
by sorry

end seed_selection_l529_529685


namespace nested_sqrt_eq_l529_529306

theorem nested_sqrt_eq (n m : ℤ) (h : ∀ k : ℕ, k < 1964 → (λ x, √(x+√(x + ... + √x))) = m) : n = 0 := sorry

end nested_sqrt_eq_l529_529306


namespace smallest_number_of_integers_l529_529413

theorem smallest_number_of_integers (a b n : ℕ) 
  (h_avg_original : 89 * n = 73 * a + 111 * b) 
  (h_group_sum : a + b = n)
  (h_ratio : 8 * a = 11 * b) : 
  n = 19 :=
sorry

end smallest_number_of_integers_l529_529413


namespace find_100th_term_l529_529710

def sequence_term (n : ℕ) : ℕ :=
  let ⟨k, hk⟩ := Nat.exists_unique (λ k, n < k * (k + 1) / 2) in k

theorem find_100th_term : sequence_term 100 = 14 :=
  sorry

end find_100th_term_l529_529710


namespace ratio_of_couch_to_table_l529_529615

theorem ratio_of_couch_to_table
    (C T X : ℝ)
    (h1 : T = 3 * C)
    (h2 : X = 300)
    (h3 : C + T + X = 380) :
  X / T = 5 := 
by 
  sorry

end ratio_of_couch_to_table_l529_529615


namespace profit_ratio_l529_529290

noncomputable def effective_capital (investment : ℕ) (months : ℕ) : ℕ := investment * months

theorem profit_ratio : 
  let P_investment := 4000
  let P_months := 12
  let Q_investment := 9000
  let Q_months := 8
  let P_effective := effective_capital P_investment P_months
  let Q_effective := effective_capital Q_investment Q_months
  (P_effective / Nat.gcd P_effective Q_effective) = 2 ∧ (Q_effective / Nat.gcd P_effective Q_effective) = 3 :=
sorry

end profit_ratio_l529_529290


namespace limit_of_sum_l529_529805

noncomputable def limit_expression (k : ℕ) : ℝ :=
  ∑ i in (finset.range n), ((i + 1) ^ k) / (n ^ (k + 1))

theorem limit_of_sum (k : ℕ) : 
  (Real.seq_limit (λ n : ℕ, (1 / n) * (∑ i in (finset.range n), (i.succ ^ k / n ^ k))) (1 / (k + 1))) :=
begin
  sorry
end

end limit_of_sum_l529_529805


namespace eval_infinite_sum_l529_529065

noncomputable def infinite_sum : ℝ := ∑' n, (n:ℝ) / (n^4 + 9)

theorem eval_infinite_sum : infinite_sum = 7 / 9 :=
sorry

end eval_infinite_sum_l529_529065


namespace Mike_journey_correct_l529_529284

-- Define types for locations and times
structure Location where
  distance_from_home : ℝ

structure Time where
  time_elapsed : ℝ

-- Define Mike's journey
noncomputable def Mike_journey : Time → Location :=
  λ t, if t.time_elapsed < park_time then Location.mk (park_slope * t.time_elapsed)
       else if t.time_elapsed < park_time + river_time then
         Location.mk (park_distance + river_slope * (t.time_elapsed - park_time))
       else if t.time_elapsed < park_time + river_time + cafe_break then
         Location.mk (park_distance + river_distance)
       else if t.time_elapsed < park_time + river_time + cafe_break + return_river_time then
         Location.mk (park_distance + river_distance - river_slope * (t.time_elapsed - park_time - river_time - cafe_break))
       else if t.time_elapsed < park_time + river_time + cafe_break + return_river_time + rest_break then
         Location.mk (park_distance + river_distance - river_distance)
       else Location.mk ((park_distance + river_distance - river_distance) - park_slope * (t.time_elapsed - park_time - river_time - cafe_break - return_river_time - rest_break))

-- Conditions
def park_time : ℝ := 1   -- Placeholder
def river_time : ℝ := 1  -- Placeholder
def rest_break : ℝ := 0.5
def cafe_break : ℝ := 2

def park_slope : ℝ := 2    -- Placeholder
def river_slope : ℝ := 4   -- Placeholder
def park_distance : ℝ := park_slope * park_time
def river_distance : ℝ := river_slope * river_time

-- The proof problem in Lean
theorem Mike_journey_correct :
  -- Starting and ending at zero
  ∀ t : Time, t.time_elapsed = 0 → Mike_journey t = Location.mk 0 ∧
              t.time_elapsed = (2*park_time + 2*river_time + cafe_break + rest_break) → Mike_journey t = Location.mk 0 :=
  sorry

end Mike_journey_correct_l529_529284


namespace Lidex_dist_scheme_count_l529_529213

noncomputable def distribution_schemes (females males : ℕ) : ℕ :=
  let total_schemes := (2 ^ females) - 2
  in 2 * total_schemes

theorem Lidex_dist_scheme_count : distribution_schemes 5 2 = 60 :=
by
  sorry

end Lidex_dist_scheme_count_l529_529213


namespace number_of_valid_x_l529_529329

-- Define the operation ⋆ as described in the problem
def star (a b : ℤ) : ℤ := (a ^ 2) / b

-- Define a function that checks if star(a, b) is a positive integer
def isPositiveInteger (a b : ℤ) : Prop := star a b > 0 ∧ ∃ k : ℤ, star a b = k

-- Define the main theorem stating that the number of integer values of x making 15 ⋆ x a positive integer is 9
theorem number_of_valid_x : ∃ n : ℕ, n = 9 ∧ ∀ x : ℤ, isPositiveInteger 15 x ↔ x ∣ 225 :=
by
  exists 9
  intros x
  constructor
  sorry  -- Proof required here

end number_of_valid_x_l529_529329


namespace largest_x_satisfies_condition_l529_529093

theorem largest_x_satisfies_condition :
  ∃ x : ℝ, (⌊x⌋ / x = 7 / 8) ∧ (∀ y : ℝ, (⌊y⌋ / y = 7 / 8) → y ≤ 48 / 7) :=
sorry

end largest_x_satisfies_condition_l529_529093


namespace conjugate_of_z_l529_529880

-- Conditions and definitions
def i : ℂ := complex.I
def z : ℂ := (3 + i) / (2 - i)
def conj_z := complex.conj z

-- Theorem statement
theorem conjugate_of_z : conj_z = 1 - i :=
by
  sorry

end conjugate_of_z_l529_529880


namespace solution_l529_529534

variable (a : ℝ)

def p : Prop := ∀ x ∈ set.Icc 1 2, x^2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem solution : p a ∧ q a → a ≤ -2 ∨ a = 1 :=
begin
  sorry
end

end solution_l529_529534


namespace lisa_marbles_l529_529981

theorem lisa_marbles : 
  let number_of_friends := 12
  let initial_marbles := 50
  let total_marbles_needed := (number_of_friends * (number_of_friends + 1)) / 2
  total_marbles_needed - initial_marbles = 28 :=
by
  let number_of_friends := 12
  let initial_marbles := 50
  let total_marbles_needed := (number_of_friends * (number_of_friends + 1)) / 2
  show total_marbles_needed - initial_marbles = 28 from sorry

end lisa_marbles_l529_529981


namespace fraction_sum_equals_mixed_number_l529_529732

theorem fraction_sum_equals_mixed_number :
  (3 / 5 : ℚ) + (2 / 3) + (16 / 15) = (7 / 3) :=
by sorry

end fraction_sum_equals_mixed_number_l529_529732


namespace bus_total_capacity_l529_529597

def bus_capacity_lower_level (left_seats right_seats back_seat standing : ℕ) : ℕ :=
  left_seats * 3 + right_seats * 3 + back_seat + standing

def bus_capacity_upper_level (left_seats non_reserved_right_seats reserved_right_seats reserved_capacity standing : ℕ) : ℕ :=
  left_seats * 2 + non_reserved_right_seats * 2 + reserved_right_seats * reserved_capacity + standing

theorem bus_total_capacity :
  bus_capacity_lower_level 15 12 11 12 + bus_capacity_upper_level 20 13 5 4 8 = 198 :=
by
  calc
    bus_capacity_lower_level 15 12 11 12
      = 15 * 3 + 12 * 3 + 11 + 12 : by rfl
    ... = 45 + 36 + 11 + 12 : by rfl
    ... = 104 : by ring
    ... + bus_capacity_upper_level 20 13 5 4 8
      = 104 + (20 * 2 + 13 * 2 + 5 * 4 + 8) : by rfl
    ... = 104 + (40 + 26 + 20 + 8) : by rfl
    ... = 104 + 94 : by ring
    ... = 198 : by ring

end bus_total_capacity_l529_529597


namespace freshmen_sophomores_without_pets_l529_529347

-- Definitions from conditions
def total_students : ℕ := 600
def percent_freshmen : ℝ := 0.30
def percent_sophomores : ℝ := 0.25
def percent_freshmen_with_cat : ℝ := 0.40
def percent_sophomores_with_dog : ℝ := 0.30
def percent_with_both_pets : ℝ := 0.10

-- Calculations based on conditions
def number_freshmen : ℕ := (percent_freshmen * total_students).to_nat
def number_sophomores : ℕ := (percent_sophomores * total_students).to_nat
def total_freshmen_and_sophomores : ℕ := number_freshmen + number_sophomores

def number_freshmen_with_cat : ℕ := (percent_freshmen_with_cat * number_freshmen).to_nat
def number_sophomores_with_dog : ℕ := (percent_sophomores_with_dog * number_sophomores).to_nat
def number_with_both_pets : ℕ := (percent_with_both_pets * total_freshmen_and_sophomores).to_nat

def total_with_pets : ℕ := number_freshmen_with_cat + number_sophomores_with_dog - number_with_both_pets
def number_without_pets : ℕ := total_freshmen_and_sophomores - total_with_pets

theorem freshmen_sophomores_without_pets : number_without_pets = 246 := by
  -- Proof is omitted
  sorry

end freshmen_sophomores_without_pets_l529_529347


namespace expected_prize_winners_l529_529596

noncomputable def harmonic_number (N : ℕ) : ℝ :=
  ∑ i in finset.range (N + 1), (1 : ℝ) / (i + 1)

theorem expected_prize_winners (N : ℕ) : 
  ∃ (H_N : ℝ), 
  H_N = ∑ i in finset.range N, (1 : ℝ) / (i + 1)
:= sorry

end expected_prize_winners_l529_529596


namespace min_value_of_algebraic_sum_l529_529314

theorem min_value_of_algebraic_sum 
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : a + 3 * b = 3) :
  ∃ (min_value : ℝ), min_value = 16 / 3 ∧ (∀ a b, a > 0 → b > 0 → a + 3 * b = 3 → 1 / a + 3 / b ≥ min_value) :=
sorry

end min_value_of_algebraic_sum_l529_529314


namespace paper_area_l529_529423

theorem paper_area (L W : ℝ) 
(h1 : 2 * L + W = 34) 
(h2 : L + 2 * W = 38) : 
L * W = 140 := by
  sorry

end paper_area_l529_529423


namespace locus_of_points_l529_529366

def point := (ℝ × ℝ)

variables (F_1 F_2 : point) (r k : ℝ)

def distance (P Q : point) : ℝ :=
  ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)^(1/2)

def on_circle (P : point) (center : point) (radius : ℝ) : Prop :=
  distance P center = radius

theorem locus_of_points
  (P : point)
  (r1 r2 PF1 PF2 : ℝ)
  (h_pF1 : r1 = distance P F_1)
  (h_pF2 : PF2 = distance P F_2)
  (h_outside_circle : PF2 = r2 + r)
  (h_inside_circle : PF2 = r - r2)
  (h_k : r1 + PF2 = k) :
  (∀ P, distance P F_1 + distance P F_2 = k →
  ( ∃ e_ellipse : Prop, on_circle P F_2 r → e_ellipse) ∨ 
  ( ∃ h_hyperbola : Prop, on_circle P F_2 r → h_hyperbola)) :=
by
  sorry

end locus_of_points_l529_529366


namespace largest_x_exists_largest_x_largest_real_number_l529_529078

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : x ≤ 48 / 7 :=
sorry

theorem exists_largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  ∃ x, (⌊x⌋ : ℝ) / x = 7 / 8 ∧ x = 48 / 7 :=
sorry

theorem largest_real_number (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  x = 48 / 7 :=
sorry

end largest_x_exists_largest_x_largest_real_number_l529_529078


namespace xiao_zhang_round_trip_time_l529_529378

def distance : ℝ := 240
def speed_uphill : ℝ := 30
def speed_downhill : ℝ := 40
def time_difference : ℝ := 10 / 60

theorem xiao_zhang_round_trip_time :
  let D := distance / 2 in
  let time_forward := (D / speed_uphill) + (D / speed_downhill) in
  let time_return := time_forward - time_difference in
  time_forward + time_return = 7 :=
by
  sorry

end xiao_zhang_round_trip_time_l529_529378


namespace largest_x_satisfies_condition_l529_529092

theorem largest_x_satisfies_condition :
  ∃ x : ℝ, (⌊x⌋ / x = 7 / 8) ∧ (∀ y : ℝ, (⌊y⌋ / y = 7 / 8) → y ≤ 48 / 7) :=
sorry

end largest_x_satisfies_condition_l529_529092


namespace cos_triple_angle_l529_529920

theorem cos_triple_angle (α : ℝ) (h : cos α = -1 / 2) : cos (3 * α) = 1 := by
  sorry

end cos_triple_angle_l529_529920


namespace Alla_Boris_meet_l529_529793

theorem Alla_Boris_meet : 
  ∀ (n m : ℕ), 
    let intervals := 399 in
    let alla_start := 1 in
    let boris_start := 400 in
    let alla_pos_55 := 55 in
    let boris_pos_321 := 321 in
    alla_pos_55 - alla_start = 54 ∧ 
    boris_start - boris_pos_321 = 79 → 
    let speed_ratio := 79 / 54 in
    let x := 399 * 54 / 133 in
    alla_start + x = 163 := 
by
  intros n m
  let intervals := 399
  let alla_start := 1
  let boris_start := 400
  let alla_pos_55 := 55
  let boris_pos_321 := 321
  assume h : alla_pos_55 - alla_start = 54 ∧ boris_start - boris_pos_321 = 79
  let speed_ratio := 79 / 54
  let x := 399 * 54 / 133
  have h1 : alla_pos_55 - alla_start = 54, from h.left
  have h2 : boris_start - boris_pos_321 = 79, from h.right
  have meet_point := alla_start + x
  show meet_point = 163, from sorry

end Alla_Boris_meet_l529_529793


namespace total_amount_shared_l529_529209

theorem total_amount_shared (x y z : ℝ) (h1 : x = 1.25 * y) (h2 : y = 1.20 * z) (hz : z = 100) :
  x + y + z = 370 := by
  sorry

end total_amount_shared_l529_529209


namespace valid_number_count_l529_529575

def is_valid_number (n : ℕ) : Prop :=
  let a := (n / 1000) % 10
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  a = 2 ∧ b + c = 9 * d + (d / 10)

def count_valid_numbers : ℕ :=
  (List.range 1000).countp (λ n, is_valid_number (2000 + n))

theorem valid_number_count : count_valid_numbers = 100 :=
sorry

end valid_number_count_l529_529575


namespace part1_solution_part2_solution_l529_529136

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) * x + abs (x - 2) * (x - a)

theorem part1_solution (a : ℝ) (h : a = 1) :
  {x : ℝ | f x a < 0} = {x : ℝ | x < 1} :=
by
  sorry

theorem part2_solution (x : ℝ) (hx : x < 1) :
  {a : ℝ | f x a < 0} = {a : ℝ | 1 ≤ a} :=
by
  sorry

end part1_solution_part2_solution_l529_529136


namespace Vasya_wins_game_l529_529291

theorem Vasya_wins_game (s : Finset (Finset (Fin 10))) :
  (∀ x : ℝ, (∀ i j, 0 ≤ x) → (∀ i j, x i ≤ x j) → 
  (∑ t in s, (∏ i in t, x i) > ∑ t in s.filter(λ t, ¬(t in s)), (∏ i in t, x i))) := 
sorry

end Vasya_wins_game_l529_529291


namespace luke_clothing_loads_luke_clothing_loads_dec_l529_529650

theorem luke_clothing_loads (total_clothing : ℕ) (first_load : ℕ) (remaining_loads : ℕ) (num_small_loads : ℕ) :
  total_clothing = 135 → 
  first_load = 29 → 
  num_small_loads = 7 →
  remaining_loads = (total_clothing - first_load) / num_small_loads →
  remaining_loads = 15 :=
by 
  intro h_total h_first h_num h_rem 
  rw [h_total, h_first, h_num] at h_rem
  norm_num at h_rem
  exact h_rem

theorem luke_clothing_loads_dec (total_clothing first_load remaining_clothing num_small_loads remain_per_load : ℕ)
  (zLtRem : ℕ) (lwF : ℕ) 
  (total_def : total_clothing = 135) (f_def : first_load = 29) (num_sl_def : num_small_loads = 7)
  (rem_cl_def : remaining_clothing = total_clothing - first_load)
  (rem_def : remain_per_load = remaining_clothing / num_small_loads) :
  remain_per_load = 15 :=
by 
  rw [total_def, f_def, num_sl_def] at rem_cl_def
  rw [rem_cl_def, num_sl_def, f_def, total_def] at rem_def 
  norm_num at rem_def 
  exact rem_def 

end luke_clothing_loads_luke_clothing_loads_dec_l529_529650


namespace triangle_type_and_area_l529_529340

theorem triangle_type_and_area (x : ℝ) (hpos : 0 < x) (h : 3 * x + 4 * x + 5 * x = 36) :
  let a := 3 * x
  let b := 4 * x
  let c := 5 * x
  a^2 + b^2 = c^2 ∧ (1 / 2) * a * b = 54 :=
by {
  sorry
}

end triangle_type_and_area_l529_529340


namespace count_positive_integers_l529_529331

def star (a b : ℤ) : ℤ := a^2 / b

theorem count_positive_integers (count_x : ℤ) :
  (∃ count_x, (∀ x, 15 ^ 2 = n * x → (15 ^ 2) / x > 0)) → 
  count_x = 9 :=
sorry

end count_positive_integers_l529_529331


namespace monotonic_intervals_min_value_x1_x2_l529_529899

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x - (1/2) * a * x^2 + (1 - a) * x + 1

noncomputable def g (x : ℝ) : ℝ := log x + x^2 + x - (11 / 2)

theorem monotonic_intervals (a : ℝ) : 
  if a ≤ 0 then 
    ∀ x : ℝ, 0 < x → (f x a) > (f x a) 
  else 
    ∀ x : ℝ, 0 < x ∧ x < (1 / a) → (f x a) > (f x a) ∧ 
             ∀ x : ℝ, x > (1 / a) → (f x a) < (f x a) :=
sorry

theorem min_value_x1_x2 (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : g x1 + g x2 + x1 * x2 = 0) : 
  x1 + x2 ≥ 3 :=
sorry

end monotonic_intervals_min_value_x1_x2_l529_529899


namespace sharks_problem_l529_529279

variable (F : ℝ)
variable (S : ℝ := 0.25 * (F + 3 * F))
variable (total_sharks : ℝ := 15)

theorem sharks_problem : 
  (0.25 * (F + 3 * F) = 15) ↔ (F = 15) :=
by 
  sorry

end sharks_problem_l529_529279


namespace number_decomposition_l529_529348

theorem number_decomposition :
  ∃ (A B : ℕ), (B = 3 * A) ∧ 
  (∑ i in finset.range 10, i) = 45 ∧ 
  (∑ i in finset.filter (λ x, x < 10) (finset.range 100), i / 10 + (i % 10)) = 450 ∧
  (B % 9 = 0) ∧
  (∃ k > 1, B = 3 * 3 * 3 * k) :=
by
  sorry

end number_decomposition_l529_529348


namespace derivative_of_y_l529_529068

noncomputable def y (x : ℝ) : ℝ :=
  sqrt (Real.tan 4) + (Real.sin (21 * x))^2 / (21 * Real.cos (42 * x))

theorem derivative_of_y (x : ℝ) :
  deriv y x = 2 * Real.tan (42 * x) * Real.sec (42 * x) :=
by sorry

end derivative_of_y_l529_529068


namespace daps_for_36_dups_l529_529583

variables (daps dops dips dups : ℝ)

-- Given conditions
def condition1 := (5 * daps = 4 * dops)
def condition2 := (3 * dops = 9 * dips)
def condition3 := (5 * dips = 2 * dups)

-- Theorem to prove
theorem daps_for_36_dups
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3) :
  ∃ daps, (36 * dups = 37.5 * daps) :=
sorry

end daps_for_36_dups_l529_529583


namespace AngiesClassGirlsCount_l529_529026

theorem AngiesClassGirlsCount (n_girls n_boys : ℕ) (total_students : ℕ)
  (h1 : n_girls = 2 * (total_students / 5))
  (h2 : n_boys = 3 * (total_students / 5))
  (h3 : n_girls + n_boys = 20)
  : n_girls = 8 :=
by
  sorry

end AngiesClassGirlsCount_l529_529026


namespace stock_loss_l529_529385

theorem stock_loss (a : ℝ) (n : ℕ) : 0 < a → (a * (0.99)^n) < a :=
by
  intros ha
  have hn : 0.99^n < 1 := by
    induction n with
    | zero => simp
    | succ n ih => calc
      0.99^n * 0.99 < 1 * 0.99 := by
        apply mul_lt_mul_of_pos_right
        exact ih
        norm_num
  calc
  a * (0.99)^n < a * 1 := by
    apply mul_lt_mul_of_pos_left
    exact hn
    exact ha
  simp

end stock_loss_l529_529385


namespace solve_for_lambda_l529_529570

def vector_dot_product : (ℤ × ℤ) → (ℤ × ℤ) → ℤ
| (x1, y1), (x2, y2) => x1 * x2 + y1 * y2

theorem solve_for_lambda
  (a : ℤ × ℤ) (b : ℤ × ℤ) (lambda : ℤ)
  (h1 : a = (3, -2))
  (h2 : b = (1, 2))
  (h3 : vector_dot_product (a.1 + lambda * b.1, a.2 + lambda * b.2) a = 0) :
  lambda = 13 :=
sorry

end solve_for_lambda_l529_529570


namespace cashier_opens_probability_l529_529446

-- Definition of the timeline and arrival times
variables {x₁ x₂ x₃ x₄ x₅ x₆ : ℝ}
-- Condition that all arrival times are between 0 and 15 minutes
def arrival_times_within_bounds : Prop := 
  0 ≤ x₁ ∧ x₁ ≤ 15 ∧ 
  0 ≤ x₂ ∧ x₂ ≤ 15 ∧
  0 ≤ x₃ ∧ x₃ ≤ 15 ∧
  0 ≤ x₄ ∧ x₄ ≤ 15 ∧
  0 ≤ x₅ ∧ x₅ ≤ 15 ∧
  0 ≤ x₆ ∧ x₆ ≤ 15

-- Condition that the Scientist arrives last
def scientist_arrives_last : Prop := 
  x₁ < x₆ ∧ x₂ < x₆ ∧ x₃ < x₆ ∧ x₄ < x₆ ∧ x₅ < x₆

-- Event A: Cashier opens no later than 3 minutes after the Scientist arrives, i.e., x₆ ≥ 12
def event_A : Prop := x₆ ≥ 12

-- The correct answer
theorem cashier_opens_probability :
  arrival_times_within_bounds ∧ scientist_arrives_last → 
  Pr[x₆ ≥ 12 | x₁, x₂, x₃, x₄, x₅ < x₆] = 0.738 :=
sorry

end cashier_opens_probability_l529_529446


namespace Buddha_hand_fruits_problems_l529_529590

/-- Given 8 Buddha's hand fruits with their weight deviations from the standard weight of 0.5 kg
    recorded as: 0.1, 0, -0.05, -0.25, 0.15, 0.2, 0.05, -0.1 (kg). Prove the following:

    1. The heaviest fruit is 0.45 kg heavier than the lightest fruit.
    2. The total deviation of these fruits' weight from the standard sum is 0.1 kg.
    3. If the selling price is ¥42 per kg, the farmer's earnings from selling these fruits are ¥172.2.
-/
theorem Buddha_hand_fruits_problems :
  let deviations := [0.1, 0, -0.05, -0.25, 0.15, 0.2, 0.05, -0.1] in
  let standard_weight := 0.5 in
  let price_per_kg := 42 in
  let heaviest := list.maximum deviations in
  let lightest := list.minimum deviations in
  heaviest - lightest = 0.45 ∧
  list.sum deviations = 0.1 ∧
  ((8 * standard_weight) + (list.sum deviations)) * price_per_kg = 172.2 :=
by
  sorry

end Buddha_hand_fruits_problems_l529_529590


namespace find_a_l529_529561

noncomputable def f (x a : ℝ) : ℝ := x + Real.exp (x - a)
noncomputable def g (x a : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem find_a (x0 a : ℝ) (h : f x0 a - g x0 a = 3) : a = -1 - Real.log 2 := sorry

end find_a_l529_529561


namespace range_of_g_l529_529060

def g : ℝ → ℝ := λ x, 1 / x^2 + 5

theorem range_of_g : Set.Ioi 5 = Set.range g :=
by
  sorry

end range_of_g_l529_529060


namespace bounded_derivative_l529_529959

open Real

theorem bounded_derivative
  (f : ℝ → ℝ)
  (M : ℝ)
  (h1 : ∀ x, |f x| ≤ M)
  (h2 : ∀ x, |f'' x| ≤ M)
  (h3 : Continuous f'') :
  ∃ C, ∀ x, |f' x| ≤ C := 
sorry

end bounded_derivative_l529_529959


namespace solve_system_of_equations_solve_system_of_inequalities_l529_529676

-- For the system of equations
theorem solve_system_of_equations (x y : ℝ) (h1 : 3 * x + 4 * y = 2) (h2 : 2 * x - y = 5) : 
    x = 2 ∧ y = -1 :=
sorry

-- For the system of inequalities
theorem solve_system_of_inequalities (x : ℝ) 
    (h1 : x - 3 * (x - 1) < 7) 
    (h2 : x - 2 ≤ (2 * x - 3) / 3) :
    -2 < x ∧ x ≤ 3 :=
sorry

end solve_system_of_equations_solve_system_of_inequalities_l529_529676


namespace sakshi_work_days_l529_529992

theorem sakshi_work_days (x : ℝ) (efficiency_tanya : ℝ) (days_tanya : ℝ) 
  (h_efficiency : efficiency_tanya = 1.25) 
  (h_days : days_tanya = 4)
  (h_relationship : x / efficiency_tanya = days_tanya) : 
  x = 5 :=
by 
  -- Lean proof would go here
  sorry

end sakshi_work_days_l529_529992


namespace ratio_owners_on_horse_l529_529401

-- Definitions based on the given conditions.
def number_of_horses : Nat := 12
def number_of_owners : Nat := 12
def total_legs_walking_on_ground : Nat := 60
def owner_leg_count : Nat := 2
def horse_leg_count : Nat := 4
def total_owners_leg_horse_count : Nat := owner_leg_count + horse_leg_count

-- Prove the ratio of the number of owners on their horses' back to the total number of owners is 1:6
theorem ratio_owners_on_horse (R W : Nat) 
  (h1 : R + W = number_of_owners)
  (h2 : total_owners_leg_horse_count * W = total_legs_walking_on_ground) :
  R = 2 → W = 10 → (R : Nat)/(number_of_owners : Nat) = (1 : Nat)/(6 : Nat) := 
sorry

end ratio_owners_on_horse_l529_529401


namespace sum_of_all_real_x_l529_529823

theorem sum_of_all_real_x (x : ℝ) :
  (x^2 - 5 * x + 3) ^ (x^2 - 6 * x + 3) = 1 → 
  (∑ x : {x : ℝ // (x^2 - 5 * x + 3) ^ (x^2 - 6 * x + 3) = 1}, x.val) = 16 := 
by 
  sorry

end sum_of_all_real_x_l529_529823


namespace angle_ADB_eq_2_angle_BEC_l529_529301

open EuclideanGeometry

-- Define the quadrilateral and relevant points
variables {P: Type} [EuclideanSpace P] 
variables {A B C D E: P}
variables (AB CD: Line P)
variables (a1 a2: ℝ)
hypotheses
  (angle_DAC_eq_60 : ∠ A D C = 60)
  (angle_CAB_eq_60 : ∠ C A B = 60)
  (AB_eq_BD_minus_AC : dist A B = dist B D - dist A C)
  (AB_intersects_CD_at_E : intersects AB CD E)

theorem angle_ADB_eq_2_angle_BEC (h: a1 = ∠ A D B) (ha2: a2 = ∠ B E C) :  a1 = 2 * a2 :=
by sorry

end angle_ADB_eq_2_angle_BEC_l529_529301


namespace prod_ge_power_of_sqrt_prod_l529_529643

theorem prod_ge_power_of_sqrt_prod (n : Nat) (m : ℝ) (x : Fin n → ℝ) 
    (h_pos : ∀ i, 0 < x i) (h_m : 0 < m) : 
    ∏ i, (m + x i) ≥ (m + real.sqrt (∏ i, x i)) ^ n := 
begin
  -- Proof goes here
  sorry
end

end prod_ge_power_of_sqrt_prod_l529_529643


namespace volume_ratio_l529_529780

theorem volume_ratio (w l h : ℝ) (V_sphere V_prism : ℝ)
  (h1 : l = 2 * w)
  (h2 : h = 2 * w)
  (h3 : V_sphere = (4 / 3) * real.pi * (w / 2)^3)
  (h4 : V_prism = l * w * h) :
  V_sphere / V_prism = real.pi / 24 := 
sorry

end volume_ratio_l529_529780


namespace perimeter_ratio_l529_529753

variables (s : ℝ)

def smaller_square_diagonal := s * real.sqrt 2
def larger_square_diagonal := 5 * smaller_square_diagonal s

def smaller_square_perimeter := 4 * s
def larger_square_perimeter := 4 * (5 * s)

theorem perimeter_ratio : larger_square_perimeter s / smaller_square_perimeter s = 5 :=
by sorry

end perimeter_ratio_l529_529753


namespace solution_pairs_l529_529478

open Int

theorem solution_pairs (a b : ℝ) (h : ∀ n : ℕ, n > 0 → a * ⌊b * n⌋ = b * ⌊a * n⌋) :
  a = 0 ∨ b = 0 ∨ a = b ∨ (∃ (a_int b_int : ℤ), a = a_int ∧ b = b_int) :=
by sorry

end solution_pairs_l529_529478


namespace investment_accumulation_l529_529796

variable (P : ℝ) -- Initial investment amount
variable (r1 r2 r3 : ℝ) -- Interest rates for the first 3 years
variable (r4 : ℝ) -- Interest rate for the fourth year
variable (r5 : ℝ) -- Interest rate for the fifth year

-- Conditions
def conditions : Prop :=
  r1 = 0.07 ∧ 
  r2 = 0.08 ∧
  r3 = 0.10 ∧
  r4 = r3 + r3 * 0.12 ∧
  r5 = r4 - r4 * 0.08

-- The accumulated amount after 5 years
def accumulated_amount : ℝ :=
  P * (1 + r1) * (1 + r2) * (1 + r3) * (1 + r4) * (1 + r5)

-- Proof problem
theorem investment_accumulation (P : ℝ) :
  conditions r1 r2 r3 r4 r5 → 
  accumulated_amount P r1 r2 r3 r4 r5 = 1.8141 * P := by
  sorry

end investment_accumulation_l529_529796


namespace area_of_region_l529_529810

theorem area_of_region (center_C : ℝ × ℝ) (center_D : ℝ × ℝ) (radius_C radius_D : ℝ)
  (hC : center_C = (3, 5))
  (hC_radius : radius_C = 5)
  (hD : center_D = (15, 5))
  (hD_radius : radius_D = 3) :
  let rect_area := 12 * 5
      semi_circle_C_area := (radius_C ^ 2) * π / 2
      semi_circle_D_area := (radius_D ^ 2) * π / 2 in
  rect_area - (semi_circle_C_area + semi_circle_D_area) = 60 - 17 * π :=
by
  sorry

end area_of_region_l529_529810


namespace minimum_xy_l529_529856

theorem minimum_xy (x y : ℝ) 
  (h : 1 + cos (x + y - 1)^2 = (x^2 + y^2 + 2 * (x + 1) * (1 - y)) / (x - y + 1)) : 
  ∃ k : ℤ, xy ≥ 1/4 := 
by 
  sorry

end minimum_xy_l529_529856


namespace solve_cubic_equation_l529_529491

theorem solve_cubic_equation (x : ℚ) : (∃ x : ℚ, ∛(5 - x) = -5/3 ∧ x = 260/27) :=
by
  sorry

end solve_cubic_equation_l529_529491


namespace solve_cubic_equation_l529_529490

theorem solve_cubic_equation (x : ℚ) : (∃ x : ℚ, ∛(5 - x) = -5/3 ∧ x = 260/27) :=
by
  sorry

end solve_cubic_equation_l529_529490


namespace binary_digit_sum_property_l529_529848

def binary_digit_sum (n : Nat) : Nat :=
  n.digits 2 |>.foldr (· + ·) 0

theorem binary_digit_sum_property (k : Nat) (h_pos : 0 < k) :
  (Finset.range (2^k)).sum (λ n => binary_digit_sum (n + 1)) = 2^(k - 1) * k + 1 := 
sorry

end binary_digit_sum_property_l529_529848


namespace CandyGivenToJanetEmily_l529_529126

noncomputable def initial_candy : ℝ := 78.5
noncomputable def candy_left_after_janet : ℝ := 68.75
noncomputable def candy_given_to_emily : ℝ := 2.25

theorem CandyGivenToJanetEmily :
  initial_candy - candy_left_after_janet + candy_given_to_emily = 12 := 
by
  sorry

end CandyGivenToJanetEmily_l529_529126


namespace math_equiv_proof_l529_529225

-- Defining the parametric equation of the line l
def line_l (t : ℝ) (α : ℝ) : ℝ × ℝ :=
  (-1 + t * Real.cos α, 1/2 + t * Real.sin α)

-- Defining the polar equation of the curve C
def curve_C (ρ : ℝ) (θ : ℝ) : Prop :=
  ρ^2 = 4 / (4 * Real.sin θ ^ 2 + Real.cos θ ^ 2)

-- Defining point P
def P := (-1 : ℝ, 1 / 2 : ℝ)

-- The Cartesian coordinate equation of curve C
def cartesian_curve_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

-- The range of values for |PA|⋅|PB|
def range_PA_PB : set ℝ :=
  set.Icc (1/2 : ℝ) (2 : ℝ)

-- Main theorem statement
theorem math_equiv_proof :
  (∀ θ ρ, curve_C ρ θ → (∃ x y, cartesian_curve_C x y ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ)) ∧
  (∀ α, ∃ A B, (let (x1, y1) := line_l (-1) α in
                 let (x2, y2) := line_l (1) α in
                 cartesian_curve_C x1 y1 ∧ cartesian_curve_C x2 y2 ∧
                 ∀ PA PB : ℝ, PA = Real.dist P (x1, y1) ∧ PB = Real.dist P (x2, y2) →
                 |PA * PB| ∈ range_PA_PB)) :=
by sorry

end math_equiv_proof_l529_529225


namespace largest_x_eq_48_div_7_l529_529112

theorem largest_x_eq_48_div_7 :
  ∃ x : ℝ, (⟨floor x / x⟩ = 7 / 8) ∧ (x = 48 / 7) := 
begin
  sorry
end

end largest_x_eq_48_div_7_l529_529112


namespace negation_of_proposition_l529_529729

theorem negation_of_proposition (a b c : ℝ) (h1 : a + b + c ≥ 0) (h2 : abc ≤ 0) : 
  (∃ x y z : ℝ, (x < 0) ∧ (y < 0) ∧ (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ (x ≠ y)) →
  ¬(∀ x y z : ℝ, (x < 0 ∨ y < 0 ∨ z < 0) → (x ≠ y → x ≠ z → y ≠ z → (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ (z = a ∨ z = b ∨ z = c))) :=
sorry

end negation_of_proposition_l529_529729


namespace largest_real_solution_l529_529099

theorem largest_real_solution (x : ℝ) (h : (⌊x⌋ / x = 7 / 8)) : x ≤ 48 / 7 := by
  sorry

end largest_real_solution_l529_529099


namespace perimeter_shaded_region_l529_529217

-- Definitions of the given conditions
def radius : ℝ := 8
def angle_BOC : ℝ := 240
def circle_radius (A B : ℝ) := radius
def central_angle (B C : ℝ) := angle_BOC

-- Lean statement for the problem
theorem perimeter_shaded_region :
  let AB := radius,
      AC := radius,
      arc_BC := (2 / 3) * (2 * Real.pi * radius) in
  AB + AC + arc_BC = 16 + (32 / 3) * Real.pi :=
by
  -- sorry is used here as placeholder 
  sorry

end perimeter_shaded_region_l529_529217


namespace bela_always_wins_l529_529051

noncomputable def game_on_interval (n : ℕ) (h : n > 10) : Prop :=
  ∀ (moves : (ℕ → ℝ)) (valid_move : ℕ → Prop),
  (valid_move 0) ∧ 
  (∀ i, valid_move i → valid_move (i + 1) →
    abs (moves (i + 1) - moves i) > 1 ∧ abs (moves (i + 1) - moves i) < 3) →
  ∃ k, ∀ j, j > k → ¬ ∃ y, valid_move y

theorem bela_always_wins (n : ℕ) (h : n > 10) : game_on_interval n h :=
begin
  sorry,
end

end bela_always_wins_l529_529051


namespace probability_books_next_to_each_other_l529_529687

theorem probability_books_next_to_each_other (A : set (Fin 10)) :
  let n := 10!
  let m := 9! * 2
  (m / n : ℝ) = 0.2 := 
by 
  sorry

end probability_books_next_to_each_other_l529_529687


namespace find_m_and_e_l529_529262

theorem find_m_and_e (m e : ℕ) (hm : 0 < m) (he : e < 10) 
(h1 : 4 * m^2 + m + e = 346) 
(h2 : 4 * m^2 + m + 6 = 442 + 7 * e) : 
  m + e = 22 := by
  sorry

end find_m_and_e_l529_529262


namespace mower_value_drop_l529_529035

theorem mower_value_drop :
  ∀ (initial_value value_six_months value_after_year : ℝ) (percentage_drop_six_months percentage_drop_next_year : ℝ),
  initial_value = 100 →
  percentage_drop_six_months = 0.25 →
  value_six_months = initial_value * (1 - percentage_drop_six_months) →
  value_after_year = 60 →
  percentage_drop_next_year = 1 - (value_after_year / value_six_months) →
  percentage_drop_next_year * 100 = 20 :=
by
  intros initial_value value_six_months value_after_year percentage_drop_six_months percentage_drop_next_year
  intros h1 h2 h3 h4 h5
  sorry

end mower_value_drop_l529_529035


namespace valid_n_values_l529_529779

-- Define the conditions and question as a proof statement in Lean
theorem valid_n_values :
  let T (n : ℕ) : ℕ := sorry -- Some function for number of teams T
  ∃ x : ℕ,
    (12 ≤ n ∧ n ≤ 2017) →
    (T n^2 = (nat.choose n 12 * nat.choose n 10) / (nat.choose (n-6) 6 * nat.choose (n-6) 4)) →
    (0 < x) :=
sorry

end valid_n_values_l529_529779


namespace elimination_is_core_method_l529_529363

def solve_two_linear_equations_core_method (A B C D : string) : Prop :=
  C = "Elimination, reducing two variables to one"

theorem elimination_is_core_method :
  solve_two_linear_equations_core_method 
    "Substitution method" 
    "Addition and subtraction method" 
    "Elimination, reducing two variables to one" 
    "Solving for one unknown to find the other unknown" := 
by 
  sorry

end elimination_is_core_method_l529_529363


namespace sequence_starting_point_l529_529720

theorem sequence_starting_point
  (n : ℕ) 
  (k : ℕ) 
  (h₁ : n * 9 ≤ 100000)
  (h₂ : k = 11110)
  (h₃ : 9 * (n + k - 1) = 99999) : 
  9 * n = 88890 :=
by 
  sorry

end sequence_starting_point_l529_529720


namespace problem_I2_1_problem_I2_2_problem_I2_3_problem_I2_4_l529_529204

-- Problem I2.1
theorem problem_I2_1 (a : ℕ) (h₁ : a > 0) (h₂ : a^2 - 1 = 123 * 125) : a = 124 :=
by {
  -- This proof needs to be filled in
  sorry
}

-- Problem I2.2
theorem problem_I2_2 (b : ℕ) (h₁ : b = (2^3 - 16*2^2 - 9*2 + 124)) : b = 50 :=
by {
  -- This proof needs to be filled in
  sorry
}

-- Problem I2.3
theorem problem_I2_3 (n : ℕ) (h₁ : (n * (n - 3)) / 2 = 54) : n = 12 :=
by {
  -- This proof needs to be filled in
  sorry
}

-- Problem I2_4
theorem problem_I2_4 (d : ℤ) (n : ℤ) (h₁ : n = 12) 
  (h₂ : (d - 1) * 2 = (1 - n) * 2) : d = -10 :=
by {
  -- This proof needs to be filled in
  sorry
}

end problem_I2_1_problem_I2_2_problem_I2_3_problem_I2_4_l529_529204


namespace tyler_double_flips_l529_529245

theorem tyler_double_flips (jen_triple_flips : ℕ) (jen_triple_flips_count : jen_triple_flips = 16)
  (half_flips : ℕ) (total_jen_flips : ℕ) (total_jen_flips_count : total_jen_flips = jen_triple_flips * 3)
  (tyler_total_flips : ℕ) (tyler_total_flips_count : tyler_total_flips = total_jen_flips / 2) 
  (tyler_double_flips : ℕ) : tyler_double_flips = 12 := 
by
  have total_jen_flips_val : total_jen_flips = 16 * 3 := by rw [jen_triple_flips_count, total_jen_flips_count]
  have tyler_total_flips_val : tyler_total_flips = 48 / 2 := by rw [total_jen_flips_val, tyler_total_flips_count]
  have tyler_flips_count : tyler_double_flips = 24 / 2 := sorry
  show tyler_double_flips = 12, from sorry

end tyler_double_flips_l529_529245


namespace digit_7_occurrences_in_range_1_to_2017_l529_529222

-- Define the predicate that checks if a digit appears in a number
def digit_occurrences (d n : Nat) : Nat :=
  Nat.digits 10 n |>.count d

-- Define the range of numbers we are interested in
def range := (List.range' 1 2017)

-- Sum up the occurrences of digit 7 in the defined range
def total_occurrences (d : Nat) (range : List Nat) : Nat :=
  range.foldr (λ n acc => digit_occurrences d n + acc) 0

-- The main theorem to prove
theorem digit_7_occurrences_in_range_1_to_2017 : total_occurrences 7 range = 602 := by
  -- The proof should go here, but we only need to define the statement.
  sorry

end digit_7_occurrences_in_range_1_to_2017_l529_529222


namespace shelby_drive_time_in_rain_l529_529304

theorem shelby_drive_time_in_rain
  (speed_sun : ℝ := 40)  -- speed in miles per hour while sunny
  (speed_rain : ℝ := 25) -- speed in miles per hour while raining
  (total_distance : ℝ := 25) -- total distance in miles
  (total_time : ℝ := 45 / 60) -- total time in hours (45 minutes)
  : ∃ (x : ℝ), x = 20 / 60 ∧
    ((speed_sun * (total_time - x) + speed_rain * x) = total_distance) :=
begin
  sorry
end

end shelby_drive_time_in_rain_l529_529304


namespace max_cities_with_connections_l529_529830

theorem max_cities_with_connections : ∃ n : ℕ, n = 10 ∧
  (∀ c : City, ∃ (direct_neighbors : Set City), 
    direct_neighbors.card ≤ 3 ∧
    (∀ c₁ c₂ : City, ((c₁ ∈ direct_neighbors ∧ c₂ ∈ direct_neighbors) ∨ 
                      ∃ c₃ ∈ direct_neighbors, c₁ = c₃ ∨ c₂ = c₃) →
                     ((c₁ ≠ c ∧ c₂ ≠ c) → c₁ ≠ c₂)) ∧
    (∀ c₂ : City, ∀ c₃ ∉ direct_neighbors, ∃ c₁ ∈ direct_neighbors, 
      c₂ = c ∨ c₃ = c ∨ c₂ = c₁ ∨ c₃ = c₁)) :=
sorry

end max_cities_with_connections_l529_529830


namespace total_shaded_area_l529_529727

theorem total_shaded_area (a1 a2 b1 b2 c1 c2: ℕ) (h₁: a1 = 4) (h₂: a2 = 12) (h₃: b1 = 5) (h₄: b2 = 7) (h₅: c1 = 4) (h₆: c2 = 5) 
: a1 * a2 + b1 * b2 - c1 * c2 = 63 := 
by 
  rw [h₁, h₂, h₃, h₄, h₅, h₆] 
  sorry

end total_shaded_area_l529_529727


namespace number_of_conditions_for_parallel_planes_l529_529259

noncomputable def condition_1 (α β : Plane) (a b : Line) : Prop :=
  skew a b ∧ a ⊆ α ∧ b ⊆ β ∧ parallel a β ∧ parallel b α

noncomputable def condition_2 (α β : Plane) : Prop :=
  ∃ (p1 p2 p3 : Point), non_collinear {p1, p2, p3} ∧ equidistant_from_plane {p1, p2, p3} β

noncomputable def condition_3 (α β γ : Plane) : Prop :=
  perpendicular α γ ∧ perpendicular β γ

theorem number_of_conditions_for_parallel_planes (α β γ : Plane) (a b : Line) :
  (condition_1 α β a b ∧ ¬ condition_2 α β ∧ ¬ condition_3 α β γ) → 1 := sorry

end number_of_conditions_for_parallel_planes_l529_529259


namespace discount_problem_l529_529404

theorem discount_problem (n : ℕ) : 
  (∀ x : ℝ, 0 < x → (1 - n / 100 : ℝ) * x < min (0.72 * x) (min (0.6724 * x) (0.681472 * x))) ↔ n ≥ 33 :=
by
  sorry

end discount_problem_l529_529404


namespace two_small_triangles_in_unit_square_l529_529742

theorem two_small_triangles_in_unit_square (points : Fin 9 → (ℝ × ℝ))
  (h_points : ∀ i, 0 ≤ (points i).fst ∧ (points i).fst ≤ 1 ∧ 0 ≤ (points i).snd ∧ (points i).snd ≤ 1) :
  ∃ (triangles : (Fin 3) → (Fin 9)), 
  ∃ (triangles' : (Fin 3) → (Fin 9)),
  let area := λ t : (Fin 3) → (Fin 9), 
          abs ((points (t 0)).fst * ((points (t 1)).snd - (points (t 2)).snd) + 
               (points (t 1)).fst * ((points (t 2)).snd - (points (t 0)).snd) +
               (points (t 2)).fst * ((points (t 0)).snd - (points (t 1)).snd)) / 2 in
  area triangles < 1 / 8 ∧ area triangles' < 1 / 8 ∧ triangles ≠ triangles' :=
sorry

end two_small_triangles_in_unit_square_l529_529742


namespace no_valid_arrangement_in_7x7_grid_l529_529236

theorem no_valid_arrangement_in_7x7_grid :
  ¬ (∃ (f : Fin 7 → Fin 7 → ℕ),
    (∀ (i j : Fin 6),
      (f i j + f i (j + 1) + f (i + 1) j + f (i + 1) (j + 1)) % 2 = 1) ∧
    (∀ (i j : Fin 5),
      (f i j + f i (j + 1) + f i (j + 2) + f (i + 1) j + f (i + 1) (j + 1) + f (i + 1) (j + 2) +
       f (i + 2) j + f (i + 2) (j + 1) + f (i + 2) (j + 2)) % 2 = 1)) := by
  sorry

end no_valid_arrangement_in_7x7_grid_l529_529236


namespace poly_has_one_positive_and_one_negative_root_l529_529993

theorem poly_has_one_positive_and_one_negative_root :
  ∃! r1, r1 > 0 ∧ (x^4 + 5 * x^3 + 15 * x - 9 = 0) ∧ 
  ∃! r2, r2 < 0 ∧ (x^4 + 5 * x^3 + 15 * x - 9 = 0) := by
sorry

end poly_has_one_positive_and_one_negative_root_l529_529993


namespace trapezoid_area_is_96_cm2_l529_529696

noncomputable def isosceles_trapezoid_area
  (AB CD : ℝ) (h : CD = AB * 13 / 3) (perimeter : ℝ) (perimeter_cond : perimeter = 42) : ℝ :=
  let BK := Math.sqrt ((AB + CD / 2)^2 - AB^2) in
  (1 / 2) * (AB + CD) * BK

theorem trapezoid_area_is_96_cm2 (AB CD : ℝ) (h : CD = AB * 13 / 3) (perimeter_cond : 42 = 3 + 2 * CD + CD)
  (perimeter_eq : (AB + 2 * AB * 13 / 3) + CD = 42) : isosceles_trapezoid_area AB CD h 42 = 96 :=
sorry

end trapezoid_area_is_96_cm2_l529_529696


namespace a_2014_l529_529185

-- Definition of the sequence
def a : ℕ → ℚ
| 1 := 2
| (n + 1) := -1 / a n

-- Theorem to prove that a_{2014} = -1/2
theorem a_2014 :
  a 2014 = -1/2 :=
sorry

end a_2014_l529_529185


namespace A_seq_integer_and_odd_iff_l529_529387

noncomputable def A_seq (k : ℕ) : ℕ → ℤ
| 1     := 1
| (n+1) := (n * A_seq k n + 2 * (n+1) ^ (2*k)) / (n+2)

theorem A_seq_integer_and_odd_iff (k n : ℕ) :
  (∀ n ≥ 1, (A_seq k n).denom = 1) ∧ 
  (A_seq k n % 2 = 1 ↔ n % 4 = 1 ∨ n % 4 = 2) :=
sorry

end A_seq_integer_and_odd_iff_l529_529387


namespace ducks_remaining_after_three_nights_l529_529719

def initial_ducks : ℕ := 320
def first_night_ducks_eaten (ducks : ℕ) : ℕ := ducks * 1 / 4
def after_first_night (ducks : ℕ) : ℕ := ducks - first_night_ducks_eaten ducks
def second_night_ducks_fly_away (ducks : ℕ) : ℕ := ducks * 1 / 6
def after_second_night (ducks : ℕ) : ℕ := ducks - second_night_ducks_fly_away ducks
def third_night_ducks_stolen (ducks : ℕ) : ℕ := ducks * 30 / 100
def after_third_night (ducks : ℕ) : ℕ := ducks - third_night_ducks_stolen ducks

theorem ducks_remaining_after_three_nights : after_third_night (after_second_night (after_first_night initial_ducks)) = 140 :=
by 
  -- replace the following sorry with the actual proof steps
  sorry

end ducks_remaining_after_three_nights_l529_529719


namespace at_least_400_discs_l529_529416

theorem at_least_400_discs (S : set (metric.ball (0 : ℝ × ℝ) 100)) (discs : set (metric.ball (0 : ℝ × ℝ) 1)) : 
  (∀ disc1 disc2 ∈ discs, disc1 ≠ disc2 → metric.ball disc1 ∩ metric.ball disc2 = ∅) → 
  (∀ segment, segment.length = 10 → ∀ p ∈ segment, ∃ disc ∈ discs, disc ∩ segment ≠ ∅) → 
  400 ≤ finset.card (S ∩ discs) :=
sorry

end at_least_400_discs_l529_529416


namespace coefficient_of_y_squared_l529_529124

/-- Given the equation ay^2 - 8y + 55 = 59 and y = 2, prove that the coefficient a is 5. -/
theorem coefficient_of_y_squared (a y : ℝ) (h_y : y = 2) (h_eq : a * y^2 - 8 * y + 55 = 59) : a = 5 := by
  sorry

end coefficient_of_y_squared_l529_529124


namespace z_is_real_iff_z_is_complex_iff_z_is_pure_imaginary_iff_l529_529852

def is_real (z : ℂ) := z.im = 0
def is_complex (z : ℂ) := z.im ≠ 0
def is_pure_imaginary (z : ℂ) := z.re = 0 ∧ z.im ≠ 0

def z (m : ℝ) : ℂ := ⟨m - 3, m^2 - 2 * m - 15⟩

theorem z_is_real_iff (m : ℝ) : is_real (z m) ↔ m = -3 ∨ m = 5 :=
by sorry

theorem z_is_complex_iff (m : ℝ) : is_complex (z m) ↔ m ≠ -3 ∧ m ≠ 5 :=
by sorry

theorem z_is_pure_imaginary_iff (m : ℝ) : is_pure_imaginary (z m) ↔ m = 3 :=
by sorry

end z_is_real_iff_z_is_complex_iff_z_is_pure_imaginary_iff_l529_529852


namespace haircut_cost_l529_529618

noncomputable def cost_of_haircut_without_tip : ℝ := sorry

theorem haircut_cost (hair_growth_monthly inches_per_cut months_per_year total_expense per_tip haircut_count yearly_cost haircut_cost : ℝ): 
  (hair_growth_monthly = 1.5) →
  (inches_per_cut = 3) →
  (months_per_year = 12) →
  (total_expense = 324) →
  (per_tip = 1.20) →
  (haircut_count = months_per_year / (inches_per_cut / hair_growth_monthly)) →
  (yearly_cost = haircut_count * per_tip * haircut_cost) →
  (yearly_cost = total_expense) →
  (haircut_cost = 45) := by
    intros hair_growth_monthly inches_per_cut months_per_year total_expense per_tip haircut_count yearly_cost haircut_cost
    intros h1 h2 h3 h4 h5 h6 h7 h8
    sorry

end haircut_cost_l529_529618


namespace geometric_seq_proof_l529_529863

noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * q ^ (n - 1)

theorem geometric_seq_proof :
  (∃ q : ℝ, ∃ a2 : ℝ, ∃ a4 : ℝ,
    q = 2 ∧
    a4 = 8 ∧
    geometric_sequence 1 q 2 * geometric_sequence 1 q 4 = geometric_sequence 1 q 5 ∧
    geometric_sequence 1 q 4 = 8 ∧
    ∑ i in (finset.range 4).map nat.succ, geometric_sequence 1 q i = 15) :=
begin
  use 2,
  use 2,
  use 8,
  split, refl,
  split, refl,
  split,
  { -- Proof that a_2 * a_4 = a_5
    have h1 := (geometric_sequence 1 2 2),
    have h4 := (geometric_sequence 1 2 4),
    have h5 := (geometric_sequence 1 2 5),
    calc h1 * h4 = (1 * 2 ^ (2 - 1)) * (1 * 2 ^ (4 - 1)) : by simp [h1, h4]
              ... = (2 ^ 1) * (2 ^ 3) : by simp
              ... = 2 ^ (1 + 3) : by rw pow_add
              ... = h5 : by simp [h5] },
  { -- Proof that a_4 = 8
    refl },
  { -- Proof that the sum of the first 4 terms is 15
    have h4 : ∑ i in (finset.range 4).map nat.succ, geometric_sequence 1 2 i = 
      geometric_sequence 1 2 1 +
      geometric_sequence 1 2 2 +
      geometric_sequence 1 2 3 +
      geometric_sequence 1 2 4 := by simp,
    calc (∑ i in (finset.range 4).map nat.succ, geometric_sequence 1 2 i)
        = 1 * (2 ^ (1 - 1)) + 1 * (2 ^ (2 - 1)) + 1 * (2 ^ (3 - 1)) + 1 * (2 ^ (4 - 1)) :
        by simp [h4, geometric_sequence]
    ... = 1 + 2 + 4 + 8 : by simp
    ... = 15 : by norm_num }
end

end geometric_seq_proof_l529_529863


namespace nonagon_diagonal_intersection_probability_l529_529941

theorem nonagon_diagonal_intersection_probability (V : Finset ℕ) (hV : V.card = 9) : 
  let num_diagonals := (V.card.choose 2) - 9,
      num_pairs := num_diagonals.choose 2,
      num_intersections := V.card.choose 4 in
  num_intersections.toRat / num_pairs.toRat = (14 : ℚ) / 39 :=
by
  sorry

end nonagon_diagonal_intersection_probability_l529_529941


namespace Trevor_tip_l529_529724

variable (Uber Lyft Taxi : ℕ)
variable (TotalCost : ℕ)

theorem Trevor_tip 
  (h1 : Uber = Lyft + 3) 
  (h2 : Lyft = Taxi + 4) 
  (h3 : Uber = 22) 
  (h4 : TotalCost = 18)
  (h5 : Taxi = 15) :
  (TotalCost - Taxi) * 100 / Taxi = 20 := by
  sorry

end Trevor_tip_l529_529724


namespace count_even_multiples_of_5_l529_529130

theorem count_even_multiples_of_5 : 
  (set_of (λ (n : ℕ), 1 ≤ n ∧ n ≤ 2023 ∧ n % 10 = 0)).finite.count = 202 :=
sorry

end count_even_multiples_of_5_l529_529130


namespace area_of_triangle_DEF_l529_529003

theorem area_of_triangle_DEF :
  ∃ (DEF : Type) (area_u1 area_u2 area_u3 area_triangle : ℝ),
  area_u1 = 25 ∧
  area_u2 = 16 ∧
  area_u3 = 64 ∧
  area_triangle = area_u1 + area_u2 + area_u3 ∧
  area_triangle = 289 :=
by
  sorry

end area_of_triangle_DEF_l529_529003


namespace speech_orders_count_l529_529008

theorem speech_orders_count {A B : Type} (others : Finset Type) (h_size : others.card = 4) :
  ∃ (L : Finset (Finset Type)), 
  (∃ s ∈ L, A ∈ s ∨ B ∈ s) ∧ L.card = 16 ∧ 
  ∑ s in L, (s.card.factorial.val : ℕ) = 96 :=
by
  -- Theorem to state the conditions
  sorry

end speech_orders_count_l529_529008


namespace max_value_x_y3_z4_l529_529266

theorem max_value_x_y3_z4 (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 2) :
  x + y^3 + z^4 ≤ 2 :=
by
  sorry

end max_value_x_y3_z4_l529_529266


namespace triangle_isosceles_of_sin_condition_l529_529233

noncomputable def isosceles_triangle (A B C : ℝ) : Prop :=
  A = B ∨ B = C ∨ C = A

theorem triangle_isosceles_of_sin_condition {A B C : ℝ} (h : 2 * Real.sin A * Real.cos B = Real.sin C) : 
  isosceles_triangle A B C :=
by
  sorry

end triangle_isosceles_of_sin_condition_l529_529233


namespace distinct_pairs_reciprocal_sum_l529_529194

theorem distinct_pairs_reciprocal_sum : 
  ∃ (S : Finset (ℕ × ℕ)), (∀ (m n : ℕ), ((m, n) ∈ S) ↔ (m > 0 ∧ n > 0 ∧ (1/m + 1/n = 1/5))) ∧ S.card = 3 :=
sorry

end distinct_pairs_reciprocal_sum_l529_529194


namespace coordinates_with_respect_to_origin_l529_529945

theorem coordinates_with_respect_to_origin :
  ∀ (point : ℝ × ℝ), point = (3, -2) → point = (3, -2) := by
  intro point h
  exact h

end coordinates_with_respect_to_origin_l529_529945


namespace periodic_values_of_n_l529_529131

theorem periodic_values_of_n :
  ∀ n : ℤ, n < 0 → (∀ x : ℝ, f (n) x = cos (7 * x) * sin (25 * x / n^2) → f n (x + 7 * π) = f n x)
    → (n = -1 ∨ n = -5) :=
by
  intros n hneg hperiod
  specialize hperiod
  sorry

end periodic_values_of_n_l529_529131


namespace max_OM_ON_value_l529_529952

noncomputable def maximum_OM_ON (a b : ℝ) : ℝ :=
  (1 + Real.sqrt 2) / 2 * (a + b)

-- Given the conditions in triangle ABC with sides BC and AC having fixed lengths a and b respectively,
-- and that AB can vary such that a square is constructed outward on side AB with center O,
-- and M and N are the midpoints of sides BC and AC respectively, prove the maximum value of OM + ON.
theorem max_OM_ON_value (a b : ℝ) : 
  ∃ OM ON : ℝ, OM + ON = maximum_OM_ON a b :=
sorry

end max_OM_ON_value_l529_529952


namespace log_power_function_l529_529892

theorem log_power_function (a : ℝ) (h : (1/2)^a = (real.sqrt 2)/2) : real.log_base a 2 = -1 :=
by
  sorry   -- Proof is omitted

end log_power_function_l529_529892


namespace area_MBCN_l529_529394

-- Define the conditions and hypothesis of the problem.
variables 
  (M A D N B C : Point)
  (ABCD_MADN_similar : similar_trapezoids M A D N A B C D)
  (area_ABCD : area_trapezoid A B C D = S)
  (angles_large_base_sum : angle A B + angle D C = 150)

-- Define the proof statement for the area of trapezoid MBCN.
theorem area_MBCN (h : similar_trapezoids M A D N A B C D)
  (h1 : area_trapezoid A B C D = S)
  (h2 : angle A B + angle D C = 150) : 
  area_trapezoid M B C N = 4 * S := 
sorry

end area_MBCN_l529_529394


namespace possible_ones_digits_count_l529_529285

theorem possible_ones_digits_count :
  {d : ℕ | ∃ n : ℕ, n % 10 = d ∧ n % 8 = 0}.finite ∧ 
  (Finset.card {d : ℕ | ∃ n : ℕ, n % 10 = d ∧ n % 8 = 0}.to_finset = 5) := 
by
  sorry

end possible_ones_digits_count_l529_529285


namespace winnie_the_pooh_apples_l529_529377

theorem winnie_the_pooh_apples :
  let winnie_rate := 4
  let tigger_rate := 7
  let winnie_time := 80
  let tigger_time := 50
  let total_apples := 2010
  Winnie_apples = 4 * 80.
  Winnie_apples + Tigger_apples = 2010
  Tigger_apples = 7 * 50 
  Winnie_apples = 960 :=
by
  let winnie_rate := 4
  let tigger_rate := 7
  let winnie_time := 80
  let tigger_time := 50
  let total_apples := 2010

  have winnie_apples : ℕ := winnie_rate * winnie_time
  have tigger_apples : ℕ := tigger_rate * tigger_time

  have total : ℕ := winnie_apples + tigger_apples

  -- Use the given that total_apples = 2010
  have eq_total : total = total_apples := by
  sorry

  -- Use winnie_apples and eq_total to prove
  have result : winnie_apples = 960 := by
  sorry

  exact result

end winnie_the_pooh_apples_l529_529377


namespace largest_x_exists_largest_x_largest_real_number_l529_529081

theorem largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : x ≤ 48 / 7 :=
sorry

theorem exists_largest_x (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  ∃ x, (⌊x⌋ : ℝ) / x = 7 / 8 ∧ x = 48 / 7 :=
sorry

theorem largest_real_number (x : ℝ) (h1 : (⌊x⌋ : ℝ) / x = 7 / 8) : 
  x = 48 / 7 :=
sorry

end largest_x_exists_largest_x_largest_real_number_l529_529081


namespace train_speed_in_m_per_s_l529_529747

theorem train_speed_in_m_per_s (speed_kmph : ℕ) (h : speed_kmph = 162) :
  (speed_kmph * 1000) / 3600 = 45 :=
by {
  sorry
}

end train_speed_in_m_per_s_l529_529747


namespace axiom1_l529_529034

-- Definition of points being on a line
def on_line {Point : Type} (A B : Point) (l : set Point) : Prop :=
  A ∈ l ∧ B ∈ l

-- Definition of points being within a plane
def in_plane {Point : Type} (A B : Point) (α : set Point) : Prop :=
  A ∈ α ∧ B ∈ α

-- Axiom 1: If two points A and B on a line l are within a plane α, then the line l is within this plane
theorem axiom1 {Point : Type} {A B : Point} {l α : set Point} :
    A ∈ l → B ∈ l → A ∈ α → B ∈ α → l ⊆ α :=
by
  sorry

end axiom1_l529_529034


namespace tyson_total_race_time_l529_529357

def lake_speed := 3        -- mph
def ocean_speed := 2.5     -- mph

def lake_races_distances := [2, 3.2, 3.5, 1.8, 4]    -- distances in miles
def ocean_races_distances := [2.5, 3.5, 2, 3.7, 4.2] -- distances in miles

def time_spent_in_lake_race (d : ℝ) : ℝ := d / lake_speed
def time_spent_in_ocean_race (d : ℝ) : ℝ := d / ocean_speed

def total_time_in_lake_races : ℝ := (lake_races_distances.map time_spent_in_lake_race).sum
def total_time_in_ocean_races : ℝ := (ocean_races_distances.map time_spent_in_ocean_race).sum

def total_time_in_races : ℝ := total_time_in_lake_races + total_time_in_ocean_races

theorem tyson_total_race_time :
  total_time_in_races = 11.1934 :=
by
  sorry

end tyson_total_race_time_l529_529357


namespace quadratic_function_solution_l529_529161

noncomputable def f (x : ℝ) : ℝ := 1/2 * x^2 + 1/2 * x

theorem quadratic_function_solution (f : ℝ → ℝ)
  (h1 : ∃ a b c : ℝ, (a ≠ 0) ∧ (∀ x, f x = a * x^2 + b * x + c))
  (h2 : f 0 = 0)
  (h3 : ∀ x, f (x+1) = f x + x + 1) :
  ∀ x, f x = 1/2 * x^2 + 1/2 * x :=
by
  sorry

end quadratic_function_solution_l529_529161


namespace three_digit_numbers_without_5_7_9_l529_529578

theorem three_digit_numbers_without_5_7_9 : 
  let digits := {d : Fin 10 | d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9},
      count_digits := λ s, ∃ (count : ℕ), count = (card s)
  in 
  (count_digits {d : Fin 10 | 1 ≤ d ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9}) * 
  (count_digits {d : Fin 10 | d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9}) * 
  (count_digits {d : Fin 10 | d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9}) = 294 := 
by
  sorry

end three_digit_numbers_without_5_7_9_l529_529578


namespace acute_angle_sine_solution_l529_529156

theorem acute_angle_sine_solution (α : ℝ) (h1 : 0 < α) (h2 : α < 90) (h3 : sin (α - 10 * real.pi / 180) = real.sqrt 3 / 2) : α = 70 * real.pi / 180 := 
by
  sorry

end acute_angle_sine_solution_l529_529156


namespace quadrilateral_construction_proof_l529_529055

open_locale classical

-- Definitions of the lengths
variables (a c e g h : ℝ)
variables (AB BC CD DA AC KM LN : ℝ)
variables (K L M N O A B C D : Type*)

-- Assumptions based on the conditions
variable (AB_eq : AB = a)
variable (CD_eq : CD = c)
variable (AC_eq : AC = e)
variable (KM_eq : KM = g)
variable (LN_eq : LN = h)
variable (parallelogram_KLMN : KLMN_is_parallelogram)

noncomputable def construct_quadrilateral : Prop :=
  ∃ (ABCD : Type*), 
  (ABCD_has_sides ABCD AB BC CD DA) ∧ 
  (midpoints_of_sides ABCD K L M N) ∧
  (side_lengths_eq ABCD a c e g h) ∧ 
  (KLMN_is_parallelogram ABCD) ∧ 
  (midsegment_ACD_eq ABCD e)

-- Statement for the proof
theorem quadrilateral_construction_proof :
  construct_quadrilateral a c e g h AB_eq CD_eq AC_eq KM_eq LN_eq :=
sorry


end quadrilateral_construction_proof_l529_529055


namespace longest_line_segment_square_in_sector_l529_529403

-- conditions
def diameter := 16 -- diameter of the pie
def radius := diameter / 2 -- radius of the pie
def sectors := 4 -- number of equal-sized sectors

-- statement: Prove that the square of the length of the longest line segment in one sector is 128
theorem longest_line_segment_square_in_sector : let l := 8 * Real.sqrt 2 in l^2 = 128 :=
by sorry

end longest_line_segment_square_in_sector_l529_529403


namespace determine_k_values_parallel_lines_l529_529540

theorem determine_k_values_parallel_lines :
  ∀ k : ℝ, ((k - 3) * x + (4 - k) * y + 1 = 0 ∧ 2 * (k - 3) * x - 2 * y + 3 = 0)
  → k = 2 ∨ k = 3 ∨ k = 6 :=
by
  sorry

end determine_k_values_parallel_lines_l529_529540


namespace analytical_expression_tangent_line_at_neg1_l529_529902

-- Definitions based on conditions
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * x
def f' (a : ℝ) (x : ℝ) : ℝ := (deriv (f a)) x
def f_at_0 (a : ℝ) : ℝ := f' a 0
def tangent_line_eq (a : ℝ) (x₀ y₀ m : ℝ) (x y : ℝ) : Prop := y - y₀ = m * (x - x₀)

-- Given conditions
axiom f'_0_eq_1 : f_at_0 a = 1

-- Theorem proof problems
theorem analytical_expression (a : ℝ) : (f a = λ x, x^2 + x) :=
by
  sorry

theorem tangent_line_at_neg1 (a : ℝ) (h : a = 1) : 
  (tangent_line_eq a (-1) 0 (f' a (-1)) x y) ↔ (x + y + 1 = 0) :=
by
  sorry

end analytical_expression_tangent_line_at_neg1_l529_529902


namespace matrix_product_zero_l529_529815

variable {R : Type} [Ring R]
variables (a b c d : R)

def M1 : Matrix (Fin 3) (Fin 3) R :=
  ![![0, 2*c, -2*b], 
    ![-2*c, 0, 2*a], 
    ![2*b, -2*a, 0]]

def M2 : Matrix (Fin 3) (Fin 3) R :=
  ![![a^2 + d, a*b, a*c], 
    ![a*b, b^2 + d, b*c], 
    ![a*c, b*c, c^2 + d]]

theorem matrix_product_zero : M1 * M2 = (0 : Matrix (Fin 3) (Fin 3) R) :=
  sorry

end matrix_product_zero_l529_529815


namespace max_spheres_in_frumst_l529_529770

noncomputable def frustumHeight : ℝ := 8
noncomputable def sphereO1radius : ℝ := 2
noncomputable def sphereO2radius : ℝ := 3

theorem max_spheres_in_frumst :=
  let original_spheres := 2 in
  let total_spheres := original_spheres + 2 in
  let max_additional_spheres := total_spheres - original_spheres in
  max_additional_spheres = 2 :=
  sorry

end max_spheres_in_frumst_l529_529770


namespace line_passes_through_fixed_point_min_area_line_eq_l529_529896

section part_one

variable (m x y : ℝ)

def line_eq := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4

theorem line_passes_through_fixed_point :
  ∀ m, line_eq m 3 1 = 0 :=
sorry

end part_one

section part_two

variable (k x y : ℝ)

def line_eq_l1 (k : ℝ) := y = k * (x - 3) + 1

theorem min_area_line_eq :
  line_eq_l1 (-1/3) x y = (x + 3 * y - 6 = 0) :=
sorry

end part_two

end line_passes_through_fixed_point_min_area_line_eq_l529_529896


namespace capacitance_infinite_chain_l529_529364

noncomputable def effective_capacitance (C : ℝ) : ℝ := C * (1 + Real.sqrt 3) / 2

theorem capacitance_infinite_chain (C : ℝ) : effective_capacitance C = C * (1 + Real.sqrt 3) / 2 := 
by
  simp [effective_capacitance]
  sorry

end capacitance_infinite_chain_l529_529364


namespace largest_real_solution_l529_529104

theorem largest_real_solution (x : ℝ) (h : (⌊x⌋ / x = 7 / 8)) : x ≤ 48 / 7 := by
  sorry

end largest_real_solution_l529_529104


namespace shelves_needed_number_of_shelves_l529_529286

-- Define the initial number of books
def initial_books : Float := 46.0

-- Define the number of additional books added by the librarian
def additional_books : Float := 10.0

-- Define the number of books each shelf can hold
def books_per_shelf : Float := 4.0

-- Define the total number of books
def total_books : Float := initial_books + additional_books

-- The mathematical proof statement for the number of shelves needed
theorem shelves_needed : Float := total_books / books_per_shelf

-- The required statement proving that the number of shelves needed is 14.0
theorem number_of_shelves : shelves_needed = 14.0 := by
  sorry

end shelves_needed_number_of_shelves_l529_529286


namespace taxi_ride_cost_l529_529426

def base_fare : ℝ := 2.00
def cost_per_mile : ℝ := 0.30
def discount_threshold : ℝ := 10
def discount_rate : ℝ := 0.10
def miles_traveled : ℝ := 12.0

theorem taxi_ride_cost :
  let total_cost_before_discount := base_fare + miles_traveled * cost_per_mile in
  let total_cost := if miles_traveled > discount_threshold then
                      total_cost_before_discount - discount_rate * total_cost_before_discount
                    else
                      total_cost_before_discount
  in
  total_cost = 5.04 := 
by sorry

end taxi_ride_cost_l529_529426


namespace proof_inequality_proof_l529_529263

noncomputable def inequality_proof (a b c : ℝ) (h1: 0 ≤ a) (h2: a ≤ 1) (h3: 0 ≤ b) (h4: b ≤ 1) 
  (h5: 0 ≤ c) (h6: c ≤ 1) (h7: 1 ≤ a + b) (h8: 1 ≤ b + c) (h9: 1 ≤ c + a) : Prop :=
  1 ≤ (1 - a)^2 + (1 - b)^2 + (1 - c)^2 + 2 * real.sqrt 2 * a * b * c / real.sqrt (a^2 + b^2 + c^2)

theorem proof_inequality_proof (a b c : ℝ) (h1: 0 ≤ a) (h2: a ≤ 1) (h3: 0 ≤ b) (h4: b ≤ 1) 
  (h5: 0 ≤ c) (h6: c ≤ 1) (h7: 1 ≤ a + b) (h8: 1 ≤ b + c) (h9: 1 ≤ c + a) : inequality_proof a b c h1 h2 h3 h4 h5 h6 h7 h8 h9 :=
  sorry

end proof_inequality_proof_l529_529263


namespace find_angle_between_vectors_l529_529965

open Real EuclideanSpace

noncomputable def angle_between_unit_vectors {n : ℕ} (a b : ℝⁿ) := 
  real.cos⁻¹ ((inner a b) / (∥a∥ * ∥b∥))

theorem find_angle_between_vectors {n : ℕ} (a b : ℝⁿ) 
  (ha : ∥a∥ = 1) (hb : ∥b∥ = 1)
  (orthogonal : inner (a + 3 • b) (4 • a - 3 • b) = 0) : 
  angle_between_unit_vectors a b = real.cos⁻¹ (5 / 9) :=
  sorry

end find_angle_between_vectors_l529_529965


namespace variance_scaled_l529_529717

theorem variance_scaled (s1 : ℝ) (c : ℝ) (h1 : s1 = 3) (h2 : c = 3) :
  s1 * (c^2) = 27 :=
by
  rw [h1, h2]
  norm_num

end variance_scaled_l529_529717


namespace find_a_of_normal_vector_l529_529700

theorem find_a_of_normal_vector (a : ℝ) : 
  (∀ x y : ℝ, 3 * x + 2 * y + 5 = 0) ∧ (∃ n : ℝ × ℝ, n = (a, a - 2)) → a = 6 := by
  sorry

end find_a_of_normal_vector_l529_529700


namespace formed_c2h6_moles_l529_529195

-- Constants representing the chemical formulae
constant C2H2 : Type
constant H2 : Type
constant C2H6 : Type

-- Reaction ratio by balanced equation
constant reaction_ratio : (C2H2 × H2 × C2H6) → Prop

-- Condition: reaction_ratio follows the balanced chemical equation
axiom balanced_equation : ∀ (c2h2 h2 c2h6 : ℕ), (reaction_ratio (c2h2, h2, c2h6)) ↔ (c2h2 + 2 * h2 = c2h6)

-- Given conditions
axiom h2_moles : ℕ := 6
axiom c2h2_moles : ℕ := 3

-- To prove
theorem formed_c2h6_moles : ∃ (c2h6_moles : ℕ), reaction_ratio (c2h2_moles, h2_moles, c2h6_moles) ∧ c2h6_moles = 3 :=
by
  sorry

end formed_c2h6_moles_l529_529195


namespace a4_value_l529_529144

axiom a_n : ℕ → ℝ
axiom S_n : ℕ → ℝ
axiom q : ℝ

-- Conditions
axiom a1_eq_1 : a_n 1 = 1
axiom S6_eq_4S3 : S_n 6 = 4 * S_n 3
axiom q_ne_1 : q ≠ 1

-- Arithmetic Sequence Sum Formula
axiom sum_formula : ∀ n, S_n n = (1 - q^n) / (1 - q)

-- nth-term Formula
axiom nth_term_formula : ∀ n, a_n n = a_n 1 * q^(n - 1)

-- Prove the value of the 4th term
theorem a4_value : a_n 4 = 3 := sorry

end a4_value_l529_529144


namespace find_conjugate_and_values_of_ab_l529_529139

variable (z : ℂ) (a b : ℝ)
variable (hz : z = 2 * complex.I / (1 - complex.I))
variable (h_eq : abs z ^ 2 + a * z + b = 1 - complex.I)

theorem find_conjugate_and_values_of_ab :
  (conj z = -1 - complex.I) ∧ (a = -1) ∧ (b = -2) :=
by
  sorry

end find_conjugate_and_values_of_ab_l529_529139


namespace circle_line_intersection_points_l529_529702

theorem circle_line_intersection_points :
  let circle := λ x y, (x + 1)^2 + (y + 2)^2 = 8
  let line := λ x y, x + y + 1 = 0
  let dist := λ x y, abs (x + y + 1) / real.sqrt 2
  ∃ xy_pairs : list (ℝ × ℝ),
  (∀ p ∈ xy_pairs, circle p.1 p.2 ∧ dist p.1 p.2 = real.sqrt 2) ∧
  xy_pairs.length = 3 :=
sorry

end circle_line_intersection_points_l529_529702


namespace binomial_expansion_sum_and_constant_l529_529587

-- Define the problem statement.
theorem binomial_expansion_sum_and_constant (n : ℕ) (T : ℕ) 
  (h1 : (2^n = 64) → n = 6) 
  (h2 : (n = 6) → T = ∑ r in range (n+1), Nat.choose 6 r * ((-1)^r * (2^(6 - r) * 1))) :
  n = 6 ∧ T = 240 :=
by 
  sorry

end binomial_expansion_sum_and_constant_l529_529587


namespace find_N_and_values_l529_529265

-- Define the given conditions
variables {a b c d e : ℝ}

-- Condition: a, b, c, d, e are positive real numbers.
axiom positive_real_numbers : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0

-- Condition: a^2 + b^2 + c^2 + d^2 + e^2 = 2020
axiom sum_of_squares : a^2 + b^2 + c^2 + d^2 + e^2 = 2020

-- Define the expression to be maximized
def expression (a b c d e : ℝ) : ℝ := ac + 3bc + 4cd + 6ce

-- Define N as the maximum value of the expression
noncomputable def N : ℝ := sorry -- Detailed calculation is omitted

-- Define a_N, b_N, c_N, d_N, e_N corresponding to the values producing maximum N
noncomputable def a_N : ℝ := sorry
noncomputable def b_N : ℝ := sorry
noncomputable def c_N : ℝ := sorry
noncomputable def d_N : ℝ := sorry
noncomputable def e_N : ℝ := sorry

-- The final theorem statement
theorem find_N_and_values :
  positive_real_numbers →
  sum_of_squares →
  N + a_N + b_N + c_N + d_N + e_N = 70 * real.sqrt 62 + real.sqrt 1010 + 1010 * real.sqrt 62 :=
sorry

end find_N_and_values_l529_529265


namespace largest_x_satisfies_condition_l529_529085

theorem largest_x_satisfies_condition (x : ℝ) (h : (⌊x⌋ / x) = 7 / 8) : x ≤ 48 / 7 :=
sorry

end largest_x_satisfies_condition_l529_529085


namespace solve_log_inequality_l529_529308

-- We're working with logarithms, inequalities, and products
open Real

theorem solve_log_inequality (a : ℝ) (x : ℝ) (h_pos : 0 < a) (h_ne : a ≠ 1) :
  (log a (x^2 - x - 2) > log a (x - 2 / a) + 1) ↔
  (a > 1 → x > 1 + a) ∧ (0 < a ∧ a < 1 → false) :=
by {
  sorry
}

end solve_log_inequality_l529_529308


namespace number_replacement_l529_529360

theorem number_replacement (n : ℕ) (h : n > 0):
  let sequence := λ (s : List ℝ), s.foldl (+) 0 = n
  → ∀ (final_number : ℝ), (final_number > 0) ∧ 
    let seq_ops := λ (s : List ℝ), 
      ∃ (a b : ℝ), ((a, b) ∈ (s.product s)) → s.replace a b ((a + b) / 4) = [final_number] 
    → final_number > 1 / (n : ℝ) :=
sorry

end number_replacement_l529_529360


namespace pyramid_volume_l529_529338

def point := (ℝ × ℝ × ℝ)

structure Rectangle (A B C D P : point) :=
  (AB : ℝ := 12 * real.sqrt 3)
  (BC : ℝ := 13 * real.sqrt 3)
  (diagonal_intersection : ℝ × ℝ × ℝ)

structure Tetrahedron (A B C D P : point) :=
  (isosceles_faces : Prop)

noncomputable def volume_of_tetrahedron (A B C D P : point) [Rectangle A B C D P] [Tetrahedron A B C D P] : ℝ := 
  594

theorem pyramid_volume 
  {A B C D P : point} 
  [hr : Rectangle A B C D P] 
  [ht : Tetrahedron A B C D P] : 
  volume_of_tetrahedron A B C D P = 594 := 
sorry

end pyramid_volume_l529_529338


namespace cashier_opens_probability_l529_529447

-- Definition of the timeline and arrival times
variables {x₁ x₂ x₃ x₄ x₅ x₆ : ℝ}
-- Condition that all arrival times are between 0 and 15 minutes
def arrival_times_within_bounds : Prop := 
  0 ≤ x₁ ∧ x₁ ≤ 15 ∧ 
  0 ≤ x₂ ∧ x₂ ≤ 15 ∧
  0 ≤ x₃ ∧ x₃ ≤ 15 ∧
  0 ≤ x₄ ∧ x₄ ≤ 15 ∧
  0 ≤ x₅ ∧ x₅ ≤ 15 ∧
  0 ≤ x₆ ∧ x₆ ≤ 15

-- Condition that the Scientist arrives last
def scientist_arrives_last : Prop := 
  x₁ < x₆ ∧ x₂ < x₆ ∧ x₃ < x₆ ∧ x₄ < x₆ ∧ x₅ < x₆

-- Event A: Cashier opens no later than 3 minutes after the Scientist arrives, i.e., x₆ ≥ 12
def event_A : Prop := x₆ ≥ 12

-- The correct answer
theorem cashier_opens_probability :
  arrival_times_within_bounds ∧ scientist_arrives_last → 
  Pr[x₆ ≥ 12 | x₁, x₂, x₃, x₄, x₅ < x₆] = 0.738 :=
sorry

end cashier_opens_probability_l529_529447


namespace power_of_fraction_l529_529037

theorem power_of_fraction : ((1/3)^5 = (1/243)) :=
by
  sorry

end power_of_fraction_l529_529037


namespace banana_distinct_arrangements_l529_529915

theorem banana_distinct_arrangements :
  let n := 6
  let f_B := 1
  let f_N := 2
  let f_A := 3
  (n.factorial) / (f_B.factorial * f_N.factorial * f_A.factorial) = 60 := by
sorry

end banana_distinct_arrangements_l529_529915


namespace correct_articles_l529_529956

-- Definitions based on conditions
def is_vowel_sound (c : Char) : Prop :=
  c ∈ ['a', 'e', 'i', 'o', 'u']

def article (word : String) : String :=
  if is_vowel_sound (String.front word) then "an" else "a"

-- Given words
def word1 : String := "MP4"
def word2 : String := "birthday"

-- Prove the correct usage of articles
theorem correct_articles :
  article word1 = "an" ∧ article word2 = "a" := by
  sorry

end correct_articles_l529_529956


namespace maximum_range_of_temperatures_l529_529389

variable (T1 T2 T3 T4 T5 : ℝ)

-- Given conditions
def average_condition : Prop := (T1 + T2 + T3 + T4 + T5) / 5 = 50
def lowest_temperature_condition : Prop := T1 = 45

-- Question to prove
def possible_maximum_range : Prop := T5 - T1 = 25

-- The final theorem statement
theorem maximum_range_of_temperatures 
  (h_avg : average_condition T1 T2 T3 T4 T5) 
  (h_lowest : lowest_temperature_condition T1) 
  : possible_maximum_range T1 T5 := by
  sorry

end maximum_range_of_temperatures_l529_529389


namespace union_and_complement_l529_529150

def A : Set ℕ := {1, 2}
def U : Set ℕ := {1, 2, 3, 4, 5}

noncomputable def is_nonempty (s: Set ℕ) : Prop := s ≠ ∅

theorem union_and_complement (a : ℕ) (h : a = 1 ∨ a = 2) : 
  (A ∩ (insert 3 (singleton a)) ≠ ∅) →
  (A ∪ (insert 3 (singleton a)) = {1, 2, 3}) ∧
  ((a = 1 → U \ (insert 3 (singleton a)) = {2, 4, 5}) ∧
  (a = 2 → U \ (insert 3 (singleton a)) = {1, 4, 5})) := 
by
  intro h1
  cases h with
  | inl ha1 =>
    have hB1 : insert 3 (singleton 1) = {3, 1} := rfl
    have hAU1 : A ∪ {3, 1} = {1, 2, 3} := by simp [A]
    have hCU1 : U \ {3, 1} = {2, 4, 5} := by simp [U]
    simp [hB1, hAU1, hCU1]
  | inr ha2 =>
    have hB2 : insert 3 (singleton 2) = {3, 2} := rfl
    have hAU2 : A ∪ {3, 2} = {1, 2, 3} := by simp [A]
    have hCU2 : U \ {3, 2} = {1, 4, 5} := by simp [U]
    simp [hB2, hAU2, hCU2]
  sorry

end union_and_complement_l529_529150


namespace ferris_wheel_seats_l529_529396

theorem ferris_wheel_seats (total_people : ℕ) (people_per_seat : ℕ) (h1 : total_people = 16) (h2 : people_per_seat = 4) : (total_people / people_per_seat) = 4 := by
  sorry

end ferris_wheel_seats_l529_529396


namespace AQPR_is_parallelogram_l529_529871

theorem AQPR_is_parallelogram
  (A B C P Q R : Type)
  [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty P] [Nonempty Q] [Nonempty R]
  (triangle_ABC : Triangle A B C)
  (isosceles_PBC : IsoscelesTriangle P B C)
  (isosceles_QCA : IsoscelesTriangle Q C A)
  (isosceles_RAB : IsoscelesTriangle R A B)
  (P_side : SameSide A B C P)
  (Q_side : OppositeSide B C A Q)
  (R_side : OppositeSide C A B R)
  (same_sides_PB_PC : PB = PC)
  (same_sides_QA_QC : QA = QC)
  (same_sides_RA_RB : RA = RB)
:
  Parallelogram A Q P R :=
sorry

end AQPR_is_parallelogram_l529_529871


namespace sum_of_given_geom_series_l529_529050

-- Define the necessary conditions
def first_term (a : ℕ) := a = 2
def common_ratio (r : ℕ) := r = 3
def number_of_terms (n : ℕ) := n = 6

-- Define the sum of the geometric series
def sum_geom_series (a r n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

-- State the theorem
theorem sum_of_given_geom_series :
  first_term 2 → common_ratio 3 → number_of_terms 6 → sum_geom_series 2 3 6 = 728 :=
by
  intros h1 h2 h3
  rw [first_term] at h1
  rw [common_ratio] at h2
  rw [number_of_terms] at h3
  have h1 : 2 = 2 := by exact h1
  have h2 : 3 = 3 := by exact h2
  have h3 : 6 = 6 := by exact h3
  exact sorry

end sum_of_given_geom_series_l529_529050


namespace solution_proof_l529_529433

noncomputable def problem_statement : Prop :=
  let x := [x1, x2, x3, x4, x5, x6] in
  let B := (∀ i < 5, x[i] < x[5]) in
  let A := (x[5] ≥ 12) in
  ∃ x1 x2 x3 x4 x5 x6 : ℝ,
    (0 ≤ x1 ∧ x1 ≤ 15) ∧ (0 ≤ x2 ∧ x2 ≤ 15) ∧ (0 ≤ x3 ∧ x3 ≤ 15) ∧
    (0 ≤ x4 ∧ x4 ≤ 15) ∧ (0 ≤ x5 ∧ x5 ≤ 15) ∧ (0 ≤ x6 ∧ x6 ≤ 15) ∧
    B ∧ (classical.some ((measure_theory.measure_space.measure (λ x, x < 12 <= x) B).to_real) = 0.738)

theorem solution_proof : problem_statement := sorry

end solution_proof_l529_529433


namespace probability_is_correct_l529_529435

noncomputable def probability_cashier_opens_early : ℝ :=
  let x1 : ℝ := sorry
  let x2 : ℝ := sorry
  let x3 : ℝ := sorry
  let x4 : ℝ := sorry
  let x5 : ℝ := sorry
  let x6 : ℝ := sorry
  if (0 <= x1) ∧ (x1 <= 15) ∧
     (0 <= x2) ∧ (x2 <= 15) ∧
     (0 <= x3) ∧ (x3 <= 15) ∧
     (0 <= x4) ∧ (x4 <= 15) ∧
     (0 <= x5) ∧ (x5 <= 15) ∧
     (0 <= x6) ∧ (x6 <= 15) ∧
     (x1 < x6) ∧ (x2 < x6) ∧ (x3 < x6) ∧ (x4 < x6) ∧ (x5 < x6) then 
    let p_not_A : ℝ := (12 / 15) ^ 6
    1 - p_not_A
  else
    0

theorem probability_is_correct : probability_cashier_opens_early = 0.738 :=
by sorry

end probability_is_correct_l529_529435


namespace trigonometric_identity_simplification_l529_529671

theorem trigonometric_identity_simplification :
  (Real.sin (15 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) + Real.cos (15 * Real.pi / 180) * Real.sin (105 * Real.pi / 180) = 1) :=
by sorry

end trigonometric_identity_simplification_l529_529671


namespace castor_chess_players_l529_529659

theorem castor_chess_players (total_players : ℕ) (never_lost_to_ai : ℕ)
  (h1 : total_players = 40) (h2 : never_lost_to_ai = total_players / 4) :
  (total_players - never_lost_to_ai) = 30 :=
by
  sorry

end castor_chess_players_l529_529659


namespace prime_triplet_l529_529506

theorem prime_triplet (a b c : ℕ) :
  nat.prime a ∧ nat.prime b ∧ nat.prime c ∧ a < b ∧ b < c ∧ c < 100 ∧ 
  ((b + 1)*(b + 1) = (a + 1)*(c + 1))
  ↔ 
  (a = 2 ∧ b = 5 ∧ c = 11) ∨ 
  (a = 2 ∧ b = 11 ∧ c = 47) ∨ 
  (a = 5 ∧ b = 11 ∧ c = 23) ∨ 
  (a = 5 ∧ b = 17 ∧ c = 53) ∨ 
  (a = 7 ∧ b = 11 ∧ c = 17) ∨ 
  (a = 7 ∧ b = 23 ∧ c = 71) ∨ 
  (a = 11 ∧ b = 23 ∧ c = 47) ∨ 
  (a = 17 ∧ b = 23 ∧ c = 31) ∨ 
  (a = 17 ∧ b = 41 ∧ c = 97) ∨ 
  (a = 31 ∧ b = 47 ∧ c = 71) ∨ 
  (a = 71 ∧ b = 83 ∧ c = 97) :=
sorry

end prime_triplet_l529_529506


namespace max_value_log_expression_l529_529580

theorem max_value_log_expression (a b : ℝ) (ha : a ≥ b) (hb : 1 < b) : 
  ∃ (c : ℝ), (c = 1 ∧ c = log a b) ∧ log a (a / b) + log b (b / a) = 0 :=
by
  sorry

end max_value_log_expression_l529_529580


namespace find_lambda_l529_529536

variables {V : Type*} [inner_product_space ℝ V]
variable {a b : V}
variable {λ : ℝ}

-- Given: a and b are unit vectors and mutually perpendicular
variable (ha : ⟪a, a⟫ = 1)
variable (hb : ⟪b, b⟫ = 1)
variable (h : ⟪a, b⟫ = 0)

-- Condition: The angle between (a + b) and (λa - b) is obtuse
def is_obtuse (u v : V) : Prop := inner u v < 0

-- Desired: Find a suitable value for λ
theorem find_lambda (h_obtuse : is_obtuse (a + b) (λ • a - b)) : λ = 0 :=
sorry

end find_lambda_l529_529536


namespace correct_order_of_numbers_l529_529932

theorem correct_order_of_numbers :
  let a := (4 / 5 : ℝ)
  let b := (81 / 100 : ℝ)
  let c := 0.801
  (a ≤ c ∧ c ≤ b) :=
by
  sorry

end correct_order_of_numbers_l529_529932


namespace probability_neither_A_nor_B_l529_529705

-- Definitions
def P (event : String) : ℝ -- Probability function (event as a string input)
| "A" := 0.25
| "B" := 0.40
| "A and B" := 0.15
| _ := sorry -- For any other event we use 'sorry' to indicate undefinedness.

-- Theorem statement
theorem probability_neither_A_nor_B :
  P("A") = 0.25 → P("B") = 0.40 → P("A and B") = 0.15 → P("not A and not B") = 0.50 :=
by
  intros hPA hPB hPAB
  have hPAnorB : P("A or B") = P("A") + P("B") - P("A and B"), sorry
  have hPnotAnotB : P("not A and not B") = 1 - P("A or B"), sorry
  exact hPnotAnotB

end probability_neither_A_nor_B_l529_529705


namespace perpendicular_midline_l529_529299

-- Definitions of Midpoints and Perpendicularity
variables {A B C D E F M : Type*}

-- Midpoints of sides of the triangle
def is_midpoint (P Q R : Type*) [has_add Q] [has_inv Q] [has_smul ℝ Q] (x y : P → Q) :=
  ∀ p : P, x p = (y p + y (R → P)) / 2

def midpoint_AB := is_midpoint D A B
def midpoint_AC := is_midpoint E A C
def midpoint_BC := is_midpoint F B C

-- Line l perpendicular to BC at F (ℝ represents the real number line)
def perp_to_BC_at_F (P Q R : Type*) [inner_product_space ℝ Q] {v w : Type*} :
  line (Q → P) → line (v → Q) → Prop :=
  ∀ (l : line (Q → P)) (bc : line (v → Q)), perpendicular l bc

-- Line DE passing through midpoints D and E
def line_DE := line (D → E)

-- Problem statement in Lean 4
theorem perpendicular_midline 
  (A B C D E F : Type*) 
  (h_midpoint1 : midpoint_AB D A B) 
  (h_midpoint2 : midpoint_AC E A C) 
  (h_midpoint3 : midpoint_BC F B C) 
  (h_perp : perp_to_BC_at_F F line_BC)
  (line_DE) : perp_to_BC_at_F line_DE :=
begin
  -- proof goes here
  sorry
end

end perpendicular_midline_l529_529299


namespace quadratic_vertex_coords_l529_529317

theorem quadratic_vertex_coords :
  ∀ x : ℝ, (y = (x-2)^2 - 1) → (2, -1) = (2, -1) :=
by
  sorry

end quadratic_vertex_coords_l529_529317


namespace largest_x_eq_48_div_7_l529_529108

theorem largest_x_eq_48_div_7 :
  ∃ x : ℝ, (⟨floor x / x⟩ = 7 / 8) ∧ (x = 48 / 7) := 
begin
  sorry
end

end largest_x_eq_48_div_7_l529_529108


namespace matrix_multiplication_correct_l529_529472

open Matrix

-- Define the first matrix
def matrixA : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 2, -4]

-- Define the second matrix
def matrixB : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; 0, 2]

-- Define the expected product of the matrices
def expectedProduct : Matrix (Fin 2) (Fin 2) ℤ := !![21, -7; 14, -14]

-- The theorem stating the matrix multiplication result
theorem matrix_multiplication_correct : matrixA ⬝ matrixB = expectedProduct :=
by
  sorry

end matrix_multiplication_correct_l529_529472


namespace quadrilateral_is_rhombus_l529_529319

-- Define the properties of a quadrilateral and equality of perimeters
variables {A B C D O : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace O]
variables (ABO BCO CDO DAO : Triangle A B O) (ABO BCO CDO DAO : Triangle B C O) (ABO BCO CDO DAO : Triangle C D O) (ABO BCO CDO DAO : Triangle D A O)

-- Assume the perimeters of the triangles ABO, BCO, CDO, DAO are equal
def equalPerimeters (ABO : Triangle A B O) (BCO : Triangle B C O) (CDO : Triangle C D O) (DAO : Triangle D A O) : Prop :=
  perimeter ABO = perimeter BCO ∧ perimeter BCO = perimeter CDO ∧ perimeter CDO = perimeter DAO 

-- State that ABCD is a quadrilateral
variables {A B C D O : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace O]
variables (h_quad : Quadrilateral A B C D O)

-- Define a rhombus condition
def isRhombus (ABCD : Quadrilateral A B C D O) : Prop := 
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A

-- Theorem statement
theorem quadrilateral_is_rhombus
  (h_intersection : intersectsDiagonals A B C D O)
  (h_same_perimeter : equalPerimeters ABO BCO CDO DAO) :
  isRhombus ABCD :=
sorry

end quadrilateral_is_rhombus_l529_529319


namespace simplify_fraction_l529_529305

-- Given
def num := 54
def denom := 972

-- Factorization condition
def factorization_54 : num = 2 * 3^3 := by 
  sorry

def factorization_972 : denom = 2^2 * 3^5 := by 
  sorry

-- GCD condition
def gcd_num_denom := 54

-- Division condition
def simplified_num := 1
def simplified_denom := 18

-- Statement to prove
theorem simplify_fraction : (num / denom) = (simplified_num / simplified_denom) := by 
  sorry

end simplify_fraction_l529_529305


namespace intersection_of_A_and_B_l529_529960

-- Define the sets A and B based on the given conditions
def A := {x : ℝ | x > 1}
def B := {x : ℝ | x ≤ 3}

-- Lean statement to prove the intersection of A and B matches the correct answer
theorem intersection_of_A_and_B : (A ∩ B) = {x : ℝ | 1 < x ∧ x ≤ 3} :=
by {
  sorry
}

end intersection_of_A_and_B_l529_529960


namespace baker_cakes_l529_529801

theorem baker_cakes (cakes_made : ℕ) (cakes_bought : ℕ) (cakes_left : ℕ) 
  (h1 : cakes_made = 155) 
  (h2 : cakes_bought = 140) 
  (h3 : cakes_left = cakes_made - cakes_bought) : 
  cakes_left = 15 :=
by {
  rw [h1, h2] at h3,
  exact h3,
  sorry
}

end baker_cakes_l529_529801


namespace cashier_opens_probability_l529_529449

-- Definition of the timeline and arrival times
variables {x₁ x₂ x₃ x₄ x₅ x₆ : ℝ}
-- Condition that all arrival times are between 0 and 15 minutes
def arrival_times_within_bounds : Prop := 
  0 ≤ x₁ ∧ x₁ ≤ 15 ∧ 
  0 ≤ x₂ ∧ x₂ ≤ 15 ∧
  0 ≤ x₃ ∧ x₃ ≤ 15 ∧
  0 ≤ x₄ ∧ x₄ ≤ 15 ∧
  0 ≤ x₅ ∧ x₅ ≤ 15 ∧
  0 ≤ x₆ ∧ x₆ ≤ 15

-- Condition that the Scientist arrives last
def scientist_arrives_last : Prop := 
  x₁ < x₆ ∧ x₂ < x₆ ∧ x₃ < x₆ ∧ x₄ < x₆ ∧ x₅ < x₆

-- Event A: Cashier opens no later than 3 minutes after the Scientist arrives, i.e., x₆ ≥ 12
def event_A : Prop := x₆ ≥ 12

-- The correct answer
theorem cashier_opens_probability :
  arrival_times_within_bounds ∧ scientist_arrives_last → 
  Pr[x₆ ≥ 12 | x₁, x₂, x₃, x₄, x₅ < x₆] = 0.738 :=
sorry

end cashier_opens_probability_l529_529449


namespace lion_figurine_arrangement_l529_529943

theorem lion_figurine_arrangement (n : ℕ) (h_n : n = 9) :
  let special_positions := 2
  let remaining_figurines := 7
  let ways_to_arrange_remaining := Nat.factorial remaining_figurines
  in special_positions * ways_to_arrange_remaining = 10080 :=
by
  sorry

end lion_figurine_arrangement_l529_529943


namespace total_cost_correct_l529_529617

-- Define the costs for each repair
def engine_labor_cost := 75 * 16
def engine_part_cost := 1200
def brake_labor_cost := 85 * 10
def brake_part_cost := 800
def tire_labor_cost := 50 * 4
def tire_part_cost := 600

-- Calculate the total costs
def engine_total_cost := engine_labor_cost + engine_part_cost
def brake_total_cost := brake_labor_cost + brake_part_cost
def tire_total_cost := tire_labor_cost + tire_part_cost

-- Calculate the total combined cost
def total_combined_cost := engine_total_cost + brake_total_cost + tire_total_cost

-- The theorem to prove
theorem total_cost_correct : total_combined_cost = 4850 := by
  sorry

end total_cost_correct_l529_529617


namespace janet_total_miles_l529_529243

theorem janet_total_miles : 
  let week1 := 5 * 8,
      week2 := 4 * 10,
      week3 := 3 * 6
  in week1 + week2 + week3 = 98 :=
by {
  let week1 := 5 * 8,
  let week2 := 4 * 10,
  let week3 := 3 * 6,
  calc
  week1 + week2 + week3 = 40 + 40 + 18 : by sorry
                     ... = 98 : by sorry
}

end janet_total_miles_l529_529243


namespace compare_abc_l529_529539

noncomputable def a := Real.sqrt 0.3
noncomputable def b := Real.sqrt 0.4
noncomputable def c := Real.log 0.6 / Real.log 3

theorem compare_abc : c < a ∧ a < b :=
by
  -- Proof goes here
  sorry

end compare_abc_l529_529539


namespace bowling_team_score_ratio_l529_529407

theorem bowling_team_score_ratio :
  ∀ (F S T : ℕ),
  F + S + T = 810 →
  F = (1 / 3 : ℚ) * S →
  T = 162 →
  S / T = 3 := 
by
  intros F S T h1 h2 h3
  sorry

end bowling_team_score_ratio_l529_529407


namespace arithmetic_sequence_sixth_term_l529_529344

theorem arithmetic_sequence_sixth_term (a d : ℤ) 
    (sum_first_five : a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) = 15)
    (fourth_term : a + 3 * d = 4) : a + 5 * d = 6 :=
by
  sorry

end arithmetic_sequence_sixth_term_l529_529344


namespace max_log2_sum_l529_529226

noncomputable def arithmetic_seq (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, (a (n + 1) - a n) = (a (m + 1) - a m)

theorem max_log2_sum (a : ℕ → ℝ) (h_arith: arithmetic_seq a) 
  (h_positive: ∀ n, 0 < a n) 
  (h_mean: (a 4 + a 14) / 2 = 8) :
  ∃ (max_val : ℝ), max_val = 6 ∧ log (a 7) / log 2 + log (a 11) / log 2 ≤ max_val := 
sorry

end max_log2_sum_l529_529226


namespace excenter_locus_l529_529392

-- Define the hyperbola and conditions
variables {a b c x y : ℝ}
variables (P : ℝ × ℝ)
variables (F₁ F₂ : ℝ × ℝ)
variables (hyperbola_eq : P.1^2 / a^2 - P.2^2 / b^2 = 1)
variables (right_branch : P.1 > max a b)
variables (foci : F₁ = (-c, 0) ∧ F₂ = (c, 0))
variables (c_positive : c > 0)

-- Define the target equation for the locus of the excenter
def locus_of_excenter (q : ℝ × ℝ) : Prop :=
  (c - a) * q.1^2 - (c + a) * q.2^2 = (c - a) * c^2 ∧ q.1 > c

-- Theorem to prove the locus equation
theorem excenter_locus : ∀ (q : ℝ × ℝ), locus_of_excenter c a (q) :=
sorry

end excenter_locus_l529_529392


namespace xy_value_l529_529425

theorem xy_value : ∀ (x y : ℕ), (x + y * 10) / 99 - (x + y * 10) / 100 < 1 / 36 → (x * 10 + y = 30) :=
by
  intro x y h
  have h1 : (x + y * 10) / 99 = (2 / 75 + (x * 10 + y) / 100) := sorry
  have h2 : (2 / 75 + (x * 10 + y) / 100) < 1 / 36 := sorry
  linarith

end xy_value_l529_529425


namespace polynomial_remainder_l529_529508

theorem polynomial_remainder 
  (p : Polynomial ℝ) : 
  (p.eval 2 = 4) ∧ (p.eval 4 = 8) → ∃ q : Polynomial ℝ, ∃ r : Polynomial ℝ, r.degree < 2 ∧ 
  (p = q * Polynomial.C (x - 2) * Polynomial.C (x - 4) + r) ∧ (r = Polynomial.C 2 * x) :=
by 
  sorry

end polynomial_remainder_l529_529508


namespace triangle_side_difference_l529_529230

theorem triangle_side_difference :
  ∀ (x : ℝ), 3 ≤ x ∧ x < 12 → (∃ (y z : ℝ), y = 11 ∧ z = 3 ∧ y - z = 8) :=
by
  intro x h
  use 11
  use 3
  split
  case left =>
    exact rfl
  case right =>
    split
    case left =>
      exact rfl
    case right =>
      have : 11 - 3 = 8 := rfl
      exact this

end triangle_side_difference_l529_529230


namespace cashier_window_open_probability_l529_529442

noncomputable def probability_window_opens_in_3_minutes_of_scientist_arrival : ℝ := 
  0.738

theorem cashier_window_open_probability :
  let x : ℝ → ℝ := λ x, if x ≥ 12 then 0.738 else 0.262144 in
  ∀ (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ),
    (∀ i ∈ [x₁, x₂, x₃, x₄, x₅], i < x₆) ∧ 0 ≤ x₆ ≤ 15 →
    x x₆ = 0.738 :=
by
  sorry

end cashier_window_open_probability_l529_529442


namespace solution_set_f_x_minus_1_leq_0_l529_529552

def f (x : ℝ) : ℝ :=
if x ≥ 1 then 2^(x - 1) - 2 else 2^(1 - x) - 2

theorem solution_set_f_x_minus_1_leq_0 :
  { x : ℝ | f (x - 1) ≤ 0 } = { x : ℝ | 1 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end solution_set_f_x_minus_1_leq_0_l529_529552


namespace eval_32_pow_5_div_2_l529_529488

theorem eval_32_pow_5_div_2 :
  32^(5/2) = 4096 * Real.sqrt 2 :=
by
  sorry

end eval_32_pow_5_div_2_l529_529488


namespace find_extrema_l529_529836

noncomputable def function_extrema (x : ℝ) : ℝ :=
  (2 / 3) * Real.cos (3 * x - Real.pi / 6)

theorem find_extrema :
  (function_extrema (Real.pi / 18) = 2 / 3 ∧
   function_extrema (7 * Real.pi / 18) = -(2 / 3)) ∧
  (0 < Real.pi / 18 ∧ Real.pi / 18 < Real.pi / 2) ∧
  (0 < 7 * Real.pi / 18 ∧ 7 * Real.pi / 18 < Real.pi / 2) :=
by
  sorry

end find_extrema_l529_529836


namespace loan_amount_l529_529988

theorem loan_amount
  (P : ℝ)
  (SI : ℝ := 704)
  (R : ℝ := 8)
  (T : ℝ := 8)
  (h : SI = (P * R * T) / 100) : P = 1100 :=
by
  sorry

end loan_amount_l529_529988


namespace constant_term_in_expansion_l529_529714

-- Define the problem parameters
def polynomial (a : ℚ) (x : ℚ) : ℚ :=
  (2 * x + a / x) * (x - 2 / x) ^ 5

-- Prove the equivalent problem in Lean 4
theorem constant_term_in_expansion : 
  (∀ x : ℚ, polynomial (-1) x = -1) → (constant_term_in_expansion (2 * x - (1 / x) * (x - 2 / x) ^ 5)) = -200 :=
by
  intro h
  sorry

end constant_term_in_expansion_l529_529714


namespace real_root_of_polynomial_l529_529507

theorem real_root_of_polynomial :
  ∀ x : ℝ, (x^4 - 4*x^3 + 5*x^2 - 2*x + 2 = 0) → x = 1 :=
by 
  intro x
  assume h
  sorry

end real_root_of_polynomial_l529_529507


namespace log_relationship_l529_529895

theorem log_relationship (a b : ℝ) (x : ℝ) (h₁ : 6 * (Real.log (x) / Real.log (a)) ^ 2 + 5 * (Real.log (x) / Real.log (b)) ^ 2 = 12 * (Real.log (x) ^ 2) / (Real.log (a) * Real.log (b))) :
  a = b^(5/3) ∨ a = b^(3/5) := by
  sorry

end log_relationship_l529_529895


namespace largest_x_satisfies_condition_l529_529096

theorem largest_x_satisfies_condition :
  ∃ x : ℝ, (⌊x⌋ / x = 7 / 8) ∧ (∀ y : ℝ, (⌊y⌋ / y = 7 / 8) → y ≤ 48 / 7) :=
sorry

end largest_x_satisfies_condition_l529_529096


namespace range_of_a_l529_529174

open Real

noncomputable def f (a x : ℝ) : ℝ := log a (x^2 - 2 * a * x)

theorem range_of_a {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x1 x2 : ℝ, (3 ≤ x1 ∧ x1 ≤ 4) → (3 ≤ x2 ∧ x2 ≤ 4) → x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) > 0) :
  1 < a ∧ a < (3 / 2) :=
sorry

end range_of_a_l529_529174


namespace mrs_hilt_snow_l529_529030

theorem mrs_hilt_snow : 
  ∀ (snow_hilt : ℕ), snow_hilt = 29 → snow_hilt = 29 :=
by
  intros snow_hilt h
  exact h
sorry

end mrs_hilt_snow_l529_529030


namespace probability_multiple_of_9_in_T_l529_529564

def T : Finset ℤ := 
  (Finset.range 31).bUnion (λ a, 
    (Finset.Icc a 30).image (λ b, 5^a + 5^b))

def multiple_of_9 (n : ℤ) : Prop := n % 9 = 0

def probability (s : Finset ℤ) (p : ℤ → Prop) : ℚ := 
  s.filter p).card.to_rat / s.card.to_rat

theorem probability_multiple_of_9_in_T : probability T multiple_of_9 = 5 / 31 :=
sorry

end probability_multiple_of_9_in_T_l529_529564


namespace mode_of_dataset_is_5_l529_529849

def dataset : Multiset ℕ := {0, 1, 2, 3, 3, 5, 5, 5}

theorem mode_of_dataset_is_5 : Multiset.mode dataset = 5 := 
by
  sorry

end mode_of_dataset_is_5_l529_529849


namespace max_xy_l529_529584

theorem max_xy (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 4 * x^2 + 9 * y^2 + 3 * x * y = 30) :
  xy ≤ 2 :=
by
  sorry

end max_xy_l529_529584


namespace circle_representation_l529_529320

theorem circle_representation (k : ℝ) :
  (∃ c1 c2 r, (∀ x y, x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0) → (x + c1)^2 + (y + c2)^2 = r^2) ↔ (k > 4 ∨ k < -1) :=
by
  sorry

end circle_representation_l529_529320


namespace sum_of_given_geom_series_l529_529048

-- Define the necessary conditions
def first_term (a : ℕ) := a = 2
def common_ratio (r : ℕ) := r = 3
def number_of_terms (n : ℕ) := n = 6

-- Define the sum of the geometric series
def sum_geom_series (a r n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

-- State the theorem
theorem sum_of_given_geom_series :
  first_term 2 → common_ratio 3 → number_of_terms 6 → sum_geom_series 2 3 6 = 728 :=
by
  intros h1 h2 h3
  rw [first_term] at h1
  rw [common_ratio] at h2
  rw [number_of_terms] at h3
  have h1 : 2 = 2 := by exact h1
  have h2 : 3 = 3 := by exact h2
  have h3 : 6 = 6 := by exact h3
  exact sorry

end sum_of_given_geom_series_l529_529048


namespace find_a_b_l529_529053

variable (a b : ℤ)

def mat1 := ![![4, -9], ![a, 14]]
def mat2 := ![![14, b], ![5, 4]]
def identity_matrix := ![![1, 0], ![0, 1]]

def matrices_are_inverses : Prop := mat1 ⬝ mat2 = identity_matrix

theorem find_a_b (h : matrices_are_inverses a b) : (a, b) = (-5, 9) :=
by sorry

end find_a_b_l529_529053


namespace triangle_side_lengths_l529_529600

theorem triangle_side_lengths 
  (A B C : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (concurrent : IsConcurrent (angleBisector A) (median B) (altitude C))
  (perpendicular : IsPerpendicular (angleBisector A) (median B))
  (AB_unit : distance A B = 1) :
  ∃ (AC BC : ℝ), AC = 2 ∧ BC = real.sqrt(33) / 3 :=
by sorry

end triangle_side_lengths_l529_529600


namespace circumcircle_tangent_circ_l529_529530

-- Conditions
variables (O₁ O₂ : Type) [MetricSpace O₁] [MetricSpace O₂]
variables (A B M : O₁) (P Q : O₁) (l₁ l₂ : Set O₁)

-- Assume properties of the geometric figures
variable [CircumCircle A B : O₁]

-- Defining initial geometric context
def midpoint_of_arc (M : O₁) (A B : O₁) (circ : O₁) := sorry

def chord_intersec (M P : O₁) (circ₁ circ₂ : O₁) := sorry

def tangent_to_circle (P : O₁) (circ : O₁) := sorry

-- Introduce circles and points
axiom h1 : M = midpoint_of_arc M A B O₁ ∧ 
           chord_intersec M P O₁ O₂ = Q ∧ 
           tangent_to_circle P O₁ = l₁ ∧ 
           tangent_to_circle Q O₂ = l₂ 

-- Problem Statement
theorem circumcircle_tangent_circ (O₁ O₂ : Type) [MetricSpace O₁] [MetricSpace O₂]
  (A B M P Q : O₁) (l₁ l₂ : Set O₁) :
  midpoint_of_arc M A B O₁ → chord_intersec M P O₁ O₂ = Q → 
  tangent_to_circle P O₁ = l₁ → tangent_to_circle Q O₂ = l₂ →
  tangent_to_circle (circumCircle l₁ l₂ A B) O₂ :=
  sorry

end circumcircle_tangent_circ_l529_529530


namespace max_length_interval_l529_529181

theorem max_length_interval
  (a : ℝ)
  (h_neg : a < 0)
  (h_bound : ∀ x ∈ [0, (1 + sqrt 5) / 2], abs (a * x^2 + 8 * x + 3) ≤ 5) :
  a = -8 ∧ (1 + sqrt 5) / 2 = (∃ l, (l = (1 + sqrt 5) / 2) ∧ ∀ x ∈ [0, l], abs (a * x^2 + 8 * x + 3) ≤ 5) :=
sorry

end max_length_interval_l529_529181


namespace ratio_of_constants_l529_529397

theorem ratio_of_constants (a b c: ℝ) (h1 : 8 = 0.02 * a) (h2 : 2 = 0.08 * b) (h3 : c = b / a) : c = 1 / 16 :=
by sorry

end ratio_of_constants_l529_529397


namespace player_A_wins_l529_529686

-- Definitions of the game conditions
def points_on_circle : ℕ := 10
def first_player : string := "A"
def second_player : string := "B"
def segment_cross (segments : list (ℕ × ℕ)) : Prop := 
  sorry -- Placeholder for the function determining if segments cross

-- Theorem: Player A has a winning strategy
theorem player_A_wins :
  ∃ strategy_A : (ℕ → (ℕ × ℕ)), ∀ strategy_B : (ℕ → (ℕ × ℕ)),
    -- Conditions ensuring the given strategies lead to a win for Player A
    let moves := (λ n, if n % 2 = 0 then strategy_A n else strategy_B n) in
    ∀ n < points_on_circle * (points_on_circle - 1) / 2,
      ¬ segment_cross (map moves (list.range n)) →
      (n % 2 = 0 → moves n = strategy_A n) ∧ 
      (n % 2 = 1 → moves n = strategy_B n) →
      segment_cross (map moves (list.range (n + 1))) →
      n % 2 = 1 
          :=
sorry

end player_A_wins_l529_529686


namespace bn_arithmetic_sum_formula_l529_529525

variable {n : ℕ} (h1 : n > 0)

-- Definition of the sequence a_n
def a : ℕ → ℚ
| 1     => 2
| (n+1) => 2 - (1 / a (n + 1))

-- Definition of the sequence b_n
def b (n : ℕ) : ℚ := 1 / (a n - 1)

-- Proving that b_n forms an arithmetic sequence
theorem bn_arithmetic : ∀ (n : ℕ), n > 0 → b (n + 1) - b n = 1 :=
  sorry

-- Definition of S_n
def S (n : ℕ) : ℚ := (1 / 3) * ∑ i in Finset.range n, b i

-- Proving the given sum formula
theorem sum_formula : ∑ k in Finset.range (n+1), 1 / S k = 6 * (1 - 1 / (n + 1)) :=
  sorry

end bn_arithmetic_sum_formula_l529_529525


namespace calc1_calc2_calc3_l529_529758

-- Problem 1
theorem calc1 : 96 * 15 / (45 * 16) = 2 :=
by sorry

-- Problem 2
theorem calc2 : 125^100 * 8^101 = 10^300 * 80 :=
by sorry

-- Problem 3
theorem calc3 : (finset.range 50).sum (λ n, 2 * (n + 1)) = 2550 :=
by sorry

end calc1_calc2_calc3_l529_529758


namespace total_cost_l529_529029

def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def num_sandwiches : ℕ := 4
def num_sodas : ℕ := 5

theorem total_cost : (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost) = 31 := by
  sorry

end total_cost_l529_529029


namespace yellow_balls_after_loss_is_zero_l529_529303

noncomputable theory -- Specify noncomputable theory if necessary.

-- Define the initial and current conditions
def initial_total_balls := 120
def current_total_balls := 110
def blue_balls := 15

-- Define the relationship between the ball colors
def red_balls := 3 * blue_balls
def green_balls := red_balls + blue_balls
def yellow_balls_before_loss := initial_total_balls - (red_balls + blue_balls + green_balls)
def yellow_balls_after_loss := yellow_balls_before_loss -- This remains the same as before because losses don't change the originally zero count of yellow balls

-- Prove that the yellow balls count is zero after the loss
theorem yellow_balls_after_loss_is_zero :
  yellow_balls_after_loss = 0 :=
by 
  sorry -- Proof placeholder

end yellow_balls_after_loss_is_zero_l529_529303


namespace cashier_window_open_probability_l529_529444

noncomputable def probability_window_opens_in_3_minutes_of_scientist_arrival : ℝ := 
  0.738

theorem cashier_window_open_probability :
  let x : ℝ → ℝ := λ x, if x ≥ 12 then 0.738 else 0.262144 in
  ∀ (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ),
    (∀ i ∈ [x₁, x₂, x₃, x₄, x₅], i < x₆) ∧ 0 ≤ x₆ ≤ 15 →
    x x₆ = 0.738 :=
by
  sorry

end cashier_window_open_probability_l529_529444


namespace number_is_46000050_l529_529414

-- Define the corresponding place values for the given digit placements
def ten_million (n : ℕ) : ℕ := n * 10000000
def hundred_thousand (n : ℕ) : ℕ := n * 100000
def hundred (n : ℕ) : ℕ := n * 100

-- Define the specific numbers given in the conditions.
def digit_4 : ℕ := ten_million 4
def digit_60 : ℕ := hundred_thousand 6
def digit_500 : ℕ := hundred 5

-- Combine these values to form the number
def combined_number : ℕ := digit_4 + digit_60 + digit_500

-- The theorem, stating the number equals 46000050
theorem number_is_46000050 : combined_number = 46000050 := by
  sorry

end number_is_46000050_l529_529414


namespace coincidence_nineteenth_time_l529_529315

-- Define the speeds of the minute and hour hands
def minute_hand_speed : ℝ := 6
def hour_hand_speed : ℝ := 0.5

-- Define the relative speed at which the minute hand approaches the hour hand
def relative_speed : ℝ := minute_hand_speed - hour_hand_speed

-- Define the time it takes for one coincidence of the minute and hour hands
def time_for_one_coincidence : ℝ := 360 / relative_speed

-- Define the total time for 19 coincidences
def total_time_for_19_coincidences : ℝ := 19 * time_for_one_coincidence

-- The target value rounded to two decimal places
def target_time : ℝ := 1243.64

-- The main theorem stating the 19th coincidence happens at the target time
theorem coincidence_nineteenth_time : total_time_for_19_coincidences = target_time :=
by sorry

end coincidence_nineteenth_time_l529_529315


namespace derivative_of_y_l529_529839

open Real

noncomputable def y (x : ℝ) : ℝ := (cos (2 * x)) ^ ((log (cos (2 * x))) / 4)

theorem derivative_of_y (x : ℝ) :
  deriv y x = -((cos (2 * x)) ^ ((log (cos (2 * x))) / 4)) * (tan (2 * x)) * (log (cos (2 * x))) := by
    sorry

end derivative_of_y_l529_529839


namespace largest_x_eq_48_div_7_l529_529107

theorem largest_x_eq_48_div_7 :
  ∃ x : ℝ, (⟨floor x / x⟩ = 7 / 8) ∧ (x = 48 / 7) := 
begin
  sorry
end

end largest_x_eq_48_div_7_l529_529107


namespace inequality_proof_l529_529970

open Real

variable {n : ℕ} (x : Fin n → ℝ)

theorem inequality_proof (h₁ : n ≥ 2) (h₂ : ∀ i, 0 < x i) (h₃ : (∑ i, x i) = 1) :
  (∑ i, x i / sqrt (1 - x i)) ≥ (1 / sqrt (n - 1)) * (∑ i, sqrt (x i)) :=
by
  sorry

end inequality_proof_l529_529970


namespace simplify_expression_l529_529370

theorem simplify_expression : (-1/343 : ℂ) ^ (-3/2) = -7 * complex.I * real.sqrt 7 := 
sorry

end simplify_expression_l529_529370


namespace solution_proof_l529_529432

noncomputable def problem_statement : Prop :=
  let x := [x1, x2, x3, x4, x5, x6] in
  let B := (∀ i < 5, x[i] < x[5]) in
  let A := (x[5] ≥ 12) in
  ∃ x1 x2 x3 x4 x5 x6 : ℝ,
    (0 ≤ x1 ∧ x1 ≤ 15) ∧ (0 ≤ x2 ∧ x2 ≤ 15) ∧ (0 ≤ x3 ∧ x3 ≤ 15) ∧
    (0 ≤ x4 ∧ x4 ≤ 15) ∧ (0 ≤ x5 ∧ x5 ≤ 15) ∧ (0 ≤ x6 ∧ x6 ≤ 15) ∧
    B ∧ (classical.some ((measure_theory.measure_space.measure (λ x, x < 12 <= x) B).to_real) = 0.738)

theorem solution_proof : problem_statement := sorry

end solution_proof_l529_529432


namespace min_value_of_f_l529_529173

def f (x : ℝ) (a : ℝ) := - x^3 + a * x^2 - 4

def f_deriv (x : ℝ) (a : ℝ) := - 3 * x^2 + 2 * a * x

theorem min_value_of_f (h : f_deriv (2) a = 0)
  (hm : ∀ m : ℝ, -1 ≤ m ∧ m ≤ 1 → f m a + f_deriv m a ≥ f 0 3 + f_deriv (-1) 3) :
  f 0 3 + f_deriv (-1) 3 = -13 :=
by sorry

end min_value_of_f_l529_529173


namespace minimum_x_plus_2y_exists_l529_529876

theorem minimum_x_plus_2y_exists (x y : ℝ) (h : x^2 + 4 * y^2 - 2 * x + 8 * y + 1 = 0) :
  ∃ z : ℝ, z = x + 2 * y ∧ z = -2 * Real.sqrt 2 - 1 :=
sorry

end minimum_x_plus_2y_exists_l529_529876


namespace hyperbola_C_properties_l529_529888

noncomputable def hyperbola_eccentricities : ℝ → Prop :=
  λ e, e = (Real.sqrt 7) / 2 ∨ e = (Real.sqrt 21) / 3

theorem hyperbola_C_properties :
  (∃ (e : ℝ), hyperbola_eccentricities e) ∧
  (∃ (m : ℝ), (2 * 2) / 8 - 1 / 6 = m ∧ m = 1 / 2) :=
by
  sorry

end hyperbola_C_properties_l529_529888


namespace smallest_integer_condition_solution_l529_529635

def smallest_integer_coloring_condition (k : ℕ) (hk : k > 1) : ℕ :=
  4 * k

theorem smallest_integer_condition_solution (k : ℕ) (hk : k > 1) :
  ∃ n, (∀ i j, (i < n ∧ j < n) → (∃ b : fin n → fin n → bool,
  (∀ i, (∑ j in finset.range n, if b i j then 1 else 0) = k) ∧
  (∀ j, (∑ i in finset.range n, if b i j then 1 else 0) = k) ∧
  (∀ i j, b i j = tt → (∀ di dj, ∃ bb, (|di| ≤ 1 ∧ |dj| ≤ 1 ∧ (di ≠ 0 ∨ dj ≠ 0)) → b (i + di) (j + dj) = bb)) )) ∧
  n = smallest_integer_coloring_condition k hk :=
  ∃ n, n = 4 * k :=
sorry

end smallest_integer_condition_solution_l529_529635


namespace valid_assignments_count_l529_529873

noncomputable def validAssignments : Nat := sorry

theorem valid_assignments_count : validAssignments = 4 := 
by {
  sorry
}

end valid_assignments_count_l529_529873


namespace find_x_l529_529934

theorem find_x (x : ℤ) (h1 : 5 < x) (h2 : x < 21) (h3 : 7 < x) (h4 : x < 18) (h5 : 2 < x) (h6 : x < 13) (h7 : 9 < x) (h8 : x < 12) (h9 : x < 12) :
  x = 10 :=
sorry

end find_x_l529_529934


namespace min_area_triangle_ABO_l529_529325

theorem min_area_triangle_ABO (k : ℝ) (hk : k > 0) :
  let A := (- (1 + 1/k), 0)
  let B := (0, k + 1)
  let O := (0, 0)
  let area := 1/2 * abs (fst A * snd B - snd A * fst B)
  area = 2 :=
by
  sorry

end min_area_triangle_ABO_l529_529325


namespace area_of_triangle_ABC_l529_529949

/--
Given a triangle ABC where BC is 12 cm and the height from A
perpendicular to BC is 15 cm, prove that the area of the triangle is 90 cm^2.
-/
theorem area_of_triangle_ABC (BC : ℝ) (hA : ℝ) (h_BC : BC = 12) (h_hA : hA = 15) : 
  1/2 * BC * hA = 90 := 
sorry

end area_of_triangle_ABC_l529_529949


namespace finite_path_directions_l529_529971

theorem finite_path_directions
  (α β γ : ℝ)
  (hα : ∃ (k1 : ℕ) (m : ℕ), α = k1 * (π / m))
  (hβ : ∃ (k2 : ℕ) (m : ℕ), β = k2 * (π / m))
  (hγ : ∃ (k3 : ℕ) (m : ℕ), γ = k3 * (π / m))
  (sum_angles : α + β + γ = π)
  (h_never_reaches_vertices : ∀ (P : Π (A B C : ℝ), P ≠ A ∧ P ≠ B ∧ P ≠ C)) :
  ∃ (n : ℕ), ∀ t : ℝ, ∃ (ϕ : ℝ), (∃ i : ℕ, ϕ = (i * π / n) ∨ ϕ = (i * π / n + φ)) :=
sorry

end finite_path_directions_l529_529971


namespace largest_x_satisfies_condition_l529_529087

theorem largest_x_satisfies_condition (x : ℝ) (h : (⌊x⌋ / x) = 7 / 8) : x ≤ 48 / 7 :=
sorry

end largest_x_satisfies_condition_l529_529087


namespace unique_solution_a_eq_sqrt_three_l529_529550

theorem unique_solution_a_eq_sqrt_three {a : ℝ} (h1 : ∀ x y : ℝ, x^2 + a * abs x + a^2 - 3 = 0 ∧ y^2 + a * abs y + a^2 - 3 = 0 → x = y)
  (h2 : a > 0) : a = Real.sqrt 3 := by
  sorry

end unique_solution_a_eq_sqrt_three_l529_529550


namespace locus_of_points_is_straight_line_l529_529657

theorem locus_of_points_is_straight_line (a R1 R2 : ℝ) :
  ∀ x y, ((x + a)^2 + y^2 - R1^2 = (x - a)^2 + y^2 - R2^2) → (x = (R1^2 - R2^2) / (4 * a)) := 
begin
  intros x y h,
  sorry
end

end locus_of_points_is_straight_line_l529_529657


namespace vectors_not_collinear_l529_529391

variables (a b c1 c2 : EuclideanSpace ℝ (Fin 3))

-- Given conditions
def a : EuclideanSpace ℝ (Fin 3) := ![-2, 7, -1]
def b : EuclideanSpace ℝ (Fin 3) := ![-3, 5, 2]
def c1 : EuclideanSpace ℝ (Fin 3) := 2 • a + 3 • b
def c2 : EuclideanSpace ℝ (Fin 3) := 3 • a + 2 • b

-- Theorem to prove
theorem vectors_not_collinear : ¬∃ γ : ℝ, c1 = γ • c2 := by 
  sorry

end vectors_not_collinear_l529_529391


namespace equalize_costs_l529_529251

theorem equalize_costs (A B : ℝ) (h_lt : A < B) :
  (B - A) / 2 = (A + B) / 2 - A :=
by sorry

end equalize_costs_l529_529251


namespace hyperbola_perpendicular_product_l529_529879

theorem hyperbola_perpendicular_product (m n : ℝ) (M : ℝ × ℝ)
  (hMn : M = (m, n)) (hM_hyperbola : m^2 - n^2 = 4) :
  let ON : ℝ := (m + n) / (real.sqrt 2),
      MN : ℝ := (m - n) / (real.sqrt 2)
  in ON * MN = 2 :=
by
  sorry

end hyperbola_perpendicular_product_l529_529879


namespace solve_fraction_equation_l529_529751

-- Defining the function f
def f (x : ℝ) : ℝ := x + 4

-- Statement of the problem
theorem solve_fraction_equation (x : ℝ) :
  (3 * f (x - 2)) / f 0 + 4 = f (2 * x + 1) ↔ x = 2 / 5 := by
  sorry

end solve_fraction_equation_l529_529751


namespace chuck_leash_area_l529_529809

noncomputable def chuck_play_area (r1 r2 : ℝ) : ℝ :=
  let A1 := (3 / 4) * Real.pi * (r1 ^ 2)
  let A2 := (1 / 4) * Real.pi * (r2 ^ 2)
  A1 + A2

theorem chuck_leash_area : chuck_play_area 3 1 = 7 * Real.pi :=
by
  sorry

end chuck_leash_area_l529_529809


namespace inequality_solution_l529_529307

theorem inequality_solution (x : ℝ) : abs ((3 * x + 2) / (x - 3)) < 4 ↔ x ∈ set.Ioo (10 / 7) 3 ∪ set.Ioo 3 14 :=
  sorry

end inequality_solution_l529_529307


namespace Ann_end_blocks_l529_529455

-- Define blocks Ann initially has and finds
def initialBlocksAnn : ℕ := 9
def foundBlocksAnn : ℕ := 44

-- Define blocks Ann ends with
def finalBlocksAnn : ℕ := initialBlocksAnn + foundBlocksAnn

-- The proof goal
theorem Ann_end_blocks : finalBlocksAnn = 53 := by
  -- Use sorry to skip the proof
  sorry

end Ann_end_blocks_l529_529455


namespace leopard_arrangement_count_l529_529651

theorem leopard_arrangement_count : 
  let leopards : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8} in
  let shortest_leopards : Finset ℕ := {1, 2} in
  let tallest_leopard : ℕ := 8 in
  let remaining_positions : Finset ℕ := {3, 4, 5, 6, 7} in
  let arrangement_count : ℕ := 2 * 1 * 5! in
  arrangement_count = 240 :=
by
  sorry

end leopard_arrangement_count_l529_529651


namespace time_to_cross_l529_529235

theorem time_to_cross : 
  let length_of_train := 240 -- in meters
  let speed_km_per_hr := 54 -- in km/hr
  let speed_m_per_s := speed_km_per_hr * 1000 / 3600 -- converting km/hr to m/s
  let expected_time := 16 -- in seconds
  in length_of_train / speed_m_per_s = expected_time :=
by
  let length_of_train := 240
  let speed_km_per_hr := 54
  let speed_m_per_s := speed_km_per_hr * 1000 / 3600
  let expected_time := 16
  sorry

end time_to_cross_l529_529235


namespace angle_sum_of_overlapping_triangles_l529_529473

theorem angle_sum_of_overlapping_triangles :
  ∀ (A B C D E F : ℝ),
  (∠A + ∠B + ∠C = 180) -> 
  (∠D + ∠E + ∠F = 180) ->
  (∠B = ∠E) ->
  ∠A + ∠B + ∠C + ∠D + ∠E + ∠F = 360 :=
by
  intros A B C D E F h1 h2 h3
  sorry

end angle_sum_of_overlapping_triangles_l529_529473


namespace number_of_sacks_after_49_days_l529_529190

def sacks_per_day : ℕ := 38
def days_of_harvest : ℕ := 49
def total_sacks_after_49_days : ℕ := 1862

theorem number_of_sacks_after_49_days :
  sacks_per_day * days_of_harvest = total_sacks_after_49_days :=
by
  sorry

end number_of_sacks_after_49_days_l529_529190


namespace min_length_segment_intersections_l529_529632

/--
Given a circle with diameter 4 and points X, Y, and Z on the described arcs,
let e be the length of the segment whose endpoints are the intersections of diameter PQ with chords XZ and YZ.

Prove that the smallest possible value of e is 6 - 5√3,
leading to u + v + w = 14 when e is written in the form u - v√w. 
-/
theorem min_length_segment_intersections 
  (diameter : ℝ)
  (midpoint_X : bool)
  (PY_length : ℝ)
  (Z_on_symmetric_arc : bool)
  (u v w : ℕ) :
  diameter = 4 →
  midpoint_X = true →
  PY_length = 5/4 →
  Z_on_symmetric_arc = true →
  u - v * Real.sqrt w = 6 - 5 * Real.sqrt 3 →
  ¬ ∃ p, p^2 ∣ w →
  u + v + w = 14 :=
by
  intros
  sorry

end min_length_segment_intersections_l529_529632


namespace map_distances_correct_l529_529655

noncomputable def RegionA_scale := 33 / 3 -- km/cm
noncomputable def RegionB_scale := 40 / 4 -- km/cm

variables (distance_XY distance_XZ distance_YZ : ℝ)
           (actual_distance_XY actual_distance_XZ actual_distance_YZ : ℝ)

-- Conditions
def condition1 : actual_distance_XY = 209 := rfl
def condition2 : actual_distance_XZ = 317 := rfl
def condition3 : actual_distance_YZ = 144 := rfl

-- Calculations based on the scale for Region A
def distance_on_map_XY := actual_distance_XY / RegionA_scale
def distance_on_map_XZ := actual_distance_XZ / RegionA_scale

-- Calculation based on the scale for Region B
def distance_on_map_YZ := actual_distance_YZ / RegionB_scale

theorem map_distances_correct :
  distance_on_map_XY = 19 ∧
  distance_on_map_XZ ≈ 28.82 ∧
  distance_on_map_YZ = 14.4 :=
by
  -- Skipping the proof part with sorry
  sorry

end map_distances_correct_l529_529655


namespace sum_inverse_values_l529_529477

def f (x : ℝ) : ℝ :=
  if x < 5 then x + 3 else x^2

noncomputable def f_inv (y : ℝ) : ℝ :=
  if y < 8 then y - 3 else real.sqrt y

theorem sum_inverse_values :
  ∑ i in finset.Ico (-2) 25, f_inv i = 5 :=
by
  sorry

end sum_inverse_values_l529_529477


namespace work_done_by_gravity_l529_529806

noncomputable def work_by_gravity (m g z_A z_B : ℝ) : ℝ :=
  m * g * (z_B - z_A)

theorem work_done_by_gravity (m g z_A z_B : ℝ) :
  work_by_gravity m g z_A z_B = m * g * (z_B - z_A) :=
by
  sorry

end work_done_by_gravity_l529_529806


namespace equilateral_triangle_sum_squares_correct_l529_529487

noncomputable def equilateral_triangle_sum_squares : ℝ :=
  let s := Real.sqrt 123 in
  let r := Real.sqrt 13 in
  4 * s^2

theorem equilateral_triangle_sum_squares_correct :
  let s := Real.sqrt 123 in
  let r := Real.sqrt 13 in
  (BD1 BD2 : ℝ) (CE1 CE2 CE3 CE4 : ℝ) (condition1 : s = Real.sqrt 123) (condition2 : BD1 = Real.sqrt 13) (condition3 : BD2 = Real.sqrt 13) :
  ∑ k in [CE1, CE2, CE3, CE4], k^2 = 492 :=
by
  sorry

end equilateral_triangle_sum_squares_correct_l529_529487


namespace range_of_a_l529_529551

-- Let f be a piecewise function
noncomputable def f (a : ℝ) : ℝ → ℝ :=
λ x : ℝ, if x ≤ 1 then (a - 2) * x - 1 else real.log x / real.log a

-- Define the statement in Lean where we check the monotonicity of f
theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) → (2 < a ∧ a ≤ 3) :=
begin
  sorry -- Proof is omitted as requested
end

end range_of_a_l529_529551


namespace radius_range_l529_529608

theorem radius_range (C A B : Type) (AC BC r : ℝ)
  (h_triangle : ∠ C = 90)
  (h_AC : AC = 5)
  (h_BC : BC = 8)
  (h_A_inside : ∀ A, dist A C < r)
  (h_B_outside : ∀ B, dist B C > r) :
  5 < r ∧ r < 8 :=
sorry

end radius_range_l529_529608


namespace total_golf_balls_l529_529692

theorem total_golf_balls :
  let dozen := 12
  let dan := 5 * dozen
  let gus := 2 * dozen
  let chris := 48
  dan + gus + chris = 132 :=
by
  let dozen := 12
  let dan := 5 * dozen
  let gus := 2 * dozen
  let chris := 48
  sorry

end total_golf_balls_l529_529692


namespace non_isosceles_triangle_has_equidistant_incenter_midpoints_l529_529610

structure Triangle (α : Type*) :=
(a b c : α)
(incenter : α)
(midpoint_a_b : α)
(midpoint_b_c : α)
(midpoint_c_a : α)
(equidistant : Bool)
(non_isosceles : Bool)

-- Define the triangle with the specified properties.
noncomputable def counterexample_triangle : Triangle ℝ :=
{ a := 3,
  b := 4,
  c := 5, 
  incenter := 1, -- incenter length for the right triangle.
  midpoint_a_b := 2.5,
  midpoint_b_c := 2,
  midpoint_c_a := 1.5,
  equidistant := true,    -- midpoints of two sides are equidistant from incenter
  non_isosceles := true } -- the triangle is not isosceles

theorem non_isosceles_triangle_has_equidistant_incenter_midpoints :
  ∃ (T : Triangle ℝ), T.equidistant ∧ T.non_isosceles := by
  use counterexample_triangle
  sorry

end non_isosceles_triangle_has_equidistant_incenter_midpoints_l529_529610


namespace pig_water_requirement_l529_529469

-- Definitions based on the conditions
def minutes_pumping := 25
def gallons_per_minute := 3
def rows := 4
def plants_per_row := 15
def water_per_plant := 0.5
def pigs := 10
def ducks := 20
def water_per_duck := 0.25

-- Total water pumped
def total_water := minutes_pumping * gallons_per_minute

-- Total water needed for corn plants
def total_water_corn := (rows * plants_per_row) * water_per_plant

-- Total water needed for ducks
def total_water_ducks := ducks * water_per_duck

-- Water remaining for pigs
def remaining_water := total_water - total_water_corn - total_water_ducks

-- Water per pig
def water_per_pig := remaining_water / pigs

-- Proof statement
theorem pig_water_requirement : water_per_pig = 4 := by sorry

end pig_water_requirement_l529_529469


namespace find_x_values_l529_529882

theorem find_x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 12) (h2 : y + 1 / x = 3 / 8) :
  x = 4 ∨ x = 8 :=
by
  sorry

end find_x_values_l529_529882


namespace indeterminable_minutes_l529_529250

noncomputable def drumming_contest (entry_fee : ℝ) (drums_to_earn : ℕ) (earn_per_drum : ℝ) (total_drums : ℕ) (net_loss : ℝ) : Prop :=
  let earnings := (total_drums - drums_to_earn) * earn_per_drum in
  net_loss = entry_fee - earnings

theorem indeterminable_minutes (entry_fee : ℝ) (drums_to_earn : ℕ) (earn_per_drum : ℝ) (total_drums : ℕ) (net_loss : ℝ) :
  drumming_contest entry_fee drums_to_earn earn_per_drum total_drums net_loss →
  entry_fee = 10 →
  drums_to_earn = 200 →
  earn_per_drum = 0.025 →
  total_drums = 300 →
  net_loss = 7.5 →
  false := by
  sorry

end indeterminable_minutes_l529_529250


namespace units_digit_of_x4_plus_inv_x4_l529_529861

theorem units_digit_of_x4_plus_inv_x4 (x : ℝ) (hx : x^2 - 13 * x + 1 = 0) : 
  (x^4 + x⁻¹ ^ 4) % 10 = 7 := sorry

end units_digit_of_x4_plus_inv_x4_l529_529861


namespace find_angle_B_find_area_l529_529951

variable (a b c : ℝ)
variables (A B C : ℝ)

/-- In triangle ABC, with sides a, b, and c opposite angles A, B, and C respectively, 
      and given the conditions -/
theorem find_angle_B (h1 : cos C = a / b - c / (2 * b)) (hB : B = π / 3) :
  B = π / 3 :=
sorry

theorem find_area (h1 : cos C = a / b - c / (2 * b))
                  (h2 : b = 2)
                  (h3 : a - c = 1)
                  (h4 : B = π / 3) :
  let ac := 3 in
  let area := 1/2 * a * b * sin C in
  area = 3 * (sqrt 3) / 4 :=
sorry

end find_angle_B_find_area_l529_529951


namespace johns_profit_l529_529248

theorem johns_profit 
  (bags_bought : ℕ)
  (discounted_price : ℕ → ℚ)
  (packaging_cost_per_bag : ℚ)
  (transportation_cost : ℚ)
  (bags_sold_to_adults : ℕ)
  (price_per_bag_adult : ℚ)
  (bags_sold_to_children : ℕ)
  (price_per_bag_child : ℚ)
  (total_profit : ℚ) :
  total_profit = 
  (bags_sold_to_adults * price_per_bag_adult + bags_sold_to_children * price_per_bag_child) 
  - (bags_bought * discounted_price bags_bought + packaging_cost_per_bag * bags_bought + transportation_cost) :=
by
  -- Given conditions and values.
  let bags_bought := 30
  let bags_sold_to_adults := 20
  let bags_sold_to_children := 10
  let original_price_per_bag := 4
  let discount := 0.1
  let discounted_price bags := if bags >= 20 then original_price_per_bag * (1 - discount) else original_price_per_bag
  let packaging_cost_per_bag := 0.5
  let transportation_cost := 10
  let price_per_bag_adult := 8
  let price_per_bag_child := 6

  -- Calculations already outlined in the solution
  let total_revenue := (bags_sold_to_adults * price_per_bag_adult) + (bags_sold_to_children * price_per_bag_child)
  let total_cost := (bags_bought * discounted_price bags_bought) + (packaging_cost_per_bag * bags_bought) + transportation_cost
  let total_profit := total_revenue - total_cost

  -- Theorem to prove
  have : total_profit = 87 := sorry
  sorry

end johns_profit_l529_529248


namespace car_drive_distance_l529_529462

-- Define the conditions as constants
def driving_speed : ℕ := 8 -- miles per hour
def driving_hours_before_cool : ℕ := 5 -- hours of constant driving
def cooling_hours : ℕ := 1 -- hours needed for cooling down
def total_time : ℕ := 13 -- hours available

-- Define the calculation for distance driven in cycles
def distance_per_cycle : ℕ := driving_speed * driving_hours_before_cool

-- Calculate the duration of one complete cycle
def cycle_duration : ℕ := driving_hours_before_cool + cooling_hours

-- Theorem statement: the car can drive 88 miles in 13 hours
theorem car_drive_distance : distance_per_cycle * (total_time / cycle_duration) + driving_speed * (total_time % cycle_duration) = 88 :=
by
  sorry

end car_drive_distance_l529_529462


namespace tan_diff_l529_529519

theorem tan_diff (x y : ℝ) (hx : Real.tan x = 3) (hy : Real.tan y = 2) : 
  Real.tan (x - y) = 1 / 7 := 
by 
  sorry

end tan_diff_l529_529519


namespace prod_ineq_geometric_mean_l529_529644

open Real

theorem prod_ineq_geometric_mean (n : ℕ) (x : Fin n → ℝ) (m : ℝ) (hx : ∀ i, 0 < x i) (hm : 0 < m) :
  (∏ i, (m + x i)) ≥ (m + (∏ i, x i) ^ (1 / n)) ^ n :=
sorry

end prod_ineq_geometric_mean_l529_529644


namespace num_prime_sums_gt_50_eq_0_l529_529703
open Nat

/-
  Define the sequence of sums of consecutive primes.
  We use Sigma_k to denote the k-th sum: Sigma_k = sum of the first k primes starting from 3.
-/
noncomputable def Σ : ℕ → ℕ
| 0       => 0
| 1       => 3
| (k + 2) => Σ (k + 1) + nth_prime (k + 1)

/-
  Prove that the number of sums among the first 12 sums that are both greater
  than 50 and prime is 0.
-/
theorem num_prime_sums_gt_50_eq_0 :
  (finset.range 12).filter (λ k => 50 < Σ k ∧ Prime (Σ k)).card = 0 := by
  sorry

end num_prime_sums_gt_50_eq_0_l529_529703


namespace dot_product_eq_2sqrt5_l529_529931

noncomputable def vector_a : ℝ × ℝ := sorry -- Define the vector_a but leave it unspecified
def vector_b : ℝ × ℝ := (1, 2)
def magnitude_a : ℝ := 4
def angle : ℝ := Real.pi / 3

theorem dot_product_eq_2sqrt5 : 
  |vector_a| = magnitude_a → 
  Real.angle vector_a vector_b = angle → 
  vector_a • vector_b = 2 * Real.sqrt 5 := 
sorry

end dot_product_eq_2sqrt5_l529_529931


namespace angle_C_max_area_triangle_l529_529609

-- Define the angles and sides of the triangle
variables (A B C a b c : ℝ)

-- Given conditions
def conditions := 2 * cos^2((A - B) / 2) - 2 * sin A * sin B = 1 - real.sqrt 2 / 2
def side_c_one := c = 1

-- Question 1: Prove that angle C is π/4
theorem angle_C :
  conditions →
  C = π / 4 := 
sorry

-- Question 2: Prove that the maximum area of triangle ABC is (sqrt(2) + 1) / 4
theorem max_area_triangle : 
  conditions → 
  side_c_one →
  ∃ (max_area : ℝ), max_area = (real.sqrt 2 + 1) / 4 := 
sorry

end angle_C_max_area_triangle_l529_529609
