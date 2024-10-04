import Mathlib

namespace valid_license_plates_l550_550707

def is_vowel (c : Char) : Bool :=
  c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

def num_valid_license_plates : Nat :=
  let num_vowels := 5
  let num_letters := 26
  let num_digits := 10
  num_vowels * num_letters * num_letters * num_digits * num_digits * num_digits

theorem valid_license_plates (h1 : ∀ plate : String, plate.length = 6)
                            (h2 : is_vowel plate[0]) :
  num_valid_license_plates = 87800000 :=
by
  sorry

end valid_license_plates_l550_550707


namespace units_digit_of_sum_factorials_l550_550477

theorem units_digit_of_sum_factorials :
  let S := ∑ i in Finset.range 100, i.factorial
  (S % 10) = 3 :=
by
  sorry

end units_digit_of_sum_factorials_l550_550477


namespace selling_price_correct_l550_550675

noncomputable def cost_price : ℝ := 241.67
noncomputable def profit_percentage : ℝ := 0.20

theorem selling_price_correct :
  let profit := profit_percentage * cost_price
  let selling_price := cost_price + profit
  floor (selling_price * 100 + 0.5) / 100 = 289.00 :=
by
  sorry

end selling_price_correct_l550_550675


namespace martin_bell_ringing_l550_550933

theorem martin_bell_ringing (B S : ℕ) (hB : B = 36) (hS : S = B / 3 + 4) : S + B = 52 :=
sorry

end martin_bell_ringing_l550_550933


namespace photograph_birds_l550_550694

theorem photograph_birds :
  let male_peacock := 1,
      females := 5,
      males_remaining_after_first := 4,
      females_remaining_after_first := 4,
      males_remaining_after_second := 3,
      females_remaining_after_second := 3,
      males_remaining_after_third := 2,
      females_remaining_after_third := 2,
      last_male := 1,
      last_female := 1 in
  (male_peacock * females * males_remaining_after_first * females_remaining_after_first
   * males_remaining_after_second * females_remaining_after_second
   * males_remaining_after_third * females_remaining_after_third
   * last_male * last_female) = 2880 :=
by sorry

end photograph_birds_l550_550694


namespace smallest_n_for_mean_equal_median_l550_550330

theorem smallest_n_for_mean_equal_median : 
  let set := {4, 8, 12, 15}
  let initial_sum := 4 + 8 + 12 + 15
  let new_sum (n: ℕ) := initial_sum + n
  let mean (n: ℕ) := (new_sum n) / 5
  let median (n: ℕ) := 
    if n < 8 then 8
    else if n < 12 then n 
    else 12
in 
  median 1 = mean 1 -> 1 = 1 :=
by 
  sorry

end smallest_n_for_mean_equal_median_l550_550330


namespace determine_m_l550_550559

-- Define the function f as given in the problem
def f (a x : ℝ) : ℝ := (a - x) * abs x

-- Define the inverse function condition
def has_inverse (f : ℝ → ℝ) := ∃ (g : ℝ → ℝ), ∀ x, f (g x) = x ∧ g (f x) = x

theorem determine_m (a m : ℝ) (f_inv : ℝ → ℝ) :
  (has_inverse (f a)) →
  (∀ x ∈ (set.Icc (-2 : ℝ) 2), f_inv (x^2 + m) < f a x) →
  m ∈ set.Ioi 12 :=
by
  sorry

end determine_m_l550_550559


namespace fourth_student_stickers_l550_550626

theorem fourth_student_stickers :
  ∀ (n : ℕ), 
    n = 1 → stickers n = 29 →
    n = 2 → stickers n = 35 →
    n = 3 → stickers n = 41 →
    -- Variable for student 4
    (n = 4 → stickers n = 47) →
    n = 5 → stickers n = 53 →
    n = 6 → stickers n = 59 →
    -- Pattern condition
    ∀ m : ℕ, m > 1 → stickers m = stickers (m - 1) + 6 :=
sorry

end fourth_student_stickers_l550_550626


namespace median_of_set_l550_550983

theorem median_of_set (y : ℝ) (h_mean : (90 + 88 + 85 + 87 + 89 + y) / 6 = 88) : 
    let l := [85, 87, 88, 89, 89, 90] 
    (∃ (m : ℝ), median l = m ∧ m = 88.5) :=
by
  sorry

end median_of_set_l550_550983


namespace color_circles_l550_550595

theorem color_circles (n : ℕ) (C : ℕ → set ℝ × ℝ) (touches : ∀ i j, i ≠ j → i < n → j < n → (dist (C i) (C j) = 2 * radius) → (C i ∩ C j ≠ ∅))
  (overlaps : ∀ i j, i ≠ j → i < n → j < n → C i ∩ C j = ∅) :
  ∃ f : ℕ → ℕ, (∀ i j, i ≠ j → i < n → j < n → (dist (C i) (C j) = 2 * radius) → f i ≠ f j) ∧ (∀ i, i < n → f i < 4) :=
by
  sorry

end color_circles_l550_550595


namespace sum_divided_among_xyz_l550_550348

noncomputable def total_amount (x_share y_share z_share : ℝ) : ℝ :=
  x_share + y_share + z_share

theorem sum_divided_among_xyz
    (x_share : ℝ) (y_share : ℝ) (z_share : ℝ)
    (y_gets_45_paisa : y_share = 0.45 * x_share)
    (z_gets_50_paisa : z_share = 0.50 * x_share)
    (y_share_is_18 : y_share = 18) :
    total_amount x_share y_share z_share = 78 := by
  sorry

end sum_divided_among_xyz_l550_550348


namespace magnitude_of_complex_l550_550155

variable (z : ℂ)
variable (h : Complex.I * z = 3 - 4 * Complex.I)

theorem magnitude_of_complex :
  Complex.abs z = 5 :=
by
  sorry

end magnitude_of_complex_l550_550155


namespace fifth_equation_in_pattern_l550_550938

theorem fifth_equation_in_pattern :
  (1 - 4 + 9 - 16 + 25) = (1 + 2 + 3 + 4 + 5) :=
sorry

end fifth_equation_in_pattern_l550_550938


namespace missing_angle_of_polygon_l550_550376

theorem missing_angle_of_polygon (sum_measured_angles : ℝ) (omitted_angle : ℝ) (h1 : sum_measured_angles = 3025) (h2 : omitted_angle = 35) :
  ∃ (n : ℕ), (n - 2) * 180 = sum_measured_angles + omitted_angle :=
begin
  use 17,
  have h : (17 - 2) * 180 = 3060 := by norm_num,
  have h3 : sum_measured_angles + omitted_angle = 3060,
  { rw [h1, h2], norm_num, },
  exact h3.trans h.symm,
end

end missing_angle_of_polygon_l550_550376


namespace sin_double_angle_l550_550447

variable {α : ℝ}

theorem sin_double_angle (h : sin α - cos α = -1 / 5) : sin (2 * α) = 24 / 25 := sorry

end sin_double_angle_l550_550447


namespace problem_l550_550149

theorem problem (p q : Prop) (hp : p) (hq : ¬ q) : ¬ q :=
by {
  exact hq,
}

end problem_l550_550149


namespace paper_folding_possible_layers_l550_550577

theorem paper_folding_possible_layers (n : ℕ) : 16 = 2 ^ n :=
by
  sorry

end paper_folding_possible_layers_l550_550577


namespace find_a_l550_550777

theorem find_a (a : ℤ) :
  (∀ x : ℤ, 6 * x + 3 > 3 * (x + a) → x > a - 1) ∧
  (∀ x : ℤ, x / 2 - 1 ≤ 7 - 3 * x / 2 → x ≤ 4) →
  (∑ x in finset.Icc (a - 1) 4, x = 9) →
  a = 2 := 
sorry

end find_a_l550_550777


namespace spherical_to_rectangular_l550_550042

theorem spherical_to_rectangular :
  ∀ (ρ θ φ : ℝ),
  ρ = 5 →
  θ = (2 * Real.pi) / 3 →
  φ = Real.pi / 4 →
  let x := ρ * Real.sin φ * Real.cos θ,
      y := ρ * Real.sin φ * Real.sin θ,
      z := ρ * Real.cos φ in
  (x, y, z) = (-5 * Real.sqrt 2 / 4, 5 * Real.sqrt 6 / 4, 5 * Real.sqrt 2 / 2) :=
by
  intros ρ θ φ hρ hθ hφ x y z
  rw [hρ, hθ, hφ]
  dsimp [x, y, z]
  sorry -- Proof omitted.


end spherical_to_rectangular_l550_550042


namespace probability_of_digit_five_l550_550943

theorem probability_of_digit_five : 
  let repeating_decimal := [8, 5, 7, 1, 4, 2] in
  let count_five := (repeating_decimal.count (λ x, x = 5)) in
  let block_length := repeating_decimal.length in
  (count_five / block_length) = (1 / 6) :=
by sorry

end probability_of_digit_five_l550_550943


namespace interest_rate_correct_l550_550484

-- Definitions based on the conditions
def interest_rate_doubles (r : ℝ) : ℝ := 70 / r

-- Given conditions
variables (r : ℝ) (initial_investment final_investment : ℝ) (years : ℝ)
hypothesis h1 : initial_investment = 5000
hypothesis h2 : final_investment = 20000
hypothesis h3 : years = 18
hypothesis h4 : final_investment = initial_investment * (2^2)

-- Statement to be proved
noncomputable def solve_interest_rate : ℝ :=
  70 / (years / 2)

theorem interest_rate_correct :
  solve_interest_rate = 7.78 :=
by
  -- omit the proof 
  sorry

end interest_rate_correct_l550_550484


namespace locus_of_M_l550_550431

noncomputable def tangent_length (M : ℝ × ℝ) (center : ℝ × ℝ) (radius : ℝ) : ℝ := 
  real.sqrt (M.1^2 + M.2^2 - radius^2)

noncomputable def distance (P1 P2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

theorem locus_of_M (Q : ℝ × ℝ) (C_radius : ℝ) (λ : ℝ) (λ_pos : 0 < λ) :
  ∃ locus : ℝ × ℝ → Prop,
    (λ = 1 ∧ locus = {M : ℝ × ℝ | M.1 = 5 / 4}) ∨
    (λ ≠ 1 ∧ 
      locus = {M : ℝ × ℝ | (λ^2 - 1) * M.1^2 + (λ^2 - 1) * M.2^2 + 4 * λ^2 * M.1 - 4 * λ^2 - 1 = 0}) :=
by {
  sorry
}

end locus_of_M_l550_550431


namespace find_q_l550_550223

noncomputable def p : ℝ := -(5 / 6)
noncomputable def g (x : ℝ) : ℝ := p * x^2 + (5 / 6) * x + 5

theorem find_q :
  (∀ x : ℝ, g x = p * x^2 + q * x + r) ∧ 
  (g (-2) = 0) ∧ 
  (g 3 = 0) ∧ 
  (g 1 = 5) 
  → q = 5 / 6 :=
sorry

end find_q_l550_550223


namespace cab_driver_income_l550_550677

theorem cab_driver_income (x : ℕ)
  (h1 : 50 + 60 + 65 + 70 + x = 5 * 58) :
  x = 45 :=
by
  sorry

end cab_driver_income_l550_550677


namespace phi_extreme_values_min_lambda_l550_550463

-- Definitions for the functions
def f (x : ℝ) : ℝ := Real.log x
def g (x : ℝ) : ℝ := x - (1 / x)
def phi (x : ℝ) : ℝ := (5 / 4) * (f x) - (1 / 2) * (g x)

-- Statement for the first proof
theorem phi_extreme_values :
  ∃ (xmin xmax : ℝ), xmin = 1/2 ∧ xmax = 2 ∧
  phi xmin = (3/4 - 5/4 * Real.log 2) ∧ phi xmax = (5/4 * Real.log 2 - 3/4) :=
sorry

-- Definition for the inequality function
def h (x : ℝ) (λ : ℝ) : ℝ := Real.log x - λ * (x - (1 / x))

-- Statement for the second proof
theorem min_lambda (λ : ℝ) :
  (∀ x ≥ 1, f x ≤ λ * g x) → λ ≥ 1 / 2 :=
sorry

end phi_extreme_values_min_lambda_l550_550463


namespace evaluate_expression_l550_550391

def diamond (a b : ℚ) : ℚ := a - (2 / b)

theorem evaluate_expression :
  ((diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4))) = -(11 / 30) :=
by
  sorry

end evaluate_expression_l550_550391


namespace meal_combinations_correct_l550_550370

-- Define the given conditions
def number_of_entrees : Nat := 4
def number_of_drinks : Nat := 4
def number_of_desserts : Nat := 2

-- Define the total number of meal combinations to prove
def total_meal_combinations : Nat := number_of_entrees * number_of_drinks * number_of_desserts

-- The theorem we want to prove
theorem meal_combinations_correct : total_meal_combinations = 32 := 
by 
  sorry

end meal_combinations_correct_l550_550370


namespace find_coordinates_of_b_l550_550446

theorem find_coordinates_of_b
  (x y : ℝ)
  (a : ℂ) (b : ℂ)
  (sqrt3 sqrt5 sqrt10 sqrt6 : ℝ)
  (h1 : sqrt3 = Real.sqrt 3)
  (h2 : sqrt5 = Real.sqrt 5)
  (h3 : sqrt10 = Real.sqrt 10)
  (h4 : sqrt6 = Real.sqrt 6)
  (h5 : a = ⟨sqrt3, sqrt5⟩)
  (h6 : ∃ x y : ℝ, b = ⟨x, y⟩ ∧ (sqrt3 * x + sqrt5 * y = 0) ∧ (Real.sqrt (x^2 + y^2) = 2))
  : b = ⟨- sqrt10 / 2, sqrt6 / 2⟩ ∨ b = ⟨sqrt10 / 2, - sqrt6 / 2⟩ := 
  sorry

end find_coordinates_of_b_l550_550446


namespace problem1_problem2_l550_550726

-- Proof problem 1
theorem problem1 :
  \left(\frac{1}{2}\right)^{-1} + (\pi - 2023)^0 - (-1)^{2023} = 4.
  sorry

-- Proof problem 2
theorem problem2 (a : ℝ) :
  (3 * a^2)^2 - (a^2 * 2 * a^2) + ((-2 * a^2)^3 / a^2) = -a^4.
  sorry

end problem1_problem2_l550_550726


namespace last_digit_of_a4_a_neg4_l550_550452

theorem last_digit_of_a4_a_neg4 (a : ℝ) (h : a^2 - 5 * a + 1 = 0) : (a^4 + a^(-4)) % 10 = 7 :=
by
  sorry

end last_digit_of_a4_a_neg4_l550_550452


namespace clock_rings_eight_times_in_a_day_l550_550719

theorem clock_rings_eight_times_in_a_day : 
  ∀ t : ℕ, t % 3 = 1 → 0 ≤ t ∧ t < 24 → ∃ n : ℕ, n = 8 := 
by 
  sorry

end clock_rings_eight_times_in_a_day_l550_550719


namespace solve_integer_pairs_l550_550601

theorem solve_integer_pairs :
  ∃ (x y : ℤ), (x = 19 ∧ y = -7) ∨ (x = -15 ∧ y = 5) ∨ (x = 7 ∧ y = 5) ∨ (x = -3 ∧ y = -7) ∧
               x^2 - 3*y^2 + 2*x*y - 2*x - 10*y + 20 = 0 :=
by
  use 19, -7,
  use -15, 5,
  use 7, 5,
  use -3, -7,
  sorry

end solve_integer_pairs_l550_550601


namespace grandfather_grandson_ages_l550_550246

theorem grandfather_grandson_ages :
  ∃ (g s : ℕ), Nat.isComposite g ∧ Nat.isComposite s ∧ (g + 1) * (s + 1) = 1610 ∧ g = 69 ∧ s = 22 :=
by
  sorry

end grandfather_grandson_ages_l550_550246


namespace proof_problem_l550_550652

/-- Statement of the proof problem: 
    Prove the equality of given mathematical expressions.
--/
theorem proof_problem :
(3:ℝ)^(4/3) = 1 / (3:ℝ)^(-4/3) ∧ 
(2:ℝ)^(1/2) = (4:ℝ)^(1/4) := 
by {
  split,
  { 
    -- Prove 3^(4/3) = 1 / 3^(-4/3)
    sorry 
  },
  { 
    -- Prove sqrt(2) = 4^(1/4)
    sorry 
  }
}

end proof_problem_l550_550652


namespace largest_number_eq_l550_550992

theorem largest_number_eq (x y z : ℚ) (h1 : x + y + z = 82) (h2 : z - y = 10) (h3 : y - x = 4) :
  z = 106 / 3 :=
sorry

end largest_number_eq_l550_550992


namespace marble_distribution_l550_550998

theorem marble_distribution (x : ℚ) (total : ℚ) (boy1 : ℚ) (boy2 : ℚ) (boy3 : ℚ) :
  (4 * x + 2) + (2 * x + 1) + (3 * x) = total → total = 62 →
  boy1 = 4 * x + 2 → boy2 = 2 * x + 1 → boy3 = 3 * x →
  boy1 = 254 / 9 ∧ boy2 = 127 / 9 ∧ boy3 = 177 / 9 :=
by
  sorry

end marble_distribution_l550_550998


namespace alice_bob_meet_in_9_turns_l550_550007

theorem alice_bob_meet_in_9_turns :
  ∃ k : ℕ, k = 9 ∧ (7 * k - 13 * k) % 18 = 0 :=
begin
  sorry
end

end alice_bob_meet_in_9_turns_l550_550007


namespace ellipse_and_circle_intersection_l550_550174

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (a^2 - b^2).sqrt = a * ((3:ℝ)^0.5 / 2)

theorem ellipse_and_circle_intersection (a b : ℝ) :
  ellipse_equation a b →
  ∃ P Q : ℝ × ℝ,
  (P.1 ^ 2 / a^2 + P.2 ^ 2 / b^2 = 1) ∧ 
  (Q.1 ^ 2 / a^2 + Q.2 ^ 2 / b^2 = 1) ∧ 
  (∀ (c d : ℝ × ℝ), 
      ((c.1 - d.1)^2 + (c.2 - d.2)^2).sqrt = 3 →
      (∀ (e f : ℝ × ℝ), 
          ((e.1 - f.1)^2 + (e.2 - f.2)^2).sqrt = 1 →
          (c = F₁ ∧ d = F₂) ∧ 
          (∀ (k : ℝ), k ≠ 0 →
              let x_1 := ... in -- Define x1 according to detailed steps
              let y_1 := k * (x_1 - 1) in
              let x_2 := ... in -- Define x2 according to detailed steps
              let y_2 := k * (x_2 - 1) in
              (line PQ through fixed point)) :=
begin
  sorry
end

end ellipse_and_circle_intersection_l550_550174


namespace complex_number_in_first_quadrant_l550_550178

-- Define the given complex number
def given_complex := 1 - (1 / complex.I)

-- State that the given complex number should be in the first quadrant.
theorem complex_number_in_first_quadrant : 
  given_complex.re > 0 ∧ given_complex.im > 0 := by
  sorry

end complex_number_in_first_quadrant_l550_550178


namespace distance_from_M0_to_plane_M1_M2_M3_l550_550062

def Point := ℝ × ℝ × ℝ

def plane_through_points (M1 M2 M3 : Point) : ℝ × ℝ × ℝ × ℝ :=
  let (x1, y1, z1) := M1
  let (x2, y2, z2) := M2
  let (x3, y3, z3) := M3
  let a := (y1 - y2) * (z1 - z3) - (y1 - y3) * (z1 - z2)
  let b := (z1 - z2) * (x1 - x3) - (z1 - z3) * (x1 - x2)
  let c := (x1 - x2) * (y1 - y3) - (x1 - x3) * (y1 - y2)
  let d := -(a * x1 + b * y1 + c * z1)
  (a, b, c, d)

def distance_point_to_plane (p : Point) (a b c d : ℝ) : ℝ :=
  let (x, y, z) := p
  (abs (a * x + b * y + c * z + d)) / (sqrt (a^2 + b^2 + c^2))

def M0 : Point := (-3, 1, 8)
def M1 : Point := (2, 1, 4)
def M2 : Point := (3, 5, -2)
def M3 : Point := (-7, -3, 2)

theorem distance_from_M0_to_plane_M1_M2_M3 :
  let (a, b, c, d) := plane_through_points M1 M2 M3
  distance_point_to_plane M0 a b c d = 4 :=
by
  sorry

end distance_from_M0_to_plane_M1_M2_M3_l550_550062


namespace magnitude_of_complex_l550_550154

variable (z : ℂ)
variable (h : Complex.I * z = 3 - 4 * Complex.I)

theorem magnitude_of_complex :
  Complex.abs z = 5 :=
by
  sorry

end magnitude_of_complex_l550_550154


namespace problem_2013_dongcheng_mock_test_l550_550318

def units_digit (n : ℕ) : ℕ := n % 10

def sequence_a : ℕ → ℕ
| 1 := 2
| 2 := 7
| (n+2) := units_digit (sequence_a n * sequence_a (n+1))

theorem problem_2013_dongcheng_mock_test :
  sequence_a 2013 = 4 :=
sorry

end problem_2013_dongcheng_mock_test_l550_550318


namespace blueberries_needed_l550_550375

noncomputable def flour_cost := 2.0
noncomputable def sugar_cost := 1.0
noncomputable def eggs_butter_cost := 1.5
noncomputable def crust_cost := flour_cost + sugar_cost + eggs_butter_cost

noncomputable def blueberry_container_oz := 8.0
noncomputable def blueberry_container_cost := 2.25
noncomputable def cherry_bag_pounds := 4.0
noncomputable def cherry_bag_cost := 14.0
noncomputable def cheapest_pie_total_cost := 18.0

/-- Calculate blueberry cost for the blueberry pie -/
noncomputable def blueberry_cost := cheapest_pie_total_cost - crust_cost

/-- Containers of blueberries required for the total cost -/
noncomputable def number_of_containers := blueberry_cost / blueberry_container_cost

/-- Pounds of blueberries needed -/
noncomputable def pounds_of_blueberries := (number_of_containers * blueberry_container_oz) / 16.0

theorem blueberries_needed (flour_cost = 2.0) (sugar_cost = 1.0) (eggs_butter_cost = 1.5) 
  (blueberry_container_cost = 2.25) (blueberry_container_oz = 8.0) 
  (cherry_bag_pounds = 4.0) (cherry_bag_cost = 14.0) 
  (cheapest_pie_total_cost = 18.0) : pounds_of_blueberries = 3 :=
by 
  sorry

end blueberries_needed_l550_550375


namespace orthocenter_on_radical_axis_l550_550213

open EuclideanGeometry

-- Define the basic entities: triangle ABC and points D and E
variables {A B C D E H : Point}

-- Prerequisite conditions mentioned in part (a)
axiom triangle_ABC : Triangle A B C
axiom D_on_AB : Online D (LineThrough A B)
axiom E_on_AC : Online E (LineThrough A C)
axiom orthocenter_H : Orthocenter H triangle_ABC

-- Define circles Γ1 and Γ2 with specific diameters
axiom Γ1 : Circle (Segment B E)
axiom Γ2 : Circle (Segment D C)

-- The theorem we need to prove
theorem orthocenter_on_radical_axis :
  LiesOnRadicalAxis H Γ1 Γ2 :=
sorry

end orthocenter_on_radical_axis_l550_550213


namespace equal_wins_draws_losses_no_five_teams_with_equal_wins_draws_losses_l550_550498

-- Part (a)
theorem equal_wins_draws_losses (n : ℕ) (h : n > 4) :
  ∃ (team1 team2 team3 team4 : ℕ), 
  (points team1 = points team2 ∧ points team2 = points team3 ∧ points team3 = points team4)
  ∧ (wins team1 = wins team2 ∧ wins team2 = wins team3 ∧ wins team3 = wins team4)
  ∧ (draws team1 = draws team2 ∧ draws team2 = draws team3 ∧ draws team3 = draws team4)
  ∧ (losses team1 = losses team2 ∧ losses team2 = losses team3 ∧ losses team3 = losses team4) := 
sorry

-- Part (b)
theorem no_five_teams_with_equal_wins_draws_losses (n : ℕ) (h : n = 10) :
  ¬ ∃ (team1 team2 team3 team4 team5 : ℕ), 
  (points team1 = points team2 ∧ points team2 = points team3 ∧ points team3 = points team4 ∧ points team4 = points team5)
  ∧ (wins team1 = wins team2 ∧ wins team2 = wins team3 ∧ wins team3 = wins team4 ∧ wins team4 = wins team5)
  ∧ (draws team1 = draws team2 ∧ draws team2 = draws team3 ∧ draws team3 = draws team4 ∧ draws team4 = draws team5)
  ∧ (losses team1 = losses team2 ∧ losses team2 = losses team3 ∧ losses team3 = losses team4 ∧ losses team4 = losses team5) :=
sorry

end equal_wins_draws_losses_no_five_teams_with_equal_wins_draws_losses_l550_550498


namespace not_always_possible_to_split_l550_550282

theorem not_always_possible_to_split :
    ∀ (A B C D E F : Point),
    is_midpoint D A B →
    is_midpoint E B C →
    is_midpoint F C A →
    ¬ (∃ (T1 T2 : set Point),
        T1.card = 3 ∧ 
        T2.card = 3 ∧ 
        (∀ (p ∈ T1), p ∈ {A, B, C, D, E, F}) ∧ 
        (∀ (p ∈ T2), p ∈ {A, B, C, D, E, F}) ∧ 
        disjoint T1 T2 ∧ 
        forms_triangle T1 ∧ 
        forms_triangle T2) := 
by
  sorry

-- Definitions for is_midpoint, Point, set, forms_triangle, etc., need to be provided or imported. 

end not_always_possible_to_split_l550_550282


namespace more_students_suggested_bacon_than_mashed_potatoes_l550_550018

-- Define the number of students suggesting each type of food
def students_suggesting_mashed_potatoes := 479
def students_suggesting_bacon := 489

-- State the theorem that needs to be proven
theorem more_students_suggested_bacon_than_mashed_potatoes :
  students_suggesting_bacon - students_suggesting_mashed_potatoes = 10 := 
  by
  sorry

end more_students_suggested_bacon_than_mashed_potatoes_l550_550018


namespace money_left_after_shopping_l550_550934

def initial_amount : ℝ := 250
def banana_cost : ℝ := 6 * 5
def pear_cost : ℝ := 8 * 1.50
def asparagus_cost : ℝ := 3 * 7.50
def chicken_cost : ℝ := 2 * 20
def strawberry_cost : ℝ := 5 * 4
def olive_oil_cost : ℝ := 1 * 15
def almond_cost : ℝ := 1 * 25
def shrimp_cost : ℝ := 0.5 * 20
def cheese_cost : ℝ := 1.2 * 10.50
def discount_threshold : ℝ := 200
def discount_amount : ℝ := 10

def total_cost_before_discount : ℝ := 
  banana_cost + pear_cost + asparagus_cost + chicken_cost + strawberry_cost + 
  olive_oil_cost + almond_cost + shrimp_cost + cheese_cost

def qualified_for_discount : Bool := total_cost_before_discount ≥ discount_threshold
def discount_applied : ℝ := if qualified_for_discount then discount_amount else 0
def total_cost_after_discount : ℝ := total_cost_before_discount - discount_applied

def final_amount_left : ℝ := initial_amount - total_cost_before_discount

-- The statement to be proven
theorem money_left_after_shopping : final_amount_left = 62.90 :=
by
  sorry

end money_left_after_shopping_l550_550934


namespace monotonic_intervals_range_a_l550_550459

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ln (x + (1 / a)) - a * x

theorem monotonic_intervals (a : ℝ) (h : a ≠ 0) :
  (a < 0 → ∀ x, f a x = ln (x + (1 / a)) - a * x ∧ x > -1 / a → deriv (f a) x > 0) ∧
  (a > 0 → ∀ x, f a x = ln (x + (1 / a)) - a * x ∧ x > -1 / a →
    ((∀ x, x < 0 → deriv (f a) x > 0) ∧ (∀ x, x > 0 → deriv (f a) x < 0))) :=
sorry

theorem range_a (a : ℝ) (h : a ≠ 0) :
  (∀ x, (a * x - f a x) > 0) → (a > exp(1) / 2) :=
sorry

end monotonic_intervals_range_a_l550_550459


namespace max_integers_no_two_differ_l550_550295

theorem max_integers_no_two_differ (n : ℕ) (a : ℕ → ℕ) (h1 : ∀ i j, 1 ≤ i → i < j → j ≤ n → a i ≠ a j)
  (h2 : ∀ i j, 1 ≤ i → i < j → j ≤ n → abs (a i - a j) ≠ 1)
  (h3 : ∀ i j, 1 ≤ i → i < j → j ≤ n → abs (a i - a j) ≠ 2)
  (h4 : ∀ i j, 1 ≤ i → i < j → j ≤ n → abs (a i - a j) ≠ 6)
  (h5 : ∀ i, 1 ≤ i → i ≤ n → a i ≤ 2016) :
  n ≤ 576 :=
by
  sorry

end max_integers_no_two_differ_l550_550295


namespace winning_candidate_percentage_l550_550284

theorem winning_candidate_percentage :
  let votes1 := 1136
  let votes2 := 7636
  let votes3 := 11628
  let total_votes := votes1 + votes2 + votes3
  let winning_candidate_votes := votes3
  (winning_candidate_votes / total_votes.toFloat) * 100 = 57.0196 :=
by
  let votes1 := 1136
  let votes2 := 7636
  let votes3 := 11628
  let total_votes := votes1 + votes2 + votes3
  let winning_candidate_votes := votes3
  have h1 : total_votes = 20400 := by sorry
  have h2 : (winning_candidate_votes.toFloat / total_votes.toFloat) * 100 = 57.0196 := by sorry
  exact h2

end winning_candidate_percentage_l550_550284


namespace kiran_currency_notes_l550_550902

theorem kiran_currency_notes :
  ∀ (n50_amount n100_amount total50 total100 : ℝ),
    n50_amount = 3500 →
    total50 = 5000 →
    total100 = 5000 - 3500 →
    n100_amount = total100 →
    (n50_amount / 50 + total100 / 100) = 85 :=
by
  intros n50_amount n100_amount total50 total100 n50_amount_eq total50_eq total100_eq n100_amount_eq
  sorry

end kiran_currency_notes_l550_550902


namespace round_trip_time_l550_550991

theorem round_trip_time (boat_speed : ℝ) (stream_speed : ℝ) (distance : ℝ) : 
  boat_speed = 8 → stream_speed = 2 → distance = 210 → 
  ((distance / (boat_speed - stream_speed)) + (distance / (boat_speed + stream_speed))) = 56 :=
by
  intros hb hs hd
  sorry

end round_trip_time_l550_550991


namespace max_inversions_n_eq_7_max_inversions_n_eq_2019_l550_550879

def is_inversion (seq : List ℕ) (i j : ℕ) : Prop :=
  i < j ∧ seq[i] > seq[j]

def count_inversions (seq : List ℕ) : ℕ :=
  (List.finRange seq.length).sum (λ i, (List.finRange seq.length).count (λ j, is_inversion seq i j))

def max_inversions (seq_sum : ℕ) : ℕ :=
  if seq_sum = 7 then 6 else
  if seq_sum = 2019 then 509545 else 0

theorem max_inversions_n_eq_7 (seq : List ℕ) (h1 : seq.sum = 7) :
  count_inversions seq ≤ max_inversions 7 :=
by
  sorry

theorem max_inversions_n_eq_2019 (seq : List ℕ) (h1 : seq.sum = 2019) :
  count_inversions seq ≤ max_inversions 2019 :=
by
  sorry

end max_inversions_n_eq_7_max_inversions_n_eq_2019_l550_550879


namespace min_dot_product_OA_OB_OC_l550_550893

open Real

/- 
  Define the setup for the problem with the given conditions.
  1. Define the points A, B, C, and the median point M.
  2. Define O as a point on the median AM.
  3. Set the length of AM as 2.
 -/
variable {A B C M O : ℝ} 

variables (AM_eq_2 : AM = 2)
          (median_AM : is_median A M B C)

-- The theorem statement proving the minimum value of the given dot product
theorem min_dot_product_OA_OB_OC :
  O ∈ segment A M →
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧
  ∀ O : ℝ, (O ∈ segment A M) → 
    ((λ x : ℝ, 2*x^2 - 4*x) x = \overrightarrow{OA} \cdot (\overrightarrow{OB} + \overrightarrow{OC})) 
    × (2*x^2 - 4*x = -2) :=
begin
  sorry
end

#exit

end min_dot_product_OA_OB_OC_l550_550893


namespace sin_alpha_neg_four_fifths_l550_550786

open Real

theorem sin_alpha_neg_four_fifths (α : ℝ) (h₁ : α ∈ Ioo π (3 * π / 2)) (h₂ : tan α = 4 / 3) : 
  sin α = -4 / 5 :=
by
  sorry

end sin_alpha_neg_four_fifths_l550_550786


namespace problem_proof_l550_550597

variables {A B C D I P : Type*}
variables [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty I] [Nonempty P]

open_locale classical

noncomputable def cyclic_quadrilateral : Prop :=
∃ (P A I B : Type*),
  (∀ (AB BC PA PB BI AI PI : ℝ), AB * PI = PA * BI + PB * AI)

noncomputable def relationship (BC AD BI CI AI DI : ℝ) : Prop :=
BC / AD = (BI * CI) / (AI * DI)

theorem problem_proof
  (P A I B C D I : Type*)
  (AB BC AI BI CI DI : ℝ)
  (h1: ∃ (P A I B : Type*), (∀ (AB BC PA PB BI AI PI : ℝ), AB * PI = PA * BI + PB * AI))
  (h2: ↑BC / ↑AD = (↑BI * ↑CI) / (↑AI * ↑DI))
  : AB * BC = BI ^ 2 + (AI * BI * CI) / DI :=
sorry

end problem_proof_l550_550597


namespace hexagon_to_square_l550_550385

-- Define a regular hexagon with side length s
structure RegularHexagon (s : ℝ) :=
  (area : ℝ)
  (is_regular : ∃ (s : ℝ), area = (3 * real.sqrt 3 / 2) * s^2)

-- Define the resulting 5 parts from the cuts described
inductive HexagonPart
| part1 : HexagonPart -- Triangle AGO
| part2 : HexagonPart -- Quadrilateral GBCF
| part3 : HexagonPart -- Triangle GOE
| part4 : HexagonPart -- Triangle ODF
| part5 : HexagonPart -- Triangle EFG

-- Define the existence of the 5 parts dividing the hexagon
def divided_hexagon (h : RegularHexagon s) : Prop :=
  ∃ (parts : fin 5 → HexagonPart),
    -- The 5 parts must cover the entire area of the hexagon
    (finset.univ.sum (λ i, match (parts i) with
                           | HexagonPart.part1 => (1 / 2) * s^2
                           -- Simplified areas, area calculation step skipped
                           -- Assume proper functions/constants representing each part's area
                           | HexagonPart.part2 => sorry -- Quadrilateral GBCF area
                           | HexagonPart.part3 => sorry -- Triangle GOE area
                           | HexagonPart.part4 => sorry -- Triangle ODF area
                           | HexagonPart.part5 => sorry -- Triangle EFG area
                           end) = (3 * real.sqrt 3 / 2) * s^2)

-- Define the square with the sides equal to sqrt of the area of one hexagon part
structure Square :=
  (side : ℝ)
  (area : ℝ := side^2)

-- The objective theorem
theorem hexagon_to_square (h : RegularHexagon s) :
  divided_hexagon h → 
  ∃ (sq : Square), sq.side = s * real.sqrt (3 * real.sqrt 3 / 2) :=
by
  sorry

end hexagon_to_square_l550_550385


namespace matrix_power_eval_l550_550204

open Matrix

variables (B : Matrix (Fin 2) (Fin 2) ℝ)

def v := ![3, -1]
def w := ![6, -2]

theorem matrix_power_eval :
  B.mul_vec v = w → (B^5).mul_vec v = ![96, -32] :=
by
  sorry

end matrix_power_eval_l550_550204


namespace areas_not_possible_l550_550684

theorem areas_not_possible (a b c d : ℕ)
  (h1 : a * c = b * d)
  (h2 : a = 2001)
  (h3 : b = 2002)
  (h4 : c = 2003) : ∃ d : ℕ, d ≠ 2001.0015 := 
sorry

end areas_not_possible_l550_550684


namespace simplify_and_evaluate_expression_l550_550599

-- Noncomputable theory is only necessary if we need noncomputable definitions. In this case, it might not be required,
-- however, we are adding it to ensure that working with real numbers does not break the proof.

noncomputable theory

-- Define the variables and the expression to be simplified and evaluated
variable (a : ℝ)

-- The main theorem we need to prove
theorem simplify_and_evaluate_expression : a = (-2)^2 → (5 * a ^ 2 - (a ^ 2 - (2 * a - 5 * a ^ 2) - 2 * (a ^ 2 - 3 * a))) = 32 :=
by
  intro h
  rw h
  sorry

end simplify_and_evaluate_expression_l550_550599


namespace line_through_intersection_of_circles_l550_550139

theorem line_through_intersection_of_circles 
  (x y : ℝ)
  (C1 : x^2 + y^2 = 10)
  (C2 : (x-1)^2 + (y-3)^2 = 20) :
  x + 3 * y = 0 :=
sorry

end line_through_intersection_of_circles_l550_550139


namespace stratified_sampling_C_units_l550_550687

theorem stratified_sampling_C_units:
  let total_units := 400 + 800 + 600
  let sampling_ratio := 90 / total_units
  let units_C := 600
  (units_C * sampling_ratio) = 30 :=
by 
  let total_units := 400 + 800 + 600
  let sampling_ratio := 90 / total_units
  let units_C := 600
  have h1 : (units_C * sampling_ratio) = 600 * (90 / (400 + 800 + 600)) := rfl
  have h2 : (90 / (400 + 800 + 600)) = 1 / 20 := by norm_num
  have h3 : 600 * (1 / 20) = 30 := by norm_num
  exact eq.trans h1 (eq.trans h2 h3)

end stratified_sampling_C_units_l550_550687


namespace largest_area_of_triangle_DEF_l550_550706

noncomputable def maxAreaTriangleDEF : Real :=
  let DE := 16.0
  let EF_to_FD := 25.0 / 24.0
  let max_area := 446.25
  max_area

theorem largest_area_of_triangle_DEF :
  ∀ (DE : Real) (EF FD : Real),
    DE = 16 ∧ EF / FD = 25 / 24 → 
    (∃ (area : Real), area ≤ maxAreaTriangleDEF) :=
by 
  sorry

end largest_area_of_triangle_DEF_l550_550706


namespace ellipse_properties_l550_550100

-- Defining the parameters and the ellipse
variables {a b : ℝ} (ha : 0 < a) (hb : 0 < b)

-- Eccentricity e
def eccentricity (a b : ℝ) : ℝ := sqrt (1 - (b/a)^2)

-- Condition for eccentricity
def e_half (a : ℝ) : Prop := eccentricity a (sqrt 3 / 2 * a) = 1/2

-- Distance from origin to line AB (derived line equation y = sqrt(3)/2 * x + sqrt(3)/2 * a)
def distance_from_origin (a b : ℝ) : ℝ := abs (sqrt(3)/2 * a) / sqrt (1 + (sqrt(3)/2)^2)

-- Main statement
theorem ellipse_properties (a : ℝ) (h_e : e_half a) (h_dist : distance_from_origin a (sqrt 3 / 2 * a) = 2 * sqrt 21 / 7) : 
  (a = 2 ∧ b = sqrt 3 ∧ ∀ (m : ℝ), ∃ c, 
    let P : ℝ × ℝ := (m * c + sqrt 3, c),
    let Q : ℝ × ℝ := (m * c + sqrt 3, -c) in
    ((P.1, -P.2) -- point P' which is symmetric to P about x-axis
     Q -- point Q
     intersect at fixed point (4 * sqrt 3 / 3, 0))) :=
sorry

end ellipse_properties_l550_550100


namespace total_savings_l550_550981

def income := 19000
def expenditure_ratio := 4 / 10
def tax_rate := 15 / 100
def long_term_investment_rate := 10 / 100
def short_term_investment_rate := 20 / 100

theorem total_savings : 
  let E := income * expenditure_ratio in
  let taxes := tax_rate * income in
  let remaining_income := income - taxes in
  let long_term_investment := long_term_investment_rate * remaining_income in
  let amount_after_lt := remaining_income - long_term_investment in
  let short_term_investment := short_term_investment_rate * amount_after_lt in
  let total_savings := amount_after_lt - short_term_investment in
  total_savings = 11628 := 
by
  sorry

end total_savings_l550_550981


namespace a_n_formula_T_n_sum_l550_550833

open Nat

noncomputable def S (n : ℕ ) : ℚ := 
  if n = 0 then 0 else (1 : ℚ) / 8 * n^2 + (9 : ℚ) / 8 * n

noncomputable def a (n : ℕ) : ℚ := 
  if n = 0 then 0 else S n - S (n - 1)

noncomputable def b (n : ℕ) : ℚ := 
  if n = 0 then 0
  else 1 / (16 * (a n - 1) * (a (n + 1) - 1))

noncomputable def T (n : ℕ) : ℚ := 
  ∑ i in range n, b i

theorem a_n_formula (n : ℕ) (hn : n > 0) : a n = (1 : ℚ) / 4 * n + 1 := 
  sorry

theorem T_n_sum (n : ℕ) (hn : n > 0) : T n = n / (n + 1 : ℚ) := 
  sorry

end a_n_formula_T_n_sum_l550_550833


namespace intersection_integer_point_l550_550175

-- Define the condition under which the intersection is an integer point
def is_integer_point (x y : ℤ) : Prop :=
  true

-- Define the main problem
theorem intersection_integer_point (k : ℤ) :
  (∃ (x y : ℤ), y = x - 2 ∧ y = k * x + k) → k = 0 ∨ k = 2 ∨ k = 4 ∨ k = -2 :=
begin
  sorry
end

end intersection_integer_point_l550_550175


namespace geom_series_correct_sum_l550_550747

-- Define the geometric series sum
noncomputable def geom_series_sum (a r : ℚ) (n : ℕ) :=
  a * (1 - r ^ n) / (1 - r)

-- Given conditions
def a := (1 : ℚ) / 4
def r := (1 : ℚ) / 4
def n := 8

-- Correct answer sum
def correct_sum := (65535 : ℚ) / 196608

-- Proof problem statement
theorem geom_series_correct_sum : geom_series_sum a r n = correct_sum := 
  sorry

end geom_series_correct_sum_l550_550747


namespace sector_area_l550_550485

theorem sector_area (l : ℝ) (α : ℝ) (r : ℝ) : 
  α = 1 ∧ l = 6 ∧ l = r * α → (1 / 2) * l * r = 18 :=
by
  intros h
  cases h with hα hr
  cases hr with hl hrα
  rw [hα, hl, hrα]
  sorry

end sector_area_l550_550485


namespace volume_of_frustum_l550_550353

-- Definition of the volume of a square pyramid
def volume_pyramid (a h : ℝ) : ℝ := (a^2 * h) / 3

-- Conditions for the problem
def base_edge_original : ℝ := 10
def height_original : ℝ := 12
def base_edge_smaller : ℝ := 5
def height_smaller : ℝ := 6

-- Volumes as per the conditions
def vol_original : ℝ := volume_pyramid base_edge_original height_original
def vol_smaller : ℝ := volume_pyramid base_edge_smaller height_smaller

-- The volume of the frustum to be proven
def vol_frustum : ℝ := vol_original - vol_smaller

-- Proof goal
theorem volume_of_frustum : vol_frustum = 350 :=
by
  sorry

end volume_of_frustum_l550_550353


namespace common_point_l550_550611

/-- The conditions on the sampling methods -/
def sampling_conditions (A B C D : Prop) := 
  (A = "Each individual is drawn one by one from the population") ∧ 
  (B = "The population is divided into several parts, and sampling is done in each part according to a predetermined rule") ∧ 
  (C = "During the sampling process, each individual has an equal chance of being selected") ∧ 
  (D = "During the sampling process, the population is divided into several layers, and sampling is done proportionally by layer")

/-- The proof statement -/
theorem common_point (A B C D : Prop) (h : sampling_conditions A B C D) :
  C = "During the sampling process, each individual has an equal chance of being selected" :=
by
  sorry

end common_point_l550_550611


namespace total_ticket_sales_is_48_l550_550285

noncomputable def ticket_sales (total_revenue : ℕ) (price_per_ticket : ℕ) (discount_1 : ℕ) (discount_2 : ℕ) : ℕ :=
  let number_first_batch := 10
  let number_second_batch := 20
  let revenue_first_batch := number_first_batch * (price_per_ticket - (price_per_ticket * discount_1 / 100))
  let revenue_second_batch := number_second_batch * (price_per_ticket - (price_per_ticket * discount_2 / 100))
  let revenue_full_price := total_revenue - (revenue_first_batch + revenue_second_batch)
  let number_full_price_tickets := revenue_full_price / price_per_ticket
  number_first_batch + number_second_batch + number_full_price_tickets

theorem total_ticket_sales_is_48 : ticket_sales 820 20 40 15 = 48 :=
by
  sorry

end total_ticket_sales_is_48_l550_550285


namespace calculator_display_after_50_presses_l550_550245

theorem calculator_display_after_50_presses :
  let initial_display := 3
  let operation (x : ℚ) := 1 / (1 - x)
  (Nat.iterate operation 50 initial_display) = 2 / 3 :=
by
  sorry

end calculator_display_after_50_presses_l550_550245


namespace max_l_good_edges_l550_550917

theorem max_l_good_edges {A B : Type} {l n : ℕ} (hA : set A) (hB : set B)
  (h_disjoint: A ∩ B = ∅) (h_size_A: |A| = n) (h_size_B: |B| = n)
  (h_no_collinear: ∀ ⦃p1 p2 p3 : A ∪ B⦄, ¬ collinear A B p1 p2 p3)
  (h_segments : ∀ a ∈ A, ∃ (b ∈ B), (a, b) ∈ S ∧ |S| ≤ l ∧ ∀ b ∈ B, ∃ (a ∈ A), (a, b) ∈ S ∧ |S| ≤ l) :
  ∃ |S|, (remove_edge S) = λ e, ∃ p, p ∈ (A ∪ B) ∧ |S - {e}| < l = 
    if (n - l ÷ 2) > l then (n-l) * 2 * l
    else (n - ⌊(n-l) / 2⌋) * (⌊(n-l) / 2⌋ + l).
:= sorry

end max_l_good_edges_l550_550917


namespace intersection_complement_of_M_l550_550810

def real_set := Set ℝ

def M : real_set := { x | 2 / x < 1 }
def N : real_set := { y | ∃ x, y = Real.sqrt (x - 1) + 1 }

def complement_of_M (R : real_set) (M : real_set) : real_set := { x | x ∈ R ∧ x ∉ M }

def intersection (A B : real_set) : real_set := { x | x ∈ A ∧ x ∈ B }

theorem intersection_complement_of_M :
  intersection N (complement_of_M { x : ℝ | True } M) = { x | 1 ≤ x ∧ x ≤ 2 } :=
  by sorry

end intersection_complement_of_M_l550_550810


namespace probability_green_ball_l550_550739

theorem probability_green_ball :
  let p := 1 / 3 * (5 / 15) + 1 / 3 * (6 / 9) + 1 / 3 * (6 / 9) in
  p = 5 / 9 :=
by
  let prob_A := 5 / 15
  let prob_B := 6 / 9
  let prob_C := 6 / 9
  let prob_total := 1 / 3 * prob_A + 1 / 3 * prob_B + 1 / 3 * prob_C
  have h : prob_total = 5 / 9 := 
  by sorry
  exact h

end probability_green_ball_l550_550739


namespace geometric_series_sum_l550_550031

theorem geometric_series_sum :
  ∀ (a r : ℤ) (n : ℕ),
  a = 3 → r = -2 → n = 10 →
  (a * ((r ^ n - 1) / (r - 1))) = -1024 :=
by
  intros a r n ha hr hn
  rw [ha, hr, hn]
  sorry

end geometric_series_sum_l550_550031


namespace largest_possible_a_l550_550921

theorem largest_possible_a (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : d < 150) (hp : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  a ≤ 8924 :=
sorry

end largest_possible_a_l550_550921


namespace max_sum_of_abcd_l550_550448

noncomputable def abcd_product (a b c d : ℕ) : ℕ := a * b * c * d

theorem max_sum_of_abcd (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
    (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) (h7 : abcd_product a b c d = 1995) : 
    a + b + c + d ≤ 142 :=
sorry

end max_sum_of_abcd_l550_550448


namespace even_integers_between_sqrt_10_and_sqrt_100_l550_550143

theorem even_integers_between_sqrt_10_and_sqrt_100 : 
  ∃ (n : ℕ), n = 4 ∧ (∀ (a : ℕ), (∃ k, (2 * k = a ∧ a > Real.sqrt 10 ∧ a < Real.sqrt 100)) ↔ 
  (a = 4 ∨ a = 6 ∨ a = 8 ∨ a = 10)) := 
by 
  sorry

end even_integers_between_sqrt_10_and_sqrt_100_l550_550143


namespace polynomial_with_complex_root_l550_550068

theorem polynomial_with_complex_root :
  ∃ P : Polynomial ℝ, P.Monic ∧ degree P = 2 ∧
  P.coeff 0 = 17 ∧ P.coeff 1 = 6 ∧ P.coeff 2 = 1 ∧
  (P.eval (-3 - Complex.i * Real.sqrt 8) = 0) :=
sorry

end polynomial_with_complex_root_l550_550068


namespace largest_expression_value_l550_550648

-- Definitions of the expressions
def expr_A : ℕ := 3 + 0 + 1 + 8
def expr_B : ℕ := 3 * 0 + 1 + 8
def expr_C : ℕ := 3 + 0 * 1 + 8
def expr_D : ℕ := 3 + 0 + 1^2 + 8
def expr_E : ℕ := 3 * 0 * 1^2 * 8

-- Statement of the theorem
theorem largest_expression_value :
  max expr_A (max expr_B (max expr_C (max expr_D expr_E))) = 12 :=
by
  sorry

end largest_expression_value_l550_550648


namespace analytic_expression_of_f_function_shift_l550_550217

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + π / 4)

theorem analytic_expression_of_f (x : ℝ) (ϕ : ℝ) (h : -π < ϕ ∧ ϕ < 0) (h_sym : 2 * (π / 8) + ϕ = k * π + π / 2) : 
  f x = Real.sin (2 * x + π / 4) :=
sorry

theorem function_shift (x : ℝ) : 
  (λ x, 2 * Real.sin (2 * x - π / 6)) = (λ x, 2 * f (x - 5 * π / 24)) :=
sorry

end analytic_expression_of_f_function_shift_l550_550217


namespace find_m_l550_550117

variable (m x1 x2 : ℝ)

def quadratic_eqn (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - m * x + 2 * m - 1 = 0

def roots_condition (m x1 x2 : ℝ) : Prop :=
  x1^2 + x2^2 = 23 ∧
  x1 + x2 = m ∧
  x1 * x2 = 2 * m - 1

theorem find_m (m x1 x2 : ℝ) : 
  quadratic_eqn m → 
  roots_condition m x1 x2 → 
  m = -3 :=
by
  intro hQ hR
  sorry

end find_m_l550_550117


namespace find_y_l550_550244

theorem find_y (t y : ℝ) (h1 : -3 = 2 - t) (h2 : y = 4 * t + 7) : y = 27 :=
sorry

end find_y_l550_550244


namespace circumcenter_on_segment_AC_l550_550427

theorem circumcenter_on_segment_AC
  (A B C D M N : Point)
  (h1 : CyclicQuadrilateral A B C D)
  (h2 : Distance A B = Distance A D)
  (h3 : OnSegment C D M)
  (h4 : OnSegment B C N)
  (h5 : Distance D M + Distance B N = Distance M N) :
  OnSegment A C (Circumcenter A M N) := 
sorry

end circumcenter_on_segment_AC_l550_550427


namespace initial_contribution_l550_550523

theorem initial_contribution (j k l : ℝ)
  (h1 : j + k + l = 1200)
  (h2 : j - 200 + 3 * (k + l) = 1800) :
  j = 800 :=
sorry

end initial_contribution_l550_550523


namespace find_x_l550_550055

theorem find_x (x : ℝ) (h : 4^real.log x / real.log 7 = 16) : x = 49 :=
by {
  sorry
}

end find_x_l550_550055


namespace value_range_for_positive_roots_l550_550274

theorem value_range_for_positive_roots (a : ℝ) :
  (∀ x : ℝ, x > 0 → a * |x| + |x + a| = 0) ↔ (-1 < a ∧ a < 0) :=
by
  sorry

end value_range_for_positive_roots_l550_550274


namespace arithmetic_sequence_sum_minimum_l550_550438

noncomputable def S_n (a1 d : ℝ) (n : ℕ) : ℝ := 
  (n * (2 * a1 + (n - 1) * d)) / 2

theorem arithmetic_sequence_sum_minimum (a1 : ℝ) (d : ℝ) :
  a1 = -20 ∧ (∀ n : ℕ, (S_n a1 d n) > (S_n a1 d 6)) → 
  (10 / 3 < d ∧ d < 4) := 
sorry

end arithmetic_sequence_sum_minimum_l550_550438


namespace quiz_answer_key_count_l550_550663

theorem quiz_answer_key_count :
  ∃ n : ℕ, n = 480 ∧
  (∃ tf_count : ℕ, tf_count = 30 ∧
   (∃ mc_count : ℕ, mc_count = 16 ∧ 
    n = tf_count * mc_count)) :=
    sorry

end quiz_answer_key_count_l550_550663


namespace closest_point_on_line_l550_550388

def closest_point_line_to_point (p1 p2 d1 d2 : ℝ) (line_point : ℝ × ℝ × ℝ) (line_dir : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x1, y1, z1) := line_point
  let (dx, dy, dz) := line_dir
  let s := -(d1 * (d2 + dx) + d2^2 + dy^2 + dz * (dx + dy)) /
              (dx^2 + dy^2 + dz^2)
  (x1 + s * dx, y1 + s * dy, z1 + s * dz)

theorem closest_point_on_line (s : ℝ) :
  closest_point_line_to_point (1 : ℝ) (2 : ℝ) 3 :=
  let point_on_line := (2, -1, 3)
  let direction := (1, -4, 2)
  let closest_point := closest_point_line_to_point 1 2 3 point_on_line direction
  closest_point = (29/21, 31/21, 45/21) :=
by
  sorry

end closest_point_on_line_l550_550388


namespace ellipse_hyperbola_foci_l550_550979

theorem ellipse_hyperbola_foci (c d : ℝ) 
  (h_ellipse : d^2 - c^2 = 25) 
  (h_hyperbola : c^2 + d^2 = 64) : |c * d| = Real.sqrt 868.5 := by
  sorry

end ellipse_hyperbola_foci_l550_550979


namespace percentage_changed_l550_550016

/-- 
At the beginning of the year, 40% of the students in Mr. Well's class answered "Yes" to the question "Do you love math?",
and 60% answered "No." A mid-year check shows that 30% answered "Yes", and 70% answered "No." However, by the end of the 
year, 60% answered "Yes" and 40% answered "No." Prove that 30% of the students gave different answers throughout the year.
-/
theorem percentage_changed (begin_yes mid_yes end_yes : ℝ) (begin_no mid_no end_no : ℝ) (h1 : begin_yes = 40) 
  (h2 : begin_no = 60) (h3 : mid_yes = 30) (h4 : mid_no = 70) (h5 : end_yes = 60) (h6 : end_no = 40) : 
  ∃ y : ℝ, y = 30 :=
by
  use 30
  sorry

end percentage_changed_l550_550016


namespace books_checked_out_on_Thursday_l550_550298

theorem books_checked_out_on_Thursday (initial_books : ℕ) (wednesday_checked_out : ℕ) 
                                      (thursday_returned : ℕ) (friday_returned : ℕ) (final_books : ℕ) 
                                      (thursday_checked_out : ℕ) : 
  (initial_books = 98) → 
  (wednesday_checked_out = 43) → 
  (thursday_returned = 23) → 
  (friday_returned = 7) → 
  (final_books = 80) → 
  (initial_books - wednesday_checked_out + thursday_returned - thursday_checked_out + friday_returned = final_books) → 
  (thursday_checked_out = 5) :=
by
  intros
  sorry

end books_checked_out_on_Thursday_l550_550298


namespace smallest_n_l550_550551

def recurrence (n : ℕ) (f : ℕ → ℤ) : ℤ :=
if (n%2 = 0) then (f n) + 3 else (f n) - 2

noncomputable def f : ℕ → ℤ
| 1     := 60
| (n+1) := recurrence n f

theorem smallest_n (n : ℕ) : (∀ m ≥ n, f m ≥ 63) ↔ n = 11 := sorry

end smallest_n_l550_550551


namespace geometric_series_sum_correct_l550_550033

-- Given conditions
def a : ℤ := 3
def r : ℤ := -2
def n : ℤ := 10

-- Sum of the geometric series formula
def geometric_series_sum (a r n : ℤ) : ℤ := 
  a * (r^n - 1) / (r - 1)

-- Goal: Prove that the sum of the series is -1023
theorem geometric_series_sum_correct : 
  geometric_series_sum a r n = -1023 := 
by
  sorry

end geometric_series_sum_correct_l550_550033


namespace solve_equation_real_l550_550239

theorem solve_equation_real (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) (h3 : x ≠ 4) :
  (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 2) * (x - 1) / ((x - 4) * (x - 2) * (x - 1)) = 1 ↔
  x = (9 + Real.sqrt 5) / 2 ∨ x = (9 - Real.sqrt 5) / 2 :=
by  
  sorry

end solve_equation_real_l550_550239


namespace coprime_in_selection_l550_550291

theorem coprime_in_selection (n : ℕ) (s : Finset ℕ) (h₁ : s.card = n + 1) (h₂ : ∀ x ∈ s, x ∈ Finset.range (2 * n + 1)) :
  ∃ a b ∈ s, a ≠ b ∧ Nat.coprime a b := 
sorry

end coprime_in_selection_l550_550291


namespace valid_three_digit_numbers_no_seven_nine_l550_550852

noncomputable def count_valid_three_digit_numbers : Nat := 
  let hundredsChoices := 7
  let tensAndUnitsChoices := 8
  hundredsChoices * tensAndUnitsChoices * tensAndUnitsChoices

theorem valid_three_digit_numbers_no_seven_nine : 
  count_valid_three_digit_numbers = 448 := by
  sorry

end valid_three_digit_numbers_no_seven_nine_l550_550852


namespace more_odd_numbers_than_even_tens_greater_than_ones_and_hundreds_l550_550731

open Finset

-- Definitions here should only come from the conditions
def digits := {1, 2, 3, 4, 5}

-- Define the three-digit numbers
def three_digit_numbers := { x : ℕ | ∃ a b c, a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ x = 100 * a + 10 * b + c }

-- Statement to prove: More odd numbers than even numbers
theorem more_odd_numbers_than_even : 
    card {x ∈ three_digit_numbers | x % 2 = 1} > card {x ∈ three_digit_numbers | x % 2 = 0} := sorry

-- Statement to prove: Numbers with tens digit greater than ones and hundreds digits count to 20
theorem tens_greater_than_ones_and_hundreds : 
    card {x ∈ three_digit_numbers | ∃ a b c, a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ x = 100 * a + 10 * b + c ∧ b > a ∧ b > c} = 20 := sorry

end more_odd_numbers_than_even_tens_greater_than_ones_and_hundreds_l550_550731


namespace student_needs_percentage_to_pass_l550_550002

theorem student_needs_percentage_to_pass :
  ∀ (obtained_marks shortfall_marks total_marks passing_marks required_percentage : ℕ),
  obtained_marks = 160 →
  shortfall_marks = 20 →
  total_marks = 300 →
  passing_marks = obtained_marks + shortfall_marks →
  required_percentage = (passing_marks * 100) / total_marks →
  required_percentage = 60 :=
by
  intros obtained_marks shortfall_marks total_marks passing_marks required_percentage
  intros h_obtained h_shortfall h_total h_passing h_required
  rw [h_obtained, h_shortfall, h_total] at *
  have h1: passing_marks = 180 := by linarith,
  have h2: required_percentage = (180 * 100) / 300 := by rw [← h1, h_passing],
  have h3: required_percentage = 60 := by norm_num,
  exact h3

end student_needs_percentage_to_pass_l550_550002


namespace part1_part2a_part2b_l550_550210

-- Given definitions for functions
def f (x : ℝ) : ℝ := x * Real.log x
def g (a x : ℝ) : ℝ := a * Real.exp x
def G (a x : ℝ) : ℝ := f x - g a x

-- Proof 1: if the tangent line of the curve y=f(x) at x=1 is also tangent to the curve y=g(x), then a = e^{-2}
theorem part1 (a : ℝ) : 
  (∃ x, g a x = x - 1 ∧ (∂' (λ x, f x) 1) = (∂' (λ x, g a x) x)) → 
   a = Real.exp (-2) :=
sorry

-- Proof 2a: If the function G(x)=f(x)-g(x) has two extreme points, find the range of values of a.
theorem part2a (a : ℝ) : 
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ G a x₁ = G a x₂ ∧ (∂' (λ x, G a x) x₁) = 0 ∧ (∂' (λ x, G a x) x₂) = 0) → 
  a ∈ Ioo 0 (1 / Real.exp 1) :=
sorry

-- Proof 2b: Prove that if ae^2 ≥ 2, then G(x) < 0.
theorem part2b (a : ℝ) : 
  a * Real.exp 2 ≥ 2 → 
  ∀ x : ℝ, G a x < 0 :=
sorry


end part1_part2a_part2b_l550_550210


namespace fred_seashells_l550_550286

-- Definitions based on conditions
def tom_seashells : Nat := 15
def total_seashells : Nat := 58

-- The theorem we want to prove
theorem fred_seashells : (15 + F = 58) → F = 43 := 
by
  intro h
  have h1 : F = 58 - 15 := by linarith
  exact h1

end fred_seashells_l550_550286


namespace circumcircle_equilateral_triangle_l550_550947

section

variables {R : ℝ} (A B C M : ℝ) (circumcenter : ℝ) (circumradius : ℝ) 

-- Define the point M on the circumcircle of an equilateral triangle ABC.
-- The lengths MA, MB, and MC are represented.

theorem circumcircle_equilateral_triangle 
  (M_on_circumcircle : ∀ (M : ℝ), ∃ (A B C : ℝ) (circumcenter circumradius : ℝ), is_circumcircle M A B C circumcenter circumradius)
  (equilateral_triangle : equilateral_triangle ABC)
  (circumradius_value : ∀ (M : ℝ), MA^4 + MB^4 + MC^4 = 18 * circumradius^4) :
  MA^4 + MB^4 + MC^4 = 18 * R^4 := 
sorry

end

end circumcircle_equilateral_triangle_l550_550947


namespace count_valid_n_l550_550774

def floor_sqrt (x : ℝ) : ℝ :=
  real.floor (real.sqrt x)

def A (n : ℤ) : ℝ :=
  ∫ x in (1 : ℝ)..(n : ℝ), floor_sqrt x * x

def valid_n (n : ℤ) : Prop :=
  2 ≤ n ∧ n ≤ 1000 ∧ is_integer (A n)

theorem count_valid_n : 
  (finset.filter valid_n (finset.range 1001)).card = 483 :=
sorry

end count_valid_n_l550_550774


namespace unique_solution_x_l550_550243

noncomputable def solveForX : ℝ :=
  let x := 0
  let y := 2^x
  if |x - log 2 y| = 2 * (x + log 2 y) then x else sorry

theorem unique_solution_x :
  ∀ x y : ℝ, |x - log 2 y| = 2 * (x + log 2 y) ∧ y = 2^x → x = 0 :=
by
  rintros x y ⟨h1, h2⟩
  have : y = 2^x := h2
  rw [h2] at h1
  have h3 : log 2 y = x
  sorry

#eval solveForX

end unique_solution_x_l550_550243


namespace trapezoid_angles_l550_550681

theorem trapezoid_angles (A B C D : ℝ) (circle : ℝ) (M N : ℝ)
  (h1 : is_trapezoid A B C D) 
  (h2 : circle.diameter = A - D)
  (h3 : midpoints_of_sides M N A B C D)
  (h4 : is_tangent circle BC) :
  angles_of_trapezoid A B C D = (75, 75, 105, 105) := 
sorry

end trapezoid_angles_l550_550681


namespace same_color_eye_proportion_l550_550367

theorem same_color_eye_proportion :
  ∀ (a b c d e f : ℝ),
  a + b + c = 0.30 →
  a + d + e = 0.40 →
  b + d + f = 0.50 →
  a + b + c + d + e + f = 1 →
  c + e + f = 0.80 :=
by
  intros a b c d e f h1 h2 h3 h4
  sorry

end same_color_eye_proportion_l550_550367


namespace find_a4_l550_550136

noncomputable def seq : ℕ → ℤ
| 0       := 0
| 1       := 1
| (n + 2) := seq (n + 1) - n - 1

theorem find_a4 : seq 4 = -5 := 
by {
  have h0 : seq 0 = 0 := rfl,
  have h1 : seq 1 = 1 := rfl,
  have h2 : seq 2 = seq 1 - 1 := rfl,
  have h3 : seq 2 = 0 := by rw [h1, sub_self],
  have h4 : seq 3 = seq 2 - 2 := rfl,
  have h5 : seq 3 = -2 := by rw [h3, zero_sub],
  show seq 4 = -5, 
  { rw [h4, h5, sub_eq_add_neg, sub_eq_add_neg],
    ring }
}

end find_a4_l550_550136


namespace triangle_side_length_l550_550491

theorem triangle_side_length 
  (A : ℝ) (a m n : ℝ) 
  (hA : A = 60) 
  (h1 : m + n = 7) 
  (h2 : m * n = 11) : a = 4 :=
by
  sorry

end triangle_side_length_l550_550491


namespace circumscribed_radius_l550_550344

-- Circle radius definition
def circle_radius : ℝ := 8

-- Definition of the radius of the circumscribed circle
def circumradius (φ : ℝ) : ℝ := 4 * Real.sec (φ / 2)

-- Theorem stating the problem
theorem circumscribed_radius (φ : ℝ) (h : φ > 0 ∧ φ < 2*Real.pi) :
  circumradius φ = 4 * Real.sec (φ / 2) :=
sorry

end circumscribed_radius_l550_550344


namespace four_people_four_chairs_l550_550885

theorem four_people_four_chairs :
  ∃ n : ℕ, n = (Nat.factorial 4) ∧ n = 24 :=
by
  use 24
  split
  · norm_num
  · rfl


end four_people_four_chairs_l550_550885


namespace geometric_progression_fourth_term_l550_550041

theorem geometric_progression_fourth_term :
  ∀ (a₁ a₂ a₃ a₄ : ℝ), a₁ = 2^(1/2) ∧ a₂ = 2^(1/4) ∧ a₃ = 2^(1/6) ∧ (a₂ / a₁ = r) ∧ (a₃ = a₂ * r⁻¹) ∧ (a₄ = a₃ * r) → a₄ = 2^(1/8) := by
intro a₁ a₂ a₃ a₄
intro h
sorry

end geometric_progression_fourth_term_l550_550041


namespace sequence_no_consecutive_ones_probability_l550_550346

/-- A sequence of length 8 that does not contain two consecutive 1s. The probability
of such a sequence can be written as m/n, where m and n are relatively prime. We prove
that m + n = 311. -/
theorem sequence_no_consecutive_ones_probability :
  ∃ m n : ℕ,
    (Nat.gcd m n = 1) ∧
    (m + n = 311) ∧
    (m : ℚ) / n = (Fib 10) / (2 ^ 8) :=
sorry

end sequence_no_consecutive_ones_probability_l550_550346


namespace shortest_fence_length_l550_550054

open Real

noncomputable def area_of_garden (length width : ℝ) : ℝ := length * width

theorem shortest_fence_length (length width : ℝ) (h : area_of_garden length width = 64) :
  4 * sqrt 64 = 32 :=
by
  -- The statement sets up the condition that the area is 64 and asks to prove minimum perimeter (fence length = perimeter).
  sorry

end shortest_fence_length_l550_550054


namespace shift_right_inverse_exp_eq_ln_l550_550237

variable (f : ℝ → ℝ)

theorem shift_right_inverse_exp_eq_ln :
  (∀ x, f (x - 1) = Real.log x) → ∀ x, f x = Real.log (x + 1) :=
by
  sorry

end shift_right_inverse_exp_eq_ln_l550_550237


namespace car_distances_l550_550635

theorem car_distances (x : ℝ) (h1 : 2 * x = 5) (h2 : 30 * x = 150) (h3 : 40 * x = 200) : 
  x = 5 ∧ 30 * x = 150 ∧ 40 * x = 200 :=
by
  split
  { exact 5 }
  split
  { exact 150 }
  { exact 200 }

end car_distances_l550_550635


namespace range_of_omega_intervals_of_monotonic_increase_sin_B_mul_sin_C_l550_550127

open Real

-- Define the function f
def f (ω x : ℝ) : ℝ := (cos (ω * x))^2 - (sin (ω * x))^2 + 2 * sqrt 3 * cos (ω * x) * sin (ω * x)

-- Given ω > 0
variable (ω : ℝ) (ω_pos : 0 < ω)

-- Given distance condition between adjacent axes of symmetry
axiom symmetry_distance_condition : (π / ω) / 2 ≥ π / 2

-- Prove range of ω
theorem range_of_omega : 0 < ω ∧ ω ≤ 1 :=
sorry

-- Prove intervals of monotonic increase for f(x)
theorem intervals_of_monotonic_increase (k : ℤ) : 
  (k * π / ω - π / (3 * ω)) ≤ x ∧ x ≤ (k * π / ω + π / (6 * ω)) →
  monotone_on (f ω) (set.Icc (k * π / ω - π / (3 * ω)) (k * π / ω + π / (6 * ω))) :=
sorry

-- Given conditions on triangle ABC
variable (a b c A B C : ℝ)
variable (a_value : a = sqrt 3) (bc_sum : b + c = 3) (fA : f 1 A = 1)
variable (A_pos : 0 < A) (A_lt_pi : A < π)

-- Prove the value of sin B · sin C
theorem sin_B_mul_sin_C : sin B * sin C = 1 / 2 := 
sorry

end range_of_omega_intervals_of_monotonic_increase_sin_B_mul_sin_C_l550_550127


namespace points_covered_by_equilateral_triangle_l550_550596

theorem points_covered_by_equilateral_triangle
  (P : set (ℝ × ℝ)) (hP : ∀ {p q : ℝ × ℝ}, p ∈ P → q ∈ P → dist p q ≤ 1) :
  ∃ (T : set (ℝ × ℝ)), (is_equilateral_triangle T ∧ side_length T = sqrt 3 ∧ ∀ p ∈ P, p ∈ T) :=
sorry

noncomputable def is_equilateral_triangle (T : set (ℝ × ℝ)) : Prop :=
  ∃ (a b c : ℝ × ℝ), (T = {a, b, c}) ∧ 
  (dist a b = dist b c ∧ dist b c = dist c a ∧ dist c a = dist a b)

noncomputable def side_length (T : set (ℝ × ℝ)) : ℝ :=
  if h : ∃ (a b c : ℝ × ℝ), (T = {a, b, c}) 
  then dist (classical.some h) (classical.some (classical.some_spec h))
  else 0

end points_covered_by_equilateral_triangle_l550_550596


namespace form_polygon_with_given_area_and_perimeter_l550_550277

def match_length := 2
def number_of_matches := 12
def total_length := number_of_matches * match_length
def polygon_area := 16
noncomputable def can_form_polygon_with_area (perimeter area : ℕ) : Prop :=
  ∃ (polygon : Type), has_perimeter polygon perimeter ∧ has_area polygon area

theorem form_polygon_with_given_area_and_perimeter :
  can_form_polygon_with_area total_length polygon_area :=
sorry

end form_polygon_with_given_area_and_perimeter_l550_550277


namespace option_c_is_quadratic_l550_550009

def is_quadratic_equation_in_one_variable (eq : String) : Prop :=
  ∃ x : ℝ, eq = "2 / 3 * x^2 + 5 = 0" ∧
  (∀ eq, eq.contains "y" → false) ∧
  (eq.degree = 2) ∧
  (eq.coefficient(2) ≠ 0) ∧
  (eq.is_polynomial)

theorem option_c_is_quadratic : is_quadratic_equation_in_one_variable "2 / 3 * x^2 + 5 = 0" :=
sorry

end option_c_is_quadratic_l550_550009


namespace sum_squares_products_eq_factorial_minus_one_l550_550040

-- Define the sum of the squares of the products of numbers in sets that do not contain two consecutive numbers
def SumOfSquaresOfProducts (n : ℕ) : ℕ :=
  sorry

-- Main theorem statement
theorem sum_squares_products_eq_factorial_minus_one (n : ℕ) :
  SumOfSquaresOfProducts n = (n+1)! - 1 :=
sorry

end sum_squares_products_eq_factorial_minus_one_l550_550040


namespace sum_first_15_l550_550797

open_locale big_operators

-- Define the arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_first_n (a : ℕ → ℤ) (n : ℕ) := ∑ i in finset.range n, a i

-- Lean 4 statement for the problem
theorem sum_first_15 {a : ℕ → ℤ}
  (h : arithmetic_seq a) 
  (h_cond : a 2 + a 12 - a 7 = 2) :  -- a_3 = a 2, a_13 = a 12, a_8 = a 7
  sum_first_n a 15 = 30 :=
sorry

end sum_first_15_l550_550797


namespace range_of_a_l550_550152

noncomputable def proof_problem (a : ℝ) : Prop :=
  ∀ (x : ℝ), arctan (sqrt (x^2 + x + 13 / 4)) ≥ π / 3 - a → a ≥ 0

theorem range_of_a (a : ℝ) : proof_problem a := 
  sorry

end range_of_a_l550_550152


namespace max_abs_sum_on_ellipse_l550_550820

theorem max_abs_sum_on_ellipse :
  ∀ (x y : ℝ), (x^2 / 4) + (y^2 / 9) = 1 → |x| + |y| ≤ 5 :=
by sorry

end max_abs_sum_on_ellipse_l550_550820


namespace correct_mean_after_adjustment_l550_550617

theorem correct_mean_after_adjustment :
  let initial_mean := 350
  let n := 50
  let incorrect_values_sum := initial_mean * n
  let corrections := [(180, 150), (235, 200), (290, 270)]
  let total_error := corrections.foldr (fun (corrected, incorrect) acc => acc + (corrected - incorrect)) 0
  let correct_sum := incorrect_values_sum + total_error
  let correct_mean := correct_sum / n
  correct_mean = 351.7 := 
by
  let initial_mean := 350
  let n := 50
  let incorrect_values_sum := initial_mean * n
  let corrections := [(180, 150), (235, 200), (290, 270)]
  let total_error := corrections.foldr (fun (corrected, incorrect) acc => acc + (corrected - incorrect)) 0
  let correct_sum := incorrect_values_sum + total_error
  let correct_mean := correct_sum / n
  show correct_mean = 351.7 from sorry

end correct_mean_after_adjustment_l550_550617


namespace find_B_l550_550435

noncomputable theory

variables {A B C : ℝ} {a b c : ℝ}
variables (sinA sinB sinC : ℝ)

-- Given conditions
def given_conditions : Prop := 
  (a + b) / sin (A + B) = (a - c) / (sin A - sin B) ∧ 
  b = 3 ∧ 
  sinA = (Real.sqrt 3) / 3

-- Prove that:
theorem find_B (h : given_conditions) : 
  B = π / 3 ∧ 
  ∃ (area : ℝ), area = (Real.sqrt 3 + 3 * Real.sqrt 2) / 2 :=
sorry

end find_B_l550_550435


namespace B_power_fours_implies_B_cubed_zero_l550_550534

open Matrix
open_locale Matrix

def is_zero_matrix_3x3 (B : Matrix (Fin 3) (Fin 3) ℝ) : Prop :=
  B = 0

theorem B_power_fours_implies_B_cubed_zero (B : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : B ^ 4 = 0) : B ^ 3 = 0 :=
by
  sorry

end B_power_fours_implies_B_cubed_zero_l550_550534


namespace sum_of_squares_inequality_l550_550214

variable {n : ℕ}
variable {a : Fin n → ℝ}

theorem sum_of_squares_inequality (h_pos : ∀ i, 0 < a i) (h_sum : (Finset.univ.sum (λ i, a i)) = 1) :
  (Finset.univ.sum (λ i, (a i)^2 / (a i + a (i + 1)%n))) ≥ (1 / 2) :=
sorry

end sum_of_squares_inequality_l550_550214


namespace choose_5_person_committee_l550_550683

theorem choose_5_person_committee (total_members ineligible_members eligible_members chosen_members : ℕ)
  (h1 : total_members = 30)
  (h2 : ineligible_members = 4)
  (h3 : eligible_members = total_members - ineligible_members)
  (h4 : chosen_members = 5)
  (h5 : eligible_members = 26) :
  nat.choose eligible_members chosen_members = 60770 :=
by
  rw [h3, h5]
  exact nat.choose_spec 26 5
  sorry

end choose_5_person_committee_l550_550683


namespace smallest_positive_period_max_value_on_interval_min_value_on_interval_l550_550823

noncomputable def f (x : ℝ) : ℝ := sin x * cos x + sqrt 3 * cos (π - x) * cos x

theorem smallest_positive_period (x : ℝ) : (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ π) := 
by
  sorry

theorem max_value_on_interval (x : ℝ) : x ∈ Icc 0 (π / 2) → f x ≤ 1 - sqrt 3 / 2 :=
by
  sorry

theorem min_value_on_interval (x : ℝ) : x ∈ Icc 0 (π / 2) → f x ≥ -sqrt 3 :=
by
  sorry

end smallest_positive_period_max_value_on_interval_min_value_on_interval_l550_550823


namespace population_net_change_l550_550004

theorem population_net_change :
  let first_year := 1.2 in
  let second_year := 0.85 in
  let third_year := 1.1 in
  let fourth_year := 0.8 in
  let fifth_year := 1.15 in
  ((first_year * second_year * third_year * fourth_year * fifth_year) - 1) * 100 = 11 :=
by
  sorry

end population_net_change_l550_550004


namespace total_sum_of_rupees_l550_550279

theorem total_sum_of_rupees :
  ∃ (total_coins : ℕ) (paise20_coins : ℕ) (paise25_coins : ℕ),
    total_coins = 344 ∧ paise20_coins = 300 ∧ paise25_coins = total_coins - paise20_coins ∧
    (60 + (44 * 0.25)) = 71 :=
by
  sorry

end total_sum_of_rupees_l550_550279


namespace infinite_set_of_midpoints_l550_550098

def is_midpoint (M : set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  ∃ a b : ℝ × ℝ, a ∈ M ∧ b ∈ M ∧ p = (a.1 + b.1) / 2 ∧ (a.2 + b.2) / 2

theorem infinite_set_of_midpoints (M : set (ℝ × ℝ)) (h : ∀ p ∈ M, is_midpoint M p) : set.infinite M :=
  sorry

end infinite_set_of_midpoints_l550_550098


namespace lisa_matching_pair_probability_l550_550756

theorem lisa_matching_pair_probability :
  let total_socks := 22
  let gray_socks := 12
  let white_socks := 10
  let total_pairs := total_socks * (total_socks - 1) / 2
  let gray_pairs := gray_socks * (gray_socks - 1) / 2
  let white_pairs := white_socks * (white_socks - 1) / 2
  let matching_pairs := gray_pairs + white_pairs
  let probability := matching_pairs / total_pairs
  probability = (111 / 231) :=
by
  sorry

end lisa_matching_pair_probability_l550_550756


namespace leak_drain_time_l550_550308

def pump_rate := 1 / 2 -- tank per hour
def combined_rate := 1 / 3 -- tank per hour

theorem leak_drain_time : 
  let leak_rate := pump_rate - combined_rate in
  1 / leak_rate = 6 :=
by
  let leak_rate := pump_rate - combined_rate
  have h1 : leak_rate = 1 / 6 := by
    calc 
      leak_rate = 1 / 2 - 1 / 3 : by rfl
      ... = 3 / 6 - 2 / 6 : by norm_num
      ... = 1 / 6 : by norm_num
  show 1 / leak_rate = 6
  calc
    1 / leak_rate = 1 / (1 / 6) : by rw h1
    ... = 6 : by norm_num

end leak_drain_time_l550_550308


namespace find_number_l550_550150

theorem find_number (y : ℝ) (h : 0.25 * 820 = 0.15 * y - 20) : y = 1500 :=
by
  sorry

end find_number_l550_550150


namespace integer_solutions_count_l550_550080

theorem integer_solutions_count :
  let circle_eq : ∀ (x y : ℤ), x^2 + y^2 = 65 → Prop := 
    λ x y h, true 
  let line_eq : ∀ (a b x y : ℤ), ax + by = 2 → Prop :=
    λ a b x y h, true 
  (∃ (a b : ℤ), ∃ (x y : ℤ), line_eq a b x y (2) ∧ circle_eq x y 65) → 
  ∃ (k : ℕ), k = 128 := sorry

end integer_solutions_count_l550_550080


namespace circle_after_transformation_l550_550395

noncomputable def transform_curve (x y : ℝ) : ℝ × ℝ :=
  (1/3 * x, 1/2 * y)

theorem circle_after_transformation (x y : ℝ) (h : x^2 / 9 + y^2 / 4 = 1) :
  let (x', y') := transform_curve x y in
  x'^2 + y'^2 = 1 :=
by
  let (x', y') := transform_curve x y
  sorry

end circle_after_transformation_l550_550395


namespace quadratic_problem_l550_550757

theorem quadratic_problem :
  ∀ x: ℝ,
  (2 * x^2 - 12 * x + 1 = 0) →
  ((x - 3)^2 = (17 / 2)) :=
by
  intros x h
  have : x^2 - 6*x + 1 / 2 = 0 := by
    field_simp [h]
  have : (x - 3)^2 = 17 / 2 := by
    sorry
  exact this

end quadratic_problem_l550_550757


namespace point_P_in_second_quadrant_l550_550862

theorem point_P_in_second_quadrant 
    (sin_2_pos : sin 2 > 0) 
    (cos_2_neg : cos 2 < 0) : 
    ∃ (quadrant : ℕ), quadrant = 2 := 
by
  use 2
  sorry

end point_P_in_second_quadrant_l550_550862


namespace sol_earns_amount_l550_550963

theorem sol_earns_amount (candy_bars_first_day : ℕ)
                         (additional_candy_bars_per_day : ℕ)
                         (sell_days_per_week : ℕ)
                         (cost_per_candy_bar_cents : ℤ) :
                         candy_bars_first_day = 10 →
                         additional_candy_bars_per_day = 4 →
                         sell_days_per_week = 6 →
                         cost_per_candy_bar_cents = 10 →
                         (∑ i in finset.range sell_days_per_week, (candy_bars_first_day + additional_candy_bars_per_day * i).to_int) * cost_per_candy_bar_cents / 100 = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end sol_earns_amount_l550_550963


namespace median_AM_of_triangle_l550_550169

theorem median_AM_of_triangle {R : ℝ} (hR : R = 4) 
    (θ : ℝ) (hθ : θ = (Real.pi / 8)) 
    (A B K C O M : ℝ → ℝ → Prop)
    (hA : A = (4, 0))
    (hB : B = (4 * cos(θ), 4 * sin(θ)))
    (hK : ∀ (x : ℝ), K x = O x ∧ O x = A x) 
    (hC : C = (4 * (1 + Real.sqrt 2), -4))
    (hM : M = midpoint (B, C)) : 
    distance (A, M) = 2 * Real.sqrt (9 + 6 * Real.sqrt 2) := 
sorry

end median_AM_of_triangle_l550_550169


namespace square_inscribed_in_isosceles_right_triangle_area_l550_550967

theorem square_inscribed_in_isosceles_right_triangle_area :
  ∀ (AG AD AB CD : ℝ), AG = AD → AB = 40 → CD = 90 →
  let x := (sqrt (AB * CD)) in
  x^2 = 3600 :=
by
  intros AG AD AB CD h1 h2 h3
  let x := sqrt (AB * CD)
  sorry

end square_inscribed_in_isosceles_right_triangle_area_l550_550967


namespace quadratic_inequality_solution_set_l550_550076

theorem quadratic_inequality_solution_set (a b c : ℝ) : 
  (∀ x : ℝ, - (a / 3) * x^2 + 2 * b * x - c < 0) ↔ (a > 0 ∧ 4 * b^2 - (4 / 3) * a * c < 0) := 
by
  sorry

end quadratic_inequality_solution_set_l550_550076


namespace trigonometric_identity_l550_550479

theorem trigonometric_identity (θ : ℝ) (h : Real.tan (θ + Real.pi / 4) = 2) : 
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = -2 := 
sorry

end trigonometric_identity_l550_550479


namespace magical_apple_tree_one_remaining_fruit_magical_apple_tree_no_remaining_fruit_l550_550333

theorem magical_apple_tree_one_remaining_fruit : 
    ∃ order : list (sum bool bool), 
    ∃ remaining_fruit : sum bool bool, 
    (fruits_after_picking order = [remaining_fruit] 
    ∧ remaining_fruit = sum.inl true) := 
sorry

theorem magical_apple_tree_no_remaining_fruit:
    ¬∃ order : list (sum bool bool), 
    fruits_after_picking order = [] := 
sorry

-- Definitions:
def fruits_after_picking (order : list (sum bool bool)) : list (sum bool bool) :=
    sorry -- definition that mimics conditions of picking

end magical_apple_tree_one_remaining_fruit_magical_apple_tree_no_remaining_fruit_l550_550333


namespace minimum_value_4x_plus_y_l550_550492

open_locale classical

noncomputable def find_min_value (x y : ℚ) :=
  let AM := x * 1  -- representing \overrightarrow{AM} = x * \overrightarrow{AB}
  let AN := y * 1  -- representing \overrightarrow{AN} = y * \overrightarrow{AC}
  (4*x + y)

theorem minimum_value_4x_plus_y : 
  ∃ (x y : ℚ), 
    (∃ (E : ℚ), ∃ (AB AC : ℚ), ∃ (M N : E), 
      let EM := (x - 1 / 4) * AB - 1 / 4 * AC in
      let EN := (y - 1 / 4) * AC - 1 / 4 * AB in
      EM = (E - 1 /4) * AB - 1 / 4 * AC ∧
      (y * 1) = (x * 1) - (1 / 4) * 1 ∧
      EM = λ * EN ∧
      4 * x + y = 9 / 4
    )
  := sorry

end minimum_value_4x_plus_y_l550_550492


namespace initial_payment_correct_l550_550570

theorem initial_payment_correct (Price_car : ℕ) (Installments : ℕ) (Installment_amount : ℕ) :
  Price_car = 18000 → Installments = 6 → Installment_amount = 2500 → 
  ∃ (initial_payment : ℕ), initial_payment = Price_car - Installments * Installment_amount ∧ initial_payment = 3000 :=
by
  intros
  use Price_car - Installments * Installment_amount
  split
  · rfl
  · sorry

end initial_payment_correct_l550_550570


namespace trigonometric_identity_solution_l550_550113

theorem trigonometric_identity_solution (α : ℝ) (h1 : sin α = 1/2 + cos α) (h2 : 0 < α ∧ α < π / 2) : 
  (cos (2 * α)) / (sin (α - π / 4)) = -sqrt 14 / 2 := 
by
  -- Proof omitted
  sorry

end trigonometric_identity_solution_l550_550113


namespace length_AB_l550_550795

theorem length_AB
  (α : ℝ) (a b : ℝ) (hα : 0 < α ∧ α < π)
  (h1 : ∃ M N : ℝ × ℝ, dist M B = a ∧ dist N B = b ∧ right_angle M A N) :
  (|AB| =
    if B lies within the given angle or its vertical opposite then
      (sqrt (a^2 + b^2 + 2 * a * b * cos α)) / (sin α)
    else
      (sqrt (a^2 + b^2 - 2 * a * b * cos α)) / (sin α)) :=
sorry

end length_AB_l550_550795


namespace determine_a_value_l550_550134

theorem determine_a_value (a : ℤ) (h : ∀ x : ℝ, x^2 + 2 * (a:ℝ) * x + 1 > 0) : a = 0 := 
sorry

end determine_a_value_l550_550134


namespace maximum_volume_pyramid_l550_550343

theorem maximum_volume_pyramid (R : ℝ) (h : ℝ) (a : ℝ) (H₁ : R > 0) (H₂ : h = (4 / 3) * R) (H₃ : a^2 = 2 * h * (2 * R - h)) :
  let V := (2 / 3) * h^2 * (2 * R - h) in
  V = (64 / 81) * R^3 :=
by
  sorry

end maximum_volume_pyramid_l550_550343


namespace general_term_formula_sum_first_n_b_l550_550536

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ q ≠ 1 ∧ ∀ n, a(n + 1) = a(n) * q

def forms_arithmetic_sequence (a1 a2 a3 : ℝ) : Prop :=
  2 * a2 = a1 + a3

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, a (k + 1)

-- Given Conditions
variables (a : ℕ → ℝ) (q : ℝ)
axiom geom_seq : is_geometric_sequence a q
axiom arith_seq : forms_arithmetic_sequence (4 * a 1) (3 * a 2) (2 * a 3)
axiom sum_first_four_terms : sum_of_first_n_terms a 4 = 15

-- Proof Problems
theorem general_term_formula :
  ∀ n, a n = 2^(n - 1) :=
sorry

def b (n : ℕ) : ℝ :=
  a n + 2 * n

theorem sum_first_n_b (n : ℕ) :
  sum_of_first_n_terms b n = 2^n + n^2 + n - 1 :=
sorry

end general_term_formula_sum_first_n_b_l550_550536


namespace emily_second_round_points_l550_550751

theorem emily_second_round_points (P : ℤ)
  (first_round_points : ℤ := 16)
  (last_round_points_lost : ℤ := 48)
  (end_points : ℤ := 1)
  (points_equation : first_round_points + P - last_round_points_lost = end_points) :
  P = 33 :=
  by {
    sorry
  }

end emily_second_round_points_l550_550751


namespace length_of_AB_l550_550692

-- Definitions of the conditions

def parabola : set (ℝ × ℝ) := {p | p.2 ^ 2 = 4 * p.1}
def focus := (1 : ℝ, 0 : ℝ)
def line_through_focus (slope : ℝ) : set (ℝ × ℝ) := 
  {p | p.2 = slope * (p.1 - focus.1)}

-- The statement which we need to prove
theorem length_of_AB :
  let A := filter (λ p, p ∈ parabola ∧ p ∈ line_through_focus (sqrt 3)) ⟨1, 0⟩,
      B := filter (λ p, p ∈ parabola ∧ p ∈ line_through_focus (sqrt 3)) ⟨1, 0⟩ in
  ∃ A B, A.1 ≠ B.1 ∧ A.2 = B.2 ∧ |A.1 - B.1| + |A.2 - B.2| = 16 / 3 :=
sorry

end length_of_AB_l550_550692


namespace marbles_problem_l550_550527

theorem marbles_problem (p : ℕ) (m n r : ℕ) 
(hp : Nat.Prime p) 
(h1 : p = 2017)
(h2 : N = p^m * n)
(h3 : ¬ p ∣ n)
(h4 : r = n % p) 
(h N : ∀ (N : ℕ), N = 3 * p * 632 - 1)
: p * m + r = 3913 := 
sorry

end marbles_problem_l550_550527


namespace milo_skateboarding_speed_l550_550569

theorem milo_skateboarding_speed (cory_speed milo_skateboarding_speed : ℝ) 
  (h1 : cory_speed = 12) 
  (h2 : cory_speed = 2 * milo_skateboarding_speed) : 
  milo_skateboarding_speed = 6 :=
by sorry

end milo_skateboarding_speed_l550_550569


namespace min_value_of_f_on_line_l550_550695

-- Definitions and conditions
def f (x y : ℝ) : ℝ := 2^x + 4^y
def on_line (x y : ℝ) : Prop := x + 2 * y = 3

-- Goal
theorem min_value_of_f_on_line :
  ∃ x y : ℝ, on_line x y ∧ ∀ a b : ℝ, on_line a b → f x y ≤ f a b ∧ f x y = 4 * real.sqrt 2 :=
sorry

end min_value_of_f_on_line_l550_550695


namespace max_distance_inequality_l550_550986

noncomputable def max_distances (AP BP CP : ℝ) : ℝ := max (max AP BP) CP

theorem max_distance_inequality 
  (P A B C : ℝ) 
  (d_a d_b d_c AP BP CP : ℝ)
  (hP_within_triangle : P ≥ 0 ∧ P ≤ 1 ∧ A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0)
  (hd_a : d_a = distance P BC)
  (hd_b : d_b = distance P CA)
  (hd_c : d_c = distance P AB)
  (hAP : AP = distance P A)
  (hBP : BP = distance P B)
  (hCP : CP = distance P C) :
  max_distances AP BP CP ≥ sqrt (d_a^2 + d_b^2 + d_c^2) := by
  sorry

end max_distance_inequality_l550_550986


namespace infinite_n_gcd_phi_floor_l550_550103

-- Definitions of the conditions
variables (a b m k : ℕ)
variable (h_k : k ≥ 2)

def phi_iterated (m : ℕ) (n : ℕ) : ℕ :=
nat.iterate nat.totient m n

-- The proof problem statement
theorem infinite_n_gcd_phi_floor (a b m k : ℕ) (h_k : k ≥ 2) :
  ∃ᶠ n in at_top, gcd (phi_iterated m n) (⌊(a * n + b)^(1:ℝ/k)⌋) = 1 := sorry

end infinite_n_gcd_phi_floor_l550_550103


namespace one_thirds_in_eight_halves_l550_550851

theorem one_thirds_in_eight_halves : (8 / 2) / (1 / 3) = 12 := by
  sorry

end one_thirds_in_eight_halves_l550_550851


namespace anthony_ate_total_l550_550712

def slices := 16

def ate_alone := 1 / slices
def shared_with_ben := (1 / 2) * (1 / slices)
def shared_with_chris := (1 / 2) * (1 / slices)

theorem anthony_ate_total :
  ate_alone + shared_with_ben + shared_with_chris = 1 / 8 :=
by
  sorry

end anthony_ate_total_l550_550712


namespace min_value_of_l_l550_550561

def f (x : ℝ) : ℝ :=
  if abs x ≤ 1 then 2 * Real.cos (π / 2 * x) else x ^ 2 - 1

def g (x l : ℝ) : ℝ :=
  abs (f x + f (x + l) - 2) + abs (f x - f (x + l))

theorem min_value_of_l : 
  ∃ l > 0, (∀ x, g x l ≥ 2) ∧ (∀ ε > 0, ¬∃ l₀ ≥ 0, l₀ < l - ε ∧ (∀ x, g x l₀ ≥ 2)) := 
sorry

end min_value_of_l_l550_550561


namespace noelle_homework_assignments_l550_550693

theorem noelle_homework_assignments :
  ∀ (n : ℕ), n ≥ 16 → ∑ i in range(16), (i // 4) + 1 = 40 := 
by
  sorry

end noelle_homework_assignments_l550_550693


namespace zoo_total_animals_l550_550354

theorem zoo_total_animals (penguins polar_bears : ℕ)
  (h1 : penguins = 21)
  (h2 : polar_bears = 2 * penguins) :
  penguins + polar_bears = 63 := by
   sorry

end zoo_total_animals_l550_550354


namespace prob_X_lt_3_l550_550832

noncomputable def normal_dist (μ σ : ℝ) : MeasureTheory.Measure ℝ := sorry

variables (X : ℝ → MeasureTheory.ProbabilityTheory.Prob) (σ : ℝ)

axiom normal_X : X = normal_dist 1 σ
axiom prob_0_to_3 : MeasureTheory.Measure.prob (normal_dist 1 σ) (set.Ioc 0 3) = 0.5
axiom prob_0_to_1 : MeasureTheory.Measure.prob (normal_dist 1 σ) (set.Ioc 0 1) = 0.2

theorem prob_X_lt_3 : MeasureTheory.Measure.prob (normal_dist 1 σ) (set.Iic 3) = 0.8 := 
sorry

end prob_X_lt_3_l550_550832


namespace line_MN_tangent_to_ω_l550_550198

open_locale euclidean_geometry

variables {AB AC : ray}
variables {ω : circle}
variables {O E F R P N M : point}
variables (parallel_1 : line(thru O).parallel(line(thru E, F)))
variables (parallel_2 : line(thru M).parallel(line(thru R, AC)))

-- Given conditions
axiom AB_AC_distinct : distinct_rays AB AC
axiom ω_tangent_AC : ω.tangent E AC
axiom ω_tangent_AB : ω.tangent F AB
axiom R_on_EF : R ∈ segment(E, F)
axiom P_on_parallel_line : P ∈ line_of(O).parallel_to(line_of(E, F))
axiom N_on_PR_AC : N = intersection(line_of(P, R), AC)
axiom M_on_parallel_AB : M = intersection(AB, line_of(R).parallel_to(AC))

-- To Prove
theorem line_MN_tangent_to_ω : line_of(M, N).tangent_to ω := by sorry

end line_MN_tangent_to_ω_l550_550198


namespace find_a_l550_550800

noncomputable def z1 := λ (a : ℝ), a + 2 * complex.I
def z2 : ℂ := 2 + complex.I

theorem find_a (a : ℝ) (h : (z1 a) * z2 ∈ {z : ℂ | z.re = 0}) : a = 1 :=
by
  sorry

end find_a_l550_550800


namespace max_x_given_max_y_l550_550275

theorem max_x_given_max_y
  (x y : ℝ)
  (h1 : (16, 11) ∈ {(x, y) | y = x - 5})
  (h2 : (14, 9) ∈ {(x, y) | y = x - 5})
  (h3 : (12, 8) ∈ {(x, y) | y = x - 5})
  (h4 : (8, 5) ∈ {(x, y) | y = x - 5})
  (max_y : 10) :
  x = 15 := 
sorry

end max_x_given_max_y_l550_550275


namespace quadratic_eqns_mod_7_l550_550240

/-- Proving the solutions for quadratic equations in arithmetic modulo 7. -/
theorem quadratic_eqns_mod_7 :
  (¬ ∃ x : ℤ, (5 * x^2 + 3 * x + 1) % 7 = 0) ∧
  (∃! x : ℤ, (x^2 + 3 * x + 4) % 7 = 0 ∧ x % 7 = 2) ∧
  (∃ x1 x2 : ℤ, (x1 ^ 2 - 2 * x1 - 3) % 7 = 0 ∧ (x2 ^ 2 - 2 * x2 - 3) % 7 = 0 ∧ 
              x1 % 7 = 3 ∧ x2 % 7 = 6) :=
by
  sorry

end quadratic_eqns_mod_7_l550_550240


namespace remainder_of_n_div_11_is_1_l550_550529

def A : ℕ := 20072009
def n : ℕ := 100 * A

theorem remainder_of_n_div_11_is_1 :
  (n % 11) = 1 :=
sorry

end remainder_of_n_div_11_is_1_l550_550529


namespace range_b_a_l550_550916

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := Real.logBase (1/2) ((1 + b * x) / (1 - 2 * x))

theorem range_b_a (a : ℝ) (b : ℝ) 
  (h₁ : f x b = -f (-x) b)  ᾿-- Odd function property
  (h₂ : ∀ x y : ℝ, x < y → f x b < f y b)  -- Monotonic function property
  (ha : 0 < a ∧ a ≤ 1/2) : 
  ∃ c : Set.Icc 1 (Real.sqrt 2), c = b^a := 
by
  sorry

end range_b_a_l550_550916


namespace decreasing_interval_l550_550132

def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative function
def f_prime (x : ℝ) : ℝ := 3*x^2 - 3

theorem decreasing_interval : ∀ x : ℝ, -1 < x ∧ x < 1 → f_prime x < 0 :=
by
  intro x h
  have h1: x^2 < 1 := by
    sorry
  have h2: 3*x^2 < 3 := by
    sorry
  have h3: 3*x^2 - 3 < 0 := by
    sorry
  exact h3

end decreasing_interval_l550_550132


namespace isosceles_triangle_of_sin_cos_cond_l550_550892

theorem isosceles_triangle_of_sin_cos_cond (A B C : ℝ) (h1 : sin C = 2 * cos A * sin B) (h2 : A + B + C = π) : 
  A = B := 
by 
  sorry

end isosceles_triangle_of_sin_cos_cond_l550_550892


namespace determine_median_in_three_sessions_l550_550369

-- Define the main problem statement
theorem determine_median_in_three_sessions (coins : Fin 5 → ℝ) :
  ∃ median, ∃ (meetings : List (Fin 4 → Fin 5)),
  (List.length meetings = 3 ∧
   (∀ meet, meet ∈ meetings → ∃ (low high : Fin 5), 
   (low ≠ high) ∧ (coins low = Finset.min' (Finset.image (λ i, coins (meet i)) (finset.univ)) (by sorry)) ∧ 
   (coins high = Finset.max' (Finset.image (λ i, coins (meet i)) (finset.univ)) (by sorry))) ∧
   (median ∉ (List.no_duplicates meetings) ∧
   (∀ meet, meet ∈ meetings → coins median ≠ Finset.min' (Finset.image (λ i, coins (meet i)) (finset.univ)) (by sorry) ∧ 
   coins median ≠ Finset.max' (Finset.image (λ i, coins (meet i)) (finset.univ)) (by sorry)))) :=
sorry

end determine_median_in_three_sessions_l550_550369


namespace find_abc_l550_550557

-- Define the function piecewise
def f (x : ℝ) (a b c : ℕ) : ℝ :=
  if x > 0 then a * x + 4
  else if x = 0 then a * b + 1
  else b * x + c

-- Define the conditions
def condition1 (a : ℕ) : Prop := f 3 a _ _ = 7
def condition2 (b : ℕ) : Prop := f 0 _ b _ = 10
def condition3 (c : ℕ) : Prop := f -3 _ _ c = -11

-- Define the main proof statement
theorem find_abc (a b c : ℕ) 
  (h1 : condition1 a) 
  (h2 : condition2 b) 
  (h3 : condition3 c) 
  (h4 : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) 
: a + b + c = 26 := 
sorry

end find_abc_l550_550557


namespace top_triangle_possibilities_l550_550306

theorem top_triangle_possibilities :
  ∃ x ∈ {8, 9}, ∃ circle_numbers triangle_numbers hexagon_numbers : finset ℕ,
    (circle_numbers ∪ triangle_numbers ∪ hexagon_numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
    (circle_numbers ∩ triangle_numbers = ∅) ∧
    (circle_numbers ∩ hexagon_numbers = ∅) ∧
    (triangle_numbers ∩ hexagon_numbers = ∅) ∧
    (2 * triangle_numbers.sum = circle_numbers.sum * 3) ∧
    (2 * hexagon_numbers.sum = triangle_numbers.sum * 2) ∧
    (triangle_numbers.contains x) ∧
    (triangle_numbers.sum = 15) ∧
    (hexagon_numbers.sum = 30) ∧
    (circle_numbers.sum = 10) :=
sorry

end top_triangle_possibilities_l550_550306


namespace cos_160_eq_neg_09397_l550_550733

theorem cos_160_eq_neg_09397 :
  Real.cos (160 * Real.pi / 180) = -0.9397 :=
by
  sorry

end cos_160_eq_neg_09397_l550_550733


namespace exists_orthogonal_projection_of_equilateral_triangle_l550_550796

theorem exists_orthogonal_projection_of_equilateral_triangle {A B C: Type*} [EuclideanGeometry] (triangle_ABC : Triangle A B C) :
  ∃ (equilateral_triangle Δ), ∃ (projection_plane : Plane), similar (orthogonal_projection projection_plane Δ) triangle_ABC := 
sorry

end exists_orthogonal_projection_of_equilateral_triangle_l550_550796


namespace books_sold_on_Thursday_l550_550521

-- Define the conditions as given in the problem
def initial_stock : ℕ := 1400
def sold_Monday : ℕ := 75
def sold_Tuesday : ℕ := 50
def sold_Wednesday : ℕ := 64
def sold_Friday : ℕ := 135
def percentage_not_sold : ℝ := 71.28571428571429 / 100

-- The main theorem statement that encapsulates the proof problem
theorem books_sold_on_Thursday :
  let not_sold := (percentage_not_sold * initial_stock).toNat in
  let total_sold := initial_stock - not_sold in
  let sold_Until_Wednesday_and_Friday := sold_Monday + sold_Tuesday + sold_Wednesday + sold_Friday in
  let sold_Thursday := total_sold - sold_Until_Wednesday_and_Friday in
  sold_Thursday = 78 :=
by {
  sorry
}

end books_sold_on_Thursday_l550_550521


namespace sum_gcd_lcm_eq_55_l550_550297

-- Define the numbers involved
def a := 45
def b := 75
def c := 40
def d := 10

-- Define the GCD and LCM functions
def gcd (x y : Nat) : Nat := Nat.gcd x y
def lcm (x y : Nat) : Nat := Nat.lcm x y

-- The proof problem statement
theorem sum_gcd_lcm_eq_55 : gcd a b + lcm c d = 55 := by
  sorry

end sum_gcd_lcm_eq_55_l550_550297


namespace total_trash_cans_paid_for_l550_550005

-- Definitions based on conditions
def trash_cans_on_streets : ℕ := 14
def trash_cans_back_of_stores : ℕ := 2 * trash_cans_on_streets

-- Theorem to prove
theorem total_trash_cans_paid_for : trash_cans_on_streets + trash_cans_back_of_stores = 42 := 
by
  -- proof would go here, but we use sorry since proof is not required
  sorry

end total_trash_cans_paid_for_l550_550005


namespace center_of_S_on_line_y_eq_x_l550_550533

noncomputable def greatest_int_le (t : ℝ) := floor t
noncomputable def T (t : ℝ) := abs (t - greatest_int_le t - 0.5)
def S (t : ℝ) := {(x, y) : ℝ × ℝ | (x - T t)^2 + (y - T t)^2 ≤ (T t)^2}

theorem center_of_S_on_line_y_eq_x (t : ℝ) (ht : 0 ≤ t) : 
  (T t, T t) ∈ S t :=
  sorry

end center_of_S_on_line_y_eq_x_l550_550533


namespace claire_distance_l550_550028

noncomputable def distance_from_starting_point (PQ QR : ℝ) (angle : ℝ) : ℝ :=
  let QR_y := QR * Real.sin (angle.toRad)
  let QR_x := QR * Real.cos (angle.toRad)
  let total_westward := PQ + QR_x
  Real.sqrt (total_westward^2 + QR_y^2)

theorem claire_distance : distance_from_starting_point 5 8 60 = 3 * Real.sqrt 43 :=
  by
    sorry

end claire_distance_l550_550028


namespace equivalent_problem_l550_550026

theorem equivalent_problem : ((Real.pi - 3.15)^0 * (-1)^2023 - (-1 / 3)^(-2)) = -10 :=
by
  sorry

end equivalent_problem_l550_550026


namespace equilateral_triangle_area_of_inscribed_triangle_l550_550732

theorem equilateral_triangle_area_of_inscribed_triangle (r : ℝ) (h : 0 < r) :
  let s := 2 * r in
  let height : ℝ := r * sqrt 3 in
  area_of_triangle s height = 64 * sqrt 3 :=
by
  let s := 2 * r
  let height := r * sqrt 3
  have area_of_right_triangle (b h : ℝ) : ℝ := (1 / 2) * b * h
  sorry

end equilateral_triangle_area_of_inscribed_triangle_l550_550732


namespace area_of_one_figure_at_least_l550_550351

theorem area_of_one_figure_at_least (n q : ℕ) (q_pos : 0 < q) (n_pos : 0 < n) 
                                  (cover : ∀ (x : ℝ) (y : ℝ), x ∈ (0, 1) ∧ y ∈ (0, 1) → 
                                       ∃ (figures : set (ℝ × ℝ)), (∀ (f ∈ figures), f ∈ (0,1) × (0,1)) ∧ figures.card ≥ q) : 
  ∃ (A : set (ℝ × ℝ)), (∀ (a ∈ A), a ∈ (0, 1) × (0, 1)) ∧ (measure_theory.measure_space A).volume ≥ q / n :=
sorry

end area_of_one_figure_at_least_l550_550351


namespace seeds_dont_germinate_without_water_l550_550654

-- Define the conditions given in the problem.
def conductor_heats_up_when_conducting (conductor : Type) : Prop :=
  ∀ (electricity : Prop), electricity → (conductor → heats_up conductor)

def three_non_collinear_points_determine_plane (points : Type) : Prop :=
  ∀ (A B C : points), non_collinear A B C → determines_plane A B C

def someone_wins_lottery_consecutive (person : Type) : Prop :=
  ∃ (weeks : ℕ), wins_lottery person weeks ∧ wins_lottery person (weeks + 1)

def seeds_germinate (seed : Type) (water : Prop) : Prop :=
  ∀ (w : water), germinate seed w

-- Proposition that we need to prove.
theorem seeds_dont_germinate_without_water (seed : Type) : ¬ ∃ (germinate : seed) (without_water : Prop), germinate → ¬ without_water := by
  sorry

end seeds_dont_germinate_without_water_l550_550654


namespace prove_problem1_prove_problem2_l550_550511

noncomputable def problem1 (a b : ℝ) (cosB : ℝ) (B : ℝ) (A : ℝ) (R : ℝ) : Prop := 
  a = 5 ∧ b = 6 ∧ cosB = -4/5 ∧ B = acos cosB ∧ R = 5 ∧ A = π/6

noncomputable def problem2 (a b : ℝ) (S : ℝ) (c : ℝ) : Prop := 
  a = 5 ∧ b = 6 ∧ S = 15 * sqrt 7 / 4 ∧ (c = 4 ∨ c = sqrt 106)

theorem prove_problem1 (a b cosB B A R : ℝ) : problem1 a b cosB B A R :=
by
  sorry

theorem prove_problem2 (a b S c : ℝ) : problem2 a b S c :=
by
  sorry

end prove_problem1_prove_problem2_l550_550511


namespace length_of_plot_is_60_l550_550616

noncomputable def plot_length (b : ℝ) : ℝ :=
  b + 20

noncomputable def plot_perimeter (b : ℝ) : ℝ :=
  2 * (plot_length b + b)

noncomputable def plot_cost_eq (b : ℝ) : Prop :=
  26.50 * plot_perimeter b = 5300

theorem length_of_plot_is_60 : ∃ b : ℝ, plot_cost_eq b ∧ plot_length b = 60 :=
sorry

end length_of_plot_is_60_l550_550616


namespace number_of_ordered_pairs_l550_550082

noncomputable def count_valid_ordered_pairs (a b: ℝ) : Prop :=
  ∃ (x y : ℤ), a * (x : ℝ) + b * (y : ℝ) = 2 ∧ x^2 + y^2 = 65

theorem number_of_ordered_pairs : ∃ s : Finset (ℝ × ℝ), s.card = 128 ∧ ∀ (p : ℝ × ℝ), p ∈ s ↔ count_valid_ordered_pairs p.1 p.2 :=
by
  sorry

end number_of_ordered_pairs_l550_550082


namespace tic_tac_toe_lines_8x8x8_l550_550738

/--
In an 8x8x8 tic-tac-toe 3D grid, the number of straight lines with 8 consecutive symbols is 244.
-/
theorem tic_tac_toe_lines_8x8x8 : 
    ∃ lines_count : ℕ, lines_count = 244 :=
begin
  use 244,
  have h : 244 = (10^3 - 8^3) / 2,
  { rw [pow_three, pow_three, ←nat.sub_eq_iff_eq_add, nat.div_eq_of_eq_mul_left, mul_comm],
    norm_num },
  exact h,
end

end tic_tac_toe_lines_8x8x8_l550_550738


namespace big_dogs_count_l550_550987

theorem big_dogs_count (B S : ℕ) (h_ratio : 3 * S = 17 * B) (h_total : B + S = 80) :
  B = 12 :=
by
  sorry

end big_dogs_count_l550_550987


namespace direction_vector_l1_l550_550140

theorem direction_vector_l1
  (m : ℝ)
  (l₁ : ∀ x y : ℝ, (m + 3) * x + 4 * y + 3 * m - 5 = 0)
  (l₂ : ∀ x y : ℝ, 2 * x + (m + 6) * y - 8 = 0)
  (h_perp : ((m + 3) * 2 = -4 * (m + 6)))
  : ∃ v : ℝ × ℝ, v = (-1, -1/2) :=
by
  sorry

end direction_vector_l1_l550_550140


namespace number_of_true_propositions_l550_550822

open Classical

variables (a b m x y : ℝ)
variables (Prop1 Prop2 Prop3 Prop4 : Prop)

def proposition1 : Prop := ∀ (a b m : ℝ), (m ≠ 0 → am^2 < bm^2 → a < b)
def proposition2 : Prop := false  -- Inverse of "The diagonals of a rectangle are equal" is false
def proposition3 : Prop := ∀ (x y : ℝ), (x * y ≠ 0 → (x ≠ 0 ∧ y ≠ 0))
def proposition4 : Prop := false  -- Variance does not change if every number is increased by the same non-zero constant

theorem number_of_true_propositions : ∀ (Prop1 Prop2 Prop3 Prop4 : Prop), 
  (Prop1 = proposition1) ∧ (Prop2 = proposition2) ∧ (Prop3 = proposition3) ∧ (Prop4 = proposition4) → 
  (Prop1 → true) + (Prop2 → true) + (Prop3 → true) + (Prop4 → true) = 2 :=
by 
  intros Prop1 Prop2 Prop3 Prop4 H,
  have h1 := propositional.proposition1,
  have h2 := propositional.proposition2,
  have h3 := propositional.proposition3,
  have h4 := propositional.proposition4,
  sorry

end number_of_true_propositions_l550_550822


namespace locus_equation_and_slope_relation_l550_550814

-- Defining the centers and equations of circles M and N
def circle_M (x y : ℝ) : Prop := (x + real.sqrt 3)^2 + y^2 = 27 / 2
def circle_N (x y : ℝ) : Prop := (x - real.sqrt 3)^2 + y^2 = 3 / 2

-- Given locus is an ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 3 = 1

-- Definitions of slope relationships given P(2,0) and Q(3,t)
def k1 (t x1 y1 : ℝ) : ℝ := (t - y1) / (3 - x1)
def k2 (t : ℝ) : ℝ := t
def k3 (t x2 y2 : ℝ) : ℝ := (t - y2) / (3 - x2)

-- Defining that the slopes form an arithmetic progression
def slopes_form_ap (t x1 y1 x2 y2 : ℝ) : Prop :=
  k1 t x1 y1 + k3 t x2 y2 = 2 * k2 t

theorem locus_equation_and_slope_relation :
  (∀ C : ℝ × ℝ, (∀ r : ℝ, 
    (circle_M C.1 C.2 ↔ (C.1 + real.sqrt 3)^2 + C.2^2 = 27 / 2) ∧ 
    (circle_N C.1 C.2 ↔ (C.1 - real.sqrt 3)^2 + C.2^2 = 3 / 2) ∧ 
    (ellipse C.1 C.2 ↔ C.1^2 / 6 + C.2^2 / 3 = 1))) ∧ 
  (∀ (A B : ℝ × ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ), 
    P = (2, 0) ∧ 
    Q.1 = 3 ∧ 
    slopes_form_ap Q.2 A.1 A.2 B.1 B.2) → 
  true :=
begin
  sorry
end

end locus_equation_and_slope_relation_l550_550814


namespace quadrilateral_trapezoid_or_parallelogram_l550_550950

-- A definition of a quadrilateral using 4 points
structure Quadrilateral (α : Type u) [Euclidean affine_space α] := 
  (A B C D : α)

-- A definition of the property of a quadrilateral being convex
@[class]
def convex_quadrilateral {α : Type u} [Euclidean affine_space α] (q : Quadrilateral α) : Prop := 
  -- some definition of convexity, commonly checking the interior angles sum to 360 degrees
  sorry

-- A definition of similarity of quadrilaterals
def similar_quadrilaterals {α : Type u} [Euclidean affine_space α] 
  (q1 q2 : Quadrilateral α) : Prop :=
  -- some definition of similarity, commonly involving angle and side ratio comparisons
  sorry

-- A definition of a quadrilateral being a trapezoid
def is_trapezoid {α : Type u} [Euclidean affine_space α] (q : Quadrilateral α) : Prop :=
  -- some definition for trapezoid, usually one pair of parallel sides
  sorry

-- A definition of a quadrilateral being a parallelogram
def is_parallelogram {α : Type u} [Euclidean affine_space α] (q : Quadrilateral α) : Prop :=
  -- some definition for parallelogram, usually two pairs of parallel sides
  sorry

-- Main theorem statement
theorem quadrilateral_trapezoid_or_parallelogram {α : Type u} [Euclidean affine_space α] 
  (ABCD : Quadrilateral α) :
  (convex_quadrilateral ABCD) →
  (∃M N : α, 
    (M ∈ line_segment ABCD.A ABCD.B) ∧
    (N ∈ line_segment ABCD.C ABCD.D) ∧ 
    (similar_quadrilaterals 
                    {A := ABCD.A, B := M, C := N, D := ABCD.D}
                    {A := ABCD.B, B := M, C := N, D := ABCD.C})) →
  (is_trapezoid ABCD ∨ is_parallelogram ABCD) :=
begin
  sorry
end

end quadrilateral_trapezoid_or_parallelogram_l550_550950


namespace problem_solution_l550_550165

noncomputable def triangle_abc (A B C a b c : ℝ) : Prop :=
  A = π / 6 ∧
  (1 + real.sqrt 3) * c = 2 * b ∧
  A + B + C = π ∧
  a / real.sin A = b / real.sin B ∧
  b / real.sin B = c / real.sin C ∧
  a = 2 * b * real.cos C

theorem problem_solution :
  ∀ (A B C a b c : ℝ), 
    triangle_abc A B C a b c →
    C = π / 4 ∧ 
    a = real.sqrt 2 ∧ 
    b = 1 + real.sqrt 3 ∧ 
    c = 2 :=
by
  intros
  sorry

end problem_solution_l550_550165


namespace marie_stamps_giveaway_l550_550929

theorem marie_stamps_giveaway :
  let notebooks := 4
  let stamps_per_notebook := 20
  let binders := 2
  let stamps_per_binder := 50
  let fraction_to_keep := 1/4
  let total_stamps := notebooks * stamps_per_notebook + binders * stamps_per_binder
  let stamps_to_keep := fraction_to_keep * total_stamps
  let stamps_to_give_away := total_stamps - stamps_to_keep
  stamps_to_give_away = 135 :=
by
  sorry

end marie_stamps_giveaway_l550_550929


namespace sequence_periodic_l550_550135

noncomputable def sequence (n : ℕ) : ℤ → ℚ
| 0 => 2
| (n + 1) => 1 - (1 / (sequence n))

theorem sequence_periodic (n : ℕ) (h : n = 2016) : sequence n (-1) := by
  sorry

end sequence_periodic_l550_550135


namespace parabola_equation_proof_l550_550828

noncomputable def hyperbola_equation (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 7) = 1

noncomputable def parabola_equation (y x : ℝ) : Prop := y^2 = 2 * 8 * x -- p = 8, so equation is y^2 = 16 * x

theorem parabola_equation_proof :
  let M (x y : ℝ) := (x^2 / 9) - (y^2 / 7) = 1 in
  let F : ℝ × ℝ := (4, 0) in -- F is the right focus of hyperbola
  let N (y x : ℝ) := y^2 = 16 * x in -- found p as 8
  let C : ℝ × ℝ := (-3, 0) in
  let D : ℝ × ℝ := (3, 0) in
  let A : ℝ × ℝ := (4, 8) in
  let B : ℝ × ℝ := (4, -8) in
  let AC : ℝ × ℝ := (-7, -8) in
  let BD : ℝ × ℝ := (-1, 8) in
  hyperbola_equation 0 0 → -- auxiliary hypothesis for simplicity
  parabola_equation 0 0 → -- auxiliary hypothesis for simplicity
  (N = λ y x => y^2 = 16 * x) ∧ -- standard equation of the parabola
  (AC.1 * BD.1 + AC.2 * BD.2 = -57) := -- value of the dot product
by
  sorry

end parabola_equation_proof_l550_550828


namespace sum_T_n_l550_550433

-- Definitions based on conditions
def seq_a (n : ℕ) : ℝ := if n = 0 then 0 else (2 : ℝ) ^ n  -- a_n = 2^n, with a_0 = 0 (since we start from a_1)

def S (n : ℕ) : ℝ := (Finset.range n).sum seq_a  -- Sum of the first n terms

def b (n : ℕ) : ℝ := Real.logb 2 (seq_a n)  -- b_n = log_2(a_n)

def c (n : ℕ) : ℝ := 1 / (b n * b (n + 1))  -- c_n = 1 / (b_n * b_(n+1))

-- The theorem to prove
theorem sum_T_n (n : ℕ) : (Finset.range n).sum c = n / (n + 1) :=
by
  -- we assume the steps to be filled in elsewhere
  sorry

end sum_T_n_l550_550433


namespace average_marks_l550_550627

/-- Given that the total marks in physics, chemistry, and mathematics is 110 more than the marks obtained in physics. -/
theorem average_marks (P C M : ℕ) (h : P + C + M = P + 110) : (C + M) / 2 = 55 :=
by
  -- The proof goes here.
  sorry

end average_marks_l550_550627


namespace triangle_congruent_reciprocal_circumcenter_orthocenter_l550_550202

-- Definitions and Conditions
variables {V : Type _} [InnerProductSpace ℝ V] [EuclideanSpace V] (A B C M A1 B1 C1 : V)

-- Existing conditions
-- M is the orthocenter of triangle ABC
def isOrthocenter_of_triangle : Prop :=
  isOrthocenter ℝ (triangle.mk A B C) M

-- Centers of the circumcircles of triangles BCM, CAM, and ABM
def circumcenter_BCM : Prop :=
  A1 = circumcenter ℝ ∠B C M
def circumcenter_CAM : Prop :=
  B1 = circumcenter ℝ ∠C A M
def circumcenter_ABM : Prop :=
  C1 = circumcenter ℝ ∠A B M

-- Proof Statements
theorem triangle_congruent :
  isOrthocenter_of_triangle ℝ A B C M →
  circumcenter_BCM ℝ A B C M A1 →
  circumcenter_CAM ℝ A B C M B1 →
  circumcenter_ABM ℝ A B C M C1 →
  triangle.congruent ℝ (triangle.mk A B C) (triangle.mk A1 B1 C1)
:= by sorry

theorem reciprocal_circumcenter_orthocenter :
  isOrthocenter_of_triangle ℝ A B C M →
  circumcenter_BCM ℝ A B C M A1 →
  circumcenter_CAM ℝ A B C M B1 →
  circumcenter_ABM ℝ A B C M C1 →
  isCircumcenter ℝ (triangle.mk A1 B1 C1) M ∧
  isOrthocenter ℝ (triangle.mk A1 B1 C1) (circumcenter ℝ ∠A B C)
:= by sorry

end triangle_congruent_reciprocal_circumcenter_orthocenter_l550_550202


namespace pyramid_top_row_missing_number_l550_550381

theorem pyramid_top_row_missing_number (a b c d e f g : ℕ)
  (h₁ : b * c = 720)
  (h₂ : a * b = 240)
  (h₃ : c * d = 1440)
  (h₄ : c = 6)
  : a = 120 :=
by
  sorry

end pyramid_top_row_missing_number_l550_550381


namespace select_test_point_l550_550319

theorem select_test_point (x1 x2 : ℝ) (h1 : x1 = 2 + 0.618 * (4 - 2)) (h2 : x2 = 2 + 4 - x1) :
  (x1 > x2 → x3 = 4 - 0.618 * (4 - x1)) ∨ (x1 < x2 → x3 = 6 - x3) :=
  sorry

end select_test_point_l550_550319


namespace sol_earns_amount_l550_550962

theorem sol_earns_amount (candy_bars_first_day : ℕ)
                         (additional_candy_bars_per_day : ℕ)
                         (sell_days_per_week : ℕ)
                         (cost_per_candy_bar_cents : ℤ) :
                         candy_bars_first_day = 10 →
                         additional_candy_bars_per_day = 4 →
                         sell_days_per_week = 6 →
                         cost_per_candy_bar_cents = 10 →
                         (∑ i in finset.range sell_days_per_week, (candy_bars_first_day + additional_candy_bars_per_day * i).to_int) * cost_per_candy_bar_cents / 100 = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end sol_earns_amount_l550_550962


namespace antonio_code_l550_550013

theorem antonio_code : ∃ (A B C : ℕ),
  A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
  B > A ∧ A < C ∧
  (10 * B + A) + (10 * B + A) = 100 * 2 + 10 * 4 + 2 :=
by
  use 1, 2, 4
  split
  . exact ne_of_lt (Nat.lt_trans one_lt_two (by norm_num))
  split
  . exact ne_of_lt (by norm_num)
  split
  . exact ne_of_gt (by norm_num)
  split
  . exact one_lt_two
  split
  . exact Nat.lt_trans (by norm_num) (by norm_num)
  norm_num


end antonio_code_l550_550013


namespace probability_second_ball_new_given_first_new_l550_550875

-- Define the conditions
def totalBalls : ℕ := 10
def newBalls : ℕ := 6
def oldBalls : ℕ := 4

-- Define the events
def firstBallIsNew : Prop := true  -- Given condition
def secondBallIsNew : Prop := true  -- Event to prove

-- Define the total probability of drawing a new ball on the first draw
def P_A := (newBalls : ℚ) / (totalBalls : ℚ)

-- Define the joint probability of drawing two new balls
def P_AB := (newBalls : ℚ) / (totalBalls : ℚ) * (newBalls - 1 : ℚ) / (totalBalls - 1 : ℚ)

-- Define the conditional probability of drawing a new ball on the second draw given the first was new
def P_B_given_A := P_AB / P_A

-- The theorem to prove
theorem probability_second_ball_new_given_first_new :
  P_B_given_A = 5 / 9 :=
by
  -- state the probabilities explicitly for clarity
  have h_P_A : P_A = 3 / 5 := by sorry
  have h_P_AB : P_AB = 1 / 3 := by sorry

  -- compute the conditional probability
  rw [P_B_given_A, h_P_A, h_P_AB]
  sorry

end probability_second_ball_new_given_first_new_l550_550875


namespace triangle_median_unique_m_l550_550384

theorem triangle_median_unique_m :
  ∃ (a b m : ℝ), a > 0 ∧ b > 0 ∧ (m = -1/4) ∧ 
    (∀ x, y = 4*x + 3 → y = b/2) ∧
    (∀ x, y = m*x + 5 → y = b/2) → (m = -1/4) :=
begin
  sorry
end

end triangle_median_unique_m_l550_550384


namespace polynomial_with_complex_root_l550_550070

theorem polynomial_with_complex_root :
  ∃ P : Polynomial ℝ, P.Monic ∧ degree P = 2 ∧
  P.coeff 0 = 17 ∧ P.coeff 1 = 6 ∧ P.coeff 2 = 1 ∧
  (P.eval (-3 - Complex.i * Real.sqrt 8) = 0) :=
sorry

end polynomial_with_complex_root_l550_550070


namespace max_m_value_l550_550856

noncomputable def f (x m : ℝ) : ℝ := x * Real.log x + x^2 - m * x + Real.exp (2 - x)

theorem max_m_value (m : ℝ) :
  (∀ x : ℝ, 0 < x → f x m ≥ 0) → m ≤ 3 :=
sorry

end max_m_value_l550_550856


namespace part1_part2_l550_550089

variables {x y : ℝ}

def A := x^2 + x * y + 2 * y - 2
def B := 2 * x^2 - 2 * x * y + x - 1

theorem part1 : 2 * A - B = 4 * x * y + 4 * y - x - 3 := by
  calc
    2 * A - B
        = 2 * (x^2 + x * y + 2 * y - 2) - (2 * x^2 - 2 * x * y + x - 1) : by rfl
    ... = 2 * x^2 + 2 * x * y + 4 * y - 4 - (2 * x^2 - 2 * x * y + x - 1)  : by ring
    ... = (2 * x^2 + 2 * x * y + 4 * y - 4) - 2 * x^2 + 2 * x * y - x + 1 : by ring
    ... = (2 * x^2 - 2 * x^2) + (2 * x * y + 2 * x * y) + 4 * y - 4 - x + 1 : by ring
    ... = 4 * x * y + 4 * y - x - 3 : by ring

theorem part2 (h : (4 * x * y + 4 * y - x - 3) = (4 * y - 1) * x + (4 * y - 3)) : y = 1 / 4 := by
  have h_coeff_x : (4 * y - 1) = 0 := by simpa using h
  have h_y : 4 * y - 1 = 0 := h_coeff_x
  have h_final : y = (1 : ℝ) / (4 : ℝ) := by linarith
  exact h_final

end part1_part2_l550_550089


namespace min_circles_to_cover_points_l550_550093

-- Define the conditions and question as a Lean theorem
theorem min_circles_to_cover_points :
  (∀ (P: set (ℝ × ℝ)), P.card = 2009 → 
    (∀ (Q: finset (ℝ × ℝ)), Q.card = 17 → 
      ∃ (R: finset (ℝ × ℝ)), R.card = 11 ∧ (∃ c: ℝ × ℝ, ∀ p ∈ R, dist p c ≤ 1/2))) 
  → ∃ (C : finset (ℝ × ℝ × ℝ)), C.card = 7 ∧ 
      ∀ p ∈ P, ∃ c ∈ C, dist p (c.1, c.2) ≤ 1 :=
begin
  sorry
end

end min_circles_to_cover_points_l550_550093


namespace function_increasing_interval_l550_550858

noncomputable def f (ω x : ℝ) := sqrt 3 * sin (ω * x) + cos (ω * x)

theorem function_increasing_interval : 
  (x : ℝ) (h_symmetry : x = π / 6) (ω := 2) 
  (∀ (x1 x2 : ℝ), 0 < x1 ∧ x1 < x2 ∧ x2 < π / 6 → f ω x1 < f ω x2) :=
sorry

end function_increasing_interval_l550_550858


namespace rachel_plants_lamps_l550_550589

-- Define the conditions as types
def plants : Type := { fern1 : Prop // true } × { fern2 : Prop // true } × { cactus : Prop // true }
def lamps : Type := { yellow1 : Prop // true } × { yellow2 : Prop // true } × { blue1 : Prop // true } × { blue2 : Prop // true }

-- A function that counts the distribution of plants under lamps
noncomputable def count_ways (p : plants) (l : lamps) : ℕ :=
  -- Here we should define the function that counts the number of configurations, 
  -- but since we are only defining the problem here we'll skip this part.
  sorry

-- The statement to prove
theorem rachel_plants_lamps :
  ∀ (p : plants) (l : lamps), count_ways p l = 14 :=
by
  sorry

end rachel_plants_lamps_l550_550589


namespace categorize_numbers_l550_550758

def is_fraction (x : ℚ) : Prop := true -- in this context, all rational numbers are considered fractions

def is_natural (x : ℕ) : Prop := true -- in this context, natural numbers include non-negative integers

def is_non_positive_integer (x : ℤ) : Prop := x ≤ 0

def is_non_negative_rational (x : ℚ) : Prop := x ≥ 0

theorem categorize_numbers :
  { x : ℚ // is_fraction x } = {-1/9, 2/15, -532/100, 2.3, 4/5} ∧
  { x : ℕ // is_natural x } = {15, 0, 5} ∧
  { x : ℤ // is_non_positive_integer x } = {-5, 0} ∧
  { x : ℚ // is_non_negative_rational x } = {15, 2/15, 0, 2.3, 4/5, 5} :=
by sorry

end categorize_numbers_l550_550758


namespace students_not_good_at_either_l550_550876

theorem students_not_good_at_either (total good_at_english good_at_chinese both_good : ℕ) 
(h₁ : total = 45) 
(h₂ : good_at_english = 35) 
(h₃ : good_at_chinese = 31) 
(h₄ : both_good = 24) : total - (good_at_english + good_at_chinese - both_good) = 3 :=
by sorry

end students_not_good_at_either_l550_550876


namespace minimal_k_exists_l550_550700

theorem minimal_k_exists :
  ∃ (k : ℕ), k = 1 ∧ ∀ d : ℕ, d >= 10000 ∧ d < 100000 ∧ (List.ofDigits (Nat.digits 10 d)).Sorted (<) → 
  ∃ n ∈ {13579}, ∃ i < 5, Nat.digits 10 d[i] = Nat.digits 10 n[i] :=
by
  sorry

end minimal_k_exists_l550_550700


namespace expression_evaluation_l550_550373

theorem expression_evaluation : 
  - (3 ^ 2) - (1 / 9)⁻¹ + (1 - 4) ^ 0 * | -3 | = -15 := 
by 
  sorry

end expression_evaluation_l550_550373


namespace fold_area_is_5_031_l550_550338

noncomputable def width (w : ℝ) : Prop := 3 * w^2 = 27

noncomputable def rectangle_folded_overlap_area (w : ℝ) : ℝ :=
  let length := 3 * w
  let height := w
  let base := real.sqrt ((1.5)^2 + (3)^2)
  (1 / 2) * base * height

theorem fold_area_is_5_031 (w : ℝ) (hw : width w) :
  rectangle_folded_overlap_area w = 5.031 :=
by sorry

end fold_area_is_5_031_l550_550338


namespace fido_ab_product_l550_550393

theorem fido_ab_product :
  ∀ (a b : ℕ), (∃ (s : ℝ) (r : ℝ), r = s/2 ∧ (π * r^2 / s^2 = (Real.sqrt a / b) * π)) → a * b = 0 :=
by
  intros a b h,
  sorry

end fido_ab_product_l550_550393


namespace sin_monotonic_on_0_max_m_l550_550867

open Real

noncomputable def max_m_f_monotonic := (2 / 3 : ℝ)

theorem sin_monotonic_on_0_max_m :
  ∀ m : ℝ, (∀ x ∈ (Icc 0 m), deriv (λ x : ℝ, sin (π * x - π / 6)) x > 0) ↔ m ≤ max_m_f_monotonic :=
by
  sorry

end sin_monotonic_on_0_max_m_l550_550867


namespace problem_statement_l550_550919

section Problem

variable {X : ℕ → ℝ} -- Sequence of random variables
variable {S : ℕ → ℝ} -- Partial sums
variable (E : ℝ → ℝ) -- Expectation operator

-- Conditions
axiom identically_distributed : ∀ n, E (X 1) = E (X n)
axiom independent : ∀ n m, n ≠ m → E (X n * X m) = E (X n) * E (X m)
axiom bounded : ∃ B, ∀ n, ∥X n∥ ≤ B
axiom mean_zero : E (X 1) = 0
axiom partial_sum : ∀ n, S n = ∑ i in finset.range n, X i 

-- Statement to Prove
theorem problem_statement (p : ℝ) (hp : p > 0) : 
    ∃ C, ∀ n, E (∥S n∥^p) ≤ C * n^(p/2) :=
sorry

end Problem

end problem_statement_l550_550919


namespace finding_b_for_infinite_solutions_l550_550087

theorem finding_b_for_infinite_solutions :
  ∀ b : ℝ, (∀ x : ℝ, 5 * (4 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 :=
by
  sorry

end finding_b_for_infinite_solutions_l550_550087


namespace total_area_approx_l550_550166

noncomputable def total_area_of_molds : ℝ :=
  let r : ℝ := 3 -- radius of the circular mold
  let A_circular : ℝ := Real.pi * r^2 -- area of the circular mold
  let length : ℝ := 8 -- length of the rectangular mold
  let width : ℝ := 4 -- width of the rectangular mold
  let A_rectangular : ℝ := length * width -- area of the rectangular mold
  A_circular + A_rectangular

theorem total_area_approx : total_area_of_molds ≈ 60.27433 :=
by
  sorry

end total_area_approx_l550_550166


namespace no_such_reals_exist_l550_550049

-- Define the existence of distinct real numbers such that the given condition holds
theorem no_such_reals_exist :
  ¬ ∃ x y z : ℝ, (x ≠ y) ∧ (y ≠ z) ∧ (z ≠ x) ∧ 
  (1 / (x^2 - y^2) + 1 / (y^2 - z^2) + 1 / (z^2 - x^2) = 0) :=
by
  -- Placeholder for proof
  sorry

end no_such_reals_exist_l550_550049


namespace min_ab_min_inv_a_plus_2_inv_b_max_sqrt_2a_plus_sqrt_b_not_max_a_plus_1_times_b_plus_1_l550_550418

-- Condition definitions
variable {a b : ℝ}
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1)

-- Minimum value of ab is 1/8
theorem min_ab (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : ∃ y, y = (a * b) ∧ y = 1 / 8 := by
  sorry

-- Minimum value of 1/a + 2/b is 8
theorem min_inv_a_plus_2_inv_b (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : ∃ y, y = (1 / a + 2 / b) ∧ y = 8 := by
  sorry

-- Maximum value of sqrt(2a) + sqrt(b) is sqrt(2)
theorem max_sqrt_2a_plus_sqrt_b (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : ∃ y, y = (Real.sqrt (2 * a) + Real.sqrt b) ∧ y = Real.sqrt 2 := by
  sorry

-- Maximum value of (a+1)(b+1) is not 2
theorem not_max_a_plus_1_times_b_plus_1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : ∃ y, y = ((a + 1) * (b + 1)) ∧ y ≠ 2 := by
  sorry


end min_ab_min_inv_a_plus_2_inv_b_max_sqrt_2a_plus_sqrt_b_not_max_a_plus_1_times_b_plus_1_l550_550418


namespace distinct_factors_count_l550_550045

-- Given conditions
def eight_squared : ℕ := 8^2
def nine_cubed : ℕ := 9^3
def seven_fifth : ℕ := 7^5
def number : ℕ := eight_squared * nine_cubed * seven_fifth

-- Proving the number of natural-number factors of the given number
theorem distinct_factors_count : 
  (number.factors.count 1 = 294) := sorry

end distinct_factors_count_l550_550045


namespace orthogonal_unit_vectors_condition_l550_550817

theorem orthogonal_unit_vectors_condition (a b : ℝ^3) (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) :
  (a ⬝ b = 0) ↔ (∥2 • a - b∥ = ∥a + 2 • b∥) :=
sorry

end orthogonal_unit_vectors_condition_l550_550817


namespace sequence_sum_property_l550_550267

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, n > 0 → a n + a (n + 1) = 1

def sum_seq (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  S 0 = 0 ∧ ∀ n : ℕ, S (n + 1) = S n + a (n + 1)

theorem sequence_sum_property {a : ℕ → ℤ} {S : ℕ → ℤ}
  (hseq : sequence a) (hsum : sum_seq a S) :
  S 2017 - 2 * S 2018 + S 2019 = 3 :=
sorry

end sequence_sum_property_l550_550267


namespace d_lt_μ_lt_M_l550_550567

noncomputable def num_days := 366
noncomputable def num_days_each := 12
noncomputable def num_days_31 := 10
noncomputable def values := [1, 2, ..., 31]

def occurences (x : ℕ) : ℕ :=
  if x = 31 then num_days_31 else num_days_each

noncomputable def dataset := 
  (values.map (λ x, (x, occurences x))).sum (λ (x, oc), (repeat x oc))

noncomputable def median_M : ℚ := 16
noncomputable def mean_μ : ℚ := 5890 / 366
noncomputable def median_d : ℚ := 15.5

theorem d_lt_μ_lt_M : median_d < mean_μ ∧ mean_μ < median_M := by
  sorry

end d_lt_μ_lt_M_l550_550567


namespace problem_l550_550481

theorem problem (a b c d : ℝ)
  (h₀ : a = -0.3^2)
  (h₁ : b = (-3)^(-2))
  (h₂ : c = (-1/3)^(-2))
  (h₃ : d = (-1/3)^0) :
  a < b ∧ b < d ∧ d < c :=
by
  rw [h₀, h₁, h₂, h₃]
  norm_num
  exact ⟨by norm_num, by norm_num, by norm_num⟩

end problem_l550_550481


namespace distance_AB_l550_550506

-- Define the polar coordinates
def A := (2, Real.pi / 6)
def B := (2, -Real.pi / 6)

-- Calculate the Cartesian coordinates without actual computation (conceptually)
noncomputable def cartesian_coords (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

-- Define Cartesian coordinates for points A and B
noncomputable def A_cartesian := cartesian_coords A.1 A.2
noncomputable def B_cartesian := cartesian_coords B.1 B.2

-- Define the Euclidean distance between two Cartesian points
noncomputable def euclidean_distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- The theorem statement
theorem distance_AB :
  euclidean_distance A_cartesian B_cartesian = 2 := by
    sorry

end distance_AB_l550_550506


namespace wax_already_have_is_28_l550_550937

-- Define conditions as variables and constants
variables (total_wax_needed still_wax_needed : ℕ)
constant hc1 : total_wax_needed = 288
constant hc2 : still_wax_needed = 260

-- Define the question and the condition to prove
def grams_wax_already_have : ℕ := total_wax_needed - still_wax_needed

-- State the theorem to be proved
theorem wax_already_have_is_28 : grams_wax_already_have total_wax_needed still_wax_needed = 28 :=
by 
  rw [hc1, hc2]
  rfl

end wax_already_have_is_28_l550_550937


namespace second_win_amount_l550_550334

theorem second_win_amount :
  ∃ W : ℝ, 
    let x₀ := 48.00000000000001 in 
    let x₁ := x₀ - (1/3) * x₀ in 
    let x₂ := x₁ + 10 in 
    let x₃ := x₂ - (1/3) * x₂ in 
    x₃ + W = x₀ ∧ W = 20 := by
  sorry

end second_win_amount_l550_550334


namespace smallest_integer_n_l550_550552

theorem smallest_integer_n (m n : ℕ) (r : ℝ) :
  (m = (n + r)^3) ∧ (0 < r) ∧ (r < 1 / 2000) ∧ (m = n^3 + 3 * n^2 * r + 3 * n * r^2 + r^3) →
  n = 26 :=
by 
  sorry

end smallest_integer_n_l550_550552


namespace marie_stamps_l550_550931

variable (n_notebooks : ℕ) (stamps_per_notebook : ℕ) (n_binders : ℕ) (stamps_per_binder : ℕ) (fraction_keep : ℚ)

theorem marie_stamps :
  n_notebooks = 4 →
  stamps_per_notebook = 20 →
  n_binders = 2 →
  stamps_per_binder = 50 →
  fraction_keep = 1/4 →
  let total_stamps := n_notebooks * stamps_per_notebook + n_binders * stamps_per_binder in
  let stamps_keep := total_stamps * fraction_keep in
  let stamps_give_away := total_stamps - stamps_keep in
  stamps_give_away = 135 :=
by
  intros h1 h2 h3 h4 h5
  let total_stamps := n_notebooks * stamps_per_notebook + n_binders * stamps_per_binder
  let stamps_keep := total_stamps * fraction_keep
  let stamps_give_away := total_stamps - stamps_keep
  have h_total_stamps : total_stamps = 180 := by simp [h1, h2, h3, h4, total_stamps]
  have h_stamps_keep : stamps_keep = 45 := by simp [h_total_stamps, h5, stamps_keep]
  have h_stamps_give_away : stamps_give_away = 135 := by simp [h_total_stamps, h_stamps_keep, stamps_give_away]
  exact h_stamps_give_away

end marie_stamps_l550_550931


namespace max_sum_of_multiplication_table_l550_550985

theorem max_sum_of_multiplication_table : ∀ a b c d e f : ℕ, 
  {a, b, c, d, e, f} = {1, 2, 4, 5, 7, 8} → 
  a ≠ b → a ≠ c → a ≠ d → a ≠ e → a ≠ f →
  b ≠ c → b ≠ d → b ≠ e → b ≠ f →
  c ≠ d → c ≠ e → c ≠ f →
  d ≠ e → d ≠ f → 
  e ≠ f →
  ((a + b + c) * (d + e + f)) ≤ 182 :=
by
  intros a b c d e f hset hneq1 hneq2 hneq3 hneq4 hneq5 hneq6 hneq7 hneq8 hneq9 hneq10 hneq11 hneq12 hneq13 hneq14
  sorry

end max_sum_of_multiplication_table_l550_550985


namespace number_of_sequences_3_classtimes_8_students_l550_550017

/-- 
  Given 8 students and 3 class meetings in a week, with the restriction that 
  a student cannot be chosen more than once in the same week, prove that 
  the number of different sequences of students (quote sharers) is 336.
-/
def number_of_sequences (n : ℕ) (k : ℕ) : ℕ := (list.range n).permutations.length

theorem number_of_sequences_3_classtimes_8_students :
  number_of_sequences 3 8 = 336 :=
by 
  unfold number_of_sequences 
  -- Calculation of 8 * 7 * 6
  sorry

end number_of_sequences_3_classtimes_8_students_l550_550017


namespace ratio_of_length_to_width_l550_550615

variable (L W : ℕ)
variable (H1 : W = 50)
variable (H2 : 2 * L + 2 * W = 240)

theorem ratio_of_length_to_width : L / W = 7 / 5 := 
by sorry

end ratio_of_length_to_width_l550_550615


namespace find_b_y_axis_find_a_angle_bisector_find_a_b_parallel_y_axis_l550_550474

-- Part 1: Finding the value of b
theorem find_b_y_axis (b : ℝ) (yB : ℝ) (H : B = (0, yB)) : b = -1 :=
by sorry

-- Part 2: Finding the value of a
theorem find_a_angle_bisector (a : ℝ) (xA : ℝ) (H1 : xA = 3) (H2 : A = (xA, a - 1)) (H3 : A lies on the angle bisector in the first and third quadrants) : a = 4 :=
by sorry

-- Part 3: Finding the values of a and b when AB is parallel to the y-axis and AB = 5
theorem find_a_b_parallel_y_axis (a : ℝ) (b : ℝ) (xA : ℝ) (yA : ℝ) (yB : ℝ) (H1 : xA = 3) (H2 : A = (xA, yA)) (H3 : B = (xA, yB)) (H4 : AB = 5) : (a = 4 ∧ b = 2) ∨ (a = -6 ∧ b = 2) :=
by sorry

end find_b_y_axis_find_a_angle_bisector_find_a_b_parallel_y_axis_l550_550474


namespace cubic_coefficient_is_two_l550_550251

noncomputable def P (a b : ℝ) : ℝ :=
  -2 * a^3 * b + 2 * a^2 * b + 7 * a * b - 9

theorem cubic_coefficient_is_two :
  (λ (a b : ℝ), P a b = -2 * a^3 * b + 2 * a^2 * b + 7 * a * b - 9) →
  2 * a^2 * b ∈ (λ (a b : ℝ), -2 * a^3 * b + 2 * a^2 * b + 7 * a * b - 9) →
  (λ (a b : ℝ), has_degree a b 3 (2 * a^2 * b)) →
  coefficient_of_term (2 * a^2 * b) = 2 := sorry

end cubic_coefficient_is_two_l550_550251


namespace interval_for_non_monotonicity_l550_550158

theorem interval_for_non_monotonicity (k : ℝ) (f : ℝ → ℝ) (h : f = λ x, x^2 + x - log x - 2) :
  (∀ (a b : ℝ), (a < b) → (a ∈ set.Ioo (2*k-1) (k+2)) → (b ∈ set.Ioo (2*k-1) (k+2)) → f b = f a → false) ↔ 
  (1/2) ≤ k ∧ k < 3/4 :=
sorry

end interval_for_non_monotonicity_l550_550158


namespace trig_identity_l550_550048

theorem trig_identity (θ : ℝ) :
  (Real.sin θ + Real.csc θ)^3 + (Real.cos θ + Real.sec θ)^3
  = 10 + 3 * (Real.sin θ + Real.cos θ + Real.csc θ + Real.sec θ + Real.cot θ + Real.tan θ + Real.cot θ^3 + Real.tan θ^3) +
    Real.sin θ^2 + Real.cos θ^2 := 
    sorry

end trig_identity_l550_550048


namespace min_positive_integer_t_l550_550131

noncomputable def y (x : ℝ) : ℝ := Real.sin (π * x / 3)

theorem min_positive_integer_t (t : ℕ) (h1 : 0 ≤ t) (h2 : ∃x1 x2 ∈ (0 : ℝ)..(t : ℝ), y x1 = 1 ∧ y x2 = 1 ∧ x1 ≠ x2) : t ≥ 8 :=
by
  sorry

end min_positive_integer_t_l550_550131


namespace collinear_A_B_D_l550_550110

variables {V : Type*} [add_comm_group V] [module ℝ V]
variables (a b : V)

def AB : V := 3 • a + 6 • b
def BC : V := -10 • a + 12 • b
def CD : V := 14 • a - 4 • b

theorem collinear_A_B_D : collinear ℝ ({(0 : V), AB, AB + BC + CD}.to_finset) :=
by
  sorry

end collinear_A_B_D_l550_550110


namespace chessboard_knight_tour_exists_Y_l550_550436

theorem chessboard_knight_tour_exists_Y (X : ℕ) (knight_tours : ℕ → list (ℕ × ℕ))
  (h_chessboard : ∀ i, 0 ≤ i ∧ i < 64) -- Representing an 8x8 chessboard
  (knight_moves : ∀ k, ∀ p q : (ℕ × ℕ), p ∈ knight_tours k → q ∈ knight_tours k → 
    (abs (p.fst - q.fst) = 1 ∧ abs (p.snd - q.snd) = 2) ∨ (abs (p.fst - q.fst) = 2 ∧ abs (p.snd - q.snd) = 1))
  (all_start_end_at_X : ∀ k, X = (knight_tours k).head ∧ X = (knight_tours k).last)
  (all_squares_visited : ∀ s, ∃ k, ∃ p, p ∈ knight_tours k ∧ p = s) :
  ∃ Y, Y ≠ X ∧ ∃ k1 k2, (k1 ≠ k2 ∨ k1 = k2) ∧ (Y ∈ knight_tours k1) ∧ (Y ∈ knight_tours k2) := 
by sorry

end chessboard_knight_tour_exists_Y_l550_550436


namespace gcd_eq_gcd_of_division_l550_550779

theorem gcd_eq_gcd_of_division (a b q r : ℕ) (h1 : a = b * q + r) (h2 : 0 < r) (h3 : r < b) (h4 : a > b) : 
  Nat.gcd a b = Nat.gcd b r :=
by
  sorry

end gcd_eq_gcd_of_division_l550_550779


namespace cone_volume_l550_550788

theorem cone_volume (r: ℝ) (θ: ℝ) (h: ℝ) (v: ℝ) (h_radius : r = 1)
                    (h_angle : θ = (4 * π) / 3)
                    (h_height : h = sqrt(1 - (2 / 3)^2))
                    (h_volume : v = (1 / 3) * (2 / 3) ^ 2 * π * (sqrt(5) / 3)): 
    v = (4 * sqrt(5) / 81) * π :=
by
  sorry

end cone_volume_l550_550788


namespace marie_stamps_giveaway_l550_550930

theorem marie_stamps_giveaway :
  let notebooks := 4
  let stamps_per_notebook := 20
  let binders := 2
  let stamps_per_binder := 50
  let fraction_to_keep := 1/4
  let total_stamps := notebooks * stamps_per_notebook + binders * stamps_per_binder
  let stamps_to_keep := fraction_to_keep * total_stamps
  let stamps_to_give_away := total_stamps - stamps_to_keep
  stamps_to_give_away = 135 :=
by
  sorry

end marie_stamps_giveaway_l550_550930


namespace fraction_students_like_swimming_but_say_dislike_l550_550368

theorem fraction_students_like_swimming_but_say_dislike (
  total_students : ℕ,
  likes_swimming_percent : ℚ,
  dislikes_swimming_percent : ℚ,
  likes_say_like_percent : ℚ,
  likes_say_dislike_percent : ℚ,
  dislikes_say_dislike_percent : ℚ,
  dislikes_say_like_percent : ℚ
) :
  likes_swimming_percent = 0.70 →
  dislikes_swimming_percent = 0.30 →
  likes_say_like_percent = 0.75 →
  likes_say_dislike_percent = 0.25 →
  dislikes_say_dislike_percent = 0.85 →
  dislikes_say_like_percent = 0.15 →
  let likes : ℚ := likes_swimming_percent * total_students,
      dislikes : ℚ := dislikes_swimming_percent * total_students,
      likes_dislike : ℚ := likes_say_dislike_percent * likes,
      dislikes_dislike : ℚ := dislikes_say_dislike_percent * dislikes,
      total_say_dislike : ℚ := likes_dislike + dislikes_dislike,
      fraction_likes_but_say_dislike : ℚ := likes_dislike / total_say_dislike
  in fraction_likes_but_say_dislike ≈ 0.407 := by
  intros;
  sorry

end fraction_students_like_swimming_but_say_dislike_l550_550368


namespace jar_clay_pot_marble_ratio_l550_550327

theorem jar_clay_pot_marble_ratio :
  ∃ (x : ℕ), 
   let jars := 16,
       pots := 16 / 2,
       marbles_per_jar := 5,
       total_marbles := 200,
       marbles_in_jars := jars * marbles_per_jar,
       marbles_in_pots := total_marbles - marbles_in_jars,
       marbles_per_pot := marbles_in_pots / pots
   in (marbles_per_pot / marbles_per_jar) = 3 :=
sorry

end jar_clay_pot_marble_ratio_l550_550327


namespace isosceles_triangles_with_perimeter_27_l550_550844

theorem isosceles_triangles_with_perimeter_27 :
  ∃ n : ℕ, n = 6 ∧ ∀ a b c : ℕ, (a = b ∧ a > 0 ∧ 2 * a + c = 27 ∧ c mod 2 = 1) → n = 6 :=
by
  sorry

end isosceles_triangles_with_perimeter_27_l550_550844


namespace probability_at_origin_after_6_moves_maximal_probability_at_point_5_l550_550337

open_locale big_operators

-- Define a function to compute probability at a specific point
def probability_at_point (n k : ℕ) : ℚ := (nat.choose n k) * (1 / 2) ^ n

-- Define a statement for the probability of being at the origin after 6 moves
theorem probability_at_origin_after_6_moves :
  probability_at_point 6 3 = 5 / 16 :=
sorry

-- Define a function to compute the probability of being at the point 5 after n moves
def probability_at_5_for_n_moves (n : ℕ) : ℚ := ∑ i in finset.range (n + 1), if 2 * i = n - 5 then probability_at_point n i else 0

-- Define a statement for the maximum probability of being at point 5
theorem maximal_probability_at_point_5 :
  probability_at_5_for_n_moves 23 = probability_at_5_for_n_moves 25 :=
sorry

end probability_at_origin_after_6_moves_maximal_probability_at_point_5_l550_550337


namespace length_CD_l550_550624

/--
  Given a region consisting of all points in three-dimensional space within 4 units
  of a line segment CD, and given that the volume of this region is 384π,
  proves that the length of the line segment CD is 18.
-/
theorem length_CD (r : ℝ) (V : ℝ) (l : ℝ) (h : ℝ) :
  r = 4 →
  V = 384 * Real.pi →
  V = (2 * (4 / 3) * Real.pi * r^3 / 2) + (Real.pi * r^2 * h) →
  h = 18 →
  l = h →
  l = 18 :=
by 
  intros hr hV hVol hh hl
  rw [hr] at hVol
  have : V = (256 / 3) * Real.pi + 16 * Real.pi * 18 := by sorry
  rw [this] at hV
  exact hl

end length_CD_l550_550624


namespace geometric_series_sum_correct_l550_550034

-- Given conditions
def a : ℤ := 3
def r : ℤ := -2
def n : ℤ := 10

-- Sum of the geometric series formula
def geometric_series_sum (a r n : ℤ) : ℤ := 
  a * (r^n - 1) / (r - 1)

-- Goal: Prove that the sum of the series is -1023
theorem geometric_series_sum_correct : 
  geometric_series_sum a r n = -1023 := 
by
  sorry

end geometric_series_sum_correct_l550_550034


namespace charlie_acorns_l550_550407

theorem charlie_acorns (x y : ℕ) (hc hs : ℕ)
  (h5 : x = 5 * hc)
  (h7 : y = 7 * hs)
  (total : x + y = 145)
  (holes : hs = hc - 3) :
  x = 70 :=
by
  sorry

end charlie_acorns_l550_550407


namespace gcd_of_ten_digit_repeated_l550_550003

theorem gcd_of_ten_digit_repeated :
  ∃ k, (∀ n : ℕ, 10000 ≤ n ∧ n < 100000 → Nat.gcd (100001 * n) k = k) ∧ k = 100001 :=
by {
  sorry
}

end gcd_of_ten_digit_repeated_l550_550003


namespace number_of_ways_to_sum_525_as_consecutive_odd_integers_l550_550500

/-- 
The mathematical statement to prove the number of ways 525 can be expressed 
as the sum of three or more consecutive odd integers.
-/
def is_consecutive_odd_sum (n k : ℕ) : Prop :=
  n ≥ 3 ∧ 525 = n * (k + n - 1) ∧ ∀ i < n, (k + 2 * i) % 2 = 1

theorem number_of_ways_to_sum_525_as_consecutive_odd_integers :
  {m : ℕ // (∃ (n k : ℕ), is_consecutive_odd_sum n k) → m = 2 } :=
begin
  sorry
end

end number_of_ways_to_sum_525_as_consecutive_odd_integers_l550_550500


namespace problem1_problem2_l550_550464

theorem problem1 (f g : ℝ → ℝ) (h : ℝ → ℝ) (m : ℤ) (x : ℝ) 
  (h_f : ∀ x, f x = Real.exp x)
  (h_g : ∀ x, g x = (1 : ℝ) * x + 1)
  (h_h : ∀ x, h x = f x - g x)
  (tangent_cond : ∃ t, f t = g t ∧ (∀ x, x = t → Real.exp t = 1 * t + 1)) :
  (∀ m, (∀ x > 0, (m - x) * h' x < x + 1 ) → m ≤ 2) :=
sorry

theorem problem2 (x : ℝ)
  (h_phi : ∀ x > 0, (m - x) * (Real.exp x - 1) < x + 1)
  (h_ineq : ∀ x > 0, (m - x) * (Real.exp x - 1) < x + 1) :
  m ≤ 2 :=
sorry

end problem1_problem2_l550_550464


namespace part1_part2_l550_550825

-- Define the given function f(x)
def f (x : ℝ) := sin (2 * x + π / 3) + (sqrt 3 / 3) * sin x ^ 2 - (sqrt 3 / 3) * cos x ^ 2

-- Smallest positive period of f(x) is π
def period π : Prop := (∀ x : ℝ, f (x + π) = f x)

-- Equation of the symmetry axis of f(x) is x = kπ/2 + π/6
def symmetry_axis (k : ℤ) : Prop := (∀ x : ℝ, 2 * x + π / 6 = k * π + π / 2 → f x = f(- x))

-- Translate the function f to obtain g
def g (x : ℝ) := f (x - π / 3)

-- Define the range of g(x) on the interval [-π/6, π/3]
def range_g : set ℝ := {y | ∃ x ∈ Icc (-π / 6) (π / 3), g x = y}

-- Prove the minimum period and the symmetry axis of f(x)
theorem part1 (k : ℤ) : period π ∧ symmetry_axis k := sorry

-- Prove the range of g(x) on the interval [-π/6, π/3]
theorem part2 : range_g = Icc (-sqrt 3 / 3) (sqrt 3 / 6) := sorry

end part1_part2_l550_550825


namespace find_omega_l550_550457

noncomputable def f (x ω : ℝ) : ℝ := 3 * sin(ω * x) - sqrt 3 * cos(ω * x)

theorem find_omega (ω : ℝ) (h1 : ∀ x : ℝ, x ∈ Ioo (-ω) (2 * ω) → f x ω ≤ f (x + ω) ω)
  (h2 : ∀ x : ℝ, f x ω = f (-ω - x) ω) : 
  ω = sqrt (3 * real.pi) / 3 :=
sorry

end find_omega_l550_550457


namespace mother_used_eggs_l550_550329

variable (initial_eggs : ℕ) (eggs_after_chickens : ℕ) (chickens : ℕ) (eggs_per_chicken : ℕ) (current_eggs : ℕ)

theorem mother_used_eggs (h1 : initial_eggs = 10)
                        (h2 : chickens = 2)
                        (h3 : eggs_per_chicken = 3)
                        (h4 : current_eggs = 11)
                        (eggs_laid : ℕ)
                        (h5 : eggs_laid = chickens * eggs_per_chicken)
                        (eggs_used : ℕ)
                        (h6 : eggs_after_chickens = initial_eggs - eggs_used + eggs_laid)
                        : eggs_used = 7 :=
by
  -- proof steps go here
  sorry

end mother_used_eggs_l550_550329


namespace number_of_initial_cards_l550_550572

theorem number_of_initial_cards (x : ℝ) (h1 : x + 276.0 = 580) : x = 304 :=
by
  sorry

end number_of_initial_cards_l550_550572


namespace find_savings_l550_550259

def income : ℕ := 15000
def expenditure (I : ℕ) : ℕ := 4 * I / 5
def savings (I E : ℕ) : ℕ := I - E

theorem find_savings : savings income (expenditure income) = 3000 := 
by
  sorry

end find_savings_l550_550259


namespace at_least_25_coins_l550_550631

noncomputable def minimum_coins (num_people : Nat) (cost_per_ticket : Nat) (denominations : List Nat) (coins_per_person : Nat) : Nat := 
  let total_fare := num_people * cost_per_ticket
  if List.sum denominations / coins_per_person >= num_people then
    List.sum denominations
  else
    0

theorem at_least_25_coins (num_people : Nat) (cost_per_ticket : Nat) (coins : Nat → List Nat) : 
  num_people = 20 ∧ cost_per_ticket = 5 ∧ (∀ i, i < num_people → coins i = [10, 15, 20]) ∧ List.sum (List.join (List.range' 0 num_people coins)) ≥ 25 :=
by
  sorry

end at_least_25_coins_l550_550631


namespace functional_eq_solution_l550_550918

noncomputable def f (b c : ℝ) (x : ℝ) : ℝ := b / ((c * x + 1) ^ 2)

theorem functional_eq_solution (b c : ℝ) (K : Set ℝ) (hK : K = Set.Ici 0 ∨ ∃ a : ℝ, K = Set.Ico 0 a) :
  (∀ k ∈ K, (1 / k * ∫ t in 0 .. k, f b c t)^2 = f b c 0 * f b c k) →
  (∀ x ∈ K, f b c x = b / ((c * x + 1)^2)) :=
by
  sorry

end functional_eq_solution_l550_550918


namespace A_and_D_independent_l550_550280

-- Define the probability events as given in the problem
def event_A :=
{ (1, x) | x ∈ Finset.range 1 6 }

def event_B :=
{ (x, 2) | x ∈ Finset.range 1 6 }

def event_C :=
{ (x, y) | x + y = 7 ∧ x ∈ Finset.range 1 6 ∧ y ∈ Finset.range 1 6 }

def event_D :=
{ (x, y) | x + y = 6 ∧ x ∈ Finset.range 1 6 ∧ y ∈ Finset.range 1 6 }

-- Define the probability function for drawing with replacement from 1 to 5
def probability (event : Set (ℕ × ℕ)) : ℚ :=
(event.toFinset.card : ℚ) / 25

-- Establish that two events are independent
def independent (e1 e2 : Set (ℕ × ℕ)) : Prop :=
probability (e1 ∩ e2) = probability e1 * probability e2

-- The target statement to be proven
theorem A_and_D_independent :
  independent event_A event_D := 
sorry

end A_and_D_independent_l550_550280


namespace proposition_D_valid_l550_550440

variables {α β γ : Type*} [plane α] [plane β] [plane γ]
variables {m l : Type*} [line m] [line l] 
variable {A : point}

-- Conditions for proposition D
def cond_D (α γ β : Type*) [plane α] [plane γ] [plane β] (m l : Type*) [line m] [line l] : Prop :=
  α ⊥ γ ∧
  (γ ∩ α = m) ∧
  (γ ∩ β = l) ∧
  (l ⊥ m)

-- Proposition D
theorem proposition_D_valid (h : cond_D α γ β m l) : l ⊥ α :=
sorry

end proposition_D_valid_l550_550440


namespace imaginary_part_of_complex_z_l550_550426

noncomputable def complex_z : ℂ := (1 - 2 * complex.I) / complex.I

theorem imaginary_part_of_complex_z :
  complex.im complex_z = -1 :=
sorry

end imaginary_part_of_complex_z_l550_550426


namespace ammeter_sum_l550_550772

variable (A1 A2 A3 A4 A5 : ℝ)
variable (I2 : ℝ)
variable (h1 : I2 = 4)
variable (h2 : A1 = I2)
variable (h3 : A3 = 2 * A1)
variable (h4 : A5 = A3 + A1)
variable (h5 : A4 = (5 / 3) * A5)

theorem ammeter_sum (A1 A2 A3 A4 A5 I2 : ℝ) (h1 : I2 = 4) (h2 : A1 = I2) (h3 : A3 = 2 * A1)
                   (h4 : A5 = A3 + A1) (h5 : A4 = (5 / 3) * A5) :
  A1 + I2 + A3 + A4 + A5 = 48 := 
sorry

end ammeter_sum_l550_550772


namespace alice_bob_not_adjacent_l550_550173

noncomputable def arrangements_excluding_alice_and_bob_adjacent : Nat :=
  let n := 10
  let total_unrestricted := (n - 1)!
  let total_adjacents := 2 * (n - 2)!
  let ways := total_unrestricted - total_adjacents
  ways * (n - 2)

theorem alice_bob_not_adjacent : arrangements_excluding_alice_and_bob_adjacent = 2_540_160 := by
  sorry

end alice_bob_not_adjacent_l550_550173


namespace a_k_lt_sqrt_2k_l550_550361

variables {a : ℕ → ℕ} -- sequence of natural numbers

-- Assume the sequence is strictly increasing
axiom sequence_increasing (n m : ℕ) (h : n < m) : a n < a m

-- Assume any natural number not in the sequence can be represented as a_k + 2k
axiom representation_condition (x : ℕ) (h : ¬ ∃ n, a n = x) : ∃ k, x = a k + 2 * k

-- The theorem to prove
theorem a_k_lt_sqrt_2k (k : ℕ) : a k < nat.sqrt (2 * k) :=
by
  sorry

end a_k_lt_sqrt_2k_l550_550361


namespace odd_nat_numbers_eq_1_l550_550550

-- Definitions of conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

theorem odd_nat_numbers_eq_1
  (a b c d : ℕ)
  (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : is_odd a) (h5 : is_odd b) (h6 : is_odd c) (h7 : is_odd d)
  (h8 : a * d = b * c)
  (h9 : is_power_of_two (a + d))
  (h10 : is_power_of_two (b + c)) :
  a = 1 :=
sorry

end odd_nat_numbers_eq_1_l550_550550


namespace chess_and_pool_l550_550365

variable (A B C : Set α)

theorem chess_and_pool :
  (A ∩ B ≠ ∅) → (C ∩ B ⊆ Aᶜ) → (A ⊈ C) :=
begin
  intros hab hcb,
  sorry,
end

end chess_and_pool_l550_550365


namespace weight_of_first_new_player_l550_550281

theorem weight_of_first_new_player 
    (original_players : ℕ) (average_weight_original : ℕ) (new_player1_weight : ℕ) (new_player2_weight : ℕ) (new_players : ℕ) 
    (average_weight_new : ℕ) (total_players_new : ℕ) :
    original_players = 7 →
    average_weight_original = 94 →
    new_player2_weight = 60 →
    new_players = 2 →
    average_weight_new = 92 →
    total_players_new = original_players + new_players →
    total_weight_new = 828 →
    new_player1_weight = 110 :=
by {
    -- all the conditions to be considered
    intros h_original_players h_average_weight_original h_new_player2_weight h_new_players h_average_weight_new h_total_players_new h_total_weight_new,
    sorry,
}

end weight_of_first_new_player_l550_550281


namespace morio_current_age_l550_550969

-- Given conditions
def teresa_current_age : ℕ := 59
def morio_age_when_michiko_born : ℕ := 38
def teresa_age_when_michiko_born : ℕ := 26

-- Definitions derived from the conditions
def michiko_age : ℕ := teresa_current_age - teresa_age_when_michiko_born

-- Statement to prove Morio's current age
theorem morio_current_age : (michiko_age + morio_age_when_michiko_born) = 71 :=
by
  sorry

end morio_current_age_l550_550969


namespace certain_number_division_l550_550761

theorem certain_number_division (N G : ℤ) : 
  G = 88 ∧ (∃ k : ℤ, N = G * k + 31) ∧ (∃ m : ℤ, 4521 = G * m + 33) → 
  N = 4519 := 
by
  sorry

end certain_number_division_l550_550761


namespace range_of_m_l550_550826

def f (m x : ℝ) : ℝ := m * Real.log x + 8 * x - x^2

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, 1 ≤ x → (f m x)' ≤ 0) ↔ m ≤ -8 :=
by
  sorry

end range_of_m_l550_550826


namespace manny_received_fraction_l550_550904

-- Conditions
def total_marbles : ℕ := 400
def marbles_per_pack : ℕ := 10
def leo_kept_packs : ℕ := 25
def neil_received_fraction : ℚ := 1 / 8

-- Definition of total packs
def total_packs : ℕ := total_marbles / marbles_per_pack

-- Proof problem: What fraction of the total packs did Manny receive?
theorem manny_received_fraction :
  (total_packs - leo_kept_packs - neil_received_fraction * total_packs) / total_packs = 1 / 4 :=
by sorry

end manny_received_fraction_l550_550904


namespace find_m_given_eccentricity_l550_550980

-- Definitions based on given conditions
def hyperbola_eq (m : ℝ) : Prop := ∀ x y : ℝ, m * x^2 + y^2 = 1
def eccentricity (e : ℝ) : Prop := e = sqrt 2

-- Lean statement
theorem find_m_given_eccentricity (m : ℝ) :
  hyperbola_eq m → eccentricity (sqrt 2) → m = -1 :=
by
  sorry

end find_m_given_eccentricity_l550_550980


namespace dartboard_central_angle_l550_550324

theorem dartboard_central_angle (A : ℝ) (x : ℝ) (P : ℝ) (h1 : P = 1 / 4) 
    (h2 : A > 0) : (x / 360 = 1 / 4) -> x = 90 :=
by
  sorry

end dartboard_central_angle_l550_550324


namespace isosceles_triangles_with_perimeter_27_l550_550845

theorem isosceles_triangles_with_perimeter_27 :
  ∃ n : ℕ, n = 6 ∧ ∀ a b c : ℕ, (a = b ∧ a > 0 ∧ 2 * a + c = 27 ∧ c mod 2 = 1) → n = 6 :=
by
  sorry

end isosceles_triangles_with_perimeter_27_l550_550845


namespace gcd_g_values_l550_550544

def g (x : ℤ) : ℤ := x^2 - 2 * x + 2023

theorem gcd_g_values : gcd (g 102) (g 103) = 1 := by
  sorry

end gcd_g_values_l550_550544


namespace sequences_of_students_l550_550935

theorem sequences_of_students : 
  let students := 15
  let times := 3
  ∀ n t, n = students → t = times → 
  ( ∏ i in (finset.range t).map (λ i, n - i), (1+i)) = 2730 := by
  intros n t h_n h_t
  rw [h_n, h_t]
  norm_num
  sorry

end sequences_of_students_l550_550935


namespace calculate_expression_l550_550729

theorem calculate_expression : 
  (1 / 2) ^ (-2: ℤ) - 3 * Real.tan (Real.pi / 6) - abs (Real.sqrt 3 - 2) = 2 := 
by
  sorry

end calculate_expression_l550_550729


namespace constant_term_of_product_is_21_l550_550640

def P (x : ℕ) : ℕ := x ^ 3 + x ^ 2 + 3
def Q (x : ℕ) : ℕ := 2 * x ^ 4 + x ^ 2 + 7

theorem constant_term_of_product_is_21 :
  (P 0) * (Q 0) = 21 :=
by
  rw [P, Q]
  simp
  rfl

end constant_term_of_product_is_21_l550_550640


namespace geometric_series_sum_l550_550038

theorem geometric_series_sum :
  let a := 3
  let r := -2
  let n := 10
  let S := a * ((r^n - 1) / (r - 1))
  S = -1023 :=
by 
  -- Sorry allows us to omit the proof details
  sorry

end geometric_series_sum_l550_550038


namespace standard_segments_odd_l550_550221

theorem standard_segments_odd (A B : ℝ) (cA cB : bool) (P : ℕ → ℝ) (cP : ℕ → bool) (n : ℕ)
  (hA : cA = tt)
  (hB : cB = ff)
  (hPoints : ∀ i, i < n → A < P i ∧ P i < B) :
  (∃ k, k = (Finset.image (λ i, (cP i ≠ cP (i + 1))) (Finset.range n)).card ∧ k % 2 = 1) :=
by
  sorry

end standard_segments_odd_l550_550221


namespace series_sum_eq_l550_550205

noncomputable def series_sum (a b : ℝ) : ℝ :=
\sum (n : ℕ) (n > 0), 
\frac{1}{[(n-1)*a - (n-3)*b] * [n*a - (2*n-3)*b]} 

theorem series_sum_eq (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a > b) :
  series_sum a b = \frac{1}{(a-b)*b} := 
sorry

end series_sum_eq_l550_550205


namespace speed_of_second_half_l550_550352

theorem speed_of_second_half (t d s1 d1 d2 : ℝ) (h_t : t = 30) (h_d : d = 672) (h_s1 : s1 = 21)
  (h_d1 : d1 = d / 2) (h_d2 : d2 = d / 2) (h_t1 : d1 / s1 = 16) (h_t2 : t - d1 / s1 = 14) :
  d2 / 14 = 24 :=
by sorry

end speed_of_second_half_l550_550352


namespace impossible_15_cents_l550_550780

theorem impossible_15_cents (a b c d : ℕ) (ha : a ≤ 4) (hb : b ≤ 4) (hc : c ≤ 4) (hd : d ≤ 4) (h : a + b + c + d = 4) : 
  1 * a + 5 * b + 10 * c + 25 * d ≠ 15 :=
by
  sorry

end impossible_15_cents_l550_550780


namespace minimum_sugar_quantity_l550_550716

theorem minimum_sugar_quantity :
  ∃ s f : ℝ, s = 4 ∧ f ≥ 4 + s / 3 ∧ f ≤ 3 * s ∧ 2 * s + 3 * f ≤ 36 :=
sorry

end minimum_sugar_quantity_l550_550716


namespace problem_1_problem_2_l550_550789

noncomputable def f (x : ℝ) : ℝ := 2^x - 1 / 2^(abs x)

theorem problem_1 
  (hx : f x = 3 / 2) 
  : x = 1 := 
sorry

theorem problem_2 
  (m : ℝ) 
  (h : ∀ t ∈ (set.Icc 1 2), 2^t * f(2 * t) + m * f t ≥ 0) 
  : m ∈ set.Ici (-5) := 
sorry

end problem_1_problem_2_l550_550789


namespace binomial_expansion_l550_550211

theorem binomial_expansion (n : ℕ) :
  ∑ k in finset.range (n + 1), (nat.choose n k) * (2 : ℕ)^(n - k) * (-1)^k = 1 :=
sorry

end binomial_expansion_l550_550211


namespace series_diverges_l550_550895

open_locale big_operators

noncomputable def series_term (n : ℕ) : ℝ :=
  let product := ∏ k in finset.range (n + 1), (3 * k - 2) in
  (product / n!) * real.sin (1 / (2 ^ (n + 1)))

def series := ∑' n : ℕ, series_term n

theorem series_diverges : ¬ summable series := by
  sorry

end series_diverges_l550_550895


namespace subset_singleton_zero_A_l550_550671

def A : Set ℝ := {x | x > -3}

theorem subset_singleton_zero_A : {0} ⊆ A := 
by
  sorry  -- Proof is not required

end subset_singleton_zero_A_l550_550671


namespace area_of_circle_l550_550948

noncomputable def Point : Type := (ℝ × ℝ)

def lies_on_circle (A B C : Point) (radius : ℝ) : Prop :=
  ∃ (O : Point), (dist O A = radius) ∧ (dist O B = radius) ∧ (dist O C = radius)

def tangent_intersect_y_axis (A B : Point) (intersect_point : Point) : Prop :=
  let y_axis_intersect := (0, snd intersect_point) in
  snd intersect_point = snd y_axis_intersect

theorem area_of_circle 
  (A B : Point)
  (radius : ℝ)
  (area : ℝ)
  (hw : lies_on_circle A B (0, 2) radius)
  (hy : tangent_intersect_y_axis A B (0, 2))
  (ha : radius^2 * π = 237.62 * π) : area = 237.62 * π :=
sorry

end area_of_circle_l550_550948


namespace vector_expression_l550_550164

variables (a b c : ℝ × ℝ)
variables (λ μ : ℝ)

-- Given conditions
def a_def : ℝ × ℝ := (1, 1)
def b_def : ℝ × ℝ := (-1, 1)
def c_def : ℝ × ℝ := (4, 2)

-- Proof statement
theorem vector_expression 
  (hac : c = λ • a + μ • b)
  (ha : a = (1, 1))
  (hb : b = (-1, 1))
  (hc : c = (4, 2)) :
  c = 3 • a + -1 • b :=
sorry

end vector_expression_l550_550164


namespace sin_B_value_cos_C_plus_pi_over_12_l550_550185

-- Given conditions
variables (a b c : ℝ)
variable (A B C : ℝ)
variable (h1 : a^2 = b^2 + c^2 - b * c)
variable (h2 : a = sqrt 15 / 2 * b)

-- Question 1: Prove \(\sin B = \dfrac{\sqrt{5}}{5}\)
theorem sin_B_value : sin B = sqrt 5 / 5 := sorry

-- Question 2: Prove \(\cos(C + \dfrac{\pi}{12}) = -\dfrac{\sqrt{10}}{10}\)
theorem cos_C_plus_pi_over_12 :
  cos (C + π / 12) = -sqrt 10 / 10 := sorry

end sin_B_value_cos_C_plus_pi_over_12_l550_550185


namespace slope_of_tangent_line_at_1_l550_550767

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / Real.exp x

theorem slope_of_tangent_line_at_1 : (Real.deriv f 1) = 1 / Real.exp 1 :=
by
  sorry

end slope_of_tangent_line_at_1_l550_550767


namespace total_students_in_class_l550_550970

def total_age_of_students (n : ℕ) (ages : list ℕ) : ℕ :=
  ages.sum

theorem total_students_in_class
  (avg_age_class : ℕ)
  (avg_age_8_students : ℕ)
  (num_8_students : ℕ)
  (avg_age_6_students : ℕ)
  (num_6_students : ℕ)
  (age_last_student : ℕ) :
  avg_age_class = 15 →
  avg_age_8_students = 14 →
  num_8_students = 8 →
  avg_age_6_students = 16 →
  num_6_students = 6 →
  age_last_student = 17 →
  15 = ((num_8_students * avg_age_8_students) + (num_6_students * avg_age_6_students) + age_last_student) / avg_age_class :=
by
  sorry

end total_students_in_class_l550_550970


namespace trader_gain_l550_550006

theorem trader_gain (C N : ℕ) (h_gain : 15 * C)
  (h_percentage : (15 / N) * 100 = 16.666666666666664)
  : N = 90 :=
begin
  sorry
end

end trader_gain_l550_550006


namespace last_digit_sum_chessboard_segments_l550_550941

theorem last_digit_sum_chessboard_segments {N : ℕ} (tile_count : ℕ) (segment_count : ℕ := 112) (dominos_per_tiling : ℕ := 32) (segments_per_domino : ℕ := 2) (N := tile_count / N) :
  (80 * N) % 10 = 0 :=
by
  sorry

end last_digit_sum_chessboard_segments_l550_550941


namespace find_m_l550_550872

theorem find_m (m : ℤ) (h1 : ∃ (x : ℝ), x ^ 2 = (2 * m - 6) ^ 2 ∧ x ^ 2 = (m + 3) ^ 2) : 
  (2 * m - 6) + (m + 3) = 0 → m = 1 :=
by
  intro h2
  simp_all
  sorry

end find_m_l550_550872


namespace polynomial_problem_l550_550539

theorem polynomial_problem 
  (d_1 d_2 d_3 d_4 e_1 e_2 e_3 e_4 : ℝ)
  (h : ∀ (x : ℝ),
    x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 =
    (x^2 + d_1 * x + e_1) * (x^2 + d_2 * x + e_2) * (x^2 + d_3 * x + e_3) * (x^2 + d_4 * x + e_4)) :
  d_1 * e_1 + d_2 * e_2 + d_3 * e_3 + d_4 * e_4 = -1 := 
by
  sorry

end polynomial_problem_l550_550539


namespace number_of_kittens_l550_550336

-- Definitions for the given conditions.
def total_animals : ℕ := 77
def hamsters : ℕ := 15
def birds : ℕ := 30

-- The proof problem statement.
theorem number_of_kittens : total_animals - hamsters - birds = 32 := by
  sorry

end number_of_kittens_l550_550336


namespace triangle_area_dodecagon_l550_550270

noncomputable def least_possible_triangle_area : ℂ := 
  let r := 2 * real.sqrt 3
  let vertices := λ k, r * complex.exp (2 * real.pi * I * k / 12)
  let D := vertices 0
  let E := vertices 1
  let F := vertices 2
  let base := complex.abs (E - F)
  let height := complex.abs (D - (E + F) / 2)
  (1 / 2) * base * height

theorem triangle_area_dodecagon :
  least_possible_triangle_area = 3 * real.sqrt 3 := by
  sorry

end triangle_area_dodecagon_l550_550270


namespace find_angle_ABC_l550_550622

-- Definitions for the conditions in problem
variables (A B C D E F O : Point) (ω : Circle)
variables (ABCD_parallelogram : Parallelogram A B C D)
variables (angle_B_lt_90 : angle B < 90)
variables (AB_lt_BC : dist A B < dist B C)
variables (E_and_F_on_circumcircle : OnCircumcircle E ω ∧ OnCircumcircle F ω)
variables (tangents_pass_through_D : TangentToCircle E ω D ∧ TangentToCircle F ω D)
variables (angle_EDA_eq_angle_FDC : angle (E D A) = angle (F D C))

-- The main theorem to be proved
theorem find_angle_ABC :
  angle A B C = 60 :=
sorry

end find_angle_ABC_l550_550622


namespace monotonic_increasing_interval_l550_550618

theorem monotonic_increasing_interval {x : ℝ} (f : ℝ → ℝ) (h₁ : f = λ x, log (1/2 : ℝ) (x^2 - 4))
  (domain : ∀ x, (f x).dom ↔ (x < -2 ∨ x > 2))
  (h₂ : ∀ x, x < -2 → x^2 - 4 < 0 ∨ x^2 - 4 > 0)
  (h₃ : ∀ t, t < 0 → log (1/2 : ℝ) t > log (1/2 : ℝ) 0)
  (h₄ : ∀ x, x < -2 → x^2 - 4 ∈ ℝ) :
  monotone_incr_on f {x | x < -2} :=
begin
  sorry
end

end monotonic_increasing_interval_l550_550618


namespace sufficient_but_not_necessary_l550_550317

-- Define the quadratic function
def f (x : ℝ) (m : ℝ) : ℝ := x^2 + 2*x + m

-- The problem statement to prove that "m < 1" is a sufficient condition
-- but not a necessary condition for the function f(x) to have a root.
theorem sufficient_but_not_necessary (m : ℝ) :
  (m < 1 → ∃ x : ℝ, f x m = 0) ∧ ¬(¬(m < 1) → ∃ x : ℝ, f x m = 0) :=
sorry

end sufficient_but_not_necessary_l550_550317


namespace tiling_possible_l550_550382

theorem tiling_possible (n : ℕ) (h : n ≥ 2) : (∃ f : ℕ → ℕ → Prop, 
  (∀ i j, i ≠ j → f i j = f j i) ∧
  (∀ i, f i i = 0) ∧
  (∀ i j, f i j ≠ 1)) ↔ (n % 3 = 0 ∨ n % 3 = 1) ∧ (n ≠ 4) ∧ (n ≠ 6) := 
sorry

end tiling_possible_l550_550382


namespace real_roots_exactly_three_l550_550085

theorem real_roots_exactly_three (m : ℝ) :
  (∀ x : ℝ, x^2 - 2 * |x| + 2 = m) → (∃ a b c : ℝ, 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  (a^2 - 2 * |a| + 2 = m) ∧ 
  (b^2 - 2 * |b| + 2 = m) ∧ 
  (c^2 - 2 * |c| + 2 = m)) → 
  m = 2 := 
sorry

end real_roots_exactly_three_l550_550085


namespace parallelogram_of_opposite_sides_equal_l550_550951

theorem parallelogram_of_opposite_sides_equal (A B C D : Type) [euclidean_geometry] 
  (AB CD AD BC : ℝ)
  (h1 : AB = CD) (h2 : AD = BC) :
  parallelogram A B C D :=
begin
  sorry
end

end parallelogram_of_opposite_sides_equal_l550_550951


namespace segment_inequality_l550_550525

open Function

-- Defining the points P_i, A, B, and M
variables {n : ℕ} -- n is the number of points
variables {P : Fin n → EuclideanSpace ℝ 2} {A B M : EuclideanSpace ℝ 2}

-- The lengths of segments |P_i M|, |P_i A|, |P_i B| are represented using the distance function dist

-- The main theorem we need to prove
theorem segment_inequality (hM : M ∈ line [A, B]) :
  (∑ i : Fin n, dist (P i) M) ≤ max (∑ i : Fin n, dist (P i) A) (∑ i : Fin n, dist (P i) B) :=
sorry

end segment_inequality_l550_550525


namespace solve_quadratic_eq_l550_550269

theorem solve_quadratic_eq (x : ℝ) : x^2 = 2 * x ↔ x = 0 ∨ x = 2 := sorry

end solve_quadratic_eq_l550_550269


namespace squirrel_travel_distance_l550_550309

def height := 12
def circumference := 3
def rise_per_circuit := 4

theorem squirrel_travel_distance : 
  let circuits := height / rise_per_circuit in
  let diagonal_distance := (rise_per_circuit^2 + circumference^2).sqrt in
  let total_distance := circuits * diagonal_distance in
  total_distance = 15 := 
by {
  sorry -- Proof goes here
}

end squirrel_travel_distance_l550_550309


namespace problem1_problem2_l550_550997

-- Given vectors
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

-- Problem 1: Prove that 3a + b - 2c = (0, 6)
theorem problem1 : 3 • a + b - 2 • c = (0, 6) := 
by {
   sorry
}

-- Problem 2: Prove that there exist real numbers m and n such that a = m • b + n • c with m = 5/9 and n = 8/9
theorem problem2 : ∃ (m n : ℝ), a = m • b + n • c ∧ m = (5/9) ∧ n = (8/9) := 
by {
   sorry
}

end problem1_problem2_l550_550997


namespace proj_w_v_plus_u_l550_550554

variable (w : ℝ) -- Assuming "w" is a scalar that parameterizes the projection function

-- Vectors in ℝ³
def v := (1, 1, 0) : ℝ × ℝ × ℝ
def u := (0, -1, 2) : ℝ × ℝ × ℝ

-- Projections
def proj_w_v : ℝ × ℝ × ℝ := (1, 1, 0)
def proj_w_u : ℝ × ℝ × ℝ := (0, -1, 2)

theorem proj_w_v_plus_u :
  proj_w_v + proj_w_u = (1, 0, 2) := 
by
  sorry

end proj_w_v_plus_u_l550_550554


namespace coeff_x2_in_expansion_l550_550972

theorem coeff_x2_in_expansion : 
  let f := (1 + (λ x, x) (λ x, 1 + sqrt(x))^5)
  the coefficient_of_x2 (expand f) = 15 :=
sorry

end coeff_x2_in_expansion_l550_550972


namespace findMonicQuadraticPolynomial_l550_550064

-- Define the root as a complex number
def root : ℂ := -3 - complex.I * real.sqrt 8

-- Define the conditions
def isMonic (p : polynomial ℝ) : Prop := p.leadingCoeff = 1
def hasRealCoefficients (p : polynomial ℝ) : Prop := ∀ a ∈ p.support, is_real (p.coeff a)

-- Define the polynomial
noncomputable def polynomial : polynomial ℝ :=
  polynomial.C 1 * polynomial.X^2 + polynomial.C 6 * polynomial.X + polynomial.C 17

-- The target statement
theorem findMonicQuadraticPolynomial :
  ∀ (p : polynomial ℝ), 
  isMonic p ∧ hasRealCoefficients p ∧ (root ∈ p.roots) →
  p = polynomial :=
by
  sorry

end findMonicQuadraticPolynomial_l550_550064


namespace total_profit_percentage_l550_550000

theorem total_profit_percentage (total_apples : ℕ) (percent_sold_10 : ℝ) (percent_sold_30 : ℝ) (profit_10 : ℝ) (profit_30 : ℝ) : 
  total_apples = 280 → 
  percent_sold_10 = 0.40 → 
  percent_sold_30 = 0.60 → 
  profit_10 = 0.10 → 
  profit_30 = 0.30 → 
  ((percent_sold_10 * total_apples * (1 + profit_10) + percent_sold_30 * total_apples * (1 + profit_30) - total_apples) / total_apples * 100) = 22 := 
by 
  intros; sorry

end total_profit_percentage_l550_550000


namespace smallest_n_rotation_matrix_l550_550768

open Matrix

noncomputable def rotation_matrix_240 := 
  ![[Real.cos (240 * Real.pi / 180), -Real.sin (240 * Real.pi / 180)],
    [Real.sin (240 * Real.pi / 180), Real.cos (240 * Real.pi / 180)]]

theorem smallest_n_rotation_matrix :
  ∃ n : ℕ, n > 0 ∧ rotation_matrix_240 ^ n = 1 ∧ ∀ m : ℕ, m > 0 → rotation_matrix_240 ^ m = 1 → m ≥ n := 
by
  sorry

end smallest_n_rotation_matrix_l550_550768


namespace equifacial_tetrahedron_iff_perpendicular_midsegments_l550_550581

variables {A B C D : Type*} [Tetrahedron A B C D]

theorem equifacial_tetrahedron_iff_perpendicular_midsegments :
  (∀ (face: triangle), is_equilateral face) ↔ 
  (∀ (e1 e2 e3 : segment), midsegment_perpendicular e1 e2 e3) :=
sorry

end equifacial_tetrahedron_iff_perpendicular_midsegments_l550_550581


namespace quadratic_eq_with_given_roots_l550_550163

variable (x1 x2 : ℝ)

def roots_quadratic (x1 x2 : ℝ) : Polynomial ℝ :=
  Polynomial.X ^ 2 - (x1 + x2) * Polynomial.X + x1 * x2

theorem quadratic_eq_with_given_roots (x1 x2 : ℝ) (h1 : x1 = 1) (h2 : x2 = 2) :
  roots_quadratic x1 x2 = Polynomial.X ^ 2 - 3 * Polynomial.X + 2 :=
by
  rw [h1, h2]
  dsimp [roots_quadratic]
  norm_num
  sorry

#check quadratic_eq_with_given_roots

end quadratic_eq_with_given_roots_l550_550163


namespace total_rehabilitation_centers_l550_550649

def lisa_visits : ℕ := 6
def jude_visits (lisa : ℕ) : ℕ := lisa / 2
def han_visits (jude : ℕ) : ℕ := 2 * jude - 2
def jane_visits (han : ℕ) : ℕ := 2 * han + 6
def total_visits (lisa jude han jane : ℕ) : ℕ := lisa + jude + han + jane

theorem total_rehabilitation_centers :
  total_visits lisa_visits (jude_visits lisa_visits) (han_visits (jude_visits lisa_visits)) 
    (jane_visits (han_visits (jude_visits lisa_visits))) = 27 :=
by
  sorry

end total_rehabilitation_centers_l550_550649


namespace sequence_property_l550_550096

-- Definition of the sequence
def sequence (a b : ℤ) : ℕ → ℤ
| 0       => a
| 1       => b
| (n + 2) => sequence (n + 1) - sequence n

-- Sum of the sequence
def sequence_sum (a b : ℤ) : ℕ → ℤ
| 0       => sequence a b 0
| n       => sequence a b n + sequence_sum a b (n - 1)

theorem sequence_property (a b : ℤ) :
  sequence a b 2007 = -a ∧ sequence_sum a b 2007 = 2 * b - a :=
sorry

end sequence_property_l550_550096


namespace sum_of_squares_le_neg_nmM_l550_550528

theorem sum_of_squares_le_neg_nmM
  (n : ℕ) (x : ℕ → ℝ) (m M : ℝ)
  (hx_interval : ∀ i, m ≤ x i ∧ x i ≤ M)
  (hx_sum : ∑ i in Finset.range n, x i = 0) :
  (∑ i in Finset.range n, (x i) ^ 2) ≤ -n * m * M := 
sorry

end sum_of_squares_le_neg_nmM_l550_550528


namespace sum_of_squares_of_four_consecutive_even_numbers_eq_344_l550_550666

theorem sum_of_squares_of_four_consecutive_even_numbers_eq_344 (n : ℤ) 
  (h : n + (n + 2) + (n + 4) + (n + 6) = 36) : 
  n^2 + (n + 2)^2 + (n + 4)^2 + (n + 6)^2 = 344 :=
by sorry

end sum_of_squares_of_four_consecutive_even_numbers_eq_344_l550_550666


namespace f_monotonicity_g_min_l550_550207

-- Definitions
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * a ^ x - 2 * a ^ (-x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a ^ (2 * x) + a ^ (-2 * x) - 2 * f x a

-- Conditions
variable {a : ℝ} 
variable (a_pos : 0 < a) (a_ne_one : a ≠ 1) (f_one : f 1 a = 3) (x : ℝ) (h : 0 ≤ x ∧ x ≤ 3)

-- Monotonicity of f(x)
theorem f_monotonicity : 
  (∀ x y, x < y → f x a < f y a) ∨ (∀ x y, x < y → f y a < f x a) :=
sorry

-- Minimum value of g(x)
theorem g_min : ∃ x' : ℝ, 0 ≤ x' ∧ x' ≤ 3 ∧ g x' a = -2 :=
sorry

end f_monotonicity_g_min_l550_550207


namespace comparison_of_large_exponents_l550_550301

theorem comparison_of_large_exponents : 2^1997 > 5^850 := sorry

end comparison_of_large_exponents_l550_550301


namespace arithmetic_sequence_problem_l550_550177

noncomputable def a_n (n : ℕ) (a d : ℝ) : ℝ := a + (n - 1) * d

theorem arithmetic_sequence_problem (a d : ℝ) 
  (h : a_n 1 a d - a_n 4 a d - a_n 8 a d - a_n 12 a d + a_n 15 a d = 2) :
  a_n 3 a d + a_n 13 a d = -4 :=
by
  sorry

end arithmetic_sequence_problem_l550_550177


namespace sum_of_set_A_star_B_l550_550742

def set_operation (A B : Set ℕ) : Set ℕ := {z | ∃ x ∈ A, ∃ y ∈ B, z = x * y}

def sum_elements (s : Set ℕ) : ℕ := s.toFinset.sum id

theorem sum_of_set_A_star_B :
  let A := {1, 2}
  let B := {0, 2}
  sum_elements (set_operation A B) = 6 :=
by
  -- Definitions and assumptions
  let A := {1, 2}
  let B := {0, 2}
  let AB := set_operation A B
  -- Conclusion
  show sum_elements AB = 6
  sorry

end sum_of_set_A_star_B_l550_550742


namespace simplify_and_evaluate_l550_550600

theorem simplify_and_evaluate :
  let x := (1 : ℝ) / 2
  let y := -(1 : ℝ) / 4
  (x, y) ∈ set.univ → 
  ([
    ((3 * x + 2 * y) * (3 * x - 2 * y) - (3 * x - 2 * y) ^ 2) / (4 * y)
  ]) = 2 :=
by
  intros x y h
  exact sorry

end simplify_and_evaluate_l550_550600


namespace geometric_seq_common_ratio_range_l550_550428

theorem geometric_seq_common_ratio_range {a_1 : ℝ} {q : ℝ} (h_pos : a_1 > 0) :
  (∀ n : ℕ, S_n > 0) → (q ≠ 1) → q ∈ set.Ioo (-1) 0 ∪ set.Ioi 0 :=
by
  sorry

end geometric_seq_common_ratio_range_l550_550428


namespace total_points_first_half_l550_550325

theorem total_points_first_half
  (r d : ℝ)
  (Tigers_scores : ℝ → List ℝ)
  (Lions_scores : List ℝ := [7, 7 + d, 7 + 2 * d, 7 + 3 * d])
  (Tigers_total, Lions_total : ℝ)
  (halftime_total : Bool := ∑t in (Tigers_scores r).take 2 ++ Lions_scores.take 2, (λx => x) = 25)
  (endgame_inequality : Tigers_total = Lions_total + 3)
  (tigers_sequence_condition : ∀ n < 4, Tigers_scores r ! n = 3 * r^n)
  (lions_sequence_condition : ∀ n < 4, Lions_scores ! n = 7 + n * d)
  (neither_exceeds_200 : Tigers_total < 200 ∧ Lions_total < 200) : 
  ∃ r d: ℝ, halftime_total → 
  (((1 : ℝ) + r + r^2 + r^3) = 10.3333333333333333 ∧ d = 7/3 ∧ (3 * (1 + r + r^2 + r^3) < 200 ∧ 28 + 6*d < 200) ∧ endgame_inequality) :=
 sorry

end total_points_first_half_l550_550325


namespace simplify_trig_expression_l550_550238

theorem simplify_trig_expression (A B : ℝ) (hA : A = 30) (hB : B = 40) :
  (sin (A * π / 180) + sin (B * π / 180)) / (cos (A * π / 180) + cos (B * π / 180)) = 
  (tan (35 * π / 180)) :=
by sorry

end simplify_trig_expression_l550_550238


namespace find_a_value_l550_550176

theorem find_a_value (a : ℝ) (t : ℝ) (x y : ℝ) : 
  (∃ t, x = - (3/5) * t + 2 ∧ y = (4/5) * t) ∧ (ρ = a * sin θ) ∧ 
  (∃ c r, (0, c) = (0, a / 2) ∧ r = a / 2 ∧ 4 * x + 3 * y - 8 = 0) ∧
  ∃ d, d = abs ((3*a/2 - 8) / 5) ∧ sqrt(3) * r = 2 * sqrt(r^2 - d^2) 
  → (a = 32 ∨ a = 32 / 11) := 
by sorry

end find_a_value_l550_550176


namespace find_m_find_S_n_find_lambda_range_l550_550444

noncomputable def f (x : ℝ) : ℝ := 1 / 2 + Real.log (x / (1 - x)) / Real.log 2

-- Point A and B are on the graph of the function f
variable {x1 x2 y1 y2 : ℝ}
axiom A_on_f : y1 = f x1
axiom B_on_f : y2 = f x2

-- Midpoint M has coordinates (1/2, m)
variable {m : ℝ}
axiom midpoint_M : (x1 + x2) / 2 = 1 / 2 ∧ m = (y1 + y2) / 2

-- Define S_n
def S_n (n : ℕ) : ℝ := (Finset.range (n-1)).sum (λ i, f ((i+1:ℕ)/n : ℝ))

-- Define sequence a_n
def a_n (n : ℕ) : ℝ :=
  if n = 0 then 0 -- this case isn't actually used due to the conditions given, this is to handle empty case
  else if n = 1 then 1 / 2 else S_n n

-- Define T_n as the sum of the first n terms of sequence a_n
def T_n (n : ℕ) : ℝ := (Finset.range n).sum (λ i, a_n (i+1))

theorem find_m : m = 1 / 2 :=
sorry

theorem find_S_n {n : ℕ} (h : 2 ≤ n) : S_n n = (n - 1) / 2 :=
sorry

theorem find_lambda_range (λ : ℝ) :
  (∀ n : ℕ, 1 ≤ n → T_n n > λ * (S_n (n + 1) + 1)) → λ < 1 / 3 :=
sorry

end find_m_find_S_n_find_lambda_range_l550_550444


namespace three_a1_gt_2n_l550_550555

open Finset

theorem three_a1_gt_2n (n : ℕ) (a : Fin n → ℕ)
  (h1 : ∀ i, 0 < a i ∧ a i ≤ 2 * n)
  (h2 : ∀ i j, i ≠ j → Nat.lcm (a i) (a j) ≥ 2 * n) :
  3 * a 0 > 2 * n := by
  sorry

end three_a1_gt_2n_l550_550555


namespace area_difference_of_square_and_circle_l550_550974

theorem area_difference_of_square_and_circle (d_square d_circle : ℝ) (h_d_square : d_square = 8) (h_d_circle : d_circle = 8) : 
  |((16 / Real.pi) - 32)| ≈ 26.9 :=
by {
  sorry 
}

end area_difference_of_square_and_circle_l550_550974


namespace medians_of_ABM_form_right_triangle_l550_550010

theorem medians_of_ABM_form_right_triangle
  (A B C H M : Point)
  (h1: is_triangle A B C)
  (h2: is_median B M C)
  (h3: is_altitude A H B C)
  (h4: bisects A H B M) :
  is_right_angle (medians_of_triangle A B M) :=
sorry

end medians_of_ABM_form_right_triangle_l550_550010


namespace find_m_of_odd_function_l550_550157

theorem find_m_of_odd_function (m : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = ((x + 3) * (x + m)) / x)
  (h₂ : ∀ x, f (-x) = -f x) : m = -3 :=
sorry

end find_m_of_odd_function_l550_550157


namespace range_of_f_on_0_pi_div_2_l550_550460

noncomputable def f (x : ℝ) : ℝ := sin (2 * x) - cos (2 * x)
def range_of_f : Set ℝ := { y : ℝ | ∃ (x : ℝ), 0 ≤ x ∧ x ≤ π / 2 ∧ f x = y }

theorem range_of_f_on_0_pi_div_2 : range_of_f = Set.Icc (-1 : ℝ) (Real.sqrt 2) :=
by
  sorry

end range_of_f_on_0_pi_div_2_l550_550460


namespace smallest_share_l550_550854

-- Definitions based on problem conditions
def total_francs := 100
def denomination_1_franc := 1
def denomination_5_franc := 5
def denomination_50_centime := 0.5

-- Let x be the number of 1-franc coins
-- Let y be the number of 5-franc coins
-- Let z be the number of 50-centime coins where z = x / 9 
def valid_distribution (x y : ℕ) : Prop :=
  x + 5 * y + 0.5 * (x / 9 : ℕ) = 100 ∧ 
  (x / 9 : ℕ) * 9 = x

theorem smallest_share (x y : ℕ) (h : valid_distribution x y) : 
  min (x * denomination_1_franc) (min (y * denomination_5_franc) ((x / 9 : ℕ) * denomination_50_centime)) = 5 :=
sorry

end smallest_share_l550_550854


namespace range_f_l550_550119

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.sqrt (a * Real.cos x ^ 2 + b * Real.sin x ^ 2) + 
  Real.sqrt (a * Real.sin x ^ 2 + b * Real.cos x ^ 2)

theorem range_f (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  Set.range (f a b) = Set.Icc (Real.sqrt a + Real.sqrt b) (Real.sqrt (2 * (a + b))) :=
sorry

end range_f_l550_550119


namespace convex_func_continuous_exists_unique_g_l550_550526

-- Part (a): Defining convexity and proving continuity
def is_convex (f : ℝ → ℝ) : Prop :=
  ∀ x y ∈ ℝ, ∀ λ ∈ Icc 0 1, f (λ * x + (1 - λ) * y) ≤ λ * f x + (1 - λ) * f y

theorem convex_func_continuous {f : ℝ → ℝ} (convex_f : is_convex f) :
  continuous f := 
sorry

-- Part (b): Existence and uniqueness of function g
theorem exists_unique_g {f : ℝ → ℝ} (convex_f : is_convex f) :
  ∃! g : Icc 0 ∞ → ℝ, ∀ x : Icc 0 ∞, f (x.1 + g x) = f (g x) - g x := 
sorry

end convex_func_continuous_exists_unique_g_l550_550526


namespace geometric_series_sum_l550_550032

theorem geometric_series_sum :
  ∀ (a r : ℤ) (n : ℕ),
  a = 3 → r = -2 → n = 10 →
  (a * ((r ^ n - 1) / (r - 1))) = -1024 :=
by
  intros a r n ha hr hn
  rw [ha, hr, hn]
  sorry

end geometric_series_sum_l550_550032


namespace plane_divides_tetrahedron_in_half_l550_550952

noncomputable def Tetrahedron (A B C D : Point) : Tetrahedron := sorry

axiom midpoint (A B : Point) : Point

def plane_through_midpoints {A B C D N M : Point} (hN : N = midpoint A B) (hM : M = midpoint C D) : Plane := sorry

def volume_partition (T : Tetrahedron) (P : Plane) : Prop :=
  (volume (region_above_plane T P) = volume (region_below_plane T P)) 

theorem plane_divides_tetrahedron_in_half {A B C D N M : Point} 
  (hN : N = midpoint A B) 
  (hM : M = midpoint C D) 
  (T : Tetrahedron) 
  (P : Plane) 
  (hplane : P = plane_through_midpoints hN hM) : volume_partition T P := 
sorry

end plane_divides_tetrahedron_in_half_l550_550952


namespace cos_alpha_minus_pi_over_2_l550_550183

theorem cos_alpha_minus_pi_over_2 (α : ℝ) 
  (h1 : ∃ k : ℤ, α = k * (2 * Real.pi) ∨ α = k * (2 * Real.pi) + Real.pi / 2 ∨ α = k * (2 * Real.pi) + Real.pi ∨ α = k * (2 * Real.pi) + 3 * Real.pi / 2)
  (h2 : Real.cos α = 4 / 5)
  (h3 : Real.sin α = -3 / 5) : 
  Real.cos (α - Real.pi / 2) = -3 / 5 := 
by 
  sorry

end cos_alpha_minus_pi_over_2_l550_550183


namespace tangent_line_at_neg1_l550_550976

noncomputable def f : ℝ → ℝ := λ x, x^4 - 3 * x^2

theorem tangent_line_at_neg1 :
  let x := -1
  let y := f x
  let df := deriv f
  2 * x -  y = 0 :=
by
  sorry

end tangent_line_at_neg1_l550_550976


namespace find_omega_l550_550558

variable (ω : ℝ)

def decreasing_on_interval (x : ℝ) :=
  2 * real.cos(ω * x)

theorem find_omega (h1 : ω > 0)
    (h2 : ∀ x1 x2, 0 ≤ x1 → x1 ≤ x2 → x2 ≤ (2 * real.pi / 3) → decreasing_on_interval ω x2 ≤ decreasing_on_interval ω x1)
    (h3 : decreasing_on_interval ω (2 * real.pi / 3) = 1) :
  ω = 1 / 2 :=
sorry

end find_omega_l550_550558


namespace circumcoronene_valid_bonds_l550_550268

-- Define the problem's conditions
def circumcoronene_hydrocarbon : Type := List (List Bool)
def carbon_atom (bonds : List Bool) : Prop := bonds.length = 4
def hydrogen_atom (bonds : List Bool) : Prop := bonds.length = 1

-- Number of valid bond arrangements for circumcircumcircumcoronene
def valid_arrangements (structure : circumcoronene_hydrocarbon) : Prop :=
  (∀ c ∈ structure, carbon_atom c) ∧ (∑ c in structure, length c) = 150 * 4 + 30 * 1

theorem circumcoronene_valid_bonds :
  ∃ structure : circumcoronene_hydrocarbon, valid_arrangements structure ∧
  (count_valid_bond_arrangements structure = 267227532) := sorry

end circumcoronene_valid_bonds_l550_550268


namespace ratio_senior_junior_l550_550014

theorem ratio_senior_junior
  (J S : ℕ)
  (h1 : ∃ k : ℕ, S = k * J)
  (h2 : (3 / 8) * S + (1 / 4) * J = (1 / 3) * (S + J)) :
  S = 2 * J :=
by
  -- The proof is to be provided
  sorry

end ratio_senior_junior_l550_550014


namespace constant_term_is_21_l550_550643

def poly1 (x : ℕ) := x^3 + x^2 + 3
def poly2 (x : ℕ) := 2*x^4 + x^2 + 7
def expanded_poly (x : ℕ) := poly1 x * poly2 x

theorem constant_term_is_21 : expanded_poly 0 = 21 := by
  sorry

end constant_term_is_21_l550_550643


namespace number_of_satisfying_subsets_l550_550378

open Set
open Finset

def nonempty_subsets_satisfying_condition : Finset (Finset ℤ) :=
  filter (λ S, S ≠ ∅ ∧ (S.card + S.min' (nonempty_of_ne_empty $ by simp) * S.max' (nonempty_of_ne_empty $ by simp) = 0))
    (powerset (range (-10) 11))

theorem number_of_satisfying_subsets : nonempty_subsets_satisfying_condition.card = 335 := 
  sorry

end number_of_satisfying_subsets_l550_550378


namespace sum_first_2016_terms_l550_550794

noncomputable def sequence (a : ℝ) : ℕ → ℝ
| 0 := 1
| 1 := a
| n+2 := |sequence (n + 1) - sequence n|

lemma sequence_periodic {a : ℝ} (h₀ : a ≤ 1) (h₁ : a ≠ 0) : ∀ n, sequence a (n + 3) = sequence a n :=
sorry

theorem sum_first_2016_terms (a : ℝ) (h₀ : a ≤ 1) (h₁ : a ≠ 0) :
  ∑ i in finset.range 2016, sequence a i = 1344 :=
sorry

end sum_first_2016_terms_l550_550794


namespace probability_multiple_of_3_l550_550673

-- Define the bag of digits
def bag : list ℕ := [0, 2, 3, 5]

-- Define a function to check if the number formed by three digits is a multiple of 3.
def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

-- Define the function to draw three digits and form a valid three-digit number.
def valid_three_digit_numbers (digits : list ℕ) : list ℕ :=
  digits.product digits.product digits 
  |>.filter (λ t, t.0 ≠ 0) -- First digit should not be zero
  |>.map (λ t, t.0 * 100 + t.1.0 * 10 + t.1.1) -- Form the three-digit number

-- Prove that the probability of forming a three-digit number that is a multiple of 3 is 0.
theorem probability_multiple_of_3 : ∀ (valid_numbers : list ℕ), 
  valid_numbers = valid_three_digit_numbers bag →
  (filter is_multiple_of_3 valid_numbers).length / valid_numbers.length = 0 :=
by
  intros valid_numbers h_valid
  sorry

end probability_multiple_of_3_l550_550673


namespace total_time_equiv_l550_550588

-- Define the number of chairs
def chairs := 7

-- Define the number of tables
def tables := 3

-- Define the time spent on each piece of furniture in minutes
def time_per_piece := 4

-- Prove the total time taken to assemble all furniture
theorem total_time_equiv : chairs + tables = 10 ∧ 4 * 10 = 40 := by
  sorry

end total_time_equiv_l550_550588


namespace calc_factorial_sum_l550_550723

theorem calc_factorial_sum : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 = 5040 := by
  sorry

end calc_factorial_sum_l550_550723


namespace students_excelled_in_both_tests_l550_550672

theorem students_excelled_in_both_tests
  (n P I N A : ℕ)
  (h₁ : n = 50)
  (h₂ : P = 40)
  (h₃ : I = 31)
  (h₄ : N = 4)
  (h₅ : P + I - A = n - N) :
  A = 25 :=
by
  sorry

end students_excelled_in_both_tests_l550_550672


namespace earthquake_energy_increase_l550_550358

theorem earthquake_energy_increase (E E1 : ℝ) (M : ℝ)
  (h1 : log10 E = 4.8 + 1.5 * M)
  (h2 : log10 E1 = 4.8 + 1.5 * (M + 1)) :
  E1 = 31.62 * E :=
by sorry

end earthquake_energy_increase_l550_550358


namespace problem_l550_550524

noncomputable def f (x : ℝ) : ℝ := 1 + x - (1 / 2) * x^2

def sequence (x₁ : ℝ) (f : ℝ → ℝ) : ℕ → ℝ
| 0     := x₁
| (n+1) := f (sequence n)

theorem problem (x₁ : ℝ) (h₀ : 1 < x₁ ∧ x₁ < 2) :
  ∀ n ≥ 3, |sequence x₁ f n - real.sqrt 2| < 2^(-n) :=
by
  -- Proof goes here
  sorry

end problem_l550_550524


namespace find_f_neg4_l550_550129

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem find_f_neg4 (a b : ℝ) (h : f a b 4 = 0) : f a b (-4) = 2 := by
  -- sorry to skip the proof
  sorry

end find_f_neg4_l550_550129


namespace perpendicular_segments_between_parallel_lines_equal_l550_550625

theorem perpendicular_segments_between_parallel_lines_equal
  (l₁ l₂ : Line) (h_parallel : l₁ ∥ l₂)
  (h_distance : ∀ p q : Point, perpendicular_segment_length l₁ l₂ p q) :
  ∀ p₁ p₂ : Point, perpendicular_to (line_through p₁ l₁) l₂ → perpendicular_to (line_through p₂ l₁) l₂ → 
  segment_length p₁ l₂ = segment_length p₂ l₂ :=
by
  sorry

end perpendicular_segments_between_parallel_lines_equal_l550_550625


namespace polish_space_defines_borel_space_l550_550582

noncomputable def is_polish_space (S : Type*) [metric_space S] : Prop :=
∃ (ρ : S → S → ℝ) (dense_set : set S), metric_space.bounded ρ ∧ metric_space.complete ρ ∧ set.countable dense_set

def borel_sigma_algebra (S : Type*) [metric_space S] : set (set S) :=
{A | ∃ X, @topological_space.is_open S sorry X → A = topological_space.generate_from X}

theorem polish_space_defines_borel_space (S : Type*) [metric_space S]
  (h : is_polish_space S) : borel_space S := sorry

end polish_space_defines_borel_space_l550_550582


namespace vector_subtraction_result_l550_550414

-- definition of vectors as pairs of integers
def OA : ℝ × ℝ := (1, -2)
def OB : ℝ × ℝ := (-3, 1)

-- definition of vector subtraction for pairs of reals
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- definition of the vector AB as the subtraction of OB and OA
def AB : ℝ × ℝ := vector_sub OB OA

-- statement to assert the expected result
theorem vector_subtraction_result : AB = (-4, 3) :=
by
  -- this is where the proof would go, but we use sorry to skip it
  sorry

end vector_subtraction_result_l550_550414


namespace convex_11_gon_diagonals_angle_le_5_degrees_l550_550229

theorem convex_11_gon_diagonals_angle_le_5_degrees :
  ∀ (P : Type*) [Plane P] (polygon : convex_polygon P 11),
  ∃ (d₁ d₂ : diagonal polygon), 
  angle_between_lines d₁.line d₂.line ≤ 5 :=
by
  sorry

end convex_11_gon_diagonals_angle_le_5_degrees_l550_550229


namespace sharing_bill_evenly_l550_550668

def total_bill : Float := 211.00
def number_of_people : Int := 7
def tip_percentage : Float := 0.15
def each_persons_share : Float := 34.66

theorem sharing_bill_evenly :
  let tip := tip_percentage * total_bill
  let total_with_tip := total_bill + tip
  let share_per_person := total_with_tip / number_of_people.toFloat
  abs (share_per_person - each_persons_share) < 0.01 :=
by
  -- Proof would go here
  sorry

end sharing_bill_evenly_l550_550668


namespace exists_x_in_interval_iff_a_ge_neg8_l550_550467

theorem exists_x_in_interval_iff_a_ge_neg8 (a : ℝ) : 
  (∃ x ∈ set.Icc (1 : ℝ) (2 : ℝ), x^2 + 2 * x + a ≥ 0) ↔ a ≥ -8 := 
by sorry

end exists_x_in_interval_iff_a_ge_neg8_l550_550467


namespace burt_net_profit_l550_550721

theorem burt_net_profit
  (cost_seeds : ℝ := 2.00)
  (cost_soil : ℝ := 8.00)
  (num_plants : ℕ := 20)
  (price_per_plant : ℝ := 5.00) :
  let total_cost := cost_seeds + cost_soil
  let total_revenue := num_plants * price_per_plant
  let net_profit := total_revenue - total_cost
  net_profit = 90.00 :=
by sorry

end burt_net_profit_l550_550721


namespace monic_quadratic_poly_l550_550075

theorem monic_quadratic_poly (x : ℂ) :
  (∃ (P : ℂ[X]), polynomial.monic P ∧ P.coeff 2 = 1 ∧ P.coeff 1 = 6 ∧ P.coeff 0 = 17 ∧ P.eval (-3 - complex.I * real.sqrt 8) = 0 ∧ P.eval (-3 + complex.I * real.sqrt 8) = 0) :=
sorry

end monic_quadratic_poly_l550_550075


namespace last_digit_3_pow_2023_l550_550575

theorem last_digit_3_pow_2023 : Nat.digits 10 (3^2023) % 10 = 7 := by
  sorry

end last_digit_3_pow_2023_l550_550575


namespace find_a3_general_formula_sum_of_first_n_terms_l550_550505

def geom_seq (a : ℕ → ℝ) : Prop :=
  ∃ q, a 1 = 1 ∧ a 4 = 27 ∧ (∀ n, a n = 1 * q^(n-1))

theorem find_a3 (a : ℕ → ℝ) (h : geom_seq a) : a 3 = 9 :=
by {
  sorry
}

theorem general_formula (a : ℕ → ℝ) (h : geom_seq a) : ∀ n, a n = 3^(n-1) :=
by {
  sorry
}

theorem sum_of_first_n_terms (a : ℕ → ℝ) (h : geom_seq a) : ∀ n, 
  (finset.range n).sum (λ k, a (k + 1)) = (3^n - 1) / 2 :=
by {
  sorry
}


end find_a3_general_formula_sum_of_first_n_terms_l550_550505


namespace simplify_and_calculate_expression_l550_550598

theorem simplify_and_calculate_expression (a b : ℤ) (ha : a = -1) (hb : b = -2) :
  (2 * a + b) * (b - 2 * a) - (a - 3 * b) ^ 2 = -25 :=
by 
  -- We can use 'by' to start the proof and 'sorry' to skip it
  sorry

end simplify_and_calculate_expression_l550_550598


namespace correct_conclusions_count_l550_550549

def closest_integer_sqrt (n : ℕ) : ℕ :=
  (Real.sqrt n).round.to_nat

def x_n (n : ℕ) : ℕ := closest_integer_sqrt n

theorem correct_conclusions_count : 
  let statements : List Bool := [
    x_n 5 = 2,
    List.length (List.filter (λ n, x_n n = 4) (List.range 21)) = 6,
    List.sum (List.mapWithIndex (λ n x, if n % 2 = 0 then x else -x) 
      (List.range' 0 21 (λ n, x_n n))) = 0,
    (∑ i in (List.range (2451)), (1 : ℝ) / ↑(x_n (i + 1))) = 98 
    ] in
  List.length (List.filter id statements) = 3 :=
by
  sorry

end correct_conclusions_count_l550_550549


namespace problem1_problem2_l550_550730

-- Proof Problem 1: Prove that (x-y)^2 - (x+y)(x-y) = -2xy + 2y^2
theorem problem1 (x y : ℝ) : (x - y) ^ 2 - (x + y) * (x - y) = -2 * x * y + 2 * y ^ 2 := 
by
  sorry

-- Proof Problem 2: Prove that (12a^2b - 6ab^2) / (-3ab) = -4a + 2b
theorem problem2 (a b : ℝ) (h : -3 * a * b ≠ 0) : (12 * a^2 * b - 6 * a * b^2) / (-3 * a * b) = -4 * a + 2 * b := 
by
  sorry

end problem1_problem2_l550_550730


namespace odd_f_neg1_l550_550541

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 
  if 0 ≤ x 
  then 2^x + 2 * x + b 
  else - (2^(-x) + 2 * (-x) + b)

theorem odd_f_neg1 (b : ℝ) (h : f 0 b = 0) : f (-1) b = -3 :=
by
  sorry

end odd_f_neg1_l550_550541


namespace burt_net_profit_l550_550720

theorem burt_net_profit
  (cost_seeds : ℝ := 2.00)
  (cost_soil : ℝ := 8.00)
  (num_plants : ℕ := 20)
  (price_per_plant : ℝ := 5.00) :
  let total_cost := cost_seeds + cost_soil
  let total_revenue := num_plants * price_per_plant
  let net_profit := total_revenue - total_cost
  net_profit = 90.00 :=
by sorry

end burt_net_profit_l550_550720


namespace shortest_travel_time_on_cube_l550_550011

theorem shortest_travel_time_on_cube :
  ∀ (edge_length : ℝ), edge_length = 1 →
  (∀ (edges_speed : ℝ), edges_speed = 5 →
  ∀ (faces_speed : ℝ), faces_speed = 4 →
  ∀ (interior_speed : ℝ), interior_speed = 3 →
  time_to_travel (vertex A) (vertex B) edges_speed faces_speed interior_speed ≤ (real.sqrt 5 / 4)) :=
by
  intros edge_length h_edge_length 
         edges_speed h_edges_speed 
         faces_speed h_faces_speed 
         interior_speed h_interior_speed 
  sorry

end shortest_travel_time_on_cube_l550_550011


namespace common_tangents_l550_550222

theorem common_tangents (r1 r2 d : ℝ) (h_r1 : r1 = 10) (h_r2 : r2 = 4) : 
  ∀ (n : ℕ), (n = 1) → ¬ (∃ (d : ℝ), 
    (6 < d ∧ d < 14 ∧ n = 2) ∨ 
    (d = 14 ∧ n = 3) ∨ 
    (d < 6 ∧ n = 0) ∨ 
    (d > 14 ∧ n = 4)) :=
by
  intro n h
  sorry

end common_tangents_l550_550222


namespace flight_routes_possible_l550_550787

theorem flight_routes_possible (n : ℕ) (h : n ≥ 5) : 
  ∃ flights : (fin n) → (fin n) → Prop, 
  (∀ i j : fin n, i ≠ j → (flights i j ∨ ∃ k : fin n, flights i k ∧ flights k j)) := 
sorry

end flight_routes_possible_l550_550787


namespace hyperbola_C_eq_correct_l550_550812

noncomputable def equation_of_hyperbola_C (C : Type) : Prop :=
  ∃ (x y : ℝ), (y^2 / 16 - x^2 / 9 = 1) ∧
    (∀ x y : ℝ, (x^2 / 27 - y^2 / 48 = 1) → asymptotes_same (C) (x^2 / 27 - y^2 / 48 = 1)) ∧
    (∀ x y : ℝ, (x^2 / 144 + y^2 / 169 = 1) → foci_same (C) (x^2 / 144 + y^2 / 169 = 1))

theorem hyperbola_C_eq_correct :
  equation_of_hyperbola_C (λ x y : ℝ, (y^2 / 16 - x^2 / 9 = 1)) :=
sorry

end hyperbola_C_eq_correct_l550_550812


namespace problem_statement_l550_550458

noncomputable def f (x a : ℝ) : ℝ := (1 / 2) * x ^ 2 - a * x + a * Real.log x

theorem problem_statement (a x₁ x₂ : ℝ) (hx₁x₂ : x₁ ≠ x₂)
  (h_extreme_pts : (derivative (λ x, f x a)) x₁ = 0 ∧ (derivative (λ x, f x a)) x₂ = 0)
  (hx₁x₂_roots : x₁ + x₂ = a ∧ x₁ * x₂ = a)
  (ha_pos : a > 4) :
  f x₁ a + f x₂ a < (1 / 4) * (x₁ ^ 2 + x₂ ^ 2) - 6 :=
sorry

end problem_statement_l550_550458


namespace backyard_area_proof_l550_550565

-- Condition: Walking the length of 40 times covers 1000 meters
def length_times_40_eq_1000 (L: ℝ) : Prop := 40 * L = 1000

-- Condition: Walking the perimeter 8 times covers 1000 meters
def perimeter_times_8_eq_1000 (P: ℝ) : Prop := 8 * P = 1000

-- Given the conditions, we need to find the Length and Width of the backyard
def is_backyard_dimensions (L W: ℝ) : Prop := 
  length_times_40_eq_1000 L ∧ 
  perimeter_times_8_eq_1000 (2 * (L + W))

-- We need to calculate the area
def backyard_area (L W: ℝ) : ℝ := L * W

-- The theorem to prove
theorem backyard_area_proof (L W: ℝ) 
  (h1: length_times_40_eq_1000 L) 
  (h2: perimeter_times_8_eq_1000 (2 * (L + W))) :
  backyard_area L W = 937.5 := 
  by 
    sorry

end backyard_area_proof_l550_550565


namespace necessary_but_not_sufficient_condition_l550_550151

-- Define the sets M and N and their relationship
variables {α : Type*} (M N : set α)
-- Assume M is non-empty
variable [h1 : set.nonempty M]
-- Assume M is a subset of N
variable [h2 : M ⊆ N]

-- Define a theorem that encapsulates the problem statement
theorem necessary_but_not_sufficient_condition {a : α} :
  (a ∈ M ∨ a ∈ N) → (a ∈ M ∩ N) :=
sorry

end necessary_but_not_sufficient_condition_l550_550151


namespace rachel_weekly_water_goal_l550_550953

variable (glasses_per_day : ℕ) (glass_in_ounces : ℕ) 
variable (X : ℕ) -- Ounces of water Rachel plans to drink on Friday and Saturday

def total_ounces_from_sunday_to_thursday (sunday monday daily : ℕ) : ℕ :=
  sunday + monday + 4 * daily

theorem rachel_weekly_water_goal
  (sunday_glasses monday_glasses daily_glasses : ℕ)
  (one_glass_ounces : ℕ)
  (X : ℕ) :
  glasses_per_day = daily_glasses ->
  glass_in_ounces = one_glass_ounces ->
  sunday_glasses = 2 ->
  monday_glasses = 4 ->
  daily_glasses = 3 ->
  one_glass_ounces = 10 ->
  let total_ounces := (total_ounces_from_sunday_to_thursday sunday_glasses monday_glasses daily_glasses) * one_glass_ounces in
  total_ounces = 180 ->
  W = total_ounces + X :=
by
  intros h1 h2 h3 h4 h5 h6 ht
  sorry

end rachel_weekly_water_goal_l550_550953


namespace distance_between_circle_centers_l550_550138

theorem distance_between_circle_centers :
  ∃ a1 a2 : ℝ, (a1, a1) ∈ {p : ℝ × ℝ | p = (5 + 2 * real.sqrt 2, 5 + 2 * real.sqrt 2) ∨ p = (5 - 2 * real.sqrt 2, 5 - 2 * real.sqrt 2)} ∧ 
  (a2, a2) ∈ {p : ℝ × ℝ | p = (5 + 2 * real.sqrt 2, 5 + 2 * real.sqrt 2) ∨ p = (5 - 2 * real.sqrt 2, 5 - 2 * real.sqrt 2)} ∧ 
  a1 ≠ a2 ∧ 
  real.sqrt ((a1 - a2)^2 + (a1 - a2)^2) = 8 := 
sorry

end distance_between_circle_centers_l550_550138


namespace contrapositive_statement_l550_550253

-- Definitions derived from conditions
def Triangle (ABC : Type) : Prop := 
  ∃ a b c : ABC, true

def IsIsosceles (ABC : Type) : Prop :=
  ∃ a b c : ABC, a = b ∨ b = c ∨ a = c

def InteriorAnglesNotEqual (ABC : Type) : Prop :=
  ∀ a b : ABC, a ≠ b

-- The contrapositive implication we need to prove
theorem contrapositive_statement (ABC : Type) (h : Triangle ABC) 
  (h_not_isosceles_implies_not_equal : ¬IsIsosceles ABC → InteriorAnglesNotEqual ABC) :
  (∃ a b c : ABC, a = b → IsIsosceles ABC) := 
sorry

end contrapositive_statement_l550_550253


namespace total_animals_is_63_l550_550356

def zoo_animals (penguins polar_bears total : ℕ) : Prop :=
  (penguins = 21) ∧
  (polar_bears = 2 * penguins) ∧
  (total = penguins + polar_bears)

theorem total_animals_is_63 :
  ∃ (penguins polar_bears total : ℕ), zoo_animals penguins polar_bears total ∧ total = 63 :=
by {
  sorry
}

end total_animals_is_63_l550_550356


namespace max_value_expression_l550_550148

noncomputable def geometric_mean (x y : ℝ) : ℝ := sqrt (x * y)

theorem max_value_expression (a b : ℝ) 
  (h1 : a = geometric_mean (1 + 2 * b) (1 - 2 * b)) :
  ∃ (c : ℝ), c = (sqrt 2 / 4) ∧ ∀ x y, (x = geometric_mean (1 + 2 * y) (1 - 2 * y)) → ∃ k, 
  k ≤ (sqrt 2 / 4) ∧ k = abs (2 * x * y) / (abs x + 2 * abs y) :=
begin
  admit, -- Placeholder for the proof
end

end max_value_expression_l550_550148


namespace num_integer_solutions_abs_ineq_l550_550847

theorem num_integer_solutions_abs_ineq : 
  ∃ (S : Set ℤ), (∀ (x : ℝ), |x - 3| ≤ 4.5 ↔ ∃ (n: ℤ), n.val = x) ∧ S.card = 9 := 
sorry

end num_integer_solutions_abs_ineq_l550_550847


namespace boxes_sold_l550_550515

theorem boxes_sold (cases boxes_per_case : ℕ) (h_cases : cases = 3) (h_boxes_per_case : boxes_per_case = 8) :
  cases * boxes_per_case = 24 :=
by
  rw [h_cases, h_boxes_per_case]
  norm_num

end boxes_sold_l550_550515


namespace drawing_outcomes_l550_550502

theorem drawing_outcomes (A B : ℕ) (hA : A = 30) (hB : B = 20) :
  (A * (A - 1) * B + B * (B - 1) * A) = 28800 :=
by
  rw [hA, hB]
  sorry

end drawing_outcomes_l550_550502


namespace coefficient_of_ab_is_correct_l550_550859

noncomputable def a : ℝ := 15 / 7
noncomputable def b : ℝ := 15 / 2
noncomputable def ab : ℝ := 674.9999999999999
noncomputable def coeff_ab := ab / (a * b)

theorem coefficient_of_ab_is_correct :
  coeff_ab = 674.9999999999999 / ((15 * 15) / (7 * 2)) := sorry

end coefficient_of_ab_is_correct_l550_550859


namespace philip_paints_2_per_day_l550_550579

def paintings_per_day (initial_paintings total_paintings days : ℕ) : ℕ :=
  (total_paintings - initial_paintings) / days

theorem philip_paints_2_per_day :
  paintings_per_day 20 80 30 = 2 :=
by
  sorry

end philip_paints_2_per_day_l550_550579


namespace cut_wire_l550_550984

theorem cut_wire (x y : ℕ) : 
  15 * x + 12 * y = 102 ↔ (x = 2 ∧ y = 6) ∨ (x = 6 ∧ y = 1) :=
by
  sorry

end cut_wire_l550_550984


namespace pyramid_volume_is_correct_l550_550891

-- Define the problem's conditions
variable (AB BC CG : ℝ)
variable (baseArea height : ℝ)

-- Define the specific values given in the problem
def AB_value : AB := 4
def BC_value : BC := 2
def CG_value : CG := 3

-- Define the centroid and its relation to the height of the pyramid
def centroid_height : height := CG_value

-- Define the area of the base rectangle BCHE
def base_area : baseArea := 2 * 5  -- 10

-- Define the volume of the pyramid
def pyramid_volume : ℝ := (1 / 3) * baseArea * height

-- The theorem to prove
theorem pyramid_volume_is_correct : pyramid_volume = 10 :=
by
  sorry

#eval pyramid_volume = 10

end pyramid_volume_is_correct_l550_550891


namespace suitcase_combination_l550_550959

noncomputable def cast_value (C A S T E : ℕ) : ℕ :=
  C * 12^3 + A * 12^2 + S * 12 + T

noncomputable def eats_value (E A T S : ℕ) : ℕ :=
  E * 12^3 + A * 12^2 + T * 12 + S

noncomputable def tea_value (T E A : ℕ) : ℕ :=
  T * 12^2 + E * 12 + A

noncomputable def seat_value (S E A T : ℕ) : ℕ :=
  S * 12^3 + E * 12^2 + A * 12 + T

theorem suitcase_combination :
  ∃ (C A S T E : ℕ),
    0 < C ∧ 0 < A ∧ 0 < S ∧ 0 < T ∧ 0 < E ∧
    C ≠ A ∧ C ≠ S ∧ C ≠ T ∧ C ≠ E ∧
    A ≠ S ∧ A ≠ T ∧ A ≠ E ∧
    S ≠ T ∧ S ≠ E ∧
    T ≠ E ∧
    cast_value C A S T = 3666 ∧
    cast_value C A S T + eats_value E A T S + tea_value T E A = seat_value S E A T :=
by
  sorry

# The theorem suitcase_combination captures the necessary conditions and asserts that there exist unique digits for C, A, S, T, and E such that the given arithmetic equation holds, and CAST is 3666. The theorem is stated, no proof is provided as instructed.

end suitcase_combination_l550_550959


namespace songs_distribution_l550_550708

-- Define the sets involved
structure Girl := (Amy Beth Jo : Prop)
axiom no_song_liked_by_all : ∀ song : Girl, ¬(song.Amy ∧ song.Beth ∧ song.Jo)
axiom no_song_disliked_by_all : ∀ song : Girl, song.Amy ∨ song.Beth ∨ song.Jo
axiom pairwise_liked : ∀ song : Girl,
  (song.Amy ∧ song.Beth ∧ ¬song.Jo) ∨
  (song.Beth ∧ song.Jo ∧ ¬song.Amy) ∨
  (song.Jo ∧ song.Amy ∧ ¬song.Beth)

-- Define the theorem to prove that there are exactly 90 ways to distribute the songs
theorem songs_distribution : ∃ ways : ℕ, ways = 90 := sorry

end songs_distribution_l550_550708


namespace coprime_permutations_count_l550_550359

theorem coprime_permutations_count :
  let nums := [1, 2, 3, 4, 5, 6, 7]
  (list.countp (λ l : List ℕ, l.adjPairs.All (λ p, Nat.coprime p.1 p.2)) (list.permutations nums)) = 864 :=
by
  sorry

end coprime_permutations_count_l550_550359


namespace no_such_function_exists_l550_550586

theorem no_such_function_exists (f : ℤ → ℤ) : 
  (∀ x : ℤ, f(f(x)) = x + 1) → false :=
by
  sorry

end no_such_function_exists_l550_550586


namespace product_of_slopes_l550_550813

def parabola_focus := C : set (ℝ × ℝ) ∧ F : ℝ × ℝ ∧ (∃(p : ℝ), p > 0 ∧ ∀(x y : ℝ), (x, y) ∈ C ↔ y ^ 2 = 2 * p * x)
def point_M_on_C := M : ℝ × ℝ ∧ x0 : ℝ ∧ (M = (x0, 1)) ∧ ((x0, 1) ∈ C)
def distance_MF := MF : ℝ ∧ (| F.1 - M.1 | ^ 2 + F.2 ^ 2 = (5 * x0 / 4) ^ 2)
def line_l_intersects_C := Q : ℝ × ℝ ∧ (Q = (3, -1)) ∧ l : ℝ → ℝ ∧ ∀ (A B : ℝ × ℝ), (A, B ≠ M) → (A.2 ^ 2 = 2 * p * A.1) ∧ (B.2 ^ 2 = 2 * p * B.1) → (l A.1 = A.2) ∧ (l B.1 = B.2)

theorem product_of_slopes (C : set (ℝ × ℝ)) (F : ℝ × ℝ) (p : ℝ) (h1 : parabola_focus) 
    (M : ℝ × ℝ) (x0 : ℝ) (h2 : point_M_on_C)
    (MF : ℝ) (h3 : distance_MF)
    (Q : ℝ × ℝ) (l : ℝ → ℝ) (h4 : line_l_intersects_C) :
    p = 1 / 2 ∧ ∀ (A B : ℝ × ℝ), (A ≠ M) ∧ (B ≠ M) → (line_l_intersects_C A) ∧ (line_l_intersects_C B) → (slope AM * slope BM = -1 / 2) :=
  sorry

end product_of_slopes_l550_550813


namespace number_of_rhombuses_l550_550887

-- Definition: A grid with 25 small equilateral triangles arranged in a larger triangular pattern
def equilateral_grid (n : ℕ) : Prop :=
  n = 25

-- Theorem: Proving the number of rhombuses that can be formed from the grid
theorem number_of_rhombuses (n : ℕ) (h : equilateral_grid n) : ℕ :=
  30 

-- Main proof statement
example (n : ℕ) (h : equilateral_grid n) : number_of_rhombuses n h = 30 :=
by
  sorry

end number_of_rhombuses_l550_550887


namespace selection_probabilities_not_equal_l550_550234

theorem selection_probabilities_not_equal :
  ∀ (students : Finset ℕ) (exclude : Finset ℕ) (remaining : Finset ℕ),
  students.card = 2010 → exclude.card = 10 → remaining.card = 2000 → 
  ∃ k, systematic_sampling students exclude remaining k ∧ ¬ all_equal_probabilities students exclude remaining :=
  by
  intros students exclude remaining hstudents hexclude hremaining
  sorry

end selection_probabilities_not_equal_l550_550234


namespace choose_starting_team_l550_550945

-- Definitions derived from the conditions
def team_size : ℕ := 18
def selected_goalie (n : ℕ) : ℕ := n
def selected_players (m : ℕ) (k : ℕ) : ℕ := Nat.choose m k

-- The number of ways to choose the starting team
theorem choose_starting_team :
  let n := team_size
  let k := 7
  selected_goalie n * selected_players (n - 1) k = 222768 :=
by
  simp only [team_size, selected_goalie, selected_players]
  sorry

end choose_starting_team_l550_550945


namespace sum_k_over_harmonic_leq_2_sum_l550_550773

theorem sum_k_over_harmonic_leq_2_sum (n : ℕ) (a : ℕ → ℝ) (hn : 0 < n) (ha : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i) : 
  (∑ k in finset.range(n + 1), k / (∑ i in finset.range(k + 1), 1 / a i)) ≤ 2 * (∑ k in finset.range(n + 1), a k) :=
sorry

end sum_k_over_harmonic_leq_2_sum_l550_550773


namespace contrapositive_example_l550_550252

theorem contrapositive_example (x : ℝ) : 
  (x = 1 ∨ x = 2) → (x^2 - 3 * x + 2 ≤ 0) :=
by
  sorry

end contrapositive_example_l550_550252


namespace geom_sequence_sum_correct_l550_550888

noncomputable def geom_sequence_sum (a₁ a₄ : ℕ) (S₅ : ℕ) :=
  ∃ q : ℕ, a₁ = 1 ∧ a₄ = a₁ * q ^ 3 ∧ S₅ = (a₁ * (1 - q ^ 5)) / (1 - q)

theorem geom_sequence_sum_correct : geom_sequence_sum 1 8 31 :=
by {
  sorry
}

end geom_sequence_sum_correct_l550_550888


namespace ann_hill_length_l550_550563

/-- Given the conditions:
1. Mary slides down a hill that is 630 feet long at a speed of 90 feet/minute.
2. Ann slides down a hill at a rate of 40 feet/minute.
3. Ann's trip takes 13 minutes longer than Mary's.
Prove that the length of the hill Ann slides down is 800 feet. -/
theorem ann_hill_length
    (distance_Mary : ℕ) (speed_Mary : ℕ) 
    (speed_Ann : ℕ) (time_diff : ℕ)
    (h1 : distance_Mary = 630)
    (h2 : speed_Mary = 90)
    (h3 : speed_Ann = 40)
    (h4 : time_diff = 13) :
    speed_Ann * ((distance_Mary / speed_Mary) + time_diff) = 800 := 
by
    sorry

end ann_hill_length_l550_550563


namespace smallest_n_extra_special_sum_one_l550_550044

def is_extra_special (x : ℝ) : Prop :=
  ∀ d ∈ (x.to_digits 10), d = 0 ∨ d = 9

theorem smallest_n_extra_special_sum_one :
  ∃ n : ℕ, (∀ xs : list ℝ, (∀ x ∈ xs, is_extra_special x) ∧ xs.sum = 1 → xs.length = n) ∧ n = 9 :=
by
  sorry

end smallest_n_extra_special_sum_one_l550_550044


namespace problem_equiv_proof_l550_550260

noncomputable def number_of_possible_n : ℕ :=
  let log10_15 := Real.log10 15
  let log10_90 := Real.log10 90
  {n : ℕ | let log10_n := Real.log10 n 
           log10_90 + log10_n > log10_15 ∧ 
           log10_15 + log10_90 > log10_n ∧ 
           log10_15 + log10_n > log10_90 ∧ 
           n > 6 ∧ n < 1350}.to_finset.card

theorem problem_equiv_proof : number_of_possible_n = 1343 := by
  sorry

end problem_equiv_proof_l550_550260


namespace sin_double_angle_l550_550415

-- Define the condition
def condition (θ : ℝ) : Prop :=
  sin (θ + π / 4) = 2 / 5

-- Define the theorem to prove
theorem sin_double_angle (θ : ℝ) (h : condition θ) : sin (2 * θ) = -17 / 25 :=
by sorry

end sin_double_angle_l550_550415


namespace cost_of_bananas_and_cantaloupe_l550_550783

theorem cost_of_bananas_and_cantaloupe (a b c d h : ℚ) 
  (h1: a + b + c + d + h = 30)
  (h2: d = 4 * a)
  (h3: c = 2 * a - b) :
  b + c = 50 / 7 := 
sorry

end cost_of_bananas_and_cantaloupe_l550_550783


namespace inclination_angle_proof_l550_550466

noncomputable def proof_problem (a b c : ℝ) (θ : ℝ) :=
  let line := λ (x y: ℝ), a * x + b * y + c = 0
  ∧ (sin θ + cos θ = 0)
  ∧ (θ = real.atan (-1)) -- This derives from the slope being -1

theorem inclination_angle_proof (a b c θ : ℝ) :
  proof_problem a b c θ → a - b = 0 :=
by
  intro h
  rw proof_problem at h
  -- We assume that the rest of the proof steps are proven here.
  sorry

end inclination_angle_proof_l550_550466


namespace abs_inequality_condition_l550_550420

theorem abs_inequality_condition (a : ℝ) : 
  (a < 2) ↔ ∀ x : ℝ, |x - 2| + |x| > a :=
sorry

end abs_inequality_condition_l550_550420


namespace probability_exceeds_175_l550_550594

theorem probability_exceeds_175 (P_lt_160 : ℝ) (P_160_to_175 : ℝ) (h : ℝ) :
  P_lt_160 = 0.2 → P_160_to_175 = 0.5 → 1 - P_lt_160 - P_160_to_175 = 0.3 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end probability_exceeds_175_l550_550594


namespace incorrect_option_is_B_l550_550405

variable (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)

-- Conditions:
axiom S_5_lt_S_6 : S 5 < S 6
axiom S_6_eq_S_7_gt_S : S 6 = S 7 ∧ S 7 > S

-- Defining properties of the sequence:
axiom S_formula : ∀ n, S (n + 1) = S n + a (n + 1)
axiom a_is_arithmetic : ∀ n, a (n + 1) = a n + d

-- Goal: Prove that S 9 < S 5
theorem incorrect_option_is_B : S 9 < S 5 :=
sorry

end incorrect_option_is_B_l550_550405


namespace balloons_remaining_l550_550897

def bags_round : ℕ := 5
def balloons_per_bag_round : ℕ := 20
def bags_long : ℕ := 4
def balloons_per_bag_long : ℕ := 30
def balloons_burst : ℕ := 5

def total_round_balloons : ℕ := bags_round * balloons_per_bag_round
def total_long_balloons : ℕ := bags_long * balloons_per_bag_long
def total_balloons : ℕ := total_round_balloons + total_long_balloons
def balloons_left : ℕ := total_balloons - balloons_burst

theorem balloons_remaining : balloons_left = 215 := by 
  -- We leave this as sorry since the proof is not required
  sorry

end balloons_remaining_l550_550897


namespace card_draw_probability_l550_550999

theorem card_draw_probability :
  let p1 := 12 / 52 * 4 / 51 * 26 / 50
  let p2 := 1 / 52 * 3 / 51 * 26 / 50
  p1 + p2 = 1 / 100 :=
by
  let p1 := (12:ℚ) / 52 * (4:ℚ) / 51 * (26:ℚ) / 50
  let p2 := (1:ℚ) / 52 * (3:ℚ) / 51 * (26:ℚ) / 50
  have p_total : p1 + p2 = 1 / 100 := sorry
  exact p_total

end card_draw_probability_l550_550999


namespace minimum_face_sum_l550_550225

def vertices : Type := Fin 8

def numbersAssignment (v : vertices) : ℕ :=
  match v with
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 2
  | 5 => 6
  | 6 => 7
  | 7 => 8

def faceVertices : Fin 6 → List vertices
  | 0 => [0, 1, 2, 3]
  | 1 => [4, 5, 6, 7]
  | 2 => [0, 1, 4, 5]
  | 3 => [2, 3, 6, 7]
  | 4 => [0, 2, 4, 6]
  | 5 => [1, 3, 5, 7]

def faceSum (f : Fin 6) : ℕ :=
  (faceVertices f).map numbersAssignment |>.sum

theorem minimum_face_sum : ∀ f : Fin 6, faceSum f ≥ 10 :=
  by
    sorry

end minimum_face_sum_l550_550225


namespace true_inverse_of_opposites_true_contrapositive_of_real_roots_l550_550799

theorem true_inverse_of_opposites (X Y : Int) :
  (X = -Y) → (X + Y = 0) :=
by 
  sorry

theorem true_contrapositive_of_real_roots (q : Real) :
  (¬ ∃ x : Real, x^2 + 2*x + q = 0) → (q > 1) :=
by
  sorry

end true_inverse_of_opposites_true_contrapositive_of_real_roots_l550_550799


namespace surface_area_of_sphere_l550_550342

noncomputable def length : ℝ := 3
noncomputable def width : ℝ := 2
noncomputable def height : ℝ := Real.sqrt 3
noncomputable def d : ℝ := Real.sqrt (length^2 + width^2 + height^2)
noncomputable def r : ℝ := d / 2

theorem surface_area_of_sphere :
  4 * Real.pi * r^2 = 14 * Real.pi := by
  sorry

end surface_area_of_sphere_l550_550342


namespace find_ab_if_odd_l550_550868

noncomputable section

open Real

def isOddFunction (f : ℝ → ℝ) : Prop := 
  ∀ x, f (-x) = -f x

def f (a b : ℝ) (x : ℝ) := log a (x + sqrt (b * x^2 + 2 * a^2))

theorem find_ab_if_odd :
  ∀ (a b : ℝ), a > 0 → a ≠ -1 → (isOddFunction (f a b) → (a, b) = (sqrt 2 / 2, 1)) := 
by 
  intros a b ha hne h_odd
  sorry

end find_ab_if_odd_l550_550868


namespace fiona_weekly_earnings_l550_550839

theorem fiona_weekly_earnings :
  let monday_hours := 1.5
  let tuesday_hours := 1.25
  let wednesday_hours := 3.1667
  let thursday_hours := 0.75
  let hourly_wage := 4
  let total_hours := monday_hours + tuesday_hours + wednesday_hours + thursday_hours
  let total_earnings := total_hours * hourly_wage
  total_earnings = 26.67 := by
  sorry

end fiona_weekly_earnings_l550_550839


namespace distance_between_city_centers_l550_550255

theorem distance_between_city_centers :
  let distance_on_map_cm := 55
  let scale_cm_to_km := 30
  let km_to_m := 1000
  (distance_on_map_cm * scale_cm_to_km * km_to_m) = 1650000 :=
by
  sorry

end distance_between_city_centers_l550_550255


namespace find_a_l550_550147

theorem find_a (a : ℝ) (b : ℝ) :
  (9 * x^2 - 27 * x + a = (3 * x + b)^2) → b = -4.5 → a = 20.25 := 
by sorry

end find_a_l550_550147


namespace monic_quadratic_poly_l550_550073

theorem monic_quadratic_poly (x : ℂ) :
  (∃ (P : ℂ[X]), polynomial.monic P ∧ P.coeff 2 = 1 ∧ P.coeff 1 = 6 ∧ P.coeff 0 = 17 ∧ P.eval (-3 - complex.I * real.sqrt 8) = 0 ∧ P.eval (-3 + complex.I * real.sqrt 8) = 0) :=
sorry

end monic_quadratic_poly_l550_550073


namespace average_divisible_by_7_l550_550061

theorem average_divisible_by_7 (numbers : List Nat) (h_subset: ∀ n ∈ numbers, 6 ≤ n ∧ n ≤ 55 ∧ n % 7 = 0):
  List.average numbers = 28 :=
by
  have numbers := [7, 14, 21, 28, 35, 42, 49]
  have h1 : ∀ n ∈ numbers, 6 ≤ n ∧ n ≤ 55 ∧ n % 7 = 0 := by
    intros n hn
    split
    repeat { sorry }
  have h2 : List.sum numbers = 196 := by sorry
  have h3 : List.length numbers = 7 := by sorry
  have h4 : List.average numbers = List.sum numbers / List.length numbers := by sorry
  show List.average numbers = 28
  calc
    List.average numbers = 196 / 7 := by sorry
    ... = 28 := by norm_num

end average_divisible_by_7_l550_550061


namespace balloons_remaining_l550_550898

def bags_round : ℕ := 5
def balloons_per_bag_round : ℕ := 20
def bags_long : ℕ := 4
def balloons_per_bag_long : ℕ := 30
def balloons_burst : ℕ := 5

def total_round_balloons : ℕ := bags_round * balloons_per_bag_round
def total_long_balloons : ℕ := bags_long * balloons_per_bag_long
def total_balloons : ℕ := total_round_balloons + total_long_balloons
def balloons_left : ℕ := total_balloons - balloons_burst

theorem balloons_remaining : balloons_left = 215 := by 
  -- We leave this as sorry since the proof is not required
  sorry

end balloons_remaining_l550_550898


namespace area_of_T_l550_550535

noncomputable def T := {z : ℂ | ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧ z = a + b * (-1/2 + (1/2) * complex.I * real.sqrt 3) + c * (-1/2 - (1/2) * complex.I * real.sqrt 3)}

theorem area_of_T : measure (T : set ℂ) = 6 * real.sqrt 3 :=
sorry

end area_of_T_l550_550535


namespace slope_of_intersection_line_l550_550404

theorem slope_of_intersection_line :
  ∀ t : ℝ, ∃ x y : ℝ, (x + 2 * y = 7 * t + 3) ∧ (x - y = 2 * t - 2) → 
    (∃ m c : ℝ, (∀ t : ℝ,
      let x := (11 * t - 1) / 3 in
      let y := (5 * t + 5) / 3 in
      y = m * x + c) ∧ m = 5 / 11) :=
begin
  intros t,
  use [x, y],
  split,
  { sorry }, -- x + 2y = 7t + 3
  { sorry }, -- x - y = 2t - 2
  split,
  { intros t,
    let x := (11 * t - 1) / 3,
    let y := (5 * t + 5) / 3,
    use [m, c],
    split,
    { sorry }, -- y = mx + c
    { exact 5 / 11 } -- m = 5 / 11
  },
end

end slope_of_intersection_line_l550_550404


namespace inf_sums_not_diffs_l550_550227

theorem inf_sums_not_diffs (n : ℕ) (h : n > 1) : 
  ∃ (S : set ℕ), S ⊆ {x | ∃ a b : ℕ, x = a^n + b^n} ∧ ∀ x ∈ S, ¬ (∃ c d : ℕ, x = c^n - d^n) ∧ set.infinite S :=
sorry

end inf_sums_not_diffs_l550_550227


namespace range_of_m_for_distinct_real_roots_l550_550864

theorem range_of_m_for_distinct_real_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2 * x₁ + m = 0 ∧ x₂^2 + 2 * x₂ + m = 0) ↔ m < 1 :=
by sorry

end range_of_m_for_distinct_real_roots_l550_550864


namespace smallest_b_l550_550769

theorem smallest_b (b : ℝ) : b^2 - 16 * b + 63 ≤ 0 → (∃ b : ℝ, b = 7) :=
sorry

end smallest_b_l550_550769


namespace inequality_solution_l550_550058

theorem inequality_solution :
  {x : ℝ | (3 * x - 8) * (x - 4) / (x - 1) ≥ 0 } = { x : ℝ | x < 1 } ∪ { x : ℝ | x ≥ 4 } :=
by {
  sorry
}

end inequality_solution_l550_550058


namespace problem1_problem2_l550_550371

theorem problem1 : sqrt 16 - (cbrt 8) + sqrt (1 / 9) = 2 + 1 / 3 :=
by
  sorry

theorem problem2 : 3 * sqrt 2 - abs (sqrt 3 - sqrt 2) = 4 * sqrt 2 - sqrt 3 :=
by
  sorry

end problem1_problem2_l550_550371


namespace num_isosceles_triangles_l550_550843

theorem num_isosceles_triangles (a b : ℕ) (h1 : 2 * a + b = 27) (h2 : a > b / 2) : 
  ∃! (n : ℕ), n = 13 :=
by 
  sorry

end num_isosceles_triangles_l550_550843


namespace minimum_cards_to_sum_eleven_l550_550628

theorem minimum_cards_to_sum_eleven (cards : Finset ℕ) 
  (h1 : cards = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) :
  ∃ n, n = 6 ∧ ∀ drawn_cards : Finset ℕ, (drawn_cards.card = n) → 
  (∃ a b ∈ drawn_cards, a + b = 11) :=
by
  sorry

end minimum_cards_to_sum_eleven_l550_550628


namespace space_mission_contribution_l550_550860

theorem space_mission_contribution 
  (mission_cost_million : ℕ := 30000) 
  (combined_population_million : ℕ := 350) : 
  mission_cost_million / combined_population_million = 86 := by
  sorry

end space_mission_contribution_l550_550860


namespace original_photo_dimensions_l550_550409

theorem original_photo_dimensions (squares_before : ℕ) 
    (squares_after : ℕ) 
    (vertical_length : ℕ) 
    (horizontal_length : ℕ) 
    (side_length : ℕ)
    (h1 : squares_before = 1812)
    (h2 : squares_after = 2018)
    (h3 : side_length = 1) :
    vertical_length = 101 ∧ horizontal_length = 803 :=
by
    sorry

end original_photo_dimensions_l550_550409


namespace negation_sin_angle_l550_550620

theorem negation_sin_angle (A B C : ℝ) (h_triangle : A + B + C = π) (h_AgtB : A > B) :
  ¬(A > B → sin A > sin B) ↔ (A > B → sin A ≤ sin B) := by
  intro h
  sorry

end negation_sin_angle_l550_550620


namespace problem_1_problem_2_l550_550830

-- Define the given line l
def line_l (t : Real) : Real × Real :=
  (2 + (1 / 2) * t, Real.sqrt 3 + (Real.sqrt 3 / 2) * t)

-- Define the curve C in polar coordinates
def curve_C_polar (ρ θ : Real) : Prop :=
  ρ = 2

-- Define the point M
def point_M : Real × Real := (2, Real.sqrt 3)

-- Define the rectangular equation of the curve C
def curve_C_rect (x y : Real) : Prop :=
  x^2 + y^2 = 4

-- Problem 1: Prove |MA| + |MB| = sqrt 13
theorem problem_1 (A B : Real × Real) (t1 t2 : Real)
  (hA : A = line_l t1)
  (hB : B = line_l t2)
  (hC_A : curve_C_rect A.1 A.2)
  (hC_B : curve_C_rect B.1 B.2) :
  (Real.sqrt ((t1 + t2)^2 - 4 * t1 * t2)) = Real.sqrt 13 :=
sorry

-- Define the transformation for curve C'
def transformation (x y : Real) : Real × Real :=
  (Real.sqrt 3 * x, y)

-- Define the equation of curve C' after the transformation
def curve_C'_rect (x' y' : Real) : Prop :=
  (1 / 12) * x'^2 + (1 / 3) * y'^2 = 1

-- Problem 2: Prove maximum perimeter of inscribed rectangle of curve C' is 16
theorem problem_2 : ∃ (x' y' : Real), curve_C'_rect x' y' ∧
  (4 * x' + 4 * y') = 16 :=
sorry

end problem_1_problem_2_l550_550830


namespace limit_of_binomial_coefficients_l550_550870

theorem limit_of_binomial_coefficients (a_n b_n : ℕ → ℝ) (h_a : ∀ n : ℕ, a_n = 3^n) (h_b : ∀ n : ℕ, b_n = 2^n) :
  tendsto (λ n, (b_n (n + 1) - a_n n) / (a_n (n + 1) + b_n n)) at_top (𝓝 (-1 / 3)) :=
by
  sorry

end limit_of_binomial_coefficients_l550_550870


namespace f_is_increasing_exists_ratio_two_range_k_for_negative_two_ratio_l550_550105

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (k x : ℝ) : ℝ := x^2 + k * x
noncomputable def a (x1 x2 : ℝ) : ℝ := (f x1 - f x2) / (x1 - x2)
noncomputable def b (z1 z2 k : ℝ) : ℝ := (g k z1 - g k z2) / (z1 - z2)

theorem f_is_increasing (x1 x2 : ℝ) (h : x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0) : a x1 x2 > 0 := by
  sorry

theorem exists_ratio_two (k : ℝ) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a x1 x2 ≠ 0 ∧ b x1 x2 k = 2 * a x1 x2 := by
  sorry

theorem range_k_for_negative_two_ratio (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a x1 x2 ≠ 0 ∧ b x1 x2 k = -2 * a x1 x2) → k < -4 := by
  sorry

end f_is_increasing_exists_ratio_two_range_k_for_negative_two_ratio_l550_550105


namespace botanical_garden_correct_path_length_l550_550676

noncomputable def correct_path_length_on_ground
  (inch_length_on_map : ℝ)
  (inch_per_error_segment : ℝ)
  (conversion_rate : ℝ) : ℝ :=
  (inch_length_on_map * conversion_rate) - (inch_per_error_segment * conversion_rate)

theorem botanical_garden_correct_path_length :
  correct_path_length_on_ground 6.5 0.75 1200 = 6900 := 
by
  sorry

end botanical_garden_correct_path_length_l550_550676


namespace sequence_is_integer_sequence_divisible_by_3_l550_550583

def a_n (n : ℕ) : ℤ :=
  (Real.sqrt 3 * (2 + Real.sqrt 3)^(n : ℤ) - Real.sqrt 3 * (2 - Real.sqrt 3)^(n : ℤ)) / (2 * Real.sqrt 3)

theorem sequence_is_integer (n : ℕ) : a_n n ∈ ℤ := sorry

theorem sequence_divisible_by_3 (n : ℕ) : (3 ∣ a_n n) ↔ (3 ∣ n) := sorry

end sequence_is_integer_sequence_divisible_by_3_l550_550583


namespace widgets_difference_l550_550386

-- Definitions according to the conditions
def widgets_monday (w t : ℕ) : ℕ := w * t
def widgets_tuesday (w t : ℕ) : ℕ := (w + 5) * (t - 3)

-- The formal statement that needs to be proved
theorem widgets_difference (t : ℕ) (h : t ≥ 3) : 
  let w := 3 * t in
  widgets_monday w t - widgets_tuesday w t = 4 * t + 15 :=
by
  sorry

end widgets_difference_l550_550386


namespace painting_faces_l550_550012

theorem painting_faces (d : Finset ℕ) (h : d = {1, 2, 3, 4, 5, 6}) : 
  ∃ n : ℕ, n = 12 ∧ ∀ (f1 f2 : ℕ), f1 ∈ d → f2 ∈ d → f1 ≠ f2 → 
  (f1 + f2 ≠ 7 → (count_faces f1 f2 = n)) := 
sorry

noncomputable def count_faces (f1 f2 : ℕ) : ℕ := 
if f1 + f2 ≠ 7 then 1 else 0


end painting_faces_l550_550012


namespace max_c_satisfies_conditions_l550_550763

noncomputable def max_c : ℝ :=
  Real.sqrt 6 / 6

theorem max_c_satisfies_conditions 
  (n : ℕ) (h : n ≥ 3)
  (A : Fin n → Set ℝ)
  (h₁ : ∀ i j k : Fin n, 
    i.val < j.val → j.val < k.val → 
    (A i ∩ A j ∩ A k).Nonempty → 
    True)
  (h₂ : n ≥ 1 / 2 * (choose n 3)) :
  ∃ I : Finset (Fin n), 
    I.card > max_c * n ∧ 
    (⋂ i ∈ I, A i).Nonempty :=
sorry

end max_c_satisfies_conditions_l550_550763


namespace minimum_n_for_factorable_polynomial_l550_550400

theorem minimum_n_for_factorable_polynomial :
  ∃ n : ℤ, (∀ A B : ℤ, 5 * A = 48 → 5 * B + A = n) ∧
  (∀ k : ℤ, (∀ A B : ℤ, 5 * A * B = 48 → 5 * B + A = k) → n ≤ k) :=
by
  sorry

end minimum_n_for_factorable_polynomial_l550_550400


namespace domain_of_f_parity_of_f_range_of_m_l550_550456

noncomputable def f (x : ℝ) : ℝ := real.log (1 + x) / real.log 2 + real.log (1 - x) / real.log 2

theorem domain_of_f : ∀ x, -1 < x ∧ x < 1 ↔ (∃ y, f y = f x) := sorry

theorem parity_of_f : ∀ x, f (-x) = f x := sorry

theorem range_of_m (m : ℝ) : (∀ x, f x ≤ m) ↔ m ≥ 0 := sorry

end domain_of_f_parity_of_f_range_of_m_l550_550456


namespace integer_a_value_l550_550775

/-- For the given system of inequalities and provided sum condition, prove that the correct value of 'a' is 2 -/
theorem integer_a_value 
  (a : ℤ)
  (h1 : ∀ x : ℤ, 6*x + 3 > 3*(x + a))
  (h2 : ∀ x : ℤ, x / 2 - 1 ≤ 7 - (3 / 2) * x)
  (h3 : ∑ (x : ℤ) in (finset.filter (λ x, x > a - 1 ∧ x ≤ 4) (finset.range 10)), x = 9) :
  a = 2 := 
sorry

end integer_a_value_l550_550775


namespace range_of_b_l550_550881

theorem range_of_b (A B C a b c : ℝ) (h_triangle : a ^ 2 + b ^ 2 > c ^ 2 ∧ b ^ 2 + c ^ 2 > a ^ 2 ∧ c ^ 2 + a ^ 2 > b ^ 2)
  (ha : a = 1) (hB : B = 60) (h_acute : 0 < A ∧ A < 90 ∧ 0 < B ∧ B < 90 ∧ 0 < C ∧ C < 90) :
  ∃ l u, l = (sqrt 3 / 2) ∧ u = sqrt 3 ∧ l < b ∧ b < u :=
by
  sorry

end range_of_b_l550_550881


namespace linear_correlation_test_l550_550578

theorem linear_correlation_test (n1 n2 n3 n4 : ℕ) (r1 r2 r3 r4 : ℝ) :
  n1 = 10 ∧ r1 = 0.9533 →
  n2 = 15 ∧ r2 = 0.3012 →
  n3 = 17 ∧ r3 = 0.9991 →
  n4 = 3  ∧ r4 = 0.9950 →
  abs r1 > abs r2 ∧ abs r3 > abs r4 →
  (abs r1 > abs r2 → abs r1 > abs r4) →
  (abs r3 > abs r2 → abs r3 > abs r4) →
  abs r1 ≠ abs r2 →
  abs r3 ≠ abs r4 →
  true := 
sorry

end linear_correlation_test_l550_550578


namespace ellipse_standard_equation_and_intersection_point_l550_550798

theorem ellipse_standard_equation_and_intersection_point (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
(F : ℝ × ℝ) (hF : F = (-1, 0)) (h3 : (a - 1) * (a + 1) = 5) :
  ∃ (a b : ℝ), (a^2 = 6 ∧ b^2 = 5 ∧ (∀ l₁ l₂ : ℝ → ℝ, ∃ M N : ℝ × ℝ, 
  (F.2 + l₁ F.1 = 0 ∧ F.2 + l₂ F.1 = 0 ∧ l₁ * l₂ = -1) → 
  (∃ fixed_point : ℝ × ℝ, fixed_point = (-6/11, 0) ∧ 
  (∃ MN : ℝ → ℝ, ∀ x : ℝ, MN x = 0 → x = fixed_point.1)))) ∧ 
  ∃ C : (ℝ × ℝ) → Prop, C = (λ p : ℝ × ℝ, p.1^2 / 6 + p.2^2 / 5 = 1)) :=
sorry

end ellipse_standard_equation_and_intersection_point_l550_550798


namespace find_z_l550_550785

open Complex

noncomputable def z_proof : Prop :=
  let z1 := 5 + 10 * I
  let z2 := 3 - 4 * I
  let inv_z1 := 1 / z1
  let inv_z2 := 1 / z2
  let inv_z := inv_z1 + inv_z2
  let z := 1 / inv_z
  z = 5 - (5 / 2) * I

theorem find_z : z_proof :=
by
  let z1 := 5 + 10 * I
  let z2 := 3 - 4 * I
  let inv_z1 := 1 / z1
  let inv_z2 := 1 / z2
  let inv_z := inv_z1 + inv_z2
  let z := 1 / inv_z
  show z = 5 - (5 / 2) * I from sorry

end find_z_l550_550785


namespace number_wall_value_l550_550880

theorem number_wall_value (x : ℤ) 
  (hlevel1 : (x + 5) = 18) 
  (hlevel2 : (x + 20) + 34 = 72) 
  (hlevel3 : 34 + 34 = 68) :
  x = -50 := 
by
  -- Given Top Level Equation
  have htop : x + 122 = 72 := 
    by 
      rw [←hlevel2, add_assoc, add_comm 34, add_assoc, hlevel3, add_comm 68]
  -- Solving for x
  linarith [htop]

end number_wall_value_l550_550880


namespace joey_read_percentage_l550_550519

theorem joey_read_percentage : 
  ∀ (total_pages read_after_break : ℕ), 
  total_pages = 30 → read_after_break = 9 → 
  ( (total_pages - read_after_break : ℕ) / (total_pages : ℕ) * 100 ) = 70 :=
by
  intros total_pages read_after_break h_total h_after
  sorry

end joey_read_percentage_l550_550519


namespace sin_shift_identity_l550_550118

theorem sin_shift_identity (α : ℝ) 
  (h1 : α > π / 2 ∧ α < π) 
  (h2 : Real.sin (-π - α) = sqrt 5 / 5) : 
  Real.sin (α - 3 * π / 2) = - (2 * sqrt 5) / 5 := 
sorry

end sin_shift_identity_l550_550118


namespace diagonal_count_l550_550086

-- Define the lengths of the sides of the quadrilateral
def a : ℕ := 7
def b : ℕ := 9
def c : ℕ := 15
def d : ℕ := 10

-- Define the diagonal length x
def x : ℕ

-- Formalize the conditions based on the triangle inequality
def triangle_inequality_abc : Prop := a + b > x ∧ x + a > b ∧ x + b > a
def triangle_inequality_cda : Prop := c + d > x ∧ x + c > d ∧ x + d > c

-- Combine the conditions
def valid_diagonal_length : Prop := 
  -- From triangle ABC
  2 < x ∧ x < 16 ∧ 
  -- From triangle CDA
  5 < x ∧ x < 25

-- Prove the final statement
theorem diagonal_count : ∃ n, n = 10 ∧ (∀ x : ℕ, valid_diagonal_length → 6 ≤ x ∧ x ≤ 15) :=
by
  sorry

end diagonal_count_l550_550086


namespace quadratic_inequality_solution_l550_550989

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 < x + 6) ↔ (-2 < x ∧ x < 3) := 
by
  sorry

end quadratic_inequality_solution_l550_550989


namespace top_leftmost_rectangle_is_B_l550_550750

-- Define the sides of the rectangles
structure Rectangle :=
  (w : ℕ)
  (x : ℕ)
  (y : ℕ)
  (z : ℕ)

-- Define the specific rectangles with their side values
noncomputable def rectA : Rectangle := ⟨2, 7, 4, 7⟩
noncomputable def rectB : Rectangle := ⟨0, 6, 8, 5⟩
noncomputable def rectC : Rectangle := ⟨6, 3, 1, 1⟩
noncomputable def rectD : Rectangle := ⟨8, 4, 0, 2⟩
noncomputable def rectE : Rectangle := ⟨5, 9, 3, 6⟩
noncomputable def rectF : Rectangle := ⟨7, 5, 9, 0⟩

-- Prove that Rectangle B is the top leftmost rectangle
theorem top_leftmost_rectangle_is_B :
  (rectB.w = 0 ∧ rectB.x = 6 ∧ rectB.y = 8 ∧ rectB.z = 5) :=
by {
  sorry
}

end top_leftmost_rectangle_is_B_l550_550750


namespace shortest_side_length_triangle_l550_550493

noncomputable def triangle_min_angle_side_length (A B : ℝ) (c : ℝ) (tanA tanB : ℝ) (ha : tanA = 1 / 4) (hb : tanB = 3 / 5) (hc : c = Real.sqrt 17) : ℝ :=
   Real.sqrt 2

theorem shortest_side_length_triangle {A B c : ℝ} {tanA tanB : ℝ} 
  (ha : tanA = 1 / 4) (hb : tanB = 3 / 5) (hc : c = Real.sqrt 17) :
  triangle_min_angle_side_length A B c tanA tanB ha hb hc = Real.sqrt 2 :=
sorry

end shortest_side_length_triangle_l550_550493


namespace combined_mean_of_sets_l550_550982

theorem combined_mean_of_sets :
  ∀ (S1 S2 : list ℝ), 
    (S1.length = 7) → (S2.length = 8) →
    (S1.sum / 7 = 15) → (S2.sum / 8 = 20) →
    ((S1 ++ S2).sum / (S1.length + S2.length) = 265 / 15) :=
by
  intros S1 S2 h_len1 h_len2 h_mean1 h_mean2
  sorry -- proof will be written here

end combined_mean_of_sets_l550_550982


namespace vector_subtraction_example_l550_550023

theorem vector_subtraction_example :
  let v1 := (3, -7)
  let scalar := 3
  let v2 := (2, -4)
  (v1.fst - scalar * v2.fst, v1.snd - scalar * v2.snd) = (-3, 5) :=
by
  let v1 := (3, -7)
  let scalar := 3
  let v2 := (2, -4)
  calc
    (v1.fst - scalar * v2.fst, v1.snd - scalar * v2.snd)
        = (3 - 3 * 2, -7 - 3 * (-4)) : by rfl
    ... = (-3, 5) : by rfl

end vector_subtraction_example_l550_550023


namespace bernardo_has_higher_probability_l550_550019

-- Define the sets from which Bernardo and Silvia pick their numbers
def bernardoSet := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def silviaSet := {1, 2, 3, 4, 5, 6, 7}

-- Define the conditions under which numbers are picked
def pickDistinctNumbers (s : Set ℕ) (n : ℕ) : Set (Set ℕ) :=
  { t | t ⊆ s ∧ t.card = n }

-- Define the probability calculation for the scenario
noncomputable def calculateProbability : ℚ :=
  let bernardoPicks := pickDistinctNumbers bernardoSet 3
  let silviaPicks := pickDistinctNumbers silviaSet 3
  let bernardoWins := { (b, s) | b ∈ bernardoPicks ∧ s ∈ silviaPicks ∧ b > s }
  bernardoWins.card.to_rat / (bernardoPicks.card * silviaPicks.card).to_rat

-- Statement of the problem to be proved
theorem bernardo_has_higher_probability : calculateProbability = 23 / 30 :=
sorry

end bernardo_has_higher_probability_l550_550019


namespace max_parts_from_blanks_l550_550497

-- Define the initial conditions and question as constants
constant initial_blanks : ℕ := 20
constant usage_fraction : ℚ := 2 / 3
constant waste_fraction : ℚ := 1 / 3

-- Define the maximum number of parts that can be produced and the remaining waste
constant max_parts_produced : ℕ := 29
constant remaining_waste : ℚ := 1 / 3

-- State the main theorem
theorem max_parts_from_blanks :
  (initial_blanks → (usage_fraction = 2 / 3) → (waste_fraction = 1 / 3) → max_parts_produced = 29 ∧ remaining_waste = 1 / 3) :=
  sorry

end max_parts_from_blanks_l550_550497


namespace long_furred_brown_count_l550_550310

variable (Dogs : Type) [Fintype Dogs]

-- Definitions based on conditions
variable (kennel_dogs : Finset Dogs) (long_furred_dogs : Finset Dogs) (brown_dogs : Finset Dogs)
variable (total_dogs : ℕ := 45) (long_fur_count : ℕ := 26) (brown_count : ℕ := 30) (neither_count : ℕ := 8)

-- Conditions
def kennel_non_empty: kennel_dogs.card = total_dogs := by sorry
def long_fur_count_condition : long_furred_dogs.card = long_fur_count := by sorry
def brown_count_condition : brown_dogs.card = brown_count := by sorry
def neither_condition: total_dogs - ((long_furred_dogs ∪ brown_dogs).card) = neither_count := by sorry

-- Prove the number of long-furred dogs that are brown
theorem long_furred_brown_count: (long_furred_dogs ∩ brown_dogs).card = 19 := by 
  sorry

end long_furred_brown_count_l550_550310


namespace triangle_BFD_ratio_l550_550501

theorem triangle_BFD_ratio (x : ℝ) : 
  let AF := 3 * x
  let FE := x
  let ED := x
  let DC := 3 * x
  let side_square := AF + FE
  let area_square := side_square^2
  let area_triangle_BFD := area_square - (1/2 * AF * side_square + 1/2 * side_square * FE + 1/2 * ED * DC)
  (area_triangle_BFD / area_square) = 7 / 16 := 
by
  sorry

end triangle_BFD_ratio_l550_550501


namespace relationship_between_m_and_n_l550_550486

theorem relationship_between_m_and_n
  (m n : ℝ)
  (circle_eq : ∀ (x y : ℝ), x^2 + y^2 - 4 * x + 2 * y - 4 = 0)
  (line_eq : ∀ (x y : ℝ), m * x + 2 * n * y - 4 = 0) :
  m - n - 2 = 0 := 
  sorry

end relationship_between_m_and_n_l550_550486


namespace marbles_left_l550_550936

def initial_marbles : ℝ := 150
def lost_marbles : ℝ := 58.5
def given_away_marbles : ℝ := 37.2
def found_marbles : ℝ := 10.8

theorem marbles_left :
  initial_marbles - lost_marbles - given_away_marbles + found_marbles = 65.1 :=
by 
  sorry

end marbles_left_l550_550936


namespace pauls_weekly_spending_l550_550224

def mowing_lawns : ℕ := 3
def weed_eating : ℕ := 3
def total_weeks : ℕ := 2
def total_money : ℕ := mowing_lawns + weed_eating
def spending_per_week : ℕ := total_money / total_weeks

theorem pauls_weekly_spending : spending_per_week = 3 := by
  sorry

end pauls_weekly_spending_l550_550224


namespace sum_of_consecutive_integers_product_2730_eq_42_l550_550265

theorem sum_of_consecutive_integers_product_2730_eq_42 :
  ∃ x : ℤ, x * (x + 1) * (x + 2) = 2730 ∧ x + (x + 1) + (x + 2) = 42 :=
by
  sorry

end sum_of_consecutive_integers_product_2730_eq_42_l550_550265


namespace slope_angle_of_line_x_equal_one_l550_550162

noncomputable def slope_angle_of_vertical_line : ℝ := 90

theorem slope_angle_of_line_x_equal_one : slope_angle_of_vertical_line = 90 := by
  sorry

end slope_angle_of_line_x_equal_one_l550_550162


namespace frustum_volume_is_correct_l550_550123

noncomputable def volume_of_frustum
  (r_top r_bottom : ℝ) 
  (h : ℝ) 
  (S_top S_bottom : ℝ) : ℝ :=
  (1 / 3) * (S_top + sqrt (S_top * S_bottom) + S_bottom) * h

theorem frustum_volume_is_correct :
  let r_top := sqrt 3 in
  let r_bottom := 3 * sqrt 3 in
  let h := 6 in
  let S_top := π * r_top^2 in
  let S_bottom := π * r_bottom^2 in
  volume_of_frustum r_top r_bottom h S_top S_bottom = 78 * π :=
by
  -- Proof skipped
  sorry

end frustum_volume_is_correct_l550_550123


namespace sum_closest_integer_sqrt_l550_550906

def closest_integer_to_sqrt (n : ℕ) : ℕ := 
  nat.floor (real.sqrt n + 0.5)

theorem sum_closest_integer_sqrt : 
  (finset.range 1980).sum (λ k, (closest_integer_to_sqrt (k+1))⁻¹) = 88 :=
sorry

end sum_closest_integer_sqrt_l550_550906


namespace distance_traveled_downstream_is_72_l550_550674

-- Conditions
def speed_boat_still_water := 14 -- speed in km/hr
def speed_stream := 6 -- speed in km/hr
def time_downstream := 3.6 -- time in hours

-- Effective speed downstream: sum of speed in still water and speed of the stream
def effective_speed_downstream := speed_boat_still_water + speed_stream

-- Distance traveled downstream: effective speed multiplied by time
def distance_downstream := effective_speed_downstream * time_downstream

-- Statement to be proved
theorem distance_traveled_downstream_is_72 :
  distance_downstream = 72 := 
  by 
    sorry

end distance_traveled_downstream_is_72_l550_550674


namespace smallest_number_of_students_l550_550495

theorem smallest_number_of_students 
  (n : ℕ)
  (h1 : 6 * 95 = 570)
  (h2 : ∀ i, i >= 1 → i <= n → score i ≥ 70)
  (h3 : (Σ i in range n, score i) / n = 80) :
  n = 15 :=
sorry

end smallest_number_of_students_l550_550495


namespace find_z_and_conjugate_find_modulus_l550_550818

noncomputable def z (b : ℝ) : ℂ := 3 + b * complex.I

noncomputable def purely_imaginary (z : ℂ) : Prop :=
  ∃ b : ℝ, z = 3 + b * complex.I ∧ (1 + 3 * complex.I) * z = 0 + (9 + b) * complex.I

theorem find_z_and_conjugate (b : ℝ) (z = z b) :
  (purely_imaginary z) → (z = 3 + complex.I ∧ z.conj = 3 - complex.I) :=
begin
  sorry
end

noncomputable def omega (z : ℂ) : ℂ := z / (2 + complex.I)
noncomputable def modulus (z : ℂ) : ℝ := complex.abs (omega z)

theorem find_modulus (z : ℂ) (h : z = 3 + complex.I):
  (purely_imaginary z) → (modulus z = real.sqrt 2) :=
begin
  sorry
end

end find_z_and_conjugate_find_modulus_l550_550818


namespace product_of_divisors_has_3_prime_factors_l550_550909

-- Definitions based only on conditions given in the problem
def is_product_of_divisors (n A : ℕ) : Prop :=
  A = ∏ d in (finset.filter (λ x, n ∣ x) (finset.range (n+1))), d

-- Definition of the problem conditions
def n : ℕ := 30

def A : ℕ := ∏ d in (finset.filter (λ x, d ∣ n) (finset.range (n+1))), d

/-- Test for the number of distinct prime factors of A -/
theorem product_of_divisors_has_3_prime_factors : 
  (∀ d ∈ (finset.filter (λ x, d ∣ n) (finset.range (n+1))), d ∈ {1, 2, 3, 5, 6, 10, 15, 30}) → -- 1
  (A = ∏ d in (finset.filter (λ x, d ∣ n) (finset.range (n+1))), d) → -- 2
  (countp is_prime A) = 3 := -- 3
by
  intros divs_prod fits
  -- This would be the proof step
  sorry

end product_of_divisors_has_3_prime_factors_l550_550909


namespace probability_two_balls_same_mass_l550_550494

-- Definitions of masses of balls numbered 1 to 5
def mass (x : ℕ) : ℕ := x ^ 2 - 5 * x + 30

-- Precomputed masses
def masses : List ℕ := List.map mass [1, 2, 3, 4, 5]

-- Predicate for checking if two balls have the same mass
def same_mass (a b : ℕ) : Prop := mass a = mass b

-- Total number of ways to draw two balls out of five
def total_combinations : ℕ := Nat.choose 5 2

-- Number of favorable outcomes
def favorable_combinations : ℕ := (if same_mass 1 4 then 1 else 0) + (if same_mass 2 3 then 1 else 0)

-- Probability that two drawn balls have the same mass
def probability_same_mass : ℚ := favorable_combinations / total_combinations

-- Lean statement for the proof problem
theorem probability_two_balls_same_mass :
  probability_same_mass = 1 / 5 := by
  unfold probability_same_mass favorable_combinations same_mass total_combinations mass masses
  have : total_combinations = 10 := by simp [Nat.choose]
  have : favorable_combinations = 2 := by simp [same_mass]; split_ifs; simp
  field_simp [this]
  norm_num


end probability_two_balls_same_mass_l550_550494


namespace isosceles_triangles_with_perimeter_27_l550_550846

theorem isosceles_triangles_with_perimeter_27 :
  ∃ n : ℕ, n = 6 ∧ ∀ a b c : ℕ, (a = b ∧ a > 0 ∧ 2 * a + c = 27 ∧ c mod 2 = 1) → n = 6 :=
by
  sorry

end isosceles_triangles_with_perimeter_27_l550_550846


namespace part_a_part_b_l550_550662

theorem part_a (c d : ℕ) : ¬ (c * (c + 1) = d * (d + 2)) :=
  by sorry
  
theorem part_b (m n : ℕ) :
  ¬ (m^4 + (m+1)^4 = n^2 + (n+1)^2) :=
  by
    have h1 : ¬ ∃ c d : ℕ, c * (c + 1) = d * (d + 2), from part_a,
    sorry

end part_a_part_b_l550_550662


namespace triangular_pyramid_range_l550_550736

theorem triangular_pyramid_range (c : ℝ) :
  (c > (Math.sqrt 5 - 1) / 2) ∧ (c < (Math.sqrt 5 + 1) / 2) ↔
  (
    ∀ (A B C D : ℝ), 
    BC = 1 ∧ CA = 1 ∧ AB = c ∧ DA = 1 ∧ DB = c ∧ DC = c →
    (
      A + B > C ∧ A + C > B ∧ B + C > A
    )
  ) :=
sorry

end triangular_pyramid_range_l550_550736


namespace max_length_slant_edge_l550_550434

theorem max_length_slant_edge {S A B C D : ℝ^3} 
(height : ℝ) (side_length : ℝ) (radius : ℝ) 
(h_height : height = sqrt 2 / 4)
(h_side : side_length = 1)
(h_radius : radius = 1)
(h_S_on_sphere : ∥S∥ = radius)
(h_A_on_sphere : ∥A∥ = radius)
(h_B_on_sphere : ∥B∥ = radius)
(h_C_on_sphere : ∥C∥ = radius)
(h_D_on_sphere : ∥D∥ = radius)
(h_base_square : |A - B| = side_length ∧ |B - C| = side_length ∧ |C - D| = side_length ∧ |D - A| = side_length) : 
  ∃ SA_max, SA_max = sqrt (6 + 2 * sqrt 7) / 2 := 
sorry

end max_length_slant_edge_l550_550434


namespace natural_solutions_3x_4y_eq_12_l550_550398

theorem natural_solutions_3x_4y_eq_12 :
  ∃ x y : ℕ, (3 * x + 4 * y = 12) ∧ ((x = 4 ∧ y = 0) ∨ (x = 0 ∧ y = 3)) := 
sorry

end natural_solutions_3x_4y_eq_12_l550_550398


namespace number_of_non_officers_l550_550248

theorem number_of_non_officers 
  (avg_salary_employees: ℝ) (avg_salary_officers: ℝ) (avg_salary_nonofficers: ℝ) 
  (num_officers: ℕ) (num_nonofficers: ℕ):
  avg_salary_employees = 120 ∧ avg_salary_officers = 440 ∧ avg_salary_nonofficers = 110 ∧
  num_officers = 15 ∧ 
  (15 * 440 + num_nonofficers * 110 = (15 + num_nonofficers) * 120)  → 
  num_nonofficers = 480 := 
by 
sorry

end number_of_non_officers_l550_550248


namespace simplify_sum_l550_550660

theorem simplify_sum (n : ℕ) :
  let S_n := n + (n-1) * 2 + (n-2) * 2^2 + ∙∙∙ + 2 * 2^(n-2) + 2^(n-1)
  in S_n = 2^(n + 1) - n - 2 := by
  sorry

end simplify_sum_l550_550660


namespace beth_lost_red_marbles_l550_550020

-- Definitions from conditions
def total_marbles : ℕ := 72
def marbles_per_color : ℕ := total_marbles / 3
variable (R : ℕ)  -- Number of red marbles Beth lost
def blue_marbles_lost : ℕ := 2 * R
def yellow_marbles_lost : ℕ := 3 * R
def marbles_left : ℕ := 42

-- Theorem we want to prove
theorem beth_lost_red_marbles (h : total_marbles - (R + blue_marbles_lost R + yellow_marbles_lost R) = marbles_left) :
  R = 5 :=
by
  sorry

end beth_lost_red_marbles_l550_550020


namespace natural_numbers_satisfy_equation_l550_550759

theorem natural_numbers_satisfy_equation:
  ∀ (n k : ℕ), (k^5 + 5 * n^4 = 81 * k) ↔ (n = 2 ∧ k = 1) :=
by
  sorry

end natural_numbers_satisfy_equation_l550_550759


namespace range_of_m_l550_550124

def circle (x y : ℝ) : Prop := x^2 + (y-1)^2 = 4

def line (x y m : ℝ) : Prop := sqrt 3 * x + y + m = 0

def distance_from_center_to_line (m : ℝ) : ℝ := abs (m + 1) / 2

theorem range_of_m :
  (-7 < m ∧ m < -3) ∨ (1 < m ∧ m < 5) ↔ ∀ x y m,
  circle x y →
  (∃ x₁ y₁, circle x₁ y₁ ∧ distance_from_center_to_line m = 1) →
  (∃ x₂ y₂, x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ circle x₂ y₂ ∧ distance_from_center_to_line m = 1) :=
by {
  sorry
}

end range_of_m_l550_550124


namespace largest_n_triangle_property_l550_550737

def satisfies_triangle_property (s : Set ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ s → b ∈ s → c ∈ s →
  a < b + c ∧ b < a + c ∧ c < a + b

theorem largest_n_triangle_property : ∃ (n : ℕ), ∀ (S : Finset ℕ), (∀ k, (3 + k - 1) ∈ S) → S.card = 7 → satisfies_triangle_property (↑S) → n = 46 :=
begin
  sorry -- Proof omitted
end

end largest_n_triangle_property_l550_550737


namespace histogram_area_sum_l550_550181

theorem histogram_area_sum {f : ℕ → ℝ} (h : ∑ i, f i = 1) :
  (∑ i, f i) = 1 :=
begin
  exact h,
end

end histogram_area_sum_l550_550181


namespace sally_pens_proof_l550_550956

variable (p : ℕ)  -- define p as a natural number for pens each student received
variable (pensLeft : ℕ)  -- define pensLeft as a natural number for pens left after distributing to students

-- Function representing Sally giving pens to each student
def pens_after_giving_students (p : ℕ) : ℕ := 342 - 44 * p

-- Condition 1: Left half of the remainder in her locker
def locker_pens (p : ℕ) : ℕ := (pens_after_giving_students p) / 2

-- Condition 2: She took 17 pens home
def home_pens : ℕ := 17

-- Main proof statement
theorem sally_pens_proof :
  (locker_pens p + home_pens = pens_after_giving_students p) → p = 7 :=
by
  sorry

end sally_pens_proof_l550_550956


namespace vector_sum_correct_l550_550412

-- Define the points A, B, and C
def A : ℝ × ℝ × ℝ := (1, 1, 0)
def B : ℝ × ℝ × ℝ := (2, 0, -1)
def C : ℝ × ℝ × ℝ := (-1, 3, -2)

-- Define the vectors AB and BC
def AB : ℝ × ℝ × ℝ := (B.1 - A.1, B.2 - A.2, B.3 - A.3)
def BC : ℝ × ℝ × ℝ := (C.1 - B.1, C.2 - B.2, C.3 - B.3)

-- Goal: Prove that the sum of vectors AB and BC equals (-2, 2, -2)
theorem vector_sum_correct : AB.1 + BC.1 = -2 ∧ AB.2 + BC.2 = 2 ∧ AB.3 + BC.3 = -2 := by
  sorry

end vector_sum_correct_l550_550412


namespace monic_quadratic_poly_l550_550074

theorem monic_quadratic_poly (x : ℂ) :
  (∃ (P : ℂ[X]), polynomial.monic P ∧ P.coeff 2 = 1 ∧ P.coeff 1 = 6 ∧ P.coeff 0 = 17 ∧ P.eval (-3 - complex.I * real.sqrt 8) = 0 ∧ P.eval (-3 + complex.I * real.sqrt 8) = 0) :=
sorry

end monic_quadratic_poly_l550_550074


namespace num_isosceles_triangles_l550_550841

theorem num_isosceles_triangles (a b : ℕ) (h1 : 2 * a + b = 27) (h2 : a > b / 2) : 
  ∃! (n : ℕ), n = 13 :=
by 
  sorry

end num_isosceles_triangles_l550_550841


namespace distance_to_right_directrix_l550_550094

noncomputable def hyperbola_param := 
  (a b c e : ℝ) (h_a : a = 5) (h_b : b = 12) (h_c : c = Real.sqrt (a^2 + b^2)) (h_e : e = c / a)

theorem distance_to_right_directrix 
  (x y : ℝ) (P : ℝ × ℝ) (dist_to_left_focus : ℝ) (d d' : ℝ)
  (hyp : P ∈ {P : ℝ × ℝ | P.1^2 / 25 - P.2^2 / 144 = 1})
  (dist_def : dist_to_left_focus = 16)
  (d'_def : d' = dist_to_left_focus + 10)
  (e_def : hyperbola_param 5 12 (Real.sqrt (5^2 + 12^2)) (Real.sqrt (5^2 + 12^2) / 5))
  (ratio_def : d' / d = 13 / 5) :
  d = 10 :=
sorry

end distance_to_right_directrix_l550_550094


namespace triangle_area_base_6_height_8_l550_550606

noncomputable def triangle_area (base height : ℕ) : ℕ :=
  (base * height) / 2

theorem triangle_area_base_6_height_8 : triangle_area 6 8 = 24 := by
  sorry

end triangle_area_base_6_height_8_l550_550606


namespace circle_radius_in_45_45_90_triangle_l550_550323

theorem circle_radius_in_45_45_90_triangle :
  ∃ r : ℝ, (∀ (O A B C : EuclideanSpace ℝ (Fin 2)), 
    (A = (0, 0)) ∧ 
    (B = (2, 0)) ∧ 
    (C = (2, 2)) ∧ 
    (O.1 = 0) ∧ 
    (O.2 = r) ∧ 
    (sqrt ((O.1 - 0)^2 + (O.2 - 0)^2) = r) ∧ 
    (sqrt ((O.1 - 2)^2 + (O.2 - 2)^2) = r)) :=
  r = 2 := 
sorry

end circle_radius_in_45_45_90_triangle_l550_550323


namespace students_attending_swimming_class_l550_550312

def totalStudents := 1000
def chessClassPercentage := 0.20
def swimmingClassPercentage := 0.10

theorem students_attending_swimming_class :
  let studentsInChess := chessClassPercentage * totalStudents
  let studentsInSwimming := swimmingClassPercentage * studentsInChess
  studentsInSwimming = 20 :=
by
  sorry

end students_attending_swimming_class_l550_550312


namespace rectangle_area_inscribed_in_semicircle_l550_550231

theorem rectangle_area_inscribed_in_semicircle
  (DA : ℝ) (GD : ℝ) (HA : ℝ)
  (hDA : DA = 19) (hGD : GD = 10) (hHA : HA = 12) :
  let CD := real.sqrt (GD * HA) in
  let Area := DA * CD in
  Area = 38 * real.sqrt 30 :=
by
  -- Definitions from problem conditions
  let DA := 19
  let GD := 10
  let HA := 12
  
  -- Placeholder for incomplete proof
  sorry

end rectangle_area_inscribed_in_semicircle_l550_550231


namespace gcf_360_180_l550_550294

theorem gcf_360_180 : Nat.gcd 360 180 = 180 :=
by
  sorry

end gcf_360_180_l550_550294


namespace point_reflection_example_l550_550503

def point := ℝ × ℝ

def reflect_x_axis (p : point) : point := (p.1, -p.2)

theorem point_reflection_example : reflect_x_axis (1, -2) = (1, 2) := sorry

end point_reflection_example_l550_550503


namespace g_is_monotonically_decreasing_on_C_l550_550462

-- Define the function f(x)
def f (x : ℝ) : ℝ := sin (2 * x) + cos (2 * x)

-- Define the function g(x) after the shift
def g (x : ℝ) : ℝ := sqrt 2 * sin (2 * x + π / 8)

-- The theorem to prove that g is monotonically decreasing on [0, π/2]
theorem g_is_monotonically_decreasing_on_C : ∀ x y : ℝ, 0 ≤ x → x < y → y ≤ π / 2 → g y ≤ g x :=
by sorry

end g_is_monotonically_decreasing_on_C_l550_550462


namespace integral_value_l550_550748

theorem integral_value :
  ∫ x in -1..1, (x * cos x + real.cbrt (x^2)) = 6 / 5 :=
by
  sorry

end integral_value_l550_550748


namespace parallel_lines_slope_m_l550_550490

theorem parallel_lines_slope_m (m : ℝ) : (∀ (x y : ℝ), (x - 2 * y + 5 = 0) ↔ (2 * x + m * y - 5 = 0)) → m = -4 :=
by
  intros h
  -- Add the necessary calculative steps here
  sorry

end parallel_lines_slope_m_l550_550490


namespace tan_theta_undefined_l550_550199

noncomputable def A := (0 : ℝ, 0 : ℝ)
noncomputable def B := (2 : ℝ, 0 : ℝ)
noncomputable def C := (1 : ℝ, Real.sqrt 3)
noncomputable def AP_length := (2 : ℝ) / 3
noncomputable def BQ_length := (2 : ℝ) / 3
noncomputable def P := ((4 : ℝ) / 3, 0)
noncomputable def Q := ((4 : ℝ) / 3, Real.sqrt 3 / 3)

noncomputable def vec_ap := (P.1 - A.1, P.2 - A.2)
noncomputable def vec_pq := (Q.1 - P.1, Q.2 - P.2)

theorem tan_theta_undefined :
  let θ := Real.arctan (vec_pq.2 / vec_pq.1) - Real.arctan (vec_ap.2 / vec_ap.1)
  Real.tan θ = 0 / vec_pq.1 := 
by
  dsimp [vec_ap, vec_pq, P, Q, A]
  sorry

end tan_theta_undefined_l550_550199


namespace Jenna_height_in_cm_l550_550512

def h_Jenna_inch : ℝ := 74
def conversion_factor : ℝ := 2.58

theorem Jenna_height_in_cm :
  Real.round (h_Jenna_inch * conversion_factor) = 191 :=
by 
  -- proof omitted
  sorry

end Jenna_height_in_cm_l550_550512


namespace nine_point_circle_l550_550585

-- Define the basic geometrical setup as specified in conditions.
variables (A B C : Type) [inhabited A] [inhabited B] [inhabited C]
noncomputable def orthocenter (A B C : Type) [inhabited A] [inhabited B] [inhabited C] := sorry
noncomputable def midpoint (x y : Type) [inhabited x] [inhabited y] := sorry
noncomputable def foot_of_altitude (x y z : Type) [inhabited x] [inhabited y] [inhabited z] := sorry

def A₁ := midpoint (orthocenter A B C) A
def B₁ := midpoint (orthocenter A B C) B
def C₁ := midpoint (orthocenter A B C) C

def A₂ := foot_of_altitude A B C
def B₂ := foot_of_altitude B A C
def C₂ := foot_of_altitude C A B

def A₃ := midpoint B C
def B₃ := midpoint A C
def C₃ := midpoint A B

theorem nine_point_circle (A B C : Type) [inhabited A] [inhabited B] [inhabited C] :
  ∃ (O : Type), ∀ (P : Type), (P = A₁) ∨ (P = B₁) ∨ (P = C₁) ∨ (P = A₂) ∨ (P = B₂) ∨ (P = C₂) ∨ (P = A₃) ∨ (P = B₃) ∨ (P = C₃) → (dist O P = dist O A₁) :=
sorry

end nine_point_circle_l550_550585


namespace problem_statement_l550_550923

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x
noncomputable def g (x : ℝ) : ℝ := x^2 + 2 * x
noncomputable def h (x : ℝ) : ℝ := 2 * Real.sin ((Real.pi / 6) * x + (2 * Real.pi / 3))

theorem problem_statement (k : ℝ) :
  (∀ x : ℝ, h x - f x ≤ k * (g x + 2)) → k ≥ 2 + 1 / Real.exp 1 :=
begin
  intro h,
  sorry
end

end problem_statement_l550_550923


namespace no_triangle_possible_l550_550276

-- Define the lengths of the sticks
def stick_lengths : List ℕ := [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

-- The theorem stating the impossibility of forming a triangle with any combination of these lengths
theorem no_triangle_possible : ¬ ∃ (a b c : ℕ), a ∈ stick_lengths ∧ b ∈ stick_lengths ∧ c ∈ stick_lengths ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (a + b > c ∧ a + c > b ∧ b + c > a) := 
by
  sorry

end no_triangle_possible_l550_550276


namespace part1_part2_l550_550416

open Classical
variables {A B C P D E F G H K M N : Point}
variable (ΔABC : Triangle A B C)
variable (inside_P : Inside P ΔABC)
variables (line_DE_parallel_BC : Parallel (Line P D E) (Line B C))
variables (line_FG_parallel_AC : Parallel (Line F G) (Line A C))
variables (line_HK_parallel_AB : Parallel (Line H K) (Line A B))
variables (intersect_lines_DE : (Line D E ∩ Line F K) = {M} ∧ (Line D E ∩ Line H G) = {N})

noncomputable def geometric_hypotheses : Prop :=
line_DE_parallel_BC ∧ line_FG_parallel_AC ∧ line_HK_parallel_AB ∧ 
intersect_lines_DE

theorem part1 (h : geometric_hypotheses) : 
  1/PM - 1/PN = 1/PD - 1/PE :=
sorry

theorem part2 (h : geometric_hypotheses) (midpoint_DE : Midpoint P D E) : 
  PM = PN :=
sorry

end part1_part2_l550_550416


namespace age_difference_l550_550667

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 13) : A = C + 13 :=
by
  sorry

end age_difference_l550_550667


namespace initial_soccer_balls_l550_550996

theorem initial_soccer_balls {x : ℕ} (h : x + 18 = 24) : x = 6 := 
sorry

end initial_soccer_balls_l550_550996


namespace percentage_difference_wages_l550_550192

variables (W1 W2 : ℝ)
variables (h1 : W1 > 0) (h2 : W2 > 0)
variables (h3 : 0.40 * W2 = 1.60 * 0.20 * W1)

theorem percentage_difference_wages (W1 W2 : ℝ) (h1 : W1 > 0) (h2 : W2 > 0) (h3 : 0.40 * W2 = 1.60 * 0.20 * W1) :
  (W1 - W2) / W1 = 0.20 :=
by
  sorry

end percentage_difference_wages_l550_550192


namespace complex_modulus_l550_550863

noncomputable def z : ℂ := (4 + 3 * Complex.I) / (2 - Complex.I)

theorem complex_modulus (z : ℂ) (h : (2 - Complex.I) * z = 4 + 3 * Complex.I) : Complex.abs z = Real.sqrt 5 :=
by
  have h_z : z = (4 + 3 * Complex.I) / (2 - Complex.I),
  rw [h_z],
  calc
    Complex.abs ((4 + 3 * Complex.I) / (2 - Complex.I)) = Complex.abs (1 + 2 * Complex.I) : by congr
    ... = Real.sqrt (1^2 + 2^2) : by sorry
    ... = Real.sqrt 5 : by sorry

end complex_modulus_l550_550863


namespace triangle_ratio_l550_550287

theorem triangle_ratio (a b c : ℝ) (P Q : ℝ) (h₁ : a ≠ b) (h₂ : a ≠ c) (h₃ : b ≠ c)
  (h₄ : P > 0) (h₅ : Q > P) (h₆ : Q < c) (h₇ : P = 21) (h₈ : Q - P = 35) (h₉ : c - Q = 100)
  (h₁₀ : P + (Q - P) + (c - Q) = c)
  (angle_trisect : ∃ x y : ℝ, x ≠ y ∧ x = a / b ∧ y = 7 / 45) :
  ∃ p q r : ℕ, p + q + r = 92 ∧ p.gcd r = 1 ∧ ¬ ∃ k : ℕ, k^2 ∣ q := sorry

end triangle_ratio_l550_550287


namespace range_of_b_l550_550882

theorem range_of_b (A B C a b c : ℝ) (h_triangle : a ^ 2 + b ^ 2 > c ^ 2 ∧ b ^ 2 + c ^ 2 > a ^ 2 ∧ c ^ 2 + a ^ 2 > b ^ 2)
  (ha : a = 1) (hB : B = 60) (h_acute : 0 < A ∧ A < 90 ∧ 0 < B ∧ B < 90 ∧ 0 < C ∧ C < 90) :
  ∃ l u, l = (sqrt 3 / 2) ∧ u = sqrt 3 ∧ l < b ∧ b < u :=
by
  sorry

end range_of_b_l550_550882


namespace optimal_partition_at_most_cubed_root_six_n_l550_550908

noncomputable def isOptimalAPartition {n : ℕ} (A : Finset ℕ) (parts : ℕ → ℕ) : Prop :=
  (∀ k : ℕ, (∃ B, B ⊆ A ∧ ∑ b in B, parts b = n ∧ B.card = k) →
  ∀ r : ℕ, r < k → ¬ (∃ B', B' ⊆ A ∧ ∑ b in B', parts b = n ∧ B'.card = r))

theorem optimal_partition_at_most_cubed_root_six_n 
  (n : ℕ)
  (A : Finset ℕ) 
  (parts : ℕ → ℕ) :
  (∀ a ∈ A, a ≤ n) →
  isOptimalAPartition A parts → 
  ∑ b in A, parts b = n → 
  B.card ≤ ∛(6 * n) :=
sorry

end optimal_partition_at_most_cubed_root_six_n_l550_550908


namespace B_pow_16_eq_I_l550_550193

noncomputable def B : Matrix (Fin 4) (Fin 4) ℝ := 
  ![
    ![Real.cos (Real.pi / 4), -Real.sin (Real.pi / 4), 0 , 0],
    ![Real.sin (Real.pi / 4), Real.cos (Real.pi / 4), 0 , 0],
    ![0, 0, Real.cos (Real.pi / 4), Real.sin (Real.pi / 4)],
    ![0, 0, -Real.sin (Real.pi / 4), Real.cos (Real.pi / 4)]
  ]

theorem B_pow_16_eq_I : B^16 = 1 := by
  sorry

end B_pow_16_eq_I_l550_550193


namespace determine_n_l550_550749

theorem determine_n 
  (n : ℕ)
  (h : 2^6 * 3^3 * 5 * n = 10!) : n = 210 :=
by
  sorry

end determine_n_l550_550749


namespace vector_ab_values_l550_550102

def planar_vectors (a b : ℝ × ℝ) : Prop :=
  a ≠ (0, 0) ∧ b ≠ (0, 0)

def vector_length (v : ℝ × ℝ) : ℝ := 
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

def vector_angle (a b : ℝ × ℝ) : ℝ := 
  real.acos (dot_product a b / (vector_length a * vector_length b))

noncomputable def ab (a b : ℝ × ℝ) : ℝ :=
  dot_product a b / dot_product b b

noncomputable def ba (a b : ℝ × ℝ) : ℝ :=
  dot_product b a / dot_product a a

def in_N_div_3 (x : ℝ) : Prop :=
  ∃ n : ℤ, x = n / 3

theorem vector_ab_values (a b : ℝ × ℝ) :
  planar_vectors a b →
  vector_length a ≥ vector_length b ∧ vector_length b > 0 →
  vector_angle a b ∈ (0, π / 6) →
  in_N_div_3 (ab a b) →
  in_N_div_3 (ba a b) →
  ab a b ∈ {4 / 3, 7 / 3, 8 / 3} :=
by
  sorry

end vector_ab_values_l550_550102


namespace equation_of_perpendicular_bisector_l550_550488

noncomputable def midpoint (A B : ℝ × ℝ) := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def slope (A B : ℝ × ℝ) := (B.2 - A.2) / (B.1 - A.1)

theorem equation_of_perpendicular_bisector :
  let A := (1 : ℝ, 3 : ℝ),
      B := (-5 : ℝ, 1 : ℝ),
      M := midpoint A B,
      m_AB := slope A B,
      m_l := -1 / m_AB,
      y_intercept := M.2 - m_l * M.1 in
  (m_l * (x - M.1) + y_intercept = 0) =
  (3 * (x : ℝ) + y + 4 = 0)
:= sorry

end equation_of_perpendicular_bisector_l550_550488


namespace num_disks_to_sell_l550_550562

-- Define the buying and selling price conditions.
def cost_per_disk := 6 / 5
def sell_per_disk := 7 / 4

-- Define the desired profit
def desired_profit := 120

-- Calculate the profit per disk.
def profit_per_disk := sell_per_disk - cost_per_disk

-- Statement of the problem: Determine number of disks to sell.
theorem num_disks_to_sell
  (h₁ : cost_per_disk = 6 / 5)
  (h₂ : sell_per_disk = 7 / 4)
  (h₃ : desired_profit = 120)
  (h₄ : profit_per_disk = 7 / 4 - 6 / 5) :
  ∃ disks_to_sell : ℕ, disks_to_sell = 219 ∧ 
  disks_to_sell * profit_per_disk ≥ 120 ∧
  (disks_to_sell - 1) * profit_per_disk < 120 :=
sorry

end num_disks_to_sell_l550_550562


namespace alpha_range_l550_550740

noncomputable def f (x : ℝ) : ℝ := if 1 ≤ x ∧ x < 2 then 2^x * (2*x + 1) else 
 sorry -- function definition for [2, +∞)

theorem alpha_range :
  (∀ x ∈ Set.Ici (1 : ℝ), f (x + 1) = α * f x) →
  (∀ x ∈ Set.Ici (1 : ℝ), f (x) ≤ f (x + 1)) →
  Set.Ici (1 : ℝ).Nonempty →
  Set {α : ℝ | ∃ α, (∀ x ∈ Set.Ici (1 : ℝ), f (x + 1) = α * f x)} ⊆ Set.Ici (10 / 3) :=
sorry

end alpha_range_l550_550740


namespace train_total_distance_l550_550705

theorem train_total_distance (x : ℝ) (h1 : x > 0) 
  (h_speed_avg : 48 = ((3 * x) / (x / 8))) : 
  3 * x = 6 := 
by
  sorry

end train_total_distance_l550_550705


namespace expected_minutes_l550_550910

-- Define initial conditions
def dodecagon := ℕ 
def vertex := ℕ 

def initial_positions : Fin 12 → vertex
| 3 => 4
| 7 => 8
| 11 => 12
| _ => 0  -- dummy value for other vertices

-- Expected number of minutes until the frogs stop jumping
noncomputable def E : ℕ → ℕ → ℕ → ℚ 
| 4, 4, 4 := 16 / 3
| _, _, _ := 0  -- dummy value for other states

-- Main theorem statement
theorem expected_minutes (a b c : ℕ) :
  initial_positions 3 = 4 ∧ initial_positions 7 = 8 ∧ initial_positions 11 = 12 ∧ a = 4 ∧ b = 4 ∧ c = 4 → E a b c = 16 / 3 :=
by {
  sorry
}

end expected_minutes_l550_550910


namespace kenneth_left_with_amount_l550_550190

theorem kenneth_left_with_amount (total_earnings : ℝ) (percentage_spent : ℝ) (amount_left : ℝ) 
    (h_total_earnings : total_earnings = 450) (h_percentage_spent : percentage_spent = 0.10) 
    (h_spent_amount : total_earnings * percentage_spent = 45) : 
    amount_left = total_earnings - total_earnings * percentage_spent :=
by sorry

end kenneth_left_with_amount_l550_550190


namespace max_x_plus_one_over_x_l550_550993

theorem max_x_plus_one_over_x (n : ℕ) (a b : ℝ) (numbers : Fin n → ℝ) 
  (h_n : n = 2023) 
  (h_pos : ∀ i, 0 < numbers i) 
  (h_sum : (∑ i, numbers i) = 2024) 
  (h_sum_reciprocal : (∑ i, (numbers i)⁻¹) = 2024) 
  (x : ℝ) 
  (h_x_in : ∀ i, numbers i = x) : 
  x + x⁻¹ ≤ 8093 / 2024 := 
sorry

end max_x_plus_one_over_x_l550_550993


namespace vector_identity_l550_550137

-- Definitions of the vectors
variables {V : Type*} [AddGroup V]

-- Conditions as Lean definitions
def cond1 (AB BO AO : V) : Prop := AB + BO = AO
def cond2 (AO OM AM : V) : Prop := AO + OM = AM
def cond3 (AM MB AB : V) : Prop := AM + MB = AB

-- The main statement to be proved
theorem vector_identity (AB MB BO BC OM AO AM AC : V) 
  (h1 : cond1 AB BO AO) 
  (h2 : cond2 AO OM AM) 
  (h3 : cond3 AM MB AB) 
  : (AB + MB) + (BO + BC) + OM = AC :=
sorry

end vector_identity_l550_550137


namespace inversion_preserves_perpendicularity_l550_550228

noncomputable def inverse_point (P : ℝ × ℝ) (C : ℝ × ℝ) : ℝ × ℝ := sorry
noncomputable def inverse_line_or_circle (g : ℝ × ℝ × ℝ × ℝ) (C : ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

theorem inversion_preserves_perpendicularity {P Q : ℝ × ℝ} {C : ℝ × ℝ} (g : ℝ × ℝ × ℝ × ℝ) (h : g ≠ (x, y, z, w)) (H1 : inverse_point P g = Q) :
  let g1 := inverse_line_or_circle g C in
  let P1 := inverse_point P C in
  let Q1 := inverse_point Q C in
  inverse_point P1 g1 = Q1 :=
by
  sorry

end inversion_preserves_perpendicularity_l550_550228


namespace range_of_b_l550_550884

theorem range_of_b (A B C a b c : ℝ)
  (h_acute : 0 < A ∧ A < real.pi / 2 ∧
             0 < B ∧ B < real.pi / 2 ∧
             0 < C ∧ C < real.pi / 2 ∧ A + B + C = real.pi)
  (h_a : a = 1)
  (h_B : B = real.pi / 3) :
  ∃ (lb ub : ℝ), lb = real.sqrt 3 / 2 ∧ ub = real.sqrt 3 ∧ lb < b ∧ b < ub :=
by
  -- Proof goes here.
  sorry

end range_of_b_l550_550884


namespace max_sum_dist_in_triangle_l550_550914

variables {A B C A' B' C' P : Type*}
variables [metric_space P] [linear_ordered_comm_ring A] [linear_ordered_comm_ring B] [linear_ordered_comm_ring C]

-- Define sides of triangle ABC
variables (a b c : ℝ) (h_cle_b : c ≤ b) (h_ble_a : b ≤ a)

-- Define lengths AA', BB', CC'
variables (AA' BB' CC' : ℝ)

-- Point P is inside triangle ABC and forms lines intersecting at A', B', C' respectively
variable (s : ℝ := AA' + BB' + CC')

theorem max_sum_dist_in_triangle (hAA' : AA' < b) (hBB' : BB' < a) (hCC' : CC' < a) : 
  s < 2 * a + b :=
begin
  -- Add the inequations
  sorry
end

end max_sum_dist_in_triangle_l550_550914


namespace no_real_solution_for_x_l550_550320

theorem no_real_solution_for_x
  (y : ℝ)
  (x : ℝ)
  (h1 : y = (x^3 - 8) / (x - 2))
  (h2 : y = 3 * x) :
  ¬ ∃ x : ℝ, y = 3*x ∧ y = (x^3 - 8) / (x - 2) :=
by {
  sorry
}

end no_real_solution_for_x_l550_550320


namespace cone_base_area_and_lateral_surface_area_l550_550699

variables (r h l : ℝ)

-- Given conditions
def volume := 36 * Real.pi
def height := 6

-- Definitions
def base_radius (r : ℝ) := (1 / 3) * Real.pi * r^2 * height = volume
def base_area (r : ℝ) := Real.pi * r^2
def slant_height (r h : ℝ) := Math.sqrt (r^2 + h^2)
def lateral_surface_area (r l : ℝ) := Real.pi * r * l

-- Theorem statement
theorem cone_base_area_and_lateral_surface_area (r := 3 * Real.sqrt 2) : 
  (base_area r = 18 * Real.pi) ∧ 
  (lateral_surface_area r (slant_height r height) = 36 * Real.pi) := 
  sorry

end cone_base_area_and_lateral_surface_area_l550_550699


namespace machine_B_has_better_quality_l550_550283

-- Let ξ1 and ξ2 represent the packaging weights of machines A and B, respectively.
variables (ξ1 ξ2 : ℝ)

-- Let Eξ1 and Eξ2 be the expected values, and Dξ1 and Dξ2 be the variances.
def expected_value (ξ : ℝ) : ℝ := sorry -- Definition of expected value, placeholder
def variance (ξ : ℝ) : ℝ := sorry -- Definition of variance, placeholder

-- Given conditions
axiom ex_ξ1_eq_ex_ξ2 : expected_value ξ1 = expected_value ξ2
axiom var_ξ1_gt_var_ξ2 : variance ξ1 > variance ξ2

-- Conclusion: Machine B has better quality
theorem machine_B_has_better_quality : better_quality ξ2 ξ1 := sorry


end machine_B_has_better_quality_l550_550283


namespace positive_difference_in_balances_l550_550711

theorem positive_difference_in_balances :
  let A_0 := 12000
  let r_A := 0.05
  let B_0 := 15000
  let r_B := 0.08
  let t := 25
  let A := A_0 * (1 + r_A)^t
  let B := B_0 * (1 + r_B * t)
  abs (B - A) = 4363 :=
by
  sorry

end positive_difference_in_balances_l550_550711


namespace count_integers_in_solution_set_l550_550850

-- Define the predicate for the condition given in the problem
def condition (x : ℝ) : Prop := abs (x - 3) ≤ 4.5

-- Define the list of integers within the range of the condition
def solution_set : List ℤ := [-1, 0, 1, 2, 3, 4, 5, 6, 7]

-- Prove that the number of integers satisfying the condition is 8
theorem count_integers_in_solution_set : solution_set.length = 8 :=
by
  sorry

end count_integers_in_solution_set_l550_550850


namespace problem_solution_l550_550647

theorem problem_solution :
  3 ^ (0 ^ (2 ^ 2)) + ((3 ^ 1) ^ 0) ^ 2 = 2 :=
by
  sorry

end problem_solution_l550_550647


namespace arithmetic_to_geometric_seq_l550_550437

theorem arithmetic_to_geometric_seq
  (d a : ℕ) 
  (h1 : d ≠ 0) 
  (a_n : ℕ → ℕ)
  (h2 : ∀ n, a_n n = a + (n - 1) * d)
  (h3 : (a + 2 * d) * (a + 2 * d) = a * (a + 8 * d))
  : (a_n 2 + a_n 4 + a_n 10) / (a_n 1 + a_n 3 + a_n 9) = 16 / 13 :=
by
  sorry

end arithmetic_to_geometric_seq_l550_550437


namespace summation_arccot_l550_550051

noncomputable def Arccot (t : ℝ) : ℝ := if h : t ≥ 0 then classical.some (exists_arccot h) else 0

theorem summation_arccot : (∑ n : ℕ, Arccot (n^2 + n + 1)) = (π / 2) :=
sorry

end summation_arccot_l550_550051


namespace find_values_of_a_l550_550472

-- Definitions for sets A and B
def A : Set ℝ := {x | x^2 - x - 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 6 = 0}

-- The theorem we want to prove
theorem find_values_of_a (a : ℝ) : (A ∪ B a = A) ↔ (a = -6 ∨ a = 0 ∨ a = 3) :=
by
  sorry

end find_values_of_a_l550_550472


namespace _l550_550678

noncomputable def total_distance (T : ℝ) (D : ℝ) (t : ℝ) : Prop :=
  D = 40 * T ∧ D = 35 * (T + t)

noncomputable def on_time := 0

noncomputable theorem car_journey_distance (T : ℝ) (D : ℝ) (t : ℝ) 
  (h1 : total_distance T D t)
  (h2 : t = 0.25)
  (h3 : T = 1.75) :
  D = 70 := 
sorry

end _l550_550678


namespace sum_of_roots_l550_550664

theorem sum_of_roots (b : ℝ) (x : ℝ) (y : ℝ) :
  (x^2 - b * x + 20 = 0) ∧ (y^2 - b * y + 20 = 0) ∧ (x * y = 20) -> (x + y = b) := 
by
  sorry

end sum_of_roots_l550_550664


namespace arithmetic_sequence_sum_properties_l550_550805

theorem arithmetic_sequence_sum_properties {S : ℕ → ℝ} {a : ℕ → ℝ} (d : ℝ)
  (h1 : S 6 > S 7) (h2 : S 7 > S 5) :
  let a6 := (S 6 - S 5)
  let a7 := (S 7 - S 6)
  (d = a7 - a6) →
  d < 0 ∧ S 12 > 0 ∧ ¬(∀ n, S n = S 11) ∧ abs a6 > abs a7 :=
by
  sorry

end arithmetic_sequence_sum_properties_l550_550805


namespace monic_quadratic_poly_l550_550072

theorem monic_quadratic_poly (x : ℂ) :
  (∃ (P : ℂ[X]), polynomial.monic P ∧ P.coeff 2 = 1 ∧ P.coeff 1 = 6 ∧ P.coeff 0 = 17 ∧ P.eval (-3 - complex.I * real.sqrt 8) = 0 ∧ P.eval (-3 + complex.I * real.sqrt 8) = 0) :=
sorry

end monic_quadratic_poly_l550_550072


namespace intersection_eq_l550_550109

-- Definitions of sets A and B
def A : Set ℤ := {0, 1}
def B : Set ℤ := {-1, 1}

-- The theorem statement
theorem intersection_eq : A ∩ B = {1} :=
by
  unfold A B
  sorry

end intersection_eq_l550_550109


namespace reverse_Holder_inequality_reverse_Minkowski_inequality_l550_550314

-- Given non-negative random variables xi and eta, prove the reverse Hölder inequality
theorem reverse_Holder_inequality
  (xi eta : ℝ → ℝ)
  (p q : ℝ)
  (hpq : 1 / p + 1 / q = 1)
  (I : Icc 0 1)
  (hxi_nonneg : ∀ x, 0 ≤ xi x)
  (heta_nonneg : ∀ x, 0 ≤ eta x)
  (hxi_finite : ∫ x, xi x ^ p < ∞)
  (heta_finite : ∫ x, eta x ^ q < ∞)
  (h_eta_positive : ∫ x, eta x ^ q > 0)
  (h_qneg : q < 0) (h_p : p ∈ I) :
  ∫ x, xi x * eta x ≥ (∫ x, xi x ^ p) ^ (1 / p) * (∫ x, eta x ^ q) ^ (1 / q) := by 
  sorry

-- Given non-negative random variables xi and eta, prove the reverse Minkowski inequality
theorem reverse_Minkowski_inequality
  (xi eta : ℝ → ℝ)
  (p : ℝ)
  (I : Icc 0 1)
  (hxi_nonneg : ∀ x, 0 ≤ xi x)
  (heta_nonneg : ∀ x, 0 ≤ eta x)
  (hxi_finite : ∫ x, xi x ^ p < ∞)
  (heta_finite : ∫ x, eta x ^ p < ∞)
  (h_p : p ∈ I) :
  (∫ x, (xi x + eta x) ^ p) ^ (1 / p) ≥ (∫ x, xi x ^ p) ^ (1 / p) + (∫ x, eta x ^ p) ^ (1 / p) := by 
  sorry

end reverse_Holder_inequality_reverse_Minkowski_inequality_l550_550314


namespace measure_of_angle_C_l550_550288

-- Definitions of the angles
def angles (A B C : ℝ) : Prop :=
  -- Conditions: measure of angle A is 1/4 of measure of angle B
  A = (1 / 4) * B ∧
  -- Lines p and q are parallel so alternate interior angles are equal
  C = A ∧
  -- Since angles B and C are supplementary
  B + C = 180

-- The problem in Lean 4 statement: Prove that C = 36 given the conditions
theorem measure_of_angle_C (A B C : ℝ) (h : angles A B C) : C = 36 := sorry

end measure_of_angle_C_l550_550288


namespace reading_time_equal_l550_550949

/--
  Alice, Bob, and Chandra are reading a 760-page book. Alice reads a page in 20 seconds, 
  Bob reads a page in 45 seconds, and Chandra reads a page in 30 seconds. Prove that if 
  they divide the book into three sections such that each reads for the same length of 
  time, then each person will read for 7200 seconds.
-/
theorem reading_time_equal 
  (rate_A : ℝ := 1/20) 
  (rate_B : ℝ := 1/45) 
  (rate_C : ℝ := 1/30) 
  (total_pages : ℝ := 760) : 
  ∃ t : ℝ, t = 7200 ∧ 
    (t * rate_A + t * rate_B + t * rate_C = total_pages) := 
by
  sorry  -- proof to be provided

end reading_time_equal_l550_550949


namespace remainder_sum_mod_15_l550_550218

variable (k j : ℤ) -- these represent any integers

def p := 60 * k + 53
def q := 75 * j + 24

theorem remainder_sum_mod_15 :
  (p k + q j) % 15 = 2 :=  
by 
  sorry

end remainder_sum_mod_15_l550_550218


namespace range_of_b_l550_550883

theorem range_of_b (A B C a b c : ℝ)
  (h_acute : 0 < A ∧ A < real.pi / 2 ∧
             0 < B ∧ B < real.pi / 2 ∧
             0 < C ∧ C < real.pi / 2 ∧ A + B + C = real.pi)
  (h_a : a = 1)
  (h_B : B = real.pi / 3) :
  ∃ (lb ub : ℝ), lb = real.sqrt 3 / 2 ∧ ub = real.sqrt 3 ∧ lb < b ∧ b < ub :=
by
  -- Proof goes here.
  sorry

end range_of_b_l550_550883


namespace math_homework_pages_l550_550954

-- Define the conditions
def reading_pages : ℕ := 7
def biology_pages : ℕ := 3
def combined_math_biology_pages : ℕ := 11

-- Define what we need to prove
theorem math_homework_pages : ∀ x : ℕ, (combined_math_biology_pages - biology_pages = x) → x = 8 :=
by
  intro x h
  have h1 : combined_math_biology_pages - biology_pages = 8 := sorry
  exact eq.trans h h1

end math_homework_pages_l550_550954


namespace rowing_distance_l550_550990

theorem rowing_distance (D : ℝ) 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (total_time : ℝ) 
  (downstream_speed : ℝ := boat_speed + stream_speed) 
  (upstream_speed : ℝ := boat_speed - stream_speed)
  (downstream_time : ℝ := D / downstream_speed)
  (upstream_time : ℝ := D / upstream_speed)
  (round_trip_time : ℝ := downstream_time + upstream_time) 
  (h1 : boat_speed = 16) 
  (h2 : stream_speed = 2) 
  (h3 : total_time = 914.2857142857143)
  (h4 : round_trip_time = total_time) :
  D = 720 :=
by sorry

end rowing_distance_l550_550990


namespace jack_ma_loan_repayment_l550_550188

-- Define the conditions: initial amount M, interest rate r, monthly repayment x, and number of months n
variables (M x : ℝ) (r : ℝ) (n : ℕ)

-- Define the scenario of the problem
def compounded_amount (M : ℝ) (r : ℝ) (n : ℕ) := M * (1 + r) ^ n
def total_repayment (x : ℝ) (n : ℕ) := x * n

-- Given conditions
axiom (initial_loan : M > 0)
axiom (monthly_interest_rate : r = 0.05)
axiom (repayment_months : n = 20)

-- Theorem to be proven
theorem jack_ma_loan_repayment : total_repayment x n = compounded_amount M r n :=
by sorry

end jack_ma_loan_repayment_l550_550188


namespace union_of_sets_l550_550443

def A : Set ℝ := {x | x < -1 ∨ x > 3}
def B : Set ℝ := {x | x ≥ 2}

theorem union_of_sets : A ∪ B = {x | x < -1 ∨ x ≥ 2} :=
by
  sorry

end union_of_sets_l550_550443


namespace cost_of_one_pack_of_gummy_bears_l550_550517

theorem cost_of_one_pack_of_gummy_bears
    (num_chocolate_bars : ℕ)
    (num_gummy_bears : ℕ)
    (num_chocolate_chips : ℕ)
    (total_cost : ℕ)
    (cost_per_chocolate_bar : ℕ)
    (cost_per_chocolate_chip : ℕ)
    (cost_of_one_gummy_bear_pack : ℕ)
    (h1 : num_chocolate_bars = 10)
    (h2 : num_gummy_bears = 10)
    (h3 : num_chocolate_chips = 20)
    (h4 : total_cost = 150)
    (h5 : cost_per_chocolate_bar = 3)
    (h6 : cost_per_chocolate_chip = 5)
    (h7 : num_chocolate_bars * cost_per_chocolate_bar +
          num_gummy_bears * cost_of_one_gummy_bear_pack +
          num_chocolate_chips * cost_per_chocolate_chip = total_cost) :
    cost_of_one_gummy_bear_pack = 2 := by
  sorry

end cost_of_one_pack_of_gummy_bears_l550_550517


namespace renu_and_suma_work_together_l550_550591

theorem renu_and_suma_work_together
  (renu_work_days : ℕ) (suma_work_days : ℕ)
  (h_renu : renu_work_days = 5) (h_suma : suma_work_days = 20) :
  let combined_work_days := 4
  in combined_work_days = 4 :=
by
  -- Proof will be filled here
  sorry

end renu_and_suma_work_together_l550_550591


namespace num_common_tangents_l550_550608

noncomputable def first_circle : set (ℝ × ℝ) := {p | let x := p.1; let y := p.2 in x^2 + y^2 - 4 * x = 0}
noncomputable def second_circle : set (ℝ × ℝ) := {p | let x := p.1; let y := p.2 in (x-3)^2 + (y+3)^2 = 9}

theorem num_common_tangents : 
  let c1_center : ℝ × ℝ := (2, 0)
  let c1_radius : ℝ := 2
  let c2_center : ℝ × ℝ := (3, -3)
  let c2_radius : ℝ := 3
  let dist_centers := real.sqrt ((c2_center.1 - c1_center.1)^2 + (c2_center.2 - c1_center.2)^2)
  ∧ dist_centers = real.sqrt 10
  → c1_radius < dist_centers ∧ dist_centers < c1_radius + c2_radius
  → ∃ tangents : finset (set (ℝ × ℝ)), tangents.card = 2 := sorry

end num_common_tangents_l550_550608


namespace highest_degree_divisor_l550_550063

def f (n : ℕ) (x : ℝ) : ℝ := n * x^(n+1) - (n+1) * x^n + 1
def g (n : ℕ) (x : ℝ) : ℝ := x^n - n * x + n - 1

theorem highest_degree_divisor (n : ℕ) :
  ∃ (h : ℝ → ℝ), ∀ (x : ℝ), f(n, x) = (x - 1)^2 * h(x) :=
sorry

end highest_degree_divisor_l550_550063


namespace probability_of_seeing_a_change_l550_550703

-- Define the traffic light cycle periods in seconds
def green_duration := 45
def yellow_duration := 5
def blinking_red_duration := 5
def solid_red_duration := 45

-- Define the total cycle duration
def total_cycle_duration := green_duration + yellow_duration + blinking_red_duration + solid_red_duration

-- Define the observation window duration
def observation_duration := 4

-- Define the intervals where changes can be observed
def change_intervals := 4 * 4

-- Define the probability that Sam sees a change at least once
def probability_change_seen := change_intervals.toRat / total_cycle_duration.toRat

-- The theorem to prove
theorem probability_of_seeing_a_change :
  probability_change_seen = 4 / 25 := by
  sorry

end probability_of_seeing_a_change_l550_550703


namespace function_equivalence_l550_550421

theorem function_equivalence (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2 * x) = 6 * x - 1) : ∀ x : ℝ, f x = 3 * x - 1 :=
by
  sorry

end function_equivalence_l550_550421


namespace max_lcm_of_set_l550_550743

open Nat

theorem max_lcm_of_set :
  max (max (max (max (max (lcm 15 3) (lcm 15 5)) (lcm 15 6)) (lcm 15 9)) (lcm 15 10)) (lcm 15 15) = 30 :=
by
  sorry

end max_lcm_of_set_l550_550743


namespace common_ratio_geometric_sequence_l550_550455

theorem common_ratio_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, 0 < a n) 
  (h_arith : 2 * (1/2 * a 5) = a 3 + a 4) : q = (1 + Real.sqrt 5) / 2 :=
sorry

end common_ratio_geometric_sequence_l550_550455


namespace total_savings_l550_550233

/-- Sally and Bob have made plans to go on a trip at the end of the year. -/
def sally_earnings_per_day : ℕ := 6
def bob_earnings_per_day : ℕ := 4
def total_days_in_year : ℕ := 365
def weekends_per_year : ℕ := 52
def weekend_days_per_weekend : ℕ := 2
def total_holidays : ℕ := 10
def savings_fraction : ℤ := 1/2

/-- They save half of what they've earned. Calculate the total savings after a year. -/
theorem total_savings : 
  let weekend_days := weekends_per_year * weekend_days_per_weekend in
  let working_days := total_days_in_year - weekend_days - total_holidays in
  let sally_total_earnings := sally_earnings_per_day * working_days in
  let bob_total_earnings := bob_earnings_per_day * working_days in
  let sally_savings := sally_total_earnings * savings_fraction in
  let bob_savings := bob_total_earnings * savings_fraction in
  (sally_savings + bob_savings) = 1255 := 
by
  sorry

end total_savings_l550_550233


namespace exterior_angle_CBE_l550_550590

/-- Given a regular square ABCD with an interior angle CAB of 90 degrees,
    and a regular triangle ABE with an interior angle BAE of 60 degrees,
    prove that the exterior angle CBE is 150 degrees. -/

theorem exterior_angle_CBE (ABCD : Type) (ABE : Type)
  [regular_square ABCD] [regular_triangle ABE]
  (CAB : angle ABCD.A ABCD.B ABCD.C = 90)
  (BAE : angle ABE.B ABE.A ABE.E = 60) :
  angle ABCD.B ABCD.C ABE.E = 150 :=
sorry

end exterior_angle_CBE_l550_550590


namespace eval_expression_when_c_eq_3_l550_550390

theorem eval_expression_when_c_eq_3 :
    (let c : ℕ := 3 
     in (c^c - c * (c-1)^c)^c) = 27 :=
by
    sorry

end eval_expression_when_c_eq_3_l550_550390


namespace problem1_problem2_l550_550821

variable (x y : ℝ)

def line_eq (a b c : ℝ) : Prop := a * x + b * y + c = 0

def point_on_line (a b c x0 y0 : ℝ) : Prop :=
  a * x0 + b * y0 + c = 0

theorem problem1:
  ∀ (x y : ℝ), (point_on_line 3 4 (-9) (-1) 3) → line_eq 3 4 (-9) :=
by
  assume x y,
  sorry

theorem problem2:
  ∀ (x y : ℝ), (4 * (-x / 4) * (y / 3) / 2 = 4) → (line_eq 4 (-3) (4 * Real.sqrt 6) ∨ line_eq 4 (-3) (-4 * Real.sqrt 6)) :=
by
  assume x y,
  sorry

end problem1_problem2_l550_550821


namespace rounding_effect_l550_550755

variable (a b c d: ℕ)

-- Conditions: a, b, c, and d are large positive integers.
-- Rounding approaches specified in the problem statement.
axiom a_pos: 1 < a
axiom b_pos: 1 < b
axiom c_pos: 1 < c
axiom d_pos: 1 < d

theorem rounding_effect:
    (round_up a + round_up b) / round_down c + round_down d < (a + b) / c + d :=
sorry

end rounding_effect_l550_550755


namespace find_distance_difference_l550_550831
-- Importing the broad math library

open Real

-- Definition of the line l
def line (t : ℝ) : ℝ × ℝ :=
  (1 + (sqrt 2) / 2 * t, (sqrt 2) / 2 * t)

-- Definition of the curve C1
def curve (θ : ℝ) : ℝ × ℝ :=
  (2 * cos θ, sqrt 3 * sin θ)

-- Given points F and F1
def point_F : (ℝ × ℝ) :=
  (1, 0)

def point_F1 : (ℝ × ℝ) :=
  (-1, 0)

-- Assertion
theorem find_distance_difference (t1 t2 : ℝ)
  (h_line : line t1 = line t2)
  (h_curve1 : line t1 = curve ?m/θ1)
  (h_curve2 : line t2 = curve ?m/θ2)
  (h_solution : t1 + t2 = -6 * (sqrt 2) / 7 ∧ t1 * t2 = -18 / 7 ∧ t1 > 0 ∧ t2 < 0) :
  abs ((-1 : ℝ) * (1 + (sqrt 2 / 2) * t1) - (-1 * (1 + (sqrt 2) / 2 * t2))) = 6 * (sqrt 2) / 7 := by
  sorry

end find_distance_difference_l550_550831


namespace sin_sum_to_product_l550_550392

-- Define the proof problem
theorem sin_sum_to_product (x : ℝ) : sin (3 * x) + sin (9 * x) = 2 * sin (6 * x) * cos (3 * x) := 
by 
  -- No need to prove, just the statement
  sorry

end sin_sum_to_product_l550_550392


namespace trig_problem_l550_550445

theorem trig_problem (α : ℝ)
  (h : (cos α + sin α) / (cos α - sin α) = 2) :
  (1 + sin (4 * α) - cos (4 * α)) / (1 + sin (4 * α) + cos (4 * α)) = 3 / 4 := 
sorry

end trig_problem_l550_550445


namespace degrees_of_freedom_exp_dist_l550_550303

theorem degrees_of_freedom_exp_dist (s : ℕ) (r : ℕ) (h1 : r = 1) :
  let k := s - 1 - r
  in k = s - 2 :=
by
  sorry

end degrees_of_freedom_exp_dist_l550_550303


namespace bottles_more_than_apples_l550_550331

theorem bottles_more_than_apples : 
  ∀ (apples regular_soda diet_soda : ℕ), 
    apples = 36 → regular_soda = 80 → diet_soda = 54 → 
    (regular_soda + diet_soda - apples) = 98 :=
by
  intros apples regular_soda diet_soda h_apples h_regular h_diet
  rw [h_apples, h_regular, h_diet]
  norm_num
  sorry

end bottles_more_than_apples_l550_550331


namespace coeffs_of_polynomial_l550_550610

def p (x y : ℚ) : ℚ := 3 * x * y^2 - 2 * y - 1

theorem coeffs_of_polynomial : 
  let p := p in 
  (coeff_of_linear_term : ℚ, coeff_of_constant_term : ℚ) = (-2, -1) :=
by
  sorry

end coeffs_of_polynomial_l550_550610


namespace sum_of_solutions_l550_550077

theorem sum_of_solutions : 
  (∑ x in {x : ℝ | 2^(x^2 - 4*x - 3) = 8^(x - 5)}, x) = 7 := 
by 
  sorry

end sum_of_solutions_l550_550077


namespace recommended_daily_serving_l550_550592

theorem recommended_daily_serving (mg_per_pill : ℕ) (pills_per_week : ℕ) (total_mg_week : ℕ) (days_per_week : ℕ) 
  (h1 : mg_per_pill = 50) (h2 : pills_per_week = 28) (h3 : total_mg_week = pills_per_week * mg_per_pill) 
  (h4 : days_per_week = 7) : 
  total_mg_week / days_per_week = 200 :=
by
  sorry

end recommended_daily_serving_l550_550592


namespace polygon_of_T_has_4_sides_l550_550538

def T (b : ℝ) (x y : ℝ) : Prop :=
  b ≤ x ∧ x ≤ 4 * b ∧
  b ≤ y ∧ y ≤ 4 * b ∧
  x + y ≥ 3 * b ∧
  x + 2 * b ≥ 2 * y ∧
  2 * y ≥ x + b

noncomputable def sides_of_T (b : ℝ) : ℕ :=
  if b > 0 then 4 else 0

theorem polygon_of_T_has_4_sides (b : ℝ) (hb : b > 0) : sides_of_T b = 4 := by
  sorry

end polygon_of_T_has_4_sides_l550_550538


namespace light_off_after_5_presses_light_on_after_50_presses_l550_550179

def initial_state : bool := true

def is_light_on_after_n_presses (n: ℕ): bool :=
  if n % 2 = 0 then initial_state else !initial_state

theorem light_off_after_5_presses : is_light_on_after_n_presses 5 = false :=
  by sorry

theorem light_on_after_50_presses : is_light_on_after_n_presses 50 = true :=
  by sorry

end light_off_after_5_presses_light_on_after_50_presses_l550_550179


namespace calculate_unshaded_perimeter_l550_550995

-- Defining the problem's conditions and results.
def total_length : ℕ := 20
def total_width : ℕ := 12
def shaded_area : ℕ := 65
def inner_shaded_width : ℕ := 5
def total_area : ℕ := total_length * total_width
def unshaded_area : ℕ := total_area - shaded_area

-- Define dimensions for the unshaded region based on the problem conditions.
def unshaded_width : ℕ := total_width - inner_shaded_width
def unshaded_height : ℕ := unshaded_area / unshaded_width

-- Calculate perimeter of the unshaded region.
def unshaded_perimeter : ℕ := 2 * (unshaded_width + unshaded_height)

-- Stating the theorem to be proved.
theorem calculate_unshaded_perimeter : unshaded_perimeter = 64 := 
sorry

end calculate_unshaded_perimeter_l550_550995


namespace calculate_factorial_expression_l550_550724

theorem calculate_factorial_expression :
  6 * nat.factorial 6 + 5 * nat.factorial 5 + nat.factorial 5 = 5040 := 
sorry

end calculate_factorial_expression_l550_550724


namespace decreasing_function_b_bound_l550_550824

theorem decreasing_function_b_bound (b : ℝ) (f : ℝ → ℝ) 
  (h : ∀ x ∈ Icc (-1:ℝ) 1, deriv (λ x, x^3 + b * x) x ≤ 0) : 
  b ≤ -3 :=
sorry

end decreasing_function_b_bound_l550_550824


namespace inscribed_sphere_radius_l550_550250

-- Define lengths of the sides and the pyramid structure
variables {AB AC BC : ℝ} {height : ℝ}
variables (SB SC : ℝ)

-- Assume given conditions
def pyramid_conditions : Prop :=
  AB = 10 ∧ AC = 10 ∧ BC = 12 ∧ height = 1.4 ∧ SB = SC

-- Define the correct answer as the radius of the inscribed sphere
def radius_of_inscribed_sphere := 12 / 19

-- State the theorem
theorem inscribed_sphere_radius (AB AC BC height SB SC : ℝ) (h : pyramid_conditions) :
  radius_of_inscribed_sphere = 12 / 19 :=
sorry

end inscribed_sphere_radius_l550_550250


namespace find_m_l550_550815

noncomputable def tangent_condition (m : ℝ) : Prop :=
  let d : ℝ := |2| / Real.sqrt (m^2 + 1)
  d = 1

theorem find_m (m : ℝ) : tangent_condition m ↔ m = Real.sqrt 3 ∨ m = -Real.sqrt 3 := by
  sorry

end find_m_l550_550815


namespace complex_modulus_value_l550_550125

-- Define the complex number z according to the given condition.
def z : ℂ := (1 + complex.i) / complex.i

-- State the theorem that needs to be proven.
theorem complex_modulus_value : complex.abs (z + 2) = real.sqrt 10 :=
sorry

end complex_modulus_value_l550_550125


namespace seventh_term_l550_550988

-- Define a sequence based on the conditions provided
def sequence : ℕ → ℕ
| 0       := 1
| 1       := 3
| 2       := 6
| 3       := 11
| 4       := 18
| 5       := 29
| (n + 1) := sequence n + (if n = 0 then 2 else sequence (n - 1) - sequence (n - 2) + 1)

-- Prove that the seventh number in the sequence is 40
theorem seventh_term :
  sequence 6 = 40 := sorry

end seventh_term_l550_550988


namespace find_sum_mod_7_l550_550480

open ZMod

-- Let a, b, and c be elements of the cyclic group modulo 7
def a : ZMod 7 := sorry
def b : ZMod 7 := sorry
def c : ZMod 7 := sorry

-- Conditions
axiom h1 : a * b * c = 1
axiom h2 : 4 * c = 5
axiom h3 : 5 * b = 4 + b

-- Goal
theorem find_sum_mod_7 : a + b + c = 2 := by
  sorry

end find_sum_mod_7_l550_550480


namespace area_of_triangle_PF1F2_l550_550532

noncomputable def area_of_triangle (P F1 F2 : Point) : Real :=
  let d1 := dist P F1
  let d2 := dist P F2
  let angle := angle F1 P F2
  0.5 * d1 * d2 * Real.sin angle

noncomputable def dist (p1 p2 : Point) : Real :=
  sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

noncomputable def angle (p1 p2 p3 : Point) : Real :=
  let a := dist p2 p3
  let b := dist p1 p3
  let c := dist p1 p2
  Real.arccos ((c^2 + b^2 - a^2) / (2 * c * b))

structure Point where
  x : Real
  y : Real

theorem area_of_triangle_PF1F2 :
  let F1 := {x := -2, y := 0}
  let F2 := {x := 2, y := 0}
  ∃ P : Point, 
    (6 * P.x^2 + 2 * P.y^2 = 12) ∧ 
    (3 * P.x^2 - P.y^2 = 3) ∧ 
    area_of_triangle P F1 F2 = sqrt 2 := 
by
  sorry

end area_of_triangle_PF1F2_l550_550532


namespace selection_with_at_least_one_girl_l550_550410

theorem selection_with_at_least_one_girl :
  ∀ (boys girls : ℕ), boys = 3 → girls = 2 →
  (∃ (selection : ℕ), selection = 3 ∧ (∃ (x : ℕ), x ≥ 1)) →
  (∑ x in {1, 2}, ∃ (ways : ℕ), ways = (Nat.choose girls x) * (Nat.choose boys (3 - x))) = 9 :=
by
  intros boys girls h1 h2 h3
  simp [Nat.choose]
  sorry

end selection_with_at_least_one_girl_l550_550410


namespace solve_for_y_l550_550966

theorem solve_for_y (y : ℤ) : 
  7 * (4 * y + 3) - 3 = -3 * (2 - 9 * y) → y = -24 :=
by
  intro h
  sorry

end solve_for_y_l550_550966


namespace no_primes_in_first_12_sums_of_consecutive_odds_l550_550264

def is_sum_of_consecutive_odds (n : ℕ) : ℕ :=
  (0).to_nat + List.sum (List.range n) * 2 + List.range n li 

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem no_primes_in_first_12_sums_of_consecutive_odds :
  ∀ n, n ∈ (List.range 12).map (λ n, is_sum_of_consecutive_odds (n + 1)) → ¬ is_prime n :=
by sorry

end no_primes_in_first_12_sums_of_consecutive_odds_l550_550264


namespace kevin_initial_cards_l550_550191

theorem kevin_initial_cards :
  ∀ (lost_cards remaining_cards initial_cards : ℝ),
  lost_cards = 7.0 → remaining_cards = 40 → initial_cards = remaining_cards + lost_cards → initial_cards = 47.0 :=
by
  intros lost_cards remaining_cards initial_cards h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end kevin_initial_cards_l550_550191


namespace compute_expression_l550_550406

theorem compute_expression (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 :=
by
  sorry

end compute_expression_l550_550406


namespace sqrt_sum_lt_sqrt_sum_l550_550442

variables {a b c d : ℝ}

-- Helper conditions to enforce the positive nature and relationship between a, b, c, d
axiom h1 : 0 < a
axiom h2 : 0 < b
axiom h3 : 0 < c
axiom h4 : 0 < d
axiom h5 : a + b = c + d
axiom h6 : a < c
axiom h7 : c ≤ d
axiom h8 : d < b

theorem sqrt_sum_lt_sqrt_sum : sqrt a + sqrt b < sqrt c + sqrt d :=
sorry

end sqrt_sum_lt_sqrt_sum_l550_550442


namespace find_f_f_5_l550_550978

noncomputable def f : ℝ → ℝ := sorry

axiom f_functional_eqn : ∀ x : ℝ, f(x + 2) = 1 / f(x)
axiom f_at_1 : f 1 = -5

theorem find_f_f_5 : f (f 5) = -1/5 :=
by
  sorry

end find_f_f_5_l550_550978


namespace max_sector_area_l550_550095

def sector_area (r : ℝ) : ℝ :=
  let l := 60 - 2 * r in
  (1 / 2) * l * r

theorem max_sector_area : ∃ r : ℝ, (0 < r ∧ r < 30) ∧ sector_area r = 225 :=
sorry

end max_sector_area_l550_550095


namespace max_consecutive_divisible_l550_550808

noncomputable def sequence_x (m : ℕ) (i : ℕ) : ℕ :=
  if 0 ≤ i ∧ i ≤ m-1 then 2^i else ∑ j in finset.range m, sequence_x m (i - (j + 1))

theorem max_consecutive_divisible (m : ℕ) (h : m > 1) :
  ∃ k, k = m - 1 ∧ ∀ i, 1 ≤ i ∧ i ≤ k → m ∣ sequence_x m i :=
sorry

end max_consecutive_divisible_l550_550808


namespace find_fake_coin_l550_550638

theorem find_fake_coin (k : ℕ) :
  ∃ strategy : (Π (tests : ℕ → set ℕ) (dogs : ℕ → ℕ → bool) (sick : ℕ), 
    {fake : ℕ // fake < 2 ^ (2 ^ k)}), 
  (∀ (tests : ℕ → set ℕ) (dogs : ℕ → ℕ → bool) (sick : ℕ),
    (∃ sick_dog : ℕ, dogs sick_dog = sick) →
    ∃ fake, strategy tests dogs sick = fake ∧ tests_card ≤ 2^k + k + 2) := 
by
  sorry

end find_fake_coin_l550_550638


namespace tiger_time_to_pass_specific_point_l550_550702

theorem tiger_time_to_pass_specific_point :
  ∀ (distance_tree : ℝ) (time_tree : ℝ) (length_tiger : ℝ),
  distance_tree = 20 →
  time_tree = 5 →
  length_tiger = 5 →
  (length_tiger / (distance_tree / time_tree)) = 1.25 :=
by
  intros distance_tree time_tree length_tiger h1 h2 h3
  rw [h1, h2, h3]
  sorry

end tiger_time_to_pass_specific_point_l550_550702


namespace total_weight_correct_l550_550942

variable (c1 c2 w2 c : Float)

def total_weight (c1 c2 w2 c : Float) (W x : Float) :=
  (c1 * x + c2 * w2) / (x + w2) = c ∧ W = x + w2

theorem total_weight_correct :
  total_weight 9 8 12 8.40 20 8 :=
by sorry

end total_weight_correct_l550_550942


namespace shortest_tangent_length_l550_550200

noncomputable def length_of_tangent_segment (x y : ℝ) :=
  let C1 := (x - 8)^2 + (y - 3)^2 = 49
  let C2 := (x + 12)^2 + (y + 4)^2 = 25
  sqrt 449

theorem shortest_tangent_length :
  let C1 := (x - 8)^2 + (y - 3)^2 = 49
  let C2 := (x + 12)^2 + (y + 4)^2 = 25
  length_of_tangent_segment x y = sqrt 449 :=
sorry

end shortest_tangent_length_l550_550200


namespace burglar_goods_value_l550_550574

theorem burglar_goods_value (V : ℝ) (S : ℝ) (S_increased : ℝ) (S_total : ℝ) (h1 : S = V / 5000) (h2 : S_increased = 1.25 * S) (h3 : S_total = S_increased + 2) (h4 : S_total = 12) : V = 40000 := by
  sorry

end burglar_goods_value_l550_550574


namespace alpha_value_l550_550091

noncomputable def f (x α : ℝ) := x^α

theorem alpha_value (α : ℝ) (h : deriv (f x α) (-1) = -4) : α = 4 :=
sorry

end alpha_value_l550_550091


namespace area_of_hexagon_l550_550380

-- Define the convex hexagon with the given conditions
def ABCDEF : Type := sorry

-- The conditions of the problem
variables (A B F : ABCDEF) (angle_A angle_B angle_F : ℝ)
variables (AB BC CD DE EF FA : ℝ)

noncomputable def hexagon_properties : Prop :=
  (angle_A = 120 * Real.pi / 180) ∧
  (angle_B = 120 * Real.pi / 180) ∧
  (angle_F = 120 * Real.pi / 180) ∧
  AB = 2 ∧
  BC = 2 ∧
  CD = 3 ∧
  DE = 3 ∧
  EF = 4 ∧
  FA = 4

-- The theorem to be proved
theorem area_of_hexagon (ABCDEF_properties : hexagon_properties)
: ℝ :=
∃ (area : ℝ), area = (63 * Real.sqrt 3) / 4

end area_of_hexagon_l550_550380


namespace geometric_series_sum_l550_550735

theorem geometric_series_sum :
  let a := 2
  let r := 2
  let n := 9
  Sum (map (λ k, a * r^k) (range n)) = 1022 := by
  -- sorry is a placeholder for proof steps which are not needed now
  sorry

end geometric_series_sum_l550_550735


namespace radius_of_wheel_l550_550266

-- Define the given conditions
def distance_covered_by_wheel := 253.44
def number_of_revolutions := 180
def pi_approx := Real.pi -- using Lean's definition of pi

-- Assuming the circumference formula and distance per revolution
def circumference (r : ℝ) : ℝ := 2 * pi_approx * r
def distance_per_revolution : ℝ := distance_covered_by_wheel / number_of_revolutions

-- Define the target radius
def target_radius : ℝ := distance_per_revolution / (2 * pi_approx)

-- The theorem we want to prove
theorem radius_of_wheel : target_radius ≈ 0.224 :=
by
  sorry

end radius_of_wheel_l550_550266


namespace find_f_inv_243_l550_550855

noncomputable def f (x : ℕ) : ℕ := sorry

axiom h₁ : f 5 = 3
axiom h₂ : ∀ x, f (3 * x) = 3 * f x

theorem find_f_inv_243 : f⁻¹ 243 = 405 := sorry

end find_f_inv_243_l550_550855


namespace men_with_tv_at_least_11_l550_550877

-- Definitions for the given conditions
def total_men : ℕ := 100
def married_men : ℕ := 81
def men_with_radio : ℕ := 85
def men_with_ac : ℕ := 70
def men_with_tv_radio_ac_and_married : ℕ := 11

-- The proposition to prove the minimum number of men with TV
theorem men_with_tv_at_least_11 :
  ∃ (T : ℕ), T ≥ men_with_tv_radio_ac_and_married := 
by
  sorry

end men_with_tv_at_least_11_l550_550877


namespace inequality_cubed_l550_550417

theorem inequality_cubed (a b : ℝ) (h : a < b ∧ b < 0) : a^3 ≤ b^3 :=
sorry

end inequality_cubed_l550_550417


namespace train_length_l550_550689

-- Definitions of conditions
noncomputable def speed_kmph : ℝ := 72
noncomputable def platform_length_meters : ℝ := 80
noncomputable def time_seconds : ℝ := 26
noncomputable def speed_mps : ℝ := (speed_kmph * 1000) / 3600

-- Main theorem statement
theorem train_length :
  let distance_covered := speed_mps * time_seconds in
  let train_length := distance_covered - platform_length_meters in
  train_length = 440 := by
  sorry

end train_length_l550_550689


namespace monotonic_increasing_interval_l550_550619

theorem monotonic_increasing_interval :
  ∀ x : ℝ, x ∈ Icc (1/2 : ℝ) 2 ↔
  (∃ (y : ℝ), y = (1/2) ^ (Real.sqrt (-x^2 + x + 2)) ∧ monotone_increasing (λ x, (1/2) ^ (Real.sqrt (-x^2 + x + 2)))) :=
by 
  sorry

end monotonic_increasing_interval_l550_550619


namespace min_x2_y2_l550_550548

theorem min_x2_y2 (x y : ℝ) (h : 2 * (x^2 + y^2) = x^2 + y + x * y) : 
  (∃ x y, x = 0 ∧ y = 0) ∨ x^2 + y^2 >= 1 := 
sorry

end min_x2_y2_l550_550548


namespace cos_1_approx_to_01_no_zero_points_in_interval_l550_550566

noncomputable def cos_approx (x : ℝ) := 1 - (x^2) / 2 + (x^4) / (4*3*2) - (x^6) / (6*5*4*3*2)  -- Compares with cos_taylor_series

theorem cos_1_approx_to_01 : abs (cos_approx 1 - cos 1) < 0.01 :=
by
  sorry

theorem no_zero_points_in_interval : ∀ x ∈ Set.Ioo ((2:ℝ)/3) 1, (exp x - 1/x) ≠ 0 :=
by
  sorry

end cos_1_approx_to_01_no_zero_points_in_interval_l550_550566


namespace clothing_price_decrease_l550_550661

theorem clothing_price_decrease (P : ℝ) (h₁ : P > 0) :
  let price_first_sale := (4 / 5) * P
  let price_second_sale := (1 / 2) * P
  let price_difference := price_first_sale - price_second_sale
  let percent_decrease := (price_difference / price_first_sale) * 100
  percent_decrease = 37.5 :=
by
  sorry

end clothing_price_decrease_l550_550661


namespace simplify_problem_l550_550960

theorem simplify_problem :
  ( (2 + complex.i) / (2 - complex.i) )^8 = (93553 - 2884 * complex.i) / 390625 :=
sorry

end simplify_problem_l550_550960


namespace range_of_m_three_zeros_l550_550857

theorem range_of_m_three_zeros (m : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (x^3 - 3*x + m = 0) ∧ (y^3 - 3*y + m = 0) ∧ (z^3 - 3*z + m = 0)) ↔ -2 < m ∧ m < 2 :=
by
  sorry

end range_of_m_three_zeros_l550_550857


namespace leak_empty_time_l550_550691

-- Define the given conditions
def tank_volume := 2160 -- Tank volume in litres
def inlet_rate := 6 * 60 -- Inlet rate in litres per hour
def combined_empty_time := 12 -- Time in hours to empty the tank with the inlet on

-- Define the derived conditions
def net_rate := tank_volume / combined_empty_time -- Net rate of emptying in litres per hour

-- Define the rate of leakage
def leak_rate := inlet_rate - net_rate -- Rate of leak in litres per hour

-- Prove the main statement
theorem leak_empty_time : (2160 / leak_rate) = 12 :=
by
  unfold leak_rate
  exact sorry

end leak_empty_time_l550_550691


namespace find_m_over_n_l550_550546

-- Define the parameters and conditions
variables (m n : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0)

-- Define the condition that (5 - 3i)(m + ni) is pure imaginary
def pure_imaginary_condition (m n : ℝ) : Prop :=
  let re_part := 5 * m + 3 * n in
  re_part = 0

-- The statement of the theorem
theorem find_m_over_n (h : pure_imaginary_condition m n) : m / n = -3 / 5 := 
by sorry

end find_m_over_n_l550_550546


namespace sum_first_9_terms_l550_550429

variable {a : ℕ → ℝ}  -- Define the geometric sequence a_n
variable {b : ℕ → ℝ}  -- Define the arithmetic sequence b_n

-- Conditions given in the problem
axiom geo_seq (a : ℕ → ℝ) : ∀ n m, a (n + m) = a n * a m
axiom condition1 : a 2 * a 8 = 4 * a 5
axiom arith_seq (b : ℕ → ℝ) : ∀ n m, m > n → b m = b n + (m - n) * (b 1 - b 0)
axiom condition2 : b 4 + b 6 = a 5

-- Statement to prove
theorem sum_first_9_terms : (Finset.range 9).sum b = 18 :=
by {
  sorry
}

end sum_first_9_terms_l550_550429


namespace initial_number_of_children_l550_550894

-- Define the initial conditions
variables {X : ℕ} -- Initial number of children on the bus
variables (got_off got_on children_after : ℕ)
variables (H1 : got_off = 10)
variables (H2 : got_on = 5)
variables (H3 : children_after = 16)

-- Define the theorem to be proved
theorem initial_number_of_children (H : X - got_off + got_on = children_after) : X = 21 :=
by sorry

end initial_number_of_children_l550_550894


namespace rectangle_area_l550_550060

theorem rectangle_area (P l w : ℝ) (h1 : P = 60) (h2 : l / w = 3 / 2) (h3 : P = 2 * l + 2 * w) : l * w = 216 :=
by
  sorry

end rectangle_area_l550_550060


namespace constant_term_is_21_l550_550642

def poly1 (x : ℕ) := x^3 + x^2 + 3
def poly2 (x : ℕ) := 2*x^4 + x^2 + 7
def expanded_poly (x : ℕ) := poly1 x * poly2 x

theorem constant_term_is_21 : expanded_poly 0 = 21 := by
  sorry

end constant_term_is_21_l550_550642


namespace compare_three_and_negfour_l550_550029

theorem compare_three_and_negfour : 3 > -4 := by
  sorry

end compare_three_and_negfour_l550_550029


namespace power_function_odd_l550_550454

theorem power_function_odd (a b : ℝ) 
  (h₁ : (a^2 - 6 * a + 10) = 1)
  (h₂ : a^b = 1 / 3)
  (h₃ : f = (λ x, x^b)) :
  let a := 3 in
  let b := -1 in
  ∀ x, f(-x) = -f(x) := 
by
  intro x
  sorry

end power_function_odd_l550_550454


namespace nebraska_license_plate_increase_l550_550571

open Nat

theorem nebraska_license_plate_increase :
  let old_plates : ℕ := 26 * 10^3
  let new_plates : ℕ := 26^2 * 10^4
  new_plates / old_plates = 260 :=
by
  -- Definitions based on conditions
  let old_plates : ℕ := 26 * 10^3
  let new_plates : ℕ := 26^2 * 10^4
  -- Assertion to prove
  show new_plates / old_plates = 260
  sorry

end nebraska_license_plate_increase_l550_550571


namespace trig_solutions_count_l550_550046

theorem trig_solutions_count :
  ∀ (x : ℝ), (0 ≤ x ∧ x ≤ 2 * Real.pi) →
  (3 * Real.sin x ^ 2 - 4 * Real.sin x * Real.cos x + Real.cos x ^ 2 = 0) →
  {x | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 3 * Real.sin x ^ 2 - 4 * Real.sin x * Real.cos x + Real.cos x ^ 2 = 0}.card = 6 := by
  sorry

end trig_solutions_count_l550_550046


namespace possible_quadrilateral_areas_l550_550713
open Real

def Point := (ℝ × ℝ)

noncomputable def side_points (p1 p2 : Point) : list Point :=
  [p1, ((p1.1 + p2.1)/4, (p1.2 + p2.2)/4), ((2 * p1.1 + 2 * p2.1)/4, (2 * p1.2 + 2 * p2.2)/4), ((3 * p1.1 + p2.1)/4, (3 * p1.2 + p2.2)/4), p2]

noncomputable def quadrilateral_points : list (list Point) :=
  let A := (0, 0)
  let B := (4, 0)
  let C := (4, 4)
  let D := (0, 4)
  [side_points A B, side_points B C, side_points C D, side_points D A]

noncomputable def area_of_quadrilateral (a b c d : Point) : ℝ :=
  0.5 * abs ((a.1 * b.2 + b.1 * c.2 + c.1 * d.2 + d.1 * a.2)
              - (a.2 * b.1 + b.2 * c.1 + c.2 * d.1 + d.2 * a.1))

theorem possible_quadrilateral_areas :
  let points := quadrilateral_points
  ∃ (areas : list ℝ), {6, 7, 7.5, 8, 8.5, 9, 10} ⊆ areas ∧
    ∀ (a ∈ points.head) (b ∈ points.nth 1) (c ∈ points.nth 2) (d ∈ points.nth 3),
      area_of_quadrilateral a b c d ∈ areas :=
by
  sorry

end possible_quadrilateral_areas_l550_550713


namespace correct_usage_of_verb_l550_550669

def verb_usage_sentence : Prop := 
  "We have been having Pizza every night for a week now."

def verb_usages : list string := ["We had Pizza every night for a week now.",
                                  "We were having Pizza every night for a week now.",
                                  "We have been having Pizza every night for a week now.",
                                  "We will be having Pizza every night for a week now."]

theorem correct_usage_of_verb : verb_usage_sentence ∈ verb_usages ∧ 
  "We have been having Pizza every night for a week now." = verb_usages.nth_le 2 (by norm_num) :=
by
  sorry

end correct_usage_of_verb_l550_550669


namespace function_range_minimum_side_length_l550_550130

-- Problem 1: Range of the function
theorem function_range (x : ℝ) (hx : x ∈ set.Icc 0 (real.pi / 2)) : 
  0 ≤ 2 * real.sin x * real.sin (x + real.pi / 6) ∧ 2 * real.sin x * real.sin (x + real.pi / 6) ≤ 1 + real.sqrt 3 / 2 :=
sorry

-- Problem 2: Minimum side length
theorem minimum_side_length (A : ℝ) (a b c : ℝ) (ha : f(A) = real.sqrt 3) (hA : 0 < A ∧ A < real.pi / 2) (hbc : (1/2) * b * c * real.sin A = 2 * real.sqrt 3) :
  a^2 ≥ 8 ∧ (a = 2 * real.sqrt 2) :=
sorry

end function_range_minimum_side_length_l550_550130


namespace right_angled_quadrilateral_solvable_l550_550829

noncomputable def construct_right_angled_quadrilateral (E F G H : Point) (GH a : ℝ) : ℝ :=
  if GH > a then 12 else 0

theorem right_angled_quadrilateral_solvable 
  (E F G H : Point) 
  (GH a : ℝ) :
  GH > a ↔ construct_right_angled_quadrilateral E F G H GH a = 12 :=
begin
  sorry
end

end right_angled_quadrilateral_solvable_l550_550829


namespace international_news_duration_l550_550614

theorem international_news_duration
  (total_duration : ℕ := 30)
  (national_news : ℕ := 12)
  (sports : ℕ := 5)
  (weather_forecasts : ℕ := 2)
  (advertising : ℕ := 6) :
  total_duration - national_news - sports - weather_forecasts - advertising = 5 :=
by
  sorry

end international_news_duration_l550_550614


namespace range_of_f_le_3_l550_550209

noncomputable def f : ℝ → ℝ :=
λ x, if x >= 8 then x^(1/3) else 2 * Real.exp(x - 8)

theorem range_of_f_le_3 :
    {x : ℝ | f x ≤ 3} = {x : ℝ | x ≤ 27} :=
by
  sorry

end range_of_f_le_3_l550_550209


namespace correct_statements_conclusion_l550_550121

variables (a m n : EuclideanSpace ℝ (Fin 3))

-- Line l is orthogonal to plane α if a is parallel to m (i.e., m is a scalar multiple of a)
def line_perpendicular_to_plane (a m : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∃ k : ℝ, m = k • a

-- Plane α is orthogonal to plane β if their normal vectors are orthogonal
def planes_perpendicular (m n : EuclideanSpace ℝ (Fin 3)) : Prop :=
  inner m n = 0

theorem correct_statements_conclusion :
  (line_perpendicular_to_plane a m → ⟪a, m⟫ = 0 → ∀ α : ℝ, α = 0) ∧
  (planes_perpendicular m n → inner m n = 0) :=
sorry

end correct_statements_conclusion_l550_550121


namespace symmetric_point_2_5_8_eq_l550_550509

def symmetric_point_xoy (M : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (M.1, M.2, -M.3)

theorem symmetric_point_2_5_8_eq : symmetric_point_xoy (2, 5, 8) = (2, 5, -8) :=
by
  sorry

end symmetric_point_2_5_8_eq_l550_550509


namespace minimum_value_of_y_l550_550261

noncomputable def y (cos_x : ℝ) : ℝ := cos_x ^ 2 - 3 * cos_x + 2

theorem minimum_value_of_y :
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ y x = 0 :=
begin
  sorry
end

end minimum_value_of_y_l550_550261


namespace problem_statement_l550_550212

open Real

theorem problem_statement (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (a : ℝ := x + x⁻¹) (b : ℝ := y + y⁻¹) (c : ℝ := z + z⁻¹) :
  a > 2 ∧ b > 2 ∧ c > 2 :=
by sorry

end problem_statement_l550_550212


namespace product_of_positive_real_part_solutions_l550_550161

def is_solution (x : ℂ) :=
  x^6 = -729

def has_positive_real_part (x : ℂ) :=
  x.re > 0

theorem product_of_positive_real_part_solutions :
  ∏ (x : ℂ) in {x : ℂ | is_solution x ∧ has_positive_real_part x}, x = 9 :=
by sorry

end product_of_positive_real_part_solutions_l550_550161


namespace odd_coeffs_sum_ge_first_l550_550403

noncomputable def is_odd (n : ℤ) : Prop := n % 2 = 1

def number_of_odd_coeffs (P : Polynomial ℤ) : ℕ :=
P.coeff.support.count (λ i, is_odd (P.coeff i))

def Q (i : ℕ) : Polynomial ℤ := (1 + X : Polynomial ℤ) ^ i

theorem odd_coeffs_sum_ge_first {n : ℕ} (i : Fin n → ℕ) (hordered : ∀ j k : Fin n, j < k → i j < i k) :
  number_of_odd_coeffs (∑ j in Finset.univ, Q (i j)) ≥ number_of_odd_coeffs (Q (i 0)) :=
sorry

end odd_coeffs_sum_ge_first_l550_550403


namespace limit_calculation_l550_550024

open Real

theorem limit_calculation :
    tendsto (fun x => (exp (tan (2 * x)) - exp (-sin (2 * x))) / (sin x - 1))
            (𝓝 (π / 2)) (𝓝 0) :=
sorry

end limit_calculation_l550_550024


namespace interior_intersection_points_12x12_l550_550262

theorem interior_intersection_points_12x12 : 
  let n := 12,
  v_lines := n - 1,
  h_lines := n - 1
  in v_lines * h_lines = 121 := by
  let n := 12
  let v_lines := n - 1
  let h_lines := n - 1
  show v_lines * h_lines = 121 from sorry

end interior_intersection_points_12x12_l550_550262


namespace geometric_series_sum_l550_550030

theorem geometric_series_sum :
  ∀ (a r : ℤ) (n : ℕ),
  a = 3 → r = -2 → n = 10 →
  (a * ((r ^ n - 1) / (r - 1))) = -1024 :=
by
  intros a r n ha hr hn
  rw [ha, hr, hn]
  sorry

end geometric_series_sum_l550_550030


namespace slope_of_line_l550_550746

theorem slope_of_line (x1 y1 x2 y2 : ℝ)
  (h1 : 4 * y1 + 6 * x1 = 0)
  (h2 : 4 * y2 + 6 * x2 = 0)
  (h1x2 : x1 ≠ x2) :
  (y2 - y1) / (x2 - x1) = -3 / 2 :=
by sorry

end slope_of_line_l550_550746


namespace coefficient_j_l550_550387

theorem coefficient_j (j k : ℝ) (p : Polynomial ℝ) (h : p = Polynomial.C 400 + Polynomial.X * Polynomial.C k + Polynomial.X^2 * Polynomial.C j + Polynomial.X^4) :
  (∃ a d : ℝ, (d ≠ 0) ∧ (0 > (4*a + 6*d)) ∧ (p.eval a = 0) ∧ (p.eval (a + d) = 0) ∧ (p.eval (a + 2*d) = 0) ∧ (p.eval (a + 3*d) = 0)) → 
  j = -40 :=
by
  sorry

end coefficient_j_l550_550387


namespace number_of_people_is_five_l550_550630

-- Define the variables and conditions
variable (n : Nat) (T : Nat)
variable (avg_age_current : Nat)
variable (youngest_age : Nat)
variable (avg_age_past : Nat)

-- Define the conditions
def condition1 := avg_age_current = 30
def condition2 := youngest_age = 6
def condition3 := avg_age_past = 24
def total_age_current := T = n * avg_age_current
def total_age_past := (T - youngest_age) = (n - 1) * avg_age_past

-- The theorem to prove
theorem number_of_people_is_five (h1 : condition1) (h2 : condition2) (h3 : condition3)
    (h4 : total_age_current) (h5 : total_age_past) : n = 5 := by
  sorry

end number_of_people_is_five_l550_550630


namespace rational_exponentiation_l550_550489

theorem rational_exponentiation (a : ℝ) (h : a > 0) : (a ^ (-1 / 4)) ^ 4 = 1 / a :=
by sorry

end rational_exponentiation_l550_550489


namespace gcd_g102_g103_eq_one_l550_550542

def g (x : ℤ) : ℤ := x^2 - 2*x + 2023

theorem gcd_g102_g103_eq_one : Nat.gcd (g 102).natAbs (g 103).natAbs = 1 := by
  sorry

end gcd_g102_g103_eq_one_l550_550542


namespace cubic_equation_sum_of_cubes_l550_550254

noncomputable def sum_of_cubes_of_roots : Prop :=
  let x1, x2, x3 : Real
  in
  (2 * x1^3 + 3 * x1^2 - 11 * x1 + 6 = 0) ∧ (2 * x2^3 + 3 * x2^2 - 11 * x2 + 6 = 0) ∧ (2 * x3^3 + 3 * x3^2 - 11 * x3 + 6 = 0) →
  x1^3 + x2^3 + x3^3 = -99 / 8

theorem cubic_equation_sum_of_cubes (x1 x2 x3 : Real) (h1 : 2 * x1^3 + 3 * x1^2 - 11 * x1 + 6 = 0) (h2 : 2 * x2^3 + 3 * x2^2 - 11 * x2 + 6 = 0) (h3 : 2 * x3^3 + 3 * x3^2 - 11 * x3 + 6 = 0) : 
  x1^3 + x2^3 + x3^3 = -99 / 8 := sorry

end cubic_equation_sum_of_cubes_l550_550254


namespace sum_of_squares_arithmetic_series_l550_550078

theorem sum_of_squares_arithmetic_series :
  let a1 := 1 in
  let d := 3 in
  let n := 45 in
  let sum_of_squares := (n / 6) * (2 * a1 + (n - 1) * d) * (2 * a1 + (n - 1) * d + 3 * d) in
  sum_of_squares = 143565 :=
by
  let a1 := 1
  let d := 3
  let n := 45
  let sum_of_squares := (n / 6) * (2 * a1 + (n - 1) * d) * (2 * a1 + (n - 1) * d + 3 * d)
  sorry

end sum_of_squares_arithmetic_series_l550_550078


namespace parabola_focus_distance_sum_correct_l550_550153

noncomputable def parabola_focus_distance_sum 
  (x1 x2 x3 x4 : ℝ) 
  (h1 : ∀ (x y : ℝ), y^2 = 8 * x → ∃ k, (x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4))
  (hx : x1 + x2 + x3 + x4 = 10) 
  : ℝ :=
  (x1 + 2) + (x2 + 2) + (x3 + 2) + (x4 + 2)

theorem parabola_focus_distance_sum_correct
  (x1 x2 x3 x4 : ℝ) 
  (h1 : ∀ (x y : ℝ), y^2 = 8 * x → ∃ k, (x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4))
  (hx : x1 + x2 + x3 + x4 = 10) 
  : parabola_focus_distance_sum x1 x2 x3 x4 h1 hx = 18 :=
by
  sorry

end parabola_focus_distance_sum_correct_l550_550153


namespace jenny_sold_boxes_l550_550514

-- Given conditions as definitions
def cases : ℕ := 3
def boxes_per_case : ℕ := 8

-- Mathematically equivalent proof problem
theorem jenny_sold_boxes : cases * boxes_per_case = 24 := by
  sorry

end jenny_sold_boxes_l550_550514


namespace findMonicQuadraticPolynomial_l550_550067

-- Define the root as a complex number
def root : ℂ := -3 - complex.I * real.sqrt 8

-- Define the conditions
def isMonic (p : polynomial ℝ) : Prop := p.leadingCoeff = 1
def hasRealCoefficients (p : polynomial ℝ) : Prop := ∀ a ∈ p.support, is_real (p.coeff a)

-- Define the polynomial
noncomputable def polynomial : polynomial ℝ :=
  polynomial.C 1 * polynomial.X^2 + polynomial.C 6 * polynomial.X + polynomial.C 17

-- The target statement
theorem findMonicQuadraticPolynomial :
  ∀ (p : polynomial ℝ), 
  isMonic p ∧ hasRealCoefficients p ∧ (root ∈ p.roots) →
  p = polynomial :=
by
  sorry

end findMonicQuadraticPolynomial_l550_550067


namespace bobby_candies_per_day_from_Mon_to_Fri_l550_550021

variable (x : ℕ)

-- Conditions
def candies_per_week (x : ℕ) := 5 * x + 2
def total_candies (weeks : ℕ) (candies_each_week : ℕ) := weeks * candies_each_week
def all_candies (packets : ℕ) (candies_per_packet : ℕ) := packets * candies_per_packet

-- Given conditions
def packets := 2
def candies_per_packet := 18
def weeks := 3
def total_candies_in_packets := all_candies packets candies_per_packet

-- Theorem statement
theorem bobby_candies_per_day_from_Mon_to_Fri : total_candies weeks (candies_per_week x) = total_candies_in_packets → x = 2 :=
by
  -- Hypothesis
  intro h,
  -- Define the hypothesis
  have : total_candies weeks (candies_per_week x) = total_candies_in_packets := h,
  sorry

end bobby_candies_per_day_from_Mon_to_Fri_l550_550021


namespace regression_equation_l550_550889

theorem regression_equation (x y : ℕ → ℝ)
(h1 : ∑ i in finset.range 5, x i = 25)
(h2 : ∑ i in finset.range 5, y i = 250)
(h3 : ∑ i in finset.range 5, (x i)^2 = 145)
(h4 : ∑ i in finset.range 5, (x i) * (y i) = 1380) :
∃ a b, (∀ x, y x = b * x + a) ∧ b = 6.5 ∧ a = 17.5 :=
begin
  sorry,
end

end regression_equation_l550_550889


namespace region_area_is_13_l550_550232

noncomputable def area_rect_outside_circles : ℝ :=
  let PQ : ℝ := 4
  let PS : ℝ := 6
  let rP : ℝ := 2
  let rQ : ℝ := 3
  let rR : ℝ := 1
  let rect_area := PQ * PS
  let circle_area (r : ℝ) : ℝ := π * r^2
  let quarter_circle_area (r : ℝ) : ℝ := (circle_area r) / 4
  let total_quarter_circle_area :=
    quarter_circle_area rP + quarter_circle_area rQ + quarter_circle_area rR
  rect_area - total_quarter_circle_area

theorem region_area_is_13 :
  area_rect_outside_circles = 13 :=
by
  sorry

end region_area_is_13_l550_550232


namespace Union_A_B_eq_l550_550108

noncomputable def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}
noncomputable def B : Set ℝ := {x | -2 < x ∧ x < 2}

theorem Union_A_B_eq : A ∪ B = {x | -2 < x ∧ x ≤ 4} :=
by
  sorry

end Union_A_B_eq_l550_550108


namespace number_of_factors_l550_550603

theorem number_of_factors (b n : ℕ) (hb1 : b = 6) (hn1 : n = 15) (hb2 : b > 0) (hb3 : b ≤ 15) (hn2 : n > 0) (hn3 : n ≤ 15) :
  let factors := (15 + 1) * (15 + 1)
  factors = 256 :=
by
  sorry

end number_of_factors_l550_550603


namespace recipe_calls_for_eight_cups_of_sugar_l550_550564

def cups_of_flour : ℕ := 6
def cups_of_salt : ℕ := 7
def additional_sugar_needed (salt : ℕ) : ℕ := salt + 1

theorem recipe_calls_for_eight_cups_of_sugar :
  additional_sugar_needed cups_of_salt = 8 :=
by
  -- condition 1: cups_of_flour = 6
  -- condition 2: cups_of_salt = 7
  -- condition 4: additional_sugar_needed = salt + 1
  -- prove formula: 7 + 1 = 8
  sorry

end recipe_calls_for_eight_cups_of_sugar_l550_550564


namespace selection_methods_correct_l550_550235

open Finset

-- Defining the set consisting of 10 village officials
def officials := range 10

-- Conditions as given in the problem
def A := 0
def B := 1
def C := 2

-- We are selecting from the remaining 9 since B is not chosen
def officials_without_B := (officials \ {B})

-- Counting the number of ways to choose 3 officials where at least one is A or C
def select_3_with_condition : ℕ :=
  (officials_without_B.choose 3).filter (λ s, A ∈ s ∨ C ∈ s).card

-- The expected answer given the conditions
def expected_answer : ℕ := 49

-- Prove that the counting is correct as per the solution
theorem selection_methods_correct :
  select_3_with_condition = expected_answer :=
sorry

end selection_methods_correct_l550_550235


namespace river_bank_bottom_width_l550_550973

/-- 
The cross-section of a river bank is a trapezium with a 12 m wide top and 
a certain width at the bottom. The area of the cross-section is 500 sq m 
and the depth is 50 m. Prove that the width at the bottom is 8 m.
-/
theorem river_bank_bottom_width (area height top_width : ℝ) (h_area: area = 500) 
(h_height : height = 50) (h_top_width : top_width = 12) : ∃ b : ℝ, (1 / 2) * (top_width + b) * height = area ∧ b = 8 :=
by
  use 8
  sorry

end river_bank_bottom_width_l550_550973


namespace geom_seq_val_l550_550504

noncomputable def is_geom_seq (a : ℕ → ℝ) : Prop :=
∃ q b, ∀ n, a n = b * q^n

variables (a : ℕ → ℝ)

axiom a_5_a_7 : a 5 * a 7 = 2
axiom a_2_plus_a_10 : a 2 + a 10 = 3

theorem geom_seq_val (a_geom : is_geom_seq a) :
  (a 12) / (a 4) = 2 ∨ (a 12) / (a 4) = 1 / 2 :=
sorry

end geom_seq_val_l550_550504


namespace total_rounded_amount_is_20_l550_550576

-- Define the rounding rule based on the specific requirements
def modified_round (x : ℝ) : ℤ :=
  if x - x.floor == 0.5 then x.floor else x.round

-- Define the individual purchases
def purchase1 := 2.99
def purchase2 := 6.51
def purchase3 := 10.49

-- Calculate the rounded total amount using the modified rounding rule
def rounded_total := modified_round purchase1 + modified_round purchase2 + modified_round purchase3

theorem total_rounded_amount_is_20 :
  rounded_total = 20 :=
by
  -- Omitted proof steps
  sorry

end total_rounded_amount_is_20_l550_550576


namespace boots_cost_more_l550_550968

theorem boots_cost_more (S B : ℝ) 
  (h1 : 22 * S + 16 * B = 460) 
  (h2 : 8 * S + 32 * B = 560) : B - S = 5 :=
by
  -- Here we provide the statement only, skipping the proof
  sorry

end boots_cost_more_l550_550968


namespace periodic_function_f_l550_550977

noncomputable def f : ℝ → ℝ :=
sorry

theorem periodic_function_f :
  (∀ x : ℝ, f(x + 2) = 1 / f(x)) →
  f(1) = -5 →
  f(f(5)) = -1 / 5 :=
by
  intros h1 h2
  -- Here we include the statement, but we do not need to construct the proof.
  sorry

end periodic_function_f_l550_550977


namespace square_contains_one_integer_point_l550_550203

noncomputable def diagonal_of_square (a b c d : ℝ) : ℝ :=
  real.sqrt ((c - a)^2 + (d - b)^2)

noncomputable def square_side_length (d : ℝ) : ℝ :=
  d / real.sqrt 2

noncomputable def probability_of_one_integer_point : ℝ :=
  1 / 100

theorem square_contains_one_integer_point 
(a b : ℝ) 
(h1 : a = 2 / 15) 
(h2 : b = 1 / 15) 
(h3 : ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1000 ∧ 0 ≤ y ∧ y ≤ 1000) 
: 
  let d := diagonal_of_square (2 / 15) (1 / 15) (- (2 / 15)) (- (1 / 15)) in 
  let s := square_side_length d in 
  ∃ v : ℝ × ℝ, 
    (0 ≤ v.1 ∧ v.1 ≤ 1000 ∧ 0 ≤ v.2 ∧ v.2 ≤ 1000) ∧ 
    ∃ T_v : set (ℝ × ℝ), 
      T_v = { p : ℝ × ℝ | true } ∧ -- Proper translation and rotation transformation
      probability_of_one_integer_point = 1 / 100 := sorry

end square_contains_one_integer_point_l550_550203


namespace sum_of_digits_eq_11_l550_550861

-- Define the problem conditions
variables (p q r : ℕ)
variables (h1 : 1 ≤ p ∧ p ≤ 9)
variables (h2 : 1 ≤ q ∧ q ≤ 9)
variables (h3 : 1 ≤ r ∧ r ≤ 9)
variables (h4 : p ≠ q ∧ p ≠ r ∧ q ≠ r)
variables (h5 : (10 * p + q) * (10 * p + r) = 221)

-- Define the theorem
theorem sum_of_digits_eq_11 : p + q + r = 11 :=
by
  sorry

end sum_of_digits_eq_11_l550_550861


namespace ceil_sub_self_eq_half_l550_550553

theorem ceil_sub_self_eq_half (n : ℤ) (x : ℝ) (h : x = n + 1/2) : ⌈x⌉ - x = 1/2 :=
by
  sorry

end ceil_sub_self_eq_half_l550_550553


namespace incorrect_reason_gene_mutation_l550_550411

def genotype := String

def individual_A_genotype : genotype := "AaB"
def individual_B_genotype : genotype := "AABb"

theorem incorrect_reason_gene_mutation : 
  ¬(gene_mutation_reason individual_A_genotype individual_B_genotype) :=
sorry

end incorrect_reason_gene_mutation_l550_550411


namespace polynomial_with_complex_root_l550_550069

theorem polynomial_with_complex_root :
  ∃ P : Polynomial ℝ, P.Monic ∧ degree P = 2 ∧
  P.coeff 0 = 17 ∧ P.coeff 1 = 6 ∧ P.coeff 2 = 1 ∧
  (P.eval (-3 - Complex.i * Real.sqrt 8) = 0) :=
sorry

end polynomial_with_complex_root_l550_550069


namespace largest_k_divisible_l550_550911

/-- Let P be the product of the first 150 positive odd integers.
    Prove that the largest integer k such that P is divisible by 3^k is 76. -/
theorem largest_k_divisible (P : ℕ) (hP : P = ∏ i in finset.range 150, 2 * i + 1) : 
  ∃ k : ℕ, (3^k ∣ P) ∧ k = 76 :=
sorry

end largest_k_divisible_l550_550911


namespace eval_expr1_eval_expr2_l550_550754

theorem eval_expr1 : (1 / Real.sin (10 * Real.pi / 180) - Real.sqrt 3 / Real.cos (10 * Real.pi / 180)) = 4 :=
by
  -- proof goes here
  sorry

theorem eval_expr2 : (Real.sin (50 * Real.pi / 180) * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)) - Real.cos (20 * Real.pi / 180)) / (Real.cos (80 * Real.pi / 180) * Real.sqrt (1 - Real.cos (20 * Real.pi / 180))) = Real.sqrt 2 :=
by
  -- proof goes here
  sorry

end eval_expr1_eval_expr2_l550_550754


namespace polynomial_divisibility_l550_550197

noncomputable def polynomial_sequence (n : ℕ) : ℕ → ℕ :=
  λ n, 
    if n = 0 then 2
    else if n = 1 then 3
    else 3 * (polynomial_sequence (n - 1)) + (1 - n - 2 * n^2) * (polynomial_sequence (n - 2))

theorem polynomial_divisibility (n : ℕ) (f : ℕ → ℕ) :
  (f 0 = 2) → 
  (f 1 = 3 * x) → 
  (∀ n ≥ 2, f n = 3 * x * (f (n - 1)) + (1 - x - 2 * x^2) * (f (n - 2))) → 
  (∃ k, n = 6 * k + 3) ↔ polynomial_sequence n \ x^3 - x^2 + x :=
by
  sorry

end polynomial_divisibility_l550_550197


namespace problem_proof_l550_550144

def is_multiple_of_13 (n : ℕ) : Prop := n % 13 = 0

def has_valid_permutation (n : ℕ) : Prop :=
  ∃ p, (permutations_of_digits n p) ∧ (1000 ≤ p ∧ p ≤ 9999 ∧ is_multiple_of_13 p)

def count_valid_integers (s : ℕ) : Prop :=
  s = 1386

theorem problem_proof :
  let count := λ (n : ℕ), 1000 ≤ n ∧ n ≤ 9999 ∧ has_valid_permutation n in
  count_valid_integers (Nat.card { n // count n}) :=
sorry

end problem_proof_l550_550144


namespace no_delightful_8_digit_integers_l550_550709

def is_delightful (n : ℕ) (digits : Fin n → ℕ) : Prop :=
  (∀ k : Fin n.succ, (∑ i in Finset.range (k + 1), digits ⟨i, Nat.lt_succ_self i⟩) % (k + 1) = 0)

theorem no_delightful_8_digit_integers :
  ∀ digits : Fin 8 → ℕ, (∀ i, digits i ∈ Finset.range 1 9) → (∀ i j, i ≠ j → digits i ≠ digits j) →
  ¬ is_delightful 8 digits :=
begin
  sorry
end

end no_delightful_8_digit_integers_l550_550709


namespace range_of_a_l550_550560

section
variable (A : Set ℝ) (B : Set ℝ) (a : ℝ)

def A : Set ℝ := {0, 1}
def B (a : ℝ) : Set ℝ := {x | x > a}

theorem range_of_a : {a | 0 ≤ a ∧ a < 1} = {a | ∃ B, (A ∩ B a).card = 1} :=
by
  sorry
end

end range_of_a_l550_550560


namespace revenue_per_investment_l550_550220

theorem revenue_per_investment (Banks_investments : ℕ) (Elizabeth_investments : ℕ) (Elizabeth_revenue_per_investment : ℕ) (revenue_difference : ℕ) :
  Banks_investments = 8 →
  Elizabeth_investments = 5 →
  Elizabeth_revenue_per_investment = 900 →
  revenue_difference = 500 →
  ∃ (R : ℤ), R = (5 * 900 - 500) / 8 :=
by
  intros h1 h2 h3 h4
  let T_elizabeth := 5 * Elizabeth_revenue_per_investment
  let T_banks := T_elizabeth - revenue_difference
  let R := T_banks / 8
  use R
  sorry

end revenue_per_investment_l550_550220


namespace triangle_identity_l550_550924

theorem triangle_identity (a b c A B C : ℝ) (h1: a = b * (sin A / sin B)) (h2: b = c * (sin B / sin C)):
  (cos (B / 2) * sin (B / 2 + C)) / (cos (C / 2) * sin (C / 2 + B)) = (a + c) / (a + b) :=
sorry

end triangle_identity_l550_550924


namespace circle_radius_l550_550043

theorem circle_radius :
  ∃ radius : ℝ, (∀ (x y : ℝ), (x - 2)^2 + (y - 1)^2 = 16 → (x - 2)^2 + (y - 1)^2 = radius^2)
  ∧ radius = 4 :=
sorry

end circle_radius_l550_550043


namespace victor_brown_sugar_l550_550302

def sugar_amounts (w b : ℝ) : Prop :=
  w = 0.25 ∧ b = w + 0.38

theorem victor_brown_sugar (w b : ℝ) (h : sugar_amounts w b) : b = 0.63 :=
by
  cases h with
  | intro hw hb =>
    rw [hw] at hb
    linarith

end victor_brown_sugar_l550_550302


namespace problem_1_problem_2_l550_550022

-- Problem 1 Lean statement
theorem problem_1 :
  (1 - 1^4 - (1/2) * (3 - (-3)^2)) = 2 :=
by sorry

-- Problem 2 Lean statement
theorem problem_2 :
  ((3/8 - 1/6 - 3/4) * 24) = -13 :=
by sorry

end problem_1_problem_2_l550_550022


namespace extreme_points_f_l550_550806

noncomputable def f (a x : ℝ) : ℝ := x * (Real.log x - 2 * a * x)

theorem extreme_points_f (a x1 x2 : ℝ) (h_a : 0 < a ∧ a < 1/4)
  (h_extremes : ∃ x1 x2, x1 < x2 ∧ ∀ x, (f a x)'' = 0 → (x = x1 ∨ x = x2)) : 
  f a x1 < 0 ∧ f a x2 > -1/2 :=
sorry

end extreme_points_f_l550_550806


namespace continuous_of_deriv_l550_550907

noncomputable def F (f : ℝ → ℝ) : ℝ → ℝ := λ x, ∫ t in 0..x, f t

theorem continuous_of_deriv (f : ℝ → ℝ) (hf_mono : Monotone f) (hf_F_diff : ∀ x, DifferentiableAt ℝ (F f) x) :
  Continuous f :=
sorry

end continuous_of_deriv_l550_550907


namespace cricket_bat_cost_price_l550_550345

theorem cricket_bat_cost_price (CP_A : ℝ) (SP_B : ℝ) (SP_C : ℝ) (h1 : SP_B = CP_A * 1.20) (h2 : SP_C = SP_B * 1.25) (h3 : SP_C = 222) : CP_A = 148 := 
by
  sorry

end cricket_bat_cost_price_l550_550345


namespace max_value_of_min_column_sum_l550_550946

open Matrix

def valid_grid (m : Matrix (Fin 5) (Fin 5) ℕ) : Prop :=
  (∀ i j, m i j ∈ {1, 2, 3, 4, 5}) ∧ 
  (∀ i, (Finset.univ : Finset (Fin 5)).sum (λ j, if m i j = 1 then 1 else 0) = 5) ∧
  (∀ i, (Finset.univ : Finset (Fin 5)).sum (λ j, if m i j = 2 then 1 else 0) = 5) ∧
  (∀ i, (Finset.univ : Finset (Fin 5)).sum (λ j, if m i j = 3 then 1 else 0) = 5) ∧
  (∀ i, (Finset.univ : Finset (Fin 5)).sum (λ j, if m i j = 4 then 1 else 0) = 5) ∧
  (∀ i, (Finset.univ : Finset (Fin 5)).sum (λ j, if m i j = 5 then 1 else 0) = 5) ∧
  (∀ j i₁ i₂,
    abs ((m i₁ j : ℤ) - (m i₂ j : ℤ)) ≤ 2)

def column_sum (m : Matrix (Fin 5) (Fin 5) ℕ) (j : Fin 5) : ℕ :=
  (Finset.univ : Finset (Fin 5)).sum (λ i, m i j)

def min_column_sum (m : Matrix (Fin 5) (Fin 5) ℕ) : ℕ :=
  Finset.fold min nat.le (Finset.univ.image (column_sum m)) 0

theorem max_value_of_min_column_sum : ∃ m : Matrix (Fin 5) (Fin 5) ℕ,
  valid_grid m ∧ (min_column_sum m = 10) :=
sorry

end max_value_of_min_column_sum_l550_550946


namespace differential_equation_solution_l550_550241

def C1 : ℝ := sorry
def C2 : ℝ := sorry

noncomputable def y (x : ℝ) : ℝ := C1 * Real.cos x + C2 * Real.sin x
noncomputable def z (x : ℝ) : ℝ := -C1 * Real.sin x + C2 * Real.cos x

theorem differential_equation_solution : 
  (∀ x : ℝ, deriv y x = z x) ∧ 
  (∀ x : ℝ, deriv z x = -y x) :=
by
  sorry

end differential_equation_solution_l550_550241


namespace solve_quadratic_sum_l550_550925

theorem solve_quadratic_sum :
  ∀ a b : ℝ, (a = 8 ∧ b = -6 ∧ a ≥ b) → (3 * a + 2 * b = 12) :=
by
  intros a b H,
  rcases H with ⟨ha, hb, hab⟩,
  rw [ha, hb],
  norm_num,
  sorry

end solve_quadratic_sum_l550_550925


namespace cheese_distribution_l550_550636

theorem cheese_distribution (total_cheese : ℕ) (initial_cut : ℕ) (remaining_cut : ℕ) :
  total_cheese = 50 ∧ initial_cut = 30 ∧ remaining_cut = 20 →
  (A_cheese : ℕ, B_cheese : ℕ) :=
begin
  -- Provided conditions
  assume h : total_cheese = 50 ∧ initial_cut = 30 ∧ remaining_cut = 20,

  -- Definitions of guarantees
  let A_cheese := 30,
  let B_cheese := 20,

  exact ⟨A_cheese, B_cheese⟩,
end

end cheese_distribution_l550_550636


namespace inradius_formula_l550_550510

variable (R : ℝ) (β γ : ℝ)

theorem inradius_formula :
  let r := 4 * R * sin (β / 2) * sin (γ / 2) * cos ((β + γ) / 2) in
  r = 4 * R * sin (β / 2) * sin (γ / 2) * cos ((β + γ) / 2) :=
by
  sorry

end inradius_formula_l550_550510


namespace sum_of_three_numbers_l550_550271

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 149) 
  (h2 : a * b + b * c + a * c = 70) : 
  a + b + c = 17 := 
begin
  sorry
end

end sum_of_three_numbers_l550_550271


namespace election_margin_of_victory_l550_550632

theorem election_margin_of_victory (T : ℕ) (H_winning_votes : T * 58 / 100 = 1044) :
  1044 - (T * 42 / 100) = 288 :=
by
  sorry

end election_margin_of_victory_l550_550632


namespace geometric_series_sum_l550_550037

theorem geometric_series_sum :
  let a := 3
  let r := -2
  let n := 10
  let S := a * ((r^n - 1) / (r - 1))
  S = -1023 :=
by 
  -- Sorry allows us to omit the proof details
  sorry

end geometric_series_sum_l550_550037


namespace general_formula_an_sum_Tn_l550_550537

def arithmeticSeq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sumFirstNTermsSeq (S : ℕ → ℝ) (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + (n - 1) * d / 2)

def condition1 (S : ℕ → ℝ) : Prop := S 5 = 20
def condition2 (a : ℕ → ℝ) : Prop := a 3 ^ 2 = a 2 * a 5

def sequenceBn (b : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  b 1 = 1 ∧ ∀ n : ℕ, b n + b (n + 1) = (real.sqrt 2) ^ (a n)

def Tn (T : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, T n = b (2 * n)

theorem general_formula_an :
  ∀ (a : ℕ → ℝ) (d : ℝ), arithmeticSeq a d →
    sumFirstNTermsSeq (λ n, n * (a 1 + (n - 1) * d / 2)) a d →
    condition1 (λ n, n * (a 1 + (n - 1) * d / 2)) →
    condition2 a →
    (λ n, a n) = λ n, 2 * (n - 1) :=
by sorry

theorem sum_Tn :
  ∀ (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) (d : ℝ),
    arithmeticSeq a d →
    sumFirstNTermsSeq (λ n, n * (a 1 + (n - 1) * d / 2)) a d →
    condition1 (λ n, n * (a 1 + (n - 1) * d / 2)) →
    condition2 a →
    sequenceBn b a →
    Tn T b →
    (λ n, T n) = λ n, (2 * 4 ^ n - 6 * n - 2) / 9 :=
by sorry

end general_formula_an_sum_Tn_l550_550537


namespace find_least_n_l550_550206

noncomputable def seq : ℕ → ℕ
| 10        := 12
| (n + 1) := 50 * seq n + (n + 1)^2

def least_n_multiple_of_121 (n : ℕ) : Prop :=
  n > 10 ∧ seq n % 121 = 0

theorem find_least_n : ∃ n > 10, seq n % 121 = 0 := by
  use 23
  split
  norm_num
  sorry

end find_least_n_l550_550206


namespace problem_solution_l550_550890

-- Definitions of the curves C1 and C2
def C1_polar (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

def C2_polar (ρ θ : ℝ) : Prop :=
  ρ^2 = 4 / (1 + 3 * (Real.sin θ)^2)

-- The range of |OM| * |ON| given θ = α
def range_OM_ON (α : ℝ) [Fact (α ≥ 0)] [Fact (α ≤ π / 4)] : Set ℝ :=
  let OM := 4 * Real.cos α
  let ON := 2 / Real.sqrt (1 + 3 * (Real.sin α)^2)
  Set.Icc (8 * Real.sqrt (1 - (Real.sin α)^2) / (Real.sqrt (1 + 3 * (Real.sin α)^2))) 8

-- Main theorem to establish the range of |OM| * |ON|
theorem problem_solution (α : ℝ) [Fact (α ≥ 0)] [Fact (α ≤ π / 4)] :
  (range_OM_ON α : Set ℝ) = (Set.Icc ((8 * Real.sqrt 5) / 5) 8) :=
sorry

end problem_solution_l550_550890


namespace probability_correct_l550_550710

def total_chips : ℕ := 15
def total_ways_to_draw_2_chips : ℕ := Nat.choose 15 2

def chips_same_color : ℕ := 3 * (Nat.choose 5 2)
def chips_same_number : ℕ := 5 * (Nat.choose 3 2)
def favorable_outcomes : ℕ := chips_same_color + chips_same_number

def probability_same_color_or_number : ℚ := favorable_outcomes / total_ways_to_draw_2_chips

theorem probability_correct :
  probability_same_color_or_number = 3 / 7 :=
by sorry

end probability_correct_l550_550710


namespace cost_of_flowers_cost_function_minimum_cost_l550_550304

-- Define the costs in terms of yuan
variables (n m : ℕ) -- n is the cost of one lily, m is the cost of one carnation.

-- Define the conditions
axiom cost_condition1 : 2 * n + m = 14
axiom cost_condition2 : 3 * m = 2 * n + 2

-- Prove the cost of one carnation and one lily
theorem cost_of_flowers : n = 5 ∧ m = 4 :=
by {
  sorry
}

-- Variables for the second part
variables (w x : ℕ) -- w is the total cost, x is the number of carnations.

-- Define the conditions
axiom total_condition : 11 = 2 + x + (11 - x)
axiom min_lilies_condition : 11 - x ≥ 2

-- State the relationship between w and x
theorem cost_function : w = 55 - x :=
by {
  sorry
}

-- Prove the minimum cost
theorem minimum_cost : ∃ x, (x ≤ 9 ∧  w = 46) :=
by {
  sorry
}

end cost_of_flowers_cost_function_minimum_cost_l550_550304


namespace find_annual_salary_l550_550568

-- Define given constants and values
def house_cost : ℝ := 450000
def downpayment_percent : ℝ := 0.20
def saving_years : ℕ := 6
def saving_rate : ℝ := 0.10
def downpayment_amount : ℝ := downpayment_percent * house_cost

-- State the problem
theorem find_annual_salary :
  ∃ (annual_salary : ℝ), (saving_rate * annual_salary) * saving_years = downpayment_amount :=
begin
  use 150000, -- This reflects the conclusion we reached above.
  -- Placeholder for the proof
  sorry
end

end find_annual_salary_l550_550568


namespace determine_a_l550_550107

noncomputable def a_value : ℤ :=
  if h : {0, 1} ⊆ {-1, 0, a+3} then -2 else 0

theorem determine_a (a : ℤ) (h : {0, 1} ⊆ {-1, 0, a+3}) : a = -2 := by
  sorry

end determine_a_l550_550107


namespace num_subsets_M_eq_eight_l550_550470

-- Define the set M
def M : Set ℕ := {1, 2, 3}

-- Statement: Prove that the number of subsets of M is 8
theorem num_subsets_M_eq_eight : M.powerset.card = 8 :=
by
  sorry

end num_subsets_M_eq_eight_l550_550470


namespace triangle_DEF_all_acute_l550_550172

theorem triangle_DEF_all_acute
  (α : ℝ)
  (hα : 0 < α ∧ α < 90)
  (DEF : Type)
  (D : DEF) (E : DEF) (F : DEF)
  (angle_DFE : DEF → DEF → DEF → ℝ) 
  (angle_FED : DEF → DEF → DEF → ℝ) 
  (angle_EFD : DEF → DEF → DEF → ℝ)
  (h1 : angle_DFE D F E = 45)
  (h2 : angle_FED F E D = 90 - α / 2)
  (h3 : angle_EFD E D F = 45 + α / 2) :
  (0 < angle_DFE D F E ∧ angle_DFE D F E < 90) ∧ 
  (0 < angle_FED F E D ∧ angle_FED F E D < 90) ∧ 
  (0 < angle_EFD E D F ∧ angle_EFD E D F < 90) := by
  sorry

end triangle_DEF_all_acute_l550_550172


namespace average_speed_of_trip_l550_550328

theorem average_speed_of_trip 
  (total_distance : ℝ)
  (first_leg_distance : ℝ)
  (first_leg_speed : ℝ)
  (second_leg_distance : ℝ)
  (second_leg_speed : ℝ)
  (h_dist : total_distance = 50)
  (h_first_leg : first_leg_distance = 25)
  (h_second_leg : second_leg_distance = 25)
  (h_first_speed : first_leg_speed = 60)
  (h_second_speed : second_leg_speed = 30) :
  (total_distance / 
   ((first_leg_distance / first_leg_speed) + (second_leg_distance / second_leg_speed)) = 40) :=
by
  sorry

end average_speed_of_trip_l550_550328


namespace find_y_l550_550483

theorem find_y 
  (x y z : ℕ) 
  (h₁ : x + y + z = 25)
  (h₂ : x + y = 19) 
  (h₃ : y + z = 18) :
  y = 12 :=
by
  sorry

end find_y_l550_550483


namespace num_integer_solutions_abs_ineq_l550_550848

theorem num_integer_solutions_abs_ineq : 
  ∃ (S : Set ℤ), (∀ (x : ℝ), |x - 3| ≤ 4.5 ↔ ∃ (n: ℤ), n.val = x) ∧ S.card = 9 := 
sorry

end num_integer_solutions_abs_ineq_l550_550848


namespace g_neither_even_nor_odd_l550_550896

def g (x : ℝ) := ⌊x⌋ + x + 1/2

theorem g_neither_even_nor_odd : 
  ¬(∀ x : ℝ, g (-x) = g x) ∧ ¬(∀ x : ℝ, g (-x) = -g x) := 
by
  sorry

end g_neither_even_nor_odd_l550_550896


namespace integer_solutions_count_l550_550081

theorem integer_solutions_count :
  let circle_eq : ∀ (x y : ℤ), x^2 + y^2 = 65 → Prop := 
    λ x y h, true 
  let line_eq : ∀ (a b x y : ℤ), ax + by = 2 → Prop :=
    λ a b x y h, true 
  (∃ (a b : ℤ), ∃ (x y : ℤ), line_eq a b x y (2) ∧ circle_eq x y 65) → 
  ∃ (k : ℕ), k = 128 := sorry

end integer_solutions_count_l550_550081


namespace min_value_x_plus_one_over_x_plus_two_l550_550423

theorem min_value_x_plus_one_over_x_plus_two (x : ℝ) (h : x > -2) : ∃ y : ℝ, y = x + 1/(x + 2) ∧ y ≥ 0 :=
by
  sorry

end min_value_x_plus_one_over_x_plus_two_l550_550423


namespace three_digit_number_is_504_l550_550349

theorem three_digit_number_is_504 (x : ℕ) [Decidable (x = 504)] :
  100 ≤ x ∧ x ≤ 999 →
  (x - 7) % 7 = 0 ∧
  (x - 8) % 8 = 0 ∧
  (x - 9) % 9 = 0 →
  x = 504 :=
by
  sorry

end three_digit_number_is_504_l550_550349


namespace all_guests_know_each_other_l550_550958

-- Define a structure for the setup
structure TableSetup :=
  (Guest : Type)
  (acquainted : Guest → Guest → Prop)
  (mutual_acquaintance : ∀ g1 g2 : Guest, acquainted g1 g2 → acquainted g2 g1)
  (equal_intervals : ∀ g : Guest, ∃ l : List Guest, l.length > 1 ∧ ∀ i j : ℕ, i < j → i < l.length → j < l.length → 
                      acquainted g (l.get i) → acquainted g (l.get j) → (j - i) % (l.length - 1) = 0)
  (common_acquaintance : ∀ g1 g2 : Guest, ∃ g3 : Guest, acquainted g1 g3 ∧ acquainted g2 g3)

-- The main theorem to be proven
theorem all_guests_know_each_other (setup : TableSetup) :
  ∀ g1 g2 : setup.Guest, setup.acquainted g1 g2 :=
  sorry

end all_guests_know_each_other_l550_550958


namespace find_value_of_2a10_minus_a12_l550_550886

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the given conditions
def condition (a : ℕ → ℝ) : Prop :=
  is_arithmetic_sequence a ∧ (a 4 + a 6 + a 8 + a 10 + a 12 = 120)

-- State the theorem
theorem find_value_of_2a10_minus_a12 (a : ℕ → ℝ) (h : condition a) : 2 * a 10 - a 12 = 24 :=
by sorry

end find_value_of_2a10_minus_a12_l550_550886


namespace cylindrical_tank_depth_l550_550311

theorem cylindrical_tank_depth (V : ℝ) (d h : ℝ) (π : ℝ) : 
  V = 1848 ∧ d = 14 ∧ π = Real.pi → h = 12 :=
by
  sorry

end cylindrical_tank_depth_l550_550311


namespace evaluate_expression_l550_550273

theorem evaluate_expression : 
  (-2 : ℤ)^2004 + 3 * (-2)^2003 = (-2)^2003 :=
by
  sorry

end evaluate_expression_l550_550273


namespace glens_speed_is_37_l550_550838

/-!
# Problem Statement
Glen and Hannah drive at constant speeds toward each other on a highway. Glen drives at a certain speed G km/h. At some point, they pass by each other, and keep driving away from each other, maintaining their constant speeds. 
Glen is 130 km away from Hannah at 6 am and again at 11 am. Hannah is driving at 15 kilometers per hour.
Prove that Glen's speed is 37 km/h.
-/

def glens_speed (G : ℝ) : Prop :=
  ∃ G: ℝ, 
    (∃ H_speed : ℝ, H_speed = 15) ∧ -- Hannah's speed
    (∃ distance : ℝ, distance = 130) ∧ -- distance at 6 am and 11 am
    G + 15 = 260 / 5 -- derived equation from conditions

theorem glens_speed_is_37 : glens_speed 37 :=
by {
  sorry -- proof to be filled in
}

end glens_speed_is_37_l550_550838


namespace a_18_is_51_l550_550256

-- Define the sequence a_n
def a : ℕ → ℕ
| 0 := 4
| (n + 1) := if (n + 1) % 2 = 1 then ((n + 3) / 2)^2 else ((n + 3) / 2^2) + 1

theorem a_18_is_51 :
  a 17 = 51 := 
sorry

end a_18_is_51_l550_550256


namespace storm_damage_in_pounds_l550_550001

-- Define the conditions
def euros_damage : ℝ := 45000000
def exchange_rate : ℝ := 1.2

-- Define the function to convert Euros to British pounds
def to_british_pounds (euros : ℝ) (rate : ℝ) : ℝ :=
  euros / rate

-- State the target problem
theorem storm_damage_in_pounds : to_british_pounds euros_damage exchange_rate = 37500000 :=
  by sorry

end storm_damage_in_pounds_l550_550001


namespace number_of_students_l550_550701

theorem number_of_students (n : ℕ) (bow_cost : ℕ) (vinegar_cost : ℕ) (baking_soda_cost : ℕ) (total_cost : ℕ) :
  bow_cost = 5 → vinegar_cost = 2 → baking_soda_cost = 1 → total_cost = 184 → 8 * n = total_cost → n = 23 :=
by
  intros h_bow h_vinegar h_baking_soda h_total_cost h_equation
  sorry

end number_of_students_l550_550701


namespace bobby_initial_candy_count_l550_550717

theorem bobby_initial_candy_count (x : ℕ) (h : x + 17 = 43) : x = 26 :=
by
  sorry

end bobby_initial_candy_count_l550_550717


namespace distance_traveled_ratio_l550_550609

def angle_per_hour (hand: String) : ℕ :=
  if hand = "hour" then 30
  else if hand = "minute" then 360
  else 0

def dist_traveled (hand: String) (hours: ℕ) : ℕ :=
  angle_per_hour hand * hours

def switch_places (p1: String, p2: String) : (String × String) :=
  (p2, p1)

theorem distance_traveled_ratio :
  ∃ (mosquito fly : String), 
    mosquito = "hour" ∧
    fly = "minute" ∧
    (∃ (ratio: ℚ), (dist_traveled mosquito 12) / (dist_traveled fly 12) = ratio ∧ ratio = 83/73) :=
by
  sorry

end distance_traveled_ratio_l550_550609


namespace range_of_m_l550_550335

def G (x y : ℤ) : ℤ :=
  if x ≥ y then x - y
  else y - x

theorem range_of_m (m : ℤ) :
  (∀ x, 0 < x → G x 1 > 4 → G (-1) x ≤ m) ↔ 9 ≤ m ∧ m < 10 :=
sorry

end range_of_m_l550_550335


namespace max_area_triangle_l550_550762

-- Problem Definitions
variables {a b c : ℝ} (h_ab : 0 < b < a)

-- Ellipse equation
def ellipse_eq (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Foci definition
def right_focus : Prop := ∃ x y, ellipse_eq x y ∧ x = c ∧ y = 0

-- Proof problem statement
theorem max_area_triangle (h1 : right_focus) (h2 : ∀ y1 : ℝ, ellipse_eq 0 y1) : 
    ∃ S : ℝ, S = b * c ∧ ∀ S' : ℝ, S' = (|y1| * |c|) → S' ≤ S  :=
sorry

end max_area_triangle_l550_550762


namespace remainder_sum_div_2008_l550_550321

theorem remainder_sum_div_2008 :
  let p := 2008
  let sum_fraction := 2 * p ^ 2 / ((2 * p - 1) * (p - 1)) in
  let m := numerator sum_fraction
  let n := denominator sum_fraction
  m + n % 2008 = 1 :=
by
  sorry

end remainder_sum_div_2008_l550_550321


namespace prove_intersection_points_l550_550104

noncomputable def sqrt5 := Real.sqrt 5

def curve1 (x y : ℝ) : Prop := x^2 + y^2 = 5 / 2
def curve2 (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1
def curve3 (x y : ℝ) : Prop := x^2 + y^2 / 4 = 1
def curve4 (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def line (x y : ℝ) : Prop := x + y = sqrt5

theorem prove_intersection_points :
  (∃! (x y : ℝ), curve1 x y ∧ line x y) ∧
  (∃! (x y : ℝ), curve3 x y ∧ line x y) ∧
  (∃! (x y : ℝ), curve4 x y ∧ line x y) :=
by
  sorry

end prove_intersection_points_l550_550104


namespace prime_square_pairs_l550_550057

theorem prime_square_pairs (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
    ∃ n : Nat, p^2 + 5 * p * q + 4 * q^2 = n^2 ↔ (p = 13 ∧ q = 3) ∨ (p = 7 ∧ q = 5) ∨ (p = 5 ∧ q = 11) ∨ (p = 3 ∧ q = 13) ∨ (p = 5 ∧ q = 7) ∨ (p = 11 ∧ q = 5) :=
by
  sorry

end prime_square_pairs_l550_550057


namespace last_two_nonzero_digits_of_factorial_100_l550_550263

theorem last_two_nonzero_digits_of_factorial_100 : let n := last_two_nonzero_digits (Nat.factorial 100) in n = 76 :=
sorry

end last_two_nonzero_digits_of_factorial_100_l550_550263


namespace valid_pairs_850_l550_550765

def contains_zero_digit (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ List.ofDigits 10 n ∧ d = 0

def valid_pair_count (sum : ℕ) : ℕ :=
  ∑ a in Finset.range (sum - 1), if ¬contains_zero_digit a ∧ ¬contains_zero_digit (sum - a) then 1 else 0

theorem valid_pairs_850 : valid_pair_count 850 = 701 := 
  sorry

end valid_pairs_850_l550_550765


namespace concurrency_of_ASa_BSb_CSc_l550_550195

-- Define the geometric structures and points involved
variable {ABC : Triangle}
variable {G : Point}
variable {Ga Gb Gc Sa Sb Sc : Point}

-- Conditions
axiom centroid_G : centroid ABC G
axiom orthogonal_projection_Ga : orthogonal_projection G Ga (side BC ABC)
axiom orthogonal_projection_Gb : orthogonal_projection G Gb (side CA ABC)
axiom orthogonal_projection_Gc : orthogonal_projection G Gc (side AB ABC)
axiom symmetrical_point_Sa : symmetrical_point G Ga Sa
axiom symmetrical_point_Sb : symmetrical_point G Gb Sb
axiom symmetrical_point_Sc : symmetrical_point G Gc Sc

-- Theorem to prove the concurrency of lines
theorem concurrency_of_ASa_BSb_CSc : concurrent (line_through A Sa) (line_through B Sb) (line_through C Sc) := sorry

end concurrency_of_ASa_BSb_CSc_l550_550195


namespace polyhedron_ratio_l550_550696

theorem polyhedron_ratio (x y : ℝ) :
  (∃ (polyhedron : Type) (faces : polyhedron → set (set ℝ)) (vertices : polyhedron → set ℝ),
    (∀ f ∈ (faces polyhedron), is_isosceles_triangle f) ∧ 
    (∀ e ∈ ⋃ f ∈ (faces polyhedron), f, e = x ∨ e = y) ∧ 
    (∀ v ∈ (vertices polyhedron), degree v = 3 ∨ degree v = 6) ∧
    (∀ (a b c : polyhedron), dihedral_angle a b c = dihedral_angle b c a) ∧
    polyhedron.has_faces 12) →
  x / y = 3 / 5 := 
sorry

end polyhedron_ratio_l550_550696


namespace midpoint_O_IJ_l550_550587

open EuclideanGeometry

/- Definitions according to the problem -/
variables (A B C D O I J : Point)
variables (P Q R S P' Q' R' S' : Point)

-- Conditions
axiom quadrilateral_inscribed_in_circle : Circle A B C D ∧ center Circle A B C D = O
axiom bisectors_internal_cyclic : CyclicQuadrilateral P Q R S ∧ circumcenter P Q R S = I
axiom bisectors_external_cyclic : CyclicQuadrilateral P' Q' R' S' ∧ circumcenter P' Q' R' S' = J

noncomputable def midpoint (P Q : Point) : Point := 
Point.mk ((P.x + Q.x) / 2) ((P.y + Q.y) / 2)

/- The statement -/
theorem midpoint_O_IJ :
  O = midpoint I J := 
sorry

end midpoint_O_IJ_l550_550587


namespace find_a_l550_550778

theorem find_a (a : ℤ) :
  (∀ x : ℤ, 6 * x + 3 > 3 * (x + a) → x > a - 1) ∧
  (∀ x : ℤ, x / 2 - 1 ≤ 7 - 3 * x / 2 → x ≤ 4) →
  (∑ x in finset.Icc (a - 1) 4, x = 9) →
  a = 2 := 
sorry

end find_a_l550_550778


namespace find_k_l550_550915

variables {ℝ : Type*} [add_comm_group ℝ] [module ℝ ℝ]

-- Define non-collinear, non-zero vectors a and b
variables (a b : ℝ) 
variables (k : ℝ)

-- Hypotheses
hypothesis h1 : a ≠ 0
hypothesis h2 : b ≠ 0
hypothesis h3 : ¬∃ λ : ℝ, λ ≠ 0 ∧ a = λ • b

-- Collinearity condition
hypothesis h4 : ∃ λ : ℝ, λ ≠ 0 ∧ (8 • a + k • b) = λ • (k • a + 2 • b)

-- Proof that k = 4
theorem find_k : k = 4 :=
sorry

end find_k_l550_550915


namespace garden_length_l550_550340

noncomputable def length_of_garden : ℝ := 300

theorem garden_length (P : ℝ) (b : ℝ) (A : ℝ) 
  (h₁ : P = 800) (h₂ : b = 100) (h₃ : A = 10000) : length_of_garden = 300 := 
by 
  sorry

end garden_length_l550_550340


namespace reduced_speed_probability_l550_550913

theorem reduced_speed_probability (α : ℝ) (n : ℕ) (hα : 0 < α) (hn : 0 < n) :
  (ε : ℝ) (hε : ε = (2 / (α * real.sqrt n))^3) ->
  (reduced_speed : ℝ) (hreduced_speed : reduced_speed < α) :
  ε < (2 / (α * real.sqrt n))^3 := 
sorry

end reduced_speed_probability_l550_550913


namespace cannot_form_triangle_sets_check_l550_550655

theorem cannot_form_triangle (a b c : ℝ) : ¬(5 + 6 > 11 ∧ 5 + 11 > 6 ∧ 6 + 11 > 5) :=
by
  -- This is where the explicit contradiction can be checked.
  -- However, for the problem statement, we are only interested in setting it up correctly.
  sorry

-- Given sets of line segment lengths
def set_a := (3, 4, 5)
def set_b := (5, 6, 11)
def set_c := (5, 6, 10)
def set_d := (2, 3, 4)

theorem sets_check :
  ∀ (a : ℝ) (b : ℝ) (c : ℝ), (a, b, c) = set_a ∨ (a, b, c) = set_b ∨ (a, b, c) = set_c ∨ (a, b, c) = set_d →
  (¬(5 + 6 > 11 ∧ 5 + 11 > 6 ∧ 6 + 11 > 5)) :=
by
  -- Again, this is only setting up the proof structure
  sorry

end cannot_form_triangle_sets_check_l550_550655


namespace zero_vector_collinear_l550_550656

noncomputable def magnitude (v : Vector ℝ) : ℝ := sorry
noncomputable def collinear (v1 v2 : Vector ℝ) : Prop := sorry

theorem zero_vector_collinear (v : Vector ℝ) :
  magnitude (0 : Vector ℝ) = 0 → collinear 0 v := sorry

end zero_vector_collinear_l550_550656


namespace findMonicQuadraticPolynomial_l550_550066

-- Define the root as a complex number
def root : ℂ := -3 - complex.I * real.sqrt 8

-- Define the conditions
def isMonic (p : polynomial ℝ) : Prop := p.leadingCoeff = 1
def hasRealCoefficients (p : polynomial ℝ) : Prop := ∀ a ∈ p.support, is_real (p.coeff a)

-- Define the polynomial
noncomputable def polynomial : polynomial ℝ :=
  polynomial.C 1 * polynomial.X^2 + polynomial.C 6 * polynomial.X + polynomial.C 17

-- The target statement
theorem findMonicQuadraticPolynomial :
  ∀ (p : polynomial ℝ), 
  isMonic p ∧ hasRealCoefficients p ∧ (root ∈ p.roots) →
  p = polynomial :=
by
  sorry

end findMonicQuadraticPolynomial_l550_550066


namespace domain_of_f_l550_550975

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (x + 1)) + Real.sqrt (4 - x^2)

theorem domain_of_f : 
  {x : ℝ | x > -1 ∧ x ≤ 2 ∧ x ≠ 0 ∧ 4 - x^2 ≥ 0} = {x : ℝ | (-1 < x ∧ x < 0) ∨ (0 < x ∧ x ≤ 2)} :=
by
  sorry

end domain_of_f_l550_550975


namespace probability_true_owner_rattle_l550_550008

theorem probability_true_owner_rattle 
  (weekday_lies : Prop)
  (sunday_truth : Prop)
  (brother_statement : ∀ (day : ℕ), day ∈ {Monday, Tuesday, Wednesday, Thursday, Friday, Saturday} → ¬ weekday_lies
  (day = Sunday) → sunday_truth)
  (expected_probability : ℚ := 13/14) :
  ∃ (prob : ℚ), prob = expected_probability :=
by
  sorry

end probability_true_owner_rattle_l550_550008


namespace geometry_problem_l550_550184

noncomputable def circle (center : ℝ × ℝ) (radius : ℝ) := 
  { p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 }

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := 
  ((p1.1 + p2.1)/2, (p1.2 + p2.2)/2)

def A : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0.5, 0)

def T1 := circle (midpoint A B) (dist A B / 2)
def T2 := circle (midpoint B C) (dist B C / 2)
def T3 := circle (midpoint A C) (dist A C / 2)

def line_through_B θ := 
  { p : ℝ × ℝ | ∃ t : ℝ, p = (B.1 + t * cos θ, B.2 + t * sin θ) }

theorem geometry_problem :
  ∀ (θ : ℝ)
    (P Q R S : ℝ × ℝ)
    (hP : P ∈ T3)
    (hQ : Q ∈ T3)
    (hR : R ∈ T1)
    (hS : S ∈ T2)
    (hPR : R ∈ line_through_B θ)
    (hQS : S ∈ line_through_B θ)
    (hLine : P ∈ line_through_B θ)
    (hLine : Q ∈ line_through_B θ),
  dist P R = dist Q S := 
by
  sorry

end geometry_problem_l550_550184


namespace trip_time_ratio_l550_550084

theorem trip_time_ratio (v : ℝ) (hv : v > 0) : 
  let t1 := 100 / v in
  let t2 := 160 / (1.5 * v) in 
  t2 / t1 = 32 / 3 := 
by 
  sorry

end trip_time_ratio_l550_550084


namespace even_ln_function_l550_550865

theorem even_ln_function {a : ℝ} (h : ∀ x : ℝ, ln (x^2 + a * x + 1) = ln (x^2 - a * x + 1)) : a = 0 :=
by
  sorry

end even_ln_function_l550_550865


namespace gcd_g102_g103_eq_one_l550_550543

def g (x : ℤ) : ℤ := x^2 - 2*x + 2023

theorem gcd_g102_g103_eq_one : Nat.gcd (g 102).natAbs (g 103).natAbs = 1 := by
  sorry

end gcd_g102_g103_eq_one_l550_550543


namespace total_visitors_600_l550_550878

variable (Enjoyed Understood : Set ℕ)
variable (TotalVisitors : ℕ)
variable (E U : ℕ)

axiom no_enjoy_no_understand : ∀ v, v ∉ Enjoyed → v ∉ Understood
axiom equal_enjoy_understand : E = U
axiom enjoy_and_understand_fraction : E = 3 / 4 * TotalVisitors
axiom total_visitors_equation : TotalVisitors = E + 150

theorem total_visitors_600 : TotalVisitors = 600 := by
  sorry

end total_visitors_600_l550_550878


namespace distinct_real_roots_m_range_root_zero_other_root_l550_550468

open Real

-- Definitions of the quadratic equation and the conditions
def quadratic_eq (m x : ℝ) := x^2 + 2 * (m - 1) * x + m^2 - 1

-- Problem (1)
theorem distinct_real_roots_m_range (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0) → m < 1 :=
by
  sorry

-- Problem (2)
theorem root_zero_other_root (m x : ℝ) :
  (quadratic_eq m 0 = 0 ∧ quadratic_eq m x = 0) → (m = 1 ∧ x = 0) ∨ (m = -1 ∧ x = 4) :=
by
  sorry

end distinct_real_roots_m_range_root_zero_other_root_l550_550468


namespace lambda_mu_constant_l550_550106

noncomputable def curve_equation (b : ℝ) : (ℝ × ℝ) → Prop :=
  λ (p : ℝ × ℝ), let (x, y) := p in x^2 + 3*y^2 = 3*b^2

def line (b : ℝ) : (ℝ × ℝ) → Prop :=
  λ (p : ℝ × ℝ), let (x, y) := p in y = x - Real.sqrt(2) * b

theorem lambda_mu_constant (b : ℝ) (λ μ : ℝ) (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : x₁^2 + 3*y₁^2 = 3 * b^2) 
  (h₂ : x₂^2 + 3*y₂^2 = 3 * b^2)
  (h_AB : line b (x₁, y₁) ∧ line b (x₂, y₂))
  (h_OP : λ * x₁ + μ * x₂ ^ 2 + 3 * (λ * y₁ + μ * y₂) ^ 2 = 3 * b ^ 2):
  λ^2 + μ^2 = 1 :=
sorry

end lambda_mu_constant_l550_550106


namespace dumpling_probability_l550_550629

theorem dumpling_probability :
  let total_dumplings := 15
  let choose4 := Nat.choose total_dumplings 4
  let choose1 := Nat.choose 3 1
  let choose5_2 := Nat.choose 5 2
  let choose5_1 := Nat.choose 5 1
  (choose1 * choose5_2 * choose5_1 * choose5_1) / choose4 = 50 / 91 := by
  sorry

end dumpling_probability_l550_550629


namespace evaluate_expression_l550_550753

theorem evaluate_expression:
  (-2)^2002 + (-1)^2003 + 2^2004 + (-1)^2005 = 3 * 2^2002 - 2 :=
by
  sorry

end evaluate_expression_l550_550753


namespace rain_probability_over_weekend_l550_550402

theorem rain_probability_over_weekend :
  let rain_sat := 0.40
  let rain_sun := 0.50
  let prob_no_rain_sat := 1.0 - rain_sat
  let prob_no_rain_sun := 1.0 - rain_sun
  let prob_no_rain_weekend := prob_no_rain_sat * prob_no_rain_sun
  let prob_rain_weekend := 1.0 - prob_no_rain_weekend
  prob_rain_weekend * 100 = 70 :=
by
  let rain_sat := 0.40
  let rain_sun := 0.50
  let prob_no_rain_sat := 1.0 - rain_sat
  let prob_no_rain_sun := 1.0 - rain_sun
  let prob_no_rain_weekend := prob_no_rain_sat * prob_no_rain_sun
  let prob_rain_weekend := 1.0 - prob_no_rain_weekend
  have h : prob_rain_weekend = 0.70 := by sorry
  exact h.symm ▸ rfl

end rain_probability_over_weekend_l550_550402


namespace distance_between_parallel_lines_eq_2_l550_550927

def line1 (x y : ℝ) : Prop := 3 * x - 4 * y + 2 = 0
def line2 (x y : ℝ) : Prop := 3 * x - 4 * y - 8 = 0

theorem distance_between_parallel_lines_eq_2 :
  let A := 3
  let B := -4
  let c1 := 2
  let c2 := -8
  let d := (|c1 - c2| / Real.sqrt (A^2 + B^2))
  d = 2 :=
by
  sorry

end distance_between_parallel_lines_eq_2_l550_550927


namespace can_weigh_1kg_with_300g_and_650g_weights_l550_550840

-- Definitions based on conditions
def balance_scale (a b : ℕ) (w₁ w₂ : ℕ) : Prop :=
  a * w₁ + b * w₂ = 1000

-- Statement to prove based on the problem and solution
theorem can_weigh_1kg_with_300g_and_650g_weights (w₁ : ℕ) (w₂ : ℕ) (a b : ℕ)
  (h_w1 : w₁ = 300) (h_w2 : w₂ = 650) (h_a : a = 1) (h_b : b = 1) :
  balance_scale a b w₁ w₂ :=
by 
  -- We are given:
  -- - w1 = 300 g
  -- - w2 = 650 g
  -- - we want to measure 1000 g using these weights
  -- - a = 1
  -- - b = 1
  -- Prove that:
  --   a * w1 + b * w2 = 1000
  -- Which is:
  --   1 * 300 + 1 * 650 = 1000
  sorry

end can_weigh_1kg_with_300g_and_650g_weights_l550_550840


namespace find_a5_of_geom_seq_l550_550791

theorem find_a5_of_geom_seq 
  (a : ℕ → ℝ) (q : ℝ)
  (hgeom : ∀ n, a (n + 1) = a n * q)
  (S : ℕ → ℝ)
  (hS3 : S 3 = a 0 * (1 - q ^ 3) / (1 - q))
  (hS6 : S 6 = a 0 * (1 - q ^ 6) / (1 - q))
  (hS9 : S 9 = a 0 * (1 - q ^ 9) / (1 - q))
  (harith : S 3 + S 6 = 2 * S 9)
  (a8 : a 8 = 3) :
  a 5 = -6 :=
by
  sorry

end find_a5_of_geom_seq_l550_550791


namespace second_smallest_integer_l550_550079

theorem second_smallest_integer (x y z w v : ℤ) (h_avg : (x + y + z + w + v) / 5 = 69)
  (h_median : z = 83) (h_mode : w = 85 ∧ v = 85) (h_range : 85 - x = 70) :
  y = 77 :=
by
  sorry

end second_smallest_integer_l550_550079


namespace orthocenter_on_circumcircle_iff_right_triangle_l550_550293

def Triangle (A B C : Type) := A ≠ B ∧ B ≠ C ∧ C ≠ A
def is_orthocenter (A B C M : Type) := 
  -- Definitions related to orthocenter here

-- Define what it means for a point to be on the circumcircle
def on_circumcircle (A B C M : Type) := 
  -- Definitions related to circumcircle here

theorem orthocenter_on_circumcircle_iff_right_triangle (A B C M : Type)
  (hTriangle : Triangle A B C)
  (hOrthocenter : is_orthocenter A B C M) :
  on_circumcircle A B C M ↔ ∠ A C B = 90 :=
sorry

end orthocenter_on_circumcircle_iff_right_triangle_l550_550293


namespace largest_value_is_D_l550_550556

def y : ℝ := 0.0002

def A : ℝ := 5 + y
def B : ℝ := 5 - y
def C : ℝ := 5 * y
def D : ℝ := 5 / y
def E : ℝ := y / 5

theorem largest_value_is_D :
  D > A ∧ D > B ∧ D > C ∧ D > E :=
sorry

end largest_value_is_D_l550_550556


namespace value_of_a_is_negative_one_l550_550201

-- Conditions
def I (a : ℤ) : Set ℤ := {2, 4, a^2 - a - 3}
def A (a : ℤ) : Set ℤ := {4, 1 - a}
def complement_I_A (a : ℤ) : Set ℤ := {x ∈ I a | x ∉ A a}

-- Theorem statement
theorem value_of_a_is_negative_one (a : ℤ) (h : complement_I_A a = {-1}) : a = -1 :=
by
  sorry

end value_of_a_is_negative_one_l550_550201


namespace joey_age_sum_of_digits_l550_550520

def sum_of_digits (n : ℕ) : ℕ :=
  nat.digits 10 n |>.sum

theorem joey_age_sum_of_digits
  (C J M : ℕ)
  (hM : M = 1)
  (hJ : J = C + 4)
  (hAgeCond : C = 8) :
  sum_of_digits (J + 1) = 4 :=
by
  sorry

end joey_age_sum_of_digits_l550_550520


namespace exists_nat_sum_of_squares_two_ways_l550_550670

theorem exists_nat_sum_of_squares_two_ways :
  ∃ n : ℕ, n < 100 ∧ ∃ a b c d : ℕ, a ≠ b ∧ c ≠ d ∧ n = a^2 + b^2 ∧ n = c^2 + d^2 :=
by {
  sorry
}

end exists_nat_sum_of_squares_two_ways_l550_550670


namespace johns_remaining_money_l550_550522

def num_days_in_april := 30
def num_sundays_in_april := 4
def num_days_john_walks := num_days_in_april - num_sundays_in_april

def weekdays_in_a_week := 5
def weekends_in_april := 4
def weekday_rate := 10
def weekend_rate := 15

def total_weekday_earnings := weekdays_in_a_week * 4 * weekday_rate
def total_weekend_earnings := weekends_in_april * weekend_rate

def total_earnings := total_weekday_earnings + total_weekend_earnings

def first_expense := 50
def second_expense := 25
def total_expenses := first_expense + second_expense

def remaining_money := total_earnings - total_expenses

def percentage_given_to_kaylee := 0.20
def money_given_to_kaylee := remaining_money * percentage_given_to_kaylee

def money_left_after_all_expenses := remaining_money - money_given_to_kaylee

theorem johns_remaining_money : money_left_after_all_expenses = 148 := 
by
  have num_days : num_days_john_walks = 26 := rfl
  have day_earnings : total_weekday_earnings + total_weekend_earnings = 260 := by
  have week_earnings : weekdays_in_a_week * 4 = 20 := rfl
  have weekend_earnings : weekends_in_april * weekend_rate = 60 := rfl
  have tot_earnings : total_weekday_earnings = 200 := by rfl
  have : 200 + 60 = 260 := by 
  exact rfl
  have tot_expenses :  total_expenses = 75 := by rfl
  have remaining_money := 260 - 75 = 185 := rfl
  have money_to_kaylee := 185 * percentage_given_to_kaylee = 37 := 
  have john_money_left := 185 - 37 = 148 := rfl
  exact rfl

end johns_remaining_money_l550_550522


namespace minimum_value_l550_550397

theorem minimum_value (x : ℝ) (h : x > 0) :
  x^3 + 12*x + 81 / x^4 = 24 := 
sorry

end minimum_value_l550_550397


namespace distance_from_epres_to_barackos_l550_550326

theorem distance_from_epres_to_barackos :
  let d_ab : ℝ := 2000             -- Almás to Barackos in meters
  let d_bc : ℝ := 1650             -- Barackos to Cseresznyés in meters
  let d_cd : ℝ := 8500             -- Cseresznyés to Dinnyés in meters
  let d_de : ℝ := 3750             -- Dinnyés to Epres in meters
  let d_ea : ℝ := 1100             -- Epres to Almás in meters
  d_ea + d_ab = 3100 :=            -- Expected answer for Epres to Barackos
begin
  sorry
end

end distance_from_epres_to_barackos_l550_550326


namespace complex_conjugate_is_correct_l550_550450

open Complex

noncomputable def i_unit : ℂ := Complex.I
noncomputable def z : ℂ := i_unit * (2 - i_unit)
noncomputable def z_conjugate : ℂ := conj z

theorem complex_conjugate_is_correct : z_conjugate = 1 - 2 * Complex.I := by
  sorry

end complex_conjugate_is_correct_l550_550450


namespace solve_for_nabla_l550_550146

theorem solve_for_nabla (nabla mu : ℤ) (h1 : 5 * (-3) = nabla + mu - 3) (h2 : mu = 4) : 
  nabla = -16 := 
by
  sorry

end solve_for_nabla_l550_550146


namespace minimum_value_g_l550_550869

-- Defining the function f
def f (x φ : ℝ) : ℝ := sin (2 * x + φ) + sqrt 3 * cos (2 * x + φ)

-- Defining the function g
def g (x : ℝ) : ℝ := cos (x + π/6)

-- Main statement of the problem
theorem minimum_value_g :
  ∀ (x : ℝ), (-π/2 ≤ x ∧ x ≤ π/6) → g x = 1/2 :=
by
  sorry

end minimum_value_g_l550_550869


namespace angles_form_geometric_sequence_l550_550744

theorem angles_form_geometric_sequence :
  {θ : ℝ // 0 < θ ∧ θ < 2 * π ∧ θ % (π / 2) ≠ 0 ∧
  (∃ (r : ℝ), r < 0 ∧ 
  ((sin θ = r * cos θ ∧ cos θ = r * tan θ) ∨
   (sin θ = r * tan θ ∧ tan θ = r * cos θ) ∨
   (cos θ = r * sin θ ∧ sin θ = r * tan θ)))}.card = 4 :=
by sorry

end angles_form_geometric_sequence_l550_550744


namespace total_meals_l550_550290

def meals := 
  let meats := 3
  let vegetables := Nat.choose 5 3
  let desserts := 4
  let drinks := 3
  meats * vegetables * desserts * drinks

theorem total_meals : meals = 360 := 
  by
    unfold meals
    conv => 
      rhs
      rw [Nat.choose]
    exact Nat.mul_comm _ _
    sorry

end total_meals_l550_550290


namespace mobile_price_two_years_ago_l550_550289

-- Definitions and conditions
def price_now : ℝ := 1000
def decrease_rate : ℝ := 0.2
def years_ago : ℝ := 2

-- Main statement
theorem mobile_price_two_years_ago :
  ∃ (a : ℝ), a * (1 - decrease_rate)^years_ago = price_now :=
sorry

end mobile_price_two_years_ago_l550_550289


namespace space_experiment_sequences_l550_550499

def number_of_sequences (A_first_or_last : Prop) (BC_adjacent : Prop) (six_procedures : Prop) : Nat :=
  if A_first_or_last ∧ BC_adjacent ∧ six_procedures then 96 else 0

theorem space_experiment_sequences (A_condition : ∀ (seq : List String), (seq.head = some "A" ∨ seq.length = 6 ∧ seq.last = some "A")) 
  (BC_condition : ∀ (seq : List String), (∀ i, seq.nth i = some "B" → seq.nth (i+1) = some "C" ∨ seq.nth i = some "C" → seq.nth (i+1) = some "B"))
  (count_condition : List String.length = 6) :
  number_of_sequences (A_first_or_last := A_condition) (BC_adjacent := BC_condition) (six_procedures := count_condition) = 96 := sorry

end space_experiment_sequences_l550_550499


namespace amount_of_food_donated_in_first_week_l550_550928

variable (F : ℝ)
variable (H1 : 0 < F)  -- Assuming F is positive as it's a physical quantity (food amount)

-- Given conditions
axiom food_second_week : (food_second_week = 2 * F)
axiom total_food : (total_food = F + food_second_week)
axiom remaining_food : (0.3 * total_food = 36)

-- Theorem to prove
theorem amount_of_food_donated_in_first_week : F = 40 :=
by 
  have h1 : food_second_week = 2 * F := by assumption
  have h2 : total_food = F + food_second_week := by assumption
  have h3 : remaining_food = 0.3 * total_food := by assumption
  sorry

end amount_of_food_donated_in_first_week_l550_550928


namespace polynomial_with_complex_root_l550_550071

theorem polynomial_with_complex_root :
  ∃ P : Polynomial ℝ, P.Monic ∧ degree P = 2 ∧
  P.coeff 0 = 17 ∧ P.coeff 1 = 6 ∧ P.coeff 2 = 1 ∧
  (P.eval (-3 - Complex.i * Real.sqrt 8) = 0) :=
sorry

end polynomial_with_complex_root_l550_550071


namespace no_solution_exists_l550_550745

theorem no_solution_exists (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : ¬ (x^y + 3 = y^x ∧ 3 * x^y = y^x + 8) :=
by
  intro h
  obtain ⟨eq1, eq2⟩ := h
  sorry

end no_solution_exists_l550_550745


namespace factorial_division_l550_550039

theorem factorial_division (n : ℕ) (hn : n ≥ 1) : (n! / (n-1)!) = n := by
  sorry

-- Special case where n = 10
example : (10! / 9!) = 10 := 
  factorial_division 10 (by decide)

end factorial_division_l550_550039


namespace B_alone_completes_in_11_14_days_Days_for_B_alone_l550_550307

variable (A B : ℝ)

-- Conditions
axiom A_and_B_together_complete_in_6_days : A + B = 1 / 6 
axiom A_alone_completes_in_13_days : A = 1 / 13

-- Statement to prove
theorem B_alone_completes_in_11_14_days :
  B = 1 / (78 / 7) :=
by
  have combined_work_rate := A_and_B_together_complete_in_6_days
  have A_work_rate := A_alone_completes_in_13_days
  have B_work_rate := combined_work_rate - A_work_rate
  sorry

-- Verifying the approximate days
theorem Days_for_B_alone : 
  (78 / 7) ≈ 11.14 :=
by
  sorry

end B_alone_completes_in_11_14_days_Days_for_B_alone_l550_550307


namespace distance_between_trains_l550_550637

-- Conditions definitions
def train1_speed := 10 -- in mph
def train2_speed := 35 -- in mph
def time_elapsed := 10 -- in hours

-- Proof statement
theorem distance_between_trains : 
  let distance1 := train1_speed * time_elapsed in
  let distance2 := train2_speed * time_elapsed in
  distance2 - distance1 = 250 := 
by 
  sorry

end distance_between_trains_l550_550637


namespace class_average_weight_l550_550278

theorem class_average_weight (n_A n_B : ℕ) (w_A w_B : ℝ) (h1 : n_A = 50) (h2 : n_B = 40) (h3 : w_A = 50) (h4 : w_B = 70) :
  (n_A * w_A + n_B * w_B) / (n_A + n_B) = 58.89 :=
by
  sorry

end class_average_weight_l550_550278


namespace price_is_219_l550_550686

noncomputable def discount_coupon1 (price : ℝ) : ℝ :=
  if price > 50 then 0.1 * price else 0

noncomputable def discount_coupon2 (price : ℝ) : ℝ :=
  if price > 100 then 20 else 0

noncomputable def discount_coupon3 (price : ℝ) : ℝ :=
  if price > 100 then 0.18 * (price - 100) else 0

noncomputable def more_savings_coupon1 (price : ℝ) : Prop :=
  discount_coupon1 price > discount_coupon2 price ∧ discount_coupon1 price > discount_coupon3 price

theorem price_is_219 (price : ℝ) :
  more_savings_coupon1 price → price = 219 :=
by
  sorry

end price_is_219_l550_550686


namespace chromosomal_variations_l550_550300

-- Define the conditions
def condition1 := "Plants grown from anther culture in vitro."
def condition2 := "Addition or deletion of DNA base pairs on chromosomes."
def condition3 := "Free combination of non-homologous chromosomes."
def condition4 := "Crossing over between non-sister chromatids in a tetrad."
def condition5 := "Cells of a patient with Down syndrome have three copies of chromosome 21."

-- Define a concept of belonging to chromosomal variations
def belongs_to_chromosomal_variations (condition: String) : Prop :=
  condition = condition1 ∨ condition = condition5

-- State the theorem
theorem chromosomal_variations :
  belongs_to_chromosomal_variations condition1 ∧ 
  belongs_to_chromosomal_variations condition5 ∧ 
  ¬ (belongs_to_chromosomal_variations condition2 ∨ 
     belongs_to_chromosomal_variations condition3 ∨ 
     belongs_to_chromosomal_variations condition4) :=
by
  sorry

end chromosomal_variations_l550_550300


namespace smallest_n_for_8820_factorial_l550_550296

theorem smallest_n_for_8820_factorial :
  ∃ (n : ℕ), 0 < n ∧ (8820 ∣ nat.factorial n) ∧ ∀ (m : ℕ), m < n → ¬ (8820 ∣ nat.factorial m) :=
sorry

end smallest_n_for_8820_factorial_l550_550296


namespace spots_combined_l550_550142

def Rover : ℕ := 46
def Cisco : ℕ := Rover / 2 - 5
def Granger : ℕ := 5 * Cisco

theorem spots_combined : Granger + Cisco = 108 := by
  sorry

end spots_combined_l550_550142


namespace marie_stamps_l550_550932

variable (n_notebooks : ℕ) (stamps_per_notebook : ℕ) (n_binders : ℕ) (stamps_per_binder : ℕ) (fraction_keep : ℚ)

theorem marie_stamps :
  n_notebooks = 4 →
  stamps_per_notebook = 20 →
  n_binders = 2 →
  stamps_per_binder = 50 →
  fraction_keep = 1/4 →
  let total_stamps := n_notebooks * stamps_per_notebook + n_binders * stamps_per_binder in
  let stamps_keep := total_stamps * fraction_keep in
  let stamps_give_away := total_stamps - stamps_keep in
  stamps_give_away = 135 :=
by
  intros h1 h2 h3 h4 h5
  let total_stamps := n_notebooks * stamps_per_notebook + n_binders * stamps_per_binder
  let stamps_keep := total_stamps * fraction_keep
  let stamps_give_away := total_stamps - stamps_keep
  have h_total_stamps : total_stamps = 180 := by simp [h1, h2, h3, h4, total_stamps]
  have h_stamps_keep : stamps_keep = 45 := by simp [h_total_stamps, h5, stamps_keep]
  have h_stamps_give_away : stamps_give_away = 135 := by simp [h_total_stamps, h_stamps_keep, stamps_give_away]
  exact h_stamps_give_away

end marie_stamps_l550_550932


namespace min_tiles_to_ensure_overlap_tiling_possible_on_removed_square_grid_l550_550315

-- Problem 1
theorem min_tiles_to_ensure_overlap : ∀ (n : ℕ), n = 8 → ∃ (k : ℕ), k = 22 ∧
  ∀ (pieces : list (finset (fin (8 * 8))), ∀ P, P ∈ pieces → (∃ (t : finset (fin (8 * 8))), t.card = 3 ∧ t ⊆ univ) → 
    (pieces.card = k → ∃ P_1, P_1 ∈ pieces → ∀ (P_2 : finset (fin (8 * 8))), P_2 = (range (8 * 8)).erase P_1 → 
      (∃ (q : finset (fin (8 * 8))), q.card = 3 ∧ q ∩ P_2 ≠ ∅))) :=
begin
  intros,
  existsi 22,
  split,
  -- k = 22
  reflexivity,
  -- ∀ (pieces : list (finset (fin (8 * 8))), ∀ P, P ∈ pieces ...
  sorry
end

-- Problem 2
theorem tiling_possible_on_removed_square_grid : ∀ (n : ℕ), n = 1987 → 
  ∀ (i j : ℕ), i < n → j < n → ∃ (t : finset (fin (n * n))), t.card = (n * n) - 1 ∧ 
  ∀ P, P ∈ t → (∃ (l : finset (fin (n * n))), l.card = 3 ∧ l ⊆ t) :=
begin
  intros,
  existsi (range (n * n)).erase (i * n + j),
  split,
  -- t.card = (n * n) - 1
  sorry,
  -- ∀ P, P ∈ t → (∃ (l : finset (fin (n * n))), l.card = 3 ∧ l ⊆ t)
  sorry
end

end min_tiles_to_ensure_overlap_tiling_possible_on_removed_square_grid_l550_550315


namespace children_in_circle_l550_550718

theorem children_in_circle (n m : ℕ) (k : ℕ) 
  (h1 : n = m) 
  (h2 : n + m = 2 * k) :
  ∃ k', n + m = 4 * k' :=
by
  sorry

end children_in_circle_l550_550718


namespace triangle_OBC_area_l550_550111

noncomputable def OBC_area (O A B C : ℝ × ℝ) : ℝ :=
  sorry  -- Placeholder for the actual computation

theorem triangle_OBC_area (O A B C : ℝ × ℝ)
  (h1: (O.1 + O.2, O.1 + O.2, O.1 + O.2) = (0,0,0))
  (h2: (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 2)
  (h3: angle A B C = 60) :
  OBC_area O B C = sqrt 3 / 3 :=
begin
  sorry
end

end triangle_OBC_area_l550_550111


namespace length_of_train_l550_550350

noncomputable def speed_kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

noncomputable def distance_covered (speed_ms : ℝ) (time_s : ℝ) : ℝ :=
  speed_ms * time_s

theorem length_of_train
  (bridge_length : ℝ)
  (time_s : ℝ)
  (speed_kmh : ℝ)
  (train_length_approx : ℝ) :
  let speed_ms := speed_kmh_to_ms speed_kmh in
  let total_distance := distance_covered speed_ms time_s in
  total_distance = train_length_approx + bridge_length :=
by
  sorry

#eval length_of_train 300 30 48 99.9 -- This should be true if the proved statement is correct

end length_of_train_l550_550350


namespace students_without_scholarship_l550_550496

theorem students_without_scholarship (x : ℝ) :
  let boys := 5 * x,
      girls := 6 * x,
      boys_with_scholarship := 0.25 * boys,
      girls_with_scholarship := 0.20 * girls,
      total_students := boys + girls,
      students_with_scholarship := boys_with_scholarship + girls_with_scholarship,
      students_without_scholarship := total_students - students_with_scholarship,
      percentage_without_scholarship := (students_without_scholarship / total_students) * 100 in
  percentage_without_scholarship ≈ 77.73 :=
by
  sorry

end students_without_scholarship_l550_550496


namespace tangent_length_integer_values_zero_l550_550782

theorem tangent_length_integer_values_zero (P : Point) (m n t : ℝ) 
  (h1 : from point P outside a circle, a tangent is drawn)
  (h2 : secant is drawn from P dividing the circle into arcs of lengths m and n)
  (h3 : m = 2 * n)
  (h4 : m + n = 18)
  (h5 : t = real.sqrt (m * n)) :
  ∃ (t : ℝ), t = 6 * real.sqrt 2 ∧ ¬ (t ∈ ℤ) → 0 :=
begin
  -- Proof goes here
  sorry
end

end tangent_length_integer_values_zero_l550_550782


namespace dinner_guest_arrangement_l550_550518

noncomputable def number_of_ways (n k : ℕ) : ℕ :=
  if n < k then 0 else Nat.factorial n / Nat.factorial (n - k)

theorem dinner_guest_arrangement :
  let total_arrangements := number_of_ways 8 5
  let unwanted_arrangements := 7 * number_of_ways 6 3 * 2
  let valid_arrangements := total_arrangements - unwanted_arrangements
  valid_arrangements = 5040 :=
by
  -- Definitions and preliminary calculations
  let total_arrangements := number_of_ways 8 5
  let unwanted_arrangements := 7 * number_of_ways 6 3 * 2
  let valid_arrangements := total_arrangements - unwanted_arrangements

  -- This is where the proof would go, but we insert sorry to skip it for now
  sorry

end dinner_guest_arrangement_l550_550518


namespace unique_solution_l550_550056

noncomputable def f : ℝ → ℝ :=
sorry

theorem unique_solution (x : ℝ) (hx : 0 ≤ x) : 
  (f : ℝ → ℝ) (2 * x + 1) = 3 * (f x) + 5 ↔ f x = -5 / 2 :=
by 
  sorry

end unique_solution_l550_550056


namespace money_given_by_school_correct_l550_550955

-- Definitions from the problem conditions
def cost_per_book : ℕ := 12
def number_of_students : ℕ := 30
def out_of_pocket : ℕ := 40

-- Derived definition from these conditions
def total_cost : ℕ := cost_per_book * number_of_students
def money_given_by_school : ℕ := total_cost - out_of_pocket

-- The theorem stating that the amount given by the school is $320
theorem money_given_by_school_correct : money_given_by_school = 320 :=
by
  sorry -- Proof placeholder

end money_given_by_school_correct_l550_550955


namespace correct_placement_l550_550182

-- Define square locations A, B, C, D, E, F, G
inductive Square
| A | B | C | D | E | F | G

open Square

def arrow_direction : Square → Square
  | Square.A => Square.G  -- Arrows in A point to G
  | Square.B => Square.E  -- Arrows in B point to E
  | Square.C => Square.D  -- Arrows in C point to D
  | Square.D => Square.A  -- Arrows in D point to A
  | Square.E => Square.C  -- Arrows in E point to C
  | Square.F => Square.B  -- Presumed direction
  | Square.G => Square.F  -- Arrows in G point to F

-- Function that gives the fixed positions based on the conditions in the problem
def position_of : ℕ → Square
  | 2 => Square.B
  | 3 => Square.E
  | 4 => Square.C
  | 5 => Square.D
  | 6 => Square.A
  | 7 => Square.G
  | 8 => Square.F

-- Prove that the placement follows the arrows condition.
theorem correct_placement :
  (position_of 2 = Square.B) ∧
  (position_of 3 = arrow_direction (position_of 2)) ∧
  (position_of 4 = arrow_direction (position_of 3)) ∧
  (position_of 5 = arrow_direction (position_of 4)) ∧
  (position_of 6 = arrow_direction (position_of 5)) ∧
  (position_of 7 = arrow_direction (position_of 6)) ∧
  (position_of 8 = arrow_direction (position_of 7)) ∧
  (arrow_direction (position_of 8) = Square.F) :=
begin
  -- Here we layout the expected positions as per the solution and a proof that each step is correct implicitly.
  split; try { refl },
  split; try { refl },
  split; try { refl },
  split; try { refl },
  split; try { refl },
  split; try { refl },
  split; try { refl },
  refl,
end

end correct_placement_l550_550182


namespace min_value_expr_l550_550419

theorem min_value_expr (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 / b = 1) : 
  ∃ c, (c = 18) ∧ (∀ a b, a > 0 ∧ b > 0 ∧ a + 4 / b = 1 → 2 / a + 2 * b ≥ c) :=
by {
  let expr := 2 / a + 2 * b,
  use 18,
  split,
  {
    refl,
  },
  {
    intros a b ha hb hab,
    -- Proof that 2 / a + 2 * b ≥ 18 goes here
    sorry,
  }
}

end min_value_expr_l550_550419


namespace chord_length_l550_550430

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem chord_length {A B : ℝ × ℝ} :
  let slope := 2
  let right_focus := (1 : ℝ, 0 : ℝ)
  let ellipse : ℝ × ℝ → Prop := λ (x, y), x^2 / 5 + y^2 / 4 = 1
  let line : ℝ × ℝ → Prop := λ (x, y), y = 2 * (x - 1)
  A = (0, -2) ∧ B = (5/3, 4/3) ∧ ellipse A ∧ ellipse B ∧ line A ∧ line B → 
    distance A B = 5 * real.sqrt 5 / 3 := 
by
  intros
  sorry

end chord_length_l550_550430


namespace sum_of_bn_lt_half_l550_550099

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + (n - 1) * d

theorem sum_of_bn_lt_half (a S b T : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a2 : a 2 = 3) (h_a3 : a 3 = 5)
  (h_S : ∀ n, S n = n * (a 1 + a n) / 2)
  (h_b : ∀ n, b n = 1 / (4 * S n - 1))
  (h_T : ∀ n, T n = ∑ i in range n, b i) :
  ∀ n, T n < 1 / 2 := 
sorry

end sum_of_bn_lt_half_l550_550099


namespace geometric_series_sum_l550_550036

theorem geometric_series_sum :
  let a := 3
  let r := -2
  let n := 10
  let S := a * ((r^n - 1) / (r - 1))
  S = -1023 :=
by 
  -- Sorry allows us to omit the proof details
  sorry

end geometric_series_sum_l550_550036


namespace sol_earnings_in_a_week_l550_550965

-- Define the number of candy bars sold each day using recurrence relation
def candies_sold (n : ℕ) : ℕ :=
  match n with
  | 0     => 10  -- Day 1
  | (n+1) => candies_sold n + 4  -- Each subsequent day

-- Define the total candies sold in a week and total earnings in dollars
def total_candies_sold_in_a_week : ℕ :=
  List.sum (List.map candies_sold [0, 1, 2, 3, 4, 5])

def total_earnings_in_dollars : ℕ :=
  (total_candies_sold_in_a_week * 10) / 100

-- Proving that Sol will earn 12 dollars in a week
theorem sol_earnings_in_a_week : total_earnings_in_dollars = 12 := by
  sorry

end sol_earnings_in_a_week_l550_550965


namespace lara_bought_52_stems_l550_550903

-- Define the conditions given in the problem:
def flowers_given_to_mom : ℕ := 15
def flowers_given_to_grandma : ℕ := flowers_given_to_mom + 6
def flowers_in_vase : ℕ := 16

-- The total number of stems of flowers Lara bought should be:
def total_flowers_bought : ℕ := flowers_given_to_mom + flowers_given_to_grandma + flowers_in_vase

-- The main theorem to prove the total number of flowers Lara bought is 52:
theorem lara_bought_52_stems : total_flowers_bought = 52 := by
  sorry

end lara_bought_52_stems_l550_550903


namespace gambler_final_winning_percentage_l550_550688

theorem gambler_final_winning_percentage : 
  (number_of_games : ℕ) (initial_win_percentage : ℝ) (initial_games_played : ℕ)
  (additional_winning_percentage : ℝ) (additional_games_played : ℕ)
  (initial_win_count : ℕ := nat.floor (initial_win_percentage * initial_games_played))
  (additional_win_count : ℕ := nat.floor (additional_winning_percentage * additional_games_played))
  (total_wins : ℕ := initial_win_count + additional_win_count)
  (total_games : ℕ := initial_games_played + additional_games_played)
  (final_winning_percentage : ℝ := total_wins / total_games * 100)
  (h1 : initial_win_percentage = 0.40)
  (h2 : initial_games_played = 30)
  (h3 : additional_winning_percentage = 0.80)
  (h4 : additional_games_played = 30) 
  : final_winning_percentage = 60 := by
    sorry

end gambler_final_winning_percentage_l550_550688


namespace area_of_triangle_bounded_by_lines_l550_550639

theorem area_of_triangle_bounded_by_lines :
  let y1 := λ x : ℝ, 2 * x
  let y2 := λ x : ℝ, -2 * x
  let y3 := λ x : ℝ, 8
  let A := (4, 8)
  let B := (-4, 8)
  let C := (0, 0)
  (y1 4 = 8) ∧
  (y2 (-4) = 8) ∧
  (y1 0 = 0) ∧ (y2 0 = 0) ∧
  (y1 = y3) → 
  (y2 = y3) →

  let base := 8
  let height := 8
  let area := (1 / 2) * base * height
  area = 32 := sorry

end area_of_triangle_bounded_by_lines_l550_550639


namespace range_of_x_l550_550801

variables {x : Real}

def P (x : Real) : Prop := (x + 1) / (x - 3) ≥ 0
def Q (x : Real) : Prop := abs (1 - x/2) < 1

theorem range_of_x (hP : P x) (hQ : ¬ Q x) : x ≤ -1 ∨ x ≥ 4 :=
  sorry

end range_of_x_l550_550801


namespace cos_sq_plus_two_sin_double_l550_550478

theorem cos_sq_plus_two_sin_double (α : ℝ) (h : Real.tan α = 3 / 4) : Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 :=
by
  sorry

end cos_sq_plus_two_sin_double_l550_550478


namespace median_of_sample_l550_550432

theorem median_of_sample (x : ℝ) (h_avg : (0 + -1 + x + 1 + 3) / 5 = 2) : median [-1, 0, 1, 3, x] = 1 :=
by
  sorry

end median_of_sample_l550_550432


namespace number_of_tables_l550_550167

noncomputable def stools_per_table : ℕ := 7
noncomputable def legs_per_stool : ℕ := 4
noncomputable def legs_per_table : ℕ := 5
noncomputable def total_legs : ℕ := 658

theorem number_of_tables : 
  ∃ t : ℕ, 
  (∃ s : ℕ, s = stools_per_table * t ∧ legs_per_stool * s + legs_per_table * t = total_legs) ∧ t = 20 :=
by {
  sorry
}

end number_of_tables_l550_550167


namespace sum_largest_and_smallest_angle_l550_550122

theorem sum_largest_and_smallest_angle (a b c : ℝ) (ha : a = 5) (hb : b = 7) (hc : c = 8) :
  ∃ A B C : ℝ, 
    (A + B + C = π ∧
     (cos B = (a^2 + c^2 - b^2) / (2 * a * c)) ∧
     ∃ largest smallest, (A + largest = π) ∧ (smallest = π - B) ∧ (largest + smallest = 2 * (π / 3))) := 
sorry

end sum_largest_and_smallest_angle_l550_550122


namespace exists_bi_iff_n_composite_l550_550408

open Nat

theorem exists_bi_iff_n_composite (n : ℕ) (h_n : n > 1) :
  (∃ (b : ℕ → ℕ), (∀ i, i < n → b i ≠ b 0) ∧ (∀ k, (∏ i in Finset.range n, (b i + k)) = m ^ e k ∧ e k > 1)) ↔ ¬Prime n :=
by
  sorry

end exists_bi_iff_n_composite_l550_550408


namespace product_of_slopes_l550_550101

noncomputable def ellipse {x y : ℝ} := (x^2 / 3) + (y^2 / 2) = 1

noncomputable def points_symmetric_about_origin (B D : ℝ × ℝ) : Prop :=
  ∃ (x₁ y₁ : ℝ), B = (x₁, y₁) ∧ D = (-x₁, -y₁)

noncomputable def parallelogram_diagonal {A B C D : ℝ × ℝ} : Prop :=
  ∃ (x₂ y₂ x₁ y₁ : ℝ), A = (x₂, y₂) ∧ C = (-x₂, -y₂) ∧ B = (x₁, y₁) ∧ D = (-x₁, -y₁) ∧
  (∃ (k₁ k₂ : ℝ), k₁ = (y₁ - y₂) / (x₁ - x₂) ∧ k₂ = (y₁ + y₂) / (x₁ + x₂))

theorem product_of_slopes 
  (A B C D : ℝ × ℝ)
  (h1 : ellipse B.1 B.2)
  (h2 : ellipse D.1 D.2)
  (h3 : points_symmetric_about_origin B D)
  (h4 : parallelogram_diagonal A B C D) :
  let k₁ : ℝ := (B.2 - A.2) / (B.1 - A.1)
      k₂ : ℝ := (B.2 + A.2) / (B.1 + A.1)
  in k₁ * k₂ = -2 / 3 := by
  sorry

end product_of_slopes_l550_550101


namespace sum_of_coefficients_l550_550727

def polynomial := 3 * (x: ℤ)^8 - 2 * x^7 + 4 * x^6 - x^4 + 6 * x^2 - 7
                  - 5 * (x^5 - 2 * x^3 + 2 * x - 8)
                  + 6 * (x^6 + x^4 - 3)

theorem sum_of_coefficients : 
  (polynomial.eval 1) = 32 := by 
  sorry

end sum_of_coefficients_l550_550727


namespace investment_percentage_change_l550_550873

/-- 
Isabel's investment problem statement:
Given an initial investment, and percentage changes over three years,
prove that the overall percentage change in Isabel's investment is 1.2% gain.
-/
theorem investment_percentage_change (initial_investment : ℝ) (gain1 : ℝ) (loss2 : ℝ) (gain3 : ℝ) 
    (final_investment : ℝ) :
    initial_investment = 500 →
    gain1 = 0.10 →
    loss2 = 0.20 →
    gain3 = 0.15 →
    final_investment = initial_investment * (1 + gain1) * (1 - loss2) * (1 + gain3) →
    ((final_investment - initial_investment) / initial_investment) * 100 = 1.2 :=
by
  intros h_init h_gain1 h_loss2 h_gain3 h_final
  sorry

end investment_percentage_change_l550_550873


namespace find_k_range_l550_550866

-- Define the function and necessary conditions 
def f (x : ℝ) : ℝ := 2 * x^2 - log x

theorem find_k_range :
  (∃ k : ℝ, ¬ monotonic_on f [k-1, k+1] ∧ (0 < k-1 ∧ k-1 < k+1)) →
  (1 < k ∧ k < 3/2) :=
by
  sorry

end find_k_range_l550_550866


namespace sphere_center_x_axis_eq_l550_550657

theorem sphere_center_x_axis_eq (a : ℝ) (R : ℝ) (x y z : ℝ) :
  (x - a) ^ 2 + y ^ 2 + z ^ 2 = R ^ 2 → (0 - a) ^ 2 + (0 - 0) ^ 2 + (0 - 0) ^ 2 = R ^ 2 →
  a = R →
  (x ^ 2 - 2 * a * x + y ^ 2 + z ^ 2 = 0) :=
by
  sorry

end sphere_center_x_axis_eq_l550_550657


namespace ratio_of_angles_range_of_cos_values_l550_550186

variable {α : Type}
variables (a b c A B C : ℝ)

-- Question 1: Prove A / B = 2 given a^2 - b^2 = bc and triangle ABC is acute
theorem ratio_of_angles (h₀ : a^2 - b^2 = b * c) 
                        (h₁ : 0 < A ∧ A < π / 2) 
                        (h₂ : 0 < B ∧ B < π / 2) 
                        (h₃ : 0 < C ∧ C < π / 2) 
                        (h₄ : A + B + C = π) :
                        A / B = 2 :=
sorry

-- Question 2: Prove the range of values for cos(C - B) + cos A is (1, 9/8]
theorem range_of_cos_values (h₀ : a^2 - b^2 = b * c) 
                            (h₁ : 0 < A ∧ A < π / 2) 
                            (h₂ : 0 < B ∧ B < π / 2) 
                            (h₃ : 0 < C ∧ C < π / 2) 
                            (h₄ : A + B + C = π) :
                            ∃ (x : ℝ), 1 < x ∧ x ≤ 9/8 ∧ x = cos(C - B) + cos A :=
sorry

end ratio_of_angles_range_of_cos_values_l550_550186


namespace marked_points_distance_l550_550659

theorem marked_points_distance (stick_length : ℝ) (red_mark_pos : ℝ) (blue_mark_pos : ℝ) :
  stick_length = 12 → 
  red_mark_pos = stick_length / 2 →
  blue_mark_pos = red_mark_pos / 2 →
  (red_mark_pos - blue_mark_pos) = 3 :=
by
  intros h_length h_red h_blue
  rw [h_length, h_red, h_blue]
  sorry

end marked_points_distance_l550_550659


namespace MBNK_parallelogram_l550_550905

-- Define the given conditions
variable (A B C K M N : Type) [Point A B C K M N]
variable [Triangle A B C]
variable [InsideTriangle K A B C]
variable [OppositeSides M K A B]
variable [OppositeSides N K B C]
variable (angle : Angle)
variable [EqAngle (Angle M A B) angle]
variable [EqAngle (Angle M B A) angle]
variable [EqAngle (Angle N B C) angle]
variable [EqAngle (Angle N C B) angle]
variable [EqAngle (Angle K A C) angle]
variable [EqAngle (Angle K C A) angle]

-- The theorem to be proved
theorem MBNK_parallelogram : Parallelogram M B N K := by
  sorry

end MBNK_parallelogram_l550_550905


namespace john_min_correct_problems_l550_550605

/-- The condition of scoring points based on correct, incorrect, and unanswered questions. -/
def score (correct_attempts : ℕ) (incorrect_attempts : ℕ) (unanswered : ℕ) : ℕ :=
  7 * correct_attempts + 0 * incorrect_attempts + 1 * unanswered

/-- Total number of problems attempted and unanswered. -/
def total_problems_attempted : ℕ := 26
def total_problems_unanswered : ℕ := 4

/-- Minimum score John needs. -/
def minimum_score : ℕ := 150

/-- The total score John needs from the attempted problems (26 problems). -/
def required_attempted_score : ℕ := minimum_score - (1 * total_problems_unanswered)

theorem john_min_correct_problems :
  ∃ (correct_attempts : ℕ), 
  correct_attempts ≤ total_problems_attempted ∧ 
  score correct_attempts (total_problems_attempted - correct_attempts) total_problems_unanswered ≥ minimum_score :=
by
  use 21
  simp [total_problems_attempted, total_problems_unanswered, minimum_score, required_attempted_score, score]
  norm_num
  linarith
  sorry

end john_min_correct_problems_l550_550605


namespace original_function_eq_l550_550413

noncomputable def OA : ℝ × ℝ := (4, 3)

def is_tangent (f : ℝ → ℝ) (l : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, f x = l x ↔ x = p.1

def original_function_satisfies_conditions (f : ℝ → ℝ) (b c : ℝ) : Prop := 
  let T' := (-3, 1) in
  let T := (1, 4) in
  let translated_f := λ x, f (x - 4) + 3 in
  is_tangent translated_f (λ x, 8 - 4 * x) T ∧ f T'.1 = T'.2 ∧ f' T'.1 = -4

theorem original_function_eq :
  ∃ (b c : ℝ), original_function_satisfies_conditions (λ x, x^2 + b * x + c) b c ∧ b = 2 ∧ c = -2 :=
by
  sorry

end original_function_eq_l550_550413


namespace b_cannot_be_2_l550_550811

variables (b : ℝ)

def condition (x : ℝ) : Prop := x ∈ set.Icc 0 1 → abs (x + b) ≤ 2

theorem b_cannot_be_2 (h : ∀ x, condition b x) : b ≠ 2 :=
by
  sorry

end b_cannot_be_2_l550_550811


namespace intersection_sets_l550_550471

variable {α : Type*} [LinearOrder α] [OrderBot α] [OrderTop α]

def setA : set ℝ := {x | x^2 - 9 * x < 0}
def setB : set ℝ := {x | 1 < 2^x ∧ 2^x < 8}

theorem intersection_sets : setA ∩ setB = {x | 0 < x ∧ x < 3} := sorry

end intersection_sets_l550_550471


namespace remainder_relation_l550_550299

theorem remainder_relation (P P' D R R' : ℕ) (hP : P > P') (h1 : P % D = R) (h2 : P' % D = R') :
  ∃ C : ℕ, ((P + C) * P') % D ≠ (P * P') % D ∧ ∃ C : ℕ, ((P + C) * P') % D = (P * P') % D :=
by sorry

end remainder_relation_l550_550299


namespace casey_decorating_time_l550_550374

theorem casey_decorating_time :
  ∀ (n : ℕ) (t_apply t_dry : ℕ), n = 3 → t_apply = 20 → t_dry = 20 → n * (t_apply + t_dry) = 120 :=
by
  intros n t_apply t_dry hn ht_apply ht_dry
  rw [hn, ht_apply, ht_dry]
  sorry

end casey_decorating_time_l550_550374


namespace calculate_n_l550_550920

theorem calculate_n (t n : ℕ) (h : (∑ i in finset.range (13 - t), (1 : ℝ) / (2^(12 - i))) = (n : ℝ) / 2^12) : n = 2^(13 - t) - 1 := 
sorry

end calculate_n_l550_550920


namespace same_function_pair_B_l550_550653

noncomputable def f1 (x : ℝ) : ℝ := x
noncomputable def g1 (x : ℝ) : ℝ := (Real.sqrt x) ^ 2

noncomputable def f2 (x : ℝ) : ℝ := x^2 + 1
noncomputable def g2 (t : ℝ) : ℝ := t^2 + 1

noncomputable def f3 (x : ℝ) : ℝ := 1
noncomputable def g3 (x : ℝ) : ℝ := if x = 0 then 0 else x / x

noncomputable def f4 (x : ℝ) : ℝ := x
noncomputable def g4 (x : ℝ) : ℝ := |x|

theorem same_function_pair_B : 
  ∀ (x : ℝ), f2 x = g2 x :=
by
  intro x
  simp only [f2, g2]
  sorry

end same_function_pair_B_l550_550653


namespace clock_angle_at_3_30_l550_550366

-- Definitions equivalent to conditions
def h : ℕ := 3
def m : ℕ := 30

-- Angle formula definition
def angle (h m : ℕ) : ℚ := abs (60 * h - 11 * m) / 2

-- Lean 4 statement
theorem clock_angle_at_3_30 : angle h m = 75 := by
  sorry

end clock_angle_at_3_30_l550_550366


namespace part1a_part1b_part2_part3a_part3b_l550_550784

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)
noncomputable def g (x : ℝ) : ℝ := x^2 + 2

-- Prove f(2) = 1/3
theorem part1a : f 2 = 1 / 3 := 
by sorry

-- Prove g(2) = 6
theorem part1b : g 2 = 6 :=
by sorry

-- Prove f[g(2)] = 1/7 
theorem part2 : f (g 2) = 1 / 7 :=
by sorry

-- Prove f[g(x)] = 1/(x^2 + 3) 
theorem part3a : ∀ x : ℝ, f (g x) = 1 / (x^2 + 3) :=
by sorry

-- Prove g[f(x)] = 1/((1 + x)^2) + 2 
theorem part3b : ∀ x : ℝ, g (f x) = 1 / (1 + x)^2 + 2 :=
by sorry

end part1a_part1b_part2_part3a_part3b_l550_550784


namespace same_grades_percentage_l550_550171

theorem same_grades_percentage (total_students same_grades_A same_grades_B same_grades_C same_grades_D : ℕ) 
  (total_eq : total_students = 50) 
  (same_A : same_grades_A = 3) 
  (same_B : same_grades_B = 6) 
  (same_C : same_grades_C = 7) 
  (same_D : same_grades_D = 2) : 
  (same_grades_A + same_grades_B + same_grades_C + same_grades_D) * 100 / total_students = 36 := 
by
  sorry

end same_grades_percentage_l550_550171


namespace average_income_increase_l550_550580

theorem average_income_increase (initial_increase : ℝ) (annual_growth_rate : ℝ) (target_increase : ℝ) (start_year : ℕ)
  (ln_3 : ℝ) (ln_10 : ℝ) (ln_11 : ℝ) :
  initial_increase = 4000 →
  annual_growth_rate = 1.1 →
  target_increase = 12000 →
  start_year = 2023 →
  ln 3 = ln_3 → ln 10 = ln_10 → ln 11 = ln_11 →
  ln_3 ≈ 1.10 → ln_10 ≈ 2.30 → ln_11 ≈ 2.40 →
  ∃ n : ℕ, start_year + n = 2035 ∧ initial_increase * annual_growth_rate ^ n ≥ target_increase := 
begin
  sorry
end

end average_income_increase_l550_550580


namespace evaluate_expression_l550_550752

theorem evaluate_expression (c : ℕ) (h : c = 4) : (c^c - c * (c - 1)^(c - 1))^c = 148^4 := 
by 
  sorry

end evaluate_expression_l550_550752


namespace increasing_power_function_only_at_m_neg1_l550_550257

-- Define the function
def power_function (m : ℝ) : ℝ → ℝ :=
  λ x, (m^2 - m - 1) * x^(m^2 - 3*m - 3)

-- State the problem as a theorem
theorem increasing_power_function_only_at_m_neg1 :
  ∀ m : ℝ, (∀ x : ℝ, 0 < x → deriv (power_function m) x > 0) ↔ m = -1 :=
sorry

end increasing_power_function_only_at_m_neg1_l550_550257


namespace find_d1_l550_550547

noncomputable def E (n : ℕ) : ℕ := sorry

theorem find_d1 :
  ∃ (d4 d3 d2 d0 : ℤ), 
  (∀ (n : ℕ), n ≥ 4 ∧ n % 2 = 0 → 
     E n = d4 * n^4 + d3 * n^3 + d2 * n^2 + (12 : ℤ) * n + d0) :=
sorry

end find_d1_l550_550547


namespace degree_of_k_is_five_l550_550482

-- Define h(x) as given
def h (x : ℝ) : ℝ := -9 * x^5 + 2 * x^4 - x^2 + 7

-- Define the condition for h(x) + k(x) having a degree of 2
def condition_on_degree (k : ℝ → ℝ) : Prop :=
  ∃ (m : ℝ → ℝ), m = h + k ∧ polynomial.degree (m : polynomial ℝ) = 2

-- Claim: the degree of k(x) is 5
theorem degree_of_k_is_five (k : ℝ → ℝ) (h_condition : condition_on_degree k) :
  polynomial.degree (k : polynomial ℝ) = 5 :=
sorry

end degree_of_k_is_five_l550_550482


namespace age_difference_zero_l550_550364

theorem age_difference_zero (A B n : ℕ) (h1 : A = B^3) (h2 : A - 1 = 6 * (B - 1)) : n = 0 :=
by
  have h3 : A = B := by sorry
  have h4 : n = A - B := by sorry
  rw [h3] at h4
  exact h4

end age_difference_zero_l550_550364


namespace log_function_extrema_l550_550160

theorem log_function_extrema (a : ℝ) (ha : 0 < a) :
  (∀ x ∈ set.Icc 2 4, log a x ≤ log a 4 ∧ log a 2 ≤ log a x) ∧
  (log a 4 = log a 2 + 1 ∨ log a 2 = log a 4 + 1) ->
  a = 2 ∨ a = 1 / 2 :=
by
  sorry

end log_function_extrema_l550_550160


namespace sequence_expression_l550_550792

def S_n (a : ℕ → ℚ) (n : ℕ) : ℚ := (finset.range n).sum a

def a_n (n : ℕ) : ℚ := (2^n - 1) / 2^(n-1)

theorem sequence_expression (a : ℕ → ℚ)
  (H : ∀ n : ℕ, n > 0 → S_n a n + a n = 2 * n) :
  ∀ n : ℕ, n > 0 → a n = (2^n - 1) / 2^(n-1) :=
by
  intro n hn
  sorry

end sequence_expression_l550_550792


namespace geom_series_mult_result_l550_550728

theorem geom_series_mult_result :
  let a := (1 / 4 : ℚ)
  let r := (1 / 4 : ℚ)
  let n := 5
  let S_n := a * (1 - r^n) / (1 - r)
  let result := S_n * (4 / 5 : ℚ)
in result = 21 / 80 := 
by {
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  let S_n := a * (1 - r^n) / (1 - r)
  let result := S_n * (4 / 5)
  have h : result = 21 / 80 := sorry
  exact h
}

end geom_series_mult_result_l550_550728


namespace exchange_1_dollar_to_lire_l550_550360

noncomputable def dollars_to_lire : ℝ := 3000 / 1.60

theorem exchange_1_dollar_to_lire :
  dollars_to_lire = 1875 :=
by
  unfold dollars_to_lire
  norm_num
  exact_eq_of_std_eq

#align dollars_to_lire exchange_1_dollar_to_lire

end exchange_1_dollar_to_lire_l550_550360


namespace distinct_solutions_square_difference_l550_550112

theorem distinct_solutions_square_difference 
  (Φ φ : ℝ) (h1 : Φ^2 = Φ + 2) (h2 : φ^2 = φ + 2) (h_distinct : Φ ≠ φ) :
  (Φ - φ)^2 = 9 :=
  sorry

end distinct_solutions_square_difference_l550_550112


namespace num_isosceles_triangles_l550_550842

theorem num_isosceles_triangles (a b : ℕ) (h1 : 2 * a + b = 27) (h2 : a > b / 2) : 
  ∃! (n : ℕ), n = 13 :=
by 
  sorry

end num_isosceles_triangles_l550_550842


namespace proof_problem_l550_550994

-- Define the conditions as propositions in Lean
def p : Prop := true
def q : Prop := true

def condition1 := p ∧ q
def condition2 := ∀ x : ℝ, x^2 - x - 1 ≥ 0
def condition3 (φ : ℝ) : Prop := φ = (Real.pi / 2) ∧ (∀ x : ℝ, y = sin (2 * x + φ) → Even Function)
def condition4 (a : ℝ) : Prop := a < 0 → StrictMonoDec (λ x : ℝ, x^a) (Set.Ioi 0)

-- Count the number of correct conclusions
def countCorrectConclusions : Nat :=
if b1 : (condition1 → ¬ p) = true then 1 else 0 +
if b2 : condition2 = true then 1 else 0 +
if b3 : condition3 (Real.pi / 2) = true then 1 else 0 +
if b4 : condition4 (-1) = true then 1 else 0

-- The theorem to prove the number of correct conclusions is 2
theorem proof_problem : countCorrectConclusions = 2 := sorry

end proof_problem_l550_550994


namespace number_of_ordered_pairs_l550_550083

noncomputable def count_valid_ordered_pairs (a b: ℝ) : Prop :=
  ∃ (x y : ℤ), a * (x : ℝ) + b * (y : ℝ) = 2 ∧ x^2 + y^2 = 65

theorem number_of_ordered_pairs : ∃ s : Finset (ℝ × ℝ), s.card = 128 ∧ ∀ (p : ℝ × ℝ), p ∈ s ↔ count_valid_ordered_pairs p.1 p.2 :=
by
  sorry

end number_of_ordered_pairs_l550_550083


namespace area_of_hexagon_l550_550531

theorem area_of_hexagon (ABCDEF : Hexagon) (G H I : Point) 
  (h1 : RegularHexagon ABCDEF)
  (h2 : Midpoint G (side AB ABCDEF))
  (h3 : Midpoint H (side CD ABCDEF))
  (h4 : Midpoint I (side EF ABCDEF))
  (h5 : Area (triangle G H I) = 400)
  : Area ABCDEF = 9600 / 9 := 
sorry

end area_of_hexagon_l550_550531


namespace count_not_containing_multiple_of_5_l550_550377

-- Defining the condition for the sequence not containing multiples of 5
def sequence_contains_multiple_of_5 (n : ℕ) : Prop :=
  ∃ p : ℕ, p < (n : nat).bitLength ∧ 5 ∣ (⌈ (n : ℝ) / (2^p : ℝ) ⌉ : ℕ)

-- Main theorem to prove the number of integers n in the range [1, 1024] satisfying the condition is 351.
theorem count_not_containing_multiple_of_5 : 
  (Finset.range 1025).filter (λ n, ¬sequence_contains_multiple_of_5 n).card = 351 :=
by
  -- This is a stub of the proof, actual proof goes here
  sorry

end count_not_containing_multiple_of_5_l550_550377


namespace inequality_solution_l550_550602

theorem inequality_solution (x : ℝ) :
  (x < -2 ∨ (-1 < x ∧ x < 1) ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 6) ∨ 7 < x) →
  (1 / (x - 1)) - (4 / (x - 2)) + (4 / (x - 3)) - (1 / (x - 4)) < 1 / 30 :=
by
  sorry

end inequality_solution_l550_550602


namespace line_eq_proof_l550_550816

-- Define the slope and x-intercept conditions
def slope : ℝ := 4
def x_intercept_point : ℝ × ℝ := (2, 0)

-- Define the equation of the line in simplified form
def line_equation (x y : ℝ) : Prop := 4 * x - y - 8 = 0

-- Statement to prove
theorem line_eq_proof : 
  ∃ (f : ℝ → ℝ), (∀ x, f x = 4 * (x - 2)) ∧ 
  (∀ x y, (x, y) = x_intercept_point → y = f x) → 
  (∀ x y, line_equation x y) :=
by
  sorry

end line_eq_proof_l550_550816


namespace equilateral_triangle_AN_relation_l550_550530

theorem equilateral_triangle_AN_relation (A B C N O : Point) (ABC_circle : Circle O)
  (h_equilateral : equilateral_triangle A B C)
  (h_inscribed : inscribed_triangle A B C O ABC_circle)
  (h_N_on_arc : N ∈ arc AC ABC_circle ∧ N ≠ C) :
  that can be equal to, less than, or greater than (dist A N) (dist B N + dist C N) :=
sorry

end equilateral_triangle_AN_relation_l550_550530


namespace gcd_g_values_l550_550545

def g (x : ℤ) : ℤ := x^2 - 2 * x + 2023

theorem gcd_g_values : gcd (g 102) (g 103) = 1 := by
  sorry

end gcd_g_values_l550_550545


namespace correct_calculation_l550_550651

/-- Conditions for the given calculations -/
def cond_a : Prop := (-2) ^ 3 = 8
def cond_b : Prop := (-3) ^ 2 = -9
def cond_c : Prop := -(3 ^ 2) = -9
def cond_d : Prop := (-2) ^ 2 = 4

/-- Prove that the correct calculation among the given is -3^2 = -9 -/
theorem correct_calculation : cond_c :=
by sorry

end correct_calculation_l550_550651


namespace triangle_area_l550_550394

variable (x y : ℝ)

def line1 : Prop := y - 4 * x = -3
def line2 : Prop := 4 * y + x = 16
def y_axis : Prop := x = 0

theorem triangle_area (h1 : line1) (h2 : line2) (hy : y_axis) : 
  (1 / 2) * (4 - (-3)) * ((8 * 4 - 4) / 255) = 98 / 17 :=
by
  sorry

end triangle_area_l550_550394


namespace exist_non_special_symmetric_concat_l550_550425

-- Define the notion of a binary series being symmetric
def is_symmetric (xs : List Bool) : Prop :=
  ∀ i, i < xs.length → xs.get? i = xs.get? (xs.length - 1 - i)

-- Define the notion of a binary series being special
def is_special (xs : List Bool) : Prop :=
  (∀ x ∈ xs, x) ∨ (∀ x ∈ xs, ¬x)

-- The main theorem statement
theorem exist_non_special_symmetric_concat (m n : ℕ) (hm : m % 2 = 1) (hn : n % 2 = 1) :
  ∃ (A B : List Bool), A.length = m ∧ B.length = n ∧ ¬is_special A ∧ ¬is_special B ∧ is_symmetric (A ++ B) :=
sorry

end exist_non_special_symmetric_concat_l550_550425


namespace elder_person_age_l550_550247

-- Definitions based on conditions
variables (y e : ℕ) 

-- Given conditions
def condition1 : Prop := e = y + 20
def condition2 : Prop := e - 5 = 5 * (y - 5)

-- Theorem stating the required proof problem
theorem elder_person_age (h1 : condition1 y e) (h2 : condition2 y e) : e = 30 :=
by
  sorry

end elder_person_age_l550_550247


namespace unique_ordered_pair_l550_550809

theorem unique_ordered_pair (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : cos (π * x) ^ 2 + 2 * sin (π * y) = 1) 
  (h2 : sin (π * x) + sin (π * y) = 0) 
  (h3 : x ^ 2 - y ^ 2 = 12) : 
  (x, y) = (4, 2) := 
by 
  sorry

end unique_ordered_pair_l550_550809


namespace smaller_solution_of_quadratic_eq_l550_550645

noncomputable def P (x : ℝ) : ℝ := x^2 + 17 * x - 60

theorem smaller_solution_of_quadratic_eq : ∃ (a b : ℝ), P a = 0 ∧ P b = 0 ∧ a ≠ b ∧ min a b = -20 :=
by
  have h1 : P (-20) = 0 := by norm_num,
  have h2 : P 3 = 0 := by norm_num,
  existsi [-20, 3],
  exact ⟨h1, h2, by norm_num, by norm_num⟩

end smaller_solution_of_quadratic_eq_l550_550645


namespace moles_HCl_formed_l550_550764

-- Define the initial moles of CH4 and Cl2
def CH4_initial : ℕ := 2
def Cl2_initial : ℕ := 4

-- Define the balanced chemical equation in terms of the number of moles
def balanced_equation (CH4 : ℕ) (Cl2 : ℕ) : Prop :=
  CH4 + 4 * Cl2 = 1 * CH4 + 4 * Cl2

-- Theorem statement: Given the conditions, prove the number of moles of HCl formed is 4
theorem moles_HCl_formed (CH4_initial Cl2_initial : ℕ) (h_CH4 : CH4_initial = 2) (h_Cl2 : Cl2_initial = 4) :
  ∃ (HCl : ℕ), HCl = 4 :=
  sorry

end moles_HCl_formed_l550_550764


namespace car_speed_is_120_l550_550704

theorem car_speed_is_120 (v t : ℝ) (h1 : v > 0) (h2 : t > 0) (h3 : v * t = 75)
  (h4 : 1.5 * v * (t - (12.5 / 60)) = 75) : v = 120 := by
  sorry

end car_speed_is_120_l550_550704


namespace planned_pigs_correct_l550_550939

-- Define initial number of animals
def initial_cows : ℕ := 2
def initial_pigs : ℕ := 3
def initial_goats : ℕ := 6

-- Define planned addition of animals
def added_cows : ℕ := 3
def added_goats : ℕ := 2
def total_animals : ℕ := 21

-- Define the total planned number of pigs to verify:
def planned_pigs := 8

-- State the final number of pigs to be proven
theorem planned_pigs_correct : 
  initial_cows + initial_pigs + initial_goats + added_cows + planned_pigs + added_goats = total_animals :=
by
  sorry

end planned_pigs_correct_l550_550939


namespace three_wheels_possible_two_wheels_not_possible_l550_550236

/-!
# Problem Statement

Given that the top view of wheels with spokes is provided in Figure 1, and
after rotation, the new top view is provided in Figure 2.

We are to prove:

1. There could have been three wheels that form the pattern in Figure 2.
2. There could not have been two wheels that form the pattern in Figure 2.
-/

-- Definitions representing the conditions in the problem
def top_view_before_rotation : Type := sorry -- Abstract representation of Figure 1
def top_view_after_rotation : Type := sorry  -- Abstract representation of Figure 2

def possible_pattern_three_wheels (view_before : top_view_before_rotation) (view_after : top_view_after_rotation) : Prop :=
  sorry -- Relation encoding that three wheels can match the pattern from Figure 1 to Figure 2

def possible_pattern_two_wheels (view_before : top_view_before_rotation) (view_after : top_view_after_rotation) : Prop :=
  sorry -- Relation encoding that two wheels can match the pattern from Figure 1 to Figure 2

-- Formalizing the proof problem
theorem three_wheels_possible (view_before : top_view_before_rotation) (view_after : top_view_after_rotation) :
  possible_pattern_three_wheels(view_before, view_after) :=
sorry

theorem two_wheels_not_possible (view_before : top_view_before_rotation) (view_after : top_view_after_rotation) :
  ¬possible_pattern_two_wheels(view_before, view_after) :=
sorry

end three_wheels_possible_two_wheels_not_possible_l550_550236


namespace ellipse_equation_exists_line_l0_l550_550439

-- Definitions for conditions as per the problem statement
def is_ellipse (a b x y : ℝ) : Prop := (a > b ∧ b > 0 ∧ (x^2 / a^2) + (y^2 / b^2) = 1)

def is_right_focus_on_line (f_x f_y : ℝ) : Prop := (f_y = 0 ∧ (sqrt 3 * f_x - f_y - 3 = 0) ∧ (f_x = sqrt 3))

def product_of_slopes (a b : ℝ) (slope_product : ℝ) : Prop := ((-b^2 / a^2) = slope_product)

-- Statement for Proof Problem 1
theorem ellipse_equation (a b f_x f_y : ℝ) (slope_product : ℝ) :
  is_ellipse a b f_x f_y →
  is_right_focus_on_line f_x f_y →
  product_of_slopes a b (-1/4) →
  (a^2 - b^2 = 3) →
  (a = 2 ∧ b = 1) :=
by
  -- Mathematically prove that a = 2 and b = 1
  sorry

-- Definitions for second proof problem
def intersection_line_ellipse (P_x P_y : ℝ) : Prop := (P_x = 1 ∧ P_y = 0)

def line_through_P_intersects_ellipse (a b k : ℝ) : Prop :=
  ∃ l l_x l_y : ℝ, 
    (l_y = k * (l_x - 1)) ∧ 
    (l_x^2 / a^2 + l_y^2 = 1)

-- Statement for Proof Problem 2
theorem exists_line_l0 (a b x0 k : ℝ) :
  is_ellipse a b 0 0 →
  (a = 2) →
  (b = 1) →
  intersection_line_ellipse 1 0 →
  line_through_P_intersects_ellipse 2 1 k →
  (x0 = 4 ∧ x0 > 2) :=
by
  -- Mathematically prove that x0 = 4 and x0 > 2
  sorry

end ellipse_equation_exists_line_l0_l550_550439


namespace original_rectangle_area_at_least_90_l550_550339

variable (a b c x y z : ℝ)
variable (hx1 : a * x = 1)
variable (hx2 : c * x = 3)
variable (hy : b * y = 10)
variable (hz : a * z = 9)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (hx : 0 < x) (hy' : 0 < y) (hz' : 0 < z)

theorem original_rectangle_area_at_least_90 : ∀ {a b c x y z : ℝ},
  (a * x = 1) →
  (c * x = 3) →
  (b * y = 10) →
  (a * z = 9) →
  (0 < a) →
  (0 < b) →
  (0 < c) →
  (0 < x) →
  (0 < y) →
  (0 < z) →
  (a + b + c) * (x + y + z) ≥ 90 :=
sorry

end original_rectangle_area_at_least_90_l550_550339


namespace jill_earnings_l550_550665

theorem jill_earnings :
  ∀ (hourly_wage : ℝ) (tip_rate : ℝ) (num_shifts : ℕ) (hours_per_shift : ℕ) (avg_orders_per_hour : ℝ),
  hourly_wage = 4.00 →
  tip_rate = 0.15 →
  num_shifts = 3 →
  hours_per_shift = 8 →
  avg_orders_per_hour = 40 →
  (num_shifts * hours_per_shift * hourly_wage + num_shifts * hours_per_shift * avg_orders_per_hour * tip_rate = 240) :=
by
  intros hourly_wage tip_rate num_shifts hours_per_shift avg_orders_per_hour
  intros hwage_eq trip_rate_eq nshifts_eq hshift_eq avgorder_eq
  sorry

end jill_earnings_l550_550665


namespace sufficient_not_necessary_l550_550090

-- Given conditions
variables {a b : ℝ} (h1 : a > 0) (h2 : b > 0)

-- Condition we want to show is sufficient
def condition1 := a^2 + b^2 < 1

-- Condition we want to deduce
def condition2 := ab + 1 > a + b

theorem sufficient_not_necessary (h1 : a > 0) (h2 : b > 0) : condition1 a b → condition2 a b :=
by
  intros h
  sorry -- Proof goes here

end sufficient_not_necessary_l550_550090


namespace find_x_in_sequence_l550_550508

theorem find_x_in_sequence :
  ∃ x y z : ℤ, 
    (z - 1 = 0) ∧ (y - z = -1) ∧ (x - y = 1) ∧ x = 1 :=
by
  sorry

end find_x_in_sequence_l550_550508


namespace Anhui_mountains_arrangement_l550_550272

/-!
The terrain in Anhui Province includes plains, plateaus (hills), hills, mountains, and other types. 
The hilly areas account for a large proportion, so there are many mountains. Some famous mountains include 
Huangshan, Jiuhuashan, and Tianzhushan. A school has organized a study tour course and plans to send 
5 outstanding students to these three places for study tours. Each mountain must have at least 
one student participating. Prove the number of different arrangement options is 150.
-/

open Nat Function

def countArrangements (n k : Nat) : Nat :=
  factorial n / factorial (n - k)

theorem Anhui_mountains_arrangement :
  let students := 5
  let mountains := 3
  ∃ arrangements options, 
  arrangements = countArrangements 5 3 * 6 ∧
  options = 150 ∧
  arrangements = options
:=
  sorry

end Anhui_mountains_arrangement_l550_550272


namespace radius_of_scrap_cookie_l550_550050

theorem radius_of_scrap_cookie :
  ∀ (r : ℝ),
    (∃ (r_dough r_cookie : ℝ),
      r_dough = 6 ∧  -- Radius of the large dough
      r_cookie = 2 ∧  -- Radius of each cookie
      8 * (π * r_cookie^2) ≤ π * r_dough^2 ∧  -- Total area of cookies is less than or equal to area of large dough
      (π * r_dough^2) - (8 * (π * r_cookie^2)) = π * r^2  -- Area of scrap dough forms a circle of radius r
    ) → r = 2 := by
  sorry

end radius_of_scrap_cookie_l550_550050


namespace sum_of_three_numbers_l550_550650

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a + b = 35) (h2 : b + c = 54) (h3 : c + a = 58) : 
  a + b + c = 73.5 :=
by
  sorry -- Proof is omitted

end sum_of_three_numbers_l550_550650


namespace zoo_total_animals_l550_550355

theorem zoo_total_animals (penguins polar_bears : ℕ)
  (h1 : penguins = 21)
  (h2 : polar_bears = 2 * penguins) :
  penguins + polar_bears = 63 := by
   sorry

end zoo_total_animals_l550_550355


namespace boxes_sold_l550_550516

theorem boxes_sold (cases boxes_per_case : ℕ) (h_cases : cases = 3) (h_boxes_per_case : boxes_per_case = 8) :
  cases * boxes_per_case = 24 :=
by
  rw [h_cases, h_boxes_per_case]
  norm_num

end boxes_sold_l550_550516


namespace range_of_a_l550_550208

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then -x^2 else x^2

theorem range_of_a :
  (∀ x : ℝ, f (x) = if x < 0 then -x^2 else x^2) ∧
  (∀ x ≥ 0, f x = x^2) ∧
  (∀ x ∈ set.Icc a (a + 2), f (x + a) ≥ 2 * f x) → a ≥ real.sqrt 2 :=
begin
  sorry
end

end range_of_a_l550_550208


namespace u_2023_is_4_l550_550126

def f (x : ℕ) : ℕ :=
  match x with
  | 1 => 5
  | 2 => 3
  | 3 => 2
  | 4 => 1
  | 5 => 4
  | _ => 0  -- f is only defined for x in {1, 2, 3, 4, 5}

def u : ℕ → ℕ
| 0 => 5
| (n + 1) => f (u n)

theorem u_2023_is_4 : u 2023 = 4 := by
  sorry

end u_2023_is_4_l550_550126


namespace permutation_count_of_word_l550_550476

-- Define the setup with the conditions
def letters : Finset (Char) := {'B', 'B', 'A', 'A', 'A', 'N', 'N'}

def count_B : ℕ := 2
def count_A : ℕ := 3
def count_N : ℕ := 2

theorem permutation_count_of_word :
  (∏ i in letters, nat.factorial (letters.count i)) = 2 * 2 * 2 → 
  finset.card (letters.permute (list.perm)) = 2520 := 
by
  intros h
  sorry

end permutation_count_of_word_l550_550476


namespace magnitude_of_complex_l550_550156

variable (z : ℂ)
variable (h : Complex.I * z = 3 - 4 * Complex.I)

theorem magnitude_of_complex :
  Complex.abs z = 5 :=
by
  sorry

end magnitude_of_complex_l550_550156


namespace minimum_xy_minimum_x_plus_y_l550_550451

variable {x y : ℝ}

-- We are going to define the conditions
def condition1 : Prop := x > 0
def condition2 : Prop := y > 0
def condition3 : Prop := 2 * x + 8 * y - x * y = 0

-- Lean statement for the first minimum value problem
theorem minimum_xy :
  condition1 →
  condition2 →
  condition3 →
  xy ≥ 64 :=
sorry

-- Lean statement for the second minimum value problem
theorem minimum_x_plus_y :
  condition1 →
  condition2 →
  condition3 →
  (x + y) ≥ 18 :=
sorry

end minimum_xy_minimum_x_plus_y_l550_550451


namespace definite_integral_value_l550_550389

def integrand (x : ℝ) : ℝ := x^2 + sin x

theorem definite_integral_value :
  ∫ x in (-1 : ℝ)..(1 : ℝ), integrand x = (2 / 3 : ℝ) :=
by
  sorry

end definite_integral_value_l550_550389


namespace polar_to_cartesian_center_l550_550612

theorem polar_to_cartesian_center :
  ∃ θ ρ, (ρ = 1 ∧ θ = π / 4) ∧
  (∃ x y, (x = sqrt 2 / 2 ∧ y = sqrt 2 / 2) ∧ 
          (x * x + y * y = sqrt 2 * (x + y)) ∧ 
          θ = arctan(y / x) ∧ ρ = sqrt(x * x + y * y)) :=
by
  sorry

end polar_to_cartesian_center_l550_550612


namespace entire_function_constant_l550_550194

open Complex

theorem entire_function_constant
  (f : ℂ → ℂ) (h_entire : ∀ z, differentiable_at ℂ f z)
  (ω1 ω2 : ℂ) (h_irrational : ω1 / ω2 ∉ ℚ)
  (h_periodic : ∀ z, f z = f (z + ω1) ∧ f z = f (z + ω2)) :
  ∃ c : ℂ, ∀ z, f z = c :=
by
  sorry

end entire_function_constant_l550_550194


namespace chord_length_problem_l550_550168

theorem chord_length_problem (d : ℝ) (m n : ℤ) (r x : ℝ) (h_deg : d < 120)
  (h_chord_d : 2 * r^2 * (1 - cos (d * (real.pi / 180))) = 484)
  (h_chord_2d : 2 * r^2 * (1 - cos (2 * d * (real.pi / 180))) = (x + 20)^2)
  (h_chord_3d : 2 * r^2 * (1 - cos (3 * d * (real.pi / 180))) = x^2)
  (h_x : x = -m + real.sqrt n)
  (hm_pos : m > 0) (hn_pos : 0 < n) :
  m + n = 178 := 
sorry

end chord_length_problem_l550_550168


namespace alex_twelfth_finger_l550_550461

def g (n : ℕ) : ℕ :=
  match n with
  | 1 => 8
  | 2 => 7
  | 3 => 6
  | 4 => 5
  | 5 => 4
  | 6 => 3
  | 7 => 2
  | 8 => 1
  | 9 => 0
  | _ => 0  -- This case should not occur for the given problem

theorem alex_twelfth_finger :
  let f : ℕ → ℕ := λ n, if n % 2 = 0 then g 2 else g 7 in
  f 12 = 7 :=
by
  sorry

end alex_twelfth_finger_l550_550461


namespace circumscribed_circle_trilinear_inscribed_circle_trilinear_excircle_trilinear_l550_550396

variables {α β γ a b c x y z : ℝ}

-- Part (a): Equation of the Circumscribed Circle in Trilinear Coordinates
theorem circumscribed_circle_trilinear (a b c x y z : ℝ) : a / x + b / y + c / z = 0 := 
sorry

-- Part (b): Equation of the Inscribed Circle in Trilinear Coordinates
theorem inscribed_circle_trilinear (α β γ x y z : ℝ) :
  cos (α / 2) * sqrt x + cos (β / 2) * sqrt y + cos (γ / 2) * sqrt z = 0 :=
sorry

-- Part (c): Equation of the Excircle Tangent to the Side BC in Trilinear Coordinates
theorem excircle_trilinear (α β γ x y z : ℝ) :
  cos (α / 2) * sqrt (-x) + cos (β / 2) * sqrt y + cos (γ / 2) * sqrt z = 0 :=
sorry

end circumscribed_circle_trilinear_inscribed_circle_trilinear_excircle_trilinear_l550_550396


namespace simplify_and_evaluate_expression_l550_550961

theorem simplify_and_evaluate_expression (a : ℚ) (h : a = -3/2) :
  (a + 2 - 5/(a - 2)) / ((2 * a^2 - 6 * a) / (a - 2)) = -1/2 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_expression_l550_550961


namespace find_sequences_and_sum_l550_550114

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, b (n + 1) = b n * q

def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  S 0 = 0 ∧ ∀ n : ℕ, S (n + 1) = S n + a (n + 1)

theorem find_sequences_and_sum :
  (∃ a b S : ℕ → ℤ, arithmetic_sequence a ∧ geometric_sequence b ∧
    sum_first_n_terms a S ∧
    a 0 = 1 ∧ b 0 = 1 ∧
    b 2 * S 2 = 36 ∧
    b 1 * S 1 = 8 ∧ 
    ∀ n, a n < a (n + 1) → 
      ∑ i in finset.range n, 1 / (a i * a (i + 1)) = n / (2 * n + 1))
  :=
begin
  sorry
end

end find_sequences_and_sum_l550_550114


namespace color_column_l550_550313

theorem color_column (n : ℕ) (color : ℕ) (board : ℕ → ℕ → ℕ) 
  (h_colors : ∀ i j, 1 ≤ board i j ∧ board i j ≤ n^2)
  (h_block : ∀ i j, (∀ k l : ℕ, k < n → l < n → ∃ c, ∀ a b : ℕ, k + a * n < n → l + b * n < n → board (i + k + a * n) (j + l + b * n) = c))
  (h_row : ∃ r, ∀ k, k < n → ∃ c, 1 ≤ c ∧ c ≤ n ∧ board r k = c) :
  ∃ c, (∀ j, 1 ≤ board c j ∧ board c j ≤ n) :=
sorry

end color_column_l550_550313


namespace min_value_a2_b2_range_of_x_l550_550115

-- Problem (I)
theorem min_value_a2_b2 (a b : ℝ) (h : |3 * a + 4 * b| = 10) : a^2 + b^2 ≥ 4 :=
sorry

-- Problem (II)
theorem range_of_x (x : ℝ) (a b : ℝ) (h : |3 * a + 4 * b| = 10) : |x + 3| - |x - 2| ≤ 4 → x ∈ Iic (3/2) ∨ x < -3 :=
sorry

end min_value_a2_b2_range_of_x_l550_550115


namespace triangle_conditions_implications_l550_550187

theorem triangle_conditions_implications 
  (a b c : ℝ) 
  (α β γ : ℝ) -- angles opposite sides a, b, c respectively
  (A : Prop := α = π/3)
  (is_equilateral : Prop := a = b ∧ b = c ∧ c = a)
  (AD_proj_AC : Prop := ∀ (D : ℝ), ∃ (k : ℝ), k = 2/3 ∧ D = k)
  (E : ℝ)
  (h1 : a * cos γ + sqrt 3 * a * sin γ - b - c = 0)
  (h2 : b^2 + c^2 = 2 * a^2)
  (h3 : 2 * ∥BD∥ = ∥DC∥):

  A ∧ is_equilateral ∧ AD_proj_AC :=
by
  sorry

end triangle_conditions_implications_l550_550187


namespace sum_of_triangle_areas_l550_550465

theorem sum_of_triangle_areas (k : ℕ) (hk_pos : 0 < k) :
    ∑ i in Finset.range k, (1 / (2 * i * (i + 1) : ℝ)) = k / (2 * (k + 1) : ℝ) := 
sorry

end sum_of_triangle_areas_l550_550465


namespace fourth_term_expansion_l550_550180

theorem fourth_term_expansion :
  (6*x + 1/(3*sqrt(x)))^9 = ∑ k in finset.range 10, (nat.choose 9 k) * (6*x)^(9 - k) * (1/(3*sqrt(x)))^k → 
  (∀ x, fourth_term (6*x + 1/(3*sqrt(x)))^9 = 224 / 9) :=
sorry

end fourth_term_expansion_l550_550180


namespace jenny_sold_boxes_l550_550513

-- Given conditions as definitions
def cases : ℕ := 3
def boxes_per_case : ℕ := 8

-- Mathematically equivalent proof problem
theorem jenny_sold_boxes : cases * boxes_per_case = 24 := by
  sorry

end jenny_sold_boxes_l550_550513


namespace cricket_team_members_l550_550971

theorem cricket_team_members (n : ℕ) 
  (avg_age_team : ℕ) 
  (age_captain : ℕ) 
  (age_wkeeper : ℕ) 
  (avg_age_remaining : ℕ) 
  (total_age_team : ℕ) 
  (total_age_excl_cw : ℕ) 
  (total_age_remaining : ℕ) :
  avg_age_team = 23 →
  age_captain = 26 →
  age_wkeeper = 29 →
  avg_age_remaining = 22 →
  total_age_team = avg_age_team * n →
  total_age_excl_cw = total_age_team - (age_captain + age_wkeeper) →
  total_age_remaining = avg_age_remaining * (n - 2) →
  total_age_excl_cw = total_age_remaining →
  n = 11 :=
by
  sorry

end cricket_team_members_l550_550971


namespace max_elements_of_valid_subset_l550_550097

def is_valid_subset (T I : set ℕ) : Prop :=
  ∀ x ∈ T, 7 * x ∉ T

theorem max_elements_of_valid_subset : 
  ∃ T ⊆ (finset.range 239).to_set, is_valid_subset T (finset.range 239).to_set ∧ T.card = 208 :=
sorry

end max_elements_of_valid_subset_l550_550097


namespace find_a_range_l550_550837

variable (a : ℝ) (A : Set ℝ)  (B : Set ℝ)

def problem_conditions : A = {x | x^2 + (a + 2) * x + 1 = 0} ∧ B = {x | x > 0} ∧ (A ∩ B = ∅) :=
  sorry

theorem find_a_range (h : problem_conditions a A B) : a > -4 :=
  sorry

end find_a_range_l550_550837


namespace findMonicQuadraticPolynomial_l550_550065

-- Define the root as a complex number
def root : ℂ := -3 - complex.I * real.sqrt 8

-- Define the conditions
def isMonic (p : polynomial ℝ) : Prop := p.leadingCoeff = 1
def hasRealCoefficients (p : polynomial ℝ) : Prop := ∀ a ∈ p.support, is_real (p.coeff a)

-- Define the polynomial
noncomputable def polynomial : polynomial ℝ :=
  polynomial.C 1 * polynomial.X^2 + polynomial.C 6 * polynomial.X + polynomial.C 17

-- The target statement
theorem findMonicQuadraticPolynomial :
  ∀ (p : polynomial ℝ), 
  isMonic p ∧ hasRealCoefficients p ∧ (root ∈ p.roots) →
  p = polynomial :=
by
  sorry

end findMonicQuadraticPolynomial_l550_550065


namespace evaluate_expression_l550_550646

theorem evaluate_expression : 3 - (3:ℝ) ^ (- 3) = (80 : ℝ) / 27 := 
by 
  sorry

end evaluate_expression_l550_550646


namespace ratio_t_T_min_max_l550_550540

def hyperbola (x : ℝ) : ℝ := 1 / x

def tangent_at_point (a : ℝ) (x : ℝ) : ℝ := -1 / (a * a) * x + 2 / a

def t (a : ℝ) : ℝ := (2 * a * 2 / a) / 2

def T (a : ℝ) : ℝ := (2 * a * sqrt (a * a + 1 / (a * a))) / 2

def t_over_T (a : ℝ) : ℝ := 4 / (1 + 1 / (a ^ 4))

theorem ratio_t_T_min_max :
  ∀ (a : ℝ), a ≥ 1 → ∃ (min_value max_bound : ℝ), 
    (min_value = 2 ∧ t_over_T 1 = min_value) ∧ 
    (∀ a, t_over_T a ≤ max_bound) ∧ 
    (max_bound = 4 ∧ ¬ ∃ a, t_over_T a > max_bound) := sorry

end ratio_t_T_min_max_l550_550540


namespace exists_set_with_properties_l550_550584

noncomputable def M_n (n : ℕ) : ℕ := sorry -- Placeholder for a sufficiently large M_n

def S : ℕ → set (ℕ × ℕ)
| 2     := { (0, 0), (1, 1) }
| (n+1) := S n ∪ { (x + 2^(n-1), y + M_n n) | (x, y) ∈ S n }

theorem exists_set_with_properties (n : ℕ) (h : n ≥ 2) :
  ∃ (S_n : set (ℕ × ℕ)), 
    S_n = S n ∧ 
    (∀ (p₁ p₂ p₃ : ℕ × ℕ), p₁ ∈ S_n → p₂ ∈ S_n → p₃ ∈ S_n → (p₁.1 * (p₂.2 - p₃.2) + p₂.1 * (p₃.2 - p₁.2) + p₃.1 * (p₁.2 - p₂.2) ≠ 0)) ∧
    (∀ (P : finset (ℕ × ℕ)), P ⊆ S n → P.card = 2 * n → ¬convex_hull P.to_set)
      := sorry -- actual proof goes here

end exists_set_with_properties_l550_550584


namespace age_difference_l550_550690

theorem age_difference 
  (A B : ℤ) 
  (h1 : B = 39) 
  (h2 : A + 10 = 2 * (B - 10)) :
  A - B = 9 := 
by 
  sorry

end age_difference_l550_550690


namespace find_m_l550_550469

theorem find_m (m : ℕ) : 
  (5 ∈ ({1, m+2, m^2+4} : set ℕ)) → (m = 1 ∨ m = 3) :=
by
  intro h,
  sorry

end find_m_l550_550469


namespace comparison_of_a_b_c_l550_550449

noncomputable def f : ℝ → ℝ := sorry -- f is some even function and not computable in this context

def a : ℝ := f (Real.log 7 / Real.log 4)

def b : ℝ := f (Real.log 3 / Real.log 2)

def c : ℝ := f (Real.exp (1.6 * Real.log 2))

theorem comparison_of_a_b_c :
  c < b ∧ b < a :=
by {
  -- Definition of even function
  have h_even : ∀ x : ℝ, f x = f (-x), from sorry,

  -- f is increasing on (-∞, 0]
  have h_increasing_negative : ∀ x y : ℝ, x < y → x ≤ 0 → y ≤ 0 → f x < f y, from sorry,

  -- f is decreasing on [0, +∞) since it's even and symmetric about the y-axis
  have h_decreasing_positive : ∀ x y : ℝ, x < y → 0 ≤ x → 0 ≤ y → f x > f y,
  {
    intros x y hxy hx hy,
    have : f x = f (-x), from h_even x,
    have : f y = f (-y), from h_even y,
    rw [this, this],
    apply h_increasing_negative; linarith,
  },

  -- Prove b = f (Real.log 9 / Real.log 4)
  have log_two_inequality : Real.log 3 / Real.log 2 = Real.log 9 / Real.log 4,
  {
    field_simp, rw [←Real.log_pow], norm_num,
  },
  have hb : b = f (Real.log 9 / Real.log 4),
  {
    rw ←log_two_inequality
  },

  -- Relationship between a, b, and c
  have h1 : Real.log 7 / Real.log 4 < Real.log 9 / Real.log 4,
  {
    apply div_lt_div_of_lt; norm_num, linarith,
  },
  have h2 : Real.log 9 / Real.log 4 < 1.6 * Real.log 2,
  {
    -- Use known logarithm properties; omitted for brevity
    sorry,
  },

  have ha_gt_hb : f (Real.log 7 / Real.log 4) > f (Real.log 9 / Real.log 4),
  {
    apply h_decreasing_positive,
    apply h1,
    simp,
  },
  have hb_gt_hc : f (Real.log 9 / Real.log 4) > f (1.6 * Real.log 2),
  {
    apply h_decreasing_positive,
    apply h2,
    simp,
  },

  exact ⟨hb_gt_hc, ha_gt_hb⟩,
}

end comparison_of_a_b_c_l550_550449


namespace cos_difference_identity_example_l550_550803

theorem cos_difference_identity_example (α : ℝ) (h1 : cos α = √2 / 10) (h2 : α ∈ Ioo (-π) 0) :
    cos (α - π / 4) = -3 / 5 := 
sorry

end cos_difference_identity_example_l550_550803


namespace blue_tickets_per_red_ticket_l550_550634

-- Definitions based on conditions
def yellow_tickets_to_win_bible : Nat := 10
def red_tickets_per_yellow_ticket : Nat := 10
def blue_tickets_needed : Nat := 163
def additional_yellow_tickets_needed (current_yellow : Nat) : Nat := yellow_tickets_to_win_bible - current_yellow
def additional_red_tickets_needed (current_red : Nat) (needed_yellow : Nat) : Nat := needed_yellow * red_tickets_per_yellow_ticket - current_red

-- Given conditions
def current_yellow_tickets : Nat := 8
def current_red_tickets : Nat := 3
def current_blue_tickets : Nat := 7
def needed_yellow_tickets : Nat := additional_yellow_tickets_needed current_yellow_tickets
def needed_red_tickets : Nat := additional_red_tickets_needed current_red_tickets needed_yellow_tickets

-- Theorem to prove
theorem blue_tickets_per_red_ticket : blue_tickets_needed / needed_red_tickets = 10 :=
by
  sorry

end blue_tickets_per_red_ticket_l550_550634


namespace course_count_l550_550347

theorem course_count (n1 n2 : ℕ) (sum_x1 sum_x2 : ℕ) :
  (n1 = 6) →
  (sum_x1 = n1 * 100) →
  (sum_x2 = n2 * 50) →
  ((sum_x1 + sum_x2) / (n1 + n2) = 77) →
  n2 = 5 :=
by
  intros h1 h2 h3 h4
  sorry

end course_count_l550_550347


namespace smaller_square_area_is_correct_l550_550226

noncomputable def smaller_square_area (larger_square_area : ℝ) : ℝ :=
  let side_of_larger_square := real.sqrt larger_square_area
  let half_side := side_of_larger_square / 2
  let side_of_smaller_square := real.sqrt (half_side ^ 2 + half_side ^ 2)
  let area_of_smaller_square := side_of_smaller_square ^ 2
  area_of_smaller_square

theorem smaller_square_area_is_correct (h : ∀ (s : ℝ), s ^ 2 = 100 → s = 10) : 
  smaller_square_area 100 = 50 :=
by
  have s := real.sqrt 100
  have half_s := s / 2
  have side_of_smaller_square := real.sqrt (half_s ^ 2 + half_s ^ 2)
  have area_of_smaller_square := side_of_smaller_square ^ 2
  have eq_s : s = 10 := h s (by linarith)
  rw [eq_s] at half_s
  have half_s_calc : half_s = 5 := by norm_num
  rw [half_s_calc] at side_of_smaller_square
  have side_of_smaller_square_calc : side_of_smaller_square = 5 * real.sqrt 2 := by norm_num
  rw [side_of_smaller_square_calc] at area_of_smaller_square
  simp only [real.sqrt_mul_self] at area_of_smaller_square
  exact area_of_smaller_square

end smaller_square_area_is_correct_l550_550226


namespace initial_investment_l550_550957

theorem initial_investment (P : ℝ) 
  (h1: ∀ (r : ℝ) (n : ℕ), r = 0.20 ∧ n = 3 → P * (1 + r)^n = P * 1.728)
  (h2: ∀ (A : ℝ), A = P * 1.728 → 3 * A = 5.184 * P)
  (h3: ∀ (P_new : ℝ) (r_new : ℝ), P_new = 5.184 * P ∧ r_new = 0.15 → P_new * (1 + r_new) = 5.9616 * P)
  (h4: 5.9616 * P = 59616)
  : P = 10000 :=
sorry

end initial_investment_l550_550957


namespace find_f_of_3_l550_550116

theorem find_f_of_3 (a b c : ℝ) (f : ℝ → ℝ) (h1 : f 1 = 7) (h2 : f 2 = 12) (h3 : ∀ x, f x = ax + bx + c) : f 3 = 17 :=
by
  sorry

end find_f_of_3_l550_550116


namespace value_of_one_house_l550_550771

theorem value_of_one_house
  (num_brothers : ℕ) (num_houses : ℕ) (payment_each : ℕ) 
  (total_money_paid : ℕ) (num_older : ℕ) (num_younger : ℕ)
  (share_per_younger : ℕ) (total_inheritance : ℕ) (value_of_house : ℕ) :
  num_brothers = 5 →
  num_houses = 3 →
  num_older = 3 →
  num_younger = 2 →
  payment_each = 800 →
  total_money_paid = num_older * payment_each →
  share_per_younger = total_money_paid / num_younger →
  total_inheritance = num_brothers * share_per_younger →
  value_of_house = total_inheritance / num_houses →
  value_of_house = 2000 :=
by {
  -- Provided conditions and statements without proofs
  sorry
}

end value_of_one_house_l550_550771


namespace sqrt_calculation_l550_550372

theorem sqrt_calculation : sqrt ((2^4 + 2^4 + 2^4 + 2^4) * 2) = 8 * sqrt 2 := 
by sorry

end sqrt_calculation_l550_550372


namespace sam_avg_speed_last_60_minutes_l550_550593

-- Define the total distance
def total_distance := 150 -- in miles

-- Define the total time
def total_time := 120 / 60 -- converted to hours

-- Define the speed for the first 30 minutes
def speed1 := 75 -- in mph

-- Define the speed for the second 30 minutes
def speed2 := 70 -- in mph

-- Define the correct average speed for the last 60 minutes
def correct_avg_speed_last_60 := 77.5 -- in mph

-- The theorem to prove
theorem sam_avg_speed_last_60_minutes :
  (total_distance - (speed1 * 0.5 + speed2 * 0.5)) / 1 = correct_avg_speed_last_60 :=
  sorry

end sam_avg_speed_last_60_minutes_l550_550593


namespace problem_l550_550422

def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 8

theorem problem 
  (a b c : ℝ) 
  (h : f a b c (-2) = 10) 
  : f a b c 2 = 6 :=
by
  sorry

end problem_l550_550422


namespace symmetric_point_l550_550487

theorem symmetric_point (a b : ℝ) (h1 : a = 2) (h2 : 3 = -b) : (a + b) ^ 2023 = -1 := 
by
  sorry

end symmetric_point_l550_550487


namespace order_of_magnitude_l550_550807

noncomputable def a := Real.log 3 / Real.log 4  -- log_4(3)
noncomputable def b := Real.log 4 / Real.log 3  -- log_3(4)
noncomputable def c := (0.3:ℝ) ^ (-2)         -- 0.3^-2

-- Prove that a < b < c
theorem order_of_magnitude : a < b ∧ b < c := 
by
  sorry

end order_of_magnitude_l550_550807


namespace red_point_connected_to_blue_and_green_l550_550940

/--
Given 1985 points on a plane, each colored either red, blue, or green, such that:
1. no three of them are collinear, and
2. each point has the same number of segments emanating from it,

prove that there exists at least one red point that is connected by segments to both a blue point and a green point.
-/
theorem red_point_connected_to_blue_and_green :
  ∃ (points : Fin 1985 → Color) (segments : Fin 1985 → Fin 1985 → Bool),
    (∀ i j, i ≠ j → segments i j = segments j i) ∧
    (∀ i, ∑ j in finset.univ, if segments i j then 1 else 0 = k) ∧
    (∀ i j k, i ≠ j ∧ i ≠ k ∧ j ≠ k → ¬(segments i j ∧ segments i k ∧ segments j k)) →
    ∃ i, ∃ j, ∃ k, points i = Color.red ∧ segments i j ∧ segments i k ∧ points j = Color.blue ∧ points k = Color.green :=
begin
  sorry
end

end red_point_connected_to_blue_and_green_l550_550940


namespace concyclic_points_l550_550216

-- Definitions for points and circles
variables {Point : Type}
variables {Circle : Type} [incidence_geometry : incidence_geom Point Circle]

-- Definitions for points A, B, C, D, X, Y, P, Q, R, S, U, V, W, Z.
variables (A B C D X Y P Q R S U V W Z : Point)

-- Definitions for circles ω₁ and ω₂.
variables (ω₁ ω₂ : Circle)

-- Hypotheses for the given problem conditions.
hypothesis h1 : quadrilateral_inscribed_in_circle A B C D ω₁
hypothesis h2 : circle_inters_circle X Y ω₁ ω₂
hypothesis h3 : circle_inter_line_at_points P Q ω₂ (line_through A B)
hypothesis h4 : circle_inter_line_at_points R S ω₂ (line_through C D)
hypothesis h5 : lines_inters_at_points U V W Z (line_through Q R) (line_through P S) (line_through A C) (line_through B D)

-- Statement to prove that X, Y, U, V, W, Z lie on the same circle.
theorem concyclic_points : concyclic X Y U V W Z :=
by sorry

end concyclic_points_l550_550216


namespace largest_x_intersection_l550_550623

-- Define the polynomial and the line
def P (x : ℝ) (r s : ℝ) : ℝ := x^6 - 12 * x^5 + 40 * x^4 - r * x^3 + s * x^2
def L (x : ℝ) (d e : ℝ) : ℝ := d * x - e

-- Condition: Polynomial intersects the line at exactly two double roots and one single root
structure IntersectionCondition (r s d e : ℝ) :=
  (double_root1 : ℝ)
  (double_root2 : ℝ)
  (single_root : ℝ)
  (is_double_root1 : P double_root1 r s = L double_root1 d e ∧ ∀ x ≠ double_root1, diff (P x r s - L x d e) double_root1 ≠ 0)
  (is_double_root2 : P double_root2 r s = L double_root2 d e ∧ ∀ x ≠ double_root2, diff (P x r s - L x d e) double_root2 ≠ 0)
  (is_single_root : P single_root r s = L single_root d e)

-- Statement of the proof problem
theorem largest_x_intersection
  (r s d e : ℝ) (h : IntersectionCondition r s d e) :
  max h.double_root1 (max h.double_root2 h.single_root) = 4 :=
sorry

end largest_x_intersection_l550_550623


namespace janeth_balloons_count_l550_550899

-- Define the conditions
def bags_round_balloons : Nat := 5
def balloons_per_bag_round : Nat := 20
def bags_long_balloons : Nat := 4
def balloons_per_bag_long : Nat := 30
def burst_round_balloons : Nat := 5

-- Proof statement
theorem janeth_balloons_count:
  let total_round_balloons := bags_round_balloons * balloons_per_bag_round
  let total_long_balloons := bags_long_balloons * balloons_per_bag_long
  let total_balloons := total_round_balloons + total_long_balloons
  total_balloons - burst_round_balloons = 215 :=
by {
  sorry
}

end janeth_balloons_count_l550_550899


namespace Tn_sum_l550_550793

noncomputable def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ := 2 - a n

noncomputable def bn (n : ℕ) : ℕ → ℝ
| 1       := 1
| (k + 2) := (2 * (k + 1) - 1) / (2 * (k + 1) + 1) * bn (k + 1)

noncomputable def an (n : ℕ) : ℝ := (1 / 2) ^ (n - 1)

noncomputable def cn (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) : ℝ := a n * b n

noncomputable def Tn (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, cn a b (i + 1)

theorem Tn_sum {a b : ℕ → ℝ} (h1 : ∀ n, Sn a n = 2 - a n)
  (h2 : ∀ n, (2 * n - 1) * b (n + 1) - (2 * n + 1) * b n = 0 ∧ b 1 = 1) :
  ∀ n, Tn a b n = 6 - (2 * n + 3) * (1 / 2) ^ (n - 1) :=
sorry

end Tn_sum_l550_550793


namespace power_function_general_form_l550_550258

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x ^ α

theorem power_function_general_form (α : ℝ) :
  ∃ y : ℝ, ∃ α : ℝ, f 3 α = y ∧ ∀ x : ℝ, f x α = x ^ α :=
by
  sorry

end power_function_general_form_l550_550258


namespace net_result_ms_c_after_transactions_l550_550332

theorem net_result_ms_c_after_transactions
  (initial_value : ℝ := 12000)
  (first_loss_percentage : ℝ := 0.15)
  (second_gain_percentage : ℝ := 0.20) :
  let first_transaction_price := initial_value * (1 - first_loss_percentage),
      second_transaction_price := first_transaction_price * (1 + second_gain_percentage),
      net_result := second_transaction_price - initial_value in
  net_result = 240 := by
  sorry

end net_result_ms_c_after_transactions_l550_550332


namespace centroid_division_proof_l550_550926

variables {A B C B₁ C₁ G : Type}
variables (triangle : Type) (centroid : triangle → Type) 
          (line : Type) (intersect : line → Type → Type → Type)
          (ratioAB: ℝ) (ratioAC: ℝ)

theorem centroid_division_proof
  (G_is_centroid : G = centroid triangle)
  (line_intersectAB₁ : intersect line A B₁)
  (line_intersectAC₁ : intersect line A C₁)
  (h1 : ratioAB = λ)
  (h2 : ratioAC = μ)
  : 1/λ + 1/μ = 3 := by tidy

end centroid_division_proof_l550_550926


namespace sample_description_l550_550679

-- Definitions based on conditions.
def middle_school_understands_vision_situation (school : Type) (students : set school) : Prop :=
  ∃ eighth_grade_students : set school, students = eighth_grade_students ∧
  ∃ selected_students, selected_students ⊆ eighth_grade_students ∧ selected_students.card = 30

-- Theorem statement.
theorem sample_description (school : Type) (students : set school)
  (H : middle_school_understands_vision_situation school students) :
  ∃ selected_students, selected_students ⊆ students ∧ selected_students.card = 30 :=
begin
  sorry
end

end sample_description_l550_550679


namespace ten_pieces_needed_board_filled_completely_impossible_to_fill_with_one_special_l550_550901

-- Part (a): Prove that 10 pieces are required to cover a 5x8 board, given each piece covers 4 squares.
theorem ten_pieces_needed (n : ℕ) (h1 : n = 5 * 8) (h2 : ∀ p, p = 4) : n / 4 = 10 := 
sorry

-- Part (b): Prove the board can be filled completely with the pieces.
theorem board_filled_completely : 
  ∃ arrangement : list (vector ℕ 2), (∀ p ∈ arrangement, true) -> true :=
sorry

-- Part (c): Prove it is impossible to fill a 5x8 board using one special piece and the rest regular pieces.
theorem impossible_to_fill_with_one_special (n : ℕ) (h1 : n = 5 * 8)
  (special_piece : vector (char × char) 4)  -- denoting color pairs
  (regular_piece : vector (char × char) 4) : 
  ∃ n_white n_grey : ℕ, n_white = 20 ∧ n_grey = 20 ∧ n_white + n_grey = 40  -> false :=
sorry

end ten_pieces_needed_board_filled_completely_impossible_to_fill_with_one_special_l550_550901


namespace sol_earnings_in_a_week_l550_550964

-- Define the number of candy bars sold each day using recurrence relation
def candies_sold (n : ℕ) : ℕ :=
  match n with
  | 0     => 10  -- Day 1
  | (n+1) => candies_sold n + 4  -- Each subsequent day

-- Define the total candies sold in a week and total earnings in dollars
def total_candies_sold_in_a_week : ℕ :=
  List.sum (List.map candies_sold [0, 1, 2, 3, 4, 5])

def total_earnings_in_dollars : ℕ :=
  (total_candies_sold_in_a_week * 10) / 100

-- Proving that Sol will earn 12 dollars in a week
theorem sol_earnings_in_a_week : total_earnings_in_dollars = 12 := by
  sorry

end sol_earnings_in_a_week_l550_550964


namespace sample_size_correct_l550_550088

-- Definitions following the conditions in the problem
def total_products : Nat := 80
def sample_products : Nat := 10

-- Statement of the proof problem
theorem sample_size_correct : sample_products = 10 :=
by
  -- The proof is replaced with a placeholder sorry to skip the proof step
  sorry

end sample_size_correct_l550_550088


namespace sqrt_224_between_14_and_15_l550_550770

theorem sqrt_224_between_14_and_15 : 14 < Real.sqrt 224 ∧ Real.sqrt 224 < 15 := by
  sorry

end sqrt_224_between_14_and_15_l550_550770


namespace fractions_sum_identity_l550_550871

theorem fractions_sum_identity (a b c : ℝ) (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / ((b - c) ^ 2) + b / ((c - a) ^ 2) + c / ((a - b) ^ 2) = 0 :=
by
  sorry

end fractions_sum_identity_l550_550871


namespace triangle_area_ratio_l550_550292

theorem triangle_area_ratio (A B C A' B' C' : Type)
  [noncomputable_area : noncomputable] -- Ensures our area calculation can be handled
  (similar : similar_triangles A B C A' B' C') -- Triangles are similar
  (ratio : side_length_ratio A B C A' B' C' = 1 / 2) -- Ratio of corresponding sides
  (A1 : ℝ) -- Area of triangle A B C
  (A2 : ℝ) -- Area of triangle A' B' C'
  (area_A : area A B C = A1) -- Define the areas
  (area_B : area A' B' C' = A2) :
  A2 = A1 / 4 := 
sorry

end triangle_area_ratio_l550_550292


namespace initial_volume_salt_solution_l550_550853

theorem initial_volume_salt_solution (V : ℝ) (V1 : ℝ) (V2 : ℝ) : 
  V1 = 0.20 * V → 
  V2 = 30 →
  V1 = 0.15 * (V + V2) →
  V = 90 := 
by 
  sorry

end initial_volume_salt_solution_l550_550853


namespace cone_base_area_l550_550159

theorem cone_base_area (r l : ℝ) (h1 : (1/2) * π * l^2 = 2 * π) (h2 : 2 * π * r = 2 * π) :
  π * r^2 = π :=
by 
  sorry

end cone_base_area_l550_550159


namespace frequency_of_function_l550_550613

theorem frequency_of_function :
  ∀ x : ℝ, ∀ y : ℝ, (y = 3 * sin(2 * x + (π / 4))) → (y = 3 * sin(2 * x + (π / 4)) → 1 / (π) = 1 / (π)) :=
by
  intros x y h
  rw h
  sorry

end frequency_of_function_l550_550613


namespace intersection_of_M_and_N_l550_550834

namespace ProofProblem

def M := { x : ℝ | x^2 < 4 }
def N := { x : ℝ | x < 1 }

theorem intersection_of_M_and_N :
  M ∩ N = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end ProofProblem

end intersection_of_M_and_N_l550_550834


namespace bus_speed_excluding_stoppages_l550_550053

noncomputable def average_speed_excluding_stoppages
  (speed_including_stoppages : ℝ)
  (stoppage_time_ratio : ℝ) : ℝ :=
  (speed_including_stoppages * 1) / (1 - stoppage_time_ratio)

theorem bus_speed_excluding_stoppages :
  average_speed_excluding_stoppages 15 (3/4) = 60 := 
by
  sorry

end bus_speed_excluding_stoppages_l550_550053


namespace ratio_areas_trapezoid_rectangle_l550_550379

variables {O : Type*} [has_dist O ℝ]
variables (A B C D P Q E F G H : O)
variable (r : ℝ) (dist_AB : ℝ) (dist_CD : ℝ)

-- Define some conditions
def circle_centered_O_with_radius_r (O : O) (r : ℝ) : Prop :=
  dist O A = r ∧ dist O B = r ∧ dist O C = r ∧ dist O D = r

def parallel_chords (A B C D : O) : Prop :=
  ∃ (m1 m2 : ℝ), (m1 = m2) ∧ (dist A B = dist C D)

def AB_4_inches_closer_than_CD (dist_AB dist_CD : ℝ) : Prop :=
  dist_AB + 4 = dist_CD

def collinear_OPQ (O P Q : O) : Prop :=
  collinear {O, P, Q}

def midpoints_PQ (A B C D P Q : O) : Prop :=
  dist A P = dist P B ∧ dist C Q = dist Q D

def length_AB_half_length_CD (dist_AB dist_CD : ℝ) : Prop :=
  dist_AB = dist_CD / 2

def rectangle_EFGH_tangent_to_circle (E F G H O : O) : Prop :=
  is_tangent E O r ∧ is_tangent F O r ∧ is_tangent G O r ∧ is_tangent H O r

-- Final theorem statement
theorem ratio_areas_trapezoid_rectangle
  (hc : circle_centered_O_with_radius_r O r)
  (hc1 : parallel_chords A B C D)
  (hc2 : AB_4_inches_closer_than_CD dist_AB dist_CD)
  (hc3 : collinear_OPQ O P Q)
  (hc4 : midpoints_PQ A B C D P Q)
  (hc5 : length_AB_half_length_CD dist_AB dist_CD)
  (hc6 : rectangle_EFGH_tangent_to_circle E F G H O) :
  (area_trapezoid A B C D) / (area_rectangle E F G H) = 7 / 32 :=
sorry

end ratio_areas_trapezoid_rectangle_l550_550379


namespace percentage_selected_B_l550_550170

-- Definitions for the given conditions
def candidates := 7900
def selected_A := (6 / 100) * candidates
def selected_B := selected_A + 79

-- The question to be answered
def P_B := (selected_B / candidates) * 100

-- Proof statement
theorem percentage_selected_B : P_B = 7 := 
by
  -- Canonical statement placeholder 
  sorry

end percentage_selected_B_l550_550170


namespace root_within_domain_l550_550128

noncomputable def n : ℝ := ∫ (t : ℝ) in 0..(π / 2), 2 * sin (t / 2) * cos (t / 2)

def f (a x : ℝ) : ℝ := (a / x) + log x - n

theorem root_within_domain (a : ℝ) (h : 0 < a) : (∃ x : ℝ, 0 < x ∧ f a x = 0) ↔ 0 < a ∧ a ≤ 1 :=
by sorry

end root_within_domain_l550_550128


namespace best_coupon1_price_l550_550685

theorem best_coupon1_price (x : ℝ) 
    (h1 : 60 ≤ x ∨ x = 60)
    (h2_1 : 25 < 0.12 * x) 
    (h2_2 : 0.12 * x > 0.2 * x - 30) :
    x = 209.95 ∨ x = 229.95 ∨ x = 249.95 :=
by sorry

end best_coupon1_price_l550_550685


namespace sales_on_second_day_l550_550682

variable (m : ℕ)

-- Define the condition for sales on the first day
def first_day_sales : ℕ := m

-- Define the condition for sales on the second day
def second_day_sales : ℕ := 2 * first_day_sales m - 3

-- The proof statement
theorem sales_on_second_day (m : ℕ) : second_day_sales m = 2 * m - 3 := by
  -- provide the actual proof here
  sorry

end sales_on_second_day_l550_550682


namespace sum_of_five_digit_binary_numbers_l550_550912

theorem sum_of_five_digit_binary_numbers : 
  let T := {n : ℕ | n >= 16 ∧ n <= 31} in
  (∑ n in T, n) = 376 :=
by 
  -- declare the set of integers
  let T := {n : ℕ | n >= 16 ∧ n <= 31} in
  -- assert the summation of elements in T
  have hT: ∑ n in T, n = (1 + 2^4 - 1) * (16 + 31) / 2 :=
    by sorry,
  show (∑ n in T, n) = 376 from hT

end sum_of_five_digit_binary_numbers_l550_550912


namespace count_satisfying_mappings_l550_550802

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℤ := {-1, -2}

def f (x : ℕ) : ℤ := sorry -- Define the mapping

theorem count_satisfying_mappings :
  (∀ b : ℤ, b ∈ B → ∃ a : ℕ, a ∈ A ∧ f a = b) → -- Condition that every element in B must be an image under f
  fintype.card ((A → B) ⊓ {f | ∀ b : ℤ, b ∈ B → ∃ a : ℕ, a ∈ A ∧ f a = b}) = 14 := sorry

end count_satisfying_mappings_l550_550802


namespace find_x_satisfies_equation_l550_550399

noncomputable theory

open Real

theorem find_x_satisfies_equation :
  ∀ x : ℝ, x > 0 → log 5 (x - 2) + log (sqrt 5) (x^3 - 2) + log (1 / 5) (x - 2) = 4 → x = 3 :=
by
  sorry

end find_x_satisfies_equation_l550_550399


namespace gcd_of_powers_of_three_l550_550644

theorem gcd_of_powers_of_three :
  let a := 3^1001 - 1
  let b := 3^1012 - 1
  gcd a b = 177146 := by
  sorry

end gcd_of_powers_of_three_l550_550644


namespace count_integers_in_solution_set_l550_550849

-- Define the predicate for the condition given in the problem
def condition (x : ℝ) : Prop := abs (x - 3) ≤ 4.5

-- Define the list of integers within the range of the condition
def solution_set : List ℤ := [-1, 0, 1, 2, 3, 4, 5, 6, 7]

-- Prove that the number of integers satisfying the condition is 8
theorem count_integers_in_solution_set : solution_set.length = 8 :=
by
  sorry

end count_integers_in_solution_set_l550_550849


namespace ellipse_C_equation_ellipse_eccentricity_quadrilateral_minimum_area_l550_550819

section EllipseProblem

variable (m : ℝ) (x y : ℝ) (x0 y0 : ℝ) (h_positive : m > 0) (h_eq : m * x^2 + 3 * m * y^2 = 1)
variable (O A P B D : Point)
variable (A_x : ℝ) (A_y : ℝ)
variable (P_x : ℝ) (P_y : ℝ)
variable (B_x : ℝ) (B_y : ℝ)
variable (D_x : ℝ) (D_y : ℝ)
variable (B_y0 : ℝ)

-- Definitions based on the conditions
def ellipse_C_eq := (m > 0) ∧ (m * x^2 + 3 * m * y^2 = 1)

def point_O : Point := (0, 0)
def point_A (x : ℝ) := (3, 0) -- since point A (3,0)
def point_P (x y : ℝ) := (x0 > 0) ∧ (m * x^2 + 3 * m * y^2 = 1)
def point_B (x : ℝ) := (0, B_y)
def point_D (x y : ℝ) := ((x0 + 3) / 2, y0 / 2)

def AP_slope := (y0 / (x0 - 3))
def BD_slope := ((3 - x0) / y0) 

def line_BD := (y - y0 / 2 = (3 - x0) / y0 * (x - (x0 + 3) / 2))

def quadrilateral_area (y0 : ℝ) : ℝ := (3 / 2 * (2 * |y0| + 3 / (2 * |y0|))) 

-- Statements to be proved
theorem ellipse_C_equation : ellipse_C_eq m x y → (x^2 / 6 + y^2 / 2 = 1) := 
sorry

theorem ellipse_eccentricity : (c : ℝ) → (a : ℝ) → (c = √6) → (a = 3) → (e = c / a) := 
sorry

theorem quadrilateral_minimum_area : (BA = BP) → (quadrilateral_area y0 ≥ 3 * √3) := 
sorry

end EllipseProblem

end ellipse_C_equation_ellipse_eccentricity_quadrilateral_minimum_area_l550_550819


namespace ratio_shelby_to_total_l550_550141

theorem ratio_shelby_to_total (total_will : ℝ) (amount_per_grandchild : ℝ) (num_remaining_grandchildren : ℕ) 
  (h_total : total_will = 124600) 
  (h_amount_grandchild : amount_per_grandchild = 6230) 
  (h_num_grandchildren : num_remaining_grandchildren = 10) : 
  (62300 / total_will = 1 / 2) := 
by
  have h_div : amount_per_grandchild * num_remaining_grandchildren = 62300 := sorry
  have h_shelby : total_will - 62300 = 62300 := sorry
  rw [h_total, h_shelby]
  sorry

end ratio_shelby_to_total_l550_550141


namespace machine_worked_yesterday_l550_550362

noncomputable def shirts_made_per_minute : ℕ := 3
noncomputable def shirts_made_yesterday : ℕ := 9

theorem machine_worked_yesterday : 
  (shirts_made_yesterday / shirts_made_per_minute) = 3 :=
sorry

end machine_worked_yesterday_l550_550362


namespace even_function_G_l550_550424

variable {a : ℝ} (F : ℝ → ℝ)

noncomputable def g (x : ℝ) : ℝ :=
  (1 / (a ^ x - 1)) + 1 / 2

noncomputable def G (x : ℝ) : ℝ :=
  F x * g x
  
theorem even_function_G (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : ∀ x, F (-x) = -F x) :
  ∀ x, G (-x) = G x :=
by
  sorry

end even_function_G_l550_550424


namespace number_of_integers_between_sqrt_10_and_sqrt_80_l550_550145

theorem number_of_integers_between_sqrt_10_and_sqrt_80 : 
  let lower := Int.ceil (Real.sqrt 10)
  let upper := Int.floor (Real.sqrt 80)
  upper - lower + 1 = 5 :=
by
  have h1 : lower = 4 := sorry
  have h2 : upper = 8 := sorry
  simp [h1, h2]
  norm_num

end number_of_integers_between_sqrt_10_and_sqrt_80_l550_550145


namespace sequence_inequality_l550_550305

def a : ℕ → ℕ
| 0 := 1
| n+1 := 4 * a n - ∑ i in range (n+1), a i

theorem sequence_inequality (n : ℕ) : a n ≥ 2^n :=
by
  sorry

end sequence_inequality_l550_550305


namespace thirtieth_change_month_is_february_l550_550052

def months_in_year := 12

def months_per_change := 7

def first_change_month := 3 -- March (if we assume January = 1, February = 2, etc.)

def nth_change_month (n : ℕ) : ℕ :=
  (first_change_month + months_per_change * (n - 1)) % months_in_year

theorem thirtieth_change_month_is_february :
  nth_change_month 30 = 2 := -- February (if we assume January = 1, February = 2, etc.)
by 
  sorry

end thirtieth_change_month_is_february_l550_550052


namespace ralph_socks_l550_550230

theorem ralph_socks (x y z : ℕ) (h1 : x + y + z = 12) (h2 : x + 3 * y + 4 * z = 24) (h3 : 1 ≤ x) (h4 : 1 ≤ y) (h5 : 1 ≤ z) : x = 7 :=
sorry

end ralph_socks_l550_550230


namespace cubic_root_equation_solution_l550_550059

theorem cubic_root_equation_solution (x : ℝ) :
  (real.cbrt (10 * x - 2) + real.cbrt (8 * x + 2) = 3 * real.cbrt x) → (x = 0) := 
by
  sorry

end cubic_root_equation_solution_l550_550059


namespace calculate_factorial_expression_l550_550725

theorem calculate_factorial_expression :
  6 * nat.factorial 6 + 5 * nat.factorial 5 + nat.factorial 5 = 5040 := 
sorry

end calculate_factorial_expression_l550_550725


namespace total_animals_is_63_l550_550357

def zoo_animals (penguins polar_bears total : ℕ) : Prop :=
  (penguins = 21) ∧
  (polar_bears = 2 * penguins) ∧
  (total = penguins + polar_bears)

theorem total_animals_is_63 :
  ∃ (penguins polar_bears total : ℕ), zoo_animals penguins polar_bears total ∧ total = 63 :=
by {
  sorry
}

end total_animals_is_63_l550_550357


namespace sum_of_roots_l550_550401

-- sum of roots of first polynomial
def S1 : ℚ := -(-6 / 3)

-- sum of roots of second polynomial
def S2 : ℚ := -(8 / 4)

-- proof statement
theorem sum_of_roots : S1 + S2 = 0 :=
by
  -- placeholders
  sorry

end sum_of_roots_l550_550401


namespace michael_scored_times_more_goals_l550_550658

theorem michael_scored_times_more_goals (x : ℕ) (hb : Bruce_goals = 4) (hm : Michael_goals = 4 * x) (ht : Bruce_goals + Michael_goals = 16) : x = 3 := by
  sorry

end michael_scored_times_more_goals_l550_550658


namespace calc_factorial_sum_l550_550722

theorem calc_factorial_sum : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 = 5040 := by
  sorry

end calc_factorial_sum_l550_550722


namespace A_3_2_eq_13_l550_550741

def A : ℕ → ℕ → ℕ
  | 0, n := n + 1
  | m+1, 0 := A m 1
  | m+1, n+1 := A m (A (m+1) n)

theorem A_3_2_eq_13 : A 3 2 = 13 := by
  sorry

end A_3_2_eq_13_l550_550741


namespace num_divisors_2002_l550_550766

def numberOfDivisors (n : ℕ) : ℕ :=
  let factors := [(2, 1), (7, 1), (11, 1), (13, 1)]
  factors.foldr (λ (p_exp : ℕ × ℕ) acc, acc * (p_exp.snd + 1)) 1

theorem num_divisors_2002 : numberOfDivisors 2002 = 16 := 
by 
  sorry

end num_divisors_2002_l550_550766


namespace width_of_rect_prism_l550_550341

theorem width_of_rect_prism (l h d : ℝ) (l_val : l = 6) (h_val : h = 8) (d_val : d = 10) :
  ∃ (w : ℝ), (l^2 + w^2 + h^2 = d^2) ∧ w = 0 :=
by
  use 0
  split
  · sorry
  · exact rfl

end width_of_rect_prism_l550_550341


namespace arcade_tickets_l550_550715

/-- At the arcade, Dave won 11 tickets, spent 5 tickets on a beanie, and later had a total of 16 tickets. Prove that Dave won 10 tickets later. -/
theorem arcade_tickets :
  (initial_tickets spent_tickets total_tickets tickets_won : ℕ)
  (h1 : initial_tickets = 11)
  (h2 : spent_tickets = 5)
  (h3 : total_tickets = 16) :
  tickets_won = total_tickets - (initial_tickets - spent_tickets) :=
by
  -- Provide numerical values
  let initial_tickets := 11
  let spent_tickets := 5
  let total_tickets := 16
  let tickets_won := total_tickets - (initial_tickets - spent_tickets)
  -- Use the hypothesis to conclude
  have h : tickets_won = 10 := by
    calc tickets_won = total_tickets - (initial_tickets - spent_tickets) : by rfl
                  ... = 16 - (11 - 5) : by rfl
                  ... = 10 : by rfl
  exact h

end arcade_tickets_l550_550715


namespace area_of_region_l550_550760

theorem area_of_region : 
  (let s := {p : ℝ × ℝ | |p.1 + 2 * p.2| + |2 * p.1 - p.2| ≤ 6} in 
   measure_theory.measure.measure s) = 5.76 :=
by
  sorry

end area_of_region_l550_550760


namespace paving_stones_needed_l550_550680

-- Definition for the dimensions of the paving stone and the courtyard
def paving_stone_length : ℝ := 2.5
def paving_stone_width : ℝ := 2
def courtyard_length : ℝ := 30
def courtyard_width : ℝ := 16.5

-- Compute areas
def paving_stone_area : ℝ := paving_stone_length * paving_stone_width
def courtyard_area : ℝ := courtyard_length * courtyard_width

-- The theorem to prove that the number of paving stones needed is 99
theorem paving_stones_needed :
  (courtyard_area / paving_stone_area) = 99 :=
by
  sorry

end paving_stones_needed_l550_550680


namespace max_integer_a_l550_550827

def f (x : ℝ) (a : ℝ) : ℝ := (x - a + 1) * Real.exp x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x a + a

theorem max_integer_a (x : ℝ) (a : ℝ) (hx : x > 0) (h : g x a > 0) : a ≤ 3 :=
by
  sorry

end max_integer_a_l550_550827


namespace geometric_sequence_sum_l550_550790

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 = 1) (h3 : a 4 * a 5 * a 6 = 8) :
  a 2 + a 5 + a 8 + a 11 = 15 :=
by
  sorry

end geometric_sequence_sum_l550_550790


namespace second_pump_drain_time_l550_550944

-- Definitions of the rates R1 and R2
def R1 : ℚ := 1 / 12  -- Rate of the first pump
def R2 : ℚ := 1 - R1  -- Rate of the second pump (from the combined rate equation)

-- The time it takes the second pump alone to drain the pond
def time_to_drain_second_pump := 1 / R2

-- The goal is to prove that this value is 12/11
theorem second_pump_drain_time : time_to_drain_second_pump = 12 / 11 := by
  -- The proof is omitted
  sorry

end second_pump_drain_time_l550_550944


namespace rhombus_BAD_angle_l550_550607

noncomputable def rhombus_angle_BAD (a : ℝ) (h_area : 0 < a ∧ a < 1): ℝ :=
  2 * real.arcsin (real.sqrt a)

theorem rhombus_BAD_angle (a : ℝ) (h_area : 0 < a ∧ a < 1) :
  (∃ area_rhombus == 2]() → THISsentence not exists* 
  (∃ inscribed_circle_ABD (touching_at_K : true))] :
  rhombus_angle_BAD a h_area = 2 * real.arcsin (real.sqrt a) :=
sorry

end rhombus_BAD_angle_l550_550607


namespace twin_primes_solution_l550_550047

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def are_twin_primes (p q : ℕ) : Prop := is_prime p ∧ is_prime q ∧ (p = q + 2 ∨ q = p + 2)

theorem twin_primes_solution (p q : ℕ) :
  are_twin_primes p q ∧ is_prime (p^2 - p * q + q^2) ↔ (p, q) = (5, 3) ∨ (p, q) = (3, 5) := by
  sorry

end twin_primes_solution_l550_550047


namespace scalene_right_triangle_area_l550_550242

-- Given problem conditions as definitions
def triangle (a b c : Type _) :=
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ ∃ (a' b' c' : ℝ), a' ≠ 0 ∧ b' ≠ 0 ∧ a' ^ 2 + b' ^ 2 = c' ^ 2

def right_triangle (A B C : Type _) (a b c : ℝ) :=
  triangle A B C ∧ a = 2 * b ∧ c = a + b

def point_on_hypotenuse (A B C P : Type _) :=
  ∃ AP CP, AP = 1 ∧ CP = 2 ∧ (angle A B P = π / 4 ∧ angle B C P = π / 4)

-- Triangle area to be computed
def triangle_area (A B C : Type _) (b c : ℝ) :=
  0.5 * b * c

-- Combining everything to form the main theorem
theorem scalene_right_triangle_area :
  ∀ (A B C P : Type _) (b c : ℝ), 
    right_triangle A B C (hypotenuse P A C) b c ∧ point_on_hypotenuse A B C P →
    triangle_area A B C b c = 9 / 5 :=
by
  intros
  sorry

end scalene_right_triangle_area_l550_550242


namespace range_of_a_l550_550835

theorem range_of_a (a : ℝ) : 
  (∃ x : ℤ, a < x ∧ x < 1 ∧ x ∈ {-2, -1, 0} ) → -3 ≤ a ∧ a < -2 :=
by
  sorry

end range_of_a_l550_550835


namespace length_of_hypotenuse_l550_550249

-- Definitions of conditions in the problem
def base : ℕ := 8
def area : ℕ := 24

-- Definition of the height using the area formula
def height (b : ℕ) (a : ℕ) : ℕ := (2 * a) / b

-- Using the Pythagorean theorem to find the hypotenuse
def hypotenuse (b h : ℕ) : ℕ := Int.natAbs (Int.sqrt (b^2 + h^2))

theorem length_of_hypotenuse :
  hypotenuse base (height base area) = 10 :=
by
  sorry

end length_of_hypotenuse_l550_550249


namespace domain_of_g_l550_550734

-- Definition of the function
def g (x : ℝ) : ℝ := 1 / ⌊x^2 - 6 * x + 10⌋

-- Lean statement to prove the domain of the function g
theorem domain_of_g : {x : ℝ | (x < 3 ∨ x > 3)} =
                      {x : ℝ | g(x) ∈ ℝ} :=
by
  -- Proof is omitted
  sorry

end domain_of_g_l550_550734


namespace pigeons_among_non_sparrows_l550_550015

theorem pigeons_among_non_sparrows (P_total P_parrots P_peacocks P_sparrows : ℝ)
    (h1 : P_total = 20)
    (h2 : P_parrots = 30)
    (h3 : P_peacocks = 15)
    (h4 : P_sparrows = 35) :
    (P_total / (100 - P_sparrows)) * 100 = 30.77 :=
by
  -- Proof will be provided here
  sorry

end pigeons_among_non_sparrows_l550_550015


namespace cyclic_points_proof_l550_550316

noncomputable def cyclic_quadrilateral (A B C D P E F I J K : Type*) 
  [euclidean_geometry.{0} A B C D P E F I J K] : Prop :=
∃ (ABCD_cyclic : cyclic A B C D) 
  (A_B_intersect_at_E : circle_intersect (circle (A, P, D)) E A ∧ on_segment A B E)
  (B_C_intersect_at_F : circle_intersect (circle (B, P, C)) F B ∧ on_segment A B F)
  (I_incenter_ADE : incenter I (A, D, E))
  (J_incenter_BCF : incenter J (B, C, F))
  (IJ_AC_meet_K : line_intersect (line I J) (line A C) K),
  cyclic A I K E

theorem cyclic_points_proof (A B C D P E F I J K : Type*) 
  [euclidean_geometry.{0} A B C D P E F I J K] :
  cyclic_quadrilateral A B C D P E F I J K :=
begin
  sorry
end

end cyclic_points_proof_l550_550316


namespace condition2_implies_perpendicular_condition3_implies_perpendicular_condition4_implies_perpendicular_l550_550092

-- Given Definitions and Conditions
variables (a b : Line) (α β : Plane)

-- Definitions & Theorems for the problem
def condition1 := a ⊆ α ∧ b ∥ β ∧ α ⊥ β
def condition2 := a ⊥ α ∧ b ⊥ β ∧ α ⊥ β
def condition3 := a ⊆ α ∧ b ⊥ β ∧ α ∥ β
def condition4 := a ⊥ α ∧ b ∥ β ∧ α ∥ β

-- Theorem statements to be proved
theorem condition2_implies_perpendicular : condition2 a b α β → a ⊥ b := sorry
theorem condition3_implies_perpendicular : condition3 a b α β → a ⊥ b := sorry
theorem condition4_implies_perpendicular : condition4 a b α β → a ⊥ b := sorry

end condition2_implies_perpendicular_condition3_implies_perpendicular_condition4_implies_perpendicular_l550_550092


namespace limit_calculation_l550_550025

open Real

theorem limit_calculation :
    tendsto (fun x => (exp (tan (2 * x)) - exp (-sin (2 * x))) / (sin x - 1))
            (𝓝 (π / 2)) (𝓝 0) :=
sorry

end limit_calculation_l550_550025


namespace area_of_inscribed_equilateral_triangle_l550_550027

theorem area_of_inscribed_equilateral_triangle 
  (r : ℝ) 
  (h_r : r = 8) : 
  let s := r * real.sqrt 3 in 
  let A := (real.sqrt 3 / 4) * s^2 in 
  A = 48 * real.sqrt 3 := 
by 
  sorry

end area_of_inscribed_equilateral_triangle_l550_550027


namespace problem_part1_problem_part2_l550_550836

open Set Real

noncomputable def A : Set ℝ := {x : ℝ | 2 * x ^ 2 - x - 6 ≥ 0}
noncomputable def B : Set ℝ := {x : ℝ | x > 0 ∧ log 2 x ≤ 2}
noncomputable def C (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < a + 1}

theorem problem_part1 (x : ℝ) : 
  (x ∈ A ∧ x ∈ B ↔ 2 ≤ x ∧ x ≤ 4) ∧
  (x ∉ B ∨ x ∈ A ↔ x ≤ 0 ∨ x ≥ 2) :=
by {
  sorry
}

theorem problem_part2 {a : ℝ} :
  (∀ {x : ℝ}, x ∈ C a → x ∈ B) ↔ 0 ≤ a ∧ a ≤ 3 :=
by {
  sorry
}

end problem_part1_problem_part2_l550_550836


namespace Lea_total_cost_l550_550219

theorem Lea_total_cost :
  let book_cost := 16 in
  let binder_cost := 3 * 2 in
  let notebook_cost := 6 * 1 in
  let pen_cost := 4 * 0.5 in
  let calculator_cost := 2 * 12 in
  book_cost + binder_cost + notebook_cost + pen_cost + calculator_cost = 54 :=
by
  sorry

end Lea_total_cost_l550_550219


namespace triangle_ABC_is_right_l550_550604

-- Definitions to set up the problem context:
variables {A B C T S : Type} [Triangle A B C] [Point T] [Line AB] [Line TC] [Point S]

-- Given conditions:
def tangents_intersect_circumcircle (ABC : Triangle) : Prop :=
  let circumcircle := circumcircle ABC in
  is_tangent circumcircle A T ∧ is_tangent circumcircle C T ∧
  lines_intersect_at (line_through A B) (line_through T C) S ∧
  area (△ ACT) = area (△ ABC) ∧ area (△ ABC) = area (△ BCS)

-- The statement to prove:
theorem triangle_ABC_is_right (ABC : Triangle) (Ht : tangents_intersect_circumcircle ABC) :
  is_right_triangle ABC :=
sorry

end triangle_ABC_is_right_l550_550604


namespace problem_statement_l550_550473

-- Define the centers and radii of the circles
def circle1_center := (0, 0)
def circle1_radius := 1

def circle2_center := (2, 0)
def circle2_radius := 2

-- Distance between the centers of the circles C1 and C2
def distance_C1_C2 : ℝ :=
  real.sqrt ((2 - 0)^2 + (0 - 0)^2) 

-- Equation of the line passing through points A and B
def line_AB_equation : ℝ :=
  1 / 4

-- Length of segment AB
def distance_AB : ℝ :=
  real.sqrt 15 / 2

-- The theorem that needs to be proved
theorem problem_statement :
  distance_C1_C2 = 2 ∧ 
  line_AB_equation = 1 / 4 ∧ 
  distance_AB = real.sqrt 15 / 2 := 
by sorry

end problem_statement_l550_550473


namespace positive_integer_tuples_count_l550_550475

theorem positive_integer_tuples_count : 
  (∑ i in Finset.range 13, (x : ℕ) → x ≤ 2006) = Nat.binomial 2006 13 := 
sorry

end positive_integer_tuples_count_l550_550475


namespace nicky_run_time_before_catched_l550_550573

theorem nicky_run_time_before_catched (vc vn t_head_start : ℝ) (t_catched : ℝ)
  (h1 : vc = 5) (h2 : vn = 3) (h3 : t_head_start = 12) (h4 : t_catched = 30) :
  ∃ t : ℝ, vn * (t + t_head_start) = vc * t ∧ (t + t_head_start) = t_catched :=
begin
  sorry
end

end nicky_run_time_before_catched_l550_550573


namespace geometric_series_sum_correct_l550_550035

-- Given conditions
def a : ℤ := 3
def r : ℤ := -2
def n : ℤ := 10

-- Sum of the geometric series formula
def geometric_series_sum (a r n : ℤ) : ℤ := 
  a * (r^n - 1) / (r - 1)

-- Goal: Prove that the sum of the series is -1023
theorem geometric_series_sum_correct : 
  geometric_series_sum a r n = -1023 := 
by
  sorry

end geometric_series_sum_correct_l550_550035


namespace log_base2_of_3_l550_550804

theorem log_base2_of_3 (a : ℝ) (h : 3^a = 4) : log 2 3 = 2 / a :=
by 
  sorry

end log_base2_of_3_l550_550804


namespace constant_term_of_product_is_21_l550_550641

def P (x : ℕ) : ℕ := x ^ 3 + x ^ 2 + 3
def Q (x : ℕ) : ℕ := 2 * x ^ 4 + x ^ 2 + 7

theorem constant_term_of_product_is_21 :
  (P 0) * (Q 0) = 21 :=
by
  rw [P, Q]
  simp
  rfl

end constant_term_of_product_is_21_l550_550641


namespace LCM_of_21_and_28_is_84_l550_550698

-- Define the conditions
def ratio : ℕ := 3 / 4
def first_number : ℕ := 21
def second_number : ℕ := 28 -- We found this from the condition 21 / x = 3 / 4

-- Definition of gcd
def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Definition of lcm using gcd
def lcm (a b : ℕ) : ℕ := (a * b) / gcd a b

-- Theorem to prove
theorem LCM_of_21_and_28_is_84 : lcm first_number second_number = 84 :=
by
  sorry

end LCM_of_21_and_28_is_84_l550_550698


namespace last_nonzero_digit_aperiodic_l550_550215

/-- Definition of the last nonzero digit of n! --/
def last_nonzero_digit (n : ℕ) : ℕ := -- implementation, e.g., taking n! modulo 10 after removing trailing zeros
  sorry

/-- The main theorem stating that the sequence of last nonzero digits of n! is aperiodic --/
theorem last_nonzero_digit_aperiodic :
  ¬ ∃ (T n₀ : ℕ), ∀ (n : ℕ), n ≥ n₀ → last_nonzero_digit (n + T) = last_nonzero_digit n :=
sorry

end last_nonzero_digit_aperiodic_l550_550215


namespace inequality_sum_l550_550196

theorem inequality_sum (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a ^ 2 + b ^ 2 + c ^ 2 = 1) :
  (a / (a ^ 3 + b * c) + b / (b ^ 3 + c * a) + c / (c ^ 3 + a * b)) > 3 :=
by
  sorry

end inequality_sum_l550_550196


namespace intersection_product_l550_550507

-- Define the parametric equations of line l
def line_l (t : ℝ) := (x, y) where
    x := (1 / 2) * t
    y := 3 + (sqrt 3 / 2) * t

-- Define the polar equation of curve C
def curve_C (θ : ℝ) : ℝ := 2 * sin θ

-- Definition of the ordinary equation of line l
def ordinary_eq_line_l (x y : ℝ) : Prop := sqrt 3 * x - y + 3 = 0

-- Definition of the rectangular coordinate equation of curve C
def rect_eq_curve_C (x y : ℝ) : Prop := x^2 + y^2 - 2 * y = 0

-- The main theorem to be proved
theorem intersection_product :
  ∀ P A B : (ℝ × ℝ), 
    P = (0, 3) → 
    (∃ t : ℝ, line_l t = A) →  
    (∃ t' : ℝ, line_l t' = B) → 
    (ordinary_eq_line_l 0 3) →
    (rect_eq_curve_C A.1 A.2) →
    (rect_eq_curve_C B.1 B.2) →
    |P.2 - A.2| * |P.2 - B.2| = 3 :=
begin
  sorry
end

end intersection_product_l550_550507


namespace find_f_9_over_2_l550_550922

variable {f : ℝ → ℝ}
variable {a b : ℝ}

-- Conditions
def domain_of_f (x : ℝ) : Prop := True  -- The domain of f is ℝ

def odd_f_of_x_plus_1 (x : ℝ) : Prop := f(x + 1) = -f(-x + 1)

def even_f_of_x_plus_2 (x : ℝ) : Prop := f(x + 2) = f(-x + 2)

def f_on_interval (x : ℝ) : Prop := x ∈ Icc (1:ℝ) 2 → f(x) = a * x^2 + b

def f_0_f_3_sum : Prop := f(0) + f(3) = 12

-- Goal
theorem find_f_9_over_2 :
  (∀ x, domain_of_f x) →
  (∀ x, odd_f_of_x_plus_1 x) →
  (∀ x, even_f_of_x_plus_2 x) →
  (∀ x, f_on_interval x) →
  (f_0_f_3_sum) →
  f(9 / 2) = 5 :=
by sorry

end find_f_9_over_2_l550_550922


namespace coordinates_of_P_l550_550120

variable (a : ℝ)

def y_coord (a : ℝ) : ℝ :=
  3 * a + 9

def x_coord (a : ℝ) : ℝ :=
  4 - a

theorem coordinates_of_P :
  (∃ a : ℝ, y_coord a = 0) → ∃ a : ℝ, (x_coord a, y_coord a) = (7, 0) :=
by
  -- The proof goes here
  sorry

end coordinates_of_P_l550_550120


namespace perpendicular_vectors_m_value_l550_550453

theorem perpendicular_vectors_m_value : 
  ∀ (m : ℝ), ((2 : ℝ) * (1 : ℝ) + (m * (1 / 2)) + (1 * 2) = 0) → m = -8 :=
by
  intro m
  intro h
  sorry

end perpendicular_vectors_m_value_l550_550453


namespace liking_sports_related_to_gender_with_99_percent_certainty_expectation_of_x_l550_550633

noncomputable def contingency_table_k_squared := 
  let n := 100
  let a := 8
  let b := 32
  let c := 32
  let d := 28
  n * (a * d - b * c) ^ 2 / ((a + b) * (c + d) * (a + c) * (b + d))

noncomputable def chi_square_test :=
  contingency_table_k_squared > 6.635

theorem liking_sports_related_to_gender_with_99_percent_certainty (h : chi_square_test) :
  True :=
by trivial

noncomputable def expectation_x :=
  let p0 := (28 / 45)
  let p1 := (16 / 45)
  let p2 := (1 / 45)
  (0 * p0 + 1 * p1 + 2 * p2 : ℝ)

theorem expectation_of_x : 
  expectation_x = 2 / 5 :=
by sorry

end liking_sports_related_to_gender_with_99_percent_certainty_expectation_of_x_l550_550633


namespace ping_pong_tournament_l550_550621

theorem ping_pong_tournament :
  ∃ n: ℕ, 
    (∃ m: ℕ, m ≥ 0 ∧ m ≤ 2 ∧ 2 * n + m = 29) ∧
    n = 14 ∧
    (n + 2 = 16) := 
by {
  sorry
}

end ping_pong_tournament_l550_550621


namespace integer_a_value_l550_550776

/-- For the given system of inequalities and provided sum condition, prove that the correct value of 'a' is 2 -/
theorem integer_a_value 
  (a : ℤ)
  (h1 : ∀ x : ℤ, 6*x + 3 > 3*(x + a))
  (h2 : ∀ x : ℤ, x / 2 - 1 ≤ 7 - (3 / 2) * x)
  (h3 : ∑ (x : ℤ) in (finset.filter (λ x, x > a - 1 ∧ x ≤ 4) (finset.range 10)), x = 9) :
  a = 2 := 
sorry

end integer_a_value_l550_550776


namespace cumulative_profit_exceeds_technical_renovation_expressions_for_A_n_B_n_l550_550322

noncomputable def A_n (n : ℕ) : ℝ :=
  490 * n - 10 * n^2

noncomputable def B_n (n : ℕ) : ℝ :=
  500 * n + 400 - 500 / 2^(n-1)

theorem cumulative_profit_exceeds_technical_renovation :
  ∀ n : ℕ, n ≥ 4 → B_n n > A_n n :=
by
  sorry  -- Proof goes here

theorem expressions_for_A_n_B_n (n : ℕ) :
  A_n n = 490 * n - 10 * n^2 ∧
  B_n n = 500 * n + 400 - 500 / 2^(n-1) :=
by
  sorry  -- Proof goes here

end cumulative_profit_exceeds_technical_renovation_expressions_for_A_n_B_n_l550_550322


namespace impossible_15_cents_l550_550781

theorem impossible_15_cents (a b c d : ℕ) (ha : a ≤ 4) (hb : b ≤ 4) (hc : c ≤ 4) (hd : d ≤ 4) (h : a + b + c + d = 4) : 
  1 * a + 5 * b + 10 * c + 25 * d ≠ 15 :=
by
  sorry

end impossible_15_cents_l550_550781


namespace crosses_in_grid_problem_l550_550874

theorem crosses_in_grid_problem :
  ∃ (placing_function : Fin 6 → (Fin 5 × Fin 5)),
  (∀ r : Fin 5, ∃ c : Fin 5, ∃ k : Fin 6, placing_function k = (r, c)) ∧
  (∀ c : Fin 5, ∃ r : Fin 5, ∃ k : Fin 6, placing_function k = (r, c)) ∧ 
  (number_of_ways_placing_crosses placing_function = 4200) := 
sorry

end crosses_in_grid_problem_l550_550874


namespace radio_price_and_total_items_l550_550714

theorem radio_price_and_total_items :
  ∃ (n : ℕ) (p : ℝ),
    (∀ (i : ℕ), (1 ≤ i ∧ i ≤ n) → (i = 1 ∨ ∃ (j : ℕ), i = j + 1 ∧ p = 1 + (j * 0.50))) ∧
    (n - 49 = 85) ∧
    (p = 43) ∧
    (n = 134) :=
by {
  sorry
}

end radio_price_and_total_items_l550_550714


namespace parabola_proof_l550_550441

noncomputable def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def parabola (p x y : ℝ) : Prop := y^2 = 2 * p * x

theorem parabola_proof (A : ℝ × ℝ) (l : ℝ) (B : ℝ × ℝ) (M : ℝ × ℝ) (p : ℝ) :
  A = (0, 2) →
  p > 0 →
  focus p = (p / 2, 0) →
  M.x = B.x →
  B = midpoint A (focus p) →
  M.y = 0 →
  A.x = 0 → A.y = 2 → (M.y = 0 → A.y = 2 → (M.x ≠ 0)) →
  parabola p B.1 B.2 →
  p = sqrt(2) :=
by
  intros
  sorry

end parabola_proof_l550_550441


namespace josh_ribbon_shortfall_l550_550189

-- Define the total amount of ribbon Josh has
def total_ribbon : ℝ := 18

-- Define the number of gifts
def num_gifts : ℕ := 6

-- Define the ribbon requirements for each gift
def ribbon_per_gift_wrapping : ℝ := 2
def ribbon_per_bow : ℝ := 1.5
def ribbon_per_tag : ℝ := 0.25
def ribbon_per_trim : ℝ := 0.5

-- Calculate the total ribbon required for all the tasks
def total_ribbon_needed : ℝ :=
  (ribbon_per_gift_wrapping * num_gifts) +
  (ribbon_per_bow * num_gifts) +
  (ribbon_per_tag * num_gifts) +
  (ribbon_per_trim * num_gifts)

-- Calculate the ribbon shortfall
def ribbon_shortfall : ℝ :=
  total_ribbon_needed - total_ribbon

-- Prove that Josh will be short by 7.5 yards of ribbon
theorem josh_ribbon_shortfall : ribbon_shortfall = 7.5 := by
  sorry

end josh_ribbon_shortfall_l550_550189


namespace janeth_balloons_count_l550_550900

-- Define the conditions
def bags_round_balloons : Nat := 5
def balloons_per_bag_round : Nat := 20
def bags_long_balloons : Nat := 4
def balloons_per_bag_long : Nat := 30
def burst_round_balloons : Nat := 5

-- Proof statement
theorem janeth_balloons_count:
  let total_round_balloons := bags_round_balloons * balloons_per_bag_round
  let total_long_balloons := bags_long_balloons * balloons_per_bag_long
  let total_balloons := total_round_balloons + total_long_balloons
  total_balloons - burst_round_balloons = 215 :=
by {
  sorry
}

end janeth_balloons_count_l550_550900


namespace find_range_of_a_l550_550133

-- Define the propositions
def p (a : ℝ) : Prop := ∀ x ∈ Icc 1 2, x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define the main theorem
theorem find_range_of_a (a : ℝ) (hp : p a) (hq : q a) : a ≤ -2 ∨ a = 1 := 
  sorry

end find_range_of_a_l550_550133


namespace Earth_angle_ACB_l550_550383

noncomputable def angle_ACB (lat_A lon_A lat_B lon_B : Real) : Real :=
  let cos_theta := Math.cos (lat_A * Real.pi / 180) * Math.cos (lat_B * Real.pi / 180) +
                   Math.sin (lat_A * Real.pi / 180) * Math.sin (lat_B * Real.pi / 180) *
                   Math.cos ((lon_A - lon_B) * Real.pi / 180)
  Real.arccos cos_theta * 180 / Real.pi

theorem Earth_angle_ACB : 
  angle_ACB 30 90 60 (-100) ≈ 89.7 := 
sorry

end Earth_angle_ACB_l550_550383


namespace circle_area_l550_550363

theorem circle_area (a b c : ℝ) (h : a = 4) (h2 : b = 4) (h3 : c = 3) :
    ∃ R : ℝ, R = 4 ∧ π * R^2 = 16 * π := by
  rw [h, h2, h3]
  use 4
  split
  · rfl
  · ring_nf
    ring

end circle_area_l550_550363


namespace max_management_fee_condition_l550_550697

theorem max_management_fee_condition :
  ∀ (x : ℝ), 
  (70 :ℝ)/100 * x ≤ 1 → 
  let price_increase := (70 * x / (100 - x)) in
  let new_sales_volume := (11.8 - x) in
  let sales_revenue := 70 + price_increase in
  let total_fee := (sales_revenue / 100 * x * new_sales_volume ) in
  total_fee ≥ 140000 →
  2 ≤ x ∧ x ≤ 10 :=
begin
  intros,
  sorry
end

end max_management_fee_condition_l550_550697
