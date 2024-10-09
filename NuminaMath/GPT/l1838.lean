import Mathlib

namespace arvin_fifth_day_running_distance_l1838_183871

theorem arvin_fifth_day_running_distance (total_km : ℕ) (first_day_km : ℕ) (increment : ℕ) (days : ℕ) 
  (h1 : total_km = 20) (h2 : first_day_km = 2) (h3 : increment = 1) (h4 : days = 5) : 
  first_day_km + (increment * (days - 1)) = 6 :=
by
  sorry

end arvin_fifth_day_running_distance_l1838_183871


namespace new_car_distance_l1838_183805

-- State the given conditions as a Lemma in Lean 4
theorem new_car_distance (d_old : ℝ) (d_new : ℝ) (h1 : d_old = 150) (h2 : d_new = d_old * 1.30) : d_new = 195 := 
by
  sorry

end new_car_distance_l1838_183805


namespace infinite_positive_integer_solutions_l1838_183802

theorem infinite_positive_integer_solutions :
  ∃ (k : ℕ), ∀ (n : ℕ), n > 24 → ∃ k > 24, k = n :=
sorry

end infinite_positive_integer_solutions_l1838_183802


namespace mail_distribution_l1838_183890

def pieces_per_block (total_pieces blocks : ℕ) : ℕ := total_pieces / blocks

theorem mail_distribution : pieces_per_block 192 4 = 48 := 
by { 
    -- Proof skipped
    sorry 
}

end mail_distribution_l1838_183890


namespace Mrs_Lara_Late_l1838_183847

noncomputable def required_speed (d t : ℝ) : ℝ := d / t

theorem Mrs_Lara_Late (d t : ℝ) (h1 : d = 50 * (t + 7 / 60)) (h2 : d = 70 * (t - 5 / 60)) :
  required_speed d t = 70 := by
  sorry

end Mrs_Lara_Late_l1838_183847


namespace Shaina_chocolate_l1838_183833

-- Definitions based on the conditions
def total_chocolate : ℚ := 72 / 7
def number_of_piles : ℚ := 6
def weight_per_pile : ℚ := total_chocolate / number_of_piles
def piles_given_to_Shaina : ℚ := 2

-- Theorem stating the problem's correct answer
theorem Shaina_chocolate :
  piles_given_to_Shaina * weight_per_pile = 24 / 7 :=
by
  sorry

end Shaina_chocolate_l1838_183833


namespace intersection_M_N_l1838_183857

def M (x : ℝ) : Prop := x^2 + 2*x - 15 < 0
def N (x : ℝ) : Prop := x^2 + 6*x - 7 ≥ 0

theorem intersection_M_N : {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | 1 ≤ x ∧ x < 3} :=
by
  sorry

end intersection_M_N_l1838_183857


namespace first_part_lent_years_l1838_183803

theorem first_part_lent_years (x n : ℕ) (total_sum second_sum : ℕ) (rate1 rate2 years2 : ℝ) :
  total_sum = 2743 →
  second_sum = 1688 →
  rate1 = 3 →
  rate2 = 5 →
  years2 = 3 →
  (x = total_sum - second_sum) →
  (x * n * rate1 / 100 = second_sum * rate2 * years2 / 100) →
  n = 8 :=
by
  sorry

end first_part_lent_years_l1838_183803


namespace circles_intersect_and_inequality_l1838_183820

variable {R r d : ℝ}

theorem circles_intersect_and_inequality (hR : R > r) (h_intersect: R - r < d ∧ d < R + r) : R - r < d ∧ d < R + r :=
by
  exact h_intersect

end circles_intersect_and_inequality_l1838_183820


namespace coordinates_of_A_l1838_183842

-- Definition of the distance function for any point (x, y)
def distance_to_x_axis (x y : ℝ) : ℝ := abs y
def distance_to_y_axis (x y : ℝ) : ℝ := abs x

-- Point A's coordinates
def point_is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- The main theorem to prove
theorem coordinates_of_A :
  ∃ (x y : ℝ), 
  point_is_in_fourth_quadrant x y ∧ 
  distance_to_x_axis x y = 3 ∧ 
  distance_to_y_axis x y = 6 ∧ 
  (x, y) = (6, -3) :=
by 
  sorry

end coordinates_of_A_l1838_183842


namespace increasing_function_geq_25_l1838_183868

theorem increasing_function_geq_25 {m : ℝ} 
  (h : ∀ x y : ℝ, x ≥ -2 ∧ x ≤ y → (4 * x^2 - m * x + 5) ≤ (4 * y^2 - m * y + 5)) :
  (4 * 1^2 - m * 1 + 5) ≥ 25 :=
by {
  -- Proof is omitted
  sorry
}

end increasing_function_geq_25_l1838_183868


namespace transformed_system_solution_l1838_183838

theorem transformed_system_solution 
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (h1 : a1 * 3 + b1 * 4 = c1)
  (h2 : a2 * 3 + b2 * 4 = c2) :
  (3 * a1 * 5 + 4 * b1 * 5 = 5 * c1) ∧ (3 * a2 * 5 + 4 * b2 * 5 = 5 * c2) :=
by 
  sorry

end transformed_system_solution_l1838_183838


namespace painting_time_eq_l1838_183825

theorem painting_time_eq (t : ℝ) :
  (1 / 6 + 1 / 8 + 1 / 12) * t = 1 ↔ t = 8 / 3 :=
by
  sorry

end painting_time_eq_l1838_183825


namespace monotonic_increase_interval_range_of_a_l1838_183876

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x + 2 * Real.exp x - a * x^2
def h (x : ℝ) : ℝ := x

theorem monotonic_increase_interval :
  ∃ I : Set ℝ, I = Set.Ioi 1 ∧ ∀ x ∈ I, ∀ y ∈ I, x ≤ y → f x ≤ f y := 
  sorry

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, (g x1 a - h x1) * (g x2 a - h x2) > 0) ↔ a ∈ Set.Iic 1 :=
  sorry

end monotonic_increase_interval_range_of_a_l1838_183876


namespace polynomial_sum_coeff_l1838_183887

-- Definitions for the polynomials given
def poly1 (d : ℤ) : ℤ := 15 * d^3 + 19 * d^2 + 17 * d + 18
def poly2 (d : ℤ) : ℤ := 3 * d^3 + 4 * d + 2

-- The main statement to prove
theorem polynomial_sum_coeff :
  let p := 18
  let q := 19
  let r := 21
  let s := 20
  p + q + r + s = 78 :=
by
  sorry

end polynomial_sum_coeff_l1838_183887


namespace find_quadratic_function_l1838_183828

def quadratic_function (c d : ℝ) (x : ℝ) : ℝ :=
  x^2 + c * x + d

theorem find_quadratic_function :
  ∃ c d, (∀ x, 
    (quadratic_function c d (quadratic_function c d x + 2 * x)) / (quadratic_function c d x) = 2 * x^2 + 1984 * x + 2024) ∧ 
    quadratic_function c d x = x^2 + 1982 * x + 21 :=
by
  sorry

end find_quadratic_function_l1838_183828


namespace fraction_value_l1838_183807

theorem fraction_value (a b c : ℕ) (h1 : a = 2200) (h2 : b = 2096) (h3 : c = 121) :
    (a - b)^2 / c = 89 := by
  sorry

end fraction_value_l1838_183807


namespace triangle_ABC_properties_l1838_183891

theorem triangle_ABC_properties
  (a b c : ℝ)
  (A B C : ℝ)
  (area_ABC : Real.sqrt 15 * 3 = 1/2 * b * c * Real.sin A)
  (cos_A : Real.cos A = -1/4)
  (b_minus_c : b - c = 2) :
  (a = 8 ∧ Real.sin C = Real.sqrt 15 / 8) ∧
  (Real.cos (2 * A + Real.pi / 6) = (Real.sqrt 15 - 7 * Real.sqrt 3) / 16) := by
  sorry

end triangle_ABC_properties_l1838_183891


namespace smallest_three_digit_pqr_l1838_183865

theorem smallest_three_digit_pqr (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  100 ≤ p * q^2 * r ∧ p * q^2 * r < 1000 → p * q^2 * r = 126 := 
sorry

end smallest_three_digit_pqr_l1838_183865


namespace weights_of_first_two_cats_l1838_183844

noncomputable def cats_weight_proof (W : ℝ) : Prop :=
  (∀ (w1 w2 : ℝ), w1 = W ∧ w2 = W ∧ (w1 + w2 + 14.7 + 9.3) / 4 = 12) → (W = 12)

theorem weights_of_first_two_cats (W : ℝ) :
  cats_weight_proof W :=
by
  sorry

end weights_of_first_two_cats_l1838_183844


namespace canoe_kayak_ratio_l1838_183874

-- Define the number of canoes and kayaks
variables (c k : ℕ)

-- Define the conditions
def rental_cost_eq : Prop := 15 * c + 18 * k = 405
def canoe_more_kayak_eq : Prop := c = k + 5

-- Statement to prove
theorem canoe_kayak_ratio (h1 : rental_cost_eq c k) (h2 : canoe_more_kayak_eq c k) : c / k = 3 / 2 :=
by sorry

end canoe_kayak_ratio_l1838_183874


namespace min_species_needed_l1838_183899

theorem min_species_needed (num_birds : ℕ) (h1 : num_birds = 2021)
  (h2 : ∀ (s : ℤ) (x y : ℕ), x ≠ y → (between_same_species : ℕ) → (h3 : between_same_species = y - x - 1) → between_same_species % 2 = 0) :
  ∃ (species : ℕ), num_birds ≤ 2 * species ∧ species = 1011 :=
by
  sorry

end min_species_needed_l1838_183899


namespace find_n_l1838_183810

theorem find_n (e n : ℕ) (h_lcm : Nat.lcm e n = 690) (h_n_not_div_3 : ¬ (3 ∣ n)) (h_e_not_div_2 : ¬ (2 ∣ e)) : n = 230 :=
by
  sorry

end find_n_l1838_183810


namespace max_touched_points_by_line_l1838_183897

noncomputable section

open Function

-- Definitions of the conditions
def coplanar_circles (circles : Set (Set ℝ)) : Prop :=
  ∀ c₁ c₂ : Set ℝ, c₁ ∈ circles → c₂ ∈ circles → c₁ ≠ c₂ → ∃ p : ℝ, p ∈ c₁ ∧ p ∈ c₂

def max_touched_points (line_circle : ℝ → ℝ) : ℕ :=
  2

-- The theorem statement that needs to be proven
theorem max_touched_points_by_line {circles : Set (Set ℝ)} (h_coplanar : coplanar_circles circles) :
  ∀ line : ℝ → ℝ, (∃ (c₁ c₂ c₃ : Set ℝ), c₁ ∈ circles ∧ c₂ ∈ circles ∧ c₃ ∈ circles ∧ c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₁ ≠ c₃) →
  ∃ (p : ℕ), p = 6 := 
sorry

end max_touched_points_by_line_l1838_183897


namespace sequence_a_2024_l1838_183879

theorem sequence_a_2024 (a : ℕ → ℝ) (h₀ : a 1 = 2) (h₁ : ∀ n : ℕ, n > 0 → a (n + 1) = 1 - 1 / a n) : a 2024 = 1 / 2 :=
by
  sorry

end sequence_a_2024_l1838_183879


namespace probability_first_three_cards_spades_l1838_183877

theorem probability_first_three_cards_spades :
  let num_spades : ℕ := 13
  let total_cards : ℕ := 52
  let prob_first_spade : ℚ := num_spades / total_cards
  let prob_second_spade_given_first : ℚ := (num_spades - 1) / (total_cards - 1)
  let prob_third_spade_given_first_two : ℚ := (num_spades - 2) / (total_cards - 2)
  let prob_all_three_spades : ℚ := prob_first_spade * prob_second_spade_given_first * prob_third_spade_given_first_two
  prob_all_three_spades = 33 / 2550 :=
by
  sorry

end probability_first_three_cards_spades_l1838_183877


namespace find_a_l1838_183814

theorem find_a :
  ∃ a : ℝ, (∀ t1 t2 : ℝ, t1 + t2 = -a ∧ t1 * t2 = -2017 ∧ 2 * t1 = 4) → a = 1006.5 :=
by
  sorry

end find_a_l1838_183814


namespace stddev_newData_l1838_183875

-- Definitions and conditions
def variance (data : List ℝ) : ℝ := sorry  -- Placeholder for variance definition
def stddev (data : List ℝ) : ℝ := sorry    -- Placeholder for standard deviation definition

-- Given data
def data : List ℝ := sorry                -- Placeholder for the data x_1, x_2, ..., x_8
def newData : List ℝ := data.map (λ x => 2 * x + 1)

-- Given condition
axiom variance_data : variance data = 16

-- Proof of the statement
theorem stddev_newData : stddev newData = 8 :=
by {
  sorry
}

end stddev_newData_l1838_183875


namespace find_x_l1838_183889

theorem find_x (a b x : ℝ) (h_a : a > 0) (h_b : b > 0) (h_x : x > 0)
  (s : ℝ) (h_s1 : s = (a ^ 2) ^ (4 * b)) (h_s2 : s = a ^ (2 * b) * x ^ (3 * b)) :
  x = a ^ 2 :=
sorry

end find_x_l1838_183889


namespace ratio_of_a_to_c_l1838_183853

theorem ratio_of_a_to_c (a b c d : ℚ) 
  (h1 : a / b = 5 / 4)
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 5) :
  a / c = 75 / 16 :=
by
  sorry

end ratio_of_a_to_c_l1838_183853


namespace total_games_l1838_183892

def joan_games_this_year : ℕ := 4
def joan_games_last_year : ℕ := 9

theorem total_games (this_year_games last_year_games : ℕ) 
    (h1 : this_year_games = joan_games_this_year) 
    (h2 : last_year_games = joan_games_last_year) : 
    this_year_games + last_year_games = 13 := 
by
  rw [h1, h2]
  exact rfl

end total_games_l1838_183892


namespace gcd_of_polynomials_l1838_183813

theorem gcd_of_polynomials (b : ℤ) (h : b % 1620 = 0) : Int.gcd (b^2 + 11 * b + 36) (b + 6) = 6 := 
by
  sorry

end gcd_of_polynomials_l1838_183813


namespace proof_problem_l1838_183821

noncomputable def A := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}
noncomputable def B := {(x, y) : ℝ × ℝ | y = x^2 + 1}

theorem proof_problem :
  ((1, 2) ∈ B) ∧
  (0 ∉ A) ∧
  ((0, 0) ∉ B) :=
by
  sorry

end proof_problem_l1838_183821


namespace correct_sampling_method_l1838_183863

-- Definitions based on conditions
def number_of_classes : ℕ := 16
def sampled_classes : ℕ := 2
def sampling_method := "Lottery then Stratified"

-- The theorem statement based on the proof problem
theorem correct_sampling_method :
  (number_of_classes = 16) ∧ (sampled_classes = 2) → (sampling_method = "Lottery then Stratified") :=
sorry

end correct_sampling_method_l1838_183863


namespace physics_marks_l1838_183845

theorem physics_marks (P C M : ℕ) 
  (h1 : (P + C + M) = 255)
  (h2 : (P + M) = 180)
  (h3 : (P + C) = 140) : 
  P = 65 :=
by
  sorry

end physics_marks_l1838_183845


namespace rhombus_area_correct_l1838_183878

/-- Define the rhombus area calculation in miles given the lengths of its diagonals -/
def scale := 250
def d1 := 6 * scale -- first diagonal in miles
def d2 := 12 * scale -- second diagonal in miles
def areaOfRhombus (d1 d2 : ℕ) : ℕ := (d1 * d2) / 2

theorem rhombus_area_correct :
  areaOfRhombus d1 d2 = 2250000 :=
by
  sorry

end rhombus_area_correct_l1838_183878


namespace sewer_runoff_capacity_l1838_183872

theorem sewer_runoff_capacity (gallons_per_hour : ℕ) (hours_per_day : ℕ) (days_till_overflow : ℕ)
  (h1 : gallons_per_hour = 1000)
  (h2 : hours_per_day = 24)
  (h3 : days_till_overflow = 10) :
  gallons_per_hour * hours_per_day * days_till_overflow = 240000 := 
by
  -- We'll use sorry here as the placeholder for the actual proof steps
  sorry

end sewer_runoff_capacity_l1838_183872


namespace exists_positive_integers_x_y_l1838_183827

theorem exists_positive_integers_x_y (x y : ℕ) : 0 < x ∧ 0 < y ∧ x^2 = y^2 + 2023 :=
  sorry

end exists_positive_integers_x_y_l1838_183827


namespace area_of_triangle_DEF_l1838_183888

theorem area_of_triangle_DEF :
  let D := (0, 2)
  let E := (6, 0)
  let F := (3, 8)
  let base1 := 6
  let height1 := 2
  let base2 := 3
  let height2 := 8
  let base3 := 3
  let height3 := 6
  let area_triangle_DE := 1 / 2 * (base1 * height1)
  let area_triangle_EF := 1 / 2 * (base2 * height2)
  let area_triangle_FD := 1 / 2 * (base3 * height3)
  let area_rectangle := 6 * 8
  ∃ area_def_triangle, 
  area_def_triangle = area_rectangle - (area_triangle_DE + area_triangle_EF + area_triangle_FD) 
  ∧ area_def_triangle = 21 :=
by 
  sorry

end area_of_triangle_DEF_l1838_183888


namespace max_length_AB_l1838_183824

theorem max_length_AB : ∀ t : ℝ, 0 ≤ t ∧ t ≤ 3 → ∃ M, M = 81 / 8 ∧ ∀ t, -2 * (t - 3/4)^2 + 81 / 8 = M :=
by sorry

end max_length_AB_l1838_183824


namespace total_legs_l1838_183896

theorem total_legs 
  (johnny_legs : ℕ := 2) 
  (son_legs : ℕ := 2) 
  (dog_legs_per_dog : ℕ := 4) 
  (number_of_dogs : ℕ := 2) :
  johnny_legs + son_legs + dog_legs_per_dog * number_of_dogs = 12 := 
sorry

end total_legs_l1838_183896


namespace stacy_height_proof_l1838_183826

noncomputable def height_last_year : ℕ := 50
noncomputable def brother_growth : ℕ := 1
noncomputable def stacy_growth : ℕ := brother_growth + 6
noncomputable def stacy_current_height : ℕ := height_last_year + stacy_growth

theorem stacy_height_proof : stacy_current_height = 57 := 
by
  sorry

end stacy_height_proof_l1838_183826


namespace solve_system_l1838_183866

theorem solve_system (x y z w : ℝ) :
  x - y + z - w = 2 ∧
  x^2 - y^2 + z^2 - w^2 = 6 ∧
  x^3 - y^3 + z^3 - w^3 = 20 ∧
  x^4 - y^4 + z^4 - w^4 = 60 ↔
  (x = 1 ∧ y = 2 ∧ z = 3 ∧ w = 0) ∨
  (x = 1 ∧ y = 0 ∧ z = 3 ∧ w = 2) ∨
  (x = 3 ∧ y = 2 ∧ z = 1 ∧ w = 0) ∨
  (x = 3 ∧ y = 0 ∧ z = 1 ∧ w = 2) :=
sorry

end solve_system_l1838_183866


namespace region_transformation_area_l1838_183864

-- Define the region T with area 15
def region_T : ℝ := 15

-- Define the transformation matrix
def matrix_M : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![ 3, 4 ],
  ![ 5, -2 ]
]

-- The determinant of the matrix
def det_matrix_M : ℝ := 3 * (-2) - 4 * 5

-- The proven target statement to show that after the transformation, the area of T' is 390
theorem region_transformation_area :
  ∃ (area_T' : ℝ), area_T' = |det_matrix_M| * region_T ∧ area_T' = 390 :=
by
  sorry

end region_transformation_area_l1838_183864


namespace min_value_of_sum_l1838_183894

theorem min_value_of_sum (x y : ℝ) (h1 : x + 4 * y = 2 * x * y) (h2 : 0 < x) (h3 : 0 < y) : 
  x + y ≥ 9 / 2 :=
sorry

end min_value_of_sum_l1838_183894


namespace avg_class_l1838_183854

-- Problem definitions
def total_students : ℕ := 40
def num_students_95 : ℕ := 8
def num_students_0 : ℕ := 5
def num_students_70 : ℕ := 10
def avg_remaining_students : ℝ := 50

-- Assuming we have these marks
def marks_95 : ℝ := 95
def marks_0 : ℝ := 0
def marks_70 : ℝ := 70

-- We need to prove that the total average is 57.75 given the above conditions
theorem avg_class (h1 : total_students = 40)
                  (h2 : num_students_95 = 8)
                  (h3 : num_students_0 = 5)
                  (h4 : num_students_70 = 10)
                  (h5 : avg_remaining_students = 50)
                  (h6 : marks_95 = 95)
                  (h7 : marks_0 = 0)
                  (h8 : marks_70 = 70) :
                  (8 * 95 + 5 * 0 + 10 * 70 + 50 * (40 - (8 + 5 + 10))) / 40 = 57.75 :=
by sorry

end avg_class_l1838_183854


namespace find_n_l1838_183858

theorem find_n (n k : ℕ) (b : ℝ) (h_n2 : n ≥ 2) (h_ab : b ≠ 0 ∧ k > 0) (h_a_eq : ∀ (a : ℝ), a = k^2 * b) :
  (∀ (S : ℕ → ℝ → ℝ), S 1 b + S 2 b = 0) →
  n = 2 * k + 1 := 
sorry

end find_n_l1838_183858


namespace conic_sections_of_equation_l1838_183831

theorem conic_sections_of_equation :
  (∀ x y : ℝ, y^6 - 6 * x^6 = 3 * y^2 - 8 → y^2 = 6 * x^2 ∨ y^2 = -6 * x^2 + 2) :=
sorry

end conic_sections_of_equation_l1838_183831


namespace sum_in_correct_range_l1838_183885

-- Define the mixed numbers
def mixed1 := 1 + 1/4
def mixed2 := 4 + 1/3
def mixed3 := 6 + 1/12

-- Their sum
def sumMixed := mixed1 + mixed2 + mixed3

-- Correct sum in mixed number form
def correctSum := 11 + 2/3

-- Range we need to check
def lowerBound := 11 + 1/2
def upperBound := 12

theorem sum_in_correct_range : sumMixed = correctSum ∧ lowerBound < correctSum ∧ correctSum < upperBound := by
  sorry

end sum_in_correct_range_l1838_183885


namespace distance_cycled_l1838_183815

variable (v t d : ℝ)

theorem distance_cycled (h1 : d = v * t)
                        (h2 : d = (v + 1) * (3 * t / 4))
                        (h3 : d = (v - 1) * (t + 3)) :
                        d = 36 :=
by
  sorry

end distance_cycled_l1838_183815


namespace fiona_prob_reaches_12_l1838_183846

/-- Lily pads are numbered from 0 to 15 -/
def is_valid_pad (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 15

/-- Predators are on lily pads 4 and 7 -/
def predator (n : ℕ) : Prop := n = 4 ∨ n = 7

/-- Fiona the frog's probability to hop to the next pad -/
def hop : ℚ := 1 / 2

/-- Fiona the frog's probability to jump 2 pads -/
def jump_two : ℚ := 1 / 2

/-- Probability that Fiona reaches pad 12 without landing on pads 4 or 7 is 1/32 -/
theorem fiona_prob_reaches_12 :
  ∀ p : ℕ, 
    (is_valid_pad p ∧ ¬ predator p ∧ (p = 12) ∧ 
    ((∀ k : ℕ, is_valid_pad k → ¬ predator k → k ≤ 3 → (hop ^ k) = 1 / 2) ∧
    hop * hop = 1 / 4 ∧ hop * jump_two = 1 / 8 ∧
    (jump_two * (hop * hop + jump_two)) = 1 / 4 → hop * 1 / 4 = 1 / 32)) := 
by intros; sorry

end fiona_prob_reaches_12_l1838_183846


namespace sheila_hourly_earnings_l1838_183823

def sheila_hours_per_day (day : String) : Nat :=
  if day = "Monday" ∨ day = "Wednesday" ∨ day = "Friday" then 8
  else if day = "Tuesday" ∨ day = "Thursday" then 6
  else 0

def sheila_weekly_hours : Nat :=
  sheila_hours_per_day "Monday" +
  sheila_hours_per_day "Tuesday" +
  sheila_hours_per_day "Wednesday" +
  sheila_hours_per_day "Thursday" +
  sheila_hours_per_day "Friday"

def sheila_weekly_earnings : Nat := 468

theorem sheila_hourly_earnings :
  sheila_weekly_earnings / sheila_weekly_hours = 13 :=
by
  sorry

end sheila_hourly_earnings_l1838_183823


namespace find_y_in_triangle_l1838_183837

theorem find_y_in_triangle (BAC ABC BCA : ℝ) (y : ℝ) (h1 : BAC = 90)
  (h2 : ABC = 2 * y) (h3 : BCA = y - 10) : y = 100 / 3 :=
by
  -- The proof will be left as sorry
  sorry

end find_y_in_triangle_l1838_183837


namespace daily_calories_burned_l1838_183860

def calories_per_pound : ℕ := 3500
def pounds_to_lose : ℕ := 5
def days : ℕ := 35
def total_calories := pounds_to_lose * calories_per_pound

theorem daily_calories_burned :
  (total_calories / days) = 500 := 
  by 
    -- calculation steps
    sorry

end daily_calories_burned_l1838_183860


namespace tricycle_count_l1838_183849

variables (b t : ℕ)

theorem tricycle_count :
  b + t = 7 ∧ 2 * b + 3 * t = 19 → t = 5 := by
  intro h
  sorry

end tricycle_count_l1838_183849


namespace find_x_l1838_183819

theorem find_x (x y z : ℕ) (h1 : x = y / 2) (h2 : y = z / 3) (h3 : z = 90) : x = 15 :=
by
  sorry

end find_x_l1838_183819


namespace dima_walking_speed_l1838_183867

def Dima_station_time := 18 * 60 -- in minutes
def Dima_actual_arrival := 17 * 60 + 5 -- in minutes
def car_speed := 60 -- in km/h
def early_arrival := 10 -- in minutes

def walking_speed (arrival_time actual_arrival car_speed early_arrival : ℕ) : ℕ :=
(car_speed * early_arrival / 60) * (60 / (arrival_time - actual_arrival - early_arrival))

theorem dima_walking_speed :
  walking_speed Dima_station_time Dima_actual_arrival car_speed early_arrival = 6 :=
sorry

end dima_walking_speed_l1838_183867


namespace value_of_a_plus_b_l1838_183881

variable (a b : ℝ)
variable (h1 : |a| = 5)
variable (h2 : |b| = 2)
variable (h3 : a < 0)
variable (h4 : b > 0)

theorem value_of_a_plus_b : a + b = -3 :=
by
  sorry

end value_of_a_plus_b_l1838_183881


namespace avg_speed_is_40_l1838_183856

noncomputable def average_speed (x : ℝ) : ℝ :=
  let time1 := x / 40
  let time2 := 2 * x / 20
  let total_time := time1 + time2
  let total_distance := 5 * x
  total_distance / total_time

theorem avg_speed_is_40 (x : ℝ) (hx : x > 0) :
  average_speed x = 40 := by
  sorry

end avg_speed_is_40_l1838_183856


namespace sum_of_roots_eq_neg2_l1838_183869

-- Define the quadratic equation.
def quadratic_equation (x : ℝ) : ℝ :=
  x^2 + 2 * x - 1

-- Define a predicate to express that x is a root of the quadratic equation.
def is_root (x : ℝ) : Prop :=
  quadratic_equation x = 0

-- Define the statement that the sum of the two roots of the quadratic equation equals -2.
theorem sum_of_roots_eq_neg2 (x1 x2 : ℝ) (h1 : is_root x1) (h2 : is_root x2) (h3 : x1 ≠ x2) :
  x1 + x2 = -2 :=
  sorry

end sum_of_roots_eq_neg2_l1838_183869


namespace find_value_l1838_183800

theorem find_value 
  (a b c d e f : ℚ)
  (h1 : a / b = 1 / 2)
  (h2 : c / d = 1 / 2)
  (h3 : e / f = 1 / 2)
  (h4 : 3 * b - 2 * d + f ≠ 0) : 
  (3 * a - 2 * c + e) / (3 * b - 2 * d + f) = 1 / 2 := 
by
  sorry

end find_value_l1838_183800


namespace find_y_l1838_183898

theorem find_y (x y : ℝ) (h1 : x ^ (3 * y) = 8) (h2 : x = 2) : y = 1 :=
by {
  sorry
}

end find_y_l1838_183898


namespace smallest_expression_value_l1838_183851

theorem smallest_expression_value (a b c : ℝ) (h₁ : b > c) (h₂ : c > 0) (h₃ : a ≠ 0) :
  (2 * a + b) ^ 2 + (b - c) ^ 2 + (c - 2 * a) ^ 2 ≥ (4 / 3) * b ^ 2 :=
by
  sorry

end smallest_expression_value_l1838_183851


namespace kerosene_cost_l1838_183861

/-- In a market, a dozen eggs cost as much as a pound of rice, and a half-liter of kerosene 
costs as much as 8 eggs. If the cost of each pound of rice is $0.33, then a liter of kerosene costs 44 cents. --/
theorem kerosene_cost : 
  let egg_cost := 0.33 / 12
  let rice_cost := 0.33
  let half_liter_kerosene_cost := 8 * egg_cost
  rice_cost = 0.33 → 1 * ((2 * half_liter_kerosene_cost) * 100) = 44 := 
by
  intros egg_cost rice_cost half_liter_kerosene_cost h_rice_cost
  sorry

end kerosene_cost_l1838_183861


namespace sum_of_primes_lt_20_eq_77_l1838_183870

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l1838_183870


namespace length_of_angle_bisector_l1838_183830

theorem length_of_angle_bisector (AB AC : ℝ) (angleBAC : ℝ) (AD : ℝ) :
  AB = 6 → AC = 3 → angleBAC = 60 → AD = 2 * Real.sqrt 3 :=
by
  intro hAB hAC hAngleBAC
  -- Consider adding proof steps here in the future
  sorry

end length_of_angle_bisector_l1838_183830


namespace total_distance_travelled_l1838_183811

-- Definitions and propositions
def distance_first_hour : ℝ := 15
def distance_second_hour : ℝ := 18
def distance_third_hour : ℝ := 1.25 * distance_second_hour

-- Conditions based on the problem
axiom second_hour_distance : distance_second_hour = 18
axiom second_hour_20_percent_more : distance_second_hour = 1.2 * distance_first_hour
axiom third_hour_25_percent_more : distance_third_hour = 1.25 * distance_second_hour

-- Proof of the total distance James traveled
theorem total_distance_travelled : 
  distance_first_hour + distance_second_hour + distance_third_hour = 55.5 :=
by
  sorry

end total_distance_travelled_l1838_183811


namespace middle_dimension_of_crate_l1838_183862

theorem middle_dimension_of_crate (middle_dimension : ℝ) : 
    (∀ r : ℝ, r = 5 → ∃ w h l : ℝ, w = 5 ∧ h = 12 ∧ l = middle_dimension ∧
        (diameter = 2 * r ∧ diameter ≤ middle_dimension ∧ h ≥ 12)) → 
    middle_dimension = 10 :=
by
  sorry

end middle_dimension_of_crate_l1838_183862


namespace non_working_games_count_l1838_183840

-- Definitions based on conditions
def totalGames : ℕ := 16
def pricePerGame : ℕ := 7
def totalEarnings : ℕ := 56

-- Statement to prove
theorem non_working_games_count : 
  totalGames - (totalEarnings / pricePerGame) = 8 :=
by
  sorry

end non_working_games_count_l1838_183840


namespace negation_proposition_l1838_183880

theorem negation_proposition:
  ¬(∃ x : ℝ, x^2 - x + 1 > 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≤ 0 :=
by
  sorry -- Proof not required as per instructions

end negation_proposition_l1838_183880


namespace non_monotonic_m_range_l1838_183806

theorem non_monotonic_m_range (m : ℝ) :
  (∃ x ∈ Set.Ioo (-1 : ℝ) 2, (3 * x^2 + 2 * x + m = 0)) →
  m ∈ Set.Ioo (-16 : ℝ) (1/3 : ℝ) :=
sorry

end non_monotonic_m_range_l1838_183806


namespace max_sum_squares_of_sides_l1838_183886

theorem max_sum_squares_of_sides
  (a : ℝ) (α : ℝ) 
  (hα1 : 0 < α) (hα2 : α < Real.pi / 2) : 
  ∃ b c : ℝ, b^2 + c^2 = a^2 / (1 - Real.cos α) := 
sorry

end max_sum_squares_of_sides_l1838_183886


namespace solve_for_x_l1838_183808

theorem solve_for_x (x y : ℝ) : 3 * x + 4 * y = 5 → x = (5 - 4 * y) / 3 :=
by
  intro h
  sorry

end solve_for_x_l1838_183808


namespace commercial_break_duration_l1838_183809

theorem commercial_break_duration (n1 n2 m1 m2 : ℕ) (h1 : n1 = 3) (h2 : m1 = 5) (h3 : n2 = 11) (h4 : m2 = 2) :
  n1 * m1 + n2 * m2 = 37 :=
by
  -- Here, in a real proof, we would substitute and show the calculations.
  sorry

end commercial_break_duration_l1838_183809


namespace right_triangle_circle_area_l1838_183818

/-- 
Given a right triangle ABC with legs AB = 6 cm and BC = 8 cm,
E is the midpoint of AB and D is the midpoint of AC.
A circle passes through points E and D and touches the hypotenuse AC.
Prove that the area of this circle is 100 * pi / 9 cm^2.
-/
theorem right_triangle_circle_area :
  ∃ (r : ℝ), 
  let AB := 6
  let BC := 8
  let AC := Real.sqrt (AB^2 + BC^2)
  let E := (AB / 2)
  let D := (AC / 2)
  let radius := (AC * (BC / 2) / AB)
  r = radius * radius * Real.pi ∧
  r = (100 * Real.pi / 9) := sorry

end right_triangle_circle_area_l1838_183818


namespace min_selling_price_l1838_183834

-- Average sales per month
def avg_sales := 50

-- Cost per refrigerator
def cost_per_fridge := 1200

-- Shipping fee per refrigerator
def shipping_fee_per_fridge := 20

-- Monthly storefront fee
def monthly_storefront_fee := 10000

-- Monthly repair costs
def monthly_repair_costs := 5000

-- Profit margin requirement
def profit_margin := 0.2

-- The minimum selling price for the shop to maintain at least 20% profit margin
theorem min_selling_price 
  (avg_sales : ℕ) 
  (cost_per_fridge : ℕ) 
  (shipping_fee_per_fridge : ℕ) 
  (monthly_storefront_fee : ℕ) 
  (monthly_repair_costs : ℕ) 
  (profit_margin : ℝ) : 
  ∃ x : ℝ, 
    (50 * x - ((cost_per_fridge + shipping_fee_per_fridge) * avg_sales + monthly_storefront_fee + monthly_repair_costs)) 
    ≥ (cost_per_fridge + shipping_fee_per_fridge) * avg_sales + monthly_storefront_fee + monthly_repair_costs * profit_margin 
    → x ≥ 1824 :=
by 
  sorry

end min_selling_price_l1838_183834


namespace find_a3_l1838_183817

noncomputable def geometric_seq (a : ℕ → ℕ) (q : ℕ) : Prop :=
∀ n, a (n+1) = a n * q

theorem find_a3 (a : ℕ → ℕ) (q : ℕ) (h_geom : geometric_seq a q) (hq : q > 1)
  (h1 : a 4 - a 0 = 15) (h2 : a 3 - a 1 = 6) :
  a 2 = 4 :=
by
  sorry

end find_a3_l1838_183817


namespace six_digit_numbers_l1838_183848

theorem six_digit_numbers :
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
  sorry

end six_digit_numbers_l1838_183848


namespace asia_fraction_correct_l1838_183841

-- Define the problem conditions
def fraction_NA (P : ℕ) : ℚ := 1/3 * P
def fraction_Europe (P : ℕ) : ℚ := 1/8 * P
def fraction_Africa (P : ℕ) : ℚ := 1/5 * P
def others : ℕ := 42
def total_passengers : ℕ := 240

-- Define the target fraction for Asia
def fraction_Asia (P: ℕ) : ℚ := 17 / 120

-- Theorem: the fraction of the passengers from Asia equals 17/120
theorem asia_fraction_correct : ∀ (P : ℕ), 
  P = total_passengers →
  fraction_NA P + fraction_Europe P + fraction_Africa P + fraction_Asia P * P + others = P →
  fraction_Asia P = 17 / 120 := 
by sorry

end asia_fraction_correct_l1838_183841


namespace solve_fractions_in_integers_l1838_183859

theorem solve_fractions_in_integers :
  ∀ (a b c : ℤ), (1 / a + 1 / b + 1 / c = 1) ↔
  (a = 3 ∧ b = 3 ∧ c = 3) ∨
  (a = 2 ∧ b = 3 ∧ c = 6) ∨
  (a = 2 ∧ b = 4 ∧ c = 4) ∨
  (a = 1 ∧ ∃ t : ℤ, b = t ∧ c = -t) :=
by {
  sorry
}

end solve_fractions_in_integers_l1838_183859


namespace comp_inter_empty_l1838_183832

section
variable {α : Type*} [DecidableEq α]
variable (I M N : Set α)
variable (a b c d e : α)
variable (hI : I = {a, b, c, d, e})
variable (hM : M = {a, c, d})
variable (hN : N = {b, d, e})

theorem comp_inter_empty : 
  (I \ M) ∩ (I \ N) = ∅ :=
by sorry
end

end comp_inter_empty_l1838_183832


namespace reducible_fraction_implies_divisibility_l1838_183812

theorem reducible_fraction_implies_divisibility
  (a b c d l k : ℤ)
  (m n : ℤ)
  (h1 : a * l + b = k * m)
  (h2 : c * l + d = k * n)
  : k ∣ (a * d - b * c) :=
by
  sorry

end reducible_fraction_implies_divisibility_l1838_183812


namespace value_of_expression_l1838_183855

theorem value_of_expression (a b c : ℚ) (h1 : a * b * c < 0) (h2 : a + b + c = 0) :
    (a - b - c) / |a| + (b - c - a) / |b| + (c - a - b) / |c| = 2 :=
by
  sorry

end value_of_expression_l1838_183855


namespace gcd_m_pow_5_plus_125_m_plus_3_l1838_183839

theorem gcd_m_pow_5_plus_125_m_plus_3 (m : ℕ) (h: m > 16) : 
  Nat.gcd (m^5 + 125) (m + 3) = Nat.gcd 27 (m + 3) :=
by
  -- Proof will be provided here
  sorry

end gcd_m_pow_5_plus_125_m_plus_3_l1838_183839


namespace min_sum_xy_l1838_183829

theorem min_sum_xy (x y : ℕ) (hx : x ≠ y) (hcond : ↑(1 / x) + ↑(1 / y) = 1 / 15) : x + y = 64 :=
sorry

end min_sum_xy_l1838_183829


namespace correct_misread_number_l1838_183850

theorem correct_misread_number (s : List ℕ) (wrong_avg correct_avg n wrong_num correct_num : ℕ) 
  (h1 : s.length = 10) 
  (h2 : (s.sum) / n = wrong_avg) 
  (h3 : wrong_num = 26) 
  (h4 : correct_avg = 16) 
  (h5 : n = 10) 
  : correct_num = 36 :=
sorry

end correct_misread_number_l1838_183850


namespace count_integer_values_of_x_l1838_183893

theorem count_integer_values_of_x (x : ℕ) (h : ⌈Real.sqrt x⌉ = 12) : ∃ (n : ℕ), n = 23 :=
by
  sorry

end count_integer_values_of_x_l1838_183893


namespace avg_remaining_two_l1838_183822

theorem avg_remaining_two (avg5 avg3 : ℝ) (h1 : avg5 = 12) (h2 : avg3 = 4) : (5 * avg5 - 3 * avg3) / 2 = 24 :=
by sorry

end avg_remaining_two_l1838_183822


namespace valuing_fraction_l1838_183843

variable {x y : ℚ}

theorem valuing_fraction (h : x / y = 1 / 2) : (x - y) / (x + y) = -1 / 3 :=
by
  sorry

end valuing_fraction_l1838_183843


namespace markers_leftover_l1838_183801

theorem markers_leftover :
  let total_markers := 154
  let num_packages := 13
  total_markers % num_packages = 11 :=
by
  sorry

end markers_leftover_l1838_183801


namespace pokemon_card_cost_l1838_183873

theorem pokemon_card_cost 
  (football_cost : ℝ)
  (num_football_packs : ℕ) 
  (baseball_cost : ℝ) 
  (total_spent : ℝ) 
  (h_football : football_cost = 2.73)
  (h_num_football_packs : num_football_packs = 2)
  (h_baseball : baseball_cost = 8.95)
  (h_total : total_spent = 18.42) :
  (total_spent - (num_football_packs * football_cost + baseball_cost) = 4.01) :=
by
  -- Proof goes here
  sorry

end pokemon_card_cost_l1838_183873


namespace coffee_y_ratio_is_1_to_5_l1838_183852

-- Define the conditions
variables {p v x y : Type}
variables (p_x p_y v_x v_y : ℕ) -- Coffee amounts in lbs
variables (total_p total_v : ℕ) -- Total amounts of p and v

-- Definitions based on conditions
def coffee_amounts_initial (total_p total_v : ℕ) : Prop :=
  total_p = 24 ∧ total_v = 25

def coffee_x_conditions (p_x v_x : ℕ) : Prop :=
  p_x = 20 ∧ 4 * v_x = p_x

def coffee_y_conditions (p_y v_y total_p total_v : ℕ) : Prop :=
  p_y = total_p - 20 ∧ v_y = total_v - (20 / 4)

-- Statement to prove
theorem coffee_y_ratio_is_1_to_5 {total_p total_v : ℕ}
  (hc1 : coffee_amounts_initial total_p total_v)
  (hc2 : coffee_x_conditions 20 5)
  (hc3 : coffee_y_conditions 4 20 total_p total_v) : 
  (4 / 20 = 1 / 5) :=
sorry

end coffee_y_ratio_is_1_to_5_l1838_183852


namespace cube_split_l1838_183882

theorem cube_split (m : ℕ) (h1 : m > 1)
  (h2 : ∃ (p : ℕ), (p = (m - 1) * (m^2 + m + 1) ∨ p = (m - 1)^2 ∨ p = (m - 1)^2 + 2) ∧ p = 2017) :
  m = 46 :=
by {
    sorry
}

end cube_split_l1838_183882


namespace plant_height_after_year_l1838_183816

theorem plant_height_after_year (current_height : ℝ) (monthly_growth : ℝ) (months_in_year : ℕ) (total_growth : ℝ)
  (h1 : current_height = 20)
  (h2 : monthly_growth = 5)
  (h3 : months_in_year = 12)
  (h4 : total_growth = monthly_growth * months_in_year) :
  current_height + total_growth = 80 :=
sorry

end plant_height_after_year_l1838_183816


namespace find_value_of_A_l1838_183836

-- Define the conditions
variable (A : ℕ)
variable (divisor : ℕ := 9)
variable (quotient : ℕ := 2)
variable (remainder : ℕ := 6)

-- The main statement of the proof problem
theorem find_value_of_A (h : A = quotient * divisor + remainder) : A = 24 :=
by
  -- Proof would go here
  sorry

end find_value_of_A_l1838_183836


namespace calculate_p_l1838_183835

variable (m n : ℤ) (p : ℤ)

theorem calculate_p (h1 : 3 * m - 2 * n = -2) (h2 : p = 3 * (m + 405) - 2 * (n - 405)) : p = 2023 := 
  sorry

end calculate_p_l1838_183835


namespace sum_of_squares_500_l1838_183884

theorem sum_of_squares_500 : (Finset.range 500).sum (λ x => (x + 1) ^ 2) = 41841791750 := by
  sorry

end sum_of_squares_500_l1838_183884


namespace solution_set_f_gt_0_l1838_183883

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 2*x - 3 else - (x^2 - 2*x - 3)

theorem solution_set_f_gt_0 :
  {x : ℝ | f x > 0} = {x : ℝ | x > 3 ∨ (-3 < x ∧ x < 0)} :=
by
  sorry

end solution_set_f_gt_0_l1838_183883


namespace b_minus_a_less_zero_l1838_183895

-- Given conditions
variables {a b : ℝ}

-- Define the condition
def a_greater_b (a b : ℝ) : Prop := a > b

-- Lean 4 proof problem statement
theorem b_minus_a_less_zero (a b : ℝ) (h : a_greater_b a b) : b - a < 0 := 
sorry

end b_minus_a_less_zero_l1838_183895


namespace sufficient_but_not_necessary_l1838_183804

variable (a : ℝ)

theorem sufficient_but_not_necessary : (a = 1 → |a| = 1) ∧ (|a| = 1 → a = 1 → False) :=
by
  sorry

end sufficient_but_not_necessary_l1838_183804
