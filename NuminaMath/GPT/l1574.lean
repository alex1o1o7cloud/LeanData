import Mathlib

namespace NUMINAMATH_GPT_max_students_gave_away_balls_more_l1574_157466

theorem max_students_gave_away_balls_more (N : ℕ) (hN : N ≤ 13) : 
  ∃(students : ℕ), students = 27 ∧ (students = 27 ∧ N ≤ students - N) :=
by
  sorry

end NUMINAMATH_GPT_max_students_gave_away_balls_more_l1574_157466


namespace NUMINAMATH_GPT_student_count_l1574_157405

noncomputable def numberOfStudents (decreaseInAverageWeight totalWeightDecrease : ℕ) : ℕ :=
  totalWeightDecrease / decreaseInAverageWeight

theorem student_count 
  (decreaseInAverageWeight : ℕ)
  (totalWeightDecrease : ℕ)
  (condition_avg_weight_decrease : decreaseInAverageWeight = 4)
  (condition_weight_difference : totalWeightDecrease = 92 - 72) :
  numberOfStudents decreaseInAverageWeight totalWeightDecrease = 5 := by 
  -- We are not providing the proof details as per the instruction
  sorry

end NUMINAMATH_GPT_student_count_l1574_157405


namespace NUMINAMATH_GPT_isosceles_triangle_inequality_degenerate_triangle_a_zero_degenerate_triangle_double_b_l1574_157416

section isosceles_triangle

variables (a b k : ℝ)

/-- Prove the inequality for an isosceles triangle -/
theorem isosceles_triangle_inequality (h_perimeter : k = a + 2 * b) (ha_pos : a > 0) :
  k / 2 < a + b ∧ a + b < 3 * k / 4 :=
sorry

/-- Prove the inequality for degenerate triangle with a = 0 -/
theorem degenerate_triangle_a_zero (b k : ℝ) (h_perimeter : k = 2 * b) :
  k / 2 ≤ b ∧ b < 3 * k / 4 :=
sorry

/-- Prove the inequality for degenerate triangle with a = 2b -/
theorem degenerate_triangle_double_b (b k : ℝ) (h_perimeter : k = 4 * b) :
  k / 2 < b ∧ b ≤ 3 * k / 4 :=
sorry

end isosceles_triangle

end NUMINAMATH_GPT_isosceles_triangle_inequality_degenerate_triangle_a_zero_degenerate_triangle_double_b_l1574_157416


namespace NUMINAMATH_GPT_evaluate_box_2_neg1_0_l1574_157406

def box (a b c : ℤ) : ℚ := a^b - b^c + c^a

theorem evaluate_box_2_neg1_0 : box 2 (-1) 0 = -1/2 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_box_2_neg1_0_l1574_157406


namespace NUMINAMATH_GPT_inequality_transformation_incorrect_l1574_157479

theorem inequality_transformation_incorrect (a b : ℝ) (h : a > b) : (3 - a > 3 - b) -> false :=
by
  intros h1
  simp at h1
  sorry

end NUMINAMATH_GPT_inequality_transformation_incorrect_l1574_157479


namespace NUMINAMATH_GPT_parking_space_area_l1574_157460

theorem parking_space_area (L W : ℕ) (h1 : L = 9) (h2 : L + 2 * W = 37) : L * W = 126 := by
  sorry

end NUMINAMATH_GPT_parking_space_area_l1574_157460


namespace NUMINAMATH_GPT_people_going_to_movie_l1574_157429

variable (people_per_car : ℕ) (number_of_cars : ℕ)

theorem people_going_to_movie (h1 : people_per_car = 6) (h2 : number_of_cars = 18) : 
    (people_per_car * number_of_cars) = 108 := 
by
  sorry

end NUMINAMATH_GPT_people_going_to_movie_l1574_157429


namespace NUMINAMATH_GPT_union_A_B_l1574_157499

open Set

def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x | x < 1}

theorem union_A_B : A ∪ B = {x | x < 2} := 
by sorry

end NUMINAMATH_GPT_union_A_B_l1574_157499


namespace NUMINAMATH_GPT_number_of_ways_to_choose_roles_l1574_157476

-- Define the problem setup
def friends := Fin 6
def cooks (maria : Fin 1) := {f : Fin 6 | f ≠ maria}
def cleaners (cooks : Fin 6 → Prop) := {f : Fin 6 | ¬cooks f}

-- The number of ways to select one additional cook from the remaining friends
def chooseSecondCook : ℕ := Nat.choose 5 1  -- 5 ways

-- The number of ways to select two cleaners from the remaining friends
def chooseCleaners : ℕ := Nat.choose 4 2  -- 6 ways

-- The final number of ways to choose roles
theorem number_of_ways_to_choose_roles (maria : Fin 1) : 
  let total_ways : ℕ := chooseSecondCook * chooseCleaners
  total_ways = 30 := sorry

end NUMINAMATH_GPT_number_of_ways_to_choose_roles_l1574_157476


namespace NUMINAMATH_GPT_find_a22_l1574_157495

-- Definitions and conditions
noncomputable def seq (n : ℕ) : ℝ := if n = 0 then 0 else sorry

axiom seq_conditions
  (a : ℕ → ℝ)
  (h1 : ∀ n ∈ Finset.range 99, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) : True

theorem find_a22 (a : ℕ → ℝ)
  (h1 : ∀ n ∈ Finset.range 99, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 :=
sorry

end NUMINAMATH_GPT_find_a22_l1574_157495


namespace NUMINAMATH_GPT_dan_spent_more_on_chocolates_l1574_157458

def price_candy_bar : ℝ := 4
def number_of_candy_bars : ℕ := 5
def candy_discount : ℝ := 0.20
def discount_threshold : ℕ := 3
def price_chocolate : ℝ := 6
def number_of_chocolates : ℕ := 4
def chocolate_tax_rate : ℝ := 0.05

def candy_cost_total : ℝ :=
  let cost_without_discount := number_of_candy_bars * price_candy_bar
  if number_of_candy_bars >= discount_threshold
  then cost_without_discount * (1 - candy_discount)
  else cost_without_discount

def chocolate_cost_total : ℝ :=
  let cost_without_tax := number_of_chocolates * price_chocolate
  cost_without_tax * (1 + chocolate_tax_rate)

def difference_in_spending : ℝ :=
  chocolate_cost_total - candy_cost_total

theorem dan_spent_more_on_chocolates :
  difference_in_spending = 9.20 :=
by
  sorry

end NUMINAMATH_GPT_dan_spent_more_on_chocolates_l1574_157458


namespace NUMINAMATH_GPT_proof_problem_l1574_157486

-- Definitions of the conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) + f x = 0

def decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f y < f x

def satisfies_neq_point (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f a = 0

-- Main problem statement to prove (with conditions)
theorem proof_problem (f : ℝ → ℝ)
  (Hodd : odd_function f)
  (Hdec : decreasing_on f {y | 0 < y})
  (Hpt : satisfies_neq_point f (-2)) :
  {x : ℝ | (x - 1) * f x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 1 < x ∧ x < 2} :=
sorry

end NUMINAMATH_GPT_proof_problem_l1574_157486


namespace NUMINAMATH_GPT_solve_frac_eq_l1574_157411

-- Define the fractional function
def frac_eq (x : ℝ) : Prop := (x + 2) / (x - 1) = 0

-- State the theorem
theorem solve_frac_eq : frac_eq (-2) :=
by
  unfold frac_eq
  -- Use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_solve_frac_eq_l1574_157411


namespace NUMINAMATH_GPT_train_length_is_correct_l1574_157449

variable (speed_km_hr : ℕ) (time_sec : ℕ)
def convert_speed (speed_km_hr : ℕ) : ℚ :=
  (speed_km_hr * 1000 : ℚ) / 3600

noncomputable def length_of_train (speed_km_hr time_sec : ℕ) : ℚ :=
  convert_speed speed_km_hr * time_sec

theorem train_length_is_correct (speed_km_hr : ℕ) (time_sec : ℕ) (h₁ : speed_km_hr = 300) (h₂ : time_sec = 33) :
  length_of_train speed_km_hr time_sec = 2750 := by
  sorry

end NUMINAMATH_GPT_train_length_is_correct_l1574_157449


namespace NUMINAMATH_GPT_total_area_of_colored_paper_l1574_157481

-- Definitions
def num_pieces : ℝ := 3.2
def side_length : ℝ := 8.5

-- Theorem statement
theorem total_area_of_colored_paper : 
  let area_one_piece := side_length * side_length
  let total_area := area_one_piece * num_pieces
  total_area = 231.2 := by
  sorry

end NUMINAMATH_GPT_total_area_of_colored_paper_l1574_157481


namespace NUMINAMATH_GPT_spherical_to_rectangular_conversion_l1574_157497

/-- Convert a point in spherical coordinates to rectangular coordinates given specific angles and distance -/
theorem spherical_to_rectangular_conversion :
  ∀ (ρ θ φ : ℝ) (x y z : ℝ), 
  ρ = 15 → θ = 225 * (Real.pi / 180) → φ = 45 * (Real.pi / 180) →
  x = ρ * Real.sin φ * Real.cos θ → y = ρ * Real.sin φ * Real.sin θ → z = ρ * Real.cos φ →
  x = -15 / 2 ∧ y = -15 / 2 ∧ z = 15 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_conversion_l1574_157497


namespace NUMINAMATH_GPT_initial_coloring_books_l1574_157453

theorem initial_coloring_books
  (x : ℝ)
  (h1 : x - 20 = 80 / 4) :
  x = 40 :=
by
  sorry

end NUMINAMATH_GPT_initial_coloring_books_l1574_157453


namespace NUMINAMATH_GPT_value_of_q_l1574_157454

theorem value_of_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1/p + 1/q = 1) (h4 : p * q = 12) : q = 6 + 2 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_q_l1574_157454


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l1574_157428

-- Define the roots of the polynomial and Vieta's conditions
variables {p q r : ℝ}

-- Given conditions from Vieta's formulas
def vieta_conditions (p q r : ℝ) : Prop :=
  p + q + r = 7 / 3 ∧
  p * q + p * r + q * r = 2 / 3 ∧
  p * q * r = 4 / 3

-- Statement that sum of squares of roots equals to 37/9 given Vieta's conditions
theorem sum_of_squares_of_roots 
  (h : vieta_conditions p q r) : 
  p^2 + q^2 + r^2 = 37 / 9 := 
sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l1574_157428


namespace NUMINAMATH_GPT_birds_landed_l1574_157465

theorem birds_landed (original_birds total_birds : ℕ) (h : original_birds = 12) (h2 : total_birds = 20) :
  total_birds - original_birds = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_birds_landed_l1574_157465


namespace NUMINAMATH_GPT_no_third_quadrant_l1574_157451

theorem no_third_quadrant {a b : ℝ} (h1 : 0 < a) (h2 : a < 1) (h3 : -1 < b) : ∀ x y : ℝ, (y = a^x + b) → ¬ (x < 0 ∧ y < 0) :=
by
  intro x y h
  sorry

end NUMINAMATH_GPT_no_third_quadrant_l1574_157451


namespace NUMINAMATH_GPT_cone_to_sphere_surface_area_ratio_l1574_157435

noncomputable def sphere_radius (r : ℝ) := r
noncomputable def cone_height (r : ℝ) := 3 * r
noncomputable def side_length_of_triangle (r : ℝ) := 2 * Real.sqrt 3 * r
noncomputable def surface_area_of_sphere (r : ℝ) := 4 * Real.pi * r^2
noncomputable def surface_area_of_cone (r : ℝ) := 9 * Real.pi * r^2
noncomputable def ratio_of_areas (cone_surface : ℝ) (sphere_surface : ℝ) := cone_surface / sphere_surface

theorem cone_to_sphere_surface_area_ratio (r : ℝ) :
    ratio_of_areas (surface_area_of_cone r) (surface_area_of_sphere r) = 9 / 4 := sorry

end NUMINAMATH_GPT_cone_to_sphere_surface_area_ratio_l1574_157435


namespace NUMINAMATH_GPT_solve_for_x_l1574_157403

theorem solve_for_x (x y : ℝ) (h : x / (x - 1) = (y^2 + 3 * y - 5) / (y^2 + 3 * y - 7)) :
  x = (y^2 + 3 * y - 5) / 2 :=
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1574_157403


namespace NUMINAMATH_GPT_find_pairs_l1574_157418

theorem find_pairs (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n)
  (h3 : (m^2 - n) ∣ (m + n^2)) (h4 : (n^2 - m) ∣ (n + m^2)) :
  (m = 2 ∧ n = 2) ∨ (m = 3 ∧ n = 3) ∨ (m = 1 ∧ n = 2) ∨ (m = 2 ∧ n = 1) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 3) := by
  sorry

end NUMINAMATH_GPT_find_pairs_l1574_157418


namespace NUMINAMATH_GPT_find_parcera_triples_l1574_157496

noncomputable def is_prime (n : ℕ) : Prop := sorry
noncomputable def parcera_triple (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧
  p ∣ q^2 - 4 ∧ q ∣ r^2 - 4 ∧ r ∣ p^2 - 4

theorem find_parcera_triples : 
  {t : ℕ × ℕ × ℕ | parcera_triple t.1 t.2.1 t.2.2} = 
  {(2, 2, 2), (5, 3, 7), (7, 5, 3), (3, 7, 5)} :=
sorry

end NUMINAMATH_GPT_find_parcera_triples_l1574_157496


namespace NUMINAMATH_GPT_girls_boys_difference_l1574_157462

variables (B G : ℕ) (x : ℕ)

-- Condition that relates boys and girls with a ratio
def ratio_condition : Prop := 3 * x = B ∧ 4 * x = G

-- Condition that the total number of students is 42
def total_students_condition : Prop := B + G = 42

-- We want to prove that the difference between the number of girls and boys is 6
theorem girls_boys_difference (h_ratio : ratio_condition B G x) (h_total : total_students_condition B G) : 
  G - B = 6 :=
sorry

end NUMINAMATH_GPT_girls_boys_difference_l1574_157462


namespace NUMINAMATH_GPT_triangle_side_length_l1574_157447

theorem triangle_side_length (a b c : ℝ)
  (h1 : 1/2 * a * c * (Real.sin (60 * Real.pi / 180)) = Real.sqrt 3)
  (h2 : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_length_l1574_157447


namespace NUMINAMATH_GPT_teachers_quit_before_lunch_percentage_l1574_157438

variables (n_initial n_after_one_hour n_after_lunch n_quit_before_lunch : ℕ)

def initial_teachers : ℕ := 60
def teachers_after_one_hour (n_initial : ℕ) : ℕ := n_initial / 2
def teachers_after_lunch : ℕ := 21
def quit_before_lunch (n_after_one_hour n_after_lunch : ℕ) : ℕ := n_after_one_hour - n_after_lunch
def percentage_quit (n_quit_before_lunch n_after_one_hour : ℕ) : ℕ := (n_quit_before_lunch * 100) / n_after_one_hour

theorem teachers_quit_before_lunch_percentage :
  ∀ n_initial n_after_one_hour n_after_lunch n_quit_before_lunch,
  n_initial = initial_teachers →
  n_after_one_hour = teachers_after_one_hour n_initial →
  n_after_lunch = teachers_after_lunch →
  n_quit_before_lunch = quit_before_lunch n_after_one_hour n_after_lunch →
  percentage_quit n_quit_before_lunch n_after_one_hour = 30 := by 
    sorry

end NUMINAMATH_GPT_teachers_quit_before_lunch_percentage_l1574_157438


namespace NUMINAMATH_GPT_point_on_y_axis_l1574_157484

theorem point_on_y_axis (x y : ℝ) (h : x = 0 ∧ y = -1) : y = -1 := by
  -- Using the conditions directly
  cases h with
  | intro hx hy =>
    -- The proof would typically follow, but we include sorry to complete the statement
    sorry

end NUMINAMATH_GPT_point_on_y_axis_l1574_157484


namespace NUMINAMATH_GPT_inequality_for_distinct_integers_l1574_157419

-- Define the necessary variables and conditions
variable {a b c : ℤ}

-- Ensure a, b, and c are pairwise distinct integers
def pairwise_distinct (a b c : ℤ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a

-- The main theorem statement
theorem inequality_for_distinct_integers 
  (h : pairwise_distinct a b c) : 
  (a^3 + b^3 + c^3) / 3 ≥ a * b * c + Real.sqrt (3 * (a * b + b * c + c * a + 1)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_for_distinct_integers_l1574_157419


namespace NUMINAMATH_GPT_height_of_E_l1574_157410

variable {h_E h_F h_G h_H : ℝ}

theorem height_of_E (h1 : h_E + h_F + h_G + h_H = 2 * (h_E + h_F))
                    (h2 : (h_E + h_F) / 2 = (h_E + h_G) / 2 - 4)
                    (h3 : h_H = h_E - 10)
                    (h4 : h_F + h_G = 288) :
  h_E = 139 :=
by
  sorry

end NUMINAMATH_GPT_height_of_E_l1574_157410


namespace NUMINAMATH_GPT_problem_1_l1574_157400

theorem problem_1 (x : ℝ) (h : x^2 = 2) : (3 * x)^2 - 4 * (x^3)^2 = -14 :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_1_l1574_157400


namespace NUMINAMATH_GPT_not_divisible_by_97_l1574_157489

theorem not_divisible_by_97 (k : ℤ) (h : k ∣ (99^3 - 99)) : k ≠ 97 :=
sorry

end NUMINAMATH_GPT_not_divisible_by_97_l1574_157489


namespace NUMINAMATH_GPT_time_fraction_l1574_157448

variable (t₅ t₁₅ : ℝ)

def total_distance (t₅ t₁₅ : ℝ) : ℝ :=
  5 * t₅ + 15 * t₁₅

def total_time (t₅ t₁₅ : ℝ) : ℝ :=
  t₅ + t₁₅

def average_speed_eq (t₅ t₁₅ : ℝ) : Prop :=
  10 * (t₅ + t₁₅) = 5 * t₅ + 15 * t₁₅

theorem time_fraction (t₅ t₁₅ : ℝ) (h : average_speed_eq t₅ t₁₅) :
  (t₁₅ / (t₅ + t₁₅)) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_time_fraction_l1574_157448


namespace NUMINAMATH_GPT_determine_m_range_l1574_157404

variable {R : Type} [OrderedCommGroup R]

-- Define the odd function f: ℝ → ℝ
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Define the increasing function f: ℝ → ℝ
def increasing_function (f : ℝ → ℝ) := ∀ x y, x < y → f x < f y

-- Define the main theorem
theorem determine_m_range (f : ℝ → ℝ) (odd_f : odd_function f) (inc_f : increasing_function f) :
    (∀ θ : ℝ, f (Real.cos (2 * θ) - 5) + f (2 * m + 4 * Real.sin θ) > 0) → m > 5 :=
by
  sorry

end NUMINAMATH_GPT_determine_m_range_l1574_157404


namespace NUMINAMATH_GPT_amount_per_person_is_correct_l1574_157469

-- Define the total amount and the number of people
def total_amount : ℕ := 2400
def number_of_people : ℕ := 9

-- State the main theorem to be proved
theorem amount_per_person_is_correct : total_amount / number_of_people = 266 := 
by sorry

end NUMINAMATH_GPT_amount_per_person_is_correct_l1574_157469


namespace NUMINAMATH_GPT_johnnys_hourly_wage_l1574_157444

def totalEarnings : ℝ := 26
def totalHours : ℝ := 8
def hourlyWage : ℝ := 3.25

theorem johnnys_hourly_wage : totalEarnings / totalHours = hourlyWage :=
by
  sorry

end NUMINAMATH_GPT_johnnys_hourly_wage_l1574_157444


namespace NUMINAMATH_GPT_new_energy_vehicle_sales_growth_l1574_157425

theorem new_energy_vehicle_sales_growth (x : ℝ) :
  let sales_jan := 64
  let sales_feb := 64 * (1 + x)
  let sales_mar := 64 * (1 + x)^2
  (sales_jan + sales_feb + sales_mar = 244) :=
sorry

end NUMINAMATH_GPT_new_energy_vehicle_sales_growth_l1574_157425


namespace NUMINAMATH_GPT_shelley_weight_l1574_157456

theorem shelley_weight (p s r : ℕ) (h1 : p + s = 151) (h2 : s + r = 132) (h3 : p + r = 115) : s = 84 := 
  sorry

end NUMINAMATH_GPT_shelley_weight_l1574_157456


namespace NUMINAMATH_GPT_complement_A_in_U_l1574_157426

universe u

-- Define the universal set U and set A.
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5, 7}

-- Define the complement of A in U.
def complement (A U: Set ℕ) : Set ℕ :=
  {x ∈ U | x ∉ A}

-- Statement to prove.
theorem complement_A_in_U :
  complement A U = {2, 4, 6} :=
sorry

end NUMINAMATH_GPT_complement_A_in_U_l1574_157426


namespace NUMINAMATH_GPT_find_f_neg_2_l1574_157468

theorem find_f_neg_2 (f : ℝ → ℝ) (b x : ℝ) (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, x ≥ 0 → f x = x^2 - 3*x + b) (h3 : f 0 = 0) : f (-2) = 2 := by
sorry

end NUMINAMATH_GPT_find_f_neg_2_l1574_157468


namespace NUMINAMATH_GPT_total_people_l1574_157492

-- Given definitions
def students : ℕ := 37500
def ratio_students_professors : ℕ := 15
def professors : ℕ := students / ratio_students_professors

-- The statement to prove
theorem total_people : students + professors = 40000 := by
  sorry

end NUMINAMATH_GPT_total_people_l1574_157492


namespace NUMINAMATH_GPT_profit_divided_equally_l1574_157443

noncomputable def Mary_investment : ℝ := 800
noncomputable def Mike_investment : ℝ := 200
noncomputable def total_profit : ℝ := 2999.9999999999995
noncomputable def Mary_extra : ℝ := 1200

theorem profit_divided_equally (E : ℝ) : 
  (E / 2 + 4 / 5 * (total_profit - E)) - (E / 2 + 1 / 5 * (total_profit - E)) = Mary_extra →
  E = 1000 :=
  by sorry

end NUMINAMATH_GPT_profit_divided_equally_l1574_157443


namespace NUMINAMATH_GPT_remainder_when_divided_by_29_l1574_157457

theorem remainder_when_divided_by_29 (k N : ℤ) (h : N = 761 * k + 173) : N % 29 = 28 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_29_l1574_157457


namespace NUMINAMATH_GPT_rectangular_box_inscribed_in_sphere_l1574_157445

noncomputable def problem_statement : Prop :=
  ∃ (a b c s : ℝ), (4 * (a + b + c) = 72) ∧ (2 * (a * b + b * c + c * a) = 216) ∧
  (a^2 + b^2 + c^2 = 108) ∧ (4 * s^2 = 108) ∧ (s = 3 * Real.sqrt 3)

theorem rectangular_box_inscribed_in_sphere : problem_statement := 
  sorry

end NUMINAMATH_GPT_rectangular_box_inscribed_in_sphere_l1574_157445


namespace NUMINAMATH_GPT_water_bottle_capacity_l1574_157437

theorem water_bottle_capacity :
  (20 * 250 + 13 * 600) / 1000 = 12.8 := 
by
  sorry

end NUMINAMATH_GPT_water_bottle_capacity_l1574_157437


namespace NUMINAMATH_GPT_initial_price_of_iphone_l1574_157439

variable (P : ℝ)

def initial_price_conditions : Prop :=
  (P > 0) ∧ (0.72 * P = 720)

theorem initial_price_of_iphone (h : initial_price_conditions P) : P = 1000 :=
by
  sorry

end NUMINAMATH_GPT_initial_price_of_iphone_l1574_157439


namespace NUMINAMATH_GPT_XYZStockPriceIs75_l1574_157407

/-- XYZ stock price model 
Starts at $50, increases by 200% in first year, 
then decreases by 50% in second year.
-/
def XYZStockPriceEndOfSecondYear : ℝ :=
  let initialPrice := 50
  let firstYearIncreaseRate := 2.0
  let secondYearDecreaseRate := 0.5
  let priceAfterFirstYear := initialPrice * (1 + firstYearIncreaseRate)
  let priceAfterSecondYear := priceAfterFirstYear * (1 - secondYearDecreaseRate)
  priceAfterSecondYear

theorem XYZStockPriceIs75 : XYZStockPriceEndOfSecondYear = 75 := by
  sorry

end NUMINAMATH_GPT_XYZStockPriceIs75_l1574_157407


namespace NUMINAMATH_GPT_inequality_proof_l1574_157455

theorem inequality_proof (a b x : ℝ) (h : a > b) : a * 2 ^ x > b * 2 ^ x :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1574_157455


namespace NUMINAMATH_GPT_speed_of_ferry_P_l1574_157452

variable (v_P v_Q : ℝ)

noncomputable def condition1 : Prop := v_Q = v_P + 4
noncomputable def condition2 : Prop := (6 * v_P) / v_Q = 4
noncomputable def condition3 : Prop := 2 + 2 = 4

theorem speed_of_ferry_P
    (h1 : condition1 v_P v_Q)
    (h2 : condition2 v_P v_Q)
    (h3 : condition3) :
    v_P = 8 := 
by 
    sorry

end NUMINAMATH_GPT_speed_of_ferry_P_l1574_157452


namespace NUMINAMATH_GPT_max_blocks_fit_l1574_157474

-- Define the dimensions of the block
def block_length : ℕ := 3
def block_width : ℕ := 1
def block_height : ℕ := 1

-- Define the dimensions of the box
def box_length : ℕ := 5
def box_width : ℕ := 3
def box_height : ℕ := 2

-- Theorem stating the maximum number of blocks that can fit in the box
theorem max_blocks_fit :
  (box_length * box_width * box_height) / (block_length * block_width * block_height) = 15 := sorry

end NUMINAMATH_GPT_max_blocks_fit_l1574_157474


namespace NUMINAMATH_GPT_predicted_customers_on_Saturday_l1574_157436

theorem predicted_customers_on_Saturday 
  (breakfast_customers : ℕ)
  (lunch_customers : ℕ)
  (dinner_customers : ℕ)
  (prediction_factor : ℕ)
  (h1 : breakfast_customers = 73)
  (h2 : lunch_customers = 127)
  (h3 : dinner_customers = 87)
  (h4 : prediction_factor = 2) :
  prediction_factor * (breakfast_customers + lunch_customers + dinner_customers) = 574 :=  
by 
  sorry 

end NUMINAMATH_GPT_predicted_customers_on_Saturday_l1574_157436


namespace NUMINAMATH_GPT_value_added_after_doubling_l1574_157414

theorem value_added_after_doubling (x v : ℝ) (h1 : x = 4) (h2 : 2 * x + v = x / 2 + 20) : v = 14 :=
by
  sorry

end NUMINAMATH_GPT_value_added_after_doubling_l1574_157414


namespace NUMINAMATH_GPT_calculate_expression_l1574_157490

theorem calculate_expression :
  (121^2 - 110^2 + 11) / 10 = 255.2 := 
sorry

end NUMINAMATH_GPT_calculate_expression_l1574_157490


namespace NUMINAMATH_GPT_batsman_average_after_17th_l1574_157488

theorem batsman_average_after_17th (A : ℤ) (h1 : 86 + 16 * A = 17 * (A + 3)) : A + 3 = 38 :=
by
  sorry

end NUMINAMATH_GPT_batsman_average_after_17th_l1574_157488


namespace NUMINAMATH_GPT_james_spends_90_dollars_per_week_l1574_157470

structure PistachioPurchasing where
  can_cost : ℕ  -- cost in dollars per can
  can_weight : ℕ -- weight in ounces per can
  consumption_oz_per_5days : ℕ -- consumption in ounces per 5 days

def cost_per_week (p : PistachioPurchasing) : ℕ :=
  let daily_consumption := p.consumption_oz_per_5days / 5
  let weekly_consumption := daily_consumption * 7
  let cans_needed := (weekly_consumption + p.can_weight - 1) / p.can_weight -- round up
  cans_needed * p.can_cost

theorem james_spends_90_dollars_per_week :
  cost_per_week ⟨10, 5, 30⟩ = 90 :=
by
  sorry

end NUMINAMATH_GPT_james_spends_90_dollars_per_week_l1574_157470


namespace NUMINAMATH_GPT_no_solution_for_vectors_l1574_157475

theorem no_solution_for_vectors {t s k : ℝ} :
  (∃ t s : ℝ, (1 + 6 * t = -1 + 3 * s) ∧ (3 + 1 * t = 4 + k * s)) ↔ k ≠ 0.5 :=
sorry

end NUMINAMATH_GPT_no_solution_for_vectors_l1574_157475


namespace NUMINAMATH_GPT_packs_of_yellow_bouncy_balls_l1574_157420

-- Define the conditions and the question in Lean
variables (GaveAwayGreen : ℝ) (BoughtGreen : ℝ) (BouncyBallsPerPack : ℝ) (TotalKeptBouncyBalls : ℝ) (Y : ℝ)

-- Assume the given conditions
axiom cond1 : GaveAwayGreen = 4.0
axiom cond2 : BoughtGreen = 4.0
axiom cond3 : BouncyBallsPerPack = 10.0
axiom cond4 : TotalKeptBouncyBalls = 80.0

-- Define the theorem statement
theorem packs_of_yellow_bouncy_balls (h1 : GaveAwayGreen = 4.0) (h2 : BoughtGreen = 4.0) (h3 : BouncyBallsPerPack = 10.0) (h4 : TotalKeptBouncyBalls = 80.0) : Y = 8 :=
sorry

end NUMINAMATH_GPT_packs_of_yellow_bouncy_balls_l1574_157420


namespace NUMINAMATH_GPT_french_students_l1574_157459

theorem french_students 
  (T : ℕ) (G : ℕ) (B : ℕ) (N : ℕ) (F : ℕ)
  (hT : T = 78) (hG : G = 22) (hB : B = 9) (hN : N = 24)
  (h_eq : F + G - B = T - N) :
  F = 41 :=
by
  sorry

end NUMINAMATH_GPT_french_students_l1574_157459


namespace NUMINAMATH_GPT_sqrt_expression_eq_neg_one_l1574_157477

theorem sqrt_expression_eq_neg_one : 
  Real.sqrt ((-2)^2) + (Real.sqrt 3)^2 - (Real.sqrt 12 * Real.sqrt 3) = -1 :=
sorry

end NUMINAMATH_GPT_sqrt_expression_eq_neg_one_l1574_157477


namespace NUMINAMATH_GPT_sum_of_roots_quadratic_specific_sum_of_roots_l1574_157461

theorem sum_of_roots_quadratic:
  ∀ a b c : ℚ, a ≠ 0 → 
  ∀ x1 x2 : ℚ, (a * x1^2 + b * x1 + c = 0) ∧ 
               (a * x2^2 + b * x2 + c = 0) → 
               x1 + x2 = -b / a := 
by
  sorry

theorem specific_sum_of_roots:
  ∀ x1 x2 : ℚ, (12 * x1^2 + 19 * x1 - 21 = 0) ∧ 
               (12 * x2^2 + 19 * x2 - 21 = 0) → 
               x1 + x2 = -19 / 12 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_quadratic_specific_sum_of_roots_l1574_157461


namespace NUMINAMATH_GPT_barry_sotter_length_increase_l1574_157464

theorem barry_sotter_length_increase (n : ℕ) : (n + 3) / 3 = 50 → n = 147 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_barry_sotter_length_increase_l1574_157464


namespace NUMINAMATH_GPT_relationship_between_m_and_n_l1574_157408

theorem relationship_between_m_and_n (f : ℝ → ℝ) (a : ℝ) 
  (h_even : ∀ x : ℝ, f x = f (-x)) 
  (h_mono_inc : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y) 
  (m_def : f (-1) = f 1) 
  (n_def : f (a^2 + 2*a + 3) > f 1) :
  f (-1) < f (a^2 + 2*a + 3) := 
by 
  sorry

end NUMINAMATH_GPT_relationship_between_m_and_n_l1574_157408


namespace NUMINAMATH_GPT_simplify_expression_correct_l1574_157450

variable (a b x y : ℝ) (i : ℂ)

noncomputable def simplify_expression (a b x y : ℝ) (i : ℂ) : Prop :=
  (a ≠ 0) ∧ (b ≠ 0) ∧ (i^2 = -1) → (a * x + b * i * y) * (a * x - b * i * y) = a^2 * x^2 + b^2 * y^2

theorem simplify_expression_correct (a b x y : ℝ) (i : ℂ) :
  simplify_expression a b x y i := by
  sorry

end NUMINAMATH_GPT_simplify_expression_correct_l1574_157450


namespace NUMINAMATH_GPT_maximum_value_expression_l1574_157480

theorem maximum_value_expression (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h_sum : a + b + c = 3) : 
  (a^2 - a * b + b^2) * (a^2 - a * c + c^2) * (b^2 - b * c + c^2) ≤ 1 :=
sorry

end NUMINAMATH_GPT_maximum_value_expression_l1574_157480


namespace NUMINAMATH_GPT_intersection_point_of_lines_l1574_157485

theorem intersection_point_of_lines :
  ∃ x y : ℝ, 3 * x + 4 * y - 2 = 0 ∧ 2 * x + y + 2 = 0 := 
by
  sorry

end NUMINAMATH_GPT_intersection_point_of_lines_l1574_157485


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l1574_157482

theorem arithmetic_geometric_sequence (d : ℤ) (a_1 a_2 a_5 : ℤ)
  (h1 : d ≠ 0)
  (h2 : a_2 = a_1 + d)
  (h3 : a_5 = a_1 + 4 * d)
  (h4 : a_2 ^ 2 = a_1 * a_5) :
  a_5 = 9 * a_1 := 
sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l1574_157482


namespace NUMINAMATH_GPT_non_empty_solution_set_range_l1574_157431

theorem non_empty_solution_set_range {a : ℝ} 
  (h : ∃ x : ℝ, |x + 2| + |x - 3| ≤ a) : 
  a ≥ 5 :=
sorry

end NUMINAMATH_GPT_non_empty_solution_set_range_l1574_157431


namespace NUMINAMATH_GPT_number_of_quarters_l1574_157487

-- Defining constants for the problem
def value_dime : ℝ := 0.10
def value_nickel : ℝ := 0.05
def value_penny : ℝ := 0.01
def value_quarter : ℝ := 0.25

-- Given conditions
def total_dimes : ℝ := 3
def total_nickels : ℝ := 4
def total_pennies : ℝ := 200
def total_amount : ℝ := 5.00

-- Theorem stating the number of quarters found
theorem number_of_quarters :
  (total_amount - (total_dimes * value_dime + total_nickels * value_nickel + total_pennies * value_penny)) / value_quarter = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_quarters_l1574_157487


namespace NUMINAMATH_GPT_solve_fraction_problem_l1574_157423

noncomputable def x_value (a b c d : ℤ) : ℝ :=
  (a + b * Real.sqrt c) / d

theorem solve_fraction_problem (a b c d : ℤ) (h1 : x_value a b c d = (5 + 5 * Real.sqrt 5) / 4)
  (h2 : (4 * x_value a b c d) / 5 - 2 = 5 / x_value a b c d) :
  (a * c * d) / b = 20 := by
  sorry

end NUMINAMATH_GPT_solve_fraction_problem_l1574_157423


namespace NUMINAMATH_GPT_fixed_monthly_fee_december_l1574_157413

theorem fixed_monthly_fee_december (x y : ℝ) 
    (h1 : x + y = 15.00) 
    (h2 : x + 2 + 3 * y = 25.40) : 
    x = 10.80 :=
by
  sorry

end NUMINAMATH_GPT_fixed_monthly_fee_december_l1574_157413


namespace NUMINAMATH_GPT_speed_limit_correct_l1574_157441

def speed_limit (distance : ℕ) (time : ℕ) (over_limit : ℕ) : ℕ :=
  let speed := distance / time
  speed - over_limit

theorem speed_limit_correct :
  speed_limit 60 1 10 = 50 :=
by
  sorry

end NUMINAMATH_GPT_speed_limit_correct_l1574_157441


namespace NUMINAMATH_GPT_compute_value_l1574_157440

def diamond_op (x y : ℕ) : ℕ := 3 * x + 5 * y
def heart_op (z x : ℕ) : ℕ := 4 * z + 2 * x

theorem compute_value : heart_op (diamond_op 4 3) 8 = 124 := by
  sorry

end NUMINAMATH_GPT_compute_value_l1574_157440


namespace NUMINAMATH_GPT_original_triangle_area_l1574_157427

-- Define the conditions
def dimensions_quadrupled (original_area new_area : ℝ) : Prop :=
  4^2 * original_area = new_area

-- Define the statement to be proved
theorem original_triangle_area {new_area : ℝ} (h : new_area = 64) :
  ∃ (original_area : ℝ), dimensions_quadrupled original_area new_area ∧ original_area = 4 :=
by
  sorry

end NUMINAMATH_GPT_original_triangle_area_l1574_157427


namespace NUMINAMATH_GPT_possible_values_of_m_plus_n_l1574_157493

theorem possible_values_of_m_plus_n (m n : ℕ) (hmn_pos : 0 < m ∧ 0 < n) 
  (cond : Nat.lcm m n - Nat.gcd m n = 103) : m + n = 21 ∨ m + n = 105 ∨ m + n = 309 := by
  sorry

end NUMINAMATH_GPT_possible_values_of_m_plus_n_l1574_157493


namespace NUMINAMATH_GPT_determinant_of_A_l1574_157401

def A : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![3, 0, -2],
  ![8, 5, -4],
  ![3, 3, 6]
]

theorem determinant_of_A : A.det = 108 := by
  sorry

end NUMINAMATH_GPT_determinant_of_A_l1574_157401


namespace NUMINAMATH_GPT_incorrect_expression_l1574_157478

theorem incorrect_expression : 
  ∀ (x y : ℚ), (x / y = 2 / 5) → (x + 3 * y) / x ≠ 17 / 2 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_incorrect_expression_l1574_157478


namespace NUMINAMATH_GPT_truck_capacity_solution_l1574_157472

variable (x y : ℝ)

theorem truck_capacity_solution (h1 : 3 * x + 4 * y = 22) (h2 : 2 * x + 6 * y = 23) :
  x + y = 6.5 := sorry

end NUMINAMATH_GPT_truck_capacity_solution_l1574_157472


namespace NUMINAMATH_GPT_feasible_measures_l1574_157434

-- Conditions for the problem
def condition1 := "Replace iron filings with iron pieces"
def condition2 := "Use excess zinc pieces instead of iron pieces"
def condition3 := "Add a small amount of CuSO₄ solution to the dilute hydrochloric acid"
def condition4 := "Add CH₃COONa solid to the dilute hydrochloric acid"
def condition5 := "Add sulfuric acid of the same molar concentration to the dilute hydrochloric acid"
def condition6 := "Add potassium sulfate solution to the dilute hydrochloric acid"
def condition7 := "Slightly heat (without considering the volatilization of HCl)"
def condition8 := "Add NaNO₃ solid to the dilute hydrochloric acid"

-- The criteria for the problem
def isFeasible (cond : String) : Prop :=
  cond = condition1 ∨ cond = condition2 ∨ cond = condition3 ∨ cond = condition7

theorem feasible_measures :
  ∀ cond, 
  cond ≠ condition4 →
  cond ≠ condition5 →
  cond ≠ condition6 →
  cond ≠ condition8 →
  isFeasible cond :=
by
  intros
  sorry

end NUMINAMATH_GPT_feasible_measures_l1574_157434


namespace NUMINAMATH_GPT_updated_mean_of_decrement_l1574_157412

theorem updated_mean_of_decrement 
  (mean_initial : ℝ)
  (num_observations : ℕ)
  (decrement_per_observation : ℝ)
  (h1 : mean_initial = 200)
  (h2 : num_observations = 50)
  (h3 : decrement_per_observation = 6) : 
  (mean_initial * num_observations - decrement_per_observation * num_observations) / num_observations = 194 :=
by
  sorry

end NUMINAMATH_GPT_updated_mean_of_decrement_l1574_157412


namespace NUMINAMATH_GPT_compute_fraction_power_mul_l1574_157430

theorem compute_fraction_power_mul : ((1 / 3: ℚ) ^ 4) * (1 / 5) = (1 / 405) := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_compute_fraction_power_mul_l1574_157430


namespace NUMINAMATH_GPT_monotonicity_of_f_solve_inequality_range_of_m_l1574_157467

variable {f : ℝ → ℝ}
variable {a b m : ℝ}

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def in_interval (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1
def f_at_one (f : ℝ → ℝ) : Prop := f 1 = 1
def positivity_condition (f : ℝ → ℝ) (a b : ℝ) : Prop := (a + b ≠ 0) → ((f a + f b) / (a + b) > 0)

-- Proof problems
theorem monotonicity_of_f 
  (h_odd : is_odd_function f) 
  (h_interval : ∀ x, in_interval x → in_interval (f x)) 
  (h_f_one : f_at_one f) 
  (h_pos : ∀ a b, in_interval a → in_interval b → positivity_condition f a b) :
  ∀ x1 x2, in_interval x1 → in_interval x2 → x1 < x2 → f x1 < f x2 :=
sorry

theorem solve_inequality 
  (h_odd : is_odd_function f) 
  (h_interval : ∀ x, in_interval x → in_interval (f x)) 
  (h_f_increasing : ∀ x1 x2, in_interval x1 → in_interval x2 → x1 < x2 → f x1 < f x2) 
  (h_f_one : f_at_one f) 
  (h_pos : ∀ a b, in_interval a → in_interval b → positivity_condition f a b) :
  ∀ x, in_interval (x + 1/2) → in_interval (1 / (x - 1)) → f (x + 1/2) < f (1 / (x - 1)) → -3/2 ≤ x ∧ x < -1 :=
sorry

theorem range_of_m 
  (h_odd : is_odd_function f) 
  (h_interval : ∀ x, in_interval x → in_interval (f x)) 
  (h_f_one : f_at_one f) 
  (h_pos : ∀ a b, in_interval a → in_interval b → positivity_condition f a b) 
  (h_f_increasing : ∀ x1 x2, in_interval x1 → in_interval x2 → x1 < x2 → f x1 < f x2) :
  (∀ a, in_interval a → f a ≤ m^2 - 2 * a * m + 1) → (m = 0 ∨ m ≤ -2 ∨ m ≥ 2) :=
sorry

end NUMINAMATH_GPT_monotonicity_of_f_solve_inequality_range_of_m_l1574_157467


namespace NUMINAMATH_GPT_hot_dogs_remainder_l1574_157422

theorem hot_dogs_remainder : 25197641 % 6 = 1 :=
by
  sorry

end NUMINAMATH_GPT_hot_dogs_remainder_l1574_157422


namespace NUMINAMATH_GPT_a_values_condition_l1574_157483

def is_subset (A B : Set ℝ) : Prop := ∀ x, x ∈ A → x ∈ B

theorem a_values_condition (a : ℝ) : 
  (2 * a + 1 ≤ 3 ∧ 3 * a - 5 ≤ 22 ∧ 2 * a + 1 ≤ 3 * a - 5) 
  ↔ (6 ≤ a ∧ a ≤ 9) :=
by 
  sorry

end NUMINAMATH_GPT_a_values_condition_l1574_157483


namespace NUMINAMATH_GPT_parabola_has_one_x_intercept_l1574_157421

-- Define the equation of the parabola.
def parabola (y : ℝ) : ℝ := -3 * y ^ 2 + 2 * y + 4

-- Prove that the number of x-intercepts of the graph of the parabola is 1.
theorem parabola_has_one_x_intercept : (∃! y : ℝ, parabola y = 4) :=
by
  sorry

end NUMINAMATH_GPT_parabola_has_one_x_intercept_l1574_157421


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_l1574_157463

theorem right_triangle_hypotenuse {a b c : ℝ} 
  (h1: a + b + c = 60) 
  (h2: a * b = 96) 
  (h3: a^2 + b^2 = c^2) : 
  c = 28.4 := 
sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_l1574_157463


namespace NUMINAMATH_GPT_gcd_lcm_ordering_l1574_157498

theorem gcd_lcm_ordering (a b p q : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_a_gt_b : a > b) 
    (h_p_gcd : p = Nat.gcd a b) (h_q_lcm : q = Nat.lcm a b) : q ≥ a ∧ a > b ∧ b ≥ p :=
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_ordering_l1574_157498


namespace NUMINAMATH_GPT_inequality_to_prove_l1574_157494

variable {r r1 r2 r3 m : ℝ}
variable {A B C : ℝ}

-- Conditions
-- r is the radius of an inscribed circle in a triangle
-- r1, r2, r3 are radii of circles each touching two sides of the triangle and the inscribed circle
-- m is a real number such that m >= 1/2

axiom r_radii_condition : r > 0
axiom r1_radii_condition : r1 > 0
axiom r2_radii_condition : r2 > 0
axiom r3_radii_condition : r3 > 0
axiom m_condition : m ≥ 1/2

-- Inequality to prove
theorem inequality_to_prove : 
  (r1 * r2) ^ m + (r2 * r3) ^ m + (r3 * r1) ^ m ≥ 3 * (r / 3) ^ (2 * m) := 
sorry

end NUMINAMATH_GPT_inequality_to_prove_l1574_157494


namespace NUMINAMATH_GPT_rectangle_length_l1574_157432

theorem rectangle_length
  (side_length_square : ℝ)
  (width_rectangle : ℝ)
  (area_equiv : side_length_square ^ 2 = width_rectangle * l)
  : l = 24 := by
  sorry

end NUMINAMATH_GPT_rectangle_length_l1574_157432


namespace NUMINAMATH_GPT_value_of_f_at_sqrt2_l1574_157417

noncomputable def f (x : ℝ) : ℝ := x^5 - 5 * x^4 + 10 * x^3 - 10 * x^2 + 5 * x - 1

theorem value_of_f_at_sqrt2 :
  f (1 + Real.sqrt 2) = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_value_of_f_at_sqrt2_l1574_157417


namespace NUMINAMATH_GPT_Susan_total_peaches_l1574_157446

-- Define the number of peaches in the knapsack
def peaches_in_knapsack : ℕ := 12

-- Define the condition that the number of peaches in the knapsack is half the number of peaches in each cloth bag
def peaches_per_cloth_bag (x : ℕ) : Prop := peaches_in_knapsack * 2 = x

-- Define the total number of peaches Susan bought
def total_peaches (x : ℕ) : ℕ := x + 2 * x

-- Theorem statement: Prove that the total number of peaches Susan bought is 60
theorem Susan_total_peaches (x : ℕ) (h : peaches_per_cloth_bag x) : total_peaches peaches_in_knapsack = 60 := by
  sorry

end NUMINAMATH_GPT_Susan_total_peaches_l1574_157446


namespace NUMINAMATH_GPT_manufacturers_price_l1574_157473

theorem manufacturers_price (M : ℝ) 
  (h1 : 0.1 ≤ 0.3) 
  (h2 : 0.2 = 0.2) 
  (h3 : 0.56 * M = 25.2) : 
  M = 45 := 
sorry

end NUMINAMATH_GPT_manufacturers_price_l1574_157473


namespace NUMINAMATH_GPT_number_of_male_students_l1574_157424

noncomputable def avg_all : ℝ := 90
noncomputable def avg_male : ℝ := 84
noncomputable def avg_female : ℝ := 92
noncomputable def count_female : ℕ := 24

theorem number_of_male_students (M : ℕ) (T : ℕ) :
  avg_all * (M + count_female) = avg_male * M + avg_female * count_female →
  T = M + count_female →
  M = 8 :=
by
  intro h_avg h_count
  sorry

end NUMINAMATH_GPT_number_of_male_students_l1574_157424


namespace NUMINAMATH_GPT_cream_ratio_l1574_157433

def joe_ends_with_cream (start_coffee : ℕ) (drank_coffee : ℕ) (added_cream : ℕ) : ℕ :=
  added_cream

def joann_cream_left (start_coffee : ℕ) (added_cream : ℕ) (drank_mix : ℕ) : ℚ :=
  added_cream - drank_mix * (added_cream / (start_coffee + added_cream))

theorem cream_ratio (start_coffee : ℕ) (joe_drinks : ℕ) (joe_adds : ℕ)
                    (joann_adds : ℕ) (joann_drinks : ℕ) :
  joe_ends_with_cream start_coffee joe_drinks joe_adds / 
  joann_cream_left start_coffee joann_adds joann_drinks = (9 : ℚ) / (7 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_cream_ratio_l1574_157433


namespace NUMINAMATH_GPT_rectangular_prism_sum_of_dimensions_l1574_157491

theorem rectangular_prism_sum_of_dimensions (a b c : ℕ) (h_volume : a * b * c = 21) 
(h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) : 
a + b + c = 11 :=
sorry

end NUMINAMATH_GPT_rectangular_prism_sum_of_dimensions_l1574_157491


namespace NUMINAMATH_GPT_max_value_f_at_e_l1574_157415

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_value_f_at_e (h : 0 < x) : 
  ∃ e : ℝ, (∀ x : ℝ, 0 < x → f x ≤ f e) ∧ e = Real.exp 1 :=
by
  sorry

end NUMINAMATH_GPT_max_value_f_at_e_l1574_157415


namespace NUMINAMATH_GPT_fifth_term_sequence_l1574_157442

theorem fifth_term_sequence 
  (a : ℕ → ℤ)
  (h1 : a 1 = 3)
  (h2 : a 2 = 6)
  (h_rec : ∀ n, a (n + 2) = a (n + 1) - a n) :
  a 5 = -6 := 
by
  sorry

end NUMINAMATH_GPT_fifth_term_sequence_l1574_157442


namespace NUMINAMATH_GPT_prize_winners_l1574_157409

theorem prize_winners (n : ℕ) (p1 p2 : ℝ) (h1 : n = 100) (h2 : p1 = 0.4) (h3 : p2 = 0.2) :
  ∃ winners : ℕ, winners = (p2 * (p1 * n)) ∧ winners = 8 :=
by
  sorry

end NUMINAMATH_GPT_prize_winners_l1574_157409


namespace NUMINAMATH_GPT_arithmetic_sequence_term_l1574_157402

theorem arithmetic_sequence_term {a : ℕ → ℤ} 
  (h1 : a 4 = -4) 
  (h2 : a 8 = 4) : 
  a 12 = 12 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_term_l1574_157402


namespace NUMINAMATH_GPT_ticket_cost_correct_l1574_157471

noncomputable def calculate_ticket_cost : ℝ :=
  let x : ℝ := 5  -- price of an adult ticket
  let child_price := x / 2  -- price of a child ticket
  let senior_price := 0.75 * x  -- price of a senior ticket
  10 * x + 8 * child_price + 5 * senior_price

theorem ticket_cost_correct :
  let x : ℝ := 5  -- price of an adult ticket
  let child_price := x / 2  -- price of a child ticket
  let senior_price := 0.75 * x  -- price of a senior ticket
  (4 * x + 3 * child_price + 2 * senior_price = 35) →
  (10 * x + 8 * child_price + 5 * senior_price = 88.75) :=
by
  intros
  sorry

end NUMINAMATH_GPT_ticket_cost_correct_l1574_157471
