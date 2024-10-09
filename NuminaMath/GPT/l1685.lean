import Mathlib

namespace distance_between_towns_l1685_168573

-- Define the custom scale for conversion
def scale_in_km := 1.05  -- 1 km + 50 meters as 1.05 km

-- Input distances on the map and their conversion
def map_distance_in_inches := 6 + 11/16

noncomputable def actual_distance_in_km : ℝ :=
  let distance_in_inches := (6 * 8 + 11) / 16
  distance_in_inches * (8 / 3)

theorem distance_between_towns :
  actual_distance_in_km = 17.85 := by
  -- Equivalent mathematical steps and tests here
  sorry

end distance_between_towns_l1685_168573


namespace symmetric_line_eq_l1685_168579

theorem symmetric_line_eq (x y : ℝ) (h : 2 * x - y = 0) : 2 * x + y = 0 :=
sorry

end symmetric_line_eq_l1685_168579


namespace xuzhou_test_2014_l1685_168576

variables (A B C D : ℝ) -- Assume A, B, C, D are real numbers.

theorem xuzhou_test_2014 :
  (C < D) → (A > B) :=
sorry

end xuzhou_test_2014_l1685_168576


namespace cuboid_height_l1685_168527

theorem cuboid_height
  (volume : ℝ)
  (width : ℝ)
  (length : ℝ)
  (height : ℝ)
  (h_volume : volume = 315)
  (h_width : width = 9)
  (h_length : length = 7)
  (h_volume_eq : volume = length * width * height) :
  height = 5 :=
by
  sorry

end cuboid_height_l1685_168527


namespace countTwoLeggedBirds_l1685_168535

def countAnimals (x y : ℕ) : Prop :=
  x + y = 200 ∧ 2 * x + 4 * y = 522

theorem countTwoLeggedBirds (x y : ℕ) (h : countAnimals x y) : x = 139 :=
by
  sorry

end countTwoLeggedBirds_l1685_168535


namespace problem_solution_l1685_168577

theorem problem_solution (x y z : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 20) (h3 : 0.6 * y = z) : 
  z = 60 := by
  sorry

end problem_solution_l1685_168577


namespace minimum_equilateral_triangles_l1685_168588

theorem minimum_equilateral_triangles (side_small : ℝ) (side_large : ℝ)
  (h_small : side_small = 1) (h_large : side_large = 15) :
  225 = (side_large / side_small)^2 :=
by
  -- Proof is skipped.
  sorry

end minimum_equilateral_triangles_l1685_168588


namespace difference_in_cans_l1685_168510

-- Definitions of the conditions
def total_cans_collected : ℕ := 9
def cans_in_bag : ℕ := 7

-- Statement of the proof problem
theorem difference_in_cans :
  total_cans_collected - cans_in_bag = 2 := by
  sorry

end difference_in_cans_l1685_168510


namespace tan_ratio_l1685_168575

theorem tan_ratio (α β : ℝ) (h : Real.sin (2 * α) = 3 * Real.sin (2 * β)) :
  (Real.tan (α - β) / Real.tan (α + β)) = 1 / 2 :=
sorry

end tan_ratio_l1685_168575


namespace probability_of_selecting_one_second_class_product_l1685_168589

def total_products : ℕ := 100
def first_class_products : ℕ := 90
def second_class_products : ℕ := 10
def selected_products : ℕ := 3
def exactly_one_second_class_probability : ℚ :=
  (Nat.choose first_class_products 2 * Nat.choose second_class_products 1) / Nat.choose total_products selected_products

theorem probability_of_selecting_one_second_class_product :
  exactly_one_second_class_probability = 0.25 := 
  sorry

end probability_of_selecting_one_second_class_product_l1685_168589


namespace algebraic_expression_evaluates_to_2_l1685_168546

theorem algebraic_expression_evaluates_to_2 (x : ℝ) (h : x^2 + x - 5 = 0) : 
(x - 1)^2 - x * (x - 3) + (x + 2) * (x - 2) = 2 := 
by 
  sorry

end algebraic_expression_evaluates_to_2_l1685_168546


namespace james_jump_height_is_16_l1685_168516

-- Define given conditions
def mark_jump_height : ℕ := 6
def lisa_jump_height : ℕ := 2 * mark_jump_height
def jacob_jump_height : ℕ := 2 * lisa_jump_height
def james_jump_height : ℕ := (2 * jacob_jump_height) / 3

-- Problem Statement to prove
theorem james_jump_height_is_16 : james_jump_height = 16 :=
by
  sorry

end james_jump_height_is_16_l1685_168516


namespace inverse_of_problem_matrix_is_zero_matrix_l1685_168523

def det (M : Matrix (Fin 2) (Fin 2) ℝ) : ℝ :=
  M 0 0 * M 1 1 - M 0 1 * M 1 0

def zero_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 0], ![0, 0]]

noncomputable def problem_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![4, -6], ![-2, 3]]

theorem inverse_of_problem_matrix_is_zero_matrix :
  det problem_matrix = 0 → problem_matrix⁻¹ = zero_matrix :=
by
  intro h
  -- Proof steps will be written here
  sorry

end inverse_of_problem_matrix_is_zero_matrix_l1685_168523


namespace Anne_weight_l1685_168569

-- Define variables
def Douglas_weight : ℕ := 52
def weight_difference : ℕ := 15

-- Theorem to prove
theorem Anne_weight : Douglas_weight + weight_difference = 67 :=
by sorry

end Anne_weight_l1685_168569


namespace fiona_shirt_number_l1685_168593

def is_two_digit_prime (n : ℕ) : Prop := 
  (n ≥ 10 ∧ n < 100 ∧ Nat.Prime n)

theorem fiona_shirt_number (d e f : ℕ) 
  (h1 : is_two_digit_prime d)
  (h2 : is_two_digit_prime e)
  (h3 : is_two_digit_prime f)
  (h4 : e + f = 36)
  (h5 : d + e = 30)
  (h6 : d + f = 32) : 
  f = 19 := 
sorry

end fiona_shirt_number_l1685_168593


namespace cos_seven_theta_l1685_168552

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : 
  Real.cos (7 * θ) = -160481 / 2097152 := by
  sorry

end cos_seven_theta_l1685_168552


namespace Ned_earning_money_l1685_168544

def total_games : Nat := 15
def non_working_games : Nat := 6
def price_per_game : Nat := 7
def working_games : Nat := total_games - non_working_games
def total_money : Nat := working_games * price_per_game

theorem Ned_earning_money : total_money = 63 := by
  sorry

end Ned_earning_money_l1685_168544


namespace NoahMealsCount_l1685_168545

-- Definition of all the choices available to Noah
def MainCourses := ["Pizza", "Burger", "Pasta"]
def Beverages := ["Soda", "Juice"]
def Snacks := ["Apple", "Banana", "Cookie"]

-- Condition that Noah avoids soda with pizza
def isValidMeal (main : String) (beverage : String) : Bool :=
  not (main = "Pizza" ∧ beverage = "Soda")

-- Total number of valid meal combinations
def totalValidMeals : Nat :=
  (if isValidMeal "Pizza" "Juice" then 1 else 0) * Snacks.length +
  (Beverages.length - 1) * Snacks.length * (MainCourses.length - 1) + -- for Pizza
  Beverages.length * Snacks.length * 2 -- for Burger and Pasta

-- The theorem that Noah can buy 15 distinct meals
theorem NoahMealsCount : totalValidMeals = 15 := by
  sorry

end NoahMealsCount_l1685_168545


namespace perimeter_of_rhombus_l1685_168550

theorem perimeter_of_rhombus (d1 d2 : ℝ) (hd1 : d1 = 8) (hd2 : d2 = 30) :
  (perimeter : ℝ) = 4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) :=
by
  simp [hd1, hd2]
  sorry

end perimeter_of_rhombus_l1685_168550


namespace sum_of_a_and_b_is_two_l1685_168597

variable (a b : ℝ)
variable (h_a_nonzero : a ≠ 0)
variable (h_fn_passes_through_point : (a * 1^2 + b * 1 - 1) = 1)

theorem sum_of_a_and_b_is_two : a + b = 2 := 
by
  sorry

end sum_of_a_and_b_is_two_l1685_168597


namespace total_males_below_50_is_2638_l1685_168539

def branchA_total_employees := 4500
def branchA_percentage_males := 60 / 100
def branchA_percentage_males_at_least_50 := 40 / 100

def branchB_total_employees := 3500
def branchB_percentage_males := 50 / 100
def branchB_percentage_males_at_least_50 := 55 / 100

def branchC_total_employees := 2200
def branchC_percentage_males := 35 / 100
def branchC_percentage_males_at_least_50 := 70 / 100

def males_below_50_branchA := (1 - branchA_percentage_males_at_least_50) * (branchA_percentage_males * branchA_total_employees)
def males_below_50_branchB := (1 - branchB_percentage_males_at_least_50) * (branchB_percentage_males * branchB_total_employees)
def males_below_50_branchC := (1 - branchC_percentage_males_at_least_50) * (branchC_percentage_males * branchC_total_employees)

def total_males_below_50 := males_below_50_branchA + males_below_50_branchB + males_below_50_branchC

theorem total_males_below_50_is_2638 : total_males_below_50 = 2638 := 
by
  -- Numerical evaluation and equality verification here
  sorry

end total_males_below_50_is_2638_l1685_168539


namespace find_a_l1685_168583

variable (a b c : ℚ)

theorem find_a (h1 : a + b + c = 150) (h2 : a - 3 = b + 4) (h3 : b + 4 = 4 * c) : 
  a = 631 / 9 :=
by
  sorry

end find_a_l1685_168583


namespace question_1_question_2_question_3_l1685_168515
-- Importing the Mathlib library for necessary functions

-- Definitions and assumptions based on the problem conditions
def z0 (m : ℝ) : ℂ := 1 - m * Complex.I
def z (x y : ℝ) : ℂ := x + y * Complex.I
def w (x' y' : ℝ) : ℂ := x' + y' * Complex.I

/-- The proof problem in Lean 4 to find necessary values and relationships -/
theorem question_1 (m : ℝ) (hm : m > 0) :
  (Complex.abs (z0 m) = 2 → m = Real.sqrt 3) ∧
  (∀ (x y : ℝ), ∃ (x' y' : ℝ), x' = x + Real.sqrt 3 * y ∧ y' = Real.sqrt 3 * x - y) :=
by
  sorry

theorem question_2 (x y : ℝ) (hx : y = x + 1) :
  ∃ x' y', x' = x + Real.sqrt 3 * y ∧ y' = Real.sqrt 3 * x - y ∧ 
  y' = (2 - Real.sqrt 3) * x' - 2 * Real.sqrt 3 + 2 :=
by
  sorry

theorem question_3 (x y : ℝ) :
  (∃ (k b : ℝ), y = k * x + b ∧ 
  (∀ (x y x' y' : ℝ), x' = x + Real.sqrt 3 * y ∧ y' = Real.sqrt 3 * x - y ∧ y' = k * x' + b → 
  y = Real.sqrt 3 / 3 * x ∨ y = - Real.sqrt 3 * x)) :=
by
  sorry

end question_1_question_2_question_3_l1685_168515


namespace richard_older_than_david_by_l1685_168514

-- Definitions based on given conditions

def richard : ℕ := sorry
def david : ℕ := 14 -- David is 14 years old.
def scott : ℕ := david - 8 -- Scott is 8 years younger than David.

-- In 8 years, Richard will be twice as old as Scott
axiom richard_in_8_years : richard + 8 = 2 * (scott + 8)

-- To prove: How many years older is Richard than David?
theorem richard_older_than_david_by : richard - david = 6 := sorry

end richard_older_than_david_by_l1685_168514


namespace cake_slices_l1685_168551

theorem cake_slices (S : ℕ) (h : 347 * S = 6 * 375 + 526) : S = 8 :=
sorry

end cake_slices_l1685_168551


namespace number_of_teachers_l1685_168560

-- Definitions from the problem conditions
def num_students : Nat := 1500
def classes_per_student : Nat := 6
def classes_per_teacher : Nat := 5
def students_per_class : Nat := 25

-- The proof problem statement
theorem number_of_teachers : 
  (num_students * classes_per_student / students_per_class) / classes_per_teacher = 72 := by
  sorry

end number_of_teachers_l1685_168560


namespace min_value_2a_plus_b_value_of_t_l1685_168509

theorem min_value_2a_plus_b (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1/a + 2/b = 2) :
  2 * a + b = 4 :=
sorry

theorem value_of_t (a b t : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1/a + 2/b = 2) (h₄ : 4^a = t) (h₅ : 3^b = t) :
  t = 6 :=
sorry

end min_value_2a_plus_b_value_of_t_l1685_168509


namespace rhombus_perimeter_l1685_168595

-- Define the conditions for the rhombus
variable (d1 d2 : ℝ) (a b s : ℝ)

-- State the condition that the diagonals of a rhombus measure 24 cm and 10 cm
def diagonal_condition := (d1 = 24) ∧ (d2 = 10)

-- State the Pythagorean theorem for the lengths of half-diagonals
def pythagorean_theorem := a^2 + b^2 = s^2

-- State the relationship of diagonals bisecting each other at right angles
def bisect_condition := (a = d1 / 2) ∧ (b = d2 / 2)

-- State the definition of the perimeter for a rhombus
def perimeter (s : ℝ) : ℝ := 4 * s

-- The theorem we want to prove
theorem rhombus_perimeter : diagonal_condition d1 d2 →
                            bisect_condition d1 d2 a b →
                            pythagorean_theorem a b s →
                            perimeter s = 52 :=
by
  intros h1 h2 h3
  -- Proof would go here, but it is omitted
  sorry

end rhombus_perimeter_l1685_168595


namespace roots_of_cubic_l1685_168525

/-- Let p, q, and r be the roots of the polynomial x^3 - 15x^2 + 10x + 24 = 0. 
   The value of (1 + p)(1 + q)(1 + r) is equal to 2. -/
theorem roots_of_cubic (p q r : ℝ)
  (h1 : p + q + r = 15)
  (h2 : p * q + q * r + r * p = 10)
  (h3 : p * q * r = -24) :
  (1 + p) * (1 + q) * (1 + r) = 2 := 
by 
  sorry

end roots_of_cubic_l1685_168525


namespace sqrt_and_cbrt_eq_self_l1685_168540

theorem sqrt_and_cbrt_eq_self (x : ℝ) (h1 : x = Real.sqrt x) (h2 : x = x^(1/3)) : x = 0 := by
  sorry

end sqrt_and_cbrt_eq_self_l1685_168540


namespace fastest_pipe_is_4_l1685_168511

/-- There are five pipes with flow rates Q_1, Q_2, Q_3, Q_4, and Q_5.
    The ordering of their flow rates is given by:
    (1) Q_1 > Q_3
    (2) Q_2 < Q_4
    (3) Q_3 < Q_5
    (4) Q_4 > Q_1
    (5) Q_5 < Q_2
    We need to prove that single pipe Q_4 will fill the pool the fastest.
 -/
theorem fastest_pipe_is_4 
  (Q1 Q2 Q3 Q4 Q5 : ℝ)
  (h1 : Q1 > Q3)
  (h2 : Q2 < Q4)
  (h3 : Q3 < Q5)
  (h4 : Q4 > Q1)
  (h5 : Q5 < Q2) :
  Q4 > Q1 ∧ Q4 > Q2 ∧ Q4 > Q3 ∧ Q4 > Q5 :=
by
  sorry

end fastest_pipe_is_4_l1685_168511


namespace conic_curve_eccentricity_l1685_168585

theorem conic_curve_eccentricity (m : ℝ) 
    (h1 : ∃ k, k ≠ 0 ∧ 1 * k = m ∧ m * k = 4)
    (h2 : m = -2) : ∃ e : ℝ, e = Real.sqrt 3 :=
by
  sorry

end conic_curve_eccentricity_l1685_168585


namespace ratio_d_s_l1685_168512

theorem ratio_d_s (s d : ℝ) 
  (h : (25 * 25 * s^2) / (25 * s + 50 * d)^2 = 0.81) :
  d / s = 1 / 18 :=
by
  sorry

end ratio_d_s_l1685_168512


namespace expr_eval_l1685_168526

theorem expr_eval : 3^3 - 3^2 + 3^1 - 3^0 = 20 := by
  sorry

end expr_eval_l1685_168526


namespace probability_scoring_less_than_8_l1685_168508

theorem probability_scoring_less_than_8 
  (P10 P9 P8 : ℝ) 
  (hP10 : P10 = 0.3) 
  (hP9 : P9 = 0.3) 
  (hP8 : P8 = 0.2) : 
  1 - (P10 + P9 + P8) = 0.2 := 
by 
  sorry

end probability_scoring_less_than_8_l1685_168508


namespace buddy_cards_on_thursday_is_32_l1685_168562

def buddy_cards_on_monday := 30
def buddy_cards_on_tuesday := buddy_cards_on_monday / 2
def buddy_cards_on_wednesday := buddy_cards_on_tuesday + 12
def buddy_cards_bought_on_thursday := buddy_cards_on_tuesday / 3
def buddy_cards_on_thursday := buddy_cards_on_wednesday + buddy_cards_bought_on_thursday

theorem buddy_cards_on_thursday_is_32 : buddy_cards_on_thursday = 32 :=
by sorry

end buddy_cards_on_thursday_is_32_l1685_168562


namespace vector_dot_product_l1685_168582

variables (a b : ℝ × ℝ)
variables (h1 : a = (1, 2))
variables (h2 : a - (1 / 5) • b = (-2, 1))

theorem vector_dot_product : (a.1 * b.1 + a.2 * b.2) = 25 :=
by
  sorry

end vector_dot_product_l1685_168582


namespace equipment_total_cost_l1685_168598

-- Definition of costs for each item of equipment
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.20
def socks_cost : ℝ := 6.80

-- Number of players
def num_players : ℕ := 16

-- Total cost for one player
def total_cost_one_player : ℝ := jersey_cost + shorts_cost + socks_cost

-- Total cost for all players
def total_cost_all_players : ℝ := total_cost_one_player * num_players

-- Theorem to prove
theorem equipment_total_cost : total_cost_all_players = 752 := by
  sorry

end equipment_total_cost_l1685_168598


namespace calculate_surface_area_of_modified_cube_l1685_168543

-- Definitions of the conditions
def edge_length_of_cube : ℕ := 5
def side_length_of_hole : ℕ := 2

-- The main theorem statement to be proven
theorem calculate_surface_area_of_modified_cube :
  let original_surface_area := 6 * (edge_length_of_cube * edge_length_of_cube)
  let area_removed_by_holes := 6 * (side_length_of_hole * side_length_of_hole)
  let area_exposed_by_holes := 6 * 6 * (side_length_of_hole * side_length_of_hole)
  original_surface_area - area_removed_by_holes + area_exposed_by_holes = 270 :=
by
  sorry

end calculate_surface_area_of_modified_cube_l1685_168543


namespace group9_40_41_right_angled_l1685_168592

theorem group9_40_41_right_angled :
  ¬ (∃ a b c : ℝ, a = 3 ∧ b = 4 ∧ c = 7 ∧ a^2 + b^2 = c^2) ∧
  ¬ (∃ a b c : ℝ, a = 1/3 ∧ b = 1/4 ∧ c = 1/5 ∧ a^2 + b^2 = c^2) ∧
  ¬ (∃ a b c : ℝ, a = 4 ∧ b = 6 ∧ c = 8 ∧ a^2 + b^2 = c^2) ∧
  (∃ a b c : ℝ, a = 9 ∧ b = 40 ∧ c = 41 ∧ a^2 + b^2 = c^2) :=
by
  sorry

end group9_40_41_right_angled_l1685_168592


namespace standard_equation_of_ellipse_l1685_168534

theorem standard_equation_of_ellipse
  (a b c : ℝ)
  (h_major_minor : 2 * a = 6 * b)
  (h_focal_distance : 2 * c = 8)
  (h_ellipse_relation : a^2 = b^2 + c^2) :
  (∀ x y : ℝ, (x^2 / 18 + y^2 / 2 = 1) ∨ (y^2 / 18 + x^2 / 2 = 1)) :=
by {
  sorry
}

end standard_equation_of_ellipse_l1685_168534


namespace complex_number_expression_l1685_168547

noncomputable def compute_expression (r : ℂ) (h1 : r^6 = 1) (h2 : r ≠ 1) :=
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1)

theorem complex_number_expression (r : ℂ) (h1 : r^6 = 1) (h2 : r ≠ 1) :
  compute_expression r h1 h2 = 5 :=
sorry

end complex_number_expression_l1685_168547


namespace function_increment_l1685_168572

theorem function_increment (f : ℝ → ℝ) 
  (h : ∀ x, f x = 2 / x) : f 1.5 - f 2 = 1 / 3 := 
by {
  sorry
}

end function_increment_l1685_168572


namespace ratio_equivalence_l1685_168520

theorem ratio_equivalence (x : ℕ) (h1 : 3 / 12 = x / 16) : x = 4 :=
by sorry

end ratio_equivalence_l1685_168520


namespace problem_statement_l1685_168578

theorem problem_statement (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (D : ℕ) (M : ℕ) (h_gcd : D = Nat.gcd (Nat.gcd a b) c) (h_lcm : M = Nat.lcm (Nat.lcm a b) c) :
  ((D * M = a * b * c) ∧ ((Nat.gcd a b = 1) ∧ (Nat.gcd b c = 1) ∧ (Nat.gcd a c = 1) → (D * M = a * b * c))) :=
by sorry

end problem_statement_l1685_168578


namespace sum_of_prime_factors_1729728_l1685_168503

def prime_factors_sum (n : ℕ) : ℕ := 
  -- Suppose that a function defined to calculate the sum of distinct prime factors
  -- In a practical setting, you would define this function or use an existing library
  sorry 

theorem sum_of_prime_factors_1729728 : prime_factors_sum 1729728 = 36 :=
by {
  -- Proof would go here
  sorry
}

end sum_of_prime_factors_1729728_l1685_168503


namespace valid_votes_l1685_168554

theorem valid_votes (V : ℝ) 
  (h1 : 0.70 * V - 0.30 * V = 176): V = 440 :=
  sorry

end valid_votes_l1685_168554


namespace total_books_l1685_168549

theorem total_books (joan_books : ℕ) (tom_books : ℕ) (h1 : joan_books = 10) (h2 : tom_books = 38) : joan_books + tom_books = 48 :=
by
  -- insert proof here
  sorry

end total_books_l1685_168549


namespace pencils_count_l1685_168517

theorem pencils_count (pens pencils : ℕ) 
  (h_ratio : 6 * pens = 5 * pencils) 
  (h_difference : pencils = pens + 6) : 
  pencils = 36 := 
by 
  sorry

end pencils_count_l1685_168517


namespace customers_at_start_l1685_168565

def initial_customers (X : ℕ) : Prop :=
  let first_hour := X + 3
  let second_hour := first_hour - 6
  second_hour = 12

theorem customers_at_start {X : ℕ} : initial_customers X → X = 15 :=
by
  sorry

end customers_at_start_l1685_168565


namespace kernels_popped_in_first_bag_l1685_168580

theorem kernels_popped_in_first_bag :
  ∀ (x : ℕ), 
    (total_kernels : ℕ := 75 + 50 + 100) →
    (total_popped : ℕ := x + 42 + 82) →
    (average_percentage_popped : ℚ := 82) →
    ((total_popped : ℚ) / total_kernels) * 100 = average_percentage_popped →
    x = 61 :=
by
  sorry

end kernels_popped_in_first_bag_l1685_168580


namespace quadratic_intersects_x_axis_l1685_168558

theorem quadratic_intersects_x_axis (a : ℝ) (h : a ≠ 0) :
  ∃ x : ℝ, a * x^2 - (3 * a + 1) * x + 3 = 0 := 
by {
  -- The proof will go here
  sorry
}

end quadratic_intersects_x_axis_l1685_168558


namespace area_within_fence_l1685_168501

theorem area_within_fence : 
  let rectangle_area := 20 * 18
  let cutout_area := 4 * 4
  rectangle_area - cutout_area = 344 := by
    -- Definitions
    let rectangle_area := 20 * 18
    let cutout_area := 4 * 4
    
    -- Computation of areas
    show rectangle_area - cutout_area = 344
    sorry

end area_within_fence_l1685_168501


namespace sam_initial_puppies_l1685_168553

theorem sam_initial_puppies (gave_away : ℝ) (now_has : ℝ) (initially : ℝ) 
    (h1 : gave_away = 2.0) (h2 : now_has = 4.0) : initially = 6.0 :=
by
  sorry

end sam_initial_puppies_l1685_168553


namespace intersection_A_B_l1685_168587

namespace SetTheory

open Set

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {0, 1} :=
by
  sorry

end SetTheory

end intersection_A_B_l1685_168587


namespace speed_ratio_l1685_168559

theorem speed_ratio (v_A v_B : ℝ) (t : ℝ) (h1 : v_A = 200 / t) (h2 : v_B = 120 / t) : 
  v_A / v_B = 5 / 3 :=
by
  sorry

end speed_ratio_l1685_168559


namespace laptop_price_l1685_168519

theorem laptop_price (upfront_percent : ℝ) (upfront_payment full_price : ℝ)
  (h1 : upfront_percent = 0.20)
  (h2 : upfront_payment = 240)
  (h3 : upfront_payment = upfront_percent * full_price) :
  full_price = 1200 := 
sorry

end laptop_price_l1685_168519


namespace chef_cooked_additional_wings_l1685_168522

def total_chicken_wings_needed (friends : ℕ) (wings_per_friend : ℕ) : ℕ :=
  friends * wings_per_friend

def additional_chicken_wings (total_needed : ℕ) (already_cooked : ℕ) : ℕ :=
  total_needed - already_cooked

theorem chef_cooked_additional_wings :
  let friends := 4
  let wings_per_friend := 4
  let already_cooked := 9
  additional_chicken_wings (total_chicken_wings_needed friends wings_per_friend) already_cooked = 7 := by
  sorry

end chef_cooked_additional_wings_l1685_168522


namespace incorrect_statement_l1685_168556

-- Conditions
variable (A B C D E F : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F]
variables (triangleABC : Triangle A B C) (triangleDEF : Triangle D E F)

-- Congruence of triangles
axiom congruent_triangles : triangleABC ≌ triangleDEF

-- Proving incorrect statement
theorem incorrect_statement : ¬ (AB = EF) := by
  sorry

end incorrect_statement_l1685_168556


namespace sum_of_digits_d_l1685_168567

theorem sum_of_digits_d (d : ℕ) (exchange_rate : 10 * d / 7 - 60 = d) : 
  (d = 140) -> (Nat.digits 10 140).sum = 5 :=
by
  sorry

end sum_of_digits_d_l1685_168567


namespace avg_height_and_variance_correct_l1685_168584

noncomputable def avg_height_and_variance
  (total_students : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (boys_avg_height : ℕ)
  (boys_variance : ℕ)
  (girls_avg_height : ℕ)
  (girls_variance : ℕ) : (ℕ × ℕ) := 
  let total_students := 300
  let boys := 180
  let girls := 120
  let boys_avg_height := 170
  let boys_variance := 14
  let girls_avg_height := 160
  let girls_variance := 24
  let avg_height := (boys * boys_avg_height + girls * girls_avg_height) / total_students 
  let variance := (boys * (boys_variance + (boys_avg_height - avg_height) ^ 2) 
                    + girls * (girls_variance + (girls_avg_height - avg_height) ^ 2)) / total_students
  (avg_height, variance)

theorem avg_height_and_variance_correct:
   avg_height_and_variance 300 180 120 170 14 160 24 = (166, 42) := 
  by {
    sorry
  }

end avg_height_and_variance_correct_l1685_168584


namespace find_radius_l1685_168528

noncomputable def sphere (center : ℝ × ℝ × ℝ) (radius : ℝ) : Prop := sorry

theorem find_radius (C1 : ℝ × ℝ × ℝ) (r1 : ℝ) (C2 : ℝ × ℝ × ℝ) (r : ℝ) :
  C1 = (3, 5, 0) →
  r1 = 2 →
  C2 = (0, 5, -8) →
  (sphere ((3, 5, -8) : ℝ × ℝ × ℝ) (2 * Real.sqrt 17)) →
  r = Real.sqrt 59 :=
by
  intros h1 h2 h3 h4
  sorry

end find_radius_l1685_168528


namespace card_probability_multiple_l1685_168502

def is_multiple_of (n k : ℕ) : Prop := k > 0 ∧ n % k = 0

def count_multiples (n k : ℕ) : ℕ :=
  if k = 0 then 0 else n / k

def inclusion_exclusion (a b c : ℕ) (n : ℕ) : ℕ :=
  (count_multiples n a) + (count_multiples n b) + (count_multiples n c) - 
  (count_multiples n (Nat.lcm a b)) - (count_multiples n (Nat.lcm a c)) - 
  (count_multiples n (Nat.lcm b c)) + 
  count_multiples n (Nat.lcm a (Nat.lcm b c))

theorem card_probability_multiple (n : ℕ) 
  (a b c : ℕ) (hne : n ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (inclusion_exclusion a b c n) / n = 47 / 100 := by
  sorry

end card_probability_multiple_l1685_168502


namespace number_of_men_in_first_group_l1685_168557

-- Definitions for the conditions
def rate_of_work (men : ℕ) (length : ℕ) (days : ℕ) : ℕ :=
  length / days / men

def work_rate_first_group (M : ℕ) : ℕ :=
  rate_of_work M 48 2

def work_rate_second_group : ℕ :=
  rate_of_work 2 36 3

theorem number_of_men_in_first_group (M : ℕ) 
  (h₁ : work_rate_first_group M = 24)
  (h₂ : work_rate_second_group = 12) :
  M = 4 :=
  sorry

end number_of_men_in_first_group_l1685_168557


namespace expression_evaluates_at_1_l1685_168530

variable (x : ℚ)

def original_expr (x : ℚ) : ℚ := (x + 2) / (x - 3)

def substituted_expr (x : ℚ) : ℚ :=
  (original_expr (original_expr x) + 2) / (original_expr (original_expr x) - 3)

theorem expression_evaluates_at_1 :
  substituted_expr 1 = -1 / 9 :=
by
  sorry

end expression_evaluates_at_1_l1685_168530


namespace ellipse_foci_distance_l1685_168518

noncomputable def distance_between_foci : ℝ := 2 * Real.sqrt 29

theorem ellipse_foci_distance : 
  ∀ (x y : ℝ), 
  (Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 25) → 
  distance_between_foci = 2 * Real.sqrt 29 := 
by
  intros x y h
  -- proof goes here (skipped)
  sorry

end ellipse_foci_distance_l1685_168518


namespace other_diagonal_length_l1685_168500

theorem other_diagonal_length (d2 : ℝ) (A : ℝ) (d1 : ℝ) 
  (h1 : d2 = 120) 
  (h2 : A = 4800) 
  (h3 : A = (d1 * d2) / 2) : d1 = 80 :=
by
  sorry

end other_diagonal_length_l1685_168500


namespace david_wins_2011th_even_l1685_168564

theorem david_wins_2011th_even :
  ∃ n : ℕ, (∃ k : ℕ, k = 2011 ∧ n = 2 * k) ∧ (∀ a b : ℕ, a < b → a + b < b * a) ∧ (n % 2 = 0) := 
sorry

end david_wins_2011th_even_l1685_168564


namespace slope_angle_135_l1685_168574

theorem slope_angle_135 (x y : ℝ) : 
  (∃ (m b : ℝ), 3 * x + 3 * y + 1 = 0 ∧ y = m * x + b ∧ m = -1) ↔ 
  (∃ α : ℝ, 0 ≤ α ∧ α < 180 ∧ Real.tan α = -1 ∧ α = 135) :=
sorry

end slope_angle_135_l1685_168574


namespace identify_incorrect_proposition_l1685_168596

-- Definitions based on problem conditions
def propositionA : Prop :=
  (∀ x : ℝ, (x^2 - 3*x + 2 = 0 → x = 1) ↔ (x ≠ 1 → x^2 - 3*x + 2 ≠ 0))

def propositionB : Prop :=
  (¬ (∃ x : ℝ, x^2 + x + 1 = 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≠ 0)

def propositionD (x : ℝ) : Prop :=
  (x > 2 → x^2 - 3*x + 2 > 0) ∧ (¬(x > 2) → ¬(x^2 - 3*x + 2 > 0))

-- Proposition C is given to be incorrect in the problem
def propositionC (p q : Prop) : Prop := ¬ (p ∧ q) → ¬p ∧ ¬q

theorem identify_incorrect_proposition (p q : Prop) : 
  (propositionA ∧ propositionB ∧ (∀ x : ℝ, propositionD x)) → 
  ¬ (propositionC p q) :=
by
  intros
  -- We know proposition C is false based on the problem's solution
  sorry

end identify_incorrect_proposition_l1685_168596


namespace original_amount_of_water_l1685_168555

variable {W : ℝ} -- Assume W is a real number representing the original amount of water

theorem original_amount_of_water (h1 : 30 * 0.02 = 0.6) (h2 : 0.6 = 0.06 * W) : W = 10 :=
by
  sorry

end original_amount_of_water_l1685_168555


namespace moneySpentOnPaintbrushes_l1685_168531

def totalExpenditure := 90
def costOfCanvases := 40
def costOfPaints := costOfCanvases / 2
def costOfEasel := 15
def costOfOthers := costOfCanvases + costOfPaints + costOfEasel

theorem moneySpentOnPaintbrushes : totalExpenditure - costOfOthers = 15 := by
  sorry

end moneySpentOnPaintbrushes_l1685_168531


namespace arithmetic_sequence_solution_geometric_sequence_solution_l1685_168504

-- Problem 1: Arithmetic sequence
noncomputable def arithmetic_general_term (n : ℕ) : ℕ := 30 - 3 * n
noncomputable def arithmetic_sum_terms (n : ℕ) : ℝ := -1.5 * n^2 + 28.5 * n

theorem arithmetic_sequence_solution (n : ℕ) (a8 a10 : ℕ) (sequence : ℕ → ℝ) :
  a8 = 6 → a10 = 0 → (sequence n = arithmetic_general_term n) ∧ (sequence n = arithmetic_sum_terms n) ∧ (n = 9 ∨ n = 10) := 
sorry

-- Problem 2: Geometric sequence
noncomputable def geometric_general_term (n : ℕ) : ℝ := 2^(n-2)
noncomputable def geometric_sum_terms (n : ℕ) : ℝ := 2^(n-1) - 0.5

theorem geometric_sequence_solution (n : ℕ) (a1 a4 : ℝ) (sequence : ℕ → ℝ):
  a1 = 0.5 → a4 = 4 → (sequence n = geometric_general_term n) ∧ (sequence n = geometric_sum_terms n) := 
sorry

end arithmetic_sequence_solution_geometric_sequence_solution_l1685_168504


namespace no_integral_solutions_l1685_168568

theorem no_integral_solutions : ∀ (x : ℤ), x^5 - 31 * x + 2015 ≠ 0 :=
by
  sorry

end no_integral_solutions_l1685_168568


namespace mandy_bike_time_l1685_168538

-- Definitions of the ratios and time spent on yoga
def ratio_gym_bike : ℕ × ℕ := (2, 3)
def ratio_yoga_exercise : ℕ × ℕ := (2, 3)
def time_yoga : ℕ := 20

-- Theorem stating that Mandy will spend 18 minutes riding her bike
theorem mandy_bike_time (r_gb : ℕ × ℕ) (r_ye : ℕ × ℕ) (t_y : ℕ) 
  (h_rgb : r_gb = (2, 3)) (h_rye : r_ye = (2, 3)) (h_ty : t_y = 20) : 
  let t_e := (r_ye.snd * t_y) / r_ye.fst
  let t_part := t_e / (r_gb.fst + r_gb.snd)
  t_part * r_gb.snd = 18 := sorry

end mandy_bike_time_l1685_168538


namespace incorrect_regression_statement_incorrect_statement_proof_l1685_168591

-- Define the regression equation and the statement about y and x
def regression_equation (x : ℝ) : ℝ := 3 - 5 * x

-- Proof statement: given the regression equation, show that when x increases by one unit, y decreases by 5 units on average
theorem incorrect_regression_statement : 
  (regression_equation (x + 1) = regression_equation x + (-5)) :=
by sorry

-- Proof statement: prove that the statement "when the variable x increases by one unit, y increases by 5 units on average" is incorrect
theorem incorrect_statement_proof :
  ¬ (regression_equation (x + 1) = regression_equation x + 5) :=
by sorry  

end incorrect_regression_statement_incorrect_statement_proof_l1685_168591


namespace simplify_expression_l1685_168507

variable (a : ℚ)
def expression := ((a + 3) / (a - 1) - 1 / (a - 1)) / ((a^2 + 4 * a + 4) / (a^2 - a))

theorem simplify_expression (h : a = 3) : expression a = 3 / 5 :=
by
  rw [h]
  -- additional simplifications would typically go here if the steps were spelled out
  sorry

end simplify_expression_l1685_168507


namespace students_above_90_l1685_168566

theorem students_above_90 (total_students : ℕ) (above_90_chinese : ℕ) (above_90_math : ℕ)
  (all_above_90_at_least_one_subject : total_students = 50 ∧ above_90_chinese = 33 ∧ above_90_math = 38 ∧ 
    ∀ (n : ℕ), n < total_students → (n < above_90_chinese ∨ n < above_90_math)) :
  (above_90_chinese + above_90_math - total_students) = 21 :=
by
  sorry

end students_above_90_l1685_168566


namespace base_k_for_repeating_series_equals_fraction_l1685_168505

-- Define the fraction 5/29
def fraction := 5 / 29

-- Define the repeating series in base k
def repeating_series (k : ℕ) : ℚ :=
  (1 / k) / (1 - 1 / k^2) + (3 / k^2) / (1 - 1 / k^2)

-- State the problem
theorem base_k_for_repeating_series_equals_fraction (k : ℕ) (hk1 : 0 < k) (hk2 : k ≠ 1):
  repeating_series k = fraction ↔ k = 8 := sorry

end base_k_for_repeating_series_equals_fraction_l1685_168505


namespace nathan_has_83_bananas_l1685_168533

def nathan_bananas (bunches_eight bananas_eight bunches_seven bananas_seven: Nat) : Nat :=
  bunches_eight * bananas_eight + bunches_seven * bananas_seven

theorem nathan_has_83_bananas (h1 : bunches_eight = 6) (h2 : bananas_eight = 8) (h3 : bunches_seven = 5) (h4 : bananas_seven = 7) : 
  nathan_bananas bunches_eight bananas_eight bunches_seven bananas_seven = 83 := by
  sorry

end nathan_has_83_bananas_l1685_168533


namespace cubic_sum_identity_l1685_168599

theorem cubic_sum_identity
  (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : ab + ac + bc = -3)
  (h3 : abc = 9) :
  a^3 + b^3 + c^3 = 22 :=
by
  sorry

end cubic_sum_identity_l1685_168599


namespace central_angle_unit_circle_l1685_168571

theorem central_angle_unit_circle :
  ∀ (θ : ℝ), (∃ (A : ℝ), A = 1 ∧ (A = 1 / 2 * θ)) → θ = 2 :=
by
  intro θ
  rintro ⟨A, hA1, hA2⟩
  sorry

end central_angle_unit_circle_l1685_168571


namespace carbonic_acid_formation_l1685_168524

-- Definition of amounts of substances involved
def moles_CO2 : ℕ := 3
def moles_H2O : ℕ := 3

-- Stoichiometric condition derived from the equation CO2 + H2O → H2CO3
def stoichiometric_ratio (a b c : ℕ) : Prop := (a = b) ∧ (a = c)

-- The main statement to prove
theorem carbonic_acid_formation : 
  stoichiometric_ratio moles_CO2 moles_H2O 3 :=
by
  sorry

end carbonic_acid_formation_l1685_168524


namespace min_additional_trains_needed_l1685_168561

-- Definitions
def current_trains : ℕ := 31
def trains_per_row : ℕ := 8
def smallest_num_additional_trains (current : ℕ) (per_row : ℕ) : ℕ :=
  let next_multiple := ((current + per_row - 1) / per_row) * per_row
  next_multiple - current

-- Theorem
theorem min_additional_trains_needed :
  smallest_num_additional_trains current_trains trains_per_row = 1 :=
by
  sorry

end min_additional_trains_needed_l1685_168561


namespace transformed_mean_stddev_l1685_168532

variables (n : ℕ) (x : Fin n → ℝ)

-- Given conditions
def mean_is_4 (mean : ℝ) : Prop :=
  mean = 4

def stddev_is_7 (stddev : ℝ) : Prop :=
  stddev = 7

-- Definitions for transformations and the results
def transformed_mean (mean : ℝ) : ℝ :=
  3 * mean + 2

def transformed_stddev (stddev : ℝ) : ℝ :=
  3 * stddev

-- The proof problem
theorem transformed_mean_stddev (mean stddev : ℝ) 
  (h_mean : mean_is_4 mean) 
  (h_stddev : stddev_is_7 stddev) :
  transformed_mean mean = 14 ∧ transformed_stddev stddev = 21 :=
by
  rw [h_mean, h_stddev]
  unfold transformed_mean transformed_stddev
  rw [← h_mean, ← h_stddev]
  sorry

end transformed_mean_stddev_l1685_168532


namespace rectangle_problem_l1685_168513

def rectangle_perimeter (L B : ℕ) : ℕ :=
  2 * (L + B)

theorem rectangle_problem (L B : ℕ) (h1 : L - B = 23) (h2 : L * B = 2520) : rectangle_perimeter L B = 206 := by
  sorry

end rectangle_problem_l1685_168513


namespace initial_mixture_volume_l1685_168594

theorem initial_mixture_volume (x : ℝ) (hx1 : 0.10 * x + 10 = 0.28 * (x + 10)) : x = 40 :=
by
  sorry

end initial_mixture_volume_l1685_168594


namespace find_b_l1685_168506

variable (a b c : ℕ)
variable (h1 : (a + b + c) / 3 = 45)
variable (h2 : (a + b) / 2 = 40)
variable (h3 : (b + c) / 2 = 43)

theorem find_b : b = 31 := sorry

end find_b_l1685_168506


namespace sum_of_cubes_l1685_168542

theorem sum_of_cubes (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + ac + bc = 3) (h3 : abc = 5) : a^3 + b^3 + c^3 = 15 :=
by
  sorry

end sum_of_cubes_l1685_168542


namespace abs_div_inequality_l1685_168581

theorem abs_div_inequality (x : ℝ) : 
  (|-((x+1)/x)| > (x+1)/x) ↔ (-1 < x ∧ x < 0) :=
sorry

end abs_div_inequality_l1685_168581


namespace tangent_line_eq_l1685_168536

/-- The equation of the tangent line to the curve y = 2x * tan x at the point x = π/4 is 
    (2 + π/2) * x - y - π^2/4 = 0. -/
theorem tangent_line_eq : ∀ x y : ℝ, 
  (y = 2 * x * Real.tan x) →
  (x = Real.pi / 4) →
  ((2 + Real.pi / 2) * x - y - Real.pi^2 / 4 = 0) :=
by
  intros x y h_curve h_point
  sorry

end tangent_line_eq_l1685_168536


namespace find_r_l1685_168548

def cubic_function (p q r x : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

theorem find_r (p q r : ℝ) (h1 : cubic_function p q r (-1) = 0) :
  r = p - 2 :=
sorry

end find_r_l1685_168548


namespace inner_tetrahedron_volume_ratio_l1685_168541

noncomputable def volume_ratio_of_tetrahedrons (s : ℝ) : ℝ :=
  let V_original := (s^3 * Real.sqrt 2) / 12
  let a := (Real.sqrt 6 / 9) * s
  let V_inner := (a^3 * Real.sqrt 2) / 12
  V_inner / V_original

theorem inner_tetrahedron_volume_ratio {s : ℝ} (hs : s > 0) : volume_ratio_of_tetrahedrons s = 1 / 243 :=
by
  sorry

end inner_tetrahedron_volume_ratio_l1685_168541


namespace cricket_run_rate_l1685_168586

theorem cricket_run_rate (run_rate_first_10_overs : ℝ) (target : ℝ) (overs_first_phase : ℕ) (overs_remaining : ℕ) :
  run_rate_first_10_overs = 4.6 → target = 282 → overs_first_phase = 10 → overs_remaining = 40 →
  (target - run_rate_first_10_overs * overs_first_phase) / overs_remaining = 5.9 :=
by
  intros
  sorry

end cricket_run_rate_l1685_168586


namespace base8_problem_l1685_168521

/--
Let A, B, and C be non-zero and distinct digits in base 8 such that
ABC_8 + BCA_8 + CAB_8 = AAA0_8 and A + B = 2C.
Prove that B + C = 14 in base 8.
-/
theorem base8_problem (A B C : ℕ) 
    (h1 : A > 0 ∧ B > 0 ∧ C > 0)
    (h2 : A < 8 ∧ B < 8 ∧ C < 8)
    (h3 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
    (bcd_sum : (8^2 * A + 8 * B + C) + (8^2 * B + 8 * C + A) + (8^2 * C + 8 * A + B) 
        = 8^3 * A + 8^2 * A + 8 * A)
    (sum_condition : A + B = 2 * C) :
    B + C = A + B := by {
  sorry
}

end base8_problem_l1685_168521


namespace distinct_stone_arrangements_l1685_168563

-- Define the set of 12 unique stones
def stones := Finset.range 12

-- Define the number of unique placements without considering symmetries
def placements : ℕ := stones.card.factorial

-- Define the number of symmetries (6 rotations and 6 reflections)
def symmetries : ℕ := 12

-- The total number of distinct configurations accounting for symmetries
def distinct_arrangements : ℕ := placements / symmetries

-- The main theorem stating the number of distinct arrangements
theorem distinct_stone_arrangements : distinct_arrangements = 39916800 := by 
  sorry

end distinct_stone_arrangements_l1685_168563


namespace seventh_term_value_l1685_168570

theorem seventh_term_value (a d : ℤ) (h1 : a = 12) (h2 : a + 3 * d = 18) : a + 6 * d = 24 := 
by
  sorry

end seventh_term_value_l1685_168570


namespace sabina_loan_l1685_168590

-- Define the conditions
def tuition_per_year : ℕ := 30000
def living_expenses_per_year : ℕ := 12000
def duration : ℕ := 4
def sabina_savings : ℕ := 10000
def grant_first_two_years_percent : ℕ := 40
def grant_last_two_years_percent : ℕ := 30
def scholarship_percent : ℕ := 20

-- Calculate total tuition for 4 years
def total_tuition : ℕ := tuition_per_year * duration

-- Calculate total living expenses for 4 years
def total_living_expenses : ℕ := living_expenses_per_year * duration

-- Calculate total cost
def total_cost : ℕ := total_tuition + total_living_expenses

-- Calculate grant coverage
def grant_first_two_years : ℕ := (grant_first_two_years_percent * tuition_per_year / 100) * 2
def grant_last_two_years : ℕ := (grant_last_two_years_percent * tuition_per_year / 100) * 2
def total_grant_coverage : ℕ := grant_first_two_years + grant_last_two_years

-- Calculate scholarship savings
def annual_scholarship_savings : ℕ := living_expenses_per_year * scholarship_percent / 100
def total_scholarship_savings : ℕ := annual_scholarship_savings * (duration - 1)

-- Calculate total reductions
def total_reductions : ℕ := total_grant_coverage + total_scholarship_savings + sabina_savings

-- Calculate the total loan needed
def total_loan_needed : ℕ := total_cost - total_reductions

theorem sabina_loan : total_loan_needed = 108800 := by
  sorry

end sabina_loan_l1685_168590


namespace circumscribed_triangle_area_relation_l1685_168537

theorem circumscribed_triangle_area_relation
  (a b c D E F : ℝ)
  (h₁ : a = 18) (h₂ : b = 24) (h₃ : c = 30)
  (triangle_right : a^2 + b^2 = c^2)
  (triangle_area : (1/2) * a * b = 216)
  (circle_area : π * (c / 2)^2 = 225 * π)
  (non_triangle_areas : D + E + 216 = F) :
  D + E + 216 = F :=
by
  sorry

end circumscribed_triangle_area_relation_l1685_168537


namespace allison_rolls_greater_probability_l1685_168529

theorem allison_rolls_greater_probability :
  let allison_roll : ℕ := 6
  let charlie_prob_less_6 := 5 / 6
  let mia_prob_rolls_3 := 4 / 6
  let combined_prob := charlie_prob_less_6 * (mia_prob_rolls_3)
  combined_prob = 5 / 9 := by
  sorry

end allison_rolls_greater_probability_l1685_168529
