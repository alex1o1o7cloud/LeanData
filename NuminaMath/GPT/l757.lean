import Mathlib

namespace interior_angle_of_arithmetic_sequence_triangle_l757_75703

theorem interior_angle_of_arithmetic_sequence_triangle :
  ∀ (α d : ℝ), (α - d) + α + (α + d) = 180 → α = 60 :=
by 
  sorry

end interior_angle_of_arithmetic_sequence_triangle_l757_75703


namespace tree_height_increase_l757_75724

theorem tree_height_increase
  (initial_height : ℝ)
  (height_increase : ℝ)
  (h6 : ℝ) :
  initial_height = 4 →
  (0 ≤ height_increase) →
  height_increase * 6 + initial_height = (height_increase * 4 + initial_height) + 1 / 7 * (height_increase * 4 + initial_height) →
  height_increase = 2 / 5 :=
by
  intro h_initial h_nonneg h_eq
  sorry

end tree_height_increase_l757_75724


namespace number_of_green_hats_l757_75792

theorem number_of_green_hats (B G : ℕ) 
  (h1 : B + G = 85) 
  (h2 : 6 * B + 7 * G = 550) : 
  G = 40 := by
  sorry

end number_of_green_hats_l757_75792


namespace initial_amount_l757_75763

theorem initial_amount (x : ℝ) (h1 : x = (2*x - 10) / 2) (h2 : x = (4*x - 30) / 2) (h3 : 8*x - 70 = 0) : x = 8.75 :=
by
  sorry

end initial_amount_l757_75763


namespace find_equation_of_l_l757_75716

open Real

/-- Define the point M(2, 1) -/
def M : ℝ × ℝ := (2, 1)

/-- Define the original line equation x - 2y + 1 = 0 as a function -/
def line1 (x y : ℝ) : Prop := x - 2 * y + 1 = 0

/-- Define the line l that passes through M and is perpendicular to line1 -/
def line_l (x y : ℝ) : Prop := 2 * x + y - 5 = 0

/-- The theorem to be proven: the line l passing through M and perpendicular to line1 has the equation 2x + y - 5 = 0 -/
theorem find_equation_of_l (x y : ℝ)
  (hM : M = (2, 1))
  (hl_perpendicular : ∀ x y : ℝ, line1 x y → line_l y (-x / 2)) :
  line_l x y ↔ (x, y) = (2, 1) :=
by
  sorry

end find_equation_of_l_l757_75716


namespace tan_domain_l757_75727

theorem tan_domain (x : ℝ) : 
  (∃ (k : ℤ), x = k * Real.pi - Real.pi / 4) ↔ 
  ¬(∃ (k : ℤ), x = k * Real.pi - Real.pi / 4) :=
sorry

end tan_domain_l757_75727


namespace exponents_mod_7_l757_75700

theorem exponents_mod_7 : (2222 ^ 5555 + 5555 ^ 2222) % 7 = 0 := 
by 
  -- sorries here because no proof is needed as stated
  sorry

end exponents_mod_7_l757_75700


namespace prob_both_hit_prob_at_least_one_hits_l757_75766

variable (pA pB : ℝ)

-- Given conditions
def prob_A_hits : Prop := pA = 0.9
def prob_B_hits : Prop := pB = 0.8

-- Proof problems
theorem prob_both_hit (hA : prob_A_hits pA) (hB : prob_B_hits pB) : 
  pA * pB = 0.72 := 
  sorry

theorem prob_at_least_one_hits (hA : prob_A_hits pA) (hB : prob_B_hits pB) : 
  1 - (1 - pA) * (1 - pB) = 0.98 := 
  sorry

end prob_both_hit_prob_at_least_one_hits_l757_75766


namespace num_adults_attended_l757_75782

-- Definitions for the conditions
def ticket_price_adult : ℕ := 26
def ticket_price_child : ℕ := 13
def num_children : ℕ := 28
def total_revenue : ℕ := 5122

-- The goal is to prove the number of adults who attended the show
theorem num_adults_attended :
  ∃ (A : ℕ), A * ticket_price_adult + num_children * ticket_price_child = total_revenue ∧ A = 183 :=
by
  sorry

end num_adults_attended_l757_75782


namespace weight_of_replaced_person_l757_75713

variable (average_weight_increase : ℝ)
variable (num_persons : ℝ)
variable (weight_new_person : ℝ)

theorem weight_of_replaced_person 
    (h1 : average_weight_increase = 2.5) 
    (h2 : num_persons = 10) 
    (h3 : weight_new_person = 90)
    : ∃ weight_replaced : ℝ, weight_replaced = 65 := 
by
  sorry

end weight_of_replaced_person_l757_75713


namespace surface_area_of_cube_l757_75778

theorem surface_area_of_cube (V : ℝ) (H : V = 125) : ∃ A : ℝ, A = 25 :=
by
  sorry

end surface_area_of_cube_l757_75778


namespace mark_money_l757_75770

theorem mark_money (M : ℝ) 
  (h1 : (1 / 2) * M + 14 + (1 / 3) * M + 16 + (1 / 4) * M + 18 = M) : 
  M = 576 := 
sorry

end mark_money_l757_75770


namespace jerrys_breakfast_calories_l757_75728

-- Define the constants based on the conditions
def pancakes : ℕ := 6
def calories_per_pancake : ℕ := 120
def strips_of_bacon : ℕ := 2
def calories_per_strip_of_bacon : ℕ := 100
def calories_in_cereal : ℕ := 200

-- Define the total calories for each category
def total_calories_from_pancakes : ℕ := pancakes * calories_per_pancake
def total_calories_from_bacon : ℕ := strips_of_bacon * calories_per_strip_of_bacon
def total_calories_from_cereal : ℕ := calories_in_cereal

-- Define the total calories in the breakfast
def total_breakfast_calories : ℕ := 
  total_calories_from_pancakes + total_calories_from_bacon + total_calories_from_cereal

-- The theorem we need to prove
theorem jerrys_breakfast_calories : total_breakfast_calories = 1120 := by sorry

end jerrys_breakfast_calories_l757_75728


namespace linear_function_points_relation_l757_75768

theorem linear_function_points_relation :
  ∀ (y1 y2 : ℝ), 
  (y1 = -3 * 2 + 1) ∧ (y2 = -3 * 3 + 1) → y1 > y2 :=
by
  intro y1 y2
  intro h
  cases h
  sorry

end linear_function_points_relation_l757_75768


namespace apple_price_36_kgs_l757_75780

theorem apple_price_36_kgs (l q : ℕ) 
  (H1 : ∀ n, n ≤ 30 → ∀ n', n' ≤ 30 → l * n' = 100)
  (H2 : 30 * l + 3 * q = 168) : 
  30 * l + 6 * q = 186 :=
by {
  sorry
}

end apple_price_36_kgs_l757_75780


namespace monomial_sum_l757_75729

variable {x y : ℝ}

theorem monomial_sum (a : ℝ) (h : -2 * x^2 * y^3 + 5 * x^(a-1) * y^3 = c * x^k * y^3) : a = 3 :=
  by
  sorry

end monomial_sum_l757_75729


namespace manny_paula_weight_l757_75711

   variable (m n o p : ℕ)

   -- Conditions
   variable (h1 : m + n = 320) 
   variable (h2 : n + o = 295) 
   variable (h3 : o + p = 310) 

   theorem manny_paula_weight : m + p = 335 :=
   by
     sorry
   
end manny_paula_weight_l757_75711


namespace solve_price_of_meat_l757_75755

def price_of_meat_per_ounce (x : ℕ) : Prop :=
  16 * x - 30 = 8 * x + 18

theorem solve_price_of_meat : ∃ x, price_of_meat_per_ounce x ∧ x = 6 :=
by
  sorry

end solve_price_of_meat_l757_75755


namespace Helen_taller_than_Amy_l757_75708

-- Definitions from conditions
def Angela_height : ℕ := 157
def Amy_height : ℕ := 150
def Helen_height := Angela_height - 4

-- Question as a theorem
theorem Helen_taller_than_Amy : Helen_height - Amy_height = 3 := by
  sorry

end Helen_taller_than_Amy_l757_75708


namespace fewest_printers_l757_75714

theorem fewest_printers (cost1 cost2 : ℕ) (h1 : cost1 = 375) (h2 : cost2 = 150) : 
  ∃ (n : ℕ), n = 2 + 5 :=
by
  have lcm_375_150 : Nat.lcm cost1 cost2 = 750 := sorry
  have n1 : 750 / 375 = 2 := sorry
  have n2 : 750 / 150 = 5 := sorry
  exact ⟨7, rfl⟩

end fewest_printers_l757_75714


namespace sum_of_numbers_in_row_l757_75709

theorem sum_of_numbers_in_row 
  (n : ℕ)
  (sum_eq : (n * (3 * n - 1)) / 2 = 20112) : 
  n = 1006 :=
sorry

end sum_of_numbers_in_row_l757_75709


namespace find_minimal_positive_n_l757_75786

-- Define the arithmetic sequence
def arithmetic_seq (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_seq (a1 d : ℤ) (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the conditions
variables (a1 d : ℤ)
axiom condition_1 : arithmetic_seq a1 d 11 / arithmetic_seq a1 d 10 < -1
axiom condition_2 : ∃ n : ℕ, ∀ k : ℕ, k ≤ n → sum_arithmetic_seq a1 d k ≤ sum_arithmetic_seq a1 d n

-- Prove the statement
theorem find_minimal_positive_n : ∃ n : ℕ, n = 19 ∧ sum_arithmetic_seq a1 d n = 0 ∧
  (∀ m : ℕ, 0 < sum_arithmetic_seq a1 d m ∧ sum_arithmetic_seq a1 d m < sum_arithmetic_seq a1 d n) :=
sorry

end find_minimal_positive_n_l757_75786


namespace min_value_expr_l757_75752

theorem min_value_expr : 
  ∀ x y : ℝ, 3 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 10 * y ≥ -3 := 
sorry

end min_value_expr_l757_75752


namespace cone_radius_height_ratio_l757_75739

theorem cone_radius_height_ratio 
  (V : ℝ) (π : ℝ) (r h : ℝ)
  (circumference : ℝ) 
  (original_height : ℝ)
  (new_volume : ℝ)
  (volume_formula : V = (1/3) * π * r^2 * h)
  (radius_from_circumference : 2 * π * r = circumference)
  (base_circumference : circumference = 28 * π)
  (original_height_eq : original_height = 45)
  (new_volume_eq : new_volume = 441 * π) :
  (r / h) = 14 / 9 :=
by
  sorry

end cone_radius_height_ratio_l757_75739


namespace diagonal_of_rectangular_solid_l757_75721

-- Define the lengths of the edges
def a : ℝ := 2
def b : ℝ := 3
def c : ℝ := 4

-- Prove that the diagonal of the rectangular solid with edges a, b, and c is sqrt(29)
theorem diagonal_of_rectangular_solid (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) : (a^2 + b^2 + c^2) = 29 := 
by 
  rw [h1, h2, h3]
  norm_num

end diagonal_of_rectangular_solid_l757_75721


namespace cone_lateral_surface_area_l757_75756

theorem cone_lateral_surface_area (r l : ℝ) (h1 : r = 2) (h2 : l = 5) : 
    0.5 * (2 * Real.pi * r * l) = 10 * Real.pi := by
    sorry

end cone_lateral_surface_area_l757_75756


namespace min_value_expression_l757_75712

open Real

/-- The minimum value of (14 - x) * (8 - x) * (14 + x) * (8 + x) is -4356. -/
theorem min_value_expression (x : ℝ) : ∃ (a : ℝ), a = (14 - x) * (8 - x) * (14 + x) * (8 + x) ∧ a ≥ -4356 :=
by
  use -4356
  sorry

end min_value_expression_l757_75712


namespace fraction_under_11_is_one_third_l757_75759

def fraction_under_11 (T : ℕ) (fraction_above_11_under_13 : ℚ) (students_above_13 : ℕ) : ℚ :=
  let fraction_under_11 := 1 - (fraction_above_11_under_13 + students_above_13 / T)
  fraction_under_11

theorem fraction_under_11_is_one_third :
  fraction_under_11 45 (2/5) 12 = 1/3 :=
by
  sorry

end fraction_under_11_is_one_third_l757_75759


namespace calcium_carbonate_required_l757_75746

theorem calcium_carbonate_required (HCl_moles CaCO3_moles CaCl2_moles CO2_moles H2O_moles : ℕ) 
  (reaction_balanced : CaCO3_moles + 2 * HCl_moles = CaCl2_moles + CO2_moles + H2O_moles) 
  (HCl_moles_value : HCl_moles = 2) : CaCO3_moles = 1 :=
by sorry

end calcium_carbonate_required_l757_75746


namespace red_not_equal_blue_l757_75725

theorem red_not_equal_blue (total_cubes : ℕ) (red_cubes : ℕ) (blue_cubes : ℕ) (edge_length : ℕ)
  (total_surface_squares : ℕ) (max_red_squares : ℕ) :
  total_cubes = 27 →
  red_cubes = 9 →
  blue_cubes = 18 →
  edge_length = 3 →
  total_surface_squares = 6 * edge_length^2 →
  max_red_squares = 26 →
  ¬ (total_surface_squares = 2 * max_red_squares) :=
by
  intros
  sorry

end red_not_equal_blue_l757_75725


namespace largest_root_eq_l757_75718

theorem largest_root_eq : ∃ x, (∀ y, (abs (Real.cos (Real.pi * y) + y^3 - 3 * y^2 + 3 * y) = 3 - y^2 - 2 * y^3) → y ≤ x) ∧ x = 1 := sorry

end largest_root_eq_l757_75718


namespace parabola_focus_l757_75742

theorem parabola_focus : ∃ f : ℝ, 
  (∀ x : ℝ, (x^2 + (2*x^2 - f)^2 = (2*x^2 - (1/4 + f))^2)) ∧
  f = 1/8 := 
by
  sorry

end parabola_focus_l757_75742


namespace caterer_preparations_l757_75738

theorem caterer_preparations :
  let b_guests := 84
  let a_guests := (2/3) * b_guests
  let total_guests := b_guests + a_guests
  let extra_plates := 10
  let total_plates := total_guests + extra_plates

  let cherry_tomatoes_per_plate := 5
  let regular_asparagus_per_plate := 8
  let vegetarian_asparagus_per_plate := 6
  let larger_asparagus_per_plate := 12
  let larger_asparagus_portion_guests := 0.1 * total_plates

  let blueberries_per_plate := 15
  let raspberries_per_plate := 8
  let blackberries_per_plate := 10

  let cherry_tomatoes_needed := cherry_tomatoes_per_plate * total_plates

  let regular_portion_guests := 0.9 * total_plates
  let regular_asparagus_needed := regular_asparagus_per_plate * regular_portion_guests
  let larger_asparagus_needed := larger_asparagus_per_plate * larger_asparagus_portion_guests
  let asparagus_needed := regular_asparagus_needed + larger_asparagus_needed

  let blueberries_needed := blueberries_per_plate * total_plates
  let raspberries_needed := raspberries_per_plate * total_plates
  let blackberries_needed := blackberries_per_plate * total_plates

  cherry_tomatoes_needed = 750 ∧
  asparagus_needed = 1260 ∧
  blueberries_needed = 2250 ∧
  raspberries_needed = 1200 ∧
  blackberries_needed = 1500 :=
by
  -- Proof goes here
  sorry

end caterer_preparations_l757_75738


namespace measure_8_liters_with_buckets_l757_75776

theorem measure_8_liters_with_buckets (capacity_B10 capacity_B6 : ℕ) (B10_target : ℕ) (B10_initial B6_initial : ℕ) : 
  capacity_B10 = 10 ∧ capacity_B6 = 6 ∧ B10_target = 8 ∧ B10_initial = 0 ∧ B6_initial = 0 →
  ∃ (B10 B6 : ℕ), B10 = 8 ∧ (B10 ≥ 0 ∧ B10 ≤ capacity_B10) ∧ (B6 ≥ 0 ∧ B6 ≤ capacity_B6) :=
by
  sorry

end measure_8_liters_with_buckets_l757_75776


namespace reconstruct_right_triangle_l757_75777

theorem reconstruct_right_triangle (c d : ℝ) (hc : c > 0) (hd : d > 0) :
  ∃ A X Y: ℝ, (A ≠ X ∧ A ≠ Y ∧ X ≠ Y) ∧ 
  -- Right triangle with hypotenuse c
  (A - X) ^ 2 + (Y - X) ^ 2 = c ^ 2 ∧ 
  -- Difference of legs is d
  ∃ AY XY: ℝ, ((AY = abs (A - Y)) ∧ (XY = abs (Y - X)) ∧ (abs (AY - XY) = d)) := 
by
  sorry

end reconstruct_right_triangle_l757_75777


namespace john_took_away_oranges_l757_75787

-- Define the initial number of oranges Melissa had.
def initial_oranges : ℕ := 70

-- Define the number of oranges Melissa has left.
def oranges_left : ℕ := 51

-- Define the expected number of oranges John took away.
def oranges_taken : ℕ := 19

-- The theorem that needs to be proven.
theorem john_took_away_oranges :
  initial_oranges - oranges_left = oranges_taken :=
by
  sorry

end john_took_away_oranges_l757_75787


namespace lcm_factor_of_hcf_and_larger_number_l757_75753

theorem lcm_factor_of_hcf_and_larger_number (A B : ℕ) (hcf : ℕ) (hlarger : A = 450) (hhcf : hcf = 30) (hwrel : A % hcf = 0) : ∃ x y, x = 15 ∧ (A * B = hcf * x * y) :=
by
  sorry

end lcm_factor_of_hcf_and_larger_number_l757_75753


namespace population_change_over_3_years_l757_75799

-- Define the initial conditions
def annual_growth_rate := 0.09
def migration_rate_year1 := -0.01
def migration_rate_year2 := -0.015
def migration_rate_year3 := -0.02
def natural_disaster_rate := -0.03

-- Lemma stating the overall percentage increase in population over three years
theorem population_change_over_3_years :
  (1 + annual_growth_rate) * (1 + migration_rate_year1) * 
  (1 + annual_growth_rate) * (1 + migration_rate_year2) * 
  (1 + annual_growth_rate) * (1 + migration_rate_year3) * 
  (1 + natural_disaster_rate) = 1.195795 := 
sorry

end population_change_over_3_years_l757_75799


namespace maximize_k_l757_75796

open Real

theorem maximize_k (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : log x + log y = 0)
  (h₄ : ∀ x y : ℝ, 0 < x → 0 < y → k * (x + 2 * y) ≤ x^2 + 4 * y^2) : k ≤ sqrt 2 :=
sorry

end maximize_k_l757_75796


namespace max_k_constant_l757_75747

theorem max_k_constant : 
  (∃ k, (∀ (x y z : ℝ), 0 < x → 0 < y → 0 < z → 
  (x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y) <= k * Real.sqrt (x + y + z))) 
  ∧ k = Real.sqrt 6 / 2) :=
sorry

end max_k_constant_l757_75747


namespace insects_total_l757_75764

def total_insects (n_leaves : ℕ) (ladybugs_per_leaf : ℕ) 
                  (n_stones : ℕ) (ants_per_stone : ℕ) 
                  (total_bees : ℕ) (n_flowers : ℕ) : ℕ :=
  let num_ladybugs := n_leaves * ladybugs_per_leaf
  let num_ants := n_stones * ants_per_stone
  let num_bees := total_bees -- already given as total_bees
  num_ladybugs + num_ants + num_bees

theorem insects_total : total_insects 345 267 178 423 498 6 = 167967 :=
  by unfold total_insects; sorry

end insects_total_l757_75764


namespace largest_three_digit_multiple_of_8_and_sum_24_is_888_l757_75745

noncomputable def largest_three_digit_multiple_of_8_with_digit_sum_24 : ℕ :=
  888

theorem largest_three_digit_multiple_of_8_and_sum_24_is_888 :
  ∃ n : ℕ, (300 ≤ n ∧ n ≤ 999) ∧ (n % 8 = 0) ∧ ((n.digits 10).sum = 24) ∧ n = largest_three_digit_multiple_of_8_with_digit_sum_24 :=
by
  existsi 888
  sorry

end largest_three_digit_multiple_of_8_and_sum_24_is_888_l757_75745


namespace each_person_gets_4_roses_l757_75793

def ricky_roses_total : Nat := 40
def roses_stolen : Nat := 4
def people : Nat := 9
def remaining_roses : Nat := ricky_roses_total - roses_stolen
def roses_per_person : Nat := remaining_roses / people

theorem each_person_gets_4_roses : roses_per_person = 4 := by
  sorry

end each_person_gets_4_roses_l757_75793


namespace simplify_expression_l757_75706

theorem simplify_expression (x : ℝ) : 
  8 * x + 15 - 3 * x + 5 * 7 = 5 * x + 50 :=
by
  sorry

end simplify_expression_l757_75706


namespace original_book_price_l757_75717

theorem original_book_price (P : ℝ) (h : 0.85 * P * 1.40 = 476) : P = 476 / (0.85 * 1.40) :=
by
  sorry

end original_book_price_l757_75717


namespace mirella_read_more_pages_l757_75705

-- Define the number of books Mirella read
def num_purple_books := 8
def num_orange_books := 7
def num_blue_books := 5

-- Define the number of pages per book for each color
def pages_per_purple_book := 320
def pages_per_orange_book := 640
def pages_per_blue_book := 450

-- Calculate the total pages for each color
def total_purple_pages := num_purple_books * pages_per_purple_book
def total_orange_pages := num_orange_books * pages_per_orange_book
def total_blue_pages := num_blue_books * pages_per_blue_book

-- Calculate the combined total of orange and blue pages
def total_orange_blue_pages := total_orange_pages + total_blue_pages

-- Define the target value
def page_difference := 4170

-- State the theorem to prove
theorem mirella_read_more_pages :
  total_orange_blue_pages - total_purple_pages = page_difference := by
  sorry

end mirella_read_more_pages_l757_75705


namespace find_y_at_neg3_l757_75702

noncomputable def quadratic_solution (y x a b : ℝ) : Prop :=
  y = x ^ 2 + a * x + b

theorem find_y_at_neg3
    (a b : ℝ)
    (h1 : 1 + a + b = 2)
    (h2 : 4 - 2 * a + b = -1)
    : quadratic_solution 2 (-3) a b :=
by
  sorry

end find_y_at_neg3_l757_75702


namespace part1_part2_l757_75765

noncomputable def f (x : ℝ) := x * Real.log x
noncomputable def g (x : ℝ) := x^2 - 1

theorem part1 {x : ℝ} (h : 1 ≤ x) : f x ≤ (1 / 2) * g x := by
  sorry

theorem part2 {m : ℝ} : (∀ x, 1 ≤ x → f x - m * g x ≤ 0) → m ≥ (1 / 2) := by
  sorry

end part1_part2_l757_75765


namespace parabola_focus_directrix_distance_l757_75751

theorem parabola_focus_directrix_distance (a : ℝ) (h_pos : a > 0) (h_dist : 1 / (2 * 2 * a) = 1) : a = 1 / 4 :=
by
  sorry

end parabola_focus_directrix_distance_l757_75751


namespace gary_profit_l757_75734

theorem gary_profit :
  let total_flour := 8 -- pounds
  let cost_flour := 4 -- dollars
  let large_cakes_flour := 5 -- pounds
  let small_cakes_flour := 3 -- pounds
  let flour_per_large_cake := 0.75 -- pounds per large cake
  let flour_per_small_cake := 0.25 -- pounds per small cake
  let cost_additional_large := 1.5 -- dollars per large cake
  let cost_additional_small := 0.75 -- dollars per small cake
  let cost_baking_equipment := 10 -- dollars
  let revenue_per_large := 6.5 -- dollars per large cake
  let revenue_per_small := 2.5 -- dollars per small cake
  let num_large_cakes := 6 -- (from calculation: ⌊5 / 0.75⌋)
  let num_small_cakes := 12 -- (from calculation: 3 / 0.25)
  let cost_additional_ingredients := num_large_cakes * cost_additional_large + num_small_cakes * cost_additional_small
  let total_revenue := num_large_cakes * revenue_per_large + num_small_cakes * revenue_per_small
  let total_cost := cost_flour + cost_baking_equipment + cost_additional_ingredients
  let profit := total_revenue - total_cost
  profit = 37 := by
  sorry

end gary_profit_l757_75734


namespace find_m_l757_75720

variable {S : ℕ → ℤ}
variable {m : ℕ}

/-- Given the sequences conditions, the value of m is 5 --/
theorem find_m (h1 : S (m - 1) = -2) (h2 : S m = 0) (h3 : S (m + 1) = 3) (h4 : 2 ≤ m) : m = 5 :=
sorry

end find_m_l757_75720


namespace ex_ineq_l757_75743

theorem ex_ineq (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 40) :
  x + y = 2 + 2 * Real.sqrt 3 ∨ x + y = 2 - 2 * Real.sqrt 3 :=
by
  sorry

end ex_ineq_l757_75743


namespace f_zero_derivative_not_extremum_l757_75704

noncomputable def f (x : ℝ) : ℝ := x ^ 3

theorem f_zero_derivative_not_extremum (x : ℝ) : 
  deriv f 0 = 0 ∧ ∀ (y : ℝ), y ≠ 0 → (∃ δ > 0, ∀ z, abs (z - 0) < δ → (f z / z : ℝ) ≠ 0) :=
by
  sorry

end f_zero_derivative_not_extremum_l757_75704


namespace polynomial_remainder_l757_75726

theorem polynomial_remainder (P : Polynomial ℝ) (a : ℝ) :
  ∃ (Q : Polynomial ℝ) (r : ℝ), P = Q * (Polynomial.X - Polynomial.C a) + Polynomial.C r ∧ r = (P.eval a) :=
by
  sorry

end polynomial_remainder_l757_75726


namespace initial_investment_l757_75737

theorem initial_investment (b : ℝ) (t_b : ℝ) (t_a : ℝ) (ratio_profit : ℝ) (x : ℝ) :
  b = 36000 → t_b = 4.5 → t_a = 12 → ratio_profit = 2 →
  (x * t_a) / (b * t_b) = ratio_profit → x = 27000 := 
by
  intros hb ht_b ht_a hr hp
  rw [hb, ht_b, ht_a, hr] at hp
  sorry

end initial_investment_l757_75737


namespace certain_event_is_A_l757_75722

def isCertainEvent (event : Prop) : Prop := event

axiom event_A : Prop
axiom event_B : Prop
axiom event_C : Prop
axiom event_D : Prop

axiom event_A_is_certain : isCertainEvent event_A
axiom event_B_is_not_certain : ¬ isCertainEvent event_B
axiom event_C_is_impossible : ¬ event_C
axiom event_D_is_not_certain : ¬ isCertainEvent event_D

theorem certain_event_is_A : isCertainEvent event_A := by
  exact event_A_is_certain

end certain_event_is_A_l757_75722


namespace age_ordered_youngest_to_oldest_l757_75784

variable (M Q S : Nat)

theorem age_ordered_youngest_to_oldest 
  (h1 : M = Q ∨ S = Q)
  (h2 : M ≥ Q)
  (h3 : S ≤ Q) : S = Q ∧ M > Q :=
by 
  sorry

end age_ordered_youngest_to_oldest_l757_75784


namespace seating_arrangements_equal_600_l757_75785

-- Definitions based on the problem conditions
def number_of_people : Nat := 4
def number_of_chairs : Nat := 8
def consecutive_empty_seats : Nat := 3

-- Theorem statement
theorem seating_arrangements_equal_600
  (h_people : number_of_people = 4)
  (h_chairs : number_of_chairs = 8)
  (h_consecutive_empty_seats : consecutive_empty_seats = 3) :
  (∃ (arrangements : Nat), arrangements = 600) :=
sorry

end seating_arrangements_equal_600_l757_75785


namespace selling_price_correct_l757_75736

/-- Define the initial cost of the gaming PC. -/
def initial_pc_cost : ℝ := 1200

/-- Define the cost of the new video card. -/
def new_video_card_cost : ℝ := 500

/-- Define the total spending after selling the old card. -/
def total_spending : ℝ := 1400

/-- Define the selling price of the old card -/
def selling_price_of_old_card : ℝ := (initial_pc_cost + new_video_card_cost) - total_spending

/-- Prove that John sold the old card for $300. -/
theorem selling_price_correct : selling_price_of_old_card = 300 := by
  sorry

end selling_price_correct_l757_75736


namespace binary_to_decimal_l757_75741

theorem binary_to_decimal : 
  (0 * 2^0 + 1 * 2^1 + 0 * 2^2 + 0 * 2^3 + 1 * 2^4) = 18 := 
by
  -- The proof is skipped
  sorry

end binary_to_decimal_l757_75741


namespace azure_valley_skirts_l757_75767

variables (P S A : ℕ)

theorem azure_valley_skirts (h1 : P = 10) 
                           (h2 : P = S / 4) 
                           (h3 : S = 2 * A / 3) : 
  A = 60 :=
by sorry

end azure_valley_skirts_l757_75767


namespace area_enclosed_l757_75781

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.sin (x - Real.pi / 3)
noncomputable def area_between (a b : ℝ) (f g : ℝ → ℝ) : ℝ :=
  ∫ x in a..b, (g x - f x)

theorem area_enclosed (h₀ : 0 ≤ 2 * Real.pi) (h₁ : 2 * Real.pi ≤ 2 * Real.pi) :
  area_between (2 * Real.pi / 3) (5 * Real.pi / 3) f g = 2 :=
by 
  sorry

end area_enclosed_l757_75781


namespace semifinalists_count_l757_75715

theorem semifinalists_count (n : ℕ) (h : (n - 2) * (n - 3) * (n - 4) = 336) : n = 10 := 
by {
  sorry
}

end semifinalists_count_l757_75715


namespace switch_pairs_bound_l757_75783

theorem switch_pairs_bound (odd_blocks_n odd_blocks_prev : ℕ) 
  (switch_pairs_n switch_pairs_prev : ℕ)
  (H1 : switch_pairs_n = 2 * odd_blocks_n)
  (H2 : odd_blocks_n ≤ switch_pairs_prev) : 
  switch_pairs_n ≤ 2 * switch_pairs_prev :=
by
  sorry

end switch_pairs_bound_l757_75783


namespace probability_of_Q_section_l757_75748

theorem probability_of_Q_section (sections : ℕ) (Q_sections : ℕ) (h1 : sections = 6) (h2 : Q_sections = 2) :
  Q_sections / sections = 2 / 6 :=
by
  -- solution proof is skipped
  sorry

end probability_of_Q_section_l757_75748


namespace total_bill_l757_75732

variable (B : ℝ)
variable (h1 : 9 * (B / 10 + 3) = B)

theorem total_bill : B = 270 :=
by
  -- proof would go here
  sorry

end total_bill_l757_75732


namespace no_solution_for_floor_eq_l757_75775

theorem no_solution_for_floor_eq :
  ∀ s : ℝ, ¬ (⌊s⌋ + s = 15.6) :=
by sorry

end no_solution_for_floor_eq_l757_75775


namespace proof_problem_l757_75707

-- Defining the statement in Lean 4.

noncomputable def p : Prop :=
  ∀ x : ℝ, x > Real.sin x

noncomputable def neg_p : Prop :=
  ∃ x : ℝ, x ≤ Real.sin x

theorem proof_problem : ¬p ↔ neg_p := 
by sorry

end proof_problem_l757_75707


namespace coin_game_goal_l757_75730

theorem coin_game_goal (a b : ℕ) (h_diff : a ≤ 3 * b ∧ b ≤ 3 * a) (h_sum : (a + b) % 4 = 0) :
  ∃ x y p q : ℕ, (a + 2 * x - 2 * y = 3 * (b + 2 * p - 2 * q)) ∨ (a + 2 * y - 2 * x = 3 * (b + 2 * q - 2 * p)) :=
sorry

end coin_game_goal_l757_75730


namespace ming_dynasty_wine_problem_l757_75794

theorem ming_dynasty_wine_problem (x y : ℕ) (h1 : x + y = 19) (h2 : 3 * x + y / 3 = 33 ) : 
  (x = 10 ∧ y = 9) :=
by {
  sorry
}

end ming_dynasty_wine_problem_l757_75794


namespace proof_problem_l757_75750

noncomputable def red_balls : ℕ := 5
noncomputable def black_balls : ℕ := 2
noncomputable def total_balls : ℕ := red_balls + black_balls
noncomputable def draws : ℕ := 3

noncomputable def prob_red_ball := red_balls / total_balls
noncomputable def prob_black_ball := black_balls / total_balls

noncomputable def E_X : ℚ := (1/7) + 2*(4/7) + 3*(2/7)
noncomputable def E_Y : ℚ := 2*(1/7) + 1*(4/7) + 0*(2/7)
noncomputable def E_xi : ℚ := 3 * (5/7)

noncomputable def D_X : ℚ := (1 - 15/7) ^ 2 * (1/7) + (2 - 15/7) ^ 2 * (4/7) + (3 - 15/7) ^ 2 * (2/7)
noncomputable def D_Y : ℚ := (2 - 6/7) ^ 2 * (1/7) + (1 - 6/7) ^ 2 * (4/7) + (0 - 6/7) ^ 2 * (2/7)
noncomputable def D_xi : ℚ := 3 * (5/7) * (1 - 5/7)

theorem proof_problem :
  (E_X / E_Y = 5 / 2) ∧ 
  (D_X ≤ D_Y) ∧ 
  (E_X = E_xi) ∧ 
  (D_X < D_xi) :=
by {
  sorry
}

end proof_problem_l757_75750


namespace ellipse_eccentricity_l757_75754

noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

def ellipse_conditions (F1 B : ℝ × ℝ) (c b : ℝ) : Prop :=
  F1 = (-2, 0) ∧ B = (0, 1) ∧ c = 2 ∧ b = 1

theorem ellipse_eccentricity (F1 B : ℝ × ℝ) (c b a : ℝ)
  (h : ellipse_conditions F1 B c b) :
  eccentricity c a = 2 * Real.sqrt 5 / 5 := by
sorry

end ellipse_eccentricity_l757_75754


namespace choosing_top_cases_l757_75795

def original_tops : Nat := 2
def bought_tops : Nat := 4
def total_tops : Nat := original_tops + bought_tops

theorem choosing_top_cases : total_tops = 6 := by
  sorry

end choosing_top_cases_l757_75795


namespace fifth_boat_more_than_average_l757_75772

theorem fifth_boat_more_than_average :
  let total_people := 2 + 4 + 3 + 5 + 6
  let num_boats := 5
  let average_people := total_people / num_boats
  let fifth_boat := 6
  (fifth_boat - average_people) = 2 :=
by
  sorry

end fifth_boat_more_than_average_l757_75772


namespace ratio_simplified_l757_75758

theorem ratio_simplified (kids_meals : ℕ) (adult_meals : ℕ) (h1 : kids_meals = 70) (h2 : adult_meals = 49) : 
  ∃ (k a : ℕ), k = 10 ∧ a = 7 ∧ kids_meals / Nat.gcd kids_meals adult_meals = k ∧ adult_meals / Nat.gcd kids_meals adult_meals = a :=
by
  sorry

end ratio_simplified_l757_75758


namespace price_increase_ratio_l757_75797

theorem price_increase_ratio 
  (c : ℝ)
  (h1 : 351 = c * 1.30) :
  (c + 351) / c = 2.3 :=
sorry

end price_increase_ratio_l757_75797


namespace combined_height_l757_75771

/-- Given that Mr. Martinez is two feet taller than Chiquita and Chiquita is 5 feet tall, prove that their combined height is 12 feet. -/
theorem combined_height (h_chiquita : ℕ) (h_martinez : ℕ) 
  (h1 : h_chiquita = 5) (h2 : h_martinez = h_chiquita + 2) : 
  h_chiquita + h_martinez = 12 :=
by sorry

end combined_height_l757_75771


namespace rectangular_garden_length_l757_75761

theorem rectangular_garden_length (w l : ℕ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 900) : l = 300 :=
by
  sorry

end rectangular_garden_length_l757_75761


namespace simplify_fraction_sum_eq_zero_l757_75790

variable (a b c : ℝ)
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)
variable (hc : c ≠ 0)
variable (h : a + b + 2 * c = 0)

theorem simplify_fraction_sum_eq_zero :
  (1 / (b^2 + 4*c^2 - a^2) + 1 / (a^2 + 4*c^2 - b^2) + 1 / (a^2 + b^2 - 4*c^2)) = 0 :=
by sorry

end simplify_fraction_sum_eq_zero_l757_75790


namespace tip_count_proof_l757_75789

def initial_customers : ℕ := 29
def additional_customers : ℕ := 20
def customers_who_tipped : ℕ := 15
def total_customers : ℕ := initial_customers + additional_customers
def customers_didn't_tip : ℕ := total_customers - customers_who_tipped

theorem tip_count_proof : customers_didn't_tip = 34 :=
by
  -- This is a proof outline, not the actual proof.
  sorry

end tip_count_proof_l757_75789


namespace part_a_part_b_part_c_l757_75788

theorem part_a (θ : ℝ) (m : ℕ) : |Real.sin (m * θ)| ≤ m * |Real.sin θ| :=
sorry

theorem part_b (θ₁ θ₂ : ℝ) (m : ℕ) (hm_even : Even m) : 
  |Real.sin (m * θ₂) - Real.sin (m * θ₁)| ≤ m * |Real.sin (θ₂ - θ₁)| :=
sorry

theorem part_c (m : ℕ) (hm_odd : Odd m) : 
  ∃ θ₁ θ₂ : ℝ, |Real.sin (m * θ₂) - Real.sin (m * θ₁)| > m * |Real.sin (θ₂ - θ₁)| :=
sorry

end part_a_part_b_part_c_l757_75788


namespace prob_at_least_one_black_without_replacement_prob_exactly_one_black_with_replacement_l757_75735

-- Definitions for the conditions
def white_balls : ℕ := 4
def black_balls : ℕ := 2

-- Total number of balls
def total_balls : ℕ := white_balls + black_balls

-- Part (I): Without Replacement
theorem prob_at_least_one_black_without_replacement : 
  (20 - 4) / 20 = 4 / 5 :=
by sorry

-- Part (II): With Replacement
theorem prob_exactly_one_black_with_replacement : 
  (3 * 2 * 4 * 4) / (6 * 6 * 6) = 4 / 9 :=
by sorry

end prob_at_least_one_black_without_replacement_prob_exactly_one_black_with_replacement_l757_75735


namespace impossible_to_half_boys_sit_with_girls_l757_75779

theorem impossible_to_half_boys_sit_with_girls:
  ∀ (g b : ℕ), 
  (g + b = 30) → 
  (∃ k, g = 2 * k) →
  (∀ (d : ℕ), 2 * d = g) →
  ¬ ∃ m, (b = 2 * m) ∧ (∀ (d : ℕ), 2 * d = b) :=
by
  sorry

end impossible_to_half_boys_sit_with_girls_l757_75779


namespace dave_deleted_apps_l757_75760

-- Definitions based on problem conditions
def original_apps : Nat := 16
def remaining_apps : Nat := 5

-- Theorem statement for proving how many apps Dave deleted
theorem dave_deleted_apps : original_apps - remaining_apps = 11 :=
by
  sorry

end dave_deleted_apps_l757_75760


namespace find_f_k_l_l757_75733

noncomputable
def f : ℕ → ℕ := sorry

axiom f_condition_1 : f 1 = 1
axiom f_condition_2 : ∀ n : ℕ, 3 * f n * f (2 * n + 1) = f (2 * n) * (1 + 3 * f n)
axiom f_condition_3 : ∀ n : ℕ, f (2 * n) < 6 * f n

theorem find_f_k_l (k l : ℕ) (h : k < l) : 
  (f k + f l = 293) ↔ 
  ((k = 121 ∧ l = 4) ∨ (k = 118 ∧ l = 4) ∨ 
   (k = 109 ∧ l = 16) ∨ (k = 16 ∧ l = 109)) := 
by 
  sorry

end find_f_k_l_l757_75733


namespace inequality_abc_d_l757_75749

theorem inequality_abc_d (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (H1 : d ≥ a) (H2 : d ≥ b) (H3 : d ≥ c) : a * (d - b) + b * (d - c) + c * (d - a) ≤ d^2 :=
by
  sorry

end inequality_abc_d_l757_75749


namespace max_fourth_term_l757_75791

open Nat

/-- Constants representing the properties of the arithmetic sequence -/
axiom a : ℕ
axiom d : ℕ
axiom pos1 : a > 0
axiom pos2 : a + d > 0
axiom pos3 : a + 2 * d > 0
axiom pos4 : a + 3 * d > 0
axiom pos5 : a + 4 * d > 0
axiom sum_condition : 5 * a + 10 * d = 75

/-- Theorem stating the maximum fourth term of the arithmetic sequence -/
theorem max_fourth_term : a + 3 * d = 22 := sorry

end max_fourth_term_l757_75791


namespace greatest_points_for_top_teams_l757_75744

-- Definitions as per the conditions
def teams := 9 -- Number of teams
def games_per_pair := 2 -- Each team plays every other team twice
def points_win := 3 -- Points for a win
def points_draw := 1 -- Points for a draw
def points_loss := 0 -- Points for a loss

-- Total number of games played
def total_games := (teams * (teams - 1) / 2) * games_per_pair

-- Total points available in the tournament
def total_points := total_games * points_win

-- Given the conditions, prove that the greatest possible number of total points each of the top three teams can accumulate is 42.
theorem greatest_points_for_top_teams :
  ∃ k, (∀ A B C : ℕ, A = B ∧ B = C → A ≤ k) ∧ k = 42 :=
sorry

end greatest_points_for_top_teams_l757_75744


namespace Nancy_seeds_l757_75740

def big_garden_seeds : ℕ := 28
def small_gardens : ℕ := 6
def seeds_per_small_garden : ℕ := 4

def total_seeds : ℕ := big_garden_seeds + small_gardens * seeds_per_small_garden

theorem Nancy_seeds : total_seeds = 52 :=
by
  -- Proof here...
  sorry

end Nancy_seeds_l757_75740


namespace find_x_l757_75762

theorem find_x (x : ℝ) (h : (1 + x) / (5 + x) = 1 / 3) : x = 1 :=
sorry

end find_x_l757_75762


namespace Carrie_can_add_turnips_l757_75757

-- Define the variables and conditions
def potatoToTurnipRatio (potatoes turnips : ℕ) : ℚ :=
  potatoes / turnips

def pastPotato : ℕ := 5
def pastTurnip : ℕ := 2
def currentPotato : ℕ := 20
def allowedTurnipAddition : ℕ := 8

-- Define the main theorem to prove, given the conditions.
theorem Carrie_can_add_turnips (past_p_ratio : potatoToTurnipRatio pastPotato pastTurnip = 2.5)
                                : potatoToTurnipRatio currentPotato allowedTurnipAddition = 2.5 :=
sorry

end Carrie_can_add_turnips_l757_75757


namespace simple_interest_rate_l757_75719

theorem simple_interest_rate 
  (SI : ℝ) (P : ℝ) (T : ℝ) (SI_eq : SI = 260)
  (P_eq : P = 910) (T_eq : T = 4)
  (H : SI = P * R * T / 100) : 
  R = 26000 / 3640 := 
by
  sorry

end simple_interest_rate_l757_75719


namespace vacation_cost_proof_l757_75723

noncomputable def vacation_cost (C : ℝ) :=
  C / 5 - C / 8 = 120

theorem vacation_cost_proof {C : ℝ} (h : vacation_cost C) : C = 1600 :=
by
  sorry

end vacation_cost_proof_l757_75723


namespace solve_inequality_zero_solve_inequality_neg_solve_inequality_pos_l757_75774

variable (a x : ℝ)

def inequality (a x : ℝ) : Prop := (1 - a * x) ^ 2 < 1

theorem solve_inequality_zero : a = 0 → ¬∃ x, inequality a x := by
  sorry

theorem solve_inequality_neg (h : a < 0) : (∃ x, inequality a x) →
  ∀ x, inequality a x ↔ (a ≠ 0 ∧ (2 / a < x ∧ x < 0)) := by
  sorry

theorem solve_inequality_pos (h : a > 0) : (∃ x, inequality a x) →
  ∀ x, inequality a x ↔ (a ≠ 0 ∧ (0 < x ∧ x < 2 / a)) := by
  sorry

end solve_inequality_zero_solve_inequality_neg_solve_inequality_pos_l757_75774


namespace container_holds_slices_l757_75701

theorem container_holds_slices (x : ℕ) 
  (h1 : x > 1) 
  (h2 : x ≠ 332) 
  (h3 : x ≠ 166) 
  (h4 : x ∣ 332) :
  x = 83 := 
sorry

end container_holds_slices_l757_75701


namespace jelly_beans_correct_l757_75731

-- Define the constants and conditions
def sandra_savings : ℕ := 10
def mother_gift : ℕ := 4
def father_gift : ℕ := 2 * mother_gift
def total_amount : ℕ := sandra_savings + mother_gift + father_gift

def candy_cost : ℕ := 5 / 10 -- == 0.5
def jelly_bean_cost : ℕ := 2 / 10 -- == 0.2

def candies_bought : ℕ := 14
def money_spent_on_candies : ℕ := candies_bought * candy_cost

def remaining_money : ℕ := total_amount - money_spent_on_candies
def money_left : ℕ := 11

-- Prove the number of jelly beans bought is 20
def number_of_jelly_beans : ℕ :=
  (remaining_money - money_left) / jelly_bean_cost

theorem jelly_beans_correct : number_of_jelly_beans = 20 :=
sorry

end jelly_beans_correct_l757_75731


namespace infinite_bad_integers_l757_75773

theorem infinite_bad_integers (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ∃ᶠ n in at_top, (¬(n^b + 1) ∣ (a^n + 1)) :=
by
  sorry

end infinite_bad_integers_l757_75773


namespace squares_total_l757_75798

def number_of_squares (figure : Type) : ℕ := sorry

theorem squares_total (figure : Type) : number_of_squares figure = 38 := sorry

end squares_total_l757_75798


namespace vartan_recreation_percent_l757_75769

noncomputable def percent_recreation_week (last_week_wages current_week_wages current_week_recreation last_week_recreation : ℝ) : ℝ :=
  (current_week_recreation / current_week_wages) * 100

theorem vartan_recreation_percent 
  (W : ℝ) 
  (h1 : last_week_wages = W)  
  (h2 : last_week_recreation = 0.15 * W)
  (h3 : current_week_wages = 0.90 * W)
  (h4 : current_week_recreation = 1.80 * last_week_recreation) :
  percent_recreation_week last_week_wages current_week_wages current_week_recreation last_week_recreation = 30 :=
by
  sorry

end vartan_recreation_percent_l757_75769


namespace tan_of_angle_in_third_quadrant_l757_75710

theorem tan_of_angle_in_third_quadrant 
  (α : ℝ) 
  (h1 : α < -π / 2 ∧ α > -π) 
  (h2 : Real.sin α = -Real.sqrt 5 / 5) :
  Real.tan α = 1 / 2 := 
sorry

end tan_of_angle_in_third_quadrant_l757_75710
