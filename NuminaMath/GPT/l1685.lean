import Mathlib

namespace NUMINAMATH_GPT_spade_to_heart_l1685_168559

-- Definition for spade and heart can be abstract geometric shapes
structure Spade := (arcs_top: ℕ) (stem_bottom: ℕ)
structure Heart := (arcs_top: ℕ) (pointed_bottom: ℕ)

-- Condition: the spade symbol must be cut into three parts
def cut_spade (s: Spade) : List (ℕ × ℕ) :=
  [(s.arcs_top, 0), (0, s.stem_bottom), (0, s.stem_bottom)]

-- Define a function to verify if the rearranged parts form a heart
def can_form_heart (pieces: List (ℕ × ℕ)) : Prop :=
  pieces = [(1, 0), (0, 1), (0, 1)]

-- The theorem that the spade parts can form a heart
theorem spade_to_heart (s: Spade) (h: Heart):
  (cut_spade s) = [(s.arcs_top, 0), (0, s.stem_bottom), (0, s.stem_bottom)] →
  can_form_heart [(s.arcs_top, 0), (s.stem_bottom, 0), (s.stem_bottom, 0)] := 
by
  sorry


end NUMINAMATH_GPT_spade_to_heart_l1685_168559


namespace NUMINAMATH_GPT_ravi_nickels_l1685_168501

variables (n q d : ℕ)

-- Defining the conditions
def quarters (n : ℕ) : ℕ := n + 2
def dimes (q : ℕ) : ℕ := q + 4

-- Using these definitions to form the Lean theorem
theorem ravi_nickels : 
  ∃ n, q = quarters n ∧ d = dimes q ∧ 
  (0.05 * n + 0.25 * q + 0.10 * d : ℝ) = 3.50 ∧ n = 6 :=
sorry

end NUMINAMATH_GPT_ravi_nickels_l1685_168501


namespace NUMINAMATH_GPT_decrement_value_each_observation_l1685_168561

theorem decrement_value_each_observation 
  (n : ℕ) 
  (original_mean updated_mean : ℝ) 
  (n_pos : n = 50) 
  (original_mean_value : original_mean = 200)
  (updated_mean_value : updated_mean = 153) :
  (original_mean * n - updated_mean * n) / n = 47 :=
by
  sorry

end NUMINAMATH_GPT_decrement_value_each_observation_l1685_168561


namespace NUMINAMATH_GPT_missing_pieces_l1685_168508

-- Definitions based on the conditions.
def total_pieces : ℕ := 500
def border_pieces : ℕ := 75
def trevor_pieces : ℕ := 105
def joe_pieces : ℕ := 3 * trevor_pieces

-- Prove the number of missing pieces is 5.
theorem missing_pieces : total_pieces - (border_pieces + trevor_pieces + joe_pieces) = 5 := by
  sorry

end NUMINAMATH_GPT_missing_pieces_l1685_168508


namespace NUMINAMATH_GPT_min_employees_to_hire_l1685_168560

-- Definitions of the given conditions
def employees_cust_service : ℕ := 95
def employees_tech_support : ℕ := 80
def employees_both : ℕ := 30

-- The theorem stating the minimum number of new employees to hire
theorem min_employees_to_hire (n : ℕ) :
  n = (employees_cust_service - employees_both) 
    + (employees_tech_support - employees_both) 
    + employees_both → 
  n = 145 := sorry

end NUMINAMATH_GPT_min_employees_to_hire_l1685_168560


namespace NUMINAMATH_GPT_shorter_leg_of_right_triangle_with_hypotenuse_65_l1685_168518

theorem shorter_leg_of_right_triangle_with_hypotenuse_65 (a b : ℕ) (h : a^2 + b^2 = 65^2) : a = 16 ∨ b = 16 :=
by sorry

end NUMINAMATH_GPT_shorter_leg_of_right_triangle_with_hypotenuse_65_l1685_168518


namespace NUMINAMATH_GPT_alex_buys_15_pounds_of_wheat_l1685_168579

theorem alex_buys_15_pounds_of_wheat (w o : ℝ) (h1 : w + o = 30) (h2 : 72 * w + 36 * o = 1620) : w = 15 :=
by
  sorry

end NUMINAMATH_GPT_alex_buys_15_pounds_of_wheat_l1685_168579


namespace NUMINAMATH_GPT_find_central_angle_l1685_168520

noncomputable def sector := 
  {R : ℝ // R > 0}

noncomputable def central_angle (R : ℝ) : ℝ := 
  (6 - 2 * R) / R

theorem find_central_angle :
  ∃ α : ℝ, (α = 1 ∨ α = 4) ∧ 
  (∃ R : ℝ, 
    (2 * R + α * R = 6) ∧ 
    (1 / 2 * R^2 * α = 2)) := 
by {
  sorry
}

end NUMINAMATH_GPT_find_central_angle_l1685_168520


namespace NUMINAMATH_GPT_crayons_in_judahs_box_l1685_168537

theorem crayons_in_judahs_box (karen_crayons beatrice_crayons gilbert_crayons judah_crayons : ℕ)
  (h1 : karen_crayons = 128)
  (h2 : beatrice_crayons = karen_crayons / 2)
  (h3 : gilbert_crayons = beatrice_crayons / 2)
  (h4 : judah_crayons = gilbert_crayons / 4) :
  judah_crayons = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_crayons_in_judahs_box_l1685_168537


namespace NUMINAMATH_GPT_factor_poly1_factor_poly2_factor_poly3_l1685_168510

-- Define the three polynomial functions.
def poly1 (x : ℝ) : ℝ := 2 * x^4 - 2
def poly2 (x : ℝ) : ℝ := x^4 - 18 * x^2 + 81
def poly3 (y : ℝ) : ℝ := (y^2 - 1)^2 + 11 * (1 - y^2) + 24

-- Formulate the goals: proving that each polynomial equals its respective factored form.
theorem factor_poly1 (x : ℝ) : poly1 x = 2 * (x^2 + 1) * (x + 1) * (x - 1) :=
sorry

theorem factor_poly2 (x : ℝ) : poly2 x = (x + 3)^2 * (x - 3)^2 :=
sorry

theorem factor_poly3 (y : ℝ) : poly3 y = (y + 2) * (y - 2) * (y + 3) * (y - 3) :=
sorry

end NUMINAMATH_GPT_factor_poly1_factor_poly2_factor_poly3_l1685_168510


namespace NUMINAMATH_GPT_product_of_integers_with_cubes_sum_189_l1685_168585

theorem product_of_integers_with_cubes_sum_189 :
  ∃ a b : ℤ, a^3 + b^3 = 189 ∧ a * b = 20 :=
by
  -- The proof is omitted for brevity.
  sorry

end NUMINAMATH_GPT_product_of_integers_with_cubes_sum_189_l1685_168585


namespace NUMINAMATH_GPT_trig_identity_l1685_168584

theorem trig_identity (x : ℝ) (h0 : -3 * Real.pi / 2 < x) (h1 : x < -Real.pi) (h2 : Real.tan x = -3) :
  Real.sin x * Real.cos x = -3 / 10 :=
sorry

end NUMINAMATH_GPT_trig_identity_l1685_168584


namespace NUMINAMATH_GPT_total_lines_correct_l1685_168513

-- Define the shapes and their corresponding lines
def triangles := 12
def squares := 8
def pentagons := 4
def hexagons := 6
def octagons := 2

def triangle_sides := 3
def square_sides := 4
def pentagon_sides := 5
def hexagon_sides := 6
def octagon_sides := 8

def lines_in_triangles := triangles * triangle_sides
def lines_in_squares := squares * square_sides
def lines_in_pentagons := pentagons * pentagon_sides
def lines_in_hexagons := hexagons * hexagon_sides
def lines_in_octagons := octagons * octagon_sides

def shared_lines_ts := 5
def shared_lines_ph := 3
def shared_lines_ho := 1

def total_lines_triangles := lines_in_triangles - shared_lines_ts
def total_lines_squares := lines_in_squares - shared_lines_ts
def total_lines_pentagons := lines_in_pentagons - shared_lines_ph
def total_lines_hexagons := lines_in_hexagons - shared_lines_ph - shared_lines_ho
def total_lines_octagons := lines_in_octagons - shared_lines_ho

-- The statement to prove
theorem total_lines_correct :
  total_lines_triangles = 31 ∧
  total_lines_squares = 27 ∧
  total_lines_pentagons = 17 ∧
  total_lines_hexagons = 32 ∧
  total_lines_octagons = 15 :=
by sorry

end NUMINAMATH_GPT_total_lines_correct_l1685_168513


namespace NUMINAMATH_GPT_ruth_train_track_length_l1685_168589

theorem ruth_train_track_length (n : ℕ) (R : ℕ)
  (h_sean : 72 = 8 * n)
  (h_ruth : 72 = R * n) : 
  R = 8 :=
by
  sorry

end NUMINAMATH_GPT_ruth_train_track_length_l1685_168589


namespace NUMINAMATH_GPT_simplify_expression_l1685_168580

theorem simplify_expression (a : ℝ) (h : a > 0) : 
  (a^2 / (a * (a^3) ^ (1 / 2)) ^ (1 / 3)) = a^(7 / 6) :=
sorry

end NUMINAMATH_GPT_simplify_expression_l1685_168580


namespace NUMINAMATH_GPT_sum_of_squares_2222_l1685_168555

theorem sum_of_squares_2222 :
  ∀ (N : ℕ), (∃ (k : ℕ), N = 2 * 10^k - 1) → (∀ (a b : ℤ), N = a^2 + b^2 ↔ N = 2) :=
by sorry

end NUMINAMATH_GPT_sum_of_squares_2222_l1685_168555


namespace NUMINAMATH_GPT_zeros_of_shifted_function_l1685_168512

def f (x : ℝ) : ℝ := x^2 - 1

theorem zeros_of_shifted_function :
  {x : ℝ | f (x - 1) = 0} = {0, 2} :=
sorry

end NUMINAMATH_GPT_zeros_of_shifted_function_l1685_168512


namespace NUMINAMATH_GPT_simplify_expression_l1685_168556

theorem simplify_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^3 + b^3 = 3 * (a + b)) :
  (a / b) + (b / a) - (3 / (a * b)) = 1 := 
sorry

end NUMINAMATH_GPT_simplify_expression_l1685_168556


namespace NUMINAMATH_GPT_slices_served_yesterday_l1685_168572

theorem slices_served_yesterday
  (lunch_slices : ℕ)
  (dinner_slices : ℕ)
  (total_slices_today : ℕ)
  (h1 : lunch_slices = 7)
  (h2 : dinner_slices = 5)
  (h3 : total_slices_today = 12) :
  (total_slices_today - (lunch_slices + dinner_slices) = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_slices_served_yesterday_l1685_168572


namespace NUMINAMATH_GPT_cafeteria_pies_l1685_168581

theorem cafeteria_pies (total_apples initial_apples_per_pie held_out_apples : ℕ) (h : total_apples = 150) (g : held_out_apples = 24) (p : initial_apples_per_pie = 15) :
  ((total_apples - held_out_apples) / initial_apples_per_pie) = 8 :=
by
  -- problem-specific proof steps would go here
  sorry

end NUMINAMATH_GPT_cafeteria_pies_l1685_168581


namespace NUMINAMATH_GPT_smallest_number_div_by_225_with_digits_0_1_l1685_168571

theorem smallest_number_div_by_225_with_digits_0_1 :
  ∃ n : ℕ, (∀ d ∈ n.digits 10, d = 0 ∨ d = 1) ∧ 225 ∣ n ∧ (∀ m : ℕ, (∀ d ∈ m.digits 10, d = 0 ∨ d = 1) ∧ 225 ∣ m → n ≤ m) ∧ n = 11111111100 :=
sorry

end NUMINAMATH_GPT_smallest_number_div_by_225_with_digits_0_1_l1685_168571


namespace NUMINAMATH_GPT_original_list_length_l1685_168565

variable (n m : ℕ)   -- number of integers and the mean respectively
variable (l : List ℤ) -- the original list of integers

def mean (l : List ℤ) : ℚ :=
  (l.sum : ℚ) / l.length

-- Condition 1: Appending 25 increases mean by 3
def condition1 (l : List ℤ) : Prop :=
  mean (25 :: l) = mean l + 3

-- Condition 2: Appending -4 to the enlarged list decreases the mean by 1.5
def condition2 (l : List ℤ) : Prop :=
  mean (-4 :: 25 :: l) = mean (25 :: l) - 1.5

theorem original_list_length (l : List ℤ) (h1 : condition1 l) (h2 : condition2 l) : l.length = 4 := by
  sorry

end NUMINAMATH_GPT_original_list_length_l1685_168565


namespace NUMINAMATH_GPT_a_and_b_together_finish_in_40_days_l1685_168540

theorem a_and_b_together_finish_in_40_days (D : ℕ) 
    (W : ℕ)
    (day_with_b : ℕ)
    (remaining_days_a : ℕ)
    (a_alone_days : ℕ)
    (a_b_together : D = 40)
    (ha : (remaining_days_a = 15) ∧ (a_alone_days = 20) ∧ (day_with_b = 10))
    (work_done_total : 10 * (W / D) + 15 * (W / a_alone_days) = W) :
    D = 40 := 
    sorry

end NUMINAMATH_GPT_a_and_b_together_finish_in_40_days_l1685_168540


namespace NUMINAMATH_GPT_plumber_salary_percentage_l1685_168521

def salary_construction_worker : ℕ := 100
def salary_electrician : ℕ := 2 * salary_construction_worker
def total_salary_without_plumber : ℕ := 2 * salary_construction_worker + salary_electrician
def total_labor_cost : ℕ := 650
def salary_plumber : ℕ := total_labor_cost - total_salary_without_plumber
def percentage_salary_plumber_as_construction_worker (x y : ℕ) : ℕ := (x * 100) / y

theorem plumber_salary_percentage :
  percentage_salary_plumber_as_construction_worker salary_plumber salary_construction_worker = 250 :=
by 
  sorry

end NUMINAMATH_GPT_plumber_salary_percentage_l1685_168521


namespace NUMINAMATH_GPT_alice_book_payment_l1685_168534

/--
Alice is in the UK and wants to purchase a book priced at £25.
If one U.S. dollar is equivalent to £0.75, 
then Alice needs to pay 33.33 USD for the book.
-/
theorem alice_book_payment :
  ∀ (price_gbp : ℝ) (conversion_rate : ℝ), 
  price_gbp = 25 → conversion_rate = 0.75 → 
  (price_gbp / conversion_rate) = 33.33 :=
by
  intros price_gbp conversion_rate hprice hrate
  rw [hprice, hrate]
  sorry

end NUMINAMATH_GPT_alice_book_payment_l1685_168534


namespace NUMINAMATH_GPT_ten_thousand_points_length_l1685_168514

theorem ten_thousand_points_length (a b : ℝ) (d : ℝ) 
  (h1 : d = a / 99) 
  (h2 : b = 9999 * d) : b = 101 * a := by
  sorry

end NUMINAMATH_GPT_ten_thousand_points_length_l1685_168514


namespace NUMINAMATH_GPT_shorter_steiner_network_l1685_168553

-- Define the variables and inequality
noncomputable def side_length (a : ℝ) : ℝ := a
noncomputable def diagonal_network_length (a : ℝ) : ℝ := 2 * a * Real.sqrt 2
noncomputable def steiner_network_length (a : ℝ) : ℝ := a * (1 + Real.sqrt 3)

theorem shorter_steiner_network {a : ℝ} (h₀ : 0 < a) :
  diagonal_network_length a > steiner_network_length a :=
by
  -- Proof to be provided (skipping it with sorry)
  sorry

end NUMINAMATH_GPT_shorter_steiner_network_l1685_168553


namespace NUMINAMATH_GPT_calculate_bus_stoppage_time_l1685_168552

variable (speed_excl_stoppages speed_incl_stoppages distance_excl_stoppages distance_incl_stoppages distance_diff time_lost_stoppages : ℝ)

def bus_stoppage_time
  (speed_excl_stoppages : ℝ)
  (speed_incl_stoppages : ℝ)
  (time_stopped : ℝ) :
  Prop :=
  speed_excl_stoppages = 32 ∧
  speed_incl_stoppages = 16 ∧
  time_stopped = 30

theorem calculate_bus_stoppage_time 
  (speed_excl_stoppages : ℝ)
  (speed_incl_stoppages : ℝ)
  (time_stopped : ℝ) :
  bus_stoppage_time speed_excl_stoppages speed_incl_stoppages time_stopped :=
by
  have h1 : speed_excl_stoppages = 32 := by
    sorry
  have h2 : speed_incl_stoppages = 16 := by
    sorry
  have h3 : time_stopped = 30 := by
    sorry
  exact ⟨h1, h2, h3⟩

end NUMINAMATH_GPT_calculate_bus_stoppage_time_l1685_168552


namespace NUMINAMATH_GPT_isosceles_triangle_side_l1685_168529

theorem isosceles_triangle_side (a : ℝ) : 
  (10 - a = 7 ∨ 10 - a = 6) ↔ (a = 3 ∨ a = 4) := 
by sorry

end NUMINAMATH_GPT_isosceles_triangle_side_l1685_168529


namespace NUMINAMATH_GPT_problem_solution_count_l1685_168526

theorem problem_solution_count (n : ℕ) (h1 : (80 * n) ^ 40 > n ^ 80) (h2 : n ^ 80 > 3 ^ 160) : 
  ∃ s : Finset ℕ, s.card = 70 ∧ ∀ x ∈ s, 10 ≤ x ∧ x ≤ 79 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_count_l1685_168526


namespace NUMINAMATH_GPT_Nils_has_300_geese_l1685_168590

variables (A x k n : ℕ)

def condition1 (A x k n : ℕ) : Prop :=
  A = k * x * n

def condition2 (A x k n : ℕ) : Prop :=
  A = (k + 20) * x * (n - 50)

def condition3 (A x k n : ℕ) : Prop :=
  A = (k - 10) * x * (n + 100)

theorem Nils_has_300_geese (A x k n : ℕ) :
  condition1 A x k n →
  condition2 A x k n →
  condition3 A x k n →
  n = 300 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_Nils_has_300_geese_l1685_168590


namespace NUMINAMATH_GPT_carpet_length_is_9_l1685_168509

noncomputable def carpet_length (width : ℝ) (living_room_area : ℝ) (coverage : ℝ) : ℝ :=
  living_room_area * coverage / width

theorem carpet_length_is_9 (width : ℝ) (living_room_area : ℝ) (coverage : ℝ) (length := carpet_length width living_room_area coverage) :
    width = 4 → living_room_area = 48 → coverage = 0.75 → length = 9 := by
  intros
  sorry

end NUMINAMATH_GPT_carpet_length_is_9_l1685_168509


namespace NUMINAMATH_GPT_roger_final_money_is_correct_l1685_168542

noncomputable def initial_money : ℝ := 84
noncomputable def birthday_money : ℝ := 56
noncomputable def found_money : ℝ := 20
noncomputable def spent_on_game : ℝ := 35
noncomputable def spent_percentage : ℝ := 0.15

noncomputable def final_money 
  (initial_money birthday_money found_money spent_on_game spent_percentage : ℝ) : ℝ :=
  let total_before_spending := initial_money + birthday_money + found_money
  let remaining_after_game := total_before_spending - spent_on_game
  let spent_on_gift := spent_percentage * remaining_after_game
  remaining_after_game - spent_on_gift

theorem roger_final_money_is_correct :
  final_money initial_money birthday_money found_money spent_on_game spent_percentage = 106.25 :=
by
  sorry

end NUMINAMATH_GPT_roger_final_money_is_correct_l1685_168542


namespace NUMINAMATH_GPT_larger_number_is_20_l1685_168533

theorem larger_number_is_20 (a b : ℕ) (h1 : a + b = 9 * (a - b)) (h2 : a + b = 36) (h3 : a > b) : a = 20 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_is_20_l1685_168533


namespace NUMINAMATH_GPT_find_length_BD_l1685_168531

theorem find_length_BD (c : ℝ) (h : c ≥ Real.sqrt 7) :
  ∃BD, BD = Real.sqrt (c^2 - 7) :=
sorry

end NUMINAMATH_GPT_find_length_BD_l1685_168531


namespace NUMINAMATH_GPT_find_number_l1685_168530

-- Define the problem constants
def total : ℝ := 1.794
def part1 : ℝ := 0.123
def part2 : ℝ := 0.321
def target : ℝ := 1.350

-- The equivalent proof problem
theorem find_number (x : ℝ) (h : part1 + part2 + x = total) : x = target := by
  -- Proof is intentionally omitted
  sorry

end NUMINAMATH_GPT_find_number_l1685_168530


namespace NUMINAMATH_GPT_carrots_total_l1685_168506

variables (initiallyPicked : Nat) (thrownOut : Nat) (pickedNextDay : Nat)

def totalCarrots (initiallyPicked : Nat) (thrownOut : Nat) (pickedNextDay : Nat) :=
  initiallyPicked - thrownOut + pickedNextDay

theorem carrots_total (h1 : initiallyPicked = 19)
                     (h2 : thrownOut = 4)
                     (h3 : pickedNextDay = 46) :
  totalCarrots initiallyPicked thrownOut pickedNextDay = 61 :=
by
  sorry

end NUMINAMATH_GPT_carrots_total_l1685_168506


namespace NUMINAMATH_GPT_volume_of_tetrahedron_equiv_l1685_168522

noncomputable def volume_tetrahedron (D1 D2 D3 : ℝ) 
  (h1 : D1 = 24) (h2 : D3 = 20) (h3 : D2 = 16) : ℝ :=
  30 * Real.sqrt 6

theorem volume_of_tetrahedron_equiv (D1 D2 D3 : ℝ) 
  (h1 : D1 = 24) (h2 : D3 = 20) (h3 : D2 = 16) :
  volume_tetrahedron D1 D2 D3 h1 h2 h3 = 30 * Real.sqrt 6 :=
  sorry

end NUMINAMATH_GPT_volume_of_tetrahedron_equiv_l1685_168522


namespace NUMINAMATH_GPT_calculate_expression_l1685_168523

theorem calculate_expression :
  (Real.sqrt 3) ^ 0 + 2 ^ (-1 : ℤ) + Real.sqrt 2 * Real.cos (Real.pi / 4) - |(-1:ℝ) / 2| = 2 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1685_168523


namespace NUMINAMATH_GPT_segment_length_eq_ten_l1685_168586

theorem segment_length_eq_ten (x : ℝ) (h : |x - 3| = 5) : |8 - (-2)| = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_segment_length_eq_ten_l1685_168586


namespace NUMINAMATH_GPT_complement_subset_lemma_l1685_168567

-- Definitions for sets P and Q
def P : Set ℝ := {x | 0 < x ∧ x < 1}

def Q : Set ℝ := {x | x^2 + x - 2 ≤ 0}

-- Definition for complement of a set
def C_ℝ (A : Set ℝ) : Set ℝ := {x | ¬(x ∈ A)}

-- Prove the required relationship
theorem complement_subset_lemma : C_ℝ Q ⊆ C_ℝ P :=
by
  -- The proof steps will go here
  sorry

end NUMINAMATH_GPT_complement_subset_lemma_l1685_168567


namespace NUMINAMATH_GPT_cookie_portion_l1685_168557

theorem cookie_portion :
  ∃ (total_cookies : ℕ) (remaining_cookies : ℕ) (cookies_senior_ate : ℕ) (cookies_senior_took_second_day : ℕ) 
    (cookies_senior_put_back : ℕ) (cookies_junior_took : ℕ),
  total_cookies = 22 ∧
  remaining_cookies = 11 ∧
  cookies_senior_ate = 3 ∧
  cookies_senior_took_second_day = 3 ∧
  cookies_senior_put_back = 2 ∧
  cookies_junior_took = 7 ∧
  4 / 22 = 2 / 11 :=
by
  existsi 22, 11, 3, 3, 2, 7
  sorry

end NUMINAMATH_GPT_cookie_portion_l1685_168557


namespace NUMINAMATH_GPT_GCF_LCM_proof_l1685_168563

-- Define GCF (greatest common factor)
def GCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Define LCM (least common multiple)
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem GCF_LCM_proof :
  GCF (LCM 9 21) (LCM 14 15) = 21 :=
by
  sorry

end NUMINAMATH_GPT_GCF_LCM_proof_l1685_168563


namespace NUMINAMATH_GPT_greatest_root_of_g_l1685_168591

noncomputable def g (x : ℝ) : ℝ := 10 * x^4 - 16 * x^2 + 6

theorem greatest_root_of_g : ∃ x : ℝ, g x = 0 ∧ ∀ y : ℝ, g y = 0 → y ≤ x := 
by
  sorry

end NUMINAMATH_GPT_greatest_root_of_g_l1685_168591


namespace NUMINAMATH_GPT_strip_covers_cube_l1685_168516

   -- Define the given conditions
   def strip_length := 12
   def strip_width := 1
   def cube_edge := 1
   def layers := 2

   -- Define the main statement to be proved
   theorem strip_covers_cube : 
     (strip_length >= 6 * cube_edge / layers) ∧ 
     (strip_width >= cube_edge) ∧ 
     (layers == 2) → 
     true :=
   by
     intro h
     sorry
   
end NUMINAMATH_GPT_strip_covers_cube_l1685_168516


namespace NUMINAMATH_GPT_Sharmila_hourly_wage_l1685_168587

def Sharmila_hours_per_day (day : String) : ℕ :=
  if day = "Monday" ∨ day = "Wednesday" ∨ day = "Friday" then 10
  else if day = "Tuesday" ∨ day = "Thursday" then 8
  else 0

def weekly_total_hours : ℕ :=
  Sharmila_hours_per_day "Monday" + Sharmila_hours_per_day "Tuesday" +
  Sharmila_hours_per_day "Wednesday" + Sharmila_hours_per_day "Thursday" +
  Sharmila_hours_per_day "Friday"

def weekly_earnings : ℤ := 460

def hourly_wage : ℚ :=
  weekly_earnings / weekly_total_hours

theorem Sharmila_hourly_wage :
  hourly_wage = (10 : ℚ) :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_Sharmila_hourly_wage_l1685_168587


namespace NUMINAMATH_GPT_inequality_solution_l1685_168538

theorem inequality_solution (a : ℝ) (h : 1 < a) : ∀ x : ℝ, a ^ (2 * x + 1) > (1 / a) ^ (2 * x) ↔ x > -1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1685_168538


namespace NUMINAMATH_GPT_constant_term_of_expansion_l1685_168500

noncomputable def constant_term := 
  (20: ℕ) * (216: ℕ) * (1/27: ℚ) = (160: ℕ)

theorem constant_term_of_expansion : constant_term :=
  by sorry

end NUMINAMATH_GPT_constant_term_of_expansion_l1685_168500


namespace NUMINAMATH_GPT_sequence_not_divisible_by_7_l1685_168597

theorem sequence_not_divisible_by_7 (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 1200) : ¬ (7 ∣ (9^n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_not_divisible_by_7_l1685_168597


namespace NUMINAMATH_GPT_negation_of_exists_eq_sin_l1685_168550

theorem negation_of_exists_eq_sin : ¬ (∃ x : ℝ, x = Real.sin x) ↔ ∀ x : ℝ, x ≠ Real.sin x :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_eq_sin_l1685_168550


namespace NUMINAMATH_GPT_pages_in_first_chapter_l1685_168515

theorem pages_in_first_chapter (x : ℕ) (h1 : x + 43 = 80) : x = 37 :=
by
  sorry

end NUMINAMATH_GPT_pages_in_first_chapter_l1685_168515


namespace NUMINAMATH_GPT_boat_upstream_speed_l1685_168573

variable (Vb Vc : ℕ)

def boat_speed_upstream (Vb Vc : ℕ) : ℕ := Vb - Vc

theorem boat_upstream_speed (hVb : Vb = 50) (hVc : Vc = 20) : boat_speed_upstream Vb Vc = 30 :=
by sorry

end NUMINAMATH_GPT_boat_upstream_speed_l1685_168573


namespace NUMINAMATH_GPT_acceptable_arrangements_correct_l1685_168564

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n + 1) * factorial n

-- Define the total number of people
def total_people := 8

-- Calculate the total arrangements of 8 people
def total_arrangements := factorial total_people

-- Calculate the arrangements where Alice and Bob are together
def reduced_people := total_people - 1
def alice_bob_arrangements := factorial reduced_people * factorial 2

-- Calculate the acceptable arrangements where Alice and Bob are not together
def acceptable_arrangements := total_arrangements - alice_bob_arrangements

-- The theorem statement, asserting the correct answer
theorem acceptable_arrangements_correct : acceptable_arrangements = 30240 :=
by
  sorry

end NUMINAMATH_GPT_acceptable_arrangements_correct_l1685_168564


namespace NUMINAMATH_GPT_max_value_of_operation_l1685_168539

theorem max_value_of_operation : 
  ∃ (n : ℤ), (10 ≤ n ∧ n ≤ 99) ∧ 4 * (300 - n) = 1160 := by
  sorry

end NUMINAMATH_GPT_max_value_of_operation_l1685_168539


namespace NUMINAMATH_GPT_double_acute_angle_l1685_168544

theorem double_acute_angle (θ : ℝ) (h : 0 < θ ∧ θ < 90) : 0 < 2 * θ ∧ 2 * θ < 180 :=
sorry

end NUMINAMATH_GPT_double_acute_angle_l1685_168544


namespace NUMINAMATH_GPT_decimal_representation_prime_has_zeros_l1685_168528

theorem decimal_representation_prime_has_zeros (p : ℕ) [Fact (Nat.Prime p)] : 
  ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, 10^2002 ∣ p^n * 10^k :=
sorry

end NUMINAMATH_GPT_decimal_representation_prime_has_zeros_l1685_168528


namespace NUMINAMATH_GPT_is_divisible_by_six_l1685_168549

/-- A stingy knight keeps gold coins in six chests. Given that he can evenly distribute the coins by opening any
two chests, any three chests, any four chests, or any five chests, prove that the total number of coins can be 
evenly distributed among all six chests. -/
theorem is_divisible_by_six (n : ℕ) 
  (h2 : ∀ (a b : ℕ), a + b = n → (a % 2 = 0 ∧ b % 2 = 0))
  (h3 : ∀ (a b c : ℕ), a + b + c = n → (a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0)) 
  (h4 : ∀ (a b c d : ℕ), a + b + c + d = n → (a % 4 = 0 ∧ b % 4 = 0 ∧ c % 4 = 0 ∧ d % 4 = 0))
  (h5 : ∀ (a b c d e : ℕ), a + b + c + d + e = n → (a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 ∧ d % 5 = 0 ∧ e % 5 = 0)) :
  n % 6 = 0 :=
sorry

end NUMINAMATH_GPT_is_divisible_by_six_l1685_168549


namespace NUMINAMATH_GPT_speed_boat_in_still_water_l1685_168569

variable (V_b V_s t : ℝ)

def speed_of_boat := V_b

axiom stream_speed : V_s = 26

axiom time_relation : 2 * (t : ℝ) = 2 * t

axiom distance_relation : (V_b + V_s) * t = (V_b - V_s) * (2 * t)

theorem speed_boat_in_still_water : V_b = 78 :=
by {
  sorry
}

end NUMINAMATH_GPT_speed_boat_in_still_water_l1685_168569


namespace NUMINAMATH_GPT_employee_y_payment_l1685_168545

theorem employee_y_payment (X Y : ℝ) (h1 : X + Y = 616) (h2 : X = 1.2 * Y) : Y = 280 :=
by
  sorry

end NUMINAMATH_GPT_employee_y_payment_l1685_168545


namespace NUMINAMATH_GPT_B_max_at_125_l1685_168504

noncomputable def B (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.3 : ℝ) ^ k

theorem B_max_at_125 :
  ∃ k, 0 ≤ k ∧ k ≤ 500 ∧ (∀ n, 0 ≤ n ∧ n ≤ 500 → B k ≥ B n) ∧ k = 125 :=
by
  sorry

end NUMINAMATH_GPT_B_max_at_125_l1685_168504


namespace NUMINAMATH_GPT_range_of_a_l1685_168575

theorem range_of_a (a : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + a * x + 1 ≥ 0) ↔ (a < -2 ∨ a > 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1685_168575


namespace NUMINAMATH_GPT_lines_perpendicular_and_intersect_l1685_168546

variable {a b : ℝ}

theorem lines_perpendicular_and_intersect 
  (h_ab_nonzero : a * b ≠ 0)
  (h_orthogonal : a + b = 0) : 
  ∃ p, p ≠ 0 ∧ 
    (∀ x y, x = -y * b^2 → y = 0 → p = (x, y)) ∧ 
    (∀ x y, y = x / a^2 → x = 0 → p = (x, y)) ∧ 
    (∀ x y, x = -y * b^2 ∧ y = x / a^2 → x = 0 ∧ y = 0) := 
sorry

end NUMINAMATH_GPT_lines_perpendicular_and_intersect_l1685_168546


namespace NUMINAMATH_GPT_solve_equation_l1685_168593

theorem solve_equation:
  ∀ x : ℝ, (x + 1) / 3 - 1 = (5 * x - 1) / 6 → x = -1 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_equation_l1685_168593


namespace NUMINAMATH_GPT_green_apples_count_l1685_168503

-- Definitions for the conditions
def total_apples : ℕ := 9
def red_apples : ℕ := 7

-- Theorem stating the number of green apples
theorem green_apples_count : total_apples - red_apples = 2 := by
  sorry

end NUMINAMATH_GPT_green_apples_count_l1685_168503


namespace NUMINAMATH_GPT_physics_experiment_l1685_168502

theorem physics_experiment (x : ℕ) (h : 1 + x + (x + 1) * x = 36) :
  1 + x + (x + 1) * x = 36 :=
  by                        
  exact h

end NUMINAMATH_GPT_physics_experiment_l1685_168502


namespace NUMINAMATH_GPT_gcd_18_30_45_l1685_168535

theorem gcd_18_30_45 : Nat.gcd (Nat.gcd 18 30) 45 = 3 :=
by
  sorry

end NUMINAMATH_GPT_gcd_18_30_45_l1685_168535


namespace NUMINAMATH_GPT_determinant_of_triangle_angles_l1685_168532

theorem determinant_of_triangle_angles (α β γ : ℝ) (h : α + β + γ = Real.pi) :
  Matrix.det ![
    ![Real.tan α, Real.sin α * Real.cos α, 1],
    ![Real.tan β, Real.sin β * Real.cos β, 1],
    ![Real.tan γ, Real.sin γ * Real.cos γ, 1]
  ] = 0 :=
by
  -- Proof statement goes here
  sorry

end NUMINAMATH_GPT_determinant_of_triangle_angles_l1685_168532


namespace NUMINAMATH_GPT_find_x4_y4_l1685_168511

theorem find_x4_y4 (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 15) : x^4 + y^4 = 175 := by
  sorry

end NUMINAMATH_GPT_find_x4_y4_l1685_168511


namespace NUMINAMATH_GPT_max_integer_a_real_roots_l1685_168547

theorem max_integer_a_real_roots :
  ∀ (a : ℤ), (∃ (x : ℝ), (a + 1 : ℝ) * x^2 - 2 * x + 3 = 0) → a ≤ -2 :=
by
  sorry

end NUMINAMATH_GPT_max_integer_a_real_roots_l1685_168547


namespace NUMINAMATH_GPT_find_deducted_salary_l1685_168554

noncomputable def dailyWage (weeklySalary : ℝ) (workingDays : ℕ) : ℝ := weeklySalary / workingDays

noncomputable def totalDeduction (dailyWage : ℝ) (absentDays : ℕ) : ℝ := dailyWage * absentDays

noncomputable def deductedSalary (weeklySalary : ℝ) (totalDeduction : ℝ) : ℝ := weeklySalary - totalDeduction

theorem find_deducted_salary
  (weeklySalary : ℝ := 791)
  (workingDays : ℕ := 5)
  (absentDays : ℕ := 4)
  (dW := dailyWage weeklySalary workingDays)
  (tD := totalDeduction dW absentDays)
  (dS := deductedSalary weeklySalary tD) :
  dS = 158.20 := 
  by
    sorry

end NUMINAMATH_GPT_find_deducted_salary_l1685_168554


namespace NUMINAMATH_GPT_find_x_l1685_168576

theorem find_x (x : ℝ) (h_pos : x > 0) (h_area : (1 / 2) * x * (3 * x) = 54) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1685_168576


namespace NUMINAMATH_GPT_comp_functions_l1685_168588

def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := 3 * x - 5

theorem comp_functions (x : ℝ) : f (g x) = 6 * x - 7 :=
by
  sorry

end NUMINAMATH_GPT_comp_functions_l1685_168588


namespace NUMINAMATH_GPT_six_digit_number_reversed_by_9_l1685_168598

-- Hypothetical function to reverse digits of a number
def reverseDigits (n : ℕ) : ℕ := sorry

theorem six_digit_number_reversed_by_9 :
  ∃ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n * 9 = reverseDigits n :=
by
  sorry

end NUMINAMATH_GPT_six_digit_number_reversed_by_9_l1685_168598


namespace NUMINAMATH_GPT_find_age_l1685_168543

-- Define the age variables
variables (P Q : ℕ)

-- Define the conditions
def condition1 : Prop := (P - 3) * 3 = (Q - 3) * 4
def condition2 : Prop := (P + 6) * 6 = (Q + 6) * 7

-- Prove that, given the conditions, P equals 15
theorem find_age (h1 : condition1 P Q) (h2 : condition2 P Q) : P = 15 :=
sorry

end NUMINAMATH_GPT_find_age_l1685_168543


namespace NUMINAMATH_GPT_negative_number_reciprocal_eq_self_l1685_168599

theorem negative_number_reciprocal_eq_self (x : ℝ) (hx : x < 0) (h : 1 / x = x) : x = -1 :=
by
  sorry

end NUMINAMATH_GPT_negative_number_reciprocal_eq_self_l1685_168599


namespace NUMINAMATH_GPT_find_original_intensity_l1685_168536

variable (I : ℝ)  -- Define intensity of the original red paint (in percentage).

-- Conditions:
variable (fractionReplaced : ℝ) (newIntensity : ℝ) (replacingIntensity : ℝ)
  (fractionReplaced_eq : fractionReplaced = 0.8)
  (newIntensity_eq : newIntensity = 30)
  (replacingIntensity_eq : replacingIntensity = 25)

-- Theorem statement:
theorem find_original_intensity :
  (1 - fractionReplaced) * I + fractionReplaced * replacingIntensity = newIntensity → I = 50 :=
sorry

end NUMINAMATH_GPT_find_original_intensity_l1685_168536


namespace NUMINAMATH_GPT_find_A_when_B_is_largest_l1685_168525

theorem find_A_when_B_is_largest :
  ∃ A : ℕ, ∃ B : ℕ, A = 17 * 25 + B ∧ B < 17 ∧ B = 16 ∧ A = 441 :=
by
  sorry

end NUMINAMATH_GPT_find_A_when_B_is_largest_l1685_168525


namespace NUMINAMATH_GPT_expressions_equal_l1685_168527

theorem expressions_equal {x y z : ℤ} : (x + 2 * y * z = (x + y) * (x + 2 * z)) ↔ (x + y + 2 * z = 1) :=
by
  sorry

end NUMINAMATH_GPT_expressions_equal_l1685_168527


namespace NUMINAMATH_GPT_perpendicular_condition_l1685_168551

def line := Type
def plane := Type

variables {α : plane} {a b : line}

-- Conditions: define parallelism and perpendicularity
def parallel (a : line) (α : plane) : Prop := sorry
def perpendicular (a : line) (α : plane) : Prop := sorry
def perpendicular_lines (a b : line) : Prop := sorry

-- Given Hypotheses
variable (h1 : parallel a α)
variable (h2 : perpendicular b α)

-- Statement to prove
theorem perpendicular_condition (h1 : parallel a α) (h2 : perpendicular b α) :
  (perpendicular_lines b a) ∧ (¬ (perpendicular_lines b a → perpendicular b α)) := 
sorry

end NUMINAMATH_GPT_perpendicular_condition_l1685_168551


namespace NUMINAMATH_GPT_factorize_x2_plus_2x_l1685_168574

theorem factorize_x2_plus_2x (x : ℝ) : x^2 + 2*x = x * (x + 2) :=
by sorry

end NUMINAMATH_GPT_factorize_x2_plus_2x_l1685_168574


namespace NUMINAMATH_GPT_isosceles_triangle_largest_angle_l1685_168548

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h_iso : A = B) (h_A : C = 50) :
  max A (max B (180 - A - B)) = 80 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_largest_angle_l1685_168548


namespace NUMINAMATH_GPT_garden_area_l1685_168524

-- Given that the garden is a square with certain properties
variables (s A P : ℕ)

-- Conditions:
-- The perimeter of the square garden is 28 feet
def perimeter_condition : Prop := P = 28

-- The area of the garden is equal to the perimeter plus 21
def area_condition : Prop := A = P + 21

-- The perimeter of a square garden with side length s
def perimeter_def : Prop := P = 4 * s

-- The area of a square garden with side length s
def area_def : Prop := A = s * s

-- Prove that the area A is 49 square feet
theorem garden_area : perimeter_condition P → area_condition P A → perimeter_def s P → area_def s A → A = 49 :=
by 
  sorry

end NUMINAMATH_GPT_garden_area_l1685_168524


namespace NUMINAMATH_GPT_below_sea_level_is_negative_l1685_168578
-- Lean 4 statement


theorem below_sea_level_is_negative 
  (above_sea_pos : ∀ x : ℝ, x > 0 → x = x)
  (below_sea_neg : ∀ x : ℝ, x < 0 → x = x) : 
  (-25 = -25) :=
by 
  -- here we are supposed to provide the proof but we are skipping it with sorry
  sorry

end NUMINAMATH_GPT_below_sea_level_is_negative_l1685_168578


namespace NUMINAMATH_GPT_walls_divided_equally_l1685_168582

-- Define the given conditions
def num_people : ℕ := 5
def num_rooms : ℕ := 9
def rooms_with_4_walls : ℕ := 5
def walls_per_room_4 : ℕ := 4
def rooms_with_5_walls : ℕ := 4
def walls_per_room_5 : ℕ := 5

-- Calculate the total number of walls
def total_walls : ℕ := (rooms_with_4_walls * walls_per_room_4) + (rooms_with_5_walls * walls_per_room_5)

-- Define the expected result
def walls_per_person : ℕ := total_walls / num_people

-- Theorem statement: Each person should paint 8 walls.
theorem walls_divided_equally : walls_per_person = 8 := by
  sorry

end NUMINAMATH_GPT_walls_divided_equally_l1685_168582


namespace NUMINAMATH_GPT_terry_spent_total_l1685_168594

def total_amount_spent (monday_spent tuesday_spent wednesday_spent : ℕ) : ℕ := 
  monday_spent + tuesday_spent + wednesday_spent

theorem terry_spent_total 
  (monday_spent : ℕ)
  (hmonday : monday_spent = 6)
  (tuesday_spent : ℕ)
  (htuesday : tuesday_spent = 2 * monday_spent)
  (wednesday_spent : ℕ)
  (hwednesday : wednesday_spent = 2 * (monday_spent + tuesday_spent)) :
  total_amount_spent monday_spent tuesday_spent wednesday_spent = 54 :=
by
  sorry

end NUMINAMATH_GPT_terry_spent_total_l1685_168594


namespace NUMINAMATH_GPT_annual_interest_payment_l1685_168505

def principal : ℝ := 10000
def quarterly_rate : ℝ := 0.05

theorem annual_interest_payment :
  (principal * quarterly_rate * 4) = 2000 :=
by sorry

end NUMINAMATH_GPT_annual_interest_payment_l1685_168505


namespace NUMINAMATH_GPT_macaroon_count_l1685_168519

def baked_red_macaroons : ℕ := 50
def baked_green_macaroons : ℕ := 40
def ate_green_macaroons : ℕ := 15
def ate_red_macaroons := 2 * ate_green_macaroons

def remaining_macaroons : ℕ := (baked_red_macaroons - ate_red_macaroons) + (baked_green_macaroons - ate_green_macaroons)

theorem macaroon_count : remaining_macaroons = 45 := by
  sorry

end NUMINAMATH_GPT_macaroon_count_l1685_168519


namespace NUMINAMATH_GPT_max_area_basketball_court_l1685_168570

theorem max_area_basketball_court : 
  ∃ l w : ℝ, 2 * l + 2 * w = 400 ∧ l ≥ 100 ∧ w ≥ 50 ∧ l * w = 10000 :=
by {
  -- We are skipping the proof for now
  sorry
}

end NUMINAMATH_GPT_max_area_basketball_court_l1685_168570


namespace NUMINAMATH_GPT_positive_difference_l1685_168558

theorem positive_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 14) : y - x = 9.714 :=
sorry

end NUMINAMATH_GPT_positive_difference_l1685_168558


namespace NUMINAMATH_GPT_intersection_A_B_subset_A_B_l1685_168595

-- Definitions for the sets A and B
def set_A (a : ℝ) : Set ℝ := {x | 2 * a - 1 ≤ x ∧ x ≤ a + 3}
def set_B : Set ℝ := {x | x < -1 ∨ x > 5}

-- First proof problem: Intersection
theorem intersection_A_B (a : ℝ) (ha : a = -2) :
  set_A a ∩ set_B = {x | -5 ≤ x ∧ x < -1} :=
sorry

-- Second proof problem: Subset
theorem subset_A_B (a : ℝ) :
  set_A a ⊆ set_B ↔ (a ≤ -4 ∨ a ≥ 3) :=
sorry

end NUMINAMATH_GPT_intersection_A_B_subset_A_B_l1685_168595


namespace NUMINAMATH_GPT_susan_probability_exactly_three_blue_marbles_l1685_168577

open ProbabilityTheory

noncomputable def probability_blue_marbles (n_blue n_red : ℕ) (total_trials drawn_blue : ℕ) : ℚ :=
  let total_marbles := n_blue + n_red
  let prob_blue := (n_blue : ℚ) / total_marbles
  let prob_red := (n_red : ℚ) / total_marbles
  let n_comb := Nat.choose total_trials drawn_blue
  (n_comb : ℚ) * (prob_blue ^ drawn_blue) * (prob_red ^ (total_trials - drawn_blue))

theorem susan_probability_exactly_three_blue_marbles :
  probability_blue_marbles 8 7 7 3 = 35 * (1225621 / 171140625) :=
by
  sorry

end NUMINAMATH_GPT_susan_probability_exactly_three_blue_marbles_l1685_168577


namespace NUMINAMATH_GPT_find_f1_plus_g1_l1685_168583

variables (f g : ℝ → ℝ)

-- Conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g x
def function_equation (f g : ℝ → ℝ) : Prop := ∀ x : ℝ, f x - g x = x^3 - 2*x^2 + 1

theorem find_f1_plus_g1 
  (hf : even_function f)
  (hg : odd_function g)
  (hfg : function_equation f g):
  f 1 + g 1 = -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_f1_plus_g1_l1685_168583


namespace NUMINAMATH_GPT_angle_sum_equal_l1685_168541

theorem angle_sum_equal 
  (AB AC DE DF : ℝ)
  (h_AB_AC : AB = AC)
  (h_DE_DF : DE = DF)
  (angle_BAC angle_EDF : ℝ)
  (h_angle_BAC : angle_BAC = 40)
  (h_angle_EDF : angle_EDF = 50)
  (angle_DAC angle_ADE : ℝ)
  (h_angle_DAC : angle_DAC = 70)
  (h_angle_ADE : angle_ADE = 65) :
  angle_DAC + angle_ADE = 135 := 
sorry

end NUMINAMATH_GPT_angle_sum_equal_l1685_168541


namespace NUMINAMATH_GPT_find_first_number_l1685_168517

def is_lcm (a b l : ℕ) : Prop := l = Nat.lcm a b

theorem find_first_number :
  ∃ (a b : ℕ), (5 * b) = a ∧ (4 * b) = b ∧ is_lcm a b 80 ∧ a = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_first_number_l1685_168517


namespace NUMINAMATH_GPT_find_principal_amount_l1685_168592

theorem find_principal_amount 
  (P₁ : ℝ) (r₁ t₁ : ℝ) (S₁ : ℝ)
  (P₂ : ℝ) (r₂ t₂ : ℝ) (C₂ : ℝ) :
  S₁ = (P₁ * r₁ * t₁) / 100 →
  C₂ = P₂ * ( (1 + r₂) ^ t₂ - 1) →
  S₁ = C₂ / 2 →
  P₁ = 2800 :=
by
  sorry

end NUMINAMATH_GPT_find_principal_amount_l1685_168592


namespace NUMINAMATH_GPT_polygon_P_properties_l1685_168596

-- Definitions of points A, B, and C
def A : (ℝ × ℝ × ℝ) := (0, 0, 0)
def B : (ℝ × ℝ × ℝ) := (1, 0.5, 0)
def C : (ℝ × ℝ × ℝ) := (0, 0.5, 1)

-- Condition of cube intersection and plane containing A, B, and C
def is_corner_of_cube (p : ℝ × ℝ × ℝ) : Prop :=
  p = A

def are_midpoints_of_cube_edges (p₁ p₂ : ℝ × ℝ × ℝ) : Prop :=
  (p₁ = B ∧ p₂ = C)

-- Polygon P resulting from the plane containing A, B, and C intersecting the cube
def num_sides_of_polygon (p : ℝ × ℝ × ℝ) : ℕ := 5 -- Given the polygon is a pentagon

-- Area of triangle ABC
noncomputable def area_triangle_ABC : ℝ :=
  (1/2) * (Real.sqrt 1.5)

-- Area of polygon P
noncomputable def area_polygon_P : ℝ :=
  (11/6) * area_triangle_ABC

-- Theorem stating that polygon P has 5 sides and the ratio of its area to the area of triangle ABC is 11/6
theorem polygon_P_properties (A B C : (ℝ × ℝ × ℝ))
  (hA : is_corner_of_cube A) (hB : are_midpoints_of_cube_edges B C) :
  num_sides_of_polygon A = 5 ∧ area_polygon_P / area_triangle_ABC = (11/6) :=
by sorry

end NUMINAMATH_GPT_polygon_P_properties_l1685_168596


namespace NUMINAMATH_GPT_factorize_expression_l1685_168562

variable {a b x y : ℝ}

theorem factorize_expression :
  (x^2 - y^2) * a^2 - (x^2 - y^2) * b^2 = (x + y) * (x - y) * (a + b) * (a - b) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1685_168562


namespace NUMINAMATH_GPT_time_to_pass_faster_train_l1685_168507

noncomputable def speed_slower_train_kmph : ℝ := 36
noncomputable def speed_faster_train_kmph : ℝ := 45
noncomputable def length_faster_train_m : ℝ := 225.018
noncomputable def kmph_to_mps_factor : ℝ := 1000 / 3600

noncomputable def relative_speed_mps : ℝ := (speed_slower_train_kmph + speed_faster_train_kmph) * kmph_to_mps_factor

theorem time_to_pass_faster_train : 
  (length_faster_train_m / relative_speed_mps) = 10.001 := 
sorry

end NUMINAMATH_GPT_time_to_pass_faster_train_l1685_168507


namespace NUMINAMATH_GPT_ellipse_foci_coordinates_l1685_168568

theorem ellipse_foci_coordinates :
  ∃ x y : Real, (3 * x^2 + 4 * y^2 = 12) ∧ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0)) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_foci_coordinates_l1685_168568


namespace NUMINAMATH_GPT_find_integers_l1685_168566

theorem find_integers (n : ℤ) : (n^2 - 13 * n + 36 < 0) ↔ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_integers_l1685_168566
