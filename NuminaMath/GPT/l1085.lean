import Mathlib

namespace complement_union_l1085_108598

open Set

-- Definitions from the given conditions
def U : Set ℕ := {x | x ≤ 9}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5, 6}

-- Statement of the proof problem
theorem complement_union :
  compl (A ∪ B) = {7, 8, 9} :=
sorry

end complement_union_l1085_108598


namespace find_a_l1085_108593

theorem find_a (a : ℝ) :
  (∀ x : ℝ, (x * x - 4 <= 0) → (2 * x + a <= 0)) ↔ (a = -4) := by
  sorry

end find_a_l1085_108593


namespace find_f_inv_486_l1085_108585

open Function

noncomputable def f (x : ℕ) : ℕ := sorry -- placeholder for function definition

axiom f_condition1 : f 5 = 2
axiom f_condition2 : ∀ (x : ℕ), f (3 * x) = 3 * f x

theorem find_f_inv_486 : f⁻¹' {486} = {1215} := sorry

end find_f_inv_486_l1085_108585


namespace repeat_block_of_7_div_13_l1085_108570

theorem repeat_block_of_7_div_13 : ∃ k : ℕ, (∀ n : ℕ, n < k → 10^n % 13 ≠ 1) ∧ 10^k % 13 = 1 ∧ k = 6 :=
by { sorry }

end repeat_block_of_7_div_13_l1085_108570


namespace temperature_below_zero_l1085_108519

-- Assume the basic definitions and context needed
def above_zero (temp : Int) := temp > 0
def below_zero (temp : Int) := temp < 0

theorem temperature_below_zero (t1 t2 : Int) (h1 : above_zero t1) (h2 : t2 = -7) :
  below_zero t2 := by 
  -- This is where the proof would go
  sorry

end temperature_below_zero_l1085_108519


namespace no_integer_cube_eq_3n_squared_plus_3n_plus_7_l1085_108500

theorem no_integer_cube_eq_3n_squared_plus_3n_plus_7 :
  ¬ ∃ x n : ℤ, x^3 = 3 * n^2 + 3 * n + 7 := 
sorry

end no_integer_cube_eq_3n_squared_plus_3n_plus_7_l1085_108500


namespace custom_op_evaluation_l1085_108571

def custom_op (x y : ℕ) : ℕ := x * y + x - y

theorem custom_op_evaluation : (custom_op 7 4) - (custom_op 4 7) = 6 := by
  sorry

end custom_op_evaluation_l1085_108571


namespace distance_from_A_to_D_l1085_108561

theorem distance_from_A_to_D 
  (A B C D : Type)
  (east_of : B → A)
  (north_of : C → B)
  (distance_AC : Real)
  (angle_BAC : ℝ)
  (north_of_D : D → C)
  (distance_CD : Real) : 
  distance_AC = 5 * Real.sqrt 5 → 
  angle_BAC = 60 → 
  distance_CD = 15 → 
  ∃ (AD : Real), AD =
    Real.sqrt (
      (5 * Real.sqrt 15 / 2) ^ 2 + 
      (5 * Real.sqrt 5 / 2 + 15) ^ 2
    ) :=
by
  intros
  sorry


end distance_from_A_to_D_l1085_108561


namespace mr_c_gain_1000_l1085_108515

-- Define the initial conditions
def initial_mr_c_cash := 15000
def initial_mr_c_house := 12000
def initial_mrs_d_cash := 16000

-- Define the changes in the house value
def house_value_appreciated := 13000
def house_value_depreciated := 11000

-- Define the cash changes after transactions
def mr_c_cash_after_first_sale := initial_mr_c_cash + house_value_appreciated
def mrs_d_cash_after_first_sale := initial_mrs_d_cash - house_value_appreciated
def mrs_d_cash_after_second_sale := mrs_d_cash_after_first_sale + house_value_depreciated
def mr_c_cash_after_second_sale := mr_c_cash_after_first_sale - house_value_depreciated

-- Define the final net worth for Mr. C
def final_mr_c_cash := mr_c_cash_after_second_sale
def final_mr_c_house := house_value_depreciated
def final_mr_c_net_worth := final_mr_c_cash + final_mr_c_house
def initial_mr_c_net_worth := initial_mr_c_cash + initial_mr_c_house

-- Statement to prove
theorem mr_c_gain_1000 : final_mr_c_net_worth = initial_mr_c_net_worth + 1000 := by
  sorry

end mr_c_gain_1000_l1085_108515


namespace work_fraction_completed_after_first_phase_l1085_108562

-- Definitions based on conditions
def total_work := 1 -- Assume total work as 1 unit
def initial_days := 100
def initial_people := 10
def first_phase_days := 20
def fired_people := 2
def remaining_days := 75
def remaining_people := initial_people - fired_people

-- Hypothesis about the rate of work initially and after firing people
def initial_rate := total_work / initial_days
def first_phase_work := first_phase_days * initial_rate
def remaining_work := total_work - first_phase_work
def remaining_rate := remaining_work / remaining_days

-- Proof problem statement: 
theorem work_fraction_completed_after_first_phase :
  (first_phase_work / total_work) = (15 / 64) :=
by
  -- This is the place where the actual formal proof should be written.
  sorry

end work_fraction_completed_after_first_phase_l1085_108562


namespace single_room_cost_l1085_108548

theorem single_room_cost (total_rooms : ℕ) (single_rooms : ℕ) (double_room_cost : ℕ) 
  (total_revenue : ℤ) (x : ℤ) : 
  total_rooms = 260 → 
  single_rooms = 64 → 
  double_room_cost = 60 → 
  total_revenue = 14000 → 
  64 * x + (total_rooms - single_rooms) * double_room_cost = total_revenue → 
  x = 35 := 
by 
  intros h_total_rooms h_single_rooms h_double_room_cost h_total_revenue h_eqn 
  -- Add steps for proving if necessary
  sorry

end single_room_cost_l1085_108548


namespace sum_of_ages_l1085_108533

-- Definitions for conditions
def age_product (a b c : ℕ) : Prop := a * b * c = 72
def younger_than_10 (k : ℕ) : Prop := k < 10

-- Main statement
theorem sum_of_ages (a b k : ℕ) (h_product : age_product a b k) (h_twin : a = b) (h_kiana : younger_than_10 k) : 
  a + b + k = 14 := sorry

end sum_of_ages_l1085_108533


namespace number_of_ordered_pairs_l1085_108547

theorem number_of_ordered_pairs (x y : ℕ) : (x * y = 1716) → 
  (∃! n : ℕ, n = 18) :=
by
  sorry

end number_of_ordered_pairs_l1085_108547


namespace terminal_side_angle_l1085_108559

open Real

theorem terminal_side_angle (α : ℝ) (m n : ℝ) (h_line : n = 3 * m) (h_radius : m^2 + n^2 = 10) (h_sin : sin α < 0) (h_coincide : tan α = 3) : m - n = 2 :=
by
  sorry

end terminal_side_angle_l1085_108559


namespace divisibility_equivalence_distinct_positive_l1085_108503

variable (a b c : ℕ)

theorem divisibility_equivalence_distinct_positive (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a + b + c) ∣ (a^3 * b + b^3 * c + c^3 * a)) ↔ ((a + b + c) ∣ (a * b^3 + b * c^3 + c * a^3)) :=
by sorry

end divisibility_equivalence_distinct_positive_l1085_108503


namespace exists_m_divisible_by_1988_l1085_108580

def f (x : ℕ) : ℕ := 3 * x + 2
def iter_function (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => f (iter_function n x)

theorem exists_m_divisible_by_1988 : ∃ m : ℕ, 1988 ∣ iter_function 100 m :=
by sorry

end exists_m_divisible_by_1988_l1085_108580


namespace can_encode_number_l1085_108516

theorem can_encode_number : ∃ (m n : ℕ), (0.07 = 1 / (m : ℝ) + 1 / (n : ℝ)) :=
by
  -- Proof omitted
  sorry

end can_encode_number_l1085_108516


namespace compute_fg_l1085_108578

def f (x : ℝ) : ℝ := 4 * x - 1
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem compute_fg : f (g (-3)) = 3 := by
  sorry

end compute_fg_l1085_108578


namespace max_distance_circle_to_line_l1085_108549

open Real

-- Definitions of polar equations and transformations to Cartesian coordinates
def circle_eq (ρ θ : ℝ) : Prop := (ρ = 8 * sin θ)
def line_eq (θ : ℝ) : Prop := (θ = π / 3)

-- Cartesian coordinate transformations
def circle_cartesian (x y : ℝ) : Prop := (x^2 + (y - 4)^2 = 16)
def line_cartesian (x y : ℝ) : Prop := (y = sqrt 3 * x)

-- Maximum distance problem statement
theorem max_distance_circle_to_line : 
  ∀ (x y : ℝ), circle_cartesian x y → 
  (∀ x y, line_cartesian x y → 
  ∃ d : ℝ, d = 6) :=
by
  sorry

end max_distance_circle_to_line_l1085_108549


namespace decimal_to_fraction_l1085_108527

theorem decimal_to_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by sorry

end decimal_to_fraction_l1085_108527


namespace sine_cosine_obtuse_angle_l1085_108595

theorem sine_cosine_obtuse_angle :
  ∀ P : (ℝ × ℝ), P = (Real.sin 2, Real.cos 2) → (Real.sin 2 > 0) ∧ (Real.cos 2 < 0) → 
  (P.1 > 0) ∧ (P.2 < 0) :=
by
  sorry

end sine_cosine_obtuse_angle_l1085_108595


namespace fraction_of_shaded_area_is_11_by_12_l1085_108537

noncomputable def shaded_fraction_of_square : ℚ :=
  let s : ℚ := 1 -- Assume the side length of the square is 1 for simplicity.
  let P := (0, s / 2)
  let Q := (s / 3, s)
  let V := (0, s)
  let base := s / 2
  let height := s / 3
  let triangle_area := (1 / 2) * base * height
  let square_area := s * s
  let shaded_area := square_area - triangle_area
  shaded_area / square_area

theorem fraction_of_shaded_area_is_11_by_12 : shaded_fraction_of_square = 11 / 12 :=
  sorry

end fraction_of_shaded_area_is_11_by_12_l1085_108537


namespace eccentricity_ellipse_l1085_108552

variable (a b : ℝ) (h1 : a > b) (h2 : b > 0)
variable (c : ℝ) (h3 : c = Real.sqrt (a ^ 2 - b ^ 2))
variable (h4 : b = c)
variable (ellipse_eq : ∀ x y, (x^2)/(a^2) + (y^2)/(b^2) = 1)

theorem eccentricity_ellipse :
  c / a = Real.sqrt 2 / 2 :=
by
  sorry

end eccentricity_ellipse_l1085_108552


namespace prod_mod_6_l1085_108517

theorem prod_mod_6 (h1 : 2015 % 6 = 3) (h2 : 2016 % 6 = 0) (h3 : 2017 % 6 = 1) (h4 : 2018 % 6 = 2) : 
  (2015 * 2016 * 2017 * 2018) % 6 = 0 := 
by 
  sorry

end prod_mod_6_l1085_108517


namespace animal_legs_count_l1085_108508

-- Let's define the conditions first.
def total_animals : ℕ := 12
def chickens : ℕ := 5
def chicken_legs : ℕ := 2
def sheep_legs : ℕ := 4

-- Define the statement that we need to prove.
theorem animal_legs_count :
  ∃ (total_legs : ℕ), total_legs = 38 :=
by
  -- Adding the condition for total number of legs
  let sheep := total_animals - chickens
  let total_legs := (chickens * chicken_legs) + (sheep * sheep_legs)
  existsi total_legs
  -- Question proves the correct answer
  sorry

end animal_legs_count_l1085_108508


namespace inequality_transformation_l1085_108591

theorem inequality_transformation (x y : ℝ) (h : x > y) : 3 * x > 3 * y :=
by sorry

end inequality_transformation_l1085_108591


namespace alice_winning_strategy_l1085_108596

theorem alice_winning_strategy (n : ℕ) (hn : n ≥ 2) : 
  (Alice_has_winning_strategy ↔ n % 4 = 3) :=
sorry

end alice_winning_strategy_l1085_108596


namespace range_of_a1_l1085_108507

theorem range_of_a1 {a : ℕ → ℝ} (h_seq : ∀ n : ℕ, 2 * a (n + 1) * a n + a (n + 1) - 3 * a n = 0)
  (h_a1_positive : a 1 > 0) :
  (0 < a 1) ∧ (a 1 < 1) ↔ ∀ m n : ℕ, m < n → a m < a n := by
  sorry

end range_of_a1_l1085_108507


namespace find_r_divisibility_l1085_108567

theorem find_r_divisibility :
  ∃ r : ℝ, (10 * r ^ 2 - 4 * r - 26 = 0 ∧ (r = (19 / 10) ∨ r = (-3 / 2))) ∧ (r = -3 / 2) ∧ (10 * r ^ 3 - 5 * r ^ 2 - 52 * r + 60 = 0) :=
by
  sorry

end find_r_divisibility_l1085_108567


namespace total_area_l1085_108557

-- Defining basic dimensions as conditions
def left_vertical_length : ℕ := 7
def top_horizontal_length_left : ℕ := 5
def left_vertical_length_near_top : ℕ := 3
def top_horizontal_length_right_of_center : ℕ := 2
def right_vertical_length_near_center : ℕ := 3
def top_horizontal_length_far_right : ℕ := 2

-- Defining areas of partitioned rectangles
def area_bottom_left_rectangle : ℕ := 7 * 8
def area_middle_rectangle : ℕ := 5 * 3
def area_top_left_rectangle : ℕ := 2 * 8
def area_top_right_rectangle : ℕ := 2 * 7
def area_bottom_right_rectangle : ℕ := 4 * 4

-- Calculate the total area of the figure
theorem total_area : 
  area_bottom_left_rectangle + area_middle_rectangle + area_top_left_rectangle + area_top_right_rectangle + area_bottom_right_rectangle = 117 := by
  -- Proof steps will go here
  sorry

end total_area_l1085_108557


namespace no_super_plus_good_exists_at_most_one_super_plus_good_l1085_108589

def is_super_plus_good (board : ℕ → ℕ → ℕ) (n : ℕ) (i j : ℕ) : Prop :=
  (∀ k, k < n → board i k ≤ board i j) ∧ 
  (∀ k, k < n → board k j ≥ board i j)

def arrangement (n : ℕ) := { board : ℕ → ℕ → ℕ // ∀ i j, i < n → j < n → 1 ≤ board i j ∧ board i j ≤ n * n }

-- Prove that in some arrangements, there is no super-plus-good number.
theorem no_super_plus_good_exists (n : ℕ) (h₁ : n = 8) :
  ∃ (b : arrangement n), ∀ i j, ¬ is_super_plus_good b.val n i j := sorry

-- Prove that in every arrangement, there is at most one super-plus-good number.
theorem at_most_one_super_plus_good (n : ℕ) (h : n = 8) :
  ∀ (b : arrangement n), ∃! i j, is_super_plus_good b.val n i j := sorry

end no_super_plus_good_exists_at_most_one_super_plus_good_l1085_108589


namespace find_m_l1085_108525

variable {a b c m : ℝ}

theorem find_m (h1 : a + b = 4)
               (h2 : a * b = m)
               (h3 : b + c = 8)
               (h4 : b * c = 5 * m) : m = 0 ∨ m = 3 :=
by {
  sorry
}

end find_m_l1085_108525


namespace certain_number_l1085_108568

theorem certain_number (x y : ℝ) (h1 : 0.20 * x = 0.15 * y - 15) (h2 : x = 1050) : y = 1500 :=
by
  sorry

end certain_number_l1085_108568


namespace trig_identity_l1085_108536

theorem trig_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  (Real.sin θ + Real.cos θ) / Real.sin θ + Real.sin θ * Real.sin θ = 23 / 10 :=
sorry

end trig_identity_l1085_108536


namespace intersection_x_sum_l1085_108551

theorem intersection_x_sum :
  ∃ x : ℤ, (0 ≤ x ∧ x < 17) ∧ (4 * x + 3 ≡ 13 * x + 14 [ZMOD 17]) ∧ x = 5 :=
by
  sorry

end intersection_x_sum_l1085_108551


namespace smallest_b_for_perfect_square_l1085_108584

theorem smallest_b_for_perfect_square (b : ℤ) (h1 : b > 4) (h2 : ∃ n : ℤ, 3 * b + 4 = n * n) : b = 7 :=
by
  sorry

end smallest_b_for_perfect_square_l1085_108584


namespace problem_statement_l1085_108532

theorem problem_statement (r p q : ℝ) (h1 : r > 0) (h2 : p * q ≠ 0) (h3 : p^2 * r > q^2 * r) : p^2 > q^2 := 
sorry

end problem_statement_l1085_108532


namespace solve_for_x_l1085_108572

-- Problem definition
def problem_statement (x : ℕ) : Prop :=
  (3 * x / 7 = 15) → x = 35

-- Theorem statement in Lean 4
theorem solve_for_x (x : ℕ) : problem_statement x :=
by
  intros h
  sorry

end solve_for_x_l1085_108572


namespace curve_intersection_l1085_108539

noncomputable def C1 (t : ℝ) (a : ℝ) : ℝ × ℝ :=
  (2 * t + 2 * a, -t)

noncomputable def C2 (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.sin θ, 1 + 2 * Real.cos θ)

theorem curve_intersection (a : ℝ) :
  (∃ t θ : ℝ, C1 t a = C2 θ) ↔ 1 - Real.sqrt 5 ≤ a ∧ a ≤ 1 + Real.sqrt 5 :=
sorry

end curve_intersection_l1085_108539


namespace nth_term_arithmetic_seq_l1085_108538

variable (a_n : Nat → Int)
variable (S : Nat → Int)
variable (a_1 : Int)

-- Conditions
def is_arithmetic_sequence (a_n : Nat → Int) : Prop :=
  ∃ d : Int, ∀ n : Nat, a_n (n + 1) = a_n n + d

def first_term (a_1 : Int) : Prop :=
  a_1 = 1

def sum_first_three_terms (S : Nat → Int) : Prop :=
  S 3 = 9

theorem nth_term_arithmetic_seq :
  (is_arithmetic_sequence a_n) →
  (first_term 1) →
  (sum_first_three_terms S) →
  ∀ n : Nat, a_n n = 2 * n - 1 :=
  sorry

end nth_term_arithmetic_seq_l1085_108538


namespace solve_for_x_l1085_108579

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
by
  sorry

end solve_for_x_l1085_108579


namespace soccer_tournament_matches_l1085_108505

theorem soccer_tournament_matches (n: ℕ):
  n = 20 → ∃ m: ℕ, m = 19 := sorry

end soccer_tournament_matches_l1085_108505


namespace cats_to_dogs_ratio_l1085_108545

theorem cats_to_dogs_ratio (cats dogs : ℕ) (h1 : 2 * dogs = 3 * cats) (h2 : cats = 14) : dogs = 21 :=
by
  sorry

end cats_to_dogs_ratio_l1085_108545


namespace solution_exists_l1085_108506

def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x + a^2

def f' (x a b : ℝ) : ℝ := 3 * x^2 - 2 * a * x - b

theorem solution_exists (a b : ℝ) :
    f 1 a b = 10 ∧ f' 1 a b = 0 ↔ (a = -4 ∧ b = 11) :=
by 
  sorry

end solution_exists_l1085_108506


namespace ratio_shiny_igneous_to_total_l1085_108543

-- Define the conditions
variable (S I SI : ℕ)
variable (SS : ℕ)
variable (h1 : I = S / 2)
variable (h2 : SI = 40)
variable (h3 : S + I = 180)
variable (h4 : SS = S / 5)

-- Statement to prove
theorem ratio_shiny_igneous_to_total (S I SI SS : ℕ) 
  (h1 : I = S / 2) 
  (h2 : SI = 40) 
  (h3 : S + I = 180) 
  (h4 : SS = S / 5) : 
  SI / I = 2 / 3 := 
sorry

end ratio_shiny_igneous_to_total_l1085_108543


namespace lego_count_l1085_108582

theorem lego_count 
  (total_legos : ℕ := 500)
  (used_legos : ℕ := total_legos / 2)
  (missing_legos : ℕ := 5) :
  total_legos - used_legos - missing_legos = 245 := 
sorry

end lego_count_l1085_108582


namespace sum_of_first_5n_l1085_108590

theorem sum_of_first_5n (n : ℕ) (h : (4 * n * (4 * n + 1)) / 2 = (2 * n * (2 * n + 1)) / 2 + 504) :
  (5 * n * (5 * n + 1)) / 2 = 1035 :=
sorry

end sum_of_first_5n_l1085_108590


namespace find_sticker_price_l1085_108509

-- Define the conditions and the question
def storeA_price (x : ℝ) : ℝ := 0.80 * x - 80
def storeB_price (x : ℝ) : ℝ := 0.70 * x - 40
def heather_saves_30 (x : ℝ) : Prop := storeA_price x = storeB_price x + 30

-- Define the main theorem
theorem find_sticker_price : ∃ x : ℝ, heather_saves_30 x ∧ x = 700 :=
by
  sorry

end find_sticker_price_l1085_108509


namespace find_special_5_digit_number_l1085_108592

theorem find_special_5_digit_number :
  ∃! (A : ℤ), (10000 ≤ A ∧ A < 100000) ∧ (A^2 % 100000 = A) ∧ A = 90625 :=
sorry

end find_special_5_digit_number_l1085_108592


namespace amount_in_cup_after_division_l1085_108554

theorem amount_in_cup_after_division (removed remaining cups : ℕ) (h : remaining + removed = 40) : 
  (40 / cups = 8) :=
by
  sorry

end amount_in_cup_after_division_l1085_108554


namespace bookseller_fiction_books_count_l1085_108550

theorem bookseller_fiction_books_count (n : ℕ) (h1 : n.factorial * 6 = 36) : n = 3 :=
sorry

end bookseller_fiction_books_count_l1085_108550


namespace compute_x_l1085_108575

theorem compute_x 
  (x : ℝ) 
  (hx : 0 < x ∧ x < 0.1)
  (hs1 : ∑' n, 4 * x^n = 4 / (1 - x))
  (hs2 : ∑' n, 4 * (10^n - 1) * x^n = 4 * (4 / (1 - x))) :
  x = 3 / 40 :=
by
  sorry

end compute_x_l1085_108575


namespace men_became_absent_l1085_108573

theorem men_became_absent (original_men planned_days actual_days : ℕ) (h1 : original_men = 48) (h2 : planned_days = 15) (h3 : actual_days = 18) :
  ∃ x : ℕ, 48 * 15 = (48 - x) * 18 ∧ x = 8 :=
by
  sorry

end men_became_absent_l1085_108573


namespace area_outside_squares_inside_triangle_l1085_108577

noncomputable def side_length_large_square : ℝ := 6
noncomputable def side_length_small_square1 : ℝ := 2
noncomputable def side_length_small_square2 : ℝ := 3
noncomputable def area_large_square := side_length_large_square ^ 2
noncomputable def area_small_square1 := side_length_small_square1 ^ 2
noncomputable def area_small_square2 := side_length_small_square2 ^ 2
noncomputable def area_triangle_EFG := area_large_square / 2
noncomputable def total_area_small_squares := area_small_square1 + area_small_square2

theorem area_outside_squares_inside_triangle :
  (area_triangle_EFG - total_area_small_squares) = 5 :=
by
  sorry

end area_outside_squares_inside_triangle_l1085_108577


namespace total_exercise_time_l1085_108556

-- Definition of constants and speeds for each day
def monday_speed := 2 -- miles per hour
def wednesday_speed := 3 -- miles per hour
def friday_speed := 6 -- miles per hour
def distance := 6 -- miles

-- Function to calculate time given distance and speed
def time (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

-- Prove the total time spent in a week
theorem total_exercise_time :
  time distance monday_speed + time distance wednesday_speed + time distance friday_speed = 6 :=
by
  -- Insert detailed proof steps here
  sorry

end total_exercise_time_l1085_108556


namespace farmer_land_l1085_108555

-- Define A to be the total land owned by the farmer
variables (A : ℝ)

-- Define the conditions of the problem
def condition_1 (A : ℝ) : ℝ := 0.90 * A
def condition_2 (cleared_land : ℝ) : ℝ := 0.20 * cleared_land
def condition_3 (cleared_land : ℝ) : ℝ := 0.70 * cleared_land
def condition_4 (cleared_land : ℝ) : ℝ := cleared_land - condition_2 cleared_land - condition_3 cleared_land

-- Define the assertion we need to prove
theorem farmer_land (h : condition_4 (condition_1 A) = 630) : A = 7000 :=
by
  sorry

end farmer_land_l1085_108555


namespace xyz_value_l1085_108586

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) : 
  x * y * z = 4 := by
  sorry

end xyz_value_l1085_108586


namespace total_roasted_marshmallows_l1085_108553

-- Definitions based on problem conditions
def dadMarshmallows : ℕ := 21
def joeMarshmallows := 4 * dadMarshmallows
def dadRoasted := dadMarshmallows / 3
def joeRoasted := joeMarshmallows / 2

-- Theorem to prove the total roasted marshmallows
theorem total_roasted_marshmallows : dadRoasted + joeRoasted = 49 := by
  sorry -- Proof omitted

end total_roasted_marshmallows_l1085_108553


namespace inequality_proof_l1085_108581

variable {m n : ℝ}

theorem inequality_proof (h1 : m < n) (h2 : n < 0) : (n / m + m / n > 2) := 
by
  sorry

end inequality_proof_l1085_108581


namespace phase_shift_of_sine_l1085_108529

theorem phase_shift_of_sine (b c : ℝ) (h_b : b = 4) (h_c : c = - (Real.pi / 2)) :
  (-c / b) = Real.pi / 8 :=
by
  rw [h_b, h_c]
  sorry

end phase_shift_of_sine_l1085_108529


namespace quadratic_inequality_solution_set_l1085_108576

theorem quadratic_inequality_solution_set (a b c : ℝ) : 
  (∀ x : ℝ, - (a / 3) * x^2 + 2 * b * x - c < 0) ↔ (a > 0 ∧ 4 * b^2 - (4 / 3) * a * c < 0) := 
by
  sorry

end quadratic_inequality_solution_set_l1085_108576


namespace original_class_strength_l1085_108546

theorem original_class_strength (x : ℕ) 
    (avg_original : ℕ)
    (num_new : ℕ) 
    (avg_new : ℕ) 
    (decrease : ℕ)
    (h1 : avg_original = 40)
    (h2 : num_new = 17)
    (h3 : avg_new = 32)
    (h4 : decrease = 4)
    (h5 : (40 * x + 17 * avg_new) = (x + num_new) * (40 - decrease))
    : x = 17 := 
by {
  sorry
}

end original_class_strength_l1085_108546


namespace polynomial_zero_pairs_l1085_108520

theorem polynomial_zero_pairs (r s : ℝ) :
  (∀ x : ℝ, (x = 0 ∨ x = 0) ↔ x^2 - 2 * r * x + r = 0) ∧
  (∀ x : ℝ, (x = 0 ∨ x = 0 ∨ x = 0) ↔ 27 * x^3 - 27 * r * x^2 + s * x - r^6 = 0) → 
  (r, s) = (0, 0) ∨ (r, s) = (1, 9) :=
by
  sorry

end polynomial_zero_pairs_l1085_108520


namespace range_of_m_l1085_108540

theorem range_of_m (m : ℝ) (h : ∃ x : ℝ, x < 0 ∧ mx^2 + 2*x + 1 = 0) : m ∈ Set.Iic 1 :=
sorry

end range_of_m_l1085_108540


namespace largest_digit_divisible_by_6_l1085_108530

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

theorem largest_digit_divisible_by_6 (N : ℕ) (hN : N ≤ 9) :
  (∃ m : ℕ, 56780 + N = m * 6) ∧ is_even N ∧ is_divisible_by_3 (26 + N) → N = 4 := by
  sorry

end largest_digit_divisible_by_6_l1085_108530


namespace Cornelia_current_age_l1085_108512

theorem Cornelia_current_age (K : ℕ) (C : ℕ) (h1 : K = 20) (h2 : C + 10 = 3 * (K + 10)) : C = 80 :=
by
  sorry

end Cornelia_current_age_l1085_108512


namespace m_range_positive_real_number_l1085_108558

theorem m_range_positive_real_number (m : ℝ) (x : ℝ) 
  (h : m * x - 1 = 2 * x) (h_pos : x > 0) : m > 2 :=
sorry

end m_range_positive_real_number_l1085_108558


namespace line_in_slope_intercept_form_l1085_108524

def vec1 : ℝ × ℝ := (3, -7)
def point : ℝ × ℝ := (-2, 4)
def line_eq (x y : ℝ) : Prop := vec1.1 * (x - point.1) + vec1.2 * (y - point.2) = 0

theorem line_in_slope_intercept_form (x y : ℝ) : line_eq x y → y = (3 / 7) * x - (34 / 7) :=
by
  sorry

end line_in_slope_intercept_form_l1085_108524


namespace ellen_smoothie_total_l1085_108569

theorem ellen_smoothie_total :
  0.2 + 0.1 + 0.2 + 0.15 + 0.05 = 0.7 :=
by sorry

end ellen_smoothie_total_l1085_108569


namespace Jeff_Jogging_Extra_Friday_l1085_108597

theorem Jeff_Jogging_Extra_Friday :
  let planned_daily_minutes := 60
  let days_in_week := 5
  let planned_weekly_minutes := days_in_week * planned_daily_minutes
  let thursday_cut_short := 20
  let actual_weekly_minutes := 290
  let thursday_run := planned_daily_minutes - thursday_cut_short
  let other_four_days_minutes := actual_weekly_minutes - thursday_run
  let mondays_to_wednesdays_run := 3 * planned_daily_minutes
  let friday_run := other_four_days_minutes - mondays_to_wednesdays_run
  let extra_run_on_friday := friday_run - planned_daily_minutes
  extra_run_on_friday = 10 := by trivial

end Jeff_Jogging_Extra_Friday_l1085_108597


namespace domain_of_h_l1085_108528

noncomputable def h (x : ℝ) : ℝ :=
  (x^2 - 9) / (abs (x - 4) + x^2 - 1)

theorem domain_of_h :
  ∀ (x : ℝ), x ≠ (1 + Real.sqrt 13) / 2 → (abs (x - 4) + x^2 - 1) ≠ 0 :=
sorry

end domain_of_h_l1085_108528


namespace nine_digit_positive_integers_l1085_108513

theorem nine_digit_positive_integers :
  (∃ n : Nat, 10^8 * 9 = n ∧ n = 900000000) :=
sorry

end nine_digit_positive_integers_l1085_108513


namespace ring_roads_count_l1085_108594

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem ring_roads_count : 
  binomial 8 4 * binomial 8 4 - (binomial 10 4 * binomial 6 4) = 1750 := by 
sorry

end ring_roads_count_l1085_108594


namespace quotient_of_division_l1085_108534

theorem quotient_of_division (S L Q : ℕ) (h1 : S = 270) (h2 : L - S = 1365) (h3 : L % S = 15) : Q = 6 :=
by
  sorry

end quotient_of_division_l1085_108534


namespace donor_multiple_l1085_108542

def cost_per_box (food_cost : ℕ) (supplies_cost : ℕ) : ℕ := food_cost + supplies_cost

def total_initial_cost (num_boxes : ℕ) (cost_per_box : ℕ) : ℕ := num_boxes * cost_per_box

def additional_boxes (total_boxes : ℕ) (initial_boxes : ℕ) : ℕ := total_boxes - initial_boxes

def donor_contribution (additional_boxes : ℕ) (cost_per_box : ℕ) : ℕ := additional_boxes * cost_per_box

def multiple (donor_contribution : ℕ) (initial_cost : ℕ) : ℕ := donor_contribution / initial_cost

theorem donor_multiple 
    (initial_boxes : ℕ) (box_cost : ℕ) (total_boxes : ℕ) (donor_multi : ℕ)
    (h1 : initial_boxes = 400) 
    (h2 : box_cost = 245) 
    (h3 : total_boxes = 2000)
    : donor_multi = 4 :=
by
    let initial_cost := total_initial_cost initial_boxes box_cost
    let additional_boxes := additional_boxes total_boxes initial_boxes
    let contribution := donor_contribution additional_boxes box_cost
    have h4 : contribution = 392000 := sorry
    have h5 : initial_cost = 98000 := sorry
    have h6 : donor_multi = contribution / initial_cost := sorry
    -- Therefore, the multiple should be 4
    exact sorry

end donor_multiple_l1085_108542


namespace three_op_six_l1085_108564

-- Define the new operation @.
def op (a b : ℕ) : ℕ := (a * a * b) / (a + b)

-- The theorem to prove that the value of 3 @ 6 is 6.
theorem three_op_six : op 3 6 = 6 := by 
  sorry

end three_op_six_l1085_108564


namespace find_x_eq_3_l1085_108501

noncomputable def f (x : ℝ) : ℝ := 4 * x - 9
noncomputable def f_inv (x : ℝ) : ℝ := (x + 9) / 4

theorem find_x_eq_3 : ∃ x : ℝ, f x = f_inv x ∧ x = 3 :=
by
  sorry

end find_x_eq_3_l1085_108501


namespace num_divisors_720_l1085_108504

-- Define the number 720 and its prime factorization
def n : ℕ := 720
def pf : List (ℕ × ℕ) := [(2, 4), (3, 2), (5, 1)]

-- Define the function to calculate the number of divisors from prime factorization
def num_divisors (pf : List (ℕ × ℕ)) : ℕ :=
  pf.foldr (λ p acc => acc * (p.snd + 1)) 1

-- Statement to prove
theorem num_divisors_720 : num_divisors pf = 30 :=
  by
  -- Placeholder for the actual proof
  sorry

end num_divisors_720_l1085_108504


namespace range_of_m_l1085_108563

def P (m : ℝ) : Prop := m^2 - 4 > 0
def Q (m : ℝ) : Prop := 16 * (m - 2)^2 - 16 < 0

theorem range_of_m (m : ℝ) :
  (P m ∨ Q m) ∧ ¬(P m ∧ Q m) ↔ (m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) :=
by
  sorry

end range_of_m_l1085_108563


namespace cannot_invert_all_signs_l1085_108511

structure RegularDecagon :=
  (vertices : Fin 10 → ℤ)
  (diagonals : Fin 45 → ℤ) -- Assume we encode the intersections as unique indices for simplicity.
  (all_positives : ∀ v, vertices v = 1 ∧ ∀ d, diagonals d = 1)

def isValidSignChange (t : List ℤ) : Prop :=
  t.length % 2 = 0

theorem cannot_invert_all_signs (D : RegularDecagon) :
  ¬ (∃ f : Fin 10 → ℤ → ℤ, ∀ (side : Fin 10) (val : ℤ), f side val = -val) :=
sorry

end cannot_invert_all_signs_l1085_108511


namespace can_determine_number_of_spies_l1085_108541

def determine_spies (V : Fin 15 → ℕ) (S : Fin 15 → ℕ) : Prop :=
  V 0 = S 0 + S 1 ∧ 
  ∀ i : Fin 13, V (Fin.succ (Fin.succ i)) = S i + S (Fin.succ i) + S (Fin.succ (Fin.succ i)) ∧
  V 14 = S 13 + S 14

theorem can_determine_number_of_spies :
  ∃ S : Fin 15 → ℕ, ∀ V : Fin 15 → ℕ, determine_spies V S :=
sorry

end can_determine_number_of_spies_l1085_108541


namespace solve_equation_in_natural_numbers_l1085_108588

-- Define the main theorem
theorem solve_equation_in_natural_numbers :
  (∃ (x y z : ℕ), 2^x + 5^y + 63 = z! ∧ ((x = 5 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6))) :=
sorry

end solve_equation_in_natural_numbers_l1085_108588


namespace calculate_difference_l1085_108521

theorem calculate_difference (x y : ℝ) (h1 : x + y = 520) (h2 : x / y = 0.75) : y - x = 74 :=
by
  sorry

end calculate_difference_l1085_108521


namespace integer_root_b_l1085_108514

theorem integer_root_b (a1 a2 a3 a4 a5 b : ℤ)
  (h_diff : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ a3 ≠ a4 ∧ a3 ≠ a5 ∧ a4 ≠ a5)
  (h_sum : a1 + a2 + a3 + a4 + a5 = 9)
  (h_prod : (b - a1) * (b - a2) * (b - a3) * (b - a4) * (b - a5) = 2009) :
  b = 10 :=
sorry

end integer_root_b_l1085_108514


namespace convert_mps_to_kmph_l1085_108522

theorem convert_mps_to_kmph (v_mps : ℝ) (c : ℝ) (h_c : c = 3.6) (h_v_mps : v_mps = 20) : (v_mps * c = 72) :=
by
  rw [h_v_mps, h_c]
  sorry

end convert_mps_to_kmph_l1085_108522


namespace factorize_x_squared_minus_one_l1085_108583

theorem factorize_x_squared_minus_one (x : Real) : (x^2 - 1) = (x + 1) * (x - 1) :=
sorry

end factorize_x_squared_minus_one_l1085_108583


namespace inequality_three_var_l1085_108526

theorem inequality_three_var
  (a b c : ℝ)
  (ha : 0 ≤ a)
  (hb : 0 ≤ b)
  (hc : 0 ≤ c) :
  2 * (a^3 + b^3 + c^3) ≥ a^2 * b + a * b^2 + a^2 * c + a * c^2 + b^2 * c + b * c^2 :=
by sorry

end inequality_three_var_l1085_108526


namespace solve_equation_l1085_108518

open Function

theorem solve_equation (m n : ℕ) (h_gcd : gcd m n = 2) (h_lcm : lcm m n = 4) :
  m * n = (gcd m n)^2 + lcm m n ↔ (m = 2 ∧ n = 4) ∨ (m = 4 ∧ n = 2) :=
by
  sorry

end solve_equation_l1085_108518


namespace necessary_and_sufficient_condition_l1085_108535

-- Definitions for sides opposite angles A, B, and C in a triangle.
variables {A B C : Real} {a b c : Real}

-- Condition p: sides a, b related to angles A, B via cosine
def condition_p (a b : Real) (A B : Real) : Prop := a / Real.cos A = b / Real.cos B

-- Condition q: sides a and b are equal
def condition_q (a b : Real) : Prop := a = b

theorem necessary_and_sufficient_condition (h1 : condition_p a b A B) : condition_q a b ↔ condition_p a b A B :=
by
  sorry

end necessary_and_sufficient_condition_l1085_108535


namespace initial_distance_l1085_108565

-- Define conditions
def fred_speed : ℝ := 4
def sam_speed : ℝ := 4
def sam_distance_when_meet : ℝ := 20

-- States that the initial distance between Fred and Sam is 40 miles considering the given conditions.
theorem initial_distance (d : ℝ) (fred_speed_eq : fred_speed = 4) (sam_speed_eq : sam_speed = 4) (sam_distance_eq : sam_distance_when_meet = 20) :
  d = 40 :=
  sorry

end initial_distance_l1085_108565


namespace base_b_prime_digits_l1085_108544

theorem base_b_prime_digits (b' : ℕ) (h1 : b'^4 ≤ 216) (h2 : 216 < b'^5) : b' = 3 :=
by {
  sorry
}

end base_b_prime_digits_l1085_108544


namespace sum_of_squares_five_consecutive_ints_not_perfect_square_l1085_108523

theorem sum_of_squares_five_consecutive_ints_not_perfect_square (n : ℤ) :
  ∀ k : ℤ, k^2 ≠ 5 * (n^2 + 2) := 
sorry

end sum_of_squares_five_consecutive_ints_not_perfect_square_l1085_108523


namespace simplify_absolute_values_l1085_108574

theorem simplify_absolute_values (a : ℝ) (h : -2 < a ∧ a < 0) : |a| + |a + 2| = 2 :=
sorry

end simplify_absolute_values_l1085_108574


namespace minimum_chess_pieces_l1085_108510

theorem minimum_chess_pieces (n : ℕ) : 
  (n % 3 = 1) ∧ (n % 5 = 3) ∧ (n % 7 = 5) → 
  n = 103 :=
by 
  sorry

end minimum_chess_pieces_l1085_108510


namespace zeros_of_quadratic_l1085_108566

def f (x : ℝ) := x^2 - 2 * x - 3

theorem zeros_of_quadratic : ∀ x, f x = 0 ↔ (x = 3 ∨ x = -1) := 
by 
  sorry

end zeros_of_quadratic_l1085_108566


namespace correct_statements_identification_l1085_108531

-- Definitions based on given conditions
def syntheticMethodCauseToEffect := True
def syntheticMethodForward := True
def analyticMethodEffectToCause := True
def analyticMethodIndirect := False
def analyticMethodBackward := True

-- The main statement to be proved
theorem correct_statements_identification :
  (syntheticMethodCauseToEffect = True) ∧ 
  (syntheticMethodForward = True) ∧ 
  (analyticMethodEffectToCause = True) ∧ 
  (analyticMethodBackward = True) ∧ 
  (analyticMethodIndirect = False) :=
by
  sorry

end correct_statements_identification_l1085_108531


namespace area_after_trimming_l1085_108560

-- Define the conditions
def original_side_length : ℝ := 22
def trim_x : ℝ := 6
def trim_y : ℝ := 5

-- Calculate dimensions after trimming
def new_length : ℝ := original_side_length - trim_x
def new_width : ℝ := original_side_length - trim_y

-- Define the goal
theorem area_after_trimming : new_length * new_width = 272 := by
  sorry

end area_after_trimming_l1085_108560


namespace derivative_at_five_l1085_108599

noncomputable def g (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3)

theorem derivative_at_five : deriv g 5 = 26 :=
sorry

end derivative_at_five_l1085_108599


namespace farm_distance_is_6_l1085_108587

noncomputable def distance_to_farm (initial_gallons : ℕ) 
  (consumption_rate : ℕ) (supermarket_distance : ℕ) 
  (outbound_distance : ℕ) (remaining_gallons : ℕ) : ℕ :=
initial_gallons * consumption_rate - 
  (2 * supermarket_distance + 2 * outbound_distance - remaining_gallons * consumption_rate)

theorem farm_distance_is_6 : 
  distance_to_farm 12 2 5 2 2 = 6 :=
by
  sorry

end farm_distance_is_6_l1085_108587


namespace smallest_multiple_of_37_smallest_multiple_of_37_verification_l1085_108502

theorem smallest_multiple_of_37 (x : ℕ) (h : 37 * x % 97 = 3) :
  x = 15 := sorry

theorem smallest_multiple_of_37_verification :
  37 * 15 = 555 := rfl

end smallest_multiple_of_37_smallest_multiple_of_37_verification_l1085_108502
