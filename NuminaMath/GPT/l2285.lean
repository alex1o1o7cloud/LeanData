import Mathlib

namespace circle_tangent_line_l2285_228570

theorem circle_tangent_line {m : ℝ} : 
  (3 * (0 : ℝ) - 4 * (1 : ℝ) - 6 = 0) ∧ 
  (∀ x y : ℝ, x^2 + y^2 - 2 * y + m = 0) → 
  m = -3 := by
  sorry

end circle_tangent_line_l2285_228570


namespace scientific_notation_43300000_l2285_228561

theorem scientific_notation_43300000 : 43300000 = 4.33 * 10^7 :=
by
  sorry

end scientific_notation_43300000_l2285_228561


namespace tan_nine_pi_over_three_l2285_228522

theorem tan_nine_pi_over_three : Real.tan (9 * Real.pi / 3) = 0 := by
  sorry

end tan_nine_pi_over_three_l2285_228522


namespace rate_of_markup_l2285_228519

theorem rate_of_markup (S : ℝ) (hS : S = 8)
  (profit_percent : ℝ) (h_profit_percent : profit_percent = 0.20)
  (expense_percent : ℝ) (h_expense_percent : expense_percent = 0.10) :
  (S - (S * (1 - profit_percent - expense_percent))) / (S * (1 - profit_percent - expense_percent)) * 100 = 42.857 :=
by
  sorry

end rate_of_markup_l2285_228519


namespace distance_from_top_correct_total_distance_covered_correct_fifth_climb_success_l2285_228588

-- We will assume the depth of the well as a constant
def well_depth : ℝ := 4.0

-- Climb and slide distances as per each climb
def first_climb : ℝ := 1.2
def first_slide : ℝ := 0.4
def second_climb : ℝ := 1.4
def second_slide : ℝ := 0.5
def third_climb : ℝ := 1.1
def third_slide : ℝ := 0.3
def fourth_climb : ℝ := 1.2
def fourth_slide : ℝ := 0.2

noncomputable def net_gain_four_climbs : ℝ :=
  (first_climb - first_slide) + (second_climb - second_slide) +
  (third_climb - third_slide) + (fourth_climb - fourth_slide)

noncomputable def distance_from_top_after_four : ℝ := 
  well_depth - net_gain_four_climbs

noncomputable def total_distance_covered_four_climbs : ℝ :=
  first_climb + first_slide + second_climb + second_slide +
  third_climb + third_slide + fourth_climb + fourth_slide

noncomputable def can_climb_out_fifth_climb : Bool :=
  well_depth < (net_gain_four_climbs + first_climb)

-- Now we state the theorems we need to prove

theorem distance_from_top_correct :
  distance_from_top_after_four = 0.5 := by
  sorry

theorem total_distance_covered_correct :
  total_distance_covered_four_climbs = 6.3 := by
  sorry

theorem fifth_climb_success :
  can_climb_out_fifth_climb = true := by
  sorry

end distance_from_top_correct_total_distance_covered_correct_fifth_climb_success_l2285_228588


namespace system_of_equations_l2285_228564

theorem system_of_equations (x y : ℝ) 
  (h1 : 2019 * x + 2020 * y = 2018) 
  (h2 : 2020 * x + 2019 * y = 2021) :
  x + y = 1 ∧ x - y = 3 :=
by sorry

end system_of_equations_l2285_228564


namespace first_positive_term_is_7_l2285_228565

-- Define the conditions and the sequence
def a1 : ℚ := -1
def d : ℚ := 1 / 5

-- Define the general term of the sequence
def a_n (n : ℕ) : ℚ := a1 + (n - 1) * d

-- Define the proposition that the 7th term is the first positive term
theorem first_positive_term_is_7 :
  ∀ n : ℕ, (0 < a_n n) → (7 <= n) :=
by
  intro n h
  sorry

end first_positive_term_is_7_l2285_228565


namespace x_is_sufficient_but_not_necessary_for_x_squared_eq_one_l2285_228553

theorem x_is_sufficient_but_not_necessary_for_x_squared_eq_one : 
  (∀ x : ℝ, x = 1 → x^2 = 1) ∧ (∃ x : ℝ, x^2 = 1 ∧ x ≠ 1) :=
by
  sorry

end x_is_sufficient_but_not_necessary_for_x_squared_eq_one_l2285_228553


namespace relationship_of_y1_y2_l2285_228574

theorem relationship_of_y1_y2 (y1 y2 : ℝ) : 
  (∃ y1 y2, (y1 = 2 / -2) ∧ (y2 = 2 / -1)) → (y1 > y2) :=
by
  sorry

end relationship_of_y1_y2_l2285_228574


namespace patio_length_l2285_228556

theorem patio_length (w l : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 100) : l = 40 := 
by 
  sorry

end patio_length_l2285_228556


namespace peggy_buys_three_folders_l2285_228527

theorem peggy_buys_three_folders 
  (red_sheets : ℕ) (green_sheets : ℕ) (blue_sheets : ℕ)
  (red_stickers_per_sheet : ℕ) (green_stickers_per_sheet : ℕ) (blue_stickers_per_sheet : ℕ)
  (total_stickers : ℕ) :
  red_sheets = 10 →
  green_sheets = 10 →
  blue_sheets = 10 →
  red_stickers_per_sheet = 3 →
  green_stickers_per_sheet = 2 →
  blue_stickers_per_sheet = 1 →
  total_stickers = 60 →
  1 + 1 + 1 = 3 :=
by 
  intros _ _ _ _ _ _ _
  sorry

end peggy_buys_three_folders_l2285_228527


namespace arithmetic_sequence_term_13_l2285_228534

variable {a : ℕ → ℝ}
variable {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_term_13 (h_arith : arithmetic_sequence a d)
  (h_a5 : a 5 = 3)
  (h_a9 : a 9 = 6) :
  a 13 = 9 := 
by 
  sorry

end arithmetic_sequence_term_13_l2285_228534


namespace B_can_finish_alone_in_27_5_days_l2285_228592

-- Definitions of work rates
variable (B A C : Type)

-- Conditions translations
def efficiency_of_A (x : ℝ) : Prop := ∀ (work_rate_A work_rate_B : ℝ), work_rate_A = 1 / (2 * x) ∧ work_rate_B = 1 / x
def efficiency_of_C (x : ℝ) : Prop := ∀ (work_rate_C work_rate_B : ℝ), work_rate_C = 1 / (3 * x) ∧ work_rate_B = 1 / x
def combined_work_rate (x : ℝ) : Prop := (1 / (2 * x) + 1 / x + 1 / (3 * x)) = 1 / 15

-- Proof statement
theorem B_can_finish_alone_in_27_5_days :
  ∃ (x : ℝ), efficiency_of_A x ∧ efficiency_of_C x ∧ combined_work_rate x ∧ x = 27.5 :=
sorry

end B_can_finish_alone_in_27_5_days_l2285_228592


namespace value_of_x_that_makes_sqrt_undefined_l2285_228505

theorem value_of_x_that_makes_sqrt_undefined (x : ℕ) (hpos : 0 < x) : (x = 1) ∨ (x = 2) ↔ (x - 3 < 0) := by
  sorry

end value_of_x_that_makes_sqrt_undefined_l2285_228505


namespace divisor_of_polynomial_l2285_228590

theorem divisor_of_polynomial (a : ℤ) (h : ∀ x : ℤ, (x^2 - x + a) ∣ (x^13 + x + 180)) : a = 1 :=
sorry

end divisor_of_polynomial_l2285_228590


namespace fifteen_percent_of_x_equals_sixty_l2285_228514

theorem fifteen_percent_of_x_equals_sixty (x : ℝ) (h : 0.15 * x = 60) : x = 400 :=
by
  sorry

end fifteen_percent_of_x_equals_sixty_l2285_228514


namespace initial_rope_length_l2285_228597

theorem initial_rope_length : 
  ∀ (π : ℝ), 
  ∀ (additional_area : ℝ) (new_rope_length : ℝ), 
  additional_area = 933.4285714285714 →
  new_rope_length = 21 →
  ∃ (initial_rope_length : ℝ), 
  additional_area = π * (new_rope_length^2 - initial_rope_length^2) ∧
  initial_rope_length = 12 :=
by
  sorry

end initial_rope_length_l2285_228597


namespace no_such_triples_l2285_228538

theorem no_such_triples : ¬ ∃ a b c : ℕ, 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  Prime ((a-2)*(b-2)*(c-2)+12) ∧ 
  ((a-2)*(b-2)*(c-2)+12) ∣ (a^2 + b^2 + c^2 + a*b*c - 2017) := 
by sorry

end no_such_triples_l2285_228538


namespace fish_size_difference_l2285_228506

variables {S J W : ℝ}

theorem fish_size_difference (h1 : S = J + 21.52) (h2 : J = W - 12.64) : S - W = 8.88 :=
sorry

end fish_size_difference_l2285_228506


namespace find_b_l2285_228549

-- Define the number 1234567 in base 36
def numBase36 : ℤ := 1 * 36^6 + 2 * 36^5 + 3 * 36^4 + 4 * 36^3 + 5 * 36^2 + 6 * 36^1 + 7 * 36^0

-- Prove that for b being an integer such that 0 ≤ b ≤ 10,
-- and given (numBase36 - b) is a multiple of 17, b must be 0
theorem find_b (b : ℤ) (h1 : 0 ≤ b) (h2 : b ≤ 10) (h3 : (numBase36 - b) % 17 = 0) : b = 0 :=
by
  sorry

end find_b_l2285_228549


namespace floor_expression_bounds_l2285_228503

theorem floor_expression_bounds (x : ℝ) (h : ⌊x * ⌊x / 2⌋⌋ = 12) : 
  4.9 ≤ x ∧ x < 5.1 :=
sorry

end floor_expression_bounds_l2285_228503


namespace exponentiation_problem_l2285_228501

theorem exponentiation_problem : (8^8 / 8^5) * 2^10 * 2^3 = 2^22 := by
  sorry

end exponentiation_problem_l2285_228501


namespace geometric_sequence_third_fourth_terms_l2285_228563

theorem geometric_sequence_third_fourth_terms
  (a : ℕ → ℝ)
  (r : ℝ)
  (ha : ∀ n, a (n + 1) = r * a n)
  (hS2 : a 0 + a 1 = 3 * a 1) :
  (a 2 + a 3) / (a 0 + a 1) = 1 / 4 :=
by
  -- proof to be filled in
  sorry

end geometric_sequence_third_fourth_terms_l2285_228563


namespace part1a_part1b_part2_part3a_part3b_l2285_228575

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

end part1a_part1b_part2_part3a_part3b_l2285_228575


namespace pentomino_symmetry_count_l2285_228539

def is_pentomino (shape : Type) : Prop :=
  -- Define the property of being a pentomino as composed of five squares edge to edge
  sorry

def has_reflectional_symmetry (shape : Type) : Prop :=
  -- Define the property of having at least one line of reflectional symmetry
  sorry

def has_rotational_symmetry_of_order_2 (shape : Type) : Prop :=
  -- Define the property of having rotational symmetry of order 2 (180 degrees rotation results in the same shape)
  sorry

noncomputable def count_valid_pentominoes : Nat :=
  -- Assume that we have a list of 18 pentominoes
  -- Count the number of pentominoes that meet both criteria
  sorry

theorem pentomino_symmetry_count :
  count_valid_pentominoes = 4 :=
sorry

end pentomino_symmetry_count_l2285_228539


namespace integer_remainder_18_l2285_228554

theorem integer_remainder_18 (n : ℤ) (h : n ∈ ({14, 15, 16, 17, 18} : Set ℤ)) : n % 7 = 4 :=
by
  sorry

end integer_remainder_18_l2285_228554


namespace angle_C_is_3pi_over_4_l2285_228595

theorem angle_C_is_3pi_over_4 (A B C : ℝ) (a b c : ℝ) (h_tri : 0 < B ∧ B < π ∧ 0 < C ∧ C < π) 
  (h_eq : b * Real.cos C + c * Real.sin B = 0) : C = 3 * π / 4 :=
by
  sorry

end angle_C_is_3pi_over_4_l2285_228595


namespace trail_length_l2285_228520

variables (a b c d e : ℕ)

theorem trail_length : 
  a + b + c = 45 ∧
  b + d = 36 ∧
  c + d + e = 60 ∧
  a + d = 32 → 
  a + b + c + d + e = 69 :=
by
  intro h
  obtain ⟨h1, h2, h3, h4⟩ := h
  sorry

end trail_length_l2285_228520


namespace sides_of_second_polygon_l2285_228578

theorem sides_of_second_polygon (s : ℝ) (n : ℕ) 
  (perimeter1_is_perimeter2 : 38 * (2 * s) = n * s) : 
  n = 76 := by
  sorry

end sides_of_second_polygon_l2285_228578


namespace largest_multiple_of_9_less_than_100_l2285_228580

theorem largest_multiple_of_9_less_than_100 : ∃ k : ℕ, 9 * k < 100 ∧ (∀ m : ℕ, 9 * m < 100 → 9 * m ≤ 9 * k) ∧ 9 * k = 99 :=
by sorry

end largest_multiple_of_9_less_than_100_l2285_228580


namespace circle_in_fourth_quadrant_l2285_228543

theorem circle_in_fourth_quadrant (a : ℝ) :
  (∃ (x y: ℝ), x^2 + y^2 - 2 * a * x + 4 * a * y + 6 * a^2 - a = 0 ∧ (a > 0) ∧ (-2 * y < 0)) → (0 < a ∧ a < 1) :=
by
  sorry

end circle_in_fourth_quadrant_l2285_228543


namespace area_shaded_region_is_correct_l2285_228573

noncomputable def radius_of_larger_circle : ℝ := 8
noncomputable def radius_of_smaller_circle := radius_of_larger_circle / 2

-- Define areas
noncomputable def area_of_larger_circle := Real.pi * radius_of_larger_circle ^ 2
noncomputable def area_of_smaller_circle := Real.pi * radius_of_smaller_circle ^ 2
noncomputable def total_area_of_smaller_circles := 2 * area_of_smaller_circle
noncomputable def area_of_shaded_region := area_of_larger_circle - total_area_of_smaller_circles

-- Prove that the area of the shaded region is 32π
theorem area_shaded_region_is_correct : area_of_shaded_region = 32 * Real.pi := by
  sorry

end area_shaded_region_is_correct_l2285_228573


namespace walk_usual_time_l2285_228502

theorem walk_usual_time (T : ℝ) (S : ℝ) (h1 : (5 / 4 : ℝ) = (T + 10) / T) : T = 40 :=
sorry

end walk_usual_time_l2285_228502


namespace perpendicular_k_value_exists_l2285_228589

open Real EuclideanSpace

def vector_a : ℝ × ℝ := (-2, 1)
def vector_b : ℝ × ℝ := (3, 2)

theorem perpendicular_k_value_exists : ∃ k : ℝ, (vector_a.1 * (vector_a.1 + k * vector_b.1) + vector_a.2 * (vector_a.2 + k * vector_b.2) = 0) ∧ k = 5 / 4 := by
  sorry

end perpendicular_k_value_exists_l2285_228589


namespace sum_infinite_series_l2285_228593

theorem sum_infinite_series : 
  ∑' n : ℕ, (3 * (n + 1) - 1) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 3)) = 73 / 12 := 
by sorry

end sum_infinite_series_l2285_228593


namespace tim_initial_balls_correct_l2285_228515

-- Defining the initial number of balls Robert had
def robert_initial_balls : ℕ := 25

-- Defining the final number of balls Robert had
def robert_final_balls : ℕ := 45

-- Defining the number of balls Tim had initially
def tim_initial_balls := 40

-- Now, we state the proof problem:
theorem tim_initial_balls_correct :
  robert_initial_balls + (tim_initial_balls / 2) = robert_final_balls :=
by
  -- This is the part where you typically write the proof.
  -- However, we put sorry here because the task does not require the proof itself.
  sorry

end tim_initial_balls_correct_l2285_228515


namespace van_helsing_removed_percentage_l2285_228550

theorem van_helsing_removed_percentage :
  ∀ (V W : ℕ), 
  (5 * V / 2 + 10 * 8 = 105) →
  (W = 4 * V) →
  8 / W * 100 = 20 := 
by
  sorry

end van_helsing_removed_percentage_l2285_228550


namespace sequence_evaluation_l2285_228568

noncomputable def a : ℕ → ℤ → ℤ
| 0, x => 1
| 1, x => x^2 + x + 1
| (n + 2), x => (x^n + 1) * a (n + 1) x - a n x 

theorem sequence_evaluation : a 2010 1 = 4021 := by
  sorry

end sequence_evaluation_l2285_228568


namespace evaluate_expression_l2285_228569

-- Define the mathematical expressions using Lean's constructs
def expr1 : ℕ := 201 * 5 + 1220 - 2 * 3 * 5 * 7

-- State the theorem we aim to prove
theorem evaluate_expression : expr1 = 2015 := by
  sorry

end evaluate_expression_l2285_228569


namespace bottles_per_person_l2285_228562

theorem bottles_per_person
  (boxes : ℕ)
  (bottles_per_box : ℕ)
  (bottles_eaten : ℕ)
  (people : ℕ)
  (total_bottles : ℕ := boxes * bottles_per_box)
  (remaining_bottles : ℕ := total_bottles - bottles_eaten)
  (bottles_per_person : ℕ := remaining_bottles / people) :
  boxes = 7 → bottles_per_box = 9 → bottles_eaten = 7 → people = 8 → bottles_per_person = 7 := 
by
  intros h1 h2 h3 h4
  sorry

end bottles_per_person_l2285_228562


namespace cubic_inequality_l2285_228537

theorem cubic_inequality 
  (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) 
  (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) :
  2 * (x^3 + y^3 + z^3) - (x^2 * y + y^2 * z + z^2 * x) ≤ 3 :=
by
  sorry

end cubic_inequality_l2285_228537


namespace lily_profit_is_correct_l2285_228552

-- Define the conditions
def first_ticket_price : ℕ := 1
def price_increment : ℕ := 1
def number_of_tickets : ℕ := 5
def prize_amount : ℕ := 11

-- Define the sum of arithmetic series formula
def total_amount_collected (n : ℕ) (a : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Calculate the total amount collected
def total : ℕ := total_amount_collected number_of_tickets first_ticket_price price_increment

-- Define the profit calculation
def profit : ℕ := total - prize_amount

-- The statement we need to prove
theorem lily_profit_is_correct : profit = 4 := by
  sorry

end lily_profit_is_correct_l2285_228552


namespace sum_of_roots_of_quadratic_l2285_228585

theorem sum_of_roots_of_quadratic :
  ∀ x : ℝ, (x - 1) * (x + 4) = 18 -> (∃ a b c : ℝ, a = 1 ∧ b = 3 ∧ c = -22 ∧ ((a * x^2 + b * x + c = 0) ∧ (-b / a = -3))) :=
by
  sorry

end sum_of_roots_of_quadratic_l2285_228585


namespace find_m_perpendicular_l2285_228551

-- Define the two vectors
def a (m : ℝ) : ℝ × ℝ := (m, -1)
def b : ℝ × ℝ := (1, 2)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Theorem stating the mathematically equivalent proof problem
theorem find_m_perpendicular (m : ℝ) (h : dot_product (a m) b = 0) : m = 2 :=
by sorry

end find_m_perpendicular_l2285_228551


namespace triangle_inequality_l2285_228559

theorem triangle_inequality
  (R r p : ℝ) (a b c : ℝ)
  (h1 : a * b + b * c + c * a = r^2 + p^2 + 4 * R * r)
  (h2 : 16 * R * r - 5 * r^2 ≤ p^2)
  (h3 : p^2 ≤ 4 * R^2 + 4 * R * r + 3 * r^2):
  20 * R * r - 4 * r^2 ≤ a * b + b * c + c * a ∧ a * b + b * c + c * a ≤ 4 * (R + r)^2 := 
  by
    sorry

end triangle_inequality_l2285_228559


namespace maximize_profit_l2285_228535

-- Conditions
def price_bound (p : ℝ) := p ≤ 22
def books_sold (p : ℝ) := 110 - 4 * p
def profit (p : ℝ) := (p - 2) * books_sold p

-- The main theorem statement
theorem maximize_profit : ∃ p : ℝ, price_bound p ∧ profit p = profit 15 :=
sorry

end maximize_profit_l2285_228535


namespace quotient_of_numbers_l2285_228591

noncomputable def larger_number : ℕ := 22
noncomputable def smaller_number : ℕ := 8

theorem quotient_of_numbers : (larger_number.toFloat / smaller_number.toFloat) = 2.75 := by
  sorry

end quotient_of_numbers_l2285_228591


namespace probability_useful_parts_l2285_228579

noncomputable def probability_three_parts_useful (pipe_length : ℝ) (min_length : ℝ) : ℝ :=
  let total_area := (pipe_length * pipe_length) / 2
  let feasible_area := ((pipe_length - min_length) * (pipe_length - min_length)) / 2
  feasible_area / total_area

theorem probability_useful_parts :
  probability_three_parts_useful 300 75 = 1 / 16 :=
by
  sorry

end probability_useful_parts_l2285_228579


namespace boat_speed_in_still_water_l2285_228521

theorem boat_speed_in_still_water (B S : ℝ) (h1 : B + S = 6) (h2 : B - S = 4) : B = 5 := by
  sorry

end boat_speed_in_still_water_l2285_228521


namespace range_of_k_l2285_228516

theorem range_of_k (k : ℝ) :
  (∀ (x1 : ℝ), x1 ∈ Set.Icc (-1 : ℝ) 3 →
    ∃ (x0 : ℝ), x0 ∈ Set.Icc (-1 : ℝ) 3 ∧ (2 * x1^2 + x1 - k) ≤ (x0^3 - 3 * x0)) →
  k ≥ 3 :=
by
  -- This is the place for the proof. 'sorry' is used to indicate that the proof is omitted.
  sorry

end range_of_k_l2285_228516


namespace m_value_l2285_228584

theorem m_value (m : ℝ) (h : (243:ℝ) ^ (1/3) = 3 ^ m) : m = 5 / 3 :=
sorry

end m_value_l2285_228584


namespace intersection_eq_union_eq_complement_union_eq_intersection_complements_eq_l2285_228557

-- Definitions for U, A, B
def U := { x : ℤ | 0 < x ∧ x <= 10 }
def A : Set ℤ := { 1, 2, 4, 5, 9 }
def B : Set ℤ := { 4, 6, 7, 8, 10 }

-- 1. Prove A ∩ B = {4}
theorem intersection_eq : A ∩ B = {4} := by
  sorry

-- 2. Prove A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}
theorem union_eq : A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10} := by
  sorry

-- 3. Prove complement_U (A ∪ B) = {3}
def complement_U (s : Set ℤ) : Set ℤ := { x ∈ U | ¬ (x ∈ s) }
theorem complement_union_eq : complement_U (A ∪ B) = {3} := by
  sorry

-- 4. Prove (complement_U A) ∩ (complement_U B) = {3}
theorem intersection_complements_eq : (complement_U A) ∩ (complement_U B) = {3} := by
  sorry

end intersection_eq_union_eq_complement_union_eq_intersection_complements_eq_l2285_228557


namespace rectangle_area_l2285_228517

theorem rectangle_area (length : ℝ) (width : ℝ) (area : ℝ) 
  (h1 : length = 24) 
  (h2 : width = 0.875 * length) 
  (h3 : area = length * width) : 
  area = 504 := 
by
  sorry

end rectangle_area_l2285_228517


namespace original_weight_of_beef_l2285_228572

theorem original_weight_of_beef (w_after : ℝ) (loss_percentage : ℝ) (w_before : ℝ) : 
  (w_after = 550) → (loss_percentage = 0.35) → (w_after = 550) → (w_before = 846.15) :=
by
  intros
  sorry

end original_weight_of_beef_l2285_228572


namespace new_mean_rent_is_880_l2285_228541

theorem new_mean_rent_is_880
  (num_friends : ℕ)
  (initial_average_rent : ℝ)
  (increase_percentage : ℝ)
  (original_rent_increased : ℝ)
  (new_mean_rent : ℝ) :
  num_friends = 4 →
  initial_average_rent = 800 →
  increase_percentage = 20 →
  original_rent_increased = 1600 →
  new_mean_rent = 880 :=
by
  intros h1 h2 h3 h4
  sorry

end new_mean_rent_is_880_l2285_228541


namespace factor_x8_minus_81_l2285_228509

theorem factor_x8_minus_81 (x : ℝ) : x^8 - 81 = (x^2 - 3) * (x^2 + 3) * (x^4 + 9) := 
by 
  sorry

end factor_x8_minus_81_l2285_228509


namespace intersection_A_B_l2285_228504

def A : Set ℝ := {y | ∃ x : ℝ, y = Real.cos x}
def B : Set ℝ := {x | x * (x + 1) ≥ 0}

theorem intersection_A_B :
  (A ∩ B) = {x | (0 ≤ x ∧ x ≤ 1) ∨ x = -1} :=
  sorry

end intersection_A_B_l2285_228504


namespace tom_age_ratio_l2285_228532

-- Definitions of given conditions
variables (T N : ℕ) -- Tom's age (T) and number of years ago (N)

-- Tom's age is T years
-- The sum of the ages of Tom's three children is also T
-- N years ago, Tom's age was twice the sum of his children's ages then

theorem tom_age_ratio (h1 : T - N = 2 * (T - 3 * N)) : T / N = 5 :=
sorry

end tom_age_ratio_l2285_228532


namespace z_is_real_iff_z_is_complex_iff_z_is_pure_imaginary_iff_l2285_228571

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

end z_is_real_iff_z_is_complex_iff_z_is_pure_imaginary_iff_l2285_228571


namespace train_speed_l2285_228512

theorem train_speed (length time : ℝ) (h_length : length = 120) (h_time : time = 11.999040076793857) :
  (length / time) * 3.6 = 36.003 :=
by
  sorry

end train_speed_l2285_228512


namespace total_kids_got_in_equals_148_l2285_228587

def total_kids : ℕ := 120 + 90 + 50

def denied_riverside : ℕ := (20 * 120) / 100
def denied_west_side : ℕ := (70 * 90) / 100
def denied_mountaintop : ℕ := 50 / 2

def got_in_riverside : ℕ := 120 - denied_riverside
def got_in_west_side : ℕ := 90 - denied_west_side
def got_in_mountaintop : ℕ := 50 - denied_mountaintop

def total_got_in : ℕ := got_in_riverside + got_in_west_side + got_in_mountaintop

theorem total_kids_got_in_equals_148 :
  total_got_in = 148 := 
by
  unfold total_got_in
  unfold got_in_riverside got_in_west_side got_in_mountaintop
  unfold denied_riverside denied_west_side denied_mountaintop
  sorry

end total_kids_got_in_equals_148_l2285_228587


namespace flowchart_basic_elements_includes_loop_l2285_228510

theorem flowchart_basic_elements_includes_loop 
  (sequence_structure : Prop)
  (condition_structure : Prop)
  (loop_structure : Prop)
  : ∃ element : ℕ, element = 2 := 
by
  -- Assume 0 is A: Judgment
  -- Assume 1 is B: Directed line
  -- Assume 2 is C: Loop
  -- Assume 3 is D: Start
  sorry

end flowchart_basic_elements_includes_loop_l2285_228510


namespace probability_both_A_and_B_selected_l2285_228583

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l2285_228583


namespace partners_in_firm_l2285_228507

theorem partners_in_firm (P A : ℕ) (h1 : P * 63 = 2 * A) (h2 : P * 34 = 1 * (A + 45)) : P = 18 :=
by
  sorry

end partners_in_firm_l2285_228507


namespace min_value_of_b_plus_3_div_a_l2285_228533

theorem min_value_of_b_plus_3_div_a (a : ℝ) (b : ℝ) :
  0 < a →
  (∀ x, 0 < x → (a * x - 2) * (-x^2 - b * x + 4) ≤ 0) →
  b + 3 / a = 2 * Real.sqrt 2 :=
by
  sorry

end min_value_of_b_plus_3_div_a_l2285_228533


namespace circumcenter_distance_two_l2285_228511

noncomputable def distance_between_circumcenter (A B C M : ℝ × ℝ)
  (hAB : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 25)
  (hBC : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 17)
  (hAC : (A.1 - C.1)^2 + (A.2 - C.2)^2 = 16)
  (hM_on_AC : M.1 = C.1 - 1 ∧ M.2 = C.2)
  (hCM : (M.1 - C.1)^2 + (M.2 - C.2)^2 = 1)
  : ℝ :=
dist ( ( (A.1 + B.1) / 2, (A.2 + B.2) / 2 ) ) ( ( (B.1 + C.1) / 2, (B.2 + C.2) / 2 )) 

theorem circumcenter_distance_two (A B C M : ℝ × ℝ)
  (hAB : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 25)
  (hBC : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 17)
  (hAC : (A.1 - C.1)^2 + (A.2 - C.2)^2 = 16)
  (hM_on_AC : M.1 = C.1 - 1 ∧ M.2 = C.2)
  (hCM : (M.1 - C.1)^2 + (M.2 - C.2)^2 = 1) 
  : distance_between_circumcenter A B C M hAB hBC hAC hM_on_AC hCM = 2 :=
sorry

end circumcenter_distance_two_l2285_228511


namespace inequality_does_not_hold_l2285_228596

noncomputable def f : ℝ → ℝ := sorry -- define f satisfying the conditions from a)

theorem inequality_does_not_hold :
  (∀ x, f (-x) = f x) ∧ -- f is even
  (∀ x, f x = f (x + 2)) ∧ -- f is periodic with period 2
  (∀ x, 3 ≤ x ∧ x ≤ 4 → f x = 2^x) → -- f(x) = 2^x when x is in [3, 4]
  ¬ (f (Real.sin 3) < f (Real.cos 3)) := by
  -- skipped proof
  sorry

end inequality_does_not_hold_l2285_228596


namespace john_pays_in_30_day_month_l2285_228581

-- The cost of one pill
def cost_per_pill : ℝ := 1.5

-- The number of pills John takes per day
def pills_per_day : ℕ := 2

-- The number of days in a month
def days_in_month : ℕ := 30

-- The insurance coverage percentage
def insurance_coverage : ℝ := 0.40

-- Calculate the total cost John has to pay after insurance coverage in a 30-day month
theorem john_pays_in_30_day_month : (2 * 30) * 1.5 * 0.60 = 54 :=
by
  sorry

end john_pays_in_30_day_month_l2285_228581


namespace quadratic_has_two_distinct_real_roots_l2285_228547

theorem quadratic_has_two_distinct_real_roots :
  let a := 1
  let b := 1
  let c := -1
  let discriminant := b^2 - 4 * a * c
  discriminant > 0 :=
by
  let a := 1
  let b := 1
  let c := -1
  let discriminant := b^2 - 4 * a * c
  show discriminant > 0
  sorry

end quadratic_has_two_distinct_real_roots_l2285_228547


namespace symmetric_point_l2285_228566

theorem symmetric_point : ∃ (x0 y0 : ℝ), 
  (x0 = -6 ∧ y0 = -3) ∧ 
  (∃ (m1 m2 : ℝ), 
    m1 = -1 ∧ 
    m2 = (y0 - 2) / (x0 + 1) ∧ 
    m1 * m2 = -1) ∧ 
  (∃ (x_mid y_mid : ℝ), 
    x_mid = (x0 - 1) / 2 ∧ 
    y_mid = (y0 + 2) / 2 ∧ 
    x_mid + y_mid + 4 = 0) := 
sorry

end symmetric_point_l2285_228566


namespace train_length_l2285_228555

theorem train_length (L : ℝ) :
  (20 * (L + 160) = 15 * (L + 250)) -> L = 110 :=
by
  intro h
  sorry

end train_length_l2285_228555


namespace width_of_wall_l2285_228542

def volume_of_brick (length width height : ℝ) : ℝ :=
  length * width * height

def volume_of_wall (length width height : ℝ) : ℝ :=
  length * width * height

theorem width_of_wall
  (l_b w_b h_b : ℝ) (n : ℝ) (L H : ℝ)
  (volume_brick := volume_of_brick l_b w_b h_b)
  (total_volume_bricks := n * volume_brick) :
  volume_of_wall L (total_volume_bricks / (L * H)) H = total_volume_bricks :=
by
  sorry

end width_of_wall_l2285_228542


namespace circular_board_area_l2285_228544

theorem circular_board_area (C : ℝ) (R T : ℝ) (h1 : R = 62.8) (h2 : T = 10) (h3 : C = R / T) (h4 : C = 2 * Real.pi) : 
  ∀ r A : ℝ, (r = C / (2 * Real.pi)) → (A = Real.pi * r^2)  → A = Real.pi :=
by
  intro r A
  intro hr hA
  sorry

end circular_board_area_l2285_228544


namespace first_discount_percentage_l2285_228526

theorem first_discount_percentage (x : ℕ) :
  let original_price := 175
  let discounted_price := original_price * (100 - x) / 100
  let final_price := discounted_price * 95 / 100
  final_price = 133 → x = 20 :=
by
  sorry

end first_discount_percentage_l2285_228526


namespace find_m_value_l2285_228558

theorem find_m_value : 
  ∃ (m : ℝ), 
  (∃ (x y : ℝ), (x - 2)^2 + (y + 1)^2 = 1 ∧ (x - y + m = 0)) → m = -3 :=
by
  sorry

end find_m_value_l2285_228558


namespace bags_total_weight_l2285_228576

noncomputable def total_weight_of_bags (x y z : ℕ) : ℕ := x + y + z

theorem bags_total_weight (x y z : ℕ) (h1 : x + y = 90) (h2 : y + z = 100) (h3 : z + x = 110) :
  total_weight_of_bags x y z = 150 :=
by
  sorry

end bags_total_weight_l2285_228576


namespace diminish_value_l2285_228500

theorem diminish_value (a b : ℕ) (h1 : a = 1015) (h2 : b = 12) (h3 : b = 16) (h4 : b = 18) (h5 : b = 21) (h6 : b = 28) :
  ∃ k, a - k = lcm (lcm (lcm b b) (lcm b b)) (lcm b b) ∧ k = 7 :=
sorry

end diminish_value_l2285_228500


namespace bottles_needed_exceed_initial_l2285_228524

-- Define the initial conditions and their relationships
def initial_bottles : ℕ := 4 * 12 -- four dozen bottles

def bottles_first_break (players : ℕ) (bottles_per_player : ℕ) : ℕ :=
  players * bottles_per_player

def bottles_second_break (total_players : ℕ) (bottles_per_player : ℕ) (exhausted_players : ℕ) (extra_bottles : ℕ) : ℕ :=
  total_players * bottles_per_player + exhausted_players * extra_bottles

def bottles_third_break (remaining_players : ℕ) (bottles_per_player : ℕ) : ℕ :=
  remaining_players * bottles_per_player

-- Prove that the bottles needed exceed the initial amount by 4
theorem bottles_needed_exceed_initial : 
  bottles_first_break 11 2 + bottles_second_break 14 1 4 1 + bottles_third_break 12 1 = initial_bottles + 4 :=
by
  -- Proof will be completed here
  sorry

end bottles_needed_exceed_initial_l2285_228524


namespace smallest_y_value_l2285_228546

theorem smallest_y_value :
  ∃ y : ℝ, (3 * y ^ 2 + 33 * y - 90 = y * (y + 16)) ∧ ∀ z : ℝ, (3 * z ^ 2 + 33 * z - 90 = z * (z + 16)) → y ≤ z :=
sorry

end smallest_y_value_l2285_228546


namespace totalPeoplePresent_is_630_l2285_228577

def totalParents : ℕ := 105
def totalPupils : ℕ := 698

def groupA_fraction : ℚ := 30 / 100
def groupB_fraction : ℚ := 25 / 100
def groupC_fraction : ℚ := 20 / 100
def groupD_fraction : ℚ := 15 / 100
def groupE_fraction : ℚ := 10 / 100

def groupA_attendance : ℚ := 90 / 100
def groupB_attendance : ℚ := 80 / 100
def groupC_attendance : ℚ := 70 / 100
def groupD_attendance : ℚ := 60 / 100
def groupE_attendance : ℚ := 50 / 100

def junior_fraction : ℚ := 30 / 100
def intermediate_fraction : ℚ := 35 / 100
def senior_fraction : ℚ := 20 / 100
def advanced_fraction : ℚ := 15 / 100

def junior_attendance : ℚ := 85 / 100
def intermediate_attendance : ℚ := 80 / 100
def senior_attendance : ℚ := 75 / 100
def advanced_attendance : ℚ := 70 / 100

noncomputable def totalPeoplePresent : ℚ := 
  totalParents * groupA_fraction * groupA_attendance +
  totalParents * groupB_fraction * groupB_attendance +
  totalParents * groupC_fraction * groupC_attendance +
  totalParents * groupD_fraction * groupD_attendance +
  totalParents * groupE_fraction * groupE_attendance +
  totalPupils * junior_fraction * junior_attendance +
  totalPupils * intermediate_fraction * intermediate_attendance +
  totalPupils * senior_fraction * senior_attendance +
  totalPupils * advanced_fraction * advanced_attendance

theorem totalPeoplePresent_is_630 : totalPeoplePresent.floor = 630 := 
by 
  sorry -- no proof required as per the instructions

end totalPeoplePresent_is_630_l2285_228577


namespace no_constant_term_in_expansion_l2285_228531

theorem no_constant_term_in_expansion : 
  ∀ (x : ℂ), ¬ ∃ (k : ℕ), ∃ (c : ℂ), c * x ^ (k / 3 - 2 * (12 - k)) = 0 :=
by sorry

end no_constant_term_in_expansion_l2285_228531


namespace eq_op_l2285_228523

-- Define the operation ⊕
def op (x y : ℝ) : ℝ := x^3 + 2 * x - y

-- State the theorem to be proven
theorem eq_op (k : ℝ) : op k (op k k) = k := sorry

end eq_op_l2285_228523


namespace perimeter_of_triangle_ABC_l2285_228598

noncomputable def triangle_perimeter (r1 r2 r3 : ℝ) (θ1 θ2 θ3 : ℝ) : ℝ :=
  let x1 := r1 * Real.cos θ1
  let y1 := r1 * Real.sin θ1
  let x2 := r2 * Real.cos θ2
  let y2 := r2 * Real.sin θ2
  let x3 := r3 * Real.cos θ3
  let y3 := r3 * Real.sin θ3
  let d12 := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  let d23 := Real.sqrt ((x3 - x2)^2 + (y3 - y2)^2)
  let d31 := Real.sqrt ((x3 - x1)^2 + (y3 - y1)^2)
  d12 + d23 + d31

--prove

theorem perimeter_of_triangle_ABC (θ1 θ2 θ3: ℝ)
  (h1: θ1 - θ2 = Real.pi / 3)
  (h2: θ2 - θ3 = Real.pi / 3) :
  triangle_perimeter 4 5 7 θ1 θ2 θ3 = sorry := 
sorry

end perimeter_of_triangle_ABC_l2285_228598


namespace seating_arrangement_l2285_228529

theorem seating_arrangement : 
  ∃ x y z : ℕ, 
  7 * x + 8 * y + 9 * z = 65 ∧ z = 1 ∧ x + y + z = r :=
sorry

end seating_arrangement_l2285_228529


namespace triangle_find_C_angle_triangle_find_perimeter_l2285_228513

variable (A B C a b c : ℝ)

theorem triangle_find_C_angle
  (h1 : 2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c) :
  C = π / 3 :=
sorry

theorem triangle_find_perimeter
  (h1 : 2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c)
  (h2 : c = Real.sqrt 7)
  (h3 : a * b = 6) :
  a + b + c = 5 + Real.sqrt 7 :=
sorry

end triangle_find_C_angle_triangle_find_perimeter_l2285_228513


namespace greatest_possible_sum_of_digits_l2285_228528

theorem greatest_possible_sum_of_digits 
  (n : ℕ) (a b d : ℕ) 
  (h_a : a ≠ 0) (h_b : b ≠ 0) (h_d : d ≠ 0)
  (h1 : ∃ n1 n2 : ℕ, n1 ≠ n2 ∧ (d * ((10 ^ (3 * n1) - 1) / 9) - b * ((10 ^ n1 - 1) / 9) = a^3 * ((10^n1 - 1) / 9)^3) 
                      ∧ (d * ((10 ^ (3 * n2) - 1) / 9) - b * ((10 ^ n2 - 1) / 9) = a^3 * ((10^n2 - 1) / 9)^3)) : 
  a + b + d = 12 := 
sorry

end greatest_possible_sum_of_digits_l2285_228528


namespace number_of_schools_l2285_228567

theorem number_of_schools (cost_per_school : ℝ) (population : ℝ) (savings_per_day_per_person : ℝ) (days_in_year : ℕ) :
  cost_per_school = 5 * 10^5 →
  population = 1.3 * 10^9 →
  savings_per_day_per_person = 0.01 →
  days_in_year = 365 →
  (population * savings_per_day_per_person * days_in_year) / cost_per_school = 9.49 * 10^3 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_schools_l2285_228567


namespace adoption_days_l2285_228540

def initial_puppies : ℕ := 15
def additional_puppies : ℕ := 62
def adoption_rate : ℕ := 7

def total_puppies : ℕ := initial_puppies + additional_puppies

theorem adoption_days :
  total_puppies / adoption_rate = 11 :=
by
  sorry

end adoption_days_l2285_228540


namespace roots_triple_relation_l2285_228594

theorem roots_triple_relation (p q r α β : ℝ) (h1 : α + β = -q / p) (h2 : α * β = r / p) (h3 : β = 3 * α) :
  3 * q ^ 2 = 16 * p * r :=
sorry

end roots_triple_relation_l2285_228594


namespace julias_preferred_number_l2285_228518

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n % 100) / 10) + (n % 10)

theorem julias_preferred_number : ∃ n : ℕ, n > 100 ∧ n < 200 ∧ n % 13 = 0 ∧ n % 3 ≠ 0 ∧ sum_of_digits n % 5 = 0 ∧ n = 104 :=
by
  sorry

end julias_preferred_number_l2285_228518


namespace select_1996_sets_l2285_228586

theorem select_1996_sets (k : ℕ) (sets : Finset (Finset ℕ)) (h : k > 1993006) (h_sets : sets.card = k) :
  ∃ (selected_sets : Finset (Finset ℕ)), selected_sets.card = 1996 ∧
  ∀ (x y z : Finset ℕ), x ∈ selected_sets → y ∈ selected_sets → z ∈ selected_sets → z = x ∪ y → false :=
sorry

end select_1996_sets_l2285_228586


namespace rectangle_shorter_side_l2285_228536

theorem rectangle_shorter_side
  (x : ℝ)
  (a b d : ℝ)
  (h₁ : a = 3 * x)
  (h₂ : b = 4 * x)
  (h₃ : d = 9) :
  a = 5.4 := 
by
  sorry

end rectangle_shorter_side_l2285_228536


namespace range_of_a_l2285_228530

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * a * x^2 + (a - 1) * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x, 1 < x ∧ x < 4 → deriv (f a) x < 0) ∧
  (∀ x, 6 < x → deriv (f a) x > 0) →
  5 ≤ a ∧ a ≤ 7 :=
sorry

end range_of_a_l2285_228530


namespace diamonds_in_F10_l2285_228560

def diamonds_in_figure (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 1 + 3 * (Nat.add (Nat.mul (n - 1) n) 0) / 2

theorem diamonds_in_F10 : diamonds_in_figure 10 = 136 :=
by
  sorry

end diamonds_in_F10_l2285_228560


namespace cost_price_of_product_is_100_l2285_228545

theorem cost_price_of_product_is_100 
  (x : ℝ) 
  (h : x * 1.2 * 0.9 - x = 8) : 
  x = 100 := 
sorry

end cost_price_of_product_is_100_l2285_228545


namespace center_of_square_l2285_228599

theorem center_of_square (O : ℝ × ℝ) (A B C D : ℝ × ℝ) 
  (hAB : dist A B = 1) 
  (hA : A = (0, 0)) 
  (hB : B = (1, 0)) 
  (hC : C = (1, 1)) 
  (hD : D = (0, 1)) 
  (h_sum_squares : (dist O A)^2 + (dist O B)^2 + (dist O C)^2 + (dist O D)^2 = 2): 
  O = (1/2, 1/2) :=
by sorry

end center_of_square_l2285_228599


namespace hou_yi_score_l2285_228548

theorem hou_yi_score (a b c : ℕ) (h1 : 2 * b + c = 29) (h2 : 2 * a + c = 43) : a + b + c = 36 := 
by 
  sorry

end hou_yi_score_l2285_228548


namespace solve_trig_eq_l2285_228525

-- Define the equation
def equation (x : ℝ) : Prop := 3 * Real.sin x = 1 + Real.cos (2 * x)

-- Define the solution set
def solution_set (x : ℝ) : Prop := ∃ k : ℤ, x = k * Real.pi + (-1)^k * (Real.pi / 6)

-- The proof problem statement
theorem solve_trig_eq {x : ℝ} : equation x ↔ solution_set x := sorry

end solve_trig_eq_l2285_228525


namespace find_percentage_l2285_228508

variable (P x : ℝ)

theorem find_percentage (h1 : x = 10)
    (h2 : (P / 100) * x = 0.05 * 500 - 20) : P = 50 := by
  sorry

end find_percentage_l2285_228508


namespace average_age_of_community_l2285_228582

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

end average_age_of_community_l2285_228582
