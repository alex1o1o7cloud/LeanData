import Mathlib

namespace sin_cos_fraction_l1024_102437

theorem sin_cos_fraction (α : ℝ) (h1 : Real.sin α - Real.cos α = 1 / 5) (h2 : α ∈ Set.Ioo (-Real.pi / 2) (Real.pi / 2)) :
    Real.sin α * Real.cos α / (Real.sin α + Real.cos α) = 12 / 35 :=
by
  sorry

end sin_cos_fraction_l1024_102437


namespace intersection_necessary_but_not_sufficient_l1024_102454

variables {M N P : Set α}

theorem intersection_necessary_but_not_sufficient : 
  (M ∩ P = N ∩ P) → (M ≠ N) :=
sorry

end intersection_necessary_but_not_sufficient_l1024_102454


namespace sum_of_roots_l1024_102455

theorem sum_of_roots (α β : ℝ) (h1 : α^2 - 4 * α + 3 = 0) (h2 : β^2 - 4 * β + 3 = 0) (h3 : α ≠ β) :
  α + β = 4 :=
sorry

end sum_of_roots_l1024_102455


namespace greatest_drop_june_increase_april_l1024_102493

-- January price change
def jan : ℝ := -1.00

-- February price change
def feb : ℝ := 3.50

-- March price change
def mar : ℝ := -3.00

-- April price change
def apr : ℝ := 4.50

-- May price change
def may : ℝ := -1.50

-- June price change
def jun : ℝ := -3.50

def greatest_drop : List (ℝ × String) := [(jan, "January"), (mar, "March"), (may, "May"), (jun, "June")]

def greatest_increase : List (ℝ × String) := [(feb, "February"), (apr, "April")]

theorem greatest_drop_june_increase_april :
  (∀ d ∈ greatest_drop, d.1 ≤ jun) ∧ (∀ i ∈ greatest_increase, i.1 ≤ apr) :=
by
  sorry

end greatest_drop_june_increase_april_l1024_102493


namespace fourth_polygon_is_square_l1024_102491

theorem fourth_polygon_is_square
  (angle_triangle angle_square angle_hexagon : ℕ)
  (h_triangle : angle_triangle = 60)
  (h_square : angle_square = 90)
  (h_hexagon : angle_hexagon = 120)
  (h_total : angle_triangle + angle_square + angle_hexagon = 270) :
  ∃ angle_fourth : ℕ, angle_fourth = 90 ∧ (angle_fourth + angle_triangle + angle_square + angle_hexagon = 360) :=
sorry

end fourth_polygon_is_square_l1024_102491


namespace remainder_of_1999_pow_11_mod_8_l1024_102476

theorem remainder_of_1999_pow_11_mod_8 :
  (1999 ^ 11) % 8 = 7 :=
  sorry

end remainder_of_1999_pow_11_mod_8_l1024_102476


namespace original_price_l1024_102427

theorem original_price (selling_price profit_percent : ℝ) (h_sell : selling_price = 63) (h_profit : profit_percent = 5) : 
  selling_price / (1 + profit_percent / 100) = 60 :=
by sorry

end original_price_l1024_102427


namespace triangle_area_half_l1024_102459

theorem triangle_area_half (AB AC BC : ℝ) (h₁ : AB = 8) (h₂ : AC = BC) (h₃ : AC * AC = AB * AB / 2) (h₄ : AC = BC) : 
  (1 / 2) * (1 / 2 * AB * AB) = 16 :=
  by
  sorry

end triangle_area_half_l1024_102459


namespace opposite_number_l1024_102442

theorem opposite_number (x : ℤ) (h : -x = -2) : x = 2 :=
sorry

end opposite_number_l1024_102442


namespace count_primes_with_squares_in_range_l1024_102421

theorem count_primes_with_squares_in_range : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, Prime n ∧ 5000 < n^2 ∧ n^2 < 9000) ∧ 
    S.card = 5 :=
by
  sorry

end count_primes_with_squares_in_range_l1024_102421


namespace simplify_expression_l1024_102484

theorem simplify_expression : 4 * (18 / 5) * (25 / -72) = -5 := by
  sorry

end simplify_expression_l1024_102484


namespace cheryl_material_usage_l1024_102403

theorem cheryl_material_usage:
  let bought := (3 / 8) + (1 / 3)
  let left := (15 / 40)
  let used := bought - left
  used = (1 / 3) := 
by
  sorry

end cheryl_material_usage_l1024_102403


namespace quadratic_integers_pairs_l1024_102428

theorem quadratic_integers_pairs (m n : ℕ) :
  (0 < m ∧ m < 9) ∧ (0 < n ∧ n < 9) ∧ (m^2 > 9 * n) ↔ ((m = 4 ∧ n = 1) ∨ (m = 5 ∧ n = 2)) :=
by {
  -- Insert proof here
  sorry
}

end quadratic_integers_pairs_l1024_102428


namespace ordering_of_powers_l1024_102497

theorem ordering_of_powers : (3 ^ 17) < (8 ^ 9) ∧ (8 ^ 9) < (4 ^ 15) := 
by 
  -- We proved (3 ^ 17) < (8 ^ 9)
  have h1 : (3 ^ 17) < (8 ^ 9) := sorry
  
  -- We proved (8 ^ 9) < (4 ^ 15)
  have h2 : (8 ^ 9) < (4 ^ 15) := sorry

  -- Therefore, combining both
  exact ⟨h1, h2⟩

end ordering_of_powers_l1024_102497


namespace perpendicular_condition_l1024_102473

theorem perpendicular_condition (a : ℝ) :
  let l1 (x y : ℝ) := x + a * y - 2
  let l2 (x y : ℝ) := x - a * y - 1
  (∀ x y, (l1 x y = 0 ↔ l2 x y ≠ 0) ↔ 1 - a * a = 0) →
  (a = -1) ∨ (a = 1) :=
by
  intro
  sorry

end perpendicular_condition_l1024_102473


namespace find_initial_population_l1024_102498

-- Define the initial population, conditions and the final population
variable (P : ℕ)

noncomputable def initial_population (P : ℕ) :=
  (0.85 * (0.92 * P) : ℝ) = 3553

theorem find_initial_population (P : ℕ) (h : initial_population P) : P = 4546 := sorry

end find_initial_population_l1024_102498


namespace range_of_a_l1024_102485

theorem range_of_a (a : ℝ) : (2 * a - 8) / 3 < 0 → a < 4 :=
by sorry

end range_of_a_l1024_102485


namespace base9_perfect_square_l1024_102474

theorem base9_perfect_square (a b d : ℕ) (h1 : a ≠ 0) (h2 : a < 9) (h3 : b < 9) (h4 : d < 9) (h5 : ∃ n : ℕ, (729 * a + 81 * b + 36 + d) = n * n) : d = 0 ∨ d = 1 ∨ d = 4 :=
by sorry

end base9_perfect_square_l1024_102474


namespace polynomial_solution_l1024_102424

theorem polynomial_solution (P : Polynomial ℝ) (h1 : P.eval 0 = 0) (h2 : ∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1) : 
  ∀ x : ℝ, P.eval x = x :=
by
  sorry

end polynomial_solution_l1024_102424


namespace line_intersects_y_axis_at_point_l1024_102483

theorem line_intersects_y_axis_at_point :
  let x1 := 3
  let y1 := 20
  let x2 := -7
  let y2 := 2

  -- line equation from 2 points: y - y1 = m * (x - x1)
  -- slope m = (y2 - y1) / (x2 - x1)
  -- y-intercept when x = 0:
  
  (0, 14.6) ∈ { p : ℝ × ℝ | ∃ m b, p.2 = m * p.1 + b ∧ 
    m = (y2 - y1) / (x2 - x1) ∧ 
    b = y1 - m * x1 }
  :=
  sorry

end line_intersects_y_axis_at_point_l1024_102483


namespace water_fee_part1_water_fee_part2_water_fee_usage_l1024_102479

theorem water_fee_part1 (x : ℕ) (h : 0 < x ∧ x ≤ 6) : y = 2 * x :=
sorry

theorem water_fee_part2 (x : ℕ) (h : x > 6) : y = 3 * x - 6 :=
sorry

theorem water_fee_usage (y : ℕ) (h : y = 27) : x = 11 :=
sorry

end water_fee_part1_water_fee_part2_water_fee_usage_l1024_102479


namespace equipment_value_decrease_l1024_102415

theorem equipment_value_decrease (a : ℝ) (b : ℝ) (n : ℕ) :
  (a * (1 - b / 100)^n) = a * (1 - b/100)^n :=
sorry

end equipment_value_decrease_l1024_102415


namespace circle_center_and_radius_l1024_102401

def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 4

theorem circle_center_and_radius :
  (∀ x y : ℝ, circle_eq x y ↔ (x - 2) ^ 2 + y ^ 2 = 4) →
  (exists (h k r : ℝ), (h, k) = (2, 0) ∧ r = 2) :=
by
  sorry

end circle_center_and_radius_l1024_102401


namespace marathon_distance_l1024_102447

theorem marathon_distance (marathons : ℕ) (miles_per_marathon : ℕ) (extra_yards_per_marathon : ℕ) (yards_per_mile : ℕ) (total_miles_run : ℕ) (total_yards_run : ℕ) (remaining_yards : ℕ) :
  marathons = 15 →
  miles_per_marathon = 26 →
  extra_yards_per_marathon = 385 →
  yards_per_mile = 1760 →
  total_miles_run = (marathons * miles_per_marathon + extra_yards_per_marathon * marathons / yards_per_mile) →
  total_yards_run = (marathons * (miles_per_marathon * yards_per_mile + extra_yards_per_marathon)) →
  remaining_yards = total_yards_run - (total_miles_run * yards_per_mile) →
  0 ≤ remaining_yards ∧ remaining_yards < yards_per_mile →
  remaining_yards = 1500 :=
by
  intros
  sorry

end marathon_distance_l1024_102447


namespace probability_of_Xiaojia_selection_l1024_102467

theorem probability_of_Xiaojia_selection : 
  let students := 2500
  let teachers := 350
  let support_staff := 150
  let total_individuals := students + teachers + support_staff
  let sampled_individuals := 300
  let student_sample := (students : ℝ)/total_individuals * sampled_individuals
  (student_sample / students) = (1 / 10) := 
by
  sorry

end probability_of_Xiaojia_selection_l1024_102467


namespace johns_pool_depth_l1024_102462

theorem johns_pool_depth : 
  ∀ (j s : ℕ), (j = 2 * s + 5) → (s = 5) → (j = 15) := 
by 
  intros j s h1 h2
  rw [h2] at h1
  exact h1

end johns_pool_depth_l1024_102462


namespace sequences_with_both_properties_are_constant_l1024_102430

-- Definitions according to the problem's conditions
def arithmetic_sequence (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) - seq n = seq (n + 2) - seq (n + 1)

def geometric_sequence (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) / seq n = seq (n + 2) / seq (n + 1)

-- Definition of the sequence properties combined
def arithmetic_and_geometric_sequence (seq : ℕ → ℝ) : Prop :=
  arithmetic_sequence seq ∧ geometric_sequence seq

-- Problem to prove
theorem sequences_with_both_properties_are_constant (seq : ℕ → ℝ) :
  arithmetic_and_geometric_sequence seq → ∀ n m : ℕ, seq n = seq m :=
sorry

end sequences_with_both_properties_are_constant_l1024_102430


namespace base8_subtraction_l1024_102434

def subtract_base_8 (a b : Nat) : Nat :=
  sorry  -- This is a placeholder for the actual implementation.

theorem base8_subtraction :
  subtract_base_8 0o5374 0o2645 = 0o1527 :=
by
  sorry

end base8_subtraction_l1024_102434


namespace number_of_trees_l1024_102465

-- Define the yard length and the distance between consecutive trees
def yard_length : ℕ := 300
def distance_between_trees : ℕ := 12

-- Prove that the number of trees planted in the garden is 26
theorem number_of_trees (yard_length distance_between_trees : ℕ) 
  (h1 : yard_length = 300) (h2 : distance_between_trees = 12) : 
  ∃ n : ℕ, n = 26 :=
by
  sorry

end number_of_trees_l1024_102465


namespace triangle_side_a_l1024_102435

theorem triangle_side_a {a b c : ℝ} (A : ℝ) (hA : A = (2 * Real.pi / 3)) (hb : b = Real.sqrt 2) 
(h_area : 1 / 2 * b * c * Real.sin A = Real.sqrt 3) :
  a = Real.sqrt 14 :=
by 
  sorry

end triangle_side_a_l1024_102435


namespace pool_capacity_l1024_102448

theorem pool_capacity (C : ℝ) (h1 : 300 = 0.30 * C) : C = 1000 :=
by
  sorry

end pool_capacity_l1024_102448


namespace max_min_values_l1024_102443

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 2

theorem max_min_values :
  let max_val := 2
  let min_val := -25
  ∃ x_max x_min, 
    0 ≤ x_max ∧ x_max ≤ 4 ∧ f x_max = max_val ∧ 
    0 ≤ x_min ∧ x_min ≤ 4 ∧ f x_min = min_val :=
sorry

end max_min_values_l1024_102443


namespace option_b_represents_factoring_l1024_102488

theorem option_b_represents_factoring (x y : ℤ) :
  x^2 - 2*x*y = x * (x - 2*y) :=
sorry

end option_b_represents_factoring_l1024_102488


namespace combined_balance_l1024_102444

theorem combined_balance (b : ℤ) (g1 g2 : ℤ) (h1 : b = 3456) (h2 : g1 = b / 4) (h3 : g2 = b / 4) : g1 + g2 = 1728 :=
by {
  sorry
}

end combined_balance_l1024_102444


namespace speed_conversion_l1024_102438

def speed_mps : ℝ := 10.0008
def conversion_factor : ℝ := 3.6

theorem speed_conversion : speed_mps * conversion_factor = 36.003 :=
by
  sorry

end speed_conversion_l1024_102438


namespace backyard_area_l1024_102429

-- Definitions from conditions
def length : ℕ := 1000 / 25
def perimeter : ℕ := 1000 / 10
def width : ℕ := (perimeter - 2 * length) / 2

-- Theorem statement: Given the conditions, the area of the backyard is 400 square meters
theorem backyard_area : length * width = 400 :=
by 
  -- Sorry to skip the proof as instructed
  sorry

end backyard_area_l1024_102429


namespace f_2012_l1024_102431

noncomputable def f : ℝ → ℝ := sorry -- provided as a 'sorry' to be determined

axiom odd_function (hf : ℝ → ℝ) : ∀ x : ℝ, hf (-x) = -hf (x)

axiom f_shift : ∀ x : ℝ, f (x + 3) = -f (x)
axiom f_one : f 1 = 2

theorem f_2012 : f 2012 = 2 :=
by
  -- proofs would go here, but 'sorry' is enough to define the theorem statement
  sorry

end f_2012_l1024_102431


namespace henri_drove_more_miles_l1024_102422

-- Defining the conditions
def Gervais_average_miles_per_day := 315
def Gervais_days_driven := 3
def Henri_total_miles := 1250

-- Total miles driven by Gervais
def Gervais_total_miles := Gervais_average_miles_per_day * Gervais_days_driven

-- The proof problem statement
theorem henri_drove_more_miles : Henri_total_miles - Gervais_total_miles = 305 := 
by 
  sorry

end henri_drove_more_miles_l1024_102422


namespace tim_total_spent_l1024_102439

-- Define the given conditions
def lunch_cost : ℝ := 50.20
def tip_percentage : ℝ := 0.20

-- Define the total amount spent
def total_amount_spent : ℝ := 60.24

-- Prove the total amount spent given the conditions
theorem tim_total_spent : lunch_cost + (tip_percentage * lunch_cost) = total_amount_spent := by
  -- This is the proof statement corresponding to the problem; the proof itself is not required for this task
  sorry

end tim_total_spent_l1024_102439


namespace largest_integer_satisfying_l1024_102412

theorem largest_integer_satisfying (x : ℤ) : 
  (∃ x, (2/7 : ℝ) < (x / 6 : ℝ) ∧ (x / 6 : ℝ) < 3/4) → x = 4 := 
by 
  sorry

end largest_integer_satisfying_l1024_102412


namespace union_of_A_and_B_l1024_102481

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} := 
by 
  sorry

end union_of_A_and_B_l1024_102481


namespace area_of_rotated_squares_l1024_102402

noncomputable def side_length : ℝ := 8
noncomputable def rotation_middle : ℝ := 45
noncomputable def rotation_top : ℝ := 75

-- Theorem: The area of the resulting 24-sided polygon.
theorem area_of_rotated_squares :
  (∃ (polygon_area : ℝ), polygon_area = 96) :=
sorry

end area_of_rotated_squares_l1024_102402


namespace number_of_male_students_drawn_l1024_102404

theorem number_of_male_students_drawn (total_students : ℕ) (total_male_students : ℕ) (total_female_students : ℕ) (sample_size : ℕ)
    (H1 : total_students = 350)
    (H2 : total_male_students = 70)
    (H3 : total_female_students = 280)
    (H4 : sample_size = 50) :
    total_male_students * sample_size / total_students = 10 :=
by
  sorry

end number_of_male_students_drawn_l1024_102404


namespace neither_jia_nor_yi_has_winning_strategy_l1024_102451

/-- 
  There are 99 points, each marked with a number from 1 to 99, placed 
  on 99 equally spaced points on a circle. Jia and Yi take turns 
  placing one piece at a time, with Jia going first. The player who 
  first makes the numbers on three consecutive points form an 
  arithmetic sequence wins. Prove that neither Jia nor Yi has a 
  guaranteed winning strategy, and both possess strategies to avoid 
  losing.
-/
theorem neither_jia_nor_yi_has_winning_strategy :
  ∀ (points : Fin 99 → ℕ), -- 99 points on the circle
  (∀ i, 1 ≤ points i ∧ points i ≤ 99) → -- Each point is numbered between 1 and 99
  ¬(∃ (player : Fin 99 → ℕ) (h : ∀ (i : Fin 99), player i ≠ 0 ∧ (player i = 1 ∨ player i = 2)),
    ∃ i : Fin 99, (points i + points (i + 1) + points (i + 2)) / 3 = points i)
:=
by
  sorry

end neither_jia_nor_yi_has_winning_strategy_l1024_102451


namespace hyperbola_slopes_l1024_102456

variables {x1 y1 x2 y2 x y k1 k2 : ℝ}

theorem hyperbola_slopes (h1 : y1^2 - (x1^2 / 2) = 1)
  (h2 : y2^2 - (x2^2 / 2) = 1)
  (hx : x1 + x2 = 2 * x)
  (hy : y1 + y2 = 2 * y)
  (hk1 : k1 = (y2 - y1) / (x2 - x1))
  (hk2 : k2 = y / x) :
  k1 * k2 = 1 / 2 :=
sorry

end hyperbola_slopes_l1024_102456


namespace min_value_l1024_102413

theorem min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 1/b = 1) : 1/(a-1) + 4/(b-1) ≥ 4 :=
by
  sorry

end min_value_l1024_102413


namespace tan_15_degree_l1024_102410

theorem tan_15_degree : 
  let a := 45 * (Real.pi / 180)
  let b := 30 * (Real.pi / 180)
  Real.tan (a - b) = 2 - Real.sqrt 3 :=
by
  sorry

end tan_15_degree_l1024_102410


namespace Monica_class_ratio_l1024_102487

theorem Monica_class_ratio : 
  (20 + 25 + 25 + x + 28 + 28 = 136) → 
  (x = 10) → 
  (x / 20 = 1 / 2) :=
by 
  intros h h_x
  sorry

end Monica_class_ratio_l1024_102487


namespace next_month_has_5_Wednesdays_l1024_102471

-- The current month characteristics
def current_month_has_5_Saturdays : Prop := ∃ month : ℕ, month = 30 ∧ ∃ day : ℕ, day = 5
def current_month_has_5_Sundays : Prop := ∃ month : ℕ, month = 30 ∧ ∃ day : ℕ, day = 5
def current_month_has_4_Mondays : Prop := ∃ month : ℕ, month = 30 ∧ ∃ day : ℕ, day = 4
def current_month_has_4_Fridays : Prop := ∃ month : ℕ, month = 30 ∧ ∃ day : ℕ, day = 4
def month_ends_on_Sunday : Prop := ∃ day : ℕ, day = 30 ∧ day % 7 = 0

-- Prove next month has 5 Wednesdays
theorem next_month_has_5_Wednesdays 
  (h1 : current_month_has_5_Saturdays) 
  (h2 : current_month_has_5_Sundays)
  (h3 : current_month_has_4_Mondays)
  (h4 : current_month_has_4_Fridays)
  (h5 : month_ends_on_Sunday) :
  ∃ month : ℕ, month = 31 ∧ ∃ day : ℕ, day = 5 := 
sorry

end next_month_has_5_Wednesdays_l1024_102471


namespace percentage_of_students_in_grade_8_combined_l1024_102406

theorem percentage_of_students_in_grade_8_combined (parkwood_students maplewood_students : ℕ)
  (parkwood_percentages maplewood_percentages : ℕ → ℕ) 
  (H_parkwood : parkwood_students = 150)
  (H_maplewood : maplewood_students = 120)
  (H_parkwood_percent : parkwood_percentages 8 = 18)
  (H_maplewood_percent : maplewood_percentages 8 = 25):
  (57 / 270) * 100 = 21.11 := 
by
  sorry  -- Proof omitted

end percentage_of_students_in_grade_8_combined_l1024_102406


namespace base6_addition_correct_l1024_102490

theorem base6_addition_correct (S H E : ℕ) (h1 : S < 6) (h2 : H < 6) (h3 : E < 6) 
  (distinct : S ≠ H ∧ H ≠ E ∧ S ≠ E) 
  (h4: S + H * 6 + E * 6^2 +  H * 6 = H + E * 6 + H * 6^2 + E * 6^1) :
  S + H + E = 12 :=
by sorry

end base6_addition_correct_l1024_102490


namespace days_passed_before_cows_ran_away_l1024_102400

def initial_cows := 1000
def initial_days := 50
def cows_left := 800
def cows_run_away := initial_cows - cows_left
def total_food := initial_cows * initial_days
def remaining_food (x : ℕ) := total_food - initial_cows * x
def food_needed := cows_left * initial_days

theorem days_passed_before_cows_ran_away (x : ℕ) :
  (remaining_food x = food_needed) → (x = 10) :=
by
  sorry

end days_passed_before_cows_ran_away_l1024_102400


namespace solve_otimes_eq_l1024_102489

def otimes (a b : ℝ) : ℝ := (a - 2) * (b + 1)

theorem solve_otimes_eq : ∃ x : ℝ, otimes (-4) (x + 3) = 6 ↔ x = -5 :=
by
  use -5
  simp [otimes]
  sorry

end solve_otimes_eq_l1024_102489


namespace area_of_second_side_l1024_102452

theorem area_of_second_side 
  (L W H : ℝ) 
  (h1 : L * H = 120) 
  (h2 : L * W = 60) 
  (h3 : L * W * H = 720) : 
  W * H = 72 :=
sorry

end area_of_second_side_l1024_102452


namespace amount_spent_on_belt_correct_l1024_102492

variable (budget shirt pants coat socks shoes remaining : ℕ)

-- Given conditions
def initial_budget : ℕ := 200
def spent_shirt : ℕ := 30
def spent_pants : ℕ := 46
def spent_coat : ℕ := 38
def spent_socks : ℕ := 11
def spent_shoes : ℕ := 41
def remaining_amount : ℕ := 16

-- The amount spent on the belt
def amount_spent_on_belt : ℕ :=
  budget - remaining - (shirt + pants + coat + socks + shoes)

-- The theorem statement we need to prove
theorem amount_spent_on_belt_correct :
  initial_budget = budget →
  spent_shirt = shirt →
  spent_pants = pants →
  spent_coat = coat →
  spent_socks = socks →
  spent_shoes = shoes →
  remaining_amount = remaining →
  amount_spent_on_belt budget shirt pants coat socks shoes remaining = 18 := by
    simp [initial_budget, spent_shirt, spent_pants, spent_coat, spent_socks, spent_shoes, remaining_amount, amount_spent_on_belt]
    sorry

end amount_spent_on_belt_correct_l1024_102492


namespace correct_exponential_rule_l1024_102494

theorem correct_exponential_rule (a : ℝ) : (a^3)^2 = a^6 :=
by sorry

end correct_exponential_rule_l1024_102494


namespace sufficient_food_supply_l1024_102466

variable {L S : ℝ}

theorem sufficient_food_supply (h1 : L + 4 * S = 14) (h2 : L > S) : L + 3 * S ≥ 11 :=
by
  sorry

end sufficient_food_supply_l1024_102466


namespace find_a_b_l1024_102468

theorem find_a_b :
  ∃ (a b : ℚ), 
    (∀ x : ℚ, x = 2 → (a * x^3 - 6 * x^2 + b * x - 5 - 3 = 0)) ∧
    (∀ x : ℚ, x = -1 → (a * x^3 - 6 * x^2 + b * x - 5 - 7 = 0)) ∧
    (a = -2/3 ∧ b = -52/3) :=
by {
  sorry
}

end find_a_b_l1024_102468


namespace collectively_behind_l1024_102495

noncomputable def sleep_hours_behind (weeknights weekend nights_ideal: ℕ) : ℕ :=
  let total_sleep := (weeknights * 5) + (weekend * 2)
  let ideal_sleep := nights_ideal * 7
  ideal_sleep - total_sleep

def tom_weeknight := 5
def tom_weekend := 6

def jane_weeknight := 7
def jane_weekend := 9

def mark_weeknight := 6
def mark_weekend := 7

def ideal_night := 8

theorem collectively_behind :
  sleep_hours_behind tom_weeknight tom_weekend ideal_night +
  sleep_hours_behind jane_weeknight jane_weekend ideal_night +
  sleep_hours_behind mark_weeknight mark_weekend ideal_night = 34 :=
by
  sorry

end collectively_behind_l1024_102495


namespace limit_perimeters_eq_l1024_102457

universe u

noncomputable def limit_perimeters (s : ℝ) : ℝ :=
  let a := 4 * s
  let r := 1 / 2
  a / (1 - r)

theorem limit_perimeters_eq (s : ℝ) : limit_perimeters s = 8 * s := by
  sorry

end limit_perimeters_eq_l1024_102457


namespace geometric_arithmetic_sequence_relation_l1024_102477

theorem geometric_arithmetic_sequence_relation 
    (a : ℕ → ℝ) (b : ℕ → ℝ) (q d a1 : ℝ)
    (h1 : a 1 = a1) (h2 : b 1 = a1) (h3 : a 3 = a1 * q^2)
    (h4 : b 3 = a1 + 2 * d) (h5 : a 3 = b 3) (h6 : a1 > 0) (h7 : q^2 ≠ 1) :
    a 5 > b 5 :=
by
  -- Proof goes here
  sorry

end geometric_arithmetic_sequence_relation_l1024_102477


namespace find_m_n_l1024_102470

theorem find_m_n (m n x1 x2 : ℕ) (hm : 0 < m) (hn : 0 < n) (hx1 : 0 < x1) (hx2 : 0 < x2) 
  (h_eq : x1 * x2 = m + n) (h_sum : x1 + x2 = m * n) :
  (m = 2 ∧ n = 3) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 2) ∨ (m = 1 ∧ n = 5) ∨ (m = 5 ∧ n = 1) := 
sorry

end find_m_n_l1024_102470


namespace possible_values_of_k_l1024_102407

theorem possible_values_of_k (n : ℕ) (h : n ≥ 3) :
  ∃ t : ℕ, k = 2 ^ t ∧ 2 ^ t ≥ n :=
sorry

end possible_values_of_k_l1024_102407


namespace simplify_expression_l1024_102458

theorem simplify_expression (y : ℝ) : 3 * y + 5 * y + 6 * y + 10 = 14 * y + 10 :=
by
  sorry

end simplify_expression_l1024_102458


namespace calculate_m_squared_l1024_102419

-- Define the conditions
def pizza_diameter := 16
def pizza_radius := pizza_diameter / 2
def num_slices := 4

-- Define the question
def longest_segment_length_in_piece := 2 * pizza_radius
def m := longest_segment_length_in_piece -- Length of the longest line segment in one piece

-- Rewrite the math proof problem
theorem calculate_m_squared :
  m^2 = 256 := 
by 
  -- Proof goes here
  sorry

end calculate_m_squared_l1024_102419


namespace total_delegates_l1024_102426

theorem total_delegates 
  (D: ℕ) 
  (h1: 16 ≤ D)
  (h2: (D - 16) % 2 = 0)
  (h3: 10 ≤ D - 16) : D = 36 := 
sorry

end total_delegates_l1024_102426


namespace inequality_solution_l1024_102482

theorem inequality_solution :
  ∀ x : ℝ, (x - 3) / (x^2 + 4 * x + 10) ≥ 0 ↔ x ≥ 3 :=
by
  sorry

end inequality_solution_l1024_102482


namespace geom_seq_sum_first_eight_l1024_102478

def geom_seq (a₀ r : ℚ) (n : ℕ) : ℚ := a₀ * r^n

def sum_geom_seq (a₀ r : ℚ) (n : ℕ) : ℚ :=
  if r = 1 then a₀ * n else a₀ * (1 - r^n) / (1 - r)

theorem geom_seq_sum_first_eight :
  let a₀ := 1 / 3
  let r := 1 / 3
  let n := 8
  sum_geom_seq a₀ r n = 3280 / 6561 :=
by
  sorry

end geom_seq_sum_first_eight_l1024_102478


namespace danielle_travel_time_is_30_l1024_102433

noncomputable def chase_speed : ℝ := sorry
noncomputable def chase_time : ℝ := 180 -- in minutes
noncomputable def cameron_speed : ℝ := 2 * chase_speed
noncomputable def danielle_speed : ℝ := 3 * cameron_speed
noncomputable def distance : ℝ := chase_speed * chase_time
noncomputable def danielle_time : ℝ := distance / danielle_speed

theorem danielle_travel_time_is_30 :
  danielle_time = 30 :=
sorry

end danielle_travel_time_is_30_l1024_102433


namespace find_f6_l1024_102436

noncomputable def f : ℝ → ℝ :=
sorry

theorem find_f6 (h1 : ∀ x y : ℝ, f (x + y) = f x + f y)
                (h2 : f 5 = 6) :
  f 6 = 36 / 5 :=
sorry

end find_f6_l1024_102436


namespace cost_price_of_apple_l1024_102408

variable (CP SP: ℝ)
variable (loss: ℝ)
variable (h1: SP = 18)
variable (h2: loss = CP / 6)
variable (h3: SP = CP - loss)

theorem cost_price_of_apple : CP = 21.6 :=
by
  sorry

end cost_price_of_apple_l1024_102408


namespace man_speed_is_correct_l1024_102420

noncomputable def speed_of_man (train_length : ℝ) (train_speed : ℝ) (cross_time : ℝ) : ℝ :=
  let train_speed_m_s := train_speed * (1000 / 3600)
  let relative_speed := train_length / cross_time
  let man_speed_m_s := relative_speed - train_speed_m_s
  man_speed_m_s * (3600 / 1000)

theorem man_speed_is_correct :
  speed_of_man 210 25 28 = 2 := by
  sorry

end man_speed_is_correct_l1024_102420


namespace roots_of_equation_l1024_102453

theorem roots_of_equation (x : ℝ) : 3 * x * (x - 1) = 2 * (x - 1) → (x = 1 ∨ x = 2 / 3) :=
by 
  intros h
  sorry

end roots_of_equation_l1024_102453


namespace emily_orange_count_l1024_102425

theorem emily_orange_count
  (betty_oranges : ℕ)
  (h1 : betty_oranges = 12)
  (sandra_oranges : ℕ)
  (h2 : sandra_oranges = 3 * betty_oranges)
  (emily_oranges : ℕ)
  (h3 : emily_oranges = 7 * sandra_oranges) :
  emily_oranges = 252 :=
by
  sorry

end emily_orange_count_l1024_102425


namespace frank_money_l1024_102446

-- Define the initial amount, expenses, and incomes as per the conditions
def initialAmount : ℕ := 11
def spentOnGame : ℕ := 3
def spentOnKeychain : ℕ := 2
def receivedFromAlice : ℕ := 4
def allowance : ℕ := 14
def spentOnBusTicket : ℕ := 5

-- Define the total money left for Frank
def finalAmount (initial : ℕ) (game : ℕ) (keychain : ℕ) (gift : ℕ) (allowance : ℕ) (bus : ℕ) : ℕ :=
  initial - game - keychain + gift + allowance - bus

-- Define the theorem stating that the final amount is 19
theorem frank_money : finalAmount initialAmount spentOnGame spentOnKeychain receivedFromAlice allowance spentOnBusTicket = 19 :=
by
  sorry

end frank_money_l1024_102446


namespace forty_percent_jacqueline_candy_l1024_102461

def fred_candy : ℕ := 12
def uncle_bob_candy : ℕ := fred_candy + 6
def total_fred_uncle_bob_candy : ℕ := fred_candy + uncle_bob_candy
def jacqueline_candy : ℕ := 10 * total_fred_uncle_bob_candy

theorem forty_percent_jacqueline_candy : (40 * jacqueline_candy) / 100 = 120 := by
  sorry

end forty_percent_jacqueline_candy_l1024_102461


namespace arithmetic_expression_l1024_102450

theorem arithmetic_expression : 4 * 6 * 8 + 18 / 3 - 2 ^ 3 = 190 :=
by
  -- Proof goes here
  sorry

end arithmetic_expression_l1024_102450


namespace algebraic_expression_domain_l1024_102417

theorem algebraic_expression_domain (x : ℝ) : 
  (x + 2 ≥ 0) ∧ (x - 3 ≠ 0) ↔ (x ≥ -2) ∧ (x ≠ 3) := by
  sorry

end algebraic_expression_domain_l1024_102417


namespace add_numerator_denominator_add_numerator_denominator_gt_one_l1024_102449

variable {a b n : ℕ}

/-- Adding the same natural number to both the numerator and the denominator of a fraction 
    increases the fraction if it is less than one, and decreases the fraction if it is greater than one. -/
theorem add_numerator_denominator (h1: a < b) : (a + n) / (b + n) > a / b := sorry

theorem add_numerator_denominator_gt_one (h2: a > b) : (a + n) / (b + n) < a / b := sorry

end add_numerator_denominator_add_numerator_denominator_gt_one_l1024_102449


namespace relationship_among_a_b_c_l1024_102423

noncomputable def a := Real.sqrt 0.5
noncomputable def b := Real.sqrt 0.3
noncomputable def c := Real.log 0.2 / Real.log 0.3

theorem relationship_among_a_b_c : b < a ∧ a < c := by
  sorry

end relationship_among_a_b_c_l1024_102423


namespace polynomial_has_root_of_multiplicity_2_l1024_102418

theorem polynomial_has_root_of_multiplicity_2 (r s k : ℝ)
  (h1 : x^3 + k * x - 128 = (x - r)^2 * (x - s)) -- polynomial has a root of multiplicity 2
  (h2 : -2 * r - s = 0)                         -- relationship from coefficient of x²
  (h3 : r^2 + 2 * r * s = k)                    -- relationship from coefficient of x
  (h4 : r^2 * s = 128)                          -- relationship from constant term
  : k = -48 := 
sorry

end polynomial_has_root_of_multiplicity_2_l1024_102418


namespace find_center_of_circle_l1024_102486

theorem find_center_of_circle :
  ∃ (a b : ℝ), a = 0 ∧ b = 3/2 ∧
  ( ∀ (x y : ℝ), ( (x = 1 ∧ y = 2) ∨ (x = 1 ∧ y = 1) ∨ (∃ t : ℝ, y = 2 * t + 3) ) → 
  (x - a)^2 + (y - b)^2 = (1 - a)^2 + (1 - b)^2 ) :=
sorry

end find_center_of_circle_l1024_102486


namespace complement_union_eq_l1024_102409

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x ≥ 2}

theorem complement_union_eq :
  U \ (A ∪ B) = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by
  sorry

end complement_union_eq_l1024_102409


namespace tyler_bird_pairs_l1024_102405

theorem tyler_bird_pairs (n_species : ℕ) (pairs_per_species : ℕ) (total_pairs : ℕ)
  (h1 : n_species = 29)
  (h2 : pairs_per_species = 7)
  (h3 : total_pairs = n_species * pairs_per_species) : total_pairs = 203 :=
by
  sorry

end tyler_bird_pairs_l1024_102405


namespace committeeFormation_l1024_102416

-- Establish the given problem conditions in Lean

open Classical

-- Noncomputable because we are working with combinations and products
noncomputable def numberOfWaysToFormCommittee (numSchools : ℕ) (membersPerSchool : ℕ) (hostSchools : ℕ) (hostReps : ℕ) (nonHostReps : ℕ) : ℕ :=
  let totalSchools := numSchools
  let chooseHostSchools := Nat.choose totalSchools hostSchools
  let chooseHostRepsPerSchool := Nat.choose membersPerSchool hostReps
  let allHostRepsChosen := chooseHostRepsPerSchool ^ hostSchools
  let chooseNonHostRepsPerSchool := Nat.choose membersPerSchool nonHostReps
  let allNonHostRepsChosen := chooseNonHostRepsPerSchool ^ (totalSchools - hostSchools)
  chooseHostSchools * allHostRepsChosen * allNonHostRepsChosen

-- We now state our theorem
theorem committeeFormation : numberOfWaysToFormCommittee 4 6 2 3 1 = 86400 :=
by
  -- This is the lemma we need to prove
  sorry

end committeeFormation_l1024_102416


namespace min_value_sum_reciprocal_squares_l1024_102414

open Real

theorem min_value_sum_reciprocal_squares 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :  
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ 27 := 
sorry

end min_value_sum_reciprocal_squares_l1024_102414


namespace butterflies_left_l1024_102463

theorem butterflies_left (initial_butterflies : ℕ) (one_third_left : ℕ)
  (h1 : initial_butterflies = 9) (h2 : one_third_left = initial_butterflies / 3) :
  initial_butterflies - one_third_left = 6 :=
by
  sorry

end butterflies_left_l1024_102463


namespace solve_base7_addition_problem_l1024_102411

noncomputable def base7_addition_problem : Prop :=
  ∃ (X Y: ℕ), 
    (5 * 7^2 + X * 7 + Y) + (3 * 7^1 + 2) = 6 * 7^2 + 2 * 7 + X ∧
    X + Y = 10 

theorem solve_base7_addition_problem : base7_addition_problem :=
by sorry

end solve_base7_addition_problem_l1024_102411


namespace rhombus_height_l1024_102472

theorem rhombus_height (a d1 d2 : ℝ) (h : ℝ)
  (h_a_positive : 0 < a)
  (h_d1_positive : 0 < d1)
  (h_d2_positive : 0 < d2)
  (h_side_geometric_mean : a^2 = d1 * d2) :
  h = a / 2 :=
sorry

end rhombus_height_l1024_102472


namespace people_and_cars_equation_l1024_102441

theorem people_and_cars_equation (x : ℕ) :
  3 * (x - 2) = 2 * x + 9 :=
sorry

end people_and_cars_equation_l1024_102441


namespace garden_length_l1024_102432

open Nat

def perimeter : ℕ → ℕ → ℕ := λ l w => 2 * (l + w)

theorem garden_length (width : ℕ) (perimeter_val : ℕ) (length : ℕ) 
  (h1 : width = 15) 
  (h2 : perimeter_val = 80) 
  (h3 : perimeter length width = perimeter_val) :
  length = 25 := by
  sorry

end garden_length_l1024_102432


namespace necessary_not_sufficient_condition_l1024_102475

theorem necessary_not_sufficient_condition (x : ℝ) : 
  x^2 - 2 * x - 3 < 0 → -2 < x ∧ x < 3 :=
by  
  sorry

end necessary_not_sufficient_condition_l1024_102475


namespace num_valid_seating_arrangements_l1024_102445

-- Define the dimensions of the examination room
def rows : Nat := 5
def columns : Nat := 6
def total_seats : Nat := rows * columns

-- Define the condition for students not sitting next to each other
def valid_seating_arrangements (rows columns : Nat) : Nat := sorry

-- The theorem to prove the number of seating arrangements
theorem num_valid_seating_arrangements : valid_seating_arrangements rows columns = 772 := 
by 
  sorry

end num_valid_seating_arrangements_l1024_102445


namespace divisible_values_l1024_102496

def is_digit (n : ℕ) : Prop :=
  n >= 0 ∧ n <= 9

def N (x y : ℕ) : ℕ :=
  30 * 10^7 + x * 10^6 + 7 * 10^4 + y * 10^3 + 3

def is_divisible_by_37 (n : ℕ) : Prop :=
  n % 37 = 0

theorem divisible_values :
  ∃ (x y : ℕ), is_digit x ∧ is_digit y ∧ is_divisible_by_37 (N x y) ∧ ((x, y) = (8, 1) ∨ (x, y) = (4, 4) ∨ (x, y) = (0, 7)) :=
by {
  sorry
}

end divisible_values_l1024_102496


namespace xy_fraction_equivalence_l1024_102464

theorem xy_fraction_equivalence
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (x^2 + 4 * x * y) / (y^2 - 4 * x * y) = 3) :
  (x^2 - 4 * x * y) / (y^2 + 4 * x * y) = -1 :=
sorry

end xy_fraction_equivalence_l1024_102464


namespace real_part_of_complex_pow_l1024_102460

open Complex

theorem real_part_of_complex_pow (a b : ℝ) : a = 1 → b = -2 → (realPart ((a : ℂ) + (b : ℂ) * Complex.I)^5) = 41 :=
by
  sorry

end real_part_of_complex_pow_l1024_102460


namespace solve_system_l1024_102440

theorem solve_system (x y : ℝ) (h1 : 5 * x + y = 19) (h2 : x + 3 * y = 1) : 3 * x + 2 * y = 10 :=
by
  sorry

end solve_system_l1024_102440


namespace problem_1_problem_2_l1024_102469

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |x + 1|

theorem problem_1 : {x : ℝ | f x < 4} = {x : ℝ | -4 / 3 < x ∧ x < 4 / 3} :=
by 
  sorry

theorem problem_2 (x₀ : ℝ) (h : ∀ t : ℝ, f x₀ < |m + t| + |t - m|) : 
  {m : ℝ | ∃ x t, f x < |m + t| + |t - m|} = {m : ℝ | m < -3 / 4 ∨ m > 3 / 4} :=
by 
  sorry

end problem_1_problem_2_l1024_102469


namespace loss_of_450_is_negative_450_l1024_102499

-- Define the concept of profit and loss based on given conditions.
def profit (x : Int) := x
def loss (x : Int) := -x

-- The mathematical statement:
theorem loss_of_450_is_negative_450 :
  (profit 1000 = 1000) → (loss 450 = -450) :=
by
  intro h
  sorry

end loss_of_450_is_negative_450_l1024_102499


namespace calculate_L_l1024_102480

theorem calculate_L (T H K : ℝ) (hT : T = 2 * Real.sqrt 5) (hH : H = 10) (hK : K = 2) :
  L = 100 :=
by
  let L := 50 * T^4 / (H^2 * K)
  have : T = 2 * Real.sqrt 5 := hT
  have : H = 10 := hH
  have : K = 2 := hK
  sorry

end calculate_L_l1024_102480
