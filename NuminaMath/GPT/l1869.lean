import Mathlib

namespace NUMINAMATH_GPT_literature_club_students_neither_english_nor_french_l1869_186926

theorem literature_club_students_neither_english_nor_french
  (total_students english_students french_students both_students : ℕ)
  (h1 : total_students = 120)
  (h2 : english_students = 72)
  (h3 : french_students = 52)
  (h4 : both_students = 12) :
  (total_students - ((english_students - both_students) + (french_students - both_students) + both_students) = 8) :=
by
  sorry

end NUMINAMATH_GPT_literature_club_students_neither_english_nor_french_l1869_186926


namespace NUMINAMATH_GPT_complement_union_correct_l1869_186965

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4})
variable (hA : A = {2, 4})
variable (hB : B = {3, 4})

theorem complement_union_correct : ((U \ A) ∪ B) = {1, 3, 4} :=
by
  rw [hU, hA, hB]
  sorry

end NUMINAMATH_GPT_complement_union_correct_l1869_186965


namespace NUMINAMATH_GPT_cash_sales_is_48_l1869_186930

variable (total_sales : ℝ) (credit_fraction : ℝ) (cash_sales : ℝ)

-- Conditions: Total sales were $80, 2/5 of the total sales were credit sales
def problem_conditions := total_sales = 80 ∧ credit_fraction = 2/5 ∧ cash_sales = (1 - credit_fraction) * total_sales

-- Question: Prove that the amount of cash sales Mr. Brandon made is $48.
theorem cash_sales_is_48 (h : problem_conditions total_sales credit_fraction cash_sales) : 
  cash_sales = 48 :=
by
  sorry

end NUMINAMATH_GPT_cash_sales_is_48_l1869_186930


namespace NUMINAMATH_GPT_cryptarithm_problem_l1869_186941

theorem cryptarithm_problem (F E D : ℤ) (h1 : F - E = D - 1) (h2 : D + E + F = 16) (h3 : F - E = D) : 
    F - E = 5 :=
by sorry

end NUMINAMATH_GPT_cryptarithm_problem_l1869_186941


namespace NUMINAMATH_GPT_cistern_length_l1869_186924

theorem cistern_length (w d A : ℝ) (h : d = 1.25 ∧ w = 4 ∧ A = 68.5) :
  ∃ L : ℝ, (L * w) + (2 * L * d) + (2 * w * d) = A ∧ L = 9 :=
by
  obtain ⟨h_d, h_w, h_A⟩ := h
  use 9
  simp [h_d, h_w, h_A]
  norm_num
  sorry

end NUMINAMATH_GPT_cistern_length_l1869_186924


namespace NUMINAMATH_GPT_sum_of_reciprocal_AP_l1869_186913

theorem sum_of_reciprocal_AP (a1 a2 a3 : ℝ) (d : ℝ)
  (h1 : a1 + a2 + a3 = 11/18)
  (h2 : 1/a1 + 1/a2 + 1/a3 = 18)
  (h3 : 1/a2 = 1/a1 + d)
  (h4 : 1/a3 = 1/a1 + 2*d) :
  (a1 = 1/9 ∧ a2 = 1/6 ∧ a3 = 1/3) ∨ (a1 = 1/3 ∧ a2 = 1/6 ∧ a3 = 1/9) :=
sorry

end NUMINAMATH_GPT_sum_of_reciprocal_AP_l1869_186913


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1869_186996

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n + 1) = r * a n)
    (h1 : a 0 + a 1 = 324) (h2 : a 2 + a 3 = 36) : a 4 + a 5 = 4 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1869_186996


namespace NUMINAMATH_GPT_contestant_final_score_l1869_186952

theorem contestant_final_score (score_content score_skills score_effects : ℕ) 
                               (weight_content weight_skills weight_effects : ℕ) :
    score_content = 90 →
    score_skills  = 80 →
    score_effects = 90 →
    weight_content = 4 →
    weight_skills  = 2 →
    weight_effects = 4 →
    (score_content * weight_content + score_skills * weight_skills + score_effects * weight_effects) / 
    (weight_content + weight_skills + weight_effects) = 88 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_contestant_final_score_l1869_186952


namespace NUMINAMATH_GPT_race_track_width_l1869_186904

noncomputable def width_of_race_track (C_inner : ℝ) (r_outer : ℝ) : ℝ :=
  let r_inner := C_inner / (2 * Real.pi)
  r_outer - r_inner

theorem race_track_width : 
  width_of_race_track 880 165.0563499208679 = 25.0492072460867 :=
by
  sorry

end NUMINAMATH_GPT_race_track_width_l1869_186904


namespace NUMINAMATH_GPT_find_other_number_product_find_third_number_sum_l1869_186995

-- First Question
theorem find_other_number_product (x : ℚ) (h : x * (1/7 : ℚ) = -2) : x = -14 :=
sorry

-- Second Question
theorem find_third_number_sum (y : ℚ) (h : (1 : ℚ) + (-4) + y = -5) : y = -2 :=
sorry

end NUMINAMATH_GPT_find_other_number_product_find_third_number_sum_l1869_186995


namespace NUMINAMATH_GPT_expand_expression_l1869_186900

theorem expand_expression : ∀ (x : ℝ), (20 * x - 25) * 3 * x = 60 * x^2 - 75 * x := 
by
  intro x
  sorry

end NUMINAMATH_GPT_expand_expression_l1869_186900


namespace NUMINAMATH_GPT_shepherd_boys_equation_l1869_186902

theorem shepherd_boys_equation (x : ℕ) :
  6 * x + 14 = 8 * x - 2 :=
by sorry

end NUMINAMATH_GPT_shepherd_boys_equation_l1869_186902


namespace NUMINAMATH_GPT_student_correct_sums_l1869_186983

-- Defining variables R and W along with the given conditions
variables (R W : ℕ)

-- Given conditions as Lean definitions
def condition1 := W = 5 * R
def condition2 := R + W = 180

-- Statement of the problem to prove R equals 30
theorem student_correct_sums :
  (W = 5 * R) → (R + W = 180) → R = 30 :=
by
  -- Import needed definitions and theorems from Mathlib
  sorry -- skipping the proof

end NUMINAMATH_GPT_student_correct_sums_l1869_186983


namespace NUMINAMATH_GPT_find_S13_l1869_186980

-- Define the arithmetic sequence
variable (a : ℕ → ℤ) (S : ℕ → ℤ)

-- The sequence is arithmetic, i.e., there exists a common difference d
variable (d : ℤ)
axiom arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + d

-- The sum of the first n terms is given by S_n
axiom sum_of_terms : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

-- Given condition
axiom given_condition : a 1 + a 8 + a 12 = 12

-- We need to prove that S_{13} = 52
theorem find_S13 : S 13 = 52 :=
sorry

end NUMINAMATH_GPT_find_S13_l1869_186980


namespace NUMINAMATH_GPT_complex_point_second_quadrant_l1869_186942

theorem complex_point_second_quadrant (i : ℂ) (h1 : i^4 = 1) :
  ∃ (z : ℂ), z = ((i^(2014))/(1 + i) * i) ∧ z.re < 0 ∧ z.im > 0 :=
by
  sorry

end NUMINAMATH_GPT_complex_point_second_quadrant_l1869_186942


namespace NUMINAMATH_GPT_find_number_of_people_l1869_186949

def number_of_people (total_shoes : Nat) (shoes_per_person : Nat) : Nat :=
  total_shoes / shoes_per_person

theorem find_number_of_people :
  number_of_people 20 2 = 10 := 
by
  sorry

end NUMINAMATH_GPT_find_number_of_people_l1869_186949


namespace NUMINAMATH_GPT_b_share_220_l1869_186960

theorem b_share_220 (A B C : ℝ) (h1 : A = B + 40) (h2 : C = A + 30) (h3 : B + A + C = 770) : B = 220 :=
by
  sorry

end NUMINAMATH_GPT_b_share_220_l1869_186960


namespace NUMINAMATH_GPT_probability_neither_red_nor_purple_l1869_186939

theorem probability_neither_red_nor_purple :
  (100 - (47 + 3)) / 100 = 0.5 :=
by sorry

end NUMINAMATH_GPT_probability_neither_red_nor_purple_l1869_186939


namespace NUMINAMATH_GPT_trigonometric_inequality_l1869_186921

theorem trigonometric_inequality (a b x : ℝ) :
  (Real.sin x + a * Real.cos x) * (Real.sin x + b * Real.cos x) ≤ 1 + ( (a + b) / 2 )^2 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_inequality_l1869_186921


namespace NUMINAMATH_GPT_distance_squared_from_B_to_origin_l1869_186970

-- Conditions:
-- 1. the radius of the circle is 10 cm
-- 2. the length of AB is 8 cm
-- 3. the length of BC is 3 cm
-- 4. the angle ABC is a right angle
-- 5. the center of the circle is at the origin
-- a^2 + b^2 is the square of the distance from B to the center of the circle (origin)

theorem distance_squared_from_B_to_origin
  (a b : ℝ)
  (h1 : a^2 + (b + 8)^2 = 100)
  (h2 : (a + 3)^2 + b^2 = 100)
  (h3 : 6 * a - 16 * b = 55) : a^2 + b^2 = 50 :=
sorry

end NUMINAMATH_GPT_distance_squared_from_B_to_origin_l1869_186970


namespace NUMINAMATH_GPT_max_value_Sn_l1869_186974

theorem max_value_Sn (a₁ : ℚ) (r : ℚ) (S : ℕ → ℚ)
  (h₀ : a₁ = 3 / 2)
  (h₁ : r = -1 / 2)
  (h₂ : ∀ n, S n = a₁ * (1 - r ^ n) / (1 - r))
  : ∀ n, S n ≤ 3 / 2 ∧ (∃ m, S m = 3 / 2) :=
by sorry

end NUMINAMATH_GPT_max_value_Sn_l1869_186974


namespace NUMINAMATH_GPT_translate_down_by_2_l1869_186964

theorem translate_down_by_2 (x y : ℝ) (h : y = -2 * x + 3) : y - 2 = -2 * x + 1 := 
by 
  sorry

end NUMINAMATH_GPT_translate_down_by_2_l1869_186964


namespace NUMINAMATH_GPT_number_of_shortest_paths_l1869_186966

-- Define the concept of shortest paths
def shortest_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

-- State the theorem that needs to be proved
theorem number_of_shortest_paths (m n : ℕ) : shortest_paths m n = Nat.choose (m + n) m :=
by 
  sorry

end NUMINAMATH_GPT_number_of_shortest_paths_l1869_186966


namespace NUMINAMATH_GPT_no_net_coin_change_l1869_186972

noncomputable def probability_no_coin_change_each_round : ℚ :=
  (1 / 3) ^ 5

theorem no_net_coin_change :
  probability_no_coin_change_each_round = 1 / 243 := by
  sorry

end NUMINAMATH_GPT_no_net_coin_change_l1869_186972


namespace NUMINAMATH_GPT_slope_product_l1869_186955

   -- Define the hyperbola
   def hyperbola (x y : ℝ) : Prop := x^2 - (2 * y^2) / (Real.sqrt 5 + 1) = 1

   -- Define the slope calculation for points P, M, N on the hyperbola
   def slopes (xP yP x0 y0 : ℝ) (hP : hyperbola xP yP) (hM : hyperbola x0 y0) (hN : hyperbola (-x0) (-y0)) :
     (Real.sqrt 5 + 1) / 2 = ((yP - y0) * (yP + y0)) / ((xP - x0) * (xP + x0)) := sorry
  
   -- Theorem to show the required relationship
   theorem slope_product (xP yP x0 y0 : ℝ) (hP : hyperbola xP yP) (hM : hyperbola x0 y0) (hN : hyperbola (-x0) (-y0)) :
     (yP^2 - y0^2) / (xP^2 - x0^2) = (Real.sqrt 5 + 1) / 2 := sorry
   
end NUMINAMATH_GPT_slope_product_l1869_186955


namespace NUMINAMATH_GPT_product_three_numbers_l1869_186922

theorem product_three_numbers 
  (a b c : ℝ)
  (h1 : a + b + c = 30)
  (h2 : a = 3 * (b + c))
  (h3 : b = 5 * c) : 
  a * b * c = 176 := 
by
  sorry

end NUMINAMATH_GPT_product_three_numbers_l1869_186922


namespace NUMINAMATH_GPT_average_speed_triathlon_l1869_186903

theorem average_speed_triathlon :
  let swimming_distance := 1.5
  let biking_distance := 3
  let running_distance := 2
  let swimming_speed := 2
  let biking_speed := 25
  let running_speed := 8

  let t_s := swimming_distance / swimming_speed
  let t_b := biking_distance / biking_speed
  let t_r := running_distance / running_speed
  let total_time := t_s + t_b + t_r

  let total_distance := swimming_distance + biking_distance + running_distance
  let average_speed := total_distance / total_time

  average_speed = 5.8 :=
  by
    sorry

end NUMINAMATH_GPT_average_speed_triathlon_l1869_186903


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1869_186994

theorem solution_set_of_inequality (x : ℝ) : (2 * x + 3) * (4 - x) > 0 ↔ -3 / 2 < x ∧ x < 4 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1869_186994


namespace NUMINAMATH_GPT_club_committee_probability_l1869_186943

noncomputable def probability_at_least_two_boys_and_two_girls (total_members boys girls committee_size : ℕ) : ℚ :=
  let total_ways := Nat.choose total_members committee_size
  let ways_fewer_than_two_boys := (Nat.choose girls committee_size) + (boys * Nat.choose girls (committee_size - 1))
  let ways_fewer_than_two_girls := (Nat.choose boys committee_size) + (girls * Nat.choose boys (committee_size - 1))
  let ways_invalid := ways_fewer_than_two_boys + ways_fewer_than_two_girls
  (total_ways - ways_invalid) / total_ways

theorem club_committee_probability :
  probability_at_least_two_boys_and_two_girls 30 12 18 6 = 457215 / 593775 :=
by
  sorry

end NUMINAMATH_GPT_club_committee_probability_l1869_186943


namespace NUMINAMATH_GPT_valid_cube_placements_count_l1869_186968

-- Define the initial cross configuration and the possible placements for the sixth square.
structure CrossConfiguration :=
  (squares : Finset (ℕ × ℕ)) -- Assume (ℕ × ℕ) represents the positions of the squares.

def valid_placements (config : CrossConfiguration) : Finset (ℕ × ℕ) :=
  -- Placeholder definition to represent the valid placements for the sixth square.
  sorry

theorem valid_cube_placements_count (config : CrossConfiguration) :
  (valid_placements config).card = 4 := 
by 
  sorry

end NUMINAMATH_GPT_valid_cube_placements_count_l1869_186968


namespace NUMINAMATH_GPT_ratio_of_investments_l1869_186935

theorem ratio_of_investments {A B C : ℝ} (x y z k : ℝ)
  (h1 : B - A = 100)
  (h2 : A + B + C = 2900)
  (h3 : A = 6 * k)
  (h4 : B = 5 * k)
  (h5 : C = 4 * k) : 
  (x / y = 6 / 5) ∧ (y / z = 5 / 4) ∧ (x / z = 6 / 4) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_investments_l1869_186935


namespace NUMINAMATH_GPT_candy_bars_per_bag_l1869_186940

def total_candy_bars : ℕ := 15
def number_of_bags : ℕ := 5

theorem candy_bars_per_bag : total_candy_bars / number_of_bags = 3 :=
by
  sorry

end NUMINAMATH_GPT_candy_bars_per_bag_l1869_186940


namespace NUMINAMATH_GPT_kerosene_sale_difference_l1869_186990

noncomputable def rice_price : ℝ := 0.33
noncomputable def price_of_dozen_eggs := rice_price
noncomputable def price_of_one_egg := rice_price / 12
noncomputable def price_of_half_liter_kerosene := 4 * price_of_one_egg
noncomputable def price_of_one_liter_kerosene := 2 * price_of_half_liter_kerosene
noncomputable def kerosene_discounted := price_of_one_liter_kerosene * 0.95
noncomputable def kerosene_diff_cents := (price_of_one_liter_kerosene - kerosene_discounted) * 100

theorem kerosene_sale_difference :
  kerosene_diff_cents = 1.1 := by sorry

end NUMINAMATH_GPT_kerosene_sale_difference_l1869_186990


namespace NUMINAMATH_GPT_pizza_volume_l1869_186976

theorem pizza_volume (h : ℝ) (d : ℝ) (n : ℕ) 
  (h_cond : h = 1/2) 
  (d_cond : d = 16) 
  (n_cond : n = 8) 
  : (π * (d / 2) ^ 2 * h / n = 4 * π) :=
by
  sorry

end NUMINAMATH_GPT_pizza_volume_l1869_186976


namespace NUMINAMATH_GPT_find_x_l1869_186901

-- Define the conditions as hypotheses
def problem_statement (x : ℤ) : Prop :=
  (3 * x > 30) ∧ (x ≥ 10) ∧ (x > 5) ∧ 
  (x = 9)

-- Define the theorem statement
theorem find_x : ∃ x : ℤ, problem_statement x :=
by
  -- Sorry to skip proof as instructed
  sorry

end NUMINAMATH_GPT_find_x_l1869_186901


namespace NUMINAMATH_GPT_arithmetic_seq_a6_l1869_186998

theorem arithmetic_seq_a6 (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (0 < q) →
  a 1 = 1 →
  S 3 = 7/4 →
  S n = (1 - q^n) / (1 - q) →
  (∀ n, a n = 1 * q^(n - 1)) →
  a 6 = 1 / 32 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_a6_l1869_186998


namespace NUMINAMATH_GPT_max_product_913_l1869_186953

-- Define the condition that ensures the digits are from the set {3, 5, 8, 9, 1}
def valid_digits (digits : List ℕ) : Prop :=
  digits = [3, 5, 8, 9, 1]

-- Define the predicate for a valid three-digit and two-digit integer
def valid_numbers (a b c d e : ℕ) : Prop :=
  valid_digits [a, b, c, d, e] ∧
  ∃ x y, 100 * x + 10 * 1 + y = 10 * d + e ∧ d ≠ 1 ∧ a ≠ 1

-- Define the product function
def product (a b c d e : ℕ) : ℕ :=
  (100 * a + 10 * b + c) * (10 * d + e)

-- State the theorem
theorem max_product_913 : ∀ (a b c d e : ℕ), valid_numbers a b c d e → 
(product a b c d e) ≤ (product 9 1 3 8 5) :=
by
  intros a b c d e
  unfold valid_numbers product 
  sorry

end NUMINAMATH_GPT_max_product_913_l1869_186953


namespace NUMINAMATH_GPT_area_below_line_l1869_186977

-- Define the conditions provided in the problem.
def graph_eq (x y : ℝ) : Prop := x^2 - 14*x + 3*y + 70 = 21 + 11*y - y^2
def line_eq (x y : ℝ) : Prop := y = x - 3

-- State the final proof problem which is to find the area under the given conditions.
theorem area_below_line :
  ∃ area : ℝ, area = 8 * Real.pi ∧ 
  (∀ x y, graph_eq x y → y ≤ x - 3 → -area / 2 ≤ y ∧ y ≤ area / 2) := 
sorry

end NUMINAMATH_GPT_area_below_line_l1869_186977


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1869_186993

theorem simplify_and_evaluate (x : ℚ) (h1 : x = -1/3) :
    (3 * x + 2) * (3 * x - 2) - 5 * x * (x - 1) - (2 * x - 1)^2 = 9 * x - 5 ∧
    (9 * x - 5) = -8 := 
by sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1869_186993


namespace NUMINAMATH_GPT_original_price_of_computer_l1869_186933

theorem original_price_of_computer (P : ℝ) (h1 : 1.20 * P = 351) (h2 : 2 * P = 585) : P = 292.5 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_computer_l1869_186933


namespace NUMINAMATH_GPT_number_of_sides_of_polygon_l1869_186978

theorem number_of_sides_of_polygon (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 := 
sorry

end NUMINAMATH_GPT_number_of_sides_of_polygon_l1869_186978


namespace NUMINAMATH_GPT_Ivanka_more_months_l1869_186906

variable (I : ℕ) (W : ℕ)

theorem Ivanka_more_months (hW : W = 18) (hI_W : I + W = 39) : I - W = 3 :=
by
  sorry

end NUMINAMATH_GPT_Ivanka_more_months_l1869_186906


namespace NUMINAMATH_GPT_simplify_expr_l1869_186987

variable (a b : ℝ)

def expr := a * b - (a^2 - a * b + b^2)

theorem simplify_expr : expr a b = - a^2 + 2 * a * b - b^2 :=
by 
  -- No proof is provided as per the instructions
  sorry

end NUMINAMATH_GPT_simplify_expr_l1869_186987


namespace NUMINAMATH_GPT_expression_evaluation_l1869_186917

theorem expression_evaluation : (3 * 15) + 47 - 27 * (2^3) / 4 = 38 := by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1869_186917


namespace NUMINAMATH_GPT_range_of_b_for_increasing_f_l1869_186956

noncomputable def f (b x : ℝ) : ℝ :=
  if x > 1 then (2 * b - 1) / x + b + 3 else -x^2 + (2 - b) * x

theorem range_of_b_for_increasing_f :
  ∀ b : ℝ, (∀ x1 x2 : ℝ, x1 < x2 → f b x1 ≤ f b x2) ↔ -1/4 ≤ b ∧ b ≤ 0 := 
sorry

end NUMINAMATH_GPT_range_of_b_for_increasing_f_l1869_186956


namespace NUMINAMATH_GPT_even_function_value_for_negative_x_l1869_186946

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_function_value_for_negative_x (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_pos : ∀ (x : ℝ), 0 < x → f x = 10^x) :
  ∀ x : ℝ, x < 0 → f x = 10^(-x) :=
by
  sorry

end NUMINAMATH_GPT_even_function_value_for_negative_x_l1869_186946


namespace NUMINAMATH_GPT_factor_expression_l1869_186999

theorem factor_expression (a : ℝ) : 198 * a ^ 2 + 36 * a + 54 = 18 * (11 * a ^ 2 + 2 * a + 3) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1869_186999


namespace NUMINAMATH_GPT_pencils_removed_l1869_186997

theorem pencils_removed (initial_pencils removed_pencils remaining_pencils : ℕ) 
  (h1 : initial_pencils = 87) 
  (h2 : remaining_pencils = 83) 
  (h3 : removed_pencils = initial_pencils - remaining_pencils) : 
  removed_pencils = 4 :=
sorry

end NUMINAMATH_GPT_pencils_removed_l1869_186997


namespace NUMINAMATH_GPT_complement_A_is_01_l1869_186971

-- Define the universal set U as the set of all real numbers
def U : Set ℝ := Set.univ

-- Define the set A given the conditions
def A : Set ℝ := {x | x ≥ 1} ∪ {x | x < 0}

-- State the theorem: complement of A is the interval [0, 1)
theorem complement_A_is_01 : Set.compl A = {x : ℝ | 0 ≤ x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_complement_A_is_01_l1869_186971


namespace NUMINAMATH_GPT_max_coins_Martha_can_take_l1869_186989

/-- 
  Suppose a total of 2010 coins are distributed in 5 boxes with quantities 
  initially forming consecutive natural numbers. Martha can perform a 
  transformation where she takes one coin from a box with at least 4 coins and 
  distributes one coin to each of the other boxes. Prove that the maximum number 
  of coins that Martha can take away is 2004.
-/
theorem max_coins_Martha_can_take : 
  ∃ (a : ℕ), 2010 = a + (a+1) + (a+2) + (a+3) + (a+4) ∧ 
  ∀ (f : ℕ → ℕ) (h : (∃ b ≥ 4, f b = 400 + b)), 
  (∃ n : ℕ, f n = 4) → (∃ n : ℕ, f n = 3) → 
  (∃ n : ℕ, f n = 2) → (∃ n : ℕ, f n = 1) → 
  (∃ m : ℕ, f m = 2004) := 
by
  sorry

end NUMINAMATH_GPT_max_coins_Martha_can_take_l1869_186989


namespace NUMINAMATH_GPT_right_triangle_area_l1869_186975

theorem right_triangle_area (a b c : ℕ) (h1 : a = 16) (h2 : b = 30) (h3 : c = 34) 
(h4 : a^2 + b^2 = c^2) : 
   1 / 2 * a * b = 240 :=
by 
  sorry

end NUMINAMATH_GPT_right_triangle_area_l1869_186975


namespace NUMINAMATH_GPT_rectangle_area_from_square_l1869_186986

theorem rectangle_area_from_square {s a : ℝ} (h1 : s^2 = 36) (h2 : a = 3 * s) :
    s * a = 108 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_rectangle_area_from_square_l1869_186986


namespace NUMINAMATH_GPT_find_a_and_b_l1869_186982

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b

theorem find_a_and_b (a b : ℝ) (h_a : a < 0) (h_max : a + b = 3) (h_min : -a + b = -1) : a = -2 ∧ b = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_and_b_l1869_186982


namespace NUMINAMATH_GPT_time_to_count_60_envelopes_is_40_time_to_count_90_envelopes_is_10_l1869_186908

noncomputable def time_to_count_envelopes (num_envelopes : ℕ) : ℕ :=
(num_envelopes / 10) * 10

theorem time_to_count_60_envelopes_is_40 :
  time_to_count_envelopes 60 = 40 := 
sorry

theorem time_to_count_90_envelopes_is_10 :
  time_to_count_envelopes 90 = 10 := 
sorry

end NUMINAMATH_GPT_time_to_count_60_envelopes_is_40_time_to_count_90_envelopes_is_10_l1869_186908


namespace NUMINAMATH_GPT_temperature_representation_l1869_186937

def represents_zero_degrees_celsius (t₁ : ℝ) : Prop := t₁ = 10

theorem temperature_representation (t₁ t₂ : ℝ) (h₀ : represents_zero_degrees_celsius t₁) 
    (h₁ : t₂ > t₁):
    t₂ = 17 :=
by
  -- Proof is omitted here
  sorry

end NUMINAMATH_GPT_temperature_representation_l1869_186937


namespace NUMINAMATH_GPT_geometric_series_sum_l1869_186945

theorem geometric_series_sum :
  let a := 1
  let r := 3
  let n := 9
  (1 * (3^n - 1) / (3 - 1)) = 9841 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1869_186945


namespace NUMINAMATH_GPT_rectangle_area_l1869_186905

theorem rectangle_area (a b : ℝ) (x : ℝ) 
  (h1 : x^2 + (x / 2)^2 = (a + b)^2) 
  (h2 : x > 0) : 
  x * (x / 2) = (2 * (a + b)^2) / 5 := 
by 
  sorry

end NUMINAMATH_GPT_rectangle_area_l1869_186905


namespace NUMINAMATH_GPT_union_when_m_is_one_range_of_m_condition_1_range_of_m_condition_2_l1869_186912

open Set

noncomputable def A := {x : ℝ | -2 < x ∧ x < 2}
noncomputable def B (m : ℝ) := {x : ℝ | (m - 2) ≤ x ∧ x ≤ (2 * m + 1)}

-- Part (1):
theorem union_when_m_is_one :
  A ∪ B 1 = {x : ℝ | -2 < x ∧ x ≤ 3} := sorry

-- Part (2):
theorem range_of_m_condition_1 :
  ∀ m : ℝ, A ∩ B m = ∅ ↔ m ∈ Iic (-3/2) ∪ Ici 4 := sorry

theorem range_of_m_condition_2 :
  ∀ m : ℝ, A ∪ B m = A ↔ m ∈ Iio (-3) ∪ Ioo 0 (1/2) := sorry

end NUMINAMATH_GPT_union_when_m_is_one_range_of_m_condition_1_range_of_m_condition_2_l1869_186912


namespace NUMINAMATH_GPT_train_length_l1869_186932

theorem train_length
  (speed_kmph : ℝ)
  (platform_length : ℝ)
  (crossing_time : ℝ)
  (conversion_factor : ℝ)
  (speed_mps : ℝ)
  (distance_covered : ℝ)
  (train_length : ℝ) :
  speed_kmph = 72 →
  platform_length = 240 →
  crossing_time = 26 →
  conversion_factor = 5 / 18 →
  speed_mps = speed_kmph * conversion_factor →
  distance_covered = speed_mps * crossing_time →
  train_length = distance_covered - platform_length →
  train_length = 280 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_train_length_l1869_186932


namespace NUMINAMATH_GPT_limit_of_nested_radical_l1869_186927

theorem limit_of_nested_radical :
  ∃ F : ℝ, F = 43 ∧ F = Real.sqrt (86 + 41 * F) :=
sorry

end NUMINAMATH_GPT_limit_of_nested_radical_l1869_186927


namespace NUMINAMATH_GPT_proof_a_plus_2b_equal_7_l1869_186916

theorem proof_a_plus_2b_equal_7 (a b : ℕ) (h1 : 82 * 1000 + a * 10 + 7 + 6 * b = 190) (h2 : 1 ≤ a) (h3 : a < 10) (h4 : 1 ≤ b) (h5 : b < 10) : 
  a + 2 * b = 7 :=
by sorry

end NUMINAMATH_GPT_proof_a_plus_2b_equal_7_l1869_186916


namespace NUMINAMATH_GPT_max_value_f1_on_interval_range_of_a_g_increasing_l1869_186925

noncomputable def f1 (x : ℝ) : ℝ := 2 * x^2 + x + 2

theorem max_value_f1_on_interval : 
  (∀ x, x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) → f1 x ≤ 5) ∧ 
  (∃ x, x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) ∧ f1 x = 5) :=
sorry

noncomputable def f2 (a x : ℝ) : ℝ := a * x^2 + (a - 1) * x + a

theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ Set.Icc (1 : ℝ) (2 : ℝ) → f2 a x / x ≥ 2) → a ≥ 1 :=
sorry

noncomputable def g (a x : ℝ) : ℝ := a * x^2 + (a - 1) * x + a + (1 - (a-1) * x^2) / x

theorem g_increasing (a : ℝ) : 
  (∀ x1 x2, (2 < x1 ∧ x1 < x2 ∧ x2 < 3) → g a x1 < g a x2) → a ≥ 1 / 16 :=
sorry

end NUMINAMATH_GPT_max_value_f1_on_interval_range_of_a_g_increasing_l1869_186925


namespace NUMINAMATH_GPT_women_bathing_suits_count_l1869_186958

theorem women_bathing_suits_count :
  ∀ (total_bathing_suits men_bathing_suits women_bathing_suits : ℕ),
    total_bathing_suits = 19766 →
    men_bathing_suits = 14797 →
    women_bathing_suits = total_bathing_suits - men_bathing_suits →
    women_bathing_suits = 4969 := by
sorry

end NUMINAMATH_GPT_women_bathing_suits_count_l1869_186958


namespace NUMINAMATH_GPT_max_m_minus_n_l1869_186973

theorem max_m_minus_n (m n : ℝ) (h : (m + 1)^2 + (n + 1)^2 = 4) : m - n ≤ 2 * Real.sqrt 2 :=
by {
  -- Here is where the proof would take place.
  sorry
}

end NUMINAMATH_GPT_max_m_minus_n_l1869_186973


namespace NUMINAMATH_GPT_cone_lateral_surface_area_eq_sqrt_17_pi_l1869_186951

theorem cone_lateral_surface_area_eq_sqrt_17_pi
  (r_cone r_sphere : ℝ) (h : ℝ)
  (V_sphere V_cone : ℝ)
  (h_cone_radius : r_cone = 1)
  (h_sphere_radius : r_sphere = 1)
  (h_volumes_eq : V_sphere = V_cone)
  (h_sphere_vol : V_sphere = (4 * π) / 3)
  (h_cone_vol : V_cone = (π * r_cone^2 * h) / 3) :
  (π * r_cone * (Real.sqrt (r_cone^2 + h^2))) = Real.sqrt 17 * π :=
sorry

end NUMINAMATH_GPT_cone_lateral_surface_area_eq_sqrt_17_pi_l1869_186951


namespace NUMINAMATH_GPT_riding_owners_ratio_l1869_186920

theorem riding_owners_ratio :
  ∃ (R W : ℕ), (R + W = 16) ∧ (4 * R + 6 * W = 80) ∧ (R : ℚ) / 16 = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_riding_owners_ratio_l1869_186920


namespace NUMINAMATH_GPT_soda_price_l1869_186957

-- We define the conditions as given in the problem
def regular_price (P : ℝ) : Prop :=
  -- Regular price per can is P
  ∃ P, 
  -- 25 percent discount on regular price when purchased in 24-can cases
  (∀ (discounted_price_per_can : ℝ), discounted_price_per_can = 0.75 * P) ∧
  -- Price of 70 cans at the discounted price is $28.875
  (70 * 0.75 * P = 28.875)

-- We state the theorem to prove that the regular price per can is $0.55
theorem soda_price (P : ℝ) (h : regular_price P) : P = 0.55 :=
by
  sorry

end NUMINAMATH_GPT_soda_price_l1869_186957


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l1869_186915

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0 :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l1869_186915


namespace NUMINAMATH_GPT_eval_expression_l1869_186931

theorem eval_expression : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 2023) / (2023 * 2024) = -4044 :=
by 
  sorry

end NUMINAMATH_GPT_eval_expression_l1869_186931


namespace NUMINAMATH_GPT_determine_x_l1869_186907

/-
  Determine \( x \) when \( y = 19 \)
  given the ratio of \( 5x - 3 \) to \( y + 10 \) is constant,
  and when \( x = 3 \), \( y = 4 \).
-/

theorem determine_x (x y k : ℚ) (h1 : ∀ x y, (5 * x - 3) / (y + 10) = k)
  (h2 : 5 * 3 - 3 / (4 + 10) = k) : x = 39 / 7 :=
sorry

end NUMINAMATH_GPT_determine_x_l1869_186907


namespace NUMINAMATH_GPT_similar_right_triangle_hypotenuse_length_l1869_186961

theorem similar_right_triangle_hypotenuse_length :
  ∀ (a b c d : ℝ), a = 15 → c = 39 → d = 45 → 
  (b^2 = c^2 - a^2) → 
  ∃ e : ℝ, e = (c * (d / b)) ∧ e = 48.75 :=
by
  intros a b c d ha hc hd hb
  sorry

end NUMINAMATH_GPT_similar_right_triangle_hypotenuse_length_l1869_186961


namespace NUMINAMATH_GPT_abc_inequality_l1869_186910

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b + b * c + a * c)^2 ≥ 3 * a * b * c * (a + b + c) :=
by sorry

end NUMINAMATH_GPT_abc_inequality_l1869_186910


namespace NUMINAMATH_GPT_cauchy_schwarz_inequality_l1869_186944

theorem cauchy_schwarz_inequality
  (x1 y1 z1 x2 y2 z2 : ℝ) :
  (x1 * x2 + y1 * y2 + z1 * z2) ^ 2 ≤ (x1 ^ 2 + y1 ^ 2 + z1 ^ 2) * (x2 ^ 2 + y2 ^ 2 + z2 ^ 2) := 
sorry

end NUMINAMATH_GPT_cauchy_schwarz_inequality_l1869_186944


namespace NUMINAMATH_GPT_resulting_polygon_sides_l1869_186938

theorem resulting_polygon_sides 
    (triangle_sides : ℕ := 3) 
    (square_sides : ℕ := 4) 
    (pentagon_sides : ℕ := 5) 
    (heptagon_sides : ℕ := 7) 
    (hexagon_sides : ℕ := 6) 
    (octagon_sides : ℕ := 8) 
    (shared_sides : ℕ := 1) :
    (2 * shared_sides + 4 * (shared_sides + 1)) = 16 := by 
  sorry

end NUMINAMATH_GPT_resulting_polygon_sides_l1869_186938


namespace NUMINAMATH_GPT_not_perfect_square_l1869_186963

theorem not_perfect_square (a b : ℤ) (h : (a % 2 ≠ b % 2)) : ¬ ∃ k : ℤ, ((a + 3 * b) * (5 * a + 7 * b) = k^2) := 
by
  sorry

end NUMINAMATH_GPT_not_perfect_square_l1869_186963


namespace NUMINAMATH_GPT_bananas_first_day_l1869_186929

theorem bananas_first_day (x : ℕ) (h : x + (x + 6) + (x + 12) + (x + 18) + (x + 24) = 100) : x = 8 := by
  sorry

end NUMINAMATH_GPT_bananas_first_day_l1869_186929


namespace NUMINAMATH_GPT_nebraska_more_plates_than_georgia_l1869_186962

theorem nebraska_more_plates_than_georgia :
  (26 ^ 2 * 10 ^ 5) - (26 ^ 4 * 10 ^ 2) = 21902400 :=
by
  sorry

end NUMINAMATH_GPT_nebraska_more_plates_than_georgia_l1869_186962


namespace NUMINAMATH_GPT_not_a_factorization_l1869_186981

open Nat

theorem not_a_factorization : ¬ (∃ (f g : ℝ → ℝ), (∀ (x : ℝ), x^2 + 6*x - 9 = f x * g x)) :=
by
  sorry

end NUMINAMATH_GPT_not_a_factorization_l1869_186981


namespace NUMINAMATH_GPT_abs_neg_2022_eq_2022_l1869_186954

theorem abs_neg_2022_eq_2022 : abs (-2022) = 2022 :=
by
  sorry

end NUMINAMATH_GPT_abs_neg_2022_eq_2022_l1869_186954


namespace NUMINAMATH_GPT_max_value_of_y_is_2_l1869_186979

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + (a - 3) * x

theorem max_value_of_y_is_2 (a : ℝ) (h : ∀ x : ℝ, (3 * x^2 + 2 * a * x + (a - 3)) = (3 * x^2 - 2 * a * x + (a - 3))) : 
  ∃ x : ℝ, f a x = 2 :=
sorry

end NUMINAMATH_GPT_max_value_of_y_is_2_l1869_186979


namespace NUMINAMATH_GPT_find_pairs_l1869_186969

theorem find_pairs (x y : ℝ) (h1 : |x| + |y| = 1340) (h2 : x^3 + y^3 + 2010 * x * y = 670^3) :
  x + y = 670 ∧ x * y = -673350 :=
sorry

end NUMINAMATH_GPT_find_pairs_l1869_186969


namespace NUMINAMATH_GPT_equation_of_parabola_l1869_186985

def parabola_vertex_form_vertex (a x y : ℝ) := y = a * (x - 3)^2 - 2
def parabola_passes_through_point (a : ℝ) := 1 = a * (0 - 3)^2 - 2
def parabola_equation (y x : ℝ) := y = (1/3) * x^2 - 2 * x + 1

theorem equation_of_parabola :
  ∃ a : ℝ,
    ∀ x y : ℝ,
      parabola_vertex_form_vertex a x y ∧
      parabola_passes_through_point a →
      parabola_equation y x :=
by
  sorry

end NUMINAMATH_GPT_equation_of_parabola_l1869_186985


namespace NUMINAMATH_GPT_number_of_adults_l1869_186948

theorem number_of_adults (A C S : ℕ) (h1 : C = A - 35) (h2 : S = 2 * C) (h3 : A + C + S = 127) : A = 58 :=
by
  sorry

end NUMINAMATH_GPT_number_of_adults_l1869_186948


namespace NUMINAMATH_GPT_ratio_of_x_to_y_l1869_186919

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + 3 * y) = 1 / 2) : x / y = 7 / 4 :=
sorry

end NUMINAMATH_GPT_ratio_of_x_to_y_l1869_186919


namespace NUMINAMATH_GPT_cylindrical_can_increase_l1869_186992

theorem cylindrical_can_increase (R H y : ℝ)
  (h₁ : R = 5)
  (h₂ : H = 4)
  (h₃ : π * (R + y)^2 * (H + y) = π * (R + 2*y)^2 * H) :
  y = Real.sqrt 76 - 5 :=
by
  sorry

end NUMINAMATH_GPT_cylindrical_can_increase_l1869_186992


namespace NUMINAMATH_GPT_gcd_mn_eq_one_l1869_186914

def m : ℤ := 123^2 + 235^2 - 347^2
def n : ℤ := 122^2 + 234^2 - 348^2

theorem gcd_mn_eq_one : Int.gcd m n = 1 := 
by
  sorry

end NUMINAMATH_GPT_gcd_mn_eq_one_l1869_186914


namespace NUMINAMATH_GPT_equation_has_at_most_one_real_root_l1869_186928

def has_inverse (f : ℝ → ℝ) : Prop := ∃ g : ℝ → ℝ, ∀ x, g (f x) = x

theorem equation_has_at_most_one_real_root (f : ℝ → ℝ) (a : ℝ) (h : has_inverse f) :
  ∀ x1 x2 : ℝ, f x1 = a ∧ f x2 = a → x1 = x2 :=
by sorry

end NUMINAMATH_GPT_equation_has_at_most_one_real_root_l1869_186928


namespace NUMINAMATH_GPT_exponent_multiplication_l1869_186947

theorem exponent_multiplication (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (a b : ℤ) (h3 : 3^m = a) (h4 : 3^n = b) : 3^(m + n) = a * b :=
by
  sorry

end NUMINAMATH_GPT_exponent_multiplication_l1869_186947


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1869_186934

theorem geometric_sequence_common_ratio 
  (a1 q : ℝ) 
  (h : (a1 * (1 - q^3) / (1 - q)) + 3 * (a1 * (1 - q^2) / (1 - q)) = 0) : 
  q = -1 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1869_186934


namespace NUMINAMATH_GPT_inequality_proof_l1869_186967

theorem inequality_proof (n k : ℕ) (h₁ : 0 < n) (h₂ : 0 < k) (h₃ : k ≤ n) :
  1 + k / n ≤ (1 + 1 / n)^k ∧ (1 + 1 / n)^k < 1 + k / n + k^2 / n^2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1869_186967


namespace NUMINAMATH_GPT_ratio_five_to_one_l1869_186950

theorem ratio_five_to_one (x : ℕ) (h : 5 / 1 = x / 9) : x = 45 :=
  sorry

end NUMINAMATH_GPT_ratio_five_to_one_l1869_186950


namespace NUMINAMATH_GPT_sum_of_digits_of_B_is_7_l1869_186911

theorem sum_of_digits_of_B_is_7 : 
  let A := 16 ^ 16
  let sum_digits (n : ℕ) : ℕ := n.digits 10 |>.sum
  let S := sum_digits
  let B := S (S A)
  sum_digits B = 7 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_of_B_is_7_l1869_186911


namespace NUMINAMATH_GPT_fraction_available_on_third_day_l1869_186909

noncomputable def liters_used_on_first_day (initial_amount : ℕ) : ℕ :=
  (initial_amount / 2)

noncomputable def liters_added_on_second_day : ℕ :=
  1

noncomputable def original_solution : ℕ :=
  4

noncomputable def remaining_solution_after_first_day : ℕ :=
  original_solution - liters_used_on_first_day original_solution

noncomputable def remaining_solution_after_second_day : ℕ :=
  remaining_solution_after_first_day + liters_added_on_second_day

noncomputable def fraction_of_original_solution : ℚ :=
  remaining_solution_after_second_day / original_solution

theorem fraction_available_on_third_day : fraction_of_original_solution = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_available_on_third_day_l1869_186909


namespace NUMINAMATH_GPT_opposite_numbers_l1869_186959

theorem opposite_numbers (a b : ℝ) (h : a = -b) : b = -a := 
by 
  sorry

end NUMINAMATH_GPT_opposite_numbers_l1869_186959


namespace NUMINAMATH_GPT_equation_B_is_quadratic_l1869_186936

theorem equation_B_is_quadratic : ∀ y : ℝ, ∃ A B C : ℝ, (5 * y ^ 2 - 5 * y = 0) ∧ A ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_equation_B_is_quadratic_l1869_186936


namespace NUMINAMATH_GPT_molly_age_is_63_l1869_186923

variable (Sandy_age Molly_age : ℕ)

theorem molly_age_is_63 (h1 : Sandy_age = 49) (h2 : Sandy_age / Molly_age = 7 / 9) : Molly_age = 63 :=
by
  sorry

end NUMINAMATH_GPT_molly_age_is_63_l1869_186923


namespace NUMINAMATH_GPT_find_x2_plus_y2_l1869_186991

theorem find_x2_plus_y2 (x y : ℕ) (h1 : xy + x + y = 35) (h2 : x^2 * y + x * y^2 = 306) : x^2 + y^2 = 290 :=
sorry

end NUMINAMATH_GPT_find_x2_plus_y2_l1869_186991


namespace NUMINAMATH_GPT_committee_count_l1869_186984

theorem committee_count (club_members : Finset ℕ) (h_count : club_members.card = 30) :
  ∃ committee_count : ℕ, committee_count = 2850360 :=
by
  sorry

end NUMINAMATH_GPT_committee_count_l1869_186984


namespace NUMINAMATH_GPT_average_math_score_first_year_students_l1869_186918

theorem average_math_score_first_year_students 
  (total_male_students : ℕ) (total_female_students : ℕ)
  (sample_size : ℕ) (avg_score_male : ℕ) (avg_score_female : ℕ)
  (male_sample_size female_sample_size : ℕ)
  (weighted_avg : ℚ) :
  total_male_students = 300 → 
  total_female_students = 200 →
  sample_size = 60 → 
  avg_score_male = 110 →
  avg_score_female = 100 →
  male_sample_size = (3 * sample_size) / 5 →
  female_sample_size = (2 * sample_size) / 5 →
  weighted_avg = (male_sample_size * avg_score_male + female_sample_size * avg_score_female : ℕ) / sample_size → 
  weighted_avg = 106 := 
by
  sorry

end NUMINAMATH_GPT_average_math_score_first_year_students_l1869_186918


namespace NUMINAMATH_GPT_sqrt_of_9_l1869_186988

theorem sqrt_of_9 : Real.sqrt 9 = 3 :=
by 
  sorry

end NUMINAMATH_GPT_sqrt_of_9_l1869_186988
