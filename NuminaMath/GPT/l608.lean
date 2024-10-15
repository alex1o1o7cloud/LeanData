import Mathlib

namespace NUMINAMATH_GPT_nine_pow_div_eighty_one_pow_l608_60870

theorem nine_pow_div_eighty_one_pow (a b : ℕ) (h1 : a = 9^2) (h2 : b = a^4) :
  (9^10 / b = 81) := by
  sorry

end NUMINAMATH_GPT_nine_pow_div_eighty_one_pow_l608_60870


namespace NUMINAMATH_GPT_denote_below_warning_level_l608_60863

-- Conditions
def warning_water_level : ℝ := 905.7
def exceed_by_10 : ℝ := 10
def below_by_5 : ℝ := -5

-- Problem statement
theorem denote_below_warning_level : below_by_5 = -5 := 
by
  sorry

end NUMINAMATH_GPT_denote_below_warning_level_l608_60863


namespace NUMINAMATH_GPT_max_value_of_expression_l608_60867

theorem max_value_of_expression (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_product : a * b * c = 16) : 
  a^b - b^c + c^a ≤ 263 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l608_60867


namespace NUMINAMATH_GPT_polynomial_coeff_sum_l608_60889

noncomputable def polynomial_expansion (x : ℝ) :=
  (2 * x + 3) * (4 * x^3 - 2 * x^2 + x - 7)

theorem polynomial_coeff_sum :
  let A := 8
  let B := 8
  let C := -4
  let D := -11
  let E := -21
  A + B + C + D + E = -20 :=
by
  -- The following proof steps are skipped
  sorry

end NUMINAMATH_GPT_polynomial_coeff_sum_l608_60889


namespace NUMINAMATH_GPT_find_g_l608_60847

variable (x : ℝ)

-- Given condition
def given_condition (g : ℝ → ℝ) : Prop :=
  5 * x^5 + 3 * x^3 - 4 * x + 2 + g x = 7 * x^3 - 9 * x^2 + x + 5

-- Goal
def goal (g : ℝ → ℝ) : Prop :=
  g x = -5 * x^5 + 4 * x^3 - 9 * x^2 + 5 * x + 3

-- The statement combining given condition and goal to prove
theorem find_g (g : ℝ → ℝ) (h : given_condition x g) : goal x g :=
by
  sorry

end NUMINAMATH_GPT_find_g_l608_60847


namespace NUMINAMATH_GPT_hyperbola_equation_l608_60872

noncomputable def focal_distance : ℝ := 10
noncomputable def c : ℝ := 5
noncomputable def point_P : (ℝ × ℝ) := (2, 1)
noncomputable def eq1 : Prop := ∀ (x y : ℝ), (x^2) / 20 - (y^2) / 5 = 1 ↔ c = 5 ∧ focal_distance = 10 ∧ point_P.1 = 2 ∧ point_P.2 = 1
noncomputable def eq2 : Prop := ∀ (x y : ℝ), (y^2) / 5 - (x^2) / 20 = 1 ↔ c = 5 ∧ focal_distance = 10 ∧ point_P.1 = 2 ∧ point_P.2 = 1

theorem hyperbola_equation :
  (∃ a b : ℝ, c = 5 ∧ focal_distance = 10 ∧ point_P.1 = 2 ∧ point_P.2 = 1 ∧ 
    (∀ x y : ℝ, (x^2) / a^2 - (y^2) / b^2 = 1) ∨ 
    (∃ a' b' : ℝ, c = 5 ∧ focal_distance = 10 ∧ point_P.1 = 2 ∧ point_P.2 = 1 ∧ 
      (∀ x y : ℝ, (y^2) / a'^2 - (x^2) / b'^2 = 1))) :=
by sorry

end NUMINAMATH_GPT_hyperbola_equation_l608_60872


namespace NUMINAMATH_GPT_smallest_five_sequential_number_greater_than_2000_is_2004_l608_60884

def fiveSequentialNumber (N : ℕ) : Prop :=
  (if 1 ∣ N then 1 else 0) + 
  (if 2 ∣ N then 1 else 0) + 
  (if 3 ∣ N then 1 else 0) + 
  (if 4 ∣ N then 1 else 0) + 
  (if 5 ∣ N then 1 else 0) + 
  (if 6 ∣ N then 1 else 0) + 
  (if 7 ∣ N then 1 else 0) + 
  (if 8 ∣ N then 1 else 0) + 
  (if 9 ∣ N then 1 else 0) ≥ 5

theorem smallest_five_sequential_number_greater_than_2000_is_2004 :
  ∀ N > 2000, fiveSequentialNumber N → N = 2004 :=
by
  intros N hn hfsn
  have hN : N = 2004 := sorry
  exact hN

end NUMINAMATH_GPT_smallest_five_sequential_number_greater_than_2000_is_2004_l608_60884


namespace NUMINAMATH_GPT_exists_triangle_with_sides_l2_l3_l4_l608_60864

theorem exists_triangle_with_sides_l2_l3_l4
  (a1 a2 a3 a4 d : ℝ)
  (h_arith_seq : a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d)
  (h_pos : a1 > 0 ∧ a2 > 0 ∧ a3 > 0 ∧ a4 > 0)
  (h_d_pos : d > 0) :
  a2 + a3 > a4 ∧ a3 + a4 > a2 ∧ a4 + a2 > a3 :=
by
  sorry

end NUMINAMATH_GPT_exists_triangle_with_sides_l2_l3_l4_l608_60864


namespace NUMINAMATH_GPT_find_principal_l608_60887

theorem find_principal (R : ℝ) (P : ℝ) (h : (P * (R + 2) * 4) / 100 = (P * R * 4) / 100 + 56) : P = 700 := 
sorry

end NUMINAMATH_GPT_find_principal_l608_60887


namespace NUMINAMATH_GPT_train_ticket_product_l608_60877

theorem train_ticket_product
  (a b c d e : ℕ)
  (h1 : b = a + 1)
  (h2 : c = a + 2)
  (h3 : d = a + 3)
  (h4 : e = a + 4)
  (h_sum : a + b + c + d + e = 120) :
  a * b * c * d * e = 7893600 :=
sorry

end NUMINAMATH_GPT_train_ticket_product_l608_60877


namespace NUMINAMATH_GPT_exists_integer_square_with_three_identical_digits_l608_60899

theorem exists_integer_square_with_three_identical_digits:
  ∃ x: ℤ, (x^2 % 1000 = 444) := by
  sorry

end NUMINAMATH_GPT_exists_integer_square_with_three_identical_digits_l608_60899


namespace NUMINAMATH_GPT_count_ordered_triples_l608_60833

def S := Finset.range 20

def succ (a b : ℕ) : Prop := 
  (0 < a - b ∧ a - b ≤ 10) ∨ (b - a > 10)

theorem count_ordered_triples 
  (h : ∃ n : ℕ, (S.card = 20) ∧
                (∀ x y z : ℕ, 
                   x ∈ S → y ∈ S → z ∈ S →
                   (succ x y) → (succ y z) → (succ z x) →
                   n = 1260)) : True := sorry

end NUMINAMATH_GPT_count_ordered_triples_l608_60833


namespace NUMINAMATH_GPT_range_of_w_l608_60880

noncomputable def f (w x : ℝ) : ℝ := Real.sin (w * x) - Real.sqrt 3 * Real.cos (w * x)

theorem range_of_w (w : ℝ) (h_w : 0 < w) :
  (∀ f_zeros : Finset ℝ, ∀ x ∈ f_zeros, (0 < x ∧ x < Real.pi) → f w x = 0 → f_zeros.card = 2) ↔
  (4 / 3 < w ∧ w ≤ 7 / 3) :=
by sorry

end NUMINAMATH_GPT_range_of_w_l608_60880


namespace NUMINAMATH_GPT_card_S_l608_60861

def a (n : ℕ) : ℕ := 2 ^ n

def b (n : ℕ) : ℕ := 5 * n - 1

def S : Finset ℕ := 
  (Finset.range 2016).image a ∩ (Finset.range (a 2015 + 1)).image b

theorem card_S : S.card = 504 := 
  sorry

end NUMINAMATH_GPT_card_S_l608_60861


namespace NUMINAMATH_GPT_problem_conditions_l608_60831

noncomputable def f (x : ℝ) : ℝ := (2 * x - x^2) * Real.exp x

theorem problem_conditions :
  (∀ x, f x > 0 ↔ 0 < x ∧ x < 2) ∧
  (∃ x_max, x_max = Real.sqrt 2 ∧ (∀ y, f y ≤ f x_max)) ∧
  ¬(∃ x_min, ∀ y, f x_min ≤ f y) :=
by sorry

end NUMINAMATH_GPT_problem_conditions_l608_60831


namespace NUMINAMATH_GPT_magician_act_reappearance_l608_60821

-- Defining the conditions as given in the problem
def total_performances : ℕ := 100

def no_one_reappears (perf : ℕ) : ℕ := perf / 10
def two_reappear (perf : ℕ) : ℕ := perf / 5
def one_reappears (perf : ℕ) : ℕ := perf - no_one_reappears perf - two_reappear perf
def total_reappeared (perf : ℕ) : ℕ := one_reappears perf + 2 * two_reappear perf

-- The statement to be proved
theorem magician_act_reappearance : total_reappeared total_performances = 110 := by
  sorry

end NUMINAMATH_GPT_magician_act_reappearance_l608_60821


namespace NUMINAMATH_GPT_osborn_friday_time_l608_60827

-- Conditions
def time_monday : ℕ := 2
def time_tuesday : ℕ := 4
def time_wednesday : ℕ := 3
def time_thursday : ℕ := 4
def old_average_time_per_day : ℕ := 3
def school_days_per_week : ℕ := 5

-- Total time needed to match old average
def total_time_needed : ℕ := old_average_time_per_day * school_days_per_week

-- Total time spent from Monday to Thursday
def time_spent_mon_to_thu : ℕ := time_monday + time_tuesday + time_wednesday + time_thursday

-- Goal: Find time on Friday
def time_friday : ℕ := total_time_needed - time_spent_mon_to_thu

theorem osborn_friday_time : time_friday = 2 :=
by
  sorry

end NUMINAMATH_GPT_osborn_friday_time_l608_60827


namespace NUMINAMATH_GPT_paint_cost_is_624_rs_l608_60857

-- Given conditions:
-- Length of floor is 21.633307652783934 meters.
-- Length is 200% more than the breadth (i.e., length = 3 * breadth).
-- Cost to paint the floor is Rs. 4 per square meter.

noncomputable def length : ℝ := 21.633307652783934
noncomputable def cost_per_sq_meter : ℝ := 4
noncomputable def breadth : ℝ := length / 3
noncomputable def area : ℝ := length * breadth
noncomputable def total_cost : ℝ := area * cost_per_sq_meter

theorem paint_cost_is_624_rs : total_cost = 624 := by
  sorry

end NUMINAMATH_GPT_paint_cost_is_624_rs_l608_60857


namespace NUMINAMATH_GPT_there_exists_triangle_part_two_l608_60824

noncomputable def exists_triangle (a b c : ℝ) : Prop :=
a > 0 ∧
4 * a - 8 * b + 4 * c ≥ 0 ∧
9 * a - 12 * b + 4 * c ≥ 0 ∧
2 * a ≤ 2 * b ∧
2 * b ≤ 3 * a ∧
b^2 ≥ a*c

theorem there_exists_triangle (a b c : ℝ) (h1 : a > 0)
  (h2 : 4 * a - 8 * b + 4 * c ≥ 0)
  (h3 : 9 * a - 12 * b + 4 * c ≥ 0)
  (h4 : 2 * a ≤ 2 * b)
  (h5 : 2 * b ≤ 3 * a)
  (h6 : b^2 ≥ a * c) : 
 a ≤ b ∧ b ≤ c ∧ a + b > c :=
sorry

theorem part_two (a b c : ℝ) (h1 : a > 0) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c < a + b) :
  ∃ h : a > 0, (a / (a + c) + b / (b + a) > c / (b + c)) :=
sorry

end NUMINAMATH_GPT_there_exists_triangle_part_two_l608_60824


namespace NUMINAMATH_GPT_minimum_value_fraction_l608_60836

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem minimum_value_fraction (a : ℕ → ℝ) (m n : ℕ) (q : ℝ) (h_geometric : geometric_sequence a q)
  (h_positive : ∀ k : ℕ, 0 < a k)
  (h_condition1 : a 7 = a 6 + 2 * a 5)
  (h_condition2 : ∃ r, r ^ 2 = a m * a n ∧ r = 2 * a 1) :
  (1 / m + 9 / n) ≥ 4 :=
  sorry

end NUMINAMATH_GPT_minimum_value_fraction_l608_60836


namespace NUMINAMATH_GPT_find_x_from_conditions_l608_60838

theorem find_x_from_conditions (a b x y s : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) :
  s = (4 * a)^(4 * b) ∧ s = a^b * y^b ∧ y = 4 * x → x = 64 * a^3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_from_conditions_l608_60838


namespace NUMINAMATH_GPT_find_x_l608_60895

theorem find_x (x y : ℤ) (h₁ : x / y = 12 / 5) (h₂ : y = 25) : x = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l608_60895


namespace NUMINAMATH_GPT_range_of_m_l608_60882

-- Definition of the quadratic function
def quadratic_function (m x : ℝ) : ℝ :=
  x^2 + (m - 1) * x + 1

-- Statement of the proof problem in Lean
theorem range_of_m (m : ℝ) : 
  (∀ x : ℤ, 0 ≤ x ∧ x ≤ 5 → quadratic_function m x ≥ quadratic_function m (x + 1)) ↔ m ≤ -8 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l608_60882


namespace NUMINAMATH_GPT_num_palindromes_is_correct_l608_60852

section Palindromes

def num_alphanumeric_chars : ℕ := 10 + 26

def num_four_char_palindromes : ℕ := num_alphanumeric_chars * num_alphanumeric_chars

theorem num_palindromes_is_correct : num_four_char_palindromes = 1296 :=
by
  sorry

end Palindromes

end NUMINAMATH_GPT_num_palindromes_is_correct_l608_60852


namespace NUMINAMATH_GPT_find_fourth_number_l608_60888

theorem find_fourth_number 
  (average : ℝ) 
  (a1 a2 a3 : ℝ) 
  (x : ℝ) 
  (n : ℝ) 
  (h1 : average = 20) 
  (h2 : a1 = 3) 
  (h3 : a2 = 16) 
  (h4 : a3 = 33) 
  (h5 : n = 27) 
  (h_avg : (a1 + a2 + a3 + x) / 4 = average) :
  x = n + 1 :=
by
  sorry

end NUMINAMATH_GPT_find_fourth_number_l608_60888


namespace NUMINAMATH_GPT_desired_depth_l608_60853

-- Define the given conditions
def men_hours_30m (d : ℕ) : ℕ := 18 * 8 * d
def men_hours_Dm (d1 : ℕ) (D : ℕ) : ℕ := 40 * 6 * d1

-- Define the proportion
def proportion (d d1 : ℕ) (D : ℕ) : Prop :=
  (men_hours_30m d) / 30 = (men_hours_Dm d1 D) / D

-- The main theorem to prove the desired depth
theorem desired_depth (d d1 : ℕ) (H : proportion d d1 50) : 50 = 50 :=
by sorry

end NUMINAMATH_GPT_desired_depth_l608_60853


namespace NUMINAMATH_GPT_rectangle_length_is_16_l608_60883

noncomputable def rectangle_length (b : ℝ) (c : ℝ) : ℝ :=
  let pi := Real.pi
  let full_circle_circumference := 2 * c
  let radius := full_circle_circumference / (2 * pi)
  let diameter := 2 * radius
  let side_length_of_square := diameter
  let perimeter_of_square := 4 * side_length_of_square
  let perimeter_of_rectangle := perimeter_of_square
  let length_of_rectangle := (perimeter_of_rectangle / 2) - b
  length_of_rectangle

theorem rectangle_length_is_16 :
  rectangle_length 14 23.56 = 16 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_length_is_16_l608_60883


namespace NUMINAMATH_GPT_find_f_x_l608_60844

def f (x : ℝ) : ℝ := sorry

theorem find_f_x (x : ℝ) (h : 2 * f x - f (-x) = 3 * x) : f x = x := 
by sorry

end NUMINAMATH_GPT_find_f_x_l608_60844


namespace NUMINAMATH_GPT_area_of_45_45_90_triangle_l608_60822

noncomputable def leg_length (hypotenuse : ℝ) : ℝ :=
  hypotenuse / Real.sqrt 2

theorem area_of_45_45_90_triangle (hypotenuse : ℝ) (h : hypotenuse = 13) : 
  (1 / 2) * (leg_length hypotenuse) * (leg_length hypotenuse) = 84.5 :=
by
  sorry

end NUMINAMATH_GPT_area_of_45_45_90_triangle_l608_60822


namespace NUMINAMATH_GPT_x_add_y_eq_neg_one_l608_60851

theorem x_add_y_eq_neg_one (x y : ℝ) (h : |x + 3| + (y - 2)^2 = 0) : x + y = -1 :=
by sorry

end NUMINAMATH_GPT_x_add_y_eq_neg_one_l608_60851


namespace NUMINAMATH_GPT_quadratic_expression_positive_intervals_l608_60876

noncomputable def quadratic_expression (x : ℝ) : ℝ := (x + 3) * (x - 1)
def interval_1 (x : ℝ) : Prop := x < (1 - Real.sqrt 13) / 2
def interval_2 (x : ℝ) : Prop := x > (1 + Real.sqrt 13) / 2

theorem quadratic_expression_positive_intervals (x : ℝ) :
  quadratic_expression x > 0 ↔ interval_1 x ∨ interval_2 x :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_expression_positive_intervals_l608_60876


namespace NUMINAMATH_GPT_interval_intersection_l608_60820

/--
  This statement asserts that the intersection of the intervals (2/4, 3/4) and (2/5, 3/5)
  results in the interval (1/2, 0.6), which is the solution to the problem.
-/
theorem interval_intersection :
  { x : ℝ | 2 < 4 * x ∧ 4 * x < 3 ∧ 2 < 5 * x ∧ 5 * x < 3 } = { x : ℝ | 0.5 < x ∧ x < 0.6 } :=
by
  sorry

end NUMINAMATH_GPT_interval_intersection_l608_60820


namespace NUMINAMATH_GPT_customer_savings_l608_60816

variables (P : ℝ) (reducedPrice negotiatedPrice savings : ℝ)

-- Conditions:
def initialReduction : reducedPrice = 0.95 * P := by sorry
def finalNegotiation : negotiatedPrice = 0.90 * reducedPrice := by sorry
def savingsCalculation : savings = P - negotiatedPrice := by sorry

-- Proof problem:
theorem customer_savings : savings = 0.145 * P :=
by {
  sorry
}

end NUMINAMATH_GPT_customer_savings_l608_60816


namespace NUMINAMATH_GPT_correct_statement_about_meiosis_and_fertilization_l608_60897

def statement_A : Prop := 
  ∃ oogonia spermatogonia zygotes : ℕ, 
    oogonia = 20 ∧ spermatogonia = 8 ∧ zygotes = 32 ∧ 
    (oogonia + spermatogonia = zygotes)

def statement_B : Prop := 
  ∀ zygote_dna mother_half father_half : ℕ,
    zygote_dna = mother_half + father_half ∧ 
    mother_half = father_half

def statement_C : Prop := 
  ∀ (meiosis stabilizes : Prop) (chromosome_count : ℕ),
    (meiosis → stabilizes) ∧ 
    (stabilizes → chromosome_count = (chromosome_count / 2 + chromosome_count / 2))

def statement_D : Prop := 
  ∀ (diversity : Prop) (gene_mutations chromosomal_variations : Prop),
    (diversity → ¬ (gene_mutations ∨ chromosomal_variations))

theorem correct_statement_about_meiosis_and_fertilization :
  ¬ statement_A ∧ ¬ statement_B ∧ statement_C ∧ ¬ statement_D :=
by
  sorry

end NUMINAMATH_GPT_correct_statement_about_meiosis_and_fertilization_l608_60897


namespace NUMINAMATH_GPT_max_watched_hours_l608_60878

-- Define the duration of one episode in minutes
def episode_duration : ℕ := 30

-- Define the number of weekdays Max watched the show
def weekdays_watched : ℕ := 4

-- Define the total minutes Max watched
def total_minutes_watched : ℕ := episode_duration * weekdays_watched

-- Define the conversion factor from minutes to hours
def minutes_to_hours_factor : ℕ := 60

-- Define the total hours watched
def total_hours_watched : ℕ := total_minutes_watched / minutes_to_hours_factor

-- Proof statement
theorem max_watched_hours : total_hours_watched = 2 :=
by
  sorry

end NUMINAMATH_GPT_max_watched_hours_l608_60878


namespace NUMINAMATH_GPT_range_of_t_l608_60845

theorem range_of_t (a b t : ℝ) (h1 : a * (-1)^2 + b * (-1) + 1 / 2 = 0)
    (h2 : (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = a * x^2 + b * x + 1 / 2))
    (h3 : t = 2 * a + b) : 
    -1 < t ∧ t < 1 / 2 :=
  sorry

end NUMINAMATH_GPT_range_of_t_l608_60845


namespace NUMINAMATH_GPT_three_point_three_six_as_fraction_l608_60873

theorem three_point_three_six_as_fraction : 3.36 = (84 : ℚ) / 25 := 
by
  sorry

end NUMINAMATH_GPT_three_point_three_six_as_fraction_l608_60873


namespace NUMINAMATH_GPT_problem1_problem2_l608_60849

def f (x : ℝ) : ℝ := abs (x - 1) - abs (x + 3)

-- Proof Problem 1
theorem problem1 (x : ℝ) (h : f x > 2) : x < -2 := sorry

-- Proof Problem 2
theorem problem2 (k : ℝ) (h : ∀ x : ℝ, -3 ≤ x ∧ x ≤ -1 → f x ≤ k * x + 1) : k ≤ -1 := sorry

end NUMINAMATH_GPT_problem1_problem2_l608_60849


namespace NUMINAMATH_GPT_total_pairs_sold_l608_60835

theorem total_pairs_sold
  (H S : ℕ)
  (price_soft : ℕ := 150)
  (price_hard : ℕ := 85)
  (diff_lenses : S = H + 5)
  (total_sales_eq : price_soft * S + price_hard * H = 1455) :
  H + S = 11 := by
sorry

end NUMINAMATH_GPT_total_pairs_sold_l608_60835


namespace NUMINAMATH_GPT_new_tax_rate_l608_60871

theorem new_tax_rate
  (old_rate : ℝ) (income : ℝ) (savings : ℝ) (new_rate : ℝ)
  (h1 : old_rate = 0.46)
  (h2 : income = 36000)
  (h3 : savings = 5040)
  (h4 : new_rate = (income * old_rate - savings) / income) :
  new_rate = 0.32 :=
by {
  sorry
}

end NUMINAMATH_GPT_new_tax_rate_l608_60871


namespace NUMINAMATH_GPT_savings_account_final_amount_l608_60860

noncomputable def final_amount (P R : ℝ) (t : ℕ) : ℝ :=
  P * (1 + R) ^ t

theorem savings_account_final_amount :
  final_amount 2500 0.06 21 = 8017.84 :=
by
  sorry

end NUMINAMATH_GPT_savings_account_final_amount_l608_60860


namespace NUMINAMATH_GPT_b_finishes_remaining_work_correct_time_for_b_l608_60815

theorem b_finishes_remaining_work (a_days : ℝ) (b_days : ℝ) (work_together_days : ℝ) (remaining_work_after : ℝ) : ℝ :=
  let a_work_rate := 1 / a_days
  let b_work_rate := 1 / b_days
  let combined_work_per_day := a_work_rate + b_work_rate
  let work_done_together := combined_work_per_day * work_together_days
  let remaining_work := 1 - work_done_together
  let b_completion_time := remaining_work / b_work_rate
  b_completion_time

theorem correct_time_for_b : b_finishes_remaining_work 2 6 1 (1 - 2/3) = 2 := 
by sorry

end NUMINAMATH_GPT_b_finishes_remaining_work_correct_time_for_b_l608_60815


namespace NUMINAMATH_GPT_meters_of_cloth_l608_60803

variable (total_cost cost_per_meter : ℝ)
variable (h1 : total_cost = 434.75)
variable (h2 : cost_per_meter = 47)

theorem meters_of_cloth : 
  total_cost / cost_per_meter = 9.25 := 
by
  sorry

end NUMINAMATH_GPT_meters_of_cloth_l608_60803


namespace NUMINAMATH_GPT_triangle_sides_l608_60837
-- Import the entire library mainly used for geometry and algebraic proofs.

-- Define the main problem statement as a theorem.
theorem triangle_sides (a b c : ℕ) (r_incircle : ℕ)
  (r_excircle_a r_excircle_b r_excircle_c : ℕ) (s : ℕ)
  (area : ℕ) : 
  r_incircle = 1 → 
  area = s →
  r_excircle_a * r_excircle_b * r_excircle_c = (s * s * s) →
  s = (a + b + c) / 2 →
  r_excircle_a = s / (s - a) →
  r_excircle_b = s / (s - b) →
  r_excircle_c = s / (s - c) →
  a * b = 12 → 
  a = 3 ∧ b = 4 ∧ c = 5 :=
by {
  -- Placeholder for the proof.
  sorry
}

end NUMINAMATH_GPT_triangle_sides_l608_60837


namespace NUMINAMATH_GPT_inscribed_sphere_l608_60862

theorem inscribed_sphere (r_base height : ℝ) (r_sphere b d : ℝ)
  (h_base : r_base = 15)
  (h_height : height = 20)
  (h_sphere : r_sphere = b * Real.sqrt d - b)
  (h_rsphere_eq : r_sphere = 120 / 11) : 
  b + d = 12 := 
sorry

end NUMINAMATH_GPT_inscribed_sphere_l608_60862


namespace NUMINAMATH_GPT_solution_set_for_inequality_l608_60866

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + (a - b) * x + 1

theorem solution_set_for_inequality (a b : ℝ) (h1 : 2*a + 4 = -(a-1)) :
  ∀ x : ℝ, (f x a b > f b a b) ↔ ((x ∈ Set.Icc (-2 : ℝ) (2 : ℝ)) ∧ ((x < -1 ∨ 1 < x))) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_for_inequality_l608_60866


namespace NUMINAMATH_GPT_clive_can_correct_time_l608_60865

def can_show_correct_time (hour_hand_angle minute_hand_angle : ℝ) :=
  ∃ θ : ℝ, θ ∈ [0, 360] ∧ hour_hand_angle + θ % 360 = minute_hand_angle + θ % 360

theorem clive_can_correct_time (hour_hand_angle minute_hand_angle : ℝ) :
  can_show_correct_time hour_hand_angle minute_hand_angle :=
sorry

end NUMINAMATH_GPT_clive_can_correct_time_l608_60865


namespace NUMINAMATH_GPT_molecular_weight_of_compound_l608_60869

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00
def num_C : ℕ := 4
def num_H : ℕ := 1
def num_O : ℕ := 1

theorem molecular_weight_of_compound : 
  (num_C * atomic_weight_C + num_H * atomic_weight_H + num_O * atomic_weight_O) = 65.048 := 
  by 
  -- proof skipped
  sorry

end NUMINAMATH_GPT_molecular_weight_of_compound_l608_60869


namespace NUMINAMATH_GPT_math_problem_proof_l608_60843

-- Define the base conversion functions
def base11_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 2471 => 1 * 11^0 + 7 * 11^1 + 4 * 11^2 + 2 * 11^3
  | _    => 0

def base5_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 121 => 1 * 5^0 + 2 * 5^1 + 1 * 5^2
  | _   => 0

def base7_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 3654 => 4 * 7^0 + 5 * 7^1 + 6 * 7^2 + 3 * 7^3
  | _    => 0

def base8_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 5680 => 0 * 8^0 + 8 * 8^1 + 6 * 8^2 + 5 * 8^3
  | _    => 0

theorem math_problem_proof :
  let x := base11_to_base10 2471
  let y := base5_to_base10 121
  let z := base7_to_base10 3654
  let w := base8_to_base10 5680
  x / y - z + w = 1736 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_proof_l608_60843


namespace NUMINAMATH_GPT_campers_afternoon_l608_60834

theorem campers_afternoon (total_campers morning_campers afternoon_campers : ℕ)
  (h1 : total_campers = 60)
  (h2 : morning_campers = 53)
  (h3 : afternoon_campers = total_campers - morning_campers) :
  afternoon_campers = 7 := by
  sorry

end NUMINAMATH_GPT_campers_afternoon_l608_60834


namespace NUMINAMATH_GPT_smallest_positive_integer_rel_prime_180_l608_60823

theorem smallest_positive_integer_rel_prime_180 : 
  ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 180 = 1 → y ≥ 7 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_rel_prime_180_l608_60823


namespace NUMINAMATH_GPT_price_of_necklace_l608_60817

-- Define the necessary conditions.
def num_charms_per_necklace : ℕ := 10
def cost_per_charm : ℕ := 15
def num_necklaces_sold : ℕ := 30
def total_profit : ℕ := 1500

-- Calculation of selling price per necklace
def cost_per_necklace := num_charms_per_necklace * cost_per_charm
def total_cost := cost_per_necklace * num_necklaces_sold
def total_revenue := total_cost + total_profit
def selling_price_per_necklace := total_revenue / num_necklaces_sold

-- Statement of the problem in Lean 4
theorem price_of_necklace : selling_price_per_necklace = 200 := by
  sorry

end NUMINAMATH_GPT_price_of_necklace_l608_60817


namespace NUMINAMATH_GPT_walkways_area_l608_60896

-- Define the conditions and prove the total walkway area is 416 square feet
theorem walkways_area (rows : ℕ) (columns : ℕ) (bed_width : ℝ) (bed_height : ℝ) (walkway_width : ℝ) 
  (h_rows : rows = 4) (h_columns : columns = 3) (h_bed_width : bed_width = 8) (h_bed_height : bed_height = 3) (h_walkway_width : walkway_width = 2) : 
  (rows * (bed_height + walkway_width) + walkway_width) * (columns * (bed_width + walkway_width) + walkway_width) - rows * columns * bed_width * bed_height = 416 := 
by 
  sorry

end NUMINAMATH_GPT_walkways_area_l608_60896


namespace NUMINAMATH_GPT_charlotte_overall_score_l608_60812

theorem charlotte_overall_score :
  (0.60 * 15 + 0.75 * 20 + 0.85 * 25).round / 60 = 0.75 :=
by
  sorry

end NUMINAMATH_GPT_charlotte_overall_score_l608_60812


namespace NUMINAMATH_GPT_line_has_equal_intercepts_find_a_l608_60807

theorem line_has_equal_intercepts (a : ℝ) :
  (∃ l : ℝ, (l = 0 → ax + y - 2 - a = 0) ∧ (l = 1 → (a = 1 ∨ a = -2))) := sorry

-- formalizing the problem
theorem find_a (a : ℝ) (h_eq_intercepts : ∀ x y : ℝ, (a * x + y - 2 - a = 0 ↔ (x = 2 + a ∧ y = -2 - a))) :
  a = 1 ∨ a = -2 := sorry

end NUMINAMATH_GPT_line_has_equal_intercepts_find_a_l608_60807


namespace NUMINAMATH_GPT_six_digit_number_theorem_l608_60890

noncomputable def six_digit_number (a b c d e f : ℕ) : ℕ :=
  10^5 * a + 10^4 * b + 10^3 * c + 10^2 * d + 10 * e + f

noncomputable def rearranged_number (a b c d e f : ℕ) : ℕ :=
  10^5 * b + 10^4 * c + 10^3 * d + 10^2 * e + 10 * f + a

theorem six_digit_number_theorem (a b c d e f : ℕ) (h_a : a ≠ 0) 
  (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) 
  (h4 : 0 ≤ d ∧ d ≤ 9) (h5 : 0 ≤ e ∧ e ≤ 9) (h6 : 0 ≤ f ∧ f ≤ 9) 
  : six_digit_number a b c d e f = 142857 ∨ six_digit_number a b c d e f = 285714 :=
by
  sorry

end NUMINAMATH_GPT_six_digit_number_theorem_l608_60890


namespace NUMINAMATH_GPT_nonzero_roots_ratio_l608_60832

theorem nonzero_roots_ratio (m : ℝ) (h : m ≠ 0) :
  (∃ (r s : ℝ), r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 ∧ r + s = 4 ∧ r * s = m) → m = 3 :=
by 
  intro h_exists
  obtain ⟨r, s, hr_ne_zero, hs_ne_zero, h_ratio, h_sum, h_prod⟩ := h_exists
  sorry

end NUMINAMATH_GPT_nonzero_roots_ratio_l608_60832


namespace NUMINAMATH_GPT_problem_inequality_l608_60808

theorem problem_inequality (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
    (y + z) / (2 * x) + (z + x) / (2 * y) + (x + y) / (2 * z) ≥
    2 * x / (y + z) + 2 * y / (z + x) + 2 * z / (x + y) :=
by
  sorry

end NUMINAMATH_GPT_problem_inequality_l608_60808


namespace NUMINAMATH_GPT_find_ab_range_m_l608_60826

-- Part 1
theorem find_ab (a b: ℝ) (h1 : 3 - 6 * a + b = 0) (h2 : -1 + 3 * a - b + a^2 = 0) :
  a = 2 ∧ b = 9 := 
sorry

-- Part 2
theorem range_m (m: ℝ) (h: ∀ x ∈ (Set.Icc (-2) 1), x^3 + 3 * 2 * x^2 + 9 * x + 4 - m ≤ 0) :
  20 ≤ m :=
sorry

end NUMINAMATH_GPT_find_ab_range_m_l608_60826


namespace NUMINAMATH_GPT_john_running_time_l608_60800

theorem john_running_time
  (x : ℚ)
  (h1 : 15 * x + 10 * (9 - x) = 100)
  (h2 : 0 ≤ x)
  (h3 : x ≤ 9) :
  x = 2 := by
  sorry

end NUMINAMATH_GPT_john_running_time_l608_60800


namespace NUMINAMATH_GPT_functional_eq_solution_l608_60874

-- Define the conditions
variables (f g : ℕ → ℕ)

-- Define the main theorem
theorem functional_eq_solution :
  (∀ n : ℕ, f n + f (n + g n) = f (n + 1)) →
  ( (∀ n, f n = 0) ∨ 
    (∃ (n₀ c : ℕ), 
      (∀ n < n₀, f n = 0) ∧ 
      (∀ n ≥ n₀, f n = c * 2^(n - n₀)) ∧
      (∀ n < n₀ - 1, ∃ ck : ℕ, g n = ck) ∧
      g (n₀ - 1) = 1 ∧
      ∀ n ≥ n₀, g n = 0 ) ) := 
by
  intro h
  /- Proof goes here -/
  sorry

end NUMINAMATH_GPT_functional_eq_solution_l608_60874


namespace NUMINAMATH_GPT_sarah_shirts_l608_60828

theorem sarah_shirts (loads : ℕ) (pieces_per_load : ℕ) (sweaters : ℕ) 
  (total_pieces : ℕ) (shirts : ℕ) : 
  loads = 9 → pieces_per_load = 5 → sweaters = 2 →
  total_pieces = loads * pieces_per_load → shirts = total_pieces - sweaters → 
  shirts = 43 :=
by
  intros h_loads h_pieces_per_load h_sweaters h_total_pieces h_shirts
  sorry

end NUMINAMATH_GPT_sarah_shirts_l608_60828


namespace NUMINAMATH_GPT_range_of_m_l608_60881

variable (m : ℝ)

def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 3 * m - 1 }
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 10}

theorem range_of_m (h : A m ∪ B = B) : m ≤ 11 / 3 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l608_60881


namespace NUMINAMATH_GPT_sugar_solution_sweeter_l608_60841

theorem sugar_solution_sweeter (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) : 
    (b + m) / (a + m) > b / a :=
sorry

end NUMINAMATH_GPT_sugar_solution_sweeter_l608_60841


namespace NUMINAMATH_GPT_ratio_of_doctors_to_nurses_l608_60830

theorem ratio_of_doctors_to_nurses (total_staff doctors nurses : ℕ) (h1 : total_staff = 456) (h2 : nurses = 264) (h3 : doctors + nurses = total_staff) :
  doctors = 192 ∧ (doctors : ℚ) / nurses = 8 / 11 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_doctors_to_nurses_l608_60830


namespace NUMINAMATH_GPT_sum_of_roots_of_quadratic_l608_60892

theorem sum_of_roots_of_quadratic :
  let a := 2
  let b := -8
  let c := 6
  let sum_of_roots := (-b / a)
  2 * (sum_of_roots) * sum_of_roots - 8 * sum_of_roots + 6 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_of_quadratic_l608_60892


namespace NUMINAMATH_GPT_total_volume_of_four_cubes_is_500_l608_60839

-- Definitions for the problem assumptions
def edge_length := 5
def volume_of_cube (edge_length : ℕ) := edge_length ^ 3
def number_of_boxes := 4

-- Main statement to prove
theorem total_volume_of_four_cubes_is_500 :
  (volume_of_cube edge_length) * number_of_boxes = 500 :=
by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_total_volume_of_four_cubes_is_500_l608_60839


namespace NUMINAMATH_GPT_james_missing_legos_l608_60846

theorem james_missing_legos  (h1 : 500 > 0) (h2 : 500 % 2 = 0) (h3 : 245 < 500)  :
  let total_legos := 500
  let used_legos := total_legos / 2
  let leftover_legos := total_legos - used_legos
  let legos_in_box := 245
  leftover_legos - legos_in_box = 5 := by
{
  sorry
}

end NUMINAMATH_GPT_james_missing_legos_l608_60846


namespace NUMINAMATH_GPT_volume_of_one_pizza_piece_l608_60848

theorem volume_of_one_pizza_piece
  (h : ℝ) (d : ℝ) (n : ℕ)
  (h_eq : h = 1 / 2)
  (d_eq : d = 16)
  (n_eq : n = 16) :
  ((π * (d / 2)^2 * h) / n) = 2 * π :=
by
  rw [h_eq, d_eq, n_eq]
  sorry

end NUMINAMATH_GPT_volume_of_one_pizza_piece_l608_60848


namespace NUMINAMATH_GPT_evaluate_expression_l608_60805

theorem evaluate_expression (a b c : ℚ) 
  (h1 : c = b - 11) 
  (h2 : b = a + 3) 
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0) 
  (h5 : b - 3 ≠ 0) 
  (h6 : c + 7 ≠ 0) : 
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 9) / (c + 7) = 10 / 7 := 
sorry

end NUMINAMATH_GPT_evaluate_expression_l608_60805


namespace NUMINAMATH_GPT_second_quadratic_roots_complex_iff_first_roots_real_distinct_l608_60818

theorem second_quadratic_roots_complex_iff_first_roots_real_distinct (q : ℝ) :
  q < 1 → (∀ x : ℂ, (3 - q) * x^2 + 2 * (1 + q) * x + (q^2 - q + 2) ≠ 0) :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_second_quadratic_roots_complex_iff_first_roots_real_distinct_l608_60818


namespace NUMINAMATH_GPT_hurricane_damage_in_GBP_l608_60829

def damage_in_AUD : ℤ := 45000000
def conversion_rate : ℚ := 1 / 2 -- 1 AUD = 1/2 GBP

theorem hurricane_damage_in_GBP : 
  (damage_in_AUD : ℚ) * conversion_rate = 22500000 := 
by
  sorry

end NUMINAMATH_GPT_hurricane_damage_in_GBP_l608_60829


namespace NUMINAMATH_GPT_tan_alpha_in_third_quadrant_l608_60858

theorem tan_alpha_in_third_quadrant (α : Real) (h1 : Real.sin α = -5/13) (h2 : ∃ k : ℕ, π < α + k * 2 * π ∧ α + k * 2 * π < 3 * π) : 
  Real.tan α = 5/12 :=
sorry

end NUMINAMATH_GPT_tan_alpha_in_third_quadrant_l608_60858


namespace NUMINAMATH_GPT_sum_of_possible_values_d_l608_60804

theorem sum_of_possible_values_d :
  let range_8 := (512, 4095)
  let digits_in_base_16 := 3
  (∀ n, n ∈ Set.Icc range_8.1 range_8.2 → (Nat.digits 16 n).length = digits_in_base_16)
  → digits_in_base_16 = 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_d_l608_60804


namespace NUMINAMATH_GPT_quadrilateral_side_difference_l608_60840

variable (a b c d : ℝ)

theorem quadrilateral_side_difference :
  a + b + c + d = 120 →
  a + c = 50 →
  (a^2 + c^2 = 1600) →
  (b + d = 70 ∧ b * d = 450) →
  |b - d| = 2 * Real.sqrt 775 :=
by
  intros ha hb hc hd
  sorry

end NUMINAMATH_GPT_quadrilateral_side_difference_l608_60840


namespace NUMINAMATH_GPT_amy_total_distance_equals_168_l608_60886

def amy_biked_monday := 12

def amy_biked_tuesday (monday: ℕ) := 2 * monday - 3

def amy_biked_other_day (previous_day: ℕ) := previous_day + 2

def total_distance_bike_week := 
  let monday := amy_biked_monday
  let tuesday := amy_biked_tuesday monday
  let wednesday := amy_biked_other_day tuesday
  let thursday := amy_biked_other_day wednesday
  let friday := amy_biked_other_day thursday
  let saturday := amy_biked_other_day friday
  let sunday := amy_biked_other_day saturday
  monday + tuesday + wednesday + thursday + friday + saturday + sunday

theorem amy_total_distance_equals_168 : 
  total_distance_bike_week = 168 := by
  sorry

end NUMINAMATH_GPT_amy_total_distance_equals_168_l608_60886


namespace NUMINAMATH_GPT_range_of_x_l608_60850

theorem range_of_x (x y : ℝ) (h : 4 * x * y + 4 * y^2 + x + 6 = 0) : x ≤ -2 ∨ x ≥ 3 :=
sorry

end NUMINAMATH_GPT_range_of_x_l608_60850


namespace NUMINAMATH_GPT_rhombus_diagonal_l608_60856

theorem rhombus_diagonal (d1 d2 : ℝ) (area_tri : ℝ) (h1 : d1 = 15) (h2 : area_tri = 75) :
  (d1 * d2) / 2 = 2 * area_tri → d2 = 20 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_l608_60856


namespace NUMINAMATH_GPT_ratio_of_ages_is_six_l608_60893

-- Definitions of ages
def Cody_age : ℕ := 14
def Grandmother_age : ℕ := 84

-- The ratio we want to prove
def age_ratio : ℕ := Grandmother_age / Cody_age

-- The theorem stating the ratio is 6
theorem ratio_of_ages_is_six : age_ratio = 6 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_ratio_of_ages_is_six_l608_60893


namespace NUMINAMATH_GPT_max_S_R_squared_l608_60891

theorem max_S_R_squared (a b c : ℝ) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) :
  (∃ a b c, DA = a ∧ DB = b ∧ DC = c ∧ S = 2 * (a * b + b * c + c * a) ∧
  R = (Real.sqrt (a^2 + b^2 + c^2)) / 2 ∧ (∃ max_val, max_val = (2 / 3) * (3 + Real.sqrt 3))) :=
sorry

end NUMINAMATH_GPT_max_S_R_squared_l608_60891


namespace NUMINAMATH_GPT_contrapositive_a_eq_b_imp_a_sq_eq_b_sq_l608_60811

theorem contrapositive_a_eq_b_imp_a_sq_eq_b_sq (a b : ℝ) :
  (a = b → a^2 = b^2) ↔ (a^2 ≠ b^2 → a ≠ b) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_a_eq_b_imp_a_sq_eq_b_sq_l608_60811


namespace NUMINAMATH_GPT_melanie_correct_coins_and_value_l608_60809

def melanie_coins_problem : Prop :=
let dimes_initial := 19
let dimes_dad := 39
let dimes_sister := 15
let dimes_mother := 25
let total_dimes := dimes_initial + dimes_dad + dimes_sister + dimes_mother

let nickels_initial := 12
let nickels_dad := 22
let nickels_sister := 7
let nickels_mother := 10
let nickels_grandmother := 30
let total_nickels := nickels_initial + nickels_dad + nickels_sister + nickels_mother + nickels_grandmother

let quarters_initial := 8
let quarters_dad := 15
let quarters_sister := 12
let quarters_grandmother := 3
let total_quarters := quarters_initial + quarters_dad + quarters_sister + quarters_grandmother

let dimes_value := total_dimes * 0.10
let nickels_value := total_nickels * 0.05
let quarters_value := total_quarters * 0.25
let total_value := dimes_value + nickels_value + quarters_value

total_dimes = 98 ∧ total_nickels = 81 ∧ total_quarters = 38 ∧ total_value = 23.35

theorem melanie_correct_coins_and_value : melanie_coins_problem :=
by sorry

end NUMINAMATH_GPT_melanie_correct_coins_and_value_l608_60809


namespace NUMINAMATH_GPT_money_problem_l608_60885

-- Define the conditions and the required proof
theorem money_problem (B S : ℕ) 
  (h1 : B = 2 * S) -- Condition 1: Brother brought twice as much money as the sister
  (h2 : B - 180 = S - 30) -- Condition 3: Remaining money of brother and sister are equal
  : B = 300 ∧ S = 150 := -- Correct answer to prove
  
sorry -- Placeholder for proof

end NUMINAMATH_GPT_money_problem_l608_60885


namespace NUMINAMATH_GPT_jess_height_l608_60810

variable (Jana_height Kelly_height Jess_height : ℕ)

-- Conditions
axiom Jana_height_eq : Jana_height = 74
axiom Jana_taller_than_Kelly : Jana_height = Kelly_height + 5
axiom Kelly_shorter_than_Jess : Kelly_height = Jess_height - 3

-- Prove Jess's height
theorem jess_height : Jess_height = 72 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_jess_height_l608_60810


namespace NUMINAMATH_GPT_speed_of_stream_l608_60801

theorem speed_of_stream (v : ℝ) (d : ℝ) :
  (∀ d : ℝ, d > 0 → (1 / (6 - v) = 2 * (1 / (6 + v)))) → v = 2 := by
  sorry

end NUMINAMATH_GPT_speed_of_stream_l608_60801


namespace NUMINAMATH_GPT_robin_photo_count_l608_60802

theorem robin_photo_count (photos_per_page : ℕ) (full_pages : ℕ) 
  (h1 : photos_per_page = 6) (h2 : full_pages = 122) :
  photos_per_page * full_pages = 732 :=
by
  sorry

end NUMINAMATH_GPT_robin_photo_count_l608_60802


namespace NUMINAMATH_GPT_blueberry_jelly_amount_l608_60819

theorem blueberry_jelly_amount (total_jelly : ℕ) (strawberry_jelly : ℕ) 
  (h_total : total_jelly = 6310) 
  (h_strawberry : strawberry_jelly = 1792) 
  : total_jelly - strawberry_jelly = 4518 := 
by 
  sorry

end NUMINAMATH_GPT_blueberry_jelly_amount_l608_60819


namespace NUMINAMATH_GPT_find_train_probability_l608_60879

-- Define the time range and parameters
def start_time : ℕ := 120
def end_time : ℕ := 240
def wait_time : ℕ := 30

-- Define the conditions
def is_in_range (t : ℕ) : Prop := start_time ≤ t ∧ t ≤ end_time

-- Define the probability function
def probability_of_finding_train : ℚ :=
  let area_triangle : ℚ := (1 / 2) * 30 * 30
  let area_parallelogram : ℚ := 90 * 30
  let shaded_area : ℚ := area_triangle + area_parallelogram
  let total_area : ℚ := (end_time - start_time) * (end_time - start_time)
  shaded_area / total_area

-- The theorem to prove
theorem find_train_probability :
  probability_of_finding_train = 7 / 32 :=
by
  sorry

end NUMINAMATH_GPT_find_train_probability_l608_60879


namespace NUMINAMATH_GPT_sandy_total_earnings_l608_60825

-- Define the conditions
def hourly_wage : ℕ := 15
def hours_friday : ℕ := 10
def hours_saturday : ℕ := 6
def hours_sunday : ℕ := 14

-- Define the total hours worked and total earnings
def total_hours := hours_friday + hours_saturday + hours_sunday
def total_earnings := total_hours * hourly_wage

-- State the theorem
theorem sandy_total_earnings : total_earnings = 450 := by
  sorry

end NUMINAMATH_GPT_sandy_total_earnings_l608_60825


namespace NUMINAMATH_GPT_find_n_l608_60868

/-- In the expansion of (1 + 3x)^n, where n is a positive integer and n >= 6, 
    if the coefficients of x^5 and x^6 are equal, then n is 7. -/
theorem find_n (n : ℕ) (h₀ : 0 < n) (h₁ : 6 ≤ n)
  (h₂ : 3^5 * Nat.choose n 5 = 3^6 * Nat.choose n 6) : 
  n = 7 := 
sorry

end NUMINAMATH_GPT_find_n_l608_60868


namespace NUMINAMATH_GPT_nina_widgets_after_reduction_is_approx_8_l608_60898

noncomputable def nina_total_money : ℝ := 16.67
noncomputable def widgets_before_reduction : ℝ := 5
noncomputable def cost_reduction_per_widget : ℝ := 1.25

noncomputable def cost_per_widget_before_reduction : ℝ := nina_total_money / widgets_before_reduction
noncomputable def cost_per_widget_after_reduction : ℝ := cost_per_widget_before_reduction - cost_reduction_per_widget
noncomputable def widgets_after_reduction : ℝ := nina_total_money / cost_per_widget_after_reduction

-- Prove that Nina can purchase approximately 8 widgets after the cost reduction
theorem nina_widgets_after_reduction_is_approx_8 : abs (widgets_after_reduction - 8) < 1 :=
by
  sorry

end NUMINAMATH_GPT_nina_widgets_after_reduction_is_approx_8_l608_60898


namespace NUMINAMATH_GPT_diminished_radius_10_percent_l608_60854

theorem diminished_radius_10_percent
  (r r' : ℝ) 
  (h₁ : r > 0)
  (h₂ : r' > 0)
  (h₃ : (π * r'^2) = 0.8100000000000001 * (π * r^2)) :
  r' = 0.9 * r :=
by sorry

end NUMINAMATH_GPT_diminished_radius_10_percent_l608_60854


namespace NUMINAMATH_GPT_line_intersects_x_axis_at_point_l608_60813

theorem line_intersects_x_axis_at_point :
  (∃ x, 5 * 0 - 2 * x = 10) ↔ (x = -5) ∧ (∃ x, 5 * y - 2 * x = 10 ∧ y = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_x_axis_at_point_l608_60813


namespace NUMINAMATH_GPT_amount_y_gets_each_rupee_x_gets_l608_60875

-- Given conditions
variables (x y z a : ℝ)
variables (h_y_share : y = 36) (h_total : x + y + z = 156) (h_z : z = 0.50 * x)

-- Proof problem
theorem amount_y_gets_each_rupee_x_gets (h : 36 / x = a) : a = 9 / 20 :=
by {
  -- The proof is omitted and replaced with 'sorry'.
  sorry
}

end NUMINAMATH_GPT_amount_y_gets_each_rupee_x_gets_l608_60875


namespace NUMINAMATH_GPT_number_of_ordered_triples_l608_60855

theorem number_of_ordered_triples (a b c : ℤ) : 
  ∃ (n : ℕ), -31 <= a ∧ a <= 31 ∧ -31 <= b ∧ b <= 31 ∧ -31 <= c ∧ c <= 31 ∧ 
  (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (a + b + c > 0) ∧ n = 117690 :=
by sorry

end NUMINAMATH_GPT_number_of_ordered_triples_l608_60855


namespace NUMINAMATH_GPT_union_setA_setB_l608_60859

noncomputable def setA : Set ℝ := { x : ℝ | 2 / (x + 1) ≥ 1 }
noncomputable def setB : Set ℝ := { y : ℝ | ∃ x : ℝ, y = 2^x ∧ x < 0 }

theorem union_setA_setB : setA ∪ setB = { x : ℝ | -1 < x ∧ x ≤ 1 } :=
by
  sorry

end NUMINAMATH_GPT_union_setA_setB_l608_60859


namespace NUMINAMATH_GPT_number_of_cows_l608_60842

-- Define the total number of legs and number of legs per cow
def total_legs : ℕ := 460
def legs_per_cow : ℕ := 4

-- Mathematical proof problem as a Lean 4 statement
theorem number_of_cows : total_legs / legs_per_cow = 115 := by
  -- This is the proof statement place. We use 'sorry' as a placeholder for the actual proof.
  sorry

end NUMINAMATH_GPT_number_of_cows_l608_60842


namespace NUMINAMATH_GPT_all_selected_prob_l608_60894

def probability_of_selection (P_ram P_ravi P_raj : ℚ) : ℚ :=
  P_ram * P_ravi * P_raj

theorem all_selected_prob :
  let P_ram := 2/7
  let P_ravi := 1/5
  let P_raj := 3/8
  probability_of_selection P_ram P_ravi P_raj = 3/140 := by
  sorry

end NUMINAMATH_GPT_all_selected_prob_l608_60894


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l608_60814

def pair_otimes (a b c d : ℚ) : ℚ := b * c - a * d

-- Problem (1)
theorem problem_1 : pair_otimes 5 3 (-2) 1 = -11 := 
by 
  unfold pair_otimes 
  sorry

-- Problem (2)
theorem problem_2 (x : ℚ) (h : pair_otimes 2 (3 * x - 1) 6 (x + 2) = 22) : x = 2 := 
by 
  unfold pair_otimes at h
  sorry

-- Problem (3)
theorem problem_3 (x k : ℤ) (h : pair_otimes 4 (k - 2) x (2 * x - 1) = 6) : 
  k = 8 ∨ k = 9 ∨ k = 11 ∨ k = 12 := 
by 
  unfold pair_otimes at h
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l608_60814


namespace NUMINAMATH_GPT_probability_of_sum_8_9_10_l608_60806

/-- The list of face values for the first die. -/
def first_die : List ℕ := [1, 1, 3, 3, 5, 6]

/-- The list of face values for the second die. -/
def second_die : List ℕ := [1, 2, 4, 5, 7, 9]

/-- The condition to verify if the sum is 8, 9, or 10. -/
def valid_sum (s : ℕ) : Bool := s = 8 ∨ s = 9 ∨ s = 10

/-- Calculate probability of the sum being 8, 9, or 10 for the two dice. -/
def calculate_probability : ℚ :=
  let total_rolls := first_die.length * second_die.length
  let valid_rolls := 
    first_die.foldl (fun acc d1 =>
      acc + second_die.foldl (fun acc' d2 => 
        if valid_sum (d1 + d2) then acc' + 1 else acc') 0) 0
  valid_rolls / total_rolls

/-- The required probability is 7/18. -/
theorem probability_of_sum_8_9_10 : calculate_probability = 7 / 18 := 
  sorry

end NUMINAMATH_GPT_probability_of_sum_8_9_10_l608_60806
