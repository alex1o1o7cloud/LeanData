import Mathlib

namespace NUMINAMATH_GPT_smallest_rel_prime_greater_than_one_l81_8129

theorem smallest_rel_prime_greater_than_one (n : ℕ) (h : n > 1) (h0: ∀ (m : ℕ), m > 1 ∧ Nat.gcd m 2100 = 1 → 11 ≤ m):
  Nat.gcd n 2100 = 1 → n = 11 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_smallest_rel_prime_greater_than_one_l81_8129


namespace NUMINAMATH_GPT_positional_relationship_l81_8184

theorem positional_relationship 
  (m n : ℝ) 
  (h_points_on_ellipse : (m^2 / 4) + (n^2 / 3) = 1)
  (h_relation : n^2 = 3 - (3/4) * m^2) : 
  (∃ x y : ℝ, (x^2 + y^2 = 1/3) ∧ (m * x + n * y + 1 = 0)) ∨ 
  (∀ x y : ℝ, (x^2 + y^2 = 1/3) → (m * x + n * y + 1 ≠ 0)) :=
sorry

end NUMINAMATH_GPT_positional_relationship_l81_8184


namespace NUMINAMATH_GPT_evaluate_absolute_value_l81_8160

theorem evaluate_absolute_value (π : ℝ) (h : π < 5.5) : |5.5 - π| = 5.5 - π :=
by
  sorry

end NUMINAMATH_GPT_evaluate_absolute_value_l81_8160


namespace NUMINAMATH_GPT_fraction_members_absent_l81_8104

variable (p : ℕ) -- Number of persons in the office
variable (W : ℝ) -- Total work amount
variable (x : ℝ) -- Fraction of members absent

theorem fraction_members_absent (h : W / (p * (1 - x)) = W / p + W / (6 * p)) : x = 1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_fraction_members_absent_l81_8104


namespace NUMINAMATH_GPT_usual_time_to_cover_distance_l81_8173

theorem usual_time_to_cover_distance (S T : ℝ) (h1 : 0.75 * S = S / (T + 24)) (h2 : S * T = 0.75 * S * (T + 24)) : T = 72 :=
by
  sorry

end NUMINAMATH_GPT_usual_time_to_cover_distance_l81_8173


namespace NUMINAMATH_GPT_banana_orange_equivalence_l81_8119

/-- Given that 3/4 of 12 bananas are worth 9 oranges,
    prove that 1/3 of 9 bananas are worth 3 oranges. -/
theorem banana_orange_equivalence :
  (3 / 4) * 12 = 9 → (1 / 3) * 9 = 3 :=
by
  intro h
  have h1 : (9 : ℝ) = 9 := by sorry -- This is from the provided condition
  have h2 : 1 * 9 = 1 * 9 := by sorry -- Deducing from h1: 9 = 9
  have h3 : 9 = 9 := by sorry -- concluding 9 bananas = 9 oranges
  have h4 : (1 / 3) * 9 = 3 := by sorry -- 1/3 of 9
  exact h4

end NUMINAMATH_GPT_banana_orange_equivalence_l81_8119


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l81_8118

theorem hyperbola_eccentricity :
  ∀ (a b c : ℝ), a = 3 → b = 4 → c = Real.sqrt (a^2 + b^2) → c / a = 5 / 3 :=
by
  intros a b c ha hb h_eq
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l81_8118


namespace NUMINAMATH_GPT_Dima_broke_more_l81_8154

theorem Dima_broke_more (D F : ℕ) (h : 2 * D + 7 * F = 3 * (D + F)) : D = 4 * F :=
sorry

end NUMINAMATH_GPT_Dima_broke_more_l81_8154


namespace NUMINAMATH_GPT_part_a_part_b_l81_8113

variable (p : ℕ)
variable (h1 : prime p)
variable (h2 : p > 3)

theorem part_a : (p + 1) % 4 = 0 ∨ (p - 1) % 4 = 0 :=
sorry

theorem part_b : ¬ ((p + 1) % 5 = 0 ∨ (p - 1) % 5 = 0) :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l81_8113


namespace NUMINAMATH_GPT_words_memorized_on_fourth_day_l81_8100

-- Definitions for the conditions
def first_three_days_words (k : ℕ) : ℕ := 3 * k
def last_four_days_words (k : ℕ) : ℕ := 4 * k
def fourth_day_words (k : ℕ) (a : ℕ) : ℕ := a
def last_three_days_words (k : ℕ) (a : ℕ) : ℕ := last_four_days_words k - a

-- Problem Statement
theorem words_memorized_on_fourth_day {k a : ℕ} (h1 : first_three_days_words k + last_four_days_words k > 100)
    (h2 : first_three_days_words k * 6 = 5 * (4 * k - a))
    (h3 : 21 * (2 * k / 3) = 100) : 
    a = 10 :=
by 
  sorry

end NUMINAMATH_GPT_words_memorized_on_fourth_day_l81_8100


namespace NUMINAMATH_GPT_cost_per_mile_l81_8135

def miles_per_week : ℕ := 3 * 50 + 4 * 100
def weeks_per_year : ℕ := 52
def miles_per_year : ℕ := miles_per_week * weeks_per_year
def weekly_fee : ℕ := 100
def yearly_total_fee : ℕ := 7800
def yearly_weekly_fees : ℕ := 52 * weekly_fee
def yearly_mile_fees := yearly_total_fee - yearly_weekly_fees
def pay_per_mile := yearly_mile_fees / miles_per_year

theorem cost_per_mile : pay_per_mile = 909 / 10000 := by
  -- proof will be added here
  sorry

end NUMINAMATH_GPT_cost_per_mile_l81_8135


namespace NUMINAMATH_GPT_symmetric_point_correct_l81_8185

-- Define the point P in a three-dimensional Cartesian coordinate system.
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the function to find the symmetric point with respect to the x-axis.
def symmetricWithRespectToXAxis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

-- Given point P(1, -2, 3).
def P : Point3D := { x := 1, y := -2, z := 3 }

-- The expected symmetric point
def symmetricP : Point3D := { x := 1, y := 2, z := -3 }

-- The proposition we need to prove
theorem symmetric_point_correct :
  symmetricWithRespectToXAxis P = symmetricP :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_correct_l81_8185


namespace NUMINAMATH_GPT_calculate_f_value_l81_8131

def f (x y : ℚ) : ℚ := x - y * ⌈x / y⌉

theorem calculate_f_value :
  f (1/3) (-3/7) = -2/21 := by
  sorry

end NUMINAMATH_GPT_calculate_f_value_l81_8131


namespace NUMINAMATH_GPT_hypotenuse_length_triangle_l81_8137

theorem hypotenuse_length_triangle (a b c : ℝ) (h1 : a + b + c = 40) (h2 : (1/2) * a * b = 30) 
  (h3 : a = b) : c = 2 * Real.sqrt 30 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_triangle_l81_8137


namespace NUMINAMATH_GPT_original_price_l81_8152

noncomputable def original_selling_price (CP : ℝ) : ℝ := CP * 1.25
noncomputable def selling_price_at_loss (CP : ℝ) : ℝ := CP * 0.5

theorem original_price (CP : ℝ) (h : selling_price_at_loss CP = 320) : original_selling_price CP = 800 :=
by
  sorry

end NUMINAMATH_GPT_original_price_l81_8152


namespace NUMINAMATH_GPT_range_m_l81_8197

namespace MathProof

noncomputable def f (x m : ℝ) : ℝ := x^3 - 3 * x + 2 + m

theorem range_m
  (m : ℝ)
  (h : m > 0)
  (a b c : ℝ)
  (ha : 0 ≤ a ∧ a ≤ 2)
  (hb : 0 ≤ b ∧ b ≤ 2)
  (hc : 0 ≤ c ∧ c ≤ 2)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (h_triangle : f a m ^ 2 + f b m ^ 2 = f c m ^ 2 ∨
                f a m ^ 2 + f c m ^ 2 = f b m ^ 2 ∨
                f b m ^ 2 + f c m ^ 2 = f a m ^ 2) :
  0 < m ∧ m < 3 + 4 * Real.sqrt 2 :=
by
  sorry

end MathProof

end NUMINAMATH_GPT_range_m_l81_8197


namespace NUMINAMATH_GPT_find_n_l81_8130

theorem find_n (n : ℕ) (hn : (n - 2) * (n - 3) / 12 = 14 / 3) : n = 10 := by
  sorry

end NUMINAMATH_GPT_find_n_l81_8130


namespace NUMINAMATH_GPT_event_B_C_mutually_exclusive_l81_8121

-- Define the events based on the given conditions
def EventA (products : Type) (is_defective : products → Prop) (x y : products) : Prop :=
  ¬is_defective x ∧ ¬is_defective y

def EventB (products : Type) (is_defective : products → Prop) (x y : products) : Prop :=
  is_defective x ∧ is_defective y

def EventC (products : Type) (is_defective : products → Prop) (x y : products) : Prop :=
  ¬(is_defective x ∧ is_defective y)

-- Prove that Event B and Event C are mutually exclusive
theorem event_B_C_mutually_exclusive (products : Type) (is_defective : products → Prop) (x y : products) :
  (EventB products is_defective x y) → ¬(EventC products is_defective x y) :=
sorry

end NUMINAMATH_GPT_event_B_C_mutually_exclusive_l81_8121


namespace NUMINAMATH_GPT_total_cows_l81_8161

variable (D C : ℕ)

-- The conditions of the problem translated to Lean definitions
def total_heads := D + C
def total_legs := 2 * D + 4 * C 

-- The main theorem based on the conditions and the result to prove
theorem total_cows (h1 : total_legs D C = 2 * total_heads D C + 40) : C = 20 :=
by
  sorry


end NUMINAMATH_GPT_total_cows_l81_8161


namespace NUMINAMATH_GPT_complete_square_example_l81_8148

theorem complete_square_example :
  ∃ c : ℝ, ∃ d : ℝ, (∀ x : ℝ, x^2 + 12 * x + 4 = (x + c)^2 - d) ∧ d = 32 := by
  sorry

end NUMINAMATH_GPT_complete_square_example_l81_8148


namespace NUMINAMATH_GPT_calculate_expression_l81_8183

theorem calculate_expression : (Real.pi - 2023)^0 - |1 - Real.sqrt 2| + 2 * Real.cos (Real.pi / 4) - (1 / 2)⁻¹ = 0 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l81_8183


namespace NUMINAMATH_GPT_find_t_l81_8191

theorem find_t (t : ℕ) : 
  t > 3 ∧ (3 * t - 10) * (4 * t - 9) = (t + 12) * (2 * t + 1) → t = 6 := 
by
  intro h
  have h1 : t > 3 := h.1
  have h2 : (3 * t - 10) * (4 * t - 9) = (t + 12) * (2 * t + 1) := h.2
  sorry

end NUMINAMATH_GPT_find_t_l81_8191


namespace NUMINAMATH_GPT_greatest_decimal_is_7391_l81_8157

noncomputable def decimal_conversion (n d : ℕ) : ℝ :=
  n / d

noncomputable def forty_two_percent_of (r : ℝ) : ℝ :=
  0.42 * r

theorem greatest_decimal_is_7391 :
  let a := forty_two_percent_of (decimal_conversion 7 11)
  let b := decimal_conversion 17 23
  let c := 0.7391
  let d := decimal_conversion 29 47
  a < b ∧ a < c ∧ a < d ∧ b = c ∧ d < b :=
by
  have dec1 := forty_two_percent_of (decimal_conversion 7 11)
  have dec2 := decimal_conversion 17 23
  have dec3 := 0.7391
  have dec4 := decimal_conversion 29 47
  sorry

end NUMINAMATH_GPT_greatest_decimal_is_7391_l81_8157


namespace NUMINAMATH_GPT_sqrt_of_sixteen_l81_8190

theorem sqrt_of_sixteen : ∃ x : ℤ, x^2 = 16 ∧ (x = 4 ∨ x = -4) := by
  sorry

end NUMINAMATH_GPT_sqrt_of_sixteen_l81_8190


namespace NUMINAMATH_GPT_number_of_people_l81_8120

theorem number_of_people
  (x y : ℕ)
  (h1 : x + y = 28)
  (h2 : 2 * x + 4 * y = 92) :
  x = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_l81_8120


namespace NUMINAMATH_GPT_base_conversion_min_sum_l81_8156

theorem base_conversion_min_sum : ∃ a b : ℕ, a > 6 ∧ b > 6 ∧ (6 * a + 3 = 3 * b + 6) ∧ (a + b = 20) :=
by
  sorry

end NUMINAMATH_GPT_base_conversion_min_sum_l81_8156


namespace NUMINAMATH_GPT_largest_num_pencils_in_package_l81_8162

theorem largest_num_pencils_in_package (Ming_pencils Catherine_pencils : ℕ) 
  (Ming_pencils := 40) 
  (Catherine_pencils := 24) 
  (H : ∃ k, Ming_pencils = k * a ∧ Catherine_pencils = k * b) :
  gcd Ming_pencils Catherine_pencils = 8 :=
by
  sorry

end NUMINAMATH_GPT_largest_num_pencils_in_package_l81_8162


namespace NUMINAMATH_GPT_right_triangle_legs_sum_l81_8114

theorem right_triangle_legs_sum
  (x : ℕ)
  (h_even : Even x)
  (h_eq : x^2 + (x + 2)^2 = 34^2) :
  x + (x + 2) = 50 := 
by
  sorry

end NUMINAMATH_GPT_right_triangle_legs_sum_l81_8114


namespace NUMINAMATH_GPT_quadratic_unique_solution_k_neg_l81_8124

theorem quadratic_unique_solution_k_neg (k : ℝ) :
  (∃ x : ℝ, 9 * x^2 + k * x + 36 = 0 ∧ ∀ y : ℝ, 9 * y^2 + k * y + 36 = 0 → y = x) →
  k = -36 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_unique_solution_k_neg_l81_8124


namespace NUMINAMATH_GPT_infinite_natural_solutions_l81_8102

theorem infinite_natural_solutions : ∀ n : ℕ, ∃ x y z : ℕ, (x + y + z)^2 + 2 * (x + y + z) = 5 * (x * y + y * z + z * x) :=
by
  sorry

end NUMINAMATH_GPT_infinite_natural_solutions_l81_8102


namespace NUMINAMATH_GPT_expression_divisible_by_9_for_any_int_l81_8170

theorem expression_divisible_by_9_for_any_int (a b : ℤ) : 9 ∣ ((3 * a + 2)^2 - (3 * b + 2)^2) := 
by 
  sorry

end NUMINAMATH_GPT_expression_divisible_by_9_for_any_int_l81_8170


namespace NUMINAMATH_GPT_margo_total_distance_l81_8186

-- Definitions based on the conditions
def time_to_friends_house_min : ℕ := 15
def time_to_return_home_min : ℕ := 25
def total_walking_time_min : ℕ := time_to_friends_house_min + time_to_return_home_min
def total_walking_time_hours : ℚ := total_walking_time_min / 60
def average_walking_rate_mph : ℚ := 3
def total_distance_miles : ℚ := average_walking_rate_mph * total_walking_time_hours

-- The statement of the proof problem
theorem margo_total_distance : total_distance_miles = 2 := by
  sorry

end NUMINAMATH_GPT_margo_total_distance_l81_8186


namespace NUMINAMATH_GPT_Robert_can_read_one_book_l81_8171

def reading_speed : ℕ := 100 -- pages per hour
def book_length : ℕ := 350 -- pages
def available_time : ℕ := 5 -- hours

theorem Robert_can_read_one_book :
  (available_time * reading_speed) >= book_length ∧ 
  (available_time * reading_speed) < 2 * book_length :=
by {
  -- The proof steps are omitted as instructed.
  sorry
}

end NUMINAMATH_GPT_Robert_can_read_one_book_l81_8171


namespace NUMINAMATH_GPT_simplify_expression_l81_8133

theorem simplify_expression (x y z : ℝ) (h1 : x = 3) (h2 : y = 2) (h3 : z = 4) :
  (12 * x^2 * y^3 * z) / (4 * x * y * z^2) = 9 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l81_8133


namespace NUMINAMATH_GPT_rhombus_area_l81_8145

theorem rhombus_area (s d1 d2 : ℝ)
  (h1 : s = Real.sqrt 113)
  (h2 : abs (d1 - d2) = 8)
  (h3 : s^2 = (d1 / 2)^2 + (d2 / 2)^2) :
  (d1 * d2) / 2 = 194 := by
  sorry

end NUMINAMATH_GPT_rhombus_area_l81_8145


namespace NUMINAMATH_GPT_hexagon_shaded_area_correct_l81_8196

theorem hexagon_shaded_area_correct :
  let side_length := 3
  let semicircle_radius := side_length / 2
  let central_circle_radius := 1
  let hexagon_area := (3 * Real.sqrt 3 / 2) * side_length ^ 2
  let semicircle_area := (π * (semicircle_radius ^ 2)) / 2
  let total_semicircle_area := 6 * semicircle_area
  let central_circle_area := π * (central_circle_radius ^ 2)
  let shaded_area := hexagon_area - (total_semicircle_area + central_circle_area)
  shaded_area = 13.5 * Real.sqrt 3 - 7.75 * π := by
  sorry

end NUMINAMATH_GPT_hexagon_shaded_area_correct_l81_8196


namespace NUMINAMATH_GPT_crayons_lost_or_given_away_l81_8125

theorem crayons_lost_or_given_away (given_away lost : ℕ) (H_given_away : given_away = 213) (H_lost : lost = 16) :
  given_away + lost = 229 :=
by
  sorry

end NUMINAMATH_GPT_crayons_lost_or_given_away_l81_8125


namespace NUMINAMATH_GPT_problem_l81_8198

noncomputable def f : ℝ → ℝ := sorry

variables (x : ℝ) (a b c : ℝ)
  (h1 : ∀ x, f (x + 1) = f (-x + 1))
  (h2 : ∀ x, 1 < x → f x ≤ f (x - 1))
  (ha : a = f 2)
  (hb : b = f (Real.log 2 / Real.log 3))
  (hc : c = f (1 / 2))

theorem problem (h : a = f 2 ∧ b = f (Real.log 2 / Real.log 3) ∧ c = f (1 / 2)) : 
  a < c ∧ c < b := sorry

end NUMINAMATH_GPT_problem_l81_8198


namespace NUMINAMATH_GPT_length_of_train_is_125_l81_8193

noncomputable def speed_kmph : ℝ := 90
noncomputable def time_sec : ℝ := 5
noncomputable def speed_mps : ℝ := speed_kmph * (1000 / 3600)
noncomputable def length_train : ℝ := speed_mps * time_sec

theorem length_of_train_is_125 :
  length_train = 125 := 
by
  sorry

end NUMINAMATH_GPT_length_of_train_is_125_l81_8193


namespace NUMINAMATH_GPT_john_max_correct_answers_l81_8144

theorem john_max_correct_answers 
  (c w b : ℕ) -- define c, w, b as natural numbers
  (h1 : c + w + b = 30) -- condition 1: total questions
  (h2 : 4 * c - 3 * w = 36) -- condition 2: scoring equation
  : c ≤ 12 := -- statement to prove
sorry

end NUMINAMATH_GPT_john_max_correct_answers_l81_8144


namespace NUMINAMATH_GPT_interest_rate_first_part_l81_8181

theorem interest_rate_first_part 
  (total_amount : ℤ) 
  (amount_at_first_rate : ℤ) 
  (amount_at_second_rate : ℤ) 
  (rate_second_part : ℤ) 
  (total_annual_interest : ℤ) 
  (r : ℤ) 
  (h_split : total_amount = amount_at_first_rate + amount_at_second_rate) 
  (h_second : rate_second_part = 5)
  (h_interest : (amount_at_first_rate * r) / 100 + (amount_at_second_rate * rate_second_part) / 100 = total_annual_interest) :
  r = 3 := 
by 
  sorry

end NUMINAMATH_GPT_interest_rate_first_part_l81_8181


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l81_8105

-- Definition of the polynomial expansion
def poly (x : ℝ) := (1 - 2*x)^7

-- Definitions capturing the conditions directly
def a_0 := 1
def sum_a_1_to_a_7 := -2
def sum_a_1_3_5_7 := -1094
def sum_abs_a_0_to_a_7 := 2187

-- Lean statements for the proof problems
theorem problem1 (x : ℝ) (a : Fin 8 → ℝ) (h : poly x = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = sum_a_1_to_a_7 :=
sorry

theorem problem2 (x : ℝ) (a : Fin 8 → ℝ) (h : poly x = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7) :
  a 1 + a 3 + a 5 + a 7 = sum_a_1_3_5_7 :=
sorry

theorem problem3 (x : ℝ) (a : Fin 8 → ℝ) (h : poly x = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7) :
  abs (a 0) + abs (a 1) + abs (a 2) + abs (a 3) + abs (a 4) + abs (a 5) + abs (a 6) + abs (a 7) = sum_abs_a_0_to_a_7 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l81_8105


namespace NUMINAMATH_GPT_radius_of_semi_circle_l81_8158

variable (r w l : ℝ)

def rectangle_inscribed_semi_circle (w l : ℝ) := 
  l = 3*w ∧ 
  2*l + 2*w = 126 ∧ 
  (∃ r, l = 2*r)

theorem radius_of_semi_circle :
  (∃ w l r, rectangle_inscribed_semi_circle w l ∧ l = 2*r) → r = 23.625 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_semi_circle_l81_8158


namespace NUMINAMATH_GPT_more_larger_boxes_l81_8163

theorem more_larger_boxes (S L : ℕ) 
  (h1 : 12 * S + 16 * L = 480)
  (h2 : S + L = 32)
  (h3 : L > S) : L - S = 16 := 
sorry

end NUMINAMATH_GPT_more_larger_boxes_l81_8163


namespace NUMINAMATH_GPT_opposite_numbers_A_l81_8153

theorem opposite_numbers_A :
  let A1 := -((-1)^2)
  let A2 := abs (-1)

  let B1 := (-2)^3
  let B2 := -(2^3)
  
  let C1 := 2
  let C2 := 1 / 2
  
  let D1 := -(-1)
  let D2 := 1
  
  (A1 = -A2 ∧ A2 = 1) ∧ ¬(B1 = -B2) ∧ ¬(C1 = -C2) ∧ ¬(D1 = -D2)
:= by
  let A1 := -((-1)^2)
  let A2 := abs (-1)

  let B1 := (-2)^3
  let B2 := -(2^3)
  
  let C1 := 2
  let C2 := 1 / 2
  
  let D1 := -(-1)
  let D2 := 1

  sorry

end NUMINAMATH_GPT_opposite_numbers_A_l81_8153


namespace NUMINAMATH_GPT_simple_interest_rate_l81_8132

/-- Prove that given Principal (P) = 750, Amount (A) = 900, and Time (T) = 5 years,
    the rate (R) such that the Simple Interest formula holds is 4 percent. -/
theorem simple_interest_rate :
  ∀ (P A T : ℕ) (R : ℕ),
    P = 750 → 
    A = 900 → 
    T = 5 → 
    A = P + (P * R * T / 100) →
    R = 4 :=
by
  intros P A T R hP hA hT h_si
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l81_8132


namespace NUMINAMATH_GPT_translation_symmetric_graphs_l81_8164

/-- The graph of the function f(x)=sin(x/π + φ) is translated to the right by θ (θ>0) units to obtain the graph of the function g(x).
    On the graph of f(x), point A is translated to point B, let x_A and x_B be the abscissas of points A and B respectively.
    If the axes of symmetry of the graphs of f(x) and g(x) coincide, then the real values that can be taken as x_A - x_B are -2π² or -π². -/
theorem translation_symmetric_graphs (θ : ℝ) (hθ : θ > 0) (x_A x_B : ℝ) (φ : ℝ) :
  ((x_A - x_B = -2 * π^2) ∨ (x_A - x_B = -π^2)) :=
sorry

end NUMINAMATH_GPT_translation_symmetric_graphs_l81_8164


namespace NUMINAMATH_GPT_andrew_correct_answer_l81_8182

variable {x : ℕ}

theorem andrew_correct_answer (h : (x - 8) / 7 = 15) : (x - 5) / 11 = 10 :=
by
  sorry

end NUMINAMATH_GPT_andrew_correct_answer_l81_8182


namespace NUMINAMATH_GPT_sum_difference_20_l81_8179

def sum_of_even_integers (n : ℕ) : ℕ := (n / 2) * (2 + 2 * (n - 1))

def sum_of_odd_integers (n : ℕ) : ℕ := (n / 2) * (1 + 2 * (n - 1))

theorem sum_difference_20 : sum_of_even_integers (20) - sum_of_odd_integers (20) = 20 := by
  sorry

end NUMINAMATH_GPT_sum_difference_20_l81_8179


namespace NUMINAMATH_GPT_angle_C_measurement_l81_8155

variables (A B C : ℝ)

theorem angle_C_measurement
  (h1 : A + C = 2 * B)
  (h2 : C - A = 80)
  (h3 : A + B + C = 180) :
  C = 100 :=
by sorry

end NUMINAMATH_GPT_angle_C_measurement_l81_8155


namespace NUMINAMATH_GPT_probability_three_white_balls_l81_8136

def total_balls := 11
def white_balls := 5
def black_balls := 6
def balls_drawn := 5
def white_balls_drawn := 3
def black_balls_drawn := 2

theorem probability_three_white_balls :
  let total_outcomes := Nat.choose total_balls balls_drawn
  let favorable_outcomes := (Nat.choose white_balls white_balls_drawn) * (Nat.choose black_balls black_balls_drawn)
  (favorable_outcomes : ℚ) / total_outcomes = 25 / 77 :=
by
  sorry

end NUMINAMATH_GPT_probability_three_white_balls_l81_8136


namespace NUMINAMATH_GPT_angle_BDC_is_15_degrees_l81_8146

theorem angle_BDC_is_15_degrees (A B C D : Type) (AB AC AD CD : ℝ) (angle_BAC : ℝ) :
  AB = AC → AC = AD → CD = 2 * AC → angle_BAC = 30 →
  ∃ angle_BDC, angle_BDC = 15 := 
by
  sorry

end NUMINAMATH_GPT_angle_BDC_is_15_degrees_l81_8146


namespace NUMINAMATH_GPT_geometric_sequence_S4_l81_8110

/-
In the geometric sequence {a_n}, S_2 = 7, S_6 = 91. Prove that S_4 = 28.
-/

theorem geometric_sequence_S4 (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n * q)
  (h_sum : ∀ n, S n = a 1 * (1 - q^n) / (1 - q))
  (h_S2 : S 2 = 7) 
  (h_S6 : S 6 = 91) :
  S 4 = 28 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_S4_l81_8110


namespace NUMINAMATH_GPT_front_wheel_revolutions_l81_8176

theorem front_wheel_revolutions (P_front P_back : ℕ) (R_back : ℕ) (H1 : P_front = 30) (H2 : P_back = 20) (H3 : R_back = 360) :
  ∃ F : ℕ, F = 240 := by
  sorry

end NUMINAMATH_GPT_front_wheel_revolutions_l81_8176


namespace NUMINAMATH_GPT_xiao_qian_has_been_to_great_wall_l81_8169

-- Define the four students
inductive Student
| XiaoZhao
| XiaoQian
| XiaoSun
| XiaoLi

open Student

-- Define the relations for their statements
def has_been (s : Student) : Prop :=
  match s with
  | XiaoZhao => false
  | XiaoQian => true
  | XiaoSun => true
  | XiaoLi => false

def said (s : Student) : Prop :=
  match s with
  | XiaoZhao => ¬has_been XiaoZhao
  | XiaoQian => has_been XiaoLi
  | XiaoSun => has_been XiaoQian
  | XiaoLi => ¬has_been XiaoLi

axiom only_one_lying : ∃ l : Student, ∀ s : Student, said s → (s ≠ l)

theorem xiao_qian_has_been_to_great_wall : has_been XiaoQian :=
by {
  sorry -- Proof elided
}

end NUMINAMATH_GPT_xiao_qian_has_been_to_great_wall_l81_8169


namespace NUMINAMATH_GPT_a_share_is_1400_l81_8142

-- Definitions for the conditions
def investment_A : ℕ := 7000
def investment_B : ℕ := 11000
def investment_C : ℕ := 18000
def share_B : ℕ := 2200

-- Definition for the ratios
def ratio_A : ℚ := investment_A / 1000
def ratio_B : ℚ := investment_B / 1000
def ratio_C : ℚ := investment_C / 1000

-- Sum of ratios
def sum_ratios : ℚ := ratio_A + ratio_B + ratio_C

-- Total profit P can be deduced from B's share
def total_profit : ℚ := share_B * sum_ratios / ratio_B

-- Goal: Prove that A's share is $1400
def share_A : ℚ := ratio_A * total_profit / sum_ratios

theorem a_share_is_1400 : share_A = 1400 :=
sorry

end NUMINAMATH_GPT_a_share_is_1400_l81_8142


namespace NUMINAMATH_GPT_fraction_division_l81_8143

theorem fraction_division: 
  ((3 + 1 / 2) / 7) / (5 / 3) = 3 / 10 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_division_l81_8143


namespace NUMINAMATH_GPT_no_such_six_tuples_exist_l81_8128

theorem no_such_six_tuples_exist :
  ∀ (a b c x y z : ℕ),
    1 ≤ c → c ≤ b → b ≤ a →
    1 ≤ z → z ≤ y → y ≤ x →
    2 * a + b + 4 * c = 4 * x * y * z →
    2 * x + y + 4 * z = 4 * a * b * c →
    False :=
by
  intros a b c x y z h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_no_such_six_tuples_exist_l81_8128


namespace NUMINAMATH_GPT_tv_price_reduction_percentage_l81_8151

noncomputable def price_reduction (x : ℝ) : Prop :=
  (1 - x / 100) * 1.80 = 1.44000000000000014

theorem tv_price_reduction_percentage : price_reduction 20 :=
by
  sorry

end NUMINAMATH_GPT_tv_price_reduction_percentage_l81_8151


namespace NUMINAMATH_GPT_divides_six_ab_l81_8138

theorem divides_six_ab 
  (a b n : ℕ) 
  (hb : b < 10) 
  (hn : n > 3) 
  (h_eq : 2^n = 10 * a + b) : 
  6 ∣ (a * b) :=
sorry

end NUMINAMATH_GPT_divides_six_ab_l81_8138


namespace NUMINAMATH_GPT_number_of_two_digit_integers_l81_8140

def digits : Finset ℕ := {2, 4, 6, 7, 8}

theorem number_of_two_digit_integers : 
  (digits.card * (digits.card - 1)) = 20 := 
by
  sorry

end NUMINAMATH_GPT_number_of_two_digit_integers_l81_8140


namespace NUMINAMATH_GPT_log_product_evaluation_l81_8189

noncomputable def evaluate_log_product : ℝ :=
  Real.log 9 / Real.log 2 * Real.log 16 / Real.log 3 * Real.log 27 / Real.log 7

theorem log_product_evaluation : evaluate_log_product = 24 := 
  sorry

end NUMINAMATH_GPT_log_product_evaluation_l81_8189


namespace NUMINAMATH_GPT_segments_form_pentagon_l81_8159

theorem segments_form_pentagon (a b c d e : ℝ) 
  (h_sum : a + b + c + d + e = 2)
  (h_a : a > 1/10)
  (h_b : b > 1/10)
  (h_c : c > 1/10)
  (h_d : d > 1/10)
  (h_e : e > 1/10) :
  a + b + c + d > e ∧ a + b + c + e > d ∧ a + b + d + e > c ∧ a + c + d + e > b ∧ b + c + d + e > a := 
sorry

end NUMINAMATH_GPT_segments_form_pentagon_l81_8159


namespace NUMINAMATH_GPT_integer_solutions_exist_l81_8178

theorem integer_solutions_exist (k : ℤ) :
  (∃ x : ℤ, 9 * x - 3 = k * x + 14) ↔ (k = 8 ∨ k = 10 ∨ k = -8 ∨ k = 26) :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_exist_l81_8178


namespace NUMINAMATH_GPT_oranges_equivalency_l81_8199

theorem oranges_equivalency :
  ∀ (w_orange w_apple w_pear : ℕ), 
  (9 * w_orange = 6 * w_apple + w_pear) →
  (36 * w_orange = 24 * w_apple + 4 * w_pear) :=
by
  -- The proof will go here; for now, we'll use sorry to skip it
  sorry

end NUMINAMATH_GPT_oranges_equivalency_l81_8199


namespace NUMINAMATH_GPT_negation_of_existence_l81_8123

theorem negation_of_existence :
  ¬ (∃ x : ℝ, 0 < x ∧ x^3 - x + 1 > 0) ↔ ∀ x : ℝ, 0 < x → x^3 - x + 1 ≤ 0 :=
by sorry

end NUMINAMATH_GPT_negation_of_existence_l81_8123


namespace NUMINAMATH_GPT_triangle_PQ_length_l81_8168

theorem triangle_PQ_length (RP PQ : ℝ) (n : ℕ) (h_rp : RP = 2.4) (h_n : n = 25) : RP = 2.4 → PQ = 3 := by
  sorry

end NUMINAMATH_GPT_triangle_PQ_length_l81_8168


namespace NUMINAMATH_GPT_calculate_expression_l81_8106

-- Define the numerator and denominator
def numerator := 11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2 + 1
def denominator := 2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10

-- Prove the expression equals 1
theorem calculate_expression : (numerator / denominator) = 1 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l81_8106


namespace NUMINAMATH_GPT_student_avg_always_greater_l81_8139

theorem student_avg_always_greater (x y z : ℝ) (h1 : x < y) (h2 : y < z) : 
  ( ( (x + y) / 2 + z) / 2 ) > ( (x + y + z) / 3 ) :=
by
  sorry

end NUMINAMATH_GPT_student_avg_always_greater_l81_8139


namespace NUMINAMATH_GPT_rectangle_area_percentage_increase_l81_8127

theorem rectangle_area_percentage_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  let A := l * w
  let len_inc := 1.3 * l
  let wid_inc := 1.15 * w
  let A_new := len_inc * wid_inc
  let percentage_increase := ((A_new - A) / A) * 100
  percentage_increase = 49.5 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_percentage_increase_l81_8127


namespace NUMINAMATH_GPT_product_modulo_seven_l81_8122

theorem product_modulo_seven (a b c d : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3)
(h3 : c % 7 = 4) (h4 : d % 7 = 5) : (a * b * c * d) % 7 = 1 := 
sorry

end NUMINAMATH_GPT_product_modulo_seven_l81_8122


namespace NUMINAMATH_GPT_arithmetic_sum_example_l81_8175

def S (n : ℕ) (a1 d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

def a (n : ℕ) (a1 d : ℤ) : ℤ := a1 + (n - 1) * d

theorem arithmetic_sum_example (a1 d : ℤ) 
  (S20_eq_340 : S 20 a1 d = 340) :
  a 6 a1 d + a 9 a1 d + a 11 a1 d + a 16 a1 d = 68 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sum_example_l81_8175


namespace NUMINAMATH_GPT_inequality_solution_l81_8180

theorem inequality_solution (x : ℝ) (h : x > -4/3) : 2 - 1 / (3 * x + 4) < 5 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l81_8180


namespace NUMINAMATH_GPT_termite_ridden_fraction_l81_8147

theorem termite_ridden_fraction (T : ℝ) 
    (h1 : 5/8 * T > 0)
    (h2 : 3/8 * T = 0.125) : T = 1/8 :=
by
  sorry

end NUMINAMATH_GPT_termite_ridden_fraction_l81_8147


namespace NUMINAMATH_GPT_direct_proportion_function_l81_8134

theorem direct_proportion_function (m : ℝ) : 
  (m^2 + 2 * m ≠ 0) ∧ (m^2 - 3 = 1) → m = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_direct_proportion_function_l81_8134


namespace NUMINAMATH_GPT_daily_salmon_l81_8195

-- Definitions of the daily consumption of trout and total fish
def daily_trout : ℝ := 0.2
def daily_total_fish : ℝ := 0.6

-- Theorem statement that the daily consumption of salmon is 0.4 buckets
theorem daily_salmon : daily_total_fish - daily_trout = 0.4 := 
by
  -- Skipping the proof, as required
  sorry

end NUMINAMATH_GPT_daily_salmon_l81_8195


namespace NUMINAMATH_GPT_multiply_expression_l81_8165

theorem multiply_expression (x : ℝ) : (x^4 + 12 * x^2 + 144) * (x^2 - 12) = x^6 - 1728 := by
  sorry

end NUMINAMATH_GPT_multiply_expression_l81_8165


namespace NUMINAMATH_GPT_hyperbola_standard_equation_equation_of_line_L_l81_8101

open Real

noncomputable def hyperbola (x y : ℝ) : Prop :=
  y^2 - x^2 / 3 = 1

noncomputable def focus_on_y_axis := ∃ c : ℝ, c = 2

noncomputable def asymptote (x y : ℝ) : Prop := 
  y = sqrt 3 / 3 * x ∨ y = - sqrt 3 / 3 * x

noncomputable def point_A := (1, 1 / 2)

noncomputable def line_L (x y : ℝ) : Prop :=
  4 * x - 6 * y - 1 = 0

theorem hyperbola_standard_equation :
  ∃ (x y: ℝ), hyperbola x y :=
sorry

theorem equation_of_line_L :
  ∀ (x y : ℝ), point_A = (1, 1 / 2) ∧ line_L x y :=
sorry

end NUMINAMATH_GPT_hyperbola_standard_equation_equation_of_line_L_l81_8101


namespace NUMINAMATH_GPT_rectangle_ratio_l81_8107

/-- Conditions:
1. There are three identical squares and two rectangles forming a large square.
2. Each rectangle shares one side with a square and another side with the edge of the large square.
3. The side length of each square is 1 unit.
4. The total side length of the large square is 5 units.
Question:
What is the ratio of the length to the width of one of the rectangles? --/

theorem rectangle_ratio (sq_len : ℝ) (large_sq_len : ℝ) (side_ratio : ℝ) :
  sq_len = 1 ∧ large_sq_len = 5 ∧ 
  (∀ (rect_len rect_wid : ℝ), 3 * sq_len + 2 * rect_len = large_sq_len ∧ side_ratio = rect_len / rect_wid) →
  side_ratio = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_ratio_l81_8107


namespace NUMINAMATH_GPT_cat_clothing_probability_l81_8150

-- Define the conditions as Lean definitions
def n_items : ℕ := 3
def total_legs : ℕ := 4
def favorable_outcomes_per_leg : ℕ := 1
def possible_outcomes_per_leg : ℕ := (n_items.factorial : ℕ)
def probability_per_leg : ℚ := favorable_outcomes_per_leg / possible_outcomes_per_leg

-- Theorem statement to show the combined probability for all legs
theorem cat_clothing_probability
    (n_items_eq : n_items = 3)
    (total_legs_eq : total_legs = 4)
    (fact_n_items : (n_items.factorial) = 6)
    (prob_leg_eq : probability_per_leg = 1 / 6) :
    (probability_per_leg ^ total_legs = 1 / 1296) := by
    sorry

end NUMINAMATH_GPT_cat_clothing_probability_l81_8150


namespace NUMINAMATH_GPT_price_reduction_percentage_price_increase_amount_l81_8192

theorem price_reduction_percentage (x : ℝ) (hx : 50 * (1 - x)^2 = 32) : x = 0.2 := 
sorry

theorem price_increase_amount (y : ℝ) 
  (hy1 : 0 < y ∧ y ≤ 8) 
  (hy2 : 6000 = (10 + y) * (500 - 20 * y)) : y = 5 := 
sorry

end NUMINAMATH_GPT_price_reduction_percentage_price_increase_amount_l81_8192


namespace NUMINAMATH_GPT_part_I_solution_part_II_solution_l81_8167

-- Defining f(x) given parameters a and b
def f (x a b : ℝ) := |x - a| + |x + b|

-- Part (I): Given a = 1 and b = 2, solve the inequality f(x) ≤ 5
theorem part_I_solution (x : ℝ) : 
  (f x 1 2) ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := 
by
  sorry

-- Part (II): Given the minimum value of f(x) is 3, find min (a^2 / b + b^2 / a)
theorem part_II_solution (a b : ℝ) (h : 3 = |a| + |b|) (ha : a > 0) (hb : b > 0) : 
  (min (a^2 / b + b^2 / a)) = 3 := 
by
  sorry

end NUMINAMATH_GPT_part_I_solution_part_II_solution_l81_8167


namespace NUMINAMATH_GPT_lesser_fraction_l81_8111

theorem lesser_fraction (x y : ℚ) (h1 : x + y = 14 / 15) (h2 : x * y = 1 / 10) : min x y = 1 / 5 :=
sorry

end NUMINAMATH_GPT_lesser_fraction_l81_8111


namespace NUMINAMATH_GPT_range_of_m_l81_8174

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem range_of_m (m : ℝ) (h : ∀ x > 0, f x > m * x) : m ≤ 2 := sorry

end NUMINAMATH_GPT_range_of_m_l81_8174


namespace NUMINAMATH_GPT_pure_imaginary_complex_number_l81_8172

theorem pure_imaginary_complex_number (m : ℝ) (h : (m^2 - 3*m) = 0) :
  (m^2 - 5*m + 6) ≠ 0 → m = 0 :=
by
  intro h_im
  have h_fact : (m = 0) ∨ (m = 3) := by
    sorry -- This is where the factorization steps would go
  cases h_fact with
  | inl h0 =>
    assumption
  | inr h3 =>
    exfalso
    have : (3^2 - 5*3 + 6) = 0 := by
      sorry -- Simplify to check that m = 3 is not a valid solution
    contradiction

end NUMINAMATH_GPT_pure_imaginary_complex_number_l81_8172


namespace NUMINAMATH_GPT_rectangle_area_problem_l81_8149

theorem rectangle_area_problem (l w l1 l2 w1 w2 : ℝ) (h1 : l = l1 + l2) (h2 : w = w1 + w2) 
  (h3 : l1 * w1 = 12) (h4 : l2 * w1 = 15) (h5 : l1 * w2 = 12) 
  (h6 : l2 * w2 = 8) (h7 : w1 * l2 = 18) (h8 : l1 * w2 = 20) :
  l2 * w1 = 18 :=
sorry

end NUMINAMATH_GPT_rectangle_area_problem_l81_8149


namespace NUMINAMATH_GPT_zachary_needs_more_money_l81_8117

def cost_of_football : ℝ := 3.75
def cost_of_shorts : ℝ := 2.40
def cost_of_shoes : ℝ := 11.85
def zachary_money : ℝ := 10.00
def total_cost : ℝ := cost_of_football + cost_of_shorts + cost_of_shoes
def amount_needed : ℝ := total_cost - zachary_money

theorem zachary_needs_more_money : amount_needed = 7.00 := by
  sorry

end NUMINAMATH_GPT_zachary_needs_more_money_l81_8117


namespace NUMINAMATH_GPT_find_unknown_number_l81_8103

theorem find_unknown_number : 
  ∃ x : ℚ, (x * 7) / (10 * 17) = 10000 ∧ x = 1700000 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_unknown_number_l81_8103


namespace NUMINAMATH_GPT_value_of_A_l81_8108

-- Definitions for values in the factor tree, ensuring each condition is respected.
def D : ℕ := 3 * 2 * 2
def E : ℕ := 5 * 2
def B : ℕ := 3 * D
def C : ℕ := 5 * E
def A : ℕ := B * C

-- Assertion of the correct value for A
theorem value_of_A : A = 1800 := by
  -- Mathematical equivalence proof problem placeholder
  sorry

end NUMINAMATH_GPT_value_of_A_l81_8108


namespace NUMINAMATH_GPT_solve_for_A_l81_8116

def clubsuit (A B : ℤ) : ℤ := 3 * A + 2 * B + 7

theorem solve_for_A (A : ℤ) : (clubsuit A 6 = 70) -> (A = 17) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_A_l81_8116


namespace NUMINAMATH_GPT_find_X_l81_8187

-- Defining the given conditions and what we need to prove
theorem find_X (X : ℝ) (h : (X + 43 / 151) * 151 = 2912) : X = 19 :=
sorry

end NUMINAMATH_GPT_find_X_l81_8187


namespace NUMINAMATH_GPT_problem_solution_eq_l81_8115

theorem problem_solution_eq : 
  { x : ℝ | (x ^ 2 - 9) / (x ^ 2 - 1) > 0 } = { x : ℝ | x > 3 ∨ x < -3 } :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_eq_l81_8115


namespace NUMINAMATH_GPT_mean_of_five_integers_l81_8112

theorem mean_of_five_integers
  (p q r s t : ℤ)
  (h1 : (p + q + r) / 3 = 9)
  (h2 : (s + t) / 2 = 14) :
  (p + q + r + s + t) / 5 = 11 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_five_integers_l81_8112


namespace NUMINAMATH_GPT_find_c_l81_8177

theorem find_c (a b c : ℝ) : 
  (a * x^2 + b * x - 5) * (a * x^2 + b * x + 25) + c = (a * x^2 + b * x + 10)^2 → 
  c = 225 :=
by sorry

end NUMINAMATH_GPT_find_c_l81_8177


namespace NUMINAMATH_GPT_unemployment_percentage_next_year_l81_8188

theorem unemployment_percentage_next_year (E U : ℝ) (h1 : E > 0) :
  ( (0.91 * (0.056 * E)) / (1.04 * E) ) * 100 = 4.9 := by
  sorry

end NUMINAMATH_GPT_unemployment_percentage_next_year_l81_8188


namespace NUMINAMATH_GPT_curve_crosses_itself_and_point_of_crossing_l81_8126

-- Define the function for x and y
def x (t : ℝ) : ℝ := t^2 + 1
def y (t : ℝ) : ℝ := t^4 - 9 * t^2 + 6

-- Definition of the curve crossing itself and the point of crossing
theorem curve_crosses_itself_and_point_of_crossing :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ x t₁ = x t₂ ∧ y t₁ = y t₂ ∧ (x t₁ = 10 ∧ y t₁ = 6) :=
by
  sorry

end NUMINAMATH_GPT_curve_crosses_itself_and_point_of_crossing_l81_8126


namespace NUMINAMATH_GPT_johns_weight_l81_8194

theorem johns_weight (j m : ℝ) (h1 : j + m = 240) (h2 : j - m = j / 3) : j = 144 :=
by
  sorry

end NUMINAMATH_GPT_johns_weight_l81_8194


namespace NUMINAMATH_GPT_max_subway_riders_l81_8141

theorem max_subway_riders:
  ∃ (P F : ℕ), P + F = 251 ∧ (1 / 11) * P + (1 / 13) * F = 22 := sorry

end NUMINAMATH_GPT_max_subway_riders_l81_8141


namespace NUMINAMATH_GPT_common_chord_of_circles_l81_8166

theorem common_chord_of_circles :
  ∀ (x y : ℝ), (x^2 + y^2 - 4 * x = 0) ∧ (x^2 + y^2 - 4 * y = 0) → (x = y) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_common_chord_of_circles_l81_8166


namespace NUMINAMATH_GPT_minimize_squares_in_rectangle_l81_8109

theorem minimize_squares_in_rectangle (w h : ℕ) (hw : w = 63) (hh : h = 42) : 
  ∃ s : ℕ, s = Nat.gcd w h ∧ s = 21 :=
by
  sorry

end NUMINAMATH_GPT_minimize_squares_in_rectangle_l81_8109
