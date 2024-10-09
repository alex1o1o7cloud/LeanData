import Mathlib

namespace mike_picked_l1042_104233

-- Define the number of pears picked by Jason, Keith, and the total number of pears picked.
def jason_picked : ℕ := 46
def keith_picked : ℕ := 47
def total_picked : ℕ := 105

-- Define the goal that we need to prove: the number of pears Mike picked.
theorem mike_picked (jason_picked keith_picked total_picked : ℕ) 
  (h1 : jason_picked = 46) 
  (h2 : keith_picked = 47) 
  (h3 : total_picked = 105) 
  : (total_picked - (jason_picked + keith_picked)) = 12 :=
by sorry

end mike_picked_l1042_104233


namespace probability_diff_by_three_l1042_104296

theorem probability_diff_by_three (r1 r2 : ℕ) (h1 : 1 ≤ r1 ∧ r1 ≤ 6) (h2 : 1 ≤ r2 ∧ r2 ≤ 6) :
  (∃ (rolls : List (ℕ × ℕ)), 
    rolls = [ (2, 5), (5, 2), (3, 6), (4, 1) ] ∧ 
    (r1, r2) ∈ rolls) →
  (4 : ℚ) / 36 = (1 / 9 : ℚ) :=
by sorry

end probability_diff_by_three_l1042_104296


namespace perpendicular_x_intercept_l1042_104203

theorem perpendicular_x_intercept (x : ℝ) :
  (∃ y : ℝ, 2 * x + 3 * y = 9) ∧ (∃ y : ℝ, y = 5) → x = -10 / 3 :=
by sorry -- Proof omitted

end perpendicular_x_intercept_l1042_104203


namespace desired_line_equation_l1042_104277

-- Define the center of the circle and the equation of the given line
def center : (ℝ × ℝ) := (-1, 0)
def line1 (x y : ℝ) : Prop := x + y = 0

-- Define the desired line passing through the center of the circle and perpendicular to line1
def line2 (x y : ℝ) : Prop := x + y + 1 = 0

-- The theorem stating that the desired line equation is x + y + 1 = 0
theorem desired_line_equation : ∀ (x y : ℝ),
  (center = (-1, 0)) → (∀ x y, line1 x y → line2 x y) :=
by
  sorry

end desired_line_equation_l1042_104277


namespace find_a_l1042_104257

theorem find_a (x y a : ℤ) (h₁ : x = 1) (h₂ : y = -1) (h₃ : 2 * x - a * y = 3) : a = 1 :=
sorry

end find_a_l1042_104257


namespace geometric_sequence_condition_l1042_104282

-- Definition of a geometric sequence
def is_geometric_sequence (x y z : ℤ) : Prop :=
  y ^ 2 = x * z

-- Lean 4 statement based on the condition and correct answer tuple
theorem geometric_sequence_condition (a : ℤ) :
  is_geometric_sequence 4 a 9 ↔ (a = 6 ∨ a = -6) :=
by 
  sorry

end geometric_sequence_condition_l1042_104282


namespace multiplicative_inverse_exists_and_is_correct_l1042_104246

theorem multiplicative_inverse_exists_and_is_correct :
  ∃ N : ℤ, N > 0 ∧ (123456 * 171717) * N % 1000003 = 1 :=
sorry

end multiplicative_inverse_exists_and_is_correct_l1042_104246


namespace exist_A_B_l1042_104223

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exist_A_B : ∃ (A B : ℕ), A = 2016 * B ∧ sum_of_digits A + 2016 * sum_of_digits B < 0 := sorry

end exist_A_B_l1042_104223


namespace product_of_roots_l1042_104267

theorem product_of_roots :
  (let a := 36
   let b := -24
   let c := -120
   a ≠ 0) →
  let roots_product := c / a
  roots_product = -10/3 :=
by
  sorry

end product_of_roots_l1042_104267


namespace find_subtracted_value_l1042_104292

theorem find_subtracted_value (N V : ℕ) (hN : N = 12) (h : 4 * N - V = 9 * (N - 7)) : V = 3 :=
by
  sorry

end find_subtracted_value_l1042_104292


namespace correct_inequality_l1042_104204

variable (a b : ℝ)

theorem correct_inequality (h : a > b) : a - 3 > b - 3 :=
by
  sorry

end correct_inequality_l1042_104204


namespace solve_k_l1042_104263

theorem solve_k (t s : ℤ) : (∃ k m, 8 * k + 4 = 7 * m ∧ k = -4 + 7 * t ∧ m = -4 + 8 * t) →
  (∃ k m, 12 * k - 8 = 7 * m ∧ k = 3 + 7 * s ∧ m = 4 + 12 * s) →
  7 * t - 4 = 7 * s + 3 →
  ∃ k, k = 3 + 7 * s :=
by
  sorry

end solve_k_l1042_104263


namespace students_that_do_not_like_either_sport_l1042_104216

def total_students : ℕ := 30
def students_like_basketball : ℕ := 15
def students_like_table_tennis : ℕ := 10
def students_like_both : ℕ := 3

theorem students_that_do_not_like_either_sport : (total_students - (students_like_basketball + students_like_table_tennis - students_like_both)) = 8 := 
by
  sorry

end students_that_do_not_like_either_sport_l1042_104216


namespace cos_4theta_l1042_104247

theorem cos_4theta (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (4 * θ) = 17 / 81 := 
by 
  sorry

end cos_4theta_l1042_104247


namespace calculation_l1042_104253

variable (x y z : ℕ)

theorem calculation (h1 : x + y + z = 20) (h2 : x + y - z = 8) :
  x + y = 14 :=
  sorry

end calculation_l1042_104253


namespace chocolates_remaining_l1042_104251

def chocolates := 24
def chocolates_first_day := 4
def chocolates_eaten_second_day := (2 * chocolates_first_day) - 3
def chocolates_eaten_third_day := chocolates_first_day - 2
def chocolates_eaten_fourth_day := chocolates_eaten_third_day - 1

theorem chocolates_remaining :
  chocolates - (chocolates_first_day + chocolates_eaten_second_day + chocolates_eaten_third_day + chocolates_eaten_fourth_day) = 12 := by
  sorry

end chocolates_remaining_l1042_104251


namespace amount_A_l1042_104293

theorem amount_A (A B C : ℝ) (h1 : A = (2/3) * B) (h2 : B = (1/4) * C) (h3 : A + B + C = 527) : A = 62 := by
  sorry

end amount_A_l1042_104293


namespace remainder_of_3y_l1042_104214

theorem remainder_of_3y (y : ℕ) (hy : y % 9 = 5) : (3 * y) % 9 = 6 :=
sorry

end remainder_of_3y_l1042_104214


namespace rectangle_area_error_percentage_l1042_104278

theorem rectangle_area_error_percentage (L W : ℝ) : 
  let measured_length := 1.06 * L
  let measured_width := 0.95 * W
  let actual_area := L * W
  let measured_area := measured_length * measured_width
  let error := measured_area - actual_area
  let error_percentage := (error / actual_area) * 100
  error_percentage = 0.7 := 
by
  sorry

end rectangle_area_error_percentage_l1042_104278


namespace exist_coprime_sums_l1042_104262

theorem exist_coprime_sums (n k : ℕ) (h1 : 0 < n) (h2 : Even (k * (n - 1))) :
  ∃ x y : ℕ, Nat.gcd x n = 1 ∧ Nat.gcd y n = 1 ∧ (x + y) % n = k % n :=
  sorry

end exist_coprime_sums_l1042_104262


namespace time_for_A_to_finish_race_l1042_104249

-- Definitions based on the conditions
def race_distance : ℝ := 120
def B_time : ℝ := 45
def B_beaten_distance : ℝ := 24

-- Proof statement: We need to show that A's time is 56.25 seconds
theorem time_for_A_to_finish_race : ∃ (t : ℝ), t = 56.25 ∧ (120 / t = 96 / 45)
  := sorry

end time_for_A_to_finish_race_l1042_104249


namespace height_of_fourth_person_l1042_104245

/-- There are 4 people of different heights standing in order of increasing height.
    The difference is 2 inches between the first person and the second person,
    and also between the second person and the third person.
    The difference between the third person and the fourth person is 6 inches.
    The average height of the four people is 76 inches.
    Prove that the height of the fourth person is 82 inches. -/
theorem height_of_fourth_person 
  (h1 h2 h3 h4 : ℕ) 
  (h2_def : h2 = h1 + 2)
  (h3_def : h3 = h2 + 2)
  (h4_def : h4 = h3 + 6)
  (average_height : (h1 + h2 + h3 + h4) / 4 = 76) 
  : h4 = 82 :=
by sorry

end height_of_fourth_person_l1042_104245


namespace four_digit_solution_l1042_104234

-- Definitions for the conditions.
def condition1 (u z x : ℕ) : Prop := u + z - 4 * x = 1
def condition2 (u z y : ℕ) : Prop := u + 10 * z - 2 * y = 14

-- The theorem to prove that the four-digit number xyz is either 1014, 2218, or 1932
theorem four_digit_solution (x y z u : ℕ) (h1 : condition1 u z x) (h2 : condition2 u z y) :
  (x = 1 ∧ y = 0 ∧ z = 1 ∧ u = 4) ∨
  (x = 2 ∧ y = 2 ∧ z = 1 ∧ u = 8) ∨
  (x = 1 ∧ y = 9 ∧ z = 3 ∧ u = 2) := 
sorry

end four_digit_solution_l1042_104234


namespace diameter_of_double_area_square_l1042_104289

-- Define the given conditions and the problem to be solved
theorem diameter_of_double_area_square (d₁ : ℝ) (d₁_eq : d₁ = 4 * Real.sqrt 2) :
  ∃ d₂ : ℝ, d₂ = 8 :=
by
  -- Define the conditions
  let s₁ := d₁ / Real.sqrt 2
  have s₁_sq : s₁ ^ 2 = (d₁ ^ 2) / 2 := by sorry -- Pythagorean theorem

  let A₁ := s₁ ^ 2
  have A₁_eq : A₁ = 16 := by sorry -- Given diagonal, thus area

  let A₂ := 2 * A₁
  have A₂_eq : A₂ = 32 := by sorry -- Double the area

  let s₂ := Real.sqrt A₂
  have s₂_eq : s₂ = 4 * Real.sqrt 2 := by sorry -- Side length of second square

  let d₂ := s₂ * Real.sqrt 2
  have d₂_eq : d₂ = 8 := by sorry -- Diameter of the second square

  -- Prove the theorem
  existsi d₂
  exact d₂_eq

end diameter_of_double_area_square_l1042_104289


namespace pole_intersection_height_l1042_104238

theorem pole_intersection_height 
  (h1 h2 d : ℝ) 
  (h1pos : h1 = 30) 
  (h2pos : h2 = 90) 
  (dpos : d = 150) : 
  ∃ y, y = 22.5 :=
by
  sorry

end pole_intersection_height_l1042_104238


namespace height_of_old_lamp_l1042_104254

theorem height_of_old_lamp (height_new_lamp : ℝ) (height_difference : ℝ) (h : height_new_lamp = 2.33) (h_diff : height_difference = 1.33) : 
  (height_new_lamp - height_difference) = 1.00 :=
by
  have height_new : height_new_lamp = 2.33 := h
  have height_diff : height_difference = 1.33 := h_diff
  sorry

end height_of_old_lamp_l1042_104254


namespace cubic_inches_in_one_cubic_foot_l1042_104220

-- Definition for the given conversion between foot and inches
def foot_to_inches : ℕ := 12

-- The theorem to prove the cubic conversion
theorem cubic_inches_in_one_cubic_foot : (foot_to_inches ^ 3) = 1728 := by
  -- Skipping the actual proof
  sorry

end cubic_inches_in_one_cubic_foot_l1042_104220


namespace sum_of_digits_B_l1042_104281

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_B (n : ℕ) (h : n = 4444^4444) : digit_sum (digit_sum (digit_sum n)) = 7 :=
by
  sorry

end sum_of_digits_B_l1042_104281


namespace compute_expression_l1042_104280

theorem compute_expression : 7^2 - 2 * 5 + 4^2 / 2 = 47 := by
  sorry

end compute_expression_l1042_104280


namespace integer_solutions_eq_l1042_104215

theorem integer_solutions_eq (x y : ℤ) (h : y^2 = x^3 + (x + 1)^2) : (x, y) = (0, 1) ∨ (x, y) = (0, -1) :=
by
  sorry

end integer_solutions_eq_l1042_104215


namespace largest_possible_s_l1042_104200

noncomputable def max_value_of_s (p q r s : ℝ) (h1 : p + q + r + s = 8) (h2 : pq + pr + ps + qr + qs + rs = 12) : ℝ :=
  2 + 3 * Real.sqrt 2

theorem largest_possible_s (p q r s : ℝ) (h1 : p + q + r + s = 8) (h2 : pq + pr + ps + qr + qs + rs = 12) :
  s ≤ max_value_of_s p q r s h1 h2 := 
sorry

end largest_possible_s_l1042_104200


namespace no_solution_system_l1042_104209

theorem no_solution_system (a : ℝ) :
  (∀ (x : ℝ), (a ≠ 0 → (a^2 * x + 2 * a) / (a * x - 2 + a^2) < 0 ∨ ax + a ≤ 5/4)) ∧ 
  (a = 0 → ¬ ∃ (x : ℝ), (a^2 * x + 2 * a) / (a * x - 2 + a^2) ≥ 0 ∧ ax + a > 5/4) ↔ 
  a ∈ Set.Iic (-1/2) ∪ {0} :=
by sorry

end no_solution_system_l1042_104209


namespace original_paint_intensity_l1042_104255

theorem original_paint_intensity
  (I : ℝ) -- Original intensity of the red paint
  (f : ℝ) -- Fraction of the original paint replaced
  (new_intensity : ℝ) -- Intensity of the new paint
  (replacement_intensity : ℝ) -- Intensity of the replacement red paint
  (hf : f = 2 / 3)
  (hreplacement_intensity : replacement_intensity = 0.30)
  (hnew_intensity : new_intensity = 0.40)
  : I = 0.60 := 
sorry

end original_paint_intensity_l1042_104255


namespace evaluate_expression_l1042_104294

theorem evaluate_expression (a b c : ℤ) (ha : a = 3) (hb : b = 2) (hc : c = 1) :
  ((a^2 + b + c)^2 - (a^2 - b - c)^2) = 108 :=
by
  sorry

end evaluate_expression_l1042_104294


namespace number_of_terms_ap_l1042_104272

variables (a d n : ℤ) 

def sum_of_first_thirteen_terms := (13 / 2) * (2 * a + 12 * d)
def sum_of_last_thirteen_terms := (13 / 2) * (2 * a + (2 * n - 14) * d)

def sum_excluding_first_three := ((n - 3) / 2) * (2 * a + (n - 4) * d)
def sum_excluding_last_three := ((n - 3) / 2) * (2 * a + (n - 1) * d)

theorem number_of_terms_ap (h1 : sum_of_first_thirteen_terms a d = (1 / 2) * sum_of_last_thirteen_terms a d)
  (h2 : sum_excluding_first_three a d / sum_excluding_last_three a d = 5 / 4) : n = 22 :=
sorry

end number_of_terms_ap_l1042_104272


namespace largest_odd_digit_multiple_of_5_lt_10000_l1042_104279

def is_odd_digit (n : ℕ) : Prop :=
  n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 9

def all_odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), is_odd_digit d

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem largest_odd_digit_multiple_of_5_lt_10000 :
  ∃ n, n < 10000 ∧ all_odd_digits n ∧ is_multiple_of_5 n ∧
        ∀ m, m < 10000 → all_odd_digits m → is_multiple_of_5 m → m ≤ n :=
  sorry

end largest_odd_digit_multiple_of_5_lt_10000_l1042_104279


namespace quadratic_condition_l1042_104291

theorem quadratic_condition (a b c : ℝ) : (a ≠ 0) ↔ ∃ (x : ℝ), ax^2 + bx + c = 0 :=
by sorry

end quadratic_condition_l1042_104291


namespace parabola_translation_l1042_104228

theorem parabola_translation :
  (∀ x : ℝ, y = x^2 → y' = (x - 1)^2 + 3) :=
sorry

end parabola_translation_l1042_104228


namespace quadratic_transformation_l1042_104287

theorem quadratic_transformation
    (a b c : ℝ)
    (h : ℝ)
    (cond : ∀ x, a * x^2 + b * x + c = 4 * (x - 5)^2 + 16) :
    (∀ x, 5 * a * x^2 + 5 * b * x + 5 * c = 20 * (x - h)^2 + 80) → h = 5 :=
by
  sorry

end quadratic_transformation_l1042_104287


namespace mean_of_remaining_four_numbers_l1042_104224

theorem mean_of_remaining_four_numbers (a b c d : ℝ) (h1 : (a + b + c + d + 106) / 5 = 92) : 
  (a + b + c + d) / 4 = 88.5 := 
sorry

end mean_of_remaining_four_numbers_l1042_104224


namespace total_ladybugs_correct_l1042_104242

noncomputable def total_ladybugs (with_spots : ℕ) (without_spots : ℕ) : ℕ :=
  with_spots + without_spots

theorem total_ladybugs_correct :
  total_ladybugs 12170 54912 = 67082 :=
by
  unfold total_ladybugs
  rfl

end total_ladybugs_correct_l1042_104242


namespace election_margin_of_victory_l1042_104261

theorem election_margin_of_victory (T : ℕ) (H_winning_votes : T * 58 / 100 = 1044) :
  1044 - (T * 42 / 100) = 288 :=
by
  sorry

end election_margin_of_victory_l1042_104261


namespace total_monkeys_l1042_104266

theorem total_monkeys (x : ℕ) (h : (1 / 8 : ℝ) * x ^ 2 + 12 = x) : x = 48 :=
sorry

end total_monkeys_l1042_104266


namespace cost_of_paving_l1042_104297

-- declaring the definitions and the problem statement
def length_of_room := 5.5
def width_of_room := 4
def rate_per_sq_meter := 700

theorem cost_of_paving (length : ℝ) (width : ℝ) (rate : ℝ) : length = 5.5 → width = 4 → rate = 700 → (length * width * rate) = 15400 :=
by
  intros h_length h_width h_rate
  rw [h_length, h_width, h_rate]
  sorry

end cost_of_paving_l1042_104297


namespace min_distance_to_line_l1042_104205

-- Given that a point P(x, y) lies on the line x - y - 1 = 0
-- We need to prove that the minimum value of (x - 2)^2 + (y - 2)^2 is 1/2
theorem min_distance_to_line (x y: ℝ) (h: x - y - 1 = 0) :
  ∃ P : ℝ, P = (x - 2)^2 + (y - 2)^2 ∧ P = 1 / 2 :=
by
  sorry

end min_distance_to_line_l1042_104205


namespace train_speed_l1042_104273

/-- 
Train A leaves the station traveling at a certain speed v. 
Two hours later, Train B leaves the same station traveling in the same direction at 36 miles per hour. 
Train A was overtaken by Train B 360 miles from the station.
We need to prove that the speed of Train A was 30 miles per hour.
-/
theorem train_speed (v : ℕ) (t : ℕ) (h1 : 36 * (t - 2) = 360) (h2 : v * t = 360) : v = 30 :=
by 
  sorry

end train_speed_l1042_104273


namespace find_natural_number_n_l1042_104207

theorem find_natural_number_n : 
  ∃ (n : ℕ), (∃ k : ℕ, n + 15 = k^2) ∧ (∃ m : ℕ, n - 14 = m^2) ∧ n = 210 :=
by
  sorry

end find_natural_number_n_l1042_104207


namespace man_l1042_104201

noncomputable def speed_of_current : ℝ := 3 -- in kmph
noncomputable def time_to_cover_100_meters_downstream : ℝ := 19.99840012798976 -- in seconds
noncomputable def distance_covered : ℝ := 0.1 -- in kilometers (100 meters)

noncomputable def speed_in_still_water : ℝ :=
  (distance_covered / (time_to_cover_100_meters_downstream / 3600)) - speed_of_current

theorem man's_speed_in_still_water :
  speed_in_still_water = 14.9997120913593 :=
  by
    sorry

end man_l1042_104201


namespace pump_without_leak_time_l1042_104286

theorem pump_without_leak_time :
  ∃ T : ℝ, (1/T - 1/5.999999999999999 = 1/3) ∧ T = 2 :=
by 
  sorry

end pump_without_leak_time_l1042_104286


namespace gcd_1729_1309_eq_7_l1042_104211

theorem gcd_1729_1309_eq_7 : Nat.gcd 1729 1309 = 7 := by
  sorry

end gcd_1729_1309_eq_7_l1042_104211


namespace greatest_divisor_arithmetic_sum_l1042_104256

theorem greatest_divisor_arithmetic_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ d, d = 15 ∧ ∀ S : ℕ, S = 15 * x + 105 * c → d ∣ S :=
by 
  sorry

end greatest_divisor_arithmetic_sum_l1042_104256


namespace maximal_segment_number_l1042_104222

theorem maximal_segment_number (n : ℕ) (h : n > 4) : 
  ∃ k, k = if n % 2 = 0 then 2 * n - 4 else 2 * n - 3 :=
sorry

end maximal_segment_number_l1042_104222


namespace range_of_expression_l1042_104227

theorem range_of_expression (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 ≤ β ∧ β ≤ π / 2) :
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π :=
sorry

end range_of_expression_l1042_104227


namespace pair_ab_l1042_104288

def students_activities_ways (n_students n_activities : Nat) : Nat :=
  n_activities ^ n_students

def championships_outcomes (n_championships n_students : Nat) : Nat :=
  n_students ^ n_championships

theorem pair_ab (a b : Nat) :
  a = students_activities_ways 4 3 ∧ b = championships_outcomes 3 4 →
  (a, b) = (3^4, 4^3) := by
  sorry

end pair_ab_l1042_104288


namespace range_of_a_l1042_104213

noncomputable def f (x : ℤ) (a : ℝ) := (3 * x^2 + a * x + 26) / (x + 1)

theorem range_of_a (a : ℝ) :
  (∃ x : ℕ+, f x a ≤ 2) → a ≤ -15 :=
by
  sorry

end range_of_a_l1042_104213


namespace regular_polygon_perimeter_l1042_104206

theorem regular_polygon_perimeter (s : ℕ) (E : ℕ) (n : ℕ) (P : ℕ)
  (h1 : s = 6)
  (h2 : E = 90)
  (h3 : E = 360 / n)
  (h4 : P = n * s) :
  P = 24 :=
by sorry

end regular_polygon_perimeter_l1042_104206


namespace y_coordinate_of_A_l1042_104226

theorem y_coordinate_of_A (a : ℝ) (y : ℝ) (h1 : y = a * 1) (h2 : y = (4 - a) / 1) : y = 2 :=
by
  sorry

end y_coordinate_of_A_l1042_104226


namespace regular_octagon_side_length_l1042_104274

theorem regular_octagon_side_length
  (side_length_pentagon : ℕ)
  (total_wire_length : ℕ)
  (side_length_octagon : ℕ) :
  side_length_pentagon = 16 →
  total_wire_length = 5 * side_length_pentagon →
  side_length_octagon = total_wire_length / 8 →
  side_length_octagon = 10 := 
sorry

end regular_octagon_side_length_l1042_104274


namespace ranking_most_economical_l1042_104229

theorem ranking_most_economical (c_T c_R c_J q_T q_R q_J : ℝ)
  (hR_cost : c_R = 1.25 * c_T)
  (hR_quantity : q_R = 0.75 * q_J)
  (hJ_quantity : q_J = 2.5 * q_T)
  (hJ_cost : c_J = 1.2 * c_R) :
  ((c_J / q_J) ≤ (c_R / q_R)) ∧ ((c_R / q_R) ≤ (c_T / q_T)) :=
by {
  sorry
}

end ranking_most_economical_l1042_104229


namespace original_visual_range_l1042_104252

theorem original_visual_range
  (V : ℝ)
  (h1 : 2.5 * V = 150) :
  V = 60 :=
by
  sorry

end original_visual_range_l1042_104252


namespace time_to_cook_one_potato_l1042_104240

-- Definitions for the conditions
def total_potatoes : ℕ := 16
def cooked_potatoes : ℕ := 7
def remaining_minutes : ℕ := 45

-- Lean theorem that asserts the equivalence of the problem statement to the correct answer
theorem time_to_cook_one_potato (total_potatoes cooked_potatoes remaining_minutes : ℕ) 
  (h_total : total_potatoes = 16) 
  (h_cooked : cooked_potatoes = 7) 
  (h_remaining : remaining_minutes = 45) :
  (remaining_minutes / (total_potatoes - cooked_potatoes) = 5) :=
by
  -- Using sorry to skip proof
  sorry

end time_to_cook_one_potato_l1042_104240


namespace sum_of_fractions_l1042_104231

theorem sum_of_fractions:
  (2 / 5) + (3 / 8) + (1 / 4) = 1 + (1 / 40) :=
by
  sorry

end sum_of_fractions_l1042_104231


namespace evaluate_expression_l1042_104268

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end evaluate_expression_l1042_104268


namespace x_squared_plus_y_squared_l1042_104241

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 17) (h2 : x * y = 6) : x^2 + y^2 = 301 :=
by sorry

end x_squared_plus_y_squared_l1042_104241


namespace product_of_fraction_l1042_104299

theorem product_of_fraction (x : ℚ) (h : x = 17 / 999) : 17 * 999 = 16983 := by sorry

end product_of_fraction_l1042_104299


namespace tyler_total_puppies_l1042_104202

/-- 
  Tyler has 15 dogs, and each dog has 5 puppies.
  We want to prove that the total number of puppies is 75.
-/
def tyler_dogs : Nat := 15
def puppies_per_dog : Nat := 5
def total_puppies_tyler_has : Nat := tyler_dogs * puppies_per_dog

theorem tyler_total_puppies : total_puppies_tyler_has = 75 := by
  sorry

end tyler_total_puppies_l1042_104202


namespace pump_out_time_l1042_104244

theorem pump_out_time
  (length : ℝ)
  (width : ℝ)
  (depth : ℝ)
  (rate : ℝ)
  (H_length : length = 50)
  (H_width : width = 30)
  (H_depth : depth = 1.8)
  (H_rate : rate = 2.5) : 
  (length * width * depth) / rate / 60 = 18 :=
by
  sorry

end pump_out_time_l1042_104244


namespace simplify_expression_l1042_104237

theorem simplify_expression :
  (3 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7) =
  6 * Real.sqrt 6 + 6 * Real.sqrt 10 - 6 * Real.sqrt 14 :=
sorry

end simplify_expression_l1042_104237


namespace rabbits_in_cage_l1042_104276

theorem rabbits_in_cage (rabbits_in_cage : ℕ) (rabbits_park : ℕ) : 
  rabbits_in_cage = 13 ∧ rabbits_park = 60 → (1/3 * rabbits_park - rabbits_in_cage) = 7 :=
by
  sorry

end rabbits_in_cage_l1042_104276


namespace compare_trig_functions_l1042_104284

theorem compare_trig_functions :
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  b < a ∧ a < c :=
by
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  sorry

end compare_trig_functions_l1042_104284


namespace domain_of_sqrt_log_function_l1042_104290

def domain_of_function (x : ℝ) : Prop :=
  (1 ≤ x ∧ x < 2) ∨ (2 < x ∧ x < 3)

theorem domain_of_sqrt_log_function :
  ∀ x : ℝ, (x - 1 ≥ 0) → (x - 2 ≠ 0) → (-x^2 + 2 * x + 3 > 0) →
    domain_of_function x :=
by
  intros x h1 h2 h3
  unfold domain_of_function
  sorry

end domain_of_sqrt_log_function_l1042_104290


namespace residue_7_pow_1234_l1042_104225

theorem residue_7_pow_1234 : (7^1234) % 13 = 4 := by
  sorry

end residue_7_pow_1234_l1042_104225


namespace a_plus_b_values_l1042_104285

theorem a_plus_b_values (a b : ℤ) (h1 : |a + 1| = 0) (h2 : b^2 = 9) :
  a + b = 2 ∨ a + b = -4 :=
by
  have ha : a = -1 := by sorry
  have hb1 : b = 3 ∨ b = -3 := by sorry
  cases hb1 with
  | inl b_pos =>
    left
    rw [ha, b_pos]
    exact sorry
  | inr b_neg =>
    right
    rw [ha, b_neg]
    exact sorry

end a_plus_b_values_l1042_104285


namespace vertex_in_fourth_quadrant_l1042_104271

theorem vertex_in_fourth_quadrant (a : ℝ) (ha : a < 0) :  
  let x_vertex := -a / 4
  let y_vertex := (-40 - a^2) / 8
  x_vertex > 0 ∧ y_vertex < 0 := by
  let x_vertex := -a / 4
  let y_vertex := (-40 - a^2) / 8
  have hx : x_vertex > 0 := by sorry
  have hy : y_vertex < 0 := by sorry
  exact And.intro hx hy

end vertex_in_fourth_quadrant_l1042_104271


namespace zion_dad_age_difference_in_10_years_l1042_104230

/-
Given:
1. Zion's age is 8 years.
2. Zion's dad's age is 3 more than 4 times Zion's age.
Prove:
In 10 years, the difference in age between Zion's dad and Zion will be 27 years.
-/

theorem zion_dad_age_difference_in_10_years :
  let zion_age := 8
  let dad_age := 4 * zion_age + 3
  (dad_age + 10) - (zion_age + 10) = 27 := by
  sorry

end zion_dad_age_difference_in_10_years_l1042_104230


namespace max_piece_length_l1042_104265

theorem max_piece_length (L1 L2 L3 L4 : ℕ) (hL1 : L1 = 48) (hL2 : L2 = 72) (hL3 : L3 = 120) (hL4 : L4 = 144) 
  (h_min_pieces : ∀ L k, L = 48 ∨ L = 72 ∨ L = 120 ∨ L = 144 → k > 0 → L / k ≥ 5) : 
  ∃ k, k = 8 ∧ ∀ L, (L = L1 ∨ L = L2 ∨ L = L3 ∨ L = L4) → L % k = 0 :=
by
  sorry

end max_piece_length_l1042_104265


namespace geometric_sequence_property_l1042_104295

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ (n m : ℕ), a (n + 1) * a (m + 1) = a n * a m

theorem geometric_sequence_property (a : ℕ → ℝ) (h_geo : is_geometric_sequence a)
(h_condition : a 2 * a 4 = 1/2) :
  a 1 * a 3 ^ 2 * a 5 = 1/4 :=
by
  sorry

end geometric_sequence_property_l1042_104295


namespace last_four_digits_pow_product_is_5856_l1042_104212

noncomputable def product : ℕ := 301 * 402 * 503 * 604 * 646 * 547 * 448 * 349

theorem last_four_digits_pow_product_is_5856 :
  (product % 10000) ^ 4 % 10000 = 5856 := by
  sorry

end last_four_digits_pow_product_is_5856_l1042_104212


namespace find_quad_function_l1042_104269

-- Define the quadratic function with the given conditions
def quad_function (a b c : ℝ) (f : ℝ → ℝ) :=
  ∀ x, f x = a * x^2 + b * x + c

-- Define the values y(-2) = -3, y(-1) = -4, y(0) = -3, y(2) = 5
def given_points (f : ℝ → ℝ) :=
  f (-2) = -3 ∧ f (-1) = -4 ∧ f 0 = -3 ∧ f 2 = 5

-- Prove that y = x^2 + 2x - 3 satisfies the given points
theorem find_quad_function : ∃ f : ℝ → ℝ, (quad_function 1 2 (-3) f) ∧ (given_points f) :=
by
  sorry

end find_quad_function_l1042_104269


namespace initial_bacteria_count_l1042_104275

theorem initial_bacteria_count 
  (double_every_30_seconds : ∀ n : ℕ, n * 2^(240 / 30) = 262144) : 
  ∃ n : ℕ, n = 1024 :=
by
  -- Define the initial number of bacteria.
  let n := 262144 / (2^8)
  -- Assert that the initial number is 1024.
  use n
  -- To skip the proof.
  sorry

end initial_bacteria_count_l1042_104275


namespace trapezoid_area_correct_l1042_104217

noncomputable def calculate_trapezoid_area : ℕ :=
  let parallel_side_1 := 6
  let parallel_side_2 := 12
  let leg := 5
  let radius := 5
  let height := radius
  let area := (1 / 2) * (parallel_side_1 + parallel_side_2) * height
  area

theorem trapezoid_area_correct :
  calculate_trapezoid_area = 45 :=
by {
  sorry
}

end trapezoid_area_correct_l1042_104217


namespace find_f_half_l1042_104248

variable {R : Type*} [LinearOrderedField R]

def odd_function (f : R → R) : Prop := ∀ x : R, f (-x) = -f x

def f_condition (f : R → R) : Prop := ∀ x : R, x < 0 → f x = 1 / (x + 1)

theorem find_f_half (f : R → R) (h_odd : odd_function f) (h_condition : f_condition f) : f (1 / 2) = -2 := by
  sorry

end find_f_half_l1042_104248


namespace cassidy_grades_below_B_l1042_104239

theorem cassidy_grades_below_B (x : ℕ) (h1 : 26 = 14 + 3 * x) : x = 4 := 
by 
  sorry

end cassidy_grades_below_B_l1042_104239


namespace percentage_paid_X_vs_Y_l1042_104218

theorem percentage_paid_X_vs_Y (X Y : ℝ) (h1 : X + Y = 528) (h2 : Y = 240) :
  ((X / Y) * 100) = 120 :=
by
  sorry

end percentage_paid_X_vs_Y_l1042_104218


namespace find_b_l1042_104232

noncomputable def a : ℂ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℂ := sorry

-- Given conditions
axiom sum_eq : a + b + c = 4
axiom prod_pairs_eq : a * b + b * c + c * a = 5
axiom prod_triple_eq : a * b * c = 6

-- Prove that b = 1
theorem find_b : b = 1 :=
by
  -- Proof omitted
  sorry

end find_b_l1042_104232


namespace exists_zero_in_interval_minus3_minus2_l1042_104264

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x - x

theorem exists_zero_in_interval_minus3_minus2 : 
  ∃ x ∈ Set.Icc (-3 : ℝ) (-2), f x = 0 :=
by
  sorry

end exists_zero_in_interval_minus3_minus2_l1042_104264


namespace small_triangles_count_l1042_104250

theorem small_triangles_count
  (sL sS : ℝ)  -- side lengths of large (sL) and small (sS) triangles
  (hL : sL = 15)  -- condition for the large triangle's side length
  (hS : sS = 3)   -- condition for the small triangle's side length
  : sL^2 / sS^2 = 25 := 
by {
  -- Definitions to skip the proof body
  -- Further mathematical steps would usually go here
  -- but 'sorry' is used to indicate the skipped proof.
  sorry
}

end small_triangles_count_l1042_104250


namespace problem_1_problem_2_l1042_104243

noncomputable def A := Real.pi / 3
noncomputable def b := 5
noncomputable def c := 4 -- derived from the solution
noncomputable def S : ℝ := 5 * Real.sqrt 3

theorem problem_1 (A : ℝ) 
  (h : Real.cos (2 * A) - 3 * Real.cos (Real.pi - A) = 1) 
  : A = Real.pi / 3 :=
sorry

theorem problem_2 (a : ℝ) 
  (b : ℝ) 
  (S : ℝ) 
  (h_b : b = 5) 
  (h_S : S = 5 * Real.sqrt 3) 
  : a = Real.sqrt 21 :=
sorry

end problem_1_problem_2_l1042_104243


namespace parabola_line_intersect_at_one_point_l1042_104236

theorem parabola_line_intersect_at_one_point (a : ℚ) :
  (∃ x : ℚ, ax^2 + 5 * x + 4 = 0) → a = 25 / 16 :=
by
  -- Conditions and computation here
  sorry

end parabola_line_intersect_at_one_point_l1042_104236


namespace wine_cost_increase_l1042_104260

noncomputable def additional_cost (initial_price : ℝ) (num_bottles : ℕ) (month1_rate : ℝ) (month2_tariff : ℝ) (month2_discount : ℝ) (month3_tariff : ℝ) (month3_rate : ℝ) : ℝ := 
  let price_month1 := initial_price * (1 + month1_rate) 
  let cost_month1 := num_bottles * price_month1
  let price_month2 := (initial_price * (1 + month2_tariff)) * (1 - month2_discount)
  let cost_month2 := num_bottles * price_month2
  let price_month3 := (initial_price * (1 + month3_tariff)) * (1 - month3_rate)
  let cost_month3 := num_bottles * price_month3
  (cost_month1 + cost_month2 + cost_month3) - (3 * num_bottles * initial_price)

theorem wine_cost_increase : 
  additional_cost 20 5 0.05 0.25 0.15 0.35 0.03 = 42.20 :=
by sorry

end wine_cost_increase_l1042_104260


namespace trapezium_hole_perimeter_correct_l1042_104258

variable (a b : ℝ)

def trapezium_hole_perimeter (a b : ℝ) : ℝ :=
  6 * a - 3 * b

theorem trapezium_hole_perimeter_correct (a b : ℝ) :
  trapezium_hole_perimeter a b = 6 * a - 3 * b :=
by
  sorry

end trapezium_hole_perimeter_correct_l1042_104258


namespace tn_range_l1042_104221

noncomputable def a (n : ℕ) : ℚ :=
  (2 * n - 1) / 10

noncomputable def b (n : ℕ) : ℚ :=
  2^(n - 1)

noncomputable def c (n : ℕ) : ℚ :=
  (1 + a n) / (4 * b n)

noncomputable def T (n : ℕ) : ℚ :=
  (1 / 10) * (2 - (n + 2) / (2^n)) + (9 / 20) * (2 - 1 / (2^(n-1)))

theorem tn_range (n : ℕ) : (101 / 400 : ℚ) ≤ T n ∧ T n < (103 / 200 : ℚ) :=
sorry

end tn_range_l1042_104221


namespace fill_sacks_times_l1042_104283

-- Define the capacities of the sacks
def father_sack_capacity : ℕ := 20
def senior_ranger_sack_capacity : ℕ := 30
def volunteer_sack_capacity : ℕ := 25
def number_of_volunteers : ℕ := 2

-- Total wood gathered
def total_wood_gathered : ℕ := 200

-- Statement of the proof problem
theorem fill_sacks_times : (total_wood_gathered / (father_sack_capacity + senior_ranger_sack_capacity + (number_of_volunteers * volunteer_sack_capacity))) = 2 := by
  sorry

end fill_sacks_times_l1042_104283


namespace f_zero_f_positive_all_f_increasing_f_range_l1042_104208

universe u

noncomputable def f : ℝ → ℝ := sorry

axiom f_nonzero : f 0 ≠ 0
axiom f_positive : ∀ x : ℝ, 0 < x → f x > 1
axiom f_add_prop : ∀ a b : ℝ, f (a + b) = f a * f b

-- Problem 1: Prove that f(0) = 1
theorem f_zero : f 0 = 1 := sorry

-- Problem 2: Prove that for any x in ℝ, f(x) > 0
theorem f_positive_all (x : ℝ) : f x > 0 := sorry

-- Problem 3: Prove that f(x) is an increasing function on ℝ
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := sorry

-- Problem 4: Given f(x) * f(2x - x²) > 1, find the range of x
theorem f_range (x : ℝ) (h : f x * f (2*x - x^2) > 1) : 0 < x ∧ x < 3 := sorry

end f_zero_f_positive_all_f_increasing_f_range_l1042_104208


namespace num_sides_regular_polygon_l1042_104235

-- Define the perimeter and side length of the polygon
def perimeter : ℝ := 160
def side_length : ℝ := 10

-- Theorem to prove the number of sides
theorem num_sides_regular_polygon : 
  (perimeter / side_length) = 16 := by
    sorry  -- Proof is omitted

end num_sides_regular_polygon_l1042_104235


namespace quadratic_factorization_l1042_104298

theorem quadratic_factorization (p q x_1 x_2 : ℝ) (h1 : x_1 = 2) (h2 : x_2 = -3) 
    (h3 : x_1 + x_2 = -p) (h4 : x_1 * x_2 = q) : 
    (x - 2) * (x + 3) = x^2 + p * x + q :=
by
  sorry

end quadratic_factorization_l1042_104298


namespace contradiction_proof_l1042_104219

theorem contradiction_proof (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by sorry

end contradiction_proof_l1042_104219


namespace bill_experience_l1042_104210

theorem bill_experience (j b : ℕ) (h1 : j - 5 = 3 * (b - 5)) (h2 : j = 2 * b) : b = 10 := 
by
  sorry

end bill_experience_l1042_104210


namespace medium_as_decoy_and_rational_choice_l1042_104270

/-- 
  Define the prices and sizes of the popcorn containers:
  Small: 50g for 200 rubles.
  Medium: 70g for 400 rubles.
  Large: 130g for 500 rubles.
-/
structure PopcornContainer where
  size : ℕ -- in grams
  price : ℕ -- in rubles

def small := PopcornContainer.mk 50 200
def medium := PopcornContainer.mk 70 400
def large := PopcornContainer.mk 130 500

/-- 
  The medium-sized popcorn container can be considered a decoy
  in the context of asymmetric dominance.
  Additionally, under certain budget constraints and preferences, 
  rational economic agents may find the medium-sized container optimal.
-/
theorem medium_as_decoy_and_rational_choice :
  (medium.price = 400 ∧ medium.size = 70) ∧ 
  (∃ (budget : ℕ) (pref : ℕ → ℕ → Prop), (budget ≥ medium.price ∧ 
    pref medium.size (budget - medium.price))) :=
by
  sorry

end medium_as_decoy_and_rational_choice_l1042_104270


namespace necessary_and_sufficient_condition_l1042_104259

theorem necessary_and_sufficient_condition (t : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ) :
    (∀ n, S n = n^2 + 5*n + t) →
    (t = 0 ↔ (∀ n, a n = 2*n + 4 ∧ (n > 0 → a n = S n - S (n - 1)))) :=
by
  sorry

end necessary_and_sufficient_condition_l1042_104259
