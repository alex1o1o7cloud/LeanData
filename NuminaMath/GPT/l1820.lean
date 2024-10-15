import Mathlib

namespace NUMINAMATH_GPT_unique_double_digit_in_range_l1820_182012

theorem unique_double_digit_in_range (a b : ℕ) (h₁ : a = 10) (h₂ : b = 40) : 
  ∃! n : ℕ, (10 ≤ n ∧ n ≤ 40) ∧ (n % 10 = n / 10) ∧ (n % 10 = 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_unique_double_digit_in_range_l1820_182012


namespace NUMINAMATH_GPT_plant_initial_mass_l1820_182096

theorem plant_initial_mass (x : ℕ) :
  (27 * x + 52 = 133) → x = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_plant_initial_mass_l1820_182096


namespace NUMINAMATH_GPT_solve_abc_l1820_182085

def f (x a b c : ℤ) : ℤ := x^3 + a*x^2 + b*x + c

theorem solve_abc (a b c : ℤ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h_fa : f a a b c = a^3) (h_fb : f b b a c = b^3) : 
  a = -2 ∧ b = 4 ∧ c = 16 := 
sorry

end NUMINAMATH_GPT_solve_abc_l1820_182085


namespace NUMINAMATH_GPT_fuel_consumption_rate_l1820_182039

theorem fuel_consumption_rate (fuel_left time_left r: ℝ) 
    (h_fuel: fuel_left = 6.3333) 
    (h_time: time_left = 0.6667) 
    (h_rate: r = fuel_left / time_left) : r = 9.5 := 
by
    sorry

end NUMINAMATH_GPT_fuel_consumption_rate_l1820_182039


namespace NUMINAMATH_GPT_roots_cubic_sum_of_cubes_l1820_182028

theorem roots_cubic_sum_of_cubes (a b c : ℝ)
  (h1 : Polynomial.eval a (Polynomial.C 1004 + Polynomial.C 502 * Polynomial.X + Polynomial.C 4 * Polynomial.X ^ 3) = 0)
  (h2 : Polynomial.eval b (Polynomial.C 1004 + Polynomial.C 502 * Polynomial.X + Polynomial.C 4 * Polynomial.X ^ 3) = 0)
  (h3 : Polynomial.eval c (Polynomial.C 1004 + Polynomial.C 502 * Polynomial.X + Polynomial.C 4 * Polynomial.X ^ 3) = 0)
  (h4 : a + b + c = 0) :
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 753 :=
by
  sorry

end NUMINAMATH_GPT_roots_cubic_sum_of_cubes_l1820_182028


namespace NUMINAMATH_GPT_function_through_point_l1820_182099

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a^x

theorem function_through_point (a : ℝ) (x : ℝ) (hx : (2 : ℝ) = x) (h : f 2 a = 4) : f x 2 = 2^x :=
by sorry

end NUMINAMATH_GPT_function_through_point_l1820_182099


namespace NUMINAMATH_GPT_Iggy_miles_on_Monday_l1820_182050

theorem Iggy_miles_on_Monday 
  (tuesday_miles : ℕ)
  (wednesday_miles : ℕ)
  (thursday_miles : ℕ)
  (friday_miles : ℕ)
  (monday_minutes : ℕ)
  (pace : ℕ)
  (total_hours : ℕ)
  (total_minutes : ℕ)
  (total_tuesday_to_friday_miles : ℕ)
  (total_tuesday_to_friday_minutes : ℕ) :
  tuesday_miles = 4 →
  wednesday_miles = 6 →
  thursday_miles = 8 →
  friday_miles = 3 →
  pace = 10 →
  total_hours = 4 →
  total_minutes = total_hours * 60 →
  total_tuesday_to_friday_miles = tuesday_miles + wednesday_miles + thursday_miles + friday_miles →
  total_tuesday_to_friday_minutes = total_tuesday_to_friday_miles * pace →
  monday_minutes = total_minutes - total_tuesday_to_friday_minutes →
  (monday_minutes / pace) = 3 := sorry

end NUMINAMATH_GPT_Iggy_miles_on_Monday_l1820_182050


namespace NUMINAMATH_GPT_placing_pencils_l1820_182005

theorem placing_pencils (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) 
    (h1 : total_pencils = 6) (h2 : num_rows = 2) : pencils_per_row = 3 :=
by
  sorry

end NUMINAMATH_GPT_placing_pencils_l1820_182005


namespace NUMINAMATH_GPT_polynomial_evaluation_l1820_182082

noncomputable def x : ℝ :=
  (3 + 3 * Real.sqrt 5) / 2

theorem polynomial_evaluation :
  (x^2 - 3 * x - 9 = 0) → (x^3 - 3 * x^2 - 9 * x + 7 = 7) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_polynomial_evaluation_l1820_182082


namespace NUMINAMATH_GPT_solution_intervals_l1820_182018

noncomputable def cubic_inequality (x : ℝ) : Prop :=
  x^3 - 3 * x^2 - 4 * x - 12 ≤ 0

noncomputable def linear_inequality (x : ℝ) : Prop :=
  2 * x + 6 > 0

theorem solution_intervals :
  { x : ℝ | cubic_inequality x ∧ linear_inequality x } = { x | -2 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end NUMINAMATH_GPT_solution_intervals_l1820_182018


namespace NUMINAMATH_GPT_staircase_steps_180_toothpicks_l1820_182033

-- Condition definition: total number of toothpicks for \( n \) steps is \( n(n + 1) \)
def total_toothpicks (n : ℕ) : ℕ := n * (n + 1)

-- Theorem statement: for 180 toothpicks, the number of steps \( n \) is 12
theorem staircase_steps_180_toothpicks : ∃ n : ℕ, total_toothpicks n = 180 ∧ n = 12 :=
by sorry

end NUMINAMATH_GPT_staircase_steps_180_toothpicks_l1820_182033


namespace NUMINAMATH_GPT_base6_to_base10_product_zero_l1820_182035

theorem base6_to_base10_product_zero
  (c d e : ℕ)
  (h : (5 * 6^2 + 3 * 6^1 + 2 * 6^0) = (100 * c + 10 * d + e)) :
  (c * e) / 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_base6_to_base10_product_zero_l1820_182035


namespace NUMINAMATH_GPT_power_six_rectangular_form_l1820_182025

noncomputable def sin (x : ℂ) : ℂ := (Complex.exp (-Complex.I * x) - Complex.exp (Complex.I * x)) / (2 * Complex.I)
noncomputable def cos (x : ℂ) : ℂ := (Complex.exp (Complex.I * x) + Complex.exp (-Complex.I * x)) / 2

theorem power_six_rectangular_form :
  (2 * cos (20 * Real.pi / 180) + 2 * Complex.I * sin (20 * Real.pi / 180))^6 = -32 + 32 * Complex.I * Real.sqrt 3 := sorry

end NUMINAMATH_GPT_power_six_rectangular_form_l1820_182025


namespace NUMINAMATH_GPT_exponential_first_quadrant_l1820_182083

theorem exponential_first_quadrant (m : ℝ) : 
  (∀ x : ℝ, y = (1 / 2)^x + m → y ≤ 0) ↔ m ≤ -1 := 
by
  sorry

end NUMINAMATH_GPT_exponential_first_quadrant_l1820_182083


namespace NUMINAMATH_GPT_total_pages_in_book_l1820_182094

def pages_already_read : ℕ := 147
def pages_left_to_read : ℕ := 416

theorem total_pages_in_book : pages_already_read + pages_left_to_read = 563 := by
  sorry

end NUMINAMATH_GPT_total_pages_in_book_l1820_182094


namespace NUMINAMATH_GPT_sum_of_coefficients_l1820_182030

theorem sum_of_coefficients (b_6 b_5 b_4 b_3 b_2 b_1 b_0 : ℤ) :
  (5 * x - 2) ^ 6 = b_6 * x ^ 6 + b_5 * x ^ 5 + b_4 * x ^ 4 + b_3 * x ^ 3 + b_2 * x ^ 2 + b_1 * x + b_0 →
  b_6 + b_5 + b_4 + b_3 + b_2 + b_1 + b_0 = 729 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1820_182030


namespace NUMINAMATH_GPT_parallel_line_plane_l1820_182074

-- Define vectors
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Dot product definition
def dotProduct (u v : Vector3D) : ℝ :=
  u.x * v.x + u.y * v.y + u.z * v.z

-- Options given
def optionA : Vector3D × Vector3D := (⟨1, 0, 0⟩, ⟨-2, 0, 0⟩)
def optionB : Vector3D × Vector3D := (⟨1, 3, 5⟩, ⟨1, 0, 1⟩)
def optionC : Vector3D × Vector3D := (⟨0, 2, 1⟩, ⟨-1, 0, -1⟩)
def optionD : Vector3D × Vector3D := (⟨1, -1, 3⟩, ⟨0, 3, 1⟩)

-- Main theorem
theorem parallel_line_plane :
  (dotProduct (optionA.fst) (optionA.snd) ≠ 0) ∧
  (dotProduct (optionB.fst) (optionB.snd) ≠ 0) ∧
  (dotProduct (optionC.fst) (optionC.snd) ≠ 0) ∧
  (dotProduct (optionD.fst) (optionD.snd) = 0) :=
by
  -- Using sorry to skip the proof
  sorry

end NUMINAMATH_GPT_parallel_line_plane_l1820_182074


namespace NUMINAMATH_GPT_inequality_example_l1820_182010

theorem inequality_example (a b c : ℝ) : a^2 + 4 * b^2 + 9 * c^2 ≥ 2 * a * b + 3 * a * c + 6 * b * c :=
by
  sorry

end NUMINAMATH_GPT_inequality_example_l1820_182010


namespace NUMINAMATH_GPT_abc_divisible_by_7_l1820_182044

theorem abc_divisible_by_7 (a b c : ℤ) (h : 7 ∣ (a^3 + b^3 + c^3)) : 7 ∣ (a * b * c) :=
sorry

end NUMINAMATH_GPT_abc_divisible_by_7_l1820_182044


namespace NUMINAMATH_GPT_quadratic_real_roots_opposite_signs_l1820_182053

theorem quadratic_real_roots_opposite_signs (c : ℝ) : 
  (c < 0 → (∃ x1 x2 : ℝ, x1 * x2 = c ∧ x1 + x2 = -1 ∧ x1 ≠ x2 ∧ (x1 < 0 ∧ x2 > 0 ∨ x1 > 0 ∧ x2 < 0))) ∧ 
  (∃ x1 x2 : ℝ, x1 * x2 = c ∧ x1 + x2 = -1 ∧ x1 ≠ x2 ∧ (x1 < 0 ∧ x2 > 0 ∨ x1 > 0 ∧ x2 < 0) → c < 0) :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_opposite_signs_l1820_182053


namespace NUMINAMATH_GPT_value_of_r_for_n_3_l1820_182031

theorem value_of_r_for_n_3 :
  ∀ (r s : ℕ), 
  (r = 4^s + 3 * s) → 
  (s = 2^3 + 2) → 
  r = 1048606 :=
by
  intros r s h1 h2
  sorry

end NUMINAMATH_GPT_value_of_r_for_n_3_l1820_182031


namespace NUMINAMATH_GPT_sum_of_three_numbers_l1820_182052

theorem sum_of_three_numbers (a b c : ℝ) (h₁ : a + b = 31) (h₂ : b + c = 48) (h₃ : c + a = 59) :
  a + b + c = 69 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l1820_182052


namespace NUMINAMATH_GPT_lattice_points_on_segment_l1820_182076

theorem lattice_points_on_segment : 
  let x1 := 5 
  let y1 := 23 
  let x2 := 47 
  let y2 := 297 
  ∃ n, n = 3 ∧ ∀ p : ℕ × ℕ, (p = (x1, y1) ∨ p = (x2, y2) ∨ ∃ t : ℕ, p = (x1 + t * (x2 - x1) / 2, y1 + t * (y2 - y1) / 2)) := 
sorry

end NUMINAMATH_GPT_lattice_points_on_segment_l1820_182076


namespace NUMINAMATH_GPT_compute_value_of_expression_l1820_182067

theorem compute_value_of_expression (p q : ℝ) (hpq : 3 * p^2 - 5 * p - 8 = 0) (hq : 3 * q^2 - 5 * q - 8 = 0) (hneq : p ≠ q) :
  3 * (p^2 - q^2) / (p - q) = 5 :=
by
  have hpq_sum : p + q = 5 / 3 := sorry
  exact sorry

end NUMINAMATH_GPT_compute_value_of_expression_l1820_182067


namespace NUMINAMATH_GPT_summer_camp_skills_l1820_182062

theorem summer_camp_skills
  (x y z a b c : ℕ)
  (h1 : x + y + z + a + b + c = 100)
  (h2 : y + z + c = 42)
  (h3 : z + x + b = 65)
  (h4 : x + y + a = 29) :
  a + b + c = 64 :=
by sorry

end NUMINAMATH_GPT_summer_camp_skills_l1820_182062


namespace NUMINAMATH_GPT_minimum_perimeter_is_728_l1820_182066

noncomputable def minimum_common_perimeter (a b c : ℤ) (h1 : 2 * a + 18 * c = 2 * b + 20 * c)
  (h2 : 9 * c * Real.sqrt (a^2 - (9 * c)^2) = 10 * c * Real.sqrt (b^2 - (10 * c)^2)) 
  (h3 : a = b + c) : ℤ :=
2 * a + 18 * c

theorem minimum_perimeter_is_728 (a b c : ℤ) 
  (h1 : 2 * a + 18 * c = 2 * b + 20 * c) 
  (h2 : 9 * c * Real.sqrt (a^2 - (9 * c)^2) = 10 * c * Real.sqrt (b^2 - (10 * c)^2)) 
  (h3 : a = b + c) : 
  minimum_common_perimeter a b c h1 h2 h3 = 728 :=
sorry

end NUMINAMATH_GPT_minimum_perimeter_is_728_l1820_182066


namespace NUMINAMATH_GPT_remainder_when_dividing_P_by_DDD_l1820_182011

variables (P D D' D'' Q Q' Q'' R R' R'' : ℕ)

-- Define the conditions
def condition1 : Prop := P = Q * D + R
def condition2 : Prop := Q = Q' * D' + R'
def condition3 : Prop := Q' = Q'' * D'' + R''

-- Theorem statement asserting the given conclusion
theorem remainder_when_dividing_P_by_DDD' 
  (H1 : condition1 P D Q R)
  (H2 : condition2 Q D' Q' R')
  (H3 : condition3 Q' D'' Q'' R'') : 
  P % (D * D' * D') = R'' * D * D' + R * D' + R := 
sorry

end NUMINAMATH_GPT_remainder_when_dividing_P_by_DDD_l1820_182011


namespace NUMINAMATH_GPT_Sarah_skateboard_speed_2160_mph_l1820_182003

-- Definitions based on the conditions
def miles_to_inches (miles : ℕ) : ℕ := miles * 63360
def minutes_to_hours (minutes : ℕ) : ℕ := minutes / 60

/-- Pete walks backwards 3 times faster than Susan walks forwards --/
def Susan_walks_forwards_speed (pete_walks_hands_speed : ℕ) : ℕ := pete_walks_hands_speed / 3

/-- Tracy does cartwheels twice as fast as Susan walks forwards --/
def Tracy_cartwheels_speed (susan_walks_forwards_speed : ℕ) : ℕ := susan_walks_forwards_speed * 2

/-- Mike swims 8 times faster than Tracy does cartwheels --/
def Mike_swims_speed (tracy_cartwheels_speed : ℕ) : ℕ := tracy_cartwheels_speed * 8

/-- Pete can walk on his hands at 1/4 the speed Tracy can do cartwheels --/
def Pete_walks_hands_speed : ℕ := 2

/-- Pete rides his bike 5 times faster than Mike swims --/
def Pete_rides_bike_speed (mike_swims_speed : ℕ) : ℕ := mike_swims_speed * 5

/-- Patty can row 3 times faster than Pete walks backwards (in feet per hour) --/
def Patty_rows_speed (pete_walks_backwards_speed : ℕ) : ℕ := pete_walks_backwards_speed * 3

/-- Sarah can skateboard 6 times faster than Patty rows (in miles per minute) --/
def Sarah_skateboards_speed (patty_rows_speed_ft_per_hr : ℕ) : ℕ := (patty_rows_speed_ft_per_hr * 6 * 60) * 63360 * 60

theorem Sarah_skateboard_speed_2160_mph : Sarah_skateboards_speed (Patty_rows_speed (Pete_walks_hands_speed * 3)) = 2160 * 63360 * 60 :=
by
  sorry

end NUMINAMATH_GPT_Sarah_skateboard_speed_2160_mph_l1820_182003


namespace NUMINAMATH_GPT_find_sum_of_a_b_l1820_182054

def star (a b : ℕ) : ℕ := a^b - a * b

theorem find_sum_of_a_b (a b : ℕ) (h1 : 2 ≤ a) (h2 : 2 ≤ b) (h3 : star a b = 2) : a + b = 5 := 
by
  sorry

end NUMINAMATH_GPT_find_sum_of_a_b_l1820_182054


namespace NUMINAMATH_GPT_books_of_jason_l1820_182069

theorem books_of_jason (M J : ℕ) (hM : M = 42) (hTotal : M + J = 60) : J = 18 :=
by
  sorry

end NUMINAMATH_GPT_books_of_jason_l1820_182069


namespace NUMINAMATH_GPT_transfer_people_correct_equation_l1820_182080

theorem transfer_people_correct_equation (A B x : ℕ) (h1 : A = 28) (h2 : B = 20) : 
  A + x = 2 * (B - x) := 
by sorry

end NUMINAMATH_GPT_transfer_people_correct_equation_l1820_182080


namespace NUMINAMATH_GPT_find_m_l1820_182091

-- Definition of vector
def vector (α : Type*) := α × α

-- Two vectors are collinear and have the same direction
def collinear_and_same_direction (a b : vector ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ a = (k * b.1, k * b.2)

-- The vectors a and b
def a (m : ℝ) : vector ℝ := (m, 1)
def b (m : ℝ) : vector ℝ := (4, m)

-- The theorem we want to prove
theorem find_m (m : ℝ) (h1 : collinear_and_same_direction (a m) (b m)) : m = 2 :=
  sorry

end NUMINAMATH_GPT_find_m_l1820_182091


namespace NUMINAMATH_GPT_time_to_cover_length_l1820_182036

/-- Define the conditions for the problem -/
def angle_deg : ℝ := 30
def escalator_speed : ℝ := 12
def length_along_incline : ℝ := 160
def person_speed : ℝ := 8

/-- Define the combined speed as the sum of the escalator speed and the person speed -/
def combined_speed : ℝ := escalator_speed + person_speed

/-- Theorem stating the time taken to cover the length of the escalator is 8 seconds -/
theorem time_to_cover_length : (length_along_incline / combined_speed) = 8 := by
  sorry

end NUMINAMATH_GPT_time_to_cover_length_l1820_182036


namespace NUMINAMATH_GPT_graph_inverse_point_sum_l1820_182019

theorem graph_inverse_point_sum 
  (f : ℝ → ℝ) (f_inv : ℝ → ℝ) 
  (h1 : ∀ x, f_inv (f x) = x) 
  (h2 : ∀ x, f (f_inv x) = x) 
  (h3 : f 2 = 6) 
  (h4 : (2, 3) ∈ {p : ℝ × ℝ | p.snd = f p.fst / 2}) :
  (6, 1) ∈ {p : ℝ × ℝ | p.snd = f_inv p.fst / 2} ∧ (6 + 1 = 7) :=
by
  sorry

end NUMINAMATH_GPT_graph_inverse_point_sum_l1820_182019


namespace NUMINAMATH_GPT_product_wavelengths_eq_n_cbrt_mn2_l1820_182009

variable (m n : ℝ)

noncomputable def common_ratio (m n : ℝ) := (n / m)^(1/3)

noncomputable def wavelength_jiazhong (m n : ℝ) := (m^2 * n)^(1/3)
noncomputable def wavelength_nanlu (m n : ℝ) := (n^4 / m)^(1/3)

theorem product_wavelengths_eq_n_cbrt_mn2
  (h : n = m * (common_ratio m n)^3) :
  (wavelength_jiazhong m n) * (wavelength_nanlu m n) = n * (m * n^2)^(1/3) :=
by
  sorry

end NUMINAMATH_GPT_product_wavelengths_eq_n_cbrt_mn2_l1820_182009


namespace NUMINAMATH_GPT_circle_reflection_l1820_182043

-- Definitions provided in conditions
def initial_center : ℝ × ℝ := (6, -5)
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.snd, p.fst)
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.fst, p.snd)

-- The final statement we need to prove
theorem circle_reflection :
  reflect_y_axis (reflect_y_eq_x initial_center) = (5, 6) :=
by
  -- By reflecting the point (6, -5) over y = x and then over the y-axis, we should get (5, 6)
  sorry

end NUMINAMATH_GPT_circle_reflection_l1820_182043


namespace NUMINAMATH_GPT_max_soccer_balls_l1820_182087

theorem max_soccer_balls (bought_balls : ℕ) (total_cost : ℕ) (available_money : ℕ) (unit_cost : ℕ)
    (h1 : bought_balls = 6) (h2 : total_cost = 168) (h3 : available_money = 500)
    (h4 : unit_cost = total_cost / bought_balls) :
    (available_money / unit_cost) = 17 := 
by
  sorry

end NUMINAMATH_GPT_max_soccer_balls_l1820_182087


namespace NUMINAMATH_GPT_find_five_dollar_bills_l1820_182000

-- Define the number of bills
def total_bills (x y : ℕ) : Prop := x + y = 126

-- Define the total value of the bills
def total_value (x y : ℕ) : Prop := 5 * x + 10 * y = 840

-- Now we state the theorem
theorem find_five_dollar_bills (x y : ℕ) (h1 : total_bills x y) (h2 : total_value x y) : x = 84 :=
by sorry

end NUMINAMATH_GPT_find_five_dollar_bills_l1820_182000


namespace NUMINAMATH_GPT_part1_tangent_line_at_x1_part2_a_range_l1820_182068

noncomputable def f (x a : ℝ) : ℝ := x * Real.exp x - a * x

theorem part1_tangent_line_at_x1 (a : ℝ) (h1 : a = 1) : 
  let f' (x : ℝ) : ℝ := (x + 1) * Real.exp x - 1
  (2 * Real.exp 1 - 1) * 1 - (f 1 1) = Real.exp 1 :=
by 
  sorry

theorem part2_a_range (a : ℝ) (h2 : ∀ x > 0, f x a ≥ Real.log x - x + 1) : 
  0 < a ∧ a ≤ 2 :=
by 
  sorry

end NUMINAMATH_GPT_part1_tangent_line_at_x1_part2_a_range_l1820_182068


namespace NUMINAMATH_GPT_P_inter_Q_eq_l1820_182056

def P (x : ℝ) : Prop := -1 < x ∧ x < 3
def Q (x : ℝ) : Prop := -2 < x ∧ x < 1

theorem P_inter_Q_eq : {x | P x} ∩ {x | Q x} = {x : ℝ | -1 < x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_P_inter_Q_eq_l1820_182056


namespace NUMINAMATH_GPT_pencil_weight_l1820_182078

theorem pencil_weight (total_weight : ℝ) (empty_case_weight : ℝ) (num_pencils : ℕ)
  (h1 : total_weight = 11.14) 
  (h2 : empty_case_weight = 0.5) 
  (h3 : num_pencils = 14) :
  (total_weight - empty_case_weight) / num_pencils = 0.76 := by
  sorry

end NUMINAMATH_GPT_pencil_weight_l1820_182078


namespace NUMINAMATH_GPT_base_conversion_is_248_l1820_182004

theorem base_conversion_is_248 (a b c n : ℕ) 
  (h1 : n = 49 * a + 7 * b + c) 
  (h2 : n = 81 * c + 9 * b + a) 
  (h3 : 0 ≤ a ∧ a ≤ 6) 
  (h4 : 0 ≤ b ∧ b ≤ 6) 
  (h5 : 0 ≤ c ∧ c ≤ 6)
  (h6 : 0 ≤ a ∧ a ≤ 8) 
  (h7 : 0 ≤ b ∧ b ≤ 8) 
  (h8 : 0 ≤ c ∧ c ≤ 8) 
  : n = 248 :=
by 
  sorry

end NUMINAMATH_GPT_base_conversion_is_248_l1820_182004


namespace NUMINAMATH_GPT_remainder_4063_div_97_l1820_182071

theorem remainder_4063_div_97 : 4063 % 97 = 86 := 
by sorry

end NUMINAMATH_GPT_remainder_4063_div_97_l1820_182071


namespace NUMINAMATH_GPT_quadratic_inequality_l1820_182097

theorem quadratic_inequality (a : ℝ) (h : 0 ≤ a ∧ a < 4) : ∀ x : ℝ, a * x^2 - a * x + 1 > 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_l1820_182097


namespace NUMINAMATH_GPT_nicholas_crackers_l1820_182038

theorem nicholas_crackers (marcus_crackers mona_crackers nicholas_crackers : ℕ) 
  (h1 : marcus_crackers = 3 * mona_crackers)
  (h2 : nicholas_crackers = mona_crackers + 6)
  (h3 : marcus_crackers = 27) : nicholas_crackers = 15 := by
  sorry

end NUMINAMATH_GPT_nicholas_crackers_l1820_182038


namespace NUMINAMATH_GPT_combined_mean_of_scores_l1820_182095

theorem combined_mean_of_scores (f s : ℕ) (mean_1 mean_2 : ℕ) (ratio : f = (2 * s) / 3) 
  (hmean1 : mean_1 = 90) (hmean2 : mean_2 = 75) :
  (135 * s) / ((2 * s) / 3 + s) = 81 := 
by
  sorry

end NUMINAMATH_GPT_combined_mean_of_scores_l1820_182095


namespace NUMINAMATH_GPT_hyperbola_equation_l1820_182013

open Real

theorem hyperbola_equation (e e' : ℝ) (h₁ : 2 * x^2 + y^2 = 2) (h₂ : e * e' = 1) :
  y^2 - x^2 = 2 :=
sorry

end NUMINAMATH_GPT_hyperbola_equation_l1820_182013


namespace NUMINAMATH_GPT_hex_prism_paintings_l1820_182084

def num_paintings : ℕ :=
  -- The total number of distinct ways to paint a hex prism according to the conditions
  3 -- Two colors case: white-red, white-blue, red-blue
  + 6 -- Three colors with pattern 121213
  + 1 -- Three colors with identical opposite faces: 123123
  + 3 -- Three colors with non-identical opposite faces: 123213

theorem hex_prism_paintings : num_paintings = 13 := by
  sorry

end NUMINAMATH_GPT_hex_prism_paintings_l1820_182084


namespace NUMINAMATH_GPT_find_value_of_c_l1820_182090

theorem find_value_of_c (c : ℝ) (h1 : c > 0) (h2 : c + ⌊c⌋ = 23.2) : c = 11.7 :=
sorry

end NUMINAMATH_GPT_find_value_of_c_l1820_182090


namespace NUMINAMATH_GPT_range_of_x_if_cos2_gt_sin2_l1820_182061

theorem range_of_x_if_cos2_gt_sin2 (x : ℝ) (h1 : x ∈ Set.Icc 0 Real.pi) (h2 : Real.cos x ^ 2 > Real.sin x ^ 2) :
  x ∈ Set.Ico 0 (Real.pi / 4) ∪ Set.Ioc (3 * Real.pi / 4) Real.pi :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_if_cos2_gt_sin2_l1820_182061


namespace NUMINAMATH_GPT_arithmetic_geometric_sum_l1820_182051

def a (n : ℕ) : ℕ := 3 * n - 2
def b (n : ℕ) : ℕ := 3 ^ (n - 1)

theorem arithmetic_geometric_sum :
  a (b 1) + a (b 2) + a (b 3) = 33 := by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sum_l1820_182051


namespace NUMINAMATH_GPT_correct_division_l1820_182023

theorem correct_division (a : ℝ) : a^8 / a^2 = a^6 := by 
  sorry

end NUMINAMATH_GPT_correct_division_l1820_182023


namespace NUMINAMATH_GPT_complement_of_set_M_l1820_182006

open Set

def universal_set : Set ℝ := univ

def set_M : Set ℝ := {x | x^2 < 2 * x}

def complement_M : Set ℝ := compl set_M

theorem complement_of_set_M :
  complement_M = {x | x ≤ 0 ∨ x ≥ 2} :=
sorry

end NUMINAMATH_GPT_complement_of_set_M_l1820_182006


namespace NUMINAMATH_GPT_inequality_proof_l1820_182055

open Real

theorem inequality_proof {x y : ℝ} (hx : x < 0) (hy : y < 0) : 
    (x ^ 4 / y ^ 4) + (y ^ 4 / x ^ 4) - (x ^ 2 / y ^ 2) - (y ^ 2 / x ^ 2) + (x / y) + (y / x) >= 2 := 
by
    sorry

end NUMINAMATH_GPT_inequality_proof_l1820_182055


namespace NUMINAMATH_GPT_tan_alpha_neg_four_over_three_l1820_182001

theorem tan_alpha_neg_four_over_three (α : ℝ) (h_cos : Real.cos α = -3/5) (h_alpha_range : α ∈ Set.Ioo (-π) 0) : Real.tan α = -4/3 :=
  sorry

end NUMINAMATH_GPT_tan_alpha_neg_four_over_three_l1820_182001


namespace NUMINAMATH_GPT_a2_value_for_cubic_expansion_l1820_182072

theorem a2_value_for_cubic_expansion (x a0 a1 a2 a3 : ℝ) : 
  (x ^ 3 = a0 + a1 * (x - 2) + a2 * (x - 2) ^ 2 + a3 * (x - 2) ^ 3) → a2 = 6 := by
  sorry

end NUMINAMATH_GPT_a2_value_for_cubic_expansion_l1820_182072


namespace NUMINAMATH_GPT_Sierra_Crest_Trail_Length_l1820_182002

theorem Sierra_Crest_Trail_Length (a b c d e : ℕ) 
(h1 : a + b + c = 36) 
(h2 : b + d = 30) 
(h3 : d + e = 38) 
(h4 : a + d = 32) : 
a + b + c + d + e = 74 := by
  sorry

end NUMINAMATH_GPT_Sierra_Crest_Trail_Length_l1820_182002


namespace NUMINAMATH_GPT_water_added_l1820_182034

theorem water_added (initial_volume : ℕ) (initial_sugar_percentage : ℝ) (final_sugar_percentage : ℝ) (V : ℝ) : 
  initial_volume = 3 →
  initial_sugar_percentage = 0.4 →
  final_sugar_percentage = 0.3 →
  V = 1 :=
by
  sorry

end NUMINAMATH_GPT_water_added_l1820_182034


namespace NUMINAMATH_GPT_intersection_complement_l1820_182070

open Set

def M : Set ℝ := {x | 0 < x ∧ x < 3}
def N : Set ℝ := {x | 2 < x}
def R_complement_N : Set ℝ := {x | x ≤ 2}

theorem intersection_complement : M ∩ R_complement_N = {x | 0 < x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l1820_182070


namespace NUMINAMATH_GPT_prove_total_rent_of_field_l1820_182049

def totalRentField (A_cows A_months B_cows B_months C_cows C_months 
                    D_cows D_months E_cows E_months F_cows F_months 
                    G_cows G_months A_rent : ℕ) : ℕ := 
  let A_cow_months := A_cows * A_months
  let B_cow_months := B_cows * B_months
  let C_cow_months := C_cows * C_months
  let D_cow_months := D_cows * D_months
  let E_cow_months := E_cows * E_months
  let F_cow_months := F_cows * F_months
  let G_cow_months := G_cows * G_months
  let total_cow_months := A_cow_months + B_cow_months + C_cow_months + 
                          D_cow_months + E_cow_months + F_cow_months + G_cow_months
  let rent_per_cow_month := A_rent / A_cow_months
  total_cow_months * rent_per_cow_month

theorem prove_total_rent_of_field : totalRentField 24 3 10 5 35 4 21 3 15 6 40 2 28 (7/2) 720 = 5930 :=
  by
  sorry

end NUMINAMATH_GPT_prove_total_rent_of_field_l1820_182049


namespace NUMINAMATH_GPT_inequality_proof_l1820_182017

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1820_182017


namespace NUMINAMATH_GPT_geom_mean_between_2_and_8_l1820_182037

theorem geom_mean_between_2_and_8 (b : ℝ) (h : b^2 = 16) : b = 4 ∨ b = -4 :=
by
  sorry

end NUMINAMATH_GPT_geom_mean_between_2_and_8_l1820_182037


namespace NUMINAMATH_GPT_new_average_age_l1820_182026

theorem new_average_age (n : ℕ) (avg_old : ℕ) (new_person_age : ℕ) (new_avg_age : ℕ)
  (h1 : avg_old = 14)
  (h2 : n = 9)
  (h3 : new_person_age = 34)
  (h4 : new_avg_age = 16) :
  (n * avg_old + new_person_age) / (n + 1) = new_avg_age :=
sorry

end NUMINAMATH_GPT_new_average_age_l1820_182026


namespace NUMINAMATH_GPT_B_score_1_probability_correct_A_wins_without_tiebreaker_probability_correct_l1820_182092

def prob_A_solve : ℝ := 0.8
def prob_B_solve : ℝ := 0.75

-- Definitions for A and B scoring in rounds
def prob_B_score_1_point : ℝ := 
  prob_B_solve * (1 - prob_B_solve) + (1 - prob_B_solve) * prob_B_solve

-- Definitions for A winning without a tiebreaker
def prob_A_score_1_point : ℝ :=
  prob_A_solve * (1 - prob_A_solve) + (1 - prob_A_solve) * prob_A_solve

def prob_A_score_2_points : ℝ :=
  prob_A_solve * prob_A_solve

def prob_B_score_0_points : ℝ :=
  (1 - prob_B_solve) * (1 - prob_B_solve)

def prob_B_score_total : ℝ :=
  prob_B_score_1_point

def prob_A_wins_without_tiebreaker : ℝ :=
  prob_A_score_2_points * prob_B_score_1_point +
  prob_A_score_2_points * prob_B_score_0_points +
  prob_A_score_1_point * prob_B_score_0_points

theorem B_score_1_probability_correct :
  prob_B_score_1_point = 3 / 8 := 
by
  sorry

theorem A_wins_without_tiebreaker_probability_correct :
  prob_A_wins_without_tiebreaker = 3 / 10 := 
by 
  sorry

end NUMINAMATH_GPT_B_score_1_probability_correct_A_wins_without_tiebreaker_probability_correct_l1820_182092


namespace NUMINAMATH_GPT_deepak_present_age_l1820_182014

theorem deepak_present_age (x : ℕ) (h : 4 * x + 6 = 26) : 3 * x = 15 := 
by 
  sorry

end NUMINAMATH_GPT_deepak_present_age_l1820_182014


namespace NUMINAMATH_GPT_time_to_save_for_downpayment_l1820_182040

-- Definitions based on conditions
def annual_saving : ℝ := 0.10 * 150000
def downpayment : ℝ := 0.20 * 450000

-- Statement of the theorem to be proved
theorem time_to_save_for_downpayment (T : ℝ) (H1 : annual_saving = 15000) (H2 : downpayment = 90000) : 
  T = 6 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_time_to_save_for_downpayment_l1820_182040


namespace NUMINAMATH_GPT_cost_per_sq_meter_l1820_182079

def tank_dimensions : ℝ × ℝ × ℝ := (25, 12, 6)
def total_plastering_cost : ℝ := 186
def total_plastering_area : ℝ :=
  let (length, width, height) := tank_dimensions
  let area_bottom := length * width
  let area_longer_walls := length * height * 2
  let area_shorter_walls := width * height * 2
  area_bottom + area_longer_walls + area_shorter_walls

theorem cost_per_sq_meter : total_plastering_cost / total_plastering_area = 0.25 := by
  sorry

end NUMINAMATH_GPT_cost_per_sq_meter_l1820_182079


namespace NUMINAMATH_GPT_four_bags_remainder_l1820_182065

theorem four_bags_remainder (n : ℤ) (hn : n % 11 = 5) : (4 * n) % 11 = 9 := 
by
  sorry

end NUMINAMATH_GPT_four_bags_remainder_l1820_182065


namespace NUMINAMATH_GPT_panthers_second_half_points_l1820_182060

theorem panthers_second_half_points (C1 P1 C2 P2 : ℕ) 
  (h1 : C1 + P1 = 38) 
  (h2 : C1 = P1 + 16) 
  (h3 : C1 + C2 + P1 + P2 = 58) 
  (h4 : C1 + C2 = P1 + P2 + 22) : 
  P2 = 7 :=
by 
  -- Definitions and substitutions are skipped here
  sorry

end NUMINAMATH_GPT_panthers_second_half_points_l1820_182060


namespace NUMINAMATH_GPT_remainder_div_13_l1820_182063

theorem remainder_div_13 {N : ℕ} (k : ℕ) (h : N = 39 * k + 18) : N % 13 = 5 := sorry

end NUMINAMATH_GPT_remainder_div_13_l1820_182063


namespace NUMINAMATH_GPT_relationship_between_lines_l1820_182047

-- Define the type for a line and a plane
structure Line where
  -- some properties (to be defined as needed, omitted for brevity)

structure Plane where
  -- some properties (to be defined as needed, omitted for brevity)

-- Define parallelism between a line and a plane
def parallel_line_plane (m : Line) (α : Plane) : Prop := sorry

-- Define line within a plane
def line_within_plane (n : Line) (α : Plane) : Prop := sorry

-- Define parallelism between two lines
def parallel_lines (m n : Line) : Prop := sorry

-- Define skewness between two lines
def skew_lines (m n : Line) : Prop := sorry

-- The mathematically equivalent proof problem
theorem relationship_between_lines (m n : Line) (α : Plane)
  (h1 : parallel_line_plane m α)
  (h2 : line_within_plane n α) :
  parallel_lines m n ∨ skew_lines m n := 
sorry

end NUMINAMATH_GPT_relationship_between_lines_l1820_182047


namespace NUMINAMATH_GPT_probability_of_matching_pair_l1820_182048

noncomputable def num_socks := 22
noncomputable def red_socks := 12
noncomputable def blue_socks := 10

def ways_to_choose_two (n : ℕ) : ℕ :=
  n * (n - 1) / 2

noncomputable def probability_same_color : ℚ :=
  (ways_to_choose_two red_socks + ways_to_choose_two blue_socks : ℚ) / ways_to_choose_two num_socks

theorem probability_of_matching_pair :
  probability_same_color = 37 / 77 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_probability_of_matching_pair_l1820_182048


namespace NUMINAMATH_GPT_customer_difference_l1820_182093

theorem customer_difference (before after : ℕ) (h1 : before = 19) (h2 : after = 4) : before - after = 15 :=
by
  sorry

end NUMINAMATH_GPT_customer_difference_l1820_182093


namespace NUMINAMATH_GPT_gear_revolutions_l1820_182015

variable (r_p : ℝ) 

theorem gear_revolutions (h1 : 40 * (1 / 6) = r_p * (1 / 6) + 5) : r_p = 10 := 
by
  sorry

end NUMINAMATH_GPT_gear_revolutions_l1820_182015


namespace NUMINAMATH_GPT_choir_members_l1820_182088

theorem choir_members (n : ℕ) : 
  (∃ k m : ℤ, n + 4 = 10 * k ∧ n + 5 = 11 * m) ∧ 200 < n ∧ n < 300 → n = 226 :=
by 
  sorry

end NUMINAMATH_GPT_choir_members_l1820_182088


namespace NUMINAMATH_GPT_business_total_profit_l1820_182086

def total_profit (investmentB periodB profitB : ℝ) (investmentA periodA profitA : ℝ) (investmentC periodC profitC : ℝ) : ℝ :=
    (investmentA * periodA * profitA) + (investmentB * periodB * profitB) + (investmentC * periodC * profitC)

theorem business_total_profit 
    (investmentB periodB profitB : ℝ)
    (investmentA periodA profitA : ℝ)
    (investmentC periodC profitC : ℝ)
    (hA_inv : investmentA = 3 * investmentB)
    (hA_period : periodA = 2 * periodB)
    (hC_inv : investmentC = 2 * investmentB)
    (hC_period : periodC = periodB / 2)
    (hA_rate : profitA = 0.10)
    (hB_rate : profitB = 0.15)
    (hC_rate : profitC = 0.12)
    (hB_profit : investmentB * periodB * profitB = 4000) :
    total_profit investmentB periodB profitB investmentA periodA profitA investmentC periodC profitC = 23200 := 
sorry

end NUMINAMATH_GPT_business_total_profit_l1820_182086


namespace NUMINAMATH_GPT_geom_seq_product_equals_16_l1820_182077

theorem geom_seq_product_equals_16
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_arith : ∀ m n, a (m + 1) - a m = a (n + 1) - a n)
  (non_zero_diff : ∃ d, d ≠ 0 ∧ ∀ n, a (n + 1) - a n = d)
  (h_cond : 2 * a 3 - (a 7) ^ 2 + 2 * a 11 = 0)
  (h_geom : ∀ m n, b (m + 1) / b m = b (n + 1) / b n)
  (h_b7 : b 7 = a 7):
  b 6 * b 8 = 16 := 
sorry

end NUMINAMATH_GPT_geom_seq_product_equals_16_l1820_182077


namespace NUMINAMATH_GPT_problem_inequality_l1820_182045

theorem problem_inequality (a b c d : ℝ) (h1 : d ≥ 0) (h2 : a + b = 2) (h3 : c + d = 2) :
  (a^2 + c^2) * (a^2 + d^2) * (b^2 + c^2) * (b^2 + d^2) ≤ 25 :=
by sorry

end NUMINAMATH_GPT_problem_inequality_l1820_182045


namespace NUMINAMATH_GPT_find_finleys_age_l1820_182081

-- Definitions for given problem
def rogers_age (J A : ℕ) := (J + A) / 2
def alex_age (F : ℕ) := 3 * (F + 10) - 5

-- Given conditions
def jills_age : ℕ := 20
def in_15_years_age_difference (R J F : ℕ) := R + 15 - (J + 15) = F - 30
def rogers_age_twice_jill_plus_five (J : ℕ) := 2 * J + 5

-- Theorem stating the problem assertion
theorem find_finleys_age (F : ℕ) :
  rogers_age jills_age (alex_age F) = rogers_age_twice_jill_plus_five jills_age ∧ 
  in_15_years_age_difference (rogers_age jills_age (alex_age F)) jills_age F →
  F = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_finleys_age_l1820_182081


namespace NUMINAMATH_GPT_total_students_l1820_182046

theorem total_students (S K : ℕ) (h1 : S = 4000) (h2 : K = 2 * S) :
  S + K = 12000 := by
  sorry

end NUMINAMATH_GPT_total_students_l1820_182046


namespace NUMINAMATH_GPT_arithmetic_mean_of_fractions_l1820_182016

theorem arithmetic_mean_of_fractions :
  (1 / 2 : ℚ) * ((2 / 5) + (4 / 7)) = 17 / 35 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_fractions_l1820_182016


namespace NUMINAMATH_GPT_find_a_iff_l1820_182059

def non_deg_ellipse (k : ℝ) : Prop :=
  ∀ x y : ℝ, 9 * (x^2) + (y^2) - 36 * x + 8 * y = k → 
  (∀ a b : ℝ, (a ≠ 0 ∧ b ≠ 0))

theorem find_a_iff (k : ℝ) : non_deg_ellipse k ↔ k > -52 := by
  sorry

end NUMINAMATH_GPT_find_a_iff_l1820_182059


namespace NUMINAMATH_GPT_find_A_l1820_182089

theorem find_A : ∃ A : ℕ, 691 - (600 + A * 10 + 7) = 4 ∧ A = 8 := by
  sorry

end NUMINAMATH_GPT_find_A_l1820_182089


namespace NUMINAMATH_GPT_companion_sets_count_l1820_182098

def companion_set (A : Set ℝ) : Prop :=
  ∀ x ∈ A, (x ≠ 0) → (1 / x) ∈ A

def M : Set ℝ := { -1, 0, 1/2, 2, 3 }

theorem companion_sets_count : 
  ∃ S : Finset (Set ℝ), (∀ A ∈ S, companion_set A) ∧ (∀ A ∈ S, A ⊆ M) ∧ S.card = 3 := 
by
  sorry

end NUMINAMATH_GPT_companion_sets_count_l1820_182098


namespace NUMINAMATH_GPT_find_ab_l1820_182007

theorem find_ab (a b : ℝ) (h1 : a - b = 5) (h2 : a^2 + b^2 = 31) : a * b = 3 := by
  sorry

end NUMINAMATH_GPT_find_ab_l1820_182007


namespace NUMINAMATH_GPT_find_side_length_of_left_square_l1820_182022

theorem find_side_length_of_left_square (x : ℕ) 
  (h1 : x + (x + 17) + (x + 11) = 52) : 
  x = 8 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_find_side_length_of_left_square_l1820_182022


namespace NUMINAMATH_GPT_antecedent_is_50_l1820_182075

theorem antecedent_is_50 (antecedent consequent : ℕ) (h_ratio : 4 * consequent = 6 * antecedent) (h_consequent : consequent = 75) : antecedent = 50 := by
  sorry

end NUMINAMATH_GPT_antecedent_is_50_l1820_182075


namespace NUMINAMATH_GPT_max_min_product_l1820_182032

theorem max_min_product (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0)
  (h_sum : p + q + r = 13) (h_prod_sum : p * q + q * r + r * p = 30) :
  ∃ n, n = min (p * q) (min (q * r) (r * p)) ∧ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_max_min_product_l1820_182032


namespace NUMINAMATH_GPT_train_pass_time_l1820_182024

noncomputable def train_speed_kmh := 36  -- Speed in km/hr
noncomputable def train_speed_ms := 10   -- Speed in m/s (converted)
noncomputable def platform_length := 180 -- Length of the platform in meters
noncomputable def platform_pass_time := 30 -- Time in seconds to pass platform
noncomputable def train_length := 120    -- Train length derived from conditions

theorem train_pass_time 
  (speed_in_kmh : ℕ) (speed_in_ms : ℕ) (platform_len : ℕ) (pass_platform_time : ℕ) (train_len : ℕ)
  (h1 : speed_in_kmh = 36)
  (h2 : speed_in_ms = 10)
  (h3 : platform_len = 180)
  (h4 : pass_platform_time = 30)
  (h5 : train_len = 120) :
  (train_len / speed_in_ms) = 12 := by
  sorry

end NUMINAMATH_GPT_train_pass_time_l1820_182024


namespace NUMINAMATH_GPT_sum_of_fourth_powers_eq_82_l1820_182020

theorem sum_of_fourth_powers_eq_82 (x y : ℝ) (hx : x + y = -2) (hy : x * y = -3) :
  x^4 + y^4 = 82 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fourth_powers_eq_82_l1820_182020


namespace NUMINAMATH_GPT_rubber_duck_cost_l1820_182073

theorem rubber_duck_cost 
  (price_large : ℕ)
  (num_regular : ℕ)
  (num_large : ℕ)
  (total_revenue : ℕ)
  (h1 : price_large = 5)
  (h2 : num_regular = 221)
  (h3 : num_large = 185)
  (h4 : total_revenue = 1588) :
  ∃ (cost_regular : ℕ), (num_regular * cost_regular + num_large * price_large = total_revenue) ∧ cost_regular = 3 :=
by
  exists 3
  sorry

end NUMINAMATH_GPT_rubber_duck_cost_l1820_182073


namespace NUMINAMATH_GPT_max_annual_profit_l1820_182064

noncomputable def annual_sales_volume (x : ℝ) : ℝ := - (1 / 3) * x^2 + 2 * x + 21

noncomputable def annual_sales_profit (x : ℝ) : ℝ := (- (1 / 3) * x^3 + 4 * x^2 + 9 * x - 126)

theorem max_annual_profit :
  ∀ x : ℝ, (x > 6) →
  (annual_sales_volume x) = - (1 / 3) * x^2 + 2 * x + 21 →
  (annual_sales_volume 10 = 23 / 3) →
  (21 - annual_sales_volume x = (1 / 3) * (x^2 - 6 * x)) →
    (annual_sales_profit x = - (1 / 3) * x^3 + 4 * x^2 + 9 * x - 126) ∧
    ∃ x_max : ℝ, 
      (annual_sales_profit x_max = 36) ∧
      x_max = 9 :=
by
  sorry

end NUMINAMATH_GPT_max_annual_profit_l1820_182064


namespace NUMINAMATH_GPT_shopkeeper_discount_l1820_182027

theorem shopkeeper_discount
  (CP LP SP : ℝ)
  (H_CP : CP = 100)
  (H_LP : LP = CP + 0.4 * CP)
  (H_SP : SP = CP + 0.33 * CP)
  (discount_percent : ℝ) :
  discount_percent = ((LP - SP) / LP) * 100 → discount_percent = 5 :=
by
  sorry

end NUMINAMATH_GPT_shopkeeper_discount_l1820_182027


namespace NUMINAMATH_GPT_arithmetic_progression_25th_term_l1820_182021

theorem arithmetic_progression_25th_term (a1 d : ℤ) (n : ℕ) (h_a1 : a1 = 5) (h_d : d = 7) (h_n : n = 25) :
  a1 + (n - 1) * d = 173 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_25th_term_l1820_182021


namespace NUMINAMATH_GPT_sara_payment_equivalence_l1820_182008

variable (cost_book1 cost_book2 change final_amount : ℝ)

theorem sara_payment_equivalence
  (h1 : cost_book1 = 5.5)
  (h2 : cost_book2 = 6.5)
  (h3 : change = 8)
  (h4 : final_amount = cost_book1 + cost_book2 + change) :
  final_amount = 20 := by
  sorry

end NUMINAMATH_GPT_sara_payment_equivalence_l1820_182008


namespace NUMINAMATH_GPT_sequence_inequality_l1820_182057

theorem sequence_inequality (a : ℕ → ℝ) (h_nonneg : ∀ n, 0 ≤ a n)
    (h_subadd : ∀ m n : ℕ, a (n + m) ≤ a n + a m) :
  ∀ (n m : ℕ), m ≤ n → a n ≤ m * a 1 + ((n : ℝ) / m - 1) * a m := 
by
  intros n m hnm
  sorry

end NUMINAMATH_GPT_sequence_inequality_l1820_182057


namespace NUMINAMATH_GPT_expand_polynomial_l1820_182029

theorem expand_polynomial :
  (5 * x^2 + 3 * x - 4) * 3 * x^3 = 15 * x^5 + 9 * x^4 - 12 * x^3 := 
by
  sorry

end NUMINAMATH_GPT_expand_polynomial_l1820_182029


namespace NUMINAMATH_GPT_min_value_expr_l1820_182041

noncomputable def expr (θ : Real) : Real :=
  3 * (Real.cos θ) + 2 / (Real.sin θ) + 2 * Real.sqrt 2 * (Real.tan θ)

theorem min_value_expr :
  ∃ (θ : Real), 0 < θ ∧ θ < Real.pi / 2 ∧ expr θ = (7 * Real.sqrt 2) / 2 := 
by
  sorry

end NUMINAMATH_GPT_min_value_expr_l1820_182041


namespace NUMINAMATH_GPT_marina_max_socks_l1820_182042

theorem marina_max_socks (white black : ℕ) (hw : white = 8) (hb : black = 15) :
  ∃ n, n = 17 ∧ ∀ w b, w + b = n → 0 ≤ w ∧ 0 ≤ b ∧ w ≤ black ∧ b ≤ black ∧ w ≤ white ∧ b ≤ black → b > w :=
sorry

end NUMINAMATH_GPT_marina_max_socks_l1820_182042


namespace NUMINAMATH_GPT_value_of_f_neg_one_l1820_182058

noncomputable def f : ℝ → ℝ := sorry

theorem value_of_f_neg_one (f_def : ∀ x, f (Real.tan x) = Real.sin (2 * x)) : f (-1) = -1 := 
by
sorry

end NUMINAMATH_GPT_value_of_f_neg_one_l1820_182058
