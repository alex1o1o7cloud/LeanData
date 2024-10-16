import Mathlib

namespace NUMINAMATH_CALUDE_min_value_of_M_l4143_414336

theorem min_value_of_M (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 →
    Real.sqrt (1 + 2 * x^2) + 2 * Real.sqrt ((5/12)^2 + y^2) ≥ 5 * Real.sqrt 34 / 12) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧
    Real.sqrt (1 + 2 * x^2) + 2 * Real.sqrt ((5/12)^2 + y^2) = 5 * Real.sqrt 34 / 12) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_M_l4143_414336


namespace NUMINAMATH_CALUDE_residue_16_pow_3030_mod_23_l4143_414348

theorem residue_16_pow_3030_mod_23 : 16^3030 ≡ 1 [ZMOD 23] := by
  sorry

end NUMINAMATH_CALUDE_residue_16_pow_3030_mod_23_l4143_414348


namespace NUMINAMATH_CALUDE_investment_plans_count_l4143_414363

/-- The number of ways to distribute projects among cities --/
def distribute_projects (n_projects : ℕ) (n_cities : ℕ) (max_per_city : ℕ) : ℕ := sorry

/-- Theorem statement --/
theorem investment_plans_count :
  distribute_projects 3 4 2 = 60 := by sorry

end NUMINAMATH_CALUDE_investment_plans_count_l4143_414363


namespace NUMINAMATH_CALUDE_linden_birch_problem_l4143_414316

theorem linden_birch_problem :
  ∃ (x y : ℕ), 
    x + y > 14 ∧ 
    y + 18 > 2 * x ∧ 
    x > 2 * y ∧ 
    x = 11 ∧ 
    y = 5 := by
  sorry

end NUMINAMATH_CALUDE_linden_birch_problem_l4143_414316


namespace NUMINAMATH_CALUDE_range_of_a_l4143_414367

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 3| + |x - a| > 5) ↔ 
  (a > 8 ∨ a < -2) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l4143_414367


namespace NUMINAMATH_CALUDE_fitness_equipment_problem_l4143_414337

/-- Unit price of type A fitness equipment -/
def unit_price_A : ℝ := 360

/-- Unit price of type B fitness equipment -/
def unit_price_B : ℝ := 540

/-- Total number of fitness equipment to be purchased -/
def total_equipment : ℕ := 50

/-- Maximum total cost allowed -/
def max_total_cost : ℝ := 21000

/-- Theorem stating the conditions and conclusions of the fitness equipment problem -/
theorem fitness_equipment_problem :
  (unit_price_B = 1.5 * unit_price_A) ∧
  (7200 / unit_price_A - 5400 / unit_price_B = 10) ∧
  (∀ x : ℕ, x ≤ total_equipment →
    unit_price_A * x + unit_price_B * (total_equipment - x) ≤ max_total_cost →
    x ≥ 34) :=
sorry

end NUMINAMATH_CALUDE_fitness_equipment_problem_l4143_414337


namespace NUMINAMATH_CALUDE_paint_left_after_three_weeks_l4143_414312

def paint_calculation (initial_paint : ℚ) : ℚ :=
  let after_week1 := initial_paint - (1/4 * initial_paint)
  let after_week2 := after_week1 - (1/2 * after_week1)
  let after_week3 := after_week2 - (2/3 * after_week2)
  after_week3

theorem paint_left_after_three_weeks :
  paint_calculation 360 = 45 := by sorry

end NUMINAMATH_CALUDE_paint_left_after_three_weeks_l4143_414312


namespace NUMINAMATH_CALUDE_negation_of_all_students_prepared_l4143_414372

variable (α : Type)
variable (student : α → Prop)
variable (prepared : α → Prop)

theorem negation_of_all_students_prepared :
  (¬ ∀ x, student x → prepared x) ↔ (∃ x, student x ∧ ¬ prepared x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_students_prepared_l4143_414372


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l4143_414349

theorem cubic_root_sum_cubes (x y z : ℝ) : 
  (x^3 - 5*x - 3 = 0) → 
  (y^3 - 5*y - 3 = 0) → 
  (z^3 - 5*z - 3 = 0) → 
  x^3 * y^3 + x^3 * z^3 + y^3 * z^3 = 99 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l4143_414349


namespace NUMINAMATH_CALUDE_regular_pay_limit_l4143_414339

theorem regular_pay_limit (regular_rate : ℝ) (overtime_hours : ℝ) (total_pay : ℝ) 
  (h1 : regular_rate = 3)
  (h2 : overtime_hours = 13)
  (h3 : total_pay = 198) :
  ∃ (regular_hours : ℝ),
    regular_hours * regular_rate + overtime_hours * (2 * regular_rate) = total_pay ∧
    regular_hours = 40 := by
  sorry

end NUMINAMATH_CALUDE_regular_pay_limit_l4143_414339


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_four_l4143_414331

theorem greatest_integer_with_gcf_four : ∃ n : ℕ, n < 200 ∧ 
  Nat.gcd n 24 = 4 ∧ 
  ∀ m : ℕ, m < 200 → Nat.gcd m 24 = 4 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_four_l4143_414331


namespace NUMINAMATH_CALUDE_min_value_of_sum_l4143_414319

theorem min_value_of_sum (x y : ℝ) (h : x^2 - 2*x*y + y^2 - Real.sqrt 2*x - Real.sqrt 2*y + 6 = 0) :
  ∃ (u : ℝ), u = x + y ∧ u ≥ 3 * Real.sqrt 2 ∧ ∀ (v : ℝ), v = x + y → v ≥ u := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l4143_414319


namespace NUMINAMATH_CALUDE_alices_total_distance_l4143_414317

/-- Alice's weekly walking distance to school and back home --/
def alices_weekly_walking_distance (days_per_week : ℕ) (distance_to_school : ℕ) (distance_from_school : ℕ) : ℕ :=
  (days_per_week * distance_to_school) + (days_per_week * distance_from_school)

/-- Theorem: Alice walks 110 miles in a week --/
theorem alices_total_distance :
  alices_weekly_walking_distance 5 10 12 = 110 := by
  sorry

end NUMINAMATH_CALUDE_alices_total_distance_l4143_414317


namespace NUMINAMATH_CALUDE_range_of_a_l4143_414311

-- Define the sets S and T
def S : Set ℝ := {x | x < -1 ∨ x > 5}
def T (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 8}

-- State the theorem
theorem range_of_a : 
  ∀ a : ℝ, (S ∪ T a = Set.univ) → (-3 < a ∧ a < -1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l4143_414311


namespace NUMINAMATH_CALUDE_basketball_only_count_l4143_414365

theorem basketball_only_count (total students_basketball students_table_tennis students_neither : ℕ) 
  (h1 : total = 30)
  (h2 : students_basketball = 15)
  (h3 : students_table_tennis = 10)
  (h4 : students_neither = 8)
  (h5 : total = students_basketball + students_table_tennis - students_both + students_neither)
  (students_both : ℕ) :
  students_basketball - students_both = 12 := by
  sorry

end NUMINAMATH_CALUDE_basketball_only_count_l4143_414365


namespace NUMINAMATH_CALUDE_work_time_calculation_l4143_414393

theorem work_time_calculation (a_time b_time : ℝ) (b_fraction : ℝ) : 
  a_time = 6 →
  b_time = 3 →
  b_fraction = 1/9 →
  (1 - b_fraction) / (1 / a_time) = 16/3 :=
by sorry

end NUMINAMATH_CALUDE_work_time_calculation_l4143_414393


namespace NUMINAMATH_CALUDE_angle_measure_l4143_414357

theorem angle_measure (x : ℝ) : 
  (180 - x = 6 * (90 - x)) → x = 72 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l4143_414357


namespace NUMINAMATH_CALUDE_tan_2alpha_values_l4143_414351

theorem tan_2alpha_values (α : Real) (h : 2 * Real.sin (2 * α) = 1 + Real.cos (2 * α)) :
  Real.tan (2 * α) = 4 / 3 ∨ Real.tan (2 * α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_2alpha_values_l4143_414351


namespace NUMINAMATH_CALUDE_sphere_surface_area_doubling_l4143_414373

/-- Given a sphere whose surface area doubles when its radius is doubled,
    prove that if the new surface area is 9856 cm², 
    then the original surface area is 2464 cm². -/
theorem sphere_surface_area_doubling (r : ℝ) :
  (4 * Real.pi * (2 * r)^2 = 9856) → (4 * Real.pi * r^2 = 2464) := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_doubling_l4143_414373


namespace NUMINAMATH_CALUDE_overlap_area_is_one_l4143_414304

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three vertices -/
structure Triangle where
  v1 : Point
  v2 : Point
  v3 : Point

/-- Calculate the area of overlap between two triangles on a 3x3 grid -/
def triangleOverlapArea (t1 t2 : Triangle) : ℝ :=
  sorry

/-- Theorem stating that the area of overlap between the given triangles is 1 square unit -/
theorem overlap_area_is_one :
  let t1 : Triangle := { v1 := {x := 0, y := 2}, v2 := {x := 2, y := 1}, v3 := {x := 0, y := 0} }
  let t2 : Triangle := { v1 := {x := 2, y := 2}, v2 := {x := 0, y := 1}, v3 := {x := 2, y := 0} }
  triangleOverlapArea t1 t2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_overlap_area_is_one_l4143_414304


namespace NUMINAMATH_CALUDE_males_not_interested_count_l4143_414398

/-- Represents the survey results for a yoga class -/
structure YogaSurvey where
  total_not_interested : ℕ
  females_not_interested : ℕ

/-- Calculates the number of males not interested in the yoga class -/
def males_not_interested (survey : YogaSurvey) : ℕ :=
  survey.total_not_interested - survey.females_not_interested

/-- Theorem stating that the number of males not interested is 110 -/
theorem males_not_interested_count (survey : YogaSurvey) 
  (h1 : survey.total_not_interested = 200)
  (h2 : survey.females_not_interested = 90) : 
  males_not_interested survey = 110 := by
  sorry

#eval males_not_interested ⟨200, 90⟩

end NUMINAMATH_CALUDE_males_not_interested_count_l4143_414398


namespace NUMINAMATH_CALUDE_max_a_squared_b_l4143_414345

theorem max_a_squared_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * (a + b) = 27) :
  a^2 * b ≤ 54 := by
sorry

end NUMINAMATH_CALUDE_max_a_squared_b_l4143_414345


namespace NUMINAMATH_CALUDE_min_exponent_sum_l4143_414387

theorem min_exponent_sum (h : ℕ+) (a b c : ℕ+) 
  (h_div_225 : 225 ∣ h)
  (h_div_216 : 216 ∣ h)
  (h_factorization : h = 2^(a:ℕ) * 3^(b:ℕ) * 5^(c:ℕ)) :
  a + b + c ≥ 8 ∧ ∃ (h' : ℕ+) (a' b' c' : ℕ+), 
    225 ∣ h' ∧ 216 ∣ h' ∧ h' = 2^(a':ℕ) * 3^(b':ℕ) * 5^(c':ℕ) ∧ a' + b' + c' = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_exponent_sum_l4143_414387


namespace NUMINAMATH_CALUDE_fred_money_left_l4143_414388

def fred_book_problem (initial_amount : ℕ) (num_books : ℕ) (avg_cost : ℕ) : ℕ :=
  initial_amount - (num_books * avg_cost)

theorem fred_money_left :
  fred_book_problem 236 6 37 = 14 := by
  sorry

end NUMINAMATH_CALUDE_fred_money_left_l4143_414388


namespace NUMINAMATH_CALUDE_water_removal_for_concentration_l4143_414329

/-- 
Proves that the amount of water removed to concentrate a 40% acidic liquid to 60% acidic liquid 
is 5 liters, given that the final volume is 5 liters less than the initial volume.
-/
theorem water_removal_for_concentration (initial_volume : ℝ) : 
  initial_volume > 0 →
  let initial_concentration : ℝ := 0.4
  let final_concentration : ℝ := 0.6
  let volume_decrease : ℝ := 5
  let final_volume : ℝ := initial_volume - volume_decrease
  let water_removed : ℝ := volume_decrease
  initial_concentration * initial_volume = final_concentration * final_volume →
  water_removed = 5 := by
sorry

end NUMINAMATH_CALUDE_water_removal_for_concentration_l4143_414329


namespace NUMINAMATH_CALUDE_min_triangles_for_100gon_l4143_414326

/-- A convex polygon with 100 sides -/
def ConvexPolygon100 : Type := Unit

/-- The number of triangles needed to represent a convex 100-gon as their intersection -/
def num_triangles (p : ConvexPolygon100) : ℕ := sorry

/-- The smallest number of triangles needed to represent any convex 100-gon as their intersection -/
def min_num_triangles : ℕ := sorry

theorem min_triangles_for_100gon :
  min_num_triangles = 50 := by sorry

end NUMINAMATH_CALUDE_min_triangles_for_100gon_l4143_414326


namespace NUMINAMATH_CALUDE_intersection_points_convex_ngon_l4143_414321

/-- The number of intersection points of the diagonals in a convex n-gon -/
def intersectionPoints (n : ℕ) : ℕ :=
  Nat.choose n 4

/-- Theorem: The number of intersection points of the diagonals in a convex n-gon
    is equal to (n choose 4) for n ≥ 4 -/
theorem intersection_points_convex_ngon (n : ℕ) (h : n ≥ 4) :
  intersectionPoints n = Nat.choose n 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_convex_ngon_l4143_414321


namespace NUMINAMATH_CALUDE_differential_of_exponential_trig_function_l4143_414347

/-- The differential of y = e^x(cos 2x + 2sin 2x) is dy = 5 e^x cos 2x · dx -/
theorem differential_of_exponential_trig_function (x : ℝ) :
  let y : ℝ → ℝ := λ x => Real.exp x * (Real.cos (2 * x) + 2 * Real.sin (2 * x))
  (deriv y) x = 5 * Real.exp x * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_differential_of_exponential_trig_function_l4143_414347


namespace NUMINAMATH_CALUDE_binomial_expansion_constant_term_l4143_414350

theorem binomial_expansion_constant_term (a b : ℝ) (n : ℕ) :
  (2 : ℝ) ^ n = 4 →
  n = 2 →
  (a + b) ^ n = a ^ 2 + 2 * a * b + 9 :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_constant_term_l4143_414350


namespace NUMINAMATH_CALUDE_equation_in_y_l4143_414359

theorem equation_in_y (x y : ℝ) 
  (eq1 : 3 * x^2 - 5 * x + 4 * y + 6 = 0)
  (eq2 : 3 * x - 2 * y + 1 = 0) :
  4 * y^2 - 2 * y + 24 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_in_y_l4143_414359


namespace NUMINAMATH_CALUDE_abs_neg_three_halves_l4143_414370

theorem abs_neg_three_halves : |(-3/2 : ℚ)| = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_halves_l4143_414370


namespace NUMINAMATH_CALUDE_square_root_sum_equals_abs_sum_l4143_414389

theorem square_root_sum_equals_abs_sum (x : ℝ) : 
  Real.sqrt ((x - 3)^2) + Real.sqrt ((x + 5)^2) = |x - 3| + |x + 5| :=
by sorry

end NUMINAMATH_CALUDE_square_root_sum_equals_abs_sum_l4143_414389


namespace NUMINAMATH_CALUDE_base_conversion_1357_to_base_5_l4143_414355

theorem base_conversion_1357_to_base_5 :
  (2 * 5^4 + 0 * 5^3 + 4 * 5^2 + 1 * 5^1 + 2 * 5^0 : ℕ) = 1357 := by
  sorry

#eval 2 * 5^4 + 0 * 5^3 + 4 * 5^2 + 1 * 5^1 + 2 * 5^0

end NUMINAMATH_CALUDE_base_conversion_1357_to_base_5_l4143_414355


namespace NUMINAMATH_CALUDE_complex_absolute_value_l4143_414341

theorem complex_absolute_value (z : ℂ) : z = 1 + I → Complex.abs (z^2 - 2*z) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l4143_414341


namespace NUMINAMATH_CALUDE_bead_probability_l4143_414308

/-- The probability that a point on a line segment of length 3 is more than 1 unit away from both endpoints is 1/3 -/
theorem bead_probability : 
  let segment_length : ℝ := 3
  let min_distance : ℝ := 1
  let favorable_length : ℝ := segment_length - 2 * min_distance
  favorable_length / segment_length = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_bead_probability_l4143_414308


namespace NUMINAMATH_CALUDE_solve_birthday_money_problem_l4143_414344

def birthday_money_problem (aunt uncle friend1 friend2 friend3 sister : ℝ)
  (mean : ℝ) (total_gifts : ℕ) (unknown_gift : ℝ) : Prop :=
  aunt = 9 ∧
  uncle = 9 ∧
  friend1 = 22 ∧
  friend2 = 22 ∧
  friend3 = 22 ∧
  sister = 7 ∧
  mean = 16.3 ∧
  total_gifts = 7 ∧
  (aunt + uncle + friend1 + unknown_gift + friend2 + friend3 + sister) / total_gifts = mean ∧
  unknown_gift = 23.1

theorem solve_birthday_money_problem :
  ∃ (aunt uncle friend1 friend2 friend3 sister : ℝ)
    (mean : ℝ) (total_gifts : ℕ) (unknown_gift : ℝ),
  birthday_money_problem aunt uncle friend1 friend2 friend3 sister mean total_gifts unknown_gift :=
by sorry

end NUMINAMATH_CALUDE_solve_birthday_money_problem_l4143_414344


namespace NUMINAMATH_CALUDE_clock_angle_division_theorem_l4143_414368

/-- The time when the second hand divides the angle between hour and minute hands -/
def clock_division_time (n : ℕ) (k : ℚ) : ℚ :=
  (43200 * (1 + k) * n) / (719 + 708 * k)

/-- Theorem stating the time when the second hand divides the angle between hour and minute hands -/
theorem clock_angle_division_theorem (n : ℕ) (k : ℚ) :
  let t := clock_division_time n k
  let second_pos := t
  let minute_pos := t / 60
  let hour_pos := t / 720
  (second_pos - hour_pos) / (minute_pos - second_pos) = k ∧
  t = (43200 * (1 + k) * n) / (719 + 708 * k) := by
  sorry


end NUMINAMATH_CALUDE_clock_angle_division_theorem_l4143_414368


namespace NUMINAMATH_CALUDE_distance_between_towns_l4143_414340

theorem distance_between_towns (total_distance : ℝ) : total_distance = 50 :=
  let petya_distance := 10 + (1/4) * (total_distance - 10)
  let kolya_distance := 20 + (1/3) * (total_distance - 20)
  have h1 : petya_distance + kolya_distance = total_distance := by sorry
  have h2 : petya_distance = 10 + (1/4) * (total_distance - 10) := by sorry
  have h3 : kolya_distance = 20 + (1/3) * (total_distance - 20) := by sorry
  sorry

end NUMINAMATH_CALUDE_distance_between_towns_l4143_414340


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_50_5005_l4143_414354

theorem gcd_lcm_sum_50_5005 : 
  Nat.gcd 50 5005 + Nat.lcm 50 5005 = 50055 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_50_5005_l4143_414354


namespace NUMINAMATH_CALUDE_factorization_valid_l4143_414352

theorem factorization_valid (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_valid_l4143_414352


namespace NUMINAMATH_CALUDE_billy_basketball_points_difference_l4143_414307

theorem billy_basketball_points_difference : 
  ∀ (billy_points friend_points : ℕ),
    billy_points = 7 →
    friend_points = 9 →
    friend_points - billy_points = 2 := by
  sorry

end NUMINAMATH_CALUDE_billy_basketball_points_difference_l4143_414307


namespace NUMINAMATH_CALUDE_equation_solution_l4143_414384

theorem equation_solution : ∃ x : ℝ, 2*x + 17 = 32 - 3*x ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l4143_414384


namespace NUMINAMATH_CALUDE_valid_parameterization_l4143_414396

/-- A vector parameterization of a line --/
structure VectorParam where
  v : ℝ × ℝ  -- point vector
  d : ℝ × ℝ  -- direction vector

/-- The line y = 2x - 5 --/
def line (x : ℝ) : ℝ := 2 * x - 5

/-- Check if a point lies on the line --/
def on_line (p : ℝ × ℝ) : Prop :=
  p.2 = line p.1

/-- Check if a vector is a scalar multiple of (1, 2) --/
def is_valid_direction (v : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k, 2 * k)

/-- A parameterization is valid if it satisfies both conditions --/
def is_valid_param (param : VectorParam) : Prop :=
  on_line param.v ∧ is_valid_direction param.d

theorem valid_parameterization (param : VectorParam) :
  is_valid_param param ↔ 
    (∀ (t : ℝ), on_line (param.v.1 + t * param.d.1, param.v.2 + t * param.d.2)) :=
sorry

end NUMINAMATH_CALUDE_valid_parameterization_l4143_414396


namespace NUMINAMATH_CALUDE_farmer_bean_seedlings_per_row_l4143_414302

/-- Represents the farmer's planting scenario -/
structure FarmPlanting where
  bean_seedlings : ℕ
  pumpkin_seeds : ℕ
  pumpkin_per_row : ℕ
  radishes : ℕ
  radishes_per_row : ℕ
  rows_per_bed : ℕ
  plant_beds : ℕ

/-- Calculates the number of bean seedlings per row -/
def bean_seedlings_per_row (fp : FarmPlanting) : ℕ :=
  fp.bean_seedlings / (fp.plant_beds * fp.rows_per_bed - 
    (fp.pumpkin_seeds / fp.pumpkin_per_row + fp.radishes / fp.radishes_per_row))

/-- Theorem stating that given the farmer's planting scenario, 
    the number of bean seedlings per row is 8 -/
theorem farmer_bean_seedlings_per_row :
  let fp : FarmPlanting := {
    bean_seedlings := 64,
    pumpkin_seeds := 84,
    pumpkin_per_row := 7,
    radishes := 48,
    radishes_per_row := 6,
    rows_per_bed := 2,
    plant_beds := 14
  }
  bean_seedlings_per_row fp = 8 := by
  sorry

end NUMINAMATH_CALUDE_farmer_bean_seedlings_per_row_l4143_414302


namespace NUMINAMATH_CALUDE_atomic_number_difference_l4143_414381

/-- Represents an element in the periodic table -/
structure Element where
  atomicNumber : ℕ

/-- Represents a main group in the periodic table -/
structure MainGroup where
  elements : Set Element

/-- 
  Given two elements A and B in the same main group of the periodic table, 
  where the atomic number of A is x, the atomic number of B cannot be x+4.
-/
theorem atomic_number_difference (g : MainGroup) (A B : Element) (x : ℕ) :
  A ∈ g.elements → B ∈ g.elements → A.atomicNumber = x → 
  B.atomicNumber ≠ x + 4 := by
  sorry

end NUMINAMATH_CALUDE_atomic_number_difference_l4143_414381


namespace NUMINAMATH_CALUDE_fraction_equality_l4143_414306

theorem fraction_equality (a b c : ℝ) (h1 : c ≠ 0) :
  a / c = b / c → a = b := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l4143_414306


namespace NUMINAMATH_CALUDE_weight_of_grapes_l4143_414366

/-- Given the weights of fruits ordered by Tommy, prove the weight of grapes. -/
theorem weight_of_grapes (total weight_apples weight_oranges weight_strawberries : ℕ) 
  (h_total : total = 10)
  (h_apples : weight_apples = 3)
  (h_oranges : weight_oranges = 1)
  (h_strawberries : weight_strawberries = 3) :
  total - (weight_apples + weight_oranges + weight_strawberries) = 3 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_grapes_l4143_414366


namespace NUMINAMATH_CALUDE_douglas_county_y_votes_l4143_414392

/-- Represents the percentage of votes Douglas won in county Y -/
def douglas_county_y_percentage : ℝ := 46

/-- Represents the total percentage of votes Douglas won in both counties -/
def total_percentage : ℝ := 58

/-- Represents the percentage of votes Douglas won in county X -/
def douglas_county_x_percentage : ℝ := 64

/-- Represents the ratio of voters in county X to county Y -/
def county_ratio : ℚ := 2 / 1

theorem douglas_county_y_votes :
  douglas_county_y_percentage = 
    (3 * total_percentage - 2 * douglas_county_x_percentage) := by sorry

end NUMINAMATH_CALUDE_douglas_county_y_votes_l4143_414392


namespace NUMINAMATH_CALUDE_range_of_m_l4143_414310

/-- The function f(x) defined as 1 / √(mx² + mx + 1) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 1 / Real.sqrt (m * x^2 + m * x + 1)

/-- The set of real numbers m for which f(x) has domain ℝ -/
def valid_m : Set ℝ := {m : ℝ | ∀ x : ℝ, m * x^2 + m * x + 1 > 0}

theorem range_of_m : valid_m = Set.Ici 0 ∩ Set.Iio 4 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l4143_414310


namespace NUMINAMATH_CALUDE_original_number_proof_l4143_414376

theorem original_number_proof (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 30 →
  (a + b + c + 50) / 4 = 40 →
  d = 10 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l4143_414376


namespace NUMINAMATH_CALUDE_f_2012_equals_cos_l4143_414377

noncomputable def f : ℕ → (ℝ → ℝ)
| 0 => λ x => Real.cos x
| (n + 1) => λ x => deriv (f n) x

theorem f_2012_equals_cos : f 2012 = λ x => Real.cos x := by sorry

end NUMINAMATH_CALUDE_f_2012_equals_cos_l4143_414377


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4143_414374

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (x > y ∧ y > 0 → x^2 > y^2) ∧
  ∃ x y : ℝ, x^2 > y^2 ∧ ¬(x > y ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4143_414374


namespace NUMINAMATH_CALUDE_student_subtraction_problem_l4143_414358

theorem student_subtraction_problem (x y : ℤ) : 
  x = 40 → 7 * x - y = 130 → y = 150 := by sorry

end NUMINAMATH_CALUDE_student_subtraction_problem_l4143_414358


namespace NUMINAMATH_CALUDE_gasoline_price_increase_l4143_414379

theorem gasoline_price_increase (initial_price initial_quantity : ℝ) 
  (h_price_increase : ℝ) (h_spending_increase : ℝ) (h_quantity_decrease : ℝ) :
  h_price_increase > 0 →
  h_spending_increase = 0.1 →
  h_quantity_decrease = 0.12 →
  initial_price * initial_quantity * (1 + h_spending_increase) = 
    initial_price * (1 + h_price_increase) * initial_quantity * (1 - h_quantity_decrease) →
  h_price_increase = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_price_increase_l4143_414379


namespace NUMINAMATH_CALUDE_root_equation_solution_l4143_414314

theorem root_equation_solution (a : ℝ) : 
  (2 : ℝ)^2 + a * 2 - 3 * a = 0 → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_solution_l4143_414314


namespace NUMINAMATH_CALUDE_largest_domain_of_g_l4143_414328

/-- A function g satisfying the given property -/
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → g x + g (1 / x^2) = x^2

/-- The domain of g is the set of all non-zero real numbers -/
theorem largest_domain_of_g :
  ∃ g : ℝ → ℝ, g_property g ∧
  ∀ S : Set ℝ, (∃ h : ℝ → ℝ, g_property h ∧ ∀ x ∈ S, h x ≠ 0) →
  S ⊆ {x : ℝ | x ≠ 0} :=
sorry

end NUMINAMATH_CALUDE_largest_domain_of_g_l4143_414328


namespace NUMINAMATH_CALUDE_paving_cost_calculation_l4143_414325

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a rectangular floor with given dimensions and rate -/
theorem paving_cost_calculation (length width rate : ℝ) 
  (h1 : length = 5.5)
  (h2 : width = 3.75)
  (h3 : rate = 1000) :
  paving_cost length width rate = 20625 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_calculation_l4143_414325


namespace NUMINAMATH_CALUDE_garrison_provision_problem_l4143_414309

/-- Calculates the initial number of days provisions would last for a garrison --/
def initial_provision_days (initial_men : ℕ) (reinforcement_men : ℕ) (days_before_reinforcement : ℕ) (days_after_reinforcement : ℕ) : ℕ :=
  (initial_men * days_before_reinforcement + (initial_men + reinforcement_men) * days_after_reinforcement) / initial_men

theorem garrison_provision_problem :
  initial_provision_days 1850 1110 12 10 = 28 := by
  sorry

end NUMINAMATH_CALUDE_garrison_provision_problem_l4143_414309


namespace NUMINAMATH_CALUDE_workshop_workers_l4143_414371

theorem workshop_workers (total_average : ℝ) (tech_count : ℕ) (tech_average : ℝ) (nontech_average : ℝ) :
  total_average = 6750 →
  tech_count = 7 →
  tech_average = 12000 →
  nontech_average = 6000 →
  ∃ (total_workers : ℕ), total_workers = 56 ∧ 
    total_average * total_workers = tech_average * tech_count + nontech_average * (total_workers - tech_count) :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_l4143_414371


namespace NUMINAMATH_CALUDE_pretzels_eaten_difference_l4143_414318

/-- The number of pretzels Marcus ate compared to John -/
def pretzels_difference (total : ℕ) (john : ℕ) (alan : ℕ) (marcus : ℕ) : ℕ :=
  marcus - john

/-- Theorem stating the difference in pretzels eaten between Marcus and John -/
theorem pretzels_eaten_difference 
  (total : ℕ) 
  (john : ℕ) 
  (alan : ℕ) 
  (marcus : ℕ) 
  (h1 : total = 95)
  (h2 : john = 28)
  (h3 : alan = john - 9)
  (h4 : marcus > john)
  (h5 : marcus = 40) :
  pretzels_difference total john alan marcus = 12 := by
  sorry

end NUMINAMATH_CALUDE_pretzels_eaten_difference_l4143_414318


namespace NUMINAMATH_CALUDE_exp_ln_eight_equals_eight_l4143_414323

theorem exp_ln_eight_equals_eight : Real.exp (Real.log 8) = 8 := by sorry

end NUMINAMATH_CALUDE_exp_ln_eight_equals_eight_l4143_414323


namespace NUMINAMATH_CALUDE_fraction_multiplication_l4143_414338

theorem fraction_multiplication : (2 : ℚ) / 3 * 4 / 7 * 9 / 11 = 24 / 77 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l4143_414338


namespace NUMINAMATH_CALUDE_ice_cream_revenue_l4143_414315

/-- Calculate the total revenue from ice cream sales with discounts --/
theorem ice_cream_revenue : 
  let chocolate : ℕ := 50
  let mango : ℕ := 54
  let vanilla : ℕ := 80
  let strawberry : ℕ := 40
  let price : ℚ := 2
  let chocolate_sold : ℚ := 3 / 5 * chocolate
  let mango_sold : ℚ := 2 / 3 * mango
  let vanilla_sold : ℚ := 75 / 100 * vanilla
  let strawberry_sold : ℚ := 5 / 8 * strawberry
  let discount : ℚ := 1 / 2
  let apply_discount (x : ℚ) : ℚ := if x ≥ 10 then x * discount else 0

  let total_revenue : ℚ := 
    (chocolate_sold + mango_sold + vanilla_sold + strawberry_sold) * price - 
    (apply_discount chocolate_sold + apply_discount mango_sold + 
     apply_discount vanilla_sold + apply_discount strawberry_sold)

  total_revenue = 226.5 := by sorry

end NUMINAMATH_CALUDE_ice_cream_revenue_l4143_414315


namespace NUMINAMATH_CALUDE_fraction_simplification_l4143_414380

theorem fraction_simplification : 
  (3 + 9 - 27 + 81 + 243 - 729) / (9 + 27 - 81 + 243 + 729 - 2187) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4143_414380


namespace NUMINAMATH_CALUDE_intersection_sum_l4143_414378

theorem intersection_sum (a b : ℚ) : 
  (2 = (1/3) * 3 + a) → 
  (3 = (1/5) * 2 + b) → 
  a + b = 18/5 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l4143_414378


namespace NUMINAMATH_CALUDE_division_of_composite_products_l4143_414333

-- Define the first six positive composite integers
def first_six_composites : List Nat := [4, 6, 8, 9, 10, 12]

-- Define the product of the first three composite integers
def product_first_three : Nat := (first_six_composites.take 3).prod

-- Define the product of the next three composite integers
def product_next_three : Nat := (first_six_composites.drop 3).prod

-- Theorem to prove
theorem division_of_composite_products :
  (product_first_three : ℚ) / product_next_three = 8 / 45 := by
  sorry

end NUMINAMATH_CALUDE_division_of_composite_products_l4143_414333


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l4143_414386

def a : ℝ × ℝ := (2, 1)
def b : ℝ → ℝ × ℝ := λ x ↦ (x, -2)

theorem parallel_vectors_sum (x : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ a = k • (b x)) →
  a + b x = (-2, -1) := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l4143_414386


namespace NUMINAMATH_CALUDE_teacher_selection_probability_l4143_414303

/-- Represents a university department -/
structure Department where
  name : String
  teachers : ℕ

/-- Represents a university -/
structure University where
  departments : List Department

/-- Calculates the total number of teachers in a university -/
def totalTeachers (u : University) : ℕ :=
  u.departments.map (·.teachers) |>.sum

/-- Calculates the probability of selecting an individual teacher -/
def selectionProbability (u : University) (numSelected : ℕ) : ℚ :=
  numSelected / (totalTeachers u)

/-- Theorem stating the probability of selecting an individual teacher -/
theorem teacher_selection_probability
  (u : University)
  (hDepartments : u.departments = [
    ⟨"A", 10⟩,
    ⟨"B", 20⟩,
    ⟨"C", 30⟩
  ])
  (hNumSelected : numSelected = 6) :
  selectionProbability u numSelected = 1 / 10 := by
  sorry


end NUMINAMATH_CALUDE_teacher_selection_probability_l4143_414303


namespace NUMINAMATH_CALUDE_solution_set_l4143_414361

def system_solution (x y : ℝ) : Prop :=
  x + y = 20 ∧ Real.log x / Real.log 4 + Real.log y / Real.log 4 = 1 + Real.log 9 / Real.log 4

theorem solution_set : 
  {(x, y) : ℝ × ℝ | system_solution x y} = {(18, 2), (2, 18)} := by sorry

end NUMINAMATH_CALUDE_solution_set_l4143_414361


namespace NUMINAMATH_CALUDE_sum_in_base_5_l4143_414369

/-- Given a base b, returns the value of a number in base 10 -/
def toBase10 (x : ℕ) (b : ℕ) : ℕ := sorry

/-- Given a base b, returns the square of a number in base b -/
def squareInBase (x : ℕ) (b : ℕ) : ℕ := sorry

/-- Given a base b, returns the sum of three numbers in base b -/
def sumInBase (x y z : ℕ) (b : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to another base -/
def fromBase10 (x : ℕ) (b : ℕ) : ℕ := sorry

theorem sum_in_base_5 (b : ℕ) : 
  (squareInBase 14 b + squareInBase 18 b + squareInBase 20 b = toBase10 2850 b) →
  (fromBase10 (sumInBase 14 18 20 b) 5 = 62) := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base_5_l4143_414369


namespace NUMINAMATH_CALUDE_multiple_problem_l4143_414330

theorem multiple_problem (x : ℝ) (m : ℝ) (h1 : x = -4.5) (h2 : 10 * x = m * x - 36) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiple_problem_l4143_414330


namespace NUMINAMATH_CALUDE_stationary_rigid_body_l4143_414346

/-- A point in a two-dimensional plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A rigid body in a two-dimensional plane -/
structure RigidBody2D where
  points : Set Point2D

/-- Three points are non-collinear if they do not lie on the same straight line -/
def NonCollinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) ≠ (p3.x - p1.x) * (p2.y - p1.y)

/-- A rigid body is stationary if it has no translational or rotational motion -/
def IsStationary (body : RigidBody2D) : Prop :=
  ∃ (p1 p2 p3 : Point2D), p1 ∈ body.points ∧ p2 ∈ body.points ∧ p3 ∈ body.points ∧
    NonCollinear p1 p2 p3

theorem stationary_rigid_body (body : RigidBody2D) :
  IsStationary body ↔ ∃ (p1 p2 p3 : Point2D), p1 ∈ body.points ∧ p2 ∈ body.points ∧ p3 ∈ body.points ∧
    NonCollinear p1 p2 p3 :=
  sorry

end NUMINAMATH_CALUDE_stationary_rigid_body_l4143_414346


namespace NUMINAMATH_CALUDE_rationalize_denominator_l4143_414375

theorem rationalize_denominator : (3 : ℝ) / Real.sqrt 75 = Real.sqrt 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l4143_414375


namespace NUMINAMATH_CALUDE_contradiction_assumption_l4143_414360

theorem contradiction_assumption (a b : ℝ) : 
  (¬(a > b → 3*a > 3*b) ↔ 3*a ≤ 3*b) :=
by sorry

end NUMINAMATH_CALUDE_contradiction_assumption_l4143_414360


namespace NUMINAMATH_CALUDE_final_price_percentage_l4143_414362

/-- Calculates the final price after applying multiple discounts -/
def final_price (original_price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (λ price discount => price * (1 - discount)) original_price

/-- Theorem stating that after applying the given discounts, the final price is 58.14% of the original -/
theorem final_price_percentage (original_price : ℝ) (original_price_pos : 0 < original_price) :
  let discounts := [0.2, 0.1, 0.05, 0.15]
  final_price original_price discounts / original_price = 0.5814 := by
sorry

#eval (final_price 100 [0.2, 0.1, 0.05, 0.15])

end NUMINAMATH_CALUDE_final_price_percentage_l4143_414362


namespace NUMINAMATH_CALUDE_complement_of_angle_A_l4143_414385

-- Define the angle A
def angle_A : ℝ := 55

-- Define the complement of an angle
def complement (angle : ℝ) : ℝ := 90 - angle

-- Theorem statement
theorem complement_of_angle_A :
  complement angle_A = 35 := by sorry

end NUMINAMATH_CALUDE_complement_of_angle_A_l4143_414385


namespace NUMINAMATH_CALUDE_inequality_solution_range_l4143_414390

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, |x| + |x - 1| ≤ a) → a ∈ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l4143_414390


namespace NUMINAMATH_CALUDE_complex_addition_l4143_414382

theorem complex_addition : (6 - 5*Complex.I) + (3 + 2*Complex.I) = 9 - 3*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_addition_l4143_414382


namespace NUMINAMATH_CALUDE_jongkook_total_points_total_questions_sum_l4143_414300

/-- The number of English problems solved by each student -/
def total_problems : ℕ := 18

/-- The number of 6-point questions Jongkook got correct -/
def correct_six_point : ℕ := 8

/-- The number of 5-point questions Jongkook got correct -/
def correct_five_point : ℕ := 6

/-- The point value of the first type of question -/
def points_type_one : ℕ := 6

/-- The point value of the second type of question -/
def points_type_two : ℕ := 5

/-- Theorem stating that Jongkook's total points is 78 -/
theorem jongkook_total_points :
  correct_six_point * points_type_one + correct_five_point * points_type_two = 78 := by
  sorry

/-- Theorem stating that the sum of correct questions equals the total number of problems -/
theorem total_questions_sum :
  correct_six_point + correct_five_point + (total_problems - correct_six_point - correct_five_point) = total_problems := by
  sorry

end NUMINAMATH_CALUDE_jongkook_total_points_total_questions_sum_l4143_414300


namespace NUMINAMATH_CALUDE_candy_distribution_l4143_414313

theorem candy_distribution (k : ℕ) : 
  (∃ q : ℕ, k = 7 * q + 3) → 
  (∃ r : ℕ, 3 * k = 7 * r + 2) :=
by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l4143_414313


namespace NUMINAMATH_CALUDE_board_cut_theorem_l4143_414301

theorem board_cut_theorem (total_length : ℝ) (shorter_piece : ℝ) : 
  total_length = 69 →
  total_length = shorter_piece + 2 * shorter_piece →
  shorter_piece = 23 := by
  sorry

end NUMINAMATH_CALUDE_board_cut_theorem_l4143_414301


namespace NUMINAMATH_CALUDE_age_difference_l4143_414324

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 11) : a = c + 11 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l4143_414324


namespace NUMINAMATH_CALUDE_prob_two_girls_from_five_l4143_414383

/-- The probability of selecting 2 girls as representatives from a group of 5 students (2 boys and 3 girls) is 3/10. -/
theorem prob_two_girls_from_five (total : ℕ) (boys : ℕ) (girls : ℕ) (representatives : ℕ) :
  total = 5 →
  boys = 2 →
  girls = 3 →
  representatives = 2 →
  (Nat.choose girls representatives : ℚ) / (Nat.choose total representatives : ℚ) = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_prob_two_girls_from_five_l4143_414383


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l4143_414327

/-- A trapezoid with an inscribed circle -/
structure InscribedCircleTrapezoid where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of BM, where M is the point where the circle touches AB -/
  bm : ℝ
  /-- The length of the top side CD -/
  cd : ℝ

/-- The area of a trapezoid with an inscribed circle -/
def trapezoidArea (t : InscribedCircleTrapezoid) : ℝ :=
  sorry

/-- Theorem: The area of the specific trapezoid is 108 -/
theorem specific_trapezoid_area :
  ∃ t : InscribedCircleTrapezoid, t.r = 4 ∧ t.bm = 16 ∧ t.cd = 3 ∧ trapezoidArea t = 108 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l4143_414327


namespace NUMINAMATH_CALUDE_microphotonics_allocation_l4143_414332

/-- Represents the budget allocation for Megatech Corporation -/
structure BudgetAllocation where
  total : ℝ
  home_electronics : ℝ
  food_additives : ℝ
  genetically_modified_microorganisms : ℝ
  industrial_lubricants : ℝ
  basic_astrophysics_degrees : ℝ
  microphotonics : ℝ

/-- The theorem stating that the microphotonics allocation is 14% given the conditions -/
theorem microphotonics_allocation
  (budget : BudgetAllocation)
  (h1 : budget.total = 100)
  (h2 : budget.home_electronics = 19)
  (h3 : budget.food_additives = 10)
  (h4 : budget.genetically_modified_microorganisms = 24)
  (h5 : budget.industrial_lubricants = 8)
  (h6 : budget.basic_astrophysics_degrees = 90)
  (h7 : budget.microphotonics = budget.total - (budget.home_electronics + budget.food_additives + budget.genetically_modified_microorganisms + budget.industrial_lubricants + (budget.basic_astrophysics_degrees / 360 * 100))) :
  budget.microphotonics = 14 := by
  sorry

end NUMINAMATH_CALUDE_microphotonics_allocation_l4143_414332


namespace NUMINAMATH_CALUDE_curve_M_properties_l4143_414394

-- Define the curve M
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1^(1/2) + p.2^(1/2) = 1}

-- Theorem statement
theorem curve_M_properties :
  (∃ (p : ℝ × ℝ), p ∈ M ∧ Real.sqrt (p.1^2 + p.2^2) < Real.sqrt 2 / 2) ∧
  (∀ (S : Set (ℝ × ℝ)), S ⊆ M → MeasureTheory.volume S ≤ 1/2) := by
  sorry

end NUMINAMATH_CALUDE_curve_M_properties_l4143_414394


namespace NUMINAMATH_CALUDE_cloth_cost_price_theorem_l4143_414334

/-- Calculates the cost price per meter of cloth given the total length,
    selling price, and profit per meter. -/
def costPricePerMeter (totalLength : ℕ) (sellingPrice : ℕ) (profitPerMeter : ℕ) : ℚ :=
  (sellingPrice - totalLength * profitPerMeter) / totalLength

/-- Theorem stating that for the given conditions, the cost price per meter is 86. -/
theorem cloth_cost_price_theorem (totalLength : ℕ) (sellingPrice : ℕ) (profitPerMeter : ℕ)
    (h1 : totalLength = 45)
    (h2 : sellingPrice = 4500)
    (h3 : profitPerMeter = 14) :
    costPricePerMeter totalLength sellingPrice profitPerMeter = 86 := by
  sorry

#eval costPricePerMeter 45 4500 14

end NUMINAMATH_CALUDE_cloth_cost_price_theorem_l4143_414334


namespace NUMINAMATH_CALUDE_pizza_slices_l4143_414335

theorem pizza_slices (total_pizzas : ℕ) (total_slices : ℕ) (slices_per_pizza : ℕ) 
  (h1 : total_pizzas = 7)
  (h2 : total_slices = 14)
  (h3 : total_slices = total_pizzas * slices_per_pizza) :
  slices_per_pizza = 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_l4143_414335


namespace NUMINAMATH_CALUDE_intersection_M_N_l4143_414342

def M : Set ℝ := {x | 2 * x - x^2 > 0}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l4143_414342


namespace NUMINAMATH_CALUDE_bracelet_bead_ratio_l4143_414395

/-- Proves that the ratio of small beads to large beads in each bracelet is 1:1 --/
theorem bracelet_bead_ratio
  (total_beads : ℕ)
  (bracelets : ℕ)
  (large_beads_per_bracelet : ℕ)
  (h1 : total_beads = 528)
  (h2 : bracelets = 11)
  (h3 : large_beads_per_bracelet = 12)
  (h4 : total_beads % 2 = 0)  -- Equal amounts of small and large beads
  (h5 : (total_beads / 2) ≥ (bracelets * large_beads_per_bracelet)) :
  (total_beads / 2 - bracelets * large_beads_per_bracelet) / bracelets = large_beads_per_bracelet :=
by sorry

end NUMINAMATH_CALUDE_bracelet_bead_ratio_l4143_414395


namespace NUMINAMATH_CALUDE_guitar_center_shipping_fee_l4143_414305

/-- The shipping fee of Guitar Center given the conditions of the guitar purchase --/
theorem guitar_center_shipping_fee :
  let suggested_price : ℚ := 1000
  let guitar_center_discount : ℚ := 15 / 100
  let sweetwater_discount : ℚ := 10 / 100
  let savings : ℚ := 50
  let guitar_center_price := suggested_price * (1 - guitar_center_discount)
  let sweetwater_price := suggested_price * (1 - sweetwater_discount)
  guitar_center_price + (sweetwater_price - guitar_center_price - savings) = guitar_center_price :=
by sorry

end NUMINAMATH_CALUDE_guitar_center_shipping_fee_l4143_414305


namespace NUMINAMATH_CALUDE_reduced_oil_price_l4143_414364

/-- Represents the price reduction scenario for oil --/
structure OilPriceReduction where
  original_price : ℝ
  reduced_price : ℝ
  price_reduction_percent : ℝ
  additional_amount : ℝ
  fixed_cost : ℝ

/-- Theorem stating the reduced price of oil given the conditions --/
theorem reduced_oil_price 
  (scenario : OilPriceReduction)
  (h1 : scenario.price_reduction_percent = 20)
  (h2 : scenario.additional_amount = 4)
  (h3 : scenario.fixed_cost = 600)
  (h4 : scenario.reduced_price = scenario.original_price * (1 - scenario.price_reduction_percent / 100))
  (h5 : scenario.fixed_cost = (scenario.fixed_cost / scenario.original_price + scenario.additional_amount) * scenario.reduced_price) :
  scenario.reduced_price = 30 := by
  sorry

#check reduced_oil_price

end NUMINAMATH_CALUDE_reduced_oil_price_l4143_414364


namespace NUMINAMATH_CALUDE_parabola_normal_min_area_l4143_414320

noncomputable def min_y_coordinate : ℝ := (-3 + Real.sqrt 33) / 24

theorem parabola_normal_min_area (x₀ : ℝ) :
  let y₀ := x₀^2
  let normal_slope := -1 / (2 * x₀)
  let x₁ := -1 / (2 * x₀) - x₀
  let y₁ := x₁^2
  let triangle_area := (1/2) * (x₀ - x₁) * (y₀ + 1/2)
  (∀ x : ℝ, triangle_area ≤ ((1/2) * (x - (-1 / (2 * x) - x)) * (x^2 + 1/2))) →
  y₀ = min_y_coordinate := by
sorry

end NUMINAMATH_CALUDE_parabola_normal_min_area_l4143_414320


namespace NUMINAMATH_CALUDE_choir_average_age_l4143_414343

theorem choir_average_age 
  (num_females : ℕ) 
  (num_males : ℕ) 
  (avg_age_females : ℚ) 
  (avg_age_males : ℚ) 
  (h1 : num_females = 8)
  (h2 : num_males = 17)
  (h3 : avg_age_females = 28)
  (h4 : avg_age_males = 32) :
  let total_people := num_females + num_males
  let total_age := num_females * avg_age_females + num_males * avg_age_males
  total_age / total_people = 768 / 25 := by
sorry

end NUMINAMATH_CALUDE_choir_average_age_l4143_414343


namespace NUMINAMATH_CALUDE_unique_perfect_square_and_cube_factor_of_1800_l4143_414353

/-- A number is a perfect square if it's equal to some integer squared. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2

/-- A number is a perfect cube if it's equal to some integer cubed. -/
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^3

/-- A number is both a perfect square and a perfect cube. -/
def is_perfect_square_and_cube (n : ℕ) : Prop :=
  is_perfect_square n ∧ is_perfect_cube n

/-- The set of positive factors of a natural number. -/
def positive_factors (n : ℕ) : Set ℕ :=
  {k : ℕ | k > 0 ∧ n % k = 0}

/-- There is exactly one positive factor of 1800 that is both a perfect square and a perfect cube. -/
theorem unique_perfect_square_and_cube_factor_of_1800 :
  ∃! x : ℕ, x ∈ positive_factors 1800 ∧ is_perfect_square_and_cube x :=
sorry

end NUMINAMATH_CALUDE_unique_perfect_square_and_cube_factor_of_1800_l4143_414353


namespace NUMINAMATH_CALUDE_circles_intersect_l4143_414399

-- Define the circles
def C1 (x y : ℝ) : Prop := (x + 2)^2 + (y - 2)^2 = 4
def C2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 5)^2 = 16

-- Define the centers and radii
def center1 : ℝ × ℝ := (-2, 2)
def center2 : ℝ × ℝ := (2, 5)
def radius1 : ℝ := 2
def radius2 : ℝ := 4

-- Theorem statement
theorem circles_intersect :
  let d := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  radius1 + radius2 > d ∧ d > abs (radius1 - radius2) := by sorry

end NUMINAMATH_CALUDE_circles_intersect_l4143_414399


namespace NUMINAMATH_CALUDE_handshake_pigeonhole_l4143_414322

theorem handshake_pigeonhole (n : ℕ) (h : n > 0) :
  ∃ (i j : Fin n) (k : ℕ), i ≠ j ∧ 
  (∃ (f : Fin n → ℕ), f i = k ∧ f j = k ∧ ∀ m, f m < n) :=
sorry

end NUMINAMATH_CALUDE_handshake_pigeonhole_l4143_414322


namespace NUMINAMATH_CALUDE_cost_price_of_article_l4143_414391

/-- The cost price of an article satisfying certain selling price conditions -/
theorem cost_price_of_article : ∃ C : ℝ, 
  (C = 400) ∧ 
  (0.8 * C = C - 0.2 * C) ∧ 
  (1.05 * C = C + 0.05 * C) ∧ 
  (1.05 * C - 0.8 * C = 100) := by
  sorry

end NUMINAMATH_CALUDE_cost_price_of_article_l4143_414391


namespace NUMINAMATH_CALUDE_sector_area_120_deg_sqrt3_radius_l4143_414356

/-- The area of a circular sector with central angle 120° and radius √3 is equal to π. -/
theorem sector_area_120_deg_sqrt3_radius (π : ℝ) : 
  let angle : ℝ := 2 * π / 3  -- 120° in radians
  let radius : ℝ := Real.sqrt 3
  let sector_area : ℝ := (1 / 2) * angle * radius^2
  sector_area = π :=
by sorry

end NUMINAMATH_CALUDE_sector_area_120_deg_sqrt3_radius_l4143_414356


namespace NUMINAMATH_CALUDE_max_cars_quotient_l4143_414397

/-- Represents the maximum number of cars that can pass a point on the highway in one hour -/
def M : ℕ :=
  -- Definition to be proved
  2000

/-- The length of each car in meters -/
def car_length : ℝ := 5

/-- Theorem stating that M divided by 10 equals 200 -/
theorem max_cars_quotient :
  M / 10 = 200 := by sorry

end NUMINAMATH_CALUDE_max_cars_quotient_l4143_414397
