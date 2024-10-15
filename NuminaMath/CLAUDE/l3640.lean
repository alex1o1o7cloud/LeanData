import Mathlib

namespace NUMINAMATH_CALUDE_largest_n_for_inequality_l3640_364003

theorem largest_n_for_inequality (z : ℕ) (h : z = 9) :
  ∃ n : ℕ, (27 ^ z > 3 ^ n ∧ ∀ m : ℕ, m > n → 27 ^ z ≤ 3 ^ m) ∧ n = 26 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_inequality_l3640_364003


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l3640_364051

-- Define the hyperbola C
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the right vertex A
def right_vertex (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the circle centered at A
def circle_at_A (a r : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + y^2 = r^2

-- Define the angle PAQ
def angle_PAQ (p q : ℝ × ℝ) : ℝ := sorry

-- Define the distance PQ
def distance_PQ (p q : ℝ × ℝ) : ℝ := sorry

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * x ∨ y = -(Real.sqrt 3 / 3) * x

theorem hyperbola_asymptote
  (a b : ℝ)
  (p q : ℝ × ℝ)
  (h1 : hyperbola a b p.1 p.2)
  (h2 : hyperbola a b q.1 q.2)
  (h3 : ∃ r, circle_at_A a r p.1 p.2 ∧ circle_at_A a r q.1 q.2)
  (h4 : angle_PAQ p q = Real.pi / 3)
  (h5 : distance_PQ p q = Real.sqrt 3 / 3 * a) :
  asymptote_equation p.1 p.2 ∧ asymptote_equation q.1 q.2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l3640_364051


namespace NUMINAMATH_CALUDE_total_spent_is_14_l3640_364064

/-- The cost of one set of barrettes in dollars -/
def barrette_cost : ℕ := 3

/-- The cost of one comb in dollars -/
def comb_cost : ℕ := 1

/-- The number of barrette sets Kristine buys -/
def kristine_barrettes : ℕ := 1

/-- The number of combs Kristine buys -/
def kristine_combs : ℕ := 1

/-- The number of barrette sets Crystal buys -/
def crystal_barrettes : ℕ := 3

/-- The number of combs Crystal buys -/
def crystal_combs : ℕ := 1

/-- The total amount spent by both Kristine and Crystal -/
def total_spent : ℕ := 
  (kristine_barrettes * barrette_cost + kristine_combs * comb_cost) +
  (crystal_barrettes * barrette_cost + crystal_combs * comb_cost)

theorem total_spent_is_14 : total_spent = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_14_l3640_364064


namespace NUMINAMATH_CALUDE_parabola_intersection_difference_l3640_364093

/-- The difference between the larger and smaller x-coordinates of the intersection points of two parabolas -/
theorem parabola_intersection_difference : ∃ (a c : ℝ),
  (∀ x y : ℝ, y = 3 * x^2 - 6 * x + 3 ↔ y = -2 * x^2 + x + 5 → x = a ∨ x = c) ∧
  c ≥ a ∧
  c - a = Real.sqrt 89 / 5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_difference_l3640_364093


namespace NUMINAMATH_CALUDE_complementary_angles_imply_right_triangle_l3640_364013

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180

-- Define complementary angles
def complementary (a b : ℝ) : Prop := a + b = 90

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop :=
  ∃ i : Fin 3, t.angles i = 90

-- Theorem statement
theorem complementary_angles_imply_right_triangle (t : Triangle) 
  (h : ∃ i j : Fin 3, i ≠ j ∧ complementary (t.angles i) (t.angles j)) : 
  is_right_triangle t := by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_imply_right_triangle_l3640_364013


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3640_364008

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ -1}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = N := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3640_364008


namespace NUMINAMATH_CALUDE_giant_slide_rides_count_l3640_364034

/-- Represents the carnival scenario with given ride times and planned rides --/
structure CarnivalScenario where
  total_time : ℕ  -- Total time in minutes
  roller_coaster_time : ℕ
  tilt_a_whirl_time : ℕ
  giant_slide_time : ℕ
  vortex_time : ℕ
  bumper_cars_time : ℕ
  roller_coaster_rides : ℕ
  tilt_a_whirl_rides : ℕ
  vortex_rides : ℕ
  bumper_cars_rides : ℕ

/-- Theorem stating that the number of giant slide rides is equal to tilt-a-whirl rides --/
theorem giant_slide_rides_count (scenario : CarnivalScenario) : 
  scenario.total_time = 240 ∧
  scenario.roller_coaster_time = 30 ∧
  scenario.tilt_a_whirl_time = 60 ∧
  scenario.giant_slide_time = 15 ∧
  scenario.vortex_time = 45 ∧
  scenario.bumper_cars_time = 25 ∧
  scenario.roller_coaster_rides = 4 ∧
  scenario.tilt_a_whirl_rides = 2 ∧
  scenario.vortex_rides = 1 ∧
  scenario.bumper_cars_rides = 3 →
  scenario.tilt_a_whirl_rides = 2 :=
by sorry

end NUMINAMATH_CALUDE_giant_slide_rides_count_l3640_364034


namespace NUMINAMATH_CALUDE_ice_cream_sales_theorem_l3640_364091

/-- Represents the ice cream cone sales scenario -/
structure IceCreamSales where
  free_cone_interval : Nat  -- Every nth customer gets a free cone
  cone_price : Nat          -- Price of each cone in dollars
  free_cones_given : Nat    -- Number of free cones given away

/-- Calculates the total sales amount for the ice cream cones -/
def calculate_sales (sales : IceCreamSales) : Nat :=
  sorry

/-- Theorem stating that given the conditions, the sales amount is $100 -/
theorem ice_cream_sales_theorem (sales : IceCreamSales) 
  (h1 : sales.free_cone_interval = 6)
  (h2 : sales.cone_price = 2)
  (h3 : sales.free_cones_given = 10) : 
  calculate_sales sales = 100 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sales_theorem_l3640_364091


namespace NUMINAMATH_CALUDE_daves_apps_l3640_364092

theorem daves_apps (initial_files : ℕ) (final_apps : ℕ) (final_files : ℕ) (deleted_apps : ℕ) :
  initial_files = 77 →
  final_apps = 5 →
  final_files = 23 →
  deleted_apps = 11 →
  final_apps + deleted_apps = 16 := by
  sorry

end NUMINAMATH_CALUDE_daves_apps_l3640_364092


namespace NUMINAMATH_CALUDE_fish_population_after_bobbit_worm_l3640_364010

/-- Calculates the number of fish remaining in James' aquarium when he discovers the Bobbit worm -/
theorem fish_population_after_bobbit_worm 
  (initial_fish : ℕ) 
  (daily_eaten : ℕ) 
  (days_before_adding : ℕ) 
  (fish_added : ℕ) 
  (total_days : ℕ) 
  (h1 : initial_fish = 60)
  (h2 : daily_eaten = 2)
  (h3 : days_before_adding = 14)
  (h4 : fish_added = 8)
  (h5 : total_days = 21) :
  initial_fish - (daily_eaten * total_days) + fish_added = 26 :=
sorry

end NUMINAMATH_CALUDE_fish_population_after_bobbit_worm_l3640_364010


namespace NUMINAMATH_CALUDE_proposition_and_converse_l3640_364074

theorem proposition_and_converse (a b : ℝ) :
  (a + b ≥ 2 → max a b ≥ 1) ∧
  ¬(max a b ≥ 1 → a + b ≥ 2) := by
sorry

end NUMINAMATH_CALUDE_proposition_and_converse_l3640_364074


namespace NUMINAMATH_CALUDE_tug_of_war_competition_l3640_364002

/-- Calculates the number of matches in a tug-of-war competition -/
def number_of_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculates the number of matches for each class in a tug-of-war competition -/
def matches_per_class (n : ℕ) : ℕ := n - 1

theorem tug_of_war_competition (n : ℕ) (h : n = 7) :
  number_of_matches n = 21 ∧ matches_per_class n = 6 := by
  sorry

#eval number_of_matches 7
#eval matches_per_class 7

end NUMINAMATH_CALUDE_tug_of_war_competition_l3640_364002


namespace NUMINAMATH_CALUDE_long_jump_competition_l3640_364070

/-- The long jump competition problem -/
theorem long_jump_competition (first second third fourth : ℝ) : 
  first = 22 →
  second = first + 1 →
  third = second - 2 →
  fourth = 24 →
  fourth - third = 3 := by
  sorry

end NUMINAMATH_CALUDE_long_jump_competition_l3640_364070


namespace NUMINAMATH_CALUDE_roger_retirement_eligibility_l3640_364023

theorem roger_retirement_eligibility 
  (roger peter tom robert mike sarah laura james : ℕ) 
  (h1 : roger = peter + tom + robert + mike + sarah + laura + james)
  (h2 : peter = 12)
  (h3 : tom = 2 * robert)
  (h4 : robert = peter - 4)
  (h5 : robert = mike + 2)
  (h6 : sarah = mike + 3)
  (h7 : sarah = tom / 2)
  (h8 : laura = robert - mike)
  (h9 : james > 0) : 
  roger > 50 := by
  sorry

end NUMINAMATH_CALUDE_roger_retirement_eligibility_l3640_364023


namespace NUMINAMATH_CALUDE_nested_expression_value_l3640_364081

theorem nested_expression_value : (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) = 1457 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l3640_364081


namespace NUMINAMATH_CALUDE_working_days_count_l3640_364040

/-- Represents a day of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a day in the month -/
structure DayInMonth where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Determines if a given day is a holiday -/
def isHoliday (d : DayInMonth) : Bool :=
  match d.dayOfWeek with
  | DayOfWeek.Sunday => true
  | DayOfWeek.Saturday => d.day % 14 == 8  -- Every second Saturday
  | _ => false

/-- Theorem: In a 30-day month starting on a Saturday, with every second Saturday 
    and all Sundays as holidays, there are 23 working days -/
theorem working_days_count : 
  let month : List DayInMonth := sorry  -- List of 30 days starting from Saturday
  (month.length = 30) →
  (month.head?.map (fun d => d.dayOfWeek) = some DayOfWeek.Saturday) →
  (month.filter (fun d => ¬isHoliday d)).length = 23 :=
by sorry

end NUMINAMATH_CALUDE_working_days_count_l3640_364040


namespace NUMINAMATH_CALUDE_geometric_mean_point_existence_l3640_364084

theorem geometric_mean_point_existence (A B C : ℝ) :
  ∃ (D : ℝ), 0 ≤ D ∧ D ≤ 1 ∧
  (Real.sin A * Real.sin B ≤ Real.sin (C / 2) ^ 2) ↔
  ∃ (CD AD DB : ℝ), CD ^ 2 = AD * DB ∧ AD + DB = 1 :=
sorry

end NUMINAMATH_CALUDE_geometric_mean_point_existence_l3640_364084


namespace NUMINAMATH_CALUDE_password_decryption_probability_l3640_364079

theorem password_decryption_probability 
  (p1 : ℝ) (p2 : ℝ) 
  (h1 : p1 = 1/5) (h2 : p2 = 1/4) 
  (h3 : 0 ≤ p1 ∧ p1 ≤ 1) (h4 : 0 ≤ p2 ∧ p2 ≤ 1) : 
  1 - (1 - p1) * (1 - p2) = 0.4 := by
sorry

end NUMINAMATH_CALUDE_password_decryption_probability_l3640_364079


namespace NUMINAMATH_CALUDE_total_traffic_tickets_l3640_364016

/-- The total number of traffic tickets Mark and Sarah have -/
def total_tickets (mark_parking : ℕ) (sarah_parking : ℕ) (mark_speeding : ℕ) (sarah_speeding : ℕ) : ℕ :=
  mark_parking + sarah_parking + mark_speeding + sarah_speeding

/-- Theorem stating the total number of traffic tickets Mark and Sarah have -/
theorem total_traffic_tickets :
  ∀ (mark_parking sarah_parking mark_speeding sarah_speeding : ℕ),
  mark_parking = 2 * sarah_parking →
  mark_speeding = sarah_speeding →
  sarah_speeding = 6 →
  mark_parking = 8 →
  total_tickets mark_parking sarah_parking mark_speeding sarah_speeding = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_traffic_tickets_l3640_364016


namespace NUMINAMATH_CALUDE_notch_volume_minimized_l3640_364077

/-- A cylindrical notch with angle θ between bounding planes -/
structure CylindricalNotch where
  θ : Real
  (θ_pos : θ > 0)
  (θ_lt_pi : θ < π)

/-- The volume of the notch given the angle φ between one bounding plane and the horizontal -/
noncomputable def notchVolume (n : CylindricalNotch) (φ : Real) : Real :=
  (2/3) * (Real.tan φ + Real.tan (n.θ - φ))

/-- Theorem: The volume of the notch is minimized when the bounding planes are at equal angles to the horizontal -/
theorem notch_volume_minimized (n : CylindricalNotch) :
  ∃ (φ_min : Real), φ_min = n.θ / 2 ∧
    ∀ (φ : Real), 0 < φ ∧ φ < n.θ → notchVolume n φ_min ≤ notchVolume n φ :=
sorry

end NUMINAMATH_CALUDE_notch_volume_minimized_l3640_364077


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l3640_364067

/-- A complex number is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The condition for z = (1+mi)(1+i) to be purely imaginary, where m is a real number. -/
theorem purely_imaginary_condition (m : ℝ) : 
  IsPurelyImaginary ((1 + m * Complex.I) * (1 + Complex.I)) ↔ m = 1 := by
  sorry

#check purely_imaginary_condition

end NUMINAMATH_CALUDE_purely_imaginary_condition_l3640_364067


namespace NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l3640_364085

theorem log_equality_implies_golden_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.log a / Real.log 8 = Real.log b / Real.log 18) ∧
  (Real.log a / Real.log 8 = Real.log (a + b) / Real.log 32) →
  b / a = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l3640_364085


namespace NUMINAMATH_CALUDE_no_real_arithmetic_progression_l3640_364017

theorem no_real_arithmetic_progression : ¬ ∃ (a b : ℝ), 
  (b - a = a - 12) ∧ (ab - b = b - a) := by
  sorry

end NUMINAMATH_CALUDE_no_real_arithmetic_progression_l3640_364017


namespace NUMINAMATH_CALUDE_triangle_problem_l3640_364026

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  -- Given conditions
  3 * Real.cos (B + C) = -1 ∧
  a = 3 ∧
  (1 / 2) * b * c * Real.sin A = 2 * Real.sqrt 2 →
  -- Conclusion
  Real.cos A = 1 / 3 ∧
  ((b = 2 ∧ c = 3) ∨ (b = 3 ∧ c = 2)) := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l3640_364026


namespace NUMINAMATH_CALUDE_quadratic_identities_max_bound_l3640_364088

/-- Given 0 ≤ p, r ≤ 1 and two identities, prove that max(a, b, c) and max(α, β, γ) are ≥ 4/9 -/
theorem quadratic_identities_max_bound {p r : ℝ} (hp : 0 ≤ p ∧ p ≤ 1) (hr : 0 ≤ r ∧ r ≤ 1)
  (h1 : ∀ x y, (p * x + (1 - p) * y)^2 = a * x^2 + b * x * y + c * y^2)
  (h2 : ∀ x y, (p * x + (1 - p) * y) * (r * x + (1 - r) * y) = α * x^2 + β * x * y + γ * y^2) :
  max a (max b c) ≥ 4/9 ∧ max α (max β γ) ≥ 4/9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_identities_max_bound_l3640_364088


namespace NUMINAMATH_CALUDE_fourth_term_of_gp_l3640_364078

def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem fourth_term_of_gp (x : ℝ) :
  let a₁ := x
  let a₂ := 3 * x + 3
  let a₃ := 5 * x + 5
  let r := a₂ / a₁
  (∀ n : ℕ, n ≥ 1 → geometric_progression a₁ r n = if n = 1 then a₁ else if n = 2 then a₂ else if n = 3 then a₃ else 0) →
  geometric_progression a₁ r 4 = -125 / 12 :=
by sorry

end NUMINAMATH_CALUDE_fourth_term_of_gp_l3640_364078


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_2_l3640_364042

theorem no_solution_implies_a_leq_2 (a : ℝ) : 
  (∀ x : ℝ, ¬(2*x - 4 > 0 ∧ x - a < 0)) → a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_2_l3640_364042


namespace NUMINAMATH_CALUDE_base_equivalence_l3640_364052

/-- Converts a number from base b to base 10 --/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The theorem stating the equivalence of the numbers in different bases --/
theorem base_equivalence (k : Nat) :
  toBase10 [5, 2, 4] 8 = toBase10 [6, 6, 4] k → k = 7 ∧ toBase10 [6, 6, 4] 7 = toBase10 [5, 2, 4] 8 := by
  sorry

end NUMINAMATH_CALUDE_base_equivalence_l3640_364052


namespace NUMINAMATH_CALUDE_hives_needed_for_candles_twenty_four_hives_for_ninety_six_candles_l3640_364059

/-- Given that 3 beehives make enough wax for 12 candles, 
    prove that 24 hives are needed to make 96 candles. -/
theorem hives_needed_for_candles : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun hives_given candles_given hives_needed candles_needed =>
    (hives_given * (candles_needed / candles_given) = hives_needed) →
    (3 * (96 / 12) = 24)

/-- The main theorem stating that 24 hives are needed for 96 candles. -/
theorem twenty_four_hives_for_ninety_six_candles :
  hives_needed_for_candles 3 12 24 96 := by
  sorry

end NUMINAMATH_CALUDE_hives_needed_for_candles_twenty_four_hives_for_ninety_six_candles_l3640_364059


namespace NUMINAMATH_CALUDE_bread_rise_time_l3640_364068

/-- The time (in minutes) Mark lets the bread rise each time -/
def rise_time : ℕ := sorry

/-- The total time (in minutes) to make bread -/
def total_time : ℕ := 280

/-- The time (in minutes) spent kneading -/
def kneading_time : ℕ := 10

/-- The time (in minutes) spent baking -/
def baking_time : ℕ := 30

/-- Theorem stating that the rise time is 120 minutes -/
theorem bread_rise_time : rise_time = 120 := by
  sorry

end NUMINAMATH_CALUDE_bread_rise_time_l3640_364068


namespace NUMINAMATH_CALUDE_equal_roots_condition_l3640_364039

theorem equal_roots_condition (m : ℝ) : 
  (∃ (x : ℝ), (x * (x - 3) - (m + 2)) / ((x - 3) * (m - 2)) = x / m ∧ 
   ∀ (y : ℝ), (y * (y - 3) - (m + 2)) / ((y - 3) * (m - 2)) = y / m → y = x) ↔ 
  (m = 2 ∨ m = (9 + Real.sqrt 57) / 8 ∨ m = (9 - Real.sqrt 57) / 8) := by
sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l3640_364039


namespace NUMINAMATH_CALUDE_line_segment_parameter_sum_of_squares_l3640_364057

/-- Given a line segment connecting (1, -3) and (4, 6), parameterized by x = pt + q and y = rt + s,
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (1, -3), prove that p^2 + q^2 + r^2 + s^2 = 100 -/
theorem line_segment_parameter_sum_of_squares :
  ∀ (p q r s : ℝ),
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = p * t + q ∧ y = r * t + s) →
  (q = 1 ∧ s = -3) →
  (p + q = 4 ∧ r + s = 6) →
  p^2 + q^2 + r^2 + s^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_parameter_sum_of_squares_l3640_364057


namespace NUMINAMATH_CALUDE_pigeonhole_mod_three_l3640_364097

theorem pigeonhole_mod_three (s : Finset ℤ) (h : s.card = 6) :
  ∃ (a b c d : ℤ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ d ∈ s ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (a * b) % 3 = (c * d) % 3 :=
by sorry

end NUMINAMATH_CALUDE_pigeonhole_mod_three_l3640_364097


namespace NUMINAMATH_CALUDE_max_a_value_l3640_364086

def A : Set ℝ := {x | x^2 + x - 6 < 0}
def B (a : ℝ) : Set ℝ := {x | x > a}

theorem max_a_value (a : ℝ) :
  (A ⊂ B a) → (∀ b, (A ⊂ B b) → a ≥ b) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_max_a_value_l3640_364086


namespace NUMINAMATH_CALUDE_triangle_data_uniqueness_l3640_364071

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)  -- sides
  (α β γ : ℝ)  -- angles

-- Define the different sets of data
def ratio_two_sides_included_angle (t : Triangle) : ℝ × ℝ × ℝ := sorry
def ratios_angle_bisectors (t : Triangle) : ℝ × ℝ × ℝ := sorry
def ratios_medians (t : Triangle) : ℝ × ℝ × ℝ := sorry
def ratios_two_altitudes_bases (t : Triangle) : ℝ × ℝ := sorry
def two_angles_ratio_side_sum (t : Triangle) : ℝ × ℝ × ℝ := sorry

-- Define a predicate for unique determination of triangle shape
def uniquely_determines_shape (f : Triangle → α) : Prop :=
  ∀ t1 t2 : Triangle, f t1 = f t2 → t1 = t2

-- State the theorem
theorem triangle_data_uniqueness :
  uniquely_determines_shape ratio_two_sides_included_angle ∧
  uniquely_determines_shape ratios_medians ∧
  uniquely_determines_shape ratios_two_altitudes_bases ∧
  uniquely_determines_shape two_angles_ratio_side_sum ∧
  ¬ uniquely_determines_shape ratios_angle_bisectors :=
sorry

end NUMINAMATH_CALUDE_triangle_data_uniqueness_l3640_364071


namespace NUMINAMATH_CALUDE_total_interest_is_350_l3640_364014

/-- Calculate the total interest amount for two loans over a specified period. -/
def totalInterest (loan1Amount : ℝ) (loan1Rate : ℝ) (loan2Amount : ℝ) (loan2Rate : ℝ) (years : ℝ) : ℝ :=
  (loan1Amount * loan1Rate * years) + (loan2Amount * loan2Rate * years)

/-- Theorem stating that the total interest for the given loans and time period is 350. -/
theorem total_interest_is_350 :
  totalInterest 1000 0.03 1200 0.05 3.888888888888889 = 350 := by
  sorry

#eval totalInterest 1000 0.03 1200 0.05 3.888888888888889

end NUMINAMATH_CALUDE_total_interest_is_350_l3640_364014


namespace NUMINAMATH_CALUDE_work_completion_time_l3640_364029

theorem work_completion_time (original_men : ℕ) (original_days : ℕ) (absent_men : ℕ) 
  (h1 : original_men = 180)
  (h2 : original_days = 55)
  (h3 : absent_men = 15) :
  (original_men * original_days) / (original_men - absent_men) = 60 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3640_364029


namespace NUMINAMATH_CALUDE_mia_nia_difference_l3640_364018

/-- Represents a driving scenario with three drivers: Leo, Nia, and Mia. -/
structure DrivingScenario where
  /-- Leo's driving time in hours -/
  t : ℝ
  /-- Leo's driving speed in miles per hour -/
  s : ℝ
  /-- Leo's total distance driven in miles -/
  d : ℝ
  /-- Nia's total distance driven in miles -/
  nia_d : ℝ
  /-- Mia's total distance driven in miles -/
  mia_d : ℝ
  /-- Leo's distance equals speed times time -/
  leo_distance : d = s * t
  /-- Nia drove 2 hours longer than Leo at 10 mph faster -/
  nia_equation : nia_d = (s + 10) * (t + 2)
  /-- Mia drove 3 hours longer than Leo at 15 mph faster -/
  mia_equation : mia_d = (s + 15) * (t + 3)
  /-- Nia drove 110 miles more than Leo -/
  nia_leo_diff : nia_d = d + 110

/-- Theorem stating that Mia drove 100 miles more than Nia -/
theorem mia_nia_difference (scenario : DrivingScenario) : 
  scenario.mia_d - scenario.nia_d = 100 := by
  sorry

end NUMINAMATH_CALUDE_mia_nia_difference_l3640_364018


namespace NUMINAMATH_CALUDE_intersection_and_inequality_l3640_364047

/-- 
Given a line y = 2x + m that intersects the x-axis at (-1, 0),
prove that the solution set of 2x + m ≤ 0 is x ≤ -1.
-/
theorem intersection_and_inequality (m : ℝ) 
  (h1 : 2 * (-1) + m = 0) -- Line intersects x-axis at (-1, 0)
  (x : ℝ) : 
  (2 * x + m ≤ 0) ↔ (x ≤ -1) := by
sorry

end NUMINAMATH_CALUDE_intersection_and_inequality_l3640_364047


namespace NUMINAMATH_CALUDE_ellipse_sum_a_k_l3640_364053

-- Define the ellipse
def Ellipse (f₁ f₂ p : ℝ × ℝ) : Prop :=
  let d₁ := Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2)
  let d₂ := Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2)
  let c := Real.sqrt ((f₂.1 - f₁.1)^2 + (f₂.2 - f₁.2)^2) / 2
  let a := (d₁ + d₂) / 2
  let b := Real.sqrt (a^2 - c^2)
  let h := (f₁.1 + f₂.1) / 2
  let k := (f₁.2 + f₂.2) / 2
  ∀ (x y : ℝ), (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

theorem ellipse_sum_a_k :
  let f₁ : ℝ × ℝ := (2, 1)
  let f₂ : ℝ × ℝ := (2, 5)
  let p : ℝ × ℝ := (-3, 3)
  Ellipse f₁ f₂ p →
  let a := (Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) +
            Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2)) / 2
  let k := (f₁.2 + f₂.2) / 2
  a + k = Real.sqrt 29 + 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_a_k_l3640_364053


namespace NUMINAMATH_CALUDE_events_A_B_independent_l3640_364087

structure GiftBox :=
  (chinese_knot : Bool)
  (notebook : Bool)
  (pencil_case : Bool)

def box1 : GiftBox := ⟨true, false, false⟩
def box2 : GiftBox := ⟨false, true, false⟩
def box3 : GiftBox := ⟨false, false, true⟩
def box4 : GiftBox := ⟨true, true, true⟩

def all_boxes : List GiftBox := [box1, box2, box3, box4]

def event_A (box : GiftBox) : Bool := box.chinese_knot
def event_B (box : GiftBox) : Bool := box.notebook

def prob_A : ℚ := (all_boxes.filter event_A).length / all_boxes.length
def prob_B : ℚ := (all_boxes.filter event_B).length / all_boxes.length
def prob_AB : ℚ := (all_boxes.filter (λ b => event_A b ∧ event_B b)).length / all_boxes.length

theorem events_A_B_independent : prob_A * prob_B = prob_AB := by sorry

end NUMINAMATH_CALUDE_events_A_B_independent_l3640_364087


namespace NUMINAMATH_CALUDE_seven_boys_without_calculators_l3640_364019

/-- Represents the number of boys who didn't bring calculators to Mrs. Luna's math class -/
def boys_without_calculators (total_boys : ℕ) (total_with_calculators : ℕ) (girls_with_calculators : ℕ) : ℕ :=
  total_boys - (total_with_calculators - girls_with_calculators)

/-- Theorem stating that 7 boys didn't bring their calculators to Mrs. Luna's math class -/
theorem seven_boys_without_calculators :
  boys_without_calculators 20 28 15 = 7 := by
  sorry

end NUMINAMATH_CALUDE_seven_boys_without_calculators_l3640_364019


namespace NUMINAMATH_CALUDE_lucy_father_age_twice_l3640_364037

theorem lucy_father_age_twice (lucy_birth_year father_birth_year : ℕ) 
  (h1 : lucy_birth_year = 2000) 
  (h2 : father_birth_year = 1960) : 
  ∃ (year : ℕ), year = 2040 ∧ 
  (year - father_birth_year = 2 * (year - lucy_birth_year)) :=
sorry

end NUMINAMATH_CALUDE_lucy_father_age_twice_l3640_364037


namespace NUMINAMATH_CALUDE_inequality1_solution_system_solution_integer_system_solution_l3640_364060

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := 3 * x - 5 > 5 * x + 3
def inequality2 (x : ℝ) : Prop := x - 1 ≥ 1 - x
def inequality3 (x : ℝ) : Prop := x + 8 > 4 * x - 1

-- Define the solution sets
def solution_set1 : Set ℝ := {x : ℝ | x < -4}
def solution_set2 : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 3}

-- Define the integer solutions
def integer_solutions : Set ℤ := {1, 2}

-- Theorem statements
theorem inequality1_solution : 
  {x : ℝ | inequality1 x} = solution_set1 :=
sorry

theorem system_solution : 
  {x : ℝ | inequality2 x ∧ inequality3 x} = solution_set2 :=
sorry

theorem integer_system_solution : 
  {x : ℤ | (x : ℝ) ∈ solution_set2} = integer_solutions :=
sorry

end NUMINAMATH_CALUDE_inequality1_solution_system_solution_integer_system_solution_l3640_364060


namespace NUMINAMATH_CALUDE_notebook_payment_possible_l3640_364069

theorem notebook_payment_possible : ∃ (a b : ℕ), 27 * a - 16 * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_notebook_payment_possible_l3640_364069


namespace NUMINAMATH_CALUDE_participation_schemes_count_l3640_364015

/-- Represents the number of students -/
def num_students : ℕ := 5

/-- Represents the number of competitions -/
def num_competitions : ℕ := 4

/-- Represents the number of competitions student A cannot participate in -/
def restricted_competitions : ℕ := 2

/-- Calculates the number of different competition participation schemes -/
def participation_schemes : ℕ := 
  (num_competitions - restricted_competitions) * 
  (Nat.factorial num_students / Nat.factorial (num_students - (num_competitions - 1)))

/-- Theorem stating the number of different competition participation schemes -/
theorem participation_schemes_count : participation_schemes = 72 := by
  sorry

end NUMINAMATH_CALUDE_participation_schemes_count_l3640_364015


namespace NUMINAMATH_CALUDE_sin_sixty_degrees_l3640_364058

theorem sin_sixty_degrees : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sixty_degrees_l3640_364058


namespace NUMINAMATH_CALUDE_fraction_division_problem_solution_l3640_364056

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) := by sorry

theorem problem_solution : (3 : ℚ) / 4 / ((2 : ℚ) / 5) = 15 / 8 := by sorry

end NUMINAMATH_CALUDE_fraction_division_problem_solution_l3640_364056


namespace NUMINAMATH_CALUDE_exactly_two_integers_satisfy_l3640_364020

-- Define the circle
def circle_center : ℝ × ℝ := (3, -3)
def circle_radius : ℝ := 8

-- Define the point (x, x+2)
def point (x : ℤ) : ℝ × ℝ := (x, x + 2)

-- Define the condition for a point to be inside or on the circle
def inside_or_on_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 ≤ circle_radius^2

-- Theorem statement
theorem exactly_two_integers_satisfy :
  ∃! (s : Finset ℤ), s.card = 2 ∧ ∀ x : ℤ, x ∈ s ↔ inside_or_on_circle (point x) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_integers_satisfy_l3640_364020


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l3640_364028

theorem meaningful_fraction_range (x : ℝ) :
  (∃ y : ℝ, y = 2 / (2 * x - 1)) ↔ x ≠ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l3640_364028


namespace NUMINAMATH_CALUDE_garden_rectangle_length_l3640_364049

theorem garden_rectangle_length :
  ∀ (perimeter width length base_triangle height_triangle : ℝ),
    perimeter = 480 →
    width = 2 * base_triangle →
    base_triangle = 50 →
    height_triangle = 100 →
    perimeter = 2 * (length + width) →
    length = 140 := by
  sorry

end NUMINAMATH_CALUDE_garden_rectangle_length_l3640_364049


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l3640_364046

def m : ℕ := 2023^2 + 2^2023

theorem units_digit_of_m_squared_plus_two_to_m (m : ℕ) : (m^2 + 2^m) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l3640_364046


namespace NUMINAMATH_CALUDE_video_game_discount_savings_l3640_364048

theorem video_game_discount_savings (original_price : ℚ) 
  (flat_discount : ℚ) (percentage_discount : ℚ) : 
  original_price = 60 →
  flat_discount = 10 →
  percentage_discount = 0.25 →
  (original_price - flat_discount) * (1 - percentage_discount) - 
  (original_price * (1 - percentage_discount) - flat_discount) = 
  250 / 100 := by
  sorry

end NUMINAMATH_CALUDE_video_game_discount_savings_l3640_364048


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3640_364024

/-- A geometric sequence {a_n} with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  (a 1 + a 2 = 30) →
  (a 3 + a 4 = 60) →
  (a 7 + a 8 = (a 1 + a 2) * q^6) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3640_364024


namespace NUMINAMATH_CALUDE_existence_of_prime_not_divisible_l3640_364076

theorem existence_of_prime_not_divisible (p : Nat) (h_prime : Prime p) (h_p_gt_2 : p > 2) :
  ∃ q : Nat, Prime q ∧ q < p ∧ ¬(p^2 ∣ q^(p-1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_prime_not_divisible_l3640_364076


namespace NUMINAMATH_CALUDE_pond_volume_l3640_364000

/-- Calculates the volume of a trapezoidal prism -/
def trapezoidalPrismVolume (length : ℝ) (avgWidth : ℝ) (avgDepth : ℝ) : ℝ :=
  length * avgWidth * avgDepth

/-- The pond dimensions -/
def pondLength : ℝ := 25
def pondAvgWidth : ℝ := 12.5
def pondAvgDepth : ℝ := 10

/-- Theorem stating the volume of the pond -/
theorem pond_volume :
  trapezoidalPrismVolume pondLength pondAvgWidth pondAvgDepth = 3125 := by
  sorry

#check pond_volume

end NUMINAMATH_CALUDE_pond_volume_l3640_364000


namespace NUMINAMATH_CALUDE_parabola_unique_coefficients_l3640_364035

/-- A parabola passing through three given points -/
structure Parabola where
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ
  p1 : eq (-2) = -16
  p2 : eq 2 = 8
  p3 : eq 4 = 36
  form : ∀ x, eq x = x^2 + b*x + c

/-- The unique values of b and c for the parabola -/
theorem parabola_unique_coefficients (p : Parabola) : p.b = 6 ∧ p.c = -8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_unique_coefficients_l3640_364035


namespace NUMINAMATH_CALUDE_complex_number_solution_binomial_expansion_coefficient_l3640_364009

-- Part 1
def complex_number (b : ℝ) : ℂ := 3 + b * Complex.I

theorem complex_number_solution (b : ℝ) (h1 : b > 0) (h2 : ∃ (k : ℝ), (complex_number b - 2)^2 = k * Complex.I) :
  complex_number b = 3 + Complex.I := by sorry

-- Part 2
def binomial_sum (n : ℕ) : ℕ := 2^n

def expansion_term (n r : ℕ) (x : ℝ) : ℝ :=
  Nat.choose n r * 3^(n - r) * x^(n - 3/2 * r)

theorem binomial_expansion_coefficient (n : ℕ) (h : binomial_sum n = 16) :
  expansion_term n 2 1 = 54 := by sorry

end NUMINAMATH_CALUDE_complex_number_solution_binomial_expansion_coefficient_l3640_364009


namespace NUMINAMATH_CALUDE_jellybean_problem_l3640_364036

/-- The number of jellybeans remaining after eating 25% --/
def eat_jellybeans (n : ℝ) : ℝ := 0.75 * n

/-- The number of jellybeans Jenny has initially --/
def initial_jellybeans : ℝ := 80

/-- The number of jellybeans added after the first day --/
def added_jellybeans : ℝ := 20

/-- The number of jellybeans remaining after three days --/
def remaining_jellybeans : ℝ := 
  eat_jellybeans (eat_jellybeans (eat_jellybeans initial_jellybeans + added_jellybeans))

theorem jellybean_problem : remaining_jellybeans = 45 := by sorry

end NUMINAMATH_CALUDE_jellybean_problem_l3640_364036


namespace NUMINAMATH_CALUDE_banana_price_reduction_theorem_l3640_364090

/-- Represents the price reduction scenario for bananas -/
structure BananaPriceReduction where
  reduced_price_per_dozen : ℝ
  additional_bananas : ℕ
  additional_cost : ℝ

/-- Calculates the percentage reduction in banana prices -/
def calculate_percentage_reduction (scenario : BananaPriceReduction) : ℝ :=
  -- The implementation is not provided as per the instructions
  sorry

/-- Theorem stating that the percentage reduction is 60% given the specified conditions -/
theorem banana_price_reduction_theorem (scenario : BananaPriceReduction) 
  (h1 : scenario.reduced_price_per_dozen = 3.84)
  (h2 : scenario.additional_bananas = 50)
  (h3 : scenario.additional_cost = 40) : 
  calculate_percentage_reduction scenario = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_price_reduction_theorem_l3640_364090


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l3640_364031

theorem chord_length_concentric_circles 
  (area_ring : ℝ) 
  (radius_small : ℝ) 
  (chord_length : ℝ) :
  area_ring = 50 * Real.pi ∧ 
  radius_small = 5 →
  chord_length = 10 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l3640_364031


namespace NUMINAMATH_CALUDE_problem_solution_l3640_364025

def A (a : ℕ) : Set ℕ := {2, 5, a + 1}
def B (a : ℕ) : Set ℕ := {1, 3, a}
def U : Set ℕ := {x | x ≤ 6}

theorem problem_solution (a : ℕ) 
  (h1 : A a ∩ B a = {2, 3}) :
  (a = 2) ∧ 
  (A a ∪ B a = {1, 2, 3, 5}) ∧ 
  ((Uᶜ ∩ (A a)ᶜ) ∩ (Uᶜ ∩ (B a)ᶜ) = {0, 4, 6}) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3640_364025


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3640_364098

/-- Given a run of 12 miles in 90 minutes, prove that the average speed is 8 miles per hour -/
theorem average_speed_calculation (distance : ℝ) (time_minutes : ℝ) (h1 : distance = 12) (h2 : time_minutes = 90) :
  distance / (time_minutes / 60) = 8 := by
sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3640_364098


namespace NUMINAMATH_CALUDE_complement_of_A_wrt_U_l3640_364061

-- Define the universal set U
def U : Set ℝ := {x | x > 1}

-- Define the set A
def A : Set ℝ := {x | x > 2}

-- Define the complement of A with respect to U
def complement_U_A : Set ℝ := {x ∈ U | x ∉ A}

-- Theorem stating the complement of A with respect to U
theorem complement_of_A_wrt_U :
  complement_U_A = {x | 1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_wrt_U_l3640_364061


namespace NUMINAMATH_CALUDE_no_solution_exists_l3640_364065

theorem no_solution_exists : ¬∃ (s c : ℕ), 
  15 ≤ s ∧ s ≤ 35 ∧ c > 0 ∧ 30 * s + 31 * c = 1200 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3640_364065


namespace NUMINAMATH_CALUDE_trailing_zeros_50_factorial_l3640_364073

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 50! is 12 -/
theorem trailing_zeros_50_factorial :
  trailingZeros 50 = 12 := by
  sorry


end NUMINAMATH_CALUDE_trailing_zeros_50_factorial_l3640_364073


namespace NUMINAMATH_CALUDE_line_inclination_45_degrees_l3640_364083

/-- The angle of inclination of the line x - y - √3 = 0 is 45° -/
theorem line_inclination_45_degrees :
  let line := {(x, y) : ℝ × ℝ | x - y - Real.sqrt 3 = 0}
  ∃ θ : ℝ, θ = 45 * π / 180 ∧ ∀ (x y : ℝ), (x, y) ∈ line → y = Real.tan θ * x + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_45_degrees_l3640_364083


namespace NUMINAMATH_CALUDE_palabras_bookstore_workers_l3640_364044

theorem palabras_bookstore_workers (W : ℕ) : 
  W / 2 = W / 2 ∧  -- Half of workers read Saramago's book
  W / 6 = W / 6 ∧  -- 1/6 of workers read Kureishi's book
  (∃ (n : ℕ), n = 12 ∧ n ≤ W / 2 ∧ n ≤ W / 6) ∧  -- 12 workers read both books
  (W - (W / 2 + W / 6 - 12)) = ((W / 2 - 12) - 1) →  -- Workers who read neither book
  W = 210 := by
sorry

end NUMINAMATH_CALUDE_palabras_bookstore_workers_l3640_364044


namespace NUMINAMATH_CALUDE_vector_ratio_theorem_l3640_364004

-- Define the plane and points
variable (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P]
variable (O A B C : P)

-- Define the non-collinearity condition
def noncollinear (O A B C : P) : Prop :=
  ¬ (∃ (a b c : ℝ), a • (A - O) + b • (B - O) + c • (C - O) = 0 ∧ (a, b, c) ≠ (0, 0, 0))

-- State the theorem
theorem vector_ratio_theorem (h_noncollinear : noncollinear P O A B C)
  (h_eq : A - O - 4 • (B - O) + 3 • (C - O) = 0) :
  ‖A - B‖ / ‖C - A‖ = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_vector_ratio_theorem_l3640_364004


namespace NUMINAMATH_CALUDE_triangle_max_area_l3640_364043

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where a = 2 and 3b*sin(C) - 5c*sin(B)*cos(A) = 0, 
    the maximum area of the triangle is 10/3. -/
theorem triangle_max_area (a b c A B C : ℝ) : 
  a = 2 → 
  3 * b * Real.sin C - 5 * c * Real.sin B * Real.cos A = 0 → 
  (∃ (S : ℝ), S = (1/2) * a * b * Real.sin C ∧ 
    ∀ (S' : ℝ), S' = (1/2) * a * b * Real.sin C → S' ≤ S) →
  (1/2) * a * b * Real.sin C ≤ 10/3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3640_364043


namespace NUMINAMATH_CALUDE_anna_cupcakes_l3640_364027

def total_cupcakes : ℕ := 150

def classmates_fraction : ℚ := 2/5
def neighbors_fraction : ℚ := 1/3
def work_friends_fraction : ℚ := 1/10
def eating_fraction : ℚ := 7/15

def remaining_cupcakes : ℕ := 14

theorem anna_cupcakes :
  let given_away := (classmates_fraction + neighbors_fraction + work_friends_fraction) * total_cupcakes
  let after_giving := total_cupcakes - ⌊given_away⌋
  let eaten := ⌊eating_fraction * after_giving⌋
  total_cupcakes - ⌊given_away⌋ - eaten = remaining_cupcakes := by
  sorry

#check anna_cupcakes

end NUMINAMATH_CALUDE_anna_cupcakes_l3640_364027


namespace NUMINAMATH_CALUDE_square_perimeters_sum_l3640_364012

theorem square_perimeters_sum (a b : ℝ) (h1 : a^2 + b^2 = 130) (h2 : a^2 - b^2 = 50) :
  4*a + 4*b = 20 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeters_sum_l3640_364012


namespace NUMINAMATH_CALUDE_parabola_directrix_l3640_364038

/-- A parabola is defined by its equation relating x and y coordinates. -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- The directrix of a parabola is a line perpendicular to its axis of symmetry. -/
structure Directrix where
  equation : ℝ → ℝ → Prop

/-- For a parabola with equation x² = (1/4)y, its directrix has equation y = -1/16. -/
theorem parabola_directrix (p : Parabola) (d : Directrix) :
  (∀ x y, p.equation x y ↔ x^2 = (1/4) * y) →
  (∀ x y, d.equation x y ↔ y = -1/16) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3640_364038


namespace NUMINAMATH_CALUDE_area_difference_triangles_l3640_364089

/-- Given a right-angled triangle with base 3 and height 9, and another right-angled triangle
    with base 6 and height 9, prove that the difference between the areas of the triangles formed
    by a line intersecting both hypotenuses is 13.5 square units. -/
theorem area_difference_triangles (A B C D F H : ℝ × ℝ) : 
  -- ΔFAH and ΔHBC are right triangles
  (F.1 - A.1) * (H.2 - A.2) = (H.1 - A.1) * (F.2 - A.2) →
  (H.1 - B.1) * (C.2 - B.2) = (C.1 - B.1) * (H.2 - B.2) →
  -- AH = 6
  (H.1 - A.1)^2 + (H.2 - A.2)^2 = 36 →
  -- HB = 3
  (B.1 - H.1)^2 + (B.2 - H.2)^2 = 9 →
  -- FC = 9
  (C.1 - F.1)^2 + (C.2 - F.2)^2 = 81 →
  -- AC and HF intersect at D
  ∃ t : ℝ, D = (1 - t) • A + t • C ∧ ∃ s : ℝ, D = (1 - s) • H + s • F →
  -- The difference between the areas of ΔADF and ΔBDC is 13.5
  abs ((A.1 * (F.2 - D.2) + D.1 * (A.2 - F.2) + F.1 * (D.2 - A.2)) / 2 -
       (B.1 * (C.2 - D.2) + D.1 * (B.2 - C.2) + C.1 * (D.2 - B.2)) / 2) = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_area_difference_triangles_l3640_364089


namespace NUMINAMATH_CALUDE_two_digit_integer_problem_l3640_364033

theorem two_digit_integer_problem :
  ∃ (m n : ℕ),
    10 ≤ m ∧ m < 100 ∧
    10 ≤ n ∧ n < 100 ∧
    (m + n : ℚ) / 2 = n + m / 100 ∧
    m + n < 150 ∧
    m = 50 ∧ n = 49 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_integer_problem_l3640_364033


namespace NUMINAMATH_CALUDE_other_factor_proof_l3640_364045

theorem other_factor_proof (a : ℕ) (h : a = 363) : 
  (a * 43 * 62 * 1311) / 33 = 38428986 := by
  sorry

end NUMINAMATH_CALUDE_other_factor_proof_l3640_364045


namespace NUMINAMATH_CALUDE_negative_double_negative_and_negative_absolute_are_opposite_l3640_364032

-- Define opposite numbers
def are_opposite (a b : ℝ) : Prop := a = -b

-- Theorem statement
theorem negative_double_negative_and_negative_absolute_are_opposite :
  are_opposite (-(-5)) (-|5|) := by
  sorry

end NUMINAMATH_CALUDE_negative_double_negative_and_negative_absolute_are_opposite_l3640_364032


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l3640_364082

theorem triangle_inequality_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + 2*b*c) / (b^2 + c^2) + (b^2 + 2*a*c) / (c^2 + a^2) + (c^2 + 2*a*b) / (a^2 + b^2) > 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l3640_364082


namespace NUMINAMATH_CALUDE_quadratic_one_root_l3640_364030

theorem quadratic_one_root (a b c d : ℝ) : 
  b = a - d →
  c = a - 3*d →
  a ≥ b →
  b ≥ c →
  c ≥ 0 →
  (∃! x : ℝ, a*x^2 + b*x + c = 0) →
  (∃ x : ℝ, a*x^2 + b*x + c = 0 ∧ x = -(1 + 3*Real.sqrt 22) / 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l3640_364030


namespace NUMINAMATH_CALUDE_train_length_calculation_l3640_364054

/-- Calculates the length of a train given the speeds of two trains traveling in opposite directions and the time taken for one train to pass an observer in the other train. -/
theorem train_length_calculation (woman_speed goods_speed : ℝ) (passing_time : ℝ) 
  (woman_speed_pos : 0 < woman_speed)
  (goods_speed_pos : 0 < goods_speed)
  (passing_time_pos : 0 < passing_time)
  (h_woman_speed : woman_speed = 25)
  (h_goods_speed : goods_speed = 142.986561075114)
  (h_passing_time : passing_time = 3) :
  ∃ (train_length : ℝ), abs (train_length - 38.932) < 0.001 :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3640_364054


namespace NUMINAMATH_CALUDE_min_trig_expression_min_trig_expression_achievable_l3640_364022

theorem min_trig_expression (θ : Real) (h_acute : 0 < θ ∧ θ < π / 2) :
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) +
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) ≥ 2 :=
by sorry

theorem min_trig_expression_achievable :
  ∃ θ : Real, 0 < θ ∧ θ < π / 2 ∧
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) +
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_trig_expression_min_trig_expression_achievable_l3640_364022


namespace NUMINAMATH_CALUDE_max_factorable_n_is_largest_l3640_364006

/-- A polynomial of the form 3x^2 + nx + 72 can be factored as (3x + A)(x + B) where A and B are integers -/
def is_factorable (n : ℤ) : Prop :=
  ∃ A B : ℤ, 3 * B + A = n ∧ A * B = 72

/-- The maximum value of n for which 3x^2 + nx + 72 can be factored as the product of two linear factors with integer coefficients -/
def max_factorable_n : ℤ := 217

/-- Theorem stating that max_factorable_n is the largest value of n for which the polynomial is factorable -/
theorem max_factorable_n_is_largest :
  is_factorable max_factorable_n ∧
  ∀ m : ℤ, m > max_factorable_n → ¬is_factorable m :=
by sorry

end NUMINAMATH_CALUDE_max_factorable_n_is_largest_l3640_364006


namespace NUMINAMATH_CALUDE_nancy_widget_production_l3640_364066

/-- Nancy's widget production problem -/
theorem nancy_widget_production (t : ℝ) (h : t > 0) : 
  let w := 2 * t
  let monday_production := w * t
  let tuesday_production := (w + 5) * (t - 3)
  monday_production - tuesday_production = t + 15 := by
sorry

end NUMINAMATH_CALUDE_nancy_widget_production_l3640_364066


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_equation_l3640_364099

theorem quadratic_inequality_and_equation (a : ℝ) :
  (∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∧ 
  (∃ x₀ : ℝ, x₀^2 - x₀ + a = 0) →
  0 ≤ a ∧ a ≤ 1/4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_and_equation_l3640_364099


namespace NUMINAMATH_CALUDE_parallelogram_distance_l3640_364050

/-- A parallelogram with given dimensions -/
structure Parallelogram where
  side1 : ℝ  -- Length of one pair of parallel sides
  side2 : ℝ  -- Length of the other pair of parallel sides
  height1 : ℝ  -- Height corresponding to side1
  height2 : ℝ  -- Height corresponding to side2 (to be proved)

/-- Theorem stating the relationship between the dimensions of the parallelogram -/
theorem parallelogram_distance (p : Parallelogram) 
  (h1 : p.side1 = 20) 
  (h2 : p.side2 = 75) 
  (h3 : p.height1 = 60) : 
  p.height2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_distance_l3640_364050


namespace NUMINAMATH_CALUDE_sphere_volume_l3640_364011

theorem sphere_volume (d r h : ℝ) (h1 : d = 2 * Real.sqrt 5) (h2 : h = 2) 
  (h3 : r^2 = (d/2)^2 + h^2) : (4/3) * Real.pi * r^3 = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_l3640_364011


namespace NUMINAMATH_CALUDE_triangle_properties_l3640_364007

noncomputable section

-- Define the triangle ABC
variable (A B C : Real)
variable (a b c : Real)

-- Define the conditions
axiom triangle_angles : A + B + C = Real.pi
axiom cos_A : Real.cos A = 1/3
axiom side_a : a = Real.sqrt 3

-- Define the theorem
theorem triangle_properties :
  (Real.sin ((B + C) / 2))^2 + Real.cos (2 * A) = -1/9 ∧
  (∀ x y : Real, x * y ≤ 9/4 ∧ (x = b ∧ y = c → x * y = 9/4)) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3640_364007


namespace NUMINAMATH_CALUDE_y_coordinates_equal_l3640_364075

/-- Parabola equation -/
def parabola (x y : ℝ) : Prop := y = 2 * (x - 3)^2 - 4

theorem y_coordinates_equal :
  ∀ y₁ y₂ : ℝ,
  parabola 2 y₁ →
  parabola 4 y₂ →
  y₁ = y₂ := by
  sorry

end NUMINAMATH_CALUDE_y_coordinates_equal_l3640_364075


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_train_passes_jogger_in_32_seconds_l3640_364072

/-- The time taken for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (initial_distance : ℝ) 
  (train_length : ℝ) : ℝ :=
by
  -- Convert speeds from km/hr to m/s
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  
  -- Calculate relative speed
  let relative_speed := train_speed_ms - jogger_speed_ms
  
  -- Calculate total distance to be covered
  let total_distance := initial_distance + train_length
  
  -- Calculate time taken
  let time_taken := total_distance / relative_speed
  
  -- Prove that the time taken is 32 seconds
  sorry

/-- The main theorem stating that the train will pass the jogger in 32 seconds -/
theorem train_passes_jogger_in_32_seconds : 
  train_passing_jogger_time 9 45 200 120 = 32 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_train_passes_jogger_in_32_seconds_l3640_364072


namespace NUMINAMATH_CALUDE_greatest_common_length_l3640_364041

theorem greatest_common_length (rope1 rope2 rope3 rope4 : ℕ) 
  (h1 : rope1 = 48) (h2 : rope2 = 64) (h3 : rope3 = 80) (h4 : rope4 = 120) :
  Nat.gcd rope1 (Nat.gcd rope2 (Nat.gcd rope3 rope4)) = 8 :=
sorry

end NUMINAMATH_CALUDE_greatest_common_length_l3640_364041


namespace NUMINAMATH_CALUDE_shopping_mall_uses_systematic_sampling_l3640_364062

/-- Represents a sampling method with given characteristics -/
structure SamplingMethod where
  initialSelection : Bool  -- True if initial selection is random
  fixedInterval : Bool     -- True if subsequent selections are at fixed intervals
  equalGroups : Bool       -- True if population is divided into equal-sized groups

/-- Definition of systematic sampling method -/
def isSystematicSampling (method : SamplingMethod) : Prop :=
  method.initialSelection ∧ method.fixedInterval ∧ method.equalGroups

/-- The sampling method used by the shopping mall -/
def shoppingMallMethod : SamplingMethod :=
  { initialSelection := true,  -- Randomly select one stub
    fixedInterval := true,     -- Sequentially take stubs at fixed intervals (every 50)
    equalGroups := true }      -- Each group has 50 invoice stubs

/-- Theorem stating that the shopping mall's method is systematic sampling -/
theorem shopping_mall_uses_systematic_sampling :
  isSystematicSampling shoppingMallMethod := by
  sorry


end NUMINAMATH_CALUDE_shopping_mall_uses_systematic_sampling_l3640_364062


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_l3640_364080

theorem lcm_gcf_ratio : 
  (Nat.lcm 180 594) / (Nat.gcd 180 594) = 330 := by sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_l3640_364080


namespace NUMINAMATH_CALUDE_wall_height_proof_l3640_364055

/-- The height of a wall built with a specific number of bricks of given dimensions. -/
theorem wall_height_proof (brick_length brick_width brick_height : ℝ)
  (wall_length wall_width : ℝ) (num_bricks : ℕ) :
  brick_length = 0.2 →
  brick_width = 0.1 →
  brick_height = 0.08 →
  wall_length = 10 →
  wall_width = 24.5 →
  num_bricks = 12250 →
  ∃ (h : ℝ), h = 0.08 ∧ num_bricks * (brick_length * brick_width * brick_height) = wall_length * h * wall_width :=
by sorry

end NUMINAMATH_CALUDE_wall_height_proof_l3640_364055


namespace NUMINAMATH_CALUDE_only_153_and_407_are_cube_sum_numbers_l3640_364095

-- Define a function to calculate the sum of cubes of digits
def sumOfCubesOfDigits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  hundreds^3 + tens^3 + ones^3

-- Define the property for a number to be a cube sum number
def isCubeSumNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ n = sumOfCubesOfDigits n

-- Theorem statement
theorem only_153_and_407_are_cube_sum_numbers :
  ∀ n : ℕ, isCubeSumNumber n ↔ n = 153 ∨ n = 407 := by sorry

end NUMINAMATH_CALUDE_only_153_and_407_are_cube_sum_numbers_l3640_364095


namespace NUMINAMATH_CALUDE_tan_equality_periodic_l3640_364001

theorem tan_equality_periodic (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1500 * π / 180) → n = 60 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_periodic_l3640_364001


namespace NUMINAMATH_CALUDE_shoe_alteration_cost_l3640_364096

def total_pairs : ℕ := 14
def sneaker_cost : ℕ := 37
def high_heel_cost : ℕ := 44
def boot_cost : ℕ := 52
def sneaker_pairs : ℕ := 5
def high_heel_pairs : ℕ := 4
def boot_pairs : ℕ := total_pairs - sneaker_pairs - high_heel_pairs
def discount_threshold : ℕ := 10
def discount_per_shoe : ℕ := 2

def total_cost : ℕ := 
  sneaker_pairs * 2 * sneaker_cost + 
  high_heel_pairs * 2 * high_heel_cost + 
  boot_pairs * 2 * boot_cost

def discounted_pairs : ℕ := max (total_pairs - discount_threshold) 0

def total_discount : ℕ := discounted_pairs * 2 * discount_per_shoe

theorem shoe_alteration_cost : 
  total_cost - total_discount = 1226 := by sorry

end NUMINAMATH_CALUDE_shoe_alteration_cost_l3640_364096


namespace NUMINAMATH_CALUDE_right_triangle_side_length_right_triangle_side_length_proof_l3640_364005

/-- Given a right triangle with hypotenuse length 13 and one non-hypotenuse side length 12,
    the length of the other side is 5. -/
theorem right_triangle_side_length : ℝ → ℝ → ℝ → Prop :=
  fun hypotenuse side1 side2 =>
    hypotenuse = 13 ∧ side1 = 12 ∧ side2 * side2 + side1 * side1 = hypotenuse * hypotenuse →
    side2 = 5

/-- Proof of the theorem -/
theorem right_triangle_side_length_proof : right_triangle_side_length 13 12 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_right_triangle_side_length_proof_l3640_364005


namespace NUMINAMATH_CALUDE_range_of_x_for_inequality_l3640_364063

theorem range_of_x_for_inequality (x : ℝ) : 
  (∀ m : ℝ, m ∈ Set.Icc 0 1 → m * x^2 - 2*x - m ≥ 2) ↔ x ∈ Set.Iic (-1) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_for_inequality_l3640_364063


namespace NUMINAMATH_CALUDE_larger_circle_radius_l3640_364094

theorem larger_circle_radius (r : ℝ) (R : ℝ) : 
  r = 2 →  -- radius of smaller circles
  R = r + r * Real.sqrt 3 →  -- radius of larger circle
  R = 2 + 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_larger_circle_radius_l3640_364094


namespace NUMINAMATH_CALUDE_coin_exchange_problem_l3640_364021

theorem coin_exchange_problem :
  ∃! (one_cent two_cent five_cent ten_cent : ℕ),
    two_cent = (3 * one_cent) / 5 ∧
    five_cent = (3 * two_cent) / 5 ∧
    ten_cent = (3 * five_cent) / 5 - 7 ∧
    50 < (one_cent + 2 * two_cent + 5 * five_cent + 10 * ten_cent) / 100 ∧
    (one_cent + 2 * two_cent + 5 * five_cent + 10 * ten_cent) / 100 < 100 ∧
    one_cent = 1375 ∧
    two_cent = 825 ∧
    five_cent = 495 ∧
    ten_cent = 290 := by
  sorry

end NUMINAMATH_CALUDE_coin_exchange_problem_l3640_364021
