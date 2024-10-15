import Mathlib

namespace NUMINAMATH_CALUDE_ana_win_probability_l3166_316623

/-- Represents a player in the coin flipping game -/
inductive Player
| Juan
| Carlos
| Manu
| Ana

/-- The coin flipping game with four players -/
def CoinFlipGame :=
  {players : List Player // players = [Player.Juan, Player.Carlos, Player.Manu, Player.Ana]}

/-- The probability of flipping heads on a single flip -/
def headsProbability : ℚ := 1/2

/-- The probability of Ana winning the game -/
def anaProbability (game : CoinFlipGame) : ℚ := 1/31

/-- Theorem stating that the probability of Ana winning is 1/31 -/
theorem ana_win_probability (game : CoinFlipGame) :
  anaProbability game = 1/31 := by
  sorry

end NUMINAMATH_CALUDE_ana_win_probability_l3166_316623


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3166_316601

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop := y = (3/4) * x ∨ y = -(3/4) * x

/-- Theorem: The asymptotes of the given hyperbola are y = ±(3/4)x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3166_316601


namespace NUMINAMATH_CALUDE_factor_expression_l3166_316667

theorem factor_expression (x : ℝ) : 54 * x^3 - 135 * x^5 = 27 * x^3 * (2 - 5 * x^2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3166_316667


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3166_316679

theorem imaginary_part_of_complex_fraction (z : ℂ) : 
  z = (Complex.I : ℂ) / (1 + 2 * Complex.I) → Complex.im z = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3166_316679


namespace NUMINAMATH_CALUDE_greatest_number_with_odd_factors_l3166_316685

-- Define a function to count positive factors
def count_positive_factors (n : ℕ) : ℕ := sorry

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := sorry

theorem greatest_number_with_odd_factors :
  ∀ n : ℕ, n < 150 → count_positive_factors n % 2 = 1 → n ≤ 144 :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_with_odd_factors_l3166_316685


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3166_316674

theorem arithmetic_sequence_sum : 2016 - 2017 + 2018 - 2019 + 2020 = 2018 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3166_316674


namespace NUMINAMATH_CALUDE_sqrt_equality_l3166_316690

theorem sqrt_equality (x : ℝ) (hx : x > 0) : -x * Real.sqrt (2 / x) = -Real.sqrt (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_l3166_316690


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l3166_316660

/-- Represents a parabola of the form x = 3y^2 - 9y + 5 -/
def Parabola := { p : ℝ × ℝ | p.1 = 3 * p.2^2 - 9 * p.2 + 5 }

/-- The x-coordinate of the x-intercept -/
def a : ℝ := 5

/-- The y-coordinates of the y-intercepts -/
def b : ℝ := sorry
def c : ℝ := sorry

theorem parabola_intercepts_sum : a + b + c = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l3166_316660


namespace NUMINAMATH_CALUDE_divisibility_by_101_l3166_316600

theorem divisibility_by_101 (n : ℕ+) :
  (∃ k : ℕ+, n = k * 101 - 1) ↔
  (101 ∣ n^3 + 1) ∧ (101 ∣ n^2 - 1) :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_101_l3166_316600


namespace NUMINAMATH_CALUDE_permutations_count_l3166_316648

/-- The total number of permutations of the string "HMMTHMMT" -/
def total_permutations : ℕ := 420

/-- The number of permutations containing the substring "HMMT" -/
def permutations_with_substring : ℕ := 60

/-- The number of cases over-counted -/
def over_counted_cases : ℕ := 1

/-- The number of permutations without the consecutive substring "HMMT" -/
def permutations_without_substring : ℕ := total_permutations - permutations_with_substring + over_counted_cases

theorem permutations_count : permutations_without_substring = 361 := by
  sorry

end NUMINAMATH_CALUDE_permutations_count_l3166_316648


namespace NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l3166_316697

theorem max_value_x_plus_reciprocal (x : ℝ) (h : 11 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 13 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l3166_316697


namespace NUMINAMATH_CALUDE_line_plane_parallelism_l3166_316607

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the containment relation for lines in planes
variable (containedIn : Line → Plane → Prop)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- State the theorem
theorem line_plane_parallelism 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β)
  (h_m_parallel_α : parallelLinePlane m α)
  (h_m_in_β : containedIn m β)
  (h_intersection : intersect α β = n) :
  parallelLine m n :=
sorry

end NUMINAMATH_CALUDE_line_plane_parallelism_l3166_316607


namespace NUMINAMATH_CALUDE_special_function_at_one_l3166_316622

/-- A monotonic function on positive real numbers satisfying f(f(x) - ln x) = 1 + e -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, ∀ y > 0, x < y → f x < f y) ∧
  (∀ x > 0, f (f x - Real.log x) = 1 + Real.exp 1)

/-- The value of f(1) for a special function f is e -/
theorem special_function_at_one (f : ℝ → ℝ) (h : special_function f) : f 1 = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_one_l3166_316622


namespace NUMINAMATH_CALUDE_hilltop_volleyball_club_members_l3166_316656

/-- Represents the Hilltop Volleyball Club inventory problem -/
theorem hilltop_volleyball_club_members :
  let sock_cost : ℕ := 6
  let tshirt_cost : ℕ := sock_cost + 7
  let items_per_member : ℕ := 3
  let cost_per_member : ℕ := items_per_member * (sock_cost + tshirt_cost)
  let total_cost : ℕ := 4026
  total_cost / cost_per_member = 71 :=
by sorry

end NUMINAMATH_CALUDE_hilltop_volleyball_club_members_l3166_316656


namespace NUMINAMATH_CALUDE_on_time_departure_rate_theorem_l3166_316699

/-- The number of flights that departed late -/
def late_flights : ℕ := 1

/-- The number of initial on-time flights -/
def initial_on_time : ℕ := 3

/-- The number of additional on-time flights needed -/
def additional_on_time : ℕ := 4

/-- The total number of flights -/
def total_flights : ℕ := late_flights + initial_on_time + additional_on_time

/-- The target on-time departure rate as a real number between 0 and 1 -/
def target_rate : ℝ := 0.875

theorem on_time_departure_rate_theorem :
  (initial_on_time + additional_on_time : ℝ) / total_flights > target_rate :=
sorry

end NUMINAMATH_CALUDE_on_time_departure_rate_theorem_l3166_316699


namespace NUMINAMATH_CALUDE_contrapositive_even_sum_l3166_316640

theorem contrapositive_even_sum (a b : ℤ) : 
  (¬(Even (a + b)) → ¬(Even a ∧ Even b)) ↔ 
  (∀ (a b : ℤ), (Even a ∧ Even b) → Even (a + b))ᶜ :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_even_sum_l3166_316640


namespace NUMINAMATH_CALUDE_inequality_for_positive_reals_l3166_316650

theorem inequality_for_positive_reals : ∀ x : ℝ, x > 0 → x + 4 / x ≥ 4 := by sorry

end NUMINAMATH_CALUDE_inequality_for_positive_reals_l3166_316650


namespace NUMINAMATH_CALUDE_area_of_circle_portion_l3166_316689

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := x^2 - 12*x + y^2 = 28

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = x - 4

/-- The region of interest -/
def region_of_interest (x y : ℝ) : Prop :=
  circle_equation x y ∧ y ≥ 0 ∧ y ≥ x - 4

/-- The area of the region of interest -/
noncomputable def area_of_region : ℝ := sorry

theorem area_of_circle_portion : area_of_region = 48 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_area_of_circle_portion_l3166_316689


namespace NUMINAMATH_CALUDE_bridge_length_l3166_316645

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 225 :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l3166_316645


namespace NUMINAMATH_CALUDE_grocery_store_bottles_l3166_316663

/-- The total number of soda bottles in a grocery store. -/
def total_bottles (regular : ℕ) (diet : ℕ) (lite : ℕ) : ℕ :=
  regular + diet + lite

/-- Theorem stating that the total number of bottles is 110. -/
theorem grocery_store_bottles : total_bottles 57 26 27 = 110 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_bottles_l3166_316663


namespace NUMINAMATH_CALUDE_ellipse_problem_l3166_316629

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The problem statement -/
theorem ellipse_problem (E : Ellipse) 
  (h_major_axis : E.a = 2 * Real.sqrt 2)
  (A B C : PointOnEllipse E)
  (h_A_vertex : A.x = E.a ∧ A.y = 0)
  (h_BC_origin : ∃ t : ℝ, B.x * t = C.x ∧ B.y * t = C.y)
  (h_B_first_quad : B.x > 0 ∧ B.y > 0)
  (h_BC_AB : Real.sqrt ((B.x - C.x)^2 + (B.y - C.y)^2) = 2 * Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2))
  (h_cos_ABC : (A.x - B.x) / Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) = 1/5) :
  (E.a^2 = 8 ∧ E.b^2 = 4) ∧
  ∃ (lower upper : ℝ), lower = Real.sqrt 14 / 2 ∧ upper = Real.sqrt 6 ∧
    ∀ (M N : PointOnEllipse E) (l : ℝ → ℝ),
      (∀ x y : ℝ, x^2 + y^2 = 1 → (y - l x) * (1 + l x * l x) = 0) →
      M ≠ N →
      (∃ t : ℝ, M.y = l M.x + t ∧ N.y = l N.x + t) →
      lower < (1/2 * Real.sqrt ((M.x - N.x)^2 + (M.y - N.y)^2)) ∧
      (1/2 * Real.sqrt ((M.x - N.x)^2 + (M.y - N.y)^2)) ≤ upper := by
  sorry

end NUMINAMATH_CALUDE_ellipse_problem_l3166_316629


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3166_316649

theorem complex_equation_solution (m : ℝ) :
  (2 : ℂ) / (1 - Complex.I) = 1 + m * Complex.I → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3166_316649


namespace NUMINAMATH_CALUDE_regular_soda_count_l3166_316611

/-- The number of bottles of regular soda in a grocery store -/
def regular_soda : ℕ := sorry

/-- The number of bottles of diet soda in a grocery store -/
def diet_soda : ℕ := 26

/-- The number of bottles of lite soda in a grocery store -/
def lite_soda : ℕ := 27

/-- The total number of soda bottles in a grocery store -/
def total_bottles : ℕ := 110

/-- Theorem stating that the number of bottles of regular soda is 57 -/
theorem regular_soda_count : regular_soda = 57 := by
  sorry

end NUMINAMATH_CALUDE_regular_soda_count_l3166_316611


namespace NUMINAMATH_CALUDE_vacant_seats_l3166_316653

/-- Given a hall with 600 seats where 62% are filled, prove that 228 seats are vacant. -/
theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) (h1 : total_seats = 600) (h2 : filled_percentage = 62/100) : 
  (total_seats : ℚ) * (1 - filled_percentage) = 228 := by
  sorry

end NUMINAMATH_CALUDE_vacant_seats_l3166_316653


namespace NUMINAMATH_CALUDE_trace_bag_weight_proof_l3166_316652

/-- The weight of one of Trace's shopping bags -/
def trace_bag_weight (
  trace_bags : ℕ)
  (gordon_bags : ℕ)
  (gordon_bag1_weight : ℕ)
  (gordon_bag2_weight : ℕ)
  (lola_bags : ℕ)
  (lola_total_weight : ℕ)
  : ℕ :=
2

theorem trace_bag_weight_proof 
  (trace_bags : ℕ)
  (gordon_bags : ℕ)
  (gordon_bag1_weight : ℕ)
  (gordon_bag2_weight : ℕ)
  (lola_bags : ℕ)
  (lola_total_weight : ℕ)
  (h1 : trace_bags = 5)
  (h2 : gordon_bags = 2)
  (h3 : gordon_bag1_weight = 3)
  (h4 : gordon_bag2_weight = 7)
  (h5 : lola_bags = 4)
  (h6 : trace_bags * trace_bag_weight trace_bags gordon_bags gordon_bag1_weight gordon_bag2_weight lola_bags lola_total_weight = gordon_bag1_weight + gordon_bag2_weight)
  (h7 : lola_total_weight = gordon_bag1_weight + gordon_bag2_weight - 2)
  : trace_bag_weight trace_bags gordon_bags gordon_bag1_weight gordon_bag2_weight lola_bags lola_total_weight = 2 := by
  sorry

#check trace_bag_weight_proof

end NUMINAMATH_CALUDE_trace_bag_weight_proof_l3166_316652


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_b_range_l3166_316628

def A : Set ℝ := {x | (x - 1) / (x + 1) < 0}
def B (a b : ℝ) : Set ℝ := {x | |x - b| < a}

theorem intersection_nonempty_implies_b_range :
  (∀ b : ℝ, (A ∩ B 1 b).Nonempty) →
  ∀ b : ℝ, -2 < b ∧ b < 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_b_range_l3166_316628


namespace NUMINAMATH_CALUDE_five_fridays_in_july_l3166_316626

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- June of year N -/
def june : Month := {
  days := 30,
  firstDay := DayOfWeek.Tuesday  -- Assuming the first Tuesday is on the 2nd
}

/-- July of year N -/
def july : Month := {
  days := 31,
  firstDay := DayOfWeek.Wednesday  -- Based on June's last day being Tuesday
}

/-- Count occurrences of a specific day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Main theorem -/
theorem five_fridays_in_july (h : countDayOccurrences june DayOfWeek.Tuesday = 5) :
  countDayOccurrences july DayOfWeek.Friday = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_fridays_in_july_l3166_316626


namespace NUMINAMATH_CALUDE_lamps_per_room_l3166_316698

/-- Given a hotel with 147 lamps and 21 rooms, prove that each room gets 7 lamps. -/
theorem lamps_per_room :
  let total_lamps : ℕ := 147
  let total_rooms : ℕ := 21
  let lamps_per_room : ℕ := total_lamps / total_rooms
  lamps_per_room = 7 := by sorry

end NUMINAMATH_CALUDE_lamps_per_room_l3166_316698


namespace NUMINAMATH_CALUDE_money_redistribution_theorem_l3166_316639

/-- Represents the money redistribution problem with Ben, Tom, and Max -/
theorem money_redistribution_theorem 
  (ben_start : ℕ) 
  (max_start_end : ℕ) 
  (ben_end : ℕ) 
  (tom_end : ℕ) 
  (max_end : ℕ) 
  (h1 : ben_start = 48)
  (h2 : max_start_end = 48)
  (h3 : max_end = max_start_end)
  (h4 : ben_end = ben_start)
  : ben_end + tom_end + max_end = 144 := by
  sorry

#check money_redistribution_theorem

end NUMINAMATH_CALUDE_money_redistribution_theorem_l3166_316639


namespace NUMINAMATH_CALUDE_simplify_expression_l3166_316643

theorem simplify_expression : (256 : ℝ) ^ (1/4) * (343 : ℝ) ^ (1/3) = 28 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3166_316643


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3166_316608

theorem inequality_solution_set (x : ℝ) : 
  (5 - x^2 > 4*x) ↔ (x > -5 ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3166_316608


namespace NUMINAMATH_CALUDE_cab_driver_income_day2_l3166_316625

def cab_driver_problem (day1 day2 day3 day4 day5 : ℕ) (average : ℚ) : Prop :=
  day1 = 250 ∧
  day3 = 750 ∧
  day4 = 400 ∧
  day5 = 500 ∧
  average = 460 ∧
  (day1 + day2 + day3 + day4 + day5) / 5 = average

theorem cab_driver_income_day2 :
  ∀ (day1 day2 day3 day4 day5 : ℕ) (average : ℚ),
    cab_driver_problem day1 day2 day3 day4 day5 average →
    day2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_income_day2_l3166_316625


namespace NUMINAMATH_CALUDE_line_not_parallel_when_planes_not_perpendicular_l3166_316627

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_not_parallel_when_planes_not_perpendicular
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : contained_in m β)
  (h3 : ¬ plane_perpendicular α β) :
  ¬ parallel l m :=
sorry

end NUMINAMATH_CALUDE_line_not_parallel_when_planes_not_perpendicular_l3166_316627


namespace NUMINAMATH_CALUDE_library_visitors_average_l3166_316681

theorem library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (month_days : ℕ) (sundays_in_month : ℕ) :
  sunday_visitors = 510 →
  other_day_visitors = 240 →
  month_days = 30 →
  sundays_in_month = 4 →
  (sundays_in_month * sunday_visitors + (month_days - sundays_in_month) * other_day_visitors) / month_days = 276 :=
by
  sorry

end NUMINAMATH_CALUDE_library_visitors_average_l3166_316681


namespace NUMINAMATH_CALUDE_wax_spilled_amount_l3166_316678

/-- The amount of wax spilled before use -/
def wax_spilled (car_wax SUV_wax initial_wax remaining_wax : ℕ) : ℕ :=
  initial_wax - (car_wax + SUV_wax) - remaining_wax

/-- Theorem stating that the amount of wax spilled is 2 ounces -/
theorem wax_spilled_amount :
  wax_spilled 3 4 11 2 = 2 := by sorry

end NUMINAMATH_CALUDE_wax_spilled_amount_l3166_316678


namespace NUMINAMATH_CALUDE_ellipse_y_axis_l3166_316612

theorem ellipse_y_axis (k : ℝ) (h : k < -1) :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  ∀ (x y : ℝ), (1 - k) * x^2 + y^2 = k^2 - 1 ↔ (x^2 / b^2) + (y^2 / a^2) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_y_axis_l3166_316612


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_3x_l3166_316670

theorem factorization_x_squared_minus_3x (x : ℝ) : x^2 - 3*x = x*(x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_3x_l3166_316670


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l3166_316609

theorem square_perimeter_ratio (s₁ s₂ : ℝ) (h : s₁ > 0 ∧ s₂ > 0) :
  s₂ = 2.5 * s₁ * Real.sqrt 2 / Real.sqrt 2 →
  (4 * s₂) / (4 * s₁) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l3166_316609


namespace NUMINAMATH_CALUDE_infinitely_many_superabundant_l3166_316686

/-- Sum of divisors function -/
def sigma (n : ℕ+) : ℕ := sorry

/-- Superabundant number -/
def is_superabundant (m : ℕ+) : Prop :=
  ∀ k : ℕ+, k < m → (sigma m : ℚ) / m > (sigma k : ℚ) / k

/-- There are infinitely many superabundant numbers -/
theorem infinitely_many_superabundant :
  ∀ N : ℕ, ∃ m : ℕ+, m > N ∧ is_superabundant m :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_superabundant_l3166_316686


namespace NUMINAMATH_CALUDE_odot_problem_l3166_316636

/-- Definition of the ⊙ operation -/
def odot (x y : ℝ) : ℝ := 2 * x + y

/-- Theorem statement -/
theorem odot_problem (a b : ℝ) (h : odot a (-6 * b) = 4) :
  odot (a - 5 * b) (a + b) = 6 := by
  sorry

end NUMINAMATH_CALUDE_odot_problem_l3166_316636


namespace NUMINAMATH_CALUDE_line_slope_angle_l3166_316666

theorem line_slope_angle (x y : ℝ) :
  x + Real.sqrt 3 * y - 2 = 0 →
  ∃ (m : ℝ), y = m * x + (2 * Real.sqrt 3) / 3 ∧
             m = -(Real.sqrt 3) / 3 ∧
             Real.tan (5 * Real.pi / 6) = m :=
by sorry

end NUMINAMATH_CALUDE_line_slope_angle_l3166_316666


namespace NUMINAMATH_CALUDE_square_sum_from_means_l3166_316673

theorem square_sum_from_means (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 24) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 168) : 
  x^2 + y^2 = 1968 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l3166_316673


namespace NUMINAMATH_CALUDE_liquid_X_percentage_l3166_316680

/-- The percentage of liquid X in solution A -/
def percent_X_in_A : ℝ := 1.464

/-- The percentage of liquid X in solution B -/
def percent_X_in_B : ℝ := 1.8

/-- The weight of solution A in grams -/
def weight_A : ℝ := 500

/-- The weight of solution B in grams -/
def weight_B : ℝ := 700

/-- The percentage of liquid X in the resulting mixture -/
def percent_X_in_mixture : ℝ := 1.66

theorem liquid_X_percentage :
  percent_X_in_A * weight_A / 100 + percent_X_in_B * weight_B / 100 =
  percent_X_in_mixture * (weight_A + weight_B) / 100 := by
  sorry

end NUMINAMATH_CALUDE_liquid_X_percentage_l3166_316680


namespace NUMINAMATH_CALUDE_typing_area_percentage_l3166_316617

/-- Calculates the percentage of a rectangular sheet used for typing, given the sheet dimensions and margins. -/
theorem typing_area_percentage (sheet_length sheet_width side_margin top_bottom_margin : ℝ) 
  (sheet_length_pos : 0 < sheet_length)
  (sheet_width_pos : 0 < sheet_width)
  (side_margin_pos : 0 < side_margin)
  (top_bottom_margin_pos : 0 < top_bottom_margin)
  (side_margin_fit : 2 * side_margin < sheet_length)
  (top_bottom_margin_fit : 2 * top_bottom_margin < sheet_width) :
  let total_area := sheet_length * sheet_width
  let typing_length := sheet_length - 2 * side_margin
  let typing_width := sheet_width - 2 * top_bottom_margin
  let typing_area := typing_length * typing_width
  (typing_area / total_area) * 100 = 64 :=
sorry

end NUMINAMATH_CALUDE_typing_area_percentage_l3166_316617


namespace NUMINAMATH_CALUDE_square_roots_sum_product_l3166_316630

theorem square_roots_sum_product (m n : ℂ) : 
  m ^ 2 = 2023 → n ^ 2 = 2023 → m + 2 * m * n + n = -4046 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_sum_product_l3166_316630


namespace NUMINAMATH_CALUDE_correct_proposition_l3166_316675

theorem correct_proposition :
  let p := ∀ x : ℝ, 2 * x < 3 * x
  let q := ∃ x : ℝ, x^3 = 1 - x^2
  ¬p ∧ q := by sorry

end NUMINAMATH_CALUDE_correct_proposition_l3166_316675


namespace NUMINAMATH_CALUDE_suit_price_increase_l3166_316695

/-- Proves that the percentage increase in the price of a suit was 20% --/
theorem suit_price_increase (original_price : ℝ) (coupon_discount : ℝ) (final_price : ℝ) :
  original_price = 150 →
  coupon_discount = 0.2 →
  final_price = 144 →
  ∃ (increase_percentage : ℝ),
    increase_percentage = 20 ∧
    final_price = (1 - coupon_discount) * (original_price * (1 + increase_percentage / 100)) :=
by sorry

end NUMINAMATH_CALUDE_suit_price_increase_l3166_316695


namespace NUMINAMATH_CALUDE_alkaline_probability_l3166_316603

/-- Represents the total number of solutions -/
def total_solutions : ℕ := 5

/-- Represents the number of alkaline solutions -/
def alkaline_solutions : ℕ := 2

/-- Represents the probability of selecting an alkaline solution -/
def probability : ℚ := alkaline_solutions / total_solutions

/-- Theorem stating that the probability of selecting an alkaline solution is 2/5 -/
theorem alkaline_probability : probability = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_alkaline_probability_l3166_316603


namespace NUMINAMATH_CALUDE_tenth_day_is_monday_l3166_316658

/-- Represents the days of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a month with its starting day and number of days -/
structure Month where
  startDay : DayOfWeek
  numDays : Nat

/-- Represents Teacher Zhang's running schedule -/
def runningDays : List DayOfWeek := [DayOfWeek.Monday, DayOfWeek.Saturday, DayOfWeek.Sunday]

/-- Calculate the day of the week for a given day in the month -/
def dayOfWeek (m : Month) (day : Nat) : DayOfWeek :=
  sorry

/-- The total running time in a month in minutes -/
def totalRunningTime : Nat := 5 * 60

/-- The theorem to be proved -/
theorem tenth_day_is_monday (m : Month) 
  (h1 : m.startDay = DayOfWeek.Saturday) 
  (h2 : m.numDays = 31) 
  (h3 : totalRunningTime = 5 * 60) : 
  dayOfWeek m 10 = DayOfWeek.Monday :=
sorry

end NUMINAMATH_CALUDE_tenth_day_is_monday_l3166_316658


namespace NUMINAMATH_CALUDE_water_one_tenth_after_pourings_l3166_316614

/-- The fraction of water remaining after n pourings -/
def waterRemaining (n : ℕ) : ℚ :=
  3 / (n + 3)

/-- The number of pourings required to reach one-tenth of the original volume -/
def pouringsToOneTenth : ℕ := 27

theorem water_one_tenth_after_pourings :
  waterRemaining pouringsToOneTenth = 1 / 10 := by
  sorry

#eval waterRemaining pouringsToOneTenth

end NUMINAMATH_CALUDE_water_one_tenth_after_pourings_l3166_316614


namespace NUMINAMATH_CALUDE_superadditive_continuous_function_is_linear_l3166_316655

/-- A function satisfying the given conditions -/
def SuperadditiveContinuousFunction (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ f 0 = 0 ∧ ∀ x y : ℝ, f (x + y) ≥ f x + f y

/-- The main theorem -/
theorem superadditive_continuous_function_is_linear
    (f : ℝ → ℝ) (hf : SuperadditiveContinuousFunction f) :
    ∃ a : ℝ, ∀ x : ℝ, f x = a * x := by
  sorry

end NUMINAMATH_CALUDE_superadditive_continuous_function_is_linear_l3166_316655


namespace NUMINAMATH_CALUDE_parabola_symmetric_axis_l3166_316691

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop :=
  y = (1/2) * x^2 - 6*x + 21

/-- The symmetric axis of the parabola -/
def symmetric_axis (x : ℝ) : Prop :=
  x = 6

/-- Theorem: The symmetric axis of the given parabola is x = 6 -/
theorem parabola_symmetric_axis :
  ∀ x y : ℝ, parabola x y → symmetric_axis x :=
by sorry

end NUMINAMATH_CALUDE_parabola_symmetric_axis_l3166_316691


namespace NUMINAMATH_CALUDE_largest_tile_size_378_525_l3166_316619

/-- The largest square tile size that can exactly pave a rectangular courtyard -/
def largest_tile_size (length width : ℕ) : ℕ :=
  Nat.gcd length width

/-- Theorem: The largest square tile size for a 378 cm by 525 cm courtyard is 21 cm -/
theorem largest_tile_size_378_525 :
  largest_tile_size 378 525 = 21 := by
  sorry

#eval largest_tile_size 378 525

end NUMINAMATH_CALUDE_largest_tile_size_378_525_l3166_316619


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3166_316661

theorem solution_set_of_inequality (x : ℝ) :
  x^2 < 2*x ↔ 0 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3166_316661


namespace NUMINAMATH_CALUDE_course_size_l3166_316646

theorem course_size (total : ℕ) 
  (grade_A : ℕ → Prop) (grade_B : ℕ → Prop) (grade_C : ℕ → Prop) (grade_D : ℕ → Prop)
  (h1 : ∀ n, grade_A n ↔ n = total / 5)
  (h2 : ∀ n, grade_B n ↔ n = total / 4)
  (h3 : ∀ n, grade_C n ↔ n = total / 2)
  (h4 : ∀ n, grade_D n ↔ n = 25)
  (h5 : ∀ n, n ≤ total → (grade_A n ∨ grade_B n ∨ grade_C n ∨ grade_D n))
  (h6 : ∀ n, (grade_A n → ¬grade_B n ∧ ¬grade_C n ∧ ¬grade_D n) ∧
             (grade_B n → ¬grade_A n ∧ ¬grade_C n ∧ ¬grade_D n) ∧
             (grade_C n → ¬grade_A n ∧ ¬grade_B n ∧ ¬grade_D n) ∧
             (grade_D n → ¬grade_A n ∧ ¬grade_B n ∧ ¬grade_C n)) :
  total = 500 := by
sorry

end NUMINAMATH_CALUDE_course_size_l3166_316646


namespace NUMINAMATH_CALUDE_cereal_consumption_time_l3166_316651

/-- The time taken for two people to consume a given amount of cereal together,
    given their individual consumption rates. -/
def time_to_consume (rate1 rate2 amount : ℚ) : ℚ :=
  amount / (rate1 + rate2)

/-- Mr. Fat's cereal consumption rate in pounds per minute -/
def fat_rate : ℚ := 1 / 25

/-- Mr. Thin's cereal consumption rate in pounds per minute -/
def thin_rate : ℚ := 1 / 35

/-- The amount of cereal to be consumed in pounds -/
def cereal_amount : ℚ := 5

theorem cereal_consumption_time :
  ∃ (t : ℚ), abs (t - time_to_consume fat_rate thin_rate cereal_amount) < 1 ∧
             t = 73 := by sorry

end NUMINAMATH_CALUDE_cereal_consumption_time_l3166_316651


namespace NUMINAMATH_CALUDE_spinach_amount_l3166_316669

/-- The initial amount of raw spinach in ounces -/
def initial_spinach : ℝ := 40

/-- The percentage of initial volume after cooking -/
def cooking_ratio : ℝ := 0.20

/-- The amount of cream cheese in ounces -/
def cream_cheese : ℝ := 6

/-- The amount of eggs in ounces -/
def eggs : ℝ := 4

/-- The total volume of the quiche in ounces -/
def total_volume : ℝ := 18

theorem spinach_amount :
  initial_spinach * cooking_ratio + cream_cheese + eggs = total_volume :=
by sorry

end NUMINAMATH_CALUDE_spinach_amount_l3166_316669


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l3166_316672

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  interval : ℕ
  first_element : ℕ

/-- Generates the nth element of a systematic sample -/
def nth_element (s : SystematicSample) (n : ℕ) : ℕ :=
  s.first_element + (n - 1) * s.interval

theorem systematic_sample_theorem (s : SystematicSample) 
  (h1 : s.population_size = 52)
  (h2 : s.sample_size = 4)
  (h3 : s.first_element = 5)
  (h4 : nth_element s 3 = 31)
  (h5 : nth_element s 4 = 44) :
  nth_element s 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_theorem_l3166_316672


namespace NUMINAMATH_CALUDE_valid_integers_count_l3166_316631

/-- The number of digits in the integers we're counting -/
def num_digits : ℕ := 8

/-- The number of choices for the first digit (2-9) -/
def first_digit_choices : ℕ := 8

/-- The number of choices for each subsequent digit (0-9) -/
def other_digit_choices : ℕ := 10

/-- The number of different 8-digit positive integers where the first digit cannot be 0 or 1 -/
def count_valid_integers : ℕ := first_digit_choices * (other_digit_choices ^ (num_digits - 1))

theorem valid_integers_count :
  count_valid_integers = 80000000 := by sorry

end NUMINAMATH_CALUDE_valid_integers_count_l3166_316631


namespace NUMINAMATH_CALUDE_complex_magnitude_l3166_316682

theorem complex_magnitude (z : ℂ) : z = 1 + 2*I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3166_316682


namespace NUMINAMATH_CALUDE_complete_square_l3166_316671

theorem complete_square (x : ℝ) : 
  (x^2 + 6*x + 5 = 0) ↔ ((x + 3)^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_complete_square_l3166_316671


namespace NUMINAMATH_CALUDE_l_shape_tiling_l3166_316644

/-- Number of ways to tile an L-shaped region with dominos -/
def tiling_count (m n : ℕ) : ℕ :=
  sorry

/-- The L-shaped region is formed by attaching two 2 by 5 rectangles to adjacent sides of a 2 by 2 square -/
theorem l_shape_tiling :
  tiling_count 5 5 = 208 :=
sorry

end NUMINAMATH_CALUDE_l_shape_tiling_l3166_316644


namespace NUMINAMATH_CALUDE_simplify_algebraic_expression_l3166_316688

theorem simplify_algebraic_expression (x : ℝ) 
  (h1 : x ≠ 1) (h2 : x ≠ -1) (h3 : x ≠ 0) : 
  (x - x / (x + 1)) / (1 + 1 / (x^2 - 1)) = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_algebraic_expression_l3166_316688


namespace NUMINAMATH_CALUDE_optimal_prevention_plan_l3166_316606

/-- Represents the cost and effectiveness of a preventive measure -/
structure PreventiveMeasure where
  cost : ℝ
  effectiveness : ℝ

/-- Calculates the total cost given preventive measures and event parameters -/
def totalCost (measures : List PreventiveMeasure) (eventProbability : ℝ) (eventLoss : ℝ) : ℝ :=
  (measures.map (·.cost)).sum + eventLoss * (1 - (measures.map (·.effectiveness)).prod)

theorem optimal_prevention_plan (eventProbability : ℝ) (eventLoss : ℝ)
  (measureA : PreventiveMeasure) (measureB : PreventiveMeasure) :
  eventProbability = 0.3 →
  eventLoss = 4 →
  measureA.cost = 0.45 →
  measureB.cost = 0.3 →
  measureA.effectiveness = 0.9 →
  measureB.effectiveness = 0.85 →
  totalCost [measureA, measureB] eventProbability eventLoss <
    min (totalCost [] eventProbability eventLoss)
      (min (totalCost [measureA] eventProbability eventLoss)
        (totalCost [measureB] eventProbability eventLoss)) :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_prevention_plan_l3166_316606


namespace NUMINAMATH_CALUDE_solution_exists_for_quadratic_cubic_congruence_l3166_316616

theorem solution_exists_for_quadratic_cubic_congruence (p : ℕ) (hp : Prime p) (a : ℤ) :
  ∃ (x y : ℤ), (x^2 + y^3) % p = a % p := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_for_quadratic_cubic_congruence_l3166_316616


namespace NUMINAMATH_CALUDE_car_rental_cost_per_mile_l3166_316624

theorem car_rental_cost_per_mile 
  (rental_cost : ℝ) 
  (gas_price : ℝ) 
  (gas_amount : ℝ) 
  (miles_driven : ℝ) 
  (total_expense : ℝ) 
  (h1 : rental_cost = 150) 
  (h2 : gas_price = 3.5) 
  (h3 : gas_amount = 8) 
  (h4 : miles_driven = 320) 
  (h5 : total_expense = 338) :
  (total_expense - (rental_cost + gas_price * gas_amount)) / miles_driven = 0.5 := by
sorry


end NUMINAMATH_CALUDE_car_rental_cost_per_mile_l3166_316624


namespace NUMINAMATH_CALUDE_rhea_children_eggs_l3166_316657

/-- The number of eggs eaten by Rhea's son and daughter every morning -/
def eggs_eaten_by_children (
  trays_per_week : ℕ)  -- Number of trays bought per week
  (eggs_per_tray : ℕ)  -- Number of eggs per tray
  (eggs_eaten_by_parents : ℕ)  -- Number of eggs eaten by parents per night
  (eggs_not_eaten : ℕ)  -- Number of eggs not eaten per week
  : ℕ :=
  trays_per_week * eggs_per_tray - 7 * eggs_eaten_by_parents - eggs_not_eaten

/-- Theorem stating that Rhea's son and daughter eat 14 eggs every morning -/
theorem rhea_children_eggs : 
  eggs_eaten_by_children 2 24 4 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_rhea_children_eggs_l3166_316657


namespace NUMINAMATH_CALUDE_last_digit_for_multiple_of_five_l3166_316694

theorem last_digit_for_multiple_of_five (n : ℕ) : 
  (71360 ≤ n ∧ n ≤ 71369) ∧ (n % 5 = 0) → (n % 10 = 0 ∨ n % 10 = 5) :=
by sorry

end NUMINAMATH_CALUDE_last_digit_for_multiple_of_five_l3166_316694


namespace NUMINAMATH_CALUDE_holly_initial_milk_l3166_316615

/-- Represents the amount of chocolate milk Holly has throughout the day -/
structure ChocolateMilk where
  initial : ℕ
  breakfast : ℕ
  lunch_purchased : ℕ
  lunch : ℕ
  dinner : ℕ
  final : ℕ

/-- The conditions of Holly's chocolate milk consumption -/
def holly_milk : ChocolateMilk where
  breakfast := 8
  lunch_purchased := 64
  lunch := 8
  dinner := 8
  final := 56
  initial := 0  -- This will be proven

/-- Theorem stating that Holly's initial amount of chocolate milk was 80 ounces -/
theorem holly_initial_milk :
  holly_milk.initial = 80 :=
by sorry

end NUMINAMATH_CALUDE_holly_initial_milk_l3166_316615


namespace NUMINAMATH_CALUDE_chef_cooks_25_wings_l3166_316634

/-- The number of additional chicken wings cooked by the chef for a group of friends -/
def additional_chicken_wings (num_friends : ℕ) (pre_cooked : ℕ) (wings_per_person : ℕ) : ℕ :=
  num_friends * wings_per_person - pre_cooked

/-- Theorem stating that for 9 friends, 2 pre-cooked wings, and 3 wings per person, 
    the chef needs to cook 25 additional wings -/
theorem chef_cooks_25_wings : additional_chicken_wings 9 2 3 = 25 := by
  sorry

end NUMINAMATH_CALUDE_chef_cooks_25_wings_l3166_316634


namespace NUMINAMATH_CALUDE_james_friends_count_l3166_316632

/-- The number of pages James writes per letter -/
def pages_per_letter : ℕ := 3

/-- The number of times James writes letters per week -/
def times_per_week : ℕ := 2

/-- The total number of pages James writes in a year -/
def total_pages_per_year : ℕ := 624

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

theorem james_friends_count :
  (total_pages_per_year / weeks_per_year / times_per_week) / pages_per_letter = 2 := by
  sorry

end NUMINAMATH_CALUDE_james_friends_count_l3166_316632


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l3166_316647

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 2| = |x - 1| + |x - 4| :=
by
  -- The unique solution is x = 4
  use 4
  constructor
  · -- Prove that x = 4 satisfies the equation
    sorry
  · -- Prove that any solution must equal 4
    sorry

#check unique_solution_absolute_value_equation

end NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l3166_316647


namespace NUMINAMATH_CALUDE_dividend_calculation_l3166_316604

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 14)
  (h2 : quotient = 12)
  (h3 : remainder = 8) :
  divisor * quotient + remainder = 176 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3166_316604


namespace NUMINAMATH_CALUDE_james_record_beat_l3166_316641

/-- James' football scoring record --/
theorem james_record_beat (touchdowns_per_game : ℕ) (points_per_touchdown : ℕ) 
  (games_in_season : ℕ) (two_point_conversions : ℕ) (old_record : ℕ) : 
  touchdowns_per_game = 4 →
  points_per_touchdown = 6 →
  games_in_season = 15 →
  two_point_conversions = 6 →
  old_record = 300 →
  (touchdowns_per_game * points_per_touchdown * games_in_season + 
   two_point_conversions * 2) - old_record = 72 := by
  sorry

#check james_record_beat

end NUMINAMATH_CALUDE_james_record_beat_l3166_316641


namespace NUMINAMATH_CALUDE_combined_work_rate_l3166_316664

/-- The combined work rate of three workers given their individual work rates -/
theorem combined_work_rate 
  (rate_A : ℚ) 
  (rate_B : ℚ) 
  (rate_C : ℚ) 
  (h_A : rate_A = 1 / 12)
  (h_B : rate_B = 1 / 6)
  (h_C : rate_C = 1 / 18) : 
  rate_A + rate_B + rate_C = 11 / 36 := by
  sorry

#check combined_work_rate

end NUMINAMATH_CALUDE_combined_work_rate_l3166_316664


namespace NUMINAMATH_CALUDE_specific_arrangement_probability_l3166_316633

/-- The number of X tiles -/
def num_x : ℕ := 5

/-- The number of O tiles -/
def num_o : ℕ := 4

/-- The total number of tiles -/
def total_tiles : ℕ := num_x + num_o

/-- The probability of the specific arrangement -/
def prob_specific_arrangement : ℚ := 1 / (total_tiles.choose num_x)

theorem specific_arrangement_probability :
  prob_specific_arrangement = 1 / 126 := by sorry

end NUMINAMATH_CALUDE_specific_arrangement_probability_l3166_316633


namespace NUMINAMATH_CALUDE_triangle_arithmetic_geometric_sequence_l3166_316692

theorem triangle_arithmetic_geometric_sequence (A B C : ℝ) (a b c : ℝ) : 
  -- Angles form an arithmetic sequence
  2 * B = A + C →
  -- Sum of angles in a triangle
  A + B + C = π →
  -- Sides form a geometric sequence
  b^2 = a * c →
  -- Law of cosines
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B →
  -- Conclusions
  Real.cos B = 1 / 2 ∧ Real.sin A * Real.sin C = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_geometric_sequence_l3166_316692


namespace NUMINAMATH_CALUDE_area_ratio_circumference_ratio_l3166_316662

-- Define a circular park
structure CircularPark where
  diameter : ℝ
  diameter_pos : diameter > 0

-- Define the enlarged park
def enlargedPark (park : CircularPark) : CircularPark :=
  { diameter := 3 * park.diameter
    diameter_pos := by
      have h : park.diameter > 0 := park.diameter_pos
      linarith }

-- Theorem for area ratio
theorem area_ratio (park : CircularPark) :
  (enlargedPark park).diameter^2 / park.diameter^2 = 9 := by
sorry

-- Theorem for circumference ratio
theorem circumference_ratio (park : CircularPark) :
  (enlargedPark park).diameter / park.diameter = 3 := by
sorry

end NUMINAMATH_CALUDE_area_ratio_circumference_ratio_l3166_316662


namespace NUMINAMATH_CALUDE_fraction_of_rotten_berries_l3166_316677

theorem fraction_of_rotten_berries 
  (total_berries : ℕ) 
  (berries_to_sell : ℕ) 
  (h1 : total_berries = 60) 
  (h2 : berries_to_sell = 20) 
  (h3 : berries_to_sell * 2 ≤ total_berries) :
  (total_berries - berries_to_sell * 2 : ℚ) / total_berries = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_rotten_berries_l3166_316677


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l3166_316668

/-- Given a line segment with one endpoint at (10, 4) and midpoint at (4, -8),
    the sum of the coordinates of the other endpoint is -22. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
    (4 = (x + 10) / 2) → 
    (-8 = (y + 4) / 2) → 
    x + y = -22 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l3166_316668


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l3166_316696

/-- Given a point (r, θ) in polar coordinates and a line θ = α, 
    the symmetric point with respect to this line has coordinates (r, 2α - θ) -/
def symmetric_point (r : ℝ) (θ : ℝ) (α : ℝ) : ℝ × ℝ := (r, 2*α - θ)

/-- The point symmetric to (3, π/2) with respect to the line θ = π/6 
    has polar coordinates (3, -π/6) -/
theorem symmetric_point_theorem : 
  symmetric_point 3 (π/2) (π/6) = (3, -π/6) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_theorem_l3166_316696


namespace NUMINAMATH_CALUDE_program_sum_equals_expected_sum_l3166_316659

def program_sum (n : ℕ) : ℕ :=
  let rec inner_sum (k : ℕ) : ℕ :=
    match k with
    | 0 => 0
    | k+1 => k+1 + inner_sum k
  let rec outer_sum (i : ℕ) : ℕ :=
    match i with
    | 0 => 0
    | i+1 => inner_sum (i+1) + outer_sum i
  outer_sum n

def expected_sum (n : ℕ) : ℕ :=
  let rec sum_of_sums (k : ℕ) : ℕ :=
    match k with
    | 0 => 0
    | k+1 => (List.range (k+1)).sum + sum_of_sums k
  sum_of_sums n

theorem program_sum_equals_expected_sum (n : ℕ) :
  program_sum n = expected_sum n := by
  sorry

end NUMINAMATH_CALUDE_program_sum_equals_expected_sum_l3166_316659


namespace NUMINAMATH_CALUDE_exam_comparison_l3166_316621

theorem exam_comparison (total_items : ℕ) (liza_percentage : ℚ) (rose_incorrect : ℕ) : 
  total_items = 60 →
  liza_percentage = 90 / 100 →
  rose_incorrect = 4 →
  (rose_incorrect : ℚ) < total_items →
  ∃ (liza_correct rose_correct : ℕ),
    (liza_correct : ℚ) = liza_percentage * total_items ∧
    rose_correct = total_items - rose_incorrect ∧
    rose_correct - liza_correct = 2 := by
sorry

end NUMINAMATH_CALUDE_exam_comparison_l3166_316621


namespace NUMINAMATH_CALUDE_det_of_matrix_l3166_316654

def matrix : Matrix (Fin 2) (Fin 2) ℤ := !![8, 5; -2, 3]

theorem det_of_matrix : Matrix.det matrix = 34 := by sorry

end NUMINAMATH_CALUDE_det_of_matrix_l3166_316654


namespace NUMINAMATH_CALUDE_julio_fish_count_l3166_316683

/-- Calculates the number of fish Julio has after fishing for a given number of hours and losing some fish. -/
def fish_count (catch_rate : ℕ) (hours : ℕ) (fish_lost : ℕ) : ℕ :=
  catch_rate * hours - fish_lost

/-- Theorem stating that Julio has 48 fish after 9 hours of fishing at 7 fish per hour and losing 15 fish. -/
theorem julio_fish_count :
  fish_count 7 9 15 = 48 := by
  sorry

end NUMINAMATH_CALUDE_julio_fish_count_l3166_316683


namespace NUMINAMATH_CALUDE_circle_properties_l3166_316602

/-- Given a circle with area 81π cm², prove its radius is 9 cm and circumference is 18π cm. -/
theorem circle_properties (A : ℝ) (h : A = 81 * Real.pi) :
  ∃ (r C : ℝ), r = 9 ∧ C = 18 * Real.pi ∧ A = Real.pi * r^2 ∧ C = 2 * Real.pi * r := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l3166_316602


namespace NUMINAMATH_CALUDE_darias_remaining_balance_l3166_316684

/-- Calculates the remaining amount owed on a credit card after an initial payment --/
def remaining_balance (saved : ℕ) (couch_price : ℕ) (table_price : ℕ) (lamp_price : ℕ) : ℕ :=
  (couch_price + table_price + lamp_price) - saved

/-- Theorem stating that Daria's remaining balance is $400 --/
theorem darias_remaining_balance :
  remaining_balance 500 750 100 50 = 400 := by
  sorry

end NUMINAMATH_CALUDE_darias_remaining_balance_l3166_316684


namespace NUMINAMATH_CALUDE_dragon_eye_centering_l3166_316635

-- Define a circle with a figure drawn on it
structure FiguredCircle where
  center : ℝ × ℝ
  radius : ℝ
  figure : Set (ℝ × ℝ)

-- Define a point that represents the dragon's eye
def dragonEye (fc : FiguredCircle) : ℝ × ℝ := 
  sorry

-- State the theorem
theorem dragon_eye_centering 
  (c1 c2 : FiguredCircle) 
  (h_congruent : c1.radius = c2.radius) 
  (h_identical_figures : c1.figure = c2.figure) 
  (h_c1_centered : dragonEye c1 = c1.center) 
  (h_c2_not_centered : dragonEye c2 ≠ c2.center) : 
  ∃ (part1 part2 : Set (ℝ × ℝ)), 
    (∃ (c3 : FiguredCircle), 
      c3.radius = c1.radius ∧ 
      c3.figure = c1.figure ∧ 
      dragonEye c3 = c3.center ∧ 
      c3.figure = part1 ∪ part2 ∧ 
      part1 ∩ part2 = ∅ ∧ 
      part1 ∪ part2 = c2.figure) :=
sorry

end NUMINAMATH_CALUDE_dragon_eye_centering_l3166_316635


namespace NUMINAMATH_CALUDE_prime_power_sum_l3166_316687

theorem prime_power_sum (n : ℕ) : Prime (n^4 + 4^n) ↔ n = 1 :=
sorry

end NUMINAMATH_CALUDE_prime_power_sum_l3166_316687


namespace NUMINAMATH_CALUDE_new_person_weight_l3166_316618

theorem new_person_weight (n : ℕ) (old_weight average_increase : ℝ) :
  n = 10 →
  old_weight = 65 →
  average_increase = 3.2 →
  ∃ (new_weight : ℝ),
    new_weight = old_weight + n * average_increase ∧
    new_weight = 97 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l3166_316618


namespace NUMINAMATH_CALUDE_rectangular_box_dimensions_l3166_316605

theorem rectangular_box_dimensions :
  ∃! (a b c : ℕ),
    2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧
    Even a ∧ Even b ∧ Even c ∧
    2 * (a * b + a * c + b * c) = 4 * (a + b + c) ∧
    a = 2 ∧ b = 2 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_dimensions_l3166_316605


namespace NUMINAMATH_CALUDE_ratio_equality_l3166_316638

theorem ratio_equality (a b : ℝ) (h : a / b = 4 / 7) : 7 * a = 4 * b := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3166_316638


namespace NUMINAMATH_CALUDE_linda_max_servings_l3166_316676

/-- Represents the recipe and available ingredients for making smoothies -/
structure SmoothieIngredients where
  recipe_bananas : ℕ        -- Bananas needed for 4 servings
  recipe_yogurt : ℕ         -- Cups of yogurt needed for 4 servings
  recipe_honey : ℕ          -- Tablespoons of honey needed for 4 servings
  available_bananas : ℕ     -- Bananas Linda has
  available_yogurt : ℕ      -- Cups of yogurt Linda has
  available_honey : ℕ       -- Tablespoons of honey Linda has

/-- Calculates the maximum number of servings that can be made -/
def max_servings (ingredients : SmoothieIngredients) : ℕ :=
  min
    (ingredients.available_bananas * 4 / ingredients.recipe_bananas)
    (min
      (ingredients.available_yogurt * 4 / ingredients.recipe_yogurt)
      (ingredients.available_honey * 4 / ingredients.recipe_honey))

/-- Theorem stating the maximum number of servings Linda can make -/
theorem linda_max_servings :
  let ingredients := SmoothieIngredients.mk 3 2 1 10 9 4
  max_servings ingredients = 13 := by
  sorry


end NUMINAMATH_CALUDE_linda_max_servings_l3166_316676


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3166_316665

def I : Finset Nat := {0, 1, 2, 3, 4}
def M : Finset Nat := {1, 2, 3}
def N : Finset Nat := {0, 3, 4}

theorem complement_intersection_theorem :
  (I \ M) ∩ N = {0, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3166_316665


namespace NUMINAMATH_CALUDE_equation_solution_l3166_316642

theorem equation_solution : ∃ x : ℝ, 3 * x + 6 = |(-5 * 4 + 2)| ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3166_316642


namespace NUMINAMATH_CALUDE_gcd_1337_382_l3166_316693

theorem gcd_1337_382 : Nat.gcd 1337 382 = 191 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1337_382_l3166_316693


namespace NUMINAMATH_CALUDE_binomial_sum_l3166_316610

theorem binomial_sum (a a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (1 + x)^6 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 63 :=
by sorry

end NUMINAMATH_CALUDE_binomial_sum_l3166_316610


namespace NUMINAMATH_CALUDE_polygon_has_12_sides_l3166_316620

/-- A polygon has n sides. -/
structure Polygon where
  n : ℕ

/-- The sum of interior angles of a polygon with n sides. -/
def sumInteriorAngles (p : Polygon) : ℝ :=
  (p.n - 2) * 180

/-- The sum of exterior angles of any polygon. -/
def sumExteriorAngles : ℝ := 360

/-- Theorem: A polygon has 12 sides if the sum of its interior angles
    is equal to five times the sum of its exterior angles. -/
theorem polygon_has_12_sides (p : Polygon) : 
  sumInteriorAngles p = 5 * sumExteriorAngles → p.n = 12 := by
  sorry

end NUMINAMATH_CALUDE_polygon_has_12_sides_l3166_316620


namespace NUMINAMATH_CALUDE_M_equals_N_l3166_316613

/-- Definition of set M -/
def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}

/-- Definition of set N -/
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

/-- Theorem stating that M equals N -/
theorem M_equals_N : M = N := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_l3166_316613


namespace NUMINAMATH_CALUDE_value_of_expression_l3166_316637

theorem value_of_expression (x : ℝ) (h : 5 * x - 3 = 7) : 3 * x^2 + 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3166_316637
