import Mathlib

namespace NUMINAMATH_CALUDE_good_numbers_in_set_l2023_202359

-- Define what a "good number" is
def isGoodNumber (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), Function.Bijective a ∧
    ∀ k : Fin n, ∃ m : ℕ, (k.val + 1 + (a k).val + 1 : ℕ) = m * m

-- Theorem statement
theorem good_numbers_in_set :
  isGoodNumber 11 = false ∧
  isGoodNumber 13 = true ∧
  isGoodNumber 15 = true ∧
  isGoodNumber 17 = true ∧
  isGoodNumber 19 = true :=
by sorry

end NUMINAMATH_CALUDE_good_numbers_in_set_l2023_202359


namespace NUMINAMATH_CALUDE_fourth_term_of_sequence_l2023_202328

theorem fourth_term_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S n = 4 * n^2) →
  a 4 = 28 :=
sorry

end NUMINAMATH_CALUDE_fourth_term_of_sequence_l2023_202328


namespace NUMINAMATH_CALUDE_power_difference_evaluation_l2023_202356

theorem power_difference_evaluation : (3^4)^3 - (4^3)^4 = -16245775 := by sorry

end NUMINAMATH_CALUDE_power_difference_evaluation_l2023_202356


namespace NUMINAMATH_CALUDE_pythagorean_triple_identification_l2023_202377

def is_pythagorean_triple (a b c : ℚ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_identification :
  ¬ is_pythagorean_triple 5 8 12 ∧
  is_pythagorean_triple 30 40 50 ∧
  ¬ is_pythagorean_triple 9 13 15 ∧
  ¬ is_pythagorean_triple (1/6) (1/8) (1/10) :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_identification_l2023_202377


namespace NUMINAMATH_CALUDE_g_composition_points_sum_l2023_202345

/-- Given a function g with specific values, prove the existence of points on g(g(x)) with a certain sum property -/
theorem g_composition_points_sum (g : ℝ → ℝ) 
  (h1 : g 2 = 4) (h2 : g 3 = 2) (h3 : g 4 = 6) :
  ∃ (p q r s : ℝ), g (g p) = q ∧ g (g r) = s ∧ p * q + r * s = 24 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_points_sum_l2023_202345


namespace NUMINAMATH_CALUDE_equation_represents_point_l2023_202384

theorem equation_represents_point :
  ∀ x y : ℝ, (Real.sqrt (x - 2) + (y + 2)^2 = 0) ↔ (x = 2 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_point_l2023_202384


namespace NUMINAMATH_CALUDE_favorite_books_probability_l2023_202342

variable (n : ℕ) (k : ℕ)

def P (n k : ℕ) : ℚ := (k.factorial * (n - k + 1).factorial) / n.factorial

theorem favorite_books_probability (h : k ≤ n) :
  (∀ m, m ≤ n → P n k ≥ P n m) ↔ (k = 1 ∨ k = n) ∧
  (n % 2 = 0 → P n k ≤ P n (n / 2)) ∧
  (n % 2 ≠ 0 → P n k ≤ P n ((n + 1) / 2)) :=
sorry

end NUMINAMATH_CALUDE_favorite_books_probability_l2023_202342


namespace NUMINAMATH_CALUDE_exam_grading_problem_l2023_202385

theorem exam_grading_problem (X : ℝ) 
  (monday_graded : X * 0.6 = X - (X * 0.4))
  (tuesday_graded : X * 0.4 * 0.75 = X * 0.4 - (X * 0.1))
  (wednesday_remaining : X * 0.1 = 12) :
  X = 120 := by
sorry

end NUMINAMATH_CALUDE_exam_grading_problem_l2023_202385


namespace NUMINAMATH_CALUDE_polynomial_roots_degree_zero_l2023_202341

theorem polynomial_roots_degree_zero (F : Type*) [Field F] :
  ∀ (P : Polynomial F),
    (∃ (S : Finset F), (∀ x ∈ S, P.eval x = 0) ∧ S.card > P.degree) →
    P = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_degree_zero_l2023_202341


namespace NUMINAMATH_CALUDE_vector_BC_proof_l2023_202383

def A : ℝ × ℝ := (0, 0)  -- Assuming A as the origin for simplicity
def B : ℝ × ℝ := (2, 4)
def C : ℝ × ℝ := (1, 3)

def vector_AB : ℝ × ℝ := B
def vector_AC : ℝ × ℝ := C

theorem vector_BC_proof :
  (C.1 - B.1, C.2 - B.2) = (-1, -1) := by
  sorry

end NUMINAMATH_CALUDE_vector_BC_proof_l2023_202383


namespace NUMINAMATH_CALUDE_solution_pairs_l2023_202369

theorem solution_pairs (x y p : ℕ) (hp : Nat.Prime p) :
  x > 0 ∧ y > 0 ∧ x ≤ y ∧ (x + y) * (x * y - 1) = p * (x * y + 1) →
  (x = 3 ∧ y = 5 ∧ p = 7) ∨ (∃ q : ℕ, Nat.Prime q ∧ x = 1 ∧ y = q + 1 ∧ p = q) :=
sorry

end NUMINAMATH_CALUDE_solution_pairs_l2023_202369


namespace NUMINAMATH_CALUDE_henry_tournament_points_l2023_202318

/-- Point system for the tic-tac-toe tournament --/
structure PointSystem where
  win_points : ℕ
  loss_points : ℕ
  draw_points : ℕ

/-- Results of Henry's tournament --/
structure TournamentResults where
  wins : ℕ
  losses : ℕ
  draws : ℕ

/-- Calculate the total points for a given point system and tournament results --/
def calculateTotalPoints (ps : PointSystem) (tr : TournamentResults) : ℕ :=
  ps.win_points * tr.wins + ps.loss_points * tr.losses + ps.draw_points * tr.draws

/-- Theorem: Henry's total points in the tournament --/
theorem henry_tournament_points :
  let ps : PointSystem := { win_points := 5, loss_points := 2, draw_points := 3 }
  let tr : TournamentResults := { wins := 2, losses := 2, draws := 10 }
  calculateTotalPoints ps tr = 44 := by
  sorry


end NUMINAMATH_CALUDE_henry_tournament_points_l2023_202318


namespace NUMINAMATH_CALUDE_complex_multiplication_l2023_202348

theorem complex_multiplication (i : ℂ) : i * i = -1 → -i * (1 - 2*i) = -2 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2023_202348


namespace NUMINAMATH_CALUDE_concert_revenue_l2023_202330

theorem concert_revenue (ticket_price : ℝ) (first_group_size : ℕ) (second_group_size : ℕ) 
  (first_discount : ℝ) (second_discount : ℝ) (total_buyers : ℕ) :
  ticket_price = 20 →
  first_group_size = 10 →
  second_group_size = 20 →
  first_discount = 0.4 →
  second_discount = 0.15 →
  total_buyers = 56 →
  let first_group_revenue := first_group_size * (ticket_price * (1 - first_discount))
  let second_group_revenue := second_group_size * (ticket_price * (1 - second_discount))
  let remaining_buyers := total_buyers - first_group_size - second_group_size
  let remaining_revenue := remaining_buyers * ticket_price
  let total_revenue := first_group_revenue + second_group_revenue + remaining_revenue
  total_revenue = 980 := by sorry

end NUMINAMATH_CALUDE_concert_revenue_l2023_202330


namespace NUMINAMATH_CALUDE_tan_C_in_special_triangle_l2023_202336

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_of_angles : A + B + C = Real.pi

-- Define the theorem
theorem tan_C_in_special_triangle (t : Triangle) (h1 : Real.tan t.A = 1) (h2 : Real.tan t.B = 2) : 
  Real.tan t.C = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_C_in_special_triangle_l2023_202336


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l2023_202362

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (60 : ℤ) = Nat.gcd 25920 213840 ∧
  ∀ (k : ℤ), k ∣ (15*x + 9) * (15*x + 15) * (15*x + 21) → k ≤ 60 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l2023_202362


namespace NUMINAMATH_CALUDE_eight_fourth_equals_sixteen_n_l2023_202306

theorem eight_fourth_equals_sixteen_n (n : ℕ) : 8^4 = 16^n → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_eight_fourth_equals_sixteen_n_l2023_202306


namespace NUMINAMATH_CALUDE_sum_bound_l2023_202346

theorem sum_bound (w x y z : ℝ) 
  (sum_zero : w + x + y + z = 0) 
  (sum_squares_one : w^2 + x^2 + y^2 + z^2 = 1) : 
  -1 ≤ w*x + x*y + y*z + z*w ∧ w*x + x*y + y*z + z*w ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_bound_l2023_202346


namespace NUMINAMATH_CALUDE_enfeoffment_probability_l2023_202333

/-- The number of nobility levels in ancient China --/
def nobility_levels : ℕ := 5

/-- The probability that two people are not enfeoffed at the same level --/
def prob_different_levels : ℚ := 4/5

/-- Theorem stating the probability of two people being enfeoffed at different levels --/
theorem enfeoffment_probability :
  (1 : ℚ) - (nobility_levels : ℚ) / (nobility_levels^2 : ℚ) = prob_different_levels :=
by sorry

end NUMINAMATH_CALUDE_enfeoffment_probability_l2023_202333


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l2023_202316

def U : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}

theorem complement_of_M_in_U :
  (U \ M) = {-3, -4} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l2023_202316


namespace NUMINAMATH_CALUDE_no_country_with_100_roads_and_3_per_city_l2023_202375

theorem no_country_with_100_roads_and_3_per_city :
  ¬ ∃ (n : ℕ), 3 * n = 200 :=
by sorry

end NUMINAMATH_CALUDE_no_country_with_100_roads_and_3_per_city_l2023_202375


namespace NUMINAMATH_CALUDE_sally_seashells_l2023_202366

theorem sally_seashells (total tom jessica : ℕ) (h1 : total = 21) (h2 : tom = 7) (h3 : jessica = 5) :
  total - tom - jessica = 9 := by
  sorry

end NUMINAMATH_CALUDE_sally_seashells_l2023_202366


namespace NUMINAMATH_CALUDE_trip_duration_is_eight_hours_l2023_202323

/-- Represents a car trip with varying speeds -/
structure CarTrip where
  initial_hours : ℝ
  initial_speed : ℝ
  additional_speed : ℝ
  average_speed : ℝ

/-- Calculates the total duration of the car trip -/
def trip_duration (trip : CarTrip) : ℝ :=
  sorry

/-- Theorem stating that the trip duration is 8 hours given the specific conditions -/
theorem trip_duration_is_eight_hours (trip : CarTrip) 
  (h1 : trip.initial_hours = 4)
  (h2 : trip.initial_speed = 50)
  (h3 : trip.additional_speed = 80)
  (h4 : trip.average_speed = 65) :
  trip_duration trip = 8 := by
  sorry

end NUMINAMATH_CALUDE_trip_duration_is_eight_hours_l2023_202323


namespace NUMINAMATH_CALUDE_pages_read_difference_l2023_202376

/-- The number of weeks required for Janet to read 2100 more pages than Belinda,
    given that Janet reads 80 pages a day and Belinda reads 30 pages a day. -/
theorem pages_read_difference (janet_daily : ℕ) (belinda_daily : ℕ) (total_difference : ℕ) :
  janet_daily = 80 →
  belinda_daily = 30 →
  total_difference = 2100 →
  (total_difference / ((janet_daily - belinda_daily) * 7) : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_pages_read_difference_l2023_202376


namespace NUMINAMATH_CALUDE_penny_count_l2023_202392

/-- Proves that given 4 nickels, 3 dimes, and a total value of $0.59, the number of pennies is 9 -/
theorem penny_count (nickels : ℕ) (dimes : ℕ) (total_cents : ℕ) (pennies : ℕ) : 
  nickels = 4 → 
  dimes = 3 → 
  total_cents = 59 → 
  5 * nickels + 10 * dimes + pennies = total_cents → 
  pennies = 9 := by
sorry

end NUMINAMATH_CALUDE_penny_count_l2023_202392


namespace NUMINAMATH_CALUDE_janice_earnings_l2023_202363

/-- Represents Janice's work schedule and earnings --/
structure WorkSchedule where
  regularDays : ℕ
  regularPayPerDay : ℕ
  overtimeShifts : ℕ
  overtimePay : ℕ

/-- Calculates the total earnings for the week --/
def totalEarnings (schedule : WorkSchedule) : ℕ :=
  schedule.regularDays * schedule.regularPayPerDay + schedule.overtimeShifts * schedule.overtimePay

/-- Janice's work schedule for the week --/
def janiceSchedule : WorkSchedule :=
  { regularDays := 5
  , regularPayPerDay := 30
  , overtimeShifts := 3
  , overtimePay := 15 }

/-- Theorem stating that Janice's total earnings for the week equal $195 --/
theorem janice_earnings : totalEarnings janiceSchedule = 195 := by
  sorry

end NUMINAMATH_CALUDE_janice_earnings_l2023_202363


namespace NUMINAMATH_CALUDE_expand_complex_product_l2023_202350

theorem expand_complex_product (x : ℂ) : (x + Complex.I) * (x - 7) = x^2 - 7*x + Complex.I*x - 7*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_expand_complex_product_l2023_202350


namespace NUMINAMATH_CALUDE_queue_arrangements_l2023_202368

/-- Represents the number of people in each category -/
def num_fathers : ℕ := 2
def num_mothers : ℕ := 2
def num_children : ℕ := 2

/-- The total number of people -/
def total_people : ℕ := num_fathers + num_mothers + num_children

/-- Represents the constraint that fathers must be at the beginning and end -/
def fathers_fixed : ℕ := 2

/-- Represents the number of units to arrange between fathers (2 mothers and 1 children unit) -/
def units_between : ℕ := num_mothers + 1

/-- Represents the number of ways to arrange children within their unit -/
def children_arrangements : ℕ := Nat.factorial num_children

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := fathers_fixed * Nat.factorial units_between * children_arrangements

/-- Theorem stating that the number of possible arrangements is 24 -/
theorem queue_arrangements : total_arrangements = 24 := by
  sorry

end NUMINAMATH_CALUDE_queue_arrangements_l2023_202368


namespace NUMINAMATH_CALUDE_initial_men_is_ten_l2023_202307

/-- The initial number of men in the camp -/
def initial_men : ℕ := sorry

/-- The number of days the food lasts for the initial number of men -/
def initial_days : ℕ := 20

/-- The number of additional men that join the camp -/
def additional_men : ℕ := 30

/-- The number of days the food lasts after additional men join -/
def final_days : ℕ := 5

/-- The total amount of food available -/
def total_food : ℕ := initial_men * initial_days

/-- Theorem stating that the initial number of men is 10 -/
theorem initial_men_is_ten : initial_men = 10 := by
  have h1 : total_food = (initial_men + additional_men) * final_days := sorry
  sorry

end NUMINAMATH_CALUDE_initial_men_is_ten_l2023_202307


namespace NUMINAMATH_CALUDE_inequality_proof_l2023_202388

theorem inequality_proof (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  x^2 + x*y + y^2 ≤ 3*(x - Real.sqrt (x*y) + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2023_202388


namespace NUMINAMATH_CALUDE_abc_product_l2023_202389

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * (b + c) = 156)
  (h2 : b * (c + a) = 168)
  (h3 : c * (a + b) = 180) :
  a * b * c = 762 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l2023_202389


namespace NUMINAMATH_CALUDE_line_equation_through_two_points_l2023_202314

/-- Given two points A and B on the line 2x + 3y = 4, 
    prove that this is the equation of the line passing through these points. -/
theorem line_equation_through_two_points 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : 2 * x₁ + 3 * y₁ = 4) 
  (h₂ : 2 * x₂ + 3 * y₂ = 4) : 
  ∀ (x y : ℝ), (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) → 2 * x + 3 * y = 4 := by
sorry

end NUMINAMATH_CALUDE_line_equation_through_two_points_l2023_202314


namespace NUMINAMATH_CALUDE_cpu_sales_count_l2023_202396

/-- Represents the sales data for a hardware store for one week -/
structure HardwareSales where
  graphics_cards : Nat
  hard_drives : Nat
  cpus : Nat
  ram_pairs : Nat
  graphics_card_price : Nat
  hard_drive_price : Nat
  cpu_price : Nat
  ram_pair_price : Nat
  total_earnings : Nat

/-- Theorem stating that given the sales data, the number of CPUs sold is 8 -/
theorem cpu_sales_count (sales : HardwareSales) : 
  sales.graphics_cards = 10 ∧
  sales.hard_drives = 14 ∧
  sales.ram_pairs = 4 ∧
  sales.graphics_card_price = 600 ∧
  sales.hard_drive_price = 80 ∧
  sales.cpu_price = 200 ∧
  sales.ram_pair_price = 60 ∧
  sales.total_earnings = 8960 →
  sales.cpus = 8 := by
  sorry

#check cpu_sales_count

end NUMINAMATH_CALUDE_cpu_sales_count_l2023_202396


namespace NUMINAMATH_CALUDE_sum_of_series_equals_25_16_l2023_202381

theorem sum_of_series_equals_25_16 : 
  (∑' n, n / 5^n) + (∑' n, (1 / 5)^n) = 25 / 16 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_series_equals_25_16_l2023_202381


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2023_202365

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 - 2x + 1 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- Theorem: f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l2023_202365


namespace NUMINAMATH_CALUDE_expected_lotus_seed_zongzi_l2023_202347

theorem expected_lotus_seed_zongzi 
  (total_zongzi : ℕ) 
  (lotus_seed_zongzi : ℕ) 
  (selected_zongzi : ℕ) 
  (h1 : total_zongzi = 180) 
  (h2 : lotus_seed_zongzi = 54) 
  (h3 : selected_zongzi = 10) :
  (selected_zongzi : ℚ) * (lotus_seed_zongzi : ℚ) / (total_zongzi : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expected_lotus_seed_zongzi_l2023_202347


namespace NUMINAMATH_CALUDE_haley_initial_trees_l2023_202371

/-- The number of trees that died during the typhoon -/
def trees_died : ℕ := 2

/-- The number of trees left after the typhoon -/
def trees_left : ℕ := 10

/-- The initial number of trees before the typhoon -/
def initial_trees : ℕ := trees_left + trees_died

theorem haley_initial_trees : initial_trees = 12 := by
  sorry

end NUMINAMATH_CALUDE_haley_initial_trees_l2023_202371


namespace NUMINAMATH_CALUDE_quadratic_function_negative_on_interval_l2023_202329

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_negative_on_interval
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : a + b + c = 0) :
  ∀ x ∈ Set.Ioo 0 1, f a b c x < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_negative_on_interval_l2023_202329


namespace NUMINAMATH_CALUDE_class_selection_theorem_l2023_202326

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of boys in the class. -/
def num_boys : ℕ := 13

/-- The total number of girls in the class. -/
def num_girls : ℕ := 10

/-- The number of boys selected. -/
def boys_selected : ℕ := 2

/-- The number of girls selected. -/
def girls_selected : ℕ := 1

/-- The total number of possible combinations. -/
def total_combinations : ℕ := 780

theorem class_selection_theorem :
  choose num_boys boys_selected * choose num_girls girls_selected = total_combinations :=
sorry

end NUMINAMATH_CALUDE_class_selection_theorem_l2023_202326


namespace NUMINAMATH_CALUDE_arrangement_count_is_240_l2023_202312

/-- The number of ways to arrange 8 distinct objects in a row,
    where the two smallest objects must be at the ends and
    the largest object must be in the middle. -/
def arrangement_count : ℕ := 240

/-- Theorem stating that the number of arrangements is 240 -/
theorem arrangement_count_is_240 : arrangement_count = 240 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_240_l2023_202312


namespace NUMINAMATH_CALUDE_parabola_vertex_l2023_202309

/-- The parabola equation -/
def parabola_equation (x : ℝ) : ℝ := -(x - 5)^2 + 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (5, 3)

/-- Theorem: The vertex of the parabola y = -(x-5)^2 + 3 is (5, 3) -/
theorem parabola_vertex :
  ∀ (x : ℝ), parabola_equation x ≤ parabola_equation (vertex.1) ∧
  parabola_equation (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2023_202309


namespace NUMINAMATH_CALUDE_increasing_function_bounds_l2023_202349

theorem increasing_function_bounds (k : ℕ+) (f : ℕ+ → ℕ+) 
  (h_increasing : ∀ m n : ℕ+, m < n → f m < f n)
  (h_composition : ∀ n : ℕ+, f (f n) = k * n) :
  ∀ n : ℕ+, (2 * k : ℚ) / (k + 1) * n ≤ f n ∧ (f n : ℚ) ≤ (k + 1) / 2 * n :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_bounds_l2023_202349


namespace NUMINAMATH_CALUDE_reciprocals_product_l2023_202395

theorem reciprocals_product (a b : ℝ) (h : a * b = 1) : 4 * a * b = 4 := by
  sorry

end NUMINAMATH_CALUDE_reciprocals_product_l2023_202395


namespace NUMINAMATH_CALUDE_alpha_sum_sixth_power_l2023_202324

theorem alpha_sum_sixth_power (α₁ α₂ α₃ : ℂ) 
  (sum_zero : α₁ + α₂ + α₃ = 0)
  (sum_squares : α₁^2 + α₂^2 + α₃^2 = 2)
  (sum_cubes : α₁^3 + α₂^3 + α₃^3 = 4) :
  α₁^6 + α₂^6 + α₃^6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_alpha_sum_sixth_power_l2023_202324


namespace NUMINAMATH_CALUDE_sixteenth_number_with_digit_sum_13_l2023_202379

/-- A function that returns the sum of digits of a positive integer -/
def digit_sum (n : ℕ+) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 13 -/
def nth_number_with_digit_sum_13 (n : ℕ+) : ℕ+ := sorry

/-- Theorem stating that the 16th number with digit sum 13 is 247 -/
theorem sixteenth_number_with_digit_sum_13 : 
  nth_number_with_digit_sum_13 16 = 247 := by sorry

end NUMINAMATH_CALUDE_sixteenth_number_with_digit_sum_13_l2023_202379


namespace NUMINAMATH_CALUDE_octal_to_binary_127_l2023_202393

theorem octal_to_binary_127 : 
  (1 * 8^2 + 2 * 8^1 + 7 * 8^0 : ℕ) = (1 * 2^6 + 0 * 2^5 + 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_octal_to_binary_127_l2023_202393


namespace NUMINAMATH_CALUDE_stacy_brother_growth_l2023_202382

/-- Proves that Stacy's brother grew 1 inch last year -/
theorem stacy_brother_growth (stacy_initial_height stacy_final_height stacy_growth_difference : ℕ) 
  (h1 : stacy_initial_height = 50)
  (h2 : stacy_final_height = 57)
  (h3 : stacy_growth_difference = 6) :
  stacy_final_height - stacy_initial_height - stacy_growth_difference = 1 := by
  sorry

end NUMINAMATH_CALUDE_stacy_brother_growth_l2023_202382


namespace NUMINAMATH_CALUDE_no_geometric_sequence_sqrt235_l2023_202358

theorem no_geometric_sequence_sqrt235 :
  ¬∃ (m n : ℕ) (q : ℝ), m > n ∧ n > 1 ∧ q > 0 ∧
    Real.sqrt 3 = q ^ n * Real.sqrt 2 ∧
    Real.sqrt 5 = q ^ m * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_no_geometric_sequence_sqrt235_l2023_202358


namespace NUMINAMATH_CALUDE_cricket_run_rate_theorem_l2023_202360

/-- Represents a cricket game scenario -/
structure CricketGame where
  totalOvers : ℕ
  firstPartOvers : ℕ
  firstPartRunRate : ℚ
  targetRuns : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstPartOvers
  let runsScored := game.firstPartRunRate * game.firstPartOvers
  let runsNeeded := game.targetRuns - runsScored
  runsNeeded / remainingOvers

/-- Theorem stating the required run rate for the given scenario -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.totalOvers = 45)
  (h2 : game.firstPartOvers = 10)
  (h3 : game.firstPartRunRate = 3.5)
  (h4 : game.targetRuns = 350) :
  requiredRunRate game = 9 := by
  sorry

end NUMINAMATH_CALUDE_cricket_run_rate_theorem_l2023_202360


namespace NUMINAMATH_CALUDE_three_tangents_range_l2023_202313

/-- The function f(x) = x^3 - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

/-- Predicate to check if a point (x, y) is on the curve y = f(x) --/
def on_curve (x y : ℝ) : Prop := y = f x

/-- Predicate to check if a line through (1, m) is tangent to the curve at some point --/
def is_tangent (m t : ℝ) : Prop := 
  ∃ x : ℝ, on_curve x (f x) ∧ (m - f 1) = f' x * (1 - x)

/-- The main theorem --/
theorem three_tangents_range (m : ℝ) : 
  (∃ t1 t2 t3 : ℝ, t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ 
    is_tangent m t1 ∧ is_tangent m t2 ∧ is_tangent m t3) → 
  m > -3 ∧ m < -2 :=
sorry

end NUMINAMATH_CALUDE_three_tangents_range_l2023_202313


namespace NUMINAMATH_CALUDE_min_value_expression_l2023_202367

theorem min_value_expression :
  ∀ x y : ℝ,
  (Real.sqrt (2 * (1 + Real.cos (2 * x))) - Real.sqrt (9 - Real.sqrt 7) * Real.sin x + 1) *
  (3 + 2 * Real.sqrt (13 - Real.sqrt 7) * Real.cos y - Real.cos (2 * y)) ≥ -19 ∧
  ∃ x₀ y₀ : ℝ,
  (Real.sqrt (2 * (1 + Real.cos (2 * x₀))) - Real.sqrt (9 - Real.sqrt 7) * Real.sin x₀ + 1) *
  (3 + 2 * Real.sqrt (13 - Real.sqrt 7) * Real.cos y₀ - Real.cos (2 * y₀)) = -19 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2023_202367


namespace NUMINAMATH_CALUDE_intersection_M_N_l2023_202302

def M : Set ℝ := { x | -1 ≤ x ∧ x < 3 }

def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2023_202302


namespace NUMINAMATH_CALUDE_smallest_with_15_divisors_l2023_202301

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Checks if a given natural number has exactly 15 positive divisors -/
def has_15_divisors (n : ℕ+) : Prop := num_divisors n = 15

theorem smallest_with_15_divisors :
  (∀ m : ℕ+, m < 24 → ¬(has_15_divisors m)) ∧ has_15_divisors 24 := by sorry

end NUMINAMATH_CALUDE_smallest_with_15_divisors_l2023_202301


namespace NUMINAMATH_CALUDE_meal_combinations_count_l2023_202331

/-- The number of items in Menu A -/
def menu_a_items : ℕ := 15

/-- The number of items in Menu B -/
def menu_b_items : ℕ := 12

/-- The total number of possible meal combinations -/
def total_combinations : ℕ := menu_a_items * menu_b_items

/-- Theorem stating that the total number of meal combinations is 180 -/
theorem meal_combinations_count : total_combinations = 180 := by
  sorry

end NUMINAMATH_CALUDE_meal_combinations_count_l2023_202331


namespace NUMINAMATH_CALUDE_school_election_votes_l2023_202361

theorem school_election_votes (eliot_votes shaun_votes other_votes : ℕ) : 
  eliot_votes = 2 * shaun_votes →
  shaun_votes = 5 * other_votes →
  eliot_votes = 160 →
  other_votes = 16 := by
sorry

end NUMINAMATH_CALUDE_school_election_votes_l2023_202361


namespace NUMINAMATH_CALUDE_toy_position_l2023_202339

/-- Given a row of toys, this function calculates the position from the left
    based on the total number of toys and the position from the right. -/
def position_from_left (total : ℕ) (position_from_right : ℕ) : ℕ :=
  total - position_from_right + 1

/-- Theorem stating that in a row of 19 toys, 
    if a toy is 8th from the right, it is 12th from the left. -/
theorem toy_position (total : ℕ) (position_from_right : ℕ) 
  (h1 : total = 19) (h2 : position_from_right = 8) : 
  position_from_left total position_from_right = 12 := by
  sorry

end NUMINAMATH_CALUDE_toy_position_l2023_202339


namespace NUMINAMATH_CALUDE_minimum_packages_for_equal_shipment_l2023_202337

theorem minimum_packages_for_equal_shipment (sarah_capacity : Nat) (ryan_capacity : Nat) (emily_capacity : Nat)
  (h1 : sarah_capacity = 18)
  (h2 : ryan_capacity = 11)
  (h3 : emily_capacity = 15) :
  Nat.lcm (Nat.lcm sarah_capacity ryan_capacity) emily_capacity = 990 :=
by sorry

end NUMINAMATH_CALUDE_minimum_packages_for_equal_shipment_l2023_202337


namespace NUMINAMATH_CALUDE_quadratic_increasing_iff_m_gt_one_l2023_202334

/-- A quadratic function of the form y = x^2 + (m-3)x + m + 1 -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := x^2 + (m-3)*x + m + 1

/-- The derivative of the quadratic function with respect to x -/
def quadratic_derivative (m : ℝ) (x : ℝ) : ℝ := 2*x + (m-3)

theorem quadratic_increasing_iff_m_gt_one (m : ℝ) :
  (∀ x > 1, ∀ h > 0, quadratic_function m (x + h) > quadratic_function m x) ↔ m > 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_increasing_iff_m_gt_one_l2023_202334


namespace NUMINAMATH_CALUDE_tangent_line_slope_l2023_202317

/-- Given a curve y = x^3 and its tangent line y = kx + 2, prove that k = 3 -/
theorem tangent_line_slope (x : ℝ) :
  let f : ℝ → ℝ := fun x => x^3
  let f' : ℝ → ℝ := fun x => 3 * x^2
  let tangent_line (k : ℝ) (x : ℝ) := k * x + 2
  ∃ m : ℝ, f m = tangent_line k m ∧ f' m = k → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l2023_202317


namespace NUMINAMATH_CALUDE_markers_given_l2023_202335

theorem markers_given (initial : ℕ) (total : ℕ) (given : ℕ) : 
  initial = 217 → total = 326 → given = total - initial → given = 109 := by
sorry

end NUMINAMATH_CALUDE_markers_given_l2023_202335


namespace NUMINAMATH_CALUDE_right_triangle_sets_l2023_202351

/-- A function to check if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that among the given sets, only (2, 5, 6) cannot form a right triangle --/
theorem right_triangle_sets :
  is_right_triangle 3 4 5 ∧
  is_right_triangle 2 2 (2 * Real.sqrt 2) ∧
  ¬is_right_triangle 2 5 6 ∧
  is_right_triangle 5 12 13 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l2023_202351


namespace NUMINAMATH_CALUDE_episode_length_proof_l2023_202394

/-- Represents the length of a single episode in minutes -/
def episode_length : ℕ := 33

/-- Represents the total number of episodes watched in a week -/
def total_episodes : ℕ := 8

/-- Represents the minutes watched on Monday -/
def monday_minutes : ℕ := 138

/-- Represents the minutes watched on Thursday -/
def thursday_minutes : ℕ := 21

/-- Represents the number of episodes watched on Friday -/
def friday_episodes : ℕ := 2

/-- Represents the minutes watched over the weekend -/
def weekend_minutes : ℕ := 105

/-- Proves that the given episode length satisfies the conditions of the problem -/
theorem episode_length_proof : 
  monday_minutes + thursday_minutes + weekend_minutes = total_episodes * episode_length := by
  sorry

end NUMINAMATH_CALUDE_episode_length_proof_l2023_202394


namespace NUMINAMATH_CALUDE_cos_135_degrees_l2023_202364

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l2023_202364


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l2023_202390

/-- Represents a trapezoid ABCD -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ

/-- The theorem statement -/
theorem trapezoid_segment_length 
  (ABCD : Trapezoid) 
  (h_ratio : (ABCD.AB / ABCD.CD) = (5 / 2)) 
  (h_sum : ABCD.AB + ABCD.CD = 280) : 
  ABCD.AB = 200 := by
  sorry

#check trapezoid_segment_length

end NUMINAMATH_CALUDE_trapezoid_segment_length_l2023_202390


namespace NUMINAMATH_CALUDE_triangle_area_rational_l2023_202343

theorem triangle_area_rational (x₁ x₂ y₂ : ℤ) :
  ∃ (a b : ℤ), b ≠ 0 ∧ (1/2 : ℚ) * |x₁ + x₂ - x₁*y₂ - x₂*y₂| = a / b :=
sorry

end NUMINAMATH_CALUDE_triangle_area_rational_l2023_202343


namespace NUMINAMATH_CALUDE_circle_through_pole_equation_l2023_202378

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  center : PolarPoint
  radius : ℝ

/-- The equation of a circle in polar coordinates -/
def circleEquation (c : PolarCircle) (p : PolarPoint) : Prop :=
  p.r = 2 * c.radius * Real.cos (p.θ - c.center.θ)

theorem circle_through_pole_equation 
  (c : PolarCircle) 
  (h1 : c.center = PolarPoint.mk (Real.sqrt 2) 0) 
  (h2 : c.radius = Real.sqrt 2) :
  ∀ (p : PolarPoint), circleEquation c p ↔ p.r = 2 * Real.sqrt 2 * Real.cos p.θ :=
by sorry

end NUMINAMATH_CALUDE_circle_through_pole_equation_l2023_202378


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2023_202325

def quadratic_inequality (x : ℝ) : Prop := 3 * x^2 + 9 * x + 6 ≤ 0

theorem quadratic_inequality_solution :
  {x : ℝ | quadratic_inequality x} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2023_202325


namespace NUMINAMATH_CALUDE_biography_increase_l2023_202327

theorem biography_increase (B : ℝ) (b n : ℝ) 
  (h1 : b = 0.20 * B)  -- Initial biographies are 20% of total
  (h2 : b + n = 0.32 * (B + n))  -- After purchase, biographies are 32% of new total
  : (n / b) * 100 = 1500 / 17 := by
  sorry

end NUMINAMATH_CALUDE_biography_increase_l2023_202327


namespace NUMINAMATH_CALUDE_eva_last_when_start_vasya_l2023_202387

/-- Represents the children in the circle -/
inductive Child : Type
| Anya : Child
| Borya : Child
| Vasya : Child
| Gena : Child
| Dasha : Child
| Eva : Child
| Zhenya : Child

/-- The number of children in the circle -/
def num_children : Nat := 7

/-- The step size for elimination -/
def step_size : Nat := 3

/-- Function to determine the last remaining child given a starting position -/
def last_remaining (start : Child) : Child :=
  sorry

/-- Theorem stating that starting from Vasya results in Eva being the last remaining -/
theorem eva_last_when_start_vasya :
  last_remaining Child.Vasya = Child.Eva :=
sorry

end NUMINAMATH_CALUDE_eva_last_when_start_vasya_l2023_202387


namespace NUMINAMATH_CALUDE_sin_neg_seven_pi_sixth_l2023_202340

theorem sin_neg_seven_pi_sixth : Real.sin (-7 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_neg_seven_pi_sixth_l2023_202340


namespace NUMINAMATH_CALUDE_unique_positive_root_in_interval_l2023_202304

-- Define the function f(x) = x^2 - x - 1
def f (x : ℝ) : ℝ := x^2 - x - 1

-- State the theorem
theorem unique_positive_root_in_interval :
  (∃! r : ℝ, r > 0 ∧ f r = 0) →  -- There exists a unique positive root
  ∃ r : ℝ, r ∈ Set.Ioo 1 2 ∧ f r = 0 :=  -- The root is in the open interval (1, 2)
by
  sorry

end NUMINAMATH_CALUDE_unique_positive_root_in_interval_l2023_202304


namespace NUMINAMATH_CALUDE_complex_reciprocal_sum_l2023_202398

theorem complex_reciprocal_sum (z z₁ z₂ : ℂ) : 
  z₁ = 5 + 10*I → z₂ = 3 - 4*I → (1 : ℂ)/z = 1/z₁ + 1/z₂ → z = 5 - (5/2)*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_reciprocal_sum_l2023_202398


namespace NUMINAMATH_CALUDE_power_mod_45_l2023_202357

theorem power_mod_45 : 14^100 % 45 = 31 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_45_l2023_202357


namespace NUMINAMATH_CALUDE_muslim_boys_percentage_l2023_202300

/-- The percentage of Muslim boys in a school -/
def percentage_muslim_boys (total_boys : ℕ) (hindu_percentage : ℚ) (sikh_percentage : ℚ) (other_boys : ℕ) : ℚ :=
  let non_muslim_boys := (hindu_percentage + sikh_percentage) * total_boys + other_boys
  let muslim_boys := total_boys - non_muslim_boys
  (muslim_boys / total_boys) * 100

/-- Theorem stating that the percentage of Muslim boys is approximately 44% -/
theorem muslim_boys_percentage :
  let total_boys : ℕ := 850
  let hindu_percentage : ℚ := 28 / 100
  let sikh_percentage : ℚ := 10 / 100
  let other_boys : ℕ := 153
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
    |percentage_muslim_boys total_boys hindu_percentage sikh_percentage other_boys - 44| < ε :=
sorry

end NUMINAMATH_CALUDE_muslim_boys_percentage_l2023_202300


namespace NUMINAMATH_CALUDE_parabola_intersection_sum_l2023_202397

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (x y : ℝ) : Prop :=
  ∃ (t : ℝ), x = 1 + t ∧ y = t

-- Define the intersection points
structure IntersectionPoint where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y
  on_line : line_through_focus x y

-- Theorem statement
theorem parabola_intersection_sum (A B : IntersectionPoint) :
  (A.x - (-1))^2 + (A.y)^2 + (B.x - (-1))^2 + (B.y)^2 = 64 →
  A.x + B.x = 6 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_sum_l2023_202397


namespace NUMINAMATH_CALUDE_power_of_product_cube_l2023_202374

theorem power_of_product_cube (m : ℝ) : (2 * m^2)^3 = 8 * m^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_cube_l2023_202374


namespace NUMINAMATH_CALUDE_no_tip_customers_l2023_202321

theorem no_tip_customers (total_customers : ℕ) (tip_amount : ℕ) (total_tips : ℕ) : 
  total_customers = 9 →
  tip_amount = 8 →
  total_tips = 32 →
  total_customers - (total_tips / tip_amount) = 5 :=
by sorry

end NUMINAMATH_CALUDE_no_tip_customers_l2023_202321


namespace NUMINAMATH_CALUDE_factor_theorem_application_l2023_202391

theorem factor_theorem_application (d : ℚ) :
  (∀ x : ℚ, (x - 3) ∣ (x^3 + 3*x^2 + d*x + 8)) → d = -62/3 := by
  sorry

end NUMINAMATH_CALUDE_factor_theorem_application_l2023_202391


namespace NUMINAMATH_CALUDE_subtracted_amount_l2023_202386

theorem subtracted_amount (N : ℝ) (A : ℝ) (h1 : N = 100) (h2 : 0.8 * N - A = 60) : A = 20 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_amount_l2023_202386


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2023_202354

theorem other_root_of_quadratic (b : ℝ) : 
  (1 : ℝ)^2 + b*(1 : ℝ) - 2 = 0 → 
  ∃ (x : ℝ), x ≠ 1 ∧ x^2 + b*x - 2 = 0 ∧ x = -2 := by
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2023_202354


namespace NUMINAMATH_CALUDE_original_group_size_l2023_202338

theorem original_group_size (n : ℕ) (W : ℝ) : 
  W = n * 35 →
  W + 40 = (n + 1) * 36 →
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_original_group_size_l2023_202338


namespace NUMINAMATH_CALUDE_x_limit_properties_l2023_202332

noncomputable def x : ℕ → ℝ
  | 0 => Real.sqrt 6
  | n + 1 => x n + 3 * Real.sqrt (x n) + (n + 1 : ℝ) / Real.sqrt (x n)

theorem x_limit_properties :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |n / x n| < ε) ∧
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |n^2 / x n - 4/9| < ε) := by
  sorry

end NUMINAMATH_CALUDE_x_limit_properties_l2023_202332


namespace NUMINAMATH_CALUDE_square_plus_double_sqrt2_minus_1_l2023_202373

theorem square_plus_double_sqrt2_minus_1 :
  let x : ℝ := Real.sqrt 2 - 1
  x^2 + 2*x = 1 := by sorry

end NUMINAMATH_CALUDE_square_plus_double_sqrt2_minus_1_l2023_202373


namespace NUMINAMATH_CALUDE_reinforcement_size_l2023_202372

/-- Calculates the size of a reinforcement given initial garrison size, initial provisions duration,
    time passed before reinforcement, and remaining provisions duration after reinforcement. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) 
                            (time_passed : ℕ) (remaining_duration : ℕ) : ℕ :=
  let provisions_left := initial_garrison * (initial_duration - time_passed)
  let total_men := provisions_left / remaining_duration
  total_men - initial_garrison

/-- Proves that given the specified conditions, the reinforcement size is 300 men. -/
theorem reinforcement_size :
  calculate_reinforcement 150 31 16 5 = 300 := by
  sorry

end NUMINAMATH_CALUDE_reinforcement_size_l2023_202372


namespace NUMINAMATH_CALUDE_nickel_difference_l2023_202355

/-- The number of cents in a nickel -/
def cents_per_nickel : ℕ := 5

/-- The total number of cents Ray has initially -/
def ray_initial_cents : ℕ := 175

/-- The number of cents Ray gives to Peter -/
def cents_to_peter : ℕ := 30

/-- Calculates the number of nickels given a number of cents -/
def cents_to_nickels (cents : ℕ) : ℕ := cents / cents_per_nickel

/-- Theorem stating the difference in nickels between Randi and Peter -/
theorem nickel_difference : 
  cents_to_nickels (2 * cents_to_peter) - cents_to_nickels cents_to_peter = 6 := by
  sorry

end NUMINAMATH_CALUDE_nickel_difference_l2023_202355


namespace NUMINAMATH_CALUDE_main_result_l2023_202310

/-- A function satisfying the given property for all real numbers -/
def satisfies_property (g : ℝ → ℝ) : Prop :=
  ∀ a c : ℝ, c^3 * g a = a^3 * g c

/-- The main theorem -/
theorem main_result (g : ℝ → ℝ) (h1 : satisfies_property g) (h2 : g 3 ≠ 0) :
  (g 6 - g 2) / g 3 = 208 / 27 := by
  sorry

end NUMINAMATH_CALUDE_main_result_l2023_202310


namespace NUMINAMATH_CALUDE_system_solution_l2023_202353

theorem system_solution :
  ∃! (x y : ℚ), 2 * x - 3 * y = 5 ∧ 4 * x - 6 * y = 10 ∧ x + 2 * y = 7 ∧ x = 31/7 ∧ y = 9/7 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2023_202353


namespace NUMINAMATH_CALUDE_water_average_l2023_202311

def water_problem (day1 day2 day3 : ℕ) : Prop :=
  day1 = 215 ∧
  day2 = day1 + 76 ∧
  day3 = day2 - 53 ∧
  (day1 + day2 + day3) / 3 = 248

theorem water_average : ∃ day1 day2 day3 : ℕ, water_problem day1 day2 day3 := by
  sorry

end NUMINAMATH_CALUDE_water_average_l2023_202311


namespace NUMINAMATH_CALUDE_max_triangle_area_in_rectangle_l2023_202352

/-- The maximum area of a right triangle with a 30° angle inside a 12x5 rectangle -/
theorem max_triangle_area_in_rectangle :
  let rectangle_length : ℝ := 12
  let rectangle_width : ℝ := 5
  let angle : ℝ := 30 * π / 180  -- 30° in radians
  ∃ (triangle_area : ℝ),
    triangle_area = 25 * Real.sqrt 3 / 4 ∧
    ∀ (a : ℝ), a ≤ rectangle_width →
      a * (2 * a) / 2 ≤ triangle_area :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_area_in_rectangle_l2023_202352


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2023_202344

def f (x : ℝ) := x^3 + 3*x - 1

theorem sum_of_a_and_b (a b : ℝ) (h1 : f (a - 3) = -3) (h2 : f (b - 3) = 1) : a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2023_202344


namespace NUMINAMATH_CALUDE_altitude_sum_of_triangle_l2023_202380

/-- The sum of altitudes of a triangle formed by the line 15x + 3y = 45 and the coordinate axes --/
theorem altitude_sum_of_triangle (x y : ℝ) : 
  (15 * x + 3 * y = 45) →  -- Line equation
  ∃ (a b c : ℝ), -- Altitudes
    (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) ∧ -- Altitudes are non-negative
    (a + b + c = (18 * Real.sqrt 26 + 15) / Real.sqrt 26) ∧ -- Sum of altitudes
    (∃ (x₁ y₁ : ℝ), 15 * x₁ + 3 * y₁ = 45 ∧ x₁ ≥ 0 ∧ y₁ ≥ 0) -- Triangle exists in the first quadrant
    :=
by sorry

end NUMINAMATH_CALUDE_altitude_sum_of_triangle_l2023_202380


namespace NUMINAMATH_CALUDE_job_candidate_probability_l2023_202322

theorem job_candidate_probability (excel_probability : Real) (day_shift_probability : Real) :
  excel_probability = 0.2 →
  day_shift_probability = 0.7 →
  (1 - day_shift_probability) * excel_probability = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_job_candidate_probability_l2023_202322


namespace NUMINAMATH_CALUDE_simplify_fraction_l2023_202399

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) : 
  (2 / (x^2 - 2*x + 1) - 1 / (x^2 - x)) / ((x + 1) / (2*x^2 - 2*x)) = 2 / (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2023_202399


namespace NUMINAMATH_CALUDE_neighborhood_vehicles_l2023_202308

theorem neighborhood_vehicles (total : Nat) (both : Nat) (car : Nat) (bike_only : Nat)
  (h1 : total = 90)
  (h2 : both = 16)
  (h3 : car = 44)
  (h4 : bike_only = 35) :
  total - (car + bike_only) = 11 := by
  sorry

end NUMINAMATH_CALUDE_neighborhood_vehicles_l2023_202308


namespace NUMINAMATH_CALUDE_complement_of_union_is_empty_l2023_202315

universe u

def U : Set Char := {'a', 'b', 'c', 'd', 'e'}
def N : Set Char := {'b', 'd', 'e'}
def M : Set Char := {'a', 'c', 'd'}

theorem complement_of_union_is_empty :
  (M ∪ N)ᶜ = ∅ :=
sorry

end NUMINAMATH_CALUDE_complement_of_union_is_empty_l2023_202315


namespace NUMINAMATH_CALUDE_total_cost_calculation_total_cost_is_832_l2023_202319

/-- Calculate the total cost of sandwiches and sodas with discount and tax -/
theorem total_cost_calculation (sandwich_price soda_price : ℚ) 
  (sandwich_quantity soda_quantity : ℕ) 
  (sandwich_discount tax_rate : ℚ) : ℚ :=
  let sandwich_cost := sandwich_price * sandwich_quantity
  let soda_cost := soda_price * soda_quantity
  let discounted_sandwich_cost := sandwich_cost * (1 - sandwich_discount)
  let subtotal := discounted_sandwich_cost + soda_cost
  let total_with_tax := subtotal * (1 + tax_rate)
  total_with_tax

/-- Prove that the total cost is $8.32 given the specific conditions -/
theorem total_cost_is_832 :
  total_cost_calculation 2.44 0.87 2 4 0.15 0.09 = 8.32 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_total_cost_is_832_l2023_202319


namespace NUMINAMATH_CALUDE_largest_angle_right_isosceles_triangle_l2023_202303

theorem largest_angle_right_isosceles_triangle (D E F : Real) :
  -- Triangle DEF is a right isosceles triangle
  D + E + F = 180 →
  D = E →
  (D = 90 ∨ E = 90 ∨ F = 90) →
  -- Angle D measures 45°
  D = 45 →
  -- The largest interior angle measures 90°
  max D (max E F) = 90 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_right_isosceles_triangle_l2023_202303


namespace NUMINAMATH_CALUDE_tangent_line_slope_l2023_202370

theorem tangent_line_slope (m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^4 + m*x
  let f' : ℝ → ℝ := λ x ↦ 4*x^3 + m
  let tangent_slope : ℝ := f' (-1)
  (2 * (-1) + f (-1) + 3 = 0) ∧ (tangent_slope = -2) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l2023_202370


namespace NUMINAMATH_CALUDE_correlation_function_is_even_l2023_202305

/-- Represents a stationary random process -/
class StationaryRandomProcess (X : ℝ → ℝ) : Prop where
  is_stationary : ∀ t₁ t₂ τ : ℝ, X (t₁ + τ) = X (t₂ + τ)

/-- Correlation function for a stationary random process -/
def correlationFunction (X : ℝ → ℝ) [StationaryRandomProcess X] (τ : ℝ) : ℝ :=
  sorry -- Definition of correlation function

/-- Theorem: The correlation function of a stationary random process is an even function -/
theorem correlation_function_is_even
  (X : ℝ → ℝ) [StationaryRandomProcess X] :
  ∀ τ : ℝ, correlationFunction X τ = correlationFunction X (-τ) := by
  sorry


end NUMINAMATH_CALUDE_correlation_function_is_even_l2023_202305


namespace NUMINAMATH_CALUDE_carrots_grown_total_l2023_202320

/-- The number of carrots grown by Sally -/
def sally_carrots : ℕ := 6

/-- The number of carrots grown by Fred -/
def fred_carrots : ℕ := 4

/-- The total number of carrots grown by Sally and Fred -/
def total_carrots : ℕ := sally_carrots + fred_carrots

theorem carrots_grown_total :
  total_carrots = 10 := by sorry

end NUMINAMATH_CALUDE_carrots_grown_total_l2023_202320
