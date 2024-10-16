import Mathlib

namespace NUMINAMATH_CALUDE_circle_center_and_point_check_l4077_407767

def circle_equation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 2*y - 11 = 0

def center : ℝ × ℝ := (3, -1)

def point : ℝ × ℝ := (5, -1)

theorem circle_center_and_point_check :
  (∀ x y : ℝ, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = 21) ∧
  ¬ circle_equation point.1 point.2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_point_check_l4077_407767


namespace NUMINAMATH_CALUDE_identity_is_unique_solution_l4077_407761

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2

/-- The theorem stating that the identity function is the only solution -/
theorem identity_is_unique_solution :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f → (∀ x : ℝ, f x = x) :=
by sorry

end NUMINAMATH_CALUDE_identity_is_unique_solution_l4077_407761


namespace NUMINAMATH_CALUDE_best_meeting_days_l4077_407732

-- Define the days of the week
inductive Day
| Mon
| Tue
| Wed
| Thu
| Fri

-- Define the team members
inductive Member
| Anna
| Bill
| Carl
| Dana

-- Define the availability function
def availability (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.Anna, Day.Mon => false
  | Member.Anna, Day.Wed => false
  | Member.Bill, Day.Tue => false
  | Member.Bill, Day.Thu => false
  | Member.Bill, Day.Fri => false
  | Member.Carl, Day.Mon => false
  | Member.Carl, Day.Tue => false
  | Member.Carl, Day.Thu => false
  | Member.Carl, Day.Fri => false
  | Member.Dana, Day.Wed => false
  | Member.Dana, Day.Thu => false
  | _, _ => true

-- Count available members for a given day
def availableCount (d : Day) : Nat :=
  (List.filter (fun m => availability m d) [Member.Anna, Member.Bill, Member.Carl, Member.Dana]).length

-- Define the maximum availability
def maxAvailability : Nat :=
  List.foldl max 0 (List.map availableCount [Day.Mon, Day.Tue, Day.Wed, Day.Thu, Day.Fri])

-- Theorem statement
theorem best_meeting_days :
  (availableCount Day.Mon = maxAvailability) ∧
  (availableCount Day.Tue = maxAvailability) ∧
  (availableCount Day.Wed = maxAvailability) ∧
  (availableCount Day.Thu < maxAvailability) ∧
  (availableCount Day.Fri = maxAvailability) := by
  sorry

end NUMINAMATH_CALUDE_best_meeting_days_l4077_407732


namespace NUMINAMATH_CALUDE_mrs_hilt_apple_pies_l4077_407708

/-- The number of apple pies Mrs. Hilt baked -/
def apple_pies : ℕ := 150 - 16

/-- Theorem: Mrs. Hilt baked 134 apple pies -/
theorem mrs_hilt_apple_pies : apple_pies = 134 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_apple_pies_l4077_407708


namespace NUMINAMATH_CALUDE_points_on_circle_l4077_407742

/-- A type representing a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a motion (isometry) of the plane -/
structure Motion where
  transform : Point → Point

/-- A system of points with the given property -/
structure PointSystem where
  points : List Point
  motion_property : ∀ (p q : Point), p ∈ points → q ∈ points → 
    ∃ (m : Motion), m.transform p = q ∧ (∀ x ∈ points, m.transform x ∈ points)

/-- Definition of a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- The main theorem -/
theorem points_on_circle (sys : PointSystem) : 
  ∃ (c : Circle), ∀ p ∈ sys.points, 
    (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2 := by
  sorry

end NUMINAMATH_CALUDE_points_on_circle_l4077_407742


namespace NUMINAMATH_CALUDE_sin_cos_sum_l4077_407756

theorem sin_cos_sum (θ : ℝ) (h : Real.sin θ ^ 3 + Real.cos θ ^ 3 = 11/16) : 
  Real.sin θ + Real.cos θ = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_sum_l4077_407756


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l4077_407751

theorem rectangle_area_ratio :
  ∀ (x y : ℝ),
  x = (3/5) * 40 →
  y = (2/3) * 20 →
  x * y = 320 ∧
  40 * 20 = 800 ∧
  (x * y) / (40 * 20) = 2/5 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l4077_407751


namespace NUMINAMATH_CALUDE_triangle_count_equality_l4077_407733

/-- The number of non-congruent triangles with positive area and integer side lengths summing to n -/
def T (n : ℕ) : ℕ := sorry

/-- The statement to prove -/
theorem triangle_count_equality : T 2022 = T 2019 := by sorry

end NUMINAMATH_CALUDE_triangle_count_equality_l4077_407733


namespace NUMINAMATH_CALUDE_train_length_proof_l4077_407796

/-- Given a train crossing a bridge and passing a lamp post, prove its length --/
theorem train_length_proof (bridge_length : ℝ) (bridge_time : ℝ) (post_time : ℝ)
  (h1 : bridge_length = 800)
  (h2 : bridge_time = 45)
  (h3 : post_time = 15) :
  let train_length := (bridge_length * post_time) / (bridge_time - post_time)
  train_length = 400 := by
sorry

end NUMINAMATH_CALUDE_train_length_proof_l4077_407796


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l4077_407773

theorem cubic_equation_solution : ∃ x : ℝ, (x - 2)^3 + (x - 6)^3 = 54 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l4077_407773


namespace NUMINAMATH_CALUDE_abc_positive_l4077_407730

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem abc_positive 
  (a b c : ℝ) 
  (ha : a ≠ 0)
  (h0 : quadratic a b c 0 = -2)
  (h1 : quadratic a b c 1 = -2)
  (hneg_half : quadratic a b c (-1/2) > 0) :
  a * b * c > 0 := by
  sorry

end NUMINAMATH_CALUDE_abc_positive_l4077_407730


namespace NUMINAMATH_CALUDE_min_sum_cotangents_l4077_407717

theorem min_sum_cotangents (A B C : ℝ) (hAcute : 0 < A ∧ A < π/2) 
  (hBcute : 0 < B ∧ B < π/2) (hCcute : 0 < C ∧ C < π/2) 
  (hSum : A + B + C = π) (hSin : 2 * Real.sin A ^ 2 + Real.sin B ^ 2 = 2 * Real.sin C ^ 2) : 
  (∀ A' B' C', A' + B' + C' = π → 2 * Real.sin A' ^ 2 + Real.sin B' ^ 2 = 2 * Real.sin C' ^ 2 →
    1 / Real.tan A + 1 / Real.tan B + 1 / Real.tan C ≤ 
    1 / Real.tan A' + 1 / Real.tan B' + 1 / Real.tan C') ∧
  1 / Real.tan A + 1 / Real.tan B + 1 / Real.tan C = Real.sqrt 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_cotangents_l4077_407717


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_odd_products_l4077_407728

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def consecutive_odd_integers (a b c d : ℕ) : Prop :=
  is_odd a ∧ is_odd b ∧ is_odd c ∧ is_odd d ∧
  b = a + 2 ∧ c = b + 2 ∧ d = c + 2

theorem largest_common_divisor_of_consecutive_odd_products :
  ∀ a b c d : ℕ,
  consecutive_odd_integers a b c d →
  (∃ k : ℕ, a * b * c * d = 3 * k) ∧
  (∀ m : ℕ, m > 3 → ∃ x y z w : ℕ, 
    consecutive_odd_integers x y z w ∧ 
    ¬(∃ k : ℕ, x * y * z * w = m * k)) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_odd_products_l4077_407728


namespace NUMINAMATH_CALUDE_work_completion_time_l4077_407740

theorem work_completion_time (y_completion_time x_remaining_time : ℕ) 
  (y_worked_days : ℕ) (h1 : y_completion_time = 16) (h2 : y_worked_days = 10) 
  (h3 : x_remaining_time = 9) : 
  (y_completion_time * x_remaining_time) / 
  (y_completion_time - y_worked_days) = 24 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l4077_407740


namespace NUMINAMATH_CALUDE_necklace_beads_l4077_407788

theorem necklace_beads (total : ℕ) (amethyst : ℕ) (amber : ℕ) (turquoise : ℕ) :
  total = 40 →
  amethyst = 7 →
  amber = 2 * amethyst →
  total = amethyst + amber + turquoise →
  turquoise = 19 := by
sorry

end NUMINAMATH_CALUDE_necklace_beads_l4077_407788


namespace NUMINAMATH_CALUDE_multiple_of_nine_implies_multiple_of_three_l4077_407707

theorem multiple_of_nine_implies_multiple_of_three 
  (h1 : ∀ n : ℕ, 9 ∣ n → 3 ∣ n) 
  (k : ℕ) 
  (h2 : Odd k) 
  (h3 : 9 ∣ k) : 
  3 ∣ k := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_nine_implies_multiple_of_three_l4077_407707


namespace NUMINAMATH_CALUDE_daisy_sales_difference_l4077_407727

/-- Represents the sales of daisies at Daisy's Flower Shop over four days -/
structure DaisySales where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ
  total : ℕ

/-- Theorem stating the difference in sales between day 2 and day 1 -/
theorem daisy_sales_difference (s : DaisySales) : 
  s.day1 = 45 ∧ 
  s.day2 > s.day1 ∧ 
  s.day3 = 2 * s.day2 - 10 ∧ 
  s.day4 = 120 ∧ 
  s.total = 350 ∧ 
  s.total = s.day1 + s.day2 + s.day3 + s.day4 →
  s.day2 - s.day1 = 20 := by
  sorry

#check daisy_sales_difference

end NUMINAMATH_CALUDE_daisy_sales_difference_l4077_407727


namespace NUMINAMATH_CALUDE_min_nSn_arithmetic_sequence_l4077_407741

def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

theorem min_nSn_arithmetic_sequence (a₁ d : ℤ) :
  (arithmetic_sequence a₁ d 7 = 5) →
  (sum_arithmetic_sequence a₁ d 5 = -55) →
  (∀ n : ℕ, n > 0 → n * (sum_arithmetic_sequence a₁ d n) ≥ -343) ∧
  (∃ n : ℕ, n > 0 ∧ n * (sum_arithmetic_sequence a₁ d n) = -343) :=
sorry

end NUMINAMATH_CALUDE_min_nSn_arithmetic_sequence_l4077_407741


namespace NUMINAMATH_CALUDE_pencils_theorem_l4077_407702

def pencils_problem (monday tuesday wednesday thursday friday : ℕ) : Prop :=
  let total_tuesday := monday + tuesday
  let total_wednesday := total_tuesday + 3 * tuesday - 20
  let total_thursday := total_wednesday + wednesday / 2
  let total_friday := total_thursday + 2 * monday
  let final_total := total_friday - 50
  (monday = 35) ∧
  (tuesday = 42) ∧
  (wednesday = 3 * tuesday) ∧
  (thursday = wednesday / 2) ∧
  (friday = 2 * monday) ∧
  (final_total = 266)

theorem pencils_theorem :
  ∃ (monday tuesday wednesday thursday friday : ℕ),
    pencils_problem monday tuesday wednesday thursday friday :=
by
  sorry

end NUMINAMATH_CALUDE_pencils_theorem_l4077_407702


namespace NUMINAMATH_CALUDE_certain_number_multiplied_by_p_l4077_407759

theorem certain_number_multiplied_by_p (x : ℕ+) (p : ℕ) (n : ℕ) : 
  Nat.Prime p → 
  (x : ℕ) / (n * p) = 2 → 
  x ≥ 48 → 
  (∀ y : ℕ+, y < x → (y : ℕ) / (n * p) ≠ 2) →
  n = 12 := by sorry

end NUMINAMATH_CALUDE_certain_number_multiplied_by_p_l4077_407759


namespace NUMINAMATH_CALUDE_snooker_tournament_ticket_difference_l4077_407711

theorem snooker_tournament_ticket_difference :
  ∀ (vip_tickets general_tickets : ℕ),
    vip_tickets + general_tickets = 320 →
    45 * vip_tickets + 20 * general_tickets = 7500 →
    general_tickets - vip_tickets = 232 :=
by
  sorry

end NUMINAMATH_CALUDE_snooker_tournament_ticket_difference_l4077_407711


namespace NUMINAMATH_CALUDE_min_Q_value_l4077_407718

def is_special_number (m : ℕ) : Prop :=
  m ≥ 10 ∧ m < 100 ∧ (m / 10) ≠ (m % 10) ∧ (m / 10) ≠ 0 ∧ (m % 10) ≠ 0

def F (m : ℕ) : ℤ :=
  let m₁ := (m % 10) * 10 + (m / 10)
  (m * 100 + m₁ - (m₁ * 100 + m)) / 99

def Q (s t : ℕ) : ℚ :=
  (t - s : ℚ) / s

theorem min_Q_value (s t : ℕ) (a b x y : ℕ) :
  is_special_number s →
  is_special_number t →
  s = 10 * a + b →
  t = 10 * x + y →
  1 ≤ b →
  b < a →
  a ≤ 7 →
  1 ≤ x →
  x ≤ 8 →
  1 ≤ y →
  y ≤ 8 →
  F s % 5 = 1 →
  F t - F s + 18 * x = 36 →
  ∀ (s' t' : ℕ), is_special_number s' → is_special_number t' → Q s' t' ≥ Q s t →
  Q s t = -42 / 73 :=
sorry

end NUMINAMATH_CALUDE_min_Q_value_l4077_407718


namespace NUMINAMATH_CALUDE_a_greater_than_b_l4077_407721

theorem a_greater_than_b (n : ℕ) (a b : ℝ) 
  (h_n : n > 1)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_eq : a^n = a + 1)
  (h_b_eq : b^(2*n) = b + 3*a) :
  a > b :=
by sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l4077_407721


namespace NUMINAMATH_CALUDE_polynomial_simplification_l4077_407709

theorem polynomial_simplification (x : ℝ) :
  (x^3 + 4*x^2 - 7*x + 11) + (-4*x^4 - x^3 + x^2 + 7*x + 3) = -4*x^4 + 5*x^2 + 14 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l4077_407709


namespace NUMINAMATH_CALUDE_reservoir_water_supply_l4077_407762

/-- Reservoir water supply problem -/
theorem reservoir_water_supply
  (reservoir_volume : ℝ)
  (initial_population : ℝ)
  (initial_sustainability : ℝ)
  (new_population : ℝ)
  (new_sustainability : ℝ)
  (h_reservoir : reservoir_volume = 120)
  (h_initial_pop : initial_population = 160000)
  (h_initial_sus : initial_sustainability = 20)
  (h_new_pop : new_population = 200000)
  (h_new_sus : new_sustainability = 15) :
  ∃ (annual_precipitation : ℝ) (annual_consumption_pp : ℝ),
    annual_precipitation = 200 ∧
    annual_consumption_pp = 50 ∧
    reservoir_volume + initial_sustainability * annual_precipitation = initial_population * initial_sustainability * annual_consumption_pp / 1000000 ∧
    reservoir_volume + new_sustainability * annual_precipitation = new_population * new_sustainability * annual_consumption_pp / 1000000 :=
by sorry


end NUMINAMATH_CALUDE_reservoir_water_supply_l4077_407762


namespace NUMINAMATH_CALUDE_two_six_minus_one_prime_divisors_l4077_407710

theorem two_six_minus_one_prime_divisors :
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧
  (∀ (r : Nat), Nat.Prime r → r ∣ (2^6 - 1) → r = p ∨ r = q) ∧
  p + q = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_six_minus_one_prime_divisors_l4077_407710


namespace NUMINAMATH_CALUDE_parabola_shift_l4077_407785

def original_function (x : ℝ) : ℝ := (x + 1)^2 + 3

def shift_right (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x => f (x - shift)

def shift_down (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x => f x - shift

def final_function (x : ℝ) : ℝ := (x - 1)^2 + 2

theorem parabola_shift :
  (shift_down (shift_right original_function 2) 1) = final_function := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l4077_407785


namespace NUMINAMATH_CALUDE_acid_solution_replacement_l4077_407745

theorem acid_solution_replacement (x : ℝ) :
  x ≥ 0 ∧ x ≤ 1 →
  0.5 * (1 - x) + 0.3 * x = 0.4 →
  x = 1/2 := by sorry

end NUMINAMATH_CALUDE_acid_solution_replacement_l4077_407745


namespace NUMINAMATH_CALUDE_amy_spelling_problems_l4077_407700

/-- The number of spelling problems Amy had to solve -/
def spelling_problems (total_problems math_problems : ℕ) : ℕ :=
  total_problems - math_problems

/-- Proof that Amy had 6 spelling problems -/
theorem amy_spelling_problems :
  spelling_problems 24 18 = 6 := by
  sorry

end NUMINAMATH_CALUDE_amy_spelling_problems_l4077_407700


namespace NUMINAMATH_CALUDE_smallest_covering_set_smallest_n_is_five_l4077_407703

theorem smallest_covering_set (n : ℕ) : Prop :=
  ∃ (k a : Fin n → ℕ),
    (∀ i j : Fin n, i < j → 1 < k i ∧ k i < k j) ∧
    (∀ N : ℤ, ∃ i : Fin n, (k i : ℤ) ∣ (N - (a i : ℤ)))

theorem smallest_n_is_five :
  (∃ n : ℕ, smallest_covering_set n) ∧
  (∀ m : ℕ, smallest_covering_set m → m ≥ 5) ∧
  smallest_covering_set 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_covering_set_smallest_n_is_five_l4077_407703


namespace NUMINAMATH_CALUDE_laura_payment_l4077_407713

/-- The amount Laura gave to the cashier --/
def amount_given_to_cashier (pants_price : ℕ) (shirts_price : ℕ) (pants_quantity : ℕ) (shirts_quantity : ℕ) (change : ℕ) : ℕ :=
  pants_price * pants_quantity + shirts_price * shirts_quantity + change

/-- Theorem stating that Laura gave $250 to the cashier --/
theorem laura_payment : amount_given_to_cashier 54 33 2 4 10 = 250 := by
  sorry

end NUMINAMATH_CALUDE_laura_payment_l4077_407713


namespace NUMINAMATH_CALUDE_equation_solution_l4077_407763

theorem equation_solution : ∃ x : ℝ, 
  (1 / (x + 9) + 1 / (x + 7) = 1 / (x + 10) + 1 / (x + 6)) ∧ 
  x = -8 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l4077_407763


namespace NUMINAMATH_CALUDE_system_no_solution_l4077_407743

theorem system_no_solution (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y = 1 ∧ 2 * x + a * y = 1) ↔ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_system_no_solution_l4077_407743


namespace NUMINAMATH_CALUDE_worksheets_turned_in_l4077_407771

/-- 
Given:
- initial_worksheets: The initial number of worksheets to grade
- graded_worksheets: The number of worksheets graded
- final_worksheets: The final number of worksheets to grade

Prove that the number of worksheets turned in after grading is 36.
-/
theorem worksheets_turned_in 
  (initial_worksheets : ℕ) 
  (graded_worksheets : ℕ) 
  (final_worksheets : ℕ) 
  (h1 : initial_worksheets = 34)
  (h2 : graded_worksheets = 7)
  (h3 : final_worksheets = 63) :
  final_worksheets - (initial_worksheets - graded_worksheets) = 36 := by
  sorry

end NUMINAMATH_CALUDE_worksheets_turned_in_l4077_407771


namespace NUMINAMATH_CALUDE_six_digit_multiple_of_nine_l4077_407704

theorem six_digit_multiple_of_nine :
  ∀ (d : ℕ), d < 10 →
  (456780 + d) % 9 = 0 ↔ d = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_multiple_of_nine_l4077_407704


namespace NUMINAMATH_CALUDE_train_passing_time_l4077_407750

/-- Proves that a train of given length and speed will pass a stationary point in the calculated time -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 175 →
  train_speed_kmh = 63 →
  passing_time = train_length / (train_speed_kmh * (5/18)) →
  passing_time = 10 := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l4077_407750


namespace NUMINAMATH_CALUDE_equation_solutions_l4077_407769

def equation (x : ℝ) : Prop :=
  1 / (x^2 + 14*x - 36) + 1 / (x^2 + 5*x - 14) + 1 / (x^2 - 16*x - 36) = 0

theorem equation_solutions :
  {x : ℝ | equation x} = {9, -4, 12, 3} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l4077_407769


namespace NUMINAMATH_CALUDE_marks_money_theorem_l4077_407723

/-- The amount of money Mark's father gave him. -/
def fathers_money : ℕ := 85

/-- The number of books Mark bought. -/
def num_books : ℕ := 10

/-- The cost of each book in dollars. -/
def book_cost : ℕ := 5

/-- The amount of money Mark has left after buying the books. -/
def money_left : ℕ := 35

/-- Theorem stating that the amount of money Mark's father gave him is correct. -/
theorem marks_money_theorem :
  fathers_money = num_books * book_cost + money_left :=
by sorry

end NUMINAMATH_CALUDE_marks_money_theorem_l4077_407723


namespace NUMINAMATH_CALUDE_petyas_journey_contradiction_l4077_407716

theorem petyas_journey_contradiction (S T : ℝ) (hS : S > 0) (hT : T > 0) : 
  ¬(∃ (S T : ℝ), 
    S / 2 = 4 * (T / 2) ∧ 
    S / 2 = 5 * (T / 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_petyas_journey_contradiction_l4077_407716


namespace NUMINAMATH_CALUDE_flea_misses_point_l4077_407786

/-- The number of points on the circle -/
def n : ℕ := 300

/-- The set of all points on the circle -/
def Circle := Fin n

/-- The jumping pattern of the flea -/
def jump (k : ℕ) : ℕ := k * (k + 1) / 2

/-- The set of points visited by the flea -/
def VisitedPoints : Set Circle :=
  {p | ∃ k : ℕ, p = ⟨jump k % n, sorry⟩}

/-- Theorem stating that there exists a point the flea never visits -/
theorem flea_misses_point : ∃ p : Circle, p ∉ VisitedPoints := by
  sorry

end NUMINAMATH_CALUDE_flea_misses_point_l4077_407786


namespace NUMINAMATH_CALUDE_system_solution_l4077_407712

theorem system_solution : ∃ (x y : ℝ), x - y = 3 ∧ x + y = 1 ∧ x = 2 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4077_407712


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l4077_407725

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ (l1.a ≠ 0 ∨ l1.b ≠ 0) ∧ (l2.a ≠ 0 ∨ l2.b ≠ 0)

theorem parallel_lines_m_value :
  let l1 : Line := { a := 3, b := 4, c := -3 }
  let l2 : Line := { a := 6, b := m, c := 14 }
  parallel l1 l2 → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l4077_407725


namespace NUMINAMATH_CALUDE_rungs_on_twenty_ladders_eq_1200_l4077_407782

/-- Calculates the number of rungs on 20 ladders given the following conditions:
  * There are 10 ladders with 50 rungs each
  * There are 20 additional ladders with an unknown number of rungs
  * Each rung costs $2
  * The total cost for all ladders is $3,400
-/
def rungs_on_twenty_ladders : ℕ :=
  let ladders_with_fifty_rungs : ℕ := 10
  let rungs_per_ladder : ℕ := 50
  let cost_per_rung : ℕ := 2
  let total_cost : ℕ := 3400
  let remaining_ladders : ℕ := 20
  
  let cost_of_fifty_rung_ladders : ℕ := ladders_with_fifty_rungs * rungs_per_ladder * cost_per_rung
  let remaining_cost : ℕ := total_cost - cost_of_fifty_rung_ladders
  remaining_cost / cost_per_rung

theorem rungs_on_twenty_ladders_eq_1200 : rungs_on_twenty_ladders = 1200 := by
  sorry

end NUMINAMATH_CALUDE_rungs_on_twenty_ladders_eq_1200_l4077_407782


namespace NUMINAMATH_CALUDE_rita_bought_three_pants_l4077_407755

/-- Represents the shopping trip of Rita -/
structure ShoppingTrip where
  dresses : Nat
  jackets : Nat
  dress_cost : Nat
  jacket_cost : Nat
  pants_cost : Nat
  transport_cost : Nat
  initial_money : Nat
  remaining_money : Nat

/-- Calculates the number of pairs of pants bought given a shopping trip -/
def pants_bought (trip : ShoppingTrip) : Nat :=
  let total_spent := trip.initial_money - trip.remaining_money
  let known_expenses := trip.dresses * trip.dress_cost + trip.jackets * trip.jacket_cost + trip.transport_cost
  let pants_expense := total_spent - known_expenses
  pants_expense / trip.pants_cost

/-- Theorem stating that Rita bought 3 pairs of pants -/
theorem rita_bought_three_pants :
  let trip : ShoppingTrip := {
    dresses := 5,
    jackets := 4,
    dress_cost := 20,
    jacket_cost := 30,
    pants_cost := 12,
    transport_cost := 5,
    initial_money := 400,
    remaining_money := 139
  }
  pants_bought trip = 3 := by sorry

end NUMINAMATH_CALUDE_rita_bought_three_pants_l4077_407755


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l4077_407720

/-- A right circular cone with a sphere inscribed inside it. -/
structure ConeWithSphere where
  /-- The diameter of the cone's base in inches. -/
  base_diameter : ℝ
  /-- The vertex angle of the cross-section triangle in degrees. -/
  vertex_angle : ℝ
  /-- The sphere is tangent to the sides of the cone and sits on the table. -/
  sphere_tangent : Bool

/-- The volume of the inscribed sphere in cubic inches. -/
def sphere_volume (cone : ConeWithSphere) : ℝ := sorry

/-- Theorem stating the volume of the inscribed sphere for the given conditions. -/
theorem inscribed_sphere_volume (cone : ConeWithSphere) 
  (h1 : cone.base_diameter = 24)
  (h2 : cone.vertex_angle = 90)
  (h3 : cone.sphere_tangent = true) :
  sphere_volume cone = 576 * Real.sqrt 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l4077_407720


namespace NUMINAMATH_CALUDE_symmetric_expressions_l4077_407744

-- Define what it means for an expression to be completely symmetric
def is_completely_symmetric (f : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ (a b c : ℝ), f a b c = f b a c ∧ f a b c = f a c b ∧ f a b c = f c b a

-- Define the three expressions
def expr1 (a b c : ℝ) : ℝ := (a - b)^2
def expr2 (a b c : ℝ) : ℝ := a * b + b * c + c * a
def expr3 (a b c : ℝ) : ℝ := a^2 * b + b^2 * c + c^2 * a

-- State the theorem
theorem symmetric_expressions :
  is_completely_symmetric expr1 ∧
  is_completely_symmetric expr2 ∧
  ¬ is_completely_symmetric expr3 := by sorry

end NUMINAMATH_CALUDE_symmetric_expressions_l4077_407744


namespace NUMINAMATH_CALUDE_max_sections_five_l4077_407787

/-- The maximum number of sections created by drawing n line segments through a rectangle -/
def max_sections (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => max_sections m + m + 1

/-- Theorem: The maximum number of sections created by drawing 5 line segments through a rectangle is 16 -/
theorem max_sections_five : max_sections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_sections_five_l4077_407787


namespace NUMINAMATH_CALUDE_car_wash_soap_cost_l4077_407770

/-- The cost of each bottle of car wash soap -/
def bottle_cost (washes_per_bottle : ℕ) (total_washes : ℕ) (total_cost : ℕ) : ℚ :=
  total_cost / (total_washes / washes_per_bottle)

/-- Theorem stating that the cost of each bottle is $4 -/
theorem car_wash_soap_cost :
  bottle_cost 4 20 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_car_wash_soap_cost_l4077_407770


namespace NUMINAMATH_CALUDE_total_value_is_305_l4077_407748

/-- The value of a gold coin in dollars -/
def gold_coin_value : ℕ := 50

/-- The value of a silver coin in dollars -/
def silver_coin_value : ℕ := 25

/-- The number of gold coins -/
def num_gold_coins : ℕ := 3

/-- The number of silver coins -/
def num_silver_coins : ℕ := 5

/-- The amount of cash in dollars -/
def cash : ℕ := 30

/-- The total value of all coins and cash -/
def total_value : ℕ := num_gold_coins * gold_coin_value + num_silver_coins * silver_coin_value + cash

theorem total_value_is_305 : total_value = 305 := by
  sorry

end NUMINAMATH_CALUDE_total_value_is_305_l4077_407748


namespace NUMINAMATH_CALUDE_smallest_union_size_l4077_407747

theorem smallest_union_size (X Y : Finset ℕ) : 
  Finset.card X = 30 → 
  Finset.card Y = 25 → 
  Finset.card (X ∩ Y) ≥ 10 → 
  45 ≤ Finset.card (X ∪ Y) ∧ ∃ X' Y' : Finset ℕ, 
    Finset.card X' = 30 ∧ 
    Finset.card Y' = 25 ∧ 
    Finset.card (X' ∩ Y') ≥ 10 ∧ 
    Finset.card (X' ∪ Y') = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_union_size_l4077_407747


namespace NUMINAMATH_CALUDE_rabbit_count_l4077_407706

/-- The number of ducks in Eunji's house -/
def num_ducks : ℕ := 52

/-- The number of chickens in Eunji's house -/
def num_chickens : ℕ := 78

/-- The number of rabbits in Eunji's house -/
def num_rabbits : ℕ := 38

/-- Theorem stating the relationship between the number of animals and proving the number of rabbits -/
theorem rabbit_count : num_chickens = num_ducks + num_rabbits - 12 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_count_l4077_407706


namespace NUMINAMATH_CALUDE_sliding_triangle_forms_ellipse_l4077_407789

/-- Triangle ABC with A and B on perpendicular lines -/
structure SlidingTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  perpendicular : A.2 = 0 ∧ B.1 = 0
  non_right_angle_at_C : ∀ (t : ℝ), (C.1 - A.1) * (C.2 - B.2) ≠ (C.2 - A.2) * (C.1 - B.1)

/-- The locus of point C forms an ellipse -/
def is_ellipse (locus : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b h k : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), (x, y) ∈ locus ↔ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- The theorem statement -/
theorem sliding_triangle_forms_ellipse (triangle : SlidingTriangle) :
  ∃ (locus : Set (ℝ × ℝ)), is_ellipse locus ∧ ∀ (t : ℝ), triangle.C ∈ locus :=
sorry

end NUMINAMATH_CALUDE_sliding_triangle_forms_ellipse_l4077_407789


namespace NUMINAMATH_CALUDE_trig_identity_l4077_407754

theorem trig_identity (α : Real) (h : Real.cos (α - π/3) = -1/2) : 
  Real.sin (π/6 + α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l4077_407754


namespace NUMINAMATH_CALUDE_brazil_championship_prob_l4077_407729

-- Define the probabilities and point system
def win_prob : ℚ := 1/2
def draw_prob : ℚ := 1/3
def loss_prob : ℚ := 1/6
def win_points : ℕ := 3
def draw_points : ℕ := 1
def loss_points : ℕ := 0

-- Define the number of group stage matches and minimum points to advance
def group_matches : ℕ := 3
def min_points : ℕ := 4

-- Define the probability of winning a penalty shootout
def penalty_win_prob : ℚ := 3/5

-- Define the number of knockout stage matches
def knockout_matches : ℕ := 4

-- Define the function to calculate the probability of winning the championship
-- with exactly one match decided by penalty shootout
def championship_prob : ℚ := sorry

-- State the theorem
theorem brazil_championship_prob : championship_prob = 1/12 := by sorry

end NUMINAMATH_CALUDE_brazil_championship_prob_l4077_407729


namespace NUMINAMATH_CALUDE_art_project_marker_distribution_l4077_407772

/-- Proves that each student in the last group receives 5 markers given the conditions of the art project. -/
theorem art_project_marker_distribution :
  let total_students : ℕ := 68
  let total_groups : ℕ := 5
  let total_marker_boxes : ℕ := 48
  let markers_per_box : ℕ := 6
  let group1_students : ℕ := 12
  let group1_markers_per_student : ℕ := 2
  let group2_students : ℕ := 20
  let group2_markers_per_student : ℕ := 3
  let group3_students : ℕ := 15
  let group3_markers_per_student : ℕ := 5
  let group4_students : ℕ := 8
  let group4_markers_per_student : ℕ := 8
  let total_markers : ℕ := total_marker_boxes * markers_per_box
  let used_markers : ℕ := group1_students * group1_markers_per_student +
                          group2_students * group2_markers_per_student +
                          group3_students * group3_markers_per_student +
                          group4_students * group4_markers_per_student
  let remaining_markers : ℕ := total_markers - used_markers
  let last_group_students : ℕ := total_students - (group1_students + group2_students + group3_students + group4_students)
  remaining_markers / last_group_students = 5 :=
by sorry

end NUMINAMATH_CALUDE_art_project_marker_distribution_l4077_407772


namespace NUMINAMATH_CALUDE_batsman_average_theorem_l4077_407739

/-- Represents a batsman's score history -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  averageIncrease : Nat
  lastInningScore : Nat

/-- Calculates the new average after the latest inning -/
def newAverage (b : Batsman) : Rat :=
  (b.totalRuns + b.lastInningScore) / (b.innings + 1)

/-- Theorem: A batsman who scores 100 runs in his 17th inning and increases his average by 5 runs will have a new average of 20 runs -/
theorem batsman_average_theorem (b : Batsman) 
  (h1 : b.innings = 16)
  (h2 : b.lastInningScore = 100)
  (h3 : b.averageIncrease = 5)
  (h4 : newAverage b = (b.totalRuns + b.lastInningScore) / (b.innings + 1)) :
  newAverage b = 20 := by
  sorry

#check batsman_average_theorem

end NUMINAMATH_CALUDE_batsman_average_theorem_l4077_407739


namespace NUMINAMATH_CALUDE_probability_of_selecting_A_and_B_l4077_407714

/-- The number of students in total -/
def total_students : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 3

/-- The probability of selecting both A and B when choosing 3 students out of 5 -/
theorem probability_of_selecting_A_and_B :
  (Nat.choose (total_students - 2) (selected_students - 2)) / 
  (Nat.choose total_students selected_students : ℚ) = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_probability_of_selecting_A_and_B_l4077_407714


namespace NUMINAMATH_CALUDE_sufficient_condition_ranges_not_sufficient_condition_ranges_l4077_407726

/-- Condition p: (x+1)(2-x) ≥ 0 -/
def p (x : ℝ) : Prop := (x + 1) * (2 - x) ≥ 0

/-- Condition q: x^2+mx-2m^2-3m-1 < 0, where m > -2/3 -/
def q (x m : ℝ) : Prop := x^2 + m*x - 2*m^2 - 3*m - 1 < 0 ∧ m > -2/3

theorem sufficient_condition_ranges (m : ℝ) :
  (∀ x, p x → q x m) ∧ (∃ x, q x m ∧ ¬p x) → m > 1 :=
sorry

theorem not_sufficient_condition_ranges (m : ℝ) :
  (∀ x, ¬p x → ¬q x m) ∧ (∃ x, ¬q x m ∧ p x) → -2/3 < m ∧ m ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_ranges_not_sufficient_condition_ranges_l4077_407726


namespace NUMINAMATH_CALUDE_integer_solution_l4077_407722

theorem integer_solution (n : ℤ) : 
  n + 15 > 16 ∧ 4 * n < 20 ∧ |n - 2| ≤ 2 → n = 2 ∨ n = 3 ∨ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_l4077_407722


namespace NUMINAMATH_CALUDE_binomial_coefficient_identity_l4077_407731

theorem binomial_coefficient_identity (n k : ℕ) (h1 : k ≤ n) (h2 : n ≥ 1) :
  Nat.choose n k = Nat.choose (n - 1) (k - 1) + Nat.choose (n - 1) k := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_identity_l4077_407731


namespace NUMINAMATH_CALUDE_student_arrangement_probabilities_l4077_407781

/-- Represents the probability of various arrangements of 4 students in a row. -/
structure StudentArrangementProbabilities where
  /-- The total number of possible arrangements for 4 students. -/
  total_arrangements : ℕ
  /-- The number of arrangements where a specific student is at one end. -/
  student_at_end : ℕ
  /-- The number of arrangements where two specific students are at both ends. -/
  two_students_at_ends : ℕ

/-- Theorem stating the probabilities of various student arrangements. -/
theorem student_arrangement_probabilities 
  (probs : StudentArrangementProbabilities)
  (h1 : probs.total_arrangements = 24)
  (h2 : probs.student_at_end = 12)
  (h3 : probs.two_students_at_ends = 4) :
  let p1 := probs.student_at_end / probs.total_arrangements
  let p2 := probs.two_students_at_ends / probs.total_arrangements
  let p3 := 1 - (probs.total_arrangements - probs.student_at_end - probs.student_at_end + probs.two_students_at_ends) / probs.total_arrangements
  let p4 := (probs.total_arrangements - probs.student_at_end - probs.student_at_end + probs.two_students_at_ends) / probs.total_arrangements
  (p1 = 1/2) ∧ 
  (p2 = 1/6) ∧ 
  (p3 = 5/6) ∧ 
  (p4 = 1/6) := by
  sorry

end NUMINAMATH_CALUDE_student_arrangement_probabilities_l4077_407781


namespace NUMINAMATH_CALUDE_t_integer_characterization_t_irreducible_characterization_l4077_407799

def t (n : ℤ) : ℚ := (5 * n + 9) / (n - 3)

def is_integer_t (n : ℤ) : Prop := ∃ (k : ℤ), t n = k

def is_irreducible_t (n : ℤ) : Prop :=
  ∃ (a b : ℤ), t n = a / b ∧ Int.gcd a b = 1

theorem t_integer_characterization (n : ℤ) (h : n > 3) :
  is_integer_t n ↔ n ∈ ({4, 5, 6, 7, 9, 11, 15, 27} : Set ℤ) :=
sorry

theorem t_irreducible_characterization (n : ℤ) (h : n > 3) :
  is_irreducible_t n ↔ (∃ (k : ℤ), k > 0 ∧ (n = 6 * k + 1 ∨ n = 6 * k + 5)) :=
sorry

end NUMINAMATH_CALUDE_t_integer_characterization_t_irreducible_characterization_l4077_407799


namespace NUMINAMATH_CALUDE_rationalize_sqrt_five_eighteenths_l4077_407783

theorem rationalize_sqrt_five_eighteenths : 
  Real.sqrt (5 / 18) = Real.sqrt 10 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_five_eighteenths_l4077_407783


namespace NUMINAMATH_CALUDE_trajectory_is_line_segment_l4077_407791

/-- The trajectory of a point M satisfying |MF₁| + |MF₂| = 8 is a line segment -/
theorem trajectory_is_line_segment (M : ℝ × ℝ) : 
  let F₁ : ℝ × ℝ := (-4, 0)
  let F₂ : ℝ × ℝ := (4, 0)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (dist M F₁ + dist M F₂ = 8) →
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (t * F₂.1 + (1 - t) * F₁.1, t * F₂.2 + (1 - t) * F₁.2) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_is_line_segment_l4077_407791


namespace NUMINAMATH_CALUDE_simplify_expression_l4077_407777

theorem simplify_expression (a b : ℝ) : a + b - (a - b) = 2 * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4077_407777


namespace NUMINAMATH_CALUDE_toy_store_solution_l4077_407760

/-- Represents the selling and cost prices of toys, and the optimal purchase strategy -/
structure ToyStore where
  sell_price_A : ℝ
  sell_price_B : ℝ
  cost_price_A : ℝ
  cost_price_B : ℝ
  optimal_purchase_A : ℕ
  optimal_purchase_B : ℕ

/-- The toy store problem with given conditions -/
def toy_store_problem : Prop :=
  ∃ (store : ToyStore),
    -- Selling price condition
    store.sell_price_B - store.sell_price_A = 30 ∧
    -- Total sales condition
    2 * store.sell_price_A + 3 * store.sell_price_B = 740 ∧
    -- Cost prices
    store.cost_price_A = 90 ∧
    store.cost_price_B = 110 ∧
    -- Total purchase constraint
    store.optimal_purchase_A + store.optimal_purchase_B = 80 ∧
    -- Total cost constraint
    store.cost_price_A * store.optimal_purchase_A + store.cost_price_B * store.optimal_purchase_B ≤ 8400 ∧
    -- Correct selling prices
    store.sell_price_A = 130 ∧
    store.sell_price_B = 160 ∧
    -- Optimal purchase strategy
    store.optimal_purchase_A = 20 ∧
    store.optimal_purchase_B = 60 ∧
    -- Profit maximization (implied by the optimal strategy)
    ∀ (m : ℕ), m + (80 - m) = 80 →
      (store.sell_price_A - store.cost_price_A) * store.optimal_purchase_A +
      (store.sell_price_B - store.cost_price_B) * store.optimal_purchase_B ≥
      (store.sell_price_A - store.cost_price_A) * m +
      (store.sell_price_B - store.cost_price_B) * (80 - m)

theorem toy_store_solution : toy_store_problem := by
  sorry


end NUMINAMATH_CALUDE_toy_store_solution_l4077_407760


namespace NUMINAMATH_CALUDE_hall_of_mirrors_glass_area_l4077_407752

/-- Calculates the total area of glass needed for James' hall of mirrors --/
theorem hall_of_mirrors_glass_area :
  let wall1_length : ℝ := 30
  let wall1_width : ℝ := 12
  let wall2_length : ℝ := 30
  let wall2_width : ℝ := 12
  let wall3_length : ℝ := 20
  let wall3_width : ℝ := 12
  let wall1_area := wall1_length * wall1_width
  let wall2_area := wall2_length * wall2_width
  let wall3_area := wall3_length * wall3_width
  let total_area := wall1_area + wall2_area + wall3_area
  total_area = 960 := by sorry

end NUMINAMATH_CALUDE_hall_of_mirrors_glass_area_l4077_407752


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l4077_407793

theorem line_slope_intercept_product (m b : ℝ) : m = 3/4 ∧ b = -2 → m * b < -1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l4077_407793


namespace NUMINAMATH_CALUDE_smallest_pair_product_bound_l4077_407705

theorem smallest_pair_product_bound (n : ℕ) (h : n = 2016) : 
  (∃ (f : ℕ → ℕ), 
    (∀ i ∈ Finset.range n, f i ∈ Finset.range n ∧ f (f i) = i) ∧ 
    (∀ i ∈ Finset.range n, i < f i → (i + 1) * (f i + 1) ≤ 1017072)) ∧ 
  (∀ m : ℕ, m < 1017072 → 
    ¬∃ (g : ℕ → ℕ), 
      (∀ i ∈ Finset.range n, g i ∈ Finset.range n ∧ g (g i) = i) ∧ 
      (∀ i ∈ Finset.range n, i < g i → (i + 1) * (g i + 1) ≤ m)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_pair_product_bound_l4077_407705


namespace NUMINAMATH_CALUDE_log_equation_sum_l4077_407736

theorem log_equation_sum (A B C : ℕ+) 
  (h_coprime : Nat.gcd A.val (Nat.gcd B.val C.val) = 1)
  (h_eq : A * Real.log 5 / Real.log 100 + B * Real.log 2 / Real.log 100 = C) : 
  A + B + C = 5 := by
sorry

end NUMINAMATH_CALUDE_log_equation_sum_l4077_407736


namespace NUMINAMATH_CALUDE_magnitude_of_vector_combination_l4077_407776

/-- Given two plane vectors a and b with the angle between them π/2 and magnitudes 1,
    prove that the magnitude of 3a - 2b is 1. -/
theorem magnitude_of_vector_combination (a b : ℝ × ℝ) :
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- angle between a and b is π/2
  (a.1^2 + a.2^2 = 1) →  -- |a| = 1
  (b.1^2 + b.2^2 = 1) →  -- |b| = 1
  ((3*a.1 - 2*b.1)^2 + (3*a.2 - 2*b.2)^2 = 1) := by
sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_combination_l4077_407776


namespace NUMINAMATH_CALUDE_product_of_solutions_l4077_407735

theorem product_of_solutions (x : ℝ) : 
  (∃ α β : ℝ, x^2 + 4*x + 49 = 0 ∧ x = α ∨ x = β) → 
  (∃ α β : ℝ, x^2 + 4*x + 49 = 0 ∧ x = α ∨ x = β ∧ α * β = -49) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_l4077_407735


namespace NUMINAMATH_CALUDE_total_precious_stones_l4077_407737

theorem total_precious_stones (agate olivine diamond : ℕ) : 
  olivine = agate + 5 →
  diamond = olivine + 11 →
  agate = 30 →
  agate + olivine + diamond = 111 := by
sorry

end NUMINAMATH_CALUDE_total_precious_stones_l4077_407737


namespace NUMINAMATH_CALUDE_square_position_after_2023_transformations_l4077_407734

-- Define the possible square positions
inductive SquarePosition
  | ABCD
  | CDAB
  | DCBA
  | BADC

-- Define the transformations
def rotate180 (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.CDAB
  | SquarePosition.CDAB => SquarePosition.ABCD
  | SquarePosition.DCBA => SquarePosition.BADC
  | SquarePosition.BADC => SquarePosition.DCBA

def reflectHorizontal (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.DCBA
  | SquarePosition.CDAB => SquarePosition.BADC
  | SquarePosition.DCBA => SquarePosition.ABCD
  | SquarePosition.BADC => SquarePosition.CDAB

-- Define the alternating transformations
def alternateTransform (n : Nat) (pos : SquarePosition) : SquarePosition :=
  match n with
  | 0 => pos
  | n + 1 => 
    if n % 2 == 0
    then reflectHorizontal (alternateTransform n pos)
    else rotate180 (alternateTransform n pos)

-- The theorem to prove
theorem square_position_after_2023_transformations :
  alternateTransform 2023 SquarePosition.ABCD = SquarePosition.DCBA := by
  sorry


end NUMINAMATH_CALUDE_square_position_after_2023_transformations_l4077_407734


namespace NUMINAMATH_CALUDE_sequence_is_increasing_l4077_407764

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

theorem sequence_is_increasing (a : ℕ → ℝ) 
  (h : ∀ n, a (n + 1) - a n - 2 = 0) : 
  is_increasing a :=
sorry

end NUMINAMATH_CALUDE_sequence_is_increasing_l4077_407764


namespace NUMINAMATH_CALUDE_effective_average_reduction_l4077_407719

theorem effective_average_reduction (initial_price : ℝ) (reduction_percent : ℝ) : 
  reduction_percent = 36 → 
  ∃ (effective_reduction : ℝ), 
    (1 - effective_reduction / 100)^2 * initial_price = 
    (1 - reduction_percent / 100)^2 * initial_price ∧
    effective_reduction = 20 := by
  sorry

end NUMINAMATH_CALUDE_effective_average_reduction_l4077_407719


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l4077_407790

theorem banana_orange_equivalence :
  ∀ (banana_value orange_value : ℝ),
  (3/4 * 12 * banana_value = 6 * orange_value) →
  (1/4 * 12 * banana_value = 2 * orange_value) :=
by
  sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l4077_407790


namespace NUMINAMATH_CALUDE_number_operation_result_l4077_407780

theorem number_operation_result : 
  let x : ℚ := 33
  (x / 4) + 9 = 17.25 := by sorry

end NUMINAMATH_CALUDE_number_operation_result_l4077_407780


namespace NUMINAMATH_CALUDE_floor_painting_theorem_l4077_407774

/-- The number of ordered pairs (a,b) satisfying the floor painting conditions -/
def floor_painting_solutions : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    let (a, b) := p
    b > a ∧ (a - 4) * (b - 4) = 2 * a * b / 3
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- Theorem stating that there are exactly 3 solutions to the floor painting problem -/
theorem floor_painting_theorem : floor_painting_solutions = 3 := by
  sorry

end NUMINAMATH_CALUDE_floor_painting_theorem_l4077_407774


namespace NUMINAMATH_CALUDE_rotten_oranges_percentage_l4077_407768

/-- Proves that the percentage of rotten oranges is 15% given the problem conditions -/
theorem rotten_oranges_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_bananas_percentage : ℚ)
  (good_fruits_percentage : ℚ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_bananas_percentage = 4 / 100)
  (h4 : good_fruits_percentage = 894 / 1000)
  : (90 : ℚ) / total_oranges = 15 / 100 := by
  sorry

#check rotten_oranges_percentage

end NUMINAMATH_CALUDE_rotten_oranges_percentage_l4077_407768


namespace NUMINAMATH_CALUDE_fourth_graders_pizza_problem_l4077_407778

theorem fourth_graders_pizza_problem :
  ∀ (n : ℕ),
  (∀ (student : ℕ), student ≤ n → 20 * 6 * student = 1200) →
  n = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_graders_pizza_problem_l4077_407778


namespace NUMINAMATH_CALUDE_difference_of_squares_l4077_407746

def digits : List Nat := [9, 8, 7, 6, 4, 2, 1, 5]

def largest_number : Nat := 98765421

def smallest_number : Nat := 12456789

theorem difference_of_squares (d : List Nat) (largest smallest : Nat) :
  d = digits →
  largest = largest_number →
  smallest = smallest_number →
  largest * largest - smallest * smallest = 9599477756293120 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l4077_407746


namespace NUMINAMATH_CALUDE_legal_fee_participants_l4077_407784

/-- The number of participants paying legal fees -/
def num_participants : ℕ := 8

/-- The total legal costs in francs -/
def total_cost : ℕ := 800

/-- The number of participants who cannot pay -/
def non_paying_participants : ℕ := 3

/-- The additional amount each paying participant contributes in francs -/
def additional_payment : ℕ := 60

/-- Theorem stating that the number of participants satisfies the given conditions -/
theorem legal_fee_participants :
  (total_cost : ℚ) / num_participants + additional_payment = 
  total_cost / (num_participants - non_paying_participants) :=
by sorry

end NUMINAMATH_CALUDE_legal_fee_participants_l4077_407784


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_7_mod_17_is_correct_l4077_407724

/-- The least five-digit positive integer congruent to 7 (mod 17) -/
def least_five_digit_congruent_to_7_mod_17 : ℕ := 10003

/-- A number is five-digit if it's between 10000 and 99999 inclusive -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem least_five_digit_congruent_to_7_mod_17_is_correct :
  is_five_digit least_five_digit_congruent_to_7_mod_17 ∧
  least_five_digit_congruent_to_7_mod_17 % 17 = 7 ∧
  ∀ n : ℕ, is_five_digit n ∧ n % 17 = 7 → n ≥ least_five_digit_congruent_to_7_mod_17 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_7_mod_17_is_correct_l4077_407724


namespace NUMINAMATH_CALUDE_white_balls_remain_odd_one_white_ball_left_l4077_407766

/-- Represents the state of the bag with black and white balls -/
structure BagState where
  white : Nat
  black : Nat

/-- The process of drawing two balls and applying the rules -/
def drawBalls (state : BagState) : BagState :=
  sorry

/-- Predicate to check if the process has ended (0 or 1 ball left) -/
def processEnded (state : BagState) : Prop :=
  state.white + state.black ≤ 1

/-- Theorem stating that the number of white balls remains odd throughout the process -/
theorem white_balls_remain_odd (initial : BagState) (final : BagState) 
    (h_initial : initial.white = 2007 ∧ initial.black = 2007)
    (h_process : final = drawBalls initial ∨ (∃ intermediate, final = drawBalls intermediate ∧ ¬processEnded intermediate)) :
  Odd final.white :=
  sorry

/-- Main theorem proving that one white ball is left at the end -/
theorem one_white_ball_left (initial : BagState) (final : BagState)
    (h_initial : initial.white = 2007 ∧ initial.black = 2007)
    (h_process : final = drawBalls initial ∨ (∃ intermediate, final = drawBalls intermediate ∧ ¬processEnded intermediate))
    (h_ended : processEnded final) :
  final.white = 1 ∧ final.black = 0 :=
  sorry

end NUMINAMATH_CALUDE_white_balls_remain_odd_one_white_ball_left_l4077_407766


namespace NUMINAMATH_CALUDE_min_value_reciprocal_product_l4077_407797

theorem min_value_reciprocal_product (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_mean : 2 = (2 * a + b) / 2) : 
  ∀ x y, x > 0 → y > 0 → (2 = (2 * x + y) / 2) → 1 / (a * b) ≤ 1 / (x * y) :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_product_l4077_407797


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l4077_407795

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) → x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l4077_407795


namespace NUMINAMATH_CALUDE_tangent_line_condition_function_upper_bound_inequality_for_reciprocal_product_l4077_407701

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := x / Real.exp x - a * x * Real.log x

theorem tangent_line_condition (a : ℝ) : 
  (deriv (f a)) 1 = -1 → a = 1 := by sorry

theorem function_upper_bound (x : ℝ) (hx : x > 0) : 
  x / Real.exp x - x * Real.log x < 2 / Real.exp 1 := by sorry

theorem inequality_for_reciprocal_product (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m * n = 1) :
  1 / Real.exp m + 1 / Real.exp n < 2 * (m + n) := by sorry

end NUMINAMATH_CALUDE_tangent_line_condition_function_upper_bound_inequality_for_reciprocal_product_l4077_407701


namespace NUMINAMATH_CALUDE_double_papers_double_time_l4077_407749

/-- Represents the time taken to check exam papers under different conditions -/
def exam_check_time (men : ℕ) (days : ℕ) (hours_per_day : ℕ) (papers : ℕ) : ℕ :=
  men * days * hours_per_day

/-- Theorem stating the relationship between different exam checking scenarios -/
theorem double_papers_double_time (men₁ days₁ hours₁ men₂ days₂ papers₁ : ℕ) :
  exam_check_time men₁ days₁ hours₁ papers₁ = 160 →
  men₁ = 4 →
  days₁ = 8 →
  hours₁ = 5 →
  men₂ = 2 →
  days₂ = 20 →
  exam_check_time men₂ days₂ 8 (2 * papers₁) = 320 := by
  sorry

#check double_papers_double_time

end NUMINAMATH_CALUDE_double_papers_double_time_l4077_407749


namespace NUMINAMATH_CALUDE_subtraction_result_l4077_407775

theorem subtraction_result (number : ℝ) (percentage : ℝ) (subtrahend : ℝ) : 
  number = 200 → 
  percentage = 95 → 
  subtrahend = 12 → 
  (percentage / 100) * number - subtrahend = 178 := by
sorry

end NUMINAMATH_CALUDE_subtraction_result_l4077_407775


namespace NUMINAMATH_CALUDE_prob_sum_le_4_l4077_407779

/-- The number of possible outcomes for a single die. -/
def die_outcomes : ℕ := 6

/-- The set of all possible outcomes when throwing two dice. -/
def all_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range die_outcomes) (Finset.range die_outcomes)

/-- The set of favorable outcomes where the sum is less than or equal to 4. -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  all_outcomes.filter (fun p => p.1 + p.2 ≤ 4)

/-- The probability of the sum of two dice being less than or equal to 4. -/
theorem prob_sum_le_4 :
    (favorable_outcomes.card : ℚ) / all_outcomes.card = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_le_4_l4077_407779


namespace NUMINAMATH_CALUDE_jenny_distance_difference_l4077_407765

theorem jenny_distance_difference (run_distance walk_distance : ℝ) 
  (h1 : run_distance = 0.6)
  (h2 : walk_distance = 0.4) :
  run_distance - walk_distance = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_jenny_distance_difference_l4077_407765


namespace NUMINAMATH_CALUDE_sum_of_equal_expressions_l4077_407794

theorem sum_of_equal_expressions (a b c d : ℝ) :
  a + 2 = b + 3 ∧ 
  b + 3 = c + 4 ∧ 
  c + 4 = d + 5 ∧ 
  d + 5 = a + b + c + d + 10 →
  a + b + c + d = -26/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_equal_expressions_l4077_407794


namespace NUMINAMATH_CALUDE_circle_equation_l4077_407792

-- Define the given circle
def given_circle (x y : ℝ) : Prop := (x + 3)^2 + (y + 3)^2 = 18

-- Define the point A
def point_A : ℝ × ℝ := (0, 2)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the circle we want to prove
def target_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

-- Theorem statement
theorem circle_equation :
  (∀ x y : ℝ, target_circle x y ↔ 
    (((x, y) = point_A ∨ (x, y) = origin) ∧ 
     (∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
       ((x' - x)^2 + (y' - y)^2 < δ^2 → 
        (target_circle x' y' ↔ ¬given_circle x' y'))))) := 
by sorry

end NUMINAMATH_CALUDE_circle_equation_l4077_407792


namespace NUMINAMATH_CALUDE_job_completion_time_l4077_407758

/-- If a group can complete a job in 20 days, twice the group can do half the job in 5 days -/
theorem job_completion_time (people : ℕ) (work : ℝ) : 
  (people * work = 20) → (2 * people) * (work / 2) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l4077_407758


namespace NUMINAMATH_CALUDE_simplify_expression_l4077_407738

theorem simplify_expression (y : ℝ) : 2*y + 8*y^2 + 6 - (3 - 2*y - 8*y^2) = 16*y^2 + 4*y + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4077_407738


namespace NUMINAMATH_CALUDE_sum_base7_equals_650_l4077_407798

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 --/
def base10ToBase7 (n : ℕ) : ℕ := sorry

/-- The sum of three numbers in base 7 --/
def sumBase7 (a b c : ℕ) : ℕ :=
  base10ToBase7 (base7ToBase10 a + base7ToBase10 b + base7ToBase10 c)

theorem sum_base7_equals_650 :
  sumBase7 543 65 6 = 650 := by sorry

end NUMINAMATH_CALUDE_sum_base7_equals_650_l4077_407798


namespace NUMINAMATH_CALUDE_expression_evaluation_l4077_407715

theorem expression_evaluation (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^2)^y * (y^3)^x / ((y^2)^y * (x^3)^x) = x^(2*y - 3*x) * y^(3*x - 2*y) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4077_407715


namespace NUMINAMATH_CALUDE_table_tennis_matches_l4077_407757

theorem table_tennis_matches (player1_matches player2_matches : ℕ) 
  (h1 : player1_matches = 10) 
  (h2 : player2_matches = 21) : ℕ := by
  -- The number of matches the third player played
  sorry

#check table_tennis_matches

end NUMINAMATH_CALUDE_table_tennis_matches_l4077_407757


namespace NUMINAMATH_CALUDE_catchup_time_correct_l4077_407753

/-- The time (in hours) it takes for the second car to catch up with the first car -/
def catchup_time : ℝ := 1.5

/-- The speed of the first car in km/h -/
def speed1 : ℝ := 60

/-- The speed of the second car in km/h -/
def speed2 : ℝ := 80

/-- The head start time of the first car in hours -/
def head_start : ℝ := 0.5

/-- Theorem stating that the catchup time is correct given the conditions -/
theorem catchup_time_correct : 
  speed1 * (catchup_time + head_start) = speed2 * catchup_time := by
  sorry

#check catchup_time_correct

end NUMINAMATH_CALUDE_catchup_time_correct_l4077_407753
