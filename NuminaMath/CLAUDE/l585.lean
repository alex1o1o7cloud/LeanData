import Mathlib

namespace NUMINAMATH_CALUDE_jessica_non_work_days_l585_58524

/-- Calculates the number of non-work days given the problem conditions -/
theorem jessica_non_work_days 
  (total_days : ℕ) 
  (full_day_earnings : ℚ) 
  (non_work_deduction : ℚ) 
  (half_days : ℕ) 
  (total_earnings : ℚ) 
  (h1 : total_days = 30)
  (h2 : full_day_earnings = 80)
  (h3 : non_work_deduction = 40)
  (h4 : half_days = 5)
  (h5 : total_earnings = 1600) :
  ∃ (non_work_days : ℕ), 
    non_work_days = 5 ∧ 
    (total_days : ℚ) = (non_work_days : ℚ) + (half_days : ℚ) + 
      ((total_earnings + non_work_deduction * (non_work_days : ℚ) - 
        (half_days : ℚ) * full_day_earnings / 2) / full_day_earnings) :=
by sorry

end NUMINAMATH_CALUDE_jessica_non_work_days_l585_58524


namespace NUMINAMATH_CALUDE_rational_root_of_cubic_l585_58531

theorem rational_root_of_cubic (b c : ℚ) :
  (∃ x : ℝ, x^3 - 4*x^2 + b*x + c = 0 ∧ x = 4 - Real.sqrt 11) →
  (∃ y : ℚ, y^3 - 4*y^2 + b*y + c = 0) →
  (∃ z : ℚ, z^3 - 4*z^2 + b*z + c = 0 ∧ z = -4) :=
by sorry

end NUMINAMATH_CALUDE_rational_root_of_cubic_l585_58531


namespace NUMINAMATH_CALUDE_water_filter_capacity_l585_58590

/-- The total capacity of a cylindrical water filter in liters. -/
def total_capacity : ℝ := 120

/-- The amount of water in the filter when it is partially filled, in liters. -/
def partial_amount : ℝ := 36

/-- The fraction of the filter that is filled when it contains the partial amount. -/
def partial_fraction : ℝ := 0.30

/-- Theorem stating that the total capacity of the water filter is 120 liters,
    given that it contains 36 liters when it is 30% full. -/
theorem water_filter_capacity :
  total_capacity * partial_fraction = partial_amount :=
by sorry

end NUMINAMATH_CALUDE_water_filter_capacity_l585_58590


namespace NUMINAMATH_CALUDE_johny_journey_distance_johny_journey_specific_distance_l585_58526

/-- Calculates the total distance of Johny's journey given his travel pattern. -/
theorem johny_journey_distance : ℕ → ℕ → ℕ
  | south_distance, east_extra_distance =>
    let east_distance := south_distance + east_extra_distance
    let north_distance := 2 * east_distance
    south_distance + east_distance + north_distance

/-- Proves that Johny's journey distance is 220 miles given the specific conditions. -/
theorem johny_journey_specific_distance :
  johny_journey_distance 40 20 = 220 := by
  sorry

end NUMINAMATH_CALUDE_johny_journey_distance_johny_journey_specific_distance_l585_58526


namespace NUMINAMATH_CALUDE_cake_price_is_twelve_l585_58537

/-- Represents the daily sales and expenses of Marie's bakery --/
structure BakeryFinances where
  cashRegisterCost : ℕ
  breadPrice : ℕ
  breadQuantity : ℕ
  cakeQuantity : ℕ
  rentCost : ℕ
  electricityCost : ℕ
  profitDays : ℕ

/-- Calculates the price of each cake based on the given finances --/
def calculateCakePrice (finances : BakeryFinances) : ℕ :=
  let dailyBreadIncome := finances.breadPrice * finances.breadQuantity
  let dailyExpenses := finances.rentCost + finances.electricityCost
  let dailyProfitWithoutCakes := dailyBreadIncome - dailyExpenses
  let totalProfit := finances.cashRegisterCost
  let profitFromCakes := totalProfit - (finances.profitDays * dailyProfitWithoutCakes)
  profitFromCakes / (finances.cakeQuantity * finances.profitDays)

/-- Theorem stating that the cake price is $12 given the specific conditions --/
theorem cake_price_is_twelve (finances : BakeryFinances)
  (h1 : finances.cashRegisterCost = 1040)
  (h2 : finances.breadPrice = 2)
  (h3 : finances.breadQuantity = 40)
  (h4 : finances.cakeQuantity = 6)
  (h5 : finances.rentCost = 20)
  (h6 : finances.electricityCost = 2)
  (h7 : finances.profitDays = 8) :
  calculateCakePrice finances = 12 := by
  sorry

end NUMINAMATH_CALUDE_cake_price_is_twelve_l585_58537


namespace NUMINAMATH_CALUDE_money_ratio_problem_l585_58572

theorem money_ratio_problem (ram_money gopal_money krishan_money : ℕ) :
  ram_money = 588 →
  krishan_money = 3468 →
  gopal_money * 17 = krishan_money * 7 →
  ∃ (a b : ℕ), a * gopal_money = b * ram_money ∧ a = 3 ∧ b = 7 :=
by sorry

end NUMINAMATH_CALUDE_money_ratio_problem_l585_58572


namespace NUMINAMATH_CALUDE_equal_opposite_angles_imag_prod_zero_l585_58501

/-- Given complex numbers a, b, c, d where the angles a 0 b and c 0 d are equal and oppositely oriented,
    the imaginary part of their product abcd is zero. -/
theorem equal_opposite_angles_imag_prod_zero
  (a b c d : ℂ)
  (h : ∃ (θ : ℝ), (b / a).arg = θ ∧ (d / c).arg = -θ) :
  (a * b * c * d).im = 0 := by
  sorry

end NUMINAMATH_CALUDE_equal_opposite_angles_imag_prod_zero_l585_58501


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l585_58598

theorem diophantine_equation_solution (x y z : ℤ) :
  x ≠ 0 → y ≠ 0 → z ≠ 0 → x + y + z ≠ 0 →
  (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = (1 : ℚ) / (x + y + z) →
  (z = -x - y) ∨ (y = -x - z) ∨ (x = -y - z) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l585_58598


namespace NUMINAMATH_CALUDE_cone_prism_volume_ratio_l585_58514

/-- The ratio of the volume of a right circular cone inscribed in a right rectangular prism to the volume of the prism -/
theorem cone_prism_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (1 / 3 * π * r^2 * h) / (6 * r^2 * h) = π / 18 := by
  sorry


end NUMINAMATH_CALUDE_cone_prism_volume_ratio_l585_58514


namespace NUMINAMATH_CALUDE_log_12_5_value_l585_58576

-- Define the given conditions
axiom a : ℝ
axiom b : ℝ
axiom lg_2_eq_a : Real.log 2 = a
axiom ten_pow_b_eq_3 : (10 : ℝ)^b = 3

-- State the theorem to be proved
theorem log_12_5_value : Real.log 5 / Real.log 12 = (1 - a) / (2 * a + b) := by sorry

end NUMINAMATH_CALUDE_log_12_5_value_l585_58576


namespace NUMINAMATH_CALUDE_exactly_two_false_l585_58528

-- Define the basic concepts
def Line : Type := sorry
def perpendicular (l1 l2 : Line) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def skew (l1 l2 : Line) : Prop := sorry
def intersects (l1 l2 : Line) : Prop := sorry

-- Define the statements
def statement1 : Prop := ∀ l1 l2 l3 : Line, perpendicular l1 l3 → perpendicular l2 l3 → parallel l1 l2
def statement2 : Prop := ∀ l1 l2 l3 : Line, parallel l1 l3 → parallel l2 l3 → parallel l1 l2
def statement3 : Prop := ∀ a b c : Line, parallel a b → perpendicular b c → perpendicular a c
def statement4 : Prop := ∀ a b l1 l2 : Line, skew a b → intersects l1 a → intersects l1 b → intersects l2 a → intersects l2 b → skew l1 l2

-- The theorem to prove
theorem exactly_two_false : 
  (¬statement1 ∧ statement2 ∧ statement3 ∧ ¬statement4) ∨
  (¬statement1 ∧ statement2 ∧ ¬statement3 ∧ ¬statement4) ∨
  (¬statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4) ∨
  (statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4) ∨
  (statement1 ∧ statement2 ∧ ¬statement3 ∧ ¬statement4) ∨
  (¬statement1 ∧ ¬statement2 ∧ statement3 ∧ statement4) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_false_l585_58528


namespace NUMINAMATH_CALUDE_volume_circumscribed_sphere_unit_cube_l585_58534

/-- The volume of a circumscribed sphere of a cube with edge length 1 -/
theorem volume_circumscribed_sphere_unit_cube :
  let edge_length : ℝ := 1
  let radius : ℝ := (Real.sqrt 3) / 2
  let volume : ℝ := (4/3) * Real.pi * radius^3
  volume = (Real.sqrt 3 / 2) * Real.pi := by
sorry

end NUMINAMATH_CALUDE_volume_circumscribed_sphere_unit_cube_l585_58534


namespace NUMINAMATH_CALUDE_min_cubes_for_valid_config_l585_58532

/-- Represents a modified cube with two protruding snaps and four receptacle holes. -/
structure ModifiedCube :=
  (snaps : Fin 2)
  (holes : Fin 4)

/-- Represents a configuration of snapped-together cubes. -/
structure CubeConfiguration :=
  (cubes : List ModifiedCube)
  (all_snaps_covered : Bool)

/-- Returns true if all snaps are covered in the given configuration. -/
def all_snaps_covered (config : CubeConfiguration) : Bool :=
  config.all_snaps_covered

/-- The minimum number of cubes required for a valid configuration. -/
def min_cubes : Nat := 6

/-- Theorem stating that the minimum number of cubes for a valid configuration is 6. -/
theorem min_cubes_for_valid_config :
  ∀ (config : CubeConfiguration),
    all_snaps_covered config →
    config.cubes.length ≥ min_cubes :=
  sorry

end NUMINAMATH_CALUDE_min_cubes_for_valid_config_l585_58532


namespace NUMINAMATH_CALUDE_worker_times_relationship_l585_58557

/-- The time it takes for two workers to load a truck together -/
def combined_time : ℝ := 3.428571428571429

/-- The time it takes for the first worker to load a truck alone -/
def worker1_time : ℝ := 6

/-- The time it takes for the second worker to load a truck alone -/
def worker2_time : ℝ := 8

/-- Theorem stating the relationship between the workers' times -/
theorem worker_times_relationship : 
  1 / combined_time = 1 / worker1_time + 1 / worker2_time :=
sorry

end NUMINAMATH_CALUDE_worker_times_relationship_l585_58557


namespace NUMINAMATH_CALUDE_total_students_shaking_hands_l585_58573

/-- The number of students from each school who participated in the debate --/
structure SchoolParticipation where
  school1 : ℕ
  school2 : ℕ
  school3 : ℕ

/-- The conditions of the debate participation --/
def debateConditions (p : SchoolParticipation) : Prop :=
  p.school1 = 2 * p.school2 ∧
  p.school2 = p.school3 + 40 ∧
  p.school3 = 200

/-- The theorem stating the total number of students who shook the mayor's hand --/
theorem total_students_shaking_hands (p : SchoolParticipation) 
  (h : debateConditions p) : p.school1 + p.school2 + p.school3 = 920 := by
  sorry

end NUMINAMATH_CALUDE_total_students_shaking_hands_l585_58573


namespace NUMINAMATH_CALUDE_minute_hand_rotation_1h50m_l585_58579

/-- Represents the rotation of a clock's minute hand in degrees -/
def minute_hand_rotation (hours : ℕ) (minutes : ℕ) : ℤ :=
  -(hours * 360 + (minutes * 360) / 60)

/-- Theorem stating that for 1 hour and 50 minutes, the minute hand rotates -660 degrees -/
theorem minute_hand_rotation_1h50m : 
  minute_hand_rotation 1 50 = -660 := by
  sorry

end NUMINAMATH_CALUDE_minute_hand_rotation_1h50m_l585_58579


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l585_58541

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l585_58541


namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l585_58538

theorem sum_remainder_mod_seven : 
  (102345 + 102346 + 102347 + 102348 + 102349 + 102350) % 7 = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l585_58538


namespace NUMINAMATH_CALUDE_sum_of_perfect_square_integers_l585_58599

theorem sum_of_perfect_square_integers : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, ∃ k : ℕ, n^2 - 19*n + 99 = k^2) ∧ 
  (∀ n : ℕ, n ∉ S → ¬∃ k : ℕ, n^2 - 19*n + 99 = k^2) ∧
  (S.sum id = 38) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_perfect_square_integers_l585_58599


namespace NUMINAMATH_CALUDE_candy_ratio_l585_58591

theorem candy_ratio : ∀ (red yellow blue : ℕ),
  red = 40 →
  yellow = 3 * red - 20 →
  red + blue = 90 →
  blue * 2 = yellow :=
by sorry

end NUMINAMATH_CALUDE_candy_ratio_l585_58591


namespace NUMINAMATH_CALUDE_probability_theorem_l585_58577

/-- The number of roots of unity for z^1997 - 1 = 0 --/
def n : ℕ := 1997

/-- The set of complex roots of z^1997 - 1 = 0 --/
def roots : Set ℂ := {z : ℂ | z^n = 1}

/-- The condition that needs to be satisfied --/
def condition (v w : ℂ) : Prop := Real.sqrt (2 + Real.sqrt 3) ≤ Complex.abs (v + w)

/-- The number of pairs (v, w) satisfying the condition --/
def satisfying_pairs : ℕ := 332 * (n - 1)

/-- The total number of possible pairs (v, w) --/
def total_pairs : ℕ := n * (n - 1)

/-- The theorem to be proved --/
theorem probability_theorem :
  (satisfying_pairs : ℚ) / total_pairs = 83 / 499 := by sorry

end NUMINAMATH_CALUDE_probability_theorem_l585_58577


namespace NUMINAMATH_CALUDE_teacher_age_l585_58519

theorem teacher_age (num_students : Nat) (student_avg_age : ℝ) (new_avg_age : ℝ) : 
  num_students = 30 →
  student_avg_age = 15 →
  new_avg_age = 16 →
  (num_students * student_avg_age + (num_students + 1) * new_avg_age - num_students * student_avg_age) = 46 := by
  sorry

end NUMINAMATH_CALUDE_teacher_age_l585_58519


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l585_58549

theorem divisibility_by_eleven (m : ℕ+) (k : ℕ) (h : 33 ∣ m ^ k) : 11 ∣ m := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l585_58549


namespace NUMINAMATH_CALUDE_trajectory_of_moving_circle_l585_58565

-- Define the two fixed circles
def C1 (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 2
def C2 (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 2

-- Define a predicate for a point being on the trajectory
def OnTrajectory (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 / 14 = 1 ∨ x = 0

-- Theorem statement
theorem trajectory_of_moving_circle :
  ∀ (x y r : ℝ),
  (∃ (x1 y1 : ℝ), C1 x1 y1 ∧ (x - x1)^2 + (y - y1)^2 = r^2) →
  (∃ (x2 y2 : ℝ), C2 x2 y2 ∧ (x - x2)^2 + (y - y2)^2 = r^2) →
  OnTrajectory x y :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_circle_l585_58565


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l585_58511

theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) : 
  let a₂ := a₁ * q
  let a₃ := a₁ * q^2
  let S₃ := a₁ + a₂ + a₃
  (S₃ = 13 ∧ 2 * (a₂ + 2) = a₁ + a₃) → (q = 3 ∨ q = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l585_58511


namespace NUMINAMATH_CALUDE_natural_numbers_satisfying_conditions_l585_58521

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def sum_of_digits (n : ℕ) : ℕ := sorry

def num_positive_divisors (n : ℕ) : ℕ := sorry

def has_form_4k_plus_3 (p : ℕ) : Prop := ∃ k : ℕ, p = 4 * k + 3

def has_prime_divisor_with_4_or_more_digits (n : ℕ) : Prop := 
  ∃ p : ℕ, is_prime p ∧ p ∣ n ∧ p ≥ 1000

theorem natural_numbers_satisfying_conditions (n : ℕ) : 
  (∀ m : ℕ, m > 1 → is_square m → ¬(m ∣ n)) ∧
  (∃! p : ℕ, is_prime p ∧ p ∣ n ∧ has_form_4k_plus_3 p) ∧
  (sum_of_digits n + 2 = num_positive_divisors n) ∧
  (is_square (n + 3)) ∧
  (¬has_prime_divisor_with_4_or_more_digits n) ↔
  (n = 222 ∨ n = 2022) := by sorry

end NUMINAMATH_CALUDE_natural_numbers_satisfying_conditions_l585_58521


namespace NUMINAMATH_CALUDE_arccos_negative_one_l585_58543

theorem arccos_negative_one : Real.arccos (-1) = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_arccos_negative_one_l585_58543


namespace NUMINAMATH_CALUDE_expression_evaluation_l585_58516

theorem expression_evaluation : -20 + 12 * (8 / 4) - 5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l585_58516


namespace NUMINAMATH_CALUDE_sin_plus_cos_value_l585_58544

theorem sin_plus_cos_value (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) 
  (h3 : Real.sin α * Real.cos α = 1 / 8) : 
  Real.sin α + Real.cos α = Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_plus_cos_value_l585_58544


namespace NUMINAMATH_CALUDE_sum_product_inequality_l585_58563

theorem sum_product_inequality (a b c d : ℝ) (h : a + b + c + d = 0) :
  5 * (a * b + b * c + c * d) + 8 * (a * c + a * d + b * d) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l585_58563


namespace NUMINAMATH_CALUDE_complex_modulus_identity_l585_58581

theorem complex_modulus_identity 
  (z₁ z₂ z₃ z₄ : ℂ) 
  (h₁ : Complex.abs z₁ = 1) 
  (h₂ : Complex.abs z₂ = 1) 
  (h₃ : Complex.abs z₃ = 1) 
  (h₄ : Complex.abs z₄ = 1) : 
  Complex.abs (z₁ - z₂) ^ 2 * Complex.abs (z₃ - z₄) ^ 2 + 
  Complex.abs (z₁ + z₄) ^ 2 * Complex.abs (z₃ - z₂) ^ 2 = 
  Complex.abs (z₁ * (z₂ - z₃) + z₃ * (z₂ - z₁) + z₄ * (z₁ - z₃)) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_identity_l585_58581


namespace NUMINAMATH_CALUDE_coloring_book_problem_l585_58536

theorem coloring_book_problem (book1 : ℕ) (book2 : ℕ) (colored : ℕ) : 
  book1 = 23 → book2 = 32 → colored = 44 → 
  (book1 + book2) - colored = 11 := by
sorry

end NUMINAMATH_CALUDE_coloring_book_problem_l585_58536


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l585_58588

open Real

/-- A function f: ℝ₊ → ℝ₊ satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧ 
  (∀ x > 0, f x < 2*x - x / (1 + x^(3/2))) ∧
  (∀ x > 0, f (f x) = (5/2) * f x - x)

/-- The theorem stating that the only function satisfying the conditions is f(x) = x/2 -/
theorem unique_satisfying_function :
  ∀ f : ℝ → ℝ, SatisfyingFunction f → (∀ x > 0, f x = x/2) :=
by sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l585_58588


namespace NUMINAMATH_CALUDE_balloon_ratio_l585_58523

theorem balloon_ratio (mary_balloons nancy_balloons : ℕ) 
  (h1 : mary_balloons = 28) (h2 : nancy_balloons = 7) :
  mary_balloons / nancy_balloons = 4 := by
  sorry

end NUMINAMATH_CALUDE_balloon_ratio_l585_58523


namespace NUMINAMATH_CALUDE_cos_sin_identity_l585_58597

open Real

theorem cos_sin_identity : 
  cos (89 * π / 180) * cos (π / 180) + sin (91 * π / 180) * sin (181 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_identity_l585_58597


namespace NUMINAMATH_CALUDE_f_2_eq_1_l585_58551

/-- The function f(x) = x^5 - 5x^4 + 10x^3 - 10x^2 + 5x - 1 -/
def f (x : ℝ) : ℝ := x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1

/-- Theorem: f(2) = 1 -/
theorem f_2_eq_1 : f 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_2_eq_1_l585_58551


namespace NUMINAMATH_CALUDE_expression_evaluation_l585_58596

theorem expression_evaluation (c a b d : ℚ) 
  (h1 : d = a + 1)
  (h2 : a = b - 3)
  (h3 : b = c + 5)
  (h4 : c = 6)
  (h5 : d + 3 ≠ 0)
  (h6 : a + 2 ≠ 0)
  (h7 : b - 5 ≠ 0)
  (h8 : c + 7 ≠ 0) :
  ((d + 5) / (d + 3)) * ((a + 3) / (a + 2)) * ((b - 3) / (b - 5)) * ((c + 10) / (c + 7)) = 1232 / 585 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l585_58596


namespace NUMINAMATH_CALUDE_jack_sugar_usage_l585_58582

/-- Represents the amount of sugar Jack has initially -/
def initial_sugar : ℕ := 65

/-- Represents the amount of sugar Jack buys after usage -/
def sugar_bought : ℕ := 50

/-- Represents the final amount of sugar Jack has -/
def final_sugar : ℕ := 97

/-- Represents the amount of sugar Jack uses -/
def sugar_used : ℕ := 18

theorem jack_sugar_usage :
  initial_sugar - sugar_used + sugar_bought = final_sugar :=
by sorry

end NUMINAMATH_CALUDE_jack_sugar_usage_l585_58582


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_trigonometric_fraction_simplification_l585_58562

-- Problem 1
theorem trigonometric_expression_equality : 
  Real.cos (2/3 * Real.pi) - Real.tan (-Real.pi/4) + 3/4 * Real.tan (Real.pi/6) - Real.sin (-31/6 * Real.pi) = Real.sqrt 3 / 4 := by
  sorry

-- Problem 2
theorem trigonometric_fraction_simplification (α : Real) : 
  (Real.sin (Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.cos (-α + 3/2 * Real.pi)) / 
  (Real.cos (Real.pi/2 - α) * Real.sin (-Real.pi - α)) = -Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_trigonometric_fraction_simplification_l585_58562


namespace NUMINAMATH_CALUDE_post_office_mail_count_l585_58592

/- Define the daily intake of letters and packages -/
def letters_per_day : ℕ := 60
def packages_per_day : ℕ := 20

/- Define the number of days in a month and the number of months -/
def days_per_month : ℕ := 30
def months : ℕ := 6

/- Define the total pieces of mail per day -/
def mail_per_day : ℕ := letters_per_day + packages_per_day

/- Theorem to prove -/
theorem post_office_mail_count :
  mail_per_day * days_per_month * months = 14400 :=
by sorry

end NUMINAMATH_CALUDE_post_office_mail_count_l585_58592


namespace NUMINAMATH_CALUDE_new_bill_total_l585_58574

/-- Calculates the new bill total after substitutions and additional charges -/
def calculate_new_bill (original_order : ℝ) 
                       (tomato_old : ℝ) (tomato_new : ℝ)
                       (lettuce_old : ℝ) (lettuce_new : ℝ)
                       (celery_old : ℝ) (celery_new : ℝ)
                       (delivery_tip : ℝ) : ℝ :=
  original_order + (tomato_new - tomato_old) + (lettuce_new - lettuce_old) + 
  (celery_new - celery_old) + delivery_tip

/-- Theorem stating that the new bill total is $35.00 -/
theorem new_bill_total : 
  calculate_new_bill 25 0.99 2.20 1.00 1.75 1.96 2.00 8.00 = 35 := by
  sorry

end NUMINAMATH_CALUDE_new_bill_total_l585_58574


namespace NUMINAMATH_CALUDE_function_properties_l585_58594

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def is_symmetric_about (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

theorem function_properties (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : ∀ x, f (x + 1) = -f x)
  (h3 : is_increasing_on f (-1) 0) :
  (∀ x, f (x + 2) = f x) ∧ 
  (is_symmetric_about f 1) ∧
  (f 2 = f 0) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l585_58594


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l585_58535

theorem system_of_equations_solution :
  ∃ (x y : ℚ), (4 * x - 6 * y = -14) ∧ (8 * x + 3 * y = -15) ∧ (x = -11/5) ∧ (y = 13/15) := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l585_58535


namespace NUMINAMATH_CALUDE_conic_is_pair_of_lines_l585_58510

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop := 9 * x^2 - 36 * y^2 = 0

/-- The first line of the pair -/
def line1 (x y : ℝ) : Prop := x = 2 * y

/-- The second line of the pair -/
def line2 (x y : ℝ) : Prop := x = -2 * y

/-- Theorem stating that the conic equation represents a pair of straight lines -/
theorem conic_is_pair_of_lines :
  ∀ x y : ℝ, conic_equation x y ↔ (line1 x y ∨ line2 x y) :=
sorry

end NUMINAMATH_CALUDE_conic_is_pair_of_lines_l585_58510


namespace NUMINAMATH_CALUDE_eric_ben_difference_l585_58506

theorem eric_ben_difference (jack ben eric : ℕ) : 
  jack = 26 → 
  ben = jack - 9 → 
  eric + ben + jack = 50 → 
  ben - eric = 10 := by
sorry

end NUMINAMATH_CALUDE_eric_ben_difference_l585_58506


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l585_58586

theorem sufficient_not_necessary 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ x y, x + y > a + b ∧ x * y > a * b ∧ ¬(x > a ∧ y > b)) ∧ 
  (∀ x y, x > a ∧ y > b → x + y > a + b ∧ x * y > a * b) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l585_58586


namespace NUMINAMATH_CALUDE_function_properties_l585_58545

def f (x : ℝ) : ℝ := |2*x + 2| - 5

def g (m : ℝ) (x : ℝ) : ℝ := f x + |x - m|

theorem function_properties (m : ℝ) (h : m > 0) :
  (∀ x, f x - |x - 1| ≥ 0 ↔ x ∈ Set.Iic (-8) ∪ Set.Ici 2) ∧
  (∃ a b c : ℝ, a < b ∧ b < c ∧
    (∀ x, x < a → g m x < 0) ∧
    (∀ x, a < x ∧ x < c → g m x > 0) ∧
    g m a = 0 ∧ g m c = 0) ↔
  3/2 ≤ m ∧ m < 4 :=
sorry

end NUMINAMATH_CALUDE_function_properties_l585_58545


namespace NUMINAMATH_CALUDE_propositions_truth_l585_58578

theorem propositions_truth :
  (∀ x : ℝ, x^2 - x + 1 > 0) ∧
  (∃ x₀ : ℝ, x₀ > 0 ∧ Real.log (1 / x₀) > -x₀ + 1) ∧
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ Real.log x₀ > x₀ - 1) ∧
  (¬ ∀ x : ℝ, x > 0 → (1/2)^x > Real.log x / Real.log (1/2)) :=
by sorry

end NUMINAMATH_CALUDE_propositions_truth_l585_58578


namespace NUMINAMATH_CALUDE_chessboard_covering_impossibility_l585_58527

/-- Represents a chessboard -/
structure Chessboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a domino -/
inductive Domino
  | TwoByTwo
  | OneByFour

/-- Represents a set of dominoes -/
def DominoSet := List Domino

/-- A function to check if a set of dominoes can cover a chessboard -/
def can_cover (board : Chessboard) (dominoes : DominoSet) : Prop :=
  sorry

/-- A function to replace one 2x2 domino with a 1x4 domino in a set -/
def replace_one_domino (dominoes : DominoSet) : DominoSet :=
  sorry

theorem chessboard_covering_impossibility (board : Chessboard) (original_dominoes : DominoSet) :
  board.rows = 2007 →
  board.cols = 2008 →
  can_cover board original_dominoes →
  ¬(can_cover board (replace_one_domino original_dominoes)) :=
  sorry

end NUMINAMATH_CALUDE_chessboard_covering_impossibility_l585_58527


namespace NUMINAMATH_CALUDE_radio_cost_price_l585_58533

/-- 
Given a radio sold for Rs. 1330 with a 30% loss, 
prove that the original cost price was Rs. 1900.
-/
theorem radio_cost_price (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1330)
  (h2 : loss_percentage = 30) : 
  (selling_price / (1 - loss_percentage / 100)) = 1900 := by
  sorry

end NUMINAMATH_CALUDE_radio_cost_price_l585_58533


namespace NUMINAMATH_CALUDE_perfect_squares_between_powers_of_three_l585_58525

theorem perfect_squares_between_powers_of_three : 
  (Finset.range (Nat.succ (Nat.sqrt (3^10 + 3))) 
    |>.filter (λ n => n^2 ≥ 3^5 + 3 ∧ n^2 ≤ 3^10 + 3)).card = 228 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_between_powers_of_three_l585_58525


namespace NUMINAMATH_CALUDE_coplanar_vectors_m_l585_58569

/-- Three vectors in ℝ³ are coplanar if and only if their scalar triple product is zero -/
def coplanar (a b c : ℝ × ℝ × ℝ) : Prop :=
  let (a₁, a₂, a₃) := a
  let (b₁, b₂, b₃) := b
  let (c₁, c₂, c₃) := c
  a₁ * (b₂ * c₃ - b₃ * c₂) - a₂ * (b₁ * c₃ - b₃ * c₁) + a₃ * (b₁ * c₂ - b₂ * c₁) = 0

theorem coplanar_vectors_m (m : ℝ) : 
  coplanar (1, -1, 0) (-1, 2, 1) (2, 1, m) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_coplanar_vectors_m_l585_58569


namespace NUMINAMATH_CALUDE_min_value_theorem_l585_58552

theorem min_value_theorem (x y z : ℝ) (h : (1 / x) + (2 / y) + (3 / z) = 1) :
  x + y / 2 + z / 3 ≥ 9 ∧
  (x + y / 2 + z / 3 = 9 ↔ x = y / 2 ∧ y / 2 = z / 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l585_58552


namespace NUMINAMATH_CALUDE_greatest_third_side_proof_l585_58593

/-- The greatest integer length of the third side of a triangle with two sides of 7 cm and 15 cm -/
def greatest_third_side : ℕ := 21

/-- Triangle inequality theorem for our specific case -/
axiom triangle_inequality (a b c : ℝ) : 
  (a = 7 ∧ b = 15) → (c < a + b ∧ c > |a - b|)

theorem greatest_third_side_proof : 
  ∀ c : ℝ, (c < 22 ∧ c > 8) → c ≤ greatest_third_side := by sorry

end NUMINAMATH_CALUDE_greatest_third_side_proof_l585_58593


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l585_58518

-- Define the radical conjugate
def radical_conjugate (a b : ℝ) : ℝ := a - b

-- Theorem statement
theorem sum_with_radical_conjugate :
  let x : ℝ := 15 - Real.sqrt 500
  let y : ℝ := radical_conjugate 15 (Real.sqrt 500)
  x + y = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l585_58518


namespace NUMINAMATH_CALUDE_robin_bracelet_cost_l585_58529

/-- Represents the types of bracelets available --/
inductive BraceletType
| Plastic
| Metal
| Beaded

/-- Represents a friend and their bracelet preference --/
structure Friend where
  name : String
  preference : List BraceletType

/-- Calculates the cost of a single bracelet --/
def braceletCost (type : BraceletType) : ℚ :=
  match type with
  | BraceletType.Plastic => 2
  | BraceletType.Metal => 3
  | BraceletType.Beaded => 5

/-- Calculates the total cost for a friend's bracelets --/
def friendCost (friend : Friend) : ℚ :=
  let numBracelets := friend.name.length
  let preferredTypes := friend.preference
  let costs := preferredTypes.map braceletCost
  let totalCost := costs.sum * numBracelets / preferredTypes.length
  totalCost

/-- Applies discount if applicable --/
def applyDiscount (total : ℚ) (numBracelets : ℕ) : ℚ :=
  if numBracelets ≥ 10 then total * (1 - 0.1) else total

/-- Applies sales tax --/
def applySalesTax (total : ℚ) : ℚ :=
  total * (1 + 0.07)

/-- The main theorem to prove --/
theorem robin_bracelet_cost : 
  let friends : List Friend := [
    ⟨"Jessica", [BraceletType.Plastic]⟩,
    ⟨"Tori", [BraceletType.Metal]⟩,
    ⟨"Lily", [BraceletType.Beaded]⟩,
    ⟨"Patrice", [BraceletType.Metal, BraceletType.Beaded]⟩
  ]
  let totalCost := friends.map friendCost |>.sum
  let numBracelets := friends.map (fun f => f.name.length) |>.sum
  let discountedCost := applyDiscount totalCost numBracelets
  let finalCost := applySalesTax discountedCost
  finalCost = 7223/100 := by
  sorry

end NUMINAMATH_CALUDE_robin_bracelet_cost_l585_58529


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l585_58566

theorem complex_expression_simplification :
  (7 - 3*Complex.I) - 4*(2 + 5*Complex.I) + 3*(1 - 4*Complex.I) = 2 - 35*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l585_58566


namespace NUMINAMATH_CALUDE_probability_three_green_marbles_l585_58520

/-- The probability of picking exactly k successes in n trials with probability p for each trial. -/
def binomialProbability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The number of green marbles -/
def greenMarbles : ℕ := 8

/-- The number of purple marbles -/
def purpleMarbles : ℕ := 7

/-- The total number of marbles -/
def totalMarbles : ℕ := greenMarbles + purpleMarbles

/-- The number of trials -/
def numTrials : ℕ := 7

/-- The number of green marbles we want to pick -/
def targetGreen : ℕ := 3

/-- The probability of picking a green marble in one trial -/
def probGreen : ℚ := greenMarbles / totalMarbles

theorem probability_three_green_marbles :
  binomialProbability numTrials targetGreen probGreen = 34454336 / 136687500 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_green_marbles_l585_58520


namespace NUMINAMATH_CALUDE_chaperones_count_l585_58540

/-- Calculates the number of volunteer chaperones given the number of children,
    additional lunches, cost per lunch, and total cost. -/
def calculate_chaperones (children : ℕ) (additional : ℕ) (cost_per_lunch : ℕ) (total_cost : ℕ) : ℕ :=
  (total_cost / cost_per_lunch) - children - additional - 1

/-- Theorem stating that the number of volunteer chaperones is 6 given the problem conditions. -/
theorem chaperones_count :
  let children : ℕ := 35
  let additional : ℕ := 3
  let cost_per_lunch : ℕ := 7
  let total_cost : ℕ := 308
  calculate_chaperones children additional cost_per_lunch total_cost = 6 := by
  sorry

#eval calculate_chaperones 35 3 7 308

end NUMINAMATH_CALUDE_chaperones_count_l585_58540


namespace NUMINAMATH_CALUDE_one_is_monomial_l585_58547

/-- A monomial is an algebraic expression with only one term. -/
def IsMonomial (expr : ℕ) : Prop :=
  expr = 1 ∨ ∃ (base : ℕ) (exponent : ℕ), expr = base ^ exponent

/-- Theorem stating that 1 is a monomial. -/
theorem one_is_monomial : IsMonomial 1 := by sorry

end NUMINAMATH_CALUDE_one_is_monomial_l585_58547


namespace NUMINAMATH_CALUDE_inequality_theorem_l585_58561

theorem inequality_theorem (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n ≥ 1) :
  x + (n^n : ℝ) / x^n ≥ n + 1 := by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l585_58561


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l585_58583

/-- Two vectors are parallel if and only if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given two parallel vectors a = (2, 3) and b = (4, y + 1), prove that y = 5 -/
theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (4, y + 1)
  parallel a b → y = 5 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l585_58583


namespace NUMINAMATH_CALUDE_euclidean_algorithm_steps_bound_l585_58542

/-- The number of steps in the Euclidean algorithm for (a, b) -/
def euclidean_steps (a b : ℕ) : ℕ := sorry

/-- The number of digits in the decimal representation of a natural number -/
def decimal_digits (n : ℕ) : ℕ := sorry

theorem euclidean_algorithm_steps_bound (a b : ℕ) (h : a > b) :
  euclidean_steps a b ≤ 5 * decimal_digits b := by sorry

end NUMINAMATH_CALUDE_euclidean_algorithm_steps_bound_l585_58542


namespace NUMINAMATH_CALUDE_region_characterization_l585_58546

def f (x : ℝ) : ℝ := x^2 - 6*x + 5

theorem region_characterization (x y : ℝ) :
  f x + f y ≤ 0 ∧ f x - f y ≥ 0 →
  (x - 3)^2 + (y - 3)^2 ≤ 8 ∧ (x - y)*(x + y - 6) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_region_characterization_l585_58546


namespace NUMINAMATH_CALUDE_range_of_m_l585_58503

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 6*x - 16

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ∈ Set.Icc (-25) (-16)) ∧
  (∀ y ∈ Set.Icc (-25) (-16), ∃ x ∈ Set.Icc 0 m, f x = y) →
  m ∈ Set.Icc 3 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l585_58503


namespace NUMINAMATH_CALUDE_intersects_iff_m_ge_neg_one_l585_58571

/-- A quadratic function f(x) = x^2 + 2x - m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x - m

/-- The graph of f intersects the x-axis -/
def intersects_x_axis (m : ℝ) : Prop :=
  ∃ x : ℝ, f m x = 0

/-- Theorem: The graph of f(x) = x^2 + 2x - m intersects the x-axis
    if and only if m ≥ -1 -/
theorem intersects_iff_m_ge_neg_one (m : ℝ) :
  intersects_x_axis m ↔ m ≥ -1 := by sorry

end NUMINAMATH_CALUDE_intersects_iff_m_ge_neg_one_l585_58571


namespace NUMINAMATH_CALUDE_sheila_hourly_wage_l585_58554

/-- Sheila's work schedule and earnings -/
structure WorkSchedule where
  hours_mwf : ℕ  -- Hours worked on Monday, Wednesday, Friday
  days_mwf : ℕ   -- Number of days worked with hours_mwf
  hours_tt : ℕ   -- Hours worked on Tuesday, Thursday
  days_tt : ℕ    -- Number of days worked with hours_tt
  weekly_earnings : ℕ  -- Weekly earnings in dollars

/-- Calculate Sheila's hourly wage -/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  let total_hours := schedule.hours_mwf * schedule.days_mwf + schedule.hours_tt * schedule.days_tt
  schedule.weekly_earnings / total_hours

/-- Theorem: Sheila's hourly wage is $12 -/
theorem sheila_hourly_wage :
  let sheila_schedule : WorkSchedule := {
    hours_mwf := 8,
    days_mwf := 3,
    hours_tt := 6,
    days_tt := 2,
    weekly_earnings := 432
  }
  hourly_wage sheila_schedule = 12 := by
  sorry


end NUMINAMATH_CALUDE_sheila_hourly_wage_l585_58554


namespace NUMINAMATH_CALUDE_complement_M_union_N_eq_nonneg_reals_l585_58556

-- Define the set of real numbers
variable (r : Set ℝ)

-- Define set M
def M : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (1 - 2/x)}

-- Define set N
def N : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x - 1)}

-- Statement to prove
theorem complement_M_union_N_eq_nonneg_reals :
  (Set.univ \ M) ∪ N = Set.Ici (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_complement_M_union_N_eq_nonneg_reals_l585_58556


namespace NUMINAMATH_CALUDE_perpendicular_angles_counterexample_l585_58500

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents an angle in 3D space -/
structure Angle3D where
  vertex : Point3D
  side1 : Point3D
  side2 : Point3D

/-- Checks if two line segments are perpendicular in 3D space -/
def isPerpendicular (a b c d : Point3D) : Prop := sorry

/-- Calculates the measure of an angle in degrees -/
def angleMeasure (angle : Angle3D) : ℝ := sorry

/-- Theorem: There exist angles with perpendicular sides that are neither equal nor sum to 180° -/
theorem perpendicular_angles_counterexample :
  ∃ (α β : Angle3D),
    isPerpendicular α.vertex α.side1 β.vertex β.side1 ∧
    isPerpendicular α.vertex α.side2 β.vertex β.side2 ∧
    angleMeasure α ≠ angleMeasure β ∧
    angleMeasure α + angleMeasure β ≠ 180 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_angles_counterexample_l585_58500


namespace NUMINAMATH_CALUDE_tangent_line_slope_l585_58507

/-- Given a curve y = x³ + ax + b and a line y = kx + 1 tangent to the curve at point (l, 3),
    prove that k = 2. -/
theorem tangent_line_slope (a b l : ℝ) : 
  (∃ k : ℝ, (3 = l^3 + a*l + b) ∧ (3 = k*l + 1) ∧ 
   (∀ x : ℝ, k*x + 1 ≤ x^3 + a*x + b) ∧
   (∃ x : ℝ, x ≠ l ∧ k*x + 1 < x^3 + a*x + b)) →
  (∃ k : ℝ, k = 2 ∧ (3 = k*l + 1)) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l585_58507


namespace NUMINAMATH_CALUDE_smallest_n_greater_than_20_l585_58589

/-- g(n) is the sum of the digits of 1/(6^n) to the right of the decimal point -/
def g (n : ℕ+) : ℕ :=
  sorry

theorem smallest_n_greater_than_20 :
  (∀ k : ℕ+, k < 4 → g k ≤ 20) ∧ g 4 > 20 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_greater_than_20_l585_58589


namespace NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l585_58539

/-- Calculates the profit percentage given the sale price including tax, tax rate, and cost price. -/
def profit_percentage (sale_price_with_tax : ℚ) (tax_rate : ℚ) (cost_price : ℚ) : ℚ :=
  let sale_price := sale_price_with_tax / (1 + tax_rate)
  let profit := sale_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that under the given conditions, the profit percentage is approximately 4.54%. -/
theorem shopkeeper_profit_percentage :
  let sale_price_with_tax : ℚ := 616
  let tax_rate : ℚ := 1/10
  let cost_price : ℚ := 535.65
  abs (profit_percentage sale_price_with_tax tax_rate cost_price - 454/100) < 1/100 := by
  sorry

#eval profit_percentage 616 (1/10) 535.65

end NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l585_58539


namespace NUMINAMATH_CALUDE_tetrahedron_acute_angles_l585_58587

/-- A tetrahedron with vertices S, A, B, and C -/
structure Tetrahedron where
  S : Point
  A : Point
  B : Point
  C : Point

/-- The dihedral angle between two faces of a tetrahedron -/
def dihedralAngle (t : Tetrahedron) (face1 face2 : Fin 4) : ℝ := sorry

/-- The planar angle at a vertex of a face in a tetrahedron -/
def planarAngle (t : Tetrahedron) (face : Fin 4) (vertex : Fin 3) : ℝ := sorry

/-- A predicate stating that an angle is acute -/
def isAcute (angle : ℝ) : Prop := angle > 0 ∧ angle < Real.pi / 2

theorem tetrahedron_acute_angles (t : Tetrahedron) :
  (∀ face1 face2, isAcute (dihedralAngle t face1 face2)) →
  (∀ face vertex, isAcute (planarAngle t face vertex)) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_acute_angles_l585_58587


namespace NUMINAMATH_CALUDE_chocolate_bars_in_large_box_l585_58580

theorem chocolate_bars_in_large_box :
  let small_boxes : ℕ := 17
  let bars_per_small_box : ℕ := 26
  let total_bars : ℕ := small_boxes * bars_per_small_box
  total_bars = 442 := by
sorry

end NUMINAMATH_CALUDE_chocolate_bars_in_large_box_l585_58580


namespace NUMINAMATH_CALUDE_simplify_fraction_l585_58548

theorem simplify_fraction (a : ℝ) (h : a ≠ 2) :
  (a - 2) * ((a^2 - 4) / (a^2 - 4*a + 4)) = a + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l585_58548


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l585_58570

theorem magnitude_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.abs (2 * i / (1 + i)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l585_58570


namespace NUMINAMATH_CALUDE_parabola_transformation_l585_58595

def original_parabola (x : ℝ) : ℝ := 3 * x^2

def transformed_parabola (x : ℝ) : ℝ := 3 * (x - 3)^2 - 1

theorem parabola_transformation :
  ∀ x : ℝ, transformed_parabola x = original_parabola (x - 3) - 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l585_58595


namespace NUMINAMATH_CALUDE_valid_words_count_l585_58584

def alphabet_size : ℕ := 25
def max_word_length : ℕ := 5

def total_words (n : ℕ) (k : ℕ) : ℕ :=
  (n^1 + n^2 + n^3 + n^4 + n^5)

def words_without_specific_letter (n : ℕ) (k : ℕ) : ℕ :=
  ((n-1)^1 + (n-1)^2 + (n-1)^3 + (n-1)^4 + (n-1)^5)

theorem valid_words_count :
  total_words alphabet_size max_word_length - words_without_specific_letter alphabet_size max_word_length = 1678698 :=
by sorry

end NUMINAMATH_CALUDE_valid_words_count_l585_58584


namespace NUMINAMATH_CALUDE_kaleb_toy_purchase_l585_58512

/-- Represents the problem of calculating how many toys Kaleb can buy -/
theorem kaleb_toy_purchase (saved : ℝ) (new_allowance : ℝ) (allowance_increase : ℝ) 
  (toy_cost : ℝ) : 
  saved = 21 → 
  new_allowance = 15 → 
  allowance_increase = 0.2 →
  toy_cost = 6 →
  (((saved + new_allowance) / 2) / toy_cost : ℝ) = 3 := by
  sorry

#check kaleb_toy_purchase

end NUMINAMATH_CALUDE_kaleb_toy_purchase_l585_58512


namespace NUMINAMATH_CALUDE_red_lucky_stars_count_l585_58585

theorem red_lucky_stars_count (blue : ℕ) (yellow : ℕ) (red : ℕ) :
  blue = 20 →
  yellow = 15 →
  (red : ℚ) / (red + blue + yellow : ℚ) = 1/2 →
  red = 35 := by
sorry

end NUMINAMATH_CALUDE_red_lucky_stars_count_l585_58585


namespace NUMINAMATH_CALUDE_probability_of_black_ball_l585_58550

theorem probability_of_black_ball (prob_red prob_white : ℝ) 
  (h_red : prob_red = 0.42)
  (h_white : prob_white = 0.28)
  (h_sum : prob_red + prob_white + (1 - prob_red - prob_white) = 1) :
  1 - prob_red - prob_white = 0.30 := by
sorry

end NUMINAMATH_CALUDE_probability_of_black_ball_l585_58550


namespace NUMINAMATH_CALUDE_y_days_to_finish_work_l585_58567

/-- The number of days x needs to finish the work alone -/
def x_days : ℕ := 36

/-- The number of days y worked before leaving -/
def y_worked : ℕ := 12

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining : ℕ := 18

/-- The work rate of x (portion of work completed per day) -/
def x_rate : ℚ := 1 / x_days

/-- The total amount of work to be done -/
def total_work : ℚ := 1

/-- The amount of work completed by x after y left -/
def x_completed : ℚ := x_rate * x_remaining

theorem y_days_to_finish_work : ℕ := by
  sorry

end NUMINAMATH_CALUDE_y_days_to_finish_work_l585_58567


namespace NUMINAMATH_CALUDE_circle_equation_l585_58504

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangent line
def tangentLine (x : ℝ) : ℝ := 2 * x + 1

-- Define the properties of the circle
def circleProperties (c : Circle) : Prop :=
  -- The center is on the x-axis
  c.center.2 = 0 ∧
  -- The circle is tangent to the line y = 2x + 1 at point (0, 1)
  c.radius^2 = c.center.1^2 + 1 ∧
  -- The tangent line is perpendicular to the radius at the point of tangency
  2 * c.center.1 + c.center.2 - 1 = 0

-- Theorem statement
theorem circle_equation (c : Circle) (h : circleProperties c) :
  ∀ (x y : ℝ), (x - 2)^2 + y^2 = 5 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l585_58504


namespace NUMINAMATH_CALUDE_problem_solution_l585_58558

theorem problem_solution (x : ℝ) (h : x + 1/x = 3) : 
  (x - 3)^2 + 36/((x - 3)^2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l585_58558


namespace NUMINAMATH_CALUDE_positive_A_value_l585_58502

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h : hash A 7 = 200) : A = Real.sqrt 151 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l585_58502


namespace NUMINAMATH_CALUDE_modular_inverse_13_mod_64_l585_58509

theorem modular_inverse_13_mod_64 :
  ∃ x : ℕ, x < 64 ∧ (13 * x) % 64 = 1 :=
by
  use 5
  sorry

end NUMINAMATH_CALUDE_modular_inverse_13_mod_64_l585_58509


namespace NUMINAMATH_CALUDE_original_number_is_nine_l585_58575

theorem original_number_is_nine (N : ℕ) : (N - 4) % 5 = 0 → N = 9 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_nine_l585_58575


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_trigonometric_expression_value_l585_58530

-- Part 1: Quadratic equation
theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 4*x + 3
  ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 1 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

-- Part 2: Trigonometric expression
theorem trigonometric_expression_value :
  4 * Real.sin (π/6) - Real.sqrt 2 * Real.cos (π/4) + Real.sqrt 3 * Real.tan (π/3) = 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_trigonometric_expression_value_l585_58530


namespace NUMINAMATH_CALUDE_oil_consumption_ranking_l585_58553

/-- Oil consumption per person for each region -/
structure OilConsumption where
  west : ℝ
  nonWest : ℝ
  russia : ℝ

/-- The ranking of oil consumption is correct if Russia > Non-West > West -/
def correctRanking (consumption : OilConsumption) : Prop :=
  consumption.russia > consumption.nonWest ∧ consumption.nonWest > consumption.west

/-- Theorem stating that the given oil consumption data results in the correct ranking -/
theorem oil_consumption_ranking (consumption : OilConsumption) 
  (h_west : consumption.west = 55.084)
  (h_nonWest : consumption.nonWest = 214.59)
  (h_russia : consumption.russia = 1038.33) :
  correctRanking consumption := by
  sorry

#check oil_consumption_ranking

end NUMINAMATH_CALUDE_oil_consumption_ranking_l585_58553


namespace NUMINAMATH_CALUDE_pool_cannot_be_filled_problem_pool_cannot_be_filled_l585_58564

/-- Represents the state of a pool being filled -/
structure PoolFilling where
  capacity : ℝ
  num_hoses : ℕ
  flow_rate_per_hose : ℝ
  leakage_rate : ℝ

/-- Determines if a pool can be filled given its filling conditions -/
def can_be_filled (p : PoolFilling) : Prop :=
  p.num_hoses * p.flow_rate_per_hose > p.leakage_rate

/-- Theorem stating that a pool cannot be filled if inflow rate equals leakage rate -/
theorem pool_cannot_be_filled (p : PoolFilling) 
  (h : p.num_hoses * p.flow_rate_per_hose = p.leakage_rate) : 
  ¬(can_be_filled p) := by
  sorry

/-- The specific pool problem instance -/
def problem_pool : PoolFilling := {
  capacity := 48000
  num_hoses := 6
  flow_rate_per_hose := 3
  leakage_rate := 18
}

/-- Theorem for the specific problem instance -/
theorem problem_pool_cannot_be_filled : 
  ¬(can_be_filled problem_pool) := by
  sorry

end NUMINAMATH_CALUDE_pool_cannot_be_filled_problem_pool_cannot_be_filled_l585_58564


namespace NUMINAMATH_CALUDE_min_value_inequality_l585_58568

theorem min_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : b + c ≥ a) :
  b / c + c / (a + b) ≥ Real.sqrt 2 - 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l585_58568


namespace NUMINAMATH_CALUDE_joe_trading_cards_l585_58555

theorem joe_trading_cards (cards_per_box : ℕ) (num_boxes : ℕ) (h1 : cards_per_box = 8) (h2 : num_boxes = 11) :
  cards_per_box * num_boxes = 88 := by
sorry

end NUMINAMATH_CALUDE_joe_trading_cards_l585_58555


namespace NUMINAMATH_CALUDE_sin_period_scaled_l585_58522

/-- The period of the function y = sin(x/3) is 6π -/
theorem sin_period_scaled (x : ℝ) : 
  ∃ (p : ℝ), p > 0 ∧ ∀ (t : ℝ), Real.sin (t / 3) = Real.sin ((t + p) / 3) ∧ p = 6 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sin_period_scaled_l585_58522


namespace NUMINAMATH_CALUDE_tire_purchase_l585_58517

theorem tire_purchase (cost_per_tire : ℚ) (total_cost : ℚ) (num_tires : ℕ) : 
  cost_per_tire = 1/2 →
  total_cost = 4 →
  num_tires = (total_cost / cost_per_tire).num →
  num_tires = 8 := by
sorry

end NUMINAMATH_CALUDE_tire_purchase_l585_58517


namespace NUMINAMATH_CALUDE_no_divisible_by_five_append_l585_58560

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Appends a digit to the left of 864 to form a four-digit number -/
def appendDigit (d : Digit) : Nat := d.val * 1000 + 864

/-- Theorem: There are no digits that can be appended to the left of 864
    to create a four-digit number divisible by 5 -/
theorem no_divisible_by_five_append :
  ∀ d : Digit, ¬(appendDigit d % 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_divisible_by_five_append_l585_58560


namespace NUMINAMATH_CALUDE_smallest_number_with_8_divisors_multiple_of_24_l585_58513

def is_multiple_of_24 (n : ℕ) : Prop := ∃ k : ℕ, n = 24 * k

def count_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_number_with_8_divisors_multiple_of_24 :
  ∀ n : ℕ, is_multiple_of_24 n ∧ count_divisors n = 8 → n ≥ 720 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_8_divisors_multiple_of_24_l585_58513


namespace NUMINAMATH_CALUDE_multiplication_with_fraction_l585_58559

theorem multiplication_with_fraction : 8 * (1 / 7) * 14 = 16 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_with_fraction_l585_58559


namespace NUMINAMATH_CALUDE_pascal_contest_certificates_l585_58508

theorem pascal_contest_certificates 
  (num_boys : ℕ) 
  (num_girls : ℕ) 
  (percent_boys_cert : ℚ) 
  (percent_girls_cert : ℚ) 
  (h1 : num_boys = 30)
  (h2 : num_girls = 20)
  (h3 : percent_boys_cert = 30 / 100)
  (h4 : percent_girls_cert = 40 / 100) :
  (num_boys * percent_boys_cert + num_girls * percent_girls_cert) / (num_boys + num_girls) = 34 / 100 := by
sorry


end NUMINAMATH_CALUDE_pascal_contest_certificates_l585_58508


namespace NUMINAMATH_CALUDE_pizza_recipe_l585_58505

theorem pizza_recipe (water flour salt : ℚ) : 
  water = 10 ∧ 
  salt = (1/2) * flour ∧ 
  water + flour + salt = 34 →
  flour = 16 := by
sorry

end NUMINAMATH_CALUDE_pizza_recipe_l585_58505


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l585_58515

theorem quadratic_inequality_solution (c : ℝ) : 
  (∀ x : ℝ, -x^2 + 6*x + c < 0 ↔ x < 2 ∨ x > 4) → c = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l585_58515
